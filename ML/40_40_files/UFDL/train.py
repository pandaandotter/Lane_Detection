from tensorflow.keras import layers, models
import importlib.util
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path
import importlib.util
import sys
import tensorflow_model_optimization as tfmot

# ---------- Config ----------
NUM_ROWS = 40
NUM_COLS = 20
INPUT_SHAPE = (40, 40, 2)
INPUT_SHAPE2 = (40, 40, 1)
NUM_LANES = 2
NUM_POINTS = 20



def masked_sparse_categorical_crossentropy(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1, NUM_COLS])

    mask = tf.not_equal(y_true, -1)
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)

    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)


# ---------- CoordConv ----------
def add_coord_channels(x):
    batch_size = tf.shape(x)[0]
    height, width = x.shape[1], x.shape[2]

    row_coords = tf.linspace(0.0, 1.0, height)
    col_coords = tf.linspace(0.0, 1.0, width)

    row_channel = tf.reshape(row_coords, (1, height, 1, 1))
    row_channel = tf.tile(row_channel, [batch_size, 1, width, 1])

    col_channel = tf.reshape(col_coords, (1, 1, width, 1))
    col_channel = tf.tile(col_channel, [batch_size, height, 1, 1])

    coords = tf.concat([row_channel, col_channel], axis=-1)
    return tf.concat([x, coords], axis=-1)


def build_grid_lane_model(processing_mode):
    if processing_mode == "DUAL":
        inputs = tf.keras.Input(shape=INPUT_SHAPE)
    else:
        inputs = tf.keras.Input(shape=INPUT_SHAPE2)
    x = layers.Lambda(add_coord_channels)(inputs)
    x = layers.Conv2D(3, 1, padding='same')(x)

    base = tf.keras.applications.MobileNet(
        input_shape=(40, 40, 3),
        include_top=False,
        alpha=0.1,
        weights=None
    )
    x = base(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)

    lane_exist = layers.Dense(NUM_LANES * NUM_POINTS, activation='sigmoid', name='lane_exist')(x)

    lane_pos_logits = layers.Dense(NUM_ROWS * NUM_COLS)(x)
    lane_pos = layers.Reshape((NUM_ROWS, NUM_COLS))(lane_pos_logits)
    lane_pos = layers.Softmax(axis=-1, name='lane_pos')(lane_pos)

    return models.Model(inputs=inputs, outputs={'lane_pos': lane_pos, 'lane_exist': lane_exist})


# ---------- Loss & Metrics ----------
def custom_loss():
    return {
        'lane_pos': tf.keras.losses.SparseCategoricalCrossentropy(),
        'lane_exist': tf.keras.losses.BinaryCrossentropy()
    }


def custom_metrics():
    return {
        'lane_pos': 'sparse_categorical_accuracy',
        'lane_exist': 'accuracy'
    }


# ---------- Dataset Loader ----------
def load_dataset(processing_mode: str, batch_size: int = 8):
    dataset_path = Path(__file__).parent / "dataset.py"
    spec = importlib.util.spec_from_file_location("dataset", dataset_path)
    dataset_module = importlib.util.module_from_spec(spec)
    sys.modules["dataset"] = dataset_module
    spec.loader.exec_module(dataset_module)

    dataset, _ = dataset_module.create_dataset40(processing_mode, batch_size=batch_size)

    return dataset

def apply_pruning(model):
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.5,
            begin_step=0,
            end_step=10000
        )
    }
    return tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

def lr_schedule(epoch, lr):
    if epoch < 50:
        return lr
    elif epoch < 75:
        return lr * 0.1
    else:
        return lr * 0.01


def train(processing_mode: str, t_set = None, val_set = None):
    print(f"Training for mode: {processing_mode}")
    if t_set is None:
        dataset = load_dataset(processing_mode)
        dataset_size = dataset.cardinality().numpy()
        train_size = int(dataset_size * 0.9)
        train_ds = dataset.take(train_size)
        val_ds = dataset.skip(train_size)
    else:
        train_ds = t_set
        val_ds = val_set

    model = build_grid_lane_model(processing_mode)
    #model = apply_pruning(model)
    from tensorflow.keras.metrics import BinaryAccuracy
    model.compile(
        optimizer="adam", #tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            'lane_exist': 'binary_crossentropy',
            'lane_pos': masked_sparse_categorical_crossentropy
        },
        loss_weights={
            'lane_exist': 2.0,
            'lane_pos': 1.0
        },
        metrics={
            'lane_exist': BinaryAccuracy(),
            'lane_pos': 'sparse_categorical_accuracy'
        }
    )

    #from tensorflow.keras.callbacks import ReduceLROnPlateau

    #callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-5, verbose=1)]
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        #tf.keras.callbacks.LearningRateScheduler(lr_schedule),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) #patience usually 10
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=120,  #originally 120... usually takes 75
        callbacks=callbacks
    )

    #model = tfmot.sparsity.keras.strip_pruning(model)
    if t_set is None:
        save_path = f'40_40_files/UFDL/models/UFDL_{processing_mode.lower()}.h5'
        model.save(save_path)
        print(f"Pruned model saved to: {save_path}")
    return model


def main(params, t_set = None, val_set = None ):
    return train(params, t_set, val_set)








