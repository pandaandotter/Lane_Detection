import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNet
import tensorflow_model_optimization as tfmot
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path
import importlib.util
import sys


def dice_focal_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(K.clip(y_pred, 0.05, 0.95))
    intersection = K.sum(y_true_f * y_pred_f)
    dice = 1 - (2. * intersection + 1e-6) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1e-6)

    gamma = 2.0
    alpha = .1 #0.25
    focal = -alpha * (1 - y_pred_f) ** gamma * K.log(y_pred_f + 1e-6)

    dice_weight = 0.5
    focal_weight = 0.5

    return dice_weight * dice + focal_weight * K.mean(focal)


def build_model_40():
    base = MobileNet(input_shape=(40, 40, 1), include_top=False, alpha=0.1, weights=None)

    # One skip connection, channel-reduced
    skip = base.get_layer('conv_pw_1_relu').output  # 20x20
    skip = layers.Conv2D(16, 1, activation='relu')(skip)

    x = base.output

    # Upsample to 10x10
    x = layers.UpSampling2D(interpolation='nearest')(x)
    x = layers.DepthwiseConv2D(3, padding='same', activation='relu')(x)
    x = layers.Conv2D(16, 1, activation='relu')(x)

    # Upsample to 20x20 and fuse
    x = layers.UpSampling2D(interpolation='nearest')(x)
    x = layers.Lambda(lambda tensors: tf.image.resize(tensors[0], tf.shape(tensors[1])[1:3]))([x, skip])
    x = layers.Concatenate()([x, skip])
    x = layers.DepthwiseConv2D(3, padding='same', activation='relu')(x)
    x = layers.Conv2D(16, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.05)(x)

    # Final upsample to 40x40
    x = layers.UpSampling2D(interpolation='nearest')(x)
    x = layers.DepthwiseConv2D(3, padding='same', activation='relu')(x)
    x = layers.Conv2D(16, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    output = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(x)

    return models.Model(inputs=base.input, outputs=output)


def load_dataset(processing_mode: str, batch_size: int = 8):
    dataset_path = Path(__file__).parent / "dataset.py"
    spec = importlib.util.spec_from_file_location("dataset", dataset_path)
    dataset_module = importlib.util.module_from_spec(spec)
    sys.modules["dataset"] = dataset_module
    spec.loader.exec_module(dataset_module)

    return dataset_module.create_dataset40(processing_mode, batch_size=batch_size)

def get_callbacks():
    return [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir='./logs'),
        EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]

def train_model(model, train_dataset, val_dataset, callbacks):
    return model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
        callbacks=callbacks
    )


def save_model(model, mode: str):
    export_model = tfmot.sparsity.keras.strip_pruning(model)
    model_dir = Path(__file__).parent / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"mobilenetv1_40_LightSeg-{mode.lower()}.h5"
    export_model.save(model_path)
    print(f"Model saved to: {model_path.resolve()}")


def main(processing_mode: str, t_set = None, val_set = None):
    print(f"Starting training with mode: {processing_mode}")
    if t_set is None:
        dataset = load_dataset(processing_mode)
        dataset_size = dataset.cardinality().numpy()
        train_size = int(dataset_size * 0.9)
        train_ds = dataset.take(train_size)
        val_ds = dataset.skip(train_size)
    else:
        train_ds = t_set
        val_ds = val_set

    model = build_model_40()
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.6,
            begin_step=1000,
            end_step=10000
        )
    }
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
    pruned_model.compile(optimizer='adam', #optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                         loss=dice_focal_loss, metrics=['accuracy'])

    # history can be used to gather data abut the training like the umber of epochs
    history = train_model(pruned_model, train_ds, val_ds, get_callbacks())

    save_model(pruned_model, processing_mode)
    return pruned_model