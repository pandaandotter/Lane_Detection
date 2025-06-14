import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNet
import tensorflow_model_optimization as tfmot
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path
import importlib.util
import sys


def dice_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(K.clip(y_pred, 0.05, 0.95))
    intersection = K.sum(y_true_f * y_pred_f)
    dice = 1 - (2. * intersection + 1e-6) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1e-6)

    gamma = 2.0
    alpha = 0.01#.05 #0.25
    focal = -alpha * (1 - y_pred_f) ** gamma * K.log(y_pred_f + 1e-6)

    dice_weight = 0.5
    focal_weight = 0.5

    return dice_weight * dice + focal_weight * K.mean(focal)


def build_model():
    base = MobileNet(input_shape=(80, 80, 1), include_top=False, alpha=0.1, weights=None)

    skips = [
        base.get_layer('conv_pw_1_relu').output,
        base.get_layer('conv_pw_3_relu').output,
        base.get_layer('conv_pw_5_relu').output,
        base.get_layer('conv_pw_11_relu').output
    ]
    x = base.output
    for skip in reversed(skips):
        x = layers.UpSampling2D(interpolation='bilinear')(x)
        x = layers.Lambda(lambda tensors: tf.image.resize(tensors[0], tf.shape(tensors[1])[1:3]))([x, skip])
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)

    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    output = layers.Conv2D(1, 1, activation='sigmoid', padding='same')(x)

    return models.Model(inputs=base.input, outputs=output)


def load_dataset(processing_mode: str, batch_size: int = 8):
    dataset_path = Path(__file__).parent / "dataset.py"
    spec = importlib.util.spec_from_file_location("dataset", dataset_path)
    dataset_module = importlib.util.module_from_spec(spec)
    sys.modules["dataset"] = dataset_module
    spec.loader.exec_module(dataset_module)

    return dataset_module.create_dataset80(processing_mode, batch_size=batch_size)


def get_callbacks():
    return [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir='./logs'),
        EarlyStopping(
            monitor='val_loss',
            min_delta=0.001, #TODO: make much lower and let run for night
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
    model_path = model_dir / f"mobilenetv1_80_U-Net_1-{mode.lower()}.h5"
    export_model.save(model_path)
    print(f"Model saved to: {model_path.resolve()}")


def main(processing_mode: str):
    print(f" Starting training with mode: {processing_mode}")

    dataset, _ = load_dataset(processing_mode)
    dataset_size = sum(1 for _ in dataset)
    train_size = int(0.9 * dataset_size)
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    print(f"Dataset size: {dataset_size * 8} samples ({train_size * 8} train, {(dataset_size - train_size) * 8} val)")

    model = build_model()
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.8,
            begin_step=1000,
            end_step=10000
        )
    }
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
    pruned_model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy'])

    history = train_model(pruned_model, train_dataset, val_dataset, get_callbacks())
    print(f"Training stopped after {len(history.history['loss'])} epochs")

    save_model(pruned_model, processing_mode)

    return pruned_model
