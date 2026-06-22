"""Model architectures for brain tumor MRI classification."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, ResNet50


def build_baseline_cnn(input_shape: tuple[int, int, int] = (224, 224, 3), num_classes: int = 4) -> tf.keras.Model:
    """Build a compact custom CNN baseline."""
    inputs = layers.Input(shape=input_shape)

    x = inputs
    for filters, dropout_rate in [(32, 0.20), (64, 0.25), (128, 0.30), (192, 0.35)]:
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Dropout(dropout_rate)(x)

    x = layers.GlobalAveragePooling2D(name="baseline_global_pool")(x)
    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.45)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="baseline_cnn")


def build_transfer_learning_model(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
    backbone: str = "efficientnet",
    train_backbone: bool = False,
) -> tf.keras.Model:
    """Build an EfficientNetB0 or ResNet50 transfer-learning classifier."""
    backbone = backbone.lower()
    inputs = layers.Input(shape=input_shape)

    if backbone in {"efficientnet", "efficientnetb0"}:
        base_model = EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
        x = layers.Rescaling(255.0, name="efficientnet_rescale_to_255")(inputs)
        x = base_model(x, training=False)
        model_name = "transfer_efficientnetb0"
    elif backbone == "resnet50":
        base_model = ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
        x = layers.Lambda(tf.keras.applications.resnet50.preprocess_input, name="resnet50_preprocess")(inputs * 255.0)
        x = base_model(x, training=False)
        model_name = "transfer_resnet50"
    else:
        raise ValueError("backbone must be one of: efficientnet, efficientnetb0, resnet50")

    base_model.trainable = train_backbone

    x = layers.GlobalAveragePooling2D(name="transfer_global_pool")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.45)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return models.Model(inputs=inputs, outputs=outputs, name=model_name)


def unfreeze_top_layers(model: tf.keras.Model, layers_to_unfreeze: int = 30) -> tf.keras.Model:
    """Unfreeze the final layers of the transfer-learning backbone for fine-tuning."""
    backbone = next(
        (
            layer
            for layer in model.layers
            if isinstance(layer, tf.keras.Model) and ("efficientnet" in layer.name or "resnet" in layer.name)
        ),
        None,
    )
    if backbone is None:
        raise ValueError("No named backbone found in the supplied model.")

    backbone.trainable = True
    for layer in backbone.layers[:-layers_to_unfreeze]:
        layer.trainable = False
    for layer in backbone.layers[-layers_to_unfreeze:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    return model


if __name__ == "__main__":
    build_baseline_cnn().summary()
    build_transfer_learning_model(backbone="efficientnet").summary()
