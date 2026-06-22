"""Grad-CAM visualizations for model interpretability."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import tensorflow as tf

from dataset import CLASSES, get_data_loaders

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def preprocess_image(image_path: str | Path, image_size: int = 224) -> tf.Tensor:
    image = tf.io.read_file(str(image_path))
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, (image_size, image_size))
    image = tf.cast(image, tf.float32) / 255.0
    return tf.expand_dims(image, axis=0)


def find_backbone(model: tf.keras.Model) -> tf.keras.Model | None:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and ("efficientnet" in layer.name or "resnet" in layer.name):
            return layer
    return None


def find_last_conv_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError(f"No Conv2D layer found in model {model.name}.")


def gradcam_for_nested_backbone(
    image_batch: tf.Tensor,
    model: tf.keras.Model,
    backbone: tf.keras.Model,
    class_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Grad-CAM when the last convolution lives inside a backbone model."""
    target_layer = find_last_conv_layer(backbone)
    backbone_grad_model = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=[target_layer.output, backbone.output],
    )

    pre_backbone_layers = []
    head_layers = []
    found_backbone = False
    for layer in model.layers:
        if found_backbone:
            head_layers.append(layer)
        elif layer is not backbone and not isinstance(layer, tf.keras.layers.InputLayer):
            pre_backbone_layers.append(layer)
        if layer is backbone:
            found_backbone = True

    with tf.GradientTape() as tape:
        x = image_batch
        for layer in pre_backbone_layers:
            try:
                x = layer(x, training=False)
            except TypeError:
                x = layer(x)
        conv_outputs, x = backbone_grad_model(x, training=False)
        tape.watch(conv_outputs)
        for layer in head_layers:
            try:
                x = layer(x, training=False)
            except TypeError:
                x = layer(x)
        predictions = x
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, conv_outputs)
    heatmap = build_heatmap(conv_outputs, gradients)
    return heatmap, predictions.numpy()[0]


def gradcam_for_flat_model(
    image_batch: tf.Tensor,
    model: tf.keras.Model,
    class_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    target_layer = find_last_conv_layer(model)
    grad_model = tf.keras.Model(inputs=model.inputs, outputs=[target_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch, training=False)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, conv_outputs)
    heatmap = build_heatmap(conv_outputs, gradients)
    return heatmap, predictions.numpy()[0]


def build_heatmap(conv_outputs: tf.Tensor, gradients: tf.Tensor) -> np.ndarray:
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    if float(max_value) == 0.0:
        return np.zeros(heatmap.shape, dtype=np.float32)
    return (heatmap / max_value).numpy()


def get_gradcam_heatmap(
    image_batch: tf.Tensor,
    model: tf.keras.Model,
    class_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    backbone = find_backbone(model)
    if backbone is not None:
        return gradcam_for_nested_backbone(image_batch, model, backbone, class_index)
    return gradcam_for_flat_model(image_batch, model, class_index)


def overlay_heatmap(
    image_path: str | Path,
    heatmap: np.ndarray,
    alpha: float = 0.38,
    colormap: int = cv2.COLORMAP_JET,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = np.clip((1 - alpha) * image + alpha * heatmap_color, 0, 255).astype(np.uint8)
    return image, heatmap_resized, overlay


def plot_gradcam_grid(
    model: tf.keras.Model,
    test_df,
    save_path: str | Path,
    image_size: int = 224,
    max_per_class: int = 1,
) -> None:
    selected = []
    for class_id in range(len(CLASSES)):
        class_rows = test_df[test_df["label_id"] == class_id]
        if not class_rows.empty:
            selected.extend(class_rows.head(max_per_class).index.tolist())

    rows = len(selected)
    fig, axes = plt.subplots(rows, 3, figsize=(12, 3.8 * rows))
    if rows == 1:
        axes = np.array([axes])

    for row, index in enumerate(selected):
        record = test_df.loc[index]
        image_batch = preprocess_image(record["path"], image_size)
        heatmap, probabilities = get_gradcam_heatmap(image_batch, model)
        pred_id = int(np.argmax(probabilities))
        confidence = float(probabilities[pred_id])
        original, heatmap_resized, overlay = overlay_heatmap(record["path"], heatmap)

        axes[row, 0].imshow(original)
        axes[row, 0].set_title(f"True: {record['label']}", weight="bold", fontsize=10)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(heatmap_resized, cmap="jet")
        axes[row, 1].set_title("Grad-CAM heatmap", weight="bold", fontsize=10)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(f"Pred: {CLASSES[pred_id]} ({confidence:.1%})", weight="bold", fontsize=10)
        axes[row, 2].axis("off")

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM examples for a trained model.")
    parser.add_argument("--model-path", required=True, help="Path to a trained .keras model.")
    parser.add_argument("--data-dir", default="data/brain_mri", help="Dataset root directory.")
    parser.add_argument("--output-dir", default="results", help="Directory for Grad-CAM figures.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = tf.keras.models.load_model(args.model_path)
    _, _, _, _, _, test_df = get_data_loaders(
        args.data_dir,
        img_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        augment=False,
    )

    output_dir = Path(args.output_dir)
    model_name = Path(args.model_path).stem
    save_path = output_dir / f"{model_name}_gradcam_examples.png"
    plot_gradcam_grid(model, test_df, save_path, image_size=args.image_size)
    if "transfer" in model_name or "efficientnet" in model_name:
        (output_dir / "gradcam_examples.png").write_bytes(save_path.read_bytes())
    print(f"Saved Grad-CAM examples to {save_path}")


if __name__ == "__main__":
    main()
