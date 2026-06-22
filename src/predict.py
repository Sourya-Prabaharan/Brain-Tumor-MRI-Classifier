"""Run inference on a single MRI image."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import tensorflow as tf

from dataset import CLASSES
from gradcam import get_gradcam_heatmap, overlay_heatmap, preprocess_image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def predict_single_image(
    image_path: str | Path,
    model_path: str | Path,
    save_path: str | Path = "results/single_prediction.png",
    image_size: int | None = None,
) -> dict:
    model = tf.keras.models.load_model(model_path)
    if image_size is None:
        input_shape = model.input_shape
        image_size = int(input_shape[1]) if input_shape[1] is not None else 224
    image_batch = preprocess_image(image_path, image_size)
    probabilities = model.predict(image_batch, verbose=0)[0]
    pred_id = int(np.argmax(probabilities))
    heatmap, _ = get_gradcam_heatmap(image_batch, model, pred_id)
    original, heatmap_resized, overlay = overlay_heatmap(image_path, heatmap)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].imshow(original)
    axes[0].set_title("Original MRI", weight="bold")
    axes[0].axis("off")
    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title("Grad-CAM heatmap", weight="bold")
    axes[1].axis("off")
    axes[2].imshow(overlay)
    axes[2].set_title(f"{CLASSES[pred_id]} ({probabilities[pred_id]:.1%})", weight="bold")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    result = {
        "image_path": str(image_path),
        "model_path": str(model_path),
        "image_size": int(image_size),
        "predicted_class": CLASSES[pred_id],
        "confidence": float(probabilities[pred_id]),
        "class_probabilities": {label: float(probabilities[index]) for index, label in enumerate(CLASSES)},
        "visualization_path": str(save_path),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict the tumor class for a single MRI image.")
    parser.add_argument("--image", "--image-path", dest="image_path", required=True, help="Path to an MRI image.")
    parser.add_argument("--model", "--model-path", dest="model_path", required=True, help="Path to a trained .keras model.")
    parser.add_argument("--save-path", default="results/single_prediction.png", help="Output visualization path.")
    parser.add_argument("--image-size", type=int, default=None, help="Input size. Defaults to the model input size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict_single_image(args.image_path, args.model_path, args.save_path, args.image_size)
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("Class probabilities:")
    for label, probability in result["class_probabilities"].items():
        print(f"  {label:12s}: {probability:.4f}")
    print(f"Saved visualization to {result['visualization_path']}")


if __name__ == "__main__":
    main()
