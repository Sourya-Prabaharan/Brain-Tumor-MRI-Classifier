"""Evaluate trained brain MRI classifiers and create result visualizations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from dataset import CLASSES, get_data_loaders

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def model_stem(model_path: str | Path) -> str:
    return Path(model_path).stem.replace("_model", "")


def predict_dataset(model: tf.keras.Model, dataset) -> np.ndarray:
    probabilities = model.predict(dataset, verbose=1)
    return np.asarray(probabilities)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
    }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str | Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 6.4))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(CLASSES)), labels=CLASSES, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(CLASSES)), labels=CLASSES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix", weight="bold")

    threshold = cm.max() / 2 if cm.max() else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
                color="white" if cm[row, col] > threshold else "black",
                weight="bold",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def read_rgb(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def plot_sample_predictions(
    test_df: pd.DataFrame,
    probabilities: np.ndarray,
    save_path: str | Path,
    max_examples: int = 6,
) -> None:
    rng = np.random.default_rng(42)
    count = min(max_examples, len(test_df))
    indices = rng.choice(len(test_df), size=count, replace=False)

    fig, axes = plt.subplots(count, 2, figsize=(11, 3.2 * count))
    if count == 1:
        axes = np.array([axes])

    for row, index in enumerate(indices):
        image = read_rgb(test_df.iloc[index]["path"])
        true_id = int(test_df.iloc[index]["label_id"])
        probs = probabilities[index]
        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])

        axes[row, 0].imshow(image)
        axes[row, 0].axis("off")
        color = "#1b7f3a" if pred_id == true_id else "#b42318"
        axes[row, 0].set_title(
            f"True: {CLASSES[true_id]} | Pred: {CLASSES[pred_id]} ({confidence:.1%})",
            color=color,
            weight="bold",
            fontsize=10,
        )

        bar_colors = ["#b7c0c7"] * len(CLASSES)
        bar_colors[pred_id] = "#4c78a8"
        if pred_id != true_id:
            bar_colors[true_id] = "#54a24b"
        axes[row, 1].barh(CLASSES, probs, color=bar_colors)
        axes[row, 1].set_xlim(0, 1)
        axes[row, 1].invert_yaxis()
        axes[row, 1].set_xlabel("Confidence")
        axes[row, 1].grid(axis="x", linestyle="--", alpha=0.35)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_misclassified_examples(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    save_path: str | Path,
    max_examples: int = 8,
) -> None:
    y_true = test_df["label_id"].to_numpy()
    misses = np.where(y_true != y_pred)[0]
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if len(misses) == 0:
        fig, ax = plt.subplots(figsize=(7, 2.5))
        ax.text(0.5, 0.5, "No misclassified test examples in this run.", ha="center", va="center", fontsize=13)
        ax.axis("off")
        plt.savefig(save_path, dpi=300)
        plt.close()
        return

    selected = misses[:max_examples]
    cols = min(4, len(selected))
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for axis, index in zip(axes, selected):
        image = read_rgb(test_df.iloc[index]["path"])
        true_id = int(y_true[index])
        pred_id = int(y_pred[index])
        axis.imshow(image)
        axis.axis("off")
        axis.set_title(
            f"True: {CLASSES[true_id]}\nPred: {CLASSES[pred_id]} ({probabilities[index][pred_id]:.1%})",
            color="#b42318",
            fontsize=10,
            weight="bold",
        )

    for axis in axes[len(selected) :]:
        axis.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def evaluate_model(args: argparse.Namespace, model_path: str | Path, test_ds, test_df: pd.DataFrame) -> dict:
    model_path = Path(model_path)
    print(f"Evaluating {model_path}")
    model = tf.keras.models.load_model(model_path)
    probabilities = predict_dataset(model, test_ds)
    y_true = test_df["label_id"].to_numpy(dtype=np.int32)
    y_pred = np.argmax(probabilities, axis=1)

    metrics = compute_metrics(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    name = model_stem(model_path)

    result = {
        "model": name,
        "model_path": str(model_path),
        **metrics,
        "classification_report": report,
    }

    output_dir = Path(args.output_dir)
    with open(output_dir / f"{name}_evaluation_metrics.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    pd.DataFrame(report).transpose().to_csv(output_dir / f"{name}_classification_report.csv")
    plot_confusion_matrix(y_true, y_pred, output_dir / f"{name}_confusion_matrix.png")
    plot_sample_predictions(test_df, probabilities, output_dir / f"{name}_sample_predictions.png")
    plot_misclassified_examples(test_df, y_pred, probabilities, output_dir / f"{name}_misclassifications.png")

    return result


def save_comparison(results: list[dict], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    rows = [
        {
            "model": result["model"],
            "accuracy": result["accuracy"],
            "precision_weighted": result["precision_weighted"],
            "recall_weighted": result["recall_weighted"],
            "f1_weighted": result["f1_weighted"],
        }
        for result in results
    ]
    comparison = pd.DataFrame(rows).sort_values("f1_weighted", ascending=False)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    with open(output_dir / "model_comparison.md", "w", encoding="utf-8") as file:
        file.write("| model | accuracy | precision_weighted | recall_weighted | f1_weighted |\n")
        file.write("|---|---:|---:|---:|---:|\n")
        for _, row in comparison.iterrows():
            file.write(
                f"| {row['model']} | {row['accuracy']:.4f} | {row['precision_weighted']:.4f} | "
                f"{row['recall_weighted']:.4f} | {row['f1_weighted']:.4f} |\n"
            )

    best_model = comparison.iloc[0]["model"]
    for suffix in ["confusion_matrix", "sample_predictions", "misclassifications"]:
        source = output_dir / f"{best_model}_{suffix}.png"
        target = output_dir / f"{suffix}.png"
        if source.exists():
            target.write_bytes(source.read_bytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained brain tumor classifiers.")
    parser.add_argument("--model-path", nargs="+", required=True, help="One or more .keras model paths.")
    parser.add_argument("--data-dir", default="data/brain_mri", help="Dataset root directory.")
    parser.add_argument("--output-dir", default="results", help="Directory for metrics and plots.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    _, _, test_ds, _, _, test_df = get_data_loaders(
        args.data_dir,
        img_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        seed=args.seed,
        augment=False,
    )
    results = [evaluate_model(args, path, test_ds, test_df) for path in args.model_path]
    save_comparison(results, args.output_dir)

    print("\nModel comparison")
    print(pd.read_csv(Path(args.output_dir) / "model_comparison.csv").to_string(index=False))


if __name__ == "__main__":
    main()
