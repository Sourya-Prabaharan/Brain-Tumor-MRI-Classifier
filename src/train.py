"""Train baseline and transfer-learning brain MRI classifiers."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf

from dataset import CLASSES, get_data_loaders
from models import build_baseline_cnn, build_transfer_learning_model, unfreeze_top_layers

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def plot_history(history_df: pd.DataFrame, save_path: str | Path) -> None:
    """Save training and validation accuracy/loss curves."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(history_df["epoch"], history_df["accuracy"], label="Training", linewidth=2)
    axes[0].plot(history_df["epoch"], history_df["val_accuracy"], label="Validation", linewidth=2)
    axes[0].set_title("Accuracy", weight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["loss"], label="Training", linewidth=2)
    axes[1].plot(history_df["epoch"], history_df["val_loss"], label="Validation", linewidth=2)
    axes[1].set_title("Loss", weight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend()

    fig.suptitle("Training Curves", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def build_model(args: argparse.Namespace) -> tf.keras.Model:
    input_shape = (args.image_size, args.image_size, 3)
    if args.resume_from:
        return tf.keras.models.load_model(args.resume_from)
    if args.model == "baseline":
        return build_baseline_cnn(input_shape=input_shape, num_classes=len(CLASSES))
    return build_transfer_learning_model(
        input_shape=input_shape,
        num_classes=len(CLASSES),
        backbone=args.backbone,
        train_backbone=False,
    )


def train(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, _, train_df, val_df, test_df = get_data_loaders(
        args.data_dir,
        img_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        seed=args.seed,
        augment=not args.no_augmentation,
    )

    model = build_model(args)
    compile_model(model, args.learning_rate)
    model.summary()

    model_name = f"{args.model}_{args.backbone}" if args.model == "transfer" else "baseline_cnn"
    checkpoint_path = model_dir / f"{model_name}.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, args.patience // 2),
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))

    if args.model == "transfer" and args.fine_tune_epochs > 0:
        print(f"Fine-tuning top {args.fine_tune_layers} backbone layers...")
        model = unfreeze_top_layers(model, args.fine_tune_layers)
        compile_model(model, args.fine_tune_lr)
        fine_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            callbacks=callbacks,
        )
        fine_df = pd.DataFrame(fine_history.history)
        fine_df.insert(0, "epoch", np.arange(len(history_df) + 1, len(history_df) + len(fine_df) + 1))
        history_df = pd.concat([history_df, fine_df], ignore_index=True)

    model.save(checkpoint_path)
    history_path = output_dir / f"{model_name}_history.csv"
    history_df.to_csv(history_path, index=False)
    plot_path = output_dir / f"{model_name}_training_curves.png"
    plot_history(history_df, plot_path)

    if args.model == "transfer":
        plot_history(history_df, output_dir / "training_curves.png")

    metadata = {
        "model": args.model,
        "backbone": args.backbone if args.model == "transfer" else None,
        "classes": CLASSES,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs_trained": int(history_df["epoch"].max()),
        "train_images": int(len(train_df)),
        "validation_images": int(len(val_df)),
        "test_images": int(len(test_df)),
        "model_path": str(checkpoint_path),
        "history_path": str(history_path),
    }
    with open(output_dir / f"{model_name}_training_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Saved model to {checkpoint_path}")
    print(f"Saved training curves to {plot_path}")
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train brain tumor classification models.")
    parser.add_argument("--data-dir", default="data/brain_mri", help="Dataset root directory.")
    parser.add_argument("--output-dir", default="results", help="Directory for models, plots, and logs.")
    parser.add_argument("--model", choices=["baseline", "transfer"], default="transfer", help="Architecture to train.")
    parser.add_argument("--backbone", choices=["efficientnet", "efficientnetb0", "resnet50"], default="efficientnet")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--fine-tune-epochs", type=int, default=0)
    parser.add_argument("--fine-tune-layers", type=int, default=30)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-5)
    parser.add_argument("--resume-from", default=None, help="Path to an existing .keras model to continue training.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augmentation", action="store_true", help="Disable training-time augmentation.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
