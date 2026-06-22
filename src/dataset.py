"""Dataset utilities for four-class brain MRI classification.

The expected dataset layout is:

data/brain_mri/
    train/
        glioma/
        meningioma/
        pituitary/
        no_tumor/
    test/
        glioma/
        meningioma/
        pituitary/
        no_tumor/

If a validation folder is not supplied, the training folder is split
stratified into train and validation subsets.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASSES)}
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")


def find_image_files(folder: str | Path) -> list[str]:
    """Return sorted image paths under a class folder."""
    folder = Path(folder)
    files: list[str] = []
    for pattern in IMAGE_EXTENSIONS:
        files.extend(glob.glob(str(folder / pattern)))
    return sorted(files)


def scan_split(split_dir: str | Path) -> pd.DataFrame:
    """Scan a split directory and return a dataframe of paths and labels."""
    split_dir = Path(split_dir)
    rows = []
    for label in CLASSES:
        class_dir = split_dir / label
        if not class_dir.exists():
            continue
        for path in find_image_files(class_dir):
            rows.append({"path": path, "label": label, "label_id": CLASS_TO_INDEX[label]})
    return pd.DataFrame(rows)


def validate_dataset(data_dir: str | Path, require_test: bool = True) -> None:
    """Raise a helpful error if the dataset is missing or empty."""
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    missing = []
    for split_dir in [train_dir, test_dir] if require_test else [train_dir]:
        for label in CLASSES:
            class_dir = split_dir / label
            if not class_dir.exists():
                missing.append(str(class_dir))

    if missing:
        missing_text = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Dataset folders are missing. Expected four class folders for each split:\n"
            f"  - {missing_text}\n\n"
            "Place MRI images under data/brain_mri/train and data/brain_mri/test, "
            "or run `python src/dataset.py --generate-demo` for a small synthetic smoke-test dataset."
        )

    train_df = scan_split(train_dir)
    test_df = scan_split(test_dir) if require_test else pd.DataFrame()
    if train_df.empty or (require_test and test_df.empty):
        raise ValueError(
            f"No images found in {data_dir}. Add MRI images or generate a demo dataset with "
            "`python src/dataset.py --generate-demo`."
        )


def build_metadata(
    data_dir: str | Path,
    validation_split: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train, validation, and test metadata dataframes."""
    validate_dataset(data_dir)
    data_dir = Path(data_dir)

    train_full = scan_split(data_dir / "train")
    provided_val = data_dir / "val"
    if provided_val.exists():
        train_df = train_full.reset_index(drop=True)
        val_df = scan_split(provided_val).reset_index(drop=True)
    else:
        train_df, val_df = train_test_split(
            train_full,
            test_size=validation_split,
            random_state=seed,
            stratify=train_full["label_id"],
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

    test_df = scan_split(data_dir / "test").reset_index(drop=True)
    return train_df, val_df, test_df


def summarize_class_distribution(*splits: tuple[str, pd.DataFrame]) -> pd.DataFrame:
    """Return a tidy class-count table for one or more splits."""
    rows = []
    for split_name, df in splits:
        counts = df["label"].value_counts().reindex(CLASSES, fill_value=0)
        for label, count in counts.items():
            rows.append({"split": split_name, "class": label, "count": int(count)})
    return pd.DataFrame(rows)


def plot_class_distribution(
    distribution: pd.DataFrame,
    save_path: str | Path = "results/class_distribution.png",
) -> None:
    """Save a grouped bar chart showing class balance by split."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    pivot = distribution.pivot(index="class", columns="split", values="count").reindex(CLASSES)
    ax = pivot.plot(kind="bar", figsize=(9, 5), width=0.78, color=["#4c78a8", "#f58518", "#54a24b"])
    ax.set_title("MRI Dataset Class Distribution", fontsize=14, weight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of images")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(title="Split")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def make_tf_dataset(
    df: pd.DataFrame,
    img_size: tuple[int, int],
    batch_size: int,
    training: bool,
    augment: bool,
):
    """Create a tf.data pipeline from a metadata dataframe."""
    import tensorflow as tf

    paths = df["path"].to_numpy()
    labels = df["label_id"].to_numpy(dtype=np.int32)

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    def augment_image(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.08)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.12)
        image = tf.image.random_saturation(image, lower=0.95, upper=1.05)
        return tf.clip_by_value(image, 0.0, 1.0), label

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(df), seed=42, reshuffle_each_iteration=True)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training and augment:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def get_data_loaders(
    data_dir: str | Path,
    img_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.15,
    seed: int = 42,
    augment: bool = True,
):
    """Return train, validation, and test tf.data datasets plus metadata."""
    train_df, val_df, test_df = build_metadata(data_dir, validation_split, seed)
    distribution = summarize_class_distribution(("train", train_df), ("validation", val_df), ("test", test_df))
    plot_class_distribution(distribution)

    print("Dataset split summary")
    print(distribution.pivot(index="class", columns="split", values="count").reindex(CLASSES).fillna(0).astype(int))

    train_ds = make_tf_dataset(train_df, img_size, batch_size, training=True, augment=augment)
    val_ds = make_tf_dataset(val_df, img_size, batch_size, training=False, augment=False)
    test_ds = make_tf_dataset(test_df, img_size, batch_size, training=False, augment=False)
    return train_ds, val_ds, test_ds, train_df, val_df, test_df


def generate_synthetic_mri(label: str, img_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Generate a simple synthetic MRI-like image for pipeline smoke tests only."""
    height, width = img_size
    image = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    axes = (int(width * 0.38), int(height * 0.44))

    cv2.ellipse(image, center, axes, 0, 0, 360, 42, -1)
    cv2.ellipse(image, center, axes, 0, 0, 360, 95, 3)
    cv2.ellipse(image, (center[0] - 14, center[1]), (7, 25), -12, 0, 360, 18, -1)
    cv2.ellipse(image, (center[0] + 14, center[1]), (7, 25), 12, 0, 360, 18, -1)

    if label == "glioma":
        tumor_center = (center[0] - int(width * 0.12), center[1] - int(height * 0.14))
        points = []
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            radius = np.random.randint(17, 31)
            points.append([int(tumor_center[0] + radius * np.cos(angle)), int(tumor_center[1] + radius * np.sin(angle))])
        cv2.fillPoly(image, [np.array(points, dtype=np.int32)], 155)
        image = cv2.GaussianBlur(image, (5, 5), 0)
    elif label == "meningioma":
        tumor_center = (center[0] + int(width * 0.23), center[1] - int(height * 0.18))
        radius = np.random.randint(14, 23)
        cv2.circle(image, tumor_center, radius, 220, -1)
        cv2.circle(image, tumor_center, radius + 8, 82, 2)
    elif label == "pituitary":
        tumor_center = (center[0], center[1] + int(height * 0.24))
        radius = np.random.randint(13, 20)
        cv2.circle(image, tumor_center, radius, 205, -1)
        cv2.circle(image, tumor_center, max(4, radius // 3), 120, -1)

    image_rgb = cv2.merge([image, image, image])
    noise = np.random.normal(0, 8, image_rgb.shape).astype(np.float32)
    image_rgb = np.clip(image_rgb.astype(np.float32) + noise, 0, 255)
    image_rgb *= np.random.uniform(0.86, 1.14)
    return np.clip(image_rgb, 0, 255).astype(np.uint8)


def generate_synthetic_dataset(
    base_dir: str | Path = "data/brain_mri",
    train_samples_per_class: int = 120,
    test_samples_per_class: int = 30,
    seed: int = 42,
) -> None:
    """Generate a small synthetic dataset for validating the project workflow."""
    np.random.seed(seed)
    base_dir = Path(base_dir)
    for split, samples in [("train", train_samples_per_class), ("test", test_samples_per_class)]:
        for label in CLASSES:
            class_dir = base_dir / split / label
            class_dir.mkdir(parents=True, exist_ok=True)
            for index in range(samples):
                image = generate_synthetic_mri(label)
                cv2.imwrite(str(class_dir / f"{label}_{split}_{index:04d}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and inspect the brain MRI dataset.")
    parser.add_argument("--data-dir", default="data/brain_mri", help="Dataset root directory.")
    parser.add_argument("--generate-demo", action="store_true", help="Generate synthetic images for pipeline smoke tests.")
    parser.add_argument("--train-samples", type=int, default=120, help="Synthetic training samples per class.")
    parser.add_argument("--test-samples", type=int, default=30, help="Synthetic test samples per class.")
    args = parser.parse_args()

    if args.generate_demo:
        generate_synthetic_dataset(args.data_dir, args.train_samples, args.test_samples)

    train_df, val_df, test_df = build_metadata(args.data_dir)
    distribution = summarize_class_distribution(("train", train_df), ("validation", val_df), ("test", test_df))
    Path("results").mkdir(exist_ok=True)
    distribution.to_csv("results/class_distribution.csv", index=False)
    plot_class_distribution(distribution)
    print(distribution)


if __name__ == "__main__":
    _main()
