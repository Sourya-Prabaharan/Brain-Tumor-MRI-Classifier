# Brain Tumor Classification from MRI Scans

This project is a reproducible deep learning pipeline for classifying brain MRI images into four categories:

- Glioma
- Meningioma
- Pituitary tumor
- No tumor

It compares a custom CNN baseline with EfficientNetB0 transfer learning, evaluates the models on a held-out real MRI test set, and includes visual analysis tools such as confusion matrices, sample predictions, misclassification review, and Grad-CAM.

> This is an educational computer vision portfolio project. It is not intended for medical diagnosis, treatment decisions, or clinical use.

## Motivation

Brain MRI classification is a useful machine learning project because the classes are visually meaningful, some categories overlap, and accuracy alone does not tell the full story. The goal of this repository is to show a clean end-to-end workflow that a strong computer science or data science student could realistically build:

- Load and validate a real image dataset.
- Build a reproducible training pipeline.
- Compare a baseline CNN with transfer learning.
- Report precision, recall, F1 score, and confusion matrices.
- Use Grad-CAM and error analysis to inspect model behavior.

## Repository Structure

```text
Brain-Tumor-Detector-and-Classifier/
├── data/
│   └── README.md
├── notebooks/
│   └── 01_dataset_exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── gradcam.py
│   ├── models.py
│   ├── predict.py
│   └── train.py
├── results/
│   ├── architecture_diagram.png
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   ├── experiment_results.csv
│   ├── experiment_results.md
│   ├── gradcam_examples.png
│   ├── misclassifications.png
│   ├── sample_predictions.png
│   └── training_curves.png
├── README.md
├── requirements.txt
└── .gitignore
```

The `data/` directory is intentionally ignored by git so the repository does not commit large medical image files. Trained model checkpoints are also ignored.

## Dataset

The experiments use the public Kaggle dataset `masoudnickparvar/brain-tumor-mri-dataset`, copied locally into this structure:

```text
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
```

The training folder is split into train and validation sets using a stratified split. The held-out test folder is used only for final evaluation.

| Split | Glioma | Meningioma | Pituitary | No Tumor | Total |
|---|---:|---:|---:|---:|---:|
| Train | 1,190 | 1,190 | 1,190 | 1,190 | 4,760 |
| Validation | 210 | 210 | 210 | 210 | 840 |
| Test | 400 | 400 | 400 | 400 | 1,600 |

![Class distribution](/Users/sourya/Brain-Tumor-Detector-and-Classifier/results/class_distribution.png)

## Methodology

The data pipeline in [src/dataset.py](/Users/sourya/Brain-Tumor-Detector-and-Classifier/src/dataset.py) handles folder validation, stratified splitting, image resizing, normalization, batching, and data augmentation.

Training images are resized to the configured input size, normalized to `[0, 1]`, and augmented with lightweight image transformations. Augmentation is used only during training, not validation or testing.

The model code lives in [src/models.py](/Users/sourya/Brain-Tumor-Detector-and-Classifier/src/models.py). Two model families are implemented:

- Baseline custom CNN: convolution blocks with batch normalization, max pooling, dropout, global average pooling, and dense classification layers.
- EfficientNetB0 transfer learning: ImageNet-pretrained EfficientNetB0 backbone with a task-specific classification head and optional fine-tuning.

![Architecture diagram](/Users/sourya/Brain-Tumor-Detector-and-Classifier/results/architecture_diagram.png)

## Training

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train the baseline CNN:

```bash
python src/train.py --model baseline --image-size 160 --epochs 20 --batch-size 32
```

Train EfficientNetB0 transfer learning:

```bash
python src/train.py \
  --model transfer \
  --backbone efficientnet \
  --image-size 160 \
  --epochs 20 \
  --fine-tune-epochs 8 \
  --fine-tune-layers 30 \
  --fine-tune-lr 1e-5 \
  --batch-size 32
```

Run an EfficientNetB0 224px experiment:

```bash
python src/train.py --model transfer --backbone efficientnet --image-size 224 --epochs 20 --batch-size 32
```

Training produces model checkpoints, history CSV files, and training curves in `results/`.

![Training curves](/Users/sourya/Brain-Tumor-Detector-and-Classifier/results/training_curves.png)

## Evaluation

Evaluate a trained model:

```bash
python src/evaluate.py --model-path results/models/transfer_efficientnet.keras --image-size 160
```

The evaluation script in [src/evaluate.py](/Users/sourya/Brain-Tumor-Detector-and-Classifier/src/evaluate.py) reports accuracy, weighted precision, weighted recall, weighted F1 score, classification report, confusion matrix, sample predictions, and misclassified examples.

## Model Comparison

These are measured results on the real held-out Kaggle test split. The best completed run reached 90.31% accuracy, not 96%. The project keeps the reported numbers honest rather than inflating them.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Baseline custom CNN, short CPU checkpoint | 0.2681 | 0.3142 | 0.2681 | 0.1401 |
| EfficientNetB0 transfer learning | 0.9031 | 0.9089 | 0.9031 | 0.9002 |
| EfficientNetB0 224px checkpoint | 0.8988 | 0.9041 | 0.8988 | 0.8960 |

The full table is saved in [results/experiment_results.md](/Users/sourya/Brain-Tumor-Detector-and-Classifier/results/experiment_results.md).

![Confusion matrix](/Users/sourya/Brain-Tumor-Detector-and-Classifier/results/confusion_matrix.png)

## Final Results

The strongest completed model is the EfficientNetB0 transfer-learning run at 160px input resolution.

| Class | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| Glioma | 0.960 | 0.715 | 0.819 | 400 |
| Meningioma | 0.815 | 0.915 | 0.862 | 400 |
| Pituitary | 0.954 | 0.990 | 0.972 | 400 |
| No tumor | 0.906 | 0.993 | 0.947 | 400 |

Pituitary and no-tumor images performed best, while glioma had noticeably lower recall. This is important because a model can have a strong overall accuracy while still missing a clinically important subset of cases.

![Sample predictions](/Users/sourya/Brain-Tumor-Detector-and-Classifier/results/sample_predictions.png)

## Prediction

Run inference on a single MRI image:

```bash
python src/predict.py --image path/to/image.jpg --model path/to/model.keras
```

The script prints the predicted class, confidence score, and probabilities for all four classes.

Example verified output:

```text
Predicted class: pituitary
Confidence: 0.9968
Class probabilities:
  glioma      : 0.0000
  meningioma  : 0.0028
  pituitary   : 0.9968
  no_tumor    : 0.0004
```

The prediction script is implemented in [src/predict.py](/Users/sourya/Brain-Tumor-Detector-and-Classifier/src/predict.py) and also saves a Grad-CAM visualization for the image.

## Grad-CAM

Grad-CAM is implemented in [src/gradcam.py](/Users/sourya/Brain-Tumor-Detector-and-Classifier/src/gradcam.py). It highlights image regions that influenced the predicted class and helps inspect whether the model is focusing on plausible MRI regions instead of scanner artifacts, borders, or labels.

Generate Grad-CAM examples:

```bash
python src/gradcam.py --model-path results/models/transfer_efficientnet.keras --image-size 160
```

Grad-CAM is an interpretability tool, not a medical explanation and not a diagnosis.

![Grad-CAM examples](/Users/sourya/Brain-Tumor-Detector-and-Classifier/results/gradcam_examples.png)

## Error Analysis

The largest source of confusion is glioma versus meningioma. These classes can share visual patterns in 2D MRI slices, and tumors may appear with similar intensity, shape, or surrounding tissue effects. In the best completed run, glioma recall was 0.715, meaning many glioma examples were classified as another tumor class, most often meningioma.

Pituitary tumor and no-tumor examples perform best. Their recalls were 0.990 and 0.993, respectively, suggesting the model learned more separable patterns for those categories in this dataset.

Accuracy alone is not enough for medical imaging because different mistakes have different consequences. A model with high overall accuracy can still underperform on one disease class, produce overconfident wrong predictions, or rely on dataset artifacts. Precision, recall, F1 score, confusion matrices, and visual review give a more complete picture.

Possible dataset limitations include duplicate or near-duplicate images, inconsistent scanner settings, preprocessing artifacts, image-level labels without tumor masks, and a test set that may not represent real clinical deployment conditions.

![Misclassified examples](/Users/sourya/Brain-Tumor-Detector-and-Classifier/results/misclassifications.png)

## Visual Assets

The `results/` directory contains generated assets used in the README:

- [Class distribution chart](/Users/sourya/Brain-Tumor-Detector-and-Classifier/results/class_distribution.png)
- [Architecture diagram](/Users/sourya/Brain-Tumor-Detector-and-Classifier/results/architecture_diagram.png)
- [Training curves](/Users/sourya/Brain-Tumor-Detector-and-Classifier/results/training_curves.png)
- [Confusion matrix](/Users/sourya/Brain-Tumor-Detector-and-Classifier/results/confusion_matrix.png)
- [Sample prediction grid](/Users/sourya/Brain-Tumor-Detector-and-Classifier/results/sample_predictions.png)
- [Grad-CAM examples](/Users/sourya/Brain-Tumor-Detector-and-Classifier/results/gradcam_examples.png)
- [Misclassified examples](/Users/sourya/Brain-Tumor-Detector-and-Classifier/results/misclassifications.png)

## Limitations

- This project uses public image-level MRI data, not hospital-validated clinical data.
- The model does not use tumor segmentation masks or patient history.
- The reported performance is based on this dataset split and should not be generalized to clinical settings.
- The baseline CNN checkpoint included in the comparison was trained briefly in the local environment and is mainly useful as a sanity-check contrast.
- Grad-CAM can suggest where a model is looking, but it cannot prove that the model learned medically correct reasoning.

## Future Improvements

- Train the baseline CNN for a full run on GPU for a fairer comparison.
- Add duplicate-image and near-duplicate detection before training.
- Compare EfficientNetB0 and ResNet50 under identical training budgets.
- Add calibration metrics to measure confidence quality.
- Evaluate robustness across scanner types, image orientations, and external datasets.
- Add segmentation-based analysis if tumor masks become available.

## Summary

- Developed a TensorFlow/Keras deep learning pipeline for four-class brain tumor MRI classification using real Kaggle MRI data.
- Improved performance with EfficientNetB0 transfer learning and image augmentation compared with a custom CNN baseline.
- Evaluated results with accuracy, precision, recall, F1 score, confusion matrices, per-class reports, and error analysis.
- Implemented Grad-CAM visualizations to inspect prediction behavior and model attention.
