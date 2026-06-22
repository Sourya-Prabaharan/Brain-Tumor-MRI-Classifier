# Results Directory

This directory stores generated artifacts from the real Kaggle MRI dataset experiments.

Key assets:

- `class_distribution.png`
- `architecture_diagram.png`
- `training_curves.png`
- `confusion_matrix.png`
- `sample_predictions.png`
- `misclassifications.png`
- `gradcam_examples.png`
- `experiment_results.csv`
- `experiment_results.md`

The best completed held-out test result is `90.31%` accuracy from the EfficientNetB0 transfer-learning model. The project does not claim 96% accuracy because that was not measured on the real test set.

Model checkpoints are ignored by git because they are large generated files.

To refresh these artifacts:

```bash
python src/train.py --model transfer --backbone efficientnet --image-size 160 --epochs 20 --fine-tune-epochs 8 --fine-tune-layers 30 --fine-tune-lr 1e-5
python src/evaluate.py --model-path results/models/transfer_efficientnet.keras --image-size 160
python src/gradcam.py --model-path results/models/transfer_efficientnet.keras --image-size 160
```
