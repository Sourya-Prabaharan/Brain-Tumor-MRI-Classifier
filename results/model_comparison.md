| Model | Accuracy | Precision | Recall | F1 | Notes |
|---|---:|---:|---:|---:|---|
| Baseline custom CNN (short CPU checkpoint) | 0.2681 | 0.3142 | 0.2681 | 0.1401 | Custom CNN baseline checkpoint evaluated after a brief local training run. |
| EfficientNetB0 transfer learning | 0.9031 | 0.9089 | 0.9031 | 0.9002 | Best completed real-data model at 160px with augmentation and fine-tuning. |
| EfficientNetB0 224px checkpoint | 0.8988 | 0.9041 | 0.8988 | 0.8960 | Best frozen-backbone checkpoint from the 224px experiment. |
