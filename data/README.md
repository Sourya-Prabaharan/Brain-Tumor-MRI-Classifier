# Data Directory

Place the MRI image dataset here using this structure:

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

The code also supports an optional `val/` split with the same class folders. If
`val/` is not present, `src/dataset.py` creates a stratified validation split
from `train/`.

The dataset itself is not committed because MRI image datasets are usually large
and may have separate licensing requirements. A common source for this project
format is the public four-class brain MRI tumor classification dataset on Kaggle.

For a quick smoke test without downloading external data, run:

```bash
python3 src/dataset.py --generate-demo --train-samples 40 --test-samples 10
```

Synthetic images are only for checking that the pipeline runs. They should not be
reported as medical model performance.

## Preparing the recorded dataset source

Download `masoudnickparvar/brain-tumor-mri-dataset` from Kaggle under its applicable terms. Extract it outside this repository, then copy `Training/` to `data/brain_mri/train/` and `Testing/` to `data/brain_mri/test/`. Rename each `notumor/` class folder to `no_tumor/`. Keep the original test split separate; do not mix it into training. Check actual class counts before comparing with the saved experiment. The original subset manifest is not available, so an exact recreation of its balanced 1,600-image evaluation cannot be guaranteed.
