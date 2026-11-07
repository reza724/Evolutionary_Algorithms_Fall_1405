# Benchmark 1 — Three-Class Classification

Train a model on the provided data and submit it as a model package.
The organizers run your package on a hidden test set and rank submissions by **Macro-F1**.

## What is in this folder

| Path | Contents |
|---|---|
| `data/train_benchmark_1.csv` | Training data. `label` is the target; every other column is an input. |
| `docs/LABELS.md` | What each label value means, and how to avoid leakage. |
| `docs/MODEL_PACKAGE.md` | How to package and submit your model. |
| `ea_sdk/` | Helper code: model interface, metrics, and the local replay tool. |
| `examples/constant/` | A minimal example package that always predicts `0`. |
| `schemas/manifest.schema.json` | JSON schema for `manifest.json`. |
| `requirements.txt` | Python packages needed by `ea_sdk`. |

## Task

Each row has a label describing the outcome of a hypothetical long entry at that row.

| Label | Meaning |
|---:|---|
| -1 | The stop level is reached first. |
| 0 | Neither level is reached within the evaluation window. |
| 1 | The target level is reached first. |

This is a classification task. For every prediction request, return exactly one class: `-1`, `0`, or `1`.
You may use any ML, DL, or RL approach, but the final output must be one of these three classes.

## Training data

- `label` is the target column. All other columns are inputs.
- Rows are in chronological order. Keep that order.
- Some feature values are missing. Handle them the same way in training and inference.
- The last rows have no label (`NaN`). Drop them from training and validation.
- Never use `label` as a model input.

## Quick start

```bash
pip install -r requirements.txt
```

```python
import pandas as pd

df = pd.read_csv("data/train_benchmark_1.csv")
df = df[df["label"].notna()]               # drop rows without a label

X = df.drop(columns=["label"])
y = df["label"].astype(int)

# Chronological split with a purge gap (no shuffling)
n = len(df)
val_start = int(n * 0.8)
purge = 600
X_train, y_train = X.iloc[:val_start - purge], y.iloc[:val_start - purge]
X_val, y_val = X.iloc[val_start:], y.iloc[val_start:]
```

Test a packaged model locally, the same way the organizers run it:

```bash
python ea_sdk/replay.py --package examples/constant --data data/train_benchmark_1.csv
```

The tool replays the last 20% of the training file row by row (after a 600-row purge gap)
and prints Macro-F1, balanced accuracy, MCC, and the confusion matrix.
Note: it runs your code directly; it is not a sandbox.

## Validation

- Use a chronological split: train on earlier rows, validate on later rows. Never split randomly.
- Leave a purge gap of **at least several hundred rows** between the two sections, and use the gap for neither.
- Fit all learned preprocessing (imputers, scalers, feature selection, dimensionality reduction) on the training section only.
- For stateful models or rolling features, process rows in order and never look ahead.

## Good practice

- Do not build features from `label`, including shifted, rolling, or aggregated versions of past labels.
- Do not compute features with centered windows, negative shifts, or any statistic over the whole dataset; every feature at a row must use only that row and earlier rows.
- Fit scalers, imputers, encoders, and feature selection on the training section only.
- Do not tune hyperparameters on the validation section repeatedly without a separate final check; overfitting to validation inflates your score.
- Handle class imbalance deliberately, for example with class weights, resampling on training data only, or threshold tuning.
- Report per-class precision, recall, and F1, not just one number.
- Compare your model against simple baselines, such as always predicting one class.
- Keep the same preprocessing at inference that you used in training, and save it in `artifacts/`.
- Make results reproducible: fix random seeds and pin package versions.
- Test your package locally with `ea_sdk/replay.py` before submitting.

## Evaluation

The organizers run your package on a hidden test set.

| Metric | Role |
|---|---|
| Macro-F1 over `[-1, 0, 1]` | **Primary score** (each class weighted equally; undefined F1 counts as 0) |
| Balanced accuracy | Secondary |
| Matthews correlation coefficient (MCC) | Secondary, used to break ties |
| Confusion matrix, ordered `[-1, 0, 1]` | Reported |

The classes are imbalanced, so accuracy alone is misleading.
Always predicting one class gives a low Macro-F1, because the other two classes get F1 = 0.

## Submission

Submit a model package (a ZIP), not a prediction file. The organizers generate predictions by running it.

- [Label definitions](docs/LABELS.md)
- [Model package specification](docs/MODEL_PACKAGE.md)
