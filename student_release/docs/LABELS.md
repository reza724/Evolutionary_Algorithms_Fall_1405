# Label Definitions

## Target column

The target column is `label`. It has three classes:

| Value | Outcome |
|---:|---|
| -1 | The stop level is reached before the target level. |
| 0 | Neither level is reached within the evaluation window. |
| 1 | The target level is reached before the stop level. |

Each label describes a hypothetical long entry at its row.
The target level is above the entry; the stop level is below it.

`0` is a real class. It does **not** mean "missing".

## Missing labels

The last rows have `label = NaN` because their outcome is not available.

- Drop these rows from training and validation.
- Do not fill them with `0`, forward-fill them, or invent labels for them.

## Preventing leakage

- `label` is the target. Never use it as an input, for the current row or for earlier rows.
- Do not use features built from labels.
- Do not use any information that would not be available at prediction time.
- Validate chronologically, with a purge gap of at least several hundred rows.
- Fit all learned preprocessing on the training section only.

## Class imbalance

The classes are imbalanced. Look at per-class precision, recall, and F1, not only accuracy.

The primary score is Macro-F1 over `[-1, 0, 1]`, so every class counts equally.
