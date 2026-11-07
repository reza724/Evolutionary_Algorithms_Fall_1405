# Model Package Specification

## Package layout

Submit **one ZIP file** with these entries at its root (no extra top-level folder):

| Entry | Purpose |
|---|---|
| `manifest.json` | Package metadata. |
| `model.py` | Defines the `Model` class. |
| `artifacts/` | Trained weights and fitted preprocessing. |
| `requirements.txt` | Python packages your model needs, with pinned versions. |

Do not include datasets, credentials, unrelated files, or precomputed predictions.
See `examples/constant/` for a working minimal package.

Example (run inside your package folder):

```bash
zip -r ../my_submission.zip manifest.json model.py artifacts requirements.txt
```

## Manifest

```json
{
  "schema_version": "1.0",
  "benchmark_id": "benchmark_1",
  "entrypoint": "model:Model",
  "artifacts_dir": "artifacts",
  "requirements_file": "requirements.txt",
  "labels": [-1, 0, 1]
}
```

Paths are relative to the package root. The schema is in `schemas/manifest.schema.json`.

## Model interface

```python
class Model:
    def __init__(self, artifacts_dir):
        """Load trained weights and fitted preprocessing from artifacts_dir."""

    def reset(self, seed):
        """Clear inference state and seed any randomness."""

    def update(self, window, now) -> int:
        """Return exactly one of -1, 0, or 1."""
```

### `__init__(artifacts_dir)`

`artifacts_dir` is the path to your packaged `artifacts/` folder.
Load everything you need from there. Do not download anything or read files outside the package.

### `reset(seed)`

Called once before an evaluation run. Clear buffers, counters, and other running state, and seed random generators.
Keep the loaded model and preprocessing.

### `update(window, now)`

Called once per row, in chronological order.

| Argument | Contents |
|---|---|
| `window` | A pandas `DataFrame` with the most recent rows, up to and including the current row (at most 288 rows). Same columns as the training CSV, **without** `label`. |
| `now` | The `time` value of the current row. |

- `window` never contains future rows or labels.
- Before the first scored row, the runner sends some earlier rows as warm-up context; their predictions are not scored.
- Do not assume the full history is available; keep your own state if you need longer history.
- Return a plain Python `int` in `{-1, 0, 1}`. Not a probability, string, `bool`, `NaN`, or list.

## Rules for causal inference

Every prediction may use only your packaged assets, your own internal state, and rows already passed to `update`.

Do not:

- read future rows;
- use hidden labels;
- fit preprocessing on the whole evaluation sequence;
- replace the model with a lookup table of answers;
- access data outside your package and the runner inputs.

## Execution environment

- No network access.
- Time, memory, and package-size limits apply; the organizers will announce them.
- Dependencies must install without downloads at run time or interactive prompts.
- An exception, timeout, or invalid prediction fails the submission.

Test loading, `reset`, and sequential `update` calls with `ea_sdk/replay.py` before you submit.

## Scoring

The runner produces the predictions; you do not submit a prediction file.

Primary score: Macro-F1 over `[-1, 0, 1]` (undefined F1 counts as 0).
Also reported: balanced accuracy, MCC, and the confusion matrix.

## Honor code

- Use only the provided training data and explicitly permitted resources.
- Do not try to identify or reconstruct the hidden test data or labels, hardcode answers, or bypass execution limits.
- Submissions may be audited. You must be able to explain your model, preprocessing, training, and artifacts.
