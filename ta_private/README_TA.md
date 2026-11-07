# TA Guide — Benchmark 1 Private Grading

**Do not share this folder with students.** It contains the hidden test set and labels.

## Folder layout

| Path | Contents |
|---|---|
| `data/test_benchmark_1.csv` | Full test set with `label` (OHLCV removed, same columns as student train). |
| `data/test_benchmark_1_features.csv` | Test set without `label`; this is what models see. |
| `data/test_benchmark_1_labels.csv` | `time,label` only; used for scoring. |
| `grader/run_submission.py` | Unzips and runs one submission on the test set. |
| `grader/evaluate.py` | Scores one run and writes `report.json`. |
| `grader/leaderboard.py` | Builds `leaderboard.csv` from all reports. |
| `submissions/` | Put student ZIPs here, named `<student_id>.zip`. |
| `runs/<student_id>/` | `predictions.csv`, `log.txt`, `status.json`, `report.json`. |

Grader code reuses `../student_release/ea_sdk` (manifest check and metrics), so TA and student scores use identical logic.
Source: the original project file `data/symbol5_5m_test_benchmark_1.csv` with `open,high,low,close,volume` dropped and `lbl` renamed to `label`.

## Grading one submission

Run from `ta_private/` with the project venv:

```bash
cd ~/Evolutionary_Algorithms_Fall_1405/ta_private
PY=../.venv/bin/python
$PY grader/run_submission.py submissions/<student_id>.zip      # optional: --timeout-min 30
$PY grader/evaluate.py <student_id>
$PY grader/leaderboard.py
```

Install the student's `requirements.txt` first if it needs extra packages (ideally in a separate venv).

## Grading all submissions

```bash
for z in submissions/*.zip; do
  id=$(basename "$z" .zip)
  $PY grader/run_submission.py "$z"
  $PY grader/evaluate.py "$id"
done
$PY grader/leaderboard.py
```

## How the runner works

1. Rejects ZIPs over 200MB (compressed or uncompressed), with absolute paths, `..`, or symlinks.
2. Extracts to a temp dir and validates `manifest.json` against the student schema.
3. Starts a **separate Python process** that loads `Model`, calls `reset(seed=0)`.
4. Warm-up: replays the last 288 rows of the student train features (no label) through `update`; not scored.
5. Calls `update(window, now)` for every test row in order: `window` = last 288 feature rows up to and including the current one, `now` = current row's `time`.
6. Every output must be an `int` in `{-1, 0, 1}` (not `bool`). Any exception, invalid output, or timeout → `status: failed`.

## Outputs

- `runs/<id>/status.json` — `ok` or `failed` with the error.
- `runs/<id>/predictions.csv` — `time,y_pred` (only when ok).
- `runs/<id>/report.json` — `macro_f1` (primary), `balanced_accuracy`, `mcc`, `confusion_matrix` ordered `[-1,0,1]`, `per_class`.
  Evaluation requires the exact set of test times, no duplicates; rows with `NaN` label are not scored.
- `leaderboard.csv` — sorted by `macro_f1` desc, ties by `mcc`; failed submissions at the bottom without a rank.

## Limits

- Total time: 30 minutes per submission (`--timeout-min` to change).
- ZIP size: 200MB.

## Security / honor code

The runner is a separate process, **not a sandbox**: student code can read files and use the network as your user.
Skim each `model.py` before running it (look for file reads outside the package, network calls, subprocesses).
A later improvement is running each submission in Docker with `--network none` and a read-only mount.
Keep this folder and the original project files (label scripts, test CSVs, metadata JSON) away from students.
