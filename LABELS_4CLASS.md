# Four-class TP/SL labels for the 5-minute datasets

`make_labels_4class.py` relabels the 5-minute train/test data with four long-side
outcomes, using the 1-minute OHLCV path. The original files are not modified.
For the full construction pipeline, source verification and validation details,
see `DATASETS.md`.

## Label definitions

For every 5-minute bar `t`, entry is the **open of the first 1-minute bar after
bar `t` closes** (the same fill convention as `bench.py`). The window is `H * 60`
one-minute bars. TP is `entry * (1 + TP%)`, SL is `entry * (1 - SL%)`.

| `lbl` | class | meaning |
|---:|---|---|
| -1 | `true_sl` | SL touched before TP inside the window |
| -2 | `horizon_sl` | neither barrier touched; close at the horizon is at or below entry |
| 1 | `true_tp` | TP touched before SL inside the window |
| 2 | `horizon_tp` | neither barrier touched; close at the horizon is above entry |

A same-minute TP/SL tie is resolved with the 1-minute candle path: a gap through
a barrier at the open wins first; otherwise a bullish candle is assumed
`O -> L -> H -> C` (SL first) and a bearish candle `O -> H -> L -> C` (TP first).
The flat case (`close == entry`) is label -2. There were only 6 same-minute ties
in the whole train set across all three settings, so this convention does not
materially affect the distributions.

These are **long-side** labels: TP is above entry and SL is below entry. If you
need a short-side target, build it separately with the barriers mirrored; do not
reuse `true_tp` / `true_sl` for both directions.

## Generated datasets (in `data/`)

| risk setting | train | test |
|---|---|---|
| SL 1 %, TP 2 %, horizon 12 h (144 bars) | `symbol5_5m_train_sl1_tp2_h12.csv` | `symbol5_5m_test_sl1_tp2_h12.csv` |
| SL 2 %, TP 4 %, horizon 24 h (288 bars) | `symbol5_5m_train_sl2_tp4_h24.csv` | `symbol5_5m_test_sl2_tp4_h24.csv` |
| SL 3 %, TP 6 %, horizon 48 h (576 bars) | `symbol5_5m_train_sl3_tp6_h48.csv` | `symbol5_5m_test_sl3_tp6_h48.csv` |

Each file is the full 5-minute dataset (all original feature columns) plus exactly
one label column:

| column | meaning |
|---|---|
| `lbl` | class -2, -1, 1, 2; `NaN` when the window is not fully covered by 1-minute data |

That is **132 columns** in total: `time` + 130 original features + 1 label. The
class names (`true_sl`, `horizon_sl`, `true_tp`, `horizon_tp`) are recorded in the
JSON sidecar.

The last `H * 12` bars of each split (144 / 288 / 576 bars) have no complete
one-minute window and therefore `lbl = NaN`. Drop them before training:

```python
import pandas as pd

df = pd.read_csv("data/symbol5_5m_train_sl1_tp2_h12.csv", parse_dates=["time"])
train = df[df["lbl"].notna()].copy()
y = train.pop("lbl").astype(int)                 # -1, -2, 1, 2
X = train.drop(columns=[c for c in train.columns if c.startswith("lbl")])
```

The `lbl` column looks into the future: use it only as a target or for analysis,
never as an input feature. The `lbl` prefix is excluded automatically by
`bench.features_only`.

## Label distributions

Train (316,224 rows; percentages of all rows):

| setting | -1 true_sl | -2 horizon_sl | 1 true_tp | 2 horizon_tp | unlabeled |
|---|---:|---:|---:|---:|---:|
| 1 % / 2 % / 12 h | 47.11 % | 11.12 % | 16.94 % | 24.79 % | 0.05 % |
| 2 % / 4 % / 24 h | 36.35 % | 17.83 % | 10.88 % | 34.86 % | 0.09 % |
| 3 % / 6 % / 48 h | 33.92 % | 18.74 % | 10.59 % | 36.56 % | 0.18 % |

Test (52,992 rows):

| setting | -1 true_sl | -2 horizon_sl | 1 true_tp | 2 horizon_tp | unlabeled |
|---|---:|---:|---:|---:|---:|
| 1 % / 2 % / 12 h | 38.64 % | 15.68 % | 12.70 % | 32.71 % | 0.27 % |
| 2 % / 4 % / 24 h | 25.29 % | 23.64 % | 6.68 % | 43.85 % | 0.54 % |
| 3 % / 6 % / 48 h | 24.15 % | 24.46 % | 6.88 % | 43.43 % | 1.09 % |

The class imbalance is expected: a 1:2 risk-reward setting with symmetric
barriers in price space hits the nearer SL more often, and longer horizons leave
more windows with no barrier touch.

## Data source and checks

The provided 1-minute files were verified against Binance `BNBUSDT` 1-minute
klines on both sides of the split. Returns correlate at 0.99999999998 (train)
and 0.9999999951 (test); prices are the Binance price multiplied by a fixed
3.375 and volumes by 295.0738. Both 1-minute files are complete minute grids
with no gaps (`symbol5_1m_train.csv`: 1,624,320 minutes; `symbol5_1m_test.csv`:
264,960 minutes).

Additional checks performed on all six outputs:

- every file has the same rows and index as its source 5-minute file, and exactly
  132 columns (`time` + 130 features + `lbl`);
- the only `lbl*` column is `lbl`;
- `lbl` is always one of `{-2, -1, 1, 2}` or `NaN`; the JSON sidecar maps each
  class to its name;
- `open/high/low/close/volume` in every labeled file are identical to the source
  5-minute file on every row (max absolute difference `0.0`), and 1-minute OHLC
  aggregates exactly to the 5-minute OHLC — the label path and the features use
  the same relative price scale;
- 100 % of 5-minute closes equal the 1-minute close at the bar's last minute
  (confirms that the time column is the bar open time);
- 100 random rows per file match an independent brute-force row-by-row labeler
  that recomputes entry, barriers and the horizon close from the 1-minute file;
- the only unlabeled rows are the trailing `H * 12` bars.

The OHLCV scale check is reproducible with `python verify_datasets.py`.

## Regenerating

```bash
python make_labels_4class.py --all
```

Or one file at a time:

```bash
python make_labels_4class.py \
  --base data/symbol5_5m_train.csv \
  --minute data/symbol5_1m_train.csv \
  --sl 1 --tp 2 --horizon 12 \
  --out data/symbol5_5m_train_sl1_tp2_h12.csv
```

The script writes a `.json` sidecar next to each CSV with the risk setting,
label counts and tie statistics. It only needs `numpy` and `pandas`.
