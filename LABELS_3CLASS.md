# Three-class TP/SL labels for the 5-minute datasets

`make_labels_3class.py` relabels the 5-minute train/test data with three long-side
outcomes, using the 1-minute OHLCV path. The original feature columns are not
modified. For the full construction pipeline, source verification and validation
details, see `DATASETS.md`.

## Label definitions

For every 5-minute bar `t`, entry is the **open of the first 1-minute bar after
bar `t` closes** (the same fill convention as `bench.py`). The window is `H * 60`
one-minute bars. TP is `entry * (1 + TP%)`, SL is `entry * (1 - SL%)`.

| `lbl` | class | meaning |
|---:|---|---|
| -1 | `hit_sl` | SL touched before TP inside the window |
| 0 | `no_hit` | neither barrier touched inside the window |
| 1 | `hit_tp` | TP touched before SL inside the window |

The direction of the close at the horizon does **not** create extra classes: if
neither barrier is touched, the outcome is `0` regardless of whether the horizon
close is above or below entry.

A same-minute TP/SL tie is resolved with the 1-minute candle path: a gap through
a barrier at the open wins first; otherwise a bullish candle is assumed
`O -> L -> H -> C` (SL first) and a bearish candle `O -> H -> L -> C` (TP first).
There were only 6 same-minute ties in the whole train set across all three
settings, and none in test, so this convention does not materially affect the
distributions.

These are **long-side** labels: TP is above entry and SL is below entry. If you
need a short-side target, build it separately with the barriers mirrored; do not
reuse `hit_tp` / `hit_sl` for both directions.

## Generated datasets (in `data/`)

| risk setting | train | test |
|---|---|---|
| SL 1 %, TP 2 %, horizon 12 h (144 bars) | `symbol5_5m_train_benchmark_1.csv` | `symbol5_5m_test_benchmark_1.csv` |
| SL 2 %, TP 4 %, horizon 24 h (288 bars) | `symbol5_5m_train_benchmark_2.csv` | `symbol5_5m_test_benchmark_2.csv` |
| SL 3 %, TP 6 %, horizon 48 h (576 bars) | `symbol5_5m_train_benchmark_3.csv` | `symbol5_5m_test_benchmark_3.csv` |

Each file is the full 5-minute dataset (all original feature columns) plus exactly
one label column:

| column | meaning |
|---|---|
| `lbl` | class -1, 0 or 1; `NaN` when the window is not fully covered by 1-minute data |

That is **132 columns** in total: `time` + 130 original features + 1 label. The
class names (`hit_sl`, `no_hit`, `hit_tp`) and the label counts are recorded in
the JSON sidecar next to each CSV.

The last `H * 12` bars of each split (144 / 288 / 576 bars) have no complete
one-minute window and therefore `lbl = NaN`. Drop them before training:

```python
import pandas as pd

df = pd.read_csv("data/symbol5_5m_train_benchmark_1.csv", parse_dates=["time"])
train = df[df["lbl"].notna()].copy()
y = train.pop("lbl").astype(int)                 # -1, 0, 1
X = train.drop(columns=[c for c in train.columns if c.startswith("lbl")])
```

The `lbl` column looks into the future: use it only as a target or for analysis,
never as an input feature. The `lbl` prefix is excluded automatically by
`bench.features_only`.

## Label distributions

Train (316,224 rows; percentages of all rows):

| setting | -1 hit_sl | 0 no_hit | 1 hit_tp | unlabeled |
|---|---:|---:|---:|---:|
| 1 % / 2 % / 12 h | 47.11 % | 35.91 % | 16.94 % | 0.05 % |
| 2 % / 4 % / 24 h | 36.35 % | 52.68 % | 10.88 % | 0.09 % |
| 3 % / 6 % / 48 h | 33.92 % | 55.31 % | 10.59 % | 0.18 % |

Test (52,992 rows):

| setting | -1 hit_sl | 0 no_hit | 1 hit_tp | unlabeled |
|---|---:|---:|---:|---:|
| 1 % / 2 % / 12 h | 38.64 % | 48.39 % | 12.70 % | 0.27 % |
| 2 % / 4 % / 24 h | 25.29 % | 67.49 % | 6.68 % | 0.54 % |
| 3 % / 6 % / 48 h | 24.15 % | 67.89 % | 6.88 % | 1.09 % |

The imbalance is expected: a 1:2 risk-reward setting with symmetric barriers in
price space hits the nearer SL more often than the farther TP, and longer
horizons leave more windows where neither barrier is touched. For modelling,
prefer macro-F1, balanced accuracy or class weights over plain accuracy.

## Data source and checks

The provided 1-minute files were verified against Binance `BNBUSDT` 1-minute
klines on both sides of the split. Returns correlate at 0.99999999998 (train)
and 0.9999999951 (test); prices are the Binance price multiplied by a fixed
3.375 and volumes by 295.0738. Both 1-minute files are complete minute grids
with no gaps (`symbol5_1m_train.csv`: 1,624,320 minutes; `symbol5_1m_test.csv`:
264,960 minutes).

Additional checks performed on all six outputs:

- every file has the same rows and index as its source 5-minute features, and
  exactly 132 columns (`time` + 130 features + `lbl`);
- the only `lbl*` column is `lbl`;
- `lbl` is always one of `{-1, 0, 1}` or `NaN`; the JSON sidecar maps each class
  to its name;
- `open/high/low/close/volume` is identical to the canonical 5-minute source on
  every row (max absolute difference `0.0`), and 1-minute OHLC aggregates
  exactly to the 5-minute OHLC — the label path and the features use the same
  relative price scale;
- 100 % of 5-minute closes equal the 1-minute close at the bar's last minute
  (confirms that the time column is the bar open time);
- the only unlabeled rows are the trailing `H * 12` bars.

The OHLCV scale check is reproducible with `python verify_datasets.py`.

## Regenerating

```bash
python make_labels_3class.py --all
```

This relabels all six benchmark datasets in place, writing each file through a
temporary copy and refreshing its JSON sidecar. Or one file at a time:

```bash
python make_labels_3class.py \
  --base data/symbol5_5m_train_benchmark_1.csv \
  --minute data/symbol5_1m_train.csv \
  --sl 1 --tp 2 --horizon 12 \
  --out /tmp/symbol5_5m_train_benchmark_1_3class.csv
```

The script writes a `.json` sidecar next to each output CSV with the risk
setting, label counts and tie statistics. It only needs `numpy` and `pandas`.
