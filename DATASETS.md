# DATASETS.md — how the 4-class 5-minute datasets were created

This document describes the full construction pipeline for the six labeled
5-minute datasets in `data/`. The generator is `make_labels_4class.py`; a shorter
label-only reference is in `LABELS_4CLASS.md`.

## 1. Summary

- **Symbol:** Binance `BNBUSDT`, anonymized and rescaled in the course data as
  `symbol5`.
- **Base timeframe:** 5-minute OHLCV with 130 feature columns.
- **Path resolution:** 1-minute OHLCV, used for entry fills, barrier touches,
  the horizon close and excursion statistics.
- **Target:** one integer class per 5-minute bar, `lbl ∈ {-2, -1, 1, 2}`.
- **Stored label column:** only `lbl` (the class-name mapping is in the JSON sidecar).
- **Risk settings:** three SL/TP/horizon combinations, each built for both train
  and test — **6 datasets total**.
- **Original files are not modified.** All labeled datasets are new files with a
  suffix describing the risk setting.

## 2. Input files

| file | role | rows | period (UTC) |
|---|---|---:|---|
| `data/symbol5_5m_train.csv` | base train bars + features | 316,224 | 2023-05-01 00:00 → 2026-05-02 23:55 |
| `data/symbol5_5m_test.csv` | base test bars + features | 52,992 | 2026-05-03 00:00 → 2026-11-02 23:55 |
| `data/symbol5_1m_train.csv` | 1-minute path for train labels | 1,624,320 | 2023-04-01 00:00 → 2026-05-02 23:59 |
| `data/symbol5_1m_test.csv` | 1-minute path for test labels | 264,960 | 2026-05-03 00:00 → 2026-11-02 23:59 |

Both 1-minute files are complete minute grids: every consecutive timestamp is
exactly one minute apart, with no duplicates and no missing minutes. The extra
month at the start of the train 1-minute file (April 2023) is not needed for
forward labels; the first 5-minute train bar is May 2023.

### Price and volume scaling

The course data is the real BNB price multiplied by a fixed constant. The labeler
only uses ratios (`high / entry`, `low / entry`, `close / entry`) and percentages,
so the scale cancels out. Nothing in the label construction depends on the dollar
price level.

## 3. Binance source verification

The provided 1-minute files were checked against Binance `BNBUSDT` 1-minute
klines on both sides of the split:

| check | train sample | test sample |
|---|---|---|
| sample window | 2024-01-01 00:00 UTC, 1,000 minutes | 2026-05-03 00:00 UTC, 1,000 minutes |
| return correlation with Binance | 0.99999999998 | 0.99999999509 |
| median price ratio (Binance / file) | 3.375 | 3.375 |
| price-ratio standard deviation | 1.04 × 10⁻⁸ | 5.41 × 10⁻⁸ |
| median volume ratio (Binance / file) | 295.0738 | 295.0738 |

In other words, the 1-minute files are Binance `BNBUSDT` data with price scaled
by `1 / 3.375` and volume scaled by `1 / 295.0738`. The 5-minute files are on the
same price scale and were also confirmed to be exact aggregations of the
1-minute bars at sampled timestamps (open, high, low and close all match).

Because the provided 1-minute files already cover the full train and test
periods, no additional Binance download was needed. A fresh download would
produce the same labels after applying the same scale.

## 4. Time convention

The `time` column is the **bar open time** in UTC, matching `bench.py` and the
course data convention.

For a 5-minute bar with open time `t`:

- the bar closes at `t + 5 minutes`;
- a decision made after that bar closes is filled at the **open of the first
  1-minute bar after the close**, i.e. at `entry_time = t + 5 minutes`;
- the entry price is `entry = open(entry_time)` from the 1-minute file.

This is the same fill convention used by `bench.py` and `make_labels.py`. The
check `5-minute close(t) == 1-minute close(t + 4 minutes)` passed for **100 %**
of rows in both train and test, which confirms the open-time convention.

## 5. Label construction

For each 5-minute bar `t`, the labeler looks forward over a window of
`W = 60 × H` one-minute bars, starting at `entry_time`.

### 5.1 Barriers

```
TP price = entry × (1 + TP% / 100)
SL price = entry × (1 - SL% / 100)
```

Let:

- `i_TP` = first minute index `k ∈ [0, W-1]` with `high(entry_time + k) ≥ TP price`;
- `i_SL` = first minute index `k ∈ [0, W-1]` with `low(entry_time + k) ≤ SL price`.

If a barrier is never touched, its index is treated as `W`.

### 5.2 The four classes

| `lbl` | class | rule |
|---:|---|---|
| -1 | `true_sl` | `i_SL < i_TP` — SL touched first |
| -2 | `horizon_sl` | neither barrier touched and `close(entry_time + W - 1) ≤ entry` |
| 1 | `true_tp` | `i_TP < i_SL` — TP touched first |
| 2 | `horizon_tp` | neither barrier touched and `close(entry_time + W - 1) > entry` |

`close(entry_time + W - 1)` is the close of the last 1-minute bar in the window.
Since that bar closes at `entry_time + W`, it is the price exactly `H` hours
after entry.

The flat case (`close == entry`, or a return that rounds to zero at 9 decimals)
is intentionally assigned to `horizon_sl` (-2): it is not a profit.

### 5.3 Same-minute ties

TP and SL can both be touched inside the same 1-minute candle. The tie is
resolved with the assumed intra-candle path:

1. If the candle **opens at or above TP**, TP was touched first → label 1.
2. Else if the candle **opens at or below SL**, SL was touched first → label -1.
3. Else if the candle is bullish (`close ≥ open`), assume `O → L → H → C`, so
   the low (SL) was touched first → label -1.
4. Else the candle is bearish (`close < open`), assume `O → H → L → C`, so the
   high (TP) was touched first → label 1.

This is the same path convention as `make_labels.py`. Ties are rare: 5, 1 and 0
in the three train settings (6 total) and 0 in test.

### 5.4 Floating-point handling

Barrier touches are checked on percentage returns rounded to 9 decimal places:

```
round((high / entry - 1) × 100, 9) ≥ TP%
round((low  / entry - 1) × 100, 9) ≤ -SL%
```

This makes an exact `1.00 %` or `2.00 %` touch count as a touch instead of
depending on floating-point noise. The horizon return used for the `2` versus
`-2` decision is rounded the same way, so a return that rounds to `0.0` is
treated as flat.

### 5.5 Stored label column

Only one label column is written to each CSV:

| column | meaning |
|---|---|
| `lbl` | class -2, -1, 1, 2, or `NaN` for an incomplete window |

The class names (`true_sl`, `horizon_sl`, `true_tp`, `horizon_tp`) are recorded in
the JSON sidecar. The entry price, horizon return, MFE/MAE, touch times and
missing-minute count are computed internally to decide the class, but they are
**not** stored in the final datasets.

## 6. Risk settings

| setting | SL | TP | horizon | horizon in 5-minute bars | horizon in 1-minute bars |
|---|---:|---:|---:|---:|---:|
| `sl1_tp2_h12` | 1 % | 2 % | 12 h | 144 | 720 |
| `sl2_tp4_h24` | 2 % | 4 % | 24 h | 288 | 1,440 |
| `sl3_tp6_h48` | 3 % | 6 % | 48 h | 576 | 2,880 |

All three use a 1:2 risk-reward ratio: TP is twice SL.

## 7. Output files and schema

### 7.1 Naming

```
data/symbol5_5m_{split}_{setting}.csv
data/symbol5_5m_{split}_{setting}.json
```

`split` is `train` or `test`. `setting` is `sl1_tp2_h12`, `sl2_tp4_h24` or
`sl3_tp6_h48`.

Example:

```
data/symbol5_5m_train_sl1_tp2_h12.csv
data/symbol5_5m_train_sl1_tp2_h12.json
data/symbol5_5m_test_sl3_tp6_h48.csv
data/symbol5_5m_test_sl3_tp6_h48.json
```

### 7.2 Columns

Each CSV contains:

- the original 5-minute dataset: `time` + **130 feature columns**;
- the **label column** `lbl`.

Total: **132 columns**, 316,224 rows for train and 52,992 rows for test.
Row order and timestamps are identical to the original 5-minute file.

The `.json` sidecar records:

- symbol, split, setting and timeframe;
- SL / TP / horizon;
- label names and counts;
- the flat-horizon and tie-break rules;
- the time-convention check result;
- the number of same-minute ties.

### 7.3 Unlabeled tail

A label is produced only when the full 1-minute window fits inside the split's
1-minute file and contains no missing minutes. Otherwise `lbl` is `NaN`. The
number of trailing unlabeled bars is exactly `H × 12`:

| setting | horizon | unlabeled tail (5-minute bars) |
|---|---:|---:|
| `sl1_tp2_h12` | 12 h | 144 |
| `sl2_tp4_h24` | 24 h | 288 |
| `sl3_tp6_h48` | 48 h | 576 |

The train and test splits are labeled independently. The last train bars are
**not** labeled using the test 1-minute file, even though the files are
contiguous, to avoid crossing the train/test boundary.

## 8. Class distributions

### Train (316,224 rows)

| setting | -1 true_sl | -2 horizon_sl | 1 true_tp | 2 horizon_tp | unlabeled |
|---|---:|---:|---:|---:|---:|
| 1 % / 2 % / 12 h | 148,958 (47.11 %) | 35,161 (11.12 %) | 53,568 (16.94 %) | 78,393 (24.79 %) | 144 (0.05 %) |
| 2 % / 4 % / 24 h | 114,942 (36.35 %) | 56,368 (17.83 %) | 34,404 (10.88 %) | 110,222 (34.86 %) | 288 (0.09 %) |
| 3 % / 6 % / 48 h | 107,268 (33.92 %) | 59,276 (18.74 %) | 33,483 (10.59 %) | 115,621 (36.56 %) | 576 (0.18 %) |

### Test (52,992 rows)

| setting | -1 true_sl | -2 horizon_sl | 1 true_tp | 2 horizon_tp | unlabeled |
|---|---:|---:|---:|---:|---:|
| 1 % / 2 % / 12 h | 20,475 (38.64 %) | 8,310 (15.68 %) | 6,729 (12.70 %) | 17,334 (32.71 %) | 144 (0.27 %) |
| 2 % / 4 % / 24 h | 13,401 (25.29 %) | 12,527 (23.64 %) | 3,538 (6.68 %) | 23,238 (43.85 %) | 288 (0.54 %) |
| 3 % / 6 % / 48 h | 12,795 (24.15 %) | 12,963 (24.46 %) | 3,644 (6.88 %) | 23,014 (43.43 %) | 576 (1.09 %) |

The imbalance is expected, not a bug:

- with a 1:2 risk-reward ratio, the nearer SL is hit more often than the farther
  TP;
- longer horizons leave more windows with no barrier touch, so the two horizon
  classes grow and the true-touch classes shrink as a share of all rows.

For modelling, prefer macro-F1, balanced accuracy or class weights over plain
accuracy.

## 9. Validation performed

The following checks were run on all six outputs:

1. **Row/index integrity:** each output has exactly the same rows, order and
   timestamps as its source 5-minute file.
2. **Column check:** each output has exactly 132 columns, and the only `lbl*`
   column is `lbl`.
3. **Time grid:** every consecutive 5-minute timestamp differs by exactly
   `00:05:00`.
4. **Label domain:** `lbl` is always one of `{-2, -1, 1, 2}` or `NaN`; the JSON
   sidecar maps each class to its name.
5. **Tail check:** the only unlabeled rows are the trailing `H × 12` bars.
6. **Brute-force check:** 100 random rows per file were re-labeled with an
   independent row-by-row Python loop that recomputes entry, barriers and the
   horizon close from the 1-minute file. All matched.
7. **Source check:** Binance return correlation, price scale, volume scale and
   complete 1-minute grids, as described in section 3.
8. **Time convention:** 100 % of 5-minute closes equal the 1-minute close at the
   bar's last minute, confirming the open-time convention.

### OHLCV consistency check

`verify_datasets.py` compares **every row** (no sampling) of every labeled file
against the source 5-minute file, and compares the source 5-minute OHLC against
the 1-minute OHLC aggregated to 5-minute bars.

Result for both train and test:

| check | result |
|---|---|
| labeled `open/high/low/close/volume` vs source 5-minute values | identical, max absolute difference `0.0` |
| 1-minute OHLC aggregated to 5-minute vs source 5-minute OHLC | identical, max absolute difference `0.0` |
| 1-minute volume sum vs 5-minute volume | fixed ratio `0.200117985`; volume is not used by the labeler |

This is the important scale check: the labeled dataset prices are exactly the
original relative/scaled 5-minute prices, and the 1-minute path used for the
labels is on that same scale. The labeler only uses percentage moves
(`high / entry`, `low / entry`, `close / entry`), so the common scale cancels
out. No raw Binance dollar prices were mixed into the labels.

The volume ratio is a constant normalization between the two files; it varies by
only about `3 × 10⁻⁹` and has no effect on TP/SL labels because labels use OHLC
only.

Run the check with:

```bash
python verify_datasets.py
```

## 10. How to use the datasets

```python
import pandas as pd

df = pd.read_csv(
    "data/symbol5_5m_train_sl1_tp2_h12.csv",
    parse_dates=["time"],
)

# Drop rows whose forward window is incomplete.
df = df[df["lbl"].notna()].copy()

y = df.pop("lbl").astype(int)          # -1, -2, 1, 2
X = df.drop(columns=[c for c in df.columns if c.startswith("lbl")])
```

Important:

- **The label column looks into the future.** Use `lbl` only as a target or for
  analysis, never as a model input. The `lbl` prefix is deliberately excluded by
  `bench.features_only`.
- **Do not split randomly.** Adjacent 5-minute bars have overlapping forward
  windows. For time-series validation, leave a gap of at least `H × 12` bars
  between the training and validation parts.
- **These are long-side labels.** TP is above entry and SL is below entry. For a
  short-side target, mirror the barriers and build a separate label column; do
  not reuse `true_tp` / `true_sl` for both directions.
- **Only the class is stored.** Entry price, horizon return, MFE/MAE and touch
  times are computed internally but not written. Recompute them from the
  1-minute data if you need them for analysis.
- **The datasets are standalone.** `bench.py` loads the original course files
  (`data/symbol5_5m_{split}.csv.gz` in the standard layout), not these labeled
  copies; these files are meant for custom data loaders. `bench.LabelModel`
  expects the older 0–5 six-class scheme, so adapt the target if you use it with
  these files.

## 11. Reproduction

From the repository root:

```bash
python make_labels_4class.py --all
```

This builds all six datasets and their JSON sidecars. It takes about 2 minutes
and needs roughly 1.4 GB of free disk space for the six CSVs. Dependencies are
only `numpy` and `pandas`.

To build one file:

```bash
python make_labels_4class.py \
  --base data/symbol5_5m_train.csv \
  --minute data/symbol5_1m_train.csv \
  --sl 1 --tp 2 --horizon 12 \
  --out data/symbol5_5m_train_sl1_tp2_h12.csv
```

### Related files

| file | purpose |
|---|---|
| `make_labels_4class.py` | generator for the four-class datasets |
| `verify_datasets.py` | all-row OHLCV / 1m-to-5m scale verification |
| `LABELS_4CLASS.md` | shorter label-only reference |
| `make_labels.py` | original six-class TP/SL labeler used by `bench.py` |
| `bench.py` | trading environment, model contract and scoring |
