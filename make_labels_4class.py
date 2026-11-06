"""Four-class TP/SL labels for 5-minute bars, resolved on 1-minute OHLCV.

The labels describe a long entry made at the open of the first 1-minute bar after
the 5-minute bar closes.  For a risk setting SL%, TP%, H hours, every base bar gets
exactly one of four labels:

   -1  true_sl      the SL barrier (-SL%) is touched before the TP barrier (+TP%)
   -2  horizon_sl   neither barrier is touched and close at the horizon is <= entry
    1  true_tp      the TP barrier (+TP%) is touched before the SL barrier (-SL%)
    2  horizon_tp   neither barrier is touched and close at the horizon is > entry

The window is H * 60 one-minute bars starting at the entry minute.  The close of the
last bar in that window is the price H hours after entry.  A same-minute tie between
TP and SL is resolved with the 1-minute candle path: a gap through a barrier at the
open wins first; otherwise a bullish candle is O->L->H->C (SL first) and a bearish
candle is O->H->L->C (TP first) - the same convention as make_labels.py.

Rows whose full window is not covered by the 1-minute file (or has missing minutes)
get lbl = NaN, so they can be dropped safely.

Run from the repository root:

    python make_labels_4class.py --all
    python make_labels_4class.py --base data/symbol5_5m_train.csv \
        --minute data/symbol5_1m_train.csv --sl 1 --tp 2 --horizon 12

The output is the full base dataset with only one label column appended - lbl
(classes -1, -2, 1, 2) - plus a small JSON sidecar with the risk setting and label
counts.  The original files are never touched.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

TIME_COLS = ("time", "datetime", "date", "timestamp", "open_time")
MINUTE = pd.Timedelta("1min")
LABEL_NAMES = {-1: "true_sl", -2: "horizon_sl", 1: "true_tp", 2: "horizon_tp"}
SETTINGS = (
    # name,           SL %, TP %, horizon (hours)
    ("sl1_tp2_h12",   1.0,  2.0,  12),
    ("sl2_tp4_h24",   2.0,  4.0,  24),
    ("sl3_tp6_h48",   3.0,  6.0,  48),
)


def parse_time(values):
    s = pd.Series(values)
    if pd.api.types.is_numeric_dtype(s):                       # epoch: s / ms / us
        unit = "us" if s.iloc[0] > 1e15 else "ms" if s.iloc[0] > 1e11 else "s"
        return pd.DatetimeIndex(pd.to_datetime(s, unit=unit, utc=True))
    return pd.DatetimeIndex(pd.to_datetime(s, utc=True, format="ISO8601"))


def read_base(path):
    """Base OHLCV csv; the time column becomes a UTC DatetimeIndex named 'time'."""
    df = pd.read_csv(path)
    time_col = next((c for c in df.columns if c.lower() in TIME_COLS), df.columns[0])
    df.index = parse_time(df.pop(time_col))
    df.index.name = "time"
    return df.sort_index()


def read_minutes(path):
    """1m OHLC on a complete minute grid; missing minutes become NaN."""
    m = pd.read_csv(path)
    time_col = next((c for c in m.columns if c.lower() in TIME_COLS), m.columns[0])
    m.index = parse_time(m.pop(time_col))
    m = m[~m.index.duplicated()].sort_index()
    m = m.reindex(pd.date_range(m.index[0], m.index[-1], freq="1min"))
    return m[["open", "high", "low", "close"]]


def check_time_convention(base, m1):
    """Share of base closes equal to the 1m close of the bar's last minute.

    ~1.0 confirms that the base time column is the bar OPEN time, which is the
    convention this labeler (and bench.py) expects.
    """
    close = next((c for c in base.columns if c.lower() == "close"), None)
    if close is None:
        return None
    tf = base.index.to_series().diff().median()
    last_min = m1["close"].reindex(base.index + tf - MINUTE).to_numpy()
    ok = np.isfinite(last_min)
    return float(np.mean(np.abs(base[close].to_numpy()[ok] / last_min[ok] - 1) < 1e-4)) if ok.any() else None


def _first(mask, none_value):
    """Index of the first True per row, none_value when a row has no True."""
    idx = mask.argmax(1)
    return np.where(mask[np.arange(len(mask)), idx], idx, none_value)


def label_path(times, tf, fine, horizon_minutes, sl_pct, tp_pct, chunk_cells=2_000_000):
    """Label base bars `times` (bar open times) from the 1m path `fine`.

    `fine` is indexed on a regular grid of resolution res (normally 1min) and has
    open/high/low/close columns.  Returns the label frame and an info dict.
    """
    res = fine.index.to_series().diff().median()
    W = int(round(horizon_minutes / (res / MINUTE)))            # window length in fine bars
    n = len(times)
    pad = np.full(W, np.nan)
    o, h, l, c = (np.concatenate([fine[k].to_numpy(dtype=float), pad])
                  for k in ("open", "high", "low", "close"))
    c_ff = pd.Series(c).ffill().to_numpy()
    hw, lw = sliding_window_view(h, W), sliding_window_view(l, W)

    start = np.floor((times + tf - fine.index[0]) / res).astype(int).to_numpy()
    valid = np.flatnonzero((start >= 0) & (start < len(fine)))

    # Missing minutes inside each window (interior NaNs and the cut at the data end).
    nan_any = np.isnan(o) | np.isnan(h) | np.isnan(l) | np.isnan(c)
    nan_cum = np.concatenate([[0], np.cumsum(nan_any)])
    missing = np.full(n, float(horizon_minutes))
    missing[valid] = (nan_cum[start[valid] + W] - nan_cum[start[valid]]) * (res / MINUTE)
    complete = valid[(missing[valid] == 0) & (start[valid] + W <= len(fine))]

    lbl = np.full(n, np.nan)
    ties = 0

    chunk_rows = max(1, int(chunk_cells // W))
    for j in range(0, len(complete), chunk_rows):
        i = complete[j: j + chunk_rows]
        s = start[i]
        e = o[s]                                               # entry = open of first minute
        ok = np.isfinite(e) & (e > 0)
        if not ok.all():
            i, s, e = i[ok], s[ok], e[ok]
            if not len(i):
                continue

        with np.errstate(invalid="ignore", divide="ignore"):
            up = np.round((hw[s] / e[:, None] - 1) * 100, 9)    # rounded: exact 1.00% counts
            dn = np.round((lw[s] / e[:, None] - 1) * 100, 9)
            tp_first = _first(up >= tp_pct, W)
            sl_first = _first(dn <= -sl_pct, W)
            ret = np.round((c_ff[s + W - 1] / e - 1) * 100, 9)

            # Resolve same-minute ties with the intra-candle path.
            tie = (tp_first == sl_first) & (tp_first < W)
            first_is_tp = tp_first < sl_first
            if tie.any():
                ti = tp_first[tie]
                ti_idx = s[tie] + ti
                o_t, c_t = o[ti_idx], c[ti_idx]
                tp_price = e[tie] * (1 + tp_pct / 100)
                sl_price = e[tie] * (1 - sl_pct / 100)
                bull = c_t >= o_t
                first_is_tp[tie] = np.where(o_t >= tp_price, True,
                                            np.where(o_t <= sl_price, False, ~bull))
                ties += int(tie.sum())

            no_touch = (tp_first == W) & (sl_first == W)
            lbl[i] = np.where(no_touch, np.where(ret > 0, 2, -2),
                              np.where(first_is_tp, 1, -1)).astype(float)

    labels = pd.DataFrame({"lbl": lbl}, index=times)
    labels["lbl"] = labels["lbl"].astype("Int64")
    return labels, {"window_min": int(horizon_minutes), "window_bars_1m": W,
                    "same_minute_ties": ties}


def make_labels(base, m1, sl_pct, tp_pct, horizon_hours):
    """Attach four-class labels to every row of `base`; incomplete windows stay NaN."""
    tf = base.index.to_series().diff().median()
    horizon_minutes = int(round(horizon_hours * 60))
    labels, info = label_path(base.index, tf, m1, horizon_minutes, sl_pct, tp_pct)
    info.update({"timeframe": str(tf), "horizon_hours": horizon_hours,
                 "sl_pct": sl_pct, "tp_pct": tp_pct})
    return labels, info


def write_one(base, m1, out_path, sl_pct, tp_pct, horizon_hours, name, match, split):
    labels, info = make_labels(base, m1, sl_pct, tp_pct, horizon_hours)
    for col in labels.columns:
        base[col] = labels[col]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(out_path, na_rep="")
    base.drop(columns=list(labels.columns), inplace=True)

    counts = labels["lbl"].map(LABEL_NAMES).value_counts(dropna=False).to_dict()
    meta = {
        "symbol": base.attrs.get("symbol", "symbol5"),
        "setting": name,
        "timeframe": "5m",
        "split": split,
        "risk": {
            "sl_pct": sl_pct,
            "tp_pct": tp_pct,
            "horizon_hours": horizon_hours,
            "horizon_bars_5m": int(horizon_hours * 12),
            "horizon_minutes": int(horizon_hours * 60),
        },
        "labels": LABEL_NAMES,
        "label_columns": ["lbl"],
        "flat_horizon_rule": "close == entry -> horizon_sl (-2)",
        "tie_break": ("same 1m candle: open beyond a barrier wins; else bullish O->L->H->C "
                      "(SL first) and bearish O->H->L->C (TP first)"),
        "time_convention_check": match,
        "rows": int(len(labels)),
        "labeled_rows": int(labels["lbl"].notna().sum()),
        "unlabeled_rows": int(labels["lbl"].isna().sum()),
        "label_counts": {str(k): int(v) for k, v in counts.items()},
        "label_pct": {str(k): round(v / len(labels), 6) for k, v in counts.items()},
        "same_minute_ties": info["same_minute_ties"],
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"{out_path.name}: rows {len(labels):,} | labeled {meta['labeled_rows']:,} | "
          f"ties {info['same_minute_ties']:,} | " +
          " ".join(f"{k}={v:,}" for k, v in meta["label_counts"].items()))
    return meta


def build_all(data_dir="data", symbol="symbol5"):
    data_dir = Path(data_dir)
    for split in ("train", "test"):
        base_path = data_dir / f"{symbol}_5m_{split}.csv"
        minute_path = data_dir / f"{symbol}_1m_{split}.csv"
        if not base_path.exists() or not minute_path.exists():
            print(f"skip {split}: missing {base_path.name} or {minute_path.name}")
            continue
        print(f"\n[{split}] reading {base_path.name} and {minute_path.name} ...", flush=True)
        base = read_base(base_path)
        base.attrs["symbol"] = symbol
        m1 = read_minutes(minute_path)
        match = check_time_convention(base, m1)
        print(f"[{split}] base/1m close match {match:.1%}" if match is not None else "")
        for name, sl, tp, hours in SETTINGS:
            out = data_dir / f"{symbol}_5m_{split}_{name}.csv"
            write_one(base, m1, out, sl, tp, hours, name, match, split)
        del base, m1
        gc.collect()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--all", action="store_true", help="build all six 5m train/test datasets")
    p.add_argument("--data-dir", default="data", help="folder containing the csv files")
    p.add_argument("--symbol", default="symbol5", help="file prefix, default symbol5")
    p.add_argument("--base", help="one 5m csv to label (instead of --all)")
    p.add_argument("--minute", help="matching 1m csv")
    p.add_argument("--sl", type=float, help="stop-loss %%")
    p.add_argument("--tp", type=float, help="take-profit %%")
    p.add_argument("--horizon", type=float, help="horizon in hours")
    p.add_argument("--out", help="output csv")
    a = p.parse_args()

    if a.all or not a.base:
        build_all(a.data_dir, a.symbol)
        return
    if not (a.minute and a.sl is not None and a.tp is not None and a.horizon is not None):
        p.error("--base needs --minute, --sl, --tp and --horizon (or use --all)")
    base = read_base(a.base)
    m1 = read_minutes(a.minute)
    match = check_time_convention(base, m1)
    if match is not None and match < 0.9:
        print(f"WARNING: only {match:.0%} of base closes match the 1m close at bar end; "
              f"is the time column the bar OPEN time?")
    out = Path(a.out or Path(a.base).with_name(
        f"{Path(a.base).stem}_sl{a.sl:g}_tp{a.tp:g}_h{a.horizon:g}.csv"))
    write_one(base, m1, out, a.sl, a.tp, a.horizon, out.stem, match, Path(a.base).stem)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
