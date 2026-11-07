"""Verify that the labeled datasets preserve the source OHLCV and that the
1-minute path used for labels is on the same price scale as the 5-minute data.

Checks, on every row (no sampling):

1. Each labeled CSV has the same timestamps as its source 5-minute CSV.
2. open/high/low/close/volume in each labeled CSV are identical to the source
   5-minute CSV.
3. Aggregating the 1-minute OHLCV to 5-minute bars reproduces the source 5-minute
   open/high/low/close exactly.  This proves the 1-minute labels were built from
   the same scaled price series as the 5-minute features.
4. lbl is always one of {-1, 0, 1}, and only the trailing H * 12 bars are NaN.

Volume is also reported.  The 5-minute volume is a fixed multiple of the
1-minute volume sum; it is not used by the labeler (labels use OHLC only).

Run from the repository root:

    python verify_datasets.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]
SETTINGS = {
    "benchmark_1": ("sl1_tp2_h12", 1.0, 2.0, 12),
    "benchmark_2": ("sl2_tp4_h24", 2.0, 4.0, 24),
    "benchmark_3": ("sl3_tp6_h48", 3.0, 6.0, 48),
}
LABEL_CLASSES = {-1.0, 0.0, 1.0}


def read_ohlcv(path):
    df = pd.read_csv(path, usecols=["time"] + OHLCV, parse_dates=["time"])
    return df.set_index("time").sort_index()


def compare(a, b, cols, label, failures):
    """Compare two aligned frames column by column; return the max absolute diff."""
    out = {}
    for c in cols:
        d = (a[c] - b[c]).abs()
        exact = bool((a[c] == b[c]).all())
        close = bool(np.allclose(a[c].to_numpy(), b[c].to_numpy(),
                                 rtol=1e-12, atol=1e-12, equal_nan=True))
        mx = float(d.max()) if len(d) else 0.0
        out[c] = (exact, close, mx)
        if not close:
            failures.append(f"{label}: {c} differs (max abs {mx:.3e})")
    return out


def fmt(res):
    return "  ".join(
        f"{c}: {'exact' if exact else ('close' if close else 'DIFF')} (max {mx:.1e})"
        for c, (exact, close, mx) in res.items()
    )


def check_split(data_dir, symbol, split, failures):
    base_path = data_dir / f"{symbol}_5m_{split}.csv"
    if not base_path.exists():                     # benchmark_1 is the canonical feature source
        base_path = data_dir / f"{symbol}_5m_{split}_benchmark_1.csv"
    minute_path = data_dir / f"{symbol}_1m_{split}.csv"
    print(f"\n=== {split} ===")
    if not base_path.exists() or not minute_path.exists():
        print(f"SKIP: missing {base_path.name} or {minute_path.name}")
        return

    print(f"source 5m: {base_path.name}")
    base = read_ohlcv(base_path)
    m1 = read_ohlcv(minute_path)
    print(f"5m rows: {len(base):,} | 1m rows: {len(m1):,} | "
          f"5m period: {base.index[0]} -> {base.index[-1]}")

    # 1-minute -> 5-minute aggregation; labels resolve the forward path on these bars.
    agg = m1.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).reindex(base.index)
    missing = int(agg[OHLCV].isna().any(axis=1).sum())
    if missing:
        failures.append(f"{split}: {missing} 5m bars have no 1m aggregation")
    print("1m -> 5m OHLC:")
    print(" ", fmt(compare(base, agg, ["open", "high", "low", "close"], f"{split} 1m->5m", failures)))

    ratio = base["volume"] / agg["volume"]
    print(f"1m -> 5m volume: ratio median {ratio.median():.9f}, "
          f"min {ratio.min():.9f}, max {ratio.max():.9f}, std {ratio.std():.3e} "
          f"(constant normalization; volume is not used for labels)")

    for benchmark, (setting, sl, tp, hours) in SETTINGS.items():
        path = data_dir / f"{symbol}_5m_{split}_{benchmark}.csv"
        if not path.exists():
            failures.append(f"{split}/{benchmark}: missing {path.name}")
            continue
        lab = read_ohlcv(path)
        if not lab.index.equals(base.index):
            failures.append(f"{split}/{benchmark}: timestamps differ from base")
            continue
        print(f"labeled {benchmark} ({setting}) vs source 5m:")
        print(" ", fmt(compare(lab, base, OHLCV, f"{split}/{benchmark}", failures)))

        labels = pd.read_csv(path, usecols=["time", "lbl"], parse_dates=["time"])
        labels = labels.set_index("time").sort_index()
        if not labels.index.equals(base.index):
            failures.append(f"{split}/{benchmark}: label timestamps differ from base")
            continue
        lbl = labels["lbl"]
        bad = lbl.dropna()
        invalid = int((~bad.isin(LABEL_CLASSES)).sum())
        if invalid:
            failures.append(f"{split}/{benchmark}: {invalid} labels outside {{-1, 0, 1}}")
        tail = int(hours * 12)
        if lbl.iloc[-tail:].notna().any() or lbl.iloc[:-tail].isna().any():
            failures.append(f"{split}/{benchmark}: unlabeled rows are not exactly the trailing {tail} bars")
        counts = lbl.value_counts(dropna=False)
        summary = " ".join(f"{k:g}: {v:,}" for k, v in counts.items())
        print(f"  lbl {summary} | tail NaN {int(lbl.iloc[-tail:].isna().sum()):,}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--symbol", default="symbol5")
    a = p.parse_args()

    failures = []
    for split in ("train", "test"):
        check_split(Path(a.data_dir), a.symbol, split, failures)

    print()
    if failures:
        print("FAIL")
        for f in failures:
            print(" -", f)
        return 1
    print("PASS: labeled OHLCV is identical to the source 5m data on every row, "
          "and 1m OHLC aggregates exactly to the 5m OHLC.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
