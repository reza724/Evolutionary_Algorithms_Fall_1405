"""Path-dependent TP/SL labels for a base OHLCV dataset of any timeframe, resolved on 1-minute bars.

Optional: supervised models get them inside fit() with bench.labels(train, horizon, levels). As a script:
    python make_labels.py --base data/symbol5_5m_train.csv.gz --minute data/symbol5_1m_train.csv.gz --horizon 12 --levels 1 2

Risk setting: horizon H (in base records) and levels L1 < L2 < ... < LK (in %).
For every base bar t:
  entry   = open of the first minute after bar t closes (the earliest real fill).
  window  = the next H base bars, i.e. H * timeframe minutes.
  1. Direction: whichever of +L1 / -L1 is touched first. -L1 is the SL of a long, +L1 the SL of a short.
  2. Magnitude: how many levels were reached in that direction before the opposite L1 (the stop) is touched.
  3. Neither L1 touched inside the window: the sign of the return at the window end.
  Two barriers inside one 1m candle: bullish candle assumed O->L->H->C, bearish O->H->L->C.
Every row gets a label:
  - last bars of the data: the window is cut at the data end (lbl_missing_min > 0).
  - bars without 1m coverage: the same rule on the base bars' own OHLC (lbl_source = "base", coarser).
Classes 0..2K+1 run from most negative to most positive. For levels 1 2:
  0 dn>2 | 1 dn1-2 | 2 dn0-1 | 3 up0-1 | 4 up1-2 | 5 up>2

Base time must be the bar OPEN time (Binance convention); the script checks this against the 1m closes.
Every lbl_* column looks into the future: use it as a target only, never as a feature.
Only needs numpy + pandas.
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

TIME_COLS = ("time", "datetime", "date", "timestamp", "open_time")
MINUTE = pd.Timedelta("1min")


def parse_time(values):
    s = pd.Series(values)
    if pd.api.types.is_numeric_dtype(s):                       # epoch: s / ms / us
        unit = "us" if s.iloc[0] > 1e15 else "ms" if s.iloc[0] > 1e11 else "s"
        return pd.DatetimeIndex(pd.to_datetime(s, unit=unit, utc=True))
    return pd.DatetimeIndex(pd.to_datetime(s, utc=True, format="ISO8601"))


def read_base(path, time_col=None):
    df = pd.read_csv(path)
    time_col = time_col or next((c for c in df.columns if c.lower() in TIME_COLS), df.columns[0])
    df.index = parse_time(df.pop(time_col))
    df.index.name = time_col
    return df.sort_index()


def read_minutes(path):
    """1m OHLC on a complete minute grid; missing minutes are NaN and simply never touch a barrier."""
    m = pd.read_csv(path, index_col=0)
    m.index = parse_time(m.index)
    m = m[~m.index.duplicated()].sort_index()
    return m.reindex(pd.date_range(m.index[0], m.index[-1], freq="1min"))[["open", "high", "low", "close"]]


def label_names(levels):
    lv = [f"{x:g}" for x in levels]
    up = [f"up0-{lv[0]}"] + [f"up{a}-{b}" for a, b in zip(lv, lv[1:])] + [f"up>{lv[-1]}"]
    return [n.replace("up", "dn") for n in up[::-1]] + up


def class_to_trade(c, K):
    """Class -> (side, TP level index 1..K). dn/up beyond L_m -> short/long with TP = L_m; the two 0..L1 classes -> flat (0, 0)."""
    c = np.asarray(c)
    side = np.where(c >= K + 2, 1, np.where(c <= K - 1, -1, 0))
    return side, np.where(side > 0, c - K - 1, np.where(side < 0, K - c, 0))


def label_path(times, tf, fine, res, horizon, levels, chunk_cells=4_000_000):
    """Apply the rule to base bars `times` using the path in `fine` (OHLC on a regular grid of resolution `res`).
    Bars near the end of the data get a shortened window (padded with NaN); bars the grid doesn't cover stay NaN."""
    levels = np.sort(np.asarray(levels, dtype=float))
    K, n = len(levels), len(times)
    W = int(horizon * tf / res)                                    # window length in fine bars
    pad = np.full(W, np.nan)

    o, h, l, c = (np.concatenate([fine[k].to_numpy(dtype=float), pad]) for k in ("open", "high", "low", "close"))
    c_ff = pd.Series(c).ffill().to_numpy()
    bull = c >= o                                                 # NaN bar -> False
    start = np.floor((times + tf - fine.index[0]) / res).astype(int).to_numpy()   # entry bar
    valid = np.flatnonzero((start >= 1) & (start <= len(fine)))   # bar t itself lies inside the grid

    cols = ["lbl", "lbl_entry", "lbl_ret_h", "lbl_mfe", "lbl_mae", "lbl_t_touch", "lbl_missing_min"]
    out = {k: np.full(n, np.nan) for k in cols}
    hw, lw = sliding_window_view(h, W), sliding_window_view(l, W)
    ties, to_min = 0, res / MINUTE

    def first(mask):                                              # first True per row, W if never
        idx = mask.argmax(1)
        return np.where(mask[np.arange(len(mask)), idx], idx, W)

    for j in range(0, len(valid), max(1, chunk_cells // W)):
        i = valid[j: j + max(1, chunk_cells // W)]
        s = start[i]
        e = np.where(np.isfinite(o[s]), o[s], c_ff[s - 1])
        with np.errstate(invalid="ignore"):                       # rounded: an exact 1.00% touch must not hinge on float noise
            up = np.round((hw[s] / e[:, None] - 1) * 100, 9)
            dn = np.round((lw[s] / e[:, None] - 1) * 100, 9)
            tu = np.stack([first(up >= x) for x in levels])          # (K, n) minute of first touch
            td = np.stack([first(dn <= -x) for x in levels])
        bull_at = lambda t: bull[s + np.minimum(t, W - 1)]

        tie = (tu[0] == td[0]) & (tu[0] < W)
        up_first = (tu[0] < td[0]) | (tie & ~bull_at(tu[0]))
        dn_first = (td[0] < tu[0]) | (tie & bull_at(td[0]))
        up_mag = ((tu < td[0]) | ((tu == td[0]) & (tu < W) & ~bull_at(td[0]))).sum(0)
        dn_mag = ((td < tu[0]) | ((td == tu[0]) & (td < W) & bull_at(tu[0]))).sum(0)
        ret = np.round((c_ff[s + W - 1] / e - 1) * 100, 9)

        out["lbl"][i] = np.select([up_first, dn_first, ret >= 0], [K + 1 + up_mag, K - dn_mag, K + 1], K)
        out["lbl_entry"][i], out["lbl_ret_h"][i] = e, ret
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # window fully past the data end -> 0
            out["lbl_mfe"][i] = np.nan_to_num(np.nanmax(up, 1))
            out["lbl_mae"][i] = np.nan_to_num(np.nanmin(dn, 1))
        t_first = np.minimum(tu[0], td[0])
        out["lbl_t_touch"][i] = np.where(t_first < W, (t_first + 1) * to_min, np.nan)
        out["lbl_missing_min"][i] = np.isnan(up).sum(1) * to_min
        ties += tie.sum()

    return pd.DataFrame(out, index=times), int(ties)


def make_labels(base, m1, horizon, levels):
    """Label every base bar: on the 1m path where 1m data covers the bar, else on the base bars' own OHLC."""
    tf = base.index.to_series().diff().median()
    res, ties = label_path(base.index, tf, m1, MINUTE, horizon, levels)
    res["lbl_source"] = np.where(res["lbl"].notna(), "1m", None)

    ohlc = {k: next((c for c in base.columns if c.lower() == k), None) for k in ("open", "high", "low", "close")}
    miss = res["lbl"].isna()
    if miss.any() and all(ohlc.values()):
        grid = base[list(ohlc.values())].set_axis(list(ohlc), axis=1)
        grid = grid[~grid.index.duplicated()].reindex(pd.date_range(base.index[0], base.index[-1], freq=tf))
        fb, _ = label_path(base.index[miss], tf, grid, tf, horizon, levels)
        res.loc[miss, fb.columns] = fb.to_numpy()
        res.loc[miss & res["lbl"].notna(), "lbl_source"] = "base"

    res["lbl"] = res["lbl"].astype("Int64")
    return res, {"timeframe": tf, "window_min": int(horizon * tf / MINUTE), "first_touch_ties_1m": ties,
                 "source": res["lbl_source"].value_counts(dropna=False).to_dict(),
                 "short_window": int((res["lbl_missing_min"] > 0).sum())}


def check_time_convention(base, m1):
    """Share of base closes equal to the 1m close of the bar's last minute (≈1.0 if base time = open time)."""
    close = next((c for c in base.columns if c.lower() == "close"), None)
    if close is None:
        return None
    tf = base.index.to_series().diff().median()
    last_min = m1["close"].reindex(base.index + tf - MINUTE).to_numpy()
    ok = np.isfinite(last_min)
    return float(np.mean(np.abs(base[close].to_numpy()[ok] / last_min[ok] - 1) < 1e-4)) if ok.any() else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="base csv, any timeframe; time column = bar open time")
    p.add_argument("--minute", required=True, help="1m OHLC csv(.gz) of the same symbol")
    p.add_argument("--horizon", type=int, required=True, help="window length in base records")
    p.add_argument("--levels", type=float, nargs="+", required=True, help="TP/SL levels in %%, e.g. 1 2")
    p.add_argument("--time-col", help="time column in base (auto-detected by default)")
    p.add_argument("--out", help="output csv (default: <base>_h<H>_L<levels>.csv next to base)")
    a = p.parse_args()

    base, m1 = read_base(a.base, a.time_col), read_minutes(a.minute)
    match = check_time_convention(base, m1)
    if match is not None and match < 0.9:
        print(f"WARNING: only {match:.0%} of base closes match the 1m close at bar end; is the time column the bar OPEN time?")

    labels, info = make_labels(base, m1, a.horizon, a.levels)
    out = Path(a.out or Path(a.base).with_name(f"{Path(a.base).name.split('.')[0]}_h{a.horizon}_L{'-'.join(f'{x:g}' for x in a.levels)}.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    base.join(labels).to_csv(out)
    risk = {"horizon": a.horizon, "levels": sorted(a.levels), "timeframe": str(info["timeframe"]), "classes": label_names(sorted(a.levels))}
    out.with_suffix(".json").write_text(json.dumps(risk, indent=1))            # the risk setting travels with the labels

    print(f"timeframe {info['timeframe']}, horizon {a.horizon} -> {info['window_min']} minutes, levels {a.levels} %")
    print(f"rows {len(labels):,} | label source {info['source']} | windows with missing data {info['short_window']:,} | "
          f"same-minute ties {info['first_touch_ties_1m']:,} | base/1m close match {match if match is None else f'{match:.1%}'}")
    if labels["lbl"].isna().any():
        print(f"WARNING: {labels['lbl'].isna().sum():,} rows unlabeled (no 1m coverage and no OHLC columns in base)")
    names = dict(enumerate(label_names(sorted(a.levels))))
    print(labels["lbl"].map(names).value_counts(normalize=True).reindex(label_names(sorted(a.levels))).round(3).to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
