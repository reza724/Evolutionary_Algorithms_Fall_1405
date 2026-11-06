"""Independent re-derivation of the four-class labels, straight from raw Binance klines.

Cross-check of the `make_labels_4class.py` outputs. Differences from that pipeline:

  * the 1-minute path is the RAW BNBUSDT spot klines (data/BNBUSDT_1m_{train,test}_raw.csv),
    downloaded separately and scaled to the course scale (real price * 8/27);
  * the resolution algorithm is time-major - it loops over the horizon minute by minute and
    sweeps every row at once - instead of the row-major sliding-window implementation;
  * a brute-force scan (`brute`) re-derives labels for random rows with an explicit loop.

    python verify_labels_from_binance.py

Per labeled csv it prints how many rows disagree with the stored label and where. Exits 1
if any file disagrees on more than 0.02 % of its rows or on any brute-forced row.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
SCALE = 8 / 27                                   # course price = raw Binance price * SCALE
MINUTE = pd.Timedelta("1min")
SETTINGS = {"sl1_tp2_h12": (1.0, 2.0, 12), "sl2_tp4_h24": (2.0, 4.0, 24), "sl3_tp6_h48": (3.0, 6.0, 48)}


def read_raw():
    """Raw Binance klines -> one course-scale minute grid (missing minutes are NaN, never touch a level)."""
    grids = []
    for p in (DATA / "BNBUSDT_1m_train_raw.csv", DATA / "BNBUSDT_1m_test_raw.csv"):
        m = pd.read_csv(p, index_col=0)
        m.index = pd.to_datetime(m.index, utc=True, format="ISO8601")
        m = m[~m.index.duplicated()].sort_index()
        grids.append(m[["open", "high", "low", "close"]].astype(float))
    g = pd.concat(grids).sort_index()
    return g[~g.index.duplicated()]


def label_time_major(times, grid, tf_min, horizon_min, sl_pct, tp_pct):
    """Time-major four-class resolution.

    tp_first / sl_first = the minute index (0-based, relative to the entry minute) of the
    first barrier touch, never = W+1. Same-minute ties follow the candle path: open beyond
    a barrier wins, else bull O->L->H->C places SL first, bear O->H->L->C places TP first.
    """
    W = int(horizon_min)
    s = ((times - grid.index[0]) // MINUTE).to_numpy(dtype=np.int64) + int(tf_min)
    o, h, l, c = (grid[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    n = len(times)

    in_data = (s >= 0) & (s + W <= len(o))
    e = o[np.clip(s, 0, len(o) - 1)]                       # entry = open of the entry minute
    valid = in_data & np.isfinite(e)
    rows = np.flatnonzero(valid)                            # rows resolved by the sweep
    s_ok = s[rows]
    tp_price = e[rows] * (1 + tp_pct / 100)
    sl_price = e[rows] * (1 - sl_pct / 100)
    tp_first = np.full(n, W + 1)
    sl_first = np.full(n, W + 1)

    for k in range(W):
        pos = s_ok + k
        o_k, h_k, l_k, c_k = o[pos], h[pos], l[pos], c[pos]
        fin = np.isfinite(h_k) & np.isfinite(l_k)
        k_tp = fin & (h_k >= tp_price)
        k_sl = fin & (l_k <= sl_price)
        both = k_tp & k_sl                                  # both barriers inside the same candle
        if both.any():                                      # resolve the tie, keep only the winner
            b = np.flatnonzero(both)
            ob, cb = o_k[b], c_k[b]
            tp_win = (ob >= tp_price[b]) | ((ob > sl_price[b]) & (cb < ob))
            kp = rows[b][tp_win]                            # these rows: TP is the first touch
            ks = rows[b][~tp_win]                           # these rows: SL is the first touch
            k_tp[rows[b]] = False
            k_sl[rows[b]] = False
            tp_first[kp] = np.minimum(k, tp_first[kp])
            sl_first[ks] = np.minimum(k, sl_first[ks])
            k_tp[kp] = True                                 # consumed, never re-recorded
            k_sl[ks] = True
        ntp = rows[np.flatnonzero(k_tp)]
        nsl = rows[np.flatnonzero(k_sl)]
        tp_first[ntp] = np.minimum(k, tp_first[ntp])
        sl_first[nsl] = np.minimum(k, sl_first[nsl])
    # horizon rows: no touch at all -> compare the window-end close with the entry
    tail = c[np.clip(s + W - 1, 0, len(o) - 1)]
    ret = tail / np.where(np.isfinite(e), e, 1) - 1
    lbl = np.select([tp_first < sl_first, sl_first < tp_first, (tp_first == W + 1) & (sl_first == W + 1) & (ret > 0)],
                    [1, 2, 3], default=4).astype(float)
    lbl[~valid] = np.nan                                    # outside data, cut window, or no entry print
    return lbl.astype(float)


def brute(times, grid, idx, tf_min, horizon_min, sl_pct, tp_pct):
    """Explicit per-row scan for rows `idx`; the trade exits at the first barrier it touches."""
    W = int(horizon_min)
    o, h, l, c = (grid[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    out = []
    for i in idx:
        s = int(((times[i] - grid.index[0]) // MINUTE) + tf_min)
        if s < 0 or s + W > len(o):
            out.append(np.nan)
            continue
        e = o[s]
        if not np.isfinite(e):
            out.append(np.nan)
            continue
        tp, sl = e * (1 + tp_pct / 100), e * (1 - sl_pct / 100)
        ft, fs = None, None
        for k in range(W):
            hk, lk, ok_, ck_ = h[s + k], l[s + k], o[s + k], c[s + k]
            tp_k = np.isfinite(hk) and hk >= tp
            sl_k = np.isfinite(lk) and lk <= sl
            if tp_k and sl_k:                               # both inside this candle -> path rule
                if ok_ >= tp:
                    ft = k
                elif ok_ <= sl:
                    fs = k
                elif ck_ >= ok_:                            # bull visits L first
                    fs = k
                else:                                       # bear visits H first
                    ft = k
            elif tp_k:
                ft = k
            elif sl_k:
                fs = k
            else:
                continue
            break                                           # first touch decides the trade
        if ft is not None:
            out.append(1.0)
        elif fs is not None:
            out.append(2.0)
        else:
            tail = c[s + W - 1]
            out.append(3.0 if np.isfinite(tail) and tail / e > 1 else 4.0)
    return np.asarray(out, dtype=float)


def compare(path, new, times, tag):
    got = pd.read_csv(path, usecols=["time", "lbl"], index_col=0)
    got.index = pd.to_datetime(got.index, utc=True, format="ISO8601")
    stored = got["lbl"].reindex(times).to_numpy(dtype=float)
    diff = (stored != new) & ~(np.isnan(stored) & np.isnan(new))
    d = int(diff.sum())
    print(f"{tag}: differing rows {d:,} / {len(times):,} ({d / len(times) * 100:.4f}%)")
    if d:
        i = np.flatnonzero(diff)
        head = pd.DataFrame({"t": times[i], "stored": stored[i], "raw": new[i]}).head(6)
        print(head.to_string(index=False))
        print(pd.crosstab(pd.Series(stored[i], name="stored"), pd.Series(new[i], name="raw")).to_string())
    return d / len(times)


def main():
    grid = read_raw()
    print(f"raw Binance grid: {len(grid):,} minutes {grid.index[0]} -> {grid.index[-1]}")

    shipped = pd.concat([pd.read_csv(DATA / "symbol5_1m_train.csv", index_col=0, parse_dates=True)["close"],
                         pd.read_csv(DATA / "symbol5_1m_test.csv", index_col=0, parse_dates=True)["close"]])
    common = shipped.index.intersection(grid.index)
    dev = np.abs(shipped.loc[common].to_numpy() / (grid["close"].reindex(common).to_numpy() * SCALE) - 1)
    print(f"scale vs shipped 1m closes on {len(common):,} minutes: max dev {np.nanmax(dev):.2e}, median {np.nanmedian(dev):.2e}")

    worst = 0.0
    for split in ("train", "test"):
        base = pd.read_csv(DATA / f"symbol5_5m_{split}.csv", usecols=["time"], index_col=0)
        base.index = pd.to_datetime(base.index, utc=True, format="ISO8601")
        times = base.index
        rng = np.random.default_rng(7)
        for name, (sl, tp, hours) in SETTINGS.items():
            lbl = label_time_major(times, grid, 5, hours * 60, sl, tp)
            frac = compare(DATA / f"symbol5_5m_{split}_{name}.csv", lbl, times, f"{split}/{name}")
            worst = max(worst, frac)
            pick = rng.choice(len(times), size=400, replace=False)
            got = pd.read_csv(DATA / f"symbol5_5m_{split}_{name}.csv", usecols=["lbl"])["lbl"].to_numpy()[pick]
            b = brute(times, grid, pick, 5, hours * 60, sl, tp)
            both_nan = np.isnan(got) & np.isnan(b)
            bm = int(np.sum((got != b) & ~both_nan))
            nm = int(np.sum(np.isnan(got) != np.isnan(b)))
            print(f"   brute-force 400 rows: mismatched labels {bm}, label-vs-NaN {nm}")
            if bm or nm:
                worst = 1.0
    print("\nPASSED" if worst <= 0.0002 else f"\nFAILED (worst divergence {worst:.4%})")
    return 0 if worst <= 0.0002 else 1


if __name__ == "__main__":
    sys.exit(main())
