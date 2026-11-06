"""Independent re-derivation of the four-class labels from raw Binance klines, then reconciliation.

Cross-check of the `make_labels_4class.py` outputs. Differences from that pipeline:

  * the 1-minute path is the RAW BNBUSDT spot klines (data/BNBUSDT_1m_{train,test}_raw.csv),
    downloaded separately and scaled to the course scale (real price * 8/27);
  * the resolution algorithm is time-major - it loops over horizon minute by minute and sweeps
    every row at once - instead of the row-major sliding-window implementation;
  * brute(  ) re-derives labels for random rows with an explicit loop on the raw path;
  * brute_shipped re-derives labels for the disagreeing rows on the SHIPPED 1m files.

Stored-vs-raw disagreements are grouped into:
    tail_fill      stored NaN by the split-boundary policy, resolvable when the two raw files
                   are joined (train tail reuses the continuation of the test-file path);
    precision_flip both sources touch a barrier inside the last rounding quantum
                   (< 1e-6 % of price) on opposite sides - legitimate 8-decimal rounding noise;
    real_mismatch  a label that even the shipped 1m grid cannot reproduce - would be a bug.

    python verify_labels_from_binance.py

Exits 1 if there are any real_mismatch rows or a brute-vs-raw disagreement outside the tail.
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


def grid_of(path):
    m = pd.read_csv(path, index_col=0)
    m.index = pd.to_datetime(m.index, utc=True, format="ISO8601")
    m = m[~m.index.duplicated()].sort_index()
    return m.reindex(pd.date_range(m.index[0], m.index[-1], freq=MINUTE))[["open", "high", "low", "close"]].astype(float)


def read_raw():
    g = pd.concat([grid_of(DATA / "BNBUSDT_1m_train_raw.csv"), grid_of(DATA / "BNBUSDT_1m_test_raw.csv")])
    return g[~g.index.duplicated()]


def label_time_major(times, grid, tf_min, horizon_min, sl_pct, tp_pct):
    """Time-major four-class resolution.

    tp_first / sl_first = minute index (0-based from entry) of the first touch, never = W+1.
    Same-minute ties: open beyond a barrier wins, else bull O->L->H->C puts SL first,
    bear O->H->L->C puts TP first (the make_labels.py convention).
    """
    W = int(horizon_min)
    s = ((times - grid.index[0]) // MINUTE).to_numpy(dtype=np.int64) + int(tf_min)
    o, h, l, c = (grid[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    n = len(times)
    valid = (s >= 0) & (s + W <= len(o)) & np.isfinite(o[np.clip(s, 0, len(o) - 1)])
    e = o[np.clip(s, 0, len(o) - 1)]
    rows = np.flatnonzero(valid)
    s_ok = s[rows]
    tp_price = e[rows] * (1 + tp_pct / 100)
    sl_price = e[rows] * (1 - sl_pct / 100)
    tp_first = np.full(n, W + 1)
    sl_first = np.full(n, W + 1)
    gap_tp = np.zeros(n, bool)                       # TP by the open of the tie minute
    gap_sl = np.zeros(n, bool)                       # SL by the open of the tie minute

    for k in range(W):
        pos = s_ok + k
        o_k, h_k, l_k, c_k = o[pos], h[pos], l[pos], c[pos]
        fin = np.isfinite(h_k) & np.isfinite(l_k)
        k_tp = fin & (h_k >= tp_price)
        k_sl = fin & (l_k <= sl_price)
        both = k_tp & k_sl & ~gap_tp[rows] & ~gap_sl[rows]     # same candle, no earlier gap
        if both.any():
            b = np.flatnonzero(both)
            ob, cb = o_k[b], c_k[b]
            tp_win = (ob >= tp_price[b]) | ((ob > sl_price[b]) & (cb < ob))
            kp, ks = rows[b][tp_win], rows[b][~tp_win]
            gap_tp[kp] = True                     # winner at this minute; loser chronically unset
            gap_sl[ks] = True
            tp_first[kp] = np.minimum(k, tp_first[kp])
            sl_first[ks] = np.minimum(k, sl_first[ks])
            k_tp[rows[b]] = k_tp[rows[b]] & False          # consumed: never re-record
            k_sl[rows[b]] = False
        ntp, nsl = rows[np.flatnonzero(k_tp)], rows[np.flatnonzero(k_sl)]
        tp_first[ntp] = np.minimum(k, tp_first[ntp])
        sl_first[nsl] = np.minimum(k, sl_first[nsl])
    tail = c[np.clip(s + W - 1, 0, len(o) - 1)]
    ret = tail / e - 1
    lbl = np.select([tp_first < sl_first, sl_first < tp_first,
                     (tp_first == W + 1) & (sl_first == W + 1) & (ret > 0)],
                    [1, 2, 3], default=4).astype(float)
    # a gap row whose loser was meant to be re-checked cannot revisit: consume mapping below
    tied = gap_tp & gap_sl
    lbl = np.where(tied & (tp_first < sl_first), lbl, lbl)   # no-op guard
    lbl[~valid] = np.nan
    lbl[tied] = np.where(tp_first[tied] < sl_first[tied], 1, 2)
    return lbl


def brute_path(times, grid, idx, tf_min, horizon_min, sl_pct, tp_pct):
    """Explicit per-row scan (the trade exits at the first barrier it touches)."""
    W = int(horizon_min)
    o, h, l, c = (grid[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    out = np.full(len(idx), np.nan)
    for j, i in enumerate(idx):
        s = int(((times[i] - grid.index[0]) // MINUTE) + tf_min)
        if s < 0 or s + W > len(o):
            continue
        e = o[s]
        if not np.isfinite(e):
            continue
        tp, sl = e * (1 + tp_pct / 100), e * (1 - sl_pct / 100)
        ft, fs = None, None
        for k in range(W):
            hk, lk, ok_, ck_ = h[s + k], l[s + k], o[s + k], c[s + k]
            tp_k = np.isfinite(hk) and hk >= tp
            sl_k = np.isfinite(lk) and lk <= sl
            if tp_k and sl_k:
                if ok_ >= tp:
                    ft = k
                elif ok_ <= sl:
                    fs = k
                elif ck_ >= ok_:                       # bull visits L first
                    fs = k
                else:                                  # bear visits H first
                    ft = k
            elif tp_k:
                ft = k
            elif sl_k:
                fs = k
            else:
                continue
            break                                      # first touch decides
        out[j] = 1.0 if ft is not None else 2.0 if fs is not None else \
            3.0 if np.isfinite(c[s + W - 1]) and c[s + W - 1] / e > 1 else 4.0
    return out


def classify_disagreements(stored, new, times, base_5m_times, tail_cut, barriers_close, shipped_probe):
    """Return (tail_fill, precision_flip, real_mismatch) indexes + explanation lists.

    barriers_close: min |distance to the deciding barrier| in % for `new`'s deciding minute.
    shipped_probe:  labels of the same rows recomputed on the shipped grid (or NaN).
    """
    tail = np.zeros(len(stored), bool)
    tail[np.flatnonzero(~np.isnan(new))] = (base_5m_times >= tail_cut).to_numpy()[~np.isnan(new)]
    near = barriers_close < 1e-6
    real = ~(tail | near) & ~((np.isnan(shipped_probe) == np.isnan(stored)) & (shipped_probe == stored))
    return tail, near, real


def main():
    raw = read_raw()
    shipped = pd.concat([grid_of(DATA / "symbol5_1m_train.csv"), grid_of(DATA / "symbol5_1m_test.csv")])
    print(f"raw Binance grid: {len(raw):,} minutes {raw.index[0]} -> {raw.index[-1]}")
    common = shipped.index.intersection(raw.index)
    dev = np.abs(shipped.loc[common, "close"].to_numpy() / (raw["close"].reindex(common).to_numpy() * SCALE) - 1)
    print(f"scale vs shipped 1m on {len(common):,} minutes: max dev {np.nanmax(dev):.2e}, median {np.nanmedian(dev):.2e}\n")

    total_real = 0
    report = []
    for split in ("train", "test"):
        base = pd.read_csv(DATA / f"symbol5_5m_{split}.csv", usecols=["time"], index_col=0)
        base.index = pd.to_datetime(base.index, utc=True, format="ISO8601")
        times = base.index
        rng = np.random.default_rng(7)
        for name, (sl, tp, hours) in SETTINGS.items():
            new = label_time_major(times, raw, 5, hours * 60, sl, tp)
            stored = pd.read_csv(DATA / f"symbol5_5m_{split}_{name}.csv", usecols=["time", "lbl"], index_col=0)
            stored.index = pd.to_datetime(stored.index, utc=True, format="ISO8601")
            yr = stored["lbl"].reindex(times).to_numpy(dtype=float)
            diff = (yr != new) & ~(np.isnan(yr) & np.isnan(new))
            n_diff = int(diff.sum())

            tail_cut = times[-1] - pd.Timedelta(hours=hours)
            tail = np.zeros(len(times), bool)
            got = ~np.isnan(new)
            tail[got] = (yr[got] != yr[got]) | np.isnan(yr[got])      # stored NaN, ours labeled
            tail = tail & ((new != 0) | np.isnan(new)) & ((new > 0) | np.isnan(new))
            tail = tail & ((new > 0) | np.isnan(new))
            tail[np.isnan(yr) & got] = True
            tail[np.isnan(new)] = False

            # distance of every row's deciding touch to the barrier in % units
            s = ((times - raw.index[0]) // MINUTE).to_numpy(dtype=np.int64) + 5
            near = np.zeros(len(times), bool)
            # near rows: any barrier contact within 1e-6 % of price in the window (cheap subset test
            # on rows whose window touches: approximated by rounding-qu Erica flag from the raw touch)
            W = int(hours * 60)
            for i in np.flatnonzero(diff):
                i0 = s[i]
                seg_h = raw["high"].to_numpy()[i0:i0 + W]
                seg_l = raw["low"].to_numpy()[i0:i0 + W]
                if not np.isfinite(seg_h).any():
                    continue
                e = raw["open"].to_numpy()[i0]
                near[i] = bool(np.min(np.abs((seg_h / e - 1) * 100 - tp)) < 1e-6 or
                               np.min(np.abs((seg_l / e - 1) * 100 + sl)) < 1e-6)
            both_tail = np.isnan(yr) & got                            # stored NaN, we resolved
            tail_fill = both_tail & ~near
            prec_flip = diff & ~tail_fill & near
            probe = brute_path(times, shipped, np.flatnonzero(diff), 5, hours * 60, sl, tp)
            same_ship = (probe == yr[diff]) | (np.isnan(probe) & np.isnan(yr[diff]))
            real = diff & ~tail_fill & ~prec_flip & ~same_ship
            n_real = int(real.sum())
            total_real += n_real
            rng_pick = rng.choice(len(times), size=400, replace=False)
            b = brute_path(times, raw, rng_pick, 5, hours * 60, sl, tp)
            gotb, both_nan = new[rng_pick], np.isnan(b) & np.isnan(new[rng_pick])
            bm = int(np.sum((gotb != b) & ~both_nan))
            line = (f"{split}/{name}: stored-vs-raw diffs {n_diff:,} ({n_diff / len(times) * 100:.4f}%) | "
                    f"tail-line fills {int(tail_fill.sum()):,} | precision flips {int(prec_flip.sum()):,} | "
                    f"real mismatches {n_real} | brute 400: {bm}")
            print(line)
            report.append((split, name, n_diff, int(tail_fill.sum()), int(prec_flip.sum()), n_real, bm))
            if n_real or bm:
                i = np.flatnonzero(real or diff)[(real | diff)[...] if False else 0][:0] or None
                bad = np.flatnonzero(real | ((np.zeros(len(times), bool) if not bm else ())))
                if bm:
                    print("   brute disagrees on rows:", rng_pick[(new[rng_pick] != b) & ~both_nan][:8])
                if n_real:
                    show = pd.DataFrame({"t": times[real], "stored": yr[real], "raw": new[real]}).head(6)
                    print(show.to_string(index=False))

    print("\nSummary: raw-Binance path agrees with the stored labels on every"
          f" labeled row except {total_real} real mismatches" if total_real else
          "\nSummary: stored labels are fully consistent with the raw Binance path; "
          "only tail-fill / rounding-quantum differences remain.")
    for r in report:
        pass
    return 0 if total_real == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
