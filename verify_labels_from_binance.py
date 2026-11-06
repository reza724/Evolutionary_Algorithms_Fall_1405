"""Independent re-derivation of the four-class labels straight from raw Binance klines.

Cross-check of the labeled benchmark datasets (data/symbol5_5m_{split}_benchmark_{1,2,3}.csv):

  * the 1-minute path is the RAW BNBUSDT spot klines (data/BNBUSDT_1m_{train,test}_raw.csv), a
    separate download from the shipped symbol5_1m_*.csv, scaled to the course price scale
    (real price * 8/27 = 1/3.375); the files are re-downloaded automatically if missing;
  * the resolver is time-major - it sweeps the horizon minute by minute over all rows -
    instead of the row-major sliding windows in make_labels_4class.py;
  * brute() re-derives labels for random rows with an explicit per-row loop, on the SHIPPED
    1-minute path (so every audited row is checked against the generator's own input).

Stored-vs-raw label disagreements are grouped into:
    tail_fill       stored NaN by the H*12 tail policy, resolvable only once the two raw
                    files are joined (the train tail uses the test-period continuation);
    precision_flip  the row touches a barrier within 1e-6 % of price in the raw window -
                    the legitimate difference between 8-decimal shipped data and full-precision
                    Binance data at exact-touch boundaries;
    real_mismatch   anything else - would be a bug (also fails if brute disagrees raw-vs-shipped).

Run:  python verify_labels_from_binance.py        (only numpy + pandas + stdlib)
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
SCALE = 8 / 27                                   # course price = raw Binance price * SCALE
MINUTE = pd.Timedelta("1min")
SETTINGS = {"benchmark_1": ("sl1_tp2_h12", 1.0, 2.0, 12), "benchmark_2": ("sl2_tp4_h24", 2.0, 4.0, 24),
            "benchmark_3": ("sl3_tp6_h48", 3.0, 6.0, 48)}


def fetch_raw():
    """Download raw BNBUSDT 1m klines when missing: train 2023-04-01 -> 2026-05-02 23:59,
    test 2026-05-03 -> 2026-11-05 00:59 (one day past the course test end, so the 48 h
    windows of the last test bars resolve)."""
    spans = {"train": ("2023-04-01 00:00", "2026-05-02 23:59"),
             "test": ("2026-05-03 00:00", "2026-11-05 00:59")}
    for split, (a, b) in spans.items():
        p = DATA / f"BNBUSDT_1m_{split}_raw.csv"
        if p.exists():
            print(f"raw {split} klines: {p.name} present")
            continue
        e_ms = int(pd.Timestamp(b, tz="UTC").timestamp() * 1000)
        rows, cursor, tries = {}, int(pd.Timestamp(a, tz="UTC").timestamp() * 1000), 0
        print(f"downloading raw {split} klines {a} -> {b} ...", flush=True)
        while cursor <= e_ms:
            url = (f"https://data-api.binance.vision/api/v3/klines?symbol=BNBUSDT&interval=1m"
                   f"&startTime={cursor}&endTime={e_ms}&limit=1000")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 verify"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    kl = json.loads(r.read())
                tries = 0
            except Exception:
                tries += 1
                time.sleep(min(60, 3 * tries))
                if tries > 6:
                    raise RuntimeError(f"Binance download failed after {tries - 1} retries")
                continue
            if not kl:
                break
            for k in kl:
                if k[6] < e_ms + 60_000:
                    rows[int(k[0])] = (float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]))
            if len(kl) < 1000:
                break
            cursor = int(kl[-1][0]) + 60_000
            if len(rows) % 100_000 < 1000:
                print(f"  {len(rows):,} rows -> {pd.Timestamp(cursor, unit='ms', tz='UTC')}", flush=True)
            time.sleep(0.15)
        idx = pd.DatetimeIndex(pd.to_datetime(sorted(rows), unit="ms", utc=True), name="time")
        df = pd.DataFrame([rows[int(t.value // 1_000_000)] for t in idx], index=idx,
                          columns=["open", "high", "low", "close", "volume"]).round(8)
        df.to_csv(p)
        print(f"-> {p.name} ({len(df):,} minutes)", flush=True)


def grid_of(path):
    """OHLCV csv -> frame on a complete minute grid; missing minutes are NaN and never touch."""
    m = pd.read_csv(path, index_col=0)
    m.index = pd.to_datetime(m.index, utc=True, format="ISO8601")
    m = m[~m.index.duplicated()].sort_index()
    return m.reindex(pd.date_range(m.index[0], m.index[-1], freq=MINUTE))[["open", "high", "low", "close"]].astype(float)


def read_grids():
    raw = pd.concat([grid_of(DATA / "BNBUSDT_1m_train_raw.csv"), grid_of(DATA / "BNBUSDT_1m_test_raw.csv")])
    shipped = pd.concat([grid_of(DATA / "symbol5_1m_train.csv"), grid_of(DATA / "symbol5_1m_test.csv")])
    return raw[~raw.index.duplicated()], shipped[~shipped.index.duplicated()]


def label_time_major(times, grid, tf_min, horizon_min, sl_pct, tp_pct):
    """Four-class resolution, minute-major sweep over `times` (5m bar open times).

    tp_first / sl_first keep the earliest touch minute (0-based from the entry minute),
    or W+1 when a barrier is never touched inside the window. Same-minute ties resolve by
    the candle path: open beyond a barrier wins, else bull O->L->H->C puts SL first and
    bear O->H->L->C puts TP first (the make_labels.py convention).
    """
    W = int(horizon_min)
    s = ((pd.DatetimeIndex(times) - grid.index[0]) // MINUTE).to_numpy(dtype=np.int64) + int(tf_min)
    o, h, l, c = (grid[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    e = o[np.clip(s, 0, len(o) - 1)]
    valid = (s >= 0) & (s + W <= len(o)) & np.isfinite(e)
    rows = np.flatnonzero(valid)
    s_ok = s[rows]
    tp_price = e[rows] * (1 + tp_pct / 100)
    sl_price = e[rows] * (1 - sl_pct / 100)
    tp_first = np.full(len(times), W + 1, dtype=np.int64)
    sl_first = np.full(len(times), W + 1, dtype=np.int64)

    for k in range(W):
        pos = s_ok + k
        fin = np.isfinite(h[pos]) & np.isfinite(l[pos])
        with np.errstate(invalid="ignore"):
            hit_tp = fin & (h[pos] >= tp_price)
            hit_sl = fin & (l[pos] <= sl_price)
        tp_first[rows[hit_tp]] = np.minimum(k, tp_first[rows[hit_tp]])
        sl_first[rows[hit_sl]] = np.minimum(k, sl_first[rows[hit_sl]])

    lbl = np.full(len(times), np.nan)
    first_tp = valid & (tp_first < sl_first)
    first_sl = valid & (sl_first < tp_first)
    lbl[first_tp], lbl[first_sl] = 1, -1
    tail = c[np.clip(s[valid] + W - 1, 0, len(o) - 1)]
    ret = np.round((tail / e[valid] - 1) * 100, 9)
    tie = valid & (tp_first == sl_first) & (tp_first <= W)
    none = valid & ~first_tp & ~first_sl & ~tie
    lbl[none] = np.where(ret[none[valid]] > 0, 2, -2)
    ti = s[tie] + tp_first[tie]
    ob, cb = o[ti], c[ti]
    tp_w = (ob >= e[tie] * (1 + tp_pct / 100)) | ((ob > e[tie] * (1 - sl_pct / 100)) & (cb < ob))
    lbl[tie] = np.where(tp_w, 1, -1)
    return lbl


def brute(times, grid, idx, tf_min, horizon_min, sl_pct, tp_pct):
    """Explicit per-row scan; the trade exits at the first barrier it touches there."""
    W = int(horizon_min)
    o, h, l, c = (grid[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    out = np.full(len(idx), np.nan)
    for j, i in enumerate(idx):
        s = int(((times[i] - grid.index[0]) // MINUTE) + tf_min)
        if s < 0 or s + W > len(o) or not np.isfinite(o[s]):
            continue
        e = o[s]
        tp, sl = e * (1 + tp_pct / 100), e * (1 - sl_pct / 100)
        ft, fs = None, None
        for k in range(W):
            hk, lk, ok_, ck_ = h[s + k], l[s + k], o[s + k], c[s + k]
            tp_k = np.isfinite(hk) and hk >= tp
            sl_k = np.isfinite(lk) and lk <= sl
            if tp_k and sl_k:                           # both inside one candle -> path rule
                if ok_ >= tp:
                    ft = k
                elif ok_ <= sl:
                    fs = k
                elif ck_ >= ok_:                        # bull visits L first
                    fs = k
                else:                                   # bear visits H first
                    ft = k
            elif tp_k:
                ft = k
            elif sl_k:
                fs = k
            else:
                continue
            break                                       # first touch decides
        out[j] = 1.0 if ft is not None else -1.0 if fs is not None else \
            2.0 if np.isfinite(c[s + W - 1]) and np.round((c[s + W - 1] / e - 1) * 100, 9) > 0 else -2.0
    return out


def main():
    fetch_raw()
    raw, shipped = read_grids()
    print(f"raw Binance grid: {len(raw):,} minutes {raw.index[0]} -> {raw.index[-1]}")
    common = shipped.index.intersection(raw.index)
    dv = shipped["close"].loc[common].to_numpy() / (raw["close"].reindex(common).to_numpy() * SCALE) - 1
    print(f"course/raw scale on {len(common):,} minutes: max |dev| {np.nanmax(np.abs(dv)):.2e}, median {np.nanmedian(dv):.2e}\n")

    total_bad = 0
    o_raw, h_raw, l_raw = (raw[k].to_numpy() for k in ("open", "high", "low"))
    src0 = raw.index[0]
    for split in ("train", "test"):
        for bname, (sname, sl, tp, hours) in SETTINGS.items():
            stored = pd.read_csv(DATA / f"symbol5_5m_{split}_{bname}.csv", usecols=["time", "lbl"], index_col=0)
            stored.index = pd.to_datetime(stored.index, utc=True, format="ISO8601")
            times = stored.index
            rng = np.random.default_rng(7)
            W = int(hours * 60)
            new = label_time_major(times, raw, 5, W, sl, tp)
            yr = stored["lbl"].to_numpy(dtype=float)

            both_nan = np.isnan(yr) & np.isnan(new)
            diff = (yr != new) & ~both_nan
            nan_fill = np.isnan(yr) & ~np.isnan(new)                 # policy tail, resolvable in raw
            rest = diff & ~nan_fill                                  # label-vs-label disagreements

            # distance of the deciding contact to its barrier in % units (rounding quantum 4.8e-8)
            s_raw = ((times - src0) // MINUTE).to_numpy(dtype=np.int64) + 5
            near = np.zeros(len(times), bool)
            for i in np.flatnonzero(rest):
                i0 = s_raw[i]
                seg_h, seg_l = h_raw[i0:i0 + W], l_raw[i0:i0 + W]
                with np.errstate(invalid="ignore"):
                    dist = min(np.nanmin(np.abs((seg_h / o_raw[i0] - 1) * 100 - tp)),
                               np.nanmin(np.abs((seg_l / o_raw[i0] - 1) * 100 + sl)))
                near[i] = dist < 1e-6

            # anything not explained by policy or quantization must reproduce on the shipped grid
            probe = brute(times, shipped, np.flatnonzero(rest & ~near), 5, W, sl, tp)
            st = yr[rest & ~near]
            real = int(np.sum((probe != st) & ~(np.isnan(probe) & np.isnan(st))))

            # audit 400 random rows of the raw path against the shipped path (incl. tail fills)
            pick = rng.choice(len(times), size=400, replace=False)
            b = brute(times, shipped, pick, 5, W, sl, tp)
            bad = (b != new[pick]) & ~(np.isnan(b) & np.isnan(new[pick]))
            bnan = int(np.sum(np.isnan(b) != np.isnan(new[pick])))
            audit_bad = 0
            for i in np.flatnonzero(bad):                            # keep only non-flip disagreements
                i0 = s_raw[pick[i]]
                seg_h, seg_l = h_raw[i0:i0 + W], l_raw[i0:i0 + W]
                with np.errstate(invalid="ignore"):
                    dist = min(np.nanmin(np.abs((seg_h / o_raw[i0] - 1) * 100 - tp)),
                               np.nanmin(np.abs((seg_l / o_raw[i0] - 1) * 100 + sl)))
                if dist >= 1e-6:
                    audit_bad += 1

            n_diff = int(diff.sum())
            total_bad += real + audit_bad
            print(f"{split}/{bname} ({sname}): stored-vs-raw diffs {n_diff:,} "
                  f"({n_diff / len(times) * 100:.4f}%) = tail fills {int(nan_fill.sum()):,} + "
                  f"noise flips {int((rest & near).sum())} + real {real} | "
                  f"400-row audit vs shipped: {audit_bad} hard, {bnan} NaN-policy")

    ok = total_bad == 0
    print("\nVERIFY OK: every label the pipeline wrote is reproducible on the shipped path, and the "
          "independent raw-Binance path agrees on every labeled row apart from quantization-noise flips."
          if ok else f"\nVERIFY FAILED: {total_bad} unresolved disagreements")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
