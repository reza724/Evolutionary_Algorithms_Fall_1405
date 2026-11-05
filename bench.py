"""bench: one trading environment for every ML / DL / RL / evolutionary model.

    python bench.py train [submissions/my_model ...]  # fit on the train file, save to submissions/my_model/artifacts/
    python bench.py valid [submissions/my_model ...]  # fit on train minus its end, score the end (artifacts untouched)
    python bench.py check [submissions/my_model ...]  # rule checks: no look-ahead, accounting, determinism

The standard (see README.md):
  * A submission is a folder submissions/<name>/ with model.py (one subclass of Model or LabelModel)
    and artifacts/ (written by `train`).
  * fit(train) learns from the train DataFrame. act(obs) returns the target position in [-1, 1].
  * Decision at the close of bar t, fill at the open of bar t+1, fee FEE_BPS per side on every position change.
  * The hidden test period is traded through this same TradingEnv, so what works on train works there.
"""
import argparse
import importlib.util
import random
import sys
import time
import traceback
from pathlib import Path

if __name__ == "__main__":                       # `import bench` in model.py must get this module, not a second copy
    sys.modules["bench"] = sys.modules[__name__]

import joblib
import numpy as np
import pandas as pd

from make_labels import class_to_trade, label_names, make_labels, read_minutes

HERE = Path(__file__).parent
DATA, DATASETS, RESULTS, SUBMISSIONS = HERE / "data", HERE / "datasets", HERE / "results", HERE / "submissions"
SYMBOL, TF, FEE_BPS = "symbol5", "5m", 10        # the course setting: Binance taker fee 0.1 % per side
LABEL_PREFIX = "lbl"


# ---------------------------------------------------------------- data
def load(split="train", symbol=SYMBOL, tf=TF) -> pd.DataFrame:
    """data/{symbol}_{tf}_{split}.csv.gz -> DataFrame indexed by UTC bar open time. Never fit or tune on split='test'."""
    df = pd.read_csv(DATA / f"{symbol}_{tf}_{split}.csv.gz", index_col=0)
    df.index = pd.to_datetime(df.index, utc=True, format="ISO8601")
    df = df.sort_index()
    df.attrs.update(symbol=symbol, tf=tf, split=split)
    return df


def features_only(df: pd.DataFrame) -> pd.DataFrame:
    """Everything a model may see while trading: all columns except the future-looking lbl_* labels."""
    return df[[c for c in df.columns if not c.startswith(LABEL_PREFIX)]]


def labels(df: pd.DataFrame, horizon: int, levels) -> pd.DataFrame:
    """TP/SL label columns of make_labels.py for the rows of `df` (a frame from load()), cached in datasets/.

    They look into the future: use them as training targets only. Rows whose window was cut at the end of the
    file have lbl_missing_min > 0; drop them.
    """
    a = df.attrs
    if not {"symbol", "tf", "split"} <= a.keys():
        raise ValueError("labels() needs a DataFrame from bench.load() (the one fit() receives), or a slice of it")
    levels = sorted(float(x) for x in levels)
    name = f"{a['symbol']}_{a['tf']}_{a['split']}"
    cache = DATASETS / f"{name}_h{horizon}_L{'-'.join(f'{x:g}' for x in levels)}_labels.csv.gz"
    if cache.exists():
        lab = pd.read_csv(cache, index_col=0)
        lab.index = pd.to_datetime(lab.index, utc=True, format="ISO8601")
    else:
        print(f"making labels h={horizon} levels={levels} for {name} (once, cached in {cache.name}) ...", flush=True)
        base = load(a["split"], a["symbol"], a["tf"])
        lab, _ = make_labels(base, read_minutes(DATA / f"{a['symbol']}_1m_{a['split']}.csv.gz"), horizon, levels)
        DATASETS.mkdir(exist_ok=True)
        lab.to_csv(cache)
    return lab.reindex(df.index)


# ---------------------------------------------------------------- the environment
class Obs:
    """What a model sees at the close of bar t. Nothing in it looks past bar t.

    x          numpy (lookback, n_features): the last `lookback` bars of features, ending at bar t (fast)
    history    the same as a DataFrame with datetime index and column names (convenient)
    obs["ret_1"]  the value of one feature at bar t
    time       open time of bar t
    position   current position in [-1, 1] (0 = flat)
    bars_held  bars since the open trade was entered (0 when flat)
    trade_ret  return of the open trade so far, entry fill -> close of bar t, per unit and before fees (0.01 = 1 %)
    equity     account value, starts at 1
    """
    __slots__ = ("x", "time", "position", "bars_held", "trade_ret", "equity", "_env", "_i")

    @property
    def columns(self):
        return self._env.columns

    @property
    def history(self) -> pd.DataFrame:
        lo = self._i - len(self.x) + 1
        return pd.DataFrame(self.x, index=self._env.times[lo: self._i + 1], columns=self._env.columns)

    def __getitem__(self, col):
        return self.x[-1, self._env.col[col]]


class TradingEnv:
    """Bar-by-bar trading on one symbol. Gym-style: obs = env.reset(); obs, reward, done = env.step(position).

    The action is the target position in [-1, 1]: 1 = all-in long, -1 = all-in short, 0 = flat, 0.5 = half long.
    A decision made at the close of bar t fills at the open of bar t+1. Every change of position pays
    fee_bps per side on the traded size. Reward = the step's equity return (after fees).
    The last open position is closed at the final close (with its fee) when the episode ends.
    """

    def __init__(self, df: pd.DataFrame, lookback: int = 1, fee_bps: float = FEE_BPS):
        feats = features_only(df)
        self.x, self.columns = feats.to_numpy(dtype=float), list(feats.columns)
        self.col = {c: j for j, c in enumerate(self.columns)}
        self.times, self.lookback, self.fee = df.index, int(lookback), fee_bps / 1e4
        o, c = df["open"].to_numpy(dtype=float), df["close"].to_numpy(dtype=float)
        self.close = c
        self.fill = np.r_[o[1:], c[-1]]          # fill price of a decision at bar i: open of bar i+1 (last one: final close)
        if self.lookback < 1 or len(df) < self.lookback + 1:
            raise ValueError(f"need lookback >= 1 and more than lookback rows, got lookback={lookback}, rows={len(df)}")

    def reset(self, start: int | None = None, steps: int | None = None) -> Obs:
        """Start flat with equity 1 at bar `start` (default: the first bar with a full lookback), for `steps` decisions."""
        self.i = self.lookback - 1 if start is None else int(start)
        if not self.lookback - 1 <= self.i < len(self.x) - 1:
            raise ValueError(f"start must be in [{self.lookback - 1}, {len(self.x) - 2}], got {start}")
        self.end = len(self.x) - 1 if steps is None else min(self.i + int(steps), len(self.x) - 1)
        self.position, self.bars_held, self.equity = 0.0, 0, 1.0
        self.entry = self.eq_entry = self.entry_time = None
        self.curve, self.actions, self.trades = [(self.times[self.i], 1.0)], [], []
        return self._obs()

    def step(self, action):
        a = self._check(action)
        i, eq0 = self.i, self.equity
        self.actions.append((self.times[i], a))
        self._trade_to(a, self.fill[i], self.times[i + 1])
        self.equity *= 1 + a * (self.fill[i + 1] / self.fill[i] - 1)
        self.bars_held = self.bars_held + 1 if a else 0
        self.i = i + 1
        done = self.i >= self.end
        if done and a:
            self._trade_to(0.0, self.fill[self.i], self.times[self.i])
        self.curve.append((self.times[self.i], self.equity))
        return self._obs(), self.equity / eq0 - 1, done

    def _check(self, a) -> float:
        if isinstance(a, (bool, np.bool_)) or not isinstance(a, (int, float, np.integer, np.floating)) or not -1 <= a <= 1:
            raise ValueError(f"act() must return one number in [-1, 1] (the target position), got {a!r}")
        return float(a)

    def _trade_to(self, a, price, t):
        pos = self.position
        if a == pos:
            return
        if pos and (a == 0 or np.sign(a) != np.sign(pos)):          # close the open trade
            self.equity *= 1 - self.fee * abs(pos)
            self.trades.append({"entry_time": self.entry_time, "exit_time": t, "side": int(np.sign(pos)),
                                "entry": self.entry, "exit": price, "bars": self.bars_held,
                                "ret": self.equity / self.eq_entry - 1})
            pos = 0.0
        if a and not pos:                                               # open a new trade
            self.entry, self.eq_entry, self.entry_time, self.bars_held = price, self.equity, t, 0
        self.equity *= 1 - self.fee * abs(a - pos)
        self.position = a

    def _obs(self) -> Obs:
        o, i = Obs(), self.i
        o._env, o._i, o.time = self, i, self.times[i]
        o.x = self.x[i - self.lookback + 1: i + 1]
        o.position, o.bars_held, o.equity = self.position, self.bars_held, self.equity
        o.trade_ret = float(np.sign(self.position) * (self.close[i] / self.entry - 1)) if self.position else 0.0
        return o


# ---------------------------------------------------------------- the model contract
class Model:
    """Subclass this in submissions/<name>/model.py (exactly one subclass per file).

    lookback    bars of history in obs.x / obs.history
    model_type  ml | dl | rl | ea | ... (shown on the leaderboard)
    fit(train)  learn from the train DataFrame of bench.load(). For RL / EA: env = bench.TradingEnv(train, self.lookback)
    act(obs)    the target position in [-1, 1] after the close of bar obs.time
    save/load   default: pickle every attribute set in fit(). Override only for something pickle can't handle.
    """
    lookback = 1
    model_type = "?"
    author = "?"

    def fit(self, train: pd.DataFrame) -> None:
        pass

    def act(self, obs: Obs) -> float:
        raise NotImplementedError

    def save(self, folder) -> None:
        Path(folder).mkdir(parents=True, exist_ok=True)
        torch = sys.modules.get("torch")
        if torch is not None:                    # a GPU-trained network must load on a CPU-only grader
            for v in self.__dict__.values():
                if isinstance(v, torch.nn.Module):
                    v.cpu()
        joblib.dump(self.__dict__, Path(folder) / "model.joblib")

    def load(self, folder) -> None:
        self.__dict__.update(joblib.load(Path(folder) / "model.joblib"))


class LabelModel(Model):
    """For classifiers of make_labels.py classes: write fit() and predict(obs) -> class; act() trades it.

    Flat: open the class's trade (long/short, TP = its level; the two 0..L1 classes stay flat).
    In a trade: exit at TP or SL = levels[0], checked at each bar close, or after `horizon` bars.
    targets(train) gives the matching labels: (X rows mask, y).
    """
    horizon, levels = 12, [1, 2]

    def predict(self, obs: Obs) -> int:
        raise NotImplementedError

    def targets(self, train):
        lab = labels(train, self.horizon, self.levels)
        ok = (lab["lbl_missing_min"] == 0) & lab["lbl"].notna()       # cut-off windows are not labels
        return ok.to_numpy(), lab.loc[ok, "lbl"].astype(int).to_numpy()

    def act(self, obs):
        L = sorted(self.levels)
        if obs.position:
            r = obs.trade_ret * 100
            done = r >= self._tp or r <= -L[0] or obs.bars_held >= self.horizon
            return 0 if done else obs.position
        c = self.predict(obs)
        if isinstance(c, (bool, np.bool_)) or not isinstance(c, (int, np.integer)) or not 0 <= c < 2 * len(L) + 2:
            raise ValueError(f"predict() must return an int class in 0..{2 * len(L) + 1}, got {c!r}")
        side, tp = (int(v) for v in class_to_trade(c, len(L)))
        self._tp = L[max(tp, 1) - 1]
        return side


class BuyAndHold(Model):
    model_type, author = "benchmark", "bench"

    def act(self, obs):
        return 1


# ---------------------------------------------------------------- running and scoring
def load_submission(folder, trained=True) -> Model:
    """Import submissions/<name>/model.py, find its one Model subclass, and load artifacts/ when trained=True."""
    folder = Path(folder)
    name = f"submission_{folder.name}"
    spec = importlib.util.spec_from_file_location(name, folder / "model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                      # pickled objects find their classes again on load
    spec.loader.exec_module(mod)
    found = [c for c in vars(mod).values() if isinstance(c, type) and issubclass(c, Model) and c.__module__ == name]
    if len(found) != 1:
        raise ValueError(f"{folder / 'model.py'} must define exactly one Model subclass, found {len(found)}")
    model = found[0]()
    model.name = folder.name
    if trained:
        if not (folder / "artifacts").exists():
            raise FileNotFoundError(f"{folder.name} is not trained: python bench.py train {folder.as_posix()}")
        model.load(folder / "artifacts")
    return model


def seed(s=0):
    random.seed(s)
    np.random.seed(s)
    if "torch" in sys.modules:
        sys.modules["torch"].manual_seed(s)


def run(model: Model, df: pd.DataFrame, start=None, fee_bps=FEE_BPS) -> TradingEnv:
    """Trade `model` through `df`, one bar at a time. Returns the finished env (curve, trades, actions)."""
    env = TradingEnv(df, model.lookback, fee_bps)
    obs, done = env.reset(start), False
    while not done:
        try:
            obs, _, done = env.step(model.act(obs))
        except Exception as e:
            raise RuntimeError(f"{getattr(model, 'name', type(model).__name__)} failed at bar {obs.time}: {e}") from e
    return env


def metrics(env: TradingEnv) -> dict:
    eq = pd.Series(dict(env.curve))
    r = pd.Series([t["ret"] for t in env.trades], dtype=float)
    daily = eq.resample("1D").last().ffill()
    dr = (daily / daily.shift(1, fill_value=1.0) - 1).to_numpy()
    loss = -r[r < 0].sum()
    acts = np.array([a for _, a in env.actions])
    return {
        "total_return": eq.iloc[-1] - 1,
        "max_drawdown": (eq / eq.cummax() - 1).min(),
        "sharpe": dr.mean() / dr.std(ddof=1) * np.sqrt(365) if len(dr) > 1 and dr.std(ddof=1) > 0 else np.nan,
        "trades": len(r),
        "win_rate": (r > 0).mean() if len(r) else np.nan,
        "avg_trade_pct": r.mean() * 100 if len(r) else np.nan,
        "profit_factor": r[r > 0].sum() / loss if loss > 0 else np.nan,
        "exposure": (acts != 0).mean() if len(acts) else np.nan,
    }


def submissions_of(paths):
    return [Path(p) for p in paths] or sorted(p for p in SUBMISSIONS.iterdir() if (p / "model.py").exists())


# ---------------------------------------------------------------- reports and commands
def evaluate(models, history, period, prefix="", title=""):
    """Trade each (name, get_model) through `period`; `history` fills the lookback of the first decisions.

    Prints a leaderboard and writes results/{prefix}leaderboard.csv, equity.csv, equity.png, trades_<name>.csv.
    A model that fails gets an `error` row; the others still run.
    """
    RESULTS.mkdir(exist_ok=True)
    rows, curves = [], {}
    for name, get in models:
        t0 = time.perf_counter()
        try:
            model = get()
            seed()
            lb = model.lookback
            env = run(model, pd.concat([history.iloc[len(history) - lb + 1:], period]), start=lb - 1)
            curves[name] = pd.Series(dict(env.curve))
            pd.DataFrame(env.trades).to_csv(RESULTS / f"{prefix}trades_{name}.csv", index=False)
            rows.append({"model": name, "type": model.model_type, **metrics(env), "seconds": time.perf_counter() - t0})
        except Exception:
            traceback.print_exc()
            rows.append({"model": name, "error": traceback.format_exc(limit=1).strip().splitlines()[-1]})
        print(f"{name}: done", flush=True)

    board = pd.DataFrame(rows).set_index("model")
    board.to_csv(RESULTS / f"{prefix}leaderboard.csv")
    eq = pd.DataFrame(curves)
    eq.to_csv(RESULTS / f"{prefix}equity.csv")
    show = board.copy()
    for c in ["total_return", "max_drawdown", "win_rate", "exposure"]:
        if c in show:
            show[c] = show[c] * 100
    show = show.rename(columns={"total_return": "profit_%", "max_drawdown": "max_dd_%", "win_rate": "win_rate_%", "exposure": "exposure_%"})
    print(f"\n{title} {period.index[0]} -> {period.index[-1]} ({len(period):,} bars) | fee {FEE_BPS} bps per side")
    print(show.round(3).to_string())
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ax = eq.ffill().plot(figsize=(10, 4), lw=1, title=f"Equity: {title}")
        ax.set_ylabel("equity (start = 1)")
        plt.tight_layout()
        plt.savefig(RESULTS / f"{prefix}equity.png", dpi=120)
        plt.close()
    except ImportError:
        pass
    print(f"-> results/{prefix}leaderboard.csv, {prefix}equity.csv, {prefix}equity.png, {prefix}trades_<model>.csv")
    return board


def cmd_train(a):
    train = load("train", a.symbol, a.tf)
    for folder in submissions_of(a.submissions):
        model, t0 = load_submission(folder, trained=False), time.perf_counter()
        seed()
        model.fit(train)
        model.save(folder / "artifacts")
        print(f"{folder.name}: trained in {time.perf_counter() - t0:.0f} s -> {(folder / 'artifacts').as_posix()}/", flush=True)


def cmd_valid(a):
    """Fit on train minus its last `valid + gap` bars, then trade the last `valid` bars. Never touches artifacts/."""
    train = load("train", a.symbol, a.tf)
    fit_part, valid = train.iloc[:-(a.valid + a.gap)], train.iloc[-a.valid:]
    print(f"fit on {fit_part.index[0]} -> {fit_part.index[-1]}, gap {a.gap} bars, validate on the last {a.valid:,} bars", flush=True)

    def fitted(folder):
        def get():
            model = load_submission(folder, trained=False)
            seed()
            model.fit(fit_part)
            return model
        return get

    models = [("buy_and_hold", BuyAndHold)] + [(f.name, fitted(f)) for f in submissions_of(a.submissions)]
    evaluate(models, train.iloc[:-a.valid], valid, prefix="valid_", title="Validation")


def cmd_check(a):
    """Rule checks on the last 3000 train bars (the test file is never touched)."""
    df = load("train", a.symbol, a.tf).iloc[-3000:].copy()
    df["lbl_secret"] = np.arange(len(df))                        # a label column the models must never see
    o, c, f = df["open"].to_numpy(), df["close"].to_numpy(), FEE_BPS / 1e4
    failed = []

    def check(name, ok):
        print(f"{'ok  ' if ok else 'FAIL'} {name}", flush=True)
        if not ok:
            failed.append(name)

    class Spy(Model):
        lookback = 20

        def __init__(self):
            self.seen = []

        def act(self, obs):
            self.seen.append((obs.time, obs.history))
            return 0

    spy = Spy()
    run(spy, df)
    check("history ends at the decision bar", all(h.index[-1] == t for t, h in spy.seen))
    truth = features_only(df)
    check("history = the true features of those bars",
          all(np.array_equal(h.to_numpy(), truth.loc[h.index].to_numpy(dtype=float), equal_nan=True) for _, h in spy.seen[::97]))
    check("history never contains label columns", not any(col.startswith(LABEL_PREFIX) for _, h in spy.seen for col in h.columns))
    check("history length = lookback, first decision has a full history", all(len(h) == 20 for _, h in spy.seen) and spy.seen[0][1].index[0] == df.index[0])

    env = run(BuyAndHold(), df)
    check("buy and hold = (1 - fee) * last close / first fill * (1 - fee)", np.isclose(env.equity, (1 - f) ** 2 * c[-1] / o[1]))

    class Half(Model):
        def act(self, obs):
            return 0.5
    env = run(Half(), df)
    p = np.r_[o[1:], c[-1]]
    check("half position: equity = fees * product of (1 + 0.5 r)", np.isclose(env.equity, (1 - f / 2) ** 2 * np.prod(1 + 0.5 * (p[1:] / p[:-1] - 1))))

    class Flat(Model):
        def act(self, obs):
            return 0
    env = run(Flat(), df)
    check("flat: equity stays 1, no trades", env.equity == 1 and not env.trades)

    class Flipper(Model):
        def act(self, obs):
            return float(np.sign(obs["ret_1"]))
    env = run(Flipper(), df)
    check(f"equity = compounded trade returns ({len(env.trades)} trades with flips)", np.isclose(env.equity, np.prod([1 + t["ret"] for t in env.trades])))
    check("every trade enters and exits at a bar open", all(t["entry"] in p and t["exit"] in p for t in env.trades))

    for bad in (1.5, -2, float("nan"), "1", None, True, np.array([1.0])):
        class Bad(Model):
            def act(self, obs, v=bad):
                return v
        try:
            run(Bad(), df)
            rejected = False
        except RuntimeError:
            rejected = True
        check(f"rejects action {bad!r}", rejected)

    for folder in submissions_of(a.submissions):
        if not (folder / "artifacts").exists():
            print(f"skip {folder.name}: not trained")
            continue
        try:
            full, half = (run(load_submission(folder), d).actions for d in (df, df.iloc[:1500]))
            check(f"{folder.name}: runs, deterministic, decisions don't use future bars ({len(full)} decisions)", full[:len(half)] == half)
        except Exception as e:
            check(f"{folder.name}: runs ({e})", False)

    print("\nall checks passed" if not failed else f"\n{len(failed)} checks FAILED")
    sys.exit(1 if failed else 0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["train", "valid", "check"])
    p.add_argument("submissions", nargs="*", help="submission folders (default: every folder in submissions/)")
    p.add_argument("--symbol", default=SYMBOL)
    p.add_argument("--tf", default=TF, help="timeframe of the base files")
    p.add_argument("--valid", type=int, default=20_000, help="valid: bars at the end of train to validate on")
    p.add_argument("--gap", type=int, default=288, help="valid: bars left out between fit and validation (> label horizon)")
    a = p.parse_args()
    {"train": cmd_train, "valid": cmd_valid, "check": cmd_check}[a.command](a)


if __name__ == "__main__":
    main()
