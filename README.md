# bench: trading model standard

Build a trading model of any kind (machine learning, deep learning, reinforcement learning or evolutionary) in one file. It trades one crypto symbol (`symbol5`) on 5-minute bars. You train it on the train data. Your trained model is then run through a hidden test period: the six months right after the train data ends, in the same trading environment and with the same rules.

This README is the full specification. A Persian guide with the same content and complete examples is shared with the data. You can give either one, with `bench.py`, to an assistant or LLM to help you build a model.

```
bench.py                     the standard: data, TradingEnv, Model, LabelModel, train / valid / check
make_labels.py               TP/SL labels for supervised models (used through bench, no need to run it)
submissions/
  ml_logreg/model.py         sample: logistic regression (ML)
  dl_lstm/model.py           sample: LSTM (DL)
  rl_qlearn/model.py         sample: Q-learning (RL)
```

## 1. Setup

Use Python 3.12 and run every command from this folder.

```bash
pip install -r requirements.txt
```

Without a GPU, the smaller torch build is enough: `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu`

## 2. Data

You get two files. Put them in `data/`:

| file | contents |
|---|---|
| `data/symbol5_5m_train.csv.gz` | 316,224 five-minute bars, 2023-05-01 to 2026-05-02, with 130 columns. Train on this. |
| `data/symbol5_1m_train.csv.gz` | 1-minute OHLCV of the same period, used only to build labels |

- The first column is the bar open time in UTC. Each row is known at the close of its bar, so no feature looks ahead.
- `open/high/low/close` are the real price times one fixed constant, so the series starts near 100. Returns, ratios and % moves are exactly real. The scale is the same across train, test and the 1-minute file, and trading PnL is computed from price ratios. Only the dollar level is hidden, so a raw price level (e.g. "buy if close > 150") has no economic meaning: use changes and ratios (returns, distance to a moving average in %, range / close, ...).
- `volume` is relative (1 = a normal bar).
- The other columns are ready-made relative features (see the appendix). Some are NaN at the start of the data.

## 3. Workflow

```bash
python bench.py train submissions/my_model    # fit on all train data, save to submissions/my_model/artifacts/
python bench.py valid submissions/my_model    # fit on train minus its last 20,000 bars, score those bars
python bench.py check submissions/my_model    # rule checks: every line must say ok
```

Leave out the folder to run every submission. `valid` doesn't touch `artifacts/`, prints the same report your model gets on the test period, and adds buy and hold as a benchmark. It writes `results/valid_*`. Use it to compare ideas, then run `train` once more for the final model.

## 4. The contract

A submission is a folder `submissions/<name>/` with one file you write, `model.py`. It defines exactly one class that extends `bench.Model`:

```python
import bench

class MyModel(bench.Model):
    model_type, author = "ml", "your name"     # ml | dl | rl | ea
    lookback = 1                                # bars of history in obs

    def fit(self, train):                       # train: DataFrame of the train file
        ...                                     # put everything you learn on self

    def act(self, obs):                         # called after every bar
        return 1                                # target position in [-1, 1]
```

- `fit(train)` is called once with the whole train DataFrame. You choose how to split, validate or cross-validate inside it. For time series, keep the order and leave a gap larger than the label horizon between parts.
- `act(obs)` returns the **target position**: 1 = all-in long, -1 = all-in short, 0 = flat, 0.5 = half long. It is not an order to add to the position. Returning the current position means hold.
- `save` and `load` are automatic: everything set on `self` in `fit` is pickled to `artifacts/model.joblib`. Torch networks stored on `self` are moved to the CPU before saving. Override `save(folder)` and `load(folder)` only for objects that can't be pickled, or for libraries with their own save format.

### obs: what the model sees after each bar

| field | meaning |
|---|---|
| `obs["ret_1"]` | one feature at the current bar (fast) |
| `obs.history` | DataFrame of the last `lookback` bars, ending at the current bar |
| `obs.x` | the same as a numpy array of shape (lookback, n_features) (fast) |
| `obs.columns` | feature names, in the order of the columns of `obs.x` |
| `obs.position` | current position in [-1, 1] |
| `obs.bars_held` | bars since the open trade was entered (0 when flat) |
| `obs.trade_ret` | return of the open trade so far, per unit and before fees (0.01 = 1 %) |
| `obs.equity` | account value, starts at 1 |
| `obs.time` | open time of the current bar |

### Trading rules

- A decision made after bar t closes fills at the open of bar t+1.
- Every change of position pays 0.1 % of the traded size. Flipping from 1 to -1 trades 2 units and pays 0.2 %. Going from 0.5 to 1 trades 0.5 units.
- If the sign changes or the position goes to 0, the trade closes. Changing the size on the same side keeps the same trade.
- Equity is marked bar by bar. A position still open at the end is closed at the last close.
- The model never sees future bars or the `lbl_*` label columns.

### Scoring

The test period is scored with these metrics: `total_return`, `max_drawdown`, `sharpe` (daily, annualized), `trades`, `win_rate`, `avg_trade_pct`, `profit_factor` and `exposure`, next to buy and hold. `valid` prints exactly this report for the end of your train data.

## 5. Recipes

**Supervised with TP/SL labels (ML, DL).** Extend `bench.LabelModel` and write `predict(obs) -> int class` instead of `act`. With `horizon, levels = 12, [1, 2]` there are 6 classes: 0 `dn>2`, 1 `dn1-2`, 2 `dn0-1`, 3 `up0-1`, 4 `up1-2`, 5 `up>2`. When flat, `LabelModel.act` opens the class's trade: `up1-2` is long with TP 1 %, `dn>2` is short with TP 2 %, and the two `0-1` classes stay flat. It exits at TP, at SL = the first level, or after `horizon` bars. TP and SL are checked at each bar close. In `fit`, `ok, y = self.targets(train)` gives a mask of rows with a complete label window and their classes. Labels are built once and cached in `datasets/`. See `ml_logreg` and `dl_lstm`.

**Supervised with your own target.** Extend `bench.Model`, build any target from future prices of `train` inside `fit`, and write `act` yourself.

**Reinforcement learning.** Make the environment from the train data inside `fit`. See `rl_qlearn`.

```python
env = bench.TradingEnv(train, self.lookback)
obs = env.reset(start=0, steps=2000)            # flat, equity 1, `steps` decisions from bar `start`
obs, reward, done = env.step(action)            # reward = the step's equity return after fees
```

Libraries such as stable-baselines3 expect a `gymnasium.Env`. Wrap `TradingEnv` in a small `gymnasium.Env` subclass inside `model.py`, and use the library's own save and load in `save` and `load`.

**Evolutionary.** Score each candidate (a `bench.Model` instance) in the same environment, and keep the best parameters on `self`:

```python
fitness = bench.metrics(bench.run(candidate, train.iloc[-20_000:]))["sharpe"]
```

Evolve on one part of train and select on a later part, or the winner fits noise.

## 6. bench API

| name | what it does |
|---|---|
| `bench.load(split="train")` | the train DataFrame, as `fit` receives it |
| `bench.labels(df, horizon, levels)` | label columns (`lbl`, `lbl_ret_h`, `lbl_mfe`, `lbl_mae`, `lbl_missing_min`, ...) for the rows of `df` |
| `bench.TradingEnv(df, lookback)` | the environment: `reset(start, steps)`, `step(position)` |
| `bench.run(model, df)` | trade a model through `df` and return the finished env (`env.curve`, `env.trades`, `env.actions`) |
| `bench.metrics(env)` | the scoring metrics of a finished env |
| `bench.load_submission(folder, trained=True)` | import a submission, and load its artifacts when `trained=True` |
| `bench.seed()` | the fixed seeds `train` uses (random, numpy, torch) |
| `bench.class_to_trade(c, K)` | label class -> (side, TP level index) |

## 7. What to hand in

Train your final model with `python bench.py train submissions/<name>`, then check that `python bench.py check submissions/<name>` prints only `ok`. Zip the folder and hand it in:

```
submissions/<name>/          use your student ID as <name>: lowercase, no spaces
  model.py                   your code, with exactly one Model subclass
  artifacts/model.joblib     written by bench.py train (or your own files if you override save/load)
```

The evaluator copies the folder into its own `submissions/`, loads it with `bench.load_submission`, and runs `act` bar by bar through the test period. The evaluator uses an unmodified `bench.py`, so everything your model needs must be in your folder.

## 8. Rules

- Learn from the train data only.
- `act` uses only `obs` and what `fit` learned. It doesn't read files, use the network, or behave randomly. The same input gives the same output.
- Keep `act` fast. The test period has about 53,000 bars, so 1 ms per bar is about a minute.
- Keep only what `act` needs on `self`. Don't store the train data.
- If you need a package that isn't in `requirements.txt`, write its name and exact version in the docstring at the top of `model.py`.

## 9. Common errors

| message | fix |
|---|---|
| `must define exactly one Model subclass` | Only your main class extends `bench.Model`. Helper classes don't. |
| `act() must return one number in [-1, 1]` | Return a single number: `float(...)`, or `.item()` for a tensor. |
| `predict() must return an int class` | Return `int(...)`. |
| `is not trained` | Run `python bench.py train` first. |
| `labels() needs a DataFrame from bench.load()` | Pass `train`, or a slice of it, to `targets` and `labels`. |
| `KeyError` in `obs[...]` | Wrong feature name. See `obs.columns`. |
| pickle error on save | Don't keep lambdas, generators or open files on `self`, or override `save` and `load`. |
| `FAIL ... deterministic ... future bars` | Remove randomness from `act`, and decide only from `obs` and trained parameters. |

## Appendix: the 130 columns

The first 100 columns:

`open, high, low, close, volume, ret_1, ret_24, gap, upper_wick, lower_wick, range, vol_24, qvol_chg, qvol_z24, qvol_z168, trades_chg, trades_z168, taker_ratio, avg_trade_chg, ret_1_vs_symbol6, ret_24_vs_symbol6, fund_rate8h, fund_z30d, prem, prem_8, prem_z168, perp_spot_qvol24, perp_taker_ratio, oi_chg_1, oi_chg_24, oi_z168, oi_to_vol24, ls_top_pos, ls_top_pos_z168, ls_top_acc, ls_top_acc_z168, ls_all_acc, ls_all_acc_z168, ob_imb1, ob_imb2, ob_imb5, ob_depth1_z168, ob_depth5_z168, hour, weekday, ret_1_rv168, symbol6_ret_1_rv168, symbol7_ret_1_rv168, symbol8_ret_1_rv168, mkt_ret_1_rv168, ret_1_vs_symbol6_rv168, z_ret_24, z_ret_72, z_ret_168, range_pos_336, wick_imb_rv168, mkt_breadth_24, symbol6_prem_z168, mkt_fund_rate8h, log_vol_24, log_vol_168, log_range, log_mkt_vol_24, hour_sin, hour_cos, weekend, oi_chg_1_sd168, oi_to_vol24_rel720, perp_spot_qvol24_rel720, ob_missing, atr_pct_14, absret_ema_ratio_20_100, vol_ratio_20_100, range_compression_20_100, vol_of_vol_ratio_20, vol_regime_pct_120, atr_ratio_14_63, vol_over_median20, dollar_vol_rel_20, volume_percentile_60, amihud_illiquidity_20_z168, upside_semivol_20, semivariance_balance_20, drawup_96_atr14, price_z_20, garman_klass_cc_ratio_20, efficiency_ratio_20, tsi_25_13, close_rank_96, up_volume_share_20, vortex_balance_14, efficiency_ratio_96, di_balance_14, drawdown_96_atr14, volume_cv_20, price_volume_corr_32, ultimate_oscillator_7_14_28, return_z_96_96, body_signed_atr14, trend_slope_t_32`

Then 30 features built from this symbol's own OHLCV only. They are all relative (ATR units, ratios, ranks, correlations, fractions of bars) and causal. A few of them only exist once their window has filled, and `liquidity_pool_dist_96` is NaN whenever there is no swing pivot on one side of the price.

| column | meaning |
|---|---|
| `delta_div_20` | z-score of buy/sell volume pressure minus z-score of the 20-bar return (flow vs price divergence) |
| `absorption_score_100` | volume rank x (1 - range rank) over 100 bars: heavy volume with little movement |
| `nvi_quiet_ret_48` | sum of log returns over the last 48 bars, counting only bars whose volume fell |
| `sweep_bear_pressure_10` | failed breaks above the 20-bar high (rejection depth in ATR), summed over 10 bars |
| `sweep_bull_pressure_10` | failed breaks below the 20-bar low (rejection depth in ATR), summed over 10 bars |
| `liquidity_pool_dist_96` | (d_up - d_down) / (d_up + d_down) to the nearest confirmed swing high above and low below, last 96 bars |
| `inside_bar_break_5` | signed breakout of inside bars in ATR, summed over 5 bars |
| `engulfing_score_20` | signed engulfing candle x body share x volume ratio (0 if no pattern) |
| `nr7_break_5` | signed breakout of narrowest-range-of-7 bars in ATR, summed over 5 bars |
| `choppiness_14` | Choppiness Index / 100: high = sideways, low = trending |
| `hurst_100` | Hurst exponent of the last 100 returns minus 0.5: > 0 trending, < 0 mean-reverting |
| `range_acf1_40` | lag-1 autocorrelation of bar range over 40 bars |
| `ret_skew_50`, `ret_kurt_50` | skewness and excess kurtosis of log returns over 50 bars |
| `ret_acf1_96`, `ret_acf5_96` | lag-1 and lag-5 autocorrelation of log returns over 96 bars |
| `vol_range_corr_50` | correlation of volume and bar range over 50 bars |
| `vol_acf1_40` | lag-1 autocorrelation of volume over 40 bars |
| `jump_freq_100` | share of the last 100 returns that are outliers (median/MAD rule) |
| `perm_entropy_30` | permutation entropy of returns over 30 bars, 0..1 (1 = no order) |
| `inside_bar_freq_30`, `doji_freq_30` | share of inside bars / doji bars in the last 30 |
| `pivot_age_diff_96` | (bars since last swing high - bars since last swing low) / 96, in [-1, 1] |
| `wick_top_frac_20`, `wick_bottom_frac_20`, `both_wicks_frac_20` | share of bars in the last 20 with a long upper wick / long lower wick / both wicks |
| `range_med_ratio_50` | median / mean bar range over 50 bars: low = a few spiky bars |
| `max_up_streak_24` | longest run of higher closes in the last 24 bars / 24 |
| `extreme_print_frac_20` | share of the last 20 bars that set a new 10-bar high or low |
| `vol_skew_50` | skewness of log volume over 50 bars |
