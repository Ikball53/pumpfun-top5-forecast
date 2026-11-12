# pumpfun-top5 — make.fun entry

Forecast the top 5 new Solana coins by market cap, 24 hours out
(competition: "Predict the top 5 new Solana coins by market cap, 24 hours out").

## Program

`main.py` (Python 3.13, standard library only) reads `task.json` + `history.ndjson`
from `$MAKEFUN_INPUT_DIR` (default `/input`) and writes `forecast.json` to
`$MAKEFUN_OUTPUT_DIR` (default `/output`).

## Model (v3)

Per-coin OLS log-cap drift (dex/hour), shrunk toward zero (`n/(n+6)`), clamped
to ±0.25 dex/h and to ±0.15 dex total; curve coins decay 0.08 dex/h, pool coins
0.03 dex/h after 2 h; realised 10-minute volatility shrunk to venue priors
(pool 0.34, curve 0.094 dex) and capped at 0.50 dex/hour; total sd capped at
0.90 dex. Monte Carlo (6,000 scenarios, fixed seed 20261112) simulates each
answer read; each read also simulates the top 3 newborn contenders (coins aged
< 60 min), drawn as the top order statistics of the young coins visible per
mark (Beta quantiles through the empirical young-coin cap distribution), since
new launches dominate the future top 5. Place probabilities are Laplace
smoothed (3% over named coins); sizes mix the Monte Carlo histogram 90/10
with a broad Normal (sd 0.60 dex); probabilities are floor-6 rounded and the
output is sorted deterministically, so two runs on the same input are
byte-identical.

## Backtest (practice window, 10-minute horizon)

Replicating `grader/forecast/ranked.ts` in Python, training on marks 0-3 and
scoring on mark 4, and training on marks 0-4 and scoring on mark 5:

| test | this program (order/size/total) | house baseline |
|---|---|---|
| mark 4 | -16.32 / -8.39 / **-24.71** | -15.07 / -14.58 / **-29.65** |
| mark 5 | -13.06 / -10.67 / **-23.73** | -15.25 / -14.62 / **-29.87** |

Beats the house baseline on both backtests (+4.9 and +6.1 nats), mostly on
size calibration. Caveat: the practice backtest is a 10-minute horizon; the
final runs 7-13 hours out, where newborn turnover matters more and the model's
long-horizon behaviour is unvalidated locally (the kit ships no reference data).
