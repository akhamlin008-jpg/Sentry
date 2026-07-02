# Sentry — Enhanced Research & Backtest Layer

Drop-in modules for the existing repo. They import your unmodified
`factor_core.py` and `risk_core.py`; nothing in your current code needs to
change to adopt them.

## What's here

| File | Purpose |
|---|---|
| `pit_layer.py` | Point-in-time snapshot contract + validators that **refuse to run** on data with lookahead or silently-dropped delistings. Includes a build guide for real PIT data (EDGAR accession dates, historical constituents, delisting returns). |
| `backtest_core.py` | Walk-forward backtester: score → hysteresis selection → trailing-only shrunk covariance → constrained weights → optional short overlay → circuit breaker → costs → realize forward returns. Forward returns are only touched *after* weights freeze. |
| `signal_research.py` | Rank-IC per factor group with Newey-West t-stats, IC decay across horizons, factor-score correlation matrix, and beta/size/sector neutralization (residualization). |
| `portfolio_core.py` | Hysteresis (enter top-20 / exit below top-40), enforced caps (per-name weight, sector weight, per-name risk contribution), **risk-reducing short overlay**, drawdown circuit breaker with hysteresis. |
| `cost_model.py` | Half-spread + √-participation impact model with a hard capacity veto (>5% ADV → trade disallowed). Parameters are labeled assumptions; run 0.5×/1×/2× sensitivity. |
| `test_enhanced.py` | 13 offline tests against closed-form / construction-guaranteed answers. |
| `demo_synthetic_backtest.py` | End-to-end run on **synthetic** data with a planted signal — proves the machinery, proves nothing about real edge. |

## Shorts policy (as requested)

Shorts are off by default (`allow_shorts=False`). When enabled, a name is
shorted only if **all** of:

1. the model dislikes it (composite in the bottom 25th percentile —
   configurable);
2. it co-moves with the long book — formally `(Σw)ᵢ > 0` — so a short of the
   variance-minimizing size `h* = (Σw)ᵢ / Σᵢᵢ` strictly reduces estimated
   portfolio volatility;
3. the improvement clears a minimum threshold and per-name / gross-short caps
   are respected. Selection is greedy on vol reduction, recomputing after
   each accepted short.

In the synthetic demo this behaved exactly as designed: annualized vol fell
(~11.6% → ~8.8%) and max drawdown fell (~7.0% → ~5.1%) at the price of lower
return — a hedge, not an alpha source. **Caveats:** "reduces risk" means
reduces variance under a shrunk *sample* covariance; correlations can break
in crises, shorts carry borrow costs and unbounded-loss mechanics the
covariance can't see, and hard-to-borrow names must be filtered upstream.

## What the demo run showed (synthetic data — not evidence of real edge)

- Hysteresis (enter 20 / exit 40) cut monthly turnover roughly in half vs
  re-buying a fresh top-20 (~0.34 vs ~0.75 in this simulation). The magnitude
  on real data depends on your signal's autocorrelation — measure it.
- Even with a **planted** premium, 24 monthly periods produced an active
  t-stat of ~0.26 — statistically nothing. This is the honest headline: at
  monthly frequency, distinguishing a modest real edge from luck takes far
  more history than a paper account will give you, which is exactly why the
  PIT backtest over long history is the priority.

## What this deliberately does NOT solve (the remaining hard 20%)

1. **Real point-in-time data.** `pit_layer` enforces the contract but cannot
   fetch history. You must build snapshots from EDGAR XBRL facts keyed by
   accession acceptance date, a dated historical constituents table, and
   delisting returns. This is days of careful work and it is where most of
   the remaining fake-alpha risk lives. Free sources for historical
   constituents/delistings are of uncertain accuracy — spot-check before
   trusting.
2. **Analyst estimates in history.** There is no free PIT archive of
   consensus estimates I'm aware of, so `analyst_g5` is banned from
   historical snapshots; the live model may still use it, meaning backtest
   and live signal will differ slightly. Document that gap rather than
   papering over it.
3. **Cost calibration.** The defaults are placeholders. Paper fills cannot
   calibrate them (they're simulated).
4. **Statistical significance.** No code fixes a small sample. Pre-register
   your config before running on real data, split history for weight
   selection vs confirmation, and count every variant you tried.

## Run it

```bash
pip install numpy pandas pytest
python -m pytest test_enhanced.py -q     # 13 passed
python demo_synthetic_backtest.py        # synthetic end-to-end
```

Wiring real data: build `PITSnapshot`s (see bottom of `pit_layer.py`), then
`backtest_core.run_backtest(snaps, ...)` and `summarize(...)`, and run
`signal_research.ic_series` / `ic_decay` / `group_score_correlation` before
trusting any composite weights.
