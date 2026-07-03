"""
exposure_core.py — volatility-targeted gross exposure.

IDEA
----
Instead of binary exposure (fully invested until a drawdown breaker fires),
scale gross exposure continuously so the PORTFOLIO'S expected volatility sits
near a target:

    scale = clip( target_vol / estimated_vol , min_scale , max_scale )

Two estimators, used together (conservatively — the final scale is the MIN):

  * EX-ANTE: sqrt(w'Σw) annualized, from the same shrunk trailing covariance
    the weights were built with. Available from day one; reacts to what you
    actually hold.
  * REALIZED: trailing standard deviation of the strategy's OWN past period
    returns. Needs history; catches risk the covariance missed (estimation
    error, regime shift already visible in your P&L).

Taking the min means either estimator can force de-risking but both must
agree before running at full size. That is a deliberately conservative choice
(it will sometimes leave return on the table); if you disagree, swap `min`
for the ex-ante-only version consciously rather than by accident.

WHY (and honest limits)
-----------------------
There is published academic work finding that volatility-managed versions of
equity factor portfolios improved risk-adjusted returns historically — I
believe the best-known reference is Moreira & Muir, "Volatility-Managed
Portfolios" (Journal of Finance, 2017); verify that citation before relying
on it, and note subsequent papers have debated how robust and implementable
the effect is after costs. What vol-targeting reliably does is shape RISK
(steadier vol, typically shallower drawdowns in vol-clustered markets). Any
raw-return improvement is regime-dependent and NOT guaranteed. It also adds
turnover (scaling trades cost money) — judge it in the backtest net of the
cost model, not on gross returns.

Anti-lookahead: both estimators use strictly past data (trailing covariance,
past realized returns). Nothing here touches the forward window.
"""
from __future__ import annotations

import numpy as np


def scale_from_ex_ante(sigma_period, target_ann_vol, periods_per_year,
                       min_scale=0.25, max_scale=1.0):
    """sigma_period: ex-ante per-period vol sqrt(w'Σw) at the trailing-return
    frequency. Returns exposure multiplier in [min_scale, max_scale]."""
    if not np.isfinite(sigma_period) or sigma_period <= 0:
        return 1.0
    ann = sigma_period * np.sqrt(periods_per_year)
    return float(np.clip(target_ann_vol / ann, min_scale, max_scale))


def scale_from_realized(past_returns, target_ann_vol, periods_per_year,
                        lookback=12, min_obs=6, min_scale=0.25, max_scale=1.0):
    """Trailing realized vol of the strategy's own past per-period returns.
    Returns 1.0 (no opinion) until min_obs observations exist — early periods
    therefore run at whatever the ex-ante estimator says."""
    r = np.asarray(past_returns, float)
    r = r[np.isfinite(r)][-lookback:]
    if len(r) < min_obs:
        return 1.0
    ann = r.std(ddof=1) * np.sqrt(periods_per_year)
    if ann <= 0:
        return 1.0
    return float(np.clip(target_ann_vol / ann, min_scale, max_scale))


def combined_scale(sigma_period, past_returns, target_ann_vol,
                   periods_per_year, **kw):
    """Conservative combination: min(ex-ante, realized)."""
    return min(scale_from_ex_ante(sigma_period, target_ann_vol,
                                  periods_per_year,
                                  min_scale=kw.get("min_scale", 0.25),
                                  max_scale=kw.get("max_scale", 1.0)),
               scale_from_realized(past_returns, target_ann_vol,
                                   periods_per_year,
                                   lookback=kw.get("lookback", 12),
                                   min_obs=kw.get("min_obs", 6),
                                   min_scale=kw.get("min_scale", 0.25),
                                   max_scale=kw.get("max_scale", 1.0)))
