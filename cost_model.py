"""
cost_model.py — explicit, parameterized transaction costs.

A backtest without costs is an upper bound, not an estimate. This model is
deliberately simple and its parameters are assumptions you must own:

    cost_fraction(trade) = half_spread + impact_coef * sqrt(participation)

  * half_spread: half the bid/ask spread as a fraction of price. For S&P 500
    large caps this is small (roughly a basis point or two for the megacaps,
    more for the smaller/less liquid names). The default below is a
    deliberately conservative flat assumption, NOT a measured number — if you
    have quote data, measure per-name spreads instead.
  * impact: square-root market-impact form, a standard practitioner shape
    (cost grows with the square root of your share of daily volume). The
    coefficient is regime- and venue-dependent; treat the default as an
    order-of-magnitude placeholder to be calibrated against your own fills
    (your Alpaca paper fills CANNOT calibrate it — paper fills are simulated).

Sensitivity discipline: always run the backtest at 0.5x, 1x, and 2x these
costs. If the strategy only survives at 0.5x, it does not survive.
"""
from __future__ import annotations

import numpy as np

DEFAULT_HALF_SPREAD = 0.0003   # 3 bps — conservative flat assumption for large caps
DEFAULT_IMPACT_COEF = 0.10     # sqrt-impact coefficient — placeholder, calibrate
DEFAULT_PARTICIPATION_CAP = 0.05  # refuse to model trading >5% of ADV in one day


def trade_cost_fraction(trade_dollars, adv_dollars,
                        half_spread=DEFAULT_HALF_SPREAD,
                        impact_coef=DEFAULT_IMPACT_COEF,
                        participation_cap=DEFAULT_PARTICIPATION_CAP):
    """Cost as a fraction of |trade_dollars|. Vectorized, nan-safe.

    If participation exceeds `participation_cap`, cost is set to nan rather
    than extrapolated — the model is not credible there and the position is
    too big for the strategy's capacity. The backtester treats nan as
    'trade disallowed', which doubles as a crude capacity constraint.
    """
    t = np.abs(np.asarray(trade_dollars, dtype="float64"))
    adv = np.asarray(adv_dollars, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        part = np.where(adv > 0, t / adv, np.inf)
    cost = half_spread + impact_coef * np.sqrt(np.maximum(part, 0.0))
    cost = np.where(part > participation_cap, np.nan, cost)
    return cost


def apply_costs(trades: dict, adv: dict, **kw) -> tuple[float, dict]:
    """trades: ticker -> signed dollars. adv: ticker -> $ADV.
    Returns (total_cost_dollars, per_name_cost). Names with unmodelable cost
    (nan) are returned with cost=None so the caller can veto the trade."""
    total, per = 0.0, {}
    for tk, d in trades.items():
        c = trade_cost_fraction(d, adv.get(tk, np.nan), **kw)
        c = float(c) if np.isfinite(c) else None
        per[tk] = c
        if c is not None:
            total += abs(d) * c
    return total, per
