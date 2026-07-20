"""
arb_core.py — three relative-value scanners for the Arbitrage page.

HONESTY HEADER — none of these is riskless arbitrage.
-----------------------------------------------------
True arbitrage (same cash flows, different prices, locked profit) is
essentially absent from free daily-close equity data. What CAN be measured
here are three statistical convergence trades, each with a failure mode:

1. RELATIVE VALUE (within-sector pairs). Spread z-score on a hedged log-price
   ratio. Fails when the spread is wide for a REASON (fraud, guidance cut,
   index deletion) — divergence risk is unbounded. A z-score is not a
   cointegration proof; the AR(1) half-life shown is the sanity check: no
   finite half-life, no trade.

2. LEVERAGED-ETF DECAY HARVEST — what "perpetual motion" actually is.
   Shorting BOTH legs of a ±3x sibling pair (e.g. TQQQ+SQQQ) collects the
   daily-reset volatility decay that bleeds from both funds. It looks like a
   money machine in backtests because decay is mathematically real — but the
   income is compensation for real risks: strong TRENDS make one leg grow
   faster than the other decays (2020, 2023 tech runs hurt), borrow fees on
   inverse-leveraged funds can eat the whole edge (often several % a year —
   check your broker's actual rate, it is not in this data), and shorts have
   unbounded loss with buy-in risk. The scanner reports historical windows
   where the trade LOST alongside the average, and assumes daily rebalancing
   back to equal legs — without which the position drifts directional.

3. SECTOR CORRELATION REVERSION. Short-window sector-pair correlation vs its
   own long baseline. A collapsed correlation flags a candidate reconvergence
   (long laggard / short leader within the pair); an abnormally tight one
   flags crowding. Correlations are regime-dependent — 2008/2020 style
   everything-to-1 events are exactly when this signal is wrongest.

All functions take a wide close DataFrame (dates x tickers) so they run on
either the live 3y cache or the long history parquet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import etf_universe as eu


# --------------------------------------------------------------------------- #
# 1. relative value pairs
# --------------------------------------------------------------------------- #
def _half_life(spread: np.ndarray) -> float:
    """AR(1) mean-reversion half-life in days; inf if not mean-reverting."""
    s = spread[np.isfinite(spread)]
    if len(s) < 40:
        return np.inf
    x, y = s[:-1] - s[:-1].mean(), np.diff(s)
    denom = (x * x).sum()
    if denom <= 0:
        return np.inf
    b = (x * y).sum() / denom
    if b >= 0:
        return np.inf
    return float(np.log(2) / -np.log(1 + b)) if (1 + b) > 0 else np.inf


def relative_value_pairs(close: pd.DataFrame, sectors: dict[str, str],
                         adv: pd.DataFrame | None = None,
                         lookback: int = 252, corr_floor: float = 0.60,
                         max_names_per_sector: int = 25,
                         top: int = 25) -> pd.DataFrame:
    """Within-sector pair dislocations, ranked by |z| of the hedged spread."""
    px = close.iloc[-lookback:].dropna(axis=1, thresh=int(lookback * 0.9))
    cols = [c for c in px.columns if c in sectors]
    if adv is not None:                       # keep the most liquid names/sector
        last_adv = adv.iloc[-1].reindex(cols).fillna(0)
    else:
        last_adv = pd.Series(1.0, index=cols)
    lp = np.log(px[cols])
    rets = lp.diff().iloc[1:]

    rows = []
    for sec in sorted(set(sectors[c] for c in cols)):
        names = sorted([c for c in cols if sectors[c] == sec],
                       key=lambda t: -last_adv[t])[:max_names_per_sector]
        if len(names) < 2:
            continue
        C = rets[names].corr().values
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if C[i, j] < corr_floor:
                    continue
                a, b = names[i], names[j]
                x, y = lp[b].values, lp[a].values
                beta = np.polyfit(x, y, 1)[0]
                spread = y - beta * x
                mu, sd = spread.mean(), spread.std(ddof=1)
                if sd <= 0:
                    continue
                z = (spread[-1] - mu) / sd
                hl = _half_life(spread)
                if abs(z) < 1.0 or not np.isfinite(hl) or hl > lookback / 2:
                    continue
                rows.append({"sector": sec, "rich": a if z > 0 else b,
                             "cheap": b if z > 0 else a, "pair": f"{a}/{b}",
                             "z": float(z), "beta": float(beta),
                             "half_life_days": round(hl, 1),
                             "corr": float(C[i, j])})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (df.reindex(df.z.abs().sort_values(ascending=False).index)
              .head(top).reset_index(drop=True))


def pair_spread_series(close: pd.DataFrame, a: str, b: str,
                       lookback: int = 252) -> pd.Series | None:
    """Z-scored spread history for charting one pair."""
    px = close[[a, b]].dropna().iloc[-lookback:]
    if len(px) < 60:
        return None
    lp = np.log(px)
    beta = np.polyfit(lp[b].values, lp[a].values, 1)[0]
    s = lp[a] - beta * lp[b]
    return (s - s.mean()) / s.std(ddof=1)


# --------------------------------------------------------------------------- #
# 2. leveraged-ETF decay harvest ("perpetual motion")
# --------------------------------------------------------------------------- #
def decay_harvest(close: pd.DataFrame, window: int = 63,
                  pairs=None) -> pd.DataFrame:
    """Historical P&L of a STATIC short of both legs of each leveraged sibling
    pair, 50/50 at entry, HELD for `window` days, per $1 gross short:

        pnl = -0.5 * (P_bull(T)/P_bull(0) - 1) - 0.5 * (P_bear(T)/P_bear(0) - 1)

    Static holding is essential: with DAILY rebalancing back to equal legs the
    daily returns of a perfect +Lx/-Lx pair cancel exactly and the trade earns
    zero by construction — the volatility decay accrues only to the position
    you LEAVE ALONE. The price of leaving it alone is drift: after a trend you
    are net short the winner with growing exposure (that is the risk being
    paid for). Rolling windows below overlap; the vol figure uses
    NON-overlapping windows to avoid understating it. Borrow fees are NOT
    modeled (no free data) — subtract your broker's actual rate for both legs
    before believing any number here; on inverse-leveraged funds it is often
    large enough to change the sign."""
    pairs = pairs or eu.DECAY_PAIRS
    rows = []
    for bull, bear in pairs:
        if bull not in close.columns or bear not in close.columns:
            rows.append({"pair": f"{bull}+{bear}", "status": "no data",
                         "note": "run crisis.yml to fetch ETF history"})
            continue
        px = close[[bull, bear]].dropna()
        if len(px) < window + 10:
            rows.append({"pair": f"{bull}+{bear}", "status": "history too short"})
            continue
        roll = (-0.5 * (px[bull] / px[bull].shift(window) - 1)
                - 0.5 * (px[bear] / px[bear].shift(window) - 1)).dropna()
        nonov = roll.iloc[::window]                 # non-overlapping sample
        lev = eu.LEVERAGED.get(bull, {}).get("lev", "?")
        rows.append({
            "pair": f"{bull}+{bear}", "status": "ok",
            "underlying": eu.LEVERAGED.get(bull, {}).get("underlying", "?"),
            "leverage": f"±{abs(lev)}x" if lev != "?" else "?",
            "days_of_data": int(len(px)),
            f"median_{window}d_ret": float(roll.median()),
            f"pct_windows_positive": float((roll > 0).mean() * 100),
            f"worst_{window}d_ret": float(roll.min()),
            "worst_window_end": str(roll.idxmin().date()),
            "ann_vol_of_trade": float(nonov.std(ddof=1)
                                      * np.sqrt(252 / window)) if len(nonov) > 2 else np.nan,
        })
    return pd.DataFrame(rows)


def decay_harvest_curve(close: pd.DataFrame, bull: str, bear: str,
                        rebalance_days: int = 63) -> pd.Series | None:
    """Chartable equity: short both legs, held static within each
    `rebalance_days` block, then reset to 50/50 (compounding block returns).
    A shorter block is closer to the zero-by-construction daily case; a longer
    block carries more drift risk."""
    if bull not in close.columns or bear not in close.columns:
        return None
    px = close[[bull, bear]].dropna()
    if len(px) < rebalance_days + 10:
        return None
    marks = px.iloc[::rebalance_days]
    block = (-0.5 * marks[bull].pct_change()
             - 0.5 * marks[bear].pct_change()).iloc[1:]
    return (1 + block).cumprod()


# --------------------------------------------------------------------------- #
# 3. sector correlation reversion
# --------------------------------------------------------------------------- #
def sector_return_index(close: pd.DataFrame, sectors: dict[str, str]) -> pd.DataFrame:
    """Equal-weight daily-return index per sector, from the stock matrix
    (works even when sector ETFs aren't fetched yet)."""
    rets = close[[c for c in close.columns if c in sectors]].pct_change().iloc[1:]
    out = {}
    for sec in sorted(set(sectors[c] for c in rets.columns)):
        cols = [c for c in rets.columns if sectors[c] == sec]
        if len(cols) >= 3:
            out[sec] = rets[cols].mean(axis=1)
    return pd.DataFrame(out)


def sector_correlation_reversion(close: pd.DataFrame, sectors: dict[str, str],
                                 short_win: int = 63, long_win: int = 504,
                                 top: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sector-pair correlation gaps (short-window minus long baseline) and the
    relative-return divergence over the short window, which orients the
    convergence trade (long laggard / short leader)."""
    sr = sector_return_index(close, sectors)
    long_win = min(long_win, len(sr) - 5)
    if long_win < short_win * 2:
        return pd.DataFrame(), sr
    Cl = sr.iloc[-long_win:].corr()
    Cs = sr.iloc[-short_win:].corr()
    perf = (1 + sr.iloc[-short_win:]).prod() - 1

    rows = []
    secs = list(sr.columns)
    for i in range(len(secs)):
        for j in range(i + 1, len(secs)):
            a, b = secs[i], secs[j]
            gap = Cs.loc[a, b] - Cl.loc[a, b]
            lead, lag = (a, b) if perf[a] > perf[b] else (b, a)
            rows.append({"pair": f"{a} vs {b}",
                         "corr_long": float(Cl.loc[a, b]),
                         f"corr_{short_win}d": float(Cs.loc[a, b]),
                         "gap": float(gap),
                         "leader": lead, "laggard": lag,
                         "perf_spread": float(perf[lead] - perf[lag]),
                         "read": ("decoupled — reconvergence candidate"
                                  if gap < -0.15 else
                                  "abnormally coupled — dispersion candidate"
                                  if gap > 0.15 else "normal")})
    df = (pd.DataFrame(rows)
          .reindex(pd.DataFrame(rows).gap.abs().sort_values(ascending=False).index)
          .head(top).reset_index(drop=True))
    return df, sr
