"""
portfolio_core.py — construction layer between the score and the orders.

Upgrades over "ERC on this month's fresh top-20":

  1. HYSTERESIS SELECTION — enter at rank <= enter_rank, but keep a holding
     until its rank falls below exit_rank. Cuts churn at the margin of the
     book, where composite ranks are noisiest and turnover buys nothing.
     (Directionally this reduces turnover materially; the exact saving depends
     on your signal's autocorrelation — the backtester measures it.)

  2. HARD CONSTRAINTS, ENFORCED — max single-name weight, max sector weight,
     max single-name risk contribution. Previously these existed only as
     reporting in risk_core; here violations change the weights.

  3. RISK-REDUCING SHORTS (opt-in) — per the design request: shorts are
     permitted ONLY when they demonstrably reduce portfolio risk under the
     estimated covariance. Mechanism, stated exactly so it can be audited:

       For long book w (>=0), portfolio variance is w'Σw. Adding a short of
       size h in name i changes variance by  -2h(Σw)_i + h²Σ_ii , which is
       negative for small h iff (Σw)_i > 0 — i.e. the name co-moves with the
       book. The variance-minimizing size is h* = (Σw)_i / Σ_ii.

     A candidate is shorted only if ALL hold:
       (a) its composite score is in the bottom `short_score_pct` percentile
           (we only short names the model also dislikes — a pure hedge with
           positive expected return would be selling alpha to buy vol
           reduction);
       (b) h* > 0 and the post-trade portfolio vol drops by at least
           `min_vol_improvement` (relative);
       (c) caps respected: per-name `max_short_w`, total `max_gross_short`.
     Selection is greedy (best vol reduction first), recomputing Σw after
     each accepted short so interactions between shorts are accounted for.

     Honest caveats, not fine print: (i) "reduces risk" means reduces
     model-estimated variance under a shrunk sample covariance — correlations
     can and do break in crises, so this is risk reduction in expectation,
     not a guarantee; (ii) shorts carry borrow fees and unlimited-loss tail
     mechanics the covariance does not see; hard-to-borrow names should be
     excluded upstream; (iii) this deliberately does NOT implement an
     alpha-seeking short book — that is a different, harder strategy.

  4. DRAWDOWN CIRCUIT BREAKER — de-gross past a drawdown threshold, restore
     on recovery. A blunt instrument; its real job is bounding the damage of
     a bug or a broken signal, not timing the market.
"""
from __future__ import annotations

import numpy as np

import risk_core as rk


# --------------------------------------------------------------------------- #
# 1. Hysteresis selection
# --------------------------------------------------------------------------- #

def hysteresis_select(tickers, composite, prev_holdings,
                      enter_rank=20, exit_rank=40, eligible=None):
    """Rank names by composite (1 = best). Keep previous holdings ranked
    <= exit_rank; fill remaining slots (up to enter_rank names total) with the
    best-ranked non-holdings. Returns list of tickers."""
    comp = np.asarray(composite, float)
    ok = np.isfinite(comp)
    if eligible is not None:
        ok &= np.asarray(eligible, bool)
    idx = np.where(ok)[0]
    order = idx[np.argsort(-comp[idx])]
    rank = {tickers[i]: pos + 1 for pos, i in enumerate(order)}
    kept = [t for t in prev_holdings if rank.get(t, 10**9) <= exit_rank]
    book = list(kept)
    for pos, i in enumerate(order):
        if len(book) >= enter_rank:
            break
        t = tickers[i]
        if t not in book and (pos + 1) <= enter_rank:
            book.append(t)
    return book


# --------------------------------------------------------------------------- #
# 2. Constrained long weights
# --------------------------------------------------------------------------- #

def cap_and_redistribute(w, cap):
    """Iteratively clip weights at `cap` and renormalize the uncapped mass.
    Converges because the capped set only grows. Infeasible caps (cap*N < 1)
    return equal weights at cap (sum < 1; remainder is cash) — explicit,
    not silent."""
    w = np.asarray(w, float).copy()
    n = len(w)
    if cap * n < 1.0 - 1e-12:
        return np.full(n, cap)
    capped = np.zeros(n, dtype=bool)             # persistent: once capped, stays capped
    for _ in range(n):
        over = (w > cap + 1e-15) & ~capped
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        capped |= over
        free = ~capped
        if free.any() and w[free].sum() > 0:
            w[free] += excess * w[free] / w[free].sum()
        # if nothing free, remainder stays uninvested (explicit cash)
    return w


def sector_cap(w, sectors, cap):
    """Scale down any sector whose summed weight exceeds `cap`, redistribute
    to other sectors pro-rata. One pass per offending sector, iterated."""
    w = np.asarray(w, float).copy()
    sectors = np.asarray(sectors, dtype=object)
    for _ in range(20):
        sums = {}
        for s in set(sectors.tolist()):
            sums[s] = w[sectors == s].sum()
        worst = max(sums, key=lambda s: sums[s])
        if sums[worst] <= cap + 1e-12:
            break
        idx = sectors == worst
        excess = sums[worst] - cap
        w[idx] *= cap / sums[worst]
        rest = ~idx
        if w[rest].sum() > 0:
            w[rest] += excess * w[rest] / w[rest].sum()
    return w


def build_long_weights(S, sectors=None, max_name_w=0.10, max_sector_w=0.30,
                       max_rc_pct=0.15):
    """ERC base (correlation-aware), then hard caps: per-name weight, sector
    weight, and per-name risk contribution. RC capping reuses the weight
    capper on an RC-rescaled basis and re-checks; if caps interact infeasibly
    the tightest binding version is returned with a flag."""
    w, converged, _it, _disp = rk.erc_weights(S)
    w = cap_and_redistribute(w, max_name_w)
    if sectors is not None:
        w = sector_cap(w, sectors, max_sector_w)
        w = cap_and_redistribute(w, max_name_w)
    flags = [] if converged else ["ERC fell back toward inverse-vol"]
    for _ in range(10):
        _rc, pct = rk.risk_contributions(w, S)
        if pct.max() <= max_rc_pct + 1e-9:
            break
        i = int(np.argmax(pct))
        w[i] *= 0.85
        w = w / w.sum()
        w = cap_and_redistribute(w, max_name_w)
        if sectors is not None:
            w = sector_cap(w, sectors, max_sector_w)
    else:
        flags.append(f"RC cap {max_rc_pct:.0%} not fully achieved (max {pct.max():.1%})")
    return w / w.sum(), flags


# --------------------------------------------------------------------------- #
# 3. Risk-reducing shorts
# --------------------------------------------------------------------------- #

def risk_reducing_shorts(w_long, S_full, long_idx, candidate_idx,
                         composite, short_score_pct=25.0,
                         max_short_w=0.05, max_gross_short=0.20,
                         min_vol_improvement=0.005):
    """Greedy risk-reducing short overlay.

    Inputs
      w_long        : weights over long_idx (sum 1)
      S_full        : covariance over the FULL candidate set (longs + shorts)
      long_idx      : indices of long names within S_full
      candidate_idx : indices of allowed short candidates (must be disjoint
                      from long_idx; borrowability filtering happens upstream)
      composite     : composite score aligned to S_full order — condition (a)
      short_score_pct : only names at/below this composite percentile shortable

    Returns (w_full_signed, accepted list of (idx, size, vol_before, vol_after)).
    """
    N = S_full.shape[0]
    w = np.zeros(N)
    w[np.asarray(long_idx, int)] = np.asarray(w_long, float)

    comp = np.asarray(composite, float)
    fin = np.isfinite(comp)
    thr = np.nanpercentile(comp[fin], short_score_pct) if fin.sum() else -np.inf
    cands = [i for i in candidate_idx
             if np.isfinite(comp[i]) and comp[i] <= thr and w[i] == 0.0]

    accepted, gross_short = [], 0.0
    for _ in range(len(cands)):
        sig0 = rk.port_vol(w, S_full)
        Sw = S_full @ w
        best = None
        for i in cands:
            if S_full[i, i] <= 0 or Sw[i] <= 0:
                continue                                # can't reduce vol
            h = min(Sw[i] / S_full[i, i], max_short_w,
                    max_gross_short - gross_short)
            if h <= 1e-6:
                continue
            w_try = w.copy(); w_try[i] = -h
            sig1 = rk.port_vol(w_try, S_full)
            if sig1 < sig0 * (1 - min_vol_improvement):
                if best is None or sig1 < best[2]:
                    best = (i, h, sig1)
        if best is None:
            break
        i, h, sig1 = best
        w[i] = -h
        gross_short += h
        accepted.append((i, h, sig0, sig1))
        cands.remove(i)
        if gross_short >= max_gross_short - 1e-9:
            break
    return w, accepted


# --------------------------------------------------------------------------- #
# 4. Drawdown circuit breaker
# --------------------------------------------------------------------------- #

def circuit_breaker_scale(equity_curve, dd_trigger=0.15, degross_to=0.5,
                          recover_dd=0.05):
    """Gross-exposure multiplier from the equity path: 1.0 normally; drop to
    `degross_to` when drawdown from peak exceeds dd_trigger; restore only when
    drawdown recovers above recover_dd. Hysteresis prevents flip-flopping.
    Applied to NEXT period's exposure (uses only past data)."""
    eq = np.asarray(equity_curve, float)
    if len(eq) == 0:
        return 1.0
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    scale, out = 1.0, 1.0
    for d in dd:
        if scale == 1.0 and d <= -dd_trigger:
            scale = degross_to
        elif scale < 1.0 and d >= -recover_dd:
            scale = 1.0
        out = scale
    return out
