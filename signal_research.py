"""
signal_research.py — validate the signal BEFORE trusting the portfolio.

Answers, from a chronological list of PITSnapshots:

  1. Does each factor group predict forward returns at all?   -> rank IC series
  2. At what horizon does the signal live/die?                 -> IC decay
  3. Are the groups distinct bets or the same bet twice?       -> score correlation
  4. Is a "factor" secretly just beta or size?                 -> neutralization

Statistical honesty, built in rather than promised:
  * A monthly IC series over a few years is a SMALL sample. `ic_summary`
    reports a t-statistic on the mean IC using Newey-West (lag-1) standard
    errors; treat |t| < 2 as "no evidence", not "small edge".
  * Nothing here fits weights to history. Selecting only groups with positive
    in-sample IC and then backtesting on the SAME history is itself mild
    overfitting — split your snapshot history (e.g. fit on the first 60%,
    confirm on the rest) before believing a weight scheme.
"""
from __future__ import annotations

import numpy as np

import factor_core as fc


def _spearman(a, b):
    """Rank correlation, nan-pairwise. Returns nan if <8 joint observations
    (below that, a single month's IC is noise dressed as a number)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8:
        return np.nan
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / den) if den > 0 else np.nan


def neutralize(scores, rows, against=("beta", "market_cap"), sectors=None):
    """Residualize a score vector against nuisance exposures via OLS.

    Regresses score on [1, beta, log(mkt cap), sector dummies] and returns the
    residual: the part of the score NOT explained by being high-beta / large /
    in a hot sector. If a 'quality' factor loses most of its IC after this,
    it was a beta tilt wearing a quality costume.
    """
    s = np.asarray(scores, float)
    n = len(s)
    cols = [np.ones(n)]
    for key in against:
        v = np.array([fc._get(r, key) for r in rows], float)
        if key == "market_cap":
            with np.errstate(invalid="ignore"):
                v = np.log(np.where(v > 0, v, np.nan))
        cols.append(v)
    if sectors is not None:
        uniq = sorted(set(sectors))
        for sec in uniq[1:]:                       # drop-one dummy coding
            cols.append(np.array([1.0 if x == sec else 0.0 for x in sectors]))
    X = np.column_stack(cols)
    m = np.isfinite(s) & np.isfinite(X).all(axis=1)
    out = np.full(n, np.nan)
    if m.sum() <= X.shape[1] + 2:
        return s                                   # too few names to residualize
    coef, *_ = np.linalg.lstsq(X[m], s[m], rcond=None)
    out[m] = s[m] - X[m] @ coef
    return out


def ic_series(snaps, group, method="rank", neutralize_against=None):
    """Per-snapshot rank IC of one factor group vs realized forward returns.
    Returns (dates, ics)."""
    dates, ics = [], []
    for sn in snaps:
        rows = sn.rows
        score, _cov = fc.standardize_group(rows, group, method=method)
        if neutralize_against:
            secs = [r.get("sector") for r in rows]
            score = neutralize(score, rows, against=neutralize_against, sectors=secs)
        fwd = np.array([sn.fwd_returns.get(r["ticker"],
                        sn.delisted.get(r["ticker"], np.nan)) for r in rows])
        ics.append(_spearman(score, fwd))
        dates.append(sn.date)
    return dates, np.array(ics)


def ic_summary(ics):
    """Mean IC, Newey-West(1) t-stat, hit rate, and n. Interpretation guide:
    with n monthly observations the standard error is ~sigma/sqrt(n); at
    n=36 even a genuinely positive factor can easily show |t|<2. Absence of
    significance is 'insufficient evidence', presence at small n is fragile."""
    x = np.asarray(ics, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return {"mean": np.nan, "t": np.nan, "hit": np.nan, "n": n}
    mu = x.mean()
    xc = x - mu
    g0 = float(xc @ xc) / n
    g1 = float(xc[1:] @ xc[:-1]) / n
    lrv = g0 + 2 * (1 - 1 / 2) * g1                 # Bartlett kernel, 1 lag
    se = np.sqrt(max(lrv, 1e-12) / n)
    return {"mean": float(mu), "t": float(mu / se),
            "hit": float((x > 0).mean()), "n": n}


def ic_decay(snaps, group, horizons=(1, 3, 6), method="rank"):
    """Mean IC of the group's score at t against cumulative returns over the
    next h snapshot-periods, for each h. A factor whose IC at h=1 dwarfs h=3
    decays fast and is being under-harvested by monthly trading; one whose IC
    only appears at long horizons doesn't need monthly turnover."""
    out = {}
    for h in horizons:
        ics = []
        for i in range(len(snaps) - h + 1):
            sn = snaps[i]
            rows = sn.rows
            score, _ = fc.standardize_group(rows, group, method=method)
            cum = np.zeros(len(rows)); alive = np.ones(len(rows), bool)
            for j in range(h):
                sj = snaps[i + j]
                for k, r in enumerate(rows):
                    tk = r["ticker"]
                    if not alive[k]:
                        continue
                    if tk in sj.fwd_returns:
                        cum[k] = (1 + cum[k]) * (1 + sj.fwd_returns[tk]) - 1
                    elif tk in sj.delisted:
                        cum[k] = (1 + cum[k]) * (1 + sj.delisted[tk]) - 1
                        alive[k] = False
                    else:
                        cum[k] = np.nan; alive[k] = False
            ics.append(_spearman(score, cum))
        out[h] = ic_summary(ics)
    return out


def group_score_correlation(snaps, groups=None, method="rank"):
    """Average cross-sectional correlation between group scores, pooled over
    snapshots. High correlation (e.g. value vs dcf) means double-counted risk:
    the composite is more concentrated than its weight table implies."""
    groups = groups or [g for g in fc.GROUP_METRICS]
    pooled = {g: [] for g in groups}
    for sn in snaps:
        for g in groups:
            s, _ = fc.standardize_group(sn.rows, g, method=method)
            pooled[g].append(s)
    G = len(groups)
    C = np.full((G, G), np.nan)
    for i in range(G):
        for j in range(i, G):
            cs = []
            for a, b in zip(pooled[groups[i]], pooled[groups[j]]):
                c = _spearman(a, b)
                if np.isfinite(c):
                    cs.append(c)
            if cs:
                C[i, j] = C[j, i] = float(np.mean(cs))
    return groups, C


# --------------------------------------------------------------------------- #
# Walk-forward validated group weights (PIT-safe replacement for hand-picked)
# --------------------------------------------------------------------------- #

def walk_forward_weights(snaps, i, groups=None, min_obs=12, min_mean_ic=0.0,
                         method="rank"):
    """Equal weights over the factor groups that have EARNED inclusion by
    snapshot index i, judged only on snapshots[0:i] (whose forward windows are
    fully realized before snaps[i].date — PIT-safe by construction).

    Inclusion rule (deliberately binary, per the anti-overfitting argument in
    this module's docstring): a group is included iff it has >= min_obs
    trailing IC observations AND trailing mean IC > min_mean_ic. Included
    groups get EQUAL weight — no optimization, because optimizing weights on
    a short IC history is curve-fitting with extra steps.

    Fallbacks, made explicit:
      * i < min_obs (not enough history to judge anything): equal weight
        across all candidate groups — an ignorance prior, not an endorsement.
      * history sufficient but NO group qualifies: also equal weight across
        all candidates, plus flag 'no_group_qualified'. Zero exposure on IC
        evidence alone would be too aggressive at these sample sizes (mean IC
        below zero over ~12 monthly points is weak evidence of anything);
        if you want "stand aside" behavior instead, act on the flag upstream.

    Returns (weights_dict, diagnostics).
    """
    import factor_core as fc
    groups = groups or [g for g, w in fc.DEFAULT_GROUP_WEIGHTS.items() if w > 0]
    diag = {}
    if i < min_obs:
        w = {g: 1.0 / len(groups) for g in groups}
        return w, {"mode": "ignorance_prior", "detail": diag}
    qualified = []
    for g in groups:
        _, ics = ic_series(snaps[:i], g, method=method)
        s = ic_summary(ics)
        diag[g] = s
        if s["n"] >= min_obs and np.isfinite(s["mean"]) and s["mean"] > min_mean_ic:
            qualified.append(g)
    if not qualified:
        w = {g: 1.0 / len(groups) for g in groups}
        return w, {"mode": "no_group_qualified", "detail": diag}
    w = {g: (1.0 / len(qualified) if g in qualified else 0.0) for g in groups}
    return w, {"mode": "validated_equal", "qualified": qualified, "detail": diag}
