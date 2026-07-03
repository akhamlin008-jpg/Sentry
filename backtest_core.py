"""
backtest_core.py — walk-forward, point-in-time backtester.

Loop, per snapshot (chronological, validated by pit_layer first):

  score (factor_core, PIT rows only)
    -> hysteresis selection vs previous book
    -> covariance from TRAILING returns only (Ledoit-Wolf shrunk)
    -> constrained long weights (name/sector/RC caps)
    -> optional risk-reducing short overlay
    -> circuit-breaker gross scaling from PAST equity only
    -> trades = target - held; costs charged via cost_model
    -> realize fwd returns (delisted names realize their delisting return)

Anti-lookahead by construction: the only forward-looking object ever touched
is `fwd_returns`, and it is touched strictly AFTER the weights are frozen.

What this backtester still cannot tell you (be honest with yourself):
  * whether your snapshots really are point-in-time — garbage in, alpha out;
  * live fill quality (the cost model is an assumption; run 0.5x/1x/2x);
  * significance: N monthly periods is a small sample. `summarize` reports a
    t-stat on mean active return; read |t| < 2 as "indistinguishable from
    luck", and remember you looked at this data while building the model,
    which biases even that.
"""
from __future__ import annotations

import numpy as np

import factor_core as fc
import risk_core as rk
import portfolio_core as pc
import cost_model as cm
import exposure_core as ex
import signal_research as sr
from pit_layer import validate_snapshot, validate_chronology


def run_backtest(snaps, *,
                 enter_rank=20, exit_rank=40, min_groups=3,
                 max_name_w=0.10, max_sector_w=0.30, max_rc_pct=0.15,
                 allow_shorts=False, short_kwargs=None, min_history=60,
                 weight_mode="default", weight_kwargs=None,
                 vol_target_ann=None, vol_target_kwargs=None, trailing_ppy=12,
                 cost_kwargs=None, start_equity=1_000_000.0,
                 dd_trigger=0.15, degross_to=0.5,
                 benchmark_returns=None, strict_pit=True):
    """Returns a dict: equity curve, per-period returns/costs/turnover,
    holdings history, and (if benchmark_returns given) active stats.

    benchmark_returns: array aligned to snaps of the benchmark's return over
    each forward window (supply cap-weighted index returns from your data;
    equal-weight of each snapshot's universe is computed automatically as a
    second, always-available baseline)."""
    validate_chronology(snaps)
    for sn in snaps:
        validate_snapshot(sn, strict=strict_pit)

    cost_kwargs = cost_kwargs or {}
    short_kwargs = short_kwargs or {}
    weight_kwargs = weight_kwargs or {}
    vol_target_kwargs = vol_target_kwargs or {}

    equity = start_equity
    eq_curve, rets, costs_frac, turnovers = [], [], [], []
    held_w: dict[str, float] = {}          # ticker -> signed weight (of equity)
    holdings_hist, ew_baseline = [], []

    weight_hist = []
    for i_sn, sn in enumerate(snaps):
        rows = sn.rows
        tickers = [r["ticker"] for r in rows]
        if weight_mode == "validated_equal":
            gw, gw_diag = sr.walk_forward_weights(snaps, i_sn, **weight_kwargs)
        else:
            gw, gw_diag = None, {"mode": "default"}
        weight_hist.append({"weights": gw, "mode": gw_diag.get("mode")})
        res = fc.score_universe(rows, weights=gw)
        comp = res["composite"]

        gmat = np.vstack([res["group_score"][g] for g in fc.GROUP_METRICS])
        n_groups = np.isfinite(gmat).sum(axis=0)
        eligible = n_groups >= min_groups

        book = pc.hysteresis_select(tickers, comp, list(held_w),
                                    enter_rank=enter_rank, exit_rank=exit_rank,
                                    eligible=eligible)

        # trailing-only covariance
        R = sn.trailing_returns
        usable = [t for t in book if R is not None and t in R.columns
                  and R[t].notna().sum() >= min_history]
        target_w: dict[str, float] = {}
        sigma_period = np.nan
        if len(usable) >= 2:
            idx_map = {t: i for i, t in enumerate(tickers)}
            Rm = R[usable].dropna(how="any").values
            S_long, _ = rk.ledoit_wolf_cc(Rm)
            secs = [rows[idx_map[t]].get("sector") for t in usable]
            w_long, _flags = pc.build_long_weights(
                S_long, sectors=secs, max_name_w=max_name_w,
                max_sector_w=max_sector_w, max_rc_pct=max_rc_pct)

            if allow_shorts:
                cand = [t for t in tickers
                        if t not in usable and R is not None and t in R.columns
                        and R[t].notna().sum() >= min_history]
                full = usable + cand
                Rf = R[full].dropna(how="any").values
                S_full, _ = rk.ledoit_wolf_cc(Rf)
                comp_full = np.array([comp[idx_map[t]] for t in full])
                w_signed, _acc = pc.risk_reducing_shorts(
                    w_long, S_full, list(range(len(usable))),
                    list(range(len(usable), len(full))), comp_full,
                    **short_kwargs)
                target_w = {t: float(x) for t, x in zip(full, w_signed)
                            if abs(x) > 1e-9}
                sigma_period = rk.port_vol(w_signed, S_full)
            else:
                target_w = {t: float(x) for t, x in zip(usable, w_long)}
                sigma_period = rk.port_vol(w_long, S_long)

        # exposure scaling from PAST data only:
        #   breaker uses past equity; vol target uses trailing covariance
        #   (ex-ante) and the strategy's own past returns (realized).
        scale = pc.circuit_breaker_scale(eq_curve, dd_trigger=dd_trigger,
                                         degross_to=degross_to)
        if vol_target_ann is not None:
            vscale = ex.combined_scale(sigma_period, rets, vol_target_ann,
                                       trailing_ppy, **vol_target_kwargs)
            scale = min(scale, vscale)
        target_w = {t: w * scale for t, w in target_w.items()}

        # trades + costs
        adv = {r["ticker"]: r.get("adv_dollars", np.nan) for r in rows}
        trades = {}
        for t in set(target_w) | set(held_w):
            d = (target_w.get(t, 0.0) - held_w.get(t, 0.0)) * equity
            if abs(d) > 1.0:
                trades[t] = d
        # veto trades whose cost is unmodelable (capacity breach)
        total_cost, per_cost = cm.apply_costs(trades, adv, **cost_kwargs)
        for t, c in per_cost.items():
            if c is None:
                target_w[t] = held_w.get(t, 0.0)     # keep old position
                total_cost += 0.0
        gross = sum(abs(v) for v in held_w.values()) or 1.0
        turnover = sum(abs(v) for v in trades.values()) / equity
        cost_frac = total_cost / equity

        # realize the forward window
        pnl = 0.0
        for t, w in target_w.items():
            if t in sn.fwd_returns:
                r = sn.fwd_returns[t]
            elif t in sn.delisted:
                r = sn.delisted[t]
            else:
                r = 0.0                              # validated: shouldn't happen
            pnl += w * r
        period_ret = pnl - cost_frac
        equity *= (1 + period_ret)

        # positions drift with returns; dead names drop
        new_held = {}
        for t, w in target_w.items():
            r = sn.fwd_returns.get(t)
            if r is None:
                continue
            new_held[t] = w * (1 + r) / (1 + period_ret if period_ret > -1 else 1)
        held_w = new_held

        eq_curve.append(equity)
        rets.append(period_ret)
        costs_frac.append(cost_frac)
        turnovers.append(turnover)
        holdings_hist.append(dict(target_w))

        # equal-weight universe baseline (survivorship-aware)
        uni_r = [sn.fwd_returns.get(t, sn.delisted.get(t)) for t in sn.universe]
        uni_r = [x for x in uni_r if x is not None and np.isfinite(x)]
        ew_baseline.append(float(np.mean(uni_r)) if uni_r else np.nan)

    out = {"equity": np.array(eq_curve), "returns": np.array(rets),
           "costs": np.array(costs_frac), "turnover": np.array(turnovers),
           "holdings": holdings_hist, "ew_universe": np.array(ew_baseline),
           "group_weights": weight_hist}
    if benchmark_returns is not None:
        out["benchmark"] = np.asarray(benchmark_returns, float)
    return out


def summarize(result, periods_per_year=12):
    """Point estimates + a t-stat, with the small-sample caveat attached to
    the numbers rather than left to the reader's optimism."""
    r = result["returns"]
    n = len(r)
    ann_ret = float((1 + r).prod() ** (periods_per_year / max(n, 1)) - 1)
    vol = float(r.std(ddof=1) * np.sqrt(periods_per_year)) if n > 1 else np.nan
    mdd, _, _ = rk.max_drawdown(r)
    bench = result.get("benchmark", result.get("ew_universe"))
    stats = {"periods": n, "ann_return": ann_ret, "ann_vol": vol,
             "sharpe_excess_of_zero": ann_ret / vol if vol and vol > 0 else np.nan,
             "max_drawdown": float(mdd),
             "avg_turnover": float(np.nanmean(result["turnover"])),
             "avg_cost_drag_ann": float(np.nanmean(result["costs"]) * periods_per_year)}
    if bench is not None and np.isfinite(bench).all():
        active = r - bench
        mu, sd = active.mean(), active.std(ddof=1)
        stats["ann_active_return"] = float(((1 + r).prod() /
                                            (1 + bench).prod()) ** (periods_per_year / max(n, 1)) - 1)
        stats["active_t_stat"] = float(mu / (sd / np.sqrt(n))) if sd > 0 and n > 2 else np.nan
        stats["note"] = ("t-stat on mean active return vs the supplied baseline. "
                         f"n={n} periods is a small sample; |t|<2 means the edge is "
                         "statistically indistinguishable from zero, and in-sample "
                         "model development biases even a large t upward.")
    return stats
