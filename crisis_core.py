"""
crisis_core.py — validate the strategy through REAL market crises.

WHAT THIS IS
------------
Builds monthly PITSnapshots from an actual daily close/volume matrix
(history_layer) and runs backtest_core through named stress windows:

    regime_2016_2019  calm-then-volmageddon regime   (needs long history)
    covid_2020        Feb-Mar 2020 crash + recovery  (needs long history)
    bear_2022         2022 rate-shock bear           (needs long history)
    crash_2025        the April 2025 drawdown        (runs on the live 3y cache)

Each window runs several variants: long-only at 0.5x / 1x / 2x transaction
costs (the cost_model sensitivity discipline), long+short overlay, and the
circuit-breaker hedge overlay (freed exposure parked in an inverse ETF).

WHAT THIS IS NOT — READ BEFORE QUOTING NUMBERS
----------------------------------------------
1. SNAPSHOTS ARE PRICE-ONLY. Free sources give no point-in-time fundamentals
   history, so these snapshots carry only price-derived factors (momentum +
   short-term reversal — the price-only sleeve of the live composite). The
   live strategy also uses value/quality/growth/DCF; this harness validates
   the MACHINERY (selection, weights, costs, breaker, hedge) and the
   price-sleeve signal under stress, not the full composite's alpha.
2. SURVIVORSHIP BIAS ON OLD WINDOWS. History is fetched for TODAY'S
   constituents, so pre-~2023 windows overstate long returns (dead names are
   invisible). Those windows are flagged `survivorship_biased: true` in the
   report. Read them for drawdown shape, cost drag, turnover, and whether the
   breaker/hedge fired correctly — not for the return level.
3. Small samples: a 15-month window is ~15 observations. Point estimates
   only; `summarize`'s t-stat caveat applies double here.
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

import backtest_core as bt
import etf_universe as eu
import history_layer as hl
from pit_layer import PITSnapshot
from universe import SECTORS

# price-only factor sleeve
PRICE_WEIGHTS = {"momentum": 0.5, "reversal": 0.5,
                 "value": 0.0, "quality": 0.0, "growth": 0.0, "dcf": 0.0,
                 "short_interest": 0.0, "insider": 0.0, "institutional": 0.0}

WINDOWS = {
    "regime_2016_2019": ("2015-12-31", "2019-12-31"),
    "covid_2020":       ("2019-09-30", "2020-12-31"),
    "bear_2022":        ("2021-11-30", "2022-12-31"),
    "crash_2025":       ("2024-06-30", "2025-12-31"),
}
# windows scored on today's constituents but ending before the live cache era
SURVIVORSHIP_FLAGGED = {"regime_2016_2019", "covid_2020", "bear_2022"}

BASE_COSTS = {"half_spread": 0.0003, "impact_coef": 0.10}
TRAIL_DAYS = 250          # trailing daily returns fed to the covariance
MIN_HIST_DAYS = 120


def _month_ends(index: pd.DatetimeIndex, start: str, end: str) -> list[pd.Timestamp]:
    idx = index[(index >= start) & (index <= end)]
    if len(idx) == 0:
        return []
    s = pd.Series(idx, index=idx)
    return list(s.groupby([idx.year, idx.month]).max())


def build_snapshots(close: pd.DataFrame, vol: pd.DataFrame | None,
                    start: str, end: str,
                    hedge_tickers: tuple[str, ...] = ("SH", "SDS", "SPXU"),
                    ) -> list[PITSnapshot]:
    """Monthly price-only snapshots. Anti-lookahead: every row field is a
    function of closes at or before the snapshot date; fwd_returns/aux are
    the only forward objects and live in their designated slots."""
    close = close.sort_index()
    stock_cols = [c for c in close.columns if c in SECTORS]      # ETFs excluded
    adv = hl.dollar_adv(close, vol)
    dates = _month_ends(close.index, start, end)
    snaps: list[PITSnapshot] = []

    for d, d_next in zip(dates[:-1], dates[1:]):
        pos = close.index.get_loc(d)
        px_t = close.iloc[pos]

        universe = [t for t in stock_cols if np.isfinite(px_t.get(t, np.nan))]
        if len(universe) < 30:
            continue

        # forward window (the ONLY forward-looking computation)
        seg = close.loc[d:d_next, universe]
        fwd, delisted = {}, {}
        for t in universe:
            p0, p1 = px_t[t], close.at[d_next, t] if t in close.columns else np.nan
            if np.isfinite(p1):
                fwd[t] = float(p1 / p0 - 1)
            else:
                last = seg[t].dropna()
                delisted[t] = float(last.iloc[-1] / p0 - 1) if len(last) else 0.0

        # trailing objects (past-only)
        hist = close.iloc[max(0, pos - 260):pos + 1]
        rows = []
        for t in universe:
            h = hist[t].dropna()
            if len(h) < MIN_HIST_DAYS:
                continue
            a = adv.at[d, t] if adv is not None and t in adv.columns else np.nan
            if not np.isfinite(a) or a <= 0:
                continue          # unmodelable cost -> not tradeable, stays in EW universe
            p = float(h.iloc[-1])

            def _ret(k_from, k_to=0):
                if len(h) <= k_from:
                    return np.nan
                return float(h.iloc[-1 - k_to] / h.iloc[-1 - k_from] - 1)

            rows.append({
                "ticker": t, "sector": SECTORS.get(t, "Unknown"), "price": p,
                "mom_12_1": _ret(252, 21), "mom_6_1": _ret(126, 21),
                "ret_5d": _ret(5), "ret_21d": _ret(21),
                "adv_dollars": float(a),
                "asof_prices": d.date(),
            })

        trail = close.iloc[max(0, pos - TRAIL_DAYS):pos + 1][
            [r["ticker"] for r in rows]].pct_change().iloc[1:]

        aux = {}
        for ht in hedge_tickers:
            if ht in close.columns and np.isfinite(px_t.get(ht, np.nan)):
                p1 = close.at[d_next, ht]
                if np.isfinite(p1):
                    a = adv.at[d, ht] if adv is not None and ht in adv.columns else np.nan
                    aux[ht] = {"fwd_return": float(p1 / px_t[ht] - 1),
                               "adv_dollars": float(a) if np.isfinite(a) else 1e9}

        snaps.append(PITSnapshot(
            date=d.date(), universe=universe, rows=rows,
            fwd_returns=fwd, delisted=delisted,
            trailing_returns=trail, aux=aux))
    return snaps


def _spy_benchmark(close: pd.DataFrame, snaps: list[PITSnapshot]) -> np.ndarray | None:
    if "SPY" not in close.columns:
        return None
    s = close["SPY"].dropna()
    out = []
    dates = [pd.Timestamp(sn.date) for sn in snaps]
    dates.append(dates[-1] + pd.offsets.MonthEnd(1))
    for d0, d1 in zip(dates[:-1], dates[1:]):
        a = s.asof(d0)
        b = s.asof(min(d1, s.index.max()))
        out.append(b / a - 1 if np.isfinite(a) and np.isfinite(b) else np.nan)
    arr = np.array(out)
    return arr if np.isfinite(arr).all() else None


def _crash_stats(res: dict, snaps: list[PITSnapshot]) -> dict:
    """Window-shape diagnostics beyond summarize()."""
    r = res["returns"]
    eq = np.concatenate([[1.0], np.cumprod(1 + r)])
    dd = eq / np.maximum.accumulate(eq) - 1
    i_tr = int(dd.argmin())
    rec = next((j for j in range(i_tr, len(dd)) if dd[j] > -1e-9), None)
    return {"max_dd": float(dd.min()),
            "trough_date": str(snaps[max(i_tr - 1, 0)].date),
            "months_to_recover": (rec - i_tr) if rec is not None else None,
            "worst_month": float(r.min()), "best_month": float(r.max())}


def run_window(close, vol, name, start, end) -> dict:
    snaps = build_snapshots(close, vol, start, end)
    if len(snaps) < 6:
        return {"window": name, "status": "insufficient data",
                "note": "needs the long-history fetch (crisis.yml) for this window"}
    bench = _spy_benchmark(close, snaps)
    common = dict(min_groups=2, weight_mode="default", strict_pit=True,
                  benchmark_returns=bench)
    variants = {}

    for mult, label in [(0.5, "long_only_0.5x_costs"),
                        (1.0, "long_only_1x_costs"),
                        (2.0, "long_only_2x_costs")]:
        ck = {k: v * mult for k, v in BASE_COSTS.items()}
        res = bt.run_backtest(snaps, cost_kwargs=ck, **common)
        variants[label] = {**bt.summarize(res), **_crash_stats(res, snaps)}

    res = bt.run_backtest(snaps, cost_kwargs=dict(BASE_COSTS),
                          allow_shorts=True, **common)
    variants["with_short_overlay_1x"] = {**bt.summarize(res), **_crash_stats(res, snaps)}

    if any("SH" in sn.aux for sn in snaps):
        res = bt.run_backtest(snaps, cost_kwargs=dict(BASE_COSTS),
                              hedge_ticker="SH", hedge_leverage=1.0, **common)
        variants["breaker_hedge_SH_1x"] = {**bt.summarize(res), **_crash_stats(res, snaps)}
    else:
        variants["breaker_hedge_SH_1x"] = {
            "status": "skipped — SH not in price matrix (run crisis.yml to fetch ETFs)"}

    # EW-universe context for the same months
    ew = bt.run_backtest(snaps, cost_kwargs=dict(BASE_COSTS), **common)["ew_universe"]
    return {"window": name, "start": str(snaps[0].date), "end": str(snaps[-1].date),
            "periods": len(snaps),
            "survivorship_biased": name in SURVIVORSHIP_FLAGGED,
            "ew_universe_cum_return": float(np.prod(1 + ew[np.isfinite(ew)]) - 1),
            "spy_available": bench is not None,
            "factor_sleeve": "price-only (momentum + reversal); see module docstring",
            "variants": variants}


def run_all(out_path: str = "crisis_report.json") -> dict:
    close, vol, source = hl.load_history()
    report = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "price_source": source,
              "data_span": [str(close.index.min().date()), str(close.index.max().date())],
              "cost_model": {"base": BASE_COSTS,
                             "note": "sensitivity run at 0.5x/1x/2x per cost_model discipline"},
              "windows": {}}
    for name, (s, e) in WINDOWS.items():
        try:
            report["windows"][name] = run_window(close, vol, name, s, e)
        except Exception as exc:                       # keep the report partial, not dead
            report["windows"][name] = {"window": name, "status": f"error: {exc}"}
        print(f"[crisis] {name}: {report['windows'][name].get('status', 'ok')}")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report


if __name__ == "__main__":
    rep = run_all()
    for w, r in rep["windows"].items():
        if "variants" not in r:
            print(w, "->", r.get("status"))
            continue
        v = r["variants"]["long_only_1x_costs"]
        print(f"{w}: ann_ret={v['ann_return']:+.1%} maxDD={v['max_dd']:+.1%} "
              f"cost_drag={v['avg_cost_drag_ann']:.2%}/yr "
              f"{'[SURVIVORSHIP-BIASED]' if r['survivorship_biased'] else ''}")
