"""
export_arb.py — distill the arbitrage scanners + crisis validation into ONE
static arb.json for the holodeck front-end (same pattern as export_snapshot.py:
runs in the Action after data refresh, zero extra network calls, the site
fetches it from raw.githubusercontent.com).

Reads the best available price matrix via history_layer.load_history()
(long 2015+ parquet if crisis.yml has run, else the live 3y cache) and
crisis_report.json if present. Emits:

  rv     : top within-sector pair dislocations + a z-spread spark series each
  decay  : leveraged short-both pairs + a compressed equity curve each
  corr   : sector correlation-reversion table
  crisis : per-window variant summaries from crisis_report.json
  meta   : provenance + the honesty flags the front-end must display
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os

import numpy as np

import arb_core as ac
import history_layer as hl
from universe import SECTORS

OUT = "arb.json"
SPARK_N = 64          # points per spark series shipped to the site


def _clean(x, nd=4):
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else round(f, nd)


def _spark(series, n=SPARK_N):
    """Downsample a pandas Series to n points, JSON-safe."""
    if series is None or len(series) == 0:
        return []
    v = series.to_numpy(dtype="float64")
    idx = np.linspace(0, len(v) - 1, min(n, len(v))).astype(int)
    return [_clean(x) for x in v[idx]]


def main():
    close, vol, source = hl.load_history()
    adv = hl.dollar_adv(close, vol)

    # ---- relative value ---------------------------------------------------
    rv_df = ac.relative_value_pairs(close, SECTORS, adv=adv, top=18)
    rv = []
    for _, r in rv_df.iterrows():
        a, b = r["pair"].split("/")
        rv.append({"a": a, "b": b, "sector": r["sector"],
                   "rich": r["rich"], "cheap": r["cheap"],
                   "z": _clean(r["z"], 3), "beta": _clean(r["beta"], 3),
                   "hl": _clean(r["half_life_days"], 1),
                   "corr": _clean(r["corr"], 3),
                   "spark": _spark(ac.pair_spread_series(close, a, b))})

    # ---- decay harvest ----------------------------------------------------
    decay = []
    for _, r in ac.decay_harvest(close, window=63).iterrows():
        item = {"pair": r["pair"], "status": r["status"]}
        if r["status"] == "ok":
            bull, bear = r["pair"].split("+")
            item.update({
                "underlying": r["underlying"], "lev": r["leverage"],
                "days": int(r["days_of_data"]),
                "median63": _clean(r["median_63d_ret"]),
                "winpct": _clean(r["pct_windows_positive"], 1),
                "worst63": _clean(r["worst_63d_ret"]),
                "worst_end": r["worst_window_end"],
                "vol": _clean(r["ann_vol_of_trade"]),
                "curve": _spark(ac.decay_harvest_curve(close, bull, bear))})
        decay.append(item)

    # ---- sector correlation reversion -------------------------------------
    corr_df, _sr = ac.sector_correlation_reversion(close, SECTORS)
    corr = [{"a": r["pair"].split(" vs ")[0], "b": r["pair"].split(" vs ")[1],
             "cl": _clean(r["corr_long"], 3), "cs": _clean(r["corr_63d"], 3),
             "gap": _clean(r["gap"], 3), "leader": r["leader"],
             "laggard": r["laggard"], "spread": _clean(r["perf_spread"]),
             "read": r["read"]}
            for _, r in corr_df.iterrows()]

    # ---- crisis validation (already-computed report, passed through) ------
    crisis = None
    if os.path.exists("crisis_report.json"):
        rep = json.load(open("crisis_report.json"))
        crisis = {"generated": rep.get("generated"), "windows": []}
        for w, d in rep.get("windows", {}).items():
            if "variants" not in d:
                crisis["windows"].append({"id": w, "status": d.get("status")})
                continue
            def _v(key):
                v = d["variants"].get(key, {})
                if "ann_return" not in v:
                    return None
                return {"ann": _clean(v["ann_return"]), "dd": _clean(v["max_dd"]),
                        "cost": _clean(v["avg_cost_drag_ann"]),
                        "worst_m": _clean(v.get("worst_month")),
                        "trough": v.get("trough_date")}
            crisis["windows"].append({
                "id": w, "start": d["start"], "end": d["end"],
                "periods": d["periods"], "biased": d["survivorship_biased"],
                "ew_cum": _clean(d["ew_universe_cum_return"]),
                "long_05x": _v("long_only_0.5x_costs"),
                "long_1x": _v("long_only_1x_costs"),
                "long_2x": _v("long_only_2x_costs"),
                "shorts": _v("with_short_overlay_1x"),
                "hedge": _v("breaker_hedge_SH_1x"),
                "sleeve": "price-only (momentum + reversal)"})

    out = {
        "as_of": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "price_source": source,
        "data_span": [str(close.index.min().date()), str(close.index.max().date())],
        "rv": rv, "decay": decay, "corr": corr, "crisis": crisis,
        "warnings": {
            "rv": "z-score reversion, not riskless arb; divergence loss unbounded",
            "decay": ("static short-both harvests daily-reset decay; borrow fees "
                      "NOT modeled and can flip the sign; loses in strong trends"),
            "corr": "correlations regime-shift toward 1 in crashes",
            "crisis": ("price-only factor sleeve; pre-2023 windows use today's "
                       "constituents (survivorship-biased) — read for drawdown/"
                       "cost behavior, not return level")}}
    with open(OUT, "w") as f:
        json.dump(out, f, separators=(",", ":"), default=str)
    print(f"arb.json: {len(rv)} rv pairs, "
          f"{sum(1 for d in decay if d['status'] == 'ok')}/{len(decay)} decay pairs, "
          f"{len(corr)} corr rows, crisis={'yes' if crisis else 'no'}, "
          f"{os.path.getsize(OUT)} bytes")


if __name__ == "__main__":
    main()
