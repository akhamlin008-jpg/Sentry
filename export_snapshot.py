"""
export_snapshot.py — turn the pipeline's output into ONE static snapshot.json
that the 3D holodeck front-end fetches. Run it right after fetch_snapshot.py in
the refresh Action (it reads the warm cache, so it makes zero extra network
calls). Commit snapshot.json alongside .cache/** and the site self-updates at
the Action's cadence.

Everything here reuses the exact engines the Streamlit app uses:
  * data_layer.load_universe()  — fundamentals + prices (cache-warm)
  * factor_core.score_universe() — 5-factor composite, percentile, grade
  * dcf_core.derive_assumptions + two_stage_dcf — same seeding as App.py

Pulse-globe fields (computed from real data, documented so the front-end and
back-end agree on meaning):
  * volZ — cross-sectional z-score of turnover (ADV / market cap). "How heavily
    is this name trading relative to its size, versus the rest of the universe."
  * dir  — tanh(10 × trailing 5-day return), bounded to [-1, +1].
    Negative = net selling (red), positive = net buying (blue).
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys

import numpy as np

import data_layer as dl
import dcf_core as core
import factor_core as fc
from universe import TICKERS

DEFAULT_TERMINAL = 0.025
DEFAULT_ERP = 0.045
DEFAULT_RF = 0.043
CAGR_CAP = (-0.10, 0.25)
FALLBACK_G1 = 0.10
MOS_IMPLAUSIBLE = 300.0           # same sanity flag as App.py
RETURN_DAYS = 250                 # trailing window shipped to the site
OUT_PATH = "snapshot.json"

# ---- staleness gate ---------------------------------------------------------
# Fail the export (non-zero exit -> red Action) if the newest price date in
# the whole board is more than STALE_MAX_CAL_DAYS calendar days behind the run
# date (spans weekends + a holiday). Individually lagging tickers are listed
# loudly and flagged in the JSON, but do not fail the run on their own.
STALE_MAX_CAL_DAYS = int(os.environ.get("STALE_MAX_CAL_DAYS", "4"))


def _clean(x, nd=4):
    """JSON-safe float: round, and map nan/inf to None."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, nd)


def _stock_from_row(r: dict) -> dict:
    """Identical mapping to App.py so DCF seeding matches the Streamlit page."""
    fs = r.get("fcf_series") or []
    s = {
        "ticker": r["ticker"], "fcf": r.get("_fcf"), "fcf_series": fs,
        "sbc": None, "shares": r.get("_shares"), "cash": r.get("_cash"),
        "debt": r.get("_debt"), "price": r.get("_price"),
        "market_cap": r.get("_mktcap"), "interest_expense": r.get("_interest"),
        "tax_rate": 0.21, "analyst_g5": None,
        "hist_cagr": core.robust_cagr(fs, cap=CAGR_CAP),
        "beta": r.get("_beta"), "sector": r.get("sector"),
    }
    return s


def main() -> int:
    started = dt.datetime.now()
    payload = dl.load_universe(TICKERS)
    rows = payload["rows"]
    rets = payload["returns"]
    adv = payload.get("adv") or {}
    rf = payload.get("rf") or DEFAULT_RF

    sector_names = sorted({(r.get("sector") or "Unknown") for r in rows})
    sec_idx = {nm: i for i, nm in enumerate(sector_names)}

    # ---- factor scores (default weights/method, same as the Factor page) ----
    result = fc.score_universe(rows)
    tickers = [r["ticker"] for r in rows]
    live_groups = [g for g, w in fc.DEFAULT_GROUP_WEIGHTS.items() if w > 0]

    # ---- pulse-globe metrics: turnover z + 5d direction ---------------------
    turnover = {}
    for r in rows:
        a, m = adv.get(r["ticker"]), r.get("_mktcap")
        turnover[r["ticker"]] = (a / m) if (a and m) else None
    tv = np.array([turnover[t] if turnover[t] is not None else np.nan
                   for t in tickers], dtype=float)
    mu, sd = np.nanmean(tv), np.nanstd(tv)
    volz = (tv - mu) / sd if sd and not math.isnan(sd) else np.zeros_like(tv)

    dir5 = {}
    if rets is not None and not rets.empty:
        tail = rets.tail(5)
        for t in tickers:
            if t in tail.columns:
                c = float((1 + tail[t].fillna(0)).prod() - 1)
                dir5[t] = math.tanh(10 * c)

    # ---- per-stock records --------------------------------------------------
    stocks, n_valued = [], 0
    for i, r in enumerate(rows):
        tk = r["ticker"]
        s = _stock_from_row(r)
        auto, g1_src = core.derive_assumptions(s, rf, DEFAULT_ERP,
                                               DEFAULT_TERMINAL, CAGR_CAP,
                                               FALLBACK_G1)
        fair = mos = None
        flags = []
        if auto["r"] is not None and not r.get("error"):
            res = core.two_stage_dcf(s["fcf"], auto["g1"], auto["g2"],
                                     auto["gt"], auto["r"], s["cash"],
                                     s["debt"], s["shares"])
            if not res.error and s["price"]:
                m = (res.fair - s["price"]) / s["price"] * 100
                if abs(m) <= MOS_IMPLAUSIBLE:
                    fair, mos = res.fair, m
                    flags = list(res.flags)
                    n_valued += 1
                else:
                    flags = ["implausible — likely data/units issue"]
            elif res.error:
                flags = [res.error]

        f_scores = {g: _clean(result["group_score"][g][i], 3)
                    for g in live_groups}
        stocks.append({
            "tk": tk,
            "sec": sec_idx.get(r.get("sector") or "Unknown", 0),
            "px": _clean(s["price"], 2), "cap": _clean(s["market_cap"], 0),
            "px_asof": r.get("price_asof"),
            "px_stale": bool(r.get("price_stale")),
            "sh": _clean(s["shares"], 0), "cash": _clean(s["cash"], 0),
            "debt": _clean(s["debt"], 0), "fcf": _clean(s["fcf"], 0),
            "beta": _clean(s["beta"], 2),
            "g1": _clean(auto["g1"]), "g2": _clean(auto["g2"]),
            "gt": _clean(auto["gt"]), "r": _clean(auto["r"]),
            "fair": _clean(fair, 2), "mos": _clean(mos, 1), "flags": flags,
            "f": f_scores,
            "comp": _clean(result["composite"][i], 3),
            "pct": _clean(result["percentile"][i], 1),
            "grade": str(result["grade"][i]),
            "volZ": _clean(volz[i], 2) if not math.isnan(volz[i]) else 0.0,
            "dir": _clean(dir5.get(tk, 0.0), 3),
            "adv": _clean(adv.get(tk), 0),
        })

    # ---- trailing returns matrix (for the risk console) ---------------------
    returns_block = {"cols": [], "m": []}
    if rets is not None and not rets.empty:
        tail = rets.tail(RETURN_DAYS)
        cols = [c for c in tail.columns]
        mat = tail[cols].fillna(0.0).round(4).values.T.tolist()
        returns_block = {"cols": cols, "m": mat, "days": int(len(tail))}

    # ---- validation gate: never ship a snapshot that pretends to be fresh ----
    asof_dates = [s["px_asof"] for s in stocks if s.get("px_asof")]
    board_latest = max(asof_dates) if asof_dates else None
    laggards = [s["tk"] for s in stocks
                if s.get("px_stale") or (board_latest and s.get("px_asof")
                                         and s["px_asof"] < board_latest)]
    if laggards:
        print(f"[export] WARNING: {len(laggards)} tickers lag the board "
              f"(board latest {board_latest}): {', '.join(sorted(laggards))}")
    if board_latest is None:
        print("[export] ERROR: no price dates at all — refusing to export.")
        return 1
    lag_days = (started.date() - dt.date.fromisoformat(board_latest)).days
    print(f"[export] board price date {board_latest} · run date "
          f"{started.date()} · lag {lag_days}d (max {STALE_MAX_CAL_DAYS}d)")
    if lag_days > STALE_MAX_CAL_DAYS:
        print(f"[export] ERROR: board is {lag_days} calendar days stale "
              f"(> {STALE_MAX_CAL_DAYS}). The price cache is not refreshing — "
              f"refusing to export a snapshot that pretends to be current.")
        return 1

    snap = {
        "as_of": started.isoformat(timespec="minutes"),
        "px_asof": board_latest,
        "px_laggards": sorted(laggards),
        "rf": _clean(rf), "benchmark": payload.get("benchmark"),
        "sectors": sector_names,
        "n_valued": n_valued,
        "source_mix": payload.get("source_mix", {}),
        "stocks": stocks,
        "returns": returns_block,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(snap, f, separators=(",", ":"), allow_nan=False)

    kb = len(json.dumps(snap, separators=(",", ":"))) / 1024
    print(f"[export] {OUT_PATH}: {len(stocks)} stocks · {n_valued} valued · "
          f"returns {returns_block.get('days', 0)}d × "
          f"{len(returns_block['cols'])} · {kb:,.0f} KB")
    if not stocks:
        print("[export] ERROR: no stocks exported.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
