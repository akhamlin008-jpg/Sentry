"""
export_brief.py — saves Sentry's scores to brief/sentry_brief.json so the
Morning Letter can read them. Runs right after fetch_snapshot.py in
refresh.yml, so it reads the fresh cache and makes no extra data calls.

Uses Sentry's own math with the app's default settings:
  composite  -> factor_core.score_universe (rank method, no sector neutralize,
                default weights)
  top 20     -> factor_core.select_candidates (min 3 factors observed)
  risk       -> risk_core Ledoit-Wolf covariance + risk-parity (ERC) weights,
                the Risk Report page's defaults
  pulse      -> 5-day return z-score. NOT in the Sentry v3 code; defined here.
                bullish z > +0.5, bearish z < -0.5, otherwise neutral.

Always writes a file. If something fails, the file says so instead of 
guessing.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys
import traceback
from zoneinfo import ZoneInfo

import numpy as np

N_TOP = 20
N_BOTTOM = 5
N_EXTREMES = 5
MIN_GROUPS = 3
PULSE_DAYS = 5
VOL_WINDOW = 60          # trading days of history used for the pulse z
PULSE_BAND = 0.5         # |z| below this = neutral
TRADING_DAYS = 252
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brief", "sentry_brief.json")
ET = ZoneInfo("America/New_York")


def num(x, nd=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) + 0.0 if math.isfinite(x) else None


def compute_pulse(rets):
    """ticker -> (5-day compounded return, z vs prior 60-day daily vol)."""
    out = {}
    if rets is None or rets.empty:
        return out
    for tk in rets.columns:
        s = rets[tk].dropna()
        if len(s) < VOL_WINDOW + PULSE_DAYS:
            continue
        r5 = float(np.prod(1.0 + s.iloc[-PULSE_DAYS:].values) - 1.0)
        vol = float(s.iloc[-(VOL_WINDOW + PULSE_DAYS):-PULSE_DAYS].std(ddof=1))
        if not (vol > 0):
            continue
        out[tk] = (r5, r5 / (vol * math.sqrt(PULSE_DAYS)))
    return out


def label(z):
    if z is None:
        return None
    return "bullish" if z > PULSE_BAND else "bearish" if z < -PULSE_BAND else "neutral"


def pulse_fields(tk, pulse):
    r5, z = pulse.get(tk, (None, None))
    z = num(z, 2)
    return {"ret_5d_pct": num(r5 * 100 if r5 is not None else None, 2),
            "z_5d": z, "pulse": label(z)}


def risk_block(rets, longs):
    import risk_core as rk
    names = [t for t in longs if t in rets.columns]
    R = rets[names].dropna(how="any")
    if len(names) < 2 or len(R) < 60:
        return {"status": "skipped", "reason": f"{len(names)} names, {len(R)} shared days"}
    S, delta = rk.ledoit_wolf_cc(R.values)
    S_ann = rk.annualize_cov(S, TRADING_DAYS)
    w, converged, _, _ = rk.erc_weights(S)
    w = w / w.sum()
    _, rc_pct = rk.risk_contributions(w, S_ann)
    C = rk.cov_to_corr(S)
    order = np.argsort(-np.asarray(rc_pct))
    return {
        "status": "ok",
        "names": len(names),
        "days_used": int(len(R)),
        "shrinkage": num(delta, 3),
        "erc_converged": bool(converged),
        "portfolio_vol_ann_pct": num(rk.port_vol(w, S_ann) * 100, 1),
        "avg_pairwise_corr": num(rk.avg_pairwise_corr(C), 2),
        "most_correlated_pairs": [
            {"pair": [str(p[0]), str(p[1])], "corr": num(p[2], 2)}
            for p in rk.most_correlated_pairs(C, names, k=5)
        ],
        "top_risk_contributors": [
            {"ticker": names[i], "weight_pct": num(w[i] * 100, 1),
             "risk_pct": num(rc_pct[i] * 100, 1)}
            for i in order[:5]
        ],
    }


def build():
    import data_layer as dl
    import factor_core as fc
    from universe import TICKERS

    payload = dl.load_universe(TICKERS)
    rows, rets = payload["rows"], payload["returns"]
    tickers = [r["ticker"] for r in rows]
    by_tk = {r["ticker"]: r for r in rows}
    idx = {t: i for i, t in enumerate(tickers)}

    result = fc.score_universe(rows)
    longs, shorts = fc.select_candidates(tickers, result, n_long=N_TOP,
                                         n_short=N_BOTTOM, min_groups_long=MIN_GROUPS,
                                         min_groups_short=MIN_GROUPS)
    pulse = compute_pulse(rets)

    def entry(rank, tk, comp, pct):
        i = idx[tk]
        return {
            "rank": rank, "ticker": tk, "sector": by_tk[tk].get("sector"),
            "composite": num(comp, 2), "percentile": num(pct, 0),
            "grade": str(result["grade"][i]),
            "factors": {g: num(result["group_score"][g][i], 2)
                        for g, w in fc.DEFAULT_GROUP_WEIGHTS.items() if w > 0},
            **pulse_fields(tk, pulse),
        }

    ranked = sorted(((tk, z) for tk, (_, z) in pulse.items()), key=lambda x: x[1])
    extremes = lambda lst: [{"ticker": t, **pulse_fields(t, pulse)} for t, _ in lst]

    try:
        risk = risk_block(rets, [t for t, _, _ in longs])
    except Exception as e:  # noqa: BLE001
        risk = {"status": "failed", "reason": f"{type(e).__name__}: {e}"}

    loaded = sum(1 for r in rows if not r.get("error"))
    return {
        "status": "ok",
        "universe": len(tickers),
        "loaded": loaded,
        "data_as_of": payload["as_of"].isoformat(timespec="minutes"),
        "top20_composite": [entry(k + 1, *x) for k, x in enumerate(longs)],
        "bottom5_composite": [entry(k + 1, *x) for k, x in enumerate(shorts)],
        "pulse_highest_z": extremes(ranked[::-1][:N_EXTREMES]),
        "pulse_lowest_z": extremes(ranked[:N_EXTREMES]),
        "risk_top20_erc": risk,
    }


def main() -> int:
    now = dt.datetime.now(ET)
    header = {
        "generated_et": now.strftime("%Y-%m-%d %H:%M %Z"),
        "notes": {
            "composite": "Sentry factor_core, app defaults (rank, 5 live factors, min 3 observed)",
            "pulse": f"{PULSE_DAYS}-day return / (prior {VOL_WINDOW}-day daily vol x sqrt({PULSE_DAYS})); "
                     f"bullish > +{PULSE_BAND}, bearish < -{PULSE_BAND}. Defined by this script.",
            "risk": "Ledoit-Wolf covariance, risk-parity (ERC) weights on the top 20",
        },
    }
    try:
        body = build()
        code = 0 if body["loaded"] > 0 else 1
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        body = {"status": "failed", "reason": f"{type(e).__name__}: {e}"}
        code = 1
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump({**header, **body}, fh, indent=1)
    print(f"[brief] {body['status']} -> {OUT}")
    return code


if __name__ == "__main__":
    sys.exit(main())
