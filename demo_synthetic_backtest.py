"""
demo_synthetic_backtest.py — end-to-end run on SYNTHETIC data.

READ THIS FIRST
---------------
This demo exists to prove the MACHINERY works: PIT validation, hysteresis,
constrained weights, risk-reducing shorts, costs, circuit breaker, and the
anti-lookahead loop, all executing together without error and producing
internally consistent numbers.

It proves NOTHING about real-world profitability. The synthetic world below
PLANTS a weak value/quality signal by construction, so of course the model
finds it. Real markets do not come with a planted signal. Do not quote any
number this prints as evidence of edge.

Synthetic world: 120 names, 36 monthly periods, one market factor, sector
structure, a small planted premium on the true (hidden) quality/value traits,
noisy observable metrics derived from the traits, occasional delistings, and
a fundamentals reporting lag — i.e. the shape of the real problem.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

import backtest_core as bt
from pit_layer import PITSnapshot

rng = np.random.default_rng(42)

N, T = 120, 36
SECTORS = ["Tech", "Fin", "Health", "Energy", "Util", "Cons"]
tickers = [f"S{i:03d}" for i in range(N)]
sector = rng.choice(SECTORS, N)
beta = rng.normal(1.0, 0.3, N).clip(0.2, 2.0)
true_quality = rng.normal(0, 1, N)               # hidden trait
true_value = rng.normal(0, 1, N)

PLANTED_MONTHLY_PREMIUM = 0.0025                 # per 1sd of trait — assumption
MKT_MU, MKT_SD, IDIO_SD = 0.006, 0.045, 0.06

# simulate monthly returns
mkt = rng.normal(MKT_MU, MKT_SD, T)
rets = np.zeros((T, N))
for t in range(T):
    rets[t] = (beta * mkt[t]
               + PLANTED_MONTHLY_PREMIUM * (true_quality + true_value)
               + rng.normal(0, IDIO_SD, N))

# a few delistings (acquisitions at +15%, one failure at -70%)
delist_events = {8: {"S007": 0.15}, 17: {"S033": -0.70}, 25: {"S090": 0.15}}
dead: set[str] = set()

start = dt.date(2022, 1, 31)
dates = [start + dt.timedelta(days=30 * k) for k in range(T)]

snaps = []
prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0),
                      columns=tickers)

for t in range(12, T):                            # first year = warmup history
    alive = [tk for tk in tickers if tk not in dead]
    rows = []
    for tk in alive:
        i = tickers.index(tk)
        # observable metrics = hidden trait + observation noise, lagged asof
        rows.append({
            "ticker": tk, "sector": sector[i], "beta": float(beta[i]),
            "price": float(prices.iloc[t - 1, i]),
            "market_cap": float(prices.iloc[t - 1, i] * 1e7),
            "adv_dollars": 5e7,
            "roe": float(0.10 + 0.05 * true_quality[i] + rng.normal(0, 0.03)),
            "op_margin": float(0.15 + 0.05 * true_quality[i] + rng.normal(0, 0.04)),
            "fcf_yield": float(0.05 + 0.02 * true_value[i] + rng.normal(0, 0.015)),
            "earnings_yield": float(0.06 + 0.02 * true_value[i] + rng.normal(0, 0.02)),
            "mom_12_1": float(prices.iloc[t - 1, i] / prices.iloc[t - 12, i] - 1),
            "ret_21d": float(rets[t - 1, i]),
            "asof_fundamentals": dates[t] - dt.timedelta(days=45),
        })
    ev = delist_events.get(t, {})
    fwd = {tk: float(rets[t, tickers.index(tk)]) for tk in alive if tk not in ev}
    snaps.append(PITSnapshot(
        date=dates[t], universe=alive, rows=rows, fwd_returns=fwd,
        delisted={tk: r for tk, r in ev.items()},
        trailing_returns=pd.DataFrame(rets[max(0, t - 24):t],
                                      columns=tickers)[alive]))
    dead |= set(ev)

if __name__ == "__main__":
    configs = [
        ("long-only baseline", {}),
        ("+ vol target 10%", dict(vol_target_ann=0.10)),
        ("+ vol target 10% + validated weights",
         dict(vol_target_ann=0.10, weight_mode="validated_equal",
              weight_kwargs=dict(min_obs=6))),
    ]
    for label, extra in configs:
        res = bt.run_backtest(
            snaps, enter_rank=20, exit_rank=40, min_history=18,
            **extra,
        )
        s = bt.summarize(res)
        print(f"\n=== {label} (SYNTHETIC DATA — numbers are not evidence of real edge) ===")
        for k, v in s.items():
            if k == "note":
                continue
            print(f"  {k:>22}: {v:.4f}" if isinstance(v, float) else f"  {k:>22}: {v}")
        if extra.get("weight_mode") == "validated_equal":
            final = res["group_weights"][-1]
            print(f"  final weight mode: {final['mode']}")
            print(f"  final group weights: " + ", ".join(
                f"{g}={w:.2f}" for g, w in (final["weights"] or {}).items() if w > 0))
