"""Offline unit tests: closed-form or construction-guaranteed answers only."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

import cost_model as cm
import portfolio_core as pc
import risk_core as rk
import signal_research as sr
from pit_layer import PITSnapshot, PITViolation, validate_snapshot


# ---------------- portfolio_core ---------------- #

def test_cap_and_redistribute_sums_and_caps():
    w = np.array([0.6, 0.2, 0.1, 0.1])
    out = pc.cap_and_redistribute(w, 0.30)
    assert out.max() <= 0.30 + 1e-9
    assert abs(out.sum() - 1.0) < 1e-9

def test_sector_cap_enforced():
    w = np.array([0.3, 0.3, 0.2, 0.2])
    secs = ["Tech", "Tech", "Fin", "Util"]
    out = pc.sector_cap(w, secs, 0.40)
    assert out[np.array(secs) == "Tech"].sum() <= 0.40 + 1e-9
    assert abs(out.sum() - 1.0) < 1e-9

def test_hysteresis_keeps_incumbent_between_bands():
    tk = [f"T{i}" for i in range(50)]
    comp = -np.arange(50, dtype=float)          # T0 best ... T49 worst
    # incumbent T25 ranks 26th: outside enter(20) but inside exit(40) -> kept
    book = pc.hysteresis_select(tk, comp, ["T25"], enter_rank=20, exit_rank=40)
    assert "T25" in book
    # incumbent T45 ranks 46th: beyond exit band -> dropped
    book = pc.hysteresis_select(tk, comp, ["T45"], enter_rank=20, exit_rank=40)
    assert "T45" not in book

def test_risk_reducing_shorts_only_accepts_vol_reducers():
    # 2 correlated longs + 1 candidate positively correlated with the book
    # + 1 candidate NEGATIVELY correlated (shorting it would ADD vol -> reject)
    rho = 0.6
    S = np.array([
        [0.04, rho*0.04, 0.5*0.04, -0.5*0.04],
        [rho*0.04, 0.04, 0.5*0.04, -0.5*0.04],
        [0.5*0.04, 0.5*0.04, 0.04, 0.0],
        [-0.5*0.04, -0.5*0.04, 0.0, 0.04]])
    comp = np.array([2.0, 1.5, -2.0, -2.0])     # both candidates "disliked"
    w, acc = pc.risk_reducing_shorts(np.array([0.5, 0.5]), S, [0, 1], [2, 3],
                                     comp, short_score_pct=60,
                                     max_short_w=0.3, max_gross_short=0.5)
    assert w[2] < 0                              # hedging short accepted
    assert w[3] == 0                             # anti-correlated short rejected
    sig_before = rk.port_vol(np.array([0.5, 0.5, 0, 0.0]), S)
    assert rk.port_vol(w, S) < sig_before        # net effect: vol strictly down

def test_shorts_respect_gross_cap():
    N = 6
    S = 0.04 * (0.7 * np.ones((N, N)) + 0.3 * np.eye(N))
    comp = np.array([1, 1, -1, -1, -1, -1.0])
    w, acc = pc.risk_reducing_shorts(np.array([0.5, 0.5]), S, [0, 1],
                                     [2, 3, 4, 5], comp, short_score_pct=80,
                                     max_short_w=0.05, max_gross_short=0.08)
    assert -w[w < 0].sum() <= 0.08 + 1e-9

def test_circuit_breaker_hysteresis():
    eq = [100, 100, 80]                          # 20% dd -> triggers at 0.15
    assert pc.circuit_breaker_scale(eq, 0.15, 0.5, 0.05) == 0.5
    eq = [100, 100, 80, 97]                      # recovered to -3% -> restored
    assert pc.circuit_breaker_scale(eq, 0.15, 0.5, 0.05) == 1.0

def test_rc_cap_binds():
    S = np.diag([0.09, 0.01, 0.01, 0.01])        # one very risky name
    w, flags = pc.build_long_weights(S, max_name_w=0.5, max_sector_w=1.0,
                                     max_rc_pct=0.30)
    _, pct = rk.risk_contributions(w, S)
    assert pct.max() <= 0.30 + 0.02              # small tolerance: iterative


# ---------------- cost_model ---------------- #

def test_cost_monotone_in_participation_and_capacity_veto():
    c1 = cm.trade_cost_fraction(1e5, 1e8)
    c2 = cm.trade_cost_fraction(1e6, 1e8)
    assert c2 > c1 > 0
    assert np.isnan(cm.trade_cost_fraction(1e7, 1e8))   # 10% ADV > 5% cap


# ---------------- pit_layer ---------------- #

def _snap(date, rows, fwd, delisted=None):
    return PITSnapshot(date=date, universe=[r["ticker"] for r in rows],
                       rows=rows, fwd_returns=fwd, delisted=delisted or {})

def test_pit_rejects_future_fundamentals():
    d = dt.date(2024, 6, 28)
    rows = [{"ticker": "AAA", "roe": 0.2, "asof_fundamentals": dt.date(2024, 6, 27)}]
    with pytest.raises(PITViolation):            # 1-day-old + 2-day lag -> not public
        validate_snapshot(_snap(d, rows, {"AAA": 0.01}))

def test_pit_rejects_non_pit_field_and_dropped_delisting():
    d = dt.date(2024, 6, 28)
    rows = [{"ticker": "AAA", "analyst_g5": 0.1},
            {"ticker": "BBB", "price": 10.0}]
    snap = _snap(d, rows, {"AAA": 0.01})          # BBB has no fwd ret, not delisted
    errs = validate_snapshot(snap, strict=False)
    assert any("non-PIT" in e for e in errs)
    assert any("survivorship" in e for e in errs)

def test_pit_accepts_clean_snapshot():
    d = dt.date(2024, 6, 28)
    rows = [{"ticker": "AAA", "roe": 0.2, "price": 5.0,
             "asof_fundamentals": dt.date(2024, 5, 1)}]
    assert validate_snapshot(_snap(d, rows, {"AAA": 0.01})) == []


# ---------------- signal_research ---------------- #

def test_spearman_perfect_and_reversed():
    a = np.arange(20.0)
    assert abs(sr._spearman(a, a) - 1.0) < 1e-12
    assert abs(sr._spearman(a, -a) + 1.0) < 1e-12

def test_neutralize_removes_beta_tilt():
    rng = np.random.default_rng(0)
    n = 300
    beta = rng.normal(1, 0.3, n)
    score = 2.0 * beta + rng.normal(0, 0.1, n)   # score ~ pure beta
    rows = [{"beta": b, "market_cap": 1e10} for b in beta]
    resid = sr.neutralize(score, rows, against=("beta",))
    assert abs(sr._spearman(resid, beta)) < 0.15


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))


# ---------------- exposure_core ---------------- #
import exposure_core as ex

def test_vol_target_ex_ante_halves_at_double_vol():
    # per-period sigma such that annualized = 24%, target 12% -> scale 0.5
    sig = 0.24 / np.sqrt(12)
    assert abs(ex.scale_from_ex_ante(sig, 0.12, 12) - 0.5) < 1e-12

def test_vol_target_caps_and_no_opinion_cases():
    assert ex.scale_from_ex_ante(1e-9, 0.12, 12, max_scale=1.0) == 1.0   # calm -> capped
    assert ex.scale_from_ex_ante(np.nan, 0.12, 12) == 1.0                # no estimate
    assert ex.scale_from_realized([0.01, -0.01], 0.12, 12, min_obs=6) == 1.0  # short history
    big = ex.scale_from_ex_ante(10.0, 0.12, 12, min_scale=0.25)
    assert big == 0.25                                                    # floor binds

def test_combined_scale_is_conservative_min():
    sig = 0.24 / np.sqrt(12)                       # ex-ante says 0.5
    calm = [0.001] * 12                            # realized says calm -> 1.0
    assert ex.combined_scale(sig, calm, 0.12, 12) == 0.5


# ---------------- walk-forward weights: no-peek property ---------------- #

def test_walk_forward_weights_ignore_the_future():
    import copy
    from demo_synthetic_backtest import snaps
    i = 18
    w1, d1 = sr.walk_forward_weights(snaps, i, min_obs=6)
    future_mangled = copy.deepcopy(snaps)
    for sn in future_mangled[i:]:                  # corrupt everything at/after i
        for tk in sn.fwd_returns:
            sn.fwd_returns[tk] = 9.9
    w2, d2 = sr.walk_forward_weights(future_mangled, i, min_obs=6)
    assert w1 == w2 and d1["mode"] == d2["mode"]

def test_walk_forward_weights_sum_to_one_and_prior_mode_early():
    from demo_synthetic_backtest import snaps
    w, d = sr.walk_forward_weights(snaps, 2, min_obs=6)
    assert d["mode"] == "ignorance_prior"
    assert abs(sum(w.values()) - 1.0) < 1e-9
    w, d = sr.walk_forward_weights(snaps, len(snaps), min_obs=6)
    assert abs(sum(w.values()) - 1.0) < 1e-9
