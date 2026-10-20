"""Validation oracles: risk_core / factor_core math checked against
independent implementations (gs-quant timeseries, scipy, statsmodels).
DEV/CI ONLY -- none of these libraries may become runtime imports
(tests/test_import_guard.py enforces that). Run with:  pytest -m oracle

Convention reconciliations, on the record:
- gs_quant.timeseries.zscores uses ddof=1 (scipy.stats.zscore); factor_core
  .zscore(robust=False) uses population std (ddof=0). Exact factor
  sqrt((n-1)/n) apart -- asserted exactly, not fudged with loose tolerance.
- gs_quant max_drawdown takes a PRICE series and returns a negative running
  drawdown; risk_core.max_drawdown takes RETURNS and reports a positive loss.
- gs_quant winsorize clips at mean +/- k*sigma, a DIFFERENT convention from
  factor_core's quantile clip, so it is not a valid oracle there; scipy's
  independently-implemented percentile is used instead.
- risk_core.ledoit_wolf_cc shrinks toward constant-correlation (LW 2004).
  sklearn's LedoitWolf shrinks toward scaled identity: a different estimator,
  NOT a valid oracle. It is deliberately absent here; its property tests
  live in test_factor_risk.py.
"""
import numpy as np
import pandas as pd
import pytest

import factor_core as fc
import risk_core as rk

gs_ts = pytest.importorskip(
    "gs_quant.timeseries", reason="oracle tests need requirements-dev.txt")
from scipy import stats as sps                      # noqa: E402  (dev dep)
import statsmodels.api as sm                        # noqa: E402  (dev dep)

pytestmark = pytest.mark.oracle

RNG = np.random.default_rng(7)
IDX = pd.bdate_range("2021-01-04", periods=500)

PX = pd.Series(100 * np.exp(np.cumsum(RNG.normal(2e-4, 0.011, 500))), index=IDX)
BENCH = pd.Series(80 * np.exp(np.cumsum(RNG.normal(1e-4, 0.014, 500))), index=IDX)
R_PX = PX.pct_change().dropna()
R_BENCH = BENCH.pct_change().dropna()


# ------------------------------------------------------------ drawdown

def test_max_drawdown_matches_gs_quant():
    ours, _, _ = rk.max_drawdown(R_PX.values)
    gs = gs_ts.max_drawdown(PX).iloc[-1]        # negative, price-based
    assert ours == pytest.approx(-gs, rel=1e-10)

def test_max_drawdown_indices_bracket_the_loss():
    mdd, peak, trough = rk.max_drawdown(R_PX.values)
    cum = np.cumprod(1.0 + R_PX.values)
    assert cum[trough] / cum[peak] - 1.0 == pytest.approx(-mdd)


# ------------------------------------------------- covariance / correlation

def test_correlation_matches_gs_quant_and_numpy():
    R = np.column_stack([R_PX.values, R_BENCH.values])
    ours = rk.cov_to_corr(rk.sample_cov(R))[0, 1]
    assert ours == pytest.approx(gs_ts.correlation(PX, BENCH).iloc[-1], abs=1e-12)
    assert ours == pytest.approx(np.corrcoef(R.T)[0, 1], abs=1e-12)

def test_sample_cov_matches_pandas():
    R = np.column_stack([R_PX.values, R_BENCH.values])
    expected = pd.DataFrame(R).cov().values     # ddof=1, independent impl
    assert np.allclose(rk.sample_cov(R), expected, atol=1e-15)


# ------------------------------------------------------------ regression

def test_single_factor_beta_matches_gs_quant():
    X = np.column_stack([np.ones(len(R_BENCH)), R_BENCH.values])
    coef, _, _ = rk.ols(R_PX.values, X)
    assert coef[1] == pytest.approx(gs_ts.beta(PX, BENCH).iloc[-1], abs=1e-12)

def test_ols_matches_gs_quant_linear_regression():
    y = pd.Series(RNG.normal(0, 1, 300), index=IDX[:300])
    x = pd.Series(RNG.normal(0, 1, 300), index=IDX[:300])
    lr = gs_ts.LinearRegression([x], y, fit_intercept=True)
    coef, _, r2 = rk.ols(y.values, np.column_stack([np.ones(300), x.values]))
    assert coef[0] == pytest.approx(lr.coefficient(0), abs=1e-12)
    assert coef[1] == pytest.approx(lr.coefficient(1), abs=1e-12)
    assert r2 == pytest.approx(lr.r_squared(), abs=1e-12)

def test_factor_betas_match_statsmodels_two_factor_ols():
    n = 400
    mkt = RNG.normal(0, 0.01, n)
    dy = RNG.normal(0, 0.0005, n)
    y = 0.8 * mkt - 3.0 * dy + RNG.normal(0, 0.005, n)
    bm, br, r2 = rk.factor_betas(y, mkt, dy)
    ref = sm.OLS(y, sm.add_constant(np.column_stack([mkt, dy]))).fit()
    assert bm == pytest.approx(ref.params[1], abs=1e-10)
    assert br == pytest.approx(ref.params[2], abs=1e-10)
    assert r2 == pytest.approx(ref.rsquared, abs=1e-10)


# --------------------------------------------------------- standardization

def test_zscore_matches_gs_quant_after_ddof_reconciliation():
    y = pd.Series(RNG.normal(0, 1, 200), index=IDX[:200])
    n = len(y)
    ours_ddof1 = fc.zscore(y.values, robust=False) * np.sqrt((n - 1) / n)
    assert np.allclose(ours_ddof1, gs_ts.zscores(y).values, atol=1e-12)

def test_robust_zscore_matches_scipy_mad():
    x = RNG.normal(5, 3, 501)
    mad = sps.median_abs_deviation(x, scale="normal")   # 1.4826 * raw MAD
    expected = (x - np.median(x)) / mad
    assert np.allclose(fc.zscore(x, robust=True), expected, atol=1e-12)

def test_winsorize_matches_scipy_percentile_clip():
    x = RNG.standard_t(3, size=400) * 10                # heavy tails on purpose
    lo, hi = sps.scoreatpercentile(x, 2), sps.scoreatpercentile(x, 98)
    assert np.allclose(fc.winsorize(x, 0.02, 0.98), np.clip(x, lo, hi), atol=1e-12)

def test_rank_to_normal_matches_scipy_ppf_on_blom_ranks():
    x = RNG.uniform(-100, 100, 999)
    order = np.argsort(np.argsort(x))
    expected = sps.norm.ppf((order + 0.5) / len(x))
    assert np.allclose(fc.rank_to_normal(x), expected, atol=1e-8)


# ------------------------------------------------- normal dist primitives

def test_acklam_ppf_matches_scipy():
    p = np.concatenate([[1e-9, 1e-6, 0.02424, 0.02426],       # tail + regime seam
                        np.linspace(0.001, 0.999, 199),
                        [1 - 1e-6, 1 - 1e-9]])
    ours = np.array([fc._norm_ppf(pi) for pi in p])
    assert np.allclose(ours, sps.norm.ppf(p), atol=1e-7)

def test_norm_cdf_matches_scipy():
    x = np.linspace(-8, 8, 321)
    ours = np.array([rk._norm_cdf(v) for v in x])
    assert np.allclose(ours, sps.norm.cdf(x), atol=1e-14)


# ------------------------------------------------------------- VaR family

def test_var_gaussian_matches_scipy_closed_form():
    r = RNG.normal(0.0002, 0.012, 1000)
    mu, sd = r.mean(), r.std(ddof=1)
    expected = -(mu + sps.norm.ppf(0.05) * sd)
    assert rk.var_gaussian(r, 0.95) == pytest.approx(expected, abs=1e-9)

def test_var_historical_matches_numpy_quantile():
    r = RNG.normal(0, 0.012, 1000)
    assert rk.var_historical(r, 0.95) == pytest.approx(-np.quantile(r, 0.05))

def test_cornish_fisher_reduces_to_gaussian_when_moments_vanish():
    # symmetric two-point mixture: skew 0; kurtosis differs, so compare on a
    # genuinely normal large sample instead
    r = RNG.normal(0.0, 0.01, 200_000)
    assert rk.var_cornish_fisher(r, 0.99) == pytest.approx(
        rk.var_gaussian(r, 0.99), rel=0.02)
