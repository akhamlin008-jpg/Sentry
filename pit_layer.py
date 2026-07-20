"""
pit_layer.py — point-in-time (PIT) data contract for backtesting.

WHY THIS FILE EXISTS
--------------------
The single largest source of fake alpha in fundamental backtests is using
information that was not knowable on the simulated trade date:

  1. LOOKAHEAD BIAS   — using a fiscal-year 10-K's numbers on a rebalance date
                        *before* the 10-K was actually filed, or using restated
                        figures instead of the originally reported ones.
  2. SURVIVORSHIP BIAS — scoring today's index constituents through history.
                        Names that were delisted, acquired at a discount, or
                        went bankrupt vanish from the sample, which inflates
                        returns of any long strategy.

This module does NOT fetch data. It defines the immutable snapshot format the
backtester consumes, plus validators that refuse to run a backtest whose data
violates the PIT contract. Fetching is the responsibility of a builder script
(see build guidance at the bottom) because true PIT data has to be assembled
carefully from filing metadata, and no free vendor hands it to you directly.

CONTRACT
--------
A backtest is a chronological list of `PITSnapshot`s. For snapshot at date D:

  * every fundamental metric in `rows` must carry `asof` = the date the figure
    became PUBLIC (EDGAR accession acceptance date, NOT fiscal period end),
    and asof + REPORTING_LAG_DAYS <= D must hold;
  * `universe` is the constituent list AS OF D (from a historical constituents
    file), never today's list;
  * `fwd_returns` are the realized returns from D to the NEXT snapshot date,
    including delisting returns for names that die inside the window
    (a name acquired for cash gets its deal return; a bankruptcy gets ~-100%
    unless you have the actual final print — never silently drop it, dropping
    IS survivorship bias);
  * analyst estimates (e.g. yfinance's current 5y growth) are FORBIDDEN in
    historical snapshots — only fields with a verifiable historical asof date
    are allowed. `validate_snapshot` enforces this via ALLOWED_PIT_FIELDS.

All of this is mechanical to check and is checked. What this module cannot do
is conjure the historical data itself; see BUILDING REAL SNAPSHOTS below.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

# Extra conservatism on top of the filing acceptance date. EDGAR acceptance
# timestamps are end-of-day-ish; a 2-day lag guarantees the market had the
# filing before we "trade" on it.
REPORTING_LAG_DAYS = 2

# Metric keys permitted in historical snapshots. Everything here can be
# reconstructed from dated primary sources (EDGAR XBRL facts + daily prices).
# Deliberately EXCLUDED: analyst_g5 (no free PIT history of consensus
# estimates), short_pct_float / days_to_cover / insider / institutional
# (FINRA short interest and Form 4/13F can be made PIT, but only with their
# own publication-date handling — add them later with explicit asof fields,
# don't sneak them in).
ALLOWED_PIT_FIELDS = {
    # identity / classification
    "ticker", "sector",
    # price-derived (asof = price date, trivially PIT)
    "price", "market_cap", "mom_12_1", "mom_6_1", "ret_5d", "ret_21d",
    "adv_dollars",
    # fundamentals (asof = filing acceptance date + lag)
    "fcf", "fcf_yield", "ebit_ev", "earnings_yield", "sales_ev",
    "gross_margin", "op_margin", "roe", "roa", "debt_to_equity",
    "interest_coverage", "accruals", "rev_cagr", "fcf_cagr", "eps_cagr",
    "shares", "cash", "debt", "beta", "mos",
    # per-field provenance
    "asof_fundamentals", "asof_prices",
}


@dataclass
class PITSnapshot:
    """Everything knowable at `date`, plus the realized forward window."""
    date: dt.date
    universe: list[str]                      # constituents AS OF date
    rows: list[dict]                         # metric dicts, keys ⊆ ALLOWED_PIT_FIELDS
    fwd_returns: dict[str, float]            # ticker -> simple return date -> next date
    delisted: dict[str, float] = field(default_factory=dict)
    # ticker -> final realized return for names that leave the universe
    # mid-window (merger cash-out, bankruptcy). These names MUST appear here
    # rather than being dropped.
    trailing_returns: object = None          # optional (T,N) DataFrame ending at date,
                                             # for covariance estimation (past-only)
    aux: dict = field(default_factory=dict)
    # Overlay instruments (hedge/inverse ETFs) that are tradeable but must NOT
    # be scored or enter the long stock book, so they are deliberately kept
    # out of `universe`/`rows`. Shape:
    #   {ticker: {"fwd_return": float, "adv_dollars": float}}
    # PIT-safe by the same rule as fwd_returns: written by the builder, read
    # by the backtester only AFTER weights are frozen.


class PITViolation(Exception):
    pass


def validate_snapshot(snap: PITSnapshot, strict: bool = True) -> list[str]:
    """Return a list of contract violations (raises in strict mode).

    Checks are necessarily partial — code can verify internal consistency and
    field provenance, but it cannot verify that the *values* you loaded truly
    came from the original (non-restated) filing. That guarantee has to come
    from how the snapshot builder sources data (original XBRL facts keyed by
    accession number, not a vendor's restated history).
    """
    errs: list[str] = []
    tickset = set(snap.universe)

    for r in snap.rows:
        tk = r.get("ticker", "?")
        bad = set(r.keys()) - ALLOWED_PIT_FIELDS
        if bad:
            errs.append(f"{tk}: non-PIT fields present: {sorted(bad)}")
        af = r.get("asof_fundamentals")
        if af is None:
            # allowed only if the row carries no fundamental fields at all
            fund_keys = {"fcf", "roe", "roa", "ebit_ev", "earnings_yield"} & set(r)
            if fund_keys:
                errs.append(f"{tk}: fundamentals present but no asof_fundamentals date")
        else:
            if af + dt.timedelta(days=REPORTING_LAG_DAYS) > snap.date:
                errs.append(f"{tk}: fundamentals asof {af} not public (with lag) by {snap.date}")
            if (snap.date - af).days > 455:
                errs.append(f"{tk}: fundamentals stale (> ~15 months old) — exclude or flag")
        if r.get("ticker") not in tickset:
            errs.append(f"{tk}: row not in as-of universe")

    # forward returns must cover the universe or be explained by `delisted`
    missing = tickset - set(snap.fwd_returns) - set(snap.delisted)
    if missing:
        errs.append(f"{len(missing)} universe names lack forward return AND are not "
                    f"in `delisted` — dropping them is survivorship bias: "
                    f"{sorted(missing)[:8]}...")

    if strict and errs:
        raise PITViolation("; ".join(errs))
    return errs


def validate_chronology(snaps: list[PITSnapshot]) -> None:
    dates = [s.date for s in snaps]
    if dates != sorted(dates) or len(set(dates)) != len(dates):
        raise PITViolation("snapshots must be strictly increasing in date")


# --------------------------------------------------------------------------- #
# BUILDING REAL SNAPSHOTS (guidance, not code — requires network + effort)
# --------------------------------------------------------------------------- #
# 1. Historical constituents: you need a dated S&P 500 membership table
#    (add/remove dates). There is no official free feed; sources people use
#    include hand-maintained GitHub datasets reconstructed from S&P press
#    releases and Wikipedia's change log table. Whatever you use, spot-check
#    a dozen known index changes against the original press releases before
#    trusting it — I can't vouch for any particular free dataset's accuracy.
# 2. Fundamentals: EDGAR "company facts" / frames APIs expose XBRL facts with
#    the accession number and acceptance datetime per fact. Key every value by
#    (concept, fiscal period, accession acceptance date) and, at simulation
#    date D, select the latest fact whose acceptance date + lag <= D. Using
#    acceptance date (not period end, not 'filed' calendar date alone) is the
#    whole ballgame.
# 3. Prices incl. delistings: free daily sources generally EXCLUDE delisted
#    tickers, which quietly reintroduces survivorship bias through the price
#    file even if your constituents file is correct. Handle each departure
#    explicitly: merger => deal terms return; bankruptcy => assume total loss
#    unless you have the final prints. Document every manual decision in the
#    snapshot's `delisted` dict so results are auditable.
# 4. Freeze snapshots to disk (e.g. parquet per date) and never regenerate
#    them silently — reproducibility of the input is as important as the code.
