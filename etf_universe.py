"""
etf_universe.py — ETF instruments the system may trade AROUND the stock book:
hedges, leveraged exposure, and the raw material for the arbitrage page.

DESIGN RULES
------------
1. ETFs are NOT scored by factor_core and NEVER enter the long stock book.
   They are overlay instruments (hedge sleeve, decay-harvest pairs, sector
   proxies). Keeping them out of `universe.TICKERS` guarantees that.
2. Leverage numbers below are the funds' STATED daily objectives. Daily-reset
   leverage compounds path-dependently: over any horizon longer than one day a
   -1x fund is NOT the mirror of the index, and a 3x fund is NOT 3x the
   period return. That gap (volatility decay) is exactly what the
   decay-harvest scanner measures — never assume it away.
3. Inception dates differ per fund; some pairs have no data before the
   mid-2010s. Code must TRIM to available history rather than assume a start
   date. (Deliberately no inception dates hardcoded here — the data decides.)
"""
from __future__ import annotations

# ---- broad market -----------------------------------------------------------
BROAD = {
    "SPY": {"underlying": "S&P 500", "lev": 1},
    "QQQ": {"underlying": "Nasdaq-100", "lev": 1},
    "IWM": {"underlying": "Russell 2000", "lev": 1},
}

# ---- sector SPDRs (GICS sector proxies for the arb page) --------------------
# Maps GICS sector name (as used in sp500_constituents.csv) -> SPDR ticker.
SECTOR_ETF = {
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Energy": "XLE",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

# ---- leveraged / inverse ----------------------------------------------------
# lev is the stated DAILY multiple of the underlying.
LEVERAGED = {
    "SSO":  {"underlying": "S&P 500",    "lev": +2},
    "SDS":  {"underlying": "S&P 500",    "lev": -2},
    "UPRO": {"underlying": "S&P 500",    "lev": +3},
    "SPXU": {"underlying": "S&P 500",    "lev": -3},
    "SH":   {"underlying": "S&P 500",    "lev": -1},
    "QLD":  {"underlying": "Nasdaq-100", "lev": +2},
    "QID":  {"underlying": "Nasdaq-100", "lev": -2},
    "TQQQ": {"underlying": "Nasdaq-100", "lev": +3},
    "SQQQ": {"underlying": "Nasdaq-100", "lev": -3},
    "SOXL": {"underlying": "PHLX Semis", "lev": +3},
    "SOXS": {"underlying": "PHLX Semis", "lev": -3},
    "TNA":  {"underlying": "Russell 2000", "lev": +3},
    "TZA":  {"underlying": "Russell 2000", "lev": -3},
}

# Bull/bear siblings on the same underlying at the same |leverage| — the pairs
# the decay-harvest scanner analyzes (short BOTH legs).
DECAY_PAIRS = [
    ("UPRO", "SPXU"),
    ("TQQQ", "SQQQ"),
    ("SSO",  "SDS"),
    ("QLD",  "QID"),
    ("SOXL", "SOXS"),
    ("TNA",  "TZA"),
]

# Hedge instruments the backtester's circuit-breaker overlay may park freed
# exposure in, with the divisor needed to express $1 of index short.
HEDGES = {
    "SH":   {"lev": -1},   # $1 of short exposure per $1
    "SDS":  {"lev": -2},   # $0.50 per $1 of desired short exposure
    "SPXU": {"lev": -3},   # $0.33 per $1
}

ALL_ETFS: list[str] = sorted(set(BROAD) | set(SECTOR_ETF.values()) | set(LEVERAGED))
