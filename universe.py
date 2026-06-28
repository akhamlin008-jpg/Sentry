"""universe.py — single source of truth: the ~500 US large-cap universe (S&P 500
constituents as a top-500-by-market-cap proxy), their GICS sectors, and SEC CIK
numbers, loaded from sp500_constituents.csv.

WHY A CSV LOADED FROM THE MODULE DIR
------------------------------------
The constituent list lives in sp500_constituents.csv next to this file and is
loaded relative to THIS module's directory (not the current working directory),
so it resolves correctly whether launched from the repo root, a Codespace, or
Streamlit Cloud. No path surprises.

The CSV ships the CIK per name, so the EDGAR layer skips its ticker->CIK lookup
entirely (faster, fewer calls at 500 names). Sector comes from the CSV too, so
the data layer no longer makes a per-name Yahoo .info scrape just to get sector
— that scrape was the heaviest, most rate-limit-prone call and is gone.

To refresh the list later, re-download S&P 500 constituents and regenerate the
CSV (columns: ticker,sector,cik; tickers in Yahoo form, e.g. BRK-B).
"""
from __future__ import annotations

import csv
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_CSV = os.path.join(_HERE, "sp500_constituents.csv")

TICKERS: list[str] = []
SECTORS: dict[str, str] = {}
CIKS: dict[str, str] = {}

try:
    with open(_CSV, newline="") as _f:
        for _r in csv.DictReader(_f):
            _t = _r["ticker"].strip().upper()
            if not _t:
                continue
            TICKERS.append(_t)
            SECTORS[_t] = _r.get("sector", "").strip() or "Unknown"
            CIKS[_t] = _r.get("cik", "").strip()
except FileNotFoundError:
    # Fallback keeps the app importable even if the CSV is missing; the data
    # layer will simply have a tiny universe rather than crashing on import.
    TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
    SECTORS = {t: "Information Technology" for t in TICKERS}
    CIKS = {}

# De-dupe, preserve order.
TICKERS = list(dict.fromkeys(TICKERS))

# All names are US filers by construction (S&P 500), so EDGAR covers every one
# and there is no foreign-currency contamination. Kept for the data layer's
# guard, now empty.
NON_US_EDGAR: set[str] = set()

NVIDIA_GREEN = "#76B900"

THEME_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', system-ui, sans-serif; }}
.stApp {{ background:
  radial-gradient(1200px 600px at 80% -10%, #14210a 0%, transparent 55%), #0a0a0a; }}
div[data-testid="stMetricValue"] {{ font-weight: 800; }}
.stButton button {{ background:{NVIDIA_GREEN}; color:#0a0a0a; border:none; font-weight:700; border-radius:8px; }}
.cov-good {{ color:{NVIDIA_GREEN}; font-weight:700; }}
.cov-mid  {{ color:#e0b000; font-weight:700; }}
.cov-bad  {{ color:#ff6b6b; font-weight:700; }}
.note {{ color:#9a9a9a; font-size:0.84rem; }}
</style>
"""


def hero(title_html, subtitle):
    return f'<div class="hero"><h1>{title_html}</h1><p>{subtitle}</p></div>'
