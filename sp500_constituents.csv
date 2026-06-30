# Sentry

DCF + multi-factor scoring + portfolio-risk engine, built to run reliably on
free infrastructure even under heavy daily use.

## Run it

```bash
pip install -r requirements.txt
streamlit run App.py
```

In a Codespace it auto-starts (see `.devcontainer/devcontainer.json`). The app
has three pages: the DCF engine (`App.py`), Factor Analysis, and Risk Report.

## Why it stays reliable

* **Disk cache** (`cache_layer.py`) — fundamentals/prices are cached to disk with
  a daily TTL and survive restarts, so opening the app 100×/day ≈ one fetch per
  ticker per day, not 100.
* **Hardened network** (`net_layer.py`) — a shared browser-impersonating session
  with exponential backoff on rate limits; low concurrency.
* **Official free data** — SEC EDGAR for fundamentals (opt-in), Stooq as a price
  fallback, Yahoo otherwise.
* **Scheduled refresh** (`.github/workflows/refresh.yml`) — a GitHub Action
  fetches data a few times a day and commits the cache, so the live app reads
  pre-fetched files and makes zero network calls.

## Headers

Page headers are rendered with native Streamlit components (`st.title` /
`st.caption`), not raw HTML — so they can't be hidden or escaped, and the theme
lives in `.streamlit/config.toml` (where Streamlit actually reads it).

## Before trusting EDGAR numbers

Set a real contact email so SEC EDGAR doesn't reject you, then smoke-test:

```bash
export EDGAR_USER_AGENT="Your Name you@email.com"
python edgar_layer.py        # should print sane AAPL fundamentals
```

Then enable EDGAR fundamentals with `USE_EDGAR=1`. See `CHANGES.md` for the full
rundown of what each file does and what to verify.

## Optional toggles (environment variables)

```bash
USE_EDGAR=1                       # SEC EDGAR fundamentals for US filers
EDGAR_USER_AGENT="You you@email"  # REQUIRED when USE_EDGAR=1
INCLUDE_HOLDERS_INSIDER=1         # re-enable the weak institutional/insider scrapes
DCF_CACHE_DIR=.cache              # cache location (default: .cache)
```

## Tests

```bash
python Test_core.py
python test_factor_risk.py
```

Both are offline (no network) and should print all-pass.
