name: refresh-data

# Fetches market data on a schedule and commits the on-disk cache, so the
# Streamlit app reads pre-fetched files instead of calling Yahoo at request time.
# A batch job only has to succeed ONCE per run (and retries next run), which is
# far more reliable than a live app that must succeed on every page load.

on:
  schedule:
    # Times are UTC. 11:00 UTC ≈ 06:00 ET (pre-market). Add more lines for
    # intraday refreshes if you want fresher prices.
    - cron: "0 11 * * 1-5"
  workflow_dispatch: {}   # manual "Run workflow" button

permissions:
  contents: write          # needed to commit the refreshed cache

concurrency:
  group: refresh-data
  cancel-in-progress: false

jobs:
  refresh:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Fetch snapshot
        env:
          # ZERO-EDIT DEFAULT: USE_EDGAR=0 runs on Yahoo and needs no email.
          # To enable SEC EDGAR fundamentals (US filers), set USE_EDGAR to "1"
          # AND put a real contact email in EDGAR_USER_AGENT below (EDGAR 403s
          # blank/placeholder emails). EDGAR is the more reliable source but
          # requires that one edit — that's the only required edit in the repo.
          USE_EDGAR: "1"
          EDGAR_USER_AGENT: "Sentry akhamlin008@gmail.com"
          INCLUDE_HOLDERS_INSIDER: "0"
          DCF_CACHE_DIR: ".cache"
        run: python fetch_snapshot.py

      - name: Commit refreshed cache
        uses: stefanzweifel/git-auto-commit-action@v5
        with:
          commit_message: "chore: data refresh"
          file_pattern: ".cache/**"
