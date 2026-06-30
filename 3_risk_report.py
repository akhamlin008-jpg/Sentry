name: paper-trade

# Monthly long-only ERC rebalance against the Alpaca PAPER account.
#
# SAFETY: paper_trader.py is DRY-RUN unless LIVE_PAPER=1. To go live on PAPER
# money, (1) add ALPACA_API_KEY and ALPACA_SECRET_KEY as repo secrets
# (Settings -> Secrets and variables -> Actions), and (2) set LIVE_PAPER to "1"
# below. Until then it just logs the intended orders.
#
# It attempts on the 1st-5th of each month at 14:35 UTC (~9:35 ET, just after
# the open) but a once-per-month marker (state/last_rebalance.txt) makes it
# idempotent — it rebalances only once even though it runs several days, so a
# holiday on the 1st doesn't cause a missed month.

on:
  schedule:
    - cron: "35 14 1-5 * *"
  workflow_dispatch: {}     # manual "Run workflow" button (use to dry-run anytime)

permissions:
  contents: write           # to commit the monthly marker + refreshed cache

concurrency:
  group: paper-trade
  cancel-in-progress: false

jobs:
  trade:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: pip install -r requirements.txt

      - name: Run paper trader
        env:
          ALPACA_API_KEY: ${{ secrets.ALPACA_API_KEY }}
          ALPACA_SECRET_KEY: ${{ secrets.ALPACA_SECRET_KEY }}
          # Flip to "1" ONLY after you've reviewed a dry-run's logged orders.
          LIVE_PAPER: "0"
          USE_EDGAR: "1"
          EDGAR_USER_AGENT: "Sentry akhamlin008@gmail.com"
          DCF_CACHE_DIR: ".cache"
        run: python paper_trader.py

      - name: Persist cache + monthly marker
        uses: stefanzweifel/git-auto-commit-action@v5
        with:
          commit_message: "chore: paper-trade run"
          file_pattern: ".cache/** state/**"
