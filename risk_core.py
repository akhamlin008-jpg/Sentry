"""
paper_trader.py — monthly, long-only, ERC-weighted paper-trading algo.

WHAT IT DOES (once per scheduled run)
-------------------------------------
1. Loads the universe, runs the 5-factor model (factor_core.score_universe),
   takes the top-N names by composite (long-only — no shorts).
2. Builds a Ledoit-Wolf-shrunk covariance on those names and computes
   Equal-Risk-Contribution (risk-parity) weights via risk_core.erc_weights.
3. Reads your Alpaca PAPER account equity and current positions.
4. Computes target dollar value per name (ERC weight x equity x INVEST_FRAC),
   diffs against current holdings, and rebalances:
     - exits names no longer in the book (close_position),
     - buys/sells the dollar difference for names in the book (notional orders).
5. Logs everything.

SAFETY MODEL  (read this)
-------------------------
* DRY-RUN BY DEFAULT. It computes and logs the full order list but submits
  NOTHING unless LIVE_PAPER=1. Watch at least one cycle's logged orders and
  sanity-check them before enabling submission.
* PAPER ONLY. TradingClient(paper=True) — it talks to paper-api.alpaca.markets.
  It physically cannot touch a live/real-money account with paper keys.
* It checks the market clock and skips submission when the market is closed
  (notional/fractional orders are market-DAY and only fill during RTH).

Verified against alpaca-py (the current SDK; alpaca-trade-api is deprecated).
Keys come from env: ALPACA_API_KEY / ALPACA_SECRET_KEY (GitHub Actions secrets).
Never commit keys.

HONEST CAVEATS
--------------
* Paper results do not reflect real fills, slippage, or liquidity and can differ
  materially from live trading.
* Monthly rebalancing yields ~12 data points/year — this validates slowly.
* Class-share symbols (e.g. BRK-B) are converted to Alpaca's dot form (BRK.B);
  if any symbol is rejected it's logged and skipped, not retried blindly.
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np

import data_layer as dl
import factor_core as fc
import risk_core as rk
from universe import TICKERS

# ---- config (env-overridable) ---------------------------------------------- #
N_LONG = int(os.environ.get("PAPER_N_LONG", "20"))
MIN_GROUPS = int(os.environ.get("PAPER_MIN_GROUPS", "3"))
INVEST_FRAC = float(os.environ.get("PAPER_INVEST_FRAC", "0.98"))  # cash buffer
MIN_TRADE_USD = float(os.environ.get("PAPER_MIN_TRADE_USD", "1"))  # skip dust
PLACEHOLDER_EQUITY = 100_000.0   # used only in dry-run when no keys present
LIVE = os.environ.get("LIVE_PAPER", "0") == "1"
FORCE = os.environ.get("PAPER_FORCE", "0") == "1"   # ignore the monthly guard

_STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state")
_MARKER = os.path.join(_STATE_DIR, "last_rebalance.txt")


def _already_rebalanced_this_month() -> bool:
    try:
        with open(_MARKER) as f:
            return f.read().strip() == dt.date.today().strftime("%Y-%m")
    except FileNotFoundError:
        return False


def _mark_rebalanced():
    os.makedirs(_STATE_DIR, exist_ok=True)
    with open(_MARKER, "w") as f:
        f.write(dt.date.today().strftime("%Y-%m"))


def log(msg: str):
    print(f"[paper] {msg}", flush=True)


def alpaca_symbol(tk: str) -> str:
    # Our universe uses Yahoo/Stooq dash form (BRK-B); Alpaca uses dot form.
    return tk.replace("-", ".")


# --------------------------------------------------------------------------- #
# 1-2. model + ERC weights
# --------------------------------------------------------------------------- #
def target_book():
    """Return {ticker: weight} for the long book, ERC-weighted, or {} on failure."""
    payload = dl.load_universe(TICKERS)
    rows = payload["rows"]
    rets = payload["returns"]
    if rets is None or rets.empty:
        log("ERROR: no returns matrix — cannot size a book. Aborting.")
        return {}, payload

    result = fc.score_universe(rows)
    tickers = [r["ticker"] for r in rows]
    longs, _ = fc.select_candidates(tickers, result, n_long=N_LONG, n_short=0,
                                    min_groups_long=MIN_GROUPS)
    names = [t for (t, _c, _p) in longs]
    log(f"top {len(names)} longs by composite: {', '.join(names)}")

    # keep only names with usable return history
    usable = [t for t in names if t in rets.columns and rets[t].notna().sum() > 60]
    dropped = [t for t in names if t not in usable]
    if dropped:
        log(f"dropped (thin price history): {', '.join(dropped)}")
    if len(usable) < 2:
        log("ERROR: fewer than 2 names with usable history. Aborting.")
        return {}, payload

    R = rets[usable].dropna(how="any").values
    S, delta = rk.ledoit_wolf_cc(R)
    w = rk.erc_weights(S)[0]
    w = np.asarray(w, float)
    w = w / w.sum()
    book = {t: float(wi) for t, wi in zip(usable, w)}
    log(f"ERC weights (shrinkage delta={delta:.2f}): " +
        ", ".join(f"{t} {wi*100:.1f}%" for t, wi in book.items()))
    return book, payload


# --------------------------------------------------------------------------- #
# 3-5. reconcile against Alpaca and (optionally) trade
# --------------------------------------------------------------------------- #
def run():
    started = dt.datetime.now()
    log(f"run {started:%Y-%m-%d %H:%M} | LIVE={LIVE} | N={N_LONG} | invest={INVEST_FRAC:.0%}")

    # Once-per-month guard: scheduled to attempt on several days (in case the
    # 1st is a holiday), but only actually rebalances once a month in LIVE mode.
    if LIVE and not FORCE and _already_rebalanced_this_month():
        log(f"already rebalanced for {dt.date.today():%Y-%m} — skipping. "
            f"(Set PAPER_FORCE=1 to override.)")
        return 0

    book, _ = target_book()
    if not book:
        return 1

    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")

    client = None
    equity = PLACEHOLDER_EQUITY
    current = {}   # symbol -> market_value (USD)

    if key and secret:
        try:
            from alpaca.trading.client import TradingClient
            client = TradingClient(key, secret, paper=True)  # paper endpoint
            acct = client.get_account()
            equity = float(acct.equity)
            for p in client.get_all_positions():
                current[p.symbol] = float(p.market_value)
            log(f"account equity ${equity:,.0f} | {len(current)} current positions")
        except Exception as e:  # noqa: BLE001
            log(f"WARN: could not reach Alpaca ({e}). Falling back to dry-run sizing.")
            client = None
    else:
        log("no API keys in env — dry-run sizing on placeholder equity.")

    investable = equity * INVEST_FRAC
    targets = {alpaca_symbol(t): investable * w for t, w in book.items()}

    # ---- build the order plan ---- #
    buys, sells, exits = [], [], []
    for sym, tgt in targets.items():
        cur = current.get(sym, 0.0)
        delta = tgt - cur
        if delta > MIN_TRADE_USD:
            buys.append((sym, delta))
        elif delta < -MIN_TRADE_USD:
            sells.append((sym, -delta))
    for sym, cur in current.items():
        if sym not in targets and cur > MIN_TRADE_USD:
            exits.append(sym)

    log(f"PLAN: {len(buys)} buys, {len(sells)} sells, {len(exits)} exits")
    for sym, d in sorted(buys, key=lambda x: -x[1]):
        log(f"  BUY  {sym:<6} ${d:,.0f}")
    for sym, d in sorted(sells, key=lambda x: -x[1]):
        log(f"  SELL {sym:<6} ${d:,.0f}")
    for sym in exits:
        log(f"  EXIT {sym:<6} (close full position)")

    if not LIVE:
        log("DRY-RUN: no orders submitted. Set LIVE_PAPER=1 to enable.")
        return 0
    if client is None:
        log("LIVE requested but no Alpaca client — aborting without trading.")
        return 1

    # ---- market-hours gate ---- #
    try:
        if not client.get_clock().is_open:
            log("market closed — skipping submission this run (will retry next run).")
            return 0
    except Exception as e:  # noqa: BLE001
        log(f"WARN: clock check failed ({e}); not submitting to be safe.")
        return 1

    # ---- submit ---- #
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    def submit(sym, notional, side):
        try:
            req = MarketOrderRequest(symbol=sym, notional=round(notional, 2),
                                     side=side, time_in_force=TimeInForce.DAY)
            client.submit_order(order_data=req)
            log(f"  submitted {side.value} {sym} ${notional:,.0f}")
        except Exception as e:  # noqa: BLE001
            log(f"  FAILED {side.value} {sym}: {e}")

    # exits first (free up buying power), then sells, then buys
    for sym in exits:
        try:
            client.close_position(sym)
            log(f"  closed {sym}")
        except Exception as e:  # noqa: BLE001
            log(f"  FAILED close {sym}: {e}")
    for sym, d in sells:
        submit(sym, d, OrderSide.SELL)
    for sym, d in buys:
        submit(sym, d, OrderSide.BUY)

    _mark_rebalanced()
    log("submission complete; monthly marker written.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
