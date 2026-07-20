"""
history_layer.py — LONG daily price/volume history (default 2015-01-01 →
present) for the stock universe + ETF overlay instruments.

WHY A SEPARATE LAYER
--------------------
data_layer's PRICE_LOOKBACK is 3y on purpose: the live app doesn't need more,
and shorter batch downloads fail less. Crisis validation (COVID 2020,
2016-2019 regime, the 2025 crash) and the arbitrage page's long-baseline
correlations need ~10 years. That fetch is heavy, so it runs in CI
(crisis.yml), commits history_close.parquet / history_volume.parquet, and
everything else just READS those files.

SURVIVORSHIP WARNING — READ BEFORE QUOTING ANY PRE-2023 NUMBER
--------------------------------------------------------------
This layer fetches TODAY'S constituents through history. Names that were
delisted or dropped from the index before today simply have no data, so any
long-portfolio return computed on pre-~2023 windows is BIASED UPWARD by
survivorship. crisis_core labels every affected window `survivorship_biased:
true`. Those windows are valid as a STRESS TEST of drawdown behavior, cost
drag, turnover, and the breaker/hedge machinery in a crash — they are NOT an
unbiased alpha estimate. A true PIT rebuild needs a dated constituents file
(see pit_layer's build guidance).

Loading order for consumers (load_history):
  1. .cache/history_close.parquet (+volume) if present  — long history from CI
  2. the live cache's 3y market_close parquet           — always present
"""
from __future__ import annotations

import glob
import os
import time

import numpy as np
import pandas as pd

HIST_START = os.environ.get("SENTRY_HIST_START", "2015-01-01")
CACHE_DIR = os.environ.get("DCF_CACHE_DIR", ".cache")
CLOSE_PATH = os.path.join(CACHE_DIR, "history_close.parquet")
VOL_PATH = os.path.join(CACHE_DIR, "history_volume.parquet")
BATCH = 100


def _yf_batched(tickers: list[str], start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Batched yfinance download. Only callable where Yahoo is reachable
    (CI / local machine) — not from restricted sandboxes."""
    import yfinance as yf
    closes, vols = [], []
    for i in range(0, len(tickers), BATCH):
        chunk = tickers[i:i + BATCH]
        df = yf.download(chunk, start=start, interval="1d", auto_adjust=True,
                         progress=False, group_by="column", threads=True)
        if df is None or df.empty:
            continue
        c = df["Close"] if "Close" in df else df
        v = df["Volume"] if "Volume" in df else None
        if isinstance(c, pd.Series):
            c = c.to_frame(chunk[0])
        closes.append(c)
        if v is not None:
            if isinstance(v, pd.Series):
                v = v.to_frame(chunk[0])
            vols.append(v)
        time.sleep(1.0)                      # be polite between batches
    close = pd.concat(closes, axis=1) if closes else pd.DataFrame()
    vol = pd.concat(vols, axis=1) if vols else pd.DataFrame()
    return close, vol


def _stooq_fallback(tickers: list[str], start: str) -> pd.DataFrame:
    """Per-name Stooq fallback for tickers Yahoo returned empty. Slow; used
    only to patch holes, never as the primary path."""
    try:
        from pandas_datareader import data as web
    except ImportError:
        return pd.DataFrame()
    out = {}
    for t in tickers:
        try:
            df = web.DataReader(t, "stooq", start=start)
            if df is not None and not df.empty:
                out[t] = df.sort_index()["Close"]
            time.sleep(0.6)
        except Exception:
            continue
    return pd.DataFrame(out)


def fetch_history(extra_tickers: list[str] | None = None,
                  start: str = HIST_START) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch, patch holes via Stooq, write both parquets, return (close, vol)."""
    from universe import TICKERS
    import etf_universe as eu
    universe = list(dict.fromkeys(TICKERS + eu.ALL_ETFS + (extra_tickers or [])))

    close, vol = _yf_batched(universe, start)
    missing = [t for t in universe
               if t not in close.columns or close[t].notna().sum() < 60]
    if missing:
        patch = _stooq_fallback(missing, start)
        for t in patch.columns:
            close[t] = patch[t]
    close = close.sort_index().dropna(how="all")
    vol = vol.reindex(close.index) if not vol.empty else vol

    os.makedirs(CACHE_DIR, exist_ok=True)
    close.to_parquet(CLOSE_PATH)
    if not vol.empty:
        vol.to_parquet(VOL_PATH)
    still = [t for t in universe if t not in close.columns]
    print(f"history: {close.shape[0]} days x {close.shape[1]} names, "
          f"{close.index.min().date()} -> {close.index.max().date()}; "
          f"unfetchable: {len(still)} {still[:10]}")
    return close, vol


def load_history() -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    """Best available close/volume matrices without any network.
    Returns (close, volume_or_None, source_label)."""
    if os.path.exists(CLOSE_PATH):
        close = pd.read_parquet(CLOSE_PATH)
        vol = pd.read_parquet(VOL_PATH) if os.path.exists(VOL_PATH) else None
        return close, vol, "long-history parquet (crisis.yml)"
    # fall back to the live 3y cache (hash-suffixed filenames)
    cands = sorted(glob.glob(os.path.join(CACHE_DIR, "market_close_*.parquet")))
    vands = sorted(glob.glob(os.path.join(CACHE_DIR, "market_vol_*.parquet")))
    if not cands:
        raise FileNotFoundError(
            "no price history found — run fetch_snapshot.py or the crisis workflow")
    close = pd.read_parquet(cands[-1])
    vol = pd.read_parquet(vands[-1]) if vands else None
    return close, vol, "live 3y cache (no ETF columns; run crisis.yml for 2015+)"


def dollar_adv(close: pd.DataFrame, vol: pd.DataFrame | None,
               window: int = 21) -> pd.DataFrame | None:
    """Rolling average daily dollar volume; None if no volume data."""
    if vol is None:
        return None
    v = vol.reindex(close.index)[[c for c in close.columns if c in vol.columns]]
    return (close[v.columns] * v).rolling(window, min_periods=5).mean()


if __name__ == "__main__":
    fetch_history()
