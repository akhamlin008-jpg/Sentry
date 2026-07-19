"""
cache_layer.py — disk-persistent, daily-TTL key/value cache.

WHY THIS EXISTS
---------------
Streamlit's @st.cache_data is in-memory and per-function. It dies on every
Codespaces rebuild / app reboot, and the Factor page and Risk page each get
their OWN copy because they wrap different functions. That is why opening the
app 100x/day means re-hammering Yahoo: every cold start starts from nothing.

This cache lives on disk and is shared by every module that imports it, so a
ticker fetched once is served from a local file for the rest of the TTL window,
no matter how many times any page is opened or whether the process restarted.

Pure stdlib (json) for dict payloads; pandas/parquet helpers for frames. No
network, no streamlit — unit-testable and reusable from the batch snapshot job.

KEY SCHEME
----------
Use hierarchical string keys: "fund:NVDA", "edgar:facts:AAPL", "market:universe".
Keys are sanitized to safe filenames. Values must be JSON-serializable for
get/put; use get_df/put_df for DataFrames.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
from pathlib import Path

# Point DCF_CACHE_DIR at a path that survives your environment's restarts.
# Under the repo (e.g. ".cache") persists in a committed Codespace; /tmp does not.
CACHE_DIR = Path(os.environ.get("DCF_CACHE_DIR", ".cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# FRESHNESS IS STORED *IN* THE PAYLOAD, NOT IN FILE MTIME.
# ---------------------------------------------------------
# The cache is committed to git and restored by actions/checkout, which sets
# every file's mtime to checkout time. An mtime-based TTL therefore reports
# age ≈ 0 seconds on every CI run, so entries NEVER expire and the "refresh"
# job silently re-serves the same data forever (this froze the whole pipeline
# on 2026-06-29). JSON entries now carry a {"_cached_at": epoch} envelope and
# parquet entries a "<file>.meta.json" sidecar; freshness is judged from that
# embedded timestamp only. Legacy files without a timestamp are treated as
# STALE, which forces exactly one full refetch after this change ships.
#
# DCF_CACHE_SERVE_STALE=1 makes get/get_df ignore TTLs and serve whatever is
# on disk (age is still reported truthfully by as_of/age_seconds). Set it for
# read-only consumers (the deployed Streamlit app) that must never hit the
# network at request time; leave it unset in the refresh Action.
SERVE_STALE = os.environ.get("DCF_CACHE_SERVE_STALE", "0") == "1"

_ENVELOPE_KEY = "_cached_at"
_ENVELOPE_VAL = "_value"


def _now() -> float:
    return time.time()


def _meta_path(key: str) -> Path:
    return CACHE_DIR / f"{_safe(key)}.meta.json"


def _embedded_ts(key: str):
    """Trustworthy write-time of a cache entry, or None if unknown (legacy)."""
    p = _path(key)
    if p.exists():
        try:
            obj = json.loads(p.read_text())
            if isinstance(obj, dict) and _ENVELOPE_KEY in obj:
                return float(obj[_ENVELOPE_KEY])
        except Exception:
            return None
        return None  # legacy JSON without envelope -> unknown age
    mp = _meta_path(key)
    if mp.exists():
        try:
            return float(json.loads(mp.read_text())[_ENVELOPE_KEY])
        except Exception:
            return None
    return None

DAY = 24 * 3600


def _safe(key: str) -> str:
    for ch in ('/', '\\', ':', ' ', '"', "'"):
        key = key.replace(ch, "_")
    return key


def _path(key: str, ext: str = "json") -> Path:
    return CACHE_DIR / f"{_safe(key)}.{ext}"


# --------------------------------------------------------------------------- #
# JSON payloads (dicts of floats / None / lists — e.g. a fetched ticker row)
# --------------------------------------------------------------------------- #
def get(key: str, max_age_sec: int = DAY):
    """Return cached value if present AND fresher than max_age_sec, else None.

    Freshness comes from the timestamp embedded at put() time — NOT the file
    mtime, which git checkout resets and which therefore proves nothing.
    Legacy entries without an embedded timestamp count as stale (unless
    DCF_CACHE_SERVE_STALE=1)."""
    p = _path(key)
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text())
    except Exception:
        return None
    if isinstance(obj, dict) and _ENVELOPE_KEY in obj:
        if SERVE_STALE or (_now() - float(obj[_ENVELOPE_KEY])) <= max_age_sec:
            return obj.get(_ENVELOPE_VAL)
        return None
    # legacy, un-enveloped payload: age unknowable -> stale by default
    return obj if SERVE_STALE else None


def put(key: str, value) -> bool:
    try:
        envelope = {_ENVELOPE_KEY: _now(), _ENVELOPE_VAL: value}
        _path(key).write_text(json.dumps(envelope, default=str))
        return True
    except Exception:
        return False


def delete(key: str) -> None:
    for ext in ("json", "parquet"):
        p = _path(key, ext)
        if p.exists():
            p.unlink()
    mp = _meta_path(key)
    if mp.exists():
        mp.unlink()


# --------------------------------------------------------------------------- #
# DataFrame payloads (returns matrix, etc.) via parquet
# --------------------------------------------------------------------------- #
def get_df(key: str, max_age_sec: int = DAY):
    import pandas as pd
    p = _path(key, "parquet")
    if not p.exists():
        return None
    ts = _embedded_ts(key)
    if not SERVE_STALE:
        if ts is None:                       # legacy parquet without sidecar
            return None
        if _now() - ts > max_age_sec:
            return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def put_df(key: str, df) -> bool:
    try:
        df.to_parquet(_path(key, "parquet"))
        _meta_path(key).write_text(json.dumps({_ENVELOPE_KEY: _now()}))
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Freshness introspection (used by the UI to show "data as of …")
# --------------------------------------------------------------------------- #
def as_of(key: str):
    ts = _embedded_ts(key)
    if ts is not None:
        return dt.datetime.fromtimestamp(ts)
    # legacy fallback: mtime (unreliable after a git checkout — see header)
    for ext in ("json", "parquet"):
        p = _path(key, ext)
        if p.exists():
            return dt.datetime.fromtimestamp(p.stat().st_mtime)
    return None


def fresh(key: str, max_age_sec: int = DAY) -> bool:
    ts = as_of(key)
    return ts is not None and (time.time() - ts.timestamp()) <= max_age_sec


def age_seconds(key: str):
    ts = as_of(key)
    return None if ts is None else time.time() - ts.timestamp()
