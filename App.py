"""
Sentry · DCF Engine — two-stage DCF over the ~500-name universe.

At 500 names this page can NO LONGER scrape Yahoo per ticker. It reads from the
same pipeline as the Factor/Risk pages: data_layer.load_universe() (SEC EDGAR
fundamentals + Stooq/Yahoo prices, all disk-cached). So the DCF page, factor
page, and risk page all share one cache and one fetch — open any of them and the
others are warm.

The per-stock editable card is now PICKER-driven: choose a ticker to model in
detail, rather than rendering 500 expanders. The summary table covers all names.

All finance math still lives in dcf_core.py (no streamlit/yfinance there).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import dcf_core as core
import cache_layer as kv
import data_layer as dl
from universe import TICKERS, NVIDIA_GREEN

DEFAULT_TERMINAL = 0.025
DEFAULT_ERP = 0.045
DEFAULT_RF = 0.043
FALLBACK_G1 = 0.10
CAGR_CAP = (-0.10, 0.25)
MOS_IMPLAUSIBLE = 300.0   # |MoS%| above this is flagged as a data/units problem


# =============================================================================
# Data — one cached load shared with the other pages
# =============================================================================
@st.cache_data(ttl=24 * 3600, show_spinner="Loading universe (EDGAR + prices, cached)…")
def _load(tickers):
    return dl.load_universe(list(tickers))


def _stock_from_row(r: dict) -> dict:
    """Map a data_layer row -> the dict dcf_core expects."""
    fs = r.get("fcf_series") or []
    s = {
        "ticker": r["ticker"],
        "fcf": r.get("_fcf"),
        "fcf_series": fs,
        "sbc": None,
        "shares": r.get("_shares"),
        "cash": r.get("_cash"),
        "debt": r.get("_debt"),
        "price": r.get("_price"),
        "market_cap": r.get("_mktcap"),
        "interest_expense": r.get("_interest"),
        "tax_rate": 0.21,
        "analyst_g5": None,
        "hist_cagr": core.robust_cagr(fs, cap=CAGR_CAP),
        "beta": r.get("_beta"),
        "sector": r.get("sector"),
        "source": r.get("_source_fund", "—"),
        "error": r.get("error"),
    }
    s["missing"] = [k for k in ("fcf", "shares", "cash", "debt", "price")
                    if s.get(k) is None]
    return s


# =============================================================================
# Presentation
# =============================================================================
st.set_page_config(page_title="Sentry · DCF Engine", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
  html, body, [class*="css"] {{ font-family: 'Inter', system-ui, sans-serif; }}
  .stApp {{ background:
      radial-gradient(1200px 600px at 80% -10%, #14210a 0%, transparent 55%),
      #0a0a0a; }}
  div[data-testid="stExpander"] {{ background: #111313; border: 1px solid #232323;
      border-radius: 14px; margin-bottom: 10px; }}
  div[data-testid="stMetricValue"] {{ font-weight: 800; }}
  .stNumberInput input {{ background:#0e0e0e; border:1px solid #2a2a2a; color:#eee; }}
  .stButton button {{ background:{NVIDIA_GREEN}; color:#0a0a0a; border:none;
      font-weight:700; border-radius:8px; }}
  .badge {{ display:inline-block; padding:2px 9px; border-radius:999px;
      font-size:0.72rem; font-weight:700; }}
  .under {{ background:rgba(118,185,0,.15); color:{NVIDIA_GREEN}; border:1px solid {NVIDIA_GREEN}; }}
  .over  {{ background:rgba(255,77,79,.12); color:#ff6b6b; border:1px solid #ff6b6b; }}
</style>
""", unsafe_allow_html=True)

# --- HEADER (native components — cannot be escaped or hidden) ---------------- #
st.title("Sentry · DCF Engine")
st.caption("build: v3-reversal")
st.caption("Two-stage DCF across ~500 US large caps · fundamentals from SEC EDGAR · "
           "verify against the 10-K before you trade on it.")
st.divider()

# --- data load -------------------------------------------------------------- #
payload = _load(tuple(TICKERS))
rows = payload["rows"]
stocks = {r["ticker"]: _stock_from_row(r) for r in rows}

_mix = payload.get("source_mix", {})
_src = ", ".join(f"{v} {k}" for k, v in sorted(_mix.items())) or "—"
_loaded = sum(1 for r in rows if not r.get("error"))

# --- sidebar ---------------------------------------------------------------- #
st.sidebar.header("Macro knobs")
rf = st.sidebar.number_input("Risk-free % (10y UST)",
                             value=round((payload.get("rf") or DEFAULT_RF) * 100, 2),
                             step=0.1) / 100
erp = st.sidebar.number_input("Equity risk premium %",
                              value=DEFAULT_ERP * 100, step=0.25) / 100
term = st.sidebar.number_input("Terminal growth %",
                               value=DEFAULT_TERMINAL * 100, step=0.25) / 100

st.sidebar.header("Modeling")
fcf_method = st.sidebar.selectbox("FCF base year", ["latest", "mean", "median"], index=0,
    help="Normalize the launch FCF to reduce one-off distortion.")
fcf_n = st.sidebar.slider("…over N years", 2, 5, 3, disabled=(fcf_method == "latest"))
burden_sbc = st.sidebar.toggle("Burden FCF with stock-based comp", value=False)

if st.sidebar.button("🔄 Refresh data"):
    _load.clear()
    for tk in TICKERS:
        kv.delete(f"fund:{tk}:{int(dl.USE_EDGAR)}:{int(dl.INCLUDE_HOLDERS_INSIDER)}")
    st.rerun()

st.caption(f"Data as of **{payload['as_of']:%Y-%m-%d %H:%M}** · "
           f"{_loaded}/{len(TICKERS)} loaded · fundamentals: {_src} · "
           f"risk-free {rf*100:.2f}% · ERP {erp*100:.2f}%")


def _base_fcf(s):
    series = s.get("fcf_series") or ([] if s.get("fcf") is None else [s["fcf"]])
    base = core.normalize_fcf(series, fcf_method, fcf_n) if series else s.get("fcf")
    if burden_sbc and base is not None and s.get("sbc"):
        base = base - abs(s["sbc"])
    return base


def _seeded(s):
    return core.derive_assumptions(s, rf, erp, term, CAGR_CAP, FALLBACK_G1)


# --- summary dashboard (all names) ------------------------------------------ #
def build_summary():
    recs = []
    for tk in TICKERS:
        s = stocks.get(tk)
        if s is None or s.get("error"):
            recs.append({"Ticker": tk, "Sector": s.get("sector") if s else "—",
                         "Fair": None, "Price": s.get("price") if s else None,
                         "MoS %": None, "Flags": "fetch failed"})
            continue
        auto, src = _seeded(s)
        price = s.get("price")
        if auto["r"] is None:
            recs.append({"Ticker": tk, "Sector": s.get("sector"), "Fair": None,
                         "Price": price, "MoS %": None, "Flags": "no WACC"})
            continue
        res = core.two_stage_dcf(_base_fcf(s), auto["g1"], auto["g2"], auto["gt"],
                                 auto["r"], s.get("cash"), s.get("debt"), s.get("shares"))
        if res.error:
            recs.append({"Ticker": tk, "Sector": s.get("sector"), "Fair": None,
                         "Price": price, "MoS %": None, "Flags": res.error})
            continue
        mos = (res.fair - price) / price * 100 if price else None
        if mos is not None and abs(mos) > MOS_IMPLAUSIBLE:
            recs.append({"Ticker": tk, "Sector": s.get("sector"), "Fair": None,
                         "Price": price, "MoS %": None,
                         "Flags": "implausible — likely data/units issue"})
            continue
        flags = "; ".join(res.flags + core.data_quality_flags(s)) or "—"
        recs.append({"Ticker": tk, "Sector": s.get("sector"), "Fair": res.fair,
                     "Price": price, "MoS %": mos, "Flags": flags})
    return pd.DataFrame(recs)


summ = build_summary()
n_valued = int(summ["MoS %"].notna().sum())
st.subheader(f"Coverage — {n_valued} of {len(TICKERS)} names valued")
st.caption("Financials and negative-FCF names show no DCF by design (FCF-based "
           "valuation doesn't apply); they're excluded here, not broken.")

# sort by MoS desc, undervalued first; Nones last
summ_sorted = summ.sort_values("MoS %", ascending=False, na_position="last").reset_index(drop=True)
styled = (summ_sorted.style
          .format({"Fair": "${:,.2f}", "Price": "${:,.2f}", "MoS %": "{:+.1f}%"}, na_rep="—")
          .background_gradient(cmap="RdYlGn", subset=["MoS %"], vmin=-60, vmax=120))
st.dataframe(styled, hide_index=True, width="stretch", height=560,
             column_config={"Flags": st.column_config.TextColumn(width="large")})

# --- single-stock detail (picker-driven) ------------------------------------ #
st.divider()
st.subheader("Model a single stock")
valid = [tk for tk in TICKERS if stocks.get(tk) and not stocks[tk].get("error")]
pick = st.selectbox("Pick a ticker to model in detail", valid,
                    index=0 if valid else None)

if pick:
    s = stocks[pick]
    auto, src = _seeded(s)
    base_fcf = _base_fcf(s)

    for fld in ("g1", "g2", "gt", "r"):
        key = f"{pick}_{fld}"
        if key not in st.session_state:
            v = auto[fld]
            st.session_state[key] = round((v if v is not None else
                                           (FALLBACK_G1 if fld == "g1" else 0.09)) * 100, 2)
    fkey = f"{pick}_fcf"
    if fkey not in st.session_state:
        st.session_state[fkey] = round(base_fcf / 1e6, 1) if base_fcf else 0.0

    st.markdown(f"**{pick}** · {s.get('sector') or '—'} · source: {s.get('source')} "
                f"· g1 src: {src} · WACC: {'auto' if auto['r'] else 'fallback'}")
    if s.get("missing"):
        st.caption(f"⚠️ missing: {', '.join(s['missing'])}")

    def m(x): return "—" if x is None else f"${x/1e6:,.0f}M"
    sh = s.get("shares"); be = s.get("beta"); pr = s.get("price")
    L, R = st.columns(2)
    with L:
        st.markdown("**Auto-filled data** (FCF editable)")
        fcf_m = st.number_input("Free Cash Flow ($M)", key=fkey, step=100.0)
        st.write(f"Shares: {'—' if sh is None else f'{sh/1e6:,.0f}M'}"
                 f"  ·  Beta: {'—' if be is None else f'{be:.2f}'}")
        st.write(f"Mkt cap: {m(s.get('market_cap'))}  ·  Cash: {m(s.get('cash'))}"
                 f"  ·  Debt: {m(s.get('debt'))}")
        st.write(f"Current price: {'—' if pr is None else f'${pr:,.2f}'}")
    with R:
        st.markdown("**Assumptions** (auto-seeded, editable)")
        g1 = st.number_input("Yr 1–5 growth %", key=f"{pick}_g1", step=0.5)
        g2 = st.number_input("Yr 6–10 growth %", key=f"{pick}_g2", step=0.5)
        gt = st.number_input("Terminal growth %", key=f"{pick}_gt", step=0.25)
        r = st.number_input("Discount rate (WACC) %", key=f"{pick}_r", step=0.25)

    res = core.two_stage_dcf(fcf_m * 1e6, g1/100, g2/100, gt/100, r/100,
                             s.get("cash"), s.get("debt"), s.get("shares"))
    if res.error:
        st.warning(res.error)
    else:
        cols = st.columns(3)
        cols[0].metric("Fair value / share", f"${res.fair:,.2f}")
        cols[1].metric("Current price", "—" if pr is None else f"${pr:,.2f}")
        if pr:
            mos = (res.fair - pr) / pr * 100
            if abs(mos) > MOS_IMPLAUSIBLE:
                cols[2].metric("Margin of safety", "n/a")
                cols[2].caption("⚠️ implausible — likely data/units issue")
            else:
                badge = "under" if mos > 0 else "over"
                cols[2].metric("Margin of safety", f"{mos:+.1f}%")
                cols[2].markdown(f"<span class='badge {badge}'>"
                                 f"{'Undervalued' if mos > 0 else 'Overvalued'}</span>",
                                 unsafe_allow_html=True)
        for f in res.flags + core.data_quality_flags(s):
            st.caption(f"⚠️ {f}")
        with st.expander("Projection detail"):
            proj = pd.DataFrame(res.rows)
            proj["FCF"] = (proj["FCF"] / 1e6).round(0)
            proj["PV of FCF"] = (proj["PV of FCF"] / 1e6).round(0)
            st.dataframe(proj, hide_index=True, width="stretch")
            st.caption(f"PV(FCF) {m(res.pv_fcf_sum)} · PV(TV) {m(res.pv_tv)} "
                       f"({res.tv_fraction:.0%} of EV) · EV {m(res.ev)} · "
                       f"Equity {m(res.equity)}")

st.divider()
st.caption("WACC = CAPM cost of equity · E/V + after-tax cost of debt · D/V. "
           "Result is hypersensitive to WACC and terminal growth — one scenario, "
           "not a price target. Verify fundamentals against SEC EDGAR filings.")