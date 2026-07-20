"""
pages/4_arbitrage.py — statistical arbitrage scanner: relative-value pairs,
leveraged-ETF decay harvest ("perpetual motion"), and sector correlation
reversion.

Truth in labeling, shown to the user and not buried: none of these are
riskless. They are convergence trades with real, named failure modes (see
arb_core's module docstring — the page surfaces the same warnings inline).
Data source is the committed price cache; the decay tab needs ETF history,
which the crisis workflow fetches.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import arb_core as ac
import history_layer as hl
from universe import SECTORS, THEME_CSS, NVIDIA_GREEN

st.set_page_config(page_title="Sentry · Arbitrage", layout="wide")
st.markdown(THEME_CSS, unsafe_allow_html=True)
st.title("Arbitrage Scanner")
st.caption("Statistical convergence trades — relative value, leveraged-ETF decay, "
           "sector correlation reversion. Nothing on this page is riskless; every "
           "edge shown is compensation for a failure mode named next to it.")
st.divider()


@st.cache_data(ttl=6 * 3600, show_spinner="Loading price history…")
def _load():
    close, vol, source = hl.load_history()
    adv = hl.dollar_adv(close, vol)
    return close, adv, source


close, adv, source = _load()
st.markdown(f'<div class="note">price matrix: {close.shape[0]} days × '
            f'{close.shape[1]} names · {close.index.min().date()} → '
            f'{close.index.max().date()} · source: {source}</div>',
            unsafe_allow_html=True)

tab_rv, tab_pm, tab_corr = st.tabs(
    ["Relative Value", "Perpetual Motion (decay harvest)", "Sector Correlation Reversion"])

# --------------------------------------------------------------------------- #
with tab_rv:
    st.subheader("Within-sector pair dislocations")
    c1, c2, c3, c4 = st.columns(4)
    lookback = c1.slider("Lookback (days)", 120, 500, 252, 10)
    corr_floor = c2.slider("Min pair correlation", 0.3, 0.9, 0.6, 0.05)
    per_sector = c3.slider("Names per sector (by liquidity)", 10, 40, 25, 5)
    top_n = c4.slider("Show top", 10, 50, 25, 5)

    pairs = ac.relative_value_pairs(close, SECTORS, adv=adv, lookback=lookback,
                                    corr_floor=corr_floor,
                                    max_names_per_sector=per_sector, top=top_n)
    if pairs.empty:
        st.info("No pairs clear the z ≥ 1 and finite-half-life filters at these settings.")
    else:
        st.dataframe(pairs.style.format({"z": "{:+.2f}", "beta": "{:.2f}",
                                         "corr": "{:.2f}"}),
                     use_container_width=True, height=420)
        sel = st.selectbox("Chart a pair's z-scored spread",
                           pairs["pair"].tolist())
        a, b = sel.split("/")
        s = ac.pair_spread_series(close, a, b, lookback)
        if s is not None:
            st.line_chart(s.rename(f"{sel} spread z"), height=260)
    st.markdown('<div class="note">Failure mode: spreads that are wide for a '
                'reason (guidance cuts, fraud, index deletion) keep widening — '
                'divergence loss is unbounded. Half-life is an AR(1) estimate, '
                'not a cointegration proof. Position both legs and size small.</div>',
                unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
with tab_pm:
    st.subheader("Short both legs of a leveraged bull/bear pair")
    st.markdown(
        "Daily-reset leveraged funds bleed volatility decay; shorting **both** "
        "siblings 50/50 and **holding the position static** for the window "
        "collects it from both sides. (Rebalancing daily back to equal legs "
        "would earn exactly zero on perfect trackers — the decay accrues only "
        "to the position you leave alone, which is why the position drifts.) "
        "This is the trade people call *perpetual motion* — it is not. The "
        "income pays you for: **trend risk** (after a run you're net short the "
        "winner with growing exposure), **borrow fees** on inverse-leveraged "
        "funds (often several percent a year — not in this data, check your "
        "broker's live rate; it can flip the sign), and **unbounded short "
        "loss / buy-in risk**.")
    window = st.slider("Rolling window (days)", 21, 252, 63, 21)
    dh = ac.decay_harvest(close, window=window)
    if dh.empty or (dh.get("status") == "no data").all():
        st.warning("No leveraged-ETF history in the price matrix yet — run the "
                   "crisis-validation workflow (Actions → crisis-validation) to "
                   "fetch ETF prices; this tab populates on the next load.")
    else:
        ok = dh[dh.status == "ok"]
        if not ok.empty:
            st.dataframe(ok.drop(columns=["status"]).style.format({
                f"median_{window}d_ret": "{:+.2%}",
                "pct_windows_positive": "{:.0f}%",
                f"worst_{window}d_ret": "{:+.2%}",
                "ann_vol_of_trade": "{:.1%}"}),
                use_container_width=True)
            pick = st.selectbox("Equity curve of the short-both trade",
                                ok["pair"].tolist())
            bull, bear = pick.split("+")
            curve = ac.decay_harvest_curve(close, bull, bear, rebalance_days=window)
            if curve is not None:
                st.line_chart(curve.rename(
                    f"short {pick}, static within {window}d blocks, $1 gross"),
                    height=260)
        missing = dh[dh.status != "ok"]
        if not missing.empty:
            st.markdown('<div class="note">Missing pairs: ' +
                        ", ".join(missing["pair"]) + " — fetched by crisis.yml.</div>",
                        unsafe_allow_html=True)
    st.markdown('<div class="note">Read the worst-window column before the '
                'median. The trade loses exactly when markets trend hard — the '
                'same regimes in which shorts get bought in.</div>',
                unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
with tab_corr:
    st.subheader("Sector-pair correlation vs its own baseline")
    c1, c2 = st.columns(2)
    short_win = c1.slider("Short window (days)", 21, 126, 63, 21)
    long_win = c2.slider("Baseline window (days)", 252, 1260, 504, 126)
    tbl, sr = ac.sector_correlation_reversion(close, SECTORS,
                                              short_win=short_win,
                                              long_win=long_win)
    if tbl.empty:
        st.info("Not enough history for this baseline — shorten it, or run the "
                "crisis workflow to fetch 2015+ data.")
    else:
        st.dataframe(tbl.style.format({"corr_long": "{:.2f}",
                                       f"corr_{short_win}d": "{:.2f}",
                                       "gap": "{:+.2f}",
                                       "perf_spread": "{:+.1%}"}),
                     use_container_width=True, height=420)
        st.markdown(
            f'<div class="note">Sector indices are equal-weight composites of '
            f'the stock universe (so this works before ETF history exists). '
            f'<b>decoupled</b> = correlation collapsed vs baseline → candidate '
            f'reconvergence: long the laggard, short the leader, sized to the '
            f'perf spread. <b>abnormally coupled</b> = crowding — dispersion '
            f'candidate. Correlations regime-shift toward 1 in crashes; that is '
            f'when this table is most wrong.</div>', unsafe_allow_html=True)
        picks = st.multiselect("Overlay sector return indices",
                               list(sr.columns), default=list(sr.columns)[:3])
        if picks:
            st.line_chart((1 + sr[picks].iloc[-long_win:]).cumprod(), height=280)
