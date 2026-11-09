"""
GEX Dashboard — ETF option levels converted to futures prices.

Computes Gamma Flip (all + 0DTE), Max Pain, Call/Put Walls (all + 0DTE),
Delta Wall, Charm Wall, Vanna Wall, and ranked gamma / charm / vanna strikes.

The headline feature: it generates a complete, pre-filled Pine script with a
one-click copy button. Paste it over the Pine Editor and every level is
already in place — no typing numbers into settings boxes.

Run locally:   streamlit run gex_dashboard.py
Deploy free:   push to GitHub -> share.streamlit.io -> point at this file
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import gex_core as core

st.set_page_config(page_title="GEX → Futures", layout="wide",
                   initial_sidebar_state="collapsed")


@st.cache_data(ttl=300, show_spinner="Pulling the option chain…")
def run(asset_name, max_dte, use_volume, conv_mode, n_gamma):
    return core.analyze(asset_name, max_dte=max_dte, use_volume=use_volume,
                        conv_mode=conv_mode, n_gamma=n_gamma)


# ──────────────────────────────── CONTROLS ─────────────────────────────────
st.title("GEX → Futures")

a1, a2 = st.columns([2, 1])
asset_name = a1.selectbox("Instrument", list(core.ASSETS.keys()))
A = core.ASSETS[asset_name]
has_index = A["index"] is not None

max_dte = a2.selectbox("Expirations", [1, 7, 30, 90], index=1,
                       format_func=lambda d: f"≤ {d} DTE",
                       help="≤7 for index futures. Metals need ≤30 — GLD and SLV rarely have weeklies.")

c1, c2, c3 = st.columns(3)
basis_src = c1.selectbox("Weight by", ["Open interest", "Volume"])
conv_mode = c2.selectbox(
    "Conversion",
    ["Ratio + basis", "Pure ratio"] if has_index else ["Pure ratio"],
    help="Ratio + basis needs a cash index. Metals have none here, so they use the direct futures/ETF ratio.",
)
n_gamma = c3.slider("Ranked gamma strikes", 0, 10, 10)

res = run(asset_name, max_dte, basis_src == "Volume", conv_mode, n_gamma)

if "error" in res:
    st.error(res["error"])
    st.stop()

conv = res["conv"]
etf_px, index_px, fut_px = res["etf_px"], res["index_px"], res["fut_px"]

# ──────────────────────────────── HEADER ───────────────────────────────────
cols = st.columns(5 if has_index else 4)
cols[0].metric(A["etf"], f"{etf_px:,.2f}")
i = 1
if has_index:
    cols[1].metric(A["index"].lstrip("^"),
                   f"{index_px:,.2f}" if not np.isnan(index_px) else "—")
    i = 2
cols[i].metric(A["label"], f"{fut_px:,.2f}" if not np.isnan(fut_px) else "—")
cols[i + 1].metric("Multiplier", f"{res['mult']:,.4f}" if not np.isnan(res["mult"]) else "—",
                   help="What a 1-point move in the ETF is worth in futures points.")
cols[i + 2].metric("Basis", f"{res['basis']:+,.1f}" if has_index else "n/a",
                   help="Futures premium over cash. Folded into the ratio when there's no index.")

tg = res["total_gex"]
st.markdown(
    f"### {'−Γ  NEGATIVE GAMMA' if res['negative_gamma'] else '+Γ  positive gamma'}"
    f"  ·  net GEX {'+' if tg >= 0 else '−'}${abs(tg)/1e9:,.2f}B per 1%"
)

# ───────────────────── COPY-AND-PASTE PINE SCRIPT ──────────────────────────
pine = core.build_pine(res)

with st.expander("📋  Ready-to-paste TradingView script  —  every level pre-filled",
                 expanded=True):
    st.caption(
        "Hover the block and click the copy icon in its top-right corner. "
        "In TradingView open Pine Editor, select all, paste over it, Save, "
        "then Add to chart. Nothing to type in."
    )
    st.code(pine, language="javascript")
    st.download_button("Download as .pine file", pine,
                       file_name=f"gex_snapshot_{A['label']}.pine",
                       mime="text/plain")

# ──────────────────────────── TABLE + CHART ────────────────────────────────
tbl = core.levels_table(res)
left, right = st.columns([3, 2])

with right:
    st.dataframe(tbl, hide_index=True, use_container_width=True, height=620)
    st.caption(
        f"If you'd rather type them in by hand, use the {A['etf']} column with the "
        f"indicator's conversion on Auto. The {A['label']} column is what those "
        "convert to right now — handy as a sanity check."
    )
    st.download_button("Download levels as CSV", tbl.to_csv(index=False),
                       file_name=f"gex_levels_{A['label']}.csv", mime="text/csv")

with left:
    agg = res["agg"]
    fig = go.Figure()
    y = [float(conv(s)) for s in agg["strike"].values]
    fig.add_trace(go.Bar(
        x=agg["gex"].values / 1e9, y=y, orientation="h",
        marker_color=["#6f9f78" if g > 0 else "#b5676b" for g in agg["gex"].values],
        hovertemplate=A["label"] + " %{y:,.2f}<br>%{x:,.3f} $B<extra></extra>",
    ))

    for _, r in tbl.iterrows():
        name = r["Level"]
        col = core.COLORS.get(name)
        if col is None:
            if "Charm" in name:
                col = core.CHARM_RAMP[0]
            elif "Vanna" in name:
                col = core.VANNA_RAMP[0]
            else:
                col = core.GAMMA_RAMP[0]
        major = name in core.COLORS
        fig.add_hline(y=r[A["label"]], line_color=col, line_width=2 if major else 1,
                      annotation_text=f"{name}  {r[A['label']]:,.2f}",
                      annotation_position="right", annotation_font_color=col,
                      annotation_font_size=10)

    if not np.isnan(fut_px):
        fig.add_hline(y=fut_px, line_color="#ffffff", line_width=1, line_dash="dot",
                      annotation_text=f"{A['label']} {fut_px:,.2f}",
                      annotation_position="left")

    fig.update_layout(
        template="plotly_dark", height=880, margin=dict(l=10, r=160, t=30, b=10),
        xaxis_title="Net gamma exposure ($B per 1% move)",
        yaxis_title=f"{A['label']} price", showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Gamma profile across spot (where the flip comes from)"):
    gf = go.Figure()
    gf.add_trace(go.Scatter(x=[float(conv(g)) for g in res["grid"]],
                            y=res["curve"] / 1e9, mode="lines",
                            line_color=core.GAMMA_RAMP[6]))
    gf.add_hline(y=0, line_color="#888", line_width=1)
    if not np.isnan(fut_px):
        gf.add_vline(x=fut_px, line_color="#fff", line_dash="dot")
    gf.update_layout(template="plotly_dark", height=320,
                     xaxis_title=f"{A['label']} price", yaxis_title="Total GEX ($B)",
                     margin=dict(l=10, r=10, t=20, b=10),
                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(gf, use_container_width=True)

st.caption(
    f"Generated {res['generated'].strftime('%Y-%m-%d %H:%M UTC')} · "
    "Data: Yahoo via yfinance, roughly 15-minute delayed. Greeks computed with "
    "Black-Scholes from chain IV. Dealer positioning is an assumed convention "
    "(short calls, long puts), not a measurement — vendors differ here, which is "
    "why their walls differ from these. Informational only, not trading advice."
)
