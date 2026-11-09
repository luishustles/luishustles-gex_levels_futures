"""
Psychological levels — round numbers around the current price, scaled against
how far the instrument actually travels in a day.

A round level on its own says nothing. What makes it useful is whether price
can plausibly reach it today, so every level here is shown with its distance
in points and measured against the recent median session range.

Lives in the pages/ folder so Streamlit shows it in the sidebar.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Psych levels", layout="wide")

SYMBOLS = {
    "Nasdaq 100 futures (NQ / MNQ)": ("NQ=F", [25, 50, 100, 250, 500]),
    "S&P 500 futures (ES / MES)": ("ES=F", [5, 10, 25, 50, 100]),
    "Gold futures (GC / MGC)": ("GC=F", [5, 10, 25, 50, 100]),
    "Russell 2000 futures (RTY)": ("RTY=F", [5, 10, 25, 50, 100]),
}

MAJOR, MINOR, PRICE = "#b5676b", "#5f8bb5", "#e8e8e3"


@st.cache_data(ttl=120, show_spinner="Pulling price…")
def load(symbol):
    intra = yf.Ticker(symbol).history(period="5d", interval="1m")
    daily = yf.Ticker(symbol).history(period="1y", interval="1d")
    if intra.empty or daily.empty:
        return None
    d = daily.dropna(subset=["High", "Low", "Close"])
    return {
        "price": float(intra["Close"].dropna().iloc[-1]),
        "asof": intra.index[-1],
        "ranges": (d["High"] - d["Low"]).values,
        "today_high": float(intra["High"].iloc[-390:].max()),
        "today_low": float(intra["Low"].iloc[-390:].min()),
    }


# ──────────────────────────────── CONTROLS ─────────────────────────────────
st.title("Psychological levels")

c1, c2, c3 = st.columns([2, 1, 1])
name = c1.selectbox("Instrument", list(SYMBOLS.keys()))
symbol, spacings = SYMBOLS[name]
step = c2.selectbox("Level spacing (points)", spacings, index=2)
span = c3.slider("Levels each side", 2, 10, 5)

data = load(symbol)
if data is None:
    st.error("No price returned. Yahoo may be rate-limiting — wait a minute and rerun.")
    st.stop()

price = data["price"]
med_range = float(np.median(data["ranges"][-60:]))
p90_range = float(np.quantile(data["ranges"][-60:], 0.90))

# ──────────────────────────────── HEADER ───────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Last", f"{price:,.2f}")
m2.metric("Median day range", f"{med_range:,.0f} pts",
          help="Median high-to-low over the last 60 sessions.")
m3.metric("Big day (p90)", f"{p90_range:,.0f} pts",
          help="One session in ten is larger than this.")
m4.metric("Nearest level",
          f"{round(price / step) * step:,.0f}",
          f"{round(price / step) * step - price:+,.0f} pts")

# ──────────────────────────── BUILD THE LADDER ─────────────────────────────
base = round(price / step) * step
levels = [base + i * step for i in range(-span, span + 1)]

rows = []
for lv in levels:
    dist = lv - price
    # a level is "major" when it's also round at a coarser scale
    coarse = step * 4 if step * 4 <= 1000 else step * 2
    major = (lv % coarse == 0)
    if abs(dist) <= med_range / 2:
        reach = "typical day"
    elif abs(dist) <= p90_range / 2:
        reach = "big day"
    else:
        reach = "multi-day"
    rows.append({
        "Level": lv,
        "Distance": dist,
        "Weight": "major" if major else "minor",
        "Reachable in": reach,
    })
tbl = pd.DataFrame(rows).sort_values("Level", ascending=False)

left, right = st.columns([3, 2])

with left:
    fig = go.Figure()

    # bands showing how far a typical and a big session travels
    fig.add_hrect(y0=price - p90_range / 2, y1=price + p90_range / 2,
                  fillcolor="#5f8bb5", opacity=0.07, line_width=0,
                  annotation_text="big day", annotation_position="top left",
                  annotation_font_color="#5f8bb5", annotation_font_size=11)
    fig.add_hrect(y0=price - med_range / 2, y1=price + med_range / 2,
                  fillcolor="#6f9f78", opacity=0.10, line_width=0,
                  annotation_text="typical day", annotation_position="bottom left",
                  annotation_font_color="#6f9f78", annotation_font_size=11)

    for _, r in tbl.iterrows():
        is_major = r["Weight"] == "major"
        col = MAJOR if is_major else MINOR
        fig.add_hline(
            y=r["Level"], line_color=col, line_width=2 if is_major else 1,
            line_dash="solid" if is_major else "dot",
            annotation_text=f"{r['Level']:,.0f}   {r['Distance']:+,.0f}",
            annotation_position="right", annotation_font_color=col,
            annotation_font_size=11)

    fig.add_hline(y=price, line_color=PRICE, line_width=2,
                  annotation_text=f"last {price:,.2f}",
                  annotation_position="left", annotation_font_color=PRICE)

    pad = step * (span + 0.8)
    fig.update_layout(
        template="plotly_dark", height=720,
        yaxis=dict(title="Price", range=[price - pad, price + pad]),
        xaxis=dict(visible=False, range=[0, 1]),
        margin=dict(l=10, r=130, t=20, b=10), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.dataframe(
        tbl.assign(Level=tbl["Level"].map("{:,.0f}".format),
                   Distance=tbl["Distance"].map("{:+,.0f}".format)),
        hide_index=True, use_container_width=True, height=560,
    )
    st.caption(
        "Distance is in points from the last print. 'Reachable in' compares "
        "that distance to half the recent session range, since price has to "
        "travel from somewhere in the middle of the day's range to get there."
    )

# ───────────────────── COPY-AND-PASTE PINE SCRIPT ──────────────────────────
lines = [
    "//@version=6",
    f'indicator("Psych levels {step}", "Psych", overlay = true, max_lines_count = 100)',
    "",
    f'step   = input.float({step}, "Spacing (points)", minval = 0.01)',
    f'span   = input.int({span}, "Levels each side", minval = 1, maxval = 20)',
    'majorN = input.int(4, "Every Nth level is major", minval = 2, maxval = 10)',
    f'cMaj   = input.color({MAJOR}, "Major")',
    f'cMin   = input.color({MINOR}, "Minor")',
    'ext    = input.int(18, "Extend right (bars)", minval = 0)',
    'back   = input.int(300, "Length (bars back)", minval = 10)',
    "",
    "var line[] L = array.new<line>()",
    "",
    "if barstate.islast",
    "    for ln in L",
    "        line.delete(ln)",
    "    array.clear(L)",
    "    float base = math.round(close / step) * step",
    "    for i = -span to span",
    "        float lv = base + i * step",
    "        bool maj = math.abs(lv % (step * majorN)) < syminfo.mintick",
    "        array.push(L, line.new(bar_index - back, lv, bar_index + ext, lv,",
    "             xloc.bar_index, extend.none, maj ? cMaj : cMin,",
    "             maj ? line.style_solid : line.style_dotted, maj ? 2 : 1))",
]
pine = "\n".join(lines) + "\n"

with st.expander("📋  Ready-to-paste TradingView script", expanded=False):
    st.caption(
        "This one recalculates its own levels from whatever the chart's price "
        "is, so unlike the GEX script it never goes stale — paste it once and "
        "leave it."
    )
    st.code(pine, language="javascript")
    st.download_button("Download as .pine file", pine,
                       file_name=f"psych_levels_{step}.pine", mime="text/plain")

st.caption(
    f"Last print {data['asof']:%Y-%m-%d %H:%M %Z} · roughly 15-minute delayed. "
    "Round numbers attract resting orders and get referenced in commentary, but "
    "they are a convention rather than a measured level like a gamma wall — "
    "treat them as places to watch, not signals. Informational only."
)
