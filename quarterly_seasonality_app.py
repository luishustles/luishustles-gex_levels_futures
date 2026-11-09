"""
Quarterly seasonal trend — one ticker, one quarter, every year on record.

For the quarter you pick it shows:
  · the total return in that quarter for each individual year
  · how often it finished up, with a confidence interval on that rate
  · mean and median return
  · every year's price path through the quarter, overlaid and normalized
  · the same summary for all four quarters side by side

Built from the notebook approach, with three fixes:
  1. Quarters are labelled correctly (a quarter starting in April is Q2).
  2. The return is measured from the last close *before* the quarter to the
     last close *in* it, so the opening gap is included. Measuring inside the
     quarter only, as pct_change does, silently drops the first day's move.
  3. The up-rate carries a confidence interval, because 8 up years out of 14
     is not meaningfully different from a coin flip and should not be
     presented as 57%.

Run locally:  streamlit run quarterly_seasonality_app.py
Deploy free:  push to GitHub -> share.streamlit.io -> point at this file
"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Quarterly seasonality", layout="wide",
                   initial_sidebar_state="collapsed")

POS, NEG, NEUTRAL, ACCENT = "#6f9f78", "#b5676b", "#8a8578", "#5f8bb5"

QUARTERS = {
    "Q1  ·  Jan – Mar": (1, 3),
    "Q2  ·  Apr – Jun": (4, 6),
    "Q3  ·  Jul – Sep": (7, 9),
    "Q4  ·  Oct – Dec": (10, 12),
}


@st.cache_data(ttl=3600, show_spinner="Pulling price history…")
def load(ticker):
    df = yf.Ticker(ticker).history(period="max", interval="1d")
    return df.dropna(subset=["Close"]) if not df.empty else pd.DataFrame()


def wilson(successes, n, z=1.96):
    """Wilson score interval — behaves properly at small n, unlike the normal approximation."""
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z / denom * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half) * 100, min(1.0, centre + half) * 100)


def quarter_returns(data, start_m, end_m, require_full=True):
    """
    Total return for the given quarter in each year.

    Measured from the last close before the quarter opens to the last close
    inside it, so the move on the quarter's first session counts.
    """
    rows = []
    for year in range(int(data.index.year.min()), int(data.index.year.max()) + 1):
        q = data[(data.index.year == year)
                 & (data.index.month >= start_m)
                 & (data.index.month <= end_m)]
        if q.empty:
            continue

        # skip part-quarters at the very start or end of the record
        if require_full:
            months_present = q.index.month.nunique()
            if months_present < (end_m - start_m + 1):
                rows.append(dict(Year=year, ret=np.nan, partial=True,
                                 start=q.index[0], end=q.index[-1]))
                continue

        before = data[data.index < q.index[0]]
        base = float(before["Close"].iloc[-1]) if not before.empty else float(q["Close"].iloc[0])
        last = float(q["Close"].iloc[-1])
        rows.append(dict(Year=year, ret=(last / base - 1.0) * 100.0, partial=False,
                         start=q.index[0], end=q.index[-1]))
    return pd.DataFrame(rows)


def summarize(qr):
    full = qr[~qr["partial"] & qr["ret"].notna()]
    r = full["ret"].values
    n = len(r)
    if n == 0:
        return None
    up = int((r >= 0).sum())
    lo, hi = wilson(up, n)
    return {
        "n": n, "up": up, "down": n - up,
        "up_rate": up / n * 100.0, "ci": (lo, hi),
        "mean": float(np.mean(r)), "median": float(np.median(r)),
        "best": float(np.max(r)), "worst": float(np.min(r)),
        "stdev": float(np.std(r, ddof=1)) if n > 1 else np.nan,
        "data": full,
    }


# ════════════════════════════════ UI ═══════════════════════════════════════
st.title("Quarterly seasonal trend")

c1, c2, c3 = st.columns([1, 2, 1])
ticker = c1.text_input("Ticker", "NCLH").strip().upper()
qlabel = c2.selectbox("Quarter", list(QUARTERS.keys()), index=3)
min_year = c3.number_input("Start from year", 1960, 2030, 1990, step=1)

if not ticker:
    st.stop()

data = load(ticker)
if data.empty:
    st.error(f"Couldn't fetch {ticker}. Check the symbol, or wait a minute if Yahoo is rate-limiting.")
    st.stop()

data = data[data.index.year >= min_year]
if data.empty:
    st.error("No data in that year range.")
    st.stop()

start_m, end_m = QUARTERS[qlabel]
qr = quarter_returns(data, start_m, end_m)
s = summarize(qr)

if s is None:
    st.warning("No complete quarters in the available history.")
    st.stop()

qname = qlabel.split("·")[0].strip()

# ──────────────────────────────── HEADLINE ─────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric(f"{ticker} — {qname}", f"{s['n']} years",
          f"{data.index.year.min()}–{data.index.year.max()}")
m2.metric("Finished up", f"{s['up_rate']:.0f}%", f"{s['up']} of {s['n']} years")
m3.metric("Median return", f"{s['median']:+.2f}%")
m4.metric("Mean return", f"{s['mean']:+.2f}%",
          help="Mean is pulled around by single extreme years; median is the steadier read.")
m5.metric("Best / worst", f"{s['best']:+.0f}% / {s['worst']:+.0f}%")

lo, hi = s["ci"]
coin_flip = lo <= 50.0 <= hi
if coin_flip:
    st.warning(
        f"**The up-rate is {s['up_rate']:.0f}%, but the 95% confidence interval runs "
        f"{lo:.0f}% to {hi:.0f}%.** That range contains 50%, so across {s['n']} years "
        f"this is not distinguishable from a coin flip. The tendency may be real — "
        f"there simply isn't enough history here to show it."
    )
else:
    direction = "above" if lo > 50 else "below"
    st.success(
        f"**Up-rate {s['up_rate']:.0f}%, 95% interval {lo:.0f}%–{hi:.0f}%** — entirely "
        f"{direction} 50%. That is a real lean in this sample. Still worth checking "
        f"the year-by-year bars below for whether one or two years are carrying it."
    )

# ──────────────────────── YEAR-BY-YEAR BAR CHART ───────────────────────────
full = s["data"]
fig = go.Figure()
fig.add_trace(go.Bar(
    x=full["Year"], y=full["ret"],
    marker_color=[POS if v >= 0 else NEG for v in full["ret"]],
    hovertemplate="%{x}<br>%{y:+.2f}%<extra></extra>",
))
fig.add_hline(y=0, line_color=NEUTRAL, line_width=1)
fig.add_hline(y=s["median"], line_color=ACCENT, line_dash="dash", line_width=1.5,
              annotation_text=f"median {s['median']:+.1f}%",
              annotation_position="right", annotation_font_color=ACCENT)
fig.update_layout(
    template="plotly_dark", height=420,
    yaxis_title=f"{qname} total return (%)", xaxis_title="",
    margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns([3, 2])

# ───────────────────── NORMALIZED PATHS THROUGH THE QUARTER ────────────────
with left:
    st.subheader(f"Every year's path through {qname}")
    figp = go.Figure()
    for year in full["Year"]:
        q = data[(data.index.year == year)
                 & (data.index.month >= start_m)
                 & (data.index.month <= end_m)]
        if q.empty:
            continue
        norm = (q["Close"].values / q["Close"].values[0] - 1.0) * 100.0
        pct_through = np.linspace(0, 100, len(norm))
        ended_up = norm[-1] >= 0
        figp.add_trace(go.Scatter(
            x=pct_through, y=norm, mode="lines", name=str(year),
            line=dict(color=POS if ended_up else NEG, width=1),
            opacity=0.55,
            hovertemplate=f"{year}<br>%{{y:+.1f}}%<extra></extra>"))

    # average shape across years, resampled onto a common grid
    grid = np.linspace(0, 100, 60)
    stacked = []
    for year in full["Year"]:
        q = data[(data.index.year == year)
                 & (data.index.month >= start_m)
                 & (data.index.month <= end_m)]
        if len(q) < 5:
            continue
        norm = (q["Close"].values / q["Close"].values[0] - 1.0) * 100.0
        stacked.append(np.interp(grid, np.linspace(0, 100, len(norm)), norm))
    if stacked:
        figp.add_trace(go.Scatter(
            x=grid, y=np.median(np.vstack(stacked), axis=0), mode="lines",
            name="median path", line=dict(color="#e8e8e3", width=3),
            hovertemplate="median %{y:+.1f}%<extra></extra>"))

    figp.add_hline(y=0, line_color=NEUTRAL, line_width=1)
    figp.update_layout(
        template="plotly_dark", height=460,
        xaxis_title=f"Progress through {qname} (%)",
        yaxis_title="Change from quarter open (%)",
        margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figp, use_container_width=True)
    st.caption(
        "Green years finished the quarter up, red finished down; the white line "
        "is the median shape. Years are stretched onto a common 0–100% axis so "
        "they overlay despite differing session counts. A median path that drifts "
        "steadily is more interesting than one that lurches — a lurch usually "
        "means a single year dominating."
    )

# ──────────────────────────── PER-YEAR TABLE ───────────────────────────────
with right:
    st.subheader("Year by year")
    tbl = pd.DataFrame({
        "Year": full["Year"].astype(int),
        "Return %": full["ret"].round(2),
        "Direction": np.where(full["ret"] >= 0, "up", "down"),
    }).sort_values("Year", ascending=False)
    st.dataframe(tbl, hide_index=True, use_container_width=True, height=400)
    st.download_button("Download CSV", tbl.to_csv(index=False),
                       file_name=f"{ticker}_{qname}_seasonality.csv", mime="text/csv")

    partials = qr[qr["partial"]]
    if not partials.empty:
        yrs = ", ".join(str(int(y)) for y in partials["Year"])
        st.caption(f"Excluded as incomplete quarters: {yrs}.")

# ──────────────────────── ALL FOUR QUARTERS AT ONCE ────────────────────────
st.subheader("All four quarters")

rows = []
for lbl, (a, b) in QUARTERS.items():
    ss = summarize(quarter_returns(data, a, b))
    if ss is None:
        continue
    clo, chi = ss["ci"]
    rows.append({
        "Quarter": lbl.split("·")[0].strip(),
        "Months": lbl.split("·")[1].strip(),
        "Years": ss["n"],
        "Up %": round(ss["up_rate"], 0),
        "95% interval": f"{clo:.0f}–{chi:.0f}%",
        "Median %": round(ss["median"], 2),
        "Mean %": round(ss["mean"], 2),
        "Std dev %": round(ss["stdev"], 1) if np.isfinite(ss["stdev"]) else None,
        "Beats a coin flip?": "no" if clo <= 50 <= chi else "yes",
    })
allq = pd.DataFrame(rows)

cq1, cq2 = st.columns([2, 3])
with cq1:
    st.dataframe(allq, hide_index=True, use_container_width=True)
with cq2:
    figq = go.Figure()
    figq.add_trace(go.Bar(
        x=allq["Quarter"], y=allq["Median %"], name="median",
        marker_color=[POS if v >= 0 else NEG for v in allq["Median %"]],
        hovertemplate="%{x}<br>median %{y:+.2f}%<extra></extra>"))
    figq.add_hline(y=0, line_color=NEUTRAL, line_width=1)
    figq.update_layout(
        template="plotly_dark", height=280, yaxis_title="Median return (%)",
        margin=dict(l=10, r=10, t=20, b=10), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(figq, use_container_width=True)

st.caption(
    f"Data: Yahoo via yfinance, full available history from {min_year}. Returns are "
    "close-to-close including dividends where Yahoo adjusts for them, and exclude "
    "costs. Four quarters tested together means the best-looking one is roughly four "
    "times more likely to be chance, which is what the interval column is there to "
    "catch. A pattern that holds on fifteen years of one stock is a small sample by "
    "any statistical standard. Informational only, not trading advice."
)
