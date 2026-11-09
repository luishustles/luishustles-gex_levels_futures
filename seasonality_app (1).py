"""
Day-of-week seasonality — NQ and friends.

Shows average return by weekday with honest error bars, plus the two tests
that usually kill an apparent pattern:

  1. Split-half stability — does the first half of the sample agree with
     the second? A real effect persists; a fluke does not.
  2. Multiple comparisons — testing five weekdays means the best-looking
     one is roughly five times more likely to be a coincidence, so the
     p-value needs adjusting before it means anything.

Lives in the pages/ folder so Streamlit shows it in the sidebar.
"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Seasonality", layout="wide")

SYMBOLS = {
    "Nasdaq 100 futures (NQ)": "NQ=F",
    "S&P 500 futures (ES)": "ES=F",
    "Gold futures (GC)": "GC=F",
    "Russell 2000 futures (RTY)": "RTY=F",
    "QQQ": "QQQ",
    "SPY": "SPY",
}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
POS, NEG, NEUTRAL = "#6f9f78", "#b5676b", "#8a8578"


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_sided_p(t):
    """Normal approximation — fine here, each weekday bucket has 100+ samples."""
    return 2.0 * (1.0 - norm_cdf(abs(t)))


@st.cache_data(ttl=3600, show_spinner="Pulling price history…")
def load(symbol, years):
    period = "max" if years == "max" else f"{years}y"
    df = yf.Ticker(symbol).history(period=period, interval="1d")
    if df.empty:
        return pd.DataFrame()

    out = pd.DataFrame({
        "open": df["Open"], "high": df["High"],
        "low": df["Low"], "close": df["Close"],
    }).dropna()

    out["ret"] = out["close"].pct_change() * 100.0          # percent
    out["chg_pts"] = out["close"].diff()                    # points, signed
    out["range_pts"] = out["high"] - out["low"]             # high-low, points
    out["body_pts"] = (out["close"] - out["open"]).abs()    # open-close, points
    out = out.dropna()
    out["weekday"] = out.index.day_name()
    return out[out["weekday"].isin(DAYS)]


def stats_by_day(df):
    rows = []
    for d in DAYS:
        r = df.loc[df["weekday"] == d, "ret"].values
        n = len(r)
        if n < 2:
            rows.append(dict(Day=d, n=0, mean=np.nan, median=np.nan,
                             win=np.nan, se=np.nan, t=np.nan, p=np.nan))
            continue
        mean, sd = float(np.mean(r)), float(np.std(r, ddof=1))
        se = sd / math.sqrt(n)
        t = mean / se if se > 0 else 0.0
        rows.append(dict(
            Day=d, n=n, mean=mean, median=float(np.median(r)),
            win=float((r > 0).mean() * 100.0), se=se, t=t, p=two_sided_p(t),
        ))
    return pd.DataFrame(rows)


# ──────────────────────────────── CONTROLS ─────────────────────────────────
st.title("Day-of-week seasonality")

c1, c2 = st.columns([2, 1])
name = c1.selectbox("Instrument", list(SYMBOLS.keys()))
years = c2.selectbox("History", [3, 5, 10, 15, "max"], index=1,
                     format_func=lambda y: "Everything Yahoo has" if y == "max" else f"{y} years")

df = load(SYMBOLS[name], years)
if df.empty:
    st.error("No price history returned. Yahoo may be rate-limiting — wait a minute and rerun.")
    st.stop()

s = stats_by_day(df)

st.caption(
    f"{len(df):,} sessions · {df.index[0]:%b %Y} to {df.index[-1]:%b %Y} · "
    "returns are close-to-close, in percent."
)

# ──────────────────────── MEAN RETURN WITH ERROR BARS ──────────────────────
sig_level = 0.05 / len(DAYS)     # Bonferroni: five weekdays tested at once

fig = go.Figure()
fig.add_trace(go.Bar(
    x=s["Day"], y=s["mean"],
    marker_color=[POS if m > 0 else NEG for m in s["mean"]],
    error_y=dict(type="data", array=1.96 * s["se"], color="#9a9a94", thickness=1.5),
    hovertemplate="%{x}<br>mean %{y:+.3f}%<extra></extra>",
))
fig.add_hline(y=0, line_color="#9a9a94", line_width=1)
fig.update_layout(
    template="plotly_dark", height=420,
    yaxis_title="Mean daily return (%)", xaxis_title="",
    margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Whiskers are 95% confidence intervals. **If a whisker crosses zero, that "
    "day is not distinguishable from a coin flip** — the bar's direction is "
    "noise, however tall it looks."
)

# ─────────────────────────── THE HONEST TABLE ──────────────────────────────
disp = pd.DataFrame({
    "Day": s["Day"],
    "Sessions": s["n"],
    "Mean %": s["mean"].round(3),
    "Median %": s["median"].round(3),
    "Win rate %": s["win"].round(1),
    "t-stat": s["t"].round(2),
    "p-value": s["p"].round(4),
    "Survives?": np.where(s["p"] < sig_level, "yes", "no"),
})
st.dataframe(disp, hide_index=True, use_container_width=True)
st.caption(
    f"'Survives?' applies a Bonferroni correction: because five weekdays are "
    f"tested together, the bar for significance is p < {sig_level:.3f}, not 0.05. "
    "Without that adjustment one day looks special in almost every sample."
)

# ─────────────────────────── SPLIT-HALF STABILITY ──────────────────────────
st.subheader("Does it hold up out of sample?")

mid = len(df) // 2
a, b = stats_by_day(df.iloc[:mid]), stats_by_day(df.iloc[mid:])

fig2 = go.Figure()
fig2.add_trace(go.Bar(name=f"{df.index[0]:%b %Y} – {df.index[mid]:%b %Y}",
                      x=a["Day"], y=a["mean"], marker_color="#5f8bb5"))
fig2.add_trace(go.Bar(name=f"{df.index[mid]:%b %Y} – {df.index[-1]:%b %Y}",
                      x=b["Day"], y=b["mean"], marker_color="#d99a3c"))
fig2.add_hline(y=0, line_color="#9a9a94", line_width=1)
fig2.update_layout(
    template="plotly_dark", height=380, barmode="group",
    yaxis_title="Mean daily return (%)",
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", y=1.12, x=0),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig2, use_container_width=True)

agree = int(np.sum(np.sign(a["mean"].values) == np.sign(b["mean"].values)))
if agree >= 4:
    verdict, colour = f"{agree} of 5 days agree in sign across halves — unusually stable for this kind of test.", "success"
elif agree == 3:
    verdict, colour = f"{agree} of 5 days agree in sign. That is what you would expect from chance alone.", "warning"
else:
    verdict, colour = f"Only {agree} of 5 days agree in sign. The pattern does not persist.", "error"
getattr(st, colour)(verdict)

# ──────────────────── CUMULATIVE — IS IT ONE OUTLIER? ──────────────────────
with st.expander("Cumulative return if you only held that weekday"):
    fig3 = go.Figure()
    ramp = ["#d94b4b", "#de7a3f", "#c0a24a", "#6f9a93", "#5f8bb5"]
    for d, col in zip(DAYS, ramp):
        sub = df[df["weekday"] == d]
        fig3.add_trace(go.Scatter(
            x=sub.index, y=(np.cumprod(1 + sub["ret"].values / 100.0) - 1) * 100.0,
            name=d, mode="lines", line_color=col, line_width=1.5))
    fig3.add_hline(y=0, line_color="#9a9a94", line_width=1)
    fig3.update_layout(
        template="plotly_dark", height=400, yaxis_title="Cumulative return (%)",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.12, x=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        "A line that jumps in one or two places is a couple of big sessions, "
        "not a weekday tendency. Steady slope is the only thing worth a second look."
    )

# ──────────────────── HOW MANY POINTS DOES IT MOVE? ────────────────────────
st.subheader("How far it actually moves, in points")

pts = []
for d in DAYS:
    sub = df[df["weekday"] == d]
    if sub.empty:
        continue
    pts.append({
        "Day": d,
        "Avg range": sub["range_pts"].mean(),
        "Median range": sub["range_pts"].median(),
        "Avg body": sub["body_pts"].mean(),
        "p25": sub["range_pts"].quantile(0.25),
        "p75": sub["range_pts"].quantile(0.75),
        "p90": sub["range_pts"].quantile(0.90),
        "Biggest": sub["range_pts"].max(),
    })
pts = pd.DataFrame(pts)

figp = go.Figure()
figp.add_trace(go.Bar(
    name="Median high−low", x=pts["Day"], y=pts["Median range"],
    marker_color="#5f8bb5",
    hovertemplate="%{x}<br>median range %{y:,.0f} pts<extra></extra>"))
figp.add_trace(go.Bar(
    name="Median open−close", x=pts["Day"], y=pts["Avg body"],
    marker_color="#8a8578",
    hovertemplate="%{x}<br>avg body %{y:,.0f} pts<extra></extra>"))
figp.update_layout(
    template="plotly_dark", height=400, barmode="group",
    yaxis_title="Points", margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", y=1.12, x=0),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(figp, use_container_width=True)

st.dataframe(
    pts.assign(**{c: pts[c].round(0).astype(int)
                  for c in ["Avg range", "Median range", "Avg body",
                            "p25", "p75", "p90", "Biggest"]}),
    hide_index=True, use_container_width=True,
)
st.caption(
    "Range is the session high minus low; body is open to close. p75 means "
    "three sessions in four were smaller than that. Points are in the "
    "instrument's own units — NQ and MNQ share the same point scale, so these "
    "read directly off your chart. Note that older history spans much lower "
    "index levels, so a 'max' window understates today's typical point move; "
    "the percent chart above is the fairer comparison across eras."
)

typical = df["range_pts"].tail(60).median()
st.info(
    f"Over the last 60 sessions the median high-to-low range was "
    f"**{typical:,.0f} points**. That is the number worth carrying into a "
    "trading day — a target more than one range away is a stretch for a single session."
)

st.caption(
    "Close-to-close returns, no costs, no slippage, no overnight gaps handled "
    "separately. Day-of-week effects in liquid futures are small, unstable, and "
    "decay once published. This page exists to test whether a pattern is real, "
    "not to supply one. Informational only, not trading advice."
)
