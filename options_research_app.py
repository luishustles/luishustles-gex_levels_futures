"""
Options research — one ticker at a time.

Type a symbol and get:
  · monthly and quarterly seasonality (median % move, win rate, sample size)
  · Altman Z-score and Piotroski F-score, with every component shown
  · the option chain around the money, with cost per contract and breakeven
  · a scenario grid: what the contract is theoretically worth at a given
    underlying price and date, with analyst targets marked
  · next earnings date, analyst target spread, recent headlines

What this is not: a forecast. The scenario grid answers "if price gets here
by then, what is this contract worth" — it says nothing about whether price
will get there. Analyst targets are reported because they are published
numbers, not because they predict well; they historically do not.

Run locally:  streamlit run options_research_app.py
Deploy free:  push to GitHub -> share.streamlit.io -> point at this file
"""

import datetime as dt
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Options research", layout="wide",
                   initial_sidebar_state="collapsed")

RISK_FREE = 0.045
POS, NEG, NEUTRAL, ACCENT = "#6f9f78", "#b5676b", "#8a8578", "#5f8bb5"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ─────────────────────────── BLACK-SCHOLES ─────────────────────────────────
def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S, K, T, sigma, kind="C", r=RISK_FREE):
    """European option value. Returns intrinsic when T or sigma collapse."""
    if S <= 0 or K <= 0:
        return 0.0
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if kind == "C" else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "C":
        return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


# ──────────────────────────── DATA HELPERS ─────────────────────────────────
@st.cache_data(ttl=900, show_spinner="Pulling history…")
def get_history(ticker, period="max"):
    df = yf.Ticker(ticker).history(period=period, interval="1mo")
    return df.dropna(subset=["Close"]) if not df.empty else pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def get_spot(ticker):
    df = yf.Ticker(ticker).history(period="5d", interval="1m")
    return float(df["Close"].dropna().iloc[-1]) if not df.empty else np.nan


@st.cache_data(ttl=900, show_spinner=False)
def get_info(ticker):
    try:
        return dict(yf.Ticker(ticker).info or {})
    except Exception:
        return {}


@st.cache_data(ttl=900, show_spinner=False)
def get_statements(ticker):
    tk = yf.Ticker(ticker)
    out = {}
    for key, attr in [("bs", "balance_sheet"), ("is", "financials"), ("cf", "cashflow")]:
        try:
            df = getattr(tk, attr)
            out[key] = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        except Exception:
            out[key] = pd.DataFrame()
    return out


@st.cache_data(ttl=900, show_spinner=False)
def get_expirations(ticker):
    try:
        return list(yf.Ticker(ticker).options or [])
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner="Pulling option chain…")
def get_chain(ticker, expiry):
    try:
        oc = yf.Ticker(ticker).option_chain(expiry)
    except Exception:
        return pd.DataFrame(), pd.DataFrame()
    return oc.calls, oc.puts


@st.cache_data(ttl=1800, show_spinner=False)
def get_news(ticker, n=8):
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []
    out = []
    for item in raw[:n]:
        # yfinance has moved this structure around between versions
        c = item.get("content", item) if isinstance(item, dict) else {}
        title = c.get("title") or item.get("title")
        if not title:
            continue
        pub = c.get("provider", {})
        publisher = pub.get("displayName") if isinstance(pub, dict) else item.get("publisher")
        link = item.get("link") or (c.get("canonicalUrl") or {}).get("url", "")
        ts = item.get("providerPublishTime") or c.get("pubDate")
        when = ""
        if isinstance(ts, (int, float)):
            when = dt.datetime.fromtimestamp(ts).strftime("%d %b")
        elif isinstance(ts, str):
            when = ts[:10]
        out.append({"title": title, "publisher": publisher or "", "link": link, "when": when})
    return out


def _row(df, *names):
    """Pull a statement line by any of several possible labels."""
    if df is None or df.empty:
        return None
    idx = {str(i).lower().strip(): i for i in df.index}
    for n in names:
        key = n.lower().strip()
        if key in idx:
            s = df.loc[idx[key]].dropna()
            if not s.empty:
                return s
    return None


def _at(series, i=0):
    if series is None or len(series) <= i:
        return np.nan
    try:
        return float(series.iloc[i])
    except Exception:
        return np.nan


# ───────────────────────────── SEASONALITY ─────────────────────────────────
def monthly_stats(hist):
    """Median and mean % change by calendar month, with win rate and sample size."""
    px = hist["Close"].copy()
    ret = px.pct_change().dropna() * 100.0
    out = []
    for i, m in enumerate(MONTHS, start=1):
        r = ret[ret.index.month == m].values
        if len(r) == 0:
            out.append(dict(Month=m, n=0, median=np.nan, mean=np.nan, win=np.nan))
            continue
        out.append(dict(Month=m, n=len(r), median=float(np.median(r)),
                        mean=float(np.mean(r)), win=float((r > 0).mean() * 100.0)))
    return pd.DataFrame(out)


def quarterly_stats(hist):
    px = hist["Close"].copy()
    q = px.resample("QE").last().pct_change().dropna() * 100.0
    out = []
    for i, label in enumerate(["Q1", "Q2", "Q3", "Q4"], start=1):
        r = q[q.index.quarter == i].values
        if len(r) == 0:
            out.append(dict(Quarter=label, n=0, median=np.nan, win=np.nan))
            continue
        out.append(dict(Quarter=label, n=len(r), median=float(np.median(r)),
                        win=float((r > 0).mean() * 100.0)))
    return pd.DataFrame(out)


# ─────────────────────────── FUNDAMENTAL SCORES ────────────────────────────
def altman_z(stmts, info):
    """
    Altman Z-score for public manufacturers:
      Z = 1.2·WC/TA + 1.4·RE/TA + 3.3·EBIT/TA + 0.6·MVE/TL + 1.0·Sales/TA
    Bands: >2.99 safe · 1.81–2.99 grey · <1.81 distress.
    """
    bs, inc = stmts["bs"], stmts["is"]
    ta = _at(_row(bs, "Total Assets"))
    ca = _at(_row(bs, "Current Assets", "Total Current Assets"))
    cl = _at(_row(bs, "Current Liabilities", "Total Current Liabilities"))
    re = _at(_row(bs, "Retained Earnings"))
    tl = _at(_row(bs, "Total Liabilities Net Minority Interest", "Total Liab",
                  "Total Liabilities"))
    ebit = _at(_row(inc, "EBIT", "Operating Income", "Ebit"))
    sales = _at(_row(inc, "Total Revenue", "Operating Revenue"))
    mve = info.get("marketCap", np.nan)

    if not np.isfinite(ta) or ta == 0:
        return None
    parts = {
        "Working capital / assets": (1.2, (ca - cl) / ta if np.isfinite(ca) and np.isfinite(cl) else np.nan),
        "Retained earnings / assets": (1.4, re / ta if np.isfinite(re) else np.nan),
        "EBIT / assets": (3.3, ebit / ta if np.isfinite(ebit) else np.nan),
        "Market cap / liabilities": (0.6, mve / tl if np.isfinite(tl) and tl and np.isfinite(mve) else np.nan),
        "Revenue / assets": (1.0, sales / ta if np.isfinite(sales) else np.nan),
    }
    rows, z = [], 0.0
    missing = 0
    for label, (w, v) in parts.items():
        ok = np.isfinite(v)
        if ok:
            z += w * v
        else:
            missing += 1
        rows.append({"Component": label, "Weight": w,
                     "Ratio": round(v, 3) if ok else None,
                     "Contribution": round(w * v, 3) if ok else None})
    return {"z": z, "parts": pd.DataFrame(rows), "missing": missing}


def piotroski_f(stmts):
    """
    Piotroski F-score, 9 binary tests across profitability, leverage and
    efficiency. Designed for cheap value stocks; less informative elsewhere.
    """
    bs, inc, cf = stmts["bs"], stmts["is"], stmts["cf"]

    ta = _row(bs, "Total Assets")
    ni = _row(inc, "Net Income", "Net Income Common Stockholders")
    ocf = _row(cf, "Operating Cash Flow", "Total Cash From Operating Activities")
    ltd = _row(bs, "Long Term Debt", "Long Term Debt And Capital Lease Obligation")
    ca = _row(bs, "Current Assets", "Total Current Assets")
    cl = _row(bs, "Current Liabilities", "Total Current Liabilities")
    sh = _row(bs, "Ordinary Shares Number", "Share Issued", "Common Stock")
    gp = _row(inc, "Gross Profit")
    rev = _row(inc, "Total Revenue", "Operating Revenue")

    ta0, ta1 = _at(ta, 0), _at(ta, 1)
    ni0, ni1 = _at(ni, 0), _at(ni, 1)
    ocf0 = _at(ocf, 0)
    roa0 = ni0 / ta0 if np.isfinite(ni0) and np.isfinite(ta0) and ta0 else np.nan
    roa1 = ni1 / ta1 if np.isfinite(ni1) and np.isfinite(ta1) and ta1 else np.nan

    cr0 = _at(ca, 0) / _at(cl, 0) if np.isfinite(_at(cl, 0)) and _at(cl, 0) else np.nan
    cr1 = _at(ca, 1) / _at(cl, 1) if np.isfinite(_at(cl, 1)) and _at(cl, 1) else np.nan
    lev0 = _at(ltd, 0) / ta0 if np.isfinite(_at(ltd, 0)) and np.isfinite(ta0) and ta0 else np.nan
    lev1 = _at(ltd, 1) / ta1 if np.isfinite(_at(ltd, 1)) and np.isfinite(ta1) and ta1 else np.nan
    gm0 = _at(gp, 0) / _at(rev, 0) if np.isfinite(_at(rev, 0)) and _at(rev, 0) else np.nan
    gm1 = _at(gp, 1) / _at(rev, 1) if np.isfinite(_at(rev, 1)) and _at(rev, 1) else np.nan
    at0 = _at(rev, 0) / ta0 if np.isfinite(_at(rev, 0)) and np.isfinite(ta0) and ta0 else np.nan
    at1 = _at(rev, 1) / ta1 if np.isfinite(_at(rev, 1)) and np.isfinite(ta1) and ta1 else np.nan

    tests = [
        ("Profitability", "Return on assets positive", roa0 > 0 if np.isfinite(roa0) else None),
        ("Profitability", "Operating cash flow positive", ocf0 > 0 if np.isfinite(ocf0) else None),
        ("Profitability", "Return on assets improving",
         roa0 > roa1 if np.isfinite(roa0) and np.isfinite(roa1) else None),
        ("Profitability", "Cash flow exceeds net income",
         ocf0 > ni0 if np.isfinite(ocf0) and np.isfinite(ni0) else None),
        ("Leverage", "Long-term debt ratio falling",
         lev0 < lev1 if np.isfinite(lev0) and np.isfinite(lev1) else None),
        ("Leverage", "Current ratio improving",
         cr0 > cr1 if np.isfinite(cr0) and np.isfinite(cr1) else None),
        ("Leverage", "No new shares issued",
         _at(sh, 0) <= _at(sh, 1) if np.isfinite(_at(sh, 0)) and np.isfinite(_at(sh, 1)) else None),
        ("Efficiency", "Gross margin improving",
         gm0 > gm1 if np.isfinite(gm0) and np.isfinite(gm1) else None),
        ("Efficiency", "Asset turnover improving",
         at0 > at1 if np.isfinite(at0) and np.isfinite(at1) else None),
    ]

    score = sum(1 for _, _, v in tests if v is True)
    unknown = sum(1 for _, _, v in tests if v is None)
    table = pd.DataFrame([
        {"Group": g, "Test": t,
         "Result": "pass" if v is True else ("fail" if v is False else "no data")}
        for g, t, v in tests
    ])
    return {"score": score, "unknown": unknown, "table": table}


# ════════════════════════════════ UI ═══════════════════════════════════════
st.title("Options research")

t1, t2 = st.columns([1, 3])
ticker = t1.text_input("Ticker", "AAPL").strip().upper()
if not ticker:
    st.stop()

spot = get_spot(ticker)
if not np.isfinite(spot):
    st.error(f"Couldn't fetch {ticker}. Check the symbol, or wait a minute if Yahoo is rate-limiting.")
    st.stop()

info = get_info(ticker)
hist = get_history(ticker)

h1, h2, h3, h4, h5 = st.columns(5)
h1.metric(ticker, f"${spot:,.2f}")
h2.metric("Market cap",
          f"${info.get('marketCap', 0)/1e9:,.1f}B" if info.get("marketCap") else "—")
h3.metric("Sector", (info.get("sector") or "—")[:18])

earn_txt, earn_days = "—", None
try:
    cal = yf.Ticker(ticker).calendar
    ed = cal.get("Earnings Date") if isinstance(cal, dict) else None
    if isinstance(ed, list) and ed:
        ed = ed[0]
    if isinstance(ed, (dt.date, dt.datetime)):
        d = ed.date() if isinstance(ed, dt.datetime) else ed
        earn_days = (d - dt.date.today()).days
        earn_txt = d.strftime("%d %b %Y")
except Exception:
    pass
h4.metric("Next earnings", earn_txt,
          f"{earn_days} days" if earn_days is not None else None)

tgt = info.get("targetMeanPrice")
h5.metric("Analyst mean target", f"${tgt:,.2f}" if tgt else "—",
          f"{(tgt/spot - 1)*100:+.1f}%" if tgt else None)

if earn_days is not None and 0 <= earn_days <= 45:
    st.warning(
        f"Earnings in {earn_days} days. Implied vol is usually elevated into the "
        "print and collapses straight after, so a long option can lose value even "
        "when the stock moves your way. Worth knowing before you pay for one."
    )

tabs = st.tabs(["Seasonality", "Fundamental scores", "Contracts & scenarios", "Analysts & news"])

# ─────────────────────────── TAB 1 · SEASONALITY ───────────────────────────
with tabs[0]:
    if hist.empty or len(hist) < 24:
        st.warning("Not enough monthly history for seasonality.")
    else:
        ms, qs = monthly_stats(hist), quarterly_stats(hist)
        yrs = len(hist) / 12.0
        st.caption(f"{hist.index[0]:%b %Y} to {hist.index[-1]:%b %Y} — about {yrs:,.0f} years.")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=ms["Month"], y=ms["median"],
            marker_color=[POS if m > 0 else NEG for m in ms["median"].fillna(0)],
            customdata=np.stack([ms["n"], ms["win"]], axis=-1),
            hovertemplate="%{x}<br>median %{y:+.2f}%<br>%{customdata[0]:.0f} years"
                          "<br>%{customdata[1]:.0f}% positive<extra></extra>",
        ))
        fig.add_hline(y=0, line_color=NEUTRAL, line_width=1)
        fig.update_layout(template="plotly_dark", height=380,
                          yaxis_title="Median monthly change (%)",
                          margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        cA, cB = st.columns(2)
        with cA:
            st.dataframe(pd.DataFrame({
                "Month": ms["Month"], "Years": ms["n"],
                "Median %": ms["median"].round(2),
                "Mean %": ms["mean"].round(2),
                "Positive %": ms["win"].round(0),
            }), hide_index=True, use_container_width=True, height=460)
        with cB:
            figq = go.Figure()
            figq.add_trace(go.Bar(
                x=qs["Quarter"], y=qs["median"],
                marker_color=[POS if m > 0 else NEG for m in qs["median"].fillna(0)],
                customdata=np.stack([qs["n"], qs["win"]], axis=-1),
                hovertemplate="%{x}<br>median %{y:+.2f}%<br>%{customdata[0]:.0f} quarters"
                              "<br>%{customdata[1]:.0f}% positive<extra></extra>"))
            figq.add_hline(y=0, line_color=NEUTRAL, line_width=1)
            figq.update_layout(template="plotly_dark", height=300,
                               yaxis_title="Median quarterly change (%)",
                               margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(figq, use_container_width=True)
            st.dataframe(pd.DataFrame({
                "Quarter": qs["Quarter"], "Quarters": qs["n"],
                "Median %": qs["median"].round(2),
                "Positive %": qs["win"].round(0),
            }), hide_index=True, use_container_width=True)

        st.info(
            f"**Read this carefully.** Each month here has only about {yrs:,.0f} "
            "observations — far too few to separate a real tendency from noise. "
            "Twelve months tested at once means the best-looking one is roughly "
            "twelve times more likely to be a coincidence. Median is shown rather "
            "than mean because one crash or squeeze distorts an average badly. "
            "Treat this as context on what the stock has done, not evidence of "
            "what it will do."
        )

# ──────────────────────── TAB 2 · FUNDAMENTAL SCORES ───────────────────────
with tabs[1]:
    stmts = get_statements(ticker)
    if all(v.empty for v in stmts.values()):
        st.warning("No financial statements available for this symbol — common for ETFs, ADRs and recent listings.")
    else:
        z = altman_z(stmts, info)
        f = piotroski_f(stmts)
        cZ, cF = st.columns(2)

        with cZ:
            st.subheader("Altman Z-score")
            if z is None:
                st.warning("Balance sheet is missing the fields this needs.")
            else:
                zv = z["z"]
                band, colour = (
                    ("Safe zone", POS) if zv > 2.99 else
                    ("Grey zone", "#c0a24a") if zv >= 1.81 else
                    ("Distress zone", NEG)
                )
                st.markdown(
                    f"<div style='font-size:2.6rem;font-weight:600;color:{colour}'>"
                    f"{zv:,.2f}</div><div style='color:{colour}'>{band}</div>",
                    unsafe_allow_html=True)
                if z["missing"]:
                    st.caption(f"{z['missing']} of 5 components unavailable — the total is understated.")
                st.dataframe(z["parts"], hide_index=True, use_container_width=True)
                st.caption(
                    "Bankruptcy-risk model built on 1960s manufacturers. It reads "
                    "poorly for banks, insurers and asset-light software companies, "
                    "where a low score often reflects the business model rather "
                    "than distress."
                )

        with cF:
            st.subheader("Piotroski F-score")
            fv = f["score"]
            band, colour = (
                ("Strong", POS) if fv >= 7 else
                ("Middling", "#c0a24a") if fv >= 4 else
                ("Weak", NEG)
            )
            st.markdown(
                f"<div style='font-size:2.6rem;font-weight:600;color:{colour}'>"
                f"{fv} / 9</div><div style='color:{colour}'>{band}</div>",
                unsafe_allow_html=True)
            if f["unknown"]:
                st.caption(f"{f['unknown']} test(s) had no data and scored zero — the total is understated.")
            st.dataframe(f["table"], hide_index=True, use_container_width=True, height=360)
            st.caption(
                "Nine accounting checks on whether the business improved year over "
                "year. Built and validated on cheap value stocks; a high score on a "
                "richly valued growth name says much less."
            )

        st.info(
            "Both scores describe the company's books, not the option. A financially "
            "healthy company can still fall, and a distressed one can rally hard. "
            "They are most useful for ruling out balance-sheet surprises on a name "
            "you plan to hold through an expiration."
        )

# ──────────────────── TAB 3 · CONTRACTS AND SCENARIOS ──────────────────────
with tabs[2]:
    exps = get_expirations(ticker)
    if not exps:
        st.warning("No listed options for this symbol.")
    else:
        e1, e2, e3 = st.columns([1, 1, 1])
        expiry = e1.selectbox("Expiration", exps)
        kind_label = e2.radio("Type", ["Call", "Put"], horizontal=True)
        kind = "C" if kind_label == "Call" else "P"
        width = e3.slider("Strikes to show each side", 3, 15, 6)

        calls, puts = get_chain(ticker, expiry)
        df = calls if kind == "C" else puts
        if df is None or df.empty:
            st.warning("Chain came back empty.")
        else:
            df = df.copy()
            df["dist"] = (df["strike"] - spot).abs()
            near = df.nsmallest(width * 2 + 1, "dist").sort_values("strike")

            exp_date = dt.datetime.strptime(expiry, "%Y-%m-%d").date()
            dte = max((exp_date - dt.date.today()).days, 0)
            T = max(dte, 0.5) / 365.0

            near["mid"] = np.where(
                (near["bid"] > 0) & (near["ask"] > 0),
                (near["bid"] + near["ask"]) / 2.0,
                near["lastPrice"])
            near["cost"] = near["mid"] * 100.0
            near["breakeven"] = (near["strike"] + near["mid"]) if kind == "C" \
                else (near["strike"] - near["mid"])
            near["be_move_%"] = (near["breakeven"] / spot - 1.0) * 100.0

            show = pd.DataFrame({
                "Strike": near["strike"].round(2),
                "Mid": near["mid"].round(2),
                "Cost / contract": near["cost"].round(0),
                "Breakeven": near["breakeven"].round(2),
                "Move needed %": near["be_move_%"].round(1),
                "IV %": (near["impliedVolatility"] * 100).round(1),
                "Open int": near["openInterest"].fillna(0).astype(int),
                "Volume": near["volume"].fillna(0).astype(int),
            })
            st.dataframe(show, hide_index=True, use_container_width=True, height=330)
            st.caption(
                f"{dte} days to expiry. Mid is the bid-ask midpoint where both sides "
                "quote, otherwise last traded. Cost assumes 100 shares per contract "
                "and excludes commission. 'Move needed' is how far the underlying has "
                "to travel just to break even at expiration."
            )

            st.subheader("Scenario grid")
            st.caption(
                "What this contract is theoretically worth at a given underlying "
                "price and date, holding implied vol constant. This is arithmetic "
                "on an assumption you supply — it is not a forecast."
            )

            s1, s2 = st.columns([1, 1])
            strike = s1.selectbox(
                "Contract", near["strike"].tolist(),
                index=int(np.argmin(np.abs(near["strike"].values - spot))),
                format_func=lambda k: f"{ticker} {expiry} {k:g} {kind_label}")
            row = near[near["strike"] == strike].iloc[0]
            iv = float(row["impliedVolatility"]) if row["impliedVolatility"] > 0 else 0.3
            paid = float(row["mid"])
            s2.metric("Cost per contract", f"${paid*100:,.0f}",
                      f"IV {iv*100:,.1f}%")

            lo_px, hi_px = spot * 0.75, spot * 1.25
            prices = np.linspace(lo_px, hi_px, 90)
            day_steps = sorted({0, max(dte // 3, 1), max(2 * dte // 3, 2), dte})

            figs = go.Figure()
            ramp = ["#5f8bb5", "#6f9a93", "#c0a24a", "#b5676b"]
            for j, d in enumerate(day_steps):
                t_left = max(dte - d, 0) / 365.0
                vals = [bs_price(p, strike, t_left, iv, kind) * 100.0 for p in prices]
                lbl = "at expiration" if d >= dte else f"in {d} days"
                figs.add_trace(go.Scatter(
                    x=prices, y=vals, mode="lines", name=lbl,
                    line_color=ramp[j % len(ramp)], line_width=2))

            figs.add_hline(y=paid * 100.0, line_color="#9a9a94", line_dash="dash",
                           annotation_text=f"paid ${paid*100:,.0f}",
                           annotation_position="right", annotation_font_size=11)
            figs.add_vline(x=spot, line_color="#e8e8e3", line_dash="dot",
                           annotation_text=f"spot {spot:,.2f}",
                           annotation_position="top", annotation_font_size=11)

            for key, col, lbl in [("targetLowPrice", NEG, "low"),
                                  ("targetMeanPrice", ACCENT, "mean"),
                                  ("targetHighPrice", POS, "high")]:
                v = info.get(key)
                if v and lo_px <= v <= hi_px:
                    figs.add_vline(x=float(v), line_color=col, line_width=1,
                                   annotation_text=f"analyst {lbl} {v:,.0f}",
                                   annotation_position="bottom",
                                   annotation_font_color=col, annotation_font_size=10)

            figs.update_layout(
                template="plotly_dark", height=460,
                xaxis_title=f"{ticker} price", yaxis_title="Contract value ($)",
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(orientation="h", y=1.12, x=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(figs, use_container_width=True)

            scen = []
            for key, lbl in [("targetLowPrice", "Analyst low"),
                             ("targetMeanPrice", "Analyst mean"),
                             ("targetHighPrice", "Analyst high")]:
                v = info.get(key)
                if not v:
                    continue
                v = float(v)
                now_val = bs_price(v, strike, T, iv, kind) * 100.0
                exp_val = bs_price(v, strike, 0.0, iv, kind) * 100.0
                scen.append({
                    "Scenario": lbl,
                    f"{ticker} price": round(v, 2),
                    "Move %": round((v / spot - 1) * 100, 1),
                    "Value if it happens today": round(now_val, 0),
                    "Value at expiration": round(exp_val, 0),
                    "P/L at expiration": round(exp_val - paid * 100.0, 0),
                })
            if scen:
                st.dataframe(pd.DataFrame(scen), hide_index=True, use_container_width=True)
                st.caption(
                    "Analyst targets are typically 12-month views, so pairing them "
                    "with a short-dated contract asks the move to arrive far sooner "
                    "than the target implies. Their hit rate is poor — treat these "
                    "rows as 'what would this be worth at that price', nothing more."
                )
            else:
                st.caption("No analyst targets published for this symbol.")

# ───────────────────────── TAB 4 · ANALYSTS AND NEWS ───────────────────────
with tabs[3]:
    cL, cR = st.columns([1, 1])

    with cL:
        st.subheader("Analyst targets")
        lo, mean, hi = (info.get("targetLowPrice"), info.get("targetMeanPrice"),
                        info.get("targetHighPrice"))
        n_an = info.get("numberOfAnalystOpinions")
        if mean:
            figt = go.Figure()
            figt.add_trace(go.Scatter(
                x=[lo or mean, hi or mean], y=[0, 0], mode="lines",
                line=dict(color=NEUTRAL, width=6), hoverinfo="skip"))
            for v, col, lbl in [(lo, NEG, "low"), (mean, ACCENT, "mean"), (hi, POS, "high")]:
                if v:
                    figt.add_trace(go.Scatter(
                        x=[v], y=[0], mode="markers+text", marker=dict(size=14, color=col),
                        text=[f"{lbl}<br>{v:,.0f}"], textposition="top center",
                        textfont=dict(color=col, size=11), hoverinfo="skip"))
            figt.add_trace(go.Scatter(
                x=[spot], y=[0], mode="markers+text", marker=dict(size=14, color="#e8e8e3",
                                                                  symbol="diamond"),
                text=[f"now<br>{spot:,.0f}"], textposition="bottom center",
                textfont=dict(color="#e8e8e3", size=11), hoverinfo="skip"))
            figt.update_layout(
                template="plotly_dark", height=200, showlegend=False,
                yaxis=dict(visible=False, range=[-1, 1]),
                xaxis_title=f"{ticker} price",
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(figt, use_container_width=True)
            st.caption(
                f"{n_an or 'an unknown number of'} analysts · consensus "
                f"'{info.get('recommendationKey', 'n/a')}'. Published targets cluster "
                "near the current price and are revised after moves rather than "
                "before them. The spread between low and high is the more useful "
                "signal — a wide one means genuine disagreement."
            )
        else:
            st.caption("No analyst coverage published for this symbol.")

    with cR:
        st.subheader("Recent headlines")
        news = get_news(ticker)
        if not news:
            st.caption("No headlines returned.")
        for n in news:
            meta = " · ".join(x for x in [n["publisher"], n["when"]] if x)
            if n["link"]:
                st.markdown(f"**[{n['title']}]({n['link']})**  \n{meta}")
            else:
                st.markdown(f"**{n['title']}**  \n{meta}")
            st.divider()
        st.caption(
            "Headlines are listed as context. I am not scoring their sentiment or "
            "converting them into a price estimate — doing that reliably from a "
            "headline feed is not something this can do honestly."
        )

st.caption(
    "Data: Yahoo via yfinance, roughly 15-minute delayed; fundamentals are the "
    "last reported annual filings and are sometimes incomplete. Option values "
    "use Black-Scholes with the chain's implied vol held constant, which ignores "
    "vol crush around earnings and any skew change. Nothing here is a "
    "recommendation or a forecast. Informational only."
)
