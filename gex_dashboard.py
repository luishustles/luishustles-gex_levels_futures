"""
GEX Dashboard — ETF option levels converted to futures prices.

Supports Nasdaq (QQQ→NQ/MNQ), S&P (SPY→ES/MES), Gold (GLD→GC/MGC),
Russell (IWM→RTY/M2K) and Silver (SLV→SI).

Computes: Gamma Flip, Max Pain, Call/Put Wall, Call/Put Wall 0DTE,
Delta Wall, Charm Wall, and ranked gamma + charm strikes.

Free data path: yfinance (no API key, no account, ~15-min delayed).

Run locally:   streamlit run gex_dashboard.py
Deploy free:   push to GitHub -> share.streamlit.io -> point at this file
"""

import math
import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ──────────────────────────────── CONFIG ───────────────────────────────────
CONTRACT_MULT = 100          # shares per option contract
RISK_FREE     = 0.045        # annualized

# etf      — where the options live
# index    — cash index for basis-adjusted conversion; None = pure ratio only
# fut      — front-month futures symbol
# tick     — futures tick size, for rounding
# band     — strike window around spot (metals need a wider one than indices)
ASSETS = {
    "Nasdaq 100  ·  QQQ → NQ / MNQ": dict(etf="QQQ", index="^NDX",  fut="NQ=F", label="MNQ", tick=0.25,  band=0.12),
    "S&P 500  ·  SPY → ES / MES":    dict(etf="SPY", index="^GSPC", fut="ES=F", label="MES", tick=0.25,  band=0.10),
    "Gold  ·  GLD → GC / MGC":       dict(etf="GLD", index=None,    fut="GC=F", label="MGC", tick=0.10,  band=0.15),
    "Russell 2000  ·  IWM → RTY":    dict(etf="IWM", index="^RUT",  fut="RTY=F", label="M2K", tick=0.10, band=0.14),
    "Silver  ·  SLV → SI":           dict(etf="SLV", index=None,    fut="SI=F", label="SI",  tick=0.005, band=0.20),
}

st.set_page_config(page_title="GEX → Futures", layout="wide",
                   initial_sidebar_state="collapsed")


# ───────────────────────── BLACK-SCHOLES GREEKS ────────────────────────────
_erf = np.vectorize(math.erf)


def _npdf(x):
    return np.exp(-0.5 * np.square(x)) / math.sqrt(2.0 * math.pi)


def _ncdf(x):
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def _d1d2(S, K, T, sigma, r=RISK_FREE):
    S = float(S)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    ok = (T > 0) & (sigma > 0) & (K > 0) & (S > 0)
    sT = np.sqrt(np.where(ok, T, 1.0))
    sg = np.where(ok, sigma, 1.0)
    d1 = (np.log(np.where(ok, S / K, 1.0)) + (r + 0.5 * sg ** 2) * np.where(ok, T, 1.0)) / (sg * sT)
    return d1, d1 - sg * sT, ok, sT, sg


def bs_gamma(S, K, T, sigma, r=RISK_FREE):
    d1, _, ok, sT, sg = _d1d2(S, K, T, sigma, r)
    return np.where(ok, _npdf(d1) / (float(S) * sg * sT), 0.0)


def bs_delta(S, K, T, sigma, kind, r=RISK_FREE):
    d1, _, ok, _, _ = _d1d2(S, K, T, sigma, r)
    nd1 = _ncdf(d1)
    return np.where(ok, np.where(np.asarray(kind) == "C", nd1, nd1 - 1.0), 0.0)


def bs_charm(S, K, T, sigma, r=RISK_FREE):
    """d(delta)/d(time). With no dividend, call and put charm are identical."""
    d1, d2, ok, sT, sg = _d1d2(S, K, T, sigma, r)
    Tz = np.where(ok, T, 1.0)
    c = -_npdf(d1) * (2.0 * r * Tz - d2 * sg * sT) / (2.0 * Tz * sg * sT)
    return np.where(ok, c, 0.0)


# ──────────────────────────────── DATA PULL ────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_spots(etf, index, fut):
    out = {}
    for sym, key in [(etf, "etf"), (index, "index"), (fut, "fut")]:
        if not sym:
            out[key] = np.nan
            continue
        try:
            h = yf.Ticker(sym).history(period="5d", interval="1m")
            out[key] = float(h["Close"].dropna().iloc[-1]) if not h.empty else np.nan
        except Exception:
            out[key] = np.nan
    return out


@st.cache_data(ttl=300, show_spinner=False)
def fetch_chain(ticker: str, max_dte: int):
    tk = yf.Ticker(ticker)
    try:
        exps = tk.options
    except Exception as e:
        st.error(f"Could not reach the {ticker} option chain: {e}")
        return pd.DataFrame()

    today = dt.date.today()
    frames = []
    for e in exps:
        try:
            edate = dt.datetime.strptime(e, "%Y-%m-%d").date()
        except ValueError:
            continue
        dte = (edate - today).days
        if dte < 0 or dte > max_dte:
            continue
        try:
            oc = tk.option_chain(e)
        except Exception:
            continue
        for df, kind in [(oc.calls, "C"), (oc.puts, "P")]:
            if df is None or df.empty:
                continue
            d = df[["strike", "openInterest", "impliedVolatility", "volume"]].copy()
            d["type"] = kind
            d["expiry"] = edate
            d["dte"] = dte
            d["T"] = max(dte, 0.5) / 365.0    # floor 0DTE so greeks stay finite
            frames.append(d)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    for c in ["openInterest", "volume", "impliedVolatility"]:
        out[c] = out[c].fillna(0)
    return out[out["openInterest"] > 0]


# ───────────────────────── EXPOSURE CALCULATIONS ───────────────────────────
def exposures(chain: pd.DataFrame, spot: float, use_volume: bool = False):
    """
    Per-strike dealer exposure.
      GEX   — $ per 1% move. Convention: dealers short calls (+), long puts (−).
      DEX   — $ delta notional.
      CHARM — delta decay notional per year.
    """
    oi = (chain["volume"] if use_volume else chain["openInterest"]).values
    K, T = chain["strike"].values, chain["T"].values
    IV, kind = chain["impliedVolatility"].values, chain["type"].values
    sign = np.where(kind == "C", 1.0, -1.0)

    df = pd.DataFrame({
        "strike": K,
        "gex": bs_gamma(spot, K, T, IV) * oi * CONTRACT_MULT * spot * spot * 0.01 * sign,
        "dex": bs_delta(spot, K, T, IV, kind) * oi * CONTRACT_MULT * spot,
        "charm": bs_charm(spot, K, T, IV) * oi * CONTRACT_MULT * spot * sign,
    })
    return df.groupby("strike", as_index=False).sum()


def gamma_flip(chain, spot, lo, hi, steps=240):
    grid = np.linspace(lo, hi, steps)
    K, T = chain["strike"].values, chain["T"].values
    IV, OI = chain["impliedVolatility"].values, chain["openInterest"].values
    sign = np.where(chain["type"].values == "C", 1.0, -1.0)

    totals = np.array([
        float(np.sum(bs_gamma(s, K, T, IV) * OI * CONTRACT_MULT * s * s * 0.01 * sign))
        for s in grid
    ])

    flip = np.nan
    cross = np.where(np.diff(np.sign(totals)) != 0)[0]
    if len(cross):
        i = cross[np.argmin(np.abs(grid[cross] - spot))]
        x0, x1, y0, y1 = grid[i], grid[i + 1], totals[i], totals[i + 1]
        flip = x0 - y0 * (x1 - x0) / (y1 - y0) if (y1 - y0) != 0 else x0
    return flip, grid, totals


def max_pain(chain):
    strikes = np.sort(chain["strike"].unique())
    calls, puts = chain[chain["type"] == "C"], chain[chain["type"] == "P"]
    pain = [
        np.sum(np.maximum(0.0, s - calls["strike"].values) * calls["openInterest"].values)
        + np.sum(np.maximum(0.0, puts["strike"].values - s) * puts["openInterest"].values)
        for s in strikes
    ]
    return float(strikes[int(np.argmin(pain))]) if len(strikes) else np.nan


def wall(df, col, mode):
    if df.empty or df[col].abs().sum() == 0:
        return np.nan
    if mode == "max":
        return float(df.loc[df[col].idxmax(), "strike"])
    if mode == "min":
        return float(df.loc[df[col].idxmin(), "strike"])
    return float(df.loc[df[col].abs().idxmax(), "strike"])


# ───────────────────────────── FUTURES MAPPING ─────────────────────────────
def make_converter(etf_px, index_px, fut_px, mode, tick):
    """
    Ratio + basis : strike × (index/etf) + (fut − index)   [needs a cash index]
    Pure ratio    : strike × (fut/etf)                     [works for anything]
    """
    has_index = index_px is not None and not np.isnan(index_px)
    mult = (index_px / etf_px) if (has_index and etf_px) else np.nan
    basis = (fut_px - index_px) if (has_index and not np.isnan(fut_px)) else 0.0

    if mode == "Pure ratio" or not has_index:
        m, b = (fut_px / etf_px if etf_px else np.nan), 0.0
    else:
        m, b = mult, basis

    def conv(x):
        v = np.asarray(x, dtype=float) * m + b
        return np.round(v / tick) * tick

    return conv, m, b


# ──────────────────────────────── INTERFACE ────────────────────────────────
st.title("GEX → Futures")

a1, a2 = st.columns([2, 1])
asset_name = a1.selectbox("Instrument", list(ASSETS.keys()))
A = ASSETS[asset_name]
has_index = A["index"] is not None

max_dte = a2.selectbox("Expirations", [1, 7, 30, 90], index=1,
                       format_func=lambda d: f"≤ {d} DTE")

c1, c2, c3 = st.columns(3)
basis_src = c1.selectbox("Weight by", ["Open interest", "Volume"])
conv_mode = c2.selectbox(
    "Conversion",
    ["Ratio + basis", "Pure ratio"] if has_index else ["Pure ratio"],
    help="Ratio + basis needs a cash index. Metals have none here, so they use the direct futures/ETF ratio.",
)
n_rank = c3.slider("Ranked gamma strikes", 0, 10, 10)

spots = fetch_spots(A["etf"], A["index"], A["fut"])
etf_px, index_px, fut_px = spots["etf"], spots["index"], spots["fut"]

if np.isnan(etf_px):
    st.error(f"Couldn't fetch {A['etf']}. Yahoo may be rate-limiting — wait a minute and rerun.")
    st.stop()

chain = fetch_chain(A["etf"], max_dte)
if chain.empty:
    st.warning("No chain data returned for that expiration window. Try a longer one — "
               "metals ETFs often have no weekly expirations.")
    st.stop()

lo, hi = etf_px * (1 - A["band"]), etf_px * (1 + A["band"])
chain = chain[(chain["strike"] >= lo) & (chain["strike"] <= hi)]

use_vol = basis_src == "Volume"
agg = exposures(chain, etf_px, use_vol)
near = chain[chain["expiry"] == chain["expiry"].min()]
agg0 = exposures(near, etf_px, use_vol)

flip, grid, curve = gamma_flip(chain, etf_px, lo, hi)

levels = {
    "Gamma Flip":     flip,
    "Max Pain":       max_pain(chain),
    "Call Wall":      wall(agg,  "gex",   "max"),
    "Call Wall 0DTE": wall(agg0, "gex",   "max"),
    "Put Wall":       wall(agg,  "gex",   "min"),
    "Put Wall 0DTE":  wall(agg0, "gex",   "min"),
    "Delta Wall":     wall(agg,  "dex",   "abs"),
    "Charm Wall":     wall(agg,  "charm", "abs"),
}

PALETTE = {
    "Gamma Flip": "#9b7bff", "Max Pain": "#c84bd1",
    "Call Wall": "#e23b4e", "Call Wall 0DTE": "#ff3d7f",
    "Put Wall": "#22c55e", "Put Wall 0DTE": "#7ee031",
    "Delta Wall": "#2bd7d7", "Charm Wall": "#f26b21",
}
GAMMA_C, CHARM_C = "#3fa9f5", "#f5a623"

conv, mult, basis = make_converter(etf_px, index_px, fut_px, conv_mode, A["tick"])
total_gex = float(agg["gex"].sum())
neg_regime = (not np.isnan(flip)) and etf_px < flip
pfx = "−" if neg_regime else "+"

ranked = agg.reindex(agg["gex"].abs().sort_values(ascending=False).index).head(n_rank)
ranked_levels = {f"{pfx}{i+1} Gamma": float(r.strike) for i, r in enumerate(ranked.itertuples())}
ch = agg.reindex(agg["charm"].abs().sort_values(ascending=False).index).head(3)
charm_levels = {f"{pfx}{i+1} Charm": float(r.strike) for i, r in enumerate(ch.itertuples())}

# ── header ──
cols = st.columns(5 if has_index else 4)
cols[0].metric(A["etf"], f"{etf_px:,.2f}")
i = 1
if has_index:
    cols[1].metric(A["index"].lstrip("^"), f"{index_px:,.2f}" if not np.isnan(index_px) else "—")
    i = 2
cols[i].metric(A["label"], f"{fut_px:,.2f}" if not np.isnan(fut_px) else "—")
cols[i + 1].metric("Multiplier", f"{mult:,.4f}" if not np.isnan(mult) else "—",
                   help="What a 1-point move in the ETF is worth in futures points.")
cols[i + 2].metric("Basis", f"{basis:+,.1f}" if has_index else "n/a",
                   help="Futures premium over cash. Folded into the ratio when there's no index.")

st.markdown(
    f"### {'−Γ  NEGATIVE GAMMA' if neg_regime else '+Γ  positive gamma'}"
    f"  ·  net GEX {'+' if total_gex >= 0 else '−'}${abs(total_gex)/1e9:,.2f}B per 1%"
)

left, right = st.columns([3, 2])

with right:
    st.dataframe(
        pd.DataFrame([
            {"Level": k,
             A["etf"]: round(v, 2) if not np.isnan(v) else None,
             A["label"]: float(conv(v)) if not np.isnan(v) else None}
            for k, v in levels.items()
        ]),
        hide_index=True, use_container_width=True, height=320,
    )
    st.caption(f"Paste the {A['etf']} column into the TradingView indicator "
               f"with conversion on Auto, or the {A['label']} column to hard-code them.")

    extras = {**ranked_levels, **charm_levels}
    if extras:
        st.dataframe(
            pd.DataFrame([
                {"Level": k, A["etf"]: round(v, 2), A["label"]: float(conv(v))}
                for k, v in extras.items()
            ]).sort_values(A["label"], ascending=False),
            hide_index=True, use_container_width=True, height=300,
        )

with left:
    fig = go.Figure()
    y = [float(conv(s)) for s in agg["strike"].values]
    fig.add_trace(go.Bar(
        x=agg["gex"].values / 1e9, y=y, orientation="h",
        marker_color=["#22c55e" if g > 0 else "#e23b4e" for g in agg["gex"].values],
        hovertemplate=A["label"] + " %{y:,.2f}<br>%{x:,.3f} $B<extra></extra>",
    ))

    for name, v in {**levels, **ranked_levels, **charm_levels}.items():
        if np.isnan(v):
            continue
        col = PALETTE.get(name, CHARM_C if "Charm" in name else GAMMA_C)
        p = float(conv(v))
        fig.add_hline(y=p, line_color=col, line_width=2 if name in PALETTE else 1,
                      annotation_text=f"{name}  {p:,.2f}",
                      annotation_position="right", annotation_font_color=col,
                      annotation_font_size=10)

    if not np.isnan(fut_px):
        fig.add_hline(y=fut_px, line_color="#ffffff", line_width=1, line_dash="dot",
                      annotation_text=f"{A['label']} {fut_px:,.2f}", annotation_position="left")

    fig.update_layout(
        template="plotly_dark", height=820, margin=dict(l=10, r=150, t=30, b=10),
        xaxis_title="Net gamma exposure ($B per 1% move)",
        yaxis_title=f"{A['label']} price", showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Gamma profile across spot (where the flip comes from)"):
    gf = go.Figure()
    gf.add_trace(go.Scatter(x=[float(conv(g)) for g in grid], y=curve / 1e9,
                            mode="lines", line_color=GAMMA_C))
    gf.add_hline(y=0, line_color="#888", line_width=1)
    if not np.isnan(fut_px):
        gf.add_vline(x=fut_px, line_color="#fff", line_dash="dot")
    gf.update_layout(template="plotly_dark", height=320,
                     xaxis_title=f"{A['label']} price", yaxis_title="Total GEX ($B)",
                     margin=dict(l=10, r=10, t=20, b=10),
                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(gf, use_container_width=True)

st.caption(
    "Data: Yahoo via yfinance, roughly 15-minute delayed. Greeks computed with "
    "Black-Scholes from chain IV. Dealer positioning is an assumed convention "
    "(short calls, long puts), not a measurement — vendors differ here, which is "
    "why their walls differ. Informational only, not trading advice."
)
