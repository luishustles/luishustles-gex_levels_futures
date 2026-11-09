"""
Shared GEX engine — data fetch, greeks, level detection, futures conversion,
and Pine-script generation.

Imported by:
  gex_dashboard.py   (interactive Streamlit app)
  generate_pine.py   (scheduled GitHub Action)

Free data path: yfinance (no API key, no account, ~15-min delayed).
"""

import math
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf

CONTRACT_MULT = 100          # shares per option contract
RISK_FREE = 0.045            # annualized

# etf   — where the options live
# index — cash index for basis-adjusted conversion; None = pure ratio only
# fut   — front-month futures symbol
# tick  — futures tick size, for rounding
# band  — strike window around spot (metals need a wider one than indices)
ASSETS = {
    "Nasdaq 100  ·  QQQ → NQ / MNQ": dict(
        etf="QQQ", index="^NDX", fut="NQ=F", label="MNQ",
        tick=0.25, band=0.12, tv_etf="NASDAQ:QQQ", tv_idx="NASDAQ:NDX"),
    "S&P 500  ·  SPY → ES / MES": dict(
        etf="SPY", index="^GSPC", fut="ES=F", label="MES",
        tick=0.25, band=0.10, tv_etf="AMEX:SPY", tv_idx="SP:SPX"),
    "Gold  ·  GLD → GC / MGC": dict(
        etf="GLD", index=None, fut="GC=F", label="MGC",
        tick=0.10, band=0.15, tv_etf="AMEX:GLD", tv_idx=""),
    "Russell 2000  ·  IWM → RTY": dict(
        etf="IWM", index="^RUT", fut="RTY=F", label="M2K",
        tick=0.10, band=0.14, tv_etf="AMEX:IWM", tv_idx="TVC:RUT"),
    "Silver  ·  SLV → SI": dict(
        etf="SLV", index=None, fut="SI=F", label="SI",
        tick=0.005, band=0.20, tv_etf="AMEX:SLV", tv_idx=""),
}

# muted palette, matched to the TradingView indicator
COLORS = {
    "Gamma Flip": "#8b86b8",
    "Gamma Flip 0DTE": "#7d78a8",
    "Max Pain": "#a87095",
    "Call Wall": "#b5676b",
    "Call Wall 0DTE": "#c98a86",
    "Put Wall": "#6f9f78",
    "Put Wall 0DTE": "#8fae75",
    "Delta Wall": "#6f9fa3",
    "Charm Wall": "#b58a5e",
    "Vanna Wall": "#9b8fc4",
}
# strength ramp for the ranked gamma slots: strongest first
GAMMA_RAMP = ["#d94b4b", "#de7a3f", "#d99a3c", "#c0a24a", "#94a365",
              "#6f9a93", "#5f8bb5", "#5878c4", "#6a74b5", "#7a7aa8"]
CHARM_RAMP = ["#b58a5e", "#9e8a6e", "#8a8578"]
VANNA_RAMP = ["#9b8fc4", "#8d85b0", "#807b9c"]


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
    d1 = (np.log(np.where(ok, S / K, 1.0))
          + (r + 0.5 * sg ** 2) * np.where(ok, T, 1.0)) / (sg * sT)
    return d1, d1 - sg * sT, ok, sT, sg


def bs_gamma(S, K, T, sigma, r=RISK_FREE):
    """d²V/dS² — identical for calls and puts."""
    d1, _, ok, sT, sg = _d1d2(S, K, T, sigma, r)
    return np.where(ok, _npdf(d1) / (float(S) * sg * sT), 0.0)


def bs_delta(S, K, T, sigma, kind, r=RISK_FREE):
    """dV/dS."""
    d1, _, ok, _, _ = _d1d2(S, K, T, sigma, r)
    nd1 = _ncdf(d1)
    return np.where(ok, np.where(np.asarray(kind) == "C", nd1, nd1 - 1.0), 0.0)


def bs_charm(S, K, T, sigma, r=RISK_FREE):
    """d(delta)/d(time) — how fast delta bleeds toward expiry."""
    d1, d2, ok, sT, sg = _d1d2(S, K, T, sigma, r)
    Tz = np.where(ok, T, 1.0)
    c = -_npdf(d1) * (2.0 * r * Tz - d2 * sg * sT) / (2.0 * Tz * sg * sT)
    return np.where(ok, c, 0.0)


def bs_vanna(S, K, T, sigma, r=RISK_FREE):
    """
    d(delta)/d(vol), equivalently d(vega)/d(spot).
    Where dealer delta shifts as implied vol moves — same for calls and puts.
    """
    d1, d2, ok, _, sg = _d1d2(S, K, T, sigma, r)
    v = -_npdf(d1) * d2 / sg
    return np.where(ok, v, 0.0)


# ──────────────────────────────── DATA PULL ────────────────────────────────
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


def fetch_chain(ticker: str, max_dte: int):
    tk = yf.Ticker(ticker)
    try:
        exps = tk.options
    except Exception:
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
            d["T"] = max(dte, 0.5) / 365.0   # floor 0DTE so greeks stay finite
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
      gex   — $ per 1% spot move
      dex   — $ delta notional
      charm — delta decay notional per year
      vanna — delta shift per vol point

    Convention throughout: dealers short calls (+), long puts (−). That is an
    assumption, not a measurement — it's where vendors' numbers diverge.
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
        "vanna": bs_vanna(spot, K, T, IV) * oi * CONTRACT_MULT * spot * 0.01 * sign,
    })
    return df.groupby("strike", as_index=False).sum()


def gamma_flip(chain, spot, lo, hi, steps=240):
    """
    Re-price total gamma exposure across a grid of hypothetical spots.
    The zero-crossing nearest spot is the flip level.
    """
    if chain.empty:
        return np.nan, np.array([]), np.array([])

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
    if chain.empty:
        return np.nan
    strikes = np.sort(chain["strike"].unique())
    calls, puts = chain[chain["type"] == "C"], chain[chain["type"] == "P"]
    pain = [
        np.sum(np.maximum(0.0, s - calls["strike"].values) * calls["openInterest"].values)
        + np.sum(np.maximum(0.0, puts["strike"].values - s) * puts["openInterest"].values)
        for s in strikes
    ]
    return float(strikes[int(np.argmin(pain))]) if len(strikes) else np.nan


def wall(df, col, mode):
    """mode: 'max' | 'min' | 'abs'"""
    if df.empty or df[col].abs().sum() == 0:
        return np.nan
    if mode == "max":
        return float(df.loc[df[col].idxmax(), "strike"])
    if mode == "min":
        return float(df.loc[df[col].idxmin(), "strike"])
    return float(df.loc[df[col].abs().idxmax(), "strike"])


def make_converter(etf_px, index_px, fut_px, mode, tick):
    """
    Ratio + basis : strike × (index/etf) + (fut − index)   [needs a cash index]
    Pure ratio    : strike × (fut/etf)                     [works for anything]
    Returns (conv, multiplier, basis).
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


# ───────────────────────── ONE-CALL ANALYSIS ───────────────────────────────
def analyze(asset_name, max_dte=7, use_volume=False, conv_mode="Ratio + basis",
            n_gamma=10, n_charm=3, n_vanna=3):
    """
    Everything in one call. Returns a dict, or {"error": "..."} on failure.
    """
    A = ASSETS[asset_name]
    spots = fetch_spots(A["etf"], A["index"], A["fut"])
    etf_px, index_px, fut_px = spots["etf"], spots["index"], spots["fut"]

    if np.isnan(etf_px):
        return {"error": f"Couldn't fetch {A['etf']} — Yahoo may be rate-limiting."}

    chain = fetch_chain(A["etf"], max_dte)
    if chain.empty:
        return {"error": "No chain data for that expiration window. "
                         "Try a longer one — metals ETFs rarely have weeklies."}

    lo, hi = etf_px * (1 - A["band"]), etf_px * (1 + A["band"])
    chain = chain[(chain["strike"] >= lo) & (chain["strike"] <= hi)]
    if chain.empty:
        return {"error": "No strikes inside the price band."}

    agg = exposures(chain, etf_px, use_volume)
    near = chain[chain["expiry"] == chain["expiry"].min()]
    agg0 = exposures(near, etf_px, use_volume)

    flip, grid, curve = gamma_flip(chain, etf_px, lo, hi)
    flip0, _, _ = gamma_flip(near, etf_px, lo, hi)

    levels = {
        "Gamma Flip": flip,
        "Gamma Flip 0DTE": flip0,
        "Max Pain": max_pain(chain),
        "Call Wall": wall(agg, "gex", "max"),
        "Call Wall 0DTE": wall(agg0, "gex", "max"),
        "Put Wall": wall(agg, "gex", "min"),
        "Put Wall 0DTE": wall(agg0, "gex", "min"),
        "Delta Wall": wall(agg, "dex", "abs"),
        "Charm Wall": wall(agg, "charm", "abs"),
        "Vanna Wall": wall(agg, "vanna", "abs"),
    }

    conv, mult, basis = make_converter(etf_px, index_px, fut_px, conv_mode, A["tick"])
    total_gex = float(agg["gex"].sum())
    neg = (not np.isnan(flip)) and etf_px < flip
    pfx = "-" if neg else "+"

    def _ranked(col, n):
        if n <= 0 or agg.empty:
            return {}
        r = agg.reindex(agg[col].abs().sort_values(ascending=False).index).head(n)
        return {i + 1: float(row.strike) for i, row in enumerate(r.itertuples())}

    return {
        "asset": A,
        "asset_name": asset_name,
        "etf_px": etf_px,
        "index_px": index_px,
        "fut_px": fut_px,
        "chain": chain,
        "agg": agg,
        "agg0": agg0,
        "levels": levels,
        "ranked_gamma": _ranked("gex", n_gamma),
        "ranked_charm": _ranked("charm", n_charm),
        "ranked_vanna": _ranked("vanna", n_vanna),
        "conv": conv,
        "mult": mult,
        "basis": basis,
        "conv_mode": conv_mode,
        "total_gex": total_gex,
        "negative_gamma": neg,
        "prefix": pfx,
        "grid": grid,
        "curve": curve,
        "max_dte": max_dte,
        "generated": dt.datetime.now(dt.timezone.utc),
    }


# ──────────────────────── PINE SCRIPT GENERATION ───────────────────────────
_PINE_HEAD = '''//@version=6
// ═══════════════════════════════════════════════════════════════════════════
//  GEX SNAPSHOT — {asset_name}
//  Generated {stamp}
//  {etf} {etf_px:,.2f}   {label} {fut_px:,.2f}   {dte_txt}
//
//  Levels below are quoted in {etf} and converted live to futures prices.
//  Regenerate from the dashboard whenever you want fresh numbers.
// ═══════════════════════════════════════════════════════════════════════════
indicator("GEX Snapshot {label}", "GEX snap", overlay = true,
     max_lines_count = 500, max_labels_count = 500, max_boxes_count = 500)

etfSym    = input.symbol("{tv_etf}", "ETF symbol")
idxSym    = input.symbol("{tv_idx_safe}", "Cash index symbol")
useBasis  = input.bool({use_basis}, "Add futures basis",
             tooltip = "On: strike × (index/ETF) + (futures − index). Off: strike × (futures/ETF).")
holdBasis = input.bool(true, "Freeze basis while cash market is closed")

zoneOn    = input.bool(true,  "Draw zones", group = "Style")
zonePct   = input.float({zone_pct}, "Zone height (% of price)", step = 0.01, minval = 0.001, group = "Style")
fillTrans = input.int({fill_trans}, "Zone fill transparency", minval = 0, maxval = 100, group = "Style")
edgeTrans = input.int(45, "Border transparency", minval = 0, maxval = 100, group = "Style")
showMid   = input.bool(true, "Draw midline", group = "Style")
midW      = input.int(1, "Midline width", minval = 1, maxval = 4, group = "Style")
lookback  = input.int(300, "Length (bars back)", minval = 10, maxval = 4000, group = "Style")
extRight  = input.int(18, "Extend right (bars)", minval = 0, maxval = 400, group = "Style")
showPrice = input.bool(true, "Show price in label", group = "Style")
showPanel = input.bool(true, "Show panel", group = "Style")

etfC = request.security(etfSym, timeframe.period, close, barmerge.gaps_off, barmerge.lookahead_off)
idxC = request.security(idxSym, timeframe.period, close, barmerge.gaps_off, barmerge.lookahead_off)

var float rawMult  = na
var float rawBasis = na

if not na(idxC) and not na(etfC) and etfC > 0
    bool ticking = na(rawMult) or idxC != nz(idxC[1], idxC)
    if ticking or not holdBasis
        rawMult  := idxC / etfC
        rawBasis := close - idxC

float mult = (useBasis and not na(rawMult)) ? rawMult : (etfC > 0 ? close / etfC : na)
float off  = (useBasis and not na(rawBasis)) ? rawBasis : 0.0

conv(float lvl) =>
    math.round_to_mintick(lvl * mult + off)

half() =>
    close * zonePct / 200.0

var line[]  LNS = array.new<line>()
var label[] LBS = array.new<label>()
var box[]   BXS = array.new<box>()

draw(float raw, string txt, color col, float zMul) =>
    if raw > 0 and not na(mult)
        float p = conv(raw)
        if not na(p)
            int L = bar_index - lookback
            int R = bar_index + extRight
            float h = half() * zMul
            if zoneOn and h > 0
                array.push(BXS, box.new(L, p + h, R, p - h, xloc = xloc.bar_index,
                     border_color = color.new(col, edgeTrans), border_width = 1,
                     bgcolor = color.new(col, fillTrans)))
                if showMid
                    array.push(LNS, line.new(L, p, R, p, xloc.bar_index, extend.none, col, line.style_dotted, midW))
            else
                array.push(LNS, line.new(L, p, R, p, xloc.bar_index, extend.none, col, line.style_solid, 2))
            string t = showPrice ? txt + "  " + str.tostring(p, format.mintick) : txt
            array.push(LBS, label.new(R, p, t, xloc.bar_index, yloc.price,
                 color.new(color.black, 100), label.style_label_left, col, size.small))

if barstate.islast
    for ln in LNS
        line.delete(ln)
    for lb in LBS
        label.delete(lb)
    for bx in BXS
        box.delete(bx)
    array.clear(LNS)
    array.clear(LBS)
    array.clear(BXS)

'''

_PINE_TAIL = '''
float flipPx = {flip_literal}
float flipConv = (flipPx > 0 and not na(mult)) ? conv(flipPx) : na
bool  negGamma = not na(flipConv) and close < flipConv

var table pnl = table.new(position.top_left, 2, 5, border_width = 1,
     frame_width = 1, frame_color = color.new(color.gray, 50))

if showPanel and barstate.islast
    color bg  = color.new(color.black, 20)
    color txc = color.new(color.white, 0)
    table.cell(pnl, 0, 0, "GEX snapshot", bgcolor = bg, text_color = txc, text_size = size.small)
    table.cell(pnl, 1, 0, "{stamp_short}", bgcolor = bg, text_color = color.new(color.gray, 0), text_size = size.small)
    table.cell(pnl, 0, 1, "Multiplier", bgcolor = bg, text_color = txc, text_size = size.small)
    table.cell(pnl, 1, 1, na(mult) ? "—" : str.tostring(mult, "#.####"), bgcolor = bg, text_color = txc, text_size = size.small)
    table.cell(pnl, 0, 2, "Basis", bgcolor = bg, text_color = txc, text_size = size.small)
    table.cell(pnl, 1, 2, str.tostring(off, "#.##"), bgcolor = bg, text_color = txc, text_size = size.small)
    table.cell(pnl, 0, 3, "Gamma Flip", bgcolor = bg, text_color = txc, text_size = size.small)
    table.cell(pnl, 1, 3, na(flipConv) ? "—" : str.tostring(flipConv, format.mintick), bgcolor = bg, text_color = txc, text_size = size.small)
    table.cell(pnl, 0, 4, "Regime", bgcolor = bg, text_color = txc, text_size = size.small)
    table.cell(pnl, 1, 4, na(flipConv) ? "—" : negGamma ? "−Γ negative gamma" : "+Γ positive gamma",
         bgcolor = bg, text_color = na(flipConv) ? txc : negGamma ? color.new(#ff4d4d, 0) : color.new(#22c55e, 0), text_size = size.small)

alertcondition(not na(flipConv) and ta.crossunder(close, flipConv), "Crossed BELOW Gamma Flip", "Price crossed below Gamma Flip")
alertcondition(not na(flipConv) and ta.crossover(close,  flipConv), "Crossed ABOVE Gamma Flip", "Price crossed above Gamma Flip")
'''

# wider zones on the levels that tend to matter most
_ZONE_MULT = {
    "Call Wall": 1.5,
    "Put Wall": 1.5,
    "Gamma Flip": 1.25,
}


def build_pine(res: dict) -> str:
    """Turn an analyze() result into a ready-to-paste Pine v6 script."""
    A = res["asset"]
    has_index = bool(A["tv_idx"])
    use_basis = "true" if (has_index and res["conv_mode"] == "Ratio + basis") else "false"
    stamp = res["generated"].strftime("%Y-%m-%d %H:%M UTC")

    head = _PINE_HEAD.format(
        asset_name=res["asset_name"],
        stamp=stamp,
        etf=A["etf"],
        etf_px=res["etf_px"],
        label=A["label"],
        fut_px=res["fut_px"] if not np.isnan(res["fut_px"]) else 0.0,
        dte_txt=f"≤{res['max_dte']} DTE",
        tv_etf=A["tv_etf"],
        tv_idx_safe=A["tv_idx"] or A["tv_etf"],
        use_basis=use_basis,
        zone_pct=0.08,
        fill_trans=88,
    )

    rows = []
    for name, v in res["levels"].items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        z = _ZONE_MULT.get(name, 1.0)
        rows.append(f'    draw({v:.2f}, "{name}", {COLORS[name]}, {z})')

    pfx = res["prefix"]
    for i, v in res["ranked_gamma"].items():
        col = GAMMA_RAMP[min(i - 1, len(GAMMA_RAMP) - 1)]
        rows.append(f'    draw({v:.2f}, "{pfx}{i} Gamma", {col}, 0.5)')
    for i, v in res["ranked_charm"].items():
        col = CHARM_RAMP[min(i - 1, len(CHARM_RAMP) - 1)]
        rows.append(f'    draw({v:.2f}, "{pfx}{i} Charm", {col}, 0.5)')
    for i, v in res["ranked_vanna"].items():
        col = VANNA_RAMP[min(i - 1, len(VANNA_RAMP) - 1)]
        rows.append(f'    draw({v:.2f}, "{pfx}{i} Vanna", {col}, 0.5)')

    flip = res["levels"].get("Gamma Flip")
    flip_lit = f"{flip:.2f}" if (flip is not None and not np.isnan(flip)) else "0.0"

    tail = _PINE_TAIL.format(
        flip_literal=flip_lit,
        stamp_short=res["generated"].strftime("%m-%d %H:%M"),
    )

    return head + "\n".join(rows) + "\n" + tail


def levels_table(res: dict) -> pd.DataFrame:
    """Flat table of every level, in ETF and futures terms."""
    A = res["asset"]
    conv = res["conv"]
    rows = []

    for name, v in res["levels"].items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        rows.append({"Level": name, A["etf"]: round(v, 2), A["label"]: float(conv(v))})

    pfx = res["prefix"]
    for label, src in [("Gamma", res["ranked_gamma"]),
                       ("Charm", res["ranked_charm"]),
                       ("Vanna", res["ranked_vanna"])]:
        for i, v in src.items():
            rows.append({"Level": f"{pfx}{i} {label}",
                         A["etf"]: round(v, 2),
                         A["label"]: float(conv(v))})

    return pd.DataFrame(rows)
