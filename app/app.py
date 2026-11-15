"""Charted: crash-risk early warning for beginner investors (Streamlit app).

    streamlit run app/app.py

Reads only small precomputed files (data/app/, results/). It never trains a model,
downloads prices or rebuilds the big parquet files.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import ROOT  # noqa: E402

APP_DIR = ROOT / "data" / "app"
RES = ROOT / "results"
DISCLAIMER = "This project is a prototype and is not financial advice."

# ---- palette: black, blood red, glowing yellow and warm white ----------------------------
BG, GRID, LINE = "#000000", "#161616", "#1f1f1f"
WHITE, TEXT, MUTED = "#f4f1ea", "#cfc9bd", "#8c867a"
YELLOW, RED, RED_TXT = "#ffcf40", "#b5121b", "#e5383b"
C_V3, C_VOL, C_MA, C_V1, C_NONE = YELLOW, "#c8102e", "#e9e4d8", "#9c8a4a", "#5f5a50"
METHOD_STYLE = {  # name in results files -> (label, colour, dash)
    "Model v3 (rule + corrections)": ("Model v3 (ours)", C_V3, "solid"),
    "Volatility rule": ("Volatility rule", C_VOL, "solid"),
    "200-day average rule": ("200-day average rule", C_MA, "solid"),
    "Model v1": ("Model v1 (first try)", C_V1, "dash"),
}
LEVEL = {"red": ("Warning", RED_TXT), "amber": ("Watch", YELLOW), "green": ("Stable", WHITE)}

st.set_page_config(page_title="Charted · crash-risk early warning", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(f"""
<style>
header[data-testid="stHeader"], section[data-testid="stSidebar"],
div[data-testid="stSidebarCollapsedControl"] {{display: none;}}
.block-container {{padding-top: 0.6rem; max-width: 1200px;}}
iframe {{display: block;}}
h1 {{font-size: 1.9rem !important; letter-spacing: -0.015em;}}
h4 {{font-size: 1.05rem !important; font-weight: 600 !important; margin-top: 1.6rem;
     color: {WHITE};}}
p, li {{color: {TEXT};}}
.num {{font-family: 'IBM Plex Mono', monospace; font-feature-settings: "tnum";}}
.muted {{color: {MUTED}; font-size: 0.88rem;}}
.glow-y {{color: {YELLOW}; text-shadow: 0 0 18px rgba(255,207,64,0.45);}}
.glow-w {{color: {WHITE}; text-shadow: 0 0 16px rgba(244,241,234,0.25);}}
.glow-r {{color: {RED_TXT}; text-shadow: 0 0 18px rgba(200,16,46,0.55);}}
.up {{color: {YELLOW};}} .down {{color: {RED_TXT};}}
.hr {{border-top: 1px solid {LINE}; margin: 0.9rem 0;}}

/* page tabs and range pickers: horizontal radios drawn as underlined text */
div[role="radiogroup"] {{gap: 0.2rem 1.5rem;}}
div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {{display: none;}}
div[role="radiogroup"] label[data-baseweb="radio"] {{padding: 0.3rem 0 0.35rem 0; margin: 0;
    border-bottom: 2px solid transparent; transition: all .15s ease;}}
div[role="radiogroup"] label[data-baseweb="radio"] p {{color: {MUTED}; font-size: 0.95rem;
    font-weight: 500;}}
div[role="radiogroup"] label[data-baseweb="radio"]:hover p {{color: {WHITE};}}
div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {{border-bottom-color: {YELLOW};}}
div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) p {{color: {YELLOW};
    text-shadow: 0 0 12px rgba(255,207,64,0.5);}}

/* search box: a soft pill */
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{background: #0b0b0b;
    border-radius: 999px; border-color: #262626; padding-left: 0.4rem;}}
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {{
    border-color: {YELLOW}; box-shadow: 0 0 0 1px rgba(255,207,64,0.35), 0 0 18px rgba(255,207,64,0.15);}}

/* watchlist chips: quiet, not blue */
span[data-baseweb="tag"] {{background: #141414 !important; border: 1px solid #2a2a2a;
    border-radius: 999px !important;}}
span[data-baseweb="tag"] span {{color: {WHITE} !important;}}
div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {{background: #070707;
    border-color: #1c1c1c;}}

/* buttons: outlined, glowing on hover */
button[kind="primary"], button[kind="secondary"] {{background: transparent !important;
    border: 1px solid #2c2c2c !important; border-radius: 999px !important;}}
button[kind="primary"] {{border-color: {YELLOW} !important;}}
button[kind="primary"] p {{color: {YELLOW} !important; font-weight: 600;}}
button[kind="primary"]:hover {{box-shadow: 0 0 18px rgba(255,207,64,0.35);}}
button[kind="secondary"] p {{color: {TEXT} !important;}}
button[kind="secondary"]:hover {{border-color: {RED} !important;}}

/* list rows */
[class*="st-key-row_"] {{border-bottom: 1px solid {LINE}; padding: 0.15rem 0;}}
[class*="st-key-row_"]:hover {{background: #070707;}}
[class*="st-key-row_"] button {{font-weight: 600; font-size: 1rem; padding: 0;}}
[class*="st-key-row_"] button p {{color: {WHITE}; font-weight: 600;}}
[class*="st-key-row_"] button:hover p {{color: {YELLOW};}}
@media (max-width: 640px) {{
  [class*="st-key-row_"] [data-testid="stHorizontalBlock"],
  [class*="st-key-hdr_"] [data-testid="stHorizontalBlock"] {{flex-wrap: nowrap !important;
      gap: 0.5rem !important;}}
  [class*="st-key-row_"] [data-testid="stColumn"],
  [class*="st-key-hdr_"] [data-testid="stColumn"] {{min-width: 0 !important; width: auto !important;
      flex: 1 1 0 !important;}}
  [class*="st-key-row_w_"] [data-testid="stColumn"]:nth-child(3),
  [class*="st-key-row_w_"] [data-testid="stColumn"]:nth-child(6),
  .st-key-hdr_w [data-testid="stColumn"]:nth-child(3),
  .st-key-hdr_w [data-testid="stColumn"]:nth-child(6) {{display: none !important;}}
  .cell, .cell-sub, .colhead {{font-size: 0.75rem !important;}}
  .pill {{font-size: 0.75rem;}}
}}
.cell {{font-size: 0.95rem; color: {WHITE}; padding-top: 0.35rem;}}
.cell-sub {{font-size: 0.8rem; color: {MUTED};}}
.colhead {{font-size: 0.75rem; color: {MUTED}; text-transform: uppercase; letter-spacing: 0.06em;
          border-bottom: 1px solid {LINE}; padding-bottom: 0.35rem;}}

/* status pill: dot + word (never colour alone) */
.pill {{display: inline-flex; align-items: center; gap: 0.45rem; font-size: 0.85rem;
        font-weight: 500;}}
.pill .dot {{width: 8px; height: 8px; border-radius: 50%; display: inline-block;}}

/* stat rows */
.stats {{display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); column-gap: 2.5rem;}}
.stat {{display: flex; justify-content: space-between; padding: 0.5rem 0;
        border-bottom: 1px solid {LINE}; font-size: 0.92rem;}}
.stat span:first-child {{color: {MUTED};}}
.stat span:last-child {{font-family: 'IBM Plex Mono', monospace; color: {WHITE};}}
@media (max-width: 760px) {{.stats {{grid-template-columns: 1fr;}}}}

.bigprice {{font-family: 'IBM Plex Mono', monospace; font-size: 2.6rem; font-weight: 600;
            letter-spacing: -0.02em; line-height: 1.05;}}
.bignum {{font-family: 'IBM Plex Mono', monospace; font-size: 3rem; font-weight: 600;
          letter-spacing: -0.03em; line-height: 1;}}
.kpi .v {{font-family: 'IBM Plex Mono', monospace; font-size: 1.9rem; font-weight: 600;}}
.kpi .k {{color: {MUTED}; font-size: 0.82rem;}}
.why-row {{padding: 0.55rem 0; border-bottom: 1px solid {LINE}; color: {TEXT};}}
.why-row b {{color: {WHITE};}}
.note {{border-left: 2px solid {YELLOW}; padding: 0.2rem 0 0.2rem 0.9rem; color: {TEXT};
        margin: 0.8rem 0;}}
.note-r {{border-left-color: {RED};}}
table.clean {{width: 100%; border-collapse: collapse; font-size: 0.92rem; border: none !important;}}
table.clean th, table.clean td {{border-left: none !important; border-right: none !important;
    border-top: none !important; background: transparent !important;}}
code {{color: {TEXT} !important; background: #111 !important;}}
table.clean th {{text-align: left; color: {MUTED}; font-weight: 500; font-size: 0.78rem;
    text-transform: uppercase; letter-spacing: 0.05em; padding: 0.4rem 0.6rem 0.4rem 0;
    border-bottom: 1px solid {LINE};}}
table.clean td {{padding: 0.55rem 0.6rem 0.55rem 0; border-bottom: 1px solid {LINE}; color: {TEXT};}}
table.clean td.n {{font-family: 'IBM Plex Mono', monospace; color: {WHITE};}}
table.clean tr.ours td {{color: {YELLOW};}}
.footer {{color: {MUTED}; font-size: 0.8rem; border-top: 1px solid {LINE}; margin-top: 3rem;
          padding-top: 0.8rem;}}
</style>""", unsafe_allow_html=True)


# ---- helpers -----------------------------------------------------------------------------
def style_fig(fig, height=420, y_title=None, x_title=None):
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, height=height,
        font=dict(color=TEXT, family="IBM Plex Sans, sans-serif", size=13),
        margin=dict(l=4, r=4, t=46, b=4),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(color=TEXT)),
        hoverlabel=dict(bgcolor="#0c0c0c", font_color=WHITE, bordercolor="#333",
                        font_family="IBM Plex Sans"),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=LINE, title=x_title,
                     automargin=True, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=LINE, title=y_title,
                     automargin=True, tickfont=dict(color=MUTED))
    return fig


def show(fig):
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False})


def html(s):
    st.markdown(s, unsafe_allow_html=True)


def footer():
    html(f'<div class="footer">{DISCLAIMER} Prices are a snapshot ({AS_OF} close), not live, '
         f'and are adjusted for dividends and splits.</div>')


def pct(x, digits=0):
    return f"{x * 100:.{digits}f}%"


def share(x):
    """Win shares: one decimal unless whole, so 0.998 never shows as 100%."""
    return f"{x * 100:.0f}%" if round(x * 1000) % 10 == 0 else f"{x * 100:.1f}%"


def signed_pct(x, digits=2):
    cls, arrow = ("up", "▲") if x >= 0 else ("down", "▼")
    return f'<span class="{cls} num">{arrow} {abs(x) * 100:.{digits}f}%</span>'


def money(x):
    return f"${x:,.2f}"


def big_count(x):
    for div, unit in [(1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if x >= div:
            return f"{x / div:.2f}{unit}"
    return f"{x:.0f}"


def pill(level):
    word, colour = LEVEL[level]
    return (f'<span class="pill" style="color:{colour}"><span class="dot" '
            f'style="background:{colour}; box-shadow:0 0 8px {colour}"></span>{word}</span>')


def table(df, fmt=None, ours=None):
    """Minimal HTML table with hairlines. fmt: {column: format string}."""
    fmt = fmt or {}
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = []
    for _, r in df.iterrows():
        cls = ' class="ours"' if ours and r.iloc[0] == ours else ""
        cells = []
        for c in df.columns:
            v = r[c]
            if c in fmt and pd.notna(v):
                cells.append(f'<td class="n">{fmt[c].format(v)}</td>')
            else:
                cells.append(f"<td>{v}</td>")
        body.append(f"<tr{cls}>{''.join(cells)}</tr>")
    html(f'<table class="clean"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>')


# ---- data (all small, precomputed by src/08_app_data.py, 09_backtest.py, 06h_baselines.py)
@st.cache_data
def load_meta():
    return json.loads((APP_DIR / "app_meta.json").read_text())


@st.cache_data
def load_latest():
    return pd.read_parquet(APP_DIR / "app_latest.parquet").set_index("ticker")


@st.cache_data
def load_history():
    return pd.read_parquet(APP_DIR / "app_history.parquet")


@st.cache_data
def load_prices():
    return pd.read_parquet(APP_DIR / "app_prices.parquet").sort_values(["ticker", "date"])


@st.cache_data
def load_csv(name):
    path = RES / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_json(name):
    path = RES / name
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def quotes():
    """Last close and one-day change for every stock."""
    p = load_prices()
    last2 = p.groupby("ticker").tail(2)
    g = last2.groupby("ticker")["close"]
    return pd.DataFrame({"price": g.last(), "chg": g.last() / g.first() - 1})


meta = load_meta()
latest = load_latest()
TH = meta["thresholds"]
BASE = TH["base_rate_test"]
AS_OF = pd.Timestamp(meta["as_of"]).strftime("%d %b %Y")
NAMES = meta["names"]
TICKERS = sorted(latest.index)


def label_of(t):
    return f"{t}  ·  {NAMES.get(t, t)}"


# ---- plain-English explanations ----------------------------------------------------------
def explain(feature, r):
    """One sentence describing the stock's situation for a v3 feature (row r = latest)."""
    up_down = lambda x: "risen" if x >= 0 else "fallen"  # noqa: E731
    if feature in ("vol_20_rank", "vol_60_rank"):
        span = "month" if feature == "vol_20_rank" else "3 months"
        q = r[feature]
        if q >= 0.5:
            return (f"Over the last {span} its price has swung more than "
                    f"{q * 100:.0f}% of large US stocks")
        return (f"Over the last {span} its price has been steadier than "
                f"{(1 - q) * 100:.0f}% of large US stocks")
    if feature in ("ret_20_z", "ret_20_rank"):
        return f"It has {up_down(r['ret_20'])} {abs(r['ret_20']) * 100:.0f}% over the last month"
    if feature == "ret_60_z":
        return f"It has {up_down(r['ret_60'])} {abs(r['ret_60']) * 100:.0f}% over the last 3 months"
    if feature in ("drawdown_z", "drawdown_252_rank"):
        dd = -r["drawdown_252"]
        if dd < 0.05:
            return f"It is close to its 1-year high (only {dd * 100:.0f}% below)"
        return f"It is already well below its 1-year high ({dd * 100:.0f}% below)"
    if feature == "dist_ma200_z":
        d = r["dist_ma200"]
        return (f"It is {abs(d) * 100:.0f}% {'above' if d >= 0 else 'below'} "
                f"its average price of the last 200 days")
    if feature == "vol_vs_mkt":
        return f"Lately it swings {r['vol_vs_mkt']:.1f}× as much as the S&P 500 index"
    if feature == "beta_60_rank":
        more = "more" if r["beta_60_rank"] >= 0.5 else "less"
        return (f"It tends to move {more} with the whole market than most large stocks "
                f"(beta {r['beta_60']:.1f})")
    if feature == "volume_ratio_20_rank":
        return f"Trading volume is {r['volume_ratio_20']:.1f}× its usual level for the past month"
    if feature == "vol_ratio_20_60":
        trend = "getting bigger" if r["vol_ratio_20_60"] >= 1 else "settling down"
        return (f"Its price swings are {trend} (last month vs last 3 months: "
                f"{r['vol_ratio_20_60']:.1f}×)")
    if feature == "rsi_14":
        side = "gains" if r["rsi_14"] >= 50 else "losses"
        return f"Its last two weeks have been mostly {side} (RSI {r['rsi_14']:.0f} out of 100)"
    if feature == "vix":
        mood = "nervous" if r["vix"] >= 20 else "fairly relaxed"
        return f"The whole market is {mood} (VIX fear index = {r['vix']:.0f})"
    if feature == "vix_chg_5":
        return (f"Market nervousness (VIX) has {up_down(r['vix_chg_5'])} "
                f"{abs(r['vix_chg_5']) * 100:.0f}% in the past week")
    if feature == "mkt_ret_20":
        return f"The S&P 500 has {up_down(r['mkt_ret_20'])} {abs(r['mkt_ret_20']) * 100:.1f}% over the last month"
    return feature


def why_steps(r):
    """Start from the volatility rule, then add the largest corrections one at a time so
    the percentage-point steps add up exactly to the final v3 probability."""
    feats = [c[len("contrib_"):] for c in r.index if c.startswith("contrib_") and c != "contrib_bias"]
    contrib = pd.Series({f: r[f"contrib_{f}"] for f in feats})
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    # the trees' constant (same for every stock) is folded into the starting point
    z = r["start_logit"] + r["contrib_bias"]
    start = sig(z)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:3]
    steps = []
    for f, c in top.items():
        before = sig(z)
        z += c
        steps.append((f, sig(z) - before))
    rest = sig(z + contrib.drop(top.index).sum()) - sig(z)
    return start, steps, rest


# ---- top strip: brand, ticker of the snapshot's biggest moves, live clock ---------------
@st.cache_data
def biggest_moves(n_days=5, n_items=24):
    """Largest one-day price moves in the last n_days trading days of the snapshot."""
    p = load_prices()[["date", "ticker", "close"]].copy()
    p["ret"] = p.groupby("ticker")["close"].pct_change()
    recent = p[p["date"].isin(sorted(p["date"].unique())[-n_days:])].dropna(subset=["ret"])
    recent = recent.assign(size=recent["ret"].abs()).nlargest(n_items, "size")
    return recent.sort_values(["date", "size"], ascending=[False, False])


def top_strip():
    items = []
    for _, m in biggest_moves().iterrows():
        up = m["ret"] >= 0
        items.append(
            f'<span class="item"><b>{m["ticker"]}</b> <span class="mono">{money(m["close"])}</span> '
            f'<span class="{"up" if up else "down"} mono">{"▲" if up else "▼"} '
            f'{abs(m["ret"]) * 100:.1f}%</span> '
            f'<span class="when">{pd.Timestamp(m["date"]):%d %b}</span></span>')
    track = "".join(items)
    components.html(f"""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&family=IBM+Plex+Mono:wght@500&display=swap" rel="stylesheet">
<style>
 body {{margin:0; background:#000; color:{WHITE}; font-family:'IBM Plex Sans',sans-serif;}}
 .bar {{display:flex; align-items:center; height:46px; border-bottom:1px solid {LINE};
        overflow:hidden; box-sizing:border-box;}}
 .brand {{padding-right:22px; font-weight:700; font-size:17px; letter-spacing:0.2em;
          white-space:nowrap; color:{YELLOW}; text-shadow:0 0 14px rgba(255,207,64,0.55);
          display:flex; align-items:center; gap:9px;}}
 .brand i {{width:8px; height:8px; border-radius:50%; background:{RED};
            box-shadow:0 0 10px {RED}, 0 0 2px {RED_TXT};}}
 .label {{padding-right:14px; font-size:10.5px; line-height:1.3; color:{MUTED};
          text-transform:uppercase; letter-spacing:0.08em; white-space:nowrap;}}
 .tape {{flex:1; overflow:hidden; height:100%; display:flex; align-items:center;
         -webkit-mask-image:linear-gradient(90deg,transparent,#000 4%,#000 96%,transparent);}}
 .track {{display:inline-flex; white-space:nowrap; animation:scroll 80s linear infinite;}}
 .tape:hover .track {{animation-play-state:paused;}}
 .item {{padding:0 20px; font-size:13px; color:{TEXT};}}
 .item b {{color:{WHITE};}}
 .up {{color:{YELLOW};}} .down {{color:{RED_TXT};}} .when {{color:{MUTED}; font-size:11px;}}
 .mono {{font-family:'IBM Plex Mono',monospace;}}
 .clock {{padding-left:18px; display:flex; flex-direction:column; align-items:flex-end;
          white-space:nowrap;}}
 #t {{font-family:'IBM Plex Mono',monospace; font-size:18px; font-weight:500; color:{WHITE};
      text-shadow:0 0 12px rgba(244,241,234,0.35);}}
 #d {{font-size:10.5px; color:{MUTED};}}
 @keyframes scroll {{from {{transform:translateX(0);}} to {{transform:translateX(-50%);}}}}
 @media (max-width:700px) {{.label {{display:none;}} .brand {{font-size:14px;}}}}
</style>
<div class="bar">
  <div class="brand"><i></i>CHARTED</div>
  <div class="label">Biggest moves<br>to {AS_OF}</div>
  <div class="tape"><div class="track">{track}{track}</div></div>
  <div class="clock"><div id="t">--:--:--</div><div id="d">your local time</div></div>
</div>
<script>
 function tick() {{
   const n = new Date();
   document.getElementById('t').textContent =
     n.toLocaleTimeString([], {{hour:'2-digit', minute:'2-digit', second:'2-digit', hour12:false}});
   document.getElementById('d').textContent =
     n.toLocaleDateString([], {{weekday:'short', day:'numeric', month:'short', year:'numeric'}}) + ' · local time';
 }}
 tick(); setInterval(tick, 1000);
</script>""", height=48)


# ---- navigation and search ---------------------------------------------------------------
PAGES = ["Watchlist", "Stock", "Does it beat simple rules?", "How it works"]
if st.session_state.get("page") not in PAGES:
    st.session_state["page"] = PAGES[0]
if "my_picks" not in st.session_state:
    st.session_state["my_picks"] = [t for t in ["AAPL", "NVDA", "KO", "JPM", "AMD"] if t in TICKERS]
if "detail" not in st.session_state:
    st.session_state["detail"] = "AAPL"


def go_stock(t):
    st.session_state["detail"] = t
    st.session_state["page"] = "Stock"


def on_search():
    t = st.session_state.get("_search")
    if t:
        go_stock(t)
    st.session_state["_search"] = None


def on_nav():
    st.session_state["page"] = st.session_state["_nav"]


def save_picks():
    st.session_state["my_picks"] = st.session_state["_picks"]


def toggle_watch(t):
    picks = st.session_state["my_picks"]
    st.session_state["my_picks"] = [p for p in picks if p != t] if t in picks else picks + [t]


top_strip()
nav_col, search_col = st.columns([3, 2], vertical_alignment="center")
with nav_col:
    st.session_state["_nav"] = st.session_state["page"]
    st.radio("Pages", PAGES, key="_nav", on_change=on_nav, horizontal=True,
             label_visibility="collapsed")
with search_col:
    st.selectbox("Search", TICKERS, index=None, key="_search", on_change=on_search,
                 format_func=label_of, label_visibility="collapsed",
                 placeholder=f"Search {len(TICKERS)} stocks by ticker or company…")
page = st.session_state["page"]


# ==== page 1: watchlist ===================================================================
def compact_row(t, q, r, key_prefix):
    with st.container(key=f"row_{key_prefix}_{t}"):
        c = st.columns([2.4, 1.3, 1.3, 1.2], vertical_alignment="center")
        with c[0]:
            st.button(t, key=f"btn_{key_prefix}_{t}", type="tertiary", on_click=go_stock, args=(t,))
            html(f"<div class='cell-sub' style='margin-top:-0.85rem'>{NAMES.get(t, t)}</div>")
        c[1].markdown(f"<div class='cell num'>{money(q['price'])}</div>"
                      f"<div class='cell-sub'>{signed_pct(q['chg'])}</div>", unsafe_allow_html=True)
        c[2].markdown(f"<div class='cell num'>{pct(r['p_v3'], 1)}</div>", unsafe_allow_html=True)
        c[3].markdown(f"<div class='cell'>{pill(r['level'])}</div>", unsafe_allow_html=True)


def compact_header(key):
    with st.container(key=f"hdr_{key}"):
        c = st.columns([2.4, 1.3, 1.3, 1.2])
        for col, h in zip(c, ["Stock", "Price", "Drop chance", "Status"]):
            col.markdown(f"<div class='colhead'>{h}</div>", unsafe_allow_html=True)


def stock_row(t, q, r, key_prefix="w"):
    with st.container(key=f"row_{key_prefix}_{t}"):
        c = st.columns([2.6, 1.3, 1.1, 1.4, 1.1, 1.2], vertical_alignment="center")
        with c[0]:
            st.button(t, key=f"btn_{key_prefix}_{t}", type="tertiary", on_click=go_stock, args=(t,))
            html(f"<div class='cell-sub' style='margin-top:-0.85rem'>{NAMES.get(t, t)}</div>")
        c[1].markdown(f"<div class='cell num'>{money(q['price'])}</div>", unsafe_allow_html=True)
        c[2].markdown(f"<div class='cell'>{signed_pct(q['chg'])}</div>", unsafe_allow_html=True)
        width = min(r["p_v3"] / 0.35, 1) * 100
        colour = LEVEL[r["level"]][1]
        c[3].markdown(
            f"<div class='cell num'>{pct(r['p_v3'], 1)}</div>"
            f"<div style='height:3px; background:#151515; border-radius:2px; margin-top:4px'>"
            f"<div style='width:{width:.0f}%; height:3px; background:{colour}; border-radius:2px;"
            f" box-shadow:0 0 8px {colour}'></div></div>", unsafe_allow_html=True)
        c[4].markdown(f"<div class='cell'>{pill(r['level'])}</div>", unsafe_allow_html=True)
        c[5].markdown(f"<div class='cell num' style='color:{MUTED}'>{pct(r['p_vol'], 1)}</div>",
                      unsafe_allow_html=True)


def row_header():
    with st.container(key="hdr_w"):
        c = st.columns([2.6, 1.3, 1.1, 1.4, 1.1, 1.2])
        for col, h in zip(c, ["Stock", "Price", "Day", "10% drop chance", "Status", "Simple rule"]):
            col.markdown(f"<div class='colhead'>{h}</div>", unsafe_allow_html=True)


def page_watchlist():
    st.title("Your watchlist")
    html(f"<p style='max-width:780px'>The estimated chance that each stock closes "
         f"<b style='color:{WHITE}'>at least 10% below today's close at some point in the next "
         f"20 trading days</b> (about a month). A typical large US stock: "
         f"<b class='glow-w'>{pct(BASE)}</b>. Prices as of the {AS_OF} close.</p>")
    st.session_state["_picks"] = st.session_state["my_picks"]
    st.multiselect("Edit watchlist", TICKERS, key="_picks", on_change=save_picks,
                   format_func=label_of, label_visibility="collapsed",
                   placeholder="Add stocks to your watchlist…")
    picks = st.session_state["my_picks"]
    q = quotes()
    if picks:
        row_header()
        for t in latest.loc[picks].sort_values("p_v3", ascending=False).index:
            stock_row(t, q.loc[t], latest.loc[t])
        html(f"<div class='muted' style='margin-top:0.6rem'>Sorted riskiest first. "
             f"<b>Warning</b>: chance ≥ {pct(TH['red'], 1)} (fixed on 2019–20 data so the model "
             f"warns as often as the simple volatility rule did). <b>Watch</b>: ≥ "
             f"{pct(TH['amber'], 1)}. <b>Stable</b>: below that. Click a ticker for details.</div>")
    else:
        html("<div class='note'>Your watchlist is empty. Add stocks above or use the search "
             "bar.</div>")

    st.markdown("#### Across all stocks right now")
    c1, c2 = st.columns(2, gap="large")
    ranked = latest.sort_values("p_v3")
    with c1:
        html(f"<div class='glow-r' style='font-weight:600; margin-bottom:0.3rem'>Highest risk</div>")
        compact_header("hi")
        for t in ranked.index[::-1][:5]:
            compact_row(t, q.loc[t], latest.loc[t], "hi")
    with c2:
        html(f"<div class='glow-w' style='font-weight:600; margin-bottom:0.3rem'>Most stable</div>")
        compact_header("lo")
        for t in ranked.index[:5]:
            compact_row(t, q.loc[t], latest.loc[t], "lo")
    footer()


# ==== page 2: stock =======================================================================
RANGES = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "3Y": 10_000}


def key_stats(p, r):
    last, prev = p.iloc[-1], p.iloc[-2]
    yr = p.tail(252)
    ytd_base = p[p["date"] < pd.Timestamp(year=last["date"].year, month=1, day=1)]["close"]
    ret = lambda n: last["close"] / p["close"].iloc[-n - 1] - 1 if len(p) > n else np.nan  # noqa: E731
    stats = [
        ("Previous close", money(prev["close"])),
        ("Open", money(last["open"])),
        ("Day's range", f"{money(last['low'])} – {money(last['high'])}"),
        ("52-week range", f"{money(yr['low'].min())} – {money(yr['high'].max())}"),
        ("Volume", big_count(last["volume"])),
        ("Avg. volume (3 months)", big_count(p["volume"].tail(63).mean())),
        ("50-day average", money(p["close"].tail(50).mean())),
        ("200-day average", money(p["close"].tail(200).mean())),
        ("1-month return", f"{ret(21) * 100:+.1f}%"),
        ("3-month return", f"{ret(63) * 100:+.1f}%"),
        ("Year to date", f"{(last['close'] / ytd_base.iloc[-1] - 1) * 100:+.1f}%"
         if len(ytd_base) else "–"),
        ("1-year return", f"{ret(252) * 100:+.1f}%"),
        ("Below 1-year high", "at its high" if -r["drawdown_252"] < 0.0005
         else f"{-r['drawdown_252'] * 100:.1f}%"),
        ("Beta (60 days, vs S&P 500)", f"{r['beta_60']:.2f}"),
        ("Volatility (20 days, yearly)", f"{r['vol_20'] * np.sqrt(252) * 100:.0f}%"),
        ("RSI (14 days)", f"{r['rsi_14']:.0f}"),
    ]
    return "".join(f"<div class='stat'><span>{k}</span><span>{v}</span></div>" for k, v in stats)


def page_stock():
    t = st.session_state["detail"]
    r = latest.loc[t]
    p = load_prices()
    p = p[p["ticker"] == t]
    last, prev = p.iloc[-1], p.iloc[-2]
    chg = last["close"] / prev["close"] - 1

    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    with c1:
        never = (" · never used to build the model" if r["universe"] == "holdout" else "")
        html(f"<div class='muted' style='margin-top:0.8rem'>{meta['sector'].get(t, '')} · US large "
             f"cap{never}</div>"
             f"<div style='font-size:1.9rem; font-weight:600; color:{WHITE}'>{NAMES.get(t, t)} "
             f"<span style='color:{MUTED}; font-weight:500'>{t}</span></div>"
             f"<div style='display:flex; align-items:baseline; gap:1rem'>"
             f"<span class='bigprice glow-w'>{money(last['close'])}</span>"
             f"<span style='font-size:1.1rem'>{signed_pct(chg)} "
             f"<span class='num {'up' if chg >= 0 else 'down'}'>"
             f"({'+' if chg >= 0 else '−'}{money(abs(last['close'] - prev['close']))})</span></span>"
             f"</div><div class='muted'>At close, {AS_OF} · snapshot, not live</div>")
    with c2:
        inside = t in st.session_state["my_picks"]
        st.button("Remove from watchlist" if inside else "Add to watchlist", key="watch_btn",
                  on_click=toggle_watch, args=(t,), type="secondary" if inside else "primary")

    rng = st.radio("Range", list(RANGES), index=3, horizontal=True, key="_range",
                   label_visibility="collapsed")
    view = p.tail(RANGES[rng])
    h = load_history()
    h = h[(h["ticker"] == t) & (h["date"] >= view["date"].iloc[0])].sort_values("date")
    period_up = view["close"].iloc[-1] >= view["close"].iloc[0]
    line_c = YELLOW if period_up else RED_TXT
    fill_c = "rgba(255,207,64,0.10)" if period_up else "rgba(200,16,46,0.14)"
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.74, 0.06, 0.2],
                        vertical_spacing=0.025)
    ymin = view["low"].min() * 0.97
    fig.add_scatter(x=view["date"], y=view["close"], name="Close", line=dict(color=line_c, width=2),
                    fill="tozeroy", fillcolor=fill_c,
                    hovertemplate="Close $%{y:,.2f}<extra></extra>", row=1, col=1)
    crash = view[["date", "close"]].merge(h.loc[h["label"] == 1, ["date"]], on="date")
    fig.add_scatter(x=crash["date"], y=crash["close"], mode="markers",
                    name="A 10%+ drop came within 20 days",
                    marker=dict(color=WHITE, size=7, symbol="triangle-down",
                                line=dict(color="#000", width=0.5)),
                    hovertemplate="A 10%+ drop followed<extra></extra>", row=1, col=1)
    # risk-status ribbon: one coloured cell per day (word in the hover, legend below)
    ribbon = {"red": RED, "amber": "#b38a1a", "green": "#d9d4c7"}
    for lvl in ["red", "amber", "green"]:
        d = h[h["level"] == lvl]
        word = LEVEL[lvl][0]
        fig.add_bar(x=d["date"], y=np.ones(len(d)), name=f"Model status: {word}",
                    marker_color=ribbon[lvl], marker_line_width=0,
                    showlegend=len(d) > 0, legendgroup=lvl,
                    customdata=(d["p_v3"] * 100).to_numpy(),
                    hovertemplate=f"Status {word} (%{{customdata:.1f}}%)<extra></extra>",
                    row=2, col=1)
    vol_c = np.where(view["close"].diff().fillna(0) >= 0, "rgba(255,207,64,0.45)",
                     "rgba(200,16,46,0.6)")
    fig.add_bar(x=view["date"], y=view["volume"], name="Volume", marker_color=vol_c,
                showlegend=False, hovertemplate="Volume %{y:,.0f}<extra></extra>", row=3, col=1)
    fig.update_yaxes(range=[ymin, view["high"].max() * 1.02], tickprefix="$", row=1, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, range=[0, 1], row=2, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, row=3, col=1)
    style_fig(fig, height=500)
    fig.update_layout(hovermode="x unified", bargap=0, barmode="overlay")
    fig.update_xaxes(hoverformat="%d %b %Y", rangebreaks=[dict(bounds=["sat", "mon"])],
                     range=[view["date"].iloc[0] - pd.Timedelta(days=1),
                            view["date"].iloc[-1] + pd.Timedelta(days=1)])
    show(fig)
    html("<div class='muted' style='margin-top:-0.4rem'>The thin band under the price shows the "
         "model's status each day (red = Warning, gold = Watch, white = Stable). White triangles "
         "mark days that were followed by a real 10%+ drop within 20 trading days.</div>")

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.markdown("#### Crash risk, next 20 trading days")
        colour = LEVEL[r["level"]][1]
        html(f"<div style='display:flex; align-items:flex-end; gap:1.2rem'>"
             f"<div class='bignum' style='color:{colour}; text-shadow:0 0 22px {colour}66'>"
             f"{pct(r['p_v3'], 1)}</div><div style='padding-bottom:0.4rem'>{pill(r['level'])}"
             f"</div></div><div class='muted'>chance of closing 10%+ below today at some point in "
             f"the next month</div>"
             f"<div class='stats' style='margin-top:0.6rem'>"
             f"<div class='stat'><span>Typical stock</span><span>{pct(BASE, 1)}</span></div>"
             f"<div class='stat'><span>Simple volatility rule</span><span>{pct(r['p_vol'], 1)}</span>"
             f"</div></div>")
        st.markdown("#### Why?")
        start, steps, rest = why_steps(r)
        rows = [f"<div class='why-row'><b>Price swings alone suggest {pct(start)}</b> "
                f"<span class='muted'>(the volatility starting point)</span></div>"]
        for f, d in steps:
            arrow, c, verb = (("↑", RED_TXT, "raises") if d >= 0 else ("↓", YELLOW, "lowers"))
            rows.append(f"<div class='why-row'><b style='color:{c}'>{arrow}</b>&nbsp; "
                        f"{explain(f, r)}. <span class='muted'>This {verb} the estimate by "
                        f"{abs(d) * 100:.1f} points.</span></div>")
        if abs(rest) >= 0.0005:
            rows.append(f"<div class='why-row'><b>{'↑' if rest >= 0 else '↓'}</b>&nbsp; All other "
                        f"factors together: {'+' if rest >= 0 else '−'}{abs(rest) * 100:.1f} "
                        f"points.</div>")
        rows.append(f"<div class='why-row'>= <b class='num'>{pct(r['p_v3'], 1)}</b> final "
                    f"estimate</div>")
        html("".join(rows))
        ratio = r["p_v3"] / BASE
        if ratio >= 1.3:
            meaning = (f"The estimated chance of a 10% drop is about <b>{ratio:.1f}× the typical "
                       f"stock's</b>. If a sudden 10% fall would hurt you, that is worth knowing "
                       f"before you put more money in.")
        elif ratio <= 0.7:
            meaning = (f"Right now {t} looks <b>more stable than the typical stock</b>. That is not "
                       f"a guarantee: news or earnings can still cause sharp falls.")
        else:
            meaning = f"Right now {t} looks <b>about as risky as the typical stock</b>."
        html(f"<div class='note'><b style='color:{WHITE}'>What this means for a beginner:</b> "
             f"{meaning} This tool does not say whether to buy or sell; it estimates short-term "
             f"risk only.</div>")
    with right:
        st.markdown("#### Key statistics")
        html(f"<div class='stats' style='grid-template-columns:1fr'>{key_stats(p, r)}</div>")

    st.markdown("#### Risk estimate over time")
    hh = load_history()
    hh = hh[hh["ticker"] == t].sort_values("date")
    fig = go.Figure()
    fig.add_scatter(x=hh["date"], y=hh["p_vol"] * 100, name="Volatility rule",
                    line=dict(color=C_VOL, width=1.6),
                    hovertemplate="%{x|%d %b %Y}: rule %{y:.1f}%<extra></extra>")
    fig.add_scatter(x=hh["date"], y=hh["p_v3"] * 100, name="Model v3 (ours)",
                    line=dict(color=C_V3, width=2.2),
                    hovertemplate="%{x|%d %b %Y}: v3 %{y:.1f}%<extra></extra>")
    fig.add_hline(y=TH["red"] * 100, line=dict(color=RED_TXT, dash="dot", width=1),
                  annotation_text="warning line", annotation_font_color=MUTED,
                  annotation_position="top left")
    style_fig(fig, height=300, y_title="Chance (%)")
    fig.update_layout(hovermode="x unified")
    show(fig)
    hit = hh.dropna(subset=["label"])
    on = hit["level"] == "red"
    if len(hit):
        parts = [f"Since {pd.Timestamp(meta['history_start']):%b %Y}, {t} had a 10%+ drop ahead "
                 f"on {hit['label'].mean():.0%} of days."]
        if on.any() and (~on).any():
            parts.append(f"On warning days it was {hit.loc[on, 'label'].mean():.0%}; on other days "
                         f"{hit.loc[~on, 'label'].mean():.0%}.")
        elif on.all():
            parts.append("The model was on warning on every one of these days.")
        else:
            parts.append("The model never gave a warning in this window.")
        html(f"<div class='muted'>{' '.join(parts)} The last ~20 days have no outcome yet. Most "
             f"of these days are inside the period used to report results, so treat this as an "
             f"illustration.</div>")
    footer()


# ==== page 3: does it beat simple rules? ==================================================
def kpi(label, value, colour=WHITE, glow=True):
    shadow = f"text-shadow:0 0 18px {colour}55;" if glow else ""
    return (f"<div class='kpi'><div class='k'>{label}</div>"
            f"<div class='v' style='color:{colour}; {shadow}'>{value}</div></div>")


def page_results():
    hm = load_csv("holdout_metrics.csv").set_index("method")
    boot = load_json("holdout_bootstrap.json")
    bb = load_json("baselines_bootstrap.json") or {}
    by = load_csv("baselines_by_year.csv")
    v3, vr, ma = (hm.loc["Model v3 (rule + corrections)"], hm.loc["Volatility rule"],
                  hm.loc["200-day average rule"])

    st.title("Does it beat simple rules?")
    html(f"<div style='font-size:1.45rem; font-weight:600; line-height:1.35; max-width:900px; "
         f"margin:0.2rem 0 1.1rem 0'><span class='glow-y'>Yes.</span> Model v3 spots 10% drops "
         f"better than <span style='color:{C_VOL}'>the volatility rule</span> and far better than "
         f"<span style='color:{C_MA}'>the 200-day average rule</span>, on stocks it has never seen."
         f"</div>")
    k = st.columns(5)
    k[0].markdown(kpi("Model v3 PR-AUC", f"{v3['PR_AUC']:.3f}", YELLOW), unsafe_allow_html=True)
    k[1].markdown(kpi("Volatility rule", f"{vr['PR_AUC']:.3f}", C_VOL), unsafe_allow_html=True)
    k[2].markdown(kpi("200-day average rule", f"{ma['PR_AUC']:.3f}", C_MA, False),
                  unsafe_allow_html=True)
    k[3].markdown(kpi("vs 200-day rule", f"+{(v3['PR_AUC'] / ma['PR_AUC'] - 1) * 100:.0f}%"),
                  unsafe_allow_html=True)
    k[4].markdown(kpi("vs volatility rule", f"+{(v3['PR_AUC'] / vr['PR_AUC'] - 1) * 100:.0f}%"),
                  unsafe_allow_html=True)
    html(f"<div class='muted' style='margin-top:0.6rem'>Main test: 49 large US stocks never used "
         f"for training, tuning or choosing the model, 2021–2025. PR-AUC: higher is better; "
         f"guessing scores {hm.loc['Never warn', 'PR_AUC']:.3f}.</div>")

    # --- every baseline, every test --------------------------------------------------------
    if by is not None:
        st.markdown("#### Every test, every baseline")
        allr = by[by["year"].astype(str) == "All"].set_index("dataset")
        order = [d for d in ["Holdout 1", "Holdout 2", "Original test"] if d in allr.index]
        labels = {"Holdout 1": "Fresh holdout 1<br>(49 stocks)",
                  "Holdout 2": "Fresh holdout 2<br>(95 stocks)",
                  "Original test": "Original test<br>(99 stocks)"}
        x = [labels[d] for d in order]
        fig = go.Figure()
        for col, name, colour in [("v3", "Model v3 (ours)", C_V3),
                                  ("volatility_rule", "Volatility rule", C_VOL),
                                  ("ma200_rule", "200-day average rule", C_MA),
                                  ("no_skill", "No skill (guessing)", C_NONE)]:
            vals = allr.loc[order, col]
            fig.add_bar(x=x, y=vals, name=name, marker_color=colour,
                        text=[f"{v:.3f}" for v in vals], textposition="outside",
                        textfont=dict(color=colour if col != "no_skill" else MUTED, size=12),
                        hovertemplate=f"{name}<br>%{{x}}<br>PR-AUC %{{y:.3f}}<extra></extra>")
        fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.06)
        fig.update_yaxes(range=[0, allr[["v3"]].values.max() * 1.25])
        show(style_fig(fig, height=400, y_title="PR-AUC (higher is better)"))

        cols = st.columns(2, gap="large")
        for c, name in zip(cols, ["Holdout 1", "Holdout 2"]):
            b = bb.get(name)
            if not b:
                continue
            bv, bm = b["v3_vs_volatility_rule"], b["v3_vs_ma200_rule"]
            c.markdown(
                f"<div style='font-weight:600; color:{WHITE}'>{name} ({b['n_stocks']} stocks): "
                f"how sure are we?</div><div class='muted' style='margin-bottom:0.3rem'>1,000 "
                f"resamples of whole months</div>"
                f"<div class='stat'><span>v3 beats the 200-day rule</span><span class='glow-y'>"
                f"{share(bm['win_share'])} of resamples</span></div>"
                f"<div class='stat'><span>&nbsp;&nbsp;gap (95% range)</span><span>"
                f"{bm['gap_mean']:+.3f} ({bm['ci_low']:+.3f} to {bm['ci_high']:+.3f})</span></div>"
                f"<div class='stat'><span>v3 beats the volatility rule</span><span class='glow-y'>"
                f"{share(bv['win_share'])} of resamples</span></div>"
                f"<div class='stat'><span>&nbsp;&nbsp;gap (95% range)</span><span>"
                f"{bv['gap_mean']:+.3f} ({bv['ci_low']:+.3f} to {bv['ci_high']:+.3f})</span></div>",
                unsafe_allow_html=True)

        st.markdown("#### Year by year")
        ds = st.radio("Dataset", order, horizontal=True, key="_yr_ds",
                      format_func=lambda d: labels[d].replace("<br>", " "),
                      label_visibility="collapsed")
        yy = by[(by["dataset"] == ds) & (by["year"].astype(str) != "All")]
        fig = go.Figure()
        for col, name, colour in [("v3", "Model v3 (ours)", C_V3),
                                  ("volatility_rule", "Volatility rule", C_VOL),
                                  ("ma200_rule", "200-day average rule", C_MA)]:
            fig.add_bar(x=yy["year"].astype(str), y=yy[col], name=name, marker_color=colour,
                        hovertemplate=f"{name}<br>%{{x}}: %{{y:.3f}}<extra></extra>")
        fig.add_scatter(x=yy["year"].astype(str), y=yy["no_skill"], name="No skill (guessing)",
                        mode="lines+markers", line=dict(color=C_NONE, dash="dot", width=2),
                        hovertemplate="No skill %{x}: %{y:.3f}<extra></extra>")
        fig.update_layout(barmode="group", bargap=0.25)
        show(style_fig(fig, height=340, y_title="PR-AUC"))
        w_ma = int((yy["v3"] > yy["ma200_rule"]).sum())
        w_vr = int((yy["v3"] > yy["volatility_rule"]).sum())
        allyy = by[by["year"].astype(str) != "All"]
        html(f"<div class='muted'>In this test v3 beat the 200-day rule in <b>{w_ma} of "
             f"{len(yy)}</b> years and the volatility rule in <b>{w_vr} of {len(yy)}</b>. Across "
             f"all three tests: 200-day rule beaten in "
             f"<b>{int((allyy['v3'] > allyy['ma200_rule']).sum())} of {len(allyy)}</b> test-years, "
             f"volatility rule in <b>{int((allyy['v3'] > allyy['volatility_rule']).sum())} of "
             f"{len(allyy)}</b>.</div>")

    st.markdown("#### Precision–recall curve (fresh holdout 1)")
    curves = load_csv("holdout_pr_curves.csv")
    fig = go.Figure()
    for key in ["200-day average rule", "Model v1", "Volatility rule", "Model v3 (rule + corrections)"]:
        lab, col, dash = METHOD_STYLE[key]
        c = curves[curves["method"] == key]
        fig.add_scatter(x=c["recall"], y=c["precision"], name=f"{lab}: {hm.loc[key, 'PR_AUC']:.3f}",
                        line=dict(color=col, width=2.4 if key.startswith("Model v3") else 1.8,
                                  dash=dash),
                        hovertemplate=f"{lab}<br>catches %{{x:.0%}} of drops<br>"
                                      f"warnings right %{{y:.1%}} of the time<extra></extra>")
    base = hm.loc["Never warn", "PR_AUC"]
    fig.add_scatter(x=[0, 1], y=[base, base], name=f"No skill: {base:.3f}",
                    line=dict(color=C_NONE, width=1.5, dash="dot"),
                    hovertemplate="No skill: %{y:.1%}<extra></extra>")
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_yaxes(range=[0, 0.5], tickformat=".0%")
    show(style_fig(fig, height=440, x_title="Recall: share of real 10% drops that got a warning",
                   y_title="Precision: warnings that were right"))

    st.markdown("#### Yes/no warnings (thresholds fixed on 2019–20)")
    tbl = hm.reset_index()
    tbl["method"] = tbl["method"].map(lambda k: METHOD_STYLE.get(k, (k,))[0])
    tbl = tbl.rename(columns={"method": "Method", "PR_AUC": "PR-AUC", "precision": "Precision",
                              "recall": "Recall", "accuracy": "Accuracy",
                              "warn_rate": "Days warned"})
    f3 = "{:.3f}"
    table(tbl[["Method", "PR-AUC", "Precision", "Recall", "F1", "Accuracy", "Days warned"]],
          {c: f3 for c in ["PR-AUC", "Precision", "Recall", "F1", "Accuracy", "Days warned"]},
          ours="Model v3 (ours)")
    html(f"<div class='note note-r'><b style='color:{WHITE}'>Why not accuracy?</b> Only about 12% "
         f"of days are followed by a 10% drop. Never warning gets the best accuracy "
         f"({hm.loc['Never warn', 'accuracy']:.0%}) and catches zero drops, which is why we "
         f"report precision and recall.</div>")
    html(f"<div class='muted'>Honest note: on holdout 1 v3 beat the volatility rule on 21 of the "
         f"49 stocks individually (39 of 95 on holdout 2). Its edge comes from ranking risk "
         f"across stocks and dates, not from winning on every single stock.</div>")

    holdout2_section()

    st.markdown("#### The full story: v1 → v2 → v3 → no v4")
    table(pd.DataFrame([
        ["v1", "LightGBM on 16 price features, trained 2010–18, tuned on 2019–20",
         "Tied the volatility rule on the 2021–25 test: 0.192 vs 0.191"],
        ["v2", "v1 retrained on 2010–20 (decided after seeing v1)",
         "Worse: 0.182. Raw feature levels shift between market regimes"],
        ["v3", "Volatility rule + small learned corrections; volatility-scaled and same-day "
               "ranked features; chosen among 7 candidates on 2019–20 only",
         "Validation 0.251 vs 0.218. Fresh holdout 0.233 vs 0.216 (main result)"],
        ["v4?", "4 pre-registered candidates using high/low-range and faster volatility "
                "estimates, judged on 2019–20 only",
         "None beat v3 on validation (best 0.241 vs 0.251), so no v4. v3 then won a second "
         "fresh test: 0.183 vs 0.169 on 95 stocks"],
    ], columns=["Version", "What", "Result"]))
    mt = load_csv("metrics_test.csv")
    with st.expander("Original 99-stock test set, 2021–25 (looked at three times; disclosed)"):
        html("<p>v3 on this set: <b>0.204</b> vs volatility rule <b>0.191</b> vs 200-day rule "
             "<b>0.138</b>. Because this set was already used to judge v1 and v2, the fresh "
             "holdouts are the fair tests.</p>")
        mt = mt.rename(columns={"method": "Method", "PR_AUC": "PR-AUC", "precision": "Precision",
                                "recall": "Recall", "accuracy": "Accuracy",
                                "warn_rate": "Days warned"})
        table(mt, {c: f3 for c in mt.columns if c != "Method"})

    backtest_section()
    footer()


def holdout2_section():
    m2, b2 = load_csv("holdout2_metrics.csv"), load_json("holdout2_bootstrap.json")
    sm = load_json("holdout2_summary.json")
    if m2 is None or b2 is None or sm is None:
        return
    st.markdown("#### Second fresh test: 95 more stocks, pre-registered")
    html(f"<p style='max-width:900px'>Before trying to improve v3, we wrote down the rules and "
         f"picked a second set of {sm['n_stocks']} large US stocks never used anywhere "
         f"(<code>results/v4_preregistration.md</code>). None of our 4 new candidate models beat "
         f"v3 on 2019–20 data, so v3 stayed. The second test ran once: v3 won again, and also "
         f"caught more drops <b>and</b> gave more accurate warnings at the fixed warning line.</p>")
    t = m2.copy()
    t["method"] = t["method"].map(lambda k: METHOD_STYLE.get(k, (k,))[0])
    t = t.rename(columns={"method": "Method", "PR_AUC": "PR-AUC", "precision": "Precision",
                          "recall": "Recall"})[["Method", "PR-AUC", "Precision", "Recall"]]
    table(t, {c: "{:.3f}" for c in ["PR-AUC", "Precision", "Recall"]}, ours="Model v3 (ours)")
    html(f"<div class='muted' style='margin-top:0.5rem'>v3 beat the volatility rule in "
         f"{share(b2['win_share'])} of {b2['n_boot']:,} resamples (gap {b2['gap_mean']:+.3f}, 95% "
         f"range {b2['ci_low']:+.3f} to {b2['ci_high']:+.3f}). Years won {sm['years_won']} of "
         f"{sm['n_years']}. Not downloaded: {', '.join(sm['missing']) or 'none'}.</div>")


def backtest_section():
    summary = load_csv("backtest_summary.csv")
    curves = load_csv("backtest_curves.csv")
    if summary is None or curves is None:
        return
    st.markdown("#### Stress test: what if you sold every time a warning came on?")
    html("<p style='max-width:900px'>v3 is built to estimate risk, not to time trades; this test "
         "is here for honesty and is not part of the main result. An equal-weight portfolio "
         "holds each stock, but moves it to cash (0% return) while its warning is on. Warnings "
         "use the thresholds fixed on 2019–20. No trading costs, no taxes, no tuning.</p>")
    names = {"v3": ("Model v3 (ours)", C_V3), "vol_rule": ("Volatility rule", C_VOL),
             "ma200": ("200-day average rule", C_MA), "buy_hold": ("Buy and hold", "#7a7468")}
    universes = list(dict.fromkeys(summary["universe"]))
    u = st.radio("Universe", universes, horizontal=True, key="_bt_u", label_visibility="collapsed")
    cu = curves[curves["universe"] == u]
    fig = go.Figure()
    for key, (lab, col) in names.items():
        c = cu[cu["strategy"] == key]
        fig.add_scatter(x=pd.to_datetime(c["date"]), y=c["value"], name=lab,
                        line=dict(color=col, width=2),
                        hovertemplate=f"{lab}<br>%{{x|%d %b %Y}}: $%{{y:.2f}}<extra></extra>")
    fig.update_layout(hovermode="x unified")
    show(style_fig(fig, height=360, y_title="Value of $1 invested"))
    s = summary[summary["universe"] == u].copy()
    s["strategy"] = s["strategy"].map(lambda k: names[k][0])
    s = s.rename(columns={"strategy": "Strategy", "total_return": "Total return",
                          "cagr": "Per year", "max_drawdown": "Worst fall",
                          "volatility": "Yearly volatility", "time_in_cash": "Days in cash"})
    pf = "{:.1%}"
    table(s[["Strategy", "Total return", "Per year", "Worst fall", "Yearly volatility",
             "Days in cash"]],
          {c: pf for c in ["Total return", "Per year", "Worst fall", "Yearly volatility",
                           "Days in cash"]}, ours="Model v3 (ours)")
    b = summary[summary["universe"] == u].set_index("strategy")
    html(f"<div class='note'>Stepping aside on v3 warnings cut the worst fall from "
         f"{pct(-b.loc['buy_hold', 'max_drawdown'])} to {pct(-b.loc['v3', 'max_drawdown'])}, but "
         f"the money ended up {pct(b.loc['v3', 'total_return'])} higher instead of "
         f"{pct(b.loc['buy_hold', 'total_return'])} with buy-and-hold (volatility rule: "
         f"{pct(b.loc['vol_rule', 'total_return'])}; 200-day rule: "
         f"{pct(b.loc['ma200', 'total_return'])}). Missing the rebounds after warnings costs a "
         f"lot: the warnings are better for understanding risk than for timing the market.</div>")
    note = RES / "backtest_note.txt"
    if note.exists():
        html(f"<div class='muted'>{note.read_text()}</div>")


# ==== page 4: how it works ================================================================
def page_how():
    st.title("How it works")
    st.markdown(f"""
#### The question we answer
For a stock on a given day: **will its closing price, at any point in the next 20 trading days
(about one month), be at least 10% below today's close?** We call that a "10% drop". On a
typical large US stock this happened on about **{pct(BASE)}** of days in 2021–25.

The app shows a **chance** (for example 18%), not a yes/no prediction. A higher chance means
drops like this have been more common in situations that looked like today.

#### What is volatility?
Volatility measures how much a price jumps around from day to day. A stock that moves 3% a day
is far more likely to fall 10% in a month than one that moves 0.5% a day. This simple idea is
already a strong warning sign, so we use it as our main **baseline**:
- *Volatility rule (chance)*: from the last 20 days' price swings, a standard formula gives the
  chance of a 10% fall; we adjust it to match how often falls really happened in 2010–18.
- *Volatility rule (yes/no)*: warn when the last 20 days' swings are bigger than on 80% of
  days in 2010–18.

Our second baseline is the classic **200-day average rule**: warn when the price is below its
average of the last 200 trading days. We also compare with **never warning**.

#### Our model (v3): the simple rule plus learned corrections
v3 starts from the volatility rule and lets a small, deliberately restrained machine-learning
model (LightGBM, 54 tiny decision trees with 3 leaves each) nudge the chance up or down using
16 signals: recent returns, how far the stock is below its 1-year high, how it ranks against
other large US stocks today, trading volume, and market mood (VIX, S&P 500). On each stock's
page, **Why?** shows the exact size of each nudge.

Warning levels, all fixed using 2019–20 data only:
- **Warning**: chance ≥ {pct(TH['red'], 1)}. At this level v3 warned on the same share of
  2019–20 days as the simple yes/no volatility rule.
- **Watch**: chance ≥ {pct(TH['amber'], 1)} (twice that share of days).
- **Stable**: below that.

#### How we measured it (and kept it honest)
- **Train on the past, test on the future:** trained on 2010–18, choices made on 2019–20
  (validation), tested on 2021–25, with 20-trading-day gaps so no answer leaks across.
- **Two fresh holdouts:** the main test uses 49 other large US stocks never used for anything
  before v3 was frozen. A second, pre-registered test on 95 more stocks confirmed it.
- **Precision** = of the warnings given, the share followed by a real 10% drop.
  **Recall** = of the real 10% drops, the share that got a warning beforehand.
  **PR-AUC** summarises both over every possible warning level (higher is better; guessing
  scores about 0.12).
- **Not accuracy:** drops are rare, so "never warn" is 88% accurate and completely useless.
- Nothing was tuned on test or holdout results. Every threshold comes from 2019–20.

#### Limitations (please read)
- **Survivorship bias:** we use today's large companies, which by definition survived. Some
  stocks (EA, IPG, BK, MMC) dropped out of the free data source. Real-world risk may be higher.
- **Overlapping labels:** neighbouring days share most of their 20-day future, so there are far
  fewer truly independent examples than rows. We used a monthly block bootstrap for this.
- **False alarms cost something:** most warnings (about 80%) are *not* followed by a 10% drop.
  Acting on every warning can mean selling too early and missing gains (see the stress test).
- **Data snapshot, not live:** prices end on **{AS_OF}**. The chances describe that date.
- **Small edge over the volatility rule:** consistent, but small. The volatility rule alone
  already captures most of the signal; the 200-day rule is clearly weaker.
- Large US stocks only; no news, earnings dates or fundamentals.

#### Sources
Prices: Yahoo Finance via the `yfinance` library. Tools: LightGBM, scikit-learn, pandas,
Streamlit, Plotly. AI assistant: Claude (Anthropic) helped write code.
""")
    footer()


{"Watchlist": page_watchlist, "Stock": page_stock, "Does it beat simple rules?": page_results,
 "How it works": page_how}[page]()
