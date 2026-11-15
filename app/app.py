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

# ---- look and feel -----------------------------------------------------------------------
BG, GRID, TEXT, MUTED = "#000000", "#262626", "#c3c2b7", "#8a897f"
C_V3, C_VOL, C_MA, C_V1, C_NONE = "#3987e5", "#d95926", "#199e70", "#c98500", "#8a897f"
METHOD_STYLE = {  # name in results files -> (label, colour, dash)
    "Model v3 (rule + corrections)": ("Model v3 (ours)", C_V3, "solid"),
    "Volatility rule": ("Volatility rule (simple)", C_VOL, "solid"),
    "200-day average rule": ("200-day average rule", C_MA, "solid"),
    "Model v1": ("Model v1 (first try)", C_V1, "dash"),
}
LEVEL = {
    "red": ("⛔", "Warning", "#e5484d"),
    "amber": ("⚠️", "Watch", "#e5a50a"),
    "green": ("✅", "Calm", "#2bb673"),
}

st.set_page_config(page_title="Charted: crash-risk early warning", page_icon="📉",
                   layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
header[data-testid="stHeader"] {display: none;}
section[data-testid="stSidebar"], div[data-testid="stSidebarCollapsedControl"] {display: none;}
.block-container {padding-top: 1rem; max-width: 1240px;}
iframe {display: block;}
h1 {font-size: 2rem !important; letter-spacing: -0.01em; margin-bottom: 0.2rem;}
h4 {font-size: 1.1rem !important; margin-top: 1.2rem;}
.num {font-family: 'IBM Plex Mono', monospace; font-feature-settings: "tnum";}
.big-num {font-family: 'IBM Plex Mono', monospace; font-size: 2.5rem; font-weight: 600;
          line-height: 1.1; margin: 0.4rem 0 0.1rem 0; letter-spacing: -0.02em;}
.badge {display: inline-block; padding: 0.12rem 0.6rem; border-radius: 999px; font-weight: 500;
        font-size: 0.85rem; border: 1px solid; margin-bottom: 0.5rem;}
.muted {color: #8a897f; font-size: 0.88rem;}
.lead {color: #c3c2b7; font-size: 1.02rem; max-width: 820px;}
.card-title {font-size: 1.15rem; font-weight: 600;}
.kv {display: flex; justify-content: space-between; font-size: 0.92rem; padding: 0.15rem 0;}
.kv b {font-family: 'IBM Plex Mono', monospace; font-weight: 500;}
.why-row {padding: 0.45rem 0; border-bottom: 1px solid #1c1c1c;}
.hero {border: 1px solid #262626; border-radius: 12px; padding: 1.1rem 1.3rem;
       background: linear-gradient(135deg, rgba(57,135,229,0.12), rgba(0,0,0,0) 60%);}
.hero .big {font-family: 'IBM Plex Mono', monospace; font-size: 2.1rem; font-weight: 600;}
.footer {color: #8a897f; font-size: 0.82rem; border-top: 1px solid #1c1c1c;
         margin-top: 2.5rem; padding-top: 0.8rem;}
div[data-testid="stSegmentedControl"] button p {font-size: 0.92rem;}
</style>""", unsafe_allow_html=True)


def style_fig(fig, height=420, y_title=None, x_title=None):
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, height=height,
        font=dict(color=TEXT, family="IBM Plex Sans, sans-serif", size=13),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#111111", font_color="#f0efec", bordercolor="#3a3a3a"),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, title=x_title,
                     automargin=True)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, title=y_title,
                     automargin=True)
    return fig


def show(fig):
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False})


def footer():
    st.markdown(f'<div class="footer">{DISCLAIMER} Data is a snapshot, not live prices.</div>',
                unsafe_allow_html=True)


def pct(x, digits=0):
    return f"{x * 100:.{digits}f}%"


# ---- data (all small, precomputed by src/08_app_data.py and src/09_backtest.py) ---------
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
def load_csv(name):
    path = RES / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_json(name):
    path = RES / name
    return json.loads(path.read_text()) if path.exists() else None


meta = load_meta()
latest = load_latest()
TH = meta["thresholds"]
BASE = TH["base_rate_test"]
AS_OF = pd.Timestamp(meta["as_of"]).strftime("%d %b %Y")
NAMES = meta["names"]
TICKERS = sorted(latest.index)


def label_of(t):
    return f"{t} · {NAMES.get(t, t)}"


def badge(level):
    icon, word, colour = LEVEL[level]
    return (f'<span class="badge" style="color:{colour}; border-color:{colour};">'
            f'{icon} {word}</span>')


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
        return (f"Over the last {span} its price has been calmer than "
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
        trend = "getting bigger" if r["vol_ratio_20_60"] >= 1 else "calming down"
        return (f"Its price swings are {trend} (last month vs last 3 months: "
                f"{r['vol_ratio_20_60']:.1f}×)")
    if feature == "rsi_14":
        side = "gains" if r["rsi_14"] >= 50 else "losses"
        return f"Its last two weeks have been mostly {side} (RSI {r['rsi_14']:.0f} out of 100)"
    if feature == "vix":
        mood = "nervous" if r["vix"] >= 20 else "fairly calm"
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


# ---- top bar: brand, ticker of the snapshot's biggest moves, live clock ------------------
@st.cache_data
def biggest_moves(n_days=5, n_items=24):
    """Largest one-day price moves in the last n_days trading days of the snapshot."""
    h = load_history()[["date", "ticker", "close"]].sort_values(["ticker", "date"])
    h["ret"] = h.groupby("ticker")["close"].pct_change()
    recent = h[h["date"].isin(sorted(h["date"].unique())[-n_days:])].dropna(subset=["ret"])
    recent = recent.assign(size=recent["ret"].abs()).nlargest(n_items, "size")
    return recent.sort_values(["date", "size"], ascending=[False, False])


def top_bar():
    items = []
    for _, m in biggest_moves().iterrows():
        up = m["ret"] >= 0
        items.append(
            f'<span class="item"><b>{m["ticker"]}</b> '
            f'<span class="{"up" if up else "down"} mono">{"▲" if up else "▼"} '
            f'{abs(m["ret"]) * 100:.1f}%</span> '
            f'<span class="when">{pd.Timestamp(m["date"]):%d %b}</span></span>')
    track = "".join(items)
    components.html(f"""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@500&display=swap" rel="stylesheet">
<style>
 body {{margin:0; background:#000; color:#f0efec; font-family:'IBM Plex Sans',sans-serif;}}
 .bar {{display:flex; align-items:center; height:50px; border:1px solid #262626;
        border-radius:12px; background:#0b0b0b; overflow:hidden; box-sizing:border-box;}}
 .brand {{padding:0 16px; font-weight:600; font-size:16px; white-space:nowrap; height:100%;
          display:flex; align-items:center; border-right:1px solid #262626;}}
 .label {{padding:0 12px; font-size:11.5px; line-height:1.25; color:#8a897f; height:100%;
          display:flex; align-items:center; border-right:1px solid #262626;}}
 .tape {{flex:1; overflow:hidden; height:100%; display:flex; align-items:center;
         -webkit-mask-image:linear-gradient(90deg,transparent,#000 3%,#000 97%,transparent);}}
 .track {{display:inline-flex; white-space:nowrap; animation:scroll 70s linear infinite;}}
 .tape:hover .track {{animation-play-state:paused;}}
 .item {{padding:0 18px; font-size:13px; border-right:1px solid #1c1c1c;}}
 .up {{color:#2bb673;}} .down {{color:#e5484d;}} .when {{color:#8a897f; font-size:11.5px;}}
 .mono {{font-family:'IBM Plex Mono',monospace;}}
 .clock {{padding:0 16px; height:100%; display:flex; flex-direction:column; justify-content:center;
          align-items:flex-end; border-left:1px solid #262626; white-space:nowrap;}}
 #t {{font-family:'IBM Plex Mono',monospace; font-size:17px; font-weight:500;}}
 #d {{font-size:11px; color:#8a897f;}}
 @keyframes scroll {{from {{transform:translateX(0);}} to {{transform:translateX(-50%);}}}}
 @media (max-width:700px) {{.label, .brand {{display:none;}}}}
</style>
<div class="bar">
  <div class="brand">📉&nbsp;Charted</div>
  <div class="label">Biggest daily moves<br>snapshot to {AS_OF}</div>
  <div class="tape"><div class="track">{track}{track}</div></div>
  <div class="clock"><div id="t">--:--:--</div><div id="d">your local time</div></div>
</div>
<script>
 function tick() {{
   const n = new Date();
   document.getElementById('t').textContent =
     n.toLocaleTimeString([], {{hour:'2-digit', minute:'2-digit', second:'2-digit', hour12:false}});
   document.getElementById('d').textContent =
     n.toLocaleDateString([], {{weekday:'short', day:'numeric', month:'short', year:'numeric'}}) + ' · your time';
 }}
 tick(); setInterval(tick, 1000);
</script>""", height=52)


PAGES = ["Check my stocks", "Stock detail", "Does it beat a simple rule?", "How it works"]
if st.session_state.get("page") not in PAGES:
    st.session_state["page"] = PAGES[0]


def nav_changed():
    if st.session_state["_nav"]:  # clicking the active page again deselects it: ignore
        st.session_state["page"] = st.session_state["_nav"]


top_bar()
nav_col, info_col = st.columns([3, 1], vertical_alignment="center")
with nav_col:
    st.session_state["_nav"] = st.session_state["page"]
    st.segmented_control("Pages", PAGES, key="_nav", on_change=nav_changed,
                         label_visibility="collapsed")
with info_col:
    st.markdown(f"<div class='muted' style='text-align:right'>Snapshot <b>{AS_OF}</b> · not live"
                f"</div>", unsafe_allow_html=True)
page = st.session_state["page"]

# Widget state is dropped when a page doesn't render that widget, so the chosen stocks
# live in "my_picks" and are copied into the multiselect each time it is drawn.
if "my_picks" not in st.session_state:
    st.session_state["my_picks"] = [t for t in ["AAPL", "NVDA", "KO"] if t in TICKERS]


def save_picks():
    st.session_state["my_picks"] = st.session_state["_picks"]


def go_detail(t):
    st.session_state["detail"] = t
    st.session_state["page"] = "Stock detail"


# ==== page 1: check my stocks =============================================================
def page_check():
    st.title("Check my stocks")
    st.markdown(
        f"For each stock: the estimated **chance its price closes at least 10% below today's "
        f"close at some point in the next 20 trading days (about a month)**. "
        f"Data snapshot: **{AS_OF}**.")
    st.session_state["_picks"] = st.session_state["my_picks"]
    picks = st.multiselect("Stocks you hold or plan to buy (type a ticker or company name)",
                           TICKERS, key="_picks", on_change=save_picks, format_func=label_of,
                           placeholder="Search e.g. Apple, TSLA, Coca-Cola…")
    if not picks:
        st.info("Pick at least one stock above.")
        return footer()

    rows = latest.loc[picks].sort_values("p_v3", ascending=False)
    st.caption(f"Sorted riskiest first. ⛔ Warning = chance at or above {pct(TH['red'], 1)}; this "
               f"line was fixed on 2019–20 data so that the model warns as often as the simple "
               f"volatility rule did, and it sits close to the typical stock's {pct(BASE)}. "
               f"⚠️ Watch = {pct(TH['amber'], 1)} or more. ✅ Calm = below that.")
    cols = st.columns(3)
    for i, (t, r) in enumerate(rows.iterrows()):
        with cols[i % 3]:
            with st.container(border=True):
                tag = (" · <span class='muted'>never used in training</span>"
                       if r["universe"] == "holdout" else "")
                st.markdown(
                    f"<div class='card-title'>{t}</div>"
                    f"<div class='muted'>{NAMES.get(t, t)} · {meta['sector'].get(t, '')}{tag}</div>"
                    f"<div class='big-num'>{pct(r['p_v3'])}</div>"
                    f"<div class='muted'>chance of a 10%+ drop in the next month</div>"
                    f"<div style='margin-top:0.5rem'>{badge(r['level'])}</div>"
                    f"<div class='kv'><span>Typical stock</span><b>{pct(BASE)}</b></div>"
                    f"<div class='kv'><span>Simple volatility rule says</span>"
                    f"<b>{pct(r['p_vol'])}</b></div>",
                    unsafe_allow_html=True)
                st.button("Why? See details →", key=f"go_{t}", on_click=go_detail, args=(t,))

    st.markdown("#### All your picks at a glance")
    fig = go.Figure()
    fig.add_bar(y=rows.index, x=rows["p_v3"] * 100, orientation="h", name="Model v3 (ours)",
                marker_color=C_V3,
                hovertemplate="%{y}: %{x:.1f}% (model v3)<extra></extra>")
    fig.add_bar(y=rows.index, x=rows["p_vol"] * 100, orientation="h",
                name="Volatility rule (simple)", marker_color=C_VOL,
                hovertemplate="%{y}: %{x:.1f}% (volatility rule)<extra></extra>")
    fig.add_vline(x=BASE * 100, line=dict(color=C_NONE, dash="dot", width=2))
    fig.add_annotation(x=BASE * 100, y=1.0, yref="paper", yanchor="bottom",
                       text=f"typical stock {pct(BASE)}", showarrow=False, xanchor="left",
                       font=dict(color=TEXT))
    fig.update_layout(barmode="group", yaxis=dict(autorange="reversed"),
                      legend=dict(orientation="h", y=1.08, x=0, yanchor="bottom"))
    show(style_fig(fig, height=140 + 45 * len(rows),
                   x_title="Chance of a 10%+ drop within 20 trading days (%)"))
    footer()


# ==== page 2: stock detail ================================================================
def page_detail():
    st.title("Stock detail")
    default = st.session_state.get("detail") or (st.session_state["my_picks"] or ["AAPL"])[0]
    st.session_state["_detail"] = default
    t = st.selectbox("Stock", TICKERS, key="_detail", format_func=label_of,
                     on_change=lambda: st.session_state.update(detail=st.session_state["_detail"]))
    r = latest.loc[t]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(
            f"<div class='muted'>{NAMES.get(t, t)} · {meta['sector'].get(t, '')}</div>"
            f"<div class='big-num'>{pct(r['p_v3'])}</div>"
            f"<div class='muted'>chance of a 10%+ drop in the next 20 trading days "
            f"(as of {AS_OF})</div><div style='margin-top:0.5rem'>{badge(r['level'])}</div>"
            f"<div>Typical stock: <b>{pct(BASE)}</b> · Volatility rule: <b>{pct(r['p_vol'])}</b></div>",
            unsafe_allow_html=True)
        if r["universe"] == "holdout":
            st.caption("This stock was never used to build or choose the model.")

    with c2:
        st.markdown("#### Why?")
        start, steps, rest = why_steps(r)
        html = [f"<div class='why-row'>📏 <b>Price swings alone suggest {pct(start)}</b> "
                f"<span class='muted'>(the volatility starting point)</span></div>"]
        for f, d in steps:
            arrow, colour, verb = (("↑", "#e5484d", "raises") if d >= 0
                                   else ("↓", "#2bb673", "lowers"))
            html.append(f"<div class='why-row'><b style='color:{colour}'>{arrow}</b> "
                        f"{explain(f, r)}. <span class='muted'>This {verb} the estimate by "
                        f"{abs(d) * 100:.1f} points.</span></div>")
        if abs(rest) >= 0.0005:
            arrow = "↑" if rest >= 0 else "↓"
            html.append(f"<div class='why-row'><b>{arrow}</b> All other factors together: "
                        f"{'+' if rest >= 0 else '−'}{abs(rest) * 100:.1f} points.</div>")
        html.append(f"<div class='why-row'>= <b>{pct(r['p_v3'], 1)}</b> final estimate</div>")
        st.markdown("".join(html), unsafe_allow_html=True)

    ratio = r["p_v3"] / BASE
    if ratio >= 1.3:
        meaning = (f"The estimated chance of a 10% drop is about **{ratio:.1f}× the typical "
                   f"stock's** (see \"Why?\" above for what drives it). If a sudden "
                   f"10% fall would hurt you, that is worth knowing before you put more money in.")
    elif ratio <= 0.7:
        meaning = (f"Right now {t} looks **calmer than the typical stock**. A calm reading is "
                   f"not a guarantee: surprises (news, earnings) can still cause sharp falls.")
    else:
        meaning = (f"Right now {t} looks **about as risky as the typical stock** "
                   f"over the next month.")
    st.info(f"**What this means for a beginner:** {meaning} This tool does not say whether to "
            f"buy or sell; it estimates short-term risk only.")

    h = load_history()
    h = h[h["ticker"] == t].sort_values("date")
    st.markdown(f"#### Price and warnings since {pd.Timestamp(meta['history_start']):%b %Y}")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38],
                        vertical_spacing=0.06)
    fig.add_scatter(x=h["date"], y=h["close"], name="Close price", line=dict(color="#f0efec", width=2),
                    hovertemplate="%{x|%d %b %Y}: $%{y:.2f}<extra></extra>", row=1, col=1)
    crash = h[h["label"] == 1]
    fig.add_scatter(x=crash["date"], y=crash["close"], mode="markers",
                    name="A 10%+ drop followed within 20 days",
                    marker=dict(color="#e5484d", size=5, symbol="triangle-down"),
                    hovertemplate="%{x|%d %b %Y}: a 10%+ drop followed<extra></extra>",
                    row=1, col=1)
    # shade runs of red warnings
    red = (h["level"] == "red").to_numpy()
    dates = h["date"].to_numpy()
    starts = np.where(red & ~np.r_[False, red[:-1]])[0]
    ends = np.where(red & ~np.r_[red[1:], False])[0]
    for s, e in zip(starts, ends):
        fig.add_vrect(x0=dates[s], x1=dates[e], fillcolor="#e5484d", opacity=0.25,
                      line_width=0, row="all", col=1)
    if len(starts):
        fig.add_scatter(x=[None], y=[None], mode="markers", name="⛔ Warning period",
                        marker=dict(color="rgba(229,72,77,0.35)", size=12, symbol="square"),
                        row=1, col=1)
    fig.add_scatter(x=h["date"], y=h["p_v3"] * 100, name="Model v3 (ours)",
                    line=dict(color=C_V3, width=2),
                    hovertemplate="%{x|%d %b %Y}: v3 %{y:.1f}%<extra></extra>", row=2, col=1)
    fig.add_scatter(x=h["date"], y=h["p_vol"] * 100, name="Volatility rule (simple)",
                    line=dict(color=C_VOL, width=2),
                    hovertemplate="%{x|%d %b %Y}: rule %{y:.1f}%<extra></extra>", row=2, col=1)
    fig.add_hline(y=TH["red"] * 100, line=dict(color="#e5484d", dash="dot", width=1), row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Chance (%)", row=2, col=1)
    style_fig(fig, height=600)
    fig.update_layout(hovermode="x unified")
    show(fig)
    hit = h.dropna(subset=["label"])
    on = hit["level"] == "red"
    if len(hit):
        parts = [f"In this window, {t} had a 10%+ drop ahead on {hit['label'].mean():.0%} of days."]
        if on.any() and (~on).any():
            parts.append(f"On ⛔ warning days it was {hit.loc[on, 'label'].mean():.0%}; "
                         f"on other days {hit.loc[~on, 'label'].mean():.0%}.")
        elif on.all():
            parts.append("The model was on ⛔ warning on every one of these days.")
        else:
            parts.append("The model never gave a ⛔ warning in this window.")
        st.caption(
            " ".join(parts) + f" Dotted line = warning threshold "
            f"({pct(TH['red'], 1)}). The last ~20 days have no outcome yet. Note: most of these "
            f"days are inside the period used to report results, so treat this as an illustration.")
    footer()


# ==== page 3: does it beat a simple rule? ================================================
def second_test_line():
    m2, b2 = load_csv("holdout2_metrics.csv"), load_json("holdout2_bootstrap.json")
    sm = load_json("holdout2_summary.json")
    if m2 is None or b2 is None or sm is None:
        return ""
    m2 = m2.set_index("method")
    return (f"<div style='margin-top:0.9rem; padding-top:0.8rem; border-top:1px solid #262626'>"
            f"<b>Confirmed on a second, pre-registered test</b> with {sm['n_stocks']} more "
            f"never-used stocks: v3 <b class='num' style='color:{C_V3}'>"
            f"{m2.loc['Model v3 (rule + corrections)', 'PR_AUC']:.3f}</b> vs rule "
            f"<b class='num' style='color:{C_VOL}'>{m2.loc['Volatility rule', 'PR_AUC']:.3f}"
            f"</b> · won <b>{pct(b2['win_share'])}</b> of bootstrap resamples · "
            f"<b>{sm['years_won']} of {sm['n_years']}</b> years.</div>")


def holdout2_section():
    m2, b2 = load_csv("holdout2_metrics.csv"), load_json("holdout2_bootstrap.json")
    sm, yr2 = load_json("holdout2_summary.json"), load_csv("holdout2_by_year.csv")
    if m2 is None or b2 is None or sm is None:
        return
    st.markdown("#### Second fresh test: 95 more stocks, pre-registered")
    st.markdown(
        f"Before trying to improve v3, we wrote down the rules and picked a second set of "
        f"{sm['n_stocks']} large US stocks never used anywhere (see "
        f"`results/v4_preregistration.md`). None of our 4 new candidate models beat v3 on "
        f"2019–20 data, so v3 stayed. The second test ran once: v3 won again, and this "
        f"time it also caught more drops (recall) **and** gave more accurate warnings "
        f"(precision) at the fixed warning line.")
    c1, c2 = st.columns([3, 2])
    with c1:
        h1 = load_csv("holdout_metrics.csv").set_index("method")
        m2i = m2.set_index("method")
        fig = go.Figure()
        for k, col in [("Model v3 (rule + corrections)", C_V3), ("Volatility rule", C_VOL)]:
            vals = [h1.loc[k, "PR_AUC"], m2i.loc[k, "PR_AUC"]]
            fig.add_bar(x=["Holdout 1 (49 stocks)", f"Holdout 2 ({sm['n_stocks']} stocks)"],
                        y=vals, name=METHOD_STYLE[k][0], marker_color=col,
                        text=[f"{v:.3f}" for v in vals], textposition="outside",
                        hovertemplate=f"{METHOD_STYLE[k][0]}<br>%{{x}}: %{{y:.3f}}<extra></extra>")
        fig.update_layout(barmode="group")
        fig.update_yaxes(range=[0, 0.28])
        show(style_fig(fig, height=340, y_title="PR-AUC (higher is better)"))
    with c2:
        t = m2.copy()
        t["method"] = t["method"].map(lambda k: METHOD_STYLE.get(k, (k,))[0])
        t = t.rename(columns={"method": "Method", "PR_AUC": "PR-AUC", "precision": "Precision",
                              "recall": "Recall"})[["Method", "PR-AUC", "Precision", "Recall"]]
        st.dataframe(t, hide_index=True, column_config={
            c: st.column_config.NumberColumn(format="%.3f") for c in ["PR-AUC", "Precision", "Recall"]})
        st.caption(f"Bootstrap: v3 won {pct(b2['win_share'])} of {b2['n_boot']:,} resamples, gap "
                   f"{b2['gap_mean']:+.3f} (95% CI {b2['ci_low']:+.3f} to {b2['ci_high']:+.3f}). "
                   f"Years won {sm['years_won']} of {sm['n_years']}. Stocks won "
                   f"{sm['stocks_won']} of {sm['stocks_compared']}. "
                   f"Not downloaded: {', '.join(sm['missing']) or 'none'}.")


def page_results():
    st.title("Does it beat a simple rule?")
    hm = load_csv("holdout_metrics.csv").set_index("method")
    boot = load_json("holdout_bootstrap.json")
    v3, vr = hm.loc["Model v3 (rule + corrections)"], hm.loc["Volatility rule"]
    yr = load_csv("holdout_by_year.csv")
    years_won = int((yr["v3"] > yr["volatility_rule"]).sum())
    gain = v3["PR_AUC"] / vr["PR_AUC"] - 1
    st.markdown(f"""
<div class="hero">
  <div style="font-size:1.25rem; font-weight:600;">Yes. Model v3 spots 10% drops better than
  the simple volatility rule, on stocks it has never seen.</div>
  <div class="muted" style="margin:0.3rem 0 0.9rem 0;">Fresh holdout: 49 large US stocks not used
  for training, tuning or choosing the model · 2021–2025 · the margin is small but consistent</div>
  <div style="display:flex; gap:2.5rem; flex-wrap:wrap;">
    <div><div class="muted">Model v3 PR-AUC</div><div class="big" style="color:{C_V3}">{v3['PR_AUC']:.3f}</div></div>
    <div><div class="muted">Volatility rule PR-AUC</div><div class="big" style="color:{C_VOL}">{vr['PR_AUC']:.3f}</div></div>
    <div><div class="muted">Improvement</div><div class="big">+{gain * 100:.0f}%</div></div>
    <div><div class="muted">Bootstrap resamples won</div><div class="big">{pct(boot['win_share'])}</div></div>
    <div><div class="muted">Years won</div><div class="big">{years_won} of {len(yr)}</div></div>
  </div>
  {second_test_line()}
</div>""", unsafe_allow_html=True)
    st.caption(f"PR-AUC: higher is better; it scores how well the warnings rank risky days "
               f"above safe ones. Guessing scores the crash rate "
               f"({hm.loc['Never warn', 'PR_AUC']:.3f}). 95% bootstrap range of the gap: "
               f"{boot['ci_low']:+.3f} to {boot['ci_high']:+.3f}, entirely above zero.")

    st.markdown("#### Precision–recall curve (fresh holdout)")
    curves = load_csv("holdout_pr_curves.csv")
    fig = go.Figure()
    for k, (lab, col, dash) in METHOD_STYLE.items():
        c = curves[curves["method"] == k]
        fig.add_scatter(x=c["recall"], y=c["precision"], name=f"{lab}: {hm.loc[k, 'PR_AUC']:.3f}",
                        line=dict(color=col, width=2, dash=dash),
                        hovertemplate=f"{lab}<br>catches %{{x:.0%}} of drops<br>"
                                      f"warnings right %{{y:.1%}} of the time<extra></extra>")
    base = hm.loc["Never warn", "PR_AUC"]
    fig.add_scatter(x=[0, 1], y=[base, base], name=f"No skill: {base:.3f}",
                    line=dict(color=C_NONE, width=2, dash="dot"),
                    hovertemplate="No skill: %{y:.1%}<extra></extra>")
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_yaxes(range=[0, 0.5], tickformat=".0%")
    show(style_fig(fig, height=460, x_title="Recall: share of real 10% drops that got a warning",
                   y_title="Precision: share of warnings that were right"))

    st.markdown("#### Yes/no warnings: precision, recall, F1, accuracy")
    tbl = hm.reset_index().rename(columns={"method": "Method", "PR_AUC": "PR-AUC",
                                           "precision": "Precision", "recall": "Recall",
                                           "warn_rate": "Share of days warned",
                                           "accuracy": "Accuracy"})
    tbl["Method"] = tbl["Method"].map(lambda k: METHOD_STYLE.get(k, (k,))[0])
    st.dataframe(tbl, hide_index=True, column_config={
        c: st.column_config.NumberColumn(format="%.3f") for c in
        ["PR-AUC", "Precision", "Recall", "F1", "Accuracy", "Share of days warned"]})
    st.warning(
        f"**Why not accuracy?** Only about 12% of days are followed by a 10% drop. A rule that "
        f"**never warns** gets the best accuracy ({hm.loc['Never warn', 'accuracy']:.0%}) but "
        f"catches **zero** drops. That is why we report precision and recall. "
        f"Thresholds for all yes/no rules were fixed on 2019–20 data, never on test data.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Year by year")
        yr = load_csv("holdout_by_year.csv")
        fig = go.Figure()
        fig.add_bar(x=yr["year"], y=yr["v3"], name="Model v3 (ours)", marker_color=C_V3,
                    hovertemplate="%{x}: v3 %{y:.3f}<extra></extra>")
        fig.add_bar(x=yr["year"], y=yr["volatility_rule"], name="Volatility rule (simple)",
                    marker_color=C_VOL, hovertemplate="%{x}: rule %{y:.3f}<extra></extra>")
        fig.update_layout(barmode="group")
        show(style_fig(fig, height=360, y_title="PR-AUC (higher is better)"))
        won = int((yr["v3"] > yr["volatility_rule"]).sum())
        st.caption(f"v3 won {won} of {len(yr)} years (2021 lost by 0.001).")
    with c2:
        st.markdown("#### How sure are we?")
        st.markdown(
            f"We re-ran the comparison **{boot['n_boot']:,} times** on resampled data, "
            f"drawing whole months at a time ({boot['n_months']} months) so that bad weeks "
            f"stay together. v3 beat the volatility rule in **{pct(boot['win_share'])}** of "
            f"them. Average gap **{boot['gap_mean']:+.3f}** PR-AUC, 95% range "
            f"**{boot['ci_low']:+.3f} to {boot['ci_high']:+.3f}** (above zero).")
        st.markdown(
            "**Honest note:** v3 beat the rule on only **21 of the 49 stocks** individually. "
            "Its edge comes from ranking risk *across* stocks and dates (which stock is riskier "
            "today, which periods are riskier), not from being better on every single stock.")

    holdout2_section()

    st.markdown("#### The full story: v1 → v2 → v3 → (no v4)")
    story = pd.DataFrame([
        ["v1", "LightGBM on 16 price features, trained 2010–18, tuned on 2019–20",
         "Tied the rule on the 2021–25 test: 0.192 vs 0.191"],
        ["v2", "v1 retrained on 2010–20 (decided after seeing v1)",
         "Worse: 0.182. Raw feature levels shift between market regimes"],
        ["v3", "Volatility rule + small learned corrections; volatility-scaled and same-day "
               "ranked features; chosen among 7 candidates on 2019–20 only",
         "Validation 0.251 vs 0.218. Fresh holdout 0.233 vs 0.216 (main result)"],
        ["v4?", "4 pre-registered candidates using high/low-range and faster volatility "
                "estimates, judged on 2019–20 only",
         "None beat v3 on validation (best 0.241 vs 0.251), so no v4. v3 then won a second "
         "fresh test: 0.183 vs 0.169 on 95 stocks"],
    ], columns=["Version", "What", "Result"])
    st.markdown("| Version | What | Result |\n|---|---|---|\n" + "\n".join(
        f"| **{v}** | {w} | {r} |" for v, w, r in story.itertuples(index=False)))
    mt = load_csv("metrics_test.csv")
    with st.expander("Original 99-stock test set, 2021–25 (looked at three times; disclosed)"):
        st.markdown("v3 on this set: **0.204** vs volatility rule **0.191**. Because this set was "
                    "already used to judge v1 and v2, the fresh holdout above is the fair test.")
        mt = mt.rename(columns={"method": "Method", "PR_AUC": "PR-AUC"})
        st.dataframe(mt, hide_index=True, column_config={
            c: st.column_config.NumberColumn(format="%.3f") for c in mt.columns if c != "Method"})
        if (RES / "pr_curve.png").exists():
            st.image(str(RES / "pr_curve.png"), caption="Original test PR curve (v1, v2, baselines)")

    backtest_section()
    footer()


def backtest_section():
    summary = load_csv("backtest_summary.csv")
    curves = load_csv("backtest_curves.csv")
    if summary is None or curves is None:
        return
    st.markdown("#### Stress test: what if you sold every time a warning came on?")
    st.caption("v3 is built to estimate risk, not to time trades. This test is included for "
               "honesty and is not part of the main result.")
    st.markdown(
        "An equal-weight portfolio that holds each stock, but moves that stock to cash (0% "
        "return) while its warning is on, and buys back when it switches off. Warnings use the "
        "thresholds fixed on 2019–20. No trading costs, no taxes, no tuning. "
        "**This is a stress test of the warnings, not a trading strategy.**")
    names = {"v3": ("Model v3 (ours)", C_V3, "solid"),
             "vol_rule": ("Volatility rule (simple)", C_VOL, "solid"),
             "ma200": ("200-day average rule", C_MA, "solid"),
             "buy_hold": ("Buy and hold", "#f0efec", "solid")}
    universes = list(dict.fromkeys(summary["universe"]))
    tabs = st.tabs([u for u in universes])
    for tab, u in zip(tabs, universes):
        with tab:
            cu = curves[curves["universe"] == u]
            fig = go.Figure()
            for k, (lab, col, dash) in names.items():
                c = cu[cu["strategy"] == k]
                fig.add_scatter(x=pd.to_datetime(c["date"]), y=c["value"], name=lab,
                                line=dict(color=col, width=2, dash=dash),
                                hovertemplate=f"{lab}<br>%{{x|%d %b %Y}}: $%{{y:.2f}}<extra></extra>")
            fig.update_layout(hovermode="x unified")
            show(style_fig(fig, height=400, y_title="Value of $1 invested"))
            s = summary[summary["universe"] == u].copy()
            s["strategy"] = s["strategy"].map(lambda k: names[k][0])
            s = s.rename(columns={"strategy": "Strategy", "total_return": "Total return",
                                  "cagr": "Per year", "max_drawdown": "Worst fall (max drawdown)",
                                  "volatility": "Yearly volatility",
                                  "time_in_cash": "Share of stock-days in cash"})
            show_cols = ["Strategy", "Total return", "Per year", "Worst fall (max drawdown)",
                         "Yearly volatility", "Share of stock-days in cash"]
            st.dataframe(s[show_cols], hide_index=True, column_config={
                c: st.column_config.NumberColumn(format="percent") for c in show_cols[1:]})
            b = summary[summary["universe"] == u].set_index("strategy")
            st.markdown(
                f"**What it shows:** stepping aside on v3 warnings cut the worst fall from "
                f"{pct(-b.loc['buy_hold', 'max_drawdown'])} to "
                f"{pct(-b.loc['v3', 'max_drawdown'])}, but the money ended up "
                f"{pct(b.loc['v3', 'total_return'])} higher instead of "
                f"{pct(b.loc['buy_hold', 'total_return'])} with simple buy-and-hold "
                f"(volatility rule: {pct(b.loc['vol_rule', 'total_return'])}). Missing the "
                f"rebounds after warnings costs a lot. The warnings are better used to "
                f"understand risk than to time the market.")
    note = RES / "backtest_note.txt"
    if note.exists():
        st.caption(note.read_text())


# ==== page 4: how it works ================================================================
def page_how():
    st.title("How it works")
    st.markdown(f"""
### The question we answer
For a stock on a given day: **will its closing price, at any point in the next 20 trading days
(about one month), be at least 10% below today's close?** We call that a "10% drop". On a
typical large US stock this happened on about **{pct(BASE)}** of days in 2021–25.

The app shows a **chance** (for example 18%), not a yes/no prediction. A higher chance means
drops like this have been more common in situations that looked like today.

### What is volatility?
Volatility measures how much a price jumps around from day to day. A stock that moves 3% a day
is far more likely to fall 10% in a month than one that moves 0.5% a day. This simple idea is
already a strong warning sign, so we use it as our main **baseline (the simple rule)**:
- *Volatility rule (chance)*: from the last 20 days' price swings, a standard formula gives the
  chance of a 10% fall; we adjust it to match how often falls really happened in 2010–18.
- *Volatility rule (yes/no)*: warn when the last 20 days' swings are bigger than on 80% of
  days in 2010–18.

We also compare with a **200-day average rule** (warn when the price is below its 200-day
average) and with **never warning**.

### Our model (v3): the simple rule plus learned corrections
v3 starts from the volatility rule and lets a small, deliberately restrained machine-learning
model (LightGBM, 54 tiny decision trees with 3 leaves each) nudge the chance up or down using
16 signals: recent returns, how far the stock is below its 1-year high, how it ranks against
other large US stocks today, trading volume, and market mood (VIX, S&P 500). On the **Stock
detail** page you can see the exact size of each nudge (these are exact for this kind of model).

Warning levels, all fixed using 2019–20 data only:
- ⛔ **Warning**: chance ≥ {pct(TH['red'], 1)}. At this level v3 warned on the same
  share of 2019–20 days as the simple yes/no volatility rule.
- ⚠️ **Watch**: chance ≥ {pct(TH['amber'], 1)} (twice that share of days).
- ✅ **Calm**: below that.

### How we measured it (and kept it honest)
- **Train on the past, test on the future:** trained on 2010–18, choices made on 2019–20
  (validation), tested on 2021–25. 20-trading-day gaps between periods so no answer leaks
  across.
- **Two fresh holdouts:** our main test uses 49 other large US stocks never used for
  anything before v3 was frozen. A second, pre-registered test on 95 more stocks confirmed
  the result (v3 0.183 vs rule 0.169, won 100% of bootstrap resamples and all 5 years).
- **Precision** = of the warnings given, the share followed by a real 10% drop.
  **Recall** = of the real 10% drops, the share that got a warning beforehand.
  **PR-AUC** summarises precision and recall over every possible warning level (higher is
  better; guessing scores about 0.12).
- **Not accuracy:** drops are rare, so "never warn" is 88% accurate and completely useless.
- Nothing was tuned on test or holdout results. Every threshold comes from 2019–20.

### Limitations (please read)
- **Survivorship bias:** we use today's large companies, which by definition survived. Some
  stocks (EA, IPG, BK) dropped out of the free data source. Real-world risk may be higher.
- **Overlapping labels:** neighbouring days share most of their 20-day future, so the
  number of truly independent examples is much smaller than the number of rows. We used a
  monthly block bootstrap to account for this.
- **False alarms cost something:** most warnings (about 80%) are *not* followed by a 10% drop.
  Acting on every warning can mean selling too early and missing gains.
- **Data snapshot, not live:** prices end on **{AS_OF}**. The chances describe that date.
- **Small edge:** v3 beats the simple rule by a small, statistically consistent margin, not
  by a lot. The simple rule alone already captures most of the signal.
- Large US stocks only; no news, earnings dates or fundamentals.

### Sources
Prices: Yahoo Finance via the `yfinance` library. Tools: LightGBM, scikit-learn, pandas,
Streamlit, Plotly. AI assistant: Claude (Anthropic) helped write code.
""")
    footer()


{"Check my stocks": page_check, "Stock detail": page_detail,
 "Does it beat a simple rule?": page_results, "How it works": page_how}[page]()
