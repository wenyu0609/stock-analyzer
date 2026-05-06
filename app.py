"""
Stock Analyzer Pro — Streamlit Web v3
Fixes: full theme CSS / mobile responsive / MACD-RSI collapsible+popout /
       smart chart zoom (scroll wheel + 2-finger pinch) / no accidental zoom on swipe
"""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings, io, time, re
from datetime import datetime

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None


# Optional PDF deps
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib import colors as rl_colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table as RLTable, TableStyle, Image as RLImage
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

L_ACCENT = "#3557b7"

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Stock Analyzer Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core import (
    run_analysis, TW_NAME_CACHE, _load_twse_bulk, _load_tpex_bulk,
    compute_drift_bias, simple_forecast, backtest_directional,
    simulate_macd_cross_strategy, recommend_top_volume_stocks,
    friendly_error_message, DEFAULT_WEIGHTS,
    _cjk, SKLEARN_OK, XGB_OK, TORCH_OK, YF_OK,
    get_intraday_analysis, read_marketdata_key, segment_to_zh,
)

# ── Session state ──────────────────────────────────────────────────────────
_DEF = dict(
    result=None, batch_results=[],
    watchlist=["2330","2454","0050","00878","2317"],
    weights=DEFAULT_WEIGHTS.copy(),
    forecast_days=30, lookback_years=3, train_ratio=0.8,
    chart_style="K棒", show_bb=True, show_sr=True, show_band=True,
    show_macd=True, show_rsi=True,
    show_ma5=True, show_ma20=True, show_ma60=True,
    chart_dragmode="pan", mobile_chart_mode=True, font_scale=100, chart_height_mode="自動",
    _show_reco=False,
    theme="暗色", names_loaded=False, _trigger=False, _pending_sym="",
    api_key="", intraday_auto_refresh=False,
    _last_params={},  # tracks params used for current result
)
for k,v in _DEF.items():
    if k not in st.session_state: st.session_state[k]=v

# Streamlit widgets are evaluated later in the script.  When the user changes
# the font slider, sync its widget key back into the canonical setting before
# the main CSS is generated, otherwise the visual update can lag by one run.
if "_font_scale_widget" in st.session_state:
    try:
        st.session_state.font_scale = int(st.session_state["_font_scale_widget"])
    except Exception:
        pass


WEIGHTS_VERSION = "2026-05-margin-multisource"
if st.session_state.get("_weights_version") != WEIGHTS_VERSION:
    st.session_state.weights = DEFAULT_WEIGHTS.copy()
    for _k in ("wt_s","wt_n","wm_s","wm_n","wn_s","wn_n","wf_s","wf_n","wu_s","wu_n","wi_s","wi_n","wg_s","wg_n"):
        st.session_state.pop(_k, None)
    st.session_state["_weights_version"] = WEIGHTS_VERSION

if not st.session_state.names_loaded:
    with st.spinner("載入股票名稱資料庫…"):
        _load_twse_bulk(); _load_tpex_bulk()
    st.session_state.names_loaded=True

# ── Theme colours ──────────────────────────────────────────────────────────
D   = st.session_state.theme == "暗色"
BG  = "#0f1120" if D else "#f0f4ff"
SB  = "#1a1d2e" if D else "#ffffff"
CD  = "#1e2240" if D else "#ffffff"
TX  = "#e2e4f0" if D else "#111827"
DM  = "#8890aa" if D else "#4b5563"
BD  = "#2d3154" if D else "#d1d5db"
AC  = "#6c8ef5" if D else "#3b5bdb"
OK  = "#34d399" if D else "#059669"
ER  = "#f87171" if D else "#dc2626"
WA  = "#fbbf24" if D else "#b45309"
CBG = "#14172a" if D else "#ffffff"
CPP = "#0f1120" if D else "#f0f4ff"
CGR = "#2d3154" if D else "#e5e7eb"
CTX = "#8890aa" if D else "#6b7280"

# ── User adjustable font size ──────────────────────────────────────────────
_FONT_SCALE = int(st.session_state.get("font_scale", 100))
_FONT_SCALE = max(85, min(130, _FONT_SCALE))
FS_BASE = round(14 * _FONT_SCALE / 100, 1)
FS_SMALL = round(11.5 * _FONT_SCALE / 100, 1)
FS_TINY = round(10.5 * _FONT_SCALE / 100, 1)
FS_H1 = round(20 * _FONT_SCALE / 100, 1)
FS_METRIC = round(17 * _FONT_SCALE / 100, 1)
FS_METRIC_MOBILE = round(15 * _FONT_SCALE / 100, 1)

# ── CSS ────────────────────────────────────────────────────────────────────
# Use string concatenation — avoids ALL f-string brace conflicts
_CSS_RULES = """
/* ================================================================
   EXPANDER ICON FIX
   Streamlit expander summary structure:
   <summary>
     <span class="st-emotion-cache-XXX">  <- THIS is the icon span
       keyboard_double_arrow_right         <- ligature text
     </span>
     <p>Label text</p>                     <- this is the label
   </summary>
   
   Problem: our `span {{ font-family: CJK }}` overrides the icon span's
   Material Icons font, breaking the ligature -> raw text shows.
   
   Fix: hide the icon span entirely via details>summary>span:first-child
   AND protect icon spans from our font override.
================================================================ */

/* Step 1: Fix expander icon — restore Material Icons font on icon span
   Root cause: our global span font-family override breaks Material Icons ligatures
   Fix: explicitly restore Material Icons font on the summary spans */
details summary {{
    list-style: none !important;
}}
details summary::-webkit-details-marker {{
    display: none !important;
}}
/* Restore Material Icons font on ALL spans inside summary
   so "keyboard_double_arrow_right" renders as an icon not text */
details summary > span {{
    font-family: 'Material Icons', 'Material Icons Round',
                 'Material Symbols Rounded', serif !important;
    font-feature-settings: "liga" 1 !important;
    -webkit-font-feature-settings: "liga" 1 !important;
    text-rendering: optimizeLegibility !important;
}}
/* The label text is in a <p> inside a <div> after the icon span — keep visible */
details summary div,
details summary p {{
    font-family: 'Microsoft JhengHei','PingFang TC','Noto Sans TC',sans-serif !important;
    color: {TX} !important;
    display: inline !important;
}}

/* Step 2: App background */
[data-testid="stAppViewContainer"] {{ background:{BG} !important; }}
[data-testid="stSidebar"] {{ background:{SB} !important; }}
[data-testid="stHeader"] {{ background:{BG} !important; }}
section[data-testid="stSidebar"] > div {{ background:{SB} !important; }}

/* Step 3: Typography - CRITICAL: do NOT apply to icon spans */
html, body, p, label, input, textarea, select,
button, h1, h2, h3, h4, h5, h6, li, a, td, th, caption {{
    font-family: 'Microsoft JhengHei','PingFang TC','Noto Sans TC',sans-serif !important;
    font-size: {FS_BASE}px !important;
}}
/* Only apply CJK font to divs/spans that are NOT icon containers */
div:not([class*="material"]) {{
    font-family: 'Microsoft JhengHei','PingFang TC','Noto Sans TC',sans-serif !important;
}}
/* Explicitly exclude icon spans from any font override */
details summary > span,
details summary > div > span {{
    font-family: 'Material Icons','Material Icons Round','Material Symbols Rounded',sans-serif !important;
}}

/* Step 4: All text colours */
body, p, div, label, li, td, th, caption, h2, h3, h4 {{
    color: {TX} !important;
    font-size: {FS_BASE}px !important;
}}
span {{ color: {TX} !important; }}
h1 {{ color: {AC} !important; font-size:{FS_H1}px !important; font-weight:700 !important; }}
[data-testid="stSidebarContent"] * {{ color: {TX} !important; }}
[data-testid="stSidebarContent"] h4 {{ color: {AC} !important; }}

/* Step 5: Buttons */
.stButton > button {{
    background:{CD} !important; border:1px solid {BD} !important;
    color:{TX} !important; border-radius:8px !important; transition:all .15s !important;
}}
.stButton > button:hover {{ border-color:{AC} !important; color:{AC} !important; }}
.stButton > button[kind="primary"] {{
    background:{AC} !important; border-color:{AC} !important; color:#fff !important;
}}
.stButton > button[kind="primary"]:hover {{ opacity:.9 !important; }}

/* Step 6: Inputs */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {{
    background:{CD} !important; border:1px solid {BD} !important;
    color:{TX} !important; border-radius:8px !important;
}}
.stTextInput > div > div > input::placeholder {{ color:{DM} !important; }}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {{ border-color:{AC} !important; }}

/* Step 7: Selectbox */
.stSelectbox > div > div {{
    background:{CD} !important; border:1px solid {BD} !important;
    color:{TX} !important; border-radius:8px !important;
}}
[data-baseweb="select"] > div {{ background:{CD} !important; color:{TX} !important; }}
[data-baseweb="select"] span {{ color:{TX} !important; }}
[role="option"] {{ background:{CD} !important; color:{TX} !important; }}
[role="option"]:hover {{ background:{BD} !important; }}
[data-baseweb="popover"],[data-baseweb="menu"] {{ background:{CD} !important; }}
[data-baseweb="menu"] li {{ color:{TX} !important; background:{CD} !important; }}
[data-baseweb="menu"] li:hover {{ background:{BD} !important; }}

/* Step 8: Sliders, Checkboxes, Radio */
.stSlider label, .stSlider > div {{ color:{TX} !important; }}
.stCheckbox > label, .stCheckbox label {{ color:{TX} !important; }}
.stRadio > div > label, .stRadio label {{ color:{TX} !important; }}

/* Step 9: Metrics */
[data-testid="stMetricValue"] {{ color:{TX} !important; font-size:{FS_METRIC}px !important; }}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] div {{ color:{DM} !important; }}
[data-testid="stMetricDelta"] {{ color:{DM} !important; }}
[data-testid="stMetric"] {{
    background:{CD}; border:1px solid {BD}; border-radius:10px; padding:10px;
}}

/* Step 10: Expander content */
[data-testid="stExpander"] {{
    background:{CD} !important; border:1px solid {BD} !important; border-radius:8px !important;
}}
[data-testid="stExpander"] summary {{
    color:{TX} !important; background:{CD} !important;
}}
[data-testid="stExpander"] summary p {{ color:{TX} !important; }}
[data-testid="stExpander"] > div {{ background:{CD} !important; }}
[data-testid="stExpander"] > div > div {{ background:{CD} !important; }}
[data-testid="stExpander"] label {{ color:{TX} !important; }}
[data-testid="stExpander"] span {{ color:{TX} !important; }}
[data-testid="stExpander"] p {{ color:{TX} !important; }}
[data-testid="stExpander"] .stCheckbox label {{ color:{TX} !important; }}
[data-testid="stExpander"] .stRadio label {{ color:{TX} !important; }}
[data-testid="stExpander"] .stSlider label {{ color:{TX} !important; }}
[data-testid="stExpander"] input {{ background:{BG} !important; color:{TX} !important; }}
[data-testid="stCaptionContainer"] p {{ color:{DM} !important; }}
[data-testid="stCaptionContainer"] {{ color:{DM} !important; }}

/* Step 11: Tabs */
[data-baseweb="tab"] span {{ color:{DM} !important; }}
[aria-selected="true"] span {{ color:{AC} !important; }}
[data-baseweb="tab-highlight"] {{ background:{AC} !important; }}
[data-baseweb="tab-border"] {{ background:{BD} !important; }}

/* Step 12: Progress, Download, Divider */
[data-testid="stProgressBar"] > div {{ background:{AC} !important; }}
[data-testid="stProgressBar"] {{ background:{BD} !important; }}
[data-testid="stDownloadButton"] button {{
    background:{CD} !important; border:1px solid {BD} !important;
    color:{TX} !important; border-radius:8px !important;
}}
hr {{ border-top:1px solid {BD} !important; }}

/* Step 13: Custom cards & badges */
.metric-card {{
    background:{CD}; border:1px solid {BD};
    border-radius:10px; padding:12px 16px; margin-bottom:8px;
}}
.metric-card .lbl {{ color:{DM}; font-size:{FS_TINY}px; margin-bottom:2px; }}
.metric-card .val {{ color:{TX}; font-size:{FS_METRIC}px; font-weight:700; }}
.metric-card .sub {{ color:{AC}; font-size:{FS_SMALL}px; }}
.bull {{
    background:{BULL_BG}; color:{OK};
    padding:3px 12px; border-radius:20px; font-weight:700; display:inline-block;
}}
.bear {{
    background:{BEAR_BG}; color:{ER};
    padding:3px 12px; border-radius:20px; font-weight:700; display:inline-block;
}}
.neut {{
    background:{NEUT_BG}; color:{WA};
    padding:3px 12px; border-radius:20px; font-weight:700; display:inline-block;
}}
.bhi {{
    background:{BHI_BG}; color:{AC};
    padding:2px 10px; border-radius:20px; display:inline-block;
}}
.blo {{
    background:{BEAR_BG}; color:{ER};
    padding:2px 10px; border-radius:20px; display:inline-block;
}}
.ni {{ border-left:3px solid {BD}; padding:5px 10px; margin:5px 0; font-size:{FS_SMALL}px; color:{DM}; }}
.ni a {{ color:{DM}; text-decoration:none; }}
.ni:hover {{ border-color:{AC}; }}

/* Step 14: Sidebar dropdown complete override */
[data-testid="stSidebar"] [data-testid="stExpander"] {{
    background:{CD} !important; border-color:{BD} !important;
}}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {{
    background:{CD} !important; color:{TX} !important;
}}
[data-testid="stSidebar"] [data-testid="stExpander"] > div {{
    background:{CD} !important;
}}
[data-testid="stSidebar"] [data-testid="stExpander"] label,
[data-testid="stSidebar"] [data-testid="stExpander"] span,
[data-testid="stSidebar"] [data-testid="stExpander"] p {{
    color:{TX} !important;
}}
[data-testid="stSidebar"] input {{
    background:{CD} !important; color:{TX} !important; border-color:{BD} !important;
}}
[data-testid="stSidebar"] [data-baseweb="select"] > div {{
    background:{CD} !important; color:{TX} !important;
}}
[data-testid="stSidebar"] [data-baseweb="select"] span {{ color:{TX} !important; }}
[data-testid="stSidebar"] [role="option"] {{
    background:{CD} !important; color:{TX} !important;
}}

/* Step 15: Mobile */
@media (max-width:768px) {{
    .block-container {{ padding-left:.55rem !important; padding-right:.55rem !important; padding-top:.55rem !important; }}
    h1 {{ font-size:{FS_H1}px !important; }}
    .metric-card {{ padding:10px 11px !important; margin-bottom:7px !important; }}
    .metric-card .val {{ font-size:{FS_METRIC_MOBILE}px !important; }}
    [data-testid="stMetricValue"] {{ font-size:{FS_METRIC_MOBILE}px !important; }}
    [data-testid="stHorizontalBlock"] {{ gap:.45rem !important; }}
    .stPlotlyChart {{ margin-left:-.25rem !important; margin-right:-.25rem !important; }}
}}
/* Mobile chart: one-finger vertical page scroll, two-finger chart pinch zoom.
   The actual chart pinch is handled by injected JS below because Plotly inside
   Streamlit does not consistently receive native pinch gestures on mobile. */
.stPlotlyChart, .js-plotly-plot, .plot-container, .svg-container {{
    overscroll-behavior: contain !important;
}}
body.chart-pan-on .stPlotlyChart,
body.chart-pan-on .js-plotly-plot,
body.chart-pan-on .plot-container,
body.chart-pan-on .svg-container {{
    touch-action: pan-y !important;
}}
body.chart-zoom-on .stPlotlyChart,
body.chart-zoom-on .js-plotly-plot,
body.chart-zoom-on .plot-container,
body.chart-zoom-on .svg-container {{
    touch-action: none !important;
}}
.pinch-hint {{
    background:{CD}; border:1px solid {BD}; border-radius:10px;
    padding:8px 10px; margin:4px 0 8px 0; color:{DM};
    font-size:{FS_SMALL}px;
}}
"""

# Inject Material Icons font directly in HTML (more reliable than @import in some environments)
st.markdown("""
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
<link href="https://fonts.googleapis.com/icon?family=Material+Icons+Round" rel="stylesheet">
<link href="https://fonts.googleapis.com/icon?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet">
""", unsafe_allow_html=True)

# Compute badge backgrounds once
BULL_BG = "rgba(5,150,105,0.12)"  if D else "#dcfce7"
BEAR_BG = "rgba(239,68,68,0.12)"  if D else "#fee2e2"
NEUT_BG = "rgba(245,158,11,0.12)" if D else "#fef9c3"
BHI_BG  = "rgba(108,142,245,0.12)"if D else "#ede9fe"

st.markdown(
    "<style>" + _CSS_RULES.format(
        BG=BG, SB=SB, CD=CD, TX=TX, DM=DM, BD=BD,
        AC=AC, OK=OK, ER=ER, WA=WA,
        CBG=CBG, CPP=CPP, CGR=CGR, CTX=CTX,
        BULL_BG=BULL_BG, BEAR_BG=BEAR_BG, NEUT_BG=NEUT_BG, BHI_BG=BHI_BG,
        FS_BASE=FS_BASE, FS_SMALL=FS_SMALL, FS_TINY=FS_TINY, FS_H1=FS_H1,
        FS_METRIC=FS_METRIC, FS_METRIC_MOBILE=FS_METRIC_MOBILE,
    ) + "</style>",
    unsafe_allow_html=True
)

# Toggle parent CSS class for mobile chart gestures and install a custom
# two-finger pinch handler.  Plotly's native pinch zoom is unreliable inside
# Streamlit's mobile iframe, so this converts two-finger distance changes into
# Plotly.relayout() calls. One-finger vertical scrolling is left to the page.
_chart_mode_cls = "chart-zoom-on" if (st.session_state.get("mobile_chart_mode", True) and st.session_state.get("chart_dragmode", "pan") == "zoom") else "chart-pan-on"
components.html(f"""
<script>
(function(){{
  const doc = window.parent.document;
  const body = doc.body;
  body.classList.remove("chart-zoom-on", "chart-pan-on");
  body.classList.add("{_chart_mode_cls}");

  function numRange(v) {{
    if (v === undefined || v === null) return null;
    if (typeof v === 'number') return v;
    const t = Date.parse(v);
    return Number.isFinite(t) ? t : Number(v);
  }}
  function outRange(v, isDate) {{
    return isDate ? new Date(v).toISOString() : v;
  }}
  function dist(t1, t2) {{
    const dx = t1.clientX - t2.clientX;
    const dy = t1.clientY - t2.clientY;
    return Math.sqrt(dx*dx + dy*dy);
  }}
  function installPinch(gd) {{
    if (!gd || gd.__saPinchInstalled) return;
    gd.__saPinchInstalled = true;
    let pinch = null;

    gd.addEventListener('touchstart', function(e) {{
      if (e.touches && e.touches.length === 2) {{
        const fl = gd._fullLayout;
        if (!fl || !fl.xaxis || !fl.yaxis) return;
        const xr0 = fl.xaxis.range || fl.xaxis._range;
        const yr0 = fl.yaxis.range || fl.yaxis._range;
        if (!xr0 || !yr0) return;
        const x0 = numRange(xr0[0]), x1 = numRange(xr0[1]);
        const y0 = numRange(yr0[0]), y1 = numRange(yr0[1]);
        if (![x0,x1,y0,y1].every(Number.isFinite)) return;
        pinch = {{
          d0: Math.max(dist(e.touches[0], e.touches[1]), 1),
          x0, x1, y0, y1,
          xDate: isNaN(Number(xr0[0])) || isNaN(Number(xr0[1])),
          lastTs: 0
        }};
      }}
    }}, {{passive:false}});

    gd.addEventListener('touchmove', function(e) {{
      if (!pinch || !e.touches || e.touches.length !== 2) return;
      e.preventDefault();
      const now = Date.now();
      if (now - pinch.lastTs < 24) return;   // throttle, keeps mobile smooth
      pinch.lastTs = now;
      const d = Math.max(dist(e.touches[0], e.touches[1]), 1);
      let scale = pinch.d0 / d;              // fingers apart => smaller range
      scale = Math.max(0.18, Math.min(4.5, scale));
      const xc = (pinch.x0 + pinch.x1) / 2;
      const yc = (pinch.y0 + pinch.y1) / 2;
      const xHalf = (pinch.x1 - pinch.x0) * scale / 2;
      const yHalf = (pinch.y1 - pinch.y0) * scale / 2;
      const update = {{
        'xaxis.range': [outRange(xc - xHalf, pinch.xDate), outRange(xc + xHalf, pinch.xDate)],
        'yaxis.range': [yc - yHalf, yc + yHalf]
      }};
      if (window.parent.Plotly) window.parent.Plotly.relayout(gd, update);
      else if (window.Plotly) window.Plotly.relayout(gd, update);
    }}, {{passive:false}});

    gd.addEventListener('touchend', function(e) {{
      if (!e.touches || e.touches.length < 2) pinch = null;
    }}, {{passive:false}});
  }}

  function scan() {{
    doc.querySelectorAll('.js-plotly-plot').forEach(installPinch);
  }}
  scan();
  setTimeout(scan, 500);
  setTimeout(scan, 1500);
  if (!window.__saPinchObserver) {{
    window.__saPinchObserver = new MutationObserver(scan);
    window.__saPinchObserver.observe(doc.body, {{childList:true, subtree:true}});
  }}
}})();
</script>
""", height=0)

# ── Mobile chart scroll fix (JavaScript) ─────────────────────────────────
# Inject JS to prevent chart from hijacking mobile scroll


# ── Helpers ────────────────────────────────────────────────────────────────
def badge(cls, t): return f"<span class='{cls}'>{t}</span>"

def _plotly_pan_config():
    return {
        "displaylogo": False,
        "responsive": True,
        "displayModeBar": True,
        "scrollZoom": True,
        "doubleClick": "reset",
        "modeBarButtonsToRemove": ["autoScale2d","lasso2d","select2d","toImage"],
        "modeBarButtonsToAdd": ["zoom2d","pan2d","resetScale2d"],
    }

def _return_model_label(name: str) -> str:
    return {
        "baseline": "基準報酬率模型",
        "return_global": "全市場報酬率模型",
        "return_segment_blend": "分段融合報酬率模型",
    }.get(str(name or ""), str(name or "—"))

def _segment_label(value: str) -> str:
    text = str(value or "").strip()
    if not text or text == "—":
        return "—"
    if "_" not in text:
        return text
    try:
        return segment_to_zh(text)
    except Exception:
        phase_map = {"early":"月初", "mid":"月中", "late":"月末"}
        trend_map = {"up":"多頭", "down":"空頭", "side":"盤整"}
        vol_map = {"normalvol":"正常波動", "highvol":"高波動"}
        p = text.split("_")
        if len(p) >= 3:
            return f"{phase_map.get(p[0],p[0])}｜{trend_map.get(p[1],p[1])}｜{vol_map.get(p[2],p[2])}"
        return text

def _replace_segment_codes(text: str) -> str:
    def repl(m):
        return _segment_label(m.group(0))
    return re.sub(r"\b(?:early|mid|late)_(?:up|down|side)_(?:normalvol|highvol)\b", repl, str(text or ""))

def _streamlit_secret_key() -> str:
    try:
        return (
            st.secrets.get("FUGLE_API_KEY", "")
            or st.secrets.get("FUBON_MARKETDATA_API_KEY", "")
        )
    except Exception:
        return ""

@st.cache_data(ttl=10, show_spinner=False)
def _cached_intraday(symbol: str, history: pd.DataFrame):
    return get_intraday_analysis(symbol, history)

def calc_score(r):
    ind=r["indicators"]; ml=r.get("ml_predict",{}); ns=r.get("news_sentiment",{})
    fund=r.get("fundamental",{}); w={**DEFAULT_WEIGHTS, **r.get("weights",st.session_state.weights)}
    wt=w.get("technical",DEFAULT_WEIGHTS["technical"])/100; wm=w.get("ml",DEFAULT_WEIGHTS["ml"])/100
    wn=w.get("news",DEFAULT_WEIGHTS["news"])/100;      wf=w.get("fundamental",DEFAULT_WEIGHTS["fundamental"])/100
    wu=w.get("us_market",DEFAULT_WEIGHTS["us_market"])/100; wi=w.get("institutional",DEFAULT_WEIGHTS["institutional"])/100
    wg=w.get("margin",DEFAULT_WEIGHTS["margin"])/100
    try:
        mh=float(ind["macd_hist"].iloc[-1]); rsi=float(ind["rsi14"].iloc[-1])
        ms=float(ind["macd_hist"].std()) if len(ind["macd_hist"])>5 else 1.0
        t=float(np.clip(mh/max(ms,1e-9),-1,1))*.55+float(np.clip((rsi-50)/50,-1,1))*.25
    except Exception:
        t=0.0
    ml_s=(ml.get("prob_up",.5)-.5)*2
    ns_s=ns.get("score", None)
    if ns_s is None:
        ns_s=.6 if ns.get("label") in ("正面","偏正面") else(-.6 if ns.get("label") in ("負面","偏負面") else 0.)
    fs=0.
    if fund.get("pe_ratio") and 0<fund["pe_ratio"]<15: fs+=.5
    if fund.get("pe_ratio") and fund["pe_ratio"]>40:   fs-=.5
    if fund.get("roe") and fund["roe"]>.15: fs+=.5
    mkt=r.get("mkt_ctx",{})
    us_s=float(np.clip(mkt.get("nasdaq_ret_1",0)/.025,-1,1)*.3+
               np.clip(mkt.get("sp500_ret_1",0)/.02,-1,1)*.2+
               np.clip(mkt.get("semis_ret_1",0)/.03,-1,1)*.35+
               np.clip(mkt.get("sox_ret_1",0)/.03,-1,1)*.15)
    inst_s=r.get("institutional",{}).get("inst_score",0.0)
    margin_s=r.get("margin",{}).get("margin_score",0.0)
    ret_s=float(np.clip(r.get("return_prediction",{}).get("expected_return",0.0)/0.12,-1,1))
    return float(np.clip(
        t*wt*1.8+(ml_s*0.7+ret_s*0.3)*wm*2+ns_s*wn*2+fs*wf*2+us_s*wu*2+inst_s*wi*2+margin_s*wg*2,
        -4,4))

def action_badge(s):
    if s>=1.5:   return badge("bull","偏多·可試單")
    if s>=0.5:   return badge("bull","偏多觀察")
    if s<=-1.5:  return badge("bear","偏空·減碼")
    if s<=-0.5:  return badge("bear","偏空觀察")
    return badge("neut","中性觀望")

# ── Chart builder ──────────────────────────────────────────────────────────
def build_chart(r, style="K棒", bb=True, sr_on=True, band=True,
                show_macd=True, show_rsi=True,
                show_ma5=True, show_ma20=True, show_ma60=True,
                dragmode="zoom"):
    df=r["df"]; ind=r["indicators"]; sr=r["sr"]; fc=r["forecast"]
    name=r["name"]; sym=r["symbol"]

    # Dynamic row layout based on which subcharts are visible
    if show_macd and show_rsi:
        rows=3; heights=[.62,.19,.19]
        titles=[f"{name} ({sym.split('.')[0]})","MACD","RSI(14)"]
    elif show_macd:
        rows=2; heights=[.72,.28]
        titles=[f"{name} ({sym.split('.')[0]})","MACD"]
    elif show_rsi:
        rows=2; heights=[.72,.28]
        titles=[f"{name} ({sym.split('.')[0]})","RSI(14)"]
    else:
        rows=1; heights=[1.0]
        titles=[f"{name} ({sym.split('.')[0]})"]

    fig=make_subplots(rows=rows,cols=1,shared_xaxes=True,
        row_heights=heights,vertical_spacing=.02,subplot_titles=titles)

    # ── Main price chart ──
    if style=="K棒":
        fig.add_trace(go.Candlestick(
            x=df.index,open=df["Open"],high=df["High"],
            low=df["Low"],close=df["Close"],
            increasing_line_color=OK,decreasing_line_color=ER,
            name="K棒",showlegend=False),row=1,col=1)
    else:
        fig.add_trace(go.Scatter(x=df.index,y=df["Close"],mode="lines",
            line=dict(color=AC,width=1.5),name="收盤"),row=1,col=1)

    # MA lines (visibility controlled)
    if show_ma5:
        fig.add_trace(go.Scatter(x=df.index,y=ind["ma5"],
            line=dict(color="rgba(108,142,245,0.8)",width=1),name="MA5"),row=1,col=1)
    if show_ma20:
        fig.add_trace(go.Scatter(x=df.index,y=ind["ma20"],
            line=dict(color="rgba(251,191,36,0.8)",width=1),name="MA20"),row=1,col=1)
    if show_ma60:
        fig.add_trace(go.Scatter(x=df.index,y=ind["ma60"],
            line=dict(color="rgba(248,113,113,0.8)",width=1),name="MA60"),row=1,col=1)

    if bb:
        fig.add_trace(go.Scatter(x=df.index,y=ind["bb_up"],
            line=dict(color="rgba(108,142,245,0.3)",width=1),name="BB上"),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=ind["bb_dn"],
            line=dict(color="rgba(108,142,245,0.3)",width=1),fill="tonexty",
            fillcolor="rgba(108,142,245,0.06)",name="BB下"),row=1,col=1)

    if sr_on:
        fig.add_hline(y=sr["resistance_hi"],line_dash="dot",
            line_color="rgba(248,113,113,0.5)",row=1,col=1,
            annotation_text="壓力",annotation_font_color=ER)
        fig.add_hline(y=sr["support_lo"],line_dash="dot",
            line_color="rgba(52,211,153,0.5)",row=1,col=1,
            annotation_text="支撐",annotation_font_color=OK)

    fd=list(fc["future_dates"])
    if band:
        fig.add_trace(go.Scatter(x=fd,y=fc["upper"],
            line=dict(color="rgba(0,0,0,0)",width=0),showlegend=False),row=1,col=1)
        fig.add_trace(go.Scatter(x=fd,y=fc["lower"],
            line=dict(color="rgba(0,0,0,0)",width=0),fill="tonexty",
            fillcolor="rgba(167,139,250,0.12)",name="預測帶"),row=1,col=1)
    fig.add_trace(go.Scatter(
        x=[df.index[-1]]+fd,
        y=[float(df["Close"].iloc[-1])]+list(fc["median"]),
        mode="lines",line=dict(color="#a78bfa",width=2,dash="dash"),
        name="預測中位"),row=1,col=1)

    # ── MACD ──
    macd_row=None
    if show_macd:
        macd_row=2
        hc=[OK if float(v)>=0 else ER for v in ind["macd_hist"]]
        fig.add_trace(go.Bar(x=df.index,y=ind["macd_hist"],marker_color=hc,
            name="Hist",showlegend=False,opacity=0.8),row=macd_row,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=ind["macd_line"],
            line=dict(color=AC,width=1.2),name="MACD"),row=macd_row,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=ind["macd_signal"],
            line=dict(color=WA,width=1.2),name="Signal"),row=macd_row,col=1)

    # ── RSI ──
    rsi_row=None
    if show_rsi:
        rsi_row=3 if show_macd else 2
        fig.add_trace(go.Scatter(x=df.index,y=ind["rsi14"],
            line=dict(color=OK,width=1.5),name="RSI(14)"),row=rsi_row,col=1)
        fig.add_hline(y=70,line_dash="dot",line_color="rgba(248,113,113,0.4)",
            row=rsi_row,col=1)
        fig.add_hline(y=50,line_dash="dot",line_color="rgba(100,100,100,0.3)",
            row=rsi_row,col=1)
        fig.add_hline(y=30,line_dash="dot",line_color="rgba(52,211,153,0.4)",
            row=rsi_row,col=1)

    # ── Layout ──
    chart_height = 560 if rows==3 else (430 if rows==2 else 380)
    if st.session_state.get("mobile_chart_mode", True):
        chart_height = 520 if rows==3 else (400 if rows==2 else 360)
    fig.update_layout(
        height=chart_height,
        paper_bgcolor=CPP, plot_bgcolor=CBG,
        font=dict(color=CTX,size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)",bordercolor=BD,
            font=dict(size=10,color=CTX),x=0.01,y=0.99),
        xaxis_rangeslider_visible=False,
        margin=dict(l=0,r=0,t=30,b=0),
        hovermode="x unified",
        # Mobile/desktop friendly: user can switch between zoom and pan
        dragmode=dragmode,
    )
    for i in range(1,rows+1):
        fig.update_xaxes(gridcolor=CGR,row=i,col=1,
            showspikes=True,spikecolor=DM,spikethickness=1)
        fig.update_yaxes(gridcolor=CGR,row=i,col=1)

    return fig

# ── Subchart popout (renders in a wide expander styled like a modal) ───────
def render_subchart_popout(r, chart_type="MACD"):
    """Render MACD or RSI as a standalone expanded chart."""
    df=r["df"]; ind=r["indicators"]
    fig=go.Figure()
    if chart_type=="MACD":
        hc=[OK if float(v)>=0 else ER for v in ind["macd_hist"]]
        fig.add_trace(go.Bar(x=df.index,y=ind["macd_hist"],marker_color=hc,
            name="Histogram",opacity=0.8))
        fig.add_trace(go.Scatter(x=df.index,y=ind["macd_line"],
            line=dict(color=AC,width=1.5),name="MACD Line"))
        fig.add_trace(go.Scatter(x=df.index,y=ind["macd_signal"],
            line=dict(color=WA,width=1.5),name="Signal"))
        title="MACD 指標詳細圖"
        last_h=float(ind["macd_hist"].iloc[-1])
        last_m=float(ind["macd_line"].iloc[-1])
        last_s=float(ind["macd_signal"].iloc[-1])
        fig.add_annotation(text=f"Hist:{last_h:+.3f}  MACD:{last_m:.3f}  Signal:{last_s:.3f}",
            xref="paper",yref="paper",x=0,y=1.08,showarrow=False,
            font=dict(size=12,color=CTX))
    else:
        fig.add_trace(go.Scatter(x=df.index,y=ind["rsi14"],
            line=dict(color=OK,width=2),name="RSI(14)",fill="tozeroy",
            fillcolor="rgba(52,211,153,0.06)"))
        for lvl,col,lbl in [(70,"rgba(248,113,113,0.5)","超買"),
                            (50,"rgba(100,100,100,0.3)","中性"),
                            (30,"rgba(52,211,153,0.5)","超賣")]:
            fig.add_hline(y=lvl,line_dash="dot",line_color=col,
                annotation_text=lbl,annotation_font_color=CTX)
        title="RSI(14) 指標詳細圖"
        last_r=float(ind["rsi14"].iloc[-1])
        lbl="超買" if last_r>=70 else ("超賣" if last_r<=30 else "中性")
        fig.add_annotation(text=f"RSI: {last_r:.1f}  ({lbl})",
            xref="paper",yref="paper",x=0,y=1.08,showarrow=False,
            font=dict(size=12,color=CTX))

    fig.update_layout(height=320,paper_bgcolor=CPP,plot_bgcolor=CBG,
        font=dict(color=CTX,size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10,color=CTX)),
        margin=dict(l=0,r=0,t=40,b=0),hovermode="x unified",
        dragmode="pan",
        xaxis=dict(gridcolor=CGR),yaxis=dict(gridcolor=CGR),
        title=dict(text=title,font=dict(color=TX,size=13)))
    st.plotly_chart(fig,use_container_width=True,
        config={"displaylogo":False,"responsive":True,"scrollZoom":True,"dragmode":"pan",
                "modeBarButtonsToRemove":["autoScale2d","lasso2d","select2d"]})

# ── Sidebar ────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        c1,c2=st.columns([4,1])
        c1.markdown(f"## 📈 Stock Analyzer")
        if c2.button("🌙" if D else "☀️",help="切換主題"):
            st.session_state.theme="淺色" if D else "暗色"; st.rerun()
        st.markdown(
            "<p style='color:"+DM+";font-size:.78rem;margin-top:-8px'>台灣股市智能分析</p>",
            unsafe_allow_html=True)
        with st.expander("外觀 / 手機顯示", expanded=False):
            _font_val = st.slider(
                "字體大小", 85, 130, int(st.session_state.font_scale), 5,
                key="_font_scale_widget",
                help="調整整個網頁的主要文字、卡片與表格字體大小。拖動後會立即套用。")
            st.session_state.font_scale = int(_font_val)
            # Late override: this is injected after the slider is evaluated, so the
            # visual font size updates on the same rerun instead of waiting one more click.
            _fs = max(85, min(130, int(_font_val))) / 100
            st.markdown(f"""
            <style>
            html, body, p, label, input, textarea, select, button, li, a, td, th, caption {{
                font-size: {14*_fs:.1f}px !important;
            }}
            .metric-card .lbl {{ font-size: {10.5*_fs:.1f}px !important; }}
            .metric-card .val, [data-testid="stMetricValue"] {{ font-size: {18*_fs:.1f}px !important; }}
            h1 {{ font-size: {23*_fs:.1f}px !important; }}
            h2, h3, h4 {{ font-size: {16*_fs:.1f}px !important; }}
            [data-testid="stDataFrame"] * {{ font-size: {13*_fs:.1f}px !important; }}
            </style>
            """, unsafe_allow_html=True)
            st.session_state.mobile_chart_mode = st.checkbox(
                "手機圖表手勢最佳化", st.session_state.mobile_chart_mode,
                key="_mobile_zoom_top",
                help="開啟後：一指可上下滑頁面，兩指在圖表上拉開/縮合即可放大縮小。")
        st.divider()

        # Search
        sym=st.text_input("","",placeholder="股票代碼（Enter 搜尋）",
            label_visibility="collapsed",key="si",
            on_change=lambda: st.session_state.update(_trigger=True))
        abtn=st.button("🔍 分析",type="primary",use_container_width=True)

        # Watchlist
        st.divider()
        st.markdown("#### 自選股")
        wl=list(st.session_state.watchlist)
        if wl:
            wl_labels=[]
            for _c in wl:
                _n=TW_NAME_CACHE.get(_c,"")
                wl_labels.append(f"{_n} ({_c})" if (_n and _n!=_c) else _c)
            _prev=st.session_state.get("_wl_sel",wl_labels[0])
            if _prev not in wl_labels: _prev=wl_labels[0]
            _sel=st.selectbox("",wl_labels,index=wl_labels.index(_prev),
                label_visibility="collapsed",key="wl_sel")
            _sel_code=wl[wl_labels.index(_sel)]
            c1,c2,c3=st.columns(3)
            if c1.button("載入",use_container_width=True):
                st.session_state._pending_sym=_sel_code; st.rerun()
            if c2.button("加入",use_container_width=True):
                cd=st.session_state.get("si","").strip().upper()
                if cd and cd not in st.session_state.watchlist:
                    st.session_state.watchlist.append(cd); st.rerun()
            if c3.button("移除",use_container_width=True):
                st.session_state.watchlist.pop(wl_labels.index(_sel)); st.rerun()
        else:
            st.caption("自選股為空")
            if st.button("加入代碼",use_container_width=True):
                cd=st.session_state.get("si","").strip().upper()
                if cd and cd not in st.session_state.watchlist:
                    st.session_state.watchlist.append(cd); st.rerun()

        if st.button("📊 批次分析全部",use_container_width=True):
            st.session_state._do_batch=True

        st.divider()

        # Chart options
        with st.expander("圖表顯示",expanded=False):
            st.session_state.chart_style = st.radio("",["K棒","曲線"],
                horizontal=True, key="_r_style",
                index=0 if st.session_state.chart_style=="K棒" else 1)
            c1,c2=st.columns(2)
            with c1:
                st.session_state.show_bb  = st.checkbox("布林帶",   st.session_state.show_bb,  key="_c_bb")
                st.session_state.show_sr  = st.checkbox("支撐壓力", st.session_state.show_sr,  key="_c_sr")
            with c2:
                st.session_state.show_band= st.checkbox("預測信賴帶",st.session_state.show_band,key="_c_band")
            st.markdown("**均線**")
            _mc1,_mc2,_mc3=st.columns(3)
            st.session_state.show_ma5 =_mc1.checkbox("MA5", st.session_state.show_ma5, key="_c_ma5")
            st.session_state.show_ma20=_mc2.checkbox("MA20",st.session_state.show_ma20,key="_c_ma20")
            st.session_state.show_ma60=_mc3.checkbox("MA60",st.session_state.show_ma60,key="_c_ma60")
            st.caption("MACD/RSI 副圖在分析頁圖表上方控制")
            st.markdown("**手機圖表操作**")
            st.caption("建議手機維持 pan：一指滑頁面，兩指在圖表上拉開/縮合會直接縮放圖表；不用先切 zoom。")
            st.session_state.chart_dragmode = st.radio(
                "主圖預設手勢", ["pan","zoom"],
                index=0 if st.session_state.chart_dragmode=="pan" else 1,
                horizontal=True, key="_dragmode",
                help="pan：一指滑頁面/桌面左鍵平移；zoom：框選放大。手機兩指縮放在 pan 模式也可用。")

        with st.expander("分析參數",expanded=False):
            _fd_prev = st.session_state.forecast_days
            _ly_prev = st.session_state.lookback_years
            st.session_state.forecast_days =st.slider("預測天數",5,90,st.session_state.forecast_days)
            st.session_state.lookback_years=st.slider("回溯年數",1,10,st.session_state.lookback_years)
            st.session_state.train_ratio   =st.slider("ML訓練佔比",.5,.95,st.session_state.train_ratio,.05)
            st.session_state.intraday_auto_refresh = st.checkbox(
                "盤中走向 10 秒自動更新",
                st.session_state.intraday_auto_refresh,
                key="_intraday_auto")
            st.session_state.api_key = st.text_input(
                "Fugle/Fubon API Key（選填）",
                value=st.session_state.get("api_key",""),
                type="password",
                help="可留空；若 Streamlit Secrets 已設定 FUGLE_API_KEY，也會自動使用。",
                key="_api_key_input")
            if _streamlit_secret_key() and not st.session_state.api_key:
                st.caption("已偵測到 Streamlit Secrets 內的 API key。")
            _params_changed = (
                st.session_state.forecast_days != _fd_prev or
                st.session_state.lookback_years != _ly_prev
            )
            if _params_changed and st.session_state.result is not None:
                st.caption("📌 參數已變更，點下方按鈕重新分析")
            if _params_changed or st.button("🔄 重新分析（套用新參數）",
                                             use_container_width=True,
                                             disabled=(st.session_state.result is None and
                                                       not st.session_state.get("si",""))):
                if st.session_state.result:
                    _sym = st.session_state.result.get("symbol","").split(".")[0]
                    if _sym:
                        st.session_state._pending_sym = _sym

        # Weight settings
        with st.expander("分析權重",expanded=False):
            st.caption("七項總計須為 100%，預設權重已依「技術趨勢 + ML + 基本面/訂單 + 法人籌碼 + 國際盤」調整")
            w={**DEFAULT_WEIGHTS, **st.session_state.weights}

            def _sync_weight(src_key, dst_key):
                # Streamlit runs callbacks before the next render, so this keeps
                # number input and slider visually synchronized in both directions.
                st.session_state[dst_key] = int(st.session_state[src_key])

            def _weight_row(label, key_s, key_n, val):
                if key_s not in st.session_state:
                    st.session_state[key_s] = int(val)
                if key_n not in st.session_state:
                    st.session_state[key_n] = int(st.session_state[key_s])
                _c1, _c2 = st.columns([3,1])
                _c1.slider(label, 0, 100, key=key_s,
                           on_change=_sync_weight, args=(key_s, key_n))
                _c2.number_input("", 0, 100, step=1, key=key_n,
                                 label_visibility="collapsed",
                                 on_change=_sync_weight, args=(key_n, key_s))
                return int(st.session_state[key_s])
            wt = _weight_row("技術指標%", "wt_s", "wt_n", w["technical"])
            wm = _weight_row("ML模型%",   "wm_s", "wm_n", w["ml"])
            wn = _weight_row("新聞情緒%","wn_s", "wn_n", w["news"])
            wf = _weight_row("基本面/EPS/營收%",  "wf_s", "wf_n", w["fundamental"])
            wu = _weight_row("美股/國際盤%", "wu_s", "wu_n", w["us_market"])
            wi = _weight_row("三大法人%", "wi_s", "wi_n", w["institutional"])
            wg = _weight_row("融資融券%", "wg_s", "wg_n", w["margin"])
            tot=wt+wm+wn+wf+wu+wi+wg
            _wc=OK if tot==100 else ER
            st.markdown(
                "<span style='color:"+_wc+"'>"
                + ("✅" if tot==100 else "⚠")
                + f" 總計 {tot}%</span>",
                unsafe_allow_html=True)
            if st.button("▶ Apply 套用",use_container_width=True,disabled=(tot!=100)):
                st.session_state.weights={
                    "technical":wt,"ml":wm,"news":wn,"fundamental":wf,
                    "us_market":wu,"institutional":wi,"margin":wg}
                if st.session_state.result:
                    r_=st.session_state.result
                    nd=compute_drift_bias(
                        r_["indicators"], r_.get("ml_predict",{}), r_.get("nn_predict",{}),
                        r_.get("news_sentiment",{}), r_.get("fundamental",{}),
                        st.session_state.weights, r_.get("mkt_ctx",{}),
                        r_.get("institutional",{}), r_.get("margin",{}))
                    nf=simple_forecast(r_["df"],days=r_["forecast_days"],
                        n_paths=180,drift_bias=nd)
                    r_["forecast"]=nf; r_["drift_bias"]=nd
                    r_["weights"]=st.session_state.weights
                st.rerun()

        st.divider()
        if st.button("🔥 每日推薦股票（掃描前100大成交量）", use_container_width=True):
            st.session_state._show_reco = True
            st.rerun()

        st.divider()
        st.caption(f"{'GB+XGB' if XGB_OK else 'GB'} · 美股/法人/融資融券特徵")
    return sym, abtn

# ── Run analysis ───────────────────────────────────────────────────────────
def analyze(sym):
    if not sym: return
    p=st.progress(0, text="分析中…")
    def cb(s,t,m): p.progress(s/t, text=f"[{s}/{t}] {m}")
    try:
        r=run_analysis(sym.strip().upper(),
            lookback_years=st.session_state.lookback_years,
            forecast_days=st.session_state.forecast_days,
            weights=st.session_state.weights,
            train_ratio=st.session_state.train_ratio,
            progress_callback=cb,
            api_key=read_marketdata_key(st.session_state.get("api_key","") or _streamlit_secret_key()))
        r["weights"]=st.session_state.weights
        st.session_state.result=r; p.empty()
    except Exception as e:
        p.empty()
        st.warning("⚠️ " + friendly_error_message(e))
        with st.expander("查看技術細節", expanded=False):
            st.code(str(e))


# ── PDF ────────────────────────────────────────────────────────────────────

def build_reason_text(analysis: dict) -> str:
    """
    Generate concise, informative local interpretation.
    Based on academic best practices for Taiwan stock prediction
    (RSI divergence, MACD cross, BB squeeze, MA alignment, volume anomaly).
    """
    df   = analysis["df"]
    ind  = analysis["indicators"]
    sr   = analysis["sr"]
    fc   = analysis["forecast"]
    ml_p = analysis.get("ml_predict", {})
    nn_p = analysis.get("nn_predict", {})
    bt   = analysis.get("backtest", {})
    ns   = analysis.get("news_sentiment", {})

    close = df["Close"]
    last  = float(close.iloc[-1])
    med_future = float(fc["median"][-1])
    chg_pct    = (med_future - last) / last * 100

    rsi   = float(ind["rsi14"].iloc[-1])
    macd_h= float(ind["macd_hist"].iloc[-1])
    macd_l= float(ind["macd_line"].iloc[-1])
    macd_s= float(ind["macd_signal"].iloc[-1])
    ma5   = float(ind["ma5"].iloc[-1])
    ma20  = float(ind["ma20"].iloc[-1])
    ma60  = float(ind["ma60"].iloc[-1])
    bb_up = float(ind["bb_up"].iloc[-1])
    bb_dn = float(ind["bb_dn"].iloc[-1])
    bb_w  = float(ind["bb_width"].iloc[-1]) if not pd.isna(ind["bb_width"].iloc[-1]) else 5.0
    atr   = float(ind["atr14"].iloc[-1]) if not pd.isna(ind["atr14"].iloc[-1]) else last * 0.02
    vol   = float(df["Volume"].iloc[-1])
    vol_ma= float(ind["vol_ma20"].iloc[-1]) if not pd.isna(ind["vol_ma20"].iloc[-1]) else vol

    bb_pos = (last - bb_dn) / max(bb_up - bb_dn, 1e-9)
    bb_prev_w = float(ind["bb_width"].iloc[-2]) if len(ind["bb_width"]) > 1 and not pd.isna(ind["bb_width"].iloc[-2]) else bb_w
    vol_ratio = vol / vol_ma if vol_ma > 0 else 1.0

    # Detect key patterns
    macd_cross_bull = (float(ind["macd_hist"].iloc[-2]) < 0 <= macd_h) if len(ind["macd_hist"]) > 1 else False
    macd_cross_bear = (float(ind["macd_hist"].iloc[-2]) > 0 >= macd_h) if len(ind["macd_hist"]) > 1 else False
    bb_squeeze      = bb_w < bb_prev_w * 0.85   # band narrowing = volatility contraction
    bb_expand       = bb_w > bb_prev_w * 1.15

    # RSI divergence (price new high but RSI lower, or vice versa)
    if len(close) >= 10:
        p_hi5  = float(close.iloc[-6:-1].max())
        r_hi5  = float(ind["rsi14"].iloc[-6:-1].max()) if not pd.isna(ind["rsi14"].iloc[-6:-1].max()) else rsi
        bear_div = (last > p_hi5 * 1.005) and (rsi < r_hi5 * 0.97)
        bull_div = (last < p_hi5 * 0.995) and (rsi > r_hi5 * 1.03)
    else:
        bear_div = bull_div = False

    lines = []

    # ── RSI ──────────────────────────────────────────────────
    if rsi >= 75:
        lines.append(f"RSI {rsi:.0f} 深度超買，短線獲利了結壓力明顯。")
    elif rsi >= 65:
        lines.append(f"RSI {rsi:.0f} 偏高，動能尚強但需注意回落。")
    elif rsi <= 25:
        lines.append(f"RSI {rsi:.0f} 深度超賣，歷史上此區間反彈機率偏高。")
    elif rsi <= 35:
        lines.append(f"RSI {rsi:.0f} 超賣區，留意企穩訊號。")
    else:
        lines.append(f"RSI {rsi:.0f} 中性區間，方向待確認。")

    if bear_div:
        lines.append("⚠ 空頭背離：價格創近高但 RSI 未跟上，動能衰竭警示。")
    elif bull_div:
        lines.append("✦ 多頭背離：價格創近低但 RSI 未跟低，下跌動能減弱。")

    # ── MACD ─────────────────────────────────────────────────
    if macd_cross_bull:
        lines.append(f"MACD 金叉（Histogram 由負轉正），短線偏多訊號。")
    elif macd_cross_bear:
        lines.append(f"MACD 死叉（Histogram 由正轉負），短線偏空訊號。")
    elif macd_l > macd_s:
        lines.append(f"MACD 多頭格局（{macd_l:+.3f}），Histogram {macd_h:+.3f}{'擴大' if macd_h > 0 else '縮小'}。")
    else:
        lines.append(f"MACD 空頭格局（{macd_l:+.3f}），動能偏弱。")

    # ── 布林帶 ───────────────────────────────────────────────
    if bb_squeeze:
        lines.append(f"布林帶收縮（寬度 {bb_w:.1f}%）→ 波動壓縮，可能醞釀突破方向。")
    elif bb_expand:
        if bb_pos > 0.6:
            lines.append(f"布林帶向上擴張（價格 {bb_pos*100:.0f}% 分位）→ 強勢突破上軌。")
        else:
            lines.append(f"布林帶向下擴張（價格 {bb_pos*100:.0f}% 分位）→ 下行動能增強。")
    elif bb_pos > 0.85:
        lines.append(f"價格貼近布林上軌，注意短線回測中軌壓力。")
    elif bb_pos < 0.15:
        lines.append(f"價格貼近布林下軌，可能出現技術反彈。")

    # ── 均線排列 ─────────────────────────────────────────────
    if ma5 > ma20 > ma60:
        lines.append(f"MA 多頭排列（5>{ma20:.0f}>60），趨勢向上。")
    elif ma5 < ma20 < ma60:
        lines.append(f"MA 空頭排列（5<{ma20:.0f}<60），趨勢向下。")

    # ── 成交量 ───────────────────────────────────────────────
    if vol_ratio >= 2.0:
        lines.append(f"成交量爆增（均量 {vol_ratio:.1f}x），留意主力動向。")
    elif vol_ratio <= 0.5:
        lines.append(f"成交量萎縮（{vol_ratio:.1f}x），行情觀望為主。")

    # ── 預測與統計 ───────────────────────────────────────────
    fc_days = analysis.get("forecast_days", 30)
    lines.append(f"預測 {fc_days} 日後中位價 {med_future:,.2f}（{chg_pct:+.1f}%），"
                 f"ATR {atr:.2f}。")

    # ── ML + NN 融合 ─────────────────────────────────────────
    ml_prob = ml_p.get("prob_up")
    nn_prob = nn_p.get("prob_up")
    if ml_prob is not None and nn_prob is not None:
        avg_p = (ml_prob + nn_prob) / 2 * 100
        lines.append(f"ML {ml_prob*100:.0f}% · NN {nn_prob*100:.0f}%（融合 {avg_p:.0f}%）。")
    elif ml_prob is not None:
        lines.append(f"ML 上漲機率 {ml_prob*100:.0f}%（{ml_p.get('label','')}）。")

    if bt and bt.get("n_trades", 0) > 0:
        lines.append(f"回測命中率 {bt.get('hit_rate_text','N/A')}（{bt['n_trades']} 筆）。")

    # ── 支撐壓力 ─────────────────────────────────────────────
    lines.append(f"支撐 {sr['support_lo']:.0f}~{sr['support_hi']:.0f}　"
                 f"壓力 {sr['resistance_lo']:.0f}~{sr['resistance_hi']:.0f}")

    return "\n".join(lines)



def generate_pdf_bytes(analysis: dict, chart_png_bytes=None) -> bytes:
    """
    Generate PDF report using reportlab. Returns True on success.
    analysis is the full dict from analyze pipeline.
    """
    if not REPORTLAB_OK:
        raise RuntimeError("reportlab 未安裝，無法輸出 PDF。請執行：pip install reportlab")

    # Register CJK font
    font_name = _register_pdf_cjk_font()

    _pdf_io = io.BytesIO()
    doc = SimpleDocTemplate(
        _pdf_io, pagesize=A4,
        topMargin=15*mm, bottomMargin=15*mm,
        leftMargin=15*mm, rightMargin=15*mm,
    )
    styles = getSampleStyleSheet()

    ps_title = ParagraphStyle(
        "title", parent=styles["Title"],
        fontName=font_name, fontSize=18, textColor=rl_colors.HexColor(L_ACCENT),
        spaceAfter=8,
    )
    ps_h2 = ParagraphStyle(
        "h2", parent=styles["Heading2"],
        fontName=font_name, fontSize=13, textColor=rl_colors.HexColor(L_ACCENT),
        spaceBefore=8, spaceAfter=4,
    )
    ps_body = ParagraphStyle(
        "body", parent=styles["Normal"],
        fontName=font_name, fontSize=10, leading=15,
    )
    ps_dim = ParagraphStyle(
        "dim", parent=styles["Normal"],
        fontName=font_name, fontSize=9, textColor=rl_colors.HexColor("#777777"),
    )

    story = []
    # Header
    story.append(Paragraph(f"{analysis['name']}  ({analysis['symbol']})", ps_title))
    story.append(Paragraph(
        f"分析時間：{datetime.now().strftime('%Y-%m-%d %H:%M')}  ·  "
        f"預測區間：{analysis.get('forecast_days', 30)} 天",
        ps_dim
    ))
    story.append(Spacer(1, 6*mm))

    # Chart
    if chart_png_bytes:
        from io import BytesIO
        img = RLImage(BytesIO(chart_png_bytes), width=180*mm, height=100*mm)
        story.append(img)
        story.append(Spacer(1, 4*mm))

    # Trade plan
    story.append(Paragraph("一、交易計畫", ps_h2))
    df = analysis["df"]
    ind = analysis["indicators"]
    sr = analysis["sr"]
    fc = analysis["forecast"]
    last_price = float(df["Close"].iloc[-1])
    med_future = float(fc["median"][-1])
    chg_pct = (med_future - last_price) / last_price * 100
    atr = float(ind["atr14"].iloc[-1]) if not pd.isna(ind["atr14"].iloc[-1]) else last_price * 0.02
    sl = last_price - atr*1.5
    tp = last_price + atr*2.5
    rr = (tp - last_price) / max(last_price - sl, 1e-9)
    action = "偏多觀察" if chg_pct > 2 else ("偏空觀察" if chg_pct < -2 else "中性")

    plan_rows = [
        ["建議",   action],
        ["現價",   f"{last_price:,.2f}"],
        ["進場",   f"{last_price:,.2f}"],
        ["停損",   f"{sl:,.2f}"],
        ["停利",   f"{tp:,.2f}"],
        ["RR 比",  f"{rr:.2f} x"],
        [f"{analysis.get('forecast_days',30)} 日預測", f"{med_future:,.2f}  ({chg_pct:+.1f}%)"],
    ]
    if "ml_stats" in analysis and "accuracy" in analysis["ml_stats"]:
        plan_rows.append(["ML 成功率", f"{analysis['ml_predict']['prob_up']*100:.1f}%"])
    if "backtest" in analysis:
        plan_rows.append(["回測方向命中率", analysis['backtest'].get('hit_rate_text', 'N/A')])

    t = RLTable(plan_rows, colWidths=[50*mm, 120*mm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (0, -1), rl_colors.HexColor("#f0f2f7")),
        ("TEXTCOLOR",  (0, 0), (0, -1), rl_colors.HexColor(L_ACCENT)),
        ("GRID",       (0, 0), (-1, -1), 0.3, rl_colors.HexColor("#c0c0c0")),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 4*mm))

    # Key levels
    story.append(Paragraph("二、關鍵價位", ps_h2))
    lvl_rows = [
        ["壓力上緣", f"{sr['resistance_hi']:,.2f}"],
        ["壓力下緣", f"{sr['resistance_lo']:,.2f}"],
        ["現價",     f"{last_price:,.2f}"],
        ["支撐上緣", f"{sr['support_hi']:,.2f}"],
        ["支撐下緣", f"{sr['support_lo']:,.2f}"],
    ]
    t2 = RLTable(lvl_rows, colWidths=[50*mm, 120*mm])
    t2.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (0, -1), rl_colors.HexColor("#f0f2f7")),
        ("TEXTCOLOR",  (0, 0), (0, -1), rl_colors.HexColor(L_ACCENT)),
        ("GRID",       (0, 0), (-1, -1), 0.3, rl_colors.HexColor("#c0c0c0")),
    ]))
    story.append(t2)
    story.append(Spacer(1, 4*mm))

    # Reasons
    story.append(Paragraph("三、趨勢判斷與理由", ps_h2))
    reason_text = _replace_segment_codes(analysis.get("reason_text", "—"))
    for line in reason_text.split("\n"):
        if line.strip():
            story.append(Paragraph(line.strip(), ps_body))

    # News
    news = analysis.get("news", [])
    if news:
        story.append(Spacer(1, 4*mm))
        story.append(Paragraph("四、新聞重點", ps_h2))
        for i, it in enumerate(news[:5], 1):
            src_tag = f"【{it.get('source','')}】" if it.get("source") else ""
            story.append(Paragraph(f"{i}. {src_tag}{it.get('title','')}", ps_body))

    # AI commentary
    ai_text = analysis.get("ai_commentary", "")
    if ai_text:
        story.append(Spacer(1, 4*mm))
        story.append(Paragraph("五、AI 綜合評論", ps_h2))
        for para in ai_text.split("\n"):
            if para.strip():
                story.append(Paragraph(para.strip(), ps_body))

    # Footer
    story.append(Spacer(1, 8*mm))
    story.append(Paragraph(
        "本報告由 Stock Analyzer Pro 自動生成，僅供研究與學習，"
        "不構成任何投資建議。投資需謹慎，盈虧自負。",
        ps_dim
    ))

    doc.build(story)
    return _pdf_io.getvalue()


_pdf_cjk_font_registered = False

def _register_pdf_cjk_font() -> str:
    """
    Register a CJK-capable font for ReportLab.
    Prefer ReportLab built-in CID fonts so PDF Chinese text does not become □□□
    on Streamlit Cloud/Linux where Windows CJK fonts are usually absent.
    """
    global _pdf_cjk_font_registered
    if not REPORTLAB_OK:
        return "Helvetica"
    if _pdf_cjk_font_registered:
        return "STSong-Light"

    # Built-in CID font: reliable for CJK in ReportLab and does not require font files.
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        _pdf_cjk_font_registered = True
        return "STSong-Light"
    except Exception:
        pass

    candidates = [
        "C:/Windows/Fonts/msjh.ttc",
        "C:/Windows/Fonts/mingliu.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKtc-Regular.otf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]
    for fp in candidates:
        try:
            if fp and __import__("os").path.exists(fp):
                if fp.endswith(".ttc"):
                    pdfmetrics.registerFont(TTFont("CJKFont", fp, subfontIndex=0))
                else:
                    pdfmetrics.registerFont(TTFont("CJKFont", fp))
                _pdf_cjk_font_registered = True
                return "CJKFont"
        except Exception:
            continue
    return "Helvetica"

# ── Daily recommendation panel ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _cached_daily_recommendations(weights_tuple):
    weights = dict(weights_tuple)
    return recommend_top_volume_stocks(
        volume_limit=100, top_n=8,
        lookback_years=1, forecast_days=20,
        weights=weights,
        progress_callback=None,
        fast_mode=True,
        finalist_count=18,
    )

def render_daily_recommendations():
    st.markdown("### 🔥 每日推薦股票")
    st.caption("掃描範圍：每日由 TWSE/TPEX 公開資料取得台股前 100 大成交量，不限自選股。流程：100 檔全部快速掃描 → 高分候選完整分析；結果快取 1 小時，重新整理或快取到期會更新。")
    c1,c2=st.columns([1,5])
    if c1.button("關閉推薦窗", use_container_width=True):
        st.session_state._show_reco=False
        st.rerun()
    try:
        weights_tuple=tuple(sorted(st.session_state.weights.items()))
        with st.spinner("快速掃描前 100 大成交量台股，並對候選股做完整分析…"):
            rows=_cached_daily_recommendations(weights_tuple)
        if not rows:
            st.info("目前沒有篩出合適標的，或資料來源暫時忙碌。")
            return
        for i, it in enumerate(rows, 1):
            border = OK if i <= 3 else BD
            st.markdown(
                f"<div class='metric-card' style='border-color:{border}'>"
                f"<div class='lbl'>Top {i} · 綜合分數 {it['score']:+.2f}</div>"
                f"<div class='val'>{it['name']} <span style='font-size:.9rem;color:{DM}'>({it['code']})</span></div>"
                f"<div class='sub'>現價 {it['last']:.2f}｜20日預測 {it['forecast_pct']:+.1f}%｜ML {it['ml_prob']:.0f}%｜法人 {it['inst_total']/1000:,.0f} 張</div>"
                f"<p style='color:{TX};margin:.4rem 0 0'>推薦原因：{it['reason']}</p>"
                f"</div>",
                unsafe_allow_html=True)
    except Exception as e:
        st.warning("推薦掃描暫時無法完成：" + friendly_error_message(e))

# ── Show result ────────────────────────────────────────────────────────────
def show(r):
    df=r["df"]; ind=r["indicators"]; sr=r["sr"]; fc=r["forecast"]
    ml_p=r.get("ml_predict",{}); bt=r.get("backtest",{})
    ns=r.get("news_sentiment",{}); news=r.get("news",[])
    fund=r.get("fundamental",{}); name=r["name"]; sym=r["symbol"]
    last=float(df["Close"].iloc[-1])
    prev=float(df["Close"].iloc[-2]) if len(df)>1 else last
    cp=(last-prev)/prev*100; med=float(fc["median"][-1]); fc_c=(med-last)/last*100
    s=calc_score(r); prob=ml_p.get("prob_up",.5)
    bhr=bt.get("hit_rate",.5) if bt.get("n_trades",0)>0 else .5
    cs=(abs(prob-.5)*2+abs(bhr-.5)*2)/2
    conf="高" if cs>=.35 else("中" if cs>=.18 else "低")
    rsi_v=float(ind["rsi14"].iloc[-1])

    # Header
    arr="▲" if cp>=0 else "▼"; hc=OK if cp>=0 else ER
    st.markdown(
        f"<h1>{name} "
        f"<span style='font-size:.85rem;color:{DM}'>({sym.split('.')[0]})</span></h1>"
        f"<div style='font-size:1.35rem;font-weight:700;color:{hc}'>{last:,.2f} "
        f"<span style='font-size:.9rem'>{arr} {abs(cp):.2f}%</span></div>",
        unsafe_allow_html=True)

    # Metric cards
    cols=st.columns(5)
    items=[
        ("建議操作", action_badge(s), ""),
        ("可信度", badge("bhi" if conf in("高","中") else "blo",conf),
         f"ML {prob*100:.0f}%"),
        (f"{r['forecast_days']}日預測",
         f"<span style='color:{OK if fc_c>0 else ER}'>{med:,.2f}</span>",
         f"{fc_c:+.1f}%"),
        ("回測命中率", bt.get("hit_rate_text","N/A"), f"{bt.get('n_trades',0)} 筆"),
        ("RSI(14)", f"{rsi_v:.1f}",
         "超買" if rsi_v>=70 else("超賣" if rsi_v<=30 else "中性")),
    ]
    for col,(lbl,val,sub) in zip(cols,items):
        col.markdown(
            f"<div class='metric-card'>"
            f"<div class='lbl'>{lbl}</div>"
            f"<div class='val'>{val}</div>"
            f"{'<div class=sub>'+sub+'</div>' if sub else ''}"
            f"</div>", unsafe_allow_html=True)

    # ── Chart controls (inline, compact) ──
    _cc1, _cc2, _cc3, _cc4, _cc5 = st.columns([1,1,1,1,3])
    # Checkboxes update state directly — Streamlit reruns automatically on widget change
    _new_macd = _cc1.checkbox("MACD", st.session_state.show_macd, key="_chk_macd")
    _new_rsi  = _cc2.checkbox("RSI",  st.session_state.show_rsi,  key="_chk_rsi")
    st.session_state.show_macd = _new_macd
    st.session_state.show_rsi  = _new_rsi
    show_macd_pop = _cc3.button("MACD 展開", use_container_width=True)
    show_rsi_pop  = _cc4.button("RSI 展開",  use_container_width=True)
    _cc5.caption("🖥 滾輪縮放・左鍵平移  📱 一指滑頁面・兩指直接縮放圖表")

    if st.session_state.get("mobile_chart_mode", True):
        st.markdown("<div class='pinch-hint'>📱 手機操作：維持 pan 模式時，一指上下滑頁面；兩指在圖表上拉開/縮合可直接放大縮小。</div>", unsafe_allow_html=True)

    # Main chart
    fig=build_chart(r,
        style    =st.session_state.chart_style,
        bb       =st.session_state.show_bb,
        sr_on    =st.session_state.show_sr,
        band     =st.session_state.show_band,
        show_macd=st.session_state.show_macd,
        show_rsi =st.session_state.show_rsi,
        show_ma5 =st.session_state.show_ma5,
        show_ma20=st.session_state.show_ma20,
        show_ma60=st.session_state.show_ma60,
        dragmode=st.session_state.chart_dragmode)

    # Chart config:
    # - scrollZoom=True: mouse wheel zooms on desktop
    # - modeBarButtons: keep zoom, pan, reset; remove unnecessary ones
    # - dragmode="pan" set in layout (left-click drags/pans)
    chart_config={
        "displaylogo": False,
        "responsive": True,
        "displayModeBar": True,
        "scrollZoom": True,          # desktop: scroll to zoom
        "doubleClick": "reset",      # double-click resets view
        "modeBarButtonsToRemove": [
            "autoScale2d","lasso2d","select2d","toImage"
        ],
        "modeBarButtonsToAdd": ["zoom2d","pan2d","resetScale2d"],
    }
    st.plotly_chart(fig, use_container_width=True, config=chart_config)

    # Subchart popouts
    if show_macd_pop:
        with st.expander("MACD 詳細圖",expanded=True):
            render_subchart_popout(r,"MACD")
    if show_rsi_pop:
        with st.expander("RSI 詳細圖",expanded=True):
            render_subchart_popout(r,"RSI")

    # Export
    e1,e2,_=st.columns([1,1,4])
    with e1:
        try:
            png=fig.to_image(format="png",width=1400,height=700,scale=2)
            st.download_button("📷 PNG", png,
                file_name=f"{sym.split('.')[0]}_{datetime.now().strftime('%Y%m%d')}.png",
                mime="image/png", use_container_width=True)
        except: st.caption("PNG需kaleido")
    with e2:
        try:
            # Get chart as PNG for PDF embedding
            try:
                _chart_png = fig.to_image(format="png", width=1400, height=700, scale=1.5)
            except Exception:
                _chart_png = None
            _pdf = generate_pdf_bytes(r, _chart_png)
            if _pdf:
                st.download_button("📄 PDF", _pdf,
                    file_name=f"{sym.split('.')[0]}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf", use_container_width=True)
            else: st.caption("PDF需reportlab")
        except Exception as _pe:
            st.caption(f"PDF: {_pe}")

    # ── Analysis tabs ──
    t1,t2,t3,t4,t5,t6,t7,t8,t9=st.tabs([
        "📊 ML / 回測","🔍 技術解讀","📋 基本面","📰 即時新聞",
        "📍 關鍵價位","💰 籌碼/美股","🧪 策略回測",
        "盤中走向","模型回測明細"
    ])

    with t1:
        ml_stats=r.get("ml_stats",{})
        a,b=st.columns(2)
        with a:
            st.markdown("**ML 模型**")
            if "error" in ml_stats: st.warning(ml_stats["error"])
            else:
                pc=OK if prob>=.6 else(ER if prob<=.4 else WA)
                st.markdown(
                    f"<span style='font-size:1.1rem;font-weight:700;color:{pc}'>"
                    f"上漲機率 {prob*100:.1f}%</span>",
                    unsafe_allow_html=True)
                st.metric("OOS準確率",f"{ml_stats.get('oos_accuracy',0)*100:.1f}%")
                st.metric("Brier Score",f"{ml_stats.get('brier',0):.3f}")
                st.metric("訓練樣本",f"{ml_stats.get('n_samples',0)} 筆")
                st.metric("演算法",ml_stats.get("algorithm","—"))
            rp=r.get("return_prediction",{})
            if rp:
                st.markdown("**未來報酬率模型**")
                if "error" in rp:
                    st.warning(rp["error"])
                else:
                    st.metric("預測報酬",f"{rp.get('expected_return',0)*100:+.2f}%")
                    st.metric("報酬上漲機率",f"{rp.get('prob_up',0.5)*100:.1f}%")
                    st.metric("分段",_segment_label(rp.get("segment","—")))
                    st.caption(f"{_return_model_label(rp.get('model',''))} · samples={rp.get('n_samples',0)}")
        with b:
            st.markdown("**回測（含台灣市場交易成本 ~0.588%）**")
            if bt.get("n_trades",0)>0:
                st.metric("交易筆數",str(bt["n_trades"]))
                st.metric("方向命中率",bt["hit_rate_text"])
                st.metric("含成本淨報酬",f"{bt.get('avg_return_after_cost',0):+.2f}%")
                st.metric("Sharpe Ratio",f"{bt.get('sharpe',0):.2f}")
                st.metric("最大回撤",f"{bt.get('max_drawdown',0):.1f}%")
                st.metric("Profit Factor",f"{bt.get('profit_factor',0):.2f}")
            else: st.info("回測樣本不足（需 > 36 筆）")
        st.caption(f"drift_bias = {r.get('drift_bias',0):+.5f}")
        st.divider()
        # Custom backtest
        st.markdown("**自訂回測**")
        h2,n2,rb=st.columns([1,1,1])
        hv=h2.number_input("天數",5,60,10,key="bh")
        nv=n2.number_input("樣本",20,500,100,key="bn")
        if rb.button("執行回測",use_container_width=True):
            with st.spinner("回測中…"):
                bt2=backtest_directional(df,ind,sr,horizon=hv,n_samples=nv)
            if bt2.get("n_trades",0)>0:
                r1,r2,r3,r4=st.columns(4)
                r1.metric("命中率",bt2["hit_rate_text"])
                r2.metric("淨報酬",f"{bt2.get('avg_return_after_cost',0):+.2f}%")
                r3.metric("Sharpe",f"{bt2.get('sharpe',0):.2f}")
                r4.metric("最大回撤",f"{bt2.get('max_drawdown',0):.1f}%")
            else: st.warning("樣本不足")
        st.divider()
        mbt=r.get("macd_cross_backtest",{})
        st.markdown("**MACD 黃金交叉買入，持有 10 天**")
        if mbt.get("n_trades",0)>0:
            m1,m2,m3,m4=st.columns(4)
            m1.metric("交易次數",mbt["n_trades"])
            m2.metric("勝率",mbt["win_rate_text"])
            m3.metric("平均報酬",f"{mbt.get('avg_return',0):+.2f}%")
            m4.metric("最差/最佳",f"{mbt.get('worst_return',0):+.1f}% / {mbt.get('best_return',0):+.1f}%")
        else:
            st.info("這段資料期間沒有足夠的 MACD 黃金交叉樣本。")

    with t2:
        # Show full reason text from desktop-quality analysis
        try:
            reason = _replace_segment_codes(r.get("reason_text") or build_reason_text(r))
            if reason:
                for line in reason.split("\n"):
                    if line.strip():
                        st.markdown(line)
                st.divider()
        except Exception: pass
        _render_tech(r,df,ind,fc,ml_p,bt,last,med,fc_c,rsi_v)

    with t3:
        if fund:
            a,b=st.columns(2)
            items_f=[("本益比P/E","pe_ratio",lambda v:f"{v:.1f}x"),
                   ("EPS","eps",lambda v:f"{v:.2f}"),
                   ("ROE","roe",lambda v:f"{v*100:.1f}%"),
                   ("殖利率","div_yield",lambda v:f"{v*100:.2f}%"),
                   ("市值","market_cap",lambda v:f"{v/1e9:.1f}B" if v>1e9 else f"{v/1e6:.0f}M")]
            for i,(l,k,f) in enumerate(items_f):
                if k in fund: (a if i%2==0 else b).metric(l,f(fund[k]))
        else: st.info("無基本面資料（ETF 或資料不足）")

    with t4:
        if news:
            lbl=ns.get("label","中性")
            lc=OK if lbl=="正面" else(ER if lbl=="負面" else WA)
            st.markdown(
                f"整體情緒：<span style='color:{lc};font-weight:700'>{lbl}</span>",
                unsafe_allow_html=True)
            for item in news:
                t_=item.get("title",""); lk=item.get("link","#")
                sc_=item.get("source","")
                st.markdown(
                    f"<div class='ni'>"
                    f"<a href='{lk}' target='_blank'>{t_}</a>"
                    f"{'<span style=\"font-size:.75rem;color:'+DM+'\">['+sc_+']</span>' if sc_ else ''}"
                    f"</div>", unsafe_allow_html=True)
        else: st.info("未取得新聞")

    with t5:
        atr=float(ind["atr14"].iloc[-1]) if not pd.isna(ind["atr14"].iloc[-1]) else last*.02
        sl=last-atr*1.5; tp=last+atr*2.5; rr=(tp-last)/max(last-sl,1e-9)
        c1,c2,c3,c4=st.columns(4)
        c1.metric("支撐區",f"{sr['support_lo']:.2f}~{sr['support_hi']:.2f}")
        c2.metric("壓力區",f"{sr['resistance_lo']:.2f}~{sr['resistance_hi']:.2f}")
        c3.metric("建議停損",f"{sl:,.2f}",
            delta=f"-{(last-sl)/last*100:.1f}%",delta_color="inverse")
        c4.metric("建議停利",f"{tp:,.2f}",
            delta=f"+{(tp-last)/last*100:.1f}%")
        st.caption(f"ATR(14)={atr:.2f}  RR={rr:.2f}x")
        st.info("💡 停損 = 現價 - ATR×1.5，停利 = 現價 + ATR×2.5，此為參考值，請依個人風險偏好調整。")

    with t6:
        inst=r.get("institutional",{})
        margin=r.get("margin",{})
        mkt=r.get("mkt_ctx",{})
        st.markdown("**台股籌碼**")
        c1,c2,c3,c4=st.columns(4)
        c1.metric("外資買賣超",f"{inst.get('foreign_net',0)/1000:,.0f} 張")
        c2.metric("投信買賣超",f"{inst.get('trust_net',0)/1000:,.0f} 張")
        c3.metric("自營商買賣超",f"{inst.get('dealer_net',0)/1000:,.0f} 張")
        c4.metric("三大法人合計",f"{inst.get('total_net',0)/1000:,.0f} 張",
                  delta=f"score {inst.get('inst_score',0):+.2f}")
        st.caption(f"資料來源：{inst.get('source','N/A')}｜日期：{inst.get('date','—')}")
        st.markdown("**融資融券**")
        m1,m2,m3,m4=st.columns(4)
        m1.metric("融資餘額",f"{margin.get('margin_balance',0):,.0f} 張")
        m2.metric("融資變化",f"{margin.get('margin_change',0):+,.0f} 張")
        m3.metric("融券餘額",f"{margin.get('short_balance',0):,.0f} 張")
        m4.metric("融券變化",f"{margin.get('short_change',0):+,.0f} 張",
                  delta=f"score {margin.get('margin_score',0):+.2f}")
        st.caption(f"資料來源：{margin.get('source','N/A')}｜日期：{margin.get('date','—')}｜非 0 才代表有實際抓到資料")
        if margin.get("note"):
            st.warning(margin.get("note"))
        st.markdown("**美股 / 國際盤背景**")
        u1,u2,u3,u4,u5=st.columns(5)
        u1.metric("NASDAQ",f"{mkt.get('nasdaq_ret_1',0)*100:+.2f}%",f"5日 {mkt.get('nasdaq_ret_5',0)*100:+.2f}%")
        u2.metric("S&P500",f"{mkt.get('sp500_ret_1',0)*100:+.2f}%",f"5日 {mkt.get('sp500_ret_5',0)*100:+.2f}%")
        u3.metric("SMH",f"{mkt.get('smh_ret_1',mkt.get('semis_ret_1',0))*100:+.2f}%",f"5日 {mkt.get('smh_ret_5',mkt.get('semis_ret_5',0))*100:+.2f}%")
        u4.metric("費半SOX",f"{mkt.get('sox_ret_1',0)*100:+.2f}%",f"5日 {mkt.get('sox_ret_5',0)*100:+.2f}%")
        u5.metric("VIX",f"{mkt.get('vix',20):.1f}")

    with t7:
        st.markdown("**MACD 黃金交叉策略模擬器**")
        st.caption("問題範例：如果我在 MACD 黃金交叉時買入，持有 N 天的勝率是多少？")
        c1,c2,c3=st.columns([1,1,2])
        hd=c1.number_input("持有天數",1,120,10,key="macd_hold")
        ct=c2.selectbox("交叉類型",["golden","death"],
                        format_func=lambda x:"黃金交叉買入" if x=="golden" else "死亡交叉放空/避開",
                        key="macd_cross_type")
        if c3.button("執行 MACD 策略回測",use_container_width=True):
            st.session_state["_macd_bt_custom"]=simulate_macd_cross_strategy(df,ind,hold_days=hd,cross_type=ct)
        mb=st.session_state.get("_macd_bt_custom") or r.get("macd_cross_backtest",{})
        if mb.get("n_trades",0)>0:
            b1,b2,b3,b4,b5=st.columns(5)
            b1.metric("交易次數",mb["n_trades"])
            b2.metric("勝率",mb["win_rate_text"])
            b3.metric("平均報酬",f"{mb.get('avg_return',0):+.2f}%")
            b4.metric("中位數報酬",f"{mb.get('median_return',0):+.2f}%")
            b5.metric("最差報酬",f"{mb.get('worst_return',0):+.2f}%")
            st.dataframe(pd.DataFrame(mb.get("trades",[])),use_container_width=True,hide_index=True)
        else:
            st.info("此區間沒有符合條件的交叉訊號。")

    with t8:
        st.markdown("**盤中 1 分鐘走向**")
        st.caption("互動模式預設 Pan：桌面可滾輪縮放、左鍵平移、雙擊重置；手機可一指滑頁面、兩指縮放圖表。")
        if st.button("手動更新盤中資料", key="refresh_intraday_now", use_container_width=True):
            try:
                _cached_intraday.clear()
            except Exception:
                pass
            st.rerun()
        try:
            intra=_cached_intraday(sym, df)
            idf=intra.get("df",pd.DataFrame()).copy()
            pred=intra.get("prediction",pd.DataFrame()).copy()
            stats=intra.get("stats",{})
            if idf.empty:
                st.warning("目前沒有可顯示的盤中資料。")
            else:
                c1,c2,c3,c4=st.columns(4)
                c1.metric("開盤",f"{stats.get('open',0):.2f}")
                c2.metric("最新",f"{stats.get('latest',0):.2f}",f"{stats.get('change_pct',0)*100:+.2f}%")
                c3.metric("最高 / 最低",f"{stats.get('high',0):.2f} / {stats.get('low',0):.2f}")
                c4.metric("累計張數",f"{stats.get('volume_lots',0):,.0f}")
                st.caption(f"來源：{intra.get('source','')}；最後實際時間：{stats.get('last_time','')}")

                lots=idf.get("Volume_lots",idf["Volume"]/1000.0)
                fig_i=make_subplots(rows=2,cols=1,shared_xaxes=True,
                    vertical_spacing=.04,row_heights=[.72,.28],
                    subplot_titles=["實際走向","1分鐘逐列張數"])
                if st.session_state.chart_style=="K棒":
                    fig_i.add_trace(go.Candlestick(
                        x=idf.index,open=idf["Open"],high=idf["High"],low=idf["Low"],close=idf["Close"],
                        increasing_line_color=OK,decreasing_line_color=ER,name="實際K棒"),row=1,col=1)
                else:
                    fig_i.add_trace(go.Scatter(
                        x=idf.index,y=idf["Close"],mode="lines+markers",name="實際價格",
                        customdata=lots,line=dict(color=AC,width=2),marker=dict(size=4),
                        hovertemplate="時間=%{x|%H:%M}<br>價格=%{y:.2f}<br>成交量=%{customdata:.0f} 張<extra></extra>"
                    ),row=1,col=1)
                fig_i.add_trace(go.Bar(
                    x=idf.index,y=lots,name="1分鐘張數",marker_color="rgba(136,144,170,0.45)",
                    hovertemplate="時間=%{x|%H:%M}<br>成交量=%{y:.0f} 張<extra></extra>"
                ),row=2,col=1)
                fig_i.update_layout(height=460,template="plotly_dark" if D else "plotly_white",
                    hovermode="x unified",xaxis_rangeslider_visible=False,
                    paper_bgcolor=CBG,plot_bgcolor=CBG,font=dict(color=TX),
                    margin=dict(l=10,r=10,t=45,b=10),
                    dragmode=st.session_state.get("chart_dragmode","pan"))
                st.plotly_chart(fig_i,use_container_width=True,config=_plotly_pan_config())

                fig_p=go.Figure()
                if not pred.empty:
                    fig_p.add_trace(go.Scatter(
                        x=pred["Time"],y=pred["Predicted"],mode="lines",name="預測走向",
                        line=dict(color="#14b8a6",width=2.4),
                        hovertemplate="時間=%{x|%H:%M}<br>預測價格=%{y:.2f}<extra></extra>"))
                    actual=pred.dropna(subset=["Actual"])
                    if not actual.empty:
                        fig_p.add_trace(go.Scatter(
                            x=actual["Time"],y=actual["Actual"],mode="lines",name="已知實際價格",
                            line=dict(color=AC,width=1.8,dash="dot"),
                            hovertemplate="時間=%{x|%H:%M}<br>實際價格=%{y:.2f}<extra></extra>"))
                fig_p.update_layout(height=360,template="plotly_dark" if D else "plotly_white",
                    hovermode="x unified",paper_bgcolor=CBG,plot_bgcolor=CBG,font=dict(color=TX),
                    margin=dict(l=10,r=10,t=35,b=10),legend=dict(orientation="h"),
                    dragmode=st.session_state.get("chart_dragmode","pan"))
                st.plotly_chart(fig_p,use_container_width=True,config=_plotly_pan_config())

                rows=idf.reset_index().rename(columns={"index":"時間"})
                rows["時間"]=pd.to_datetime(rows["時間"]).dt.strftime("%H:%M")
                rows=rows[["時間","Open","High","Low","Close","Volume_lots"]].rename(columns={
                    "Open":"開盤","High":"最高","Low":"最低","Close":"價格","Volume_lots":"1分鐘張數"})
                st.dataframe(rows.tail(160),use_container_width=True,hide_index=True)
        except Exception as e:
            st.warning("目前抓不到盤中資料：" + friendly_error_message(e))

    with t9:
        st.markdown("**Walk-forward 報酬率模型明細**")
        rp=r.get("return_prediction",{})
        rb=r.get("return_backtest",{})
        d=r.get("backtest_detail",pd.DataFrame())
        c1,c2,c3,c4=st.columns(4)
        c1.metric("預測報酬",f"{rp.get('expected_return',0)*100:+.2f}%")
        c2.metric("上漲機率",f"{rp.get('prob_up',0.5)*100:.1f}%")
        c3.metric("WF 命中率",rb.get("hit_rate_text","N/A"))
        c4.metric("MAE",f"{rb.get('mae',0)*100:.2f}%")
        st.caption(f"目前分段：{_segment_label(rp.get('segment','—'))}；模型：{_return_model_label(rp.get('model',''))}")
        if isinstance(d,pd.DataFrame) and not d.empty:
            out=d.copy()
            out["date"]=pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
            out["predicted_return"]=out["predicted_return"]*100
            out["actual_return"]=out["actual_return"]*100
            out["hit"]=out["hit"].map({True:"命中",False:"未命中"})
            if "segment" in out:
                out["segment"]=out["segment"].map(_segment_label)
            st.dataframe(out.rename(columns={
                "date":"日期",
                "segment":"分段",
                "predicted_return":"預測報酬(%)",
                "actual_return":"實際報酬(%)",
                "hit":"是否命中",
            }),use_container_width=True,hide_index=True)
        else:
            st.info("資料筆數不足，尚無 walk-forward 明細。")

def _render_tech(r,df,ind,fc,ml_p,bt,last,med,fc_c,rsi_v):
    """Technical analysis explanation tab."""
    mh=float(ind["macd_hist"].iloc[-1])
    mp2=float(ind["macd_hist"].iloc[-2]) if len(ind["macd_hist"])>1 else mh
    m5=float(ind["ma5"].iloc[-1]); m20=float(ind["ma20"].iloc[-1])
    m60=float(ind["ma60"].iloc[-1])
    bbu=float(ind["bb_up"].iloc[-1]); bbd=float(ind["bb_dn"].iloc[-1])
    bpos=(last-bbd)/max(bbu-bbd,1e-9)
    bw=float(ind["bb_width"].iloc[-1])
    bw2=float(ind["bb_width"].iloc[-2]) if len(ind["bb_width"])>1 else bw
    vr=float(df["Volume"].iloc[-1])/max(float(ind["vol_ma20"].iloc[-1]),1)
    prob=ml_p.get("prob_up",.5)

    rows=[]
    # RSI
    if rsi_v>=75:
        rows.append(("🔴","RSI 超買",f"RSI **{rsi_v:.0f}** 進入深度超買區（>75），短線獲利了結壓力明顯，"
            "歷史統計顯示此區域後續修正機率較高。"))
    elif rsi_v>=65:
        rows.append(("🟡","RSI 偏高",f"RSI **{rsi_v:.0f}** 偏高（65-75），動能尚強但留意高檔疲態。"))
    elif rsi_v<=25:
        rows.append(("🟢","RSI 深度超賣",f"RSI **{rsi_v:.0f}** 深度超賣（<25），"
            "技術反彈機率偏高，但需確認量能支撐。"))
    elif rsi_v<=35:
        rows.append(("🟡","RSI 超賣",f"RSI **{rsi_v:.0f}** 進入超賣區（<35），留意企穩訊號。"))
    else:
        rows.append(("⚪","RSI 中性",f"RSI **{rsi_v:.0f}** 位於中性區間（35-65），無明顯超買超賣訊號。"))

    # MACD
    if mh>=0 and mp2<0:
        rows.append(("🟢","MACD 金叉",
            f"MACD Histogram 由負轉正（**金叉**），為短線偏多訊號。"
            f"Histogram 目前值 {mh:+.3f}，建議觀察後續柱狀是否持續放大。"))
    elif mh<0 and mp2>=0:
        rows.append(("🔴","MACD 死叉",
            f"MACD Histogram 由正轉負（**死叉**），短線偏空訊號。"
            f"當前值 {mh:+.3f}，留意跌勢是否加速。"))
    elif mh>0:
        rows.append(("🟢","MACD 多頭",
            f"MACD Histogram 持續為正（{mh:+.3f}），多頭動能維持中。"))
    else:
        rows.append(("🔴","MACD 空頭",
            f"MACD Histogram 持續為負（{mh:+.3f}），空頭動能主導。"))

    # Bollinger Bands
    if bw<bw2*0.85:
        rows.append(("⚡","布林帶收縮",
            "布林帶顯著收縮，波動壓縮可能醞釀大方向突破。"
            "通常收縮後的第一根大K棒方向為後續趨勢方向。"))
    elif bpos>0.9:
        rows.append(("⚠","貼近布林上軌",
            f"收盤價貼近布林上軌（位置 {bpos:.0%}），注意短線回測中軌壓力。"))
    elif bpos<0.1:
        rows.append(("🟢","貼近布林下軌",
            f"收盤價貼近布林下軌（位置 {bpos:.0%}），可能出現技術性反彈。"))
    else:
        rows.append(("ℹ","布林帶正常",
            f"收盤價位於布林帶 {bpos:.0%} 位置，帶寬正常。"))

    # MA alignment
    if m5>m20>m60:
        rows.append(("📈","均線多頭排列",
            f"MA5({m5:.1f}) > MA20({m20:.1f}) > MA60({m60:.1f})，"
            "均線呈多頭排列，趨勢向上。"))
    elif m5<m20<m60:
        rows.append(("📉","均線空頭排列",
            f"MA5({m5:.1f}) < MA20({m20:.1f}) < MA60({m60:.1f})，"
            "均線呈空頭排列，趨勢向下。"))
    elif m5>m20:
        rows.append(("🟡","短線偏多",
            f"MA5 > MA20 但 MA20 < MA60，中線整理中，短線偏多但中長線待確認。"))
    else:
        rows.append(("🟡","短線偏空",
            f"MA5 < MA20，短線偏空，等待方向確認。"))

    # Volume
    if vr>=2.5:
        rows.append(("🔊","成交量爆增",
            f"成交量為 20日均量的 **{vr:.1f}x**，資金大幅流入，留意主力動向。"))
    elif vr>=1.5:
        rows.append(("🔊","成交量放大",
            f"成交量為均量的 **{vr:.1f}x**，行情活躍。"))
    elif vr<=0.4:
        rows.append(("🔇","成交量極度萎縮",
            f"成交量僅均量的 **{vr:.1f}x**，市場觀望，行情可信度低。"))
    elif vr<=0.7:
        rows.append(("🔇","成交量萎縮",
            f"成交量為均量的 **{vr:.1f}x**，動能不足。"))

    # Forecast
    fc_col=OK if fc_c>0 else ER
    rows.append(("🎯","趨勢預測",
        f"Monte Carlo 模擬預測 {r['forecast_days']} 日後中位價 **{med:,.2f}**，"
        f"較現價 <span style='color:{fc_col}'><b>{fc_c:+.1f}%</b></span>。"
        f"漂移偏差 drift={r.get('drift_bias',0):+.5f}（由分析權重決定）。"))

    # ML
    if prob!=.5:
        pc=OK if prob>=.6 else(ER if prob<=.4 else WA)
        rows.append(("🤖","ML 模型預測",
            f"GradBoost+XGBoost 融合模型預測上漲機率 "
            f"<span style='color:{pc}'><b>{prob*100:.1f}%</b></span>（{ml_p.get('label','')}）。"
            f"OOS 準確率 {r.get('ml_stats',{}).get('oos_accuracy',0)*100:.0f}%。"))

    # Backtest
    if bt.get("n_trades",0)>0:
        bc=OK if bt.get("hit_rate",.5)>=.5 else ER
        rows.append(("📊","回測命中率",
            f"過去 {bt['n_trades']} 筆信號方向命中率 "
            f"<span style='color:{bc}'><b>{bt['hit_rate_text']}</b></span>，"
            f"含交易成本淨報酬 {bt.get('avg_return_after_cost',0):+.2f}%，"
            f"Sharpe {bt.get('sharpe',0):.2f}。"))

    for icon,title,desc in rows:
        with st.expander(title, expanded=False):
            st.markdown(desc, unsafe_allow_html=True)

# ── Batch ──────────────────────────────────────────────────────────────────
def batch():
    wl=st.session_state.watchlist
    if not wl: st.warning("自選股為空"); return
    st.markdown(f"## 批次分析 {len(wl)} 支股票")
    p=st.progress(0); rows=[]
    for i,code in enumerate(wl):
        p.progress((i+1)/len(wl),text=f"{i+1}/{len(wl)}: {code}")
        try:
            r=run_analysis(code,
                lookback_years=st.session_state.lookback_years,
                forecast_days=st.session_state.forecast_days,
                weights=st.session_state.weights,
                train_ratio=st.session_state.train_ratio,
                api_key=read_marketdata_key(st.session_state.get("api_key","") or _streamlit_secret_key()))
            r["weights"]=st.session_state.weights
            df_=r["df"]; fc_=r["forecast"]; ml_p_=r.get("ml_predict",{})
            bt_=r.get("backtest",{})
            last_=float(df_["Close"].iloc[-1])
            med_=float(fc_["median"][-1]); chg_=(med_-last_)/last_*100
            prob_=ml_p_.get("prob_up",.5); s_=calc_score(r)
            if s_>=1.5: act="偏多·可試單"
            elif s_>=.5: act="偏多觀察"
            elif s_<=-1.5: act="偏空·減碼"
            elif s_<=-.5: act="偏空觀察"
            else: act="中性觀望"
            bhr_=bt_.get("hit_rate",.5) if bt_.get("n_trades",0)>0 else .5
            cs_=(abs(prob_-.5)*2+abs(bhr_-.5)*2)/2
            conf_="高" if cs_>=.35 else("中" if cs_>=.18 else "低")
            rows.append({"代碼":r["symbol"].split(".")[0],"名稱":r["name"],
                "建議":act,"可信度":conf_,"現價":f"{last_:,.2f}",
                "ML%":f"{prob_*100:.1f}%","預測%":f"{chg_:+.1f}%",
                "命中率":bt_.get("hit_rate_text","N/A")})
        except Exception as e:
            rows.append({"代碼":code,"名稱":code,"建議":"錯誤","可信度":"—",
                "現價":"—","ML%":"—","預測%":"—","命中率":str(e)[:40]})
    p.empty()
    df_out=pd.DataFrame(rows)
    st.dataframe(df_out,use_container_width=True,hide_index=True)
    csv_bytes = df_out.to_csv(index=False).encode("utf-8-sig")
    st.download_button("💾 下載 CSV", csv_bytes,
        file_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv; charset=utf-8")
    try:
        xlsx_io = io.BytesIO()
        with pd.ExcelWriter(xlsx_io, engine="openpyxl") as writer:
            df_out.to_excel(writer, index=False, sheet_name="批次分析")
        st.download_button("📗 下載 Excel（避免中文亂碼）", xlsx_io.getvalue(),
            file_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        st.caption("若 CSV 在 Excel 顯示亂碼，請用資料匯入選 UTF-8，或安裝 openpyxl 後下載 Excel 檔。")

# ── Main ───────────────────────────────────────────────────────────────────
sym,abtn = sidebar()

if st.session_state.get("intraday_auto_refresh") and st.session_state.get("result") is not None:
    if st_autorefresh is not None:
        st_autorefresh(interval=10_000, key="intraday_autorefresh_tick")
    else:
        st.warning("若要使用 10 秒自動更新，requirements.txt 需包含 streamlit-autorefresh。")

if st.session_state.get("_show_reco", False):
    render_daily_recommendations()
    st.divider()

# Handle watchlist load
if st.session_state.get("_pending_sym",""):
    _ps=st.session_state._pending_sym
    st.session_state._pending_sym=""
    with st.spinner(f"分析 {_ps}…"): analyze(_ps)
    st.rerun()

# Handle search
_s=st.session_state.get("si","").strip().upper()
if (abtn or st.session_state.get("_trigger")) and _s:
    st.session_state._trigger=False
    with st.spinner(f"分析 {_s}…"): analyze(_s)
    st.rerun()

# Render
if st.session_state.get("_do_batch"):
    st.session_state._do_batch=False
    try: batch()
    except Exception as e:
        st.warning("批次錯誤：" + friendly_error_message(e))
elif st.session_state.result:
    try: show(st.session_state.result)
    except Exception as e:
        st.warning("顯示結果時發生問題：" + friendly_error_message(e))
        with st.expander("查看技術細節", expanded=False):
            st.code(str(e))
else:
    # Landing page
    st.markdown(
        f"<div style='text-align:center;padding:50px 20px'>"
        f"<div style='font-size:3.5rem'>📈</div>"
        f"<h1 style='font-size:1.8rem;color:{AC}'>Stock Analyzer Pro</h1>"
        f"<p style='color:{DM};max-width:480px;margin:0 auto'>"
        f"台灣股市智能分析系統<br>左側輸入代碼，按 Enter 或 🔍 開始</p>"
        f"<p style='color:{DM};font-size:.82rem;margin-top:10px'>"
        f"支援：上市 · 上櫃 · ETF · 槓桿反向 · 受益憑證 · 美股</p>"
        f"</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("#### 快速選股")
    ex=[("台積電","2330"),("元大台灣50","0050"),
        ("國泰永續高股息","00878"),("元大台灣50正2","00631L"),
        ("聯發科","2454"),("長榮","2603")]
    cols=st.columns(len(ex))
    for col,(n,c) in zip(cols,ex):
        if col.button(f"{n}\n({c})",use_container_width=True):
            with st.spinner(f"分析 {c}…"): analyze(c)
            st.rerun()
