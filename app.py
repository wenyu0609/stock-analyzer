"""
Stock Analyzer Pro — Streamlit Web v3
Fixes: full theme CSS / mobile responsive / MACD-RSI collapsible+popout /
       smart chart zoom (scroll wheel + 2-finger pinch) / no accidental zoom on swipe
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings, io
from datetime import datetime

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
    _cjk, SKLEARN_OK, XGB_OK, TORCH_OK, YF_OK,
)

# ── Session state ──────────────────────────────────────────────────────────
_DEF = dict(
    result=None, batch_results=[],
    watchlist=["2330","2454","0050","00878","2317"],
    weights={"technical":40,"ml":35,"news":15,"fundamental":10},
    forecast_days=30, lookback_years=3, train_ratio=0.8,
    chart_style="K棒", show_bb=True, show_sr=True, show_band=True,
    show_macd=True, show_rsi=True,
    theme="暗色", names_loaded=False, _trigger=False, _pending_sym="",
)
for k,v in _DEF.items():
    if k not in st.session_state: st.session_state[k]=v

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

# ── CSS ────────────────────────────────────────────────────────────────────
# Use string concatenation — avoids ALL f-string brace conflicts
st.markdown("""<style>
/* ── Reset & Base ── */
html, body, [class*="css"], [class*="st-"], div, span, p, label,
input, textarea, select, button, h1, h2, h3, h4, h5, h6, li, a,
[data-testid], [data-baseweb] {
    font-family: 'Microsoft JhengHei', 'PingFang TC', 'Noto Sans TC', sans-serif !important;
}
</style>""" +
"""<style>
/* ── App background ── */
[data-testid="stAppViewContainer"] { background:"""+BG+""" !important; }
[data-testid="stSidebar"] { background:"""+SB+""" !important; }
[data-testid="stHeader"] { background:"""+BG+""" !important; }
section[data-testid="stSidebar"] > div { background:"""+SB+""" !important; }

/* ── All text ── */
body, p, span, div, label, li, td, th, caption {
    color:"""+TX+""" !important;
}
h1 { color:"""+AC+""" !important; font-size:1.4rem !important; font-weight:700 !important; }
h2, h3, h4 { color:"""+TX+""" !important; }

/* ── Sidebar text ── */
[data-testid="stSidebarContent"] { color:"""+TX+""" !important; }
[data-testid="stSidebarContent"] * { color:"""+TX+""" !important; }
[data-testid="stSidebarContent"] h4 { color:"""+AC+""" !important; }

/* ── Buttons ── */
.stButton > button {
    background:"""+CD+""" !important;
    border:1px solid """+BD+""" !important;
    color:"""+TX+""" !important;
    border-radius:8px !important;
    transition: all .15s !important;
}
.stButton > button:hover {
    border-color:"""+AC+""" !important;
    color:"""+AC+""" !important;
}
.stButton > button[kind="primary"] {
    background:"""+AC+""" !important;
    border-color:"""+AC+""" !important;
    color:#ffffff !important;
}
.stButton > button[kind="primary"]:hover {
    opacity: 0.9 !important;
    color:#ffffff !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input {
    background:"""+CD+""" !important;
    border:1px solid """+BD+""" !important;
    color:"""+TX+""" !important;
    border-radius:8px !important;
}
.stTextInput > div > div > input::placeholder { color:"""+DM+""" !important; }
.stTextInput > div > div > input:focus { border-color:"""+AC+""" !important; }
.stNumberInput > div > div > input {
    background:"""+CD+""" !important;
    border:1px solid """+BD+""" !important;
    color:"""+TX+""" !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background:"""+CD+""" !important;
    border:1px solid """+BD+""" !important;
    color:"""+TX+""" !important;
    border-radius:8px !important;
}
.stSelectbox > div > div > div { color:"""+TX+""" !important; }
[data-baseweb="select"] > div { background:"""+CD+""" !important; color:"""+TX+""" !important; }
[data-baseweb="select"] span { color:"""+TX+""" !important; }
[role="option"] {
    background:"""+CD+""" !important;
    color:"""+TX+""" !important;
}
[role="option"]:hover { background:"""+BD+""" !important; }

/* ── Sliders ── */
.stSlider > div { color:"""+TX+""" !important; }
.stSlider label { color:"""+TX+""" !important; }
[data-testid="stSlider"] { color:"""+TX+""" !important; }

/* ── Checkboxes & Radio ── */
.stCheckbox > label { color:"""+TX+""" !important; }
.stRadio > div > label { color:"""+TX+""" !important; }
.stRadio label { color:"""+TX+""" !important; }

/* ── Metrics ── */
[data-testid="stMetricValue"] { color:"""+TX+""" !important; font-size:1.2rem !important; }
[data-testid="stMetricLabel"] { color:"""+DM+""" !important; }
[data-testid="stMetricLabel"] div { color:"""+DM+""" !important; }
[data-testid="stMetricDelta"] { color:"""+DM+""" !important; }
[data-testid="stMetric"] { background:"""+CD+"""; border:1px solid """+BD+"""; border-radius:10px; padding:10px; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background:"""+CD+""" !important;
    border:1px solid """+BD+""" !important;
    border-radius:8px !important;
}
[data-testid="stExpander"] summary { color:"""+TX+""" !important; }
[data-testid="stExpander"] summary p { color:"""+TX+""" !important; }
[data-testid="stExpander"] p { color:"""+TX+""" !important; }

/* ── Tabs ── */
[data-baseweb="tab"] span { color:"""+DM+""" !important; }
[aria-selected="true"] span { color:"""+AC+""" !important; }
[data-baseweb="tab-highlight"] { background:"""+AC+""" !important; }
[data-baseweb="tab-border"] { background:"""+BD+""" !important; }

/* ── Caption / small text ── */
[data-testid="stCaptionContainer"] { color:"""+DM+""" !important; }
[data-testid="stCaptionContainer"] p { color:"""+DM+""" !important; }
.stCaption { color:"""+DM+""" !important; }

/* ── Progress ── */
[data-testid="stProgressBar"] > div { background:"""+AC+""" !important; }
[data-testid="stProgressBar"] { background:"""+BD+""" !important; }

/* ── Download buttons ── */
[data-testid="stDownloadButton"] button {
    background:"""+CD+""" !important;
    border:1px solid """+BD+""" !important;
    color:"""+TX+""" !important;
    border-radius:8px !important;
}

/* ── Divider ── */
hr { border-top:1px solid """+BD+""" !important; }

/* ── Custom cards ── */
.metric-card {
    background:"""+CD+""";
    border:1px solid """+BD+""";
    border-radius:10px;
    padding:12px 16px;
    margin-bottom:8px;
}
.metric-card .lbl { color:"""+DM+"""; font-size:.74rem; margin-bottom:2px; }
.metric-card .val { color:"""+TX+"""; font-size:1.2rem; font-weight:700; }
.metric-card .sub { color:"""+AC+"""; font-size:.8rem; }

/* ── Badges ── */
.bull { background:"""+("rgba(5,150,105,0.12)" if D else "#dcfce7")+"""; color:"""+OK+""";
    padding:3px 12px; border-radius:20px; font-weight:700; display:inline-block; }
.bear { background:"""+("rgba(239,68,68,0.12)" if D else "#fee2e2")+"""; color:"""+ER+""";
    padding:3px 12px; border-radius:20px; font-weight:700; display:inline-block; }
.neut { background:"""+("rgba(245,158,11,0.12)" if D else "#fef9c3")+"""; color:"""+WA+""";
    padding:3px 12px; border-radius:20px; font-weight:700; display:inline-block; }
.bhi  { background:"""+("rgba(108,142,245,0.12)" if D else "#ede9fe")+"""; color:"""+AC+""";
    padding:2px 10px; border-radius:20px; display:inline-block; }
.blo  { background:"""+("rgba(239,68,68,0.12)" if D else "#fee2e2")+"""; color:"""+ER+""";
    padding:2px 10px; border-radius:20px; display:inline-block; }

/* ── News items ── */
.ni { border-left:3px solid """+BD+"""; padding:5px 10px; margin:5px 0;
    font-size:.84rem; color:"""+DM+"""; }
.ni a { color:"""+DM+"""; text-decoration:none; }
.ni:hover { border-color:"""+AC+"""; }

/* ── Mobile responsive ── */
@media (max-width: 768px) {
    h1 { font-size:1.2rem !important; }
    .metric-card .val { font-size:1rem !important; }
    [data-testid="stMetricValue"] { font-size:1rem !important; }
    /* Prevent page scroll hijack when swiping chart on mobile */
    .js-plotly-plot { touch-action: pan-x pan-y !important; }
}

/* ── Prevent plotly chart from capturing scroll on mobile ── */
.stPlotlyChart { touch-action: pan-y !important; }

/* ── Markdown text ── */
[data-testid="stMarkdownContainer"] p { color:"""+TX+""" !important; }
[data-testid="stMarkdownContainer"] span { color:"""+TX+""" !important; }
[data-testid="stMarkdownContainer"] li { color:"""+TX+""" !important; }

</style>""", unsafe_allow_html=True)

# ── Mobile chart scroll fix (JavaScript) ─────────────────────────────────
# Inject JS to prevent chart from hijacking mobile scroll
st.markdown("""
<script>
// Prevent chart from blocking mobile scroll
document.addEventListener('DOMContentLoaded', function() {
    function fixChartScroll() {
        var charts = document.querySelectorAll('.js-plotly-plot');
        charts.forEach(function(chart) {
            chart.addEventListener('touchstart', function(e) {
                if (e.touches.length === 1) {
                    // Single finger = page scroll, not chart zoom
                    this._singleTouch = true;
                }
            }, {passive: true});
        });
    }
    setTimeout(fixChartScroll, 2000);
    var observer = new MutationObserver(fixChartScroll);
    observer.observe(document.body, {childList: true, subtree: true});
});
</script>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────
def badge(cls, t): return f"<span class='{cls}'>{t}</span>"

def calc_score(r):
    ind=r["indicators"]; ml=r.get("ml_predict",{}); ns=r.get("news_sentiment",{})
    fund=r.get("fundamental",{}); w=r.get("weights",st.session_state.weights)
    wt=w.get("technical",40)/100; wm=w.get("ml",35)/100
    wn=w.get("news",15)/100;      wf=w.get("fundamental",10)/100
    try:
        mh=float(ind["macd_hist"].iloc[-1]); rsi=float(ind["rsi14"].iloc[-1])
        ms=float(ind["macd_hist"].std()) if len(ind["macd_hist"])>5 else 1.0
        t=float(np.clip(mh/max(ms,1e-9),-1,1))*.5+float(np.clip((50-rsi)/50,-1,1))*(-.3)
    except: t=0.0
    ml_s=(ml.get("prob_up",.5)-.5)*2
    ns_s=.6 if ns.get("label")=="正面" else(-.6 if ns.get("label")=="負面" else 0.)
    fs=0.
    if fund.get("pe_ratio") and 0<fund["pe_ratio"]<15: fs+=.5
    if fund.get("pe_ratio") and fund["pe_ratio"]>40:   fs-=.5
    if fund.get("roe") and fund["roe"]>.15: fs+=.5
    return float(np.clip(t*wt*2+ml_s*wm*2+ns_s*wn*2+fs*wf*2,-4,4))

def action_badge(s):
    if s>=1.5:   return badge("bull","偏多·可試單")
    if s>=0.5:   return badge("bull","偏多觀察")
    if s<=-1.5:  return badge("bear","偏空·減碼")
    if s<=-0.5:  return badge("bear","偏空觀察")
    return badge("neut","中性觀望")

# ── Chart builder ──────────────────────────────────────────────────────────
def build_chart(r, style="K棒", bb=True, sr_on=True, band=True,
                show_macd=True, show_rsi=True):
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

    # MA lines
    fig.add_trace(go.Scatter(x=df.index,y=ind["ma5"],
        line=dict(color="rgba(108,142,245,0.7)",width=1),name="MA5"),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=ind["ma20"],
        line=dict(color="rgba(251,191,36,0.7)",width=1),name="MA20"),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=ind["ma60"],
        line=dict(color="rgba(248,113,113,0.7)",width=1),name="MA60"),row=1,col=1)

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
    chart_height = 580 if rows==3 else (440 if rows==2 else 380)
    fig.update_layout(
        height=chart_height,
        paper_bgcolor=CPP, plot_bgcolor=CBG,
        font=dict(color=CTX,size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)",bordercolor=BD,
            font=dict(size=10,color=CTX),x=0.01,y=0.99),
        xaxis_rangeslider_visible=False,
        margin=dict(l=0,r=0,t=30,b=0),
        hovermode="x unified",
        # Scroll zoom enabled — mouse wheel to zoom on desktop
        dragmode="pan",  # default drag = pan (move chart)
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
        xaxis=dict(gridcolor=CGR),yaxis=dict(gridcolor=CGR),
        title=dict(text=title,font=dict(color=TX,size=13)))
    st.plotly_chart(fig,use_container_width=True,
        config={"displaylogo":False,"scrollZoom":True,
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
        with st.expander("📊 圖表顯示",expanded=False):
            st.session_state.chart_style=st.radio("",["K棒","曲線"],
                horizontal=True,
                index=0 if st.session_state.chart_style=="K棒" else 1)
            c1,c2=st.columns(2)
            with c1:
                st.session_state.show_bb   =st.checkbox("布林帶",  st.session_state.show_bb)
                st.session_state.show_sr   =st.checkbox("支撐壓力",st.session_state.show_sr)
            with c2:
                st.session_state.show_band =st.checkbox("預測信賴帶",st.session_state.show_band)
            st.markdown("**副圖**")
            c1,c2=st.columns(2)
            with c1:
                st.session_state.show_macd=st.checkbox("MACD",st.session_state.show_macd)
            with c2:
                st.session_state.show_rsi =st.checkbox("RSI", st.session_state.show_rsi)
            st.caption("💡 取消勾選可隱藏副圖；可在分析結果頁獨立放大")

        # Analysis params
        with st.expander("⚙ 分析參數",expanded=False):
            st.session_state.forecast_days =st.slider("預測天數",5,90,st.session_state.forecast_days)
            st.session_state.lookback_years=st.slider("回溯年數",1,10,st.session_state.lookback_years)
            st.session_state.train_ratio   =st.slider("ML訓練佔比",.5,.95,st.session_state.train_ratio,.05)

        # Weight settings
        with st.expander("⚖ 分析權重",expanded=False):
            st.caption("四項總計須為 100%，影響趨勢預測方向")
            w=st.session_state.weights
            wt=st.slider("技術指標%",0,100,w["technical"],key="wt")
            wm=st.slider("ML/NN%",   0,100,w["ml"],       key="wm")
            wn=st.slider("新聞情緒%",0,100,w["news"],     key="wn")
            wf=st.slider("基本面%",  0,100,w["fundamental"],key="wf")
            tot=wt+wm+wn+wf
            c=OK if tot==100 else ER
            st.markdown(
                f"<span style='color:{c}'>{'✅' if tot==100 else '⚠'} 總計 {tot}%</span>",
                unsafe_allow_html=True)
            if st.button("▶ Apply 套用",use_container_width=True,disabled=(tot!=100)):
                st.session_state.weights={
                    "technical":wt,"ml":wm,"news":wn,"fundamental":wf}
                if st.session_state.result:
                    r_=st.session_state.result
                    nd=compute_drift_bias(r_["indicators"],r_.get("ml_predict",{}),{},
                        r_.get("news_sentiment",{}),r_.get("fundamental",{}),
                        st.session_state.weights)
                    nf=simple_forecast(r_["df"],days=r_["forecast_days"],
                        n_paths=150,drift_bias=nd)
                    r_["forecast"]=nf; r_["drift_bias"]=nd
                    r_["weights"]=st.session_state.weights
                st.rerun()

        st.divider()
        st.caption(f"{'GB+XGB' if XGB_OK else 'GB'}{'+LSTM' if TORCH_OK else ''}")
    return sym, abtn

# ── Run analysis ───────────────────────────────────────────────────────────
def analyze(sym):
    if not sym: return
    p=st.progress(0,text="準備…")
    def cb(s,t,m): p.progress(s/t,text=f"[{s}/{t}] {m}")
    try:
        r=run_analysis(sym.strip().upper(),
            lookback_years=st.session_state.lookback_years,
            forecast_days=st.session_state.forecast_days,
            weights=st.session_state.weights,
            train_ratio=st.session_state.train_ratio,
            progress_callback=cb)
        r["weights"]=st.session_state.weights
        st.session_state.result=r; p.empty()
    except Exception as e:
        p.empty(); st.error(f"❌ {type(e).__name__}: {e}")

# ── PDF ────────────────────────────────────────────────────────────────────
def to_pdf(r,fig):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image as RI
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib import colors
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        import os
        fn="Helvetica"
        for fp in ["C:/Windows/Fonts/msjh.ttc","/System/Library/Fonts/PingFang.ttc",
                   "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"]:
            if os.path.exists(fp):
                try: pdfmetrics.registerFont(TTFont("CJK",fp,subfontIndex=0)); fn="CJK"; break
                except: pass
    except ImportError: return b""
    buf=io.BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=A4,leftMargin=2*cm,rightMargin=2*cm,
        topMargin=2*cm,bottomMargin=2*cm)
    H1=ParagraphStyle("H1",fontSize=16,fontName=fn,
        textColor=colors.HexColor("#3b5bdb"),spaceAfter=8)
    H2=ParagraphStyle("H2",fontSize=12,fontName=fn,
        textColor=colors.HexColor("#111827"),spaceBefore=12,spaceAfter=6)
    NM=ParagraphStyle("NM",fontSize=9,fontName=fn,
        textColor=colors.HexColor("#374151"),leading=14)
    elems=[]
    name=r["name"]; sym=r["symbol"]; df=r["df"]; fc=r["forecast"]
    ml_p=r.get("ml_predict",{}); bt=r.get("backtest",{})
    ind=r["indicators"]
    last=float(df["Close"].iloc[-1]); med=float(fc["median"][-1])
    chg=(med-last)/last*100
    elems.append(Paragraph(f"股票分析報告：{name} ({sym.split('.')[0]})",H1))
    elems.append(Paragraph(f"產生時間：{datetime.now().strftime('%Y-%m-%d %H:%M')}",NM))
    elems.append(Spacer(1,.3*cm))
    try:
        png=fig.to_image(format="png",width=1400,height=700,scale=1.5)
        elems.append(RI(io.BytesIO(png),width=16*cm,height=8*cm))
    except: pass
    elems.append(Spacer(1,.3*cm))
    elems.append(Paragraph("分析摘要",H2))
    s=calc_score(r)
    if s>=1.5: act="偏多·可試單"
    elif s>=.5: act="偏多觀察"
    elif s<=-1.5: act="偏空·減碼"
    elif s<=-.5: act="偏空觀察"
    else: act="中性觀望"
    atr=float(ind["atr14"].iloc[-1]) if not pd.isna(ind["atr14"].iloc[-1]) else last*.02
    td=[["項目","數值","項目","數值"],
        ["現價",f"{last:,.2f}","建議",act],
        [f"{r['forecast_days']}日預測",f"{med:,.2f}({chg:+.1f}%)",
         "ML機率",f"{ml_p.get('prob_up',.5)*100:.1f}%"],
        ["RSI",f"{float(ind['rsi14'].iloc[-1]):.1f}",
         "命中率",bt.get("hit_rate_text","N/A")],
        ["停損",f"{last-atr*1.5:,.2f}","停利",f"{last+atr*2.5:,.2f}"]]
    t=Table(td,colWidths=[3.5*cm,4*cm,3.5*cm,4*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#3b5bdb")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,-1),fn),("FONTSIZE",(0,0),(-1,-1),9),
        ("GRID",(0,0),(-1,-1),.5,colors.HexColor("#e5e7eb")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f8fafc"),colors.white]),
        ("ALIGN",(1,0),(-1,-1),"CENTER"),("PADDING",(0,0),(-1,-1),4)]))
    elems.append(t)
    if bt.get("n_trades",0)>0:
        elems.append(Paragraph("回測統計（含交易成本）",H2))
        bd=[["筆數",str(bt["n_trades"]),"命中率",bt["hit_rate_text"]],
            ["淨報酬",f"{bt.get('avg_return_after_cost',0):+.2f}%",
             "Sharpe",f"{bt.get('sharpe',0):.2f}"],
            ["最大回撤",f"{bt.get('max_drawdown',0):.1f}%",
             "Profit Factor",f"{bt.get('profit_factor',0):.2f}"]]
        b=Table([["指標","數值","指標","數值"]]+bd,colWidths=[3.5*cm,4*cm,3.5*cm,4*cm])
        b.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#059669")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTNAME",(0,0),(-1,-1),fn),("FONTSIZE",(0,0),(-1,-1),9),
            ("GRID",(0,0),(-1,-1),.5,colors.HexColor("#e5e7eb")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f0fdf4"),colors.white]),
            ("ALIGN",(1,0),(-1,-1),"CENTER"),("PADDING",(0,0),(-1,-1),4)]))
        elems.append(b)
    elems.append(Spacer(1,.5*cm))
    elems.append(Paragraph("⚠ 本報告僅供研究參考，不構成投資建議，操作須自行評估風險。",NM))
    doc.build(elems); return buf.getvalue()

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

    # ── Chart controls row ──
    cc1,cc2,cc3,cc4,cc5 = st.columns([1,1,1,1,2])
    with cc1:
        if st.button("MACD " + ("✅" if st.session_state.show_macd else "⬜"),
                     use_container_width=True, help="切換 MACD 副圖"):
            st.session_state.show_macd = not st.session_state.show_macd
            st.rerun()
    with cc2:
        if st.button("RSI " + ("✅" if st.session_state.show_rsi else "⬜"),
                     use_container_width=True, help="切換 RSI 副圖"):
            st.session_state.show_rsi = not st.session_state.show_rsi
            st.rerun()
    with cc3:
        show_macd_pop = st.button("📊 MACD ↗", use_container_width=True,
                                   help="獨立放大 MACD 圖")
    with cc4:
        show_rsi_pop  = st.button("📈 RSI ↗",  use_container_width=True,
                                   help="獨立放大 RSI 圖")
    with cc5:
        st.caption(
            "🖥 滾輪縮放・左鍵平移  |  📱 雙指縮放・單指平移")

    # Main chart
    fig=build_chart(r,
        style    =st.session_state.chart_style,
        bb       =st.session_state.show_bb,
        sr_on    =st.session_state.show_sr,
        band     =st.session_state.show_band,
        show_macd=st.session_state.show_macd,
        show_rsi =st.session_state.show_rsi)

    # Chart config:
    # - scrollZoom=True: mouse wheel zooms on desktop
    # - modeBarButtons: keep zoom, pan, reset; remove unnecessary ones
    # - dragmode="pan" set in layout (left-click drags/pans)
    chart_config={
        "displaylogo": False,
        "scrollZoom": True,          # desktop: scroll to zoom
        "doubleClick": "reset",      # double-click resets view
        "modeBarButtonsToRemove": [
            "autoScale2d","lasso2d","select2d","toImage"
        ],
        "modeBarButtonsToAdd": ["resetScale2d"],
    }
    st.plotly_chart(fig, use_container_width=True, config=chart_config)

    # Subchart popouts
    if show_macd_pop:
        with st.expander("📊 MACD 詳細圖（點此收合）", expanded=True):
            render_subchart_popout(r,"MACD")
    if show_rsi_pop:
        with st.expander("📈 RSI(14) 詳細圖（點此收合）", expanded=True):
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
            pdf=to_pdf(r,fig)
            if pdf:
                st.download_button("📄 PDF", pdf,
                    file_name=f"{sym.split('.')[0]}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf", use_container_width=True)
            else: st.caption("PDF需reportlab")
        except: st.caption("PDF需reportlab")

    # ── Analysis tabs ──
    t1,t2,t3,t4,t5=st.tabs(["📊 ML / 回測","🔍 技術解讀","📋 基本面","📰 新聞","📍 關鍵價位"])

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

    with t2:
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
        with st.expander(f"{icon} {title}", expanded=False):
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
                train_ratio=st.session_state.train_ratio)
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
    csv=df_out.to_csv(index=False,encoding="utf-8-sig")
    st.download_button("💾 下載 CSV",csv,
        file_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv")

# ── Main ───────────────────────────────────────────────────────────────────
sym,abtn = sidebar()

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
        st.error(f"批次錯誤：{e}")
elif st.session_state.result:
    try: show(st.session_state.result)
    except Exception as e:
        import traceback
        st.error(f"顯示錯誤：{type(e).__name__}: {e}")
        st.code(traceback.format_exc())
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
