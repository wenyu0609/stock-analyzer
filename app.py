"""
Stock Analyzer Pro — Streamlit Web Version v2
全功能：分析/回測/PNG/PDF匯出/批次/深色淺色主題/Enter搜尋/自選股修正
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
    theme="暗色", names_loaded=False, _trigger=False,
)
for k,v in _DEF.items():
    if k not in st.session_state: st.session_state[k]=v

if not st.session_state.names_loaded:
    with st.spinner("載入股票名稱資料庫…"):
        _load_twse_bulk(); _load_tpex_bulk()
    st.session_state.names_loaded=True

# ── Theme colours ──────────────────────────────────────────────────────────
D = st.session_state.theme=="暗色"
BG  = "#0f1120" if D else "#f5f7fc"
SB  = "#1a1d2e" if D else "#ffffff"
CD  = "#1e2240" if D else "#ffffff"
TX  = "#e2e4f0" if D else "#1a1d2e"
DM  = "#8890aa" if D else "#6b7280"
BD  = "#2d3154" if D else "#e2e8f0"
AC  = "#6c8ef5" if D else "#4f6ef5"
OK  = "#34d399" if D else "#059669"
ER  = "#f87171" if D else "#dc2626"
WA  = "#fbbf24" if D else "#d97706"
CBG = "#14172a" if D else "#ffffff"
CPP = "#0f1120" if D else "#f5f7fc"
CGR = "#2d3154" if D else "#e2e8f0"
CTX = "#8890aa" if D else "#6b7280"

st.markdown(f"""<style>
[data-testid="stAppViewContainer"]{{background:{BG}}}
[data-testid="stSidebar"]{{background:{SB}}}
[data-testid="stHeader"]{{background:{BG}}}
html,body,[class*="css"]{{font-family:'Microsoft JhengHei','PingFang TC',sans-serif;color:{TX}}}
h1{{color:{AC};font-size:1.45rem;font-weight:700;margin-bottom:2px}}
.stButton>button{{background:{CD};border:1px solid {BD};color:{TX};border-radius:8px;transition:.15s}}
.stButton>button:hover{{border-color:{AC};color:{AC}}}
.stTextInput>div>div>input{{background:{CD};border:1px solid {BD};color:{TX};border-radius:8px}}
.stTextInput>div>div>input:focus{{border-color:{AC}}}
.card{{background:{CD};border:1px solid {BD};border-radius:10px;padding:12px 16px;margin-bottom:8px}}
.lbl{{color:{DM};font-size:.74rem;margin-bottom:2px}}
.val{{color:{TX};font-size:1.2rem;font-weight:700}}
.sub{{color:{AC};font-size:.8rem}}
.bull{{background:{"#05966918" if D else "#dcfce7"};color:{OK};padding:3px 12px;border-radius:20px;font-weight:700;display:inline-block}}
.bear{{background:{"#ef444418" if D else "#fee2e2"};color:{ER};padding:3px 12px;border-radius:20px;font-weight:700;display:inline-block}}
.neut{{background:{"#f59e0b18" if D else "#fef9c3"};color:{WA};padding:3px 12px;border-radius:20px;font-weight:700;display:inline-block}}
.bhi{{background:{"#6c8ef518" if D else "#ede9fe"};color:{AC};padding:2px 10px;border-radius:20px;display:inline-block}}
.blo{{background:{"#ef444418" if D else "#fee2e2"};color:{ER};padding:2px 10px;border-radius:20px;display:inline-block}}
.ni{{border-left:3px solid {BD};padding:5px 10px;margin:5px 0;font-size:.84rem;color:{DM}}}
.ni a{color:{DM};text-decoration:none}
.ni:hover{border-color:{AC}}
/* ── Streamlit 原生元件文字顏色強制覆蓋 ── */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stText"],
[data-testid="stCaption"],
label,
.stSlider label,
.stCheckbox label,
.stRadio label,
[data-baseweb="select"] *,
[data-baseweb="input"] *,
[data-testid="stSelectbox"] *,
[data-testid="stNumberInput"] *,
[data-testid="stTextInput"] * {
    color:{TX} !important;
}
/* Metric */
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
[data-testid="stMetricDelta"] {
    color:{TX} !important;
}
/* Expander */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] p {
    color:{TX} !important;
}
/* Tab labels */
[data-baseweb="tab"] button,
[data-baseweb="tab"] span {
    color:{TX} !important;
}
/* Selectbox dropdown items */
[data-baseweb="popover"] *,
[role="option"] {
    background:{CD} !important;
    color:{TX} !important;
}
/* Sidebar text */
[data-testid="stSidebarContent"] * {
    color:{TX};
}
[data-testid="stSidebarContent"] h2,
[data-testid="stSidebarContent"] h3,
[data-testid="stSidebarContent"] h4 {
    color:{AC} !important;
}
/* Caption / dim text */
[data-testid="stCaptionContainer"] {
    color:{DM} !important;
}
/* Download button */
[data-testid="stDownloadButton"] button {
    background:{CD};border:1px solid {BD};color:{TX};border-radius:8px;
}
/* Progress bar */
[data-testid="stProgressBar"] > div {
    background:{AC};
}
/* Divider */
hr {border-top:1px solid {BD} !important;}

/* ── Force Streamlit native widget text colours ── */
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
[data-testid="stMetricDelta"],
[data-testid="baseButton-secondary"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
.stSelectbox label, .stSlider label,
.stCheckbox label, .stRadio label,
.stTextInput label, .stNumberInput label,
.stCaption, .stText, p, span, label, div {{
    color: {TX} !important;
}}
[data-testid="stExpander"] summary p {{
    color: {TX} !important;
}}
[data-testid="stSelectbox"] div[data-baseweb="select"] span,
[data-testid="stSelectbox"] div[data-baseweb="select"] div {{
    color: {TX} !important;
    background: {CD} !important;
}}
[data-testid="stMetricValue"] {{
    color: {TX} !important;
    font-size: 1.4rem !important;
}}
[data-testid="stMetricLabel"] div {{
    color: {DM} !important;
}}
</style>""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────
def badge(cls, t): return f"<span class='{cls}'>{t}</span>"

def score(r):
    ind=r["indicators"]; ml=r.get("ml_predict",{}); ns=r.get("news_sentiment",{})
    fund=r.get("fundamental",{}); w=r.get("weights",st.session_state.weights)
    wt=w.get("technical",40)/100; wm=w.get("ml",35)/100
    wn=w.get("news",15)/100; wf=w.get("fundamental",10)/100
    try:
        mh=float(ind["macd_hist"].iloc[-1]); rsi=float(ind["rsi14"].iloc[-1])
        ms=float(ind["macd_hist"].std()) if len(ind["macd_hist"])>5 else 1.0
        t=float(np.clip(mh/max(ms,1e-9),-1,1))*.5+float(np.clip((50-rsi)/50,-1,1))*(-.3)
    except: t=0.0
    ml_s=(ml.get("prob_up",.5)-.5)*2
    ns_s=.6 if ns.get("label")=="正面" else (-.6 if ns.get("label")=="負面" else 0.)
    fs=0.
    if fund.get("pe_ratio") and 0<fund["pe_ratio"]<15: fs+=.5
    if fund.get("pe_ratio") and fund["pe_ratio"]>40: fs-=.5
    if fund.get("roe") and fund["roe"]>.15: fs+=.5
    return float(np.clip(t*wt*2+ml_s*wm*2+ns_s*wn*2+fs*wf*2,-4,4))

def action(s):
    if s>=1.5: return badge("bull","偏多·可試單")
    if s>=0.5: return badge("bull","偏多觀察")
    if s<=-1.5: return badge("bear","偏空·減碼")
    if s<=-0.5: return badge("bear","偏空觀察")
    return badge("neut","中性觀望")

# ── Chart ──────────────────────────────────────────────────────────────────
def chart(r,style="K棒",bb=True,sr_=True,band=True):
    df=r["df"]; ind=r["indicators"]; sr=r["sr"]; fc=r["forecast"]
    name=r["name"]; sym=r["symbol"]
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,row_heights=[.65,.18,.17],
        vertical_spacing=.02,
        subplot_titles=[f"{name} ({sym.split('.')[0]})","MACD","RSI(14)"])
    if style=="K棒":
        fig.add_trace(go.Candlestick(x=df.index,open=df["Open"],high=df["High"],
            low=df["Low"],close=df["Close"],increasing_line_color=OK,
            decreasing_line_color=ER,name="K棒",showlegend=False),row=1,col=1)
    else:
        fig.add_trace(go.Scatter(x=df.index,y=df["Close"],mode="lines",
            line=dict(color=AC,width=1.5),name="收盤"),row=1,col=1)
    if bb:
        fig.add_trace(go.Scatter(x=df.index,y=ind["bb_up"],
            line=dict(color="rgba(108,142,245,0.3)",width=1),name="BB上"),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=ind["bb_dn"],
            line=dict(color="rgba(108,142,245,0.3)",width=1),fill="tonexty",
            fillcolor="rgba(108,142,245,0.06)",name="BB下"),row=1,col=1)
    if sr_:
        fig.add_hline(y=sr["resistance_hi"],line_dash="dot",
            line_color="rgba(248,113,113,0.5)",row=1,col=1,
            annotation_text="壓力",annotation_font_color=ER)
        fig.add_hline(y=sr["support_lo"],line_dash="dot",
            line_color="rgba(52,211,153,0.5)",row=1,col=1,
            annotation_text="支撐",annotation_font_color=OK)
    fd=list(fc["future_dates"])
    if band:
        fig.add_trace(go.Scatter(x=fd,y=fc["upper"],
            line=dict(color="rgba(167,139,250,0)",width=0),name="上限"),row=1,col=1)
        fig.add_trace(go.Scatter(x=fd,y=fc["lower"],
            line=dict(color="rgba(167,139,250,0)",width=0),fill="tonexty",
            fillcolor="rgba(167,139,250,0.1)",name="預測帶"),row=1,col=1)
    fig.add_trace(go.Scatter(
        x=[df.index[-1]]+fd,y=[float(df["Close"].iloc[-1])]+list(fc["median"]),
        mode="lines",line=dict(color="#a78bfa",width=2,dash="dash"),
        name=f"預測"),row=1,col=1)
    hc=[OK if float(v)>=0 else ER for v in ind["macd_hist"]]
    fig.add_trace(go.Bar(x=df.index,y=ind["macd_hist"],marker_color=hc,name="Hist",showlegend=False),row=2,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=ind["macd_line"],line=dict(color=AC,width=1),name="MACD"),row=2,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=ind["macd_signal"],line=dict(color=WA,width=1),name="Signal"),row=2,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=ind["rsi14"],line=dict(color=OK,width=1.5),name="RSI"),row=3,col=1)
    fig.add_hline(y=70,line_dash="dot",line_color="rgba(248,113,113,0.4)",row=3,col=1)
    fig.add_hline(y=30,line_dash="dot",line_color="rgba(52,211,153,0.4)",row=3,col=1)
    fig.update_layout(height=640,paper_bgcolor=CPP,plot_bgcolor=CBG,
        font=dict(color=CTX,size=11),
        legend=dict(bgcolor=CD,bordercolor=BD,font=dict(size=10),x=.01,y=.99),
        xaxis_rangeslider_visible=False,margin=dict(l=0,r=0,t=30,b=0),
        hovermode="x unified")
    for i in range(1,4):
        fig.update_xaxes(gridcolor=CGR,row=i,col=1)
        fig.update_yaxes(gridcolor=CGR,row=i,col=1)
    return fig

# ── Sidebar ────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        c1,c2=st.columns([3,1])
        c1.markdown(f"## 📈 Stock Analyzer")
        if c2.button("🌙" if D else "☀️"):
            st.session_state.theme="淺色" if D else "暗色"; st.rerun()
        st.markdown(f"<p style='color:{DM};font-size:.78rem;margin-top:-8px'>台灣股市智能分析</p>",
                    unsafe_allow_html=True)
        st.divider()

        sym=st.text_input("","",placeholder="代碼（Enter 搜尋）",
                          label_visibility="collapsed",key="si",
                          on_change=lambda: st.session_state.update(_trigger=True))
        abtn=st.button("🔍 分析",type="primary",use_container_width=True)

        st.divider()
        st.markdown("#### 自選股")
        wl = list(st.session_state.watchlist)   # snapshot, avoids mutation mid-render
        if wl:
            # Build static label list — NO lambda, NO format_func
            wl_labels = []
            for _c in wl:
                _n = TW_NAME_CACHE.get(_c, "")
                wl_labels.append(f"{_n} ({_c})" if (_n and _n != _c) else _c)

            # selectbox on string labels; derive index from selected value
            _prev = st.session_state.get("_wl_sel", wl_labels[0])
            if _prev not in wl_labels:
                _prev = wl_labels[0]
            _sel = st.selectbox("", wl_labels,
                index=wl_labels.index(_prev),
                label_visibility="collapsed", key="wl_sel")
            # Decode selected code from label
            _sel_code = wl[wl_labels.index(_sel)]

            c1,c2,c3 = st.columns(3)
            if c1.button("載入", use_container_width=True):
                st.session_state._pending_sym = _sel_code
                st.rerun()
            if c2.button("加入", use_container_width=True):
                cd = st.session_state.get("si","").strip().upper()
                if cd and cd not in st.session_state.watchlist:
                    st.session_state.watchlist.append(cd)
                    st.rerun()
            if c3.button("移除", use_container_width=True):
                _ri = wl_labels.index(_sel)
                st.session_state.watchlist.pop(_ri)
                st.rerun()
        else:
            st.caption("自選股為空")
            if st.button("加入當前代碼", use_container_width=True):
                cd = st.session_state.get("si","").strip().upper()
                if cd and cd not in st.session_state.watchlist:
                    st.session_state.watchlist.append(cd)
                    st.rerun()
        if st.button("📊 批次分析全部",use_container_width=True):
            st.session_state._do_batch=True

        st.divider()
        with st.expander("⚙ 分析參數",expanded=False):
            st.session_state.forecast_days =st.slider("預測天數",5,90,st.session_state.forecast_days)
            st.session_state.lookback_years=st.slider("回溯年數",1,10,st.session_state.lookback_years)
            st.session_state.train_ratio   =st.slider("訓練佔比",.5,.95,st.session_state.train_ratio,.05)
        with st.expander("📊 圖表顯示",expanded=False):
            st.session_state.chart_style=st.radio("",["K棒","曲線"],horizontal=True,
                index=0 if st.session_state.chart_style=="K棒" else 1)
            st.session_state.show_bb  =st.checkbox("布林帶",  st.session_state.show_bb)
            st.session_state.show_sr  =st.checkbox("支撐壓力",st.session_state.show_sr)
            st.session_state.show_band=st.checkbox("預測信賴帶",st.session_state.show_band)
        with st.expander("⚖ 分析權重",expanded=False):
            st.caption("四項總計須為 100%")
            w=st.session_state.weights
            wt=st.slider("技術指標%",0,100,w["technical"],key="wt")
            wm=st.slider("ML/NN%",   0,100,w["ml"],       key="wm")
            wn=st.slider("新聞情緒%",0,100,w["news"],     key="wn")
            wf=st.slider("基本面%",  0,100,w["fundamental"],key="wf")
            tot=wt+wm+wn+wf
            st.markdown(f"{'✅' if tot==100 else '⚠'} **總計 {tot}%**")
            if st.button("▶ Apply",use_container_width=True,disabled=(tot!=100)):
                st.session_state.weights={"technical":wt,"ml":wm,"news":wn,"fundamental":wf}
                if st.session_state.result:
                    r=st.session_state.result
                    nd=compute_drift_bias(r["indicators"],r.get("ml_predict",{}),{},
                        r.get("news_sentiment",{}),r.get("fundamental",{}),
                        st.session_state.weights)
                    nf=simple_forecast(r["df"],days=r["forecast_days"],n_paths=150,drift_bias=nd)
                    r["forecast"]=nf; r["drift_bias"]=nd; r["weights"]=st.session_state.weights
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
        p.empty(); st.error(f"❌ {e}")

# ── PDF export ─────────────────────────────────────────────────────────────
def to_pdf(r,fig):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image as RI
        from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
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
    doc=SimpleDocTemplate(buf,pagesize=A4,leftMargin=2*cm,rightMargin=2*cm,topMargin=2*cm,bottomMargin=2*cm)
    H1=ParagraphStyle("H1",fontSize=16,fontName=fn,textColor=colors.HexColor("#4f6ef5"),spaceAfter=8)
    H2=ParagraphStyle("H2",fontSize=12,fontName=fn,textColor=colors.HexColor("#1a1d2e"),spaceBefore=12,spaceAfter=6)
    NM=ParagraphStyle("NM",fontSize=9,fontName=fn,textColor=colors.HexColor("#374151"),leading=14)
    elems=[]
    name=r["name"]; sym=r["symbol"]; df=r["df"]; fc=r["forecast"]
    ml_p=r.get("ml_predict",{}); bt=r.get("backtest",{})
    last=float(df["Close"].iloc[-1]); med=float(fc["median"][-1]); chg=(med-last)/last*100
    elems.append(Paragraph(f"股票分析報告：{name} ({sym.split('.')[0]})",H1))
    elems.append(Paragraph(f"產生時間：{datetime.now().strftime('%Y-%m-%d %H:%M')}",NM))
    elems.append(Spacer(1,.3*cm))
    try:
        png=fig.to_image(format="png",width=1400,height=700,scale=1.5)
        elems.append(RI(io.BytesIO(png),width=16*cm,height=8*cm))
    except: pass
    elems.append(Spacer(1,.3*cm))
    elems.append(Paragraph("分析摘要",H2))
    s=score(r)
    if s>=1.5: act="偏多·可試單"
    elif s>=.5: act="偏多觀察"
    elif s<=-1.5: act="偏空·減碼"
    elif s<=-.5: act="偏空觀察"
    else: act="中性觀望"
    ind=r["indicators"]; atr=float(ind["atr14"].iloc[-1]) if not pd.isna(ind["atr14"].iloc[-1]) else last*.02
    td=[["項目","數值","項目","數值"],
        ["現價",f"{last:,.2f}","建議",act],
        [f"{r['forecast_days']}日預測",f"{med:,.2f}({chg:+.1f}%)","ML機率",f"{ml_p.get('prob_up',.5)*100:.1f}%"],
        ["RSI",f"{float(ind['rsi14'].iloc[-1]):.1f}","命中率",bt.get("hit_rate_text","N/A")],
        ["停損",f"{last-atr*1.5:,.2f}","停利",f"{last+atr*2.5:,.2f}"]]
    t=Table(td,colWidths=[3.5*cm,4*cm,3.5*cm,4*cm])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#4f6ef5")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,-1),fn),
        ("FONTSIZE",(0,0),(-1,-1),9),("GRID",(0,0),(-1,-1),.5,colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f8fafc"),colors.white]),
        ("ALIGN",(1,0),(-1,-1),"CENTER"),("PADDING",(0,0),(-1,-1),4)]))
    elems.append(t)
    if bt.get("n_trades",0)>0:
        elems.append(Paragraph("回測統計",H2))
        bd=[["筆數",str(bt["n_trades"]),"命中率",bt["hit_rate_text"]],
            ["淨報酬",f"{bt.get('avg_return_after_cost',0):+.2f}%","Sharpe",f"{bt.get('sharpe',0):.2f}"],
            ["最大回撤",f"{bt.get('max_drawdown',0):.1f}%","PF",f"{bt.get('profit_factor',0):.2f}"]]
        b=Table([["筆數","數值","筆數","數值"]]+bd,colWidths=[3.5*cm,4*cm,3.5*cm,4*cm])
        b.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#059669")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,-1),fn),
            ("FONTSIZE",(0,0),(-1,-1),9),("GRID",(0,0),(-1,-1),.5,colors.HexColor("#e2e8f0")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f0fdf4"),colors.white]),
            ("ALIGN",(1,0),(-1,-1),"CENTER"),("PADDING",(0,0),(-1,-1),4)]))
        elems.append(b)
    elems.append(Spacer(1,.5*cm))
    elems.append(Paragraph("⚠ 本報告僅供研究參考，不構成投資建議。",NM))
    doc.build(elems); return buf.getvalue()

# ── Render result ──────────────────────────────────────────────────────────
def show(r):
    df=r["df"]; ind=r["indicators"]; sr=r["sr"]; fc=r["forecast"]
    ml_p=r.get("ml_predict",{}); bt=r.get("backtest",{})
    ns=r.get("news_sentiment",{}); news=r.get("news",[])
    fund=r.get("fundamental",{}); name=r["name"]; sym=r["symbol"]
    last=float(df["Close"].iloc[-1]); prev=float(df["Close"].iloc[-2]) if len(df)>1 else last
    cp=(last-prev)/prev*100; med=float(fc["median"][-1]); fc_c=(med-last)/last*100
    s=score(r); prob=ml_p.get("prob_up",.5)
    bhr=bt.get("hit_rate",.5) if bt.get("n_trades",0)>0 else .5
    cs=(abs(prob-.5)*2+abs(bhr-.5)*2)/2
    conf="高" if cs>=.35 else ("中" if cs>=.18 else "低")
    rsi_v=float(ind["rsi14"].iloc[-1])

    arr="▲" if cp>=0 else "▼"; hc=OK if cp>=0 else ER
    st.markdown(f"<h1>{name} <span style='font-size:.88rem;color:{DM}'>({sym.split('.')[0]})</span></h1>"
        f"<div style='font-size:1.4rem;font-weight:700;color:{hc}'>{last:,.2f} "
        f"<span style='font-size:.9rem'>{arr} {abs(cp):.2f}%</span></div>",
        unsafe_allow_html=True)

    def card(col,lbl,val_h,sub=""):
        col.markdown(f"<div class='card'><div class='lbl'>{lbl}</div>"
            f"<div class='val'>{val_h}</div>"
            f"{'<div class=sub>'+sub+'</div>' if sub else ''}</div>",
            unsafe_allow_html=True)

    c1,c2,c3,c4,c5=st.columns(5)
    card(c1,"建議操作",action(s))
    ccls="bhi" if conf in("高","中") else "blo"
    card(c2,"可信度",badge(ccls,conf),f"ML {prob*100:.0f}%")
    fc_col=OK if fc_c>0 else ER
    card(c3,f"{r['forecast_days']}日預測",f"<span style='color:{fc_col}'>{med:,.2f}</span>",f"{fc_c:+.1f}%")
    card(c4,"回測命中率",bt.get("hit_rate_text","N/A"),f"{bt.get('n_trades',0)} 筆")
    rl="超買" if rsi_v>=70 else("超賣" if rsi_v<=30 else "中性")
    card(c5,"RSI(14)",f"{rsi_v:.1f}",rl)

    fig=chart(r,st.session_state.chart_style,st.session_state.show_bb,
              st.session_state.show_sr,st.session_state.show_band)
    st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})

    # Export
    e1,e2,_=st.columns([1,1,3])
    with e1:
        try:
            png=fig.to_image(format="png",width=1400,height=700,scale=2)
            st.download_button("📷 PNG",png,
                file_name=f"{sym.split('.')[0]}_{datetime.now().strftime('%Y%m%d')}.png",
                mime="image/png",use_container_width=True)
        except: st.caption("PNG需kaleido")
    with e2:
        try:
            pdf=to_pdf(r,fig)
            if pdf:
                st.download_button("📄 PDF",pdf,
                    file_name=f"{sym.split('.')[0]}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",use_container_width=True)
            else: st.caption("PDF需reportlab")
        except: st.caption("PDF需reportlab")

    # Tabs
    t1,t2,t3,t4,t5=st.tabs(["📊 ML/回測","🔍 技術解讀","📋 基本面","📰 新聞","📍 關鍵價位"])

    with t1:
        ml_stats=r.get("ml_stats",{})
        a,b=st.columns(2)
        with a:
            st.markdown("**ML 模型**")
            if "error" in ml_stats: st.warning(ml_stats["error"])
            else:
                pc=OK if prob>=.6 else(ER if prob<=.4 else WA)
                st.markdown(f"<span style='font-size:1.2rem;font-weight:700;color:{pc}'>上漲機率 {prob*100:.1f}%</span>",
                            unsafe_allow_html=True)
                st.metric("OOS準確率",f"{ml_stats.get('oos_accuracy',0)*100:.1f}%")
                st.metric("Brier",f"{ml_stats.get('brier',0):.3f}")
                st.metric("樣本",f"{ml_stats.get('n_samples',0)} 筆")
        with b:
            st.markdown("**回測（含交易成本 ~0.588%/來回）**")
            if bt.get("n_trades",0)>0:
                st.metric("筆數",str(bt["n_trades"]))
                st.metric("命中率",bt["hit_rate_text"])
                st.metric("淨報酬",f"{bt.get('avg_return_after_cost',0):+.2f}%")
                st.metric("Sharpe",f"{bt.get('sharpe',0):.2f}")
                st.metric("最大回撤",f"{bt.get('max_drawdown',0):.1f}%")
                st.metric("Profit Factor",f"{bt.get('profit_factor',0):.2f}")
            else: st.info("回測樣本不足")
        st.caption(f"drift_bias={r.get('drift_bias',0):+.5f}")
        st.divider()
        st.markdown("**自訂回測**")
        h2,n2,rb=st.columns([1,1,1])
        hv=h2.number_input("天數",5,60,10,key="bh")
        nv=n2.number_input("樣本",20,500,100,key="bn")
        if rb.button("執行",use_container_width=True):
            with st.spinner("回測中…"):
                bt2=backtest_directional(r["df"],r["indicators"],r["sr"],horizon=hv,n_samples=nv)
            if bt2.get("n_trades",0)>0:
                r1,r2,r3,r4=st.columns(4)
                r1.metric("命中率",bt2["hit_rate_text"])
                r2.metric("淨報酬",f"{bt2.get('avg_return_after_cost',0):+.2f}%")
                r3.metric("Sharpe",f"{bt2.get('sharpe',0):.2f}")
                r4.metric("最大回撤",f"{bt2.get('max_drawdown',0):.1f}%")
            else: st.warning("樣本不足")

    with t2:
        mh=float(ind["macd_hist"].iloc[-1]); mp2=float(ind["macd_hist"].iloc[-2]) if len(ind["macd_hist"])>1 else mh
        m5=float(ind["ma5"].iloc[-1]); m20=float(ind["ma20"].iloc[-1]); m60=float(ind["ma60"].iloc[-1])
        bbu=float(ind["bb_up"].iloc[-1]); bbd=float(ind["bb_dn"].iloc[-1])
        bpos=(last-bbd)/max(bbu-bbd,1e-9)
        bw=float(ind["bb_width"].iloc[-1]); bw2=float(ind["bb_width"].iloc[-2]) if len(ind["bb_width"])>1 else bw
        vr=float(df["Volume"].iloc[-1])/max(float(ind["vol_ma20"].iloc[-1]),1)
        lines=[]
        if rsi_v>=75:    lines.append(("🔴",f"RSI **{rsi_v:.0f}** 深度超買，短線壓力明顯"))
        elif rsi_v>=65:  lines.append(("🟡",f"RSI **{rsi_v:.0f}** 偏高，留意回落"))
        elif rsi_v<=25:  lines.append(("🟢",f"RSI **{rsi_v:.0f}** 深度超賣，反彈機率高"))
        elif rsi_v<=35:  lines.append(("🟡",f"RSI **{rsi_v:.0f}** 超賣區"))
        else:            lines.append(("⚪",f"RSI **{rsi_v:.0f}** 中性區間"))
        if mh>=0 and mp2<0:  lines.append(("🟢","MACD **金叉**，偏多訊號"))
        elif mh<0 and mp2>=0:lines.append(("🔴","MACD **死叉**，偏空訊號"))
        elif mh>0: lines.append(("🟢",f"MACD 多頭 {mh:+.3f}"))
        else:      lines.append(("🔴",f"MACD 空頭 {mh:+.3f}"))
        if bw<bw2*.85:  lines.append(("⚡","布林帶**收縮**，可能醞釀突破"))
        elif bpos>.85:  lines.append(("⚠","貼近布林上軌，注意壓力"))
        elif bpos<.15:  lines.append(("🟢","貼近布林下軌，可能反彈"))
        if m5>m20>m60:  lines.append(("📈","均線**多頭排列**"))
        elif m5<m20<m60:lines.append(("📉","均線**空頭排列**"))
        if vr>=2:   lines.append(("🔊",f"成交量爆增 **{vr:.1f}x**"))
        elif vr<=.5:lines.append(("🔇",f"成交量萎縮 **{vr:.1f}x**"))
        fc_col2=OK if fc_c>0 else ER
        lines.append(("🎯",f"預測 {r['forecast_days']}日 **{med:,.2f}** <span style='color:{fc_col2}'>({fc_c:+.1f}%)</span>"))
        if prob!=.5:
            pc=OK if prob>=.6 else(ER if prob<=.4 else WA)
            lines.append(("🤖",f"ML <span style='color:{pc}'><b>{prob*100:.1f}%</b></span> ({ml_p.get('label','')})"))
        if bt.get("n_trades",0)>0:
            lines.append(("📊",f"回測命中 **{bt['hit_rate_text']}** ({bt['n_trades']} 筆)"))
        for icon,txt in lines:
            st.markdown(f"{icon} {txt}", unsafe_allow_html=True)

    with t3:
        if fund:
            a,b=st.columns(2)
            items=[("本益比","pe_ratio",lambda v:f"{v:.1f}x"),("EPS","eps",lambda v:f"{v:.2f}"),
                   ("ROE","roe",lambda v:f"{v*100:.1f}%"),("殖利率","div_yield",lambda v:f"{v*100:.2f}%"),
                   ("市值","market_cap",lambda v:f"{v/1e9:.1f}B" if v>1e9 else f"{v/1e6:.0f}M")]
            for i,(l,k,f) in enumerate(items):
                if k in fund: (a if i%2==0 else b).metric(l,f(fund[k]))
        else: st.info("無基本面資料")

    with t4:
        if news:
            lbl=ns.get("label","中性"); lc=OK if lbl=="正面" else(ER if lbl=="負面" else WA)
            st.markdown(f"**整體情緒：<span style='color:{lc}'>{lbl}</span>**",unsafe_allow_html=True)
            for item in news:
                t=item.get("title",""); lk=item.get("link","#"); sc=item.get("source","")
                st.markdown(f"<div class='ni'><a href='{lk}' target='_blank'>{t}</a>"
                    f"{'<span style=\"font-size:.75rem;color:'+DM+'\">['+sc+']</span>' if sc else ''}"
                    f"</div>",unsafe_allow_html=True)
        else: st.info("未取得新聞")

    with t5:
        atr=float(ind["atr14"].iloc[-1]) if not pd.isna(ind["atr14"].iloc[-1]) else last*.02
        sl=last-atr*1.5; tp=last+atr*2.5; rr=(tp-last)/max(last-sl,1e-9)
        c1,c2,c3,c4=st.columns(4)
        c1.metric("支撐區",f"{sr['support_lo']:.2f}~{sr['support_hi']:.2f}")
        c2.metric("壓力區",f"{sr['resistance_lo']:.2f}~{sr['resistance_hi']:.2f}")
        c3.metric("建議停損",f"{sl:,.2f}",delta=f"-{(last-sl)/last*100:.1f}%",delta_color="inverse")
        c4.metric("建議停利",f"{tp:,.2f}",delta=f"+{(tp-last)/last*100:.1f}%")
        st.caption(f"ATR={atr:.2f}　RR={rr:.2f}x")

# ── Batch ──────────────────────────────────────────────────────────────────
def batch():
    wl=st.session_state.watchlist
    if not wl: st.warning("自選股為空"); return
    st.markdown(f"## 批次分析 {len(wl)} 支股票")
    p=st.progress(0); rows=[]
    for i,code in enumerate(wl):
        p.progress((i+1)/len(wl),text=f"{i+1}/{len(wl)}: {code}")
        try:
            r=run_analysis(code,lookback_years=st.session_state.lookback_years,
                forecast_days=st.session_state.forecast_days,
                weights=st.session_state.weights,
                train_ratio=st.session_state.train_ratio)
            r["weights"]=st.session_state.weights
            df=r["df"]; fc=r["forecast"]; ml_p=r.get("ml_predict",{}); bt=r.get("backtest",{})
            last=float(df["Close"].iloc[-1]); med=float(fc["median"][-1]); chg=(med-last)/last*100
            prob=ml_p.get("prob_up",.5); s_=score(r)
            if s_>=1.5: act="偏多·可試單"
            elif s_>=.5: act="偏多觀察"
            elif s_<=-1.5: act="偏空·減碼"
            elif s_<=-.5: act="偏空觀察"
            else: act="中性觀望"
            bhr=bt.get("hit_rate",.5) if bt.get("n_trades",0)>0 else .5
            cs=(abs(prob-.5)*2+abs(bhr-.5)*2)/2
            conf="高" if cs>=.35 else("中" if cs>=.18 else "低")
            rows.append({"代碼":r["symbol"].split(".")[0],"名稱":r["name"],
                "建議":act,"可信度":conf,"現價":f"{last:,.2f}",
                "ML%":f"{prob*100:.1f}%","預測%":f"{chg:+.1f}%",
                "命中率":bt.get("hit_rate_text","N/A")})
        except Exception as e:
            rows.append({"代碼":code,"名稱":code,"建議":"錯誤","可信度":"—",
                "現價":"—","ML%":"—","預測%":"—","命中率":str(e)[:30]})
    p.empty()
    df_out=pd.DataFrame(rows)
    st.dataframe(df_out,use_container_width=True,hide_index=True)
    csv=df_out.to_csv(index=False,encoding="utf-8-sig")
    st.download_button("💾 下載 CSV",csv,
        file_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv")

# ── Main ───────────────────────────────────────────────────────────────────
sym,abtn=sidebar()

# 處理自選股載入（用中間變數，不直接改 widget key）
if st.session_state.get("_pending_sym"):
    _pending = st.session_state.pop("_pending_sym")
    with st.spinner(f"分析 {_pending}…"):
        analyze(_pending)
    st.rerun()

_s=st.session_state.get("si","").strip().upper()
if (abtn or st.session_state.get("_trigger")) and _s:
    st.session_state._trigger=False
    with st.spinner(f"分析 {_s}…"):
        analyze(_s)
    st.rerun()
if st.session_state.get("_do_batch"):
    st.session_state._do_batch=False; batch()
elif st.session_state.result:
    show(st.session_state.result)
else:
    st.markdown(f"""<div style='text-align:center;padding:50px 20px'>
        <div style='font-size:3.5rem'>📈</div>
        <h1 style='font-size:1.8rem;color:{AC}'>Stock Analyzer Pro</h1>
        <p style='color:{DM};max-width:480px;margin:0 auto'>
        台灣股市智能分析系統<br>左側輸入代碼，按 Enter 或 🔍 開始</p>
        <p style='color:{DM};font-size:.82rem;margin-top:10px'>
        支援：上市 · 上櫃 · ETF · 槓桿反向 · 受益憑證 · 美股</p>
    </div>""", unsafe_allow_html=True)
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
