"""
Stock Analyzer Pro — Streamlit Web Version
手機 / 電腦 瀏覽器皆可使用
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings, io

warnings.filterwarnings("ignore")

# ── Page config (must be FIRST streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Stock Analyzer Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Stock Analyzer Pro — 台灣股市智能分析系統"},
)

from core import (
    run_analysis, TW_NAME_CACHE, _load_twse_bulk, _load_tpex_bulk,
    compute_drift_bias, simple_forecast, _cjk,
    SKLEARN_OK, XGB_OK, TORCH_OK,
)

# ═══════════════════════════════════════════════════════════════════════════
# Theme / CSS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] {background:#0f1120;}
[data-testid="stSidebar"] {background:#1a1d2e;}
[data-testid="stHeader"] {background:#0f1120;}

/* ── Typography ── */
html, body, [class*="css"] {
    font-family: 'Microsoft JhengHei', 'PingFang TC', sans-serif;
    color: #e2e4f0;
}
h1 {color:#6c8ef5; font-size:1.6rem; font-weight:700; margin-bottom:0.2rem;}
h2 {color:#a5b4fc; font-size:1.1rem; font-weight:600;}
h3 {color:#8890aa; font-size:0.95rem; font-weight:500;}

/* ── Cards ── */
.metric-card {
    background:#1a1d2e; border:1px solid #2d3154;
    border-radius:12px; padding:14px 18px;
    margin-bottom:10px;
}
.metric-card .label {color:#8890aa; font-size:0.8rem; margin-bottom:4px;}
.metric-card .value {color:#e2e4f0; font-size:1.3rem; font-weight:700;}
.metric-card .sub   {color:#6c8ef5; font-size:0.85rem;}

/* ── Action badges ── */
.badge-bull  {background:#05966922;color:#34d399;padding:4px 14px;border-radius:20px;font-weight:700;}
.badge-bear  {background:#ef444422;color:#f87171;padding:4px 14px;border-radius:20px;font-weight:700;}
.badge-neut  {background:#f59e0b22;color:#fbbf24;padding:4px 14px;border-radius:20px;font-weight:700;}
.badge-high  {background:#6c8ef522;color:#6c8ef5;padding:4px 10px;border-radius:20px;}
.badge-mid   {background:#f59e0b22;color:#fbbf24;padding:4px 10px;border-radius:20px;}
.badge-low   {background:#ef444422;color:#f87171;padding:4px 10px;border-radius:20px;}

/* ── News ── */
.news-item {
    border-left:3px solid #2d3154; padding:6px 10px;
    margin:6px 0; font-size:0.88rem; color:#b0b8d0;
}
.news-item:hover {border-color:#6c8ef5; color:#e2e4f0;}

/* ── Sidebar buttons ── */
.stButton button {
    background:#1e2240; border:1px solid #3d4580;
    color:#c8d0f0; border-radius:8px; width:100%;
    transition:all 0.2s;
}
.stButton button:hover {background:#2a3060; border-color:#6c8ef5; color:#fff;}

/* ── Tables ── */
.stDataFrame {background:#1a1d2e !important;}

/* ── Progress ── */
.stProgress > div > div {background:#6c8ef5;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Session state initialisation
# ═══════════════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "result":         None,
        "batch_results":  [],
        "watchlist":      ["2330","2454","0050","00878","2317"],
        "weights":        {"technical":40,"ml":35,"news":15,"fundamental":10},
        "forecast_days":  30,
        "lookback_years": 3,
        "train_ratio":    0.8,
        "chart_style":    "K棒",
        "show_bb":        True,
        "show_sr":        True,
        "show_band":      True,
        "names_loaded":   False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# Pre-load name tables once
if not st.session_state.names_loaded:
    with st.spinner("載入股票名稱資料庫…"):
        _load_twse_bulk(); _load_tpex_bulk()
    st.session_state.names_loaded = True


# ═══════════════════════════════════════════════════════════════════════════
# Helper: Plotly chart
# ═══════════════════════════════════════════════════════════════════════════
ACCENT = "#6c8ef5"
SUCCESS = "#34d399"
DANGER  = "#f87171"
BG1     = "#0f1120"
BG2     = "#1a1d2e"
BG3     = "#14172a"
TEXT_DIM = "#8890aa"

def build_chart(result: dict, style: str = "K棒",
                show_bb: bool = True, show_sr: bool = True,
                show_band: bool = True) -> go.Figure:
    df   = result["df"]
    ind  = result["indicators"]
    sr   = result["sr"]
    fc   = result["forecast"]
    name = result["name"]
    sym  = result["symbol"]

    # Row heights: main, MACD, RSI
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.18, 0.17],
        vertical_spacing=0.02,
        subplot_titles=[f"{name} ({sym.split('.')[0]})", "MACD", "RSI(14)"],
    )

    # ── Main chart ────────────────────────────────────────────────
    if style == "K棒":
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"],  close=df["Close"],
            increasing_line_color=SUCCESS, decreasing_line_color=DANGER,
            name="K棒", showlegend=False,
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"], mode="lines",
            line=dict(color=ACCENT, width=1.5), name="收盤"),
            row=1, col=1)

    # Bollinger Bands
    if show_bb:
        fig.add_trace(go.Scatter(
            x=df.index, y=ind["bb_up"],
            line=dict(color="rgba(108,142,245,0.38)", width=1), name="BB上軌"),
            row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=ind["bb_dn"],
            line=dict(color="rgba(108,142,245,0.38)", width=1), fill="tonexty",
            fillcolor="rgba(108,142,245,0.06)", name="BB下軌"),
            row=1, col=1)

    # Support / Resistance
    if show_sr:
        for lvl, color, lbl in [
            (sr["resistance_hi"], "rgba(248,113,113,0.31)", "壓力"),
            (sr["support_lo"],    "rgba(52,211,153,0.31)", "支撐"),
        ]:
            fig.add_hline(y=lvl, line_dash="dot",
                          line_color=color, row=1, col=1,
                          annotation_text=lbl, annotation_font_color=color)

    # Forecast zone
    future_dates = list(fc["future_dates"])
    if show_band:
        fig.add_trace(go.Scatter(
            x=future_dates, y=fc["upper"],
            line=dict(color="rgba(167,139,250,0.25)", width=0), name="上限"),
            row=1, col=1)
        fig.add_trace(go.Scatter(
            x=future_dates, y=fc["lower"],
            line=dict(color="rgba(167,139,250,0.25)", width=0), fill="tonexty",
            fillcolor="rgba(167,139,250,0.08)", name="預測帶"),
            row=1, col=1)

    # Forecast median
    fig.add_trace(go.Scatter(
        x=[df.index[-1]] + future_dates,
        y=[float(df["Close"].iloc[-1])] + list(fc["median"]),
        mode="lines",
        line=dict(color="#a78bfa", width=2, dash="dash"),
        name=f"預測(drift={fc.get('drift_bias',0):+.4f})",
    ), row=1, col=1)

    # ── MACD ──────────────────────────────────────────────────────
    colors = [SUCCESS if v >= 0 else DANGER for v in ind["macd_hist"]]
    fig.add_trace(go.Bar(
        x=df.index, y=ind["macd_hist"], marker_color=colors,
        name="Histogram", showlegend=False),
        row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=ind["macd_line"],
        line=dict(color=ACCENT, width=1), name="MACD"),
        row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=ind["macd_signal"],
        line=dict(color="#f59e0b", width=1), name="Signal"),
        row=2, col=1)

    # ── RSI ───────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=ind["rsi14"],
        line=dict(color="#34d399", width=1.5), name="RSI"),
        row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="rgba(248,113,113,0.38)", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="rgba(52,211,153,0.38)", row=3, col=1)

    # ── Layout ────────────────────────────────────────────────────
    fig.update_layout(
        height=640,
        paper_bgcolor=BG1,
        plot_bgcolor=BG3,
        font=dict(color=TEXT_DIM, size=11),
        legend=dict(bgcolor=BG2, bordercolor="#2d3154",
                    font=dict(size=10), x=0.01, y=0.99),
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=30, b=0),
        hovermode="x unified",
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor="#2d3154", row=i, col=1)
        fig.update_yaxes(gridcolor="#2d3154", row=i, col=1)

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("## 📈 Stock Analyzer Pro")
        st.markdown("<p style='color:#8890aa;font-size:0.8rem;margin-top:-10px'>台灣股市智能分析系統</p>", unsafe_allow_html=True)
        st.divider()

        # ── Stock input ──────────────────────────────────────────
        st.markdown("### 股票代碼")
        col1, col2 = st.columns([3, 1])
        with col1:
            symbol = st.text_input("", placeholder="例：2330 / 0050 / AAPL",
                                   label_visibility="collapsed",
                                   key="symbol_input")
        with col2:
            analyze_btn = st.button("分析", type="primary", use_container_width=True)

        # Watchlist
        st.markdown("### 自選股")
        wl = st.session_state.watchlist
        wl_display = []
        for code in wl:
            n = TW_NAME_CACHE.get(code, code)
            wl_display.append(f"{n} ({code})")
        selected_wl = st.selectbox("", ["— 選擇 —"] + wl_display,
                                   label_visibility="collapsed", key="wl_select")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("載入", use_container_width=True) and selected_wl != "— 選擇 —":
                code = selected_wl.split("(")[-1].rstrip(")")
                st.session_state.symbol_input = code
                analyze_btn = True
        with c2:
            add_btn = st.button("加入", use_container_width=True)
        with c3:
            del_btn = st.button("移除", use_container_width=True)

        if add_btn and symbol:
            code = symbol.strip().upper()
            if code not in st.session_state.watchlist:
                st.session_state.watchlist.append(code)
                st.rerun()
        if del_btn and selected_wl != "— 選擇 —":
            code = selected_wl.split("(")[-1].rstrip(")")
            if code in st.session_state.watchlist:
                st.session_state.watchlist.remove(code)
                st.rerun()

        # Batch
        st.divider()
        if st.button("📊 批次分析自選股", use_container_width=True):
            st.session_state._do_batch = True

        st.divider()

        # ── Analysis params ──────────────────────────────────────
        st.markdown("### 分析參數")
        st.session_state.forecast_days  = st.slider("預測天數", 5, 90, st.session_state.forecast_days)
        st.session_state.lookback_years = st.slider("回溯年數", 1, 10, st.session_state.lookback_years)
        st.session_state.train_ratio    = st.slider("訓練佔比", 0.5, 0.95, st.session_state.train_ratio, 0.05)

        # ── Chart options ────────────────────────────────────────
        st.markdown("### 圖表顯示")
        st.session_state.chart_style = st.radio("圖形", ["K棒","曲線"],
                                                  horizontal=True,
                                                  index=0 if st.session_state.chart_style=="K棒" else 1)
        st.session_state.show_bb   = st.checkbox("布林帶", st.session_state.show_bb)
        st.session_state.show_sr   = st.checkbox("支撐壓力", st.session_state.show_sr)
        st.session_state.show_band = st.checkbox("預測信賴帶", st.session_state.show_band)

        # ── Weight settings ──────────────────────────────────────
        st.divider()
        st.markdown("### ⚖ 分析權重設定")
        st.caption("總計須為 100%，影響趨勢方向與建議操作")
        w = st.session_state.weights
        w_tech = st.slider("技術指標 %", 0, 100, w["technical"], key="w_tech")
        w_ml   = st.slider("ML / NN %",  0, 100, w["ml"],        key="w_ml")
        w_news = st.slider("新聞情緒 %", 0, 100, w["news"],      key="w_news")
        w_fund = st.slider("基本面 %",   0, 100, w["fundamental"],key="w_fund")
        total  = w_tech + w_ml + w_news + w_fund
        color  = "normal" if total == 100 else "error"
        st.markdown(f"**總計：{total}%** {'✅' if total==100 else '← 請調整至 100%'}")

        apply_weights = st.button("▶ Apply 套用權重", use_container_width=True,
                                  disabled=(total != 100))
        if apply_weights:
            st.session_state.weights = {
                "technical": w_tech, "ml": w_ml,
                "news": w_news, "fundamental": w_fund,
            }
            # Recompute forecast with new weights if result exists
            if st.session_state.result is not None:
                r = st.session_state.result
                new_drift = compute_drift_bias(
                    r["indicators"], r.get("ml_predict",{}),
                    {}, r.get("news_sentiment",{}),
                    r.get("fundamental",{}), st.session_state.weights)
                new_fc = simple_forecast(
                    r["df"], days=r["forecast_days"],
                    n_paths=150, drift_bias=new_drift)
                st.session_state.result["forecast"] = new_fc
                st.session_state.result["drift_bias"] = new_drift
                st.success(f"✔ 套用完成  drift={new_drift:+.5f}")
            st.rerun()

        # Model info
        st.divider()
        st.caption(
            f"模型：{'GB+XGB' if XGB_OK else 'GB'}"
            f"{'+LSTM' if TORCH_OK else ''}"
        )

    return symbol, analyze_btn


# ═══════════════════════════════════════════════════════════════════════════
# Analysis runner
# ═══════════════════════════════════════════════════════════════════════════
def do_analysis(symbol: str):
    progress = st.progress(0, text="準備分析…")
    status   = st.empty()
    steps    = ["抓取資料","計算指標","市場背景","訓練ML","趨勢預測","歷史回測","新聞情緒"]

    def prog(s, t, msg):
        progress.progress(s/t, text=f"[{s}/{t}] {msg}")
        status.markdown(f"<span style='color:#8890aa;font-size:0.85rem'>{msg}</span>",
                        unsafe_allow_html=True)

    try:
        result = run_analysis(
            symbol,
            lookback_years  = st.session_state.lookback_years,
            forecast_days   = st.session_state.forecast_days,
            weights         = st.session_state.weights,
            train_ratio     = st.session_state.train_ratio,
            progress_callback = prog,
        )
        st.session_state.result = result
        progress.empty(); status.empty()
        return result
    except Exception as e:
        progress.empty(); status.empty()
        st.error(f"分析失敗：{e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Result rendering
# ═══════════════════════════════════════════════════════════════════════════
def render_result(result: dict):
    df   = result["df"]
    ind  = result["indicators"]
    sr   = result["sr"]
    fc   = result["forecast"]
    ml_p = result.get("ml_predict", {})
    bt   = result.get("backtest", {})
    ns   = result.get("news_sentiment", {})
    news = result.get("news", [])
    fund = result.get("fundamental", {})
    name = result["name"]
    sym  = result["symbol"]
    w    = result.get("weights", st.session_state.weights)

    last  = float(df["Close"].iloc[-1])
    prev  = float(df["Close"].iloc[-2])
    chg   = last - prev
    chg_p = chg / prev * 100
    med   = float(fc["median"][-1])
    fc_chg = (med - last) / last * 100
    drift  = result.get("drift_bias", 0.0)

    # ── Action score ──────────────────────────────────────────────
    w_t = w.get("technical",40)/100; w_m = w.get("ml",35)/100
    w_n = w.get("news",15)/100;      w_f = w.get("fundamental",10)/100
    try:
        mh   = float(ind["macd_hist"].iloc[-1])
        rsi  = float(ind["rsi14"].iloc[-1])
        mstd = float(ind["macd_hist"].std()) if len(ind["macd_hist"])>5 else 1.0
        tech_sig = float(np.clip(mh/max(mstd,1e-9),-1,1))*0.5
        tech_sig += float(np.clip((50-rsi)/50,-1,1))*(-0.3)
    except Exception:
        tech_sig = 0.0
    ml_prob = ml_p.get("prob_up", 0.5)
    ml_sig  = (ml_prob - 0.5)*2
    ns_sig  = 0.6 if ns.get("label")=="正面" else (-0.6 if ns.get("label")=="負面" else 0.0)
    score = tech_sig*w_t*2 + ml_sig*w_m*2 + ns_sig*w_n*2
    if score >= 1.5:    action, badge_cls = "偏多·可試單", "badge-bull"
    elif score >= 0.5:  action, badge_cls = "偏多觀察",   "badge-bull"
    elif score <= -1.5: action, badge_cls = "偏空·減碼",  "badge-bear"
    elif score <= -0.5: action, badge_cls = "偏空觀察",   "badge-bear"
    else:               action, badge_cls = "中性觀望",   "badge-neut"

    bt_hr = bt.get("hit_rate", 0.5) if bt.get("n_trades",0) > 0 else 0.5
    conf_s = (abs(ml_prob-0.5)*2 + abs(bt_hr-0.5)*2)/2
    if conf_s >= 0.35:   conf_lbl, conf_cls = "高", "badge-high"
    elif conf_s >= 0.18: conf_lbl, conf_cls = "中", "badge-mid"
    else:                conf_lbl, conf_cls = "低", "badge-low"

    # ── Header ───────────────────────────────────────────────────
    arrow = "▲" if chg >= 0 else "▼"
    color = SUCCESS if chg >= 0 else DANGER
    st.markdown(
        f"<h1>{name} "
        f"<span style='font-size:1rem;color:#8890aa'>({sym.split('.')[0]})</span></h1>"
        f"<div style='font-size:1.6rem;font-weight:700;color:{color}'>"
        f"{last:,.2f} "
        f"<span style='font-size:1rem'>{arrow} {abs(chg_p):.2f}%</span></div>",
        unsafe_allow_html=True,
    )

    # ── Quick metrics row ─────────────────────────────────────────
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='label'>建議操作</div>
            <div class='value'><span class='{badge_cls}'>{action}</span></div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div class='label'>可信度</div>
            <div class='value'><span class='{conf_cls}'>{conf_lbl}</span></div>
            <div class='sub'>ML {ml_prob*100:.0f}%</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        fc_color = SUCCESS if fc_chg > 0 else DANGER
        st.markdown(f"""<div class='metric-card'>
            <div class='label'>{result['forecast_days']}日預測</div>
            <div class='value' style='color:{fc_color}'>{med:,.2f}</div>
            <div class='sub' style='color:{fc_color}'>{fc_chg:+.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        hr_txt = bt.get("hit_rate_text","N/A")
        st.markdown(f"""<div class='metric-card'>
            <div class='label'>回測命中率</div>
            <div class='value'>{hr_txt}</div>
            <div class='sub'>{bt.get('n_trades',0)} 筆</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        rsi_v = float(ind["rsi14"].iloc[-1])
        rsi_lbl = "超買" if rsi_v>=70 else ("超賣" if rsi_v<=30 else "中性")
        st.markdown(f"""<div class='metric-card'>
            <div class='label'>RSI(14)</div>
            <div class='value'>{rsi_v:.1f}</div>
            <div class='sub'>{rsi_lbl}</div>
        </div>""", unsafe_allow_html=True)

    # ── Chart ─────────────────────────────────────────────────────
    fig = build_chart(result,
                      style     = st.session_state.chart_style,
                      show_bb   = st.session_state.show_bb,
                      show_sr   = st.session_state.show_sr,
                      show_band = st.session_state.show_band)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo":False})

    # ── Tabs: ML/回測 | 基本面 | 技術解讀 | 新聞 ────────────────
    tab_ml, tab_fund, tab_reason, tab_news = st.tabs(
        ["📊 ML / 回測", "📋 基本面", "🔍 技術解讀", "📰 新聞"])

    with tab_ml:
        ml_stats = result.get("ml_stats", {})
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**ML 模型**")
            if "error" in ml_stats:
                st.warning(ml_stats["error"])
            else:
                st.metric("演算法", ml_stats.get("algorithm","—"))
                st.metric("OOS 準確率", f"{ml_stats.get('oos_accuracy',0)*100:.1f}%")
                st.metric("Brier Score", f"{ml_stats.get('brier',0):.3f}")
                st.metric("訓練樣本", f"{ml_stats.get('n_samples',0)} 筆")
                prob_c = SUCCESS if ml_prob >= 0.6 else (DANGER if ml_prob <= 0.4 else "#fbbf24")
                st.markdown(f"**上漲機率：<span style='color:{prob_c}'>{ml_prob*100:.1f}%</span>**",
                            unsafe_allow_html=True)
        with c2:
            st.markdown("**回測統計（含交易成本）**")
            if bt.get("n_trades",0) > 0:
                st.metric("交易筆數",    f"{bt['n_trades']}")
                st.metric("方向命中率",  bt["hit_rate_text"])
                st.metric("含成本淨報酬",f"{bt.get('avg_return_after_cost',0):+.2f}%")
                st.metric("Sharpe Ratio", f"{bt.get('sharpe',0):.2f}")
                st.metric("最大回撤",    f"{bt.get('max_drawdown',0):.1f}%")
                st.metric("Profit Factor",f"{bt.get('profit_factor',0):.2f}")
            else:
                st.info("回測樣本不足")
        # Drift info
        st.divider()
        st.caption(f"預測漂移偏差 drift_bias = {drift:+.5f}（由當前分析權重決定）")

    with tab_fund:
        if fund:
            c1, c2 = st.columns(2)
            fund_items = [
                ("本益比 P/E",   "pe_ratio",   lambda v: f"{v:.1f}x"),
                ("EPS",          "eps",         lambda v: f"{v:.2f}"),
                ("ROE",          "roe",         lambda v: f"{v*100:.1f}%"),
                ("殖利率",       "div_yield",   lambda v: f"{v*100:.2f}%"),
                ("市值",         "market_cap",  lambda v: f"{v/1e9:.1f}B" if v>1e9 else f"{v/1e6:.0f}M"),
            ]
            for i, (lbl, key, fmt) in enumerate(fund_items):
                col = c1 if i % 2 == 0 else c2
                if key in fund:
                    col.metric(lbl, fmt(fund[key]))
        else:
            st.info("無基本面資料（此股票可能為 ETF 或資料不足）")

    with tab_reason:
        _render_reason(result)

    with tab_news:
        if news:
            ns_label = ns.get("label","中性")
            ns_color = SUCCESS if ns_label=="正面" else (DANGER if ns_label=="負面" else "#fbbf24")
            st.markdown(f"**整體情緒：<span style='color:{ns_color}'>{ns_label}</span>**",
                        unsafe_allow_html=True)
            for item in news:
                title = item.get("title","")
                link  = item.get("link","#")
                src   = item.get("source","")
                st.markdown(
                    f"<div class='news-item'>"
                    f"<a href='{link}' target='_blank' style='color:#b0b8d0;text-decoration:none'>"
                    f"{title}</a>"
                    f"{'  <span style=\"color:#8890aa;font-size:0.75rem\">['+src+']</span>' if src else ''}"
                    f"</div>",
                    unsafe_allow_html=True)
        else:
            st.info("未取得相關新聞")

    # ── Key levels ────────────────────────────────────────────────
    with st.expander("📍 關鍵價位 / 交易計畫"):
        atr = float(ind["atr14"].iloc[-1]) if not pd.isna(ind["atr14"].iloc[-1]) else last*0.02
        sl  = last - atr*1.5
        tp  = last + atr*2.5
        rr  = (tp-last)/max(last-sl,1e-9)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("支撐區",   f"{sr['support_lo']:.2f} ~ {sr['support_hi']:.2f}")
        c2.metric("壓力區",   f"{sr['resistance_lo']:.2f} ~ {sr['resistance_hi']:.2f}")
        c3.metric("建議停損", f"{sl:,.2f}", delta=f"-{(last-sl)/last*100:.1f}%", delta_color="inverse")
        c4.metric("建議停利", f"{tp:,.2f}", delta=f"+{(tp-last)/last*100:.1f}%")
        st.caption(f"ATR = {atr:.2f}　停損 ATR×1.5　停利 ATR×2.5　RR = {rr:.2f}x")


def _render_reason(result: dict):
    """Generate technical analysis text."""
    df   = result["df"]
    ind  = result["indicators"]
    fc   = result["forecast"]
    ml_p = result.get("ml_predict", {})
    bt   = result.get("backtest", {})

    last  = float(df["Close"].iloc[-1])
    med   = float(fc["median"][-1])
    chg   = (med-last)/last*100
    rsi   = float(ind["rsi14"].iloc[-1])
    mh    = float(ind["macd_hist"].iloc[-1])
    ml_p2 = float(ind["macd_hist"].iloc[-2]) if len(ind["macd_hist"])>1 else mh
    m5    = float(ind["ma5"].iloc[-1])
    m20   = float(ind["ma20"].iloc[-1])
    m60   = float(ind["ma60"].iloc[-1])
    bb_up = float(ind["bb_up"].iloc[-1])
    bb_dn = float(ind["bb_dn"].iloc[-1])
    bb_pos = (last-bb_dn)/max(bb_up-bb_dn,1e-9)
    bb_w  = float(ind["bb_width"].iloc[-1])
    bb_w2 = float(ind["bb_width"].iloc[-2]) if len(ind["bb_width"])>1 else bb_w
    vol   = float(df["Volume"].iloc[-1])
    vol_m = float(ind["vol_ma20"].iloc[-1])

    lines = []
    # RSI
    if rsi>=75: lines.append(f"🔴 RSI **{rsi:.0f}** 深度超買，短線獲利了結壓力明顯")
    elif rsi>=65: lines.append(f"🟡 RSI **{rsi:.0f}** 偏高，動能尚強但留意回落")
    elif rsi<=25: lines.append(f"🟢 RSI **{rsi:.0f}** 深度超賣，歷史上此區間反彈機率偏高")
    elif rsi<=35: lines.append(f"🟡 RSI **{rsi:.0f}** 超賣區，留意企穩訊號")
    else: lines.append(f"⚪ RSI **{rsi:.0f}** 中性區間，方向待確認")
    # MACD
    if mh >= 0 and ml_p2 < 0: lines.append("🟢 MACD **金叉**（Histogram 由負轉正），短線偏多訊號")
    elif mh < 0 and ml_p2 >= 0: lines.append("🔴 MACD **死叉**（Histogram 由正轉負），短線偏空訊號")
    elif mh > 0: lines.append(f"🟢 MACD 多頭格局，Histogram {mh:+.3f}")
    else: lines.append(f"🔴 MACD 空頭格局，動能偏弱")
    # BB
    if bb_w < bb_w2*0.85: lines.append("⚡ 布林帶**收縮**，波動壓縮，可能醞釀突破方向")
    elif bb_w > bb_w2*1.15 and bb_pos > 0.6: lines.append("🚀 布林帶**向上擴張**，強勢突破上軌")
    elif bb_pos > 0.85: lines.append("⚠ 價格貼近布林上軌，注意短線回測中軌壓力")
    elif bb_pos < 0.15: lines.append("🟢 價格貼近布林下軌，可能出現技術反彈")
    # MA
    if m5>m20>m60: lines.append("📈 均線**多頭排列**（MA5>MA20>MA60），趨勢向上")
    elif m5<m20<m60: lines.append("📉 均線**空頭排列**（MA5<MA20<MA60），趨勢向下")
    # Volume
    vr = vol/max(vol_m,1)
    if vr>=2.0: lines.append(f"🔊 成交量**爆增**（{vr:.1f}x 均量），留意主力動向")
    elif vr<=0.5: lines.append(f"🔇 成交量**萎縮**（{vr:.1f}x 均量），行情觀望為主")
    # Forecast
    fc_c = SUCCESS if chg>0 else DANGER
    lines.append(f"🎯 預測 {result['forecast_days']} 日後中位價 **{med:,.2f}** "
                 f"（**{chg:+.1f}%**），drift={result.get('drift_bias',0):+.5f}")
    # ML
    if ml_p.get("prob_up") is not None:
        p = ml_p["prob_up"]*100
        lbl = ml_p.get("label","")
        lines.append(f"🤖 ML 融合上漲機率 **{p:.1f}%**（{lbl}）")
    if bt.get("n_trades",0) > 0:
        lines.append(f"📊 回測命中率 **{bt['hit_rate_text']}**（{bt['n_trades']} 筆，含交易成本）")

    for line in lines:
        st.markdown(f"- {line}")


# ═══════════════════════════════════════════════════════════════════════════
# Batch analysis
# ═══════════════════════════════════════════════════════════════════════════
def render_batch():
    wl = st.session_state.watchlist
    if not wl:
        st.warning("自選股為空")
        return
    st.markdown(f"## 批次分析  —  {len(wl)} 支股票")
    results = []
    prog = st.progress(0)
    for idx, code in enumerate(wl):
        prog.progress((idx+1)/len(wl), text=f"分析中 {idx+1}/{len(wl)}: {code}")
        try:
            r = run_analysis(code,
                             lookback_years = st.session_state.lookback_years,
                             forecast_days  = st.session_state.forecast_days,
                             weights        = st.session_state.weights,
                             train_ratio    = st.session_state.train_ratio)
            results.append(r)
        except Exception as e:
            results.append({"symbol":code,"name":code,"error":str(e)})
    prog.empty()
    st.session_state.batch_results = results

    # Build summary table
    rows = []
    for r in results:
        if "error" in r:
            rows.append({"代碼":r["symbol"],"名稱":r["name"],
                         "建議":"—","可信度":"—","現價":"—","ML%":"—","預測%":"—",
                         "命中率":"—","錯誤":r["error"]})
            continue
        df   = r["df"]
        ml_p = r.get("ml_predict",{})
        bt   = r.get("backtest",{})
        fc   = r["forecast"]
        last = float(df["Close"].iloc[-1])
        med  = float(fc["median"][-1])
        chg  = (med-last)/last*100
        ml_prob = ml_p.get("prob_up",0.5)
        # Score
        ind = r["indicators"]
        mh  = float(ind["macd_hist"].iloc[-1]) if not pd.isna(ind["macd_hist"].iloc[-1]) else 0
        rsi = float(ind["rsi14"].iloc[-1]) if not pd.isna(ind["rsi14"].iloc[-1]) else 50
        w   = r.get("weights", st.session_state.weights)
        tech_s = (0.5 if mh>0 else -0.5) + (0.3 if rsi<35 else (-0.3 if rsi>65 else 0))
        ml_s   = (ml_prob-0.5)*2
        score  = tech_s*w.get("technical",40)/100*2 + ml_s*w.get("ml",35)/100*2
        if score>=1.5:   act="偏多·可試單"
        elif score>=0.5: act="偏多觀察"
        elif score<=-1.5:act="偏空·減碼"
        elif score<=-0.5:act="偏空觀察"
        else:            act="中性觀望"
        bt_hr = bt.get("hit_rate",0.5) if bt.get("n_trades",0)>0 else 0.5
        cs = (abs(ml_prob-0.5)*2 + abs(bt_hr-0.5)*2)/2
        conf = "高" if cs>=0.35 else ("中" if cs>=0.18 else "低")
        rows.append({
            "代碼":    r["symbol"].split(".")[0],
            "名稱":    r["name"],
            "建議":    act,
            "可信度":  conf,
            "現價":    f"{last:,.2f}",
            "ML%":     f"{ml_prob*100:.1f}%",
            "預測%":   f"{chg:+.1f}%",
            "命中率":  bt.get("hit_rate_text","N/A"),
        })

    df_batch = pd.DataFrame(rows)
    st.dataframe(df_batch, use_container_width=True, hide_index=True)

    # Download
    csv = df_batch.to_csv(index=False, encoding="utf-8-sig")
    st.download_button("💾 下載 CSV", csv,
                       file_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                       mime="text/csv")


# ═══════════════════════════════════════════════════════════════════════════
# Main layout
# ═══════════════════════════════════════════════════════════════════════════
from datetime import datetime

symbol, analyze_btn = render_sidebar()

# Trigger analysis
if analyze_btn and symbol:
    with st.spinner(""):
        do_analysis(symbol.strip().upper())

# Batch
if st.session_state.get("_do_batch"):
    st.session_state._do_batch = False
    render_batch()
elif st.session_state.result is not None:
    render_result(st.session_state.result)
else:
    # Landing page
    st.markdown("""
    <div style='text-align:center;padding:60px 20px;'>
        <div style='font-size:4rem'>📈</div>
        <h1 style='font-size:2rem'>Stock Analyzer Pro</h1>
        <p style='color:#8890aa;font-size:1.1rem;max-width:500px;margin:0 auto'>
            台灣股市智能分析系統<br>
            在左側輸入股票代碼（如 2330、0050、00878）按「分析」開始
        </p>
        <div style='margin-top:30px;color:#6c8ef5;font-size:0.9rem'>
            支援：上市 · 上櫃 · ETF · 槓桿反向ETF · 美股
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick examples
    st.divider()
    st.markdown("#### 快速選股")
    examples = [
        ("台積電","2330"),("元大台灣50","0050"),
        ("國泰永續高股息","00878"),("台積電正2","00631L"),
        ("聯發科","2454"),("長榮","2603"),
    ]
    cols = st.columns(len(examples))
    for col, (name, code) in zip(cols, examples):
        if col.button(f"{name}\n{code}", use_container_width=True):
            with st.spinner(""):
                do_analysis(code)
            st.rerun()
