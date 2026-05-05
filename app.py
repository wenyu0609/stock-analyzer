from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core import (
    analyze_stock,
    format_pct,
    get_intraday,
    read_api_key,
    support_resistance,
)

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None


st.set_page_config(
    page_title="台股分析器",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _secret_key() -> str:
    try:
        return (
            st.secrets.get("FUGLE_API_KEY", "")
            or st.secrets.get("FUBON_MARKETDATA_API_KEY", "")
        )
    except Exception:
        return os.getenv("FUGLE_API_KEY", "") or os.getenv("FUBON_MARKETDATA_API_KEY", "")


@st.cache_data(ttl=15 * 60, show_spinner=False)
def cached_analysis(symbol: str, years: int, forecast_days: int, api_key: str):
    return analyze_stock(symbol, years=years, forecast_days=forecast_days, api_key=api_key)


@st.cache_data(ttl=10, show_spinner=False)
def cached_intraday(symbol: str, history: pd.DataFrame):
    return get_intraday(symbol, history)


def history_figure(result, chart_style: str) -> go.Figure:
    hist = result.indicators.copy()
    fc = result.forecast.copy()
    fig = go.Figure()

    if chart_style == "K棒":
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist["Open"],
            high=hist["High"],
            low=hist["Low"],
            close=hist["Close"],
            name="歷史K棒",
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=hist["Close"],
            mode="lines",
            name="收盤價",
            line=dict(color="#2563eb", width=2),
            hovertemplate="日期=%{x|%Y-%m-%d}<br>收盤=%{y:.2f}<extra></extra>",
        ))

    for col, color in [("MA20", "#f59e0b"), ("MA60", "#7c3aed")]:
        if col in hist:
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist[col],
                mode="lines",
                name=col,
                line=dict(color=color, width=1.4),
                hovertemplate=f"日期=%{{x|%Y-%m-%d}}<br>{col}=%{{y:.2f}}<extra></extra>",
            ))

    if not fc.empty:
        fig.add_trace(go.Scatter(
            x=fc["Date"],
            y=fc["Median"],
            mode="lines",
            name="預測中位價",
            line=dict(color="#0f766e", width=2.5, dash="dash"),
            hovertemplate="日期=%{x|%Y-%m-%d}<br>預測=%{y:.2f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=fc["Date"],
            y=fc["Lower"],
            mode="lines",
            name="預測下緣",
            line=dict(color="rgba(15,118,110,0.15)", width=0),
            hoverinfo="skip",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=fc["Date"],
            y=fc["Upper"],
            mode="lines",
            name="預測區間",
            fill="tonexty",
            fillcolor="rgba(15,118,110,0.14)",
            line=dict(color="rgba(15,118,110,0.15)", width=0),
            hoverinfo="skip",
        ))

    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=20, b=10),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="價格")
    return fig


def intraday_actual_figure(intra, chart_style: str) -> go.Figure:
    df = intra.data.copy()
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.72, 0.28],
    )
    lots = df.get("Volume_lots", df["Volume"] / 1000.0)

    if chart_style == "K棒":
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="實際K棒",
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines+markers",
            name="實際價格",
            customdata=lots,
            line=dict(color="#2563eb", width=2),
            marker=dict(size=4),
            hovertemplate="時間=%{x|%H:%M}<br>價格=%{y:.2f}<br>成交量=%{customdata:.0f} 張<extra></extra>",
        ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df.index,
        y=lots,
        name="1分鐘張數",
        marker_color="rgba(100,116,139,0.45)",
        hovertemplate="時間=%{x|%H:%M}<br>成交量=%{y:.0f} 張<extra></extra>",
    ), row=2, col=1)
    fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=20, b=10),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", y=1.04),
    )
    fig.update_yaxes(title_text="價格", row=1, col=1)
    fig.update_yaxes(title_text="張數", row=2, col=1)
    return fig


def intraday_prediction_figure(intra) -> go.Figure:
    pred = intra.prediction.copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pred["Time"],
        y=pred["Predicted"],
        mode="lines",
        name="預測走向",
        line=dict(color="#0f766e", width=2.4),
        hovertemplate="時間=%{x|%H:%M}<br>預測價格=%{y:.2f}<extra></extra>",
    ))
    actual = pred.dropna(subset=["Actual"])
    if not actual.empty:
        fig.add_trace(go.Scatter(
            x=actual["Time"],
            y=actual["Actual"],
            mode="lines",
            name="已知實際價格",
            line=dict(color="#2563eb", width=1.8, dash="dot"),
            hovertemplate="時間=%{x|%H:%M}<br>實際價格=%{y:.2f}<extra></extra>",
        ))
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=20, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.04),
    )
    fig.update_yaxes(title_text="價格")
    return fig


def show_summary(result):
    hist = result.history
    latest = float(hist["Close"].iloc[-1])
    prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else latest
    pred = result.prediction
    bt = result.backtest

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("最新收盤", f"{latest:.2f}", f"{(latest / prev - 1) * 100:+.2f}%")
    c2.metric("預測報酬", format_pct(pred.get("expected_return", 0.0)))
    c3.metric("上漲機率", f"{pred.get('prob_up', 0.5) * 100:.1f}%")
    c4.metric("回測命中", f"{bt.get('hit_rate', 0) * 100:.1f}%" if bt.get("n", 0) else "N/A")
    c5.metric("資料筆數", f"{len(hist):,}")


def show_technical_snapshot(result):
    ind = result.indicators
    sr = support_resistance(ind)
    last = ind.iloc[-1]
    rows = [
        ("RSI14", f"{last.get('RSI14', 0):.1f}"),
        ("MACD柱", f"{last.get('MACD_HIST', 0):+.3f}"),
        ("MA20", f"{last.get('MA20', 0):.2f}"),
        ("MA60", f"{last.get('MA60', 0):.2f}"),
        ("支撐", f"{sr.get('support', 0):.2f}"),
        ("壓力", f"{sr.get('resistance', 0):.2f}"),
        ("量比20日", f"{last.get('VOL_RATIO20', 0):.2f}"),
        ("ATR14", f"{last.get('ATR14', 0):.2f}"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["項目", "數值"]), hide_index=True, width="stretch")


def _style_dataframe_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out:
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    for col in ("predicted_return", "actual_return"):
        if col in out:
            out[col] = out[col] * 100
    if "hit" in out:
        out["hit"] = out["hit"].map({True: "命中", False: "未命中"})
    return out


st.title("台股分析器")

with st.sidebar:
    symbol = st.text_input("股票代碼", value="2330", placeholder="例如 2330、8069、00981A")
    years = st.slider("歷史資料年數", min_value=1, max_value=10, value=3)
    forecast_days = st.slider("預測天數", min_value=1, max_value=60, value=20)
    chart_style = st.radio("圖表型態", ["曲線", "K棒"], horizontal=True)
    auto_refresh = st.toggle("盤中資料 10 秒自動更新", value=False)
    manual_key = st.text_input("Fugle API Key（可留空）", type="password")
    api_key = read_api_key(manual_key or _secret_key())

    if auto_refresh:
        if st_autorefresh is not None:
            st_autorefresh(interval=10_000, key="intraday_refresh")
        else:
            st.warning("requirements.txt 需包含 streamlit-autorefresh 才能自動更新。")

    if st.button("手動更新 / 清除快取", type="primary", width="stretch"):
        st.cache_data.clear()
        st.rerun()

if not symbol.strip():
    st.info("請輸入股票代碼。")
    st.stop()

try:
    with st.spinner("抓取資料、訓練報酬率模型與執行 walk-forward 回測中..."):
        result = cached_analysis(symbol.strip(), years, forecast_days, api_key)
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.subheader(f"{result.name} ({result.symbol})")
st.caption(f"資料來源：{result.source}｜模型目標：未來 {forecast_days} 日報酬率，圖上的價格線是由預測報酬率換算而來。")
show_summary(result)

tab_main, tab_intraday, tab_backtest, tab_data = st.tabs(["總覽", "盤中走向", "模型回測明細", "資料"])

with tab_main:
    left, right = st.columns([3.2, 1])
    with left:
        st.plotly_chart(history_figure(result, chart_style), width="stretch")
    with right:
        st.markdown("#### 技術快照")
        show_technical_snapshot(result)
        st.markdown("#### 模型")
        st.write({
            "model": result.prediction.get("model", ""),
            "segment": result.prediction.get("segment", ""),
            "residual_std": round(float(result.prediction.get("residual_std", 0)), 4),
        })

with tab_intraday:
    try:
        intra = cached_intraday(result.symbol, result.history)
        stats = intra.stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("開盤", f"{stats.get('open', 0):.2f}")
        c2.metric("最新", f"{stats.get('latest', 0):.2f}", format_pct(stats.get("change_pct", 0)))
        c3.metric("最高 / 最低", f"{stats.get('high', 0):.2f} / {stats.get('low', 0):.2f}")
        c4.metric("累計張數", f"{stats.get('volume_lots', 0):,.0f}")
        st.caption(f"盤中來源：{intra.source}｜最後時間：{stats.get('last_time')}")
        st.plotly_chart(intraday_actual_figure(intra, chart_style), width="stretch")
        st.plotly_chart(intraday_prediction_figure(intra), width="stretch")

        rows = intra.data.copy()
        rows = rows.reset_index().rename(columns={"index": "Time"})
        rows["Time"] = pd.to_datetime(rows["Time"]).dt.strftime("%H:%M")
        rows = rows[["Time", "Open", "High", "Low", "Close", "Volume_lots"]].rename(columns={
            "Open": "開盤", "High": "最高", "Low": "最低", "Close": "價格", "Volume_lots": "1分鐘張數",
        })
        st.markdown("#### 1分鐘逐列張數")
        st.dataframe(rows.tail(120), hide_index=True, width="stretch")
    except Exception as exc:
        st.warning(f"目前抓不到盤中資料：{exc}")

with tab_backtest:
    bt = result.backtest
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("測試次數", f"{bt.get('n', 0)}")
    c2.metric("方向命中率", f"{bt.get('hit_rate', 0) * 100:.1f}%" if bt.get("n", 0) else "N/A")
    c3.metric("平均絕對誤差", f"{bt.get('mae', 0) * 100:.2f}%")
    c4.metric("平均實際報酬", f"{bt.get('avg_actual_return', 0) * 100:+.2f}%")
    detail = _style_dataframe_returns(result.backtest_detail)
    if detail.empty:
        st.info("資料筆數不足，尚無 walk-forward 明細。")
    else:
        st.dataframe(
            detail.rename(columns={
                "date": "日期",
                "segment": "segment",
                "predicted_return": "預測報酬(%)",
                "actual_return": "實際報酬(%)",
                "hit": "是否命中",
            }),
            hide_index=True,
            width="stretch",
        )

with tab_data:
    st.markdown("#### 歷史資料")
    hist = result.history.reset_index().rename(columns={"index": "Date"})
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.strftime("%Y-%m-%d")
    st.dataframe(hist.tail(300), hide_index=True, width="stretch")
    st.markdown("#### 預測路徑")
    fc = result.forecast.copy()
    fc["Date"] = pd.to_datetime(fc["Date"]).dt.strftime("%Y-%m-%d")
    st.dataframe(fc, hide_index=True, width="stretch")

st.caption("本工具僅供研究與風險評估，不構成投資建議。")
