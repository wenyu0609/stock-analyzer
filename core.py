# core.py — Backend engine extracted from Stock Analyzer Pro v11
# All analysis logic: indicators, ML, forecast, backtest, names
# This file has ZERO PySide6 / GUI dependencies — pure Python data layer

import os, sys, json, time, warnings, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Optional deps ──────────────────────────────────────────────────────────
try:
    import yfinance as yf
    YF_OK = True
except ImportError:
    yf = None; YF_OK = False

try:
    import requests
    REQUESTS_OK = True
except ImportError:
    requests = None; REQUESTS_OK = False

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import brier_score_loss, accuracy_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    import torch
    import torch.nn as nn_torch
    import torch.optim as optim
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

# ── Paths ──────────────────────────────────────────────────────────────────
SETTINGS_DIR   = Path.home() / ".stock_analyzer_pro"
NN_SAMPLES_PATH = SETTINGS_DIR / "nn_samples.npz"
NN_MODEL_PATH   = SETTINGS_DIR / "nn_lstm_model.pt"
_NAMES_CACHE_FILE = SETTINGS_DIR / "tw_names.json"
SETTINGS_DIR.mkdir(parents=True, exist_ok=True)


DEFAULT_WEIGHTS = {
    "technical": 25,
    "ml": 25,
    "news": 10,
    "fundamental": 10,
    "us_market": 10,
    "institutional": 15,
    "margin": 5,
}

class StockAnalysisError(RuntimeError):
    """User-facing stock analysis error."""

class StockNotFoundError(StockAnalysisError):
    """Raised when a symbol cannot be resolved or has no usable data."""

class DataFetchTimeout(StockAnalysisError):
    """Raised when an external data source times out."""

def friendly_error_message(exc: Exception) -> str:
    """Convert low-level exceptions into Streamlit-friendly Chinese messages."""
    msg = str(exc)
    low = msg.lower()
    if isinstance(exc, StockNotFoundError) or "查無" in msg or "no data" in low or "無資料" in msg:
        return "找不到這個股票代碼，請確認代碼是否正確；台股可輸入 2330、0050、00631L，或直接輸入 2330.TW。"
    if isinstance(exc, DataFetchTimeout) or "timeout" in low or "timed out" in low or "read timed out" in low:
        return "資料來源連線逾時，可能是 Yahoo Finance / TWSE 暫時忙碌。請稍後重試，或先縮短回溯年數。"
    if "429" in msg or "rate" in low or "too many" in low:
        return "資料來源暫時限制查詢頻率，請等一下再重新分析。"
    if "yfinance 未安裝" in msg:
        return "目前環境缺少 yfinance，請先安裝：pip install yfinance"
    return f"分析時發生問題：{msg[:180]}"

_ANALYSIS_CACHE = {}
_ANALYSIS_CACHE_TTL = 900  # 15 minutes

def _safe_float(v, default=0.0):
    try:
        if v is None: return default
        if isinstance(v, str):
            v = v.replace(",", "").replace("--", "").strip()
            if v in ("", "-", "NaN", "nan"): return default
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default

def _tw_date_candidates(days: int = 10):
    base = pd.Timestamp.today().normalize()
    return [(base - pd.Timedelta(days=i)).strftime("%Y%m%d") for i in range(days)]

TX_ROUND_TRIP  = 0.001425 * 2 + 0.003
LSTM_SEQ_LEN   = 20
LSTM_FEATURES  = 19
LSTM_HIDDEN    = 64
LSTM_LAYERS    = 2
LSTM_DROPOUT   = 0.25
LSTM_EPOCHS    = 30
LSTM_LR        = 0.001
LSTM_BATCH     = 32
MAX_NN_SAMPLES = 25000

# ═══════════════════════════════════════════════════════════════════════════
# Stock name system
# ═══════════════════════════════════════════════════════════════════════════
TW_NAME_CACHE: dict = {
    "2330":"台積電","2454":"聯發科","2303":"聯電","2308":"台達電",
    "3008":"大立光","2382":"廣達","3017":"奇鋐","2379":"瑞昱",
    "2317":"鴻海","2357":"華碩","2324":"仁寶","2356":"英業達",
    "2376":"技嘉","3231":"緯創","3702":"大聯大","3034":"聯詠",
    "4938":"和碩","2395":"研華","2412":"中華電","1216":"統一",
    "1301":"台塑","1303":"南亞","2002":"中鋼","2207":"和泰車",
    "2603":"長榮","2609":"陽明","2615":"萬海","6446":"藥華藥",
    "4743":"合一","1101":"台泥","2912":"統一超","5274":"信驊",
    "6669":"緯穎","2881":"富邦金","2882":"國泰金","2891":"中信金",
    "2886":"兆豐金","2884":"玉山金","2885":"元大金","2880":"華南金",
    "2883":"開發金","2892":"第一金","2890":"永豐金","2887":"台新金",
    "5880":"合庫金",
    "0050":"元大台灣50","0056":"元大高股息","006208":"富邦台50",
    "00878":"國泰永續高股息","00919":"群益台灣精選高息",
    "00929":"復華台灣科技優息","00940":"元大台灣價值高息",
    "00713":"元大台灣高息低波","00692":"富邦公司治理",
    "00631L":"元大台灣50正2","00632R":"元大台灣50反1",
    "00633L":"富邦台灣加強","00634R":"富邦台灣反1",
    "00663L":"國泰臺灣加強","00664R":"國泰臺灣反1",
    "00670L":"富邦NASDAQ正2","00671R":"富邦NASDAQ反1",
    "00672L":"元大S&P正2","00673R":"元大S&P反1",
    "00685L":"群益臺灣加權正2","00686R":"群益臺灣加權反1",
    "00980A":"中信高優息成長ETF",
    "00981A":"主動統一台股增長",
    "00982A":"主動群益台灣強棒",
}

_twse_bulk_loaded = False
_tpex_bulk_loaded = False

def _load_names_disk_cache():
    try:
        if _NAMES_CACHE_FILE.exists():
            with open(_NAMES_CACHE_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            for k, v in saved.items():
                if k not in TW_NAME_CACHE:
                    TW_NAME_CACHE[k] = v
    except Exception:
        pass

def _save_names_disk_cache():
    try:
        with open(_NAMES_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(TW_NAME_CACHE, f, ensure_ascii=False, indent=1)
    except Exception:
        pass

_load_names_disk_cache()

def _cjk(s: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in str(s))

CODE_KEYS = ("Code","SecuritiesCode","股票代號","code","stockCode","companyCode")
NAME_KEYS = ("Name","SecuritiesName","有價證券名稱","name","companyName","shortName")

def _load_twse_bulk():
    global _twse_bulk_loaded
    if _twse_bulk_loaded or not REQUESTS_OK: return
    _twse_bulk_loaded = True
    hdrs = {"User-Agent":"Mozilla/5.0","Accept":"application/json"}
    endpoints = [
        "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
        "https://openapi.twse.com.tw/v1/opendata/t187ap03_L",
        "https://openapi.twse.com.tw/v1/opendata/t187ap05_L",
    ]
    added = 0
    for url in endpoints:
        try:
            r = requests.get(url, timeout=10, headers=hdrs, verify=False)
            if r.status_code != 200: continue
            data = r.json()
            if not isinstance(data, list): continue
            for item in data:
                code = next((str(item.get(k,"")).strip() for k in CODE_KEYS if item.get(k)), "")
                name = next((str(item.get(k,"")).strip() for k in NAME_KEYS if item.get(k)), "")
                if (code and name and len(code)<=7 and code[0].isdigit()
                        and code not in TW_NAME_CACHE):
                    TW_NAME_CACHE[code] = name; added += 1
        except Exception:
            continue
    if added > 0: _save_names_disk_cache()

def _load_tpex_bulk():
    global _tpex_bulk_loaded
    if _tpex_bulk_loaded or not REQUESTS_OK: return
    _tpex_bulk_loaded = True
    hdrs = {"User-Agent":"Mozilla/5.0","Accept":"application/json"}
    for url in ["https://www.tpex.org.tw/openapi/v1/tpex_mainboard_perday_quotes",
                "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"]:
        try:
            r = requests.get(url, timeout=10, headers=hdrs, verify=False)
            if r.status_code != 200: continue
            data = r.json()
            if not isinstance(data, list): continue
            for item in data:
                code = next((str(item.get(k,"")).strip() for k in CODE_KEYS if item.get(k)), "")
                name = next((str(item.get(k,"")).strip() for k in NAME_KEYS if item.get(k)), "")
                if code and name and code not in TW_NAME_CACHE:
                    TW_NAME_CACHE[code] = name
        except Exception:
            continue

def _yf_history_with_retry(ticker, period: str, max_retries: int = 3) -> pd.DataFrame:
    """Call ticker.history() with exponential backoff on rate limit."""
    for attempt in range(max_retries):
        try:
            df = ticker.history(period=period, auto_adjust=True)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "429" in err or "too many" in err:
                wait = 5 * (2 ** attempt)
                time.sleep(wait)
                continue
            raise
    # Last attempt without catching
    return ticker.history(period=period, auto_adjust=True)


def resolve_and_fetch(symbol: str, lookback_years: int = 3) -> Tuple[str, str, pd.DataFrame]:
    """Fetch history + resolve name. Returns (resolved_sym, name, df)."""
    if not YF_OK:
        raise RuntimeError("yfinance 未安裝")
    _load_twse_bulk(); _load_tpex_bulk()
    raw = symbol.strip().upper().split(".")[0]
    candidates = []
    if symbol.endswith((".TW",".TWO")):
        candidates = [symbol]
    elif "." in symbol:
        candidates = [symbol]
    else:
        candidates = [f"{raw}.TW", f"{raw}.TWO", raw]

    HARD_MIN = 30
    period   = f"{max(lookback_years,1)}y"
    best     = None
    last_err = ""
    for sym in candidates:
        try:
            ticker = yf.Ticker(sym)
            df = _yf_history_with_retry(ticker, period)
            if df is None or df.empty: last_err = f"{sym}: 無資料"; continue
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df[["Open","High","Low","Close","Volume"]].astype(float)
            n = len(df)
            if n < HARD_MIN: last_err = f"{sym}: {n}筆不足"; continue
            # Resolve name
            rc = sym.split(".")[0]
            name = TW_NAME_CACHE.get(rc, "")
            if not name or not _cjk(name):
                try:
                    info = ticker.info or {}
                    for key in ("shortName","longName","displayName"):
                        v = str(info.get(key,"") or "").strip()
                        if v and v != rc and _cjk(v):
                            TW_NAME_CACHE[rc] = v; _save_names_disk_cache()
                            name = v; break
                except Exception:
                    pass
            if not name or not _cjk(name):
                name = rc
            if n >= 80:
                return sym, name, df
            if best is None:
                best = (sym, name, df)
                last_err = f"{sym}: {n}筆（可分析）"
        except Exception as e:
            last_err = f"{sym}: {e}"
    if best:
        return best
    raise RuntimeError(f"查無 {symbol}：{last_err}")


# ═══════════════════════════════════════════════════════════════════════════
# Technical Indicators
# ═══════════════════════════════════════════════════════════════════════════
def compute_indicators(df: pd.DataFrame) -> dict:
    c = df["Close"]; h = df["High"]; lo = df["Low"]; v = df["Volume"]
    e12  = c.ewm(span=12,adjust=False).mean()
    e26  = c.ewm(span=26,adjust=False).mean()
    macd_l = e12 - e26
    macd_s = macd_l.ewm(span=9,adjust=False).mean()
    macd_h = macd_l - macd_s
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/14,min_periods=14,adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/14,min_periods=14,adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi14 = (100 - 100/(1+rs)).fillna(50)
    ma5   = c.rolling(5,min_periods=1).mean()
    ma20  = c.rolling(20,min_periods=1).mean()
    ma60  = c.rolling(60,min_periods=1).mean()
    bb_mid = c.rolling(20,min_periods=1).mean()
    bb_std = c.rolling(20,min_periods=1).std().fillna(0)
    bb_up  = bb_mid + 2*bb_std
    bb_dn  = bb_mid - 2*bb_std
    bb_w   = ((bb_up - bb_dn) / bb_mid.replace(0, np.nan) * 100).fillna(5)
    atr14  = (h - lo).rolling(14,min_periods=1).mean()
    vol_ma20 = v.rolling(20,min_periods=1).mean()
    return dict(
        macd_line=macd_l, macd_signal=macd_s, macd_hist=macd_h,
        macd_hist_change=macd_h.diff().fillna(0),
        rsi14=rsi14,
        ma5=ma5, ma20=ma20, ma60=ma60,
        bb_up=bb_up, bb_dn=bb_dn, bb_width=bb_w,
        atr14=atr14, vol_ma20=vol_ma20,
    )

def compute_support_resistance(df: pd.DataFrame, lookback: int = 60) -> dict:
    """Recent support/resistance zones based on actual lows/highs."""
    recent = df.tail(min(lookback, len(df)))
    hi = recent["High"]
    lo = recent["Low"]
    return dict(
        support_lo=float(lo.min()),
        support_hi=float(lo.quantile(0.15)),
        resistance_lo=float(hi.quantile(0.85)),
        resistance_hi=float(hi.max()),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Feature engineering
# ═══════════════════════════════════════════════════════════════════════════
def build_feature_row(df, ind, sr, idx) -> Optional[np.ndarray]:
    if idx < 20: return None
    try:
        c = df["Close"]; v = df["Volume"]
        p = float(c.iloc[idx])
        ret_1  = p / float(c.iloc[idx-1]) - 1 if idx >= 1 else 0
        ret_5  = p / float(c.iloc[idx-5]) - 1 if idx >= 5 else 0
        ret_20 = p / float(c.iloc[idx-20]) - 1 if idx >= 20 else 0
        vm = float(ind["vol_ma20"].iloc[idx])
        vol_ratio = float(v.iloc[idx]) / max(vm, 1)
        rsi = float(ind["rsi14"].iloc[idx]) if not pd.isna(ind["rsi14"].iloc[idx]) else 50.0
        macd_h = float(ind["macd_hist"].iloc[idx]) if not pd.isna(ind["macd_hist"].iloc[idx]) else 0.0
        macd_hc = float(ind["macd_hist_change"].iloc[idx]) if not pd.isna(ind["macd_hist_change"].iloc[idx]) else 0.0
        bb_up = float(ind["bb_up"].iloc[idx]); bb_dn = float(ind["bb_dn"].iloc[idx])
        bb_pos = (p - bb_dn) / max(bb_up - bb_dn, 1e-9)
        bb_w   = float(ind["bb_width"].iloc[idx]) if not pd.isna(ind["bb_width"].iloc[idx]) else 5.0
        atr    = float(ind["atr14"].iloc[idx]) if not pd.isna(ind["atr14"].iloc[idx]) else p*0.02
        atr_pct = atr / p
        m5 = float(ind["ma5"].iloc[idx]); m20 = float(ind["ma20"].iloc[idx])
        m60 = float(ind["ma60"].iloc[idx])
        ma_cross = (m5 - m20) / p
        sr_d_sup = (p - sr["support_hi"])  / p
        sr_d_res = (sr["resistance_lo"] - p) / p
        mtf = 0
        if p > m5 > m20 > m60: mtf = 1
        elif p < m5 < m20 < m60: mtf = -1
        dt = df.index[idx]
        month_sin = float(np.sin(2*np.pi*dt.month/12))
        month_cos = float(np.cos(2*np.pi*dt.month/12))
        dow_sin   = float(np.sin(2*np.pi*dt.dayofweek/5))
        win = min(20, idx)
        if win > 0:
            rh = float(df["High"].iloc[idx-win:idx].max())
            rl = float(df["Low"].iloc[idx-win:idx].min())
            dist_hi = (p - rh) / max(rh, 1e-9)
            dist_lo = (p - rl) / max(rl, 1e-9)
        else:
            dist_hi = dist_lo = 0.0
        w52 = min(252, idx)
        if w52 >= 10:
            hi52 = float(df["High"].iloc[idx-w52:idx].max())
            lo52 = float(df["Low"].iloc[idx-w52:idx].min())
            brk_hi = 1.0 if p >= hi52*0.99 else 0.0
            brk_lo = 1.0 if p <= lo52*1.01 else 0.0
        else:
            brk_hi = brk_lo = 0.0
        vec = np.array([
            ret_1, ret_5, ret_20, vol_ratio, rsi,
            macd_h, macd_hc, bb_pos, bb_w, atr_pct,
            ma_cross, sr_d_sup, sr_d_res, mtf,
            month_sin, month_cos, dow_sin,
            dist_hi, dist_lo, brk_hi, brk_lo,
            0., 0., 0., 0., 0., 0.,
        ], dtype=float)
        if not np.all(np.isfinite(vec)): return None
        return vec
    except Exception:
        return None

def build_full_feature_row(df, ind, sr, idx, mkt_ctx=None, inst=None) -> Optional[np.ndarray]:
    """
    27-dim feature row.
    Dims 0-20: technical + calendar + price pattern.
    Dims 21-26: market/institution/margin context.
    """
    if mkt_ctx is None: mkt_ctx = {}
    if inst    is None: inst    = {}
    margin = mkt_ctx.get("_margin", {}) or {}
    vec = build_feature_row(df, ind, sr, idx)
    if vec is None or len(vec) != 27: return None
    vec[21] = float(np.clip(mkt_ctx.get("taiex_ret_1",0.0), -0.1, 0.1))
    vec[22] = float(np.clip(mkt_ctx.get("nasdaq_ret_1",0.0), -0.1, 0.1))
    vec[23] = float(np.clip(mkt_ctx.get("semis_ret_1",0.0), -0.1, 0.1))
    vec[24] = float(mkt_ctx.get("vix",20.0)) / 40.0
    vec[25] = float(np.clip(inst.get("inst_score", inst.get("inst_combo",0.0)), -1, 1))
    vec[26] = float(np.clip(margin.get("margin_score",0.0), -1, 1))
    if not np.all(np.isfinite(vec)): return None
    return vec


# ═══════════════════════════════════════════════════════════════════════════
# Forecast
# ═══════════════════════════════════════════════════════════════════════════
def compute_drift_bias(ind, ml_pred, nn_pred, ns, fund, weights,
                       mkt_ctx=None, inst=None, margin=None) -> float:
    """
    Convert weighted signals into a conservative daily drift adjustment.
    Signals included: technical, ML/NN, news, fundamentals, US market,
    institutional flow, and margin financing/short data.
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    mkt_ctx = mkt_ctx or {}
    inst = inst or {}
    margin = margin or {}

    w_tech = weights.get("technical", 25) / 100.0
    w_ml   = weights.get("ml", 25) / 100.0
    w_news = weights.get("news", 10) / 100.0
    w_fund = weights.get("fundamental", 10) / 100.0
    w_us   = weights.get("us_market", 10) / 100.0
    w_inst = weights.get("institutional", 15) / 100.0
    w_marg = weights.get("margin", 5) / 100.0

    tech_sig = 0.0
    try:
        mh  = float(ind["macd_hist"].iloc[-1])
        rsi = float(ind["rsi14"].iloc[-1])
        bbu = float(ind["bb_up"].iloc[-1]); bbd = float(ind["bb_dn"].iloc[-1])
        cl  = float(ind["ma5"].iloc[-1])
        mstd = float(ind["macd_hist"].std()) if len(ind["macd_hist"]) > 5 else 1.0
        tech_sig += float(np.clip(mh / max(mstd, 1e-9), -1, 1)) * 0.45
        tech_sig += float(np.clip((rsi - 50) / 50, -1, 1)) * 0.25
        bpos = (cl - bbd) / max(bbu - bbd, 1e-9)
        tech_sig += float(np.clip((0.5 - abs(bpos - 0.5)) * 2, -1, 1)) * 0.15
        if len(ind["ma5"]) and float(ind["ma5"].iloc[-1]) > float(ind["ma20"].iloc[-1]) > float(ind["ma60"].iloc[-1]):
            tech_sig += 0.15
        elif len(ind["ma5"]) and float(ind["ma5"].iloc[-1]) < float(ind["ma20"].iloc[-1]) < float(ind["ma60"].iloc[-1]):
            tech_sig -= 0.15
    except Exception:
        pass
    tech_sig = float(np.clip(tech_sig, -1, 1))

    gb_p = (ml_pred or {}).get("prob_up", 0.5)
    nn_p = (nn_pred or {}).get("prob_up", gb_p)
    ml_sig = float(np.clip((gb_p * 0.55 + nn_p * 0.45 - 0.5) * 2, -1, 1))

    lbl = (ns or {}).get("label", "中性")
    news_score = (ns or {}).get("score", None)
    if news_score is None:
        news_sig = 0.6 if lbl in ("正面", "偏正面") else (-0.6 if lbl in ("負面", "偏負面") else 0.0)
    else:
        news_sig = float(np.clip(news_score, -1, 1))

    fs = 0.0
    if fund:
        pe = fund.get("pe_ratio")
        roe = fund.get("roe")
        dy = fund.get("div_yield")
        if pe and 0 < pe < 15: fs += 0.45
        if pe and pe > 40: fs -= 0.45
        if roe and roe > 0.15: fs += 0.45
        if roe and roe < 0.05: fs -= 0.25
        if dy and dy > 0.04: fs += 0.15
    fs = float(np.clip(fs, -1, 1))

    us_sig = (
        np.clip(mkt_ctx.get("nasdaq_ret_1", 0.0) / 0.025, -1, 1) * 0.30 +
        np.clip(mkt_ctx.get("sp500_ret_1", 0.0) / 0.020, -1, 1) * 0.20 +
        np.clip(mkt_ctx.get("semis_ret_1", 0.0) / 0.030, -1, 1) * 0.35 +
        np.clip(mkt_ctx.get("taiex_ret_1", 0.0) / 0.020, -1, 1) * 0.15
    )
    us_sig = float(np.clip(us_sig, -1, 1))

    inst_sig = float(np.clip(inst.get("inst_score", inst.get("inst_combo", 0.0)), -1, 1))
    margin_sig = float(np.clip(margin.get("margin_score", 0.0), -1, 1))

    composite = float(np.clip(
        tech_sig * w_tech +
        ml_sig   * w_ml +
        news_sig * w_news +
        fs       * w_fund +
        us_sig   * w_us +
        inst_sig * w_inst +
        margin_sig * w_marg,
        -1, 1
    ))
    return composite * 0.003

def simple_forecast(df, days=30, n_paths=150, lower_pct=15, upper_pct=85, drift_bias=0.0):
    close = df["Close"].values
    if len(close) < 30: raise RuntimeError("歷史資料不足")
    rets = np.diff(np.log(close))
    mu   = float(np.mean(rets)) + drift_bias
    sig  = float(np.std(rets))
    if not np.isfinite(sig) or sig <= 0: sig = 0.01
    last_price = float(close[-1])
    rng    = np.random.default_rng(seed=42)
    shocks = rng.normal(mu, sig, (n_paths, days))
    paths  = last_price * np.exp(np.cumsum(shocks, axis=1))
    last_date    = df.index[-1]
    future_dates = pd.bdate_range(start=last_date+pd.Timedelta(days=1), periods=days)
    return {
        "future_dates": future_dates,
        "median":  np.median(paths, axis=0),
        "lower":   np.percentile(paths, lower_pct, axis=0),
        "upper":   np.percentile(paths, upper_pct, axis=0),
        "paths":   paths,
        "drift_bias": drift_bias,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Backtest
# ═══════════════════════════════════════════════════════════════════════════
def backtest_directional(df, ind, sr, horizon=10, n_samples=100) -> dict:
    results = []; equity = [1.0]
    min_start = 20
    available = len(df) - horizon - min_start
    if available <= 0:
        return {"n_trades":0,"hit_rate":0.0,"hit_rate_text":"N/A",
                "sharpe":0.0,"max_drawdown":0.0,"avg_return_after_cost":0.0}
    if available < n_samples: n_samples = available
    start_idx = max(min_start, len(df)-n_samples-horizon)
    close  = df["Close"]; macd_h = ind["macd_hist"]; rsi14 = ind["rsi14"]
    for i in range(start_idx, len(df)-horizon):
        try:
            mh = float(macd_h.iloc[i]) if not pd.isna(macd_h.iloc[i]) else 0.0
            rs = float(rsi14.iloc[i])  if not pd.isna(rsi14.iloc[i]) else 50.0
        except Exception:
            continue
        if   mh > 0 and rs < 72: direction = 1
        elif mh < 0 and rs > 28: direction = -1
        else: continue
        entry = float(close.iloc[i]); exit_p = float(close.iloc[i+horizon])
        gross = (exit_p-entry)/entry*direction
        net   = gross - TX_ROUND_TRIP
        equity.append(equity[-1]*(1+net))
        results.append({"direction":direction,"net_ret":net,"gross_ret":gross,"hit":net>0})
    if not results:
        return {"n_trades":0,"hit_rate":0.0,"hit_rate_text":"N/A",
                "sharpe":0.0,"max_drawdown":0.0,"avg_return_after_cost":0.0}
    n = len(results); hits = sum(r["hit"] for r in results)
    rets = np.array([r["net_ret"] for r in results])
    ppyr = 252/max(horizon,1)
    sharpe = float(np.mean(rets)/max(np.std(rets),1e-9)*np.sqrt(ppyr))
    eq = np.array(equity); peak = np.maximum.accumulate(eq)
    dd = (eq-peak)/peak
    wins   = sum(r["gross_ret"] for r in results if r["gross_ret"]>0)
    losses = abs(sum(r["gross_ret"] for r in results if r["gross_ret"]<0))
    pf = wins/losses if losses>0 else float("inf")
    hr = hits/n
    return {
        "n_trades":n,"hit_rate":hr,"hit_rate_text":f"{hr*100:.1f}%",
        "avg_return_after_cost":float(np.mean(rets))*100,
        "sharpe":sharpe,"max_drawdown":float(dd.min())*100,
        "profit_factor":pf,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MLPredictor
# ═══════════════════════════════════════════════════════════════════════════
class MLPredictor:
    N_FEATURES = 27
    def __init__(self, horizon=10):
        self.horizon = horizon
        self.model_gb = self.model_xgb = self.scaler = None

    def _build_samples(self, df, ind, sr, mkt_ctx=None, inst=None):
        if mkt_ctx is None: mkt_ctx = {}
        if inst    is None: inst    = {}
        X, y = [], []
        close = df["Close"]; h = self.horizon
        for i in range(20, len(df)-h):
            vec = build_full_feature_row(df, ind, sr, i, mkt_ctx, inst)
            if vec is None:
                vec = build_feature_row(df, ind, sr, i)
                if vec is None: continue
                if len(vec) < self.N_FEATURES:
                    vec = np.concatenate([vec, np.zeros(self.N_FEATURES-len(vec))])
                vec = vec[:self.N_FEATURES]
                if not np.all(np.isfinite(vec)): continue
            X.append(vec)
            y.append(1 if float(close.iloc[i+h]) > float(close.iloc[i]) else 0)
        if not X: return np.empty((0,self.N_FEATURES)), np.empty(0)
        return np.array(X), np.array(y)

    def train(self, df, ind, sr, mkt_ctx=None, inst=None):
        if not SKLEARN_OK: return {"error":"sklearn 未安裝"}
        X, y = self._build_samples(df, ind, sr, mkt_ctx, inst)
        if len(X) < 10: return {"error":f"樣本不足({len(X)})"}
        if len(np.unique(y)) < 2: return {"error":"標籤單一"}
        tr = float((mkt_ctx or {}).get("_train_ratio", 0.8))
        split = int(len(X)*tr)
        Xtr, ytr = X[:split], y[:split]
        Xoo, yoo = X[split:], y[split:]
        self.scaler  = StandardScaler()
        Xtr_s = self.scaler.fit_transform(Xtr)
        Xoo_s = self.scaler.transform(Xoo) if len(Xoo) > 0 else Xtr_s
        n_est = min(120, max(40, len(Xtr)//8))
        np.random.seed(42)
        self.model_gb = GradientBoostingClassifier(
            n_estimators=n_est, max_depth=3, learning_rate=0.08,
            subsample=0.85, random_state=42)
        self.model_gb.fit(Xtr_s, ytr)
        self.model_xgb = None
        if XGB_OK:
            self.model_xgb = xgb.XGBClassifier(
                n_estimators=n_est, max_depth=3, learning_rate=0.08,
                subsample=0.85, colsample_bytree=0.85,
                eval_metric="logloss", random_state=42, verbosity=0)
            self.model_xgb.fit(Xtr_s, ytr)
        if len(Xoo) == 0:
            Xoo_s, yoo = Xtr_s, ytr
        prob_gb = self.model_gb.predict_proba(Xoo_s)[:,1]
        prob_ens = (0.5*prob_gb + 0.5*self.model_xgb.predict_proba(Xoo_s)[:,1]
                   if self.model_xgb else prob_gb)
        oos_pred = (prob_ens >= 0.5).astype(int)
        return {
            "algorithm": "GB+XGB(OOS)" if XGB_OK else "GB(OOS)",
            "n_samples": len(Xtr), "n_oos": len(yoo),
            "oos_accuracy": float(accuracy_score(yoo, oos_pred)),
            "accuracy":     float(accuracy_score(yoo, oos_pred)),
            "brier":        float(brier_score_loss(yoo, prob_ens)),
        }

    def predict_current(self, df, ind, sr, mkt_ctx=None, inst=None):
        if self.model_gb is None or self.scaler is None:
            return {"prob_up":0.5,"label":"未訓練"}
        vec = build_full_feature_row(df, ind, sr, len(df)-1, mkt_ctx or {}, inst or {})
        if vec is None or len(vec) != self.N_FEATURES:
            v = build_feature_row(df, ind, sr, len(df)-1)
            if v is None: return {"prob_up":0.5,"label":"特徵不足"}
            vec = np.concatenate([v, np.zeros(max(0,self.N_FEATURES-len(v)))])[:self.N_FEATURES]
        try:
            Xs   = self.scaler.transform(vec.reshape(1,-1))
            p_gb = float(self.model_gb.predict_proba(Xs)[0,1])
            prob = (0.5*p_gb + 0.5*float(self.model_xgb.predict_proba(Xs)[0,1])
                    if self.model_xgb else p_gb)
            label = "看漲" if prob>=0.6 else ("看跌" if prob<=0.4 else "中性")
            return {"prob_up":prob,"label":label}
        except Exception:
            return {"prob_up":0.5,"label":"預測失敗"}


# ═══════════════════════════════════════════════════════════════════════════
# Market / TW institutional / margin / news / strategy helpers
# ═══════════════════════════════════════════════════════════════════════════
try:
    import xml.etree.ElementTree as ET
    ET_OK = True
except ImportError:
    ET = None; ET_OK = False

def _yf_return(ticker: str, period: str = "10d") -> dict:
    out = {"ret_1": 0.0, "ret_5": 0.0, "last": None}
    if not YF_OK: return out
    try:
        df = _yf_history_with_retry(yf.Ticker(ticker), period)
        if df is None or len(df) < 2: return out
        c = df["Close"].astype(float)
        out["last"] = float(c.iloc[-1])
        out["ret_1"] = float(c.iloc[-1] / c.iloc[-2] - 1)
        if len(c) >= 6:
            out["ret_5"] = float(c.iloc[-1] / c.iloc[-6] - 1)
    except Exception:
        pass
    return out

def fetch_market_context() -> dict:
    """Fetch TAIEX, VIX, US indices, semiconductor ETF context."""
    ctx = {
        "taiex_ret_1":0.0, "taiex_ret_5":0.0, "taiex_vol":0.01,
        "vix":20.0, "usd_twd":31.5,
        "nasdaq_ret_1":0.0, "nasdaq_ret_5":0.0,
        "sp500_ret_1":0.0, "sp500_ret_5":0.0,
        "semis_ret_1":0.0, "semis_ret_5":0.0,
    }
    if not YF_OK: return ctx
    try:
        tw = _yf_history_with_retry(yf.Ticker("^TWII"), "20d")
        if tw is not None and len(tw) >= 6:
            c = tw["Close"].astype(float)
            ctx["taiex_ret_1"] = float(c.iloc[-1]/c.iloc[-2]-1)
            ctx["taiex_ret_5"] = float(c.iloc[-1]/c.iloc[-6]-1)
            ctx["taiex_vol"] = float(c.pct_change().dropna().tail(5).std())
    except Exception:
        pass
    try:
        vix = _yf_return("^VIX", "5d")
        if vix.get("last"): ctx["vix"] = float(vix["last"])
    except Exception:
        pass
    try:
        fx = _yf_return("TWD=X", "5d")
        if fx.get("last"): ctx["usd_twd"] = float(fx["last"])
    except Exception:
        pass
    ndq = _yf_return("^IXIC", "10d")
    spx = _yf_return("^GSPC", "10d")
    sem = _yf_return("SMH", "10d")
    ctx.update({
        "nasdaq_ret_1": ndq["ret_1"], "nasdaq_ret_5": ndq["ret_5"],
        "sp500_ret_1": spx["ret_1"], "sp500_ret_5": spx["ret_5"],
        "semis_ret_1": sem["ret_1"], "semis_ret_5": sem["ret_5"],
    })
    return ctx

def fetch_institutional_flow(raw_code: str) -> dict:
    """Fetch latest TWSE/TPEX institutional flow. Best-effort, no hard failure."""
    result = {
        "date": "", "foreign_net":0.0, "trust_net":0.0, "dealer_net":0.0,
        "total_net":0.0, "inst_score":0.0, "source":"N/A",
    }
    if not REQUESTS_OK: return result
    # TWSE T86; loop recent calendar days to survive holidays.
    for date in _tw_date_candidates(12):
        try:
            url = f"https://www.twse.com.tw/fund/T86?response=json&date={date}&selectType=ALLBUT0999"
            r = requests.get(url, timeout=7, headers={"User-Agent":"Mozilla/5.0"}, verify=False)
            if r.status_code != 200: continue
            rows = (r.json() or {}).get("data", [])
            for row in rows:
                if not row or str(row[0]).strip() != raw_code: continue
                foreign = _safe_float(row[4]) if len(row) > 4 else 0.0
                trust   = _safe_float(row[7]) if len(row) > 7 else 0.0
                dealer  = _safe_float(row[10]) if len(row) > 10 else 0.0
                total   = _safe_float(row[11]) if len(row) > 11 else foreign + trust + dealer
                # shares -> thousand shares-ish signal, saturated
                score = float(np.clip(total / 20000000.0, -1, 1))
                return {
                    "date": date, "foreign_net": foreign, "trust_net": trust,
                    "dealer_net": dealer, "total_net": total,
                    "inst_score": score, "inst_combo": score, "source":"TWSE",
                }
        except Exception:
            continue
    # TPEX fallback is structurally different and changes over time; keep safe zero if not found.
    return result

def fetch_margin_balance(raw_code: str) -> dict:
    """Fetch latest TWSE margin financing/short balance. Best-effort."""
    result = {
        "date":"", "margin_balance":0.0, "margin_change":0.0,
        "short_balance":0.0, "short_change":0.0, "margin_score":0.0,
        "source":"N/A",
    }
    if not REQUESTS_OK: return result
    for date in _tw_date_candidates(12):
        try:
            url = f"https://www.twse.com.tw/exchangeReport/MI_MARGN?response=json&date={date}&selectType=MS"
            r = requests.get(url, timeout=7, headers={"User-Agent":"Mozilla/5.0"}, verify=False)
            if r.status_code != 200: continue
            data = r.json() or {}
            rows = data.get("data", [])
            for row in rows:
                if not row or str(row[0]).strip() != raw_code: continue
                # Common TWSE row:
                # code, name, margin buy, margin sell, cash repay, prev bal, today bal,
                # margin limit, short sell, short buy, short repay, prev short, today short, short limit...
                margin_change = _safe_float(row[2]) - _safe_float(row[3]) - _safe_float(row[4]) if len(row) > 4 else 0.0
                margin_bal    = _safe_float(row[6]) if len(row) > 6 else 0.0
                short_change  = _safe_float(row[8]) - _safe_float(row[9]) - _safe_float(row[10]) if len(row) > 10 else 0.0
                short_bal     = _safe_float(row[12]) if len(row) > 12 else 0.0
                # Rising shorts can be bullish squeeze, but rising margin is overheated; use conservative blend.
                score = np.clip((-margin_change / 5000.0) + (short_change / 3000.0), -1, 1)
                return {
                    "date":date, "margin_balance":margin_bal, "margin_change":margin_change,
                    "short_balance":short_bal, "short_change":short_change,
                    "margin_score":float(score), "source":"TWSE",
                }
        except Exception:
            continue
    return result

def fetch_news(symbol: str, name: str, max_items: int = 8) -> list:
    """Fetch near-real-time Google News RSS and parse lightweight metadata."""
    if not REQUESTS_OK or not ET_OK: return []
    raw = symbol.split(".")[0]
    queries = []
    if _cjk(name): queries.append(f'"{name}" 股票')
    queries.append(f'{raw} 股票')
    queries.append(f'"{name}" 財報 OR 法人 OR 營收' if _cjk(name) else f'{raw} stock news')
    out, seen = [], set()
    for q in queries:
        try:
            url = "https://news.google.com/rss/search?q=" + requests.utils.quote(q) + "&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
            r = requests.get(url, timeout=7, verify=False, headers={"User-Agent":"Mozilla/5.0"})
            if r.status_code != 200: continue
            root = ET.fromstring(r.text)
            for item in root.findall(".//item"):
                title = re.sub(r"\s+", " ", item.findtext("title","")).strip()
                if not title or title in seen: continue
                link = item.findtext("link","")
                pub = item.findtext("pubDate","")
                src = item.find("source")
                src_t = src.text if src is not None else ""
                summary = re.sub(r"<[^>]+>", " ", item.findtext("description","") or "")
                summary = re.sub(r"\s+", " ", summary).strip()[:220]
                out.append({"title":title, "link":link, "date":pub, "source":src_t, "summary":summary})
                seen.add(title)
                if len(out) >= max_items: return out
        except Exception:
            continue
    return out[:max_items]

def news_sentiment(news: list) -> dict:
    pos_kw = ["上漲","看多","利多","突破","創高","成長","獲利","買進","強勢","升","優於","增","旺","加碼","上修"]
    neg_kw = ["下跌","看空","利空","跌破","虧損","賣出","弱勢","降","警示","風險","衰退","減碼","下修","疲弱"]
    pos = neg = 0
    for item in news:
        t = (item.get("title","") + " " + item.get("summary",""))
        pos += sum(1 for k in pos_kw if k in t)
        neg += sum(1 for k in neg_kw if k in t)
    total = max(pos + neg, 1)
    score = float(np.clip((pos - neg) / total, -1, 1))
    if score >= 0.25: label = "正面"
    elif score <= -0.25: label = "負面"
    else: label = "中性"
    return {"label":label, "score":score, "positive_ct":pos, "negative_ct":neg}

def macd_rsi_diagnosis(df: pd.DataFrame, ind: dict) -> dict:
    """Automatic MACD + RSI diagnosis text."""
    close = df["Close"]
    last = float(close.iloc[-1])
    rsi = float(ind["rsi14"].iloc[-1])
    mh = float(ind["macd_hist"].iloc[-1])
    mh_prev = float(ind["macd_hist"].iloc[-2]) if len(ind["macd_hist"]) > 1 else mh
    ml = float(ind["macd_line"].iloc[-1]); ms = float(ind["macd_signal"].iloc[-1])
    ma5 = float(ind["ma5"].iloc[-1]); ma20 = float(ind["ma20"].iloc[-1]); ma60 = float(ind["ma60"].iloc[-1])
    bb_up = float(ind["bb_up"].iloc[-1]); bb_dn = float(ind["bb_dn"].iloc[-1])
    bb_pos = (last - bb_dn) / max(bb_up - bb_dn, 1e-9)
    score = 0.0
    lines = []

    if mh_prev < 0 <= mh:
        lines.append("MACD Histogram 由負轉正，出現黃金交叉初期訊號，短線動能轉強。")
        score += 0.35
    elif mh_prev > 0 >= mh:
        lines.append("MACD Histogram 由正轉負，出現死亡交叉初期訊號，短線動能轉弱。")
        score -= 0.35
    elif mh > 0 and ml > ms:
        lines.append(f"MACD 維持多頭排列，Histogram {mh:+.3f}，上升動能仍在。")
        score += 0.22
    elif mh < 0 and ml < ms:
        lines.append(f"MACD 維持空頭排列，Histogram {mh:+.3f}，下跌動能尚未解除。")
        score -= 0.22
    else:
        lines.append("MACD 訊號混合，趨勢仍待確認。")

    if rsi >= 75:
        lines.append(f"RSI {rsi:.1f} 深度超買，短線容易震盪或拉回。")
        score -= 0.20
    elif rsi >= 60:
        lines.append(f"RSI {rsi:.1f} 偏強，買盤動能仍優於賣壓。")
        score += 0.18
    elif rsi <= 25:
        lines.append(f"RSI {rsi:.1f} 深度超賣，可能有技術反彈，但需量價確認。")
        score += 0.10
    elif rsi <= 40:
        lines.append(f"RSI {rsi:.1f} 偏弱，短線反彈力道不足。")
        score -= 0.18
    else:
        lines.append(f"RSI {rsi:.1f} 位於中性區間，方向仍需等待量價突破。")

    if ma5 > ma20 > ma60:
        lines.append(f"均線呈多頭排列：MA5 {ma5:.2f} > MA20 {ma20:.2f} > MA60 {ma60:.2f}。")
        score += 0.22
    elif ma5 < ma20 < ma60:
        lines.append(f"均線呈空頭排列：MA5 {ma5:.2f} < MA20 {ma20:.2f} < MA60 {ma60:.2f}。")
        score -= 0.22
    elif ma5 > ma20:
        lines.append("短均線站上中期均線，短線轉強但中線仍需確認。")
        score += 0.10
    else:
        lines.append("短均線低於中期均線，短線偏弱。")
        score -= 0.10

    if bb_pos > 0.90:
        lines.append("價格貼近布林上軌，屬強勢區，但追價需注意風險。")
        score += 0.05
    elif bb_pos < 0.10:
        lines.append("價格貼近布林下軌，短線可能醞釀反彈。")
        score += 0.03

    score = float(np.clip(score, -1, 1))
    if score >= 0.35: label = "偏多"
    elif score <= -0.35: label = "偏空"
    else: label = "中性"
    return {"label": label, "score": score, "text": "\n".join(lines)}

def simulate_macd_cross_strategy(df: pd.DataFrame, ind: dict, hold_days: int = 10,
                                 cross_type: str = "golden") -> dict:
    """
    Backtest: buy on MACD golden cross and hold N days.
    If cross_type='death', simulate short direction for comparison.
    """
    hold_days = int(max(1, min(120, hold_days)))
    hist = ind["macd_hist"]
    close = df["Close"]
    trades = []
    for i in range(1, len(df) - hold_days):
        prev_h = float(hist.iloc[i-1]); cur_h = float(hist.iloc[i])
        is_cross = (prev_h < 0 <= cur_h) if cross_type == "golden" else (prev_h > 0 >= cur_h)
        if not is_cross: continue
        entry = float(close.iloc[i])
        exit_p = float(close.iloc[i + hold_days])
        raw_ret = (exit_p - entry) / entry
        if cross_type != "golden": raw_ret *= -1
        net_ret = raw_ret - TX_ROUND_TRIP
        trades.append({
            "date": df.index[i].strftime("%Y-%m-%d"),
            "entry": entry, "exit": exit_p,
            "return_pct": net_ret * 100,
            "win": net_ret > 0,
        })
    if not trades:
        return {"n_trades":0, "win_rate":0.0, "win_rate_text":"N/A", "avg_return":0.0, "trades":[]}
    rets = np.array([t["return_pct"] for t in trades])
    wins = sum(t["win"] for t in trades)
    return {
        "n_trades": len(trades),
        "win_rate": wins / len(trades),
        "win_rate_text": f"{wins / len(trades) * 100:.1f}%",
        "avg_return": float(np.mean(rets)),
        "median_return": float(np.median(rets)),
        "best_return": float(np.max(rets)),
        "worst_return": float(np.min(rets)),
        "hold_days": hold_days,
        "cross_type": cross_type,
        "trades": trades[-30:],
    }

def get_top_volume_stocks(limit: int = 100) -> list:
    """Return top Taiwan volume stocks from TWSE + TPEX, fallback to common list."""
    fallback = ["2330","2317","2454","2303","2881","2882","2891","2886","0050","00631L",
                "2603","2615","2382","3231","2379","3037","2308","3661","3711","3443"]
    if not REQUESTS_OK: return fallback[:limit]
    parsed = []

    def _parse_rows(rows, source=""):
        for it in rows:
            if not isinstance(it, dict): continue
            code = str(
                it.get("Code") or it.get("SecuritiesCode") or it.get("股票代號") or
                it.get("SecuritiesCompanyCode") or it.get("stock_id") or ""
            ).strip()
            name = str(
                it.get("Name") or it.get("SecuritiesName") or it.get("股票名稱") or
                it.get("CompanyName") or it.get("公司名稱") or ""
            ).strip()
            vol = _safe_float(
                it.get("TradeVolume") or it.get("成交股數") or it.get("TradingShares") or
                it.get("成交量") or it.get("Volume") or 0
            )
            if code and code[0].isdigit() and vol > 0:
                if name: TW_NAME_CACHE.setdefault(code, name)
                parsed.append((code, vol, source))

    endpoints = [
        ("TWSE", "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"),
        ("TPEX", "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_perday_quotes"),
    ]
    for src, url in endpoints:
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent":"Mozilla/5.0"}, verify=False)
            if r.status_code != 200: continue
            rows = r.json()
            if isinstance(rows, list):
                _parse_rows(rows, src)
        except Exception:
            continue
    parsed.sort(key=lambda x: x[1], reverse=True)
    codes, seen = [], set()
    for c, _, _ in parsed:
        if c not in seen:
            codes.append(c); seen.add(c)
        if len(codes) >= limit: break
    return codes or fallback[:limit]

def recommendation_score(r: dict) -> float:
    df = r["df"]; ind = r["indicators"]; fc = r["forecast"]
    last = float(df["Close"].iloc[-1])
    med = float(fc["median"][-1])
    fc_pct = (med - last) / max(last, 1e-9)
    ml_p = r.get("ml_predict", {}).get("prob_up", 0.5)
    diag = r.get("tech_diagnosis", {}).get("score", 0.0)
    inst = r.get("institutional", {}).get("inst_score", 0.0)
    margin = r.get("margin", {}).get("margin_score", 0.0)
    news = r.get("news_sentiment", {}).get("score", 0.0)
    us = np.clip(r.get("mkt_ctx", {}).get("semis_ret_1", 0.0) / 0.03, -1, 1)
    return float(
        fc_pct * 3.0 +
        (ml_p - 0.5) * 1.8 +
        diag * 0.9 +
        inst * 0.8 +
        margin * 0.35 +
        news * 0.45 +
        us * 0.25
    )

def recommend_top_volume_stocks(volume_limit: int = 100, top_n: int = 5,
                                lookback_years: int = 2, forecast_days: int = 20,
                                weights: dict = None, progress_callback=None) -> list:
    """
    Analyze top-volume Taiwan stocks and return recommended candidates.
    This intentionally scans ALL top `volume_limit` symbols, not just watchlist.
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    codes = get_top_volume_stocks(volume_limit)
    rows = []
    total = len(codes)
    def _one(code):
        try:
            r = run_analysis(code, lookback_years=lookback_years,
                             forecast_days=forecast_days, weights=weights,
                             train_ratio=0.8, progress_callback=None,
                             use_cache=True, light_news=True)
            sc = recommendation_score(r)
            df = r["df"]; last = float(df["Close"].iloc[-1])
            med = float(r["forecast"]["median"][-1])
            reason = []
            diag = r.get("tech_diagnosis", {})
            if diag.get("label"): reason.append(f"技術診斷{diag['label']}")
            if r.get("ml_predict", {}).get("prob_up", 0.5) >= 0.58:
                reason.append(f"ML上漲機率{r['ml_predict']['prob_up']*100:.0f}%")
            if r.get("institutional", {}).get("total_net", 0) > 0:
                reason.append("三大法人買超")
            if r.get("news_sentiment", {}).get("label") == "正面":
                reason.append("新聞情緒正面")
            if med > last:
                reason.append(f"預測{forecast_days}日約{(med-last)/last*100:+.1f}%")
            return {
                "code": r["symbol"].split(".")[0], "name": r["name"],
                "score": sc, "last": last,
                "forecast_pct": (med-last)/last*100,
                "ml_prob": r.get("ml_predict", {}).get("prob_up", 0.5)*100,
                "inst_total": r.get("institutional", {}).get("total_net", 0),
                "diagnosis": diag.get("label", "中性"),
                "reason": "、".join(reason) if reason else "綜合分數較高",
            }
        except Exception as e:
            return None

    # Conservative workers to reduce Yahoo rate-limit. This still checks all candidates.
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_one, c): c for c in codes}
        done = 0
        for fut in as_completed(futs):
            done += 1
            if progress_callback:
                progress_callback(done, total, f"掃描前 {volume_limit} 大成交量：{futs[fut]}")
            item = fut.result()
            if item is not None:
                rows.append(item)
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows[:top_n]


# ═══════════════════════════════════════════════════════════════════════════
# Main analysis pipeline (single function, no QThread needed)
# ═══════════════════════════════════════════════════════════════════════════
def run_analysis(symbol: str, lookback_years: int = 3,
                 forecast_days: int = 30, weights: dict = None,
                 train_ratio: float = 0.8,
                 progress_callback=None,
                 use_cache: bool = True,
                 light_news: bool = False) -> dict:
    """
    Full Streamlit analysis pipeline.
    Returns dict with all results needed by the UI.
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    cache_key = json.dumps({
        "symbol": symbol.strip().upper(),
        "lookback_years": lookback_years,
        "forecast_days": forecast_days,
        "weights": weights,
        "train_ratio": train_ratio,
        "light_news": light_news,
    }, sort_keys=True, ensure_ascii=False)
    now = time.time()
    if use_cache and cache_key in _ANALYSIS_CACHE:
        ts, cached = _ANALYSIS_CACHE[cache_key]
        if now - ts < _ANALYSIS_CACHE_TTL:
            return cached

    def _prog(s, t, msg):
        if progress_callback:
            progress_callback(s, t, msg)

    TOTAL = 8
    try:
        # 1. Fetch
        _prog(1, TOTAL, f"抓取 {symbol} 資料…")
        sym, name, df = resolve_and_fetch(symbol, lookback_years)

        # 2. Indicators
        _prog(2, TOTAL, "計算技術指標與支撐壓力…")
        ind = compute_indicators(df)
        sr  = compute_support_resistance(df)
        diag = macd_rsi_diagnosis(df, ind)

        raw_code = sym.split(".")[0]

        # 3. Market + TW-specific + news/fundamental
        _prog(3, TOTAL, "抓取美股/台股法人/融資融券/新聞…")
        def _do_market(): return fetch_market_context()
        def _do_inst(): return fetch_institutional_flow(raw_code)
        def _do_margin(): return fetch_margin_balance(raw_code)
        def _do_news():
            items = fetch_news(sym, name, max_items=4 if light_news else 8)
            return items, news_sentiment(items)
        def _do_fund():
            fund = {}
            if not YF_OK: return fund
            try:
                info = yf.Ticker(sym).info or {}
                for k, lbl in [("trailingPE","pe_ratio"),("trailingEps","eps"),
                               ("returnOnEquity","roe"),("dividendYield","div_yield"),
                               ("marketCap","market_cap"),("revenueGrowth","rev_growth"),
                               ("earningsGrowth","earn_growth")]:
                    v = info.get(k)
                    if v is not None:
                        try: fund[lbl] = float(v)
                        except Exception: pass
            except Exception:
                pass
            return fund

        with ThreadPoolExecutor(max_workers=5) as ex:
            fm = ex.submit(_do_market)
            fi = ex.submit(_do_inst)
            fmg = ex.submit(_do_margin)
            fn = ex.submit(_do_news)
            ff = ex.submit(_do_fund)
            mkt_ctx = fm.result()
            inst = fi.result()
            margin = fmg.result()
            news, ns = fn.result()
            fund = ff.result()

        mkt_ctx["_train_ratio"] = train_ratio
        mkt_ctx["_margin"] = margin

        # 4. ML
        _prog(4, TOTAL, "訓練 ML 模型（含美股/法人/融資融券特徵）…")
        horizon = min(forecast_days, 10)
        ml = MLPredictor(horizon=horizon)
        ml_stats = ml.train(df, ind, sr, mkt_ctx, inst)
        ml_pred  = ml.predict_current(df, ind, sr, mkt_ctx, inst) if "error" not in ml_stats else {}

        # 5. Forecast
        _prog(5, TOTAL, "Monte-Carlo 趨勢預測（整合權重）…")
        drift = compute_drift_bias(ind, ml_pred, {}, ns, fund, weights, mkt_ctx, inst, margin)
        fc = simple_forecast(df, days=forecast_days, n_paths=180, drift_bias=drift)

        # 6. Backtests
        _prog(6, TOTAL, "回測與 MACD 黃金交叉策略…")
        bt = backtest_directional(df, ind, sr, horizon=horizon)
        macd_bt = simulate_macd_cross_strategy(df, ind, hold_days=10, cross_type="golden")

        # 7. Reason
        _prog(7, TOTAL, "產生自動診斷文字…")
        # UI also has its own rich rendering; core provides stable text.
        reason_lines = [
            diag["text"],
            f"三大法人：合計買賣超 {inst.get('total_net',0)/1000:,.0f} 張，分數 {inst.get('inst_score',0):+.2f}。",
            f"融資融券：融資變化 {margin.get('margin_change',0):,.0f} 張、融券變化 {margin.get('short_change',0):,.0f} 張，分數 {margin.get('margin_score',0):+.2f}。",
            f"美股背景：NASDAQ {mkt_ctx.get('nasdaq_ret_1',0)*100:+.2f}%、S&P500 {mkt_ctx.get('sp500_ret_1',0)*100:+.2f}%、半導體ETF {mkt_ctx.get('semis_ret_1',0)*100:+.2f}%。",
            f"新聞情緒：{ns.get('label','中性')}（正面 {ns.get('positive_ct',0)} / 負面 {ns.get('negative_ct',0)}）。",
        ]

        # 8. Assemble
        _prog(8, TOTAL, "整理分析結果…")
        result = {
            "symbol": sym, "name": name, "df": df,
            "indicators": ind, "sr": sr,
            "forecast": fc, "forecast_days": forecast_days,
            "ml_stats": ml_stats, "ml_predict": ml_pred,
            "nn_predict": {}, "nn_stats": {"note":"web core currently uses GB/XGB; LSTM not enabled in this backend"},
            "backtest": bt, "macd_cross_backtest": macd_bt,
            "news": news, "news_sentiment": ns,
            "fundamental": fund, "drift_bias": drift,
            "weights": weights, "mkt_ctx": mkt_ctx,
            "institutional": inst, "margin": margin,
            "tech_diagnosis": diag,
            "reason_text": "\n".join(reason_lines),
        }
        if use_cache:
            _ANALYSIS_CACHE[cache_key] = (time.time(), result)
        return result
    except StockAnalysisError:
        raise
    except Exception as e:
        msg = friendly_error_message(e)
        if "找不到" in msg:
            raise StockNotFoundError(msg) from e
        if "逾時" in msg:
            raise DataFetchTimeout(msg) from e
        raise StockAnalysisError(msg) from e
