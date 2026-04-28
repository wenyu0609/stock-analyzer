# core.py — Backend engine extracted from Stock Analyzer Pro v11
# All analysis logic: indicators, ML, forecast, backtest, names
# This file has ZERO PySide6 / GUI dependencies — pure Python data layer

import os, sys, json, time, warnings
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
    hi = df["High"].tail(lookback); lo = df["Low"].tail(lookback)
    q25h = float(hi.quantile(0.25)); q75h = float(hi.quantile(0.75))
    q25l = float(lo.quantile(0.25)); q75l = float(lo.quantile(0.75))
    return dict(support_lo=q25l, support_hi=q25h,
                resistance_lo=q75l, resistance_hi=q75h)


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
    if mkt_ctx is None: mkt_ctx = {}
    if inst    is None: inst    = {}
    vec = build_feature_row(df, ind, sr, idx)
    if vec is None or len(vec) != 27: return None
    vec[21] = float(np.clip(mkt_ctx.get("taiex_ret_1",0.0), -0.1, 0.1))
    vec[22] = float(np.clip(mkt_ctx.get("taiex_ret_5",0.0), -0.2, 0.2))
    vec[23] = float(np.clip(mkt_ctx.get("taiex_vol",0.01), 0, 0.05))
    vec[24] = float(mkt_ctx.get("vix",20.0)) / 40.0
    vec[25] = float(np.clip(inst.get("inst_combo",0.0), -1, 1))
    vec[26] = float(np.clip(inst.get("foreign_net",0.0), -50, 50)) / 50.0
    if not np.all(np.isfinite(vec)): return None
    return vec


# ═══════════════════════════════════════════════════════════════════════════
# Forecast
# ═══════════════════════════════════════════════════════════════════════════
def compute_drift_bias(ind, ml_pred, nn_pred, ns, fund, weights) -> float:
    w_tech = weights.get("technical", 40) / 100.0
    w_ml   = weights.get("ml", 35)        / 100.0
    w_news = weights.get("news", 15)      / 100.0
    w_fund = weights.get("fundamental",10)/ 100.0
    tech_sig = 0.0
    try:
        mh  = float(ind["macd_hist"].iloc[-1])
        rsi = float(ind["rsi14"].iloc[-1])
        bbu = float(ind["bb_up"].iloc[-1]); bbd = float(ind["bb_dn"].iloc[-1])
        cl  = float(ind["ma5"].iloc[-1])
        mstd = float(ind["macd_hist"].std()) if len(ind["macd_hist"])>5 else 1.0
        tech_sig += float(np.clip(mh/max(mstd,1e-9),-1,1))*0.5
        tech_sig += float(np.clip((50-rsi)/50,-1,1))*(-0.3)
        bpos = (cl-bbd)/max(bbu-bbd,1e-9)
        tech_sig += float(np.clip((0.5-bpos)*2,-1,1))*0.2
    except Exception:
        pass
    gb_p   = (ml_pred or {}).get("prob_up", 0.5)
    nn_p   = (nn_pred or {}).get("prob_up", gb_p)
    ml_sig = (gb_p*0.45 + nn_p*0.55 - 0.5)*2
    lbl    = (ns or {}).get("label","中性")
    ns_sig = 0.6 if lbl=="正面" else (-0.6 if lbl=="負面" else 0.0)
    fs = 0.0
    if fund:
        if fund.get("pe_ratio") and 0<fund["pe_ratio"]<15: fs += 0.5
        if fund.get("pe_ratio") and fund["pe_ratio"]>40:   fs -= 0.5
        if fund.get("roe")      and fund["roe"]>0.15:      fs += 0.5
    composite = float(np.clip(
        tech_sig*w_tech + ml_sig*w_ml + ns_sig*w_news + fs*w_fund, -1, 1))
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
# News Fetcher
# ═══════════════════════════════════════════════════════════════════════════
try:
    import xml.etree.ElementTree as ET
    ET_OK = True
except ImportError:
    ET = None; ET_OK = False

def fetch_news(symbol: str, name: str, max_items: int = 5) -> list:
    if not REQUESTS_OK or not ET_OK: return []
    queries = [name[:4] if _cjk(name) else symbol.split(".")[0]]
    results = []
    for q in queries:
        try:
            url = (f"https://news.google.com/rss/search?q={q}+股票"
                   f"&hl=zh-TW&gl=TW&ceid=TW:zh-Hant")
            r = requests.get(url, timeout=5, verify=False,
                             headers={"User-Agent":"Mozilla/5.0"})
            if r.status_code != 200: continue
            root = ET.fromstring(r.text)
            for item in root.findall(".//item")[:max_items]:
                title = item.findtext("title","")
                link  = item.findtext("link","")
                pub   = item.findtext("pubDate","")
                src   = item.find("source")
                src_t = src.text if src is not None else ""
                if title:
                    results.append({"title":title,"link":link,
                                    "date":pub,"source":src_t})
        except Exception:
            continue
    return results[:max_items]

def news_sentiment(news: list) -> dict:
    pos_kw = ["上漲","看多","利多","突破","創高","成長","獲利","買進","強勢","升"]
    neg_kw = ["下跌","看空","利空","跌破","虧損","賣出","弱勢","降","警示","風險"]
    pos = neg = 0
    for item in news:
        t = item.get("title","")
        pos += sum(1 for k in pos_kw if k in t)
        neg += sum(1 for k in neg_kw if k in t)
    if pos > neg*1.5:   label = "正面"
    elif neg > pos*1.5: label = "負面"
    else:               label = "中性"
    return {"label":label,"positive_ct":pos,"negative_ct":neg}


# ═══════════════════════════════════════════════════════════════════════════
# Main analysis pipeline (single function, no QThread needed)
# ═══════════════════════════════════════════════════════════════════════════
def run_analysis(symbol: str, lookback_years: int = 3,
                 forecast_days: int = 30, weights: dict = None,
                 train_ratio: float = 0.8,
                 progress_callback=None) -> dict:
    """
    Full analysis pipeline. progress_callback(step, total, msg) optional.
    Returns dict with all results needed by the Streamlit UI.
    """
    if weights is None:
        weights = {"technical":40,"ml":35,"news":15,"fundamental":10}

    def _prog(s, t, msg):
        if progress_callback: progress_callback(s, t, msg)

    TOTAL = 7
    # 1. Fetch
    _prog(1, TOTAL, f"抓取 {symbol} 資料…")
    sym, name, df = resolve_and_fetch(symbol, lookback_years)

    # 2. Indicators
    _prog(2, TOTAL, "計算技術指標…")
    ind = compute_indicators(df)
    sr  = compute_support_resistance(df)

    # 3. Market context
    _prog(3, TOTAL, "抓取市場背景…")
    mkt_ctx = {"taiex_ret_1":0.0,"taiex_ret_5":0.0,"taiex_vol":0.01,
               "vix":20.0,"usd_twd":31.5,"_train_ratio":train_ratio}
    inst = {}
    if REQUESTS_OK and YF_OK:
        try:
            vix_df = _yf_history_with_retry(yf.Ticker("^VIX"), "5d")
            if vix_df is not None and not vix_df.empty:
                mkt_ctx["vix"] = float(vix_df["Close"].iloc[-1])
        except Exception: pass

    # 4. ML
    _prog(4, TOTAL, "訓練 ML 模型…")
    horizon = min(forecast_days, 10)
    ml = MLPredictor(horizon=horizon)
    ml_stats = ml.train(df, ind, sr, mkt_ctx, inst)
    ml_pred  = ml.predict_current(df, ind, sr, mkt_ctx, inst) if "error" not in ml_stats else {}

    # 5. Forecast (uses ML result for drift)
    _prog(5, TOTAL, "Monte-Carlo 趨勢預測…")
    drift = compute_drift_bias(ind, ml_pred, {}, {}, {}, weights)
    fc = simple_forecast(df, days=forecast_days, n_paths=150, drift_bias=drift)

    # 6+7. Backtest + News + Fundamental in parallel
    _prog(6, TOTAL, "回測 / 新聞 / 基本面（平行抓取）…")

    def _do_backtest(): return backtest_directional(df, ind, sr, horizon=horizon)
    def _do_news():
        items = fetch_news(sym, name)
        return items, news_sentiment(items)
    def _do_fund():
        fund = {}
        if not YF_OK: return fund
        try:
            info = yf.Ticker(sym).info or {}
            for k, lbl in [("trailingPE","pe_ratio"),("trailingEps","eps"),
                           ("returnOnEquity","roe"),("dividendYield","div_yield"),
                           ("marketCap","market_cap")]:
                v = info.get(k)
                if v is not None:
                    try: fund[lbl] = float(v)
                    except Exception: pass
        except Exception: pass
        return fund

    bt = news = ns = fund = None
    with ThreadPoolExecutor(max_workers=3) as ex:
        fbt   = ex.submit(_do_backtest)
        fnews = ex.submit(_do_news)
        ffund = ex.submit(_do_fund)
        bt   = fbt.result()
        news, ns = fnews.result()
        fund = ffund.result()

    return {
        "symbol": sym, "name": name, "df": df,
        "indicators": ind, "sr": sr,
        "forecast": fc, "forecast_days": forecast_days,
        "ml_stats": ml_stats, "ml_predict": ml_pred,
        "backtest": bt, "news": news, "news_sentiment": ns,
        "fundamental": fund, "drift_bias": drift,
        "weights": weights,
    }
