from __future__ import annotations

import html
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, time as dtime
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor

try:
    requests.packages.urllib3.disable_warnings()
except Exception:
    pass


TW_TZ = "Asia/Taipei"
MARKET_OPEN = dtime(9, 0)
MARKET_CLOSE = dtime(13, 30)

VERIFIED_NAMES = {
    "2330": "台積電", "2454": "聯發科", "2317": "鴻海", "2303": "聯電",
    "2308": "台達電", "2382": "廣達", "3711": "日月光投控", "3034": "聯詠",
    "8069": "元太", "6488": "環球晶", "3227": "原相", "4743": "合一",
    "0050": "元大台灣50", "0056": "元大高股息", "006208": "富邦台50",
    "006201": "元大富櫃50", "00631L": "元大台灣50正2",
    "00632R": "元大台灣50反1", "00980A": "中信高優息成長ETF",
    "00981A": "主動統一台股增長", "00982A": "主動群益台灣強棒",
}

FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60",
    "volatility_5", "volatility_20", "volatility_60",
    "vol_ratio_5", "vol_ratio_20", "range_pct", "body_pct",
    "close_location", "gap_pct", "rsi14", "macd_hist_pct",
    "bb_pos", "bb_width", "atr_pct", "ma5_dist", "ma20_dist",
    "ma60_dist", "ma20_slope", "ma60_slope", "vwap20_dist",
    "dist_hi20", "dist_lo20", "trend_regime", "vol_regime",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
]


@dataclass
class AnalysisResult:
    input_symbol: str
    symbol: str
    raw_code: str
    name: str
    source: str
    history: pd.DataFrame
    indicators: pd.DataFrame
    forecast: pd.DataFrame
    prediction: Dict[str, float]
    backtest: Dict[str, float]
    backtest_detail: pd.DataFrame


@dataclass
class IntradayResult:
    symbol: str
    name: str
    source: str
    data: pd.DataFrame
    prediction: pd.DataFrame
    stats: Dict[str, float]


def _http_get(url: str, *, timeout: int = 8, headers: Optional[dict] = None):
    h = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/html,text/plain,*/*",
    }
    if headers:
        h.update(headers)
    return requests.get(url, timeout=timeout, headers=h, verify=False)


def tw_code_key(value: str) -> str:
    s = str(value or "").strip().upper().replace("\u3000", " ")
    s = re.sub(r"\.(TW|TWO)$", "", s)
    m = re.search(r"(?<!\d)(\d{4,6}[A-Z]?)(?!\d)", s)
    return m.group(1) if m else re.sub(r"[^0-9A-Z]", "", s)


def is_tw_code(symbol: str) -> bool:
    c = tw_code_key(symbol)
    return bool(re.fullmatch(r"\d{4,6}[A-Z]?", c))


def candidate_symbols(symbol: str) -> List[str]:
    s = str(symbol or "").strip().upper()
    if not s:
        return []
    if "." in s:
        return [s]
    if is_tw_code(s):
        return [f"{tw_code_key(s)}.TW", f"{tw_code_key(s)}.TWO"]
    return [s]


def _has_cjk(value: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(value or ""))


def clean_tw_name(value: str) -> str:
    s = html.unescape(str(value or "")).strip()
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" \t\r\n　-–—:：")
    s = re.sub(r"\(\s*\d{4,6}[A-Z]?\.(TW|TWO)\s*\).*", "", s, flags=re.I).strip()
    s = re.sub(r"\b\d{4,6}[A-Z]?\b", "", s).strip(" ()（）-–—:：")
    for suffix in ("股份有限公司", "有限公司", "公司"):
        if s.endswith(suffix) and len(s) - len(suffix) >= 2:
            s = s[:-len(suffix)].strip()
            break
    if "�" in s or not _has_cjk(s):
        return ""
    return s


def _name_ok(name: str, code: str = "") -> bool:
    s = clean_tw_name(name)
    if not s:
        return False
    if re.fullmatch(r"[0-9A-Z._-]+", s.upper()):
        return False
    if s.upper() == str(code or "").upper():
        return False
    return True


def _extract_code_name(item: dict) -> Tuple[str, str]:
    code_keys = (
        "Code", "code", "StockNo", "stockNo", "SecuritiesCode",
        "SecuritiesCompanyCode", "SecurityCode", "股票代號", "證券代號",
        "有價證券代號", "公司代號", "代號",
    )
    name_keys = (
        "Name", "name", "StockName", "stockName", "CompanyAbbreviation",
        "CompanyShortName", "SecuritiesName", "CompanyName", "SecurityName",
        "股票名稱", "證券名稱", "有價證券名稱", "公司簡稱", "公司名稱", "名稱",
    )
    code = ""
    name = ""
    for key in code_keys:
        if key in item:
            code = tw_code_key(item.get(key))
            if code:
                break
    for key in name_keys:
        if key in item:
            candidate = clean_tw_name(item.get(key))
            if _name_ok(candidate, code):
                name = candidate
                break
    return code, name


def _fetch_json_list(url: str) -> list:
    try:
        r = _http_get(url, timeout=10)
        if r.status_code != 200:
            return []
        ctype = str(r.headers.get("content-type", "")).lower()
        sample = (r.text or "").lstrip()[:1]
        if "json" not in ctype and sample not in ("[", "{"):
            return []
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception:
        return []


@lru_cache(maxsize=1)
def official_tw_name_map() -> Dict[str, str]:
    urls = [
        "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
        "https://openapi.twse.com.tw/v1/opendata/t187ap03_L",
        "https://openapi.twse.com.tw/v1/opendata/t187ap05_L",
        "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes",
        "https://www.tpex.org.tw/openapi/v1/tpex_daily_market_value",
        "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O",
        "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap05_O",
        "https://www.tpex.org.tw/openapi/v1/tpex_esb_latest_statistics",
        "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_R",
    ]
    out = dict(VERIFIED_NAMES)
    for url in urls:
        for item in _fetch_json_list(url):
            if not isinstance(item, dict):
                continue
            code, name = _extract_code_name(item)
            if code and _name_ok(name, code):
                out.setdefault(code, name)
    return out


@lru_cache(maxsize=512)
def yahoo_tw_name(code: str) -> str:
    raw = tw_code_key(code)
    if not raw:
        return ""
    for suffix in ("TW", "TWO"):
        try:
            url = f"https://tw.stock.yahoo.com/quote/{raw}.{suffix}"
            r = _http_get(url, timeout=7, headers={"Accept": "text/html,*/*"})
            if r.status_code != 200:
                continue
            m = re.search(r"<title[^>]*>(.*?)</title>", r.text or "", flags=re.I | re.S)
            title = html.unescape(m.group(1)) if m else (r.text or "")[:300]
            title = re.sub(r"<[^>]+>", " ", title)
            pats = [
                rf"^\s*(.*?)\s*\(\s*{re.escape(raw)}\.(?:TW|TWO)\s*\)",
                r"^\s*([\u4e00-\u9fff][\w\u4e00-\u9fff＋+\-－*（）()·‧& ]{1,30})\s*(?:走勢圖|即時行情)",
            ]
            for pat in pats:
                mm = re.search(pat, title, flags=re.I)
                if mm:
                    name = clean_tw_name(mm.group(1))
                    if _name_ok(name, raw):
                        return name
        except Exception:
            continue
    return ""


def fetch_display_name(symbol: str, ticker=None) -> str:
    raw = tw_code_key(symbol)
    if is_tw_code(symbol):
        if raw in VERIFIED_NAMES:
            return VERIFIED_NAMES[raw]
        name = yahoo_tw_name(raw)
        if _name_ok(name, raw):
            return name
        return official_tw_name_map().get(raw, raw)
    if ticker is not None:
        try:
            info = ticker.info or {}
            for key in ("shortName", "longName", "displayName"):
                value = str(info.get(key, "") or "").strip()
                if value:
                    return value
        except Exception:
            pass
    return str(symbol or "").upper()


def _normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    try:
        if out.index.tz is not None:
            out.index = out.index.tz_convert(TW_TZ).tz_localize(None)
    except Exception:
        try:
            out.index = out.index.tz_localize(None)
        except Exception:
            pass
    out = out.rename(columns={c: str(c).capitalize() for c in out.columns})
    keep = ["Open", "High", "Low", "Close", "Volume"]
    for col in keep:
        if col not in out.columns:
            out[col] = 0.0 if col == "Volume" else np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Close"])
    out = out[out["Close"] > 0].sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out["Open"] = out["Open"].where(out["Open"] > 0, out["Close"])
    out["High"] = out["High"].where(out["High"] > 0, out[["Open", "Close"]].max(axis=1))
    out["Low"] = out["Low"].where(out["Low"] > 0, out[["Open", "Close"]].min(axis=1))
    out["Volume"] = out["Volume"].fillna(0).clip(lower=0)
    return out[keep]


def fetch_fugle_history(raw_code: str, years: int, api_key: str) -> pd.DataFrame:
    if not api_key or not is_tw_code(raw_code):
        return pd.DataFrame()
    end = pd.Timestamp.now(tz=TW_TZ).date()
    start = (pd.Timestamp(end) - pd.DateOffset(years=max(1, int(years))) - pd.Timedelta(days=10)).date()
    url = f"https://api.fugle.tw/marketdata/v1.0/stock/historical/candles/{tw_code_key(raw_code)}"
    params = {
        "from": str(start), "to": str(end), "timeframe": "D",
        "adjusted": "true", "fields": "open,high,low,close,volume", "sort": "asc",
    }
    headers = {"X-API-KEY": api_key, "User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        if r.status_code != 200:
            return pd.DataFrame()
        payload = r.json()
        rows = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            return pd.DataFrame()
        records = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            records.append({
                "Date": pd.to_datetime(item.get("date"), errors="coerce"),
                "Open": item.get("open"),
                "High": item.get("high"),
                "Low": item.get("low"),
                "Close": item.get("close"),
                "Volume": item.get("volume", 0),
            })
        df = pd.DataFrame(records).dropna(subset=["Date"])
        if df.empty:
            return pd.DataFrame()
        return _normalize_history(df.set_index("Date"))
    except Exception:
        return pd.DataFrame()


def fetch_history(symbol: str, years: int = 3, min_rows: int = 80, api_key: str = "") -> Tuple[str, str, pd.DataFrame, str]:
    raw = tw_code_key(symbol)
    if api_key and is_tw_code(symbol):
        df = fetch_fugle_history(raw, years, api_key)
        if len(df) >= min_rows:
            resolved = f"{raw}.TW"
            return resolved, fetch_display_name(raw), df, "Fugle MarketData"

    errors = []
    for candidate in candidate_symbols(symbol):
        try:
            ticker = yf.Ticker(candidate)
            df = ticker.history(period=f"{max(1, int(years))}y", auto_adjust=True)
            df = _normalize_history(df)
            if len(df) < min_rows:
                errors.append(f"{candidate}: {len(df)} rows")
                continue
            return candidate, fetch_display_name(candidate, ticker), df, "Yahoo Finance"
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
    detail = "；".join(errors[-3:]) if errors else "no data"
    raise RuntimeError(f"查無 {symbol} 歷史資料：{detail}")


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = _normalize_history(df)
    close = out["Close"]
    high = out["High"]
    low = out["Low"]
    volume = out["Volume"]
    out["MA5"] = close.rolling(5).mean()
    out["MA20"] = close.rolling(20).mean()
    out["MA60"] = close.rolling(60).mean()
    out["MA120"] = close.rolling(120).mean()
    out["RSI14"] = _rsi(close, 14)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_HIST"] = out["MACD"] - out["MACD_SIGNAL"]
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    out["BB_MID"] = mid
    out["BB_UP"] = mid + 2 * std
    out["BB_DN"] = mid - 2 * std
    out["BB_WIDTH"] = (out["BB_UP"] - out["BB_DN"]) / close.replace(0, np.nan)
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    out["ATR14"] = tr.rolling(14).mean()
    out["VOL_MA20"] = volume.rolling(20).mean()
    out["VOL_RATIO20"] = volume / out["VOL_MA20"].replace(0, np.nan)
    return out


def support_resistance(ind: pd.DataFrame, window: int = 60) -> Dict[str, float]:
    recent = ind.tail(max(20, window))
    return {
        "support": float(recent["Low"].rolling(5).min().dropna().tail(10).median()),
        "resistance": float(recent["High"].rolling(5).max().dropna().tail(10).median()),
    }


def _build_features(ind: pd.DataFrame) -> pd.DataFrame:
    close = ind["Close"]
    high = ind["High"]
    low = ind["Low"]
    open_ = ind["Open"]
    volume = ind["Volume"].fillna(0)
    typical = (high + low + close) / 3
    vol_sum = volume.rolling(20).sum().replace(0, np.nan)
    vwap20 = (typical * volume).rolling(20).sum() / vol_sum
    rng = (high - low).replace(0, np.nan)
    rets = close.pct_change()
    f = pd.DataFrame(index=ind.index)
    for n in (1, 3, 5, 10, 20, 60):
        f[f"ret_{n}"] = close.pct_change(n)
    f["volatility_5"] = rets.rolling(5).std()
    f["volatility_20"] = rets.rolling(20).std()
    f["volatility_60"] = rets.rolling(60).std()
    f["vol_ratio_5"] = volume / volume.rolling(5).mean().replace(0, np.nan)
    f["vol_ratio_20"] = volume / volume.rolling(20).mean().replace(0, np.nan)
    f["range_pct"] = (high - low) / close
    f["body_pct"] = (close - open_) / close
    f["close_location"] = (close - low) / rng
    f["gap_pct"] = open_ / close.shift(1).replace(0, np.nan) - 1
    f["rsi14"] = ind["RSI14"] / 100
    f["macd_hist_pct"] = ind["MACD_HIST"] / close
    f["bb_pos"] = (close - ind["BB_DN"]) / (ind["BB_UP"] - ind["BB_DN"]).replace(0, np.nan)
    f["bb_width"] = ind["BB_WIDTH"]
    f["atr_pct"] = ind["ATR14"] / close
    f["ma5_dist"] = (close - ind["MA5"]) / close
    f["ma20_dist"] = (close - ind["MA20"]) / close
    f["ma60_dist"] = (close - ind["MA60"]) / close
    f["ma20_slope"] = ind["MA20"].pct_change(5)
    f["ma60_slope"] = ind["MA60"].pct_change(10)
    f["vwap20_dist"] = (close - vwap20) / close
    f["dist_hi20"] = (close - high.rolling(20).max()) / high.rolling(20).max()
    f["dist_lo20"] = (close - low.rolling(20).min()) / low.rolling(20).min()
    f["trend_regime"] = np.where((close >= ind["MA20"]) & (ind["MA20"] >= ind["MA60"]), 1,
                                 np.where((close <= ind["MA20"]) & (ind["MA20"] <= ind["MA60"]), -1, 0))
    f["vol_regime"] = (f["volatility_20"] / f["volatility_60"].replace(0, np.nan)).clip(0, 4) / 4
    month = pd.Series(ind.index.month, index=ind.index)
    dow = pd.Series(ind.index.dayofweek, index=ind.index)
    f["month_sin"] = np.sin(2 * np.pi * month / 12)
    f["month_cos"] = np.cos(2 * np.pi * month / 12)
    f["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 5)
    f = f.replace([np.inf, -np.inf], np.nan)
    return f[FEATURE_COLUMNS]


def _segments(ind: pd.DataFrame) -> pd.Series:
    close = ind["Close"]
    trend = np.where((close >= ind["MA20"]) & (ind["MA20"] >= ind["MA60"]), "up",
                     np.where((close <= ind["MA20"]) & (ind["MA20"] <= ind["MA60"]), "down", "side"))
    vol20 = close.pct_change().rolling(20).std()
    vol60 = close.pct_change().rolling(60).std()
    vol = np.where(vol20 > vol60 * 1.15, "highvol", "normalvol")
    day = pd.Series(ind.index.day, index=ind.index)
    phase = np.where(day <= 10, "early", np.where(day >= 21, "late", "mid"))
    return pd.Series([f"{a}_{b}_{c}" for a, b, c in zip(phase, trend, vol)], index=ind.index)


def _fit_model(X: pd.DataFrame, y: pd.Series) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(
        n_estimators=80,
        learning_rate=0.055,
        max_depth=2,
        random_state=42,
        subsample=0.9,
    )
    model.fit(X, y)
    return model


def predict_future_return(df: pd.DataFrame, horizon: int) -> Dict[str, float]:
    ind = compute_indicators(df)
    X = _build_features(ind)
    y = ind["Close"].shift(-horizon) / ind["Close"] - 1
    seg = _segments(ind)
    train = X.join(y.rename("target")).dropna()
    if len(train) < 90:
        baseline = float(ind["Close"].pct_change(horizon).dropna().tail(120).median())
        return {
            "expected_return": float(np.clip(baseline, -0.25, 0.25)),
            "prob_up": 0.5 if baseline == 0 else float(1 / (1 + math.exp(-baseline / 0.04))),
            "segment": str(seg.iloc[-1]),
            "model": "baseline",
            "residual_std": float(ind["Close"].pct_change().tail(60).std() * math.sqrt(max(1, horizon))),
        }
    X_train = train[FEATURE_COLUMNS]
    y_train = train["target"].clip(-0.35, 0.35)
    current_x = X.iloc[[-1]].fillna(0)
    current_seg = str(seg.iloc[-1])
    global_model = _fit_model(X_train, y_train)
    pred_global = float(global_model.predict(current_x)[0])
    seg_mask = seg.reindex(train.index).eq(current_seg)
    pred = pred_global
    model_name = "global"
    if seg_mask.sum() >= 55:
        seg_model = _fit_model(X_train.loc[seg_mask], y_train.loc[seg_mask])
        pred_seg = float(seg_model.predict(current_x)[0])
        pred = 0.65 * pred_seg + 0.35 * pred_global
        model_name = f"segment+global ({current_seg})"
    fitted = pd.Series(global_model.predict(X_train), index=X_train.index)
    resid_std = float((y_train - fitted).std())
    resid_std = resid_std if np.isfinite(resid_std) and resid_std > 0 else 0.05
    pred = float(np.clip(pred, -0.35, 0.35))
    prob_up = float(1 / (1 + math.exp(-pred / max(resid_std * 0.8, 0.015))))
    return {
        "expected_return": pred,
        "prob_up": prob_up,
        "segment": current_seg,
        "model": model_name,
        "residual_std": resid_std,
    }


def walk_forward_backtest(df: pd.DataFrame, horizon: int, max_folds: int = 36) -> Tuple[Dict[str, float], pd.DataFrame]:
    ind = compute_indicators(df)
    X = _build_features(ind)
    y = (ind["Close"].shift(-horizon) / ind["Close"] - 1).rename("actual_return")
    seg = _segments(ind).rename("segment")
    data = X.join([y, seg]).dropna()
    min_train = max(120, horizon * 4)
    if len(data) <= min_train + 5:
        empty = pd.DataFrame(columns=["date", "segment", "predicted_return", "actual_return", "hit"])
        return {"hit_rate": 0.0, "mae": 0.0, "n": 0}, empty
    step = max(5, min(20, horizon))
    positions = list(range(min_train, len(data), step))[-max_folds:]
    rows = []
    for pos in positions:
        train = data.iloc[:pos]
        test = data.iloc[pos]
        X_train = train[FEATURE_COLUMNS]
        y_train = train["actual_return"].clip(-0.35, 0.35)
        model = _fit_model(X_train, y_train)
        pred_global = float(model.predict(test[FEATURE_COLUMNS].to_frame().T)[0])
        same_seg = train["segment"].eq(test["segment"])
        pred = pred_global
        if same_seg.sum() >= 55:
            seg_model = _fit_model(X_train.loc[same_seg], y_train.loc[same_seg])
            pred_seg = float(seg_model.predict(test[FEATURE_COLUMNS].to_frame().T)[0])
            pred = 0.65 * pred_seg + 0.35 * pred_global
        actual = float(test["actual_return"])
        rows.append({
            "date": data.index[pos],
            "segment": str(test["segment"]),
            "predicted_return": float(np.clip(pred, -0.35, 0.35)),
            "actual_return": actual,
            "hit": bool(np.sign(pred) == np.sign(actual)) if actual != 0 else False,
        })
    detail = pd.DataFrame(rows)
    if detail.empty:
        return {"hit_rate": 0.0, "mae": 0.0, "n": 0}, detail
    summary = {
        "hit_rate": float(detail["hit"].mean()),
        "mae": float((detail["predicted_return"] - detail["actual_return"]).abs().mean()),
        "avg_predicted_return": float(detail["predicted_return"].mean()),
        "avg_actual_return": float(detail["actual_return"].mean()),
        "n": int(len(detail)),
    }
    return summary, detail


def forecast_price_path(df: pd.DataFrame, prediction: dict, days: int) -> pd.DataFrame:
    last_date = pd.to_datetime(df.index[-1])
    last_price = float(df["Close"].iloc[-1])
    expected_return = float(prediction.get("expected_return", 0.0))
    resid = float(prediction.get("residual_std", 0.04))
    dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=max(1, int(days)))
    progress = np.linspace(1 / len(dates), 1, len(dates))
    total_log_ret = math.log(max(0.05, 1 + expected_return))
    median = last_price * np.exp(total_log_ret * progress)
    band = np.maximum(resid, 0.015) * np.sqrt(progress)
    lower = median * np.exp(-1.15 * band)
    upper = median * np.exp(1.15 * band)
    return pd.DataFrame({"Date": dates, "Median": median, "Lower": lower, "Upper": upper})


def analyze_stock(symbol: str, years: int = 3, forecast_days: int = 20, api_key: str = "") -> AnalysisResult:
    resolved, name, history, source = fetch_history(symbol, years=years, min_rows=90, api_key=api_key)
    indicators = compute_indicators(history)
    prediction = predict_future_return(history, horizon=max(1, int(forecast_days)))
    backtest, detail = walk_forward_backtest(history, horizon=max(1, int(forecast_days)))
    forecast = forecast_price_path(history, prediction, max(1, int(forecast_days)))
    return AnalysisResult(
        input_symbol=symbol,
        symbol=resolved,
        raw_code=tw_code_key(resolved),
        name=name,
        source=source,
        history=history,
        indicators=indicators,
        forecast=forecast,
        prediction=prediction,
        backtest=backtest,
        backtest_detail=detail,
    )


def fetch_intraday(symbol: str) -> Tuple[str, str, pd.DataFrame, str]:
    errors = []
    for candidate in candidate_symbols(symbol):
        try:
            ticker = yf.Ticker(candidate)
            df = ticker.history(period="5d", interval="1m", auto_adjust=False, prepost=False)
            df = _normalize_history(df)
            if df.empty:
                errors.append(f"{candidate}: empty")
                continue
            df.index = pd.to_datetime(df.index)
            df.index = df.index.floor("min")
            df = df[~df.index.duplicated(keep="last")]
            latest_date = df.index.max().date()
            start = pd.Timestamp.combine(latest_date, MARKET_OPEN)
            end = pd.Timestamp.combine(latest_date, MARKET_CLOSE)
            day = df[(df.index >= start) & (df.index <= end)].copy()
            if day.empty:
                day = df[df.index.date == latest_date].copy()
            if day.empty:
                errors.append(f"{candidate}: no regular-session rows")
                continue
            day["Volume_lots"] = day["Volume"].fillna(0) / 1000.0
            return candidate, fetch_display_name(candidate, ticker), day, "Yahoo Finance 1m"
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError("查無盤中資料：" + "；".join(errors[-3:]))


def intraday_stats(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {}
    open_price = float(df["Open"].dropna().iloc[0])
    latest = float(df["Close"].dropna().iloc[-1])
    high = float(df["High"].max())
    low = float(df["Low"].min())
    lots = float(df.get("Volume_lots", df["Volume"] / 1000.0).sum())
    return {
        "open": open_price,
        "latest": latest,
        "high": high,
        "low": low,
        "change_pct": latest / open_price - 1 if open_price else 0.0,
        "volume_lots": lots,
        "last_time": df.index.max(),
    }


def make_intraday_prediction(intraday: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    if intraday is None or intraday.empty:
        return pd.DataFrame(columns=["Time", "Predicted", "Actual"])
    date = intraday.index.max().date()
    start = pd.Timestamp.combine(date, MARKET_OPEN)
    end = pd.Timestamp.combine(date, MARKET_CLOSE)
    full_index = pd.date_range(start, end, freq="1min")
    actual = intraday["Close"].reindex(full_index)
    pred = actual.ffill()
    last_time = intraday.index.max().floor("min")
    last_price = float(intraday["Close"].iloc[-1])
    open_price = float(intraday["Open"].dropna().iloc[0])
    daily_mu = float(history["Close"].pct_change().tail(60).mean()) if history is not None and not history.empty else 0.0
    current_ret = last_price / open_price - 1 if open_price else 0.0
    elapsed = max((last_time - start).total_seconds(), 0)
    total = max((end - start).total_seconds(), 1)
    remaining_frac = max(0.0, 1.0 - elapsed / total)
    target_ret = current_ret + remaining_frac * (0.25 * current_ret + 0.50 * daily_mu)
    target_ret = float(np.clip(target_ret, current_ret - 0.04, current_ret + 0.04))
    target_price = open_price * (1 + target_ret)
    future_mask = full_index > last_time
    n_future = int(future_mask.sum())
    if n_future > 0:
        future_path = np.linspace(last_price, target_price, n_future + 1)[1:]
        pred.loc[future_mask] = future_path
    out = pd.DataFrame({"Time": full_index, "Predicted": pred.values, "Actual": actual.values})
    return out


def get_intraday(symbol: str, history: Optional[pd.DataFrame] = None) -> IntradayResult:
    resolved, name, data, source = fetch_intraday(symbol)
    pred = make_intraday_prediction(data, history if history is not None else pd.DataFrame())
    return IntradayResult(
        symbol=resolved,
        name=name,
        source=source,
        data=data,
        prediction=pred,
        stats=intraday_stats(data),
    )


def format_pct(value: float) -> str:
    try:
        return f"{float(value) * 100:+.2f}%"
    except Exception:
        return "N/A"


def read_api_key(explicit_key: str = "") -> str:
    return explicit_key or os.getenv("FUGLE_API_KEY", "") or os.getenv("FUBON_MARKETDATA_API_KEY", "")
