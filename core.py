# core.py — Backend engine extracted from Stock Analyzer Pro v11
# All analysis logic: indicators, ML, forecast, backtest, names
# This file has ZERO PySide6 / GUI dependencies — pure Python data layer

import os, sys, json, time, warnings, re, html
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, time as dtime
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
    try:
        requests.packages.urllib3.disable_warnings()
    except Exception:
        pass
except ImportError:
    requests = None; REQUESTS_OK = False

try:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
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
    # Balanced default: trend + model are primary; fundamentals/news/chips confirm; margin is a risk overlay.
    "technical": 20,
    "ml": 24,
    "news": 12,
    "fundamental": 14,
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
_TOP_VOLUME_CACHE = {}
_TOP_VOLUME_CACHE_TTL = 1800  # 30 minutes

TW_TZ = "Asia/Taipei"
MARKET_OPEN = dtime(9, 0)
MARKET_CLOSE = dtime(13, 30)

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


def _json_row_sets(payload):
    """Yield table-like row lists from TWSE/TPEX JSON payloads.
    TWSE endpoints have changed key names across years (data/data1/tables),
    so this keeps the parser tolerant instead of returning all-zero values.
    """
    if isinstance(payload, list):
        if payload and all(isinstance(x, (list, tuple, dict)) for x in payload):
            yield payload
        return
    if not isinstance(payload, dict):
        return
    for key in ("data", "data1", "data2", "aaData", "items"):
        rows = payload.get(key)
        if isinstance(rows, list) and rows:
            yield rows
    tables = payload.get("tables")
    if isinstance(tables, list):
        for t in tables:
            if isinstance(t, dict):
                rows = t.get("data") or t.get("rows")
                if isinstance(rows, list) and rows:
                    yield rows

def _extract_twse_margin_row(row):
    """Parse one TWSE MI_MARGN row into margin/short fields."""
    if isinstance(row, dict):
        code = str(row.get("股票代號") or row.get("Code") or row.get("code") or row.get("SecuritiesCode") or "").strip()
        def g(*keys):
            for k in keys:
                if k in row:
                    return _safe_float(row.get(k))
            return 0.0
        margin_bal = g("融資今日餘額", "融資餘額", "MarginPurchaseTodayBalance", "TodayBalance")
        margin_prev = g("融資前日餘額", "MarginPurchasePreviousBalance", "PreviousBalance")
        margin_buy = g("融資買進", "MarginPurchaseBuy")
        margin_sell = g("融資賣出", "MarginPurchaseSell")
        margin_repay = g("融資現金償還", "CashRepayment")
        short_bal = g("融券今日餘額", "融券餘額", "ShortSaleTodayBalance")
        short_prev = g("融券前日餘額", "ShortSalePreviousBalance")
        short_sell = g("融券賣出", "ShortSaleSell")
        short_buy = g("融券買進", "ShortSaleBuy")
        short_repay = g("融券現券償還", "StockRepayment")
    else:
        vals = [str(x).strip() for x in row]
        code = vals[0] if vals else ""
        margin_buy = _safe_float(vals[2]) if len(vals) > 2 else 0.0
        margin_sell = _safe_float(vals[3]) if len(vals) > 3 else 0.0
        margin_repay = _safe_float(vals[4]) if len(vals) > 4 else 0.0
        margin_prev = _safe_float(vals[5]) if len(vals) > 5 else 0.0
        margin_bal = _safe_float(vals[6]) if len(vals) > 6 else 0.0
        short_sell = _safe_float(vals[8]) if len(vals) > 8 else 0.0
        short_buy = _safe_float(vals[9]) if len(vals) > 9 else 0.0
        short_repay = _safe_float(vals[10]) if len(vals) > 10 else 0.0
        short_prev = _safe_float(vals[11]) if len(vals) > 11 else 0.0
        short_bal = _safe_float(vals[12]) if len(vals) > 12 else 0.0
    margin_change = margin_bal - margin_prev if (margin_bal or margin_prev) else margin_buy - margin_sell - margin_repay
    short_change = short_bal - short_prev if (short_bal or short_prev) else short_sell - short_buy - short_repay
    return code, margin_bal, margin_change, short_bal, short_change


def _normalize_lot_units(x):
    """Normalize Taiwan margin figures to trading lots (張) when source returns shares."""
    x = _safe_float(x)
    return x / 1000.0 if abs(x) >= 1_000_000 else x


def _margin_score_from_values(margin_change, short_change, margin_balance=0.0, short_balance=0.0):
    """Contrarian-ish credit signal: financing increase is mild negative; short increase may be squeeze-positive."""
    margin_change = _normalize_lot_units(margin_change)
    short_change = _normalize_lot_units(short_change)
    margin_balance = max(abs(_normalize_lot_units(margin_balance)), 1.0)
    short_balance = max(abs(_normalize_lot_units(short_balance)), 1.0)
    m_pressure = np.clip(margin_change / max(margin_balance * 0.08, 1500.0), -1, 1)
    s_pressure = np.clip(short_change / max(short_balance * 0.25, 500.0), -1, 1)
    return float(np.clip(-0.55 * m_pressure + 0.45 * s_pressure, -1, 1))


def _make_margin_result(date, margin_balance, margin_change, short_balance, short_change, source, note=""):
    margin_balance = _normalize_lot_units(margin_balance)
    margin_change = _normalize_lot_units(margin_change)
    short_balance = _normalize_lot_units(short_balance)
    short_change = _normalize_lot_units(short_change)
    return {
        "date": str(date or ""),
        "margin_balance": float(margin_balance),
        "margin_change": float(margin_change),
        "short_balance": float(short_balance),
        "short_change": float(short_change),
        "margin_score": _margin_score_from_values(margin_change, short_change, margin_balance, short_balance),
        "source": source,
        "note": note,
    }


def _looks_like_valid_margin(m):
    return bool(m) and any(abs(float(m.get(k, 0) or 0)) > 1e-9 for k in ("margin_balance", "short_balance", "margin_change", "short_change"))


def _fetch_finmind_margin(raw_code: str) -> dict:
    """Fallback: FinMind open-data API, dataset TaiwanStockMarginPurchaseShortSale."""
    if not REQUESTS_OK:
        return {}
    try:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
        start = (pd.Timestamp.today() - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {"dataset": "TaiwanStockMarginPurchaseShortSale", "data_id": str(raw_code), "start_date": start, "end_date": end}
        r = requests.get(url, params=params, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return {}
        payload = r.json() or {}
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        if not rows:
            return {}
        rows = sorted(rows, key=lambda x: str(x.get("date", "")))
        last = rows[-1]
        prev = rows[-2] if len(rows) >= 2 else {}
        mb = _safe_float(last.get("MarginPurchaseTodayBalance") or last.get("margin_purchase_today_balance") or last.get("MarginPurchaseTodayBalanceShares"))
        sb = _safe_float(last.get("ShortSaleTodayBalance") or last.get("short_sale_today_balance") or last.get("ShortSaleTodayBalanceShares"))
        pm = _safe_float(prev.get("MarginPurchaseTodayBalance") or prev.get("margin_purchase_today_balance") or prev.get("MarginPurchaseTodayBalanceShares")) if prev else 0.0
        ps = _safe_float(prev.get("ShortSaleTodayBalance") or prev.get("short_sale_today_balance") or prev.get("ShortSaleTodayBalanceShares")) if prev else 0.0
        mc = mb - pm if (mb or pm) else (_safe_float(last.get("MarginPurchaseBuy")) - _safe_float(last.get("MarginPurchaseSell")) - _safe_float(last.get("MarginPurchaseCashRepayment")))
        sc = sb - ps if (sb or ps) else (_safe_float(last.get("ShortSaleSell")) - _safe_float(last.get("ShortSaleBuy")) - _safe_float(last.get("ShortSaleCashRepayment")))
        if any(abs(x) > 0 for x in (mb, sb, mc, sc)):
            return _make_margin_result(last.get("date", "latest"), mb, mc, sb, sc, "FinMind", "")
    except Exception:
        return {}
    return {}


def _extract_first_number_after(text: str, labels) -> float:
    for label in labels:
        pattern = re.escape(label) + r"[^0-9+\-]{0,30}([+\-]?[0-9][0-9,]*\.?[0-9]*)"
        m = re.search(pattern, text)
        if m:
            return _safe_float(m.group(1))
    return 0.0


def _fetch_public_web_margin(raw_code: str) -> dict:
    """Fallback scrape from public web pages such as Yahoo Finance TW and WantGoo."""
    if not REQUESTS_OK:
        return {}
    urls = [
        (f"https://tw.stock.yahoo.com/quote/{raw_code}.TW/margin", "Yahoo股市"),
        (f"https://tw.stock.yahoo.com/quote/{raw_code}.TWO/margin", "Yahoo股市"),
        (f"https://www.wantgoo.com/stock/{raw_code}/margin-trading/synopsis", "WantGoo"),
    ]
    for url, source in urls:
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"}, verify=False)
            if r.status_code != 200 or not r.text:
                continue
            text = re.sub(r"<[^>]+>", " ", r.text)
            text = re.sub(r"\s+", " ", text)
            mb = _extract_first_number_after(text, ["融資餘額(張)", "融資餘額", "資餘額"])
            sb = _extract_first_number_after(text, ["融券餘額(張)", "融券餘額", "券餘額"])
            mc = _extract_first_number_after(text, ["融資增減(張)", "融資增減", "資增減", "融資變化"])
            sc = _extract_first_number_after(text, ["融券增減(張)", "融券增減", "券增減", "融券變化"])
            dm = re.search(r"(20\d{2}[/-]\d{1,2}[/-]\d{1,2})", text)
            if any(abs(x) > 0 for x in (mb, sb, mc, sc)):
                return _make_margin_result(dm.group(1) if dm else "latest", mb, mc, sb, sc, source, "")
        except Exception:
            continue
    return {}


def company_event_score(news: list, fund: dict = None) -> float:
    """Score company-specific technology/order/business momentum from news and fundamentals."""
    fund = fund or {}
    text = " ".join((it.get("title", "") + " " + it.get("summary", "")) for it in (news or []))
    pos_kw = ["接單", "訂單", "大單", "出貨", "量產", "擴產", "先進製程", "AI", "CoWoS", "HBM", "伺服器", "ASIC", "光通訊", "新產品", "營收創高", "法說上修", "展望樂觀"]
    neg_kw = ["砍單", "降價", "庫存", "延後", "展望保守", "營收衰退", "毛利率下滑", "產能利用率下降", "競爭加劇", "客戶流失"]
    pos = sum(1 for k in pos_kw if k.lower() in text.lower())
    neg = sum(1 for k in neg_kw if k.lower() in text.lower())
    s = (pos - neg) / max(pos + neg, 1)
    if fund.get("rev_growth") is not None:
        s += float(np.clip(fund.get("rev_growth", 0) / 0.25, -0.35, 0.35))
    if fund.get("earn_growth") is not None:
        s += float(np.clip(fund.get("earn_growth", 0) / 0.35, -0.35, 0.35))
    if fund.get("profit_margin") is not None:
        s += float(np.clip((fund.get("profit_margin", 0) - 0.08) / 0.25, -0.20, 0.25))
    return float(np.clip(s, -1, 1))


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
_VERIFIED_NAMES = {
    "00631L": "元大台灣50正2", "00632R": "元大台灣50反1",
    "00633L": "富邦台灣加強", "00634R": "富邦台灣反1",
    "00663L": "國泰臺灣加強", "00664R": "國泰臺灣反1",
    "00670L": "富邦NASDAQ正2", "00671R": "富邦NASDAQ反1",
    "00672L": "元大S&P正2", "00673R": "元大S&P反1",
    "00980A": "中信高優息成長ETF",
    "00981A": "主動統一台股增長",
    "00982A": "主動群益台灣強棒",
    "8069": "元太",
    "6488": "環球晶",
    "006201": "元大富櫃50",
}
TW_NAME_CACHE.update(_VERIFIED_NAMES)

def _cjk(s: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in str(s or ""))

def _tw_code_key(value) -> str:
    s = str(value or "").strip().upper().replace("\u3000", " ")
    s = re.sub(r"\.(TW|TWO)$", "", s)
    m = re.search(r"(?<!\d)(\d{4,6}[A-Z]?)(?!\d)", s)
    return m.group(1) if m else re.sub(r"[^0-9A-Z]", "", s)

def _clean_tw_name(raw: str) -> str:
    s = html.unescape(str(raw or "")).strip()
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" \t\r\n　-–—:：")
    s = re.sub(r"\(\s*\d{4,6}[A-Z]?\.(TW|TWO)\s*\).*", "", s, flags=re.I).strip()
    s = re.sub(r"\b\d{4,6}[A-Z]?\b", "", s).strip(" ()（）-–—:：")
    for suffix in ("股份有限公司", "有限公司", "公司"):
        if s.endswith(suffix) and len(s) - len(suffix) >= 2:
            s = s[:-len(suffix)].strip()
            break
    if "�" in s or not _cjk(s):
        return ""
    if s.count("?") >= max(2, len(s) // 2):
        return ""
    return s

def _tw_name_ok(name: str, code: str = "") -> bool:
    s = _clean_tw_name(name)
    if not s:
        return False
    if s.upper() in {"N/A", "NA", "NULL", "NONE", str(code or "").upper()}:
        return False
    if re.fullmatch(r"[0-9A-Z._-]+", s.upper()):
        return False
    return True

def _remember_tw_name(code, name, overwrite: bool = False) -> str:
    code = _tw_code_key(code)
    name = _clean_tw_name(name)
    if not code or not _tw_name_ok(name, code):
        return ""
    if code in _VERIFIED_NAMES and _tw_name_ok(_VERIFIED_NAMES[code], code):
        name = _clean_tw_name(_VERIFIED_NAMES[code])
        overwrite = True
    old = TW_NAME_CACHE.get(code, "")
    if overwrite or not _tw_name_ok(old, code):
        TW_NAME_CACHE[code] = name
        return name
    return _clean_tw_name(old)

CODE_KEYS = (
    "Code", "code", "StockNo", "stockNo", "StockID",
    "SecuritiesCode", "SecuritiesCompanyCode", "SecurityCode",
    "股票代號", "證券代號", "有價證券代號", "公司代號", "代號",
)
NAME_KEYS = (
    "Name", "name", "StockName", "stockName", "shortName",
    "CompanyAbbreviation", "CompanyShortName", "SecuritiesName",
    "CompanyName", "SecurityName", "股票名稱", "證券名稱",
    "有價證券名稱", "公司簡稱", "公司名稱", "名稱",
)

def _extract_code_name_from_item(item) -> tuple:
    if not isinstance(item, dict):
        return "", ""
    code = ""
    name = ""
    for key in CODE_KEYS:
        if key in item:
            code = _tw_code_key(item.get(key))
            if code:
                break
    if not code:
        for key, value in item.items():
            low = str(key).lower()
            if "代號" in str(key) or "code" in low or "stockno" in low:
                code = _tw_code_key(value)
                if code:
                    break
    for key in NAME_KEYS:
        if key in item:
            cand = _clean_tw_name(item.get(key))
            if _tw_name_ok(cand, code):
                name = cand
                break
    if not name:
        for key, value in item.items():
            low = str(key).lower()
            if "名稱" in str(key) or "簡稱" in str(key) or "name" in low or "abbreviation" in low:
                cand = _clean_tw_name(value)
                if _tw_name_ok(cand, code):
                    name = cand
                    break
    return code, name

def _fetch_json_list(url: str, timeout: int = 8) -> list:
    if not REQUESTS_OK:
        return []
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json,text/plain,*/*"}
    try:
        r = requests.get(url, timeout=timeout, headers=headers, verify=False)
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

def _load_names_disk_cache():
    try:
        if _NAMES_CACHE_FILE.exists():
            with open(_NAMES_CACHE_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            for k, v in saved.items():
                code = _tw_code_key(k)
                if code and _tw_name_ok(v, code) and not _tw_name_ok(TW_NAME_CACHE.get(code, ""), code):
                    TW_NAME_CACHE[code] = _clean_tw_name(v)
    except Exception:
        pass

def _save_names_disk_cache():
    try:
        clean = {
            _tw_code_key(k): _clean_tw_name(v)
            for k, v in TW_NAME_CACHE.items()
            if _tw_code_key(k) and _tw_name_ok(v, _tw_code_key(k))
        }
        with open(_NAMES_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False, indent=1)
    except Exception:
        pass

_load_names_disk_cache()

def _load_twse_bulk():
    global _twse_bulk_loaded
    if _twse_bulk_loaded or not REQUESTS_OK: return
    _twse_bulk_loaded = True
    endpoints = [
        "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
        "https://openapi.twse.com.tw/v1/opendata/t187ap03_L",
        "https://openapi.twse.com.tw/v1/opendata/t187ap05_L",
    ]
    added = 0
    for url in endpoints:
        for item in _fetch_json_list(url, timeout=10):
            code, name = _extract_code_name_from_item(item)
            before = TW_NAME_CACHE.get(code, "")
            stored = _remember_tw_name(code, name)
            if stored and stored != before:
                added += 1
    if added > 0: _save_names_disk_cache()

def _load_tpex_bulk():
    global _tpex_bulk_loaded
    if _tpex_bulk_loaded or not REQUESTS_OK: return
    _tpex_bulk_loaded = True
    endpoints = [
        "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes",
        "https://www.tpex.org.tw/openapi/v1/tpex_daily_market_value",
        "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O",
        "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap05_O",
        "https://www.tpex.org.tw/openapi/v1/tpex_esb_latest_statistics",
        "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_R",
    ]
    added = 0
    for url in endpoints:
        for item in _fetch_json_list(url, timeout=10):
            code, name = _extract_code_name_from_item(item)
            before = TW_NAME_CACHE.get(code, "")
            stored = _remember_tw_name(code, name)
            if stored and stored != before:
                added += 1
    if added > 0: _save_names_disk_cache()

def _fetch_yahoo_tw_name(raw_code: str) -> str:
    code = _tw_code_key(raw_code)
    if not code or not REQUESTS_OK:
        return ""
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/html,*/*"}
    for suffix in ("TW", "TWO"):
        try:
            url = f"https://tw.stock.yahoo.com/quote/{code}.{suffix}"
            r = requests.get(url, timeout=7, headers=headers)
            if r.status_code != 200:
                continue
            m = re.search(r"<title[^>]*>(.*?)</title>", r.text or "", flags=re.I | re.S)
            title = html.unescape(m.group(1)) if m else (r.text or "")[:400]
            title = re.sub(r"<[^>]+>", " ", title)
            for pat in (
                rf"^\s*(.*?)\s*\(\s*{re.escape(code)}\.(?:TW|TWO)\s*\)",
                r"^\s*([\u4e00-\u9fff][\w\u4e00-\u9fff＋+\-－*（）()·‧& ]{1,30})\s*(?:走勢圖|即時行情)",
            ):
                mm = re.search(pat, title, flags=re.I)
                if mm:
                    name = _clean_tw_name(mm.group(1))
                    if _tw_name_ok(name, code):
                        return _remember_tw_name(code, name, overwrite=True)
        except Exception:
            continue
    return ""

def resolve_tw_display_name(raw_code: str, ticker=None, symbol: str = "", prefer_cache: bool = True) -> str:
    code = _tw_code_key(raw_code or symbol)
    if not code:
        return ""
    if code in _VERIFIED_NAMES and _tw_name_ok(_VERIFIED_NAMES[code], code):
        return _remember_tw_name(code, _VERIFIED_NAMES[code], overwrite=True)
    cached = TW_NAME_CACHE.get(code, "")
    if prefer_cache and _tw_name_ok(cached, code):
        return _clean_tw_name(cached)
    name = _fetch_yahoo_tw_name(code)
    if _tw_name_ok(name, code):
        _save_names_disk_cache()
        return _clean_tw_name(name)
    _load_twse_bulk()
    cached = TW_NAME_CACHE.get(code, "")
    if _tw_name_ok(cached, code):
        return _clean_tw_name(cached)
    _load_tpex_bulk()
    cached = TW_NAME_CACHE.get(code, "")
    if _tw_name_ok(cached, code):
        return _clean_tw_name(cached)
    if ticker is not None:
        try:
            info = ticker.info or {}
            for key in ("shortName", "longName", "displayName"):
                candidate = _clean_tw_name(info.get(key, ""))
                if _tw_name_ok(candidate, code):
                    _remember_tw_name(code, candidate, overwrite=True)
                    _save_names_disk_cache()
                    return candidate
        except Exception:
            pass
    return code

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


def _normalize_history_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    if not isinstance(out.index, pd.DatetimeIndex):
        if "Date" in out.columns:
            out.index = pd.to_datetime(out.pop("Date"), errors="coerce")
        else:
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
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in out.columns:
            out[col] = 0.0 if col == "Volume" else np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Close"])
    out = out[out["Close"] > 0].sort_index()
    out = out[~out.index.duplicated(keep="last")]
    if out.empty:
        return pd.DataFrame()
    out["Open"] = out["Open"].where(out["Open"] > 0, out["Close"])
    out["High"] = out["High"].where(out["High"] > 0, out[["Open", "Close"]].max(axis=1))
    out["Low"] = out["Low"].where(out["Low"] > 0, out[["Open", "Close"]].min(axis=1))
    out["Volume"] = out["Volume"].fillna(0).clip(lower=0)
    return out[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def read_marketdata_key(explicit_key: str = "") -> str:
    return (
        explicit_key
        or os.getenv("FUGLE_API_KEY", "")
        or os.getenv("FUBON_MARKETDATA_API_KEY", "")
    )


def _fetch_authorized_history(raw_code: str, lookback_years: int = 3, api_key: str = "") -> pd.DataFrame:
    key = read_marketdata_key(api_key)
    raw = _tw_code_key(raw_code)
    if not key or not raw or not REQUESTS_OK:
        return pd.DataFrame()
    try:
        end = pd.Timestamp.now(tz=TW_TZ).date()
        start = (pd.Timestamp(end) - pd.DateOffset(years=max(1, int(lookback_years))) - pd.Timedelta(days=10)).date()
        url = f"https://api.fugle.tw/marketdata/v1.0/stock/historical/candles/{raw}"
        params = {
            "from": str(start), "to": str(end), "timeframe": "D",
            "adjusted": "true", "fields": "open,high,low,close,volume", "sort": "asc",
        }
        headers = {"X-API-KEY": key, "User-Agent": "Mozilla/5.0", "Accept": "application/json"}
        r = requests.get(url, params=params, headers=headers, timeout=10)
        if r.status_code != 200:
            return pd.DataFrame()
        payload = r.json()
        rows = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(rows, list) or not rows:
            return pd.DataFrame()
        recs = []
        for it in rows:
            if not isinstance(it, dict):
                continue
            recs.append({
                "Date": pd.to_datetime(it.get("date"), errors="coerce"),
                "Open": it.get("open"),
                "High": it.get("high"),
                "Low": it.get("low"),
                "Close": it.get("close"),
                "Volume": it.get("volume", 0),
            })
        return _normalize_history_df(pd.DataFrame(recs).dropna(subset=["Date"]).set_index("Date"))
    except Exception:
        return pd.DataFrame()


def resolve_and_fetch(symbol: str, lookback_years: int = 3, api_key: str = "") -> Tuple[str, str, pd.DataFrame]:
    """Fetch history + resolve name. Returns (resolved_sym, name, df)."""
    if not YF_OK:
        raise RuntimeError("yfinance 未安裝")
    raw = _tw_code_key(symbol)
    if raw and re.fullmatch(r"\d{4,6}[A-Z]?", raw):
        auth_df = _fetch_authorized_history(raw, lookback_years, api_key)
        if len(auth_df) >= 30:
            return f"{raw}.TW", resolve_tw_display_name(raw), auth_df

    candidates = []
    s = str(symbol or "").strip().upper()
    if s.endswith((".TW",".TWO")):
        candidates = [s]
    elif "." in s:
        candidates = [s]
    elif raw and re.fullmatch(r"\d{4,6}[A-Z]?", raw):
        candidates = [f"{raw}.TW", f"{raw}.TWO"]
    else:
        candidates = [s]

    HARD_MIN = 30
    period   = f"{max(lookback_years,1)}y"
    best     = None
    last_err = ""
    for sym in candidates:
        try:
            ticker = yf.Ticker(sym)
            df = _yf_history_with_retry(ticker, period)
            if df is None or df.empty: last_err = f"{sym}: 無資料"; continue
            df = _normalize_history_df(df)
            n = len(df)
            if n < HARD_MIN: last_err = f"{sym}: {n}筆不足"; continue
            # Resolve name
            rc = _tw_code_key(sym)
            if rc and re.fullmatch(r"\d{4,6}[A-Z]?", rc):
                name = resolve_tw_display_name(rc, ticker=ticker, symbol=sym, prefer_cache=True)
            else:
                name = ""
                try:
                    info = ticker.info or {}
                    for key in ("shortName","longName","displayName"):
                        v = str(info.get(key,"") or "").strip()
                        if v and v != rc:
                            name = v; break
                except Exception:
                    pass
            is_tw = bool(rc and re.fullmatch(r"\d{4,6}[A-Z]?", rc))
            if not name or (is_tw and not _cjk(name)):
                name = rc or sym
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

    w_tech = weights.get("technical", DEFAULT_WEIGHTS["technical"]) / 100.0
    w_ml   = weights.get("ml", DEFAULT_WEIGHTS["ml"]) / 100.0
    w_news = weights.get("news", DEFAULT_WEIGHTS["news"]) / 100.0
    w_fund = weights.get("fundamental", DEFAULT_WEIGHTS["fundamental"]) / 100.0
    w_us   = weights.get("us_market", DEFAULT_WEIGHTS["us_market"]) / 100.0
    w_inst = weights.get("institutional", DEFAULT_WEIGHTS["institutional"]) / 100.0
    w_marg = weights.get("margin", DEFAULT_WEIGHTS["margin"]) / 100.0

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
        if fund.get("eps") and fund.get("eps") > 0: fs += 0.08
        if fund.get("rev_growth") is not None: fs += float(np.clip(fund.get("rev_growth", 0) / 0.30, -0.25, 0.30))
        if fund.get("earn_growth") is not None: fs += float(np.clip(fund.get("earn_growth", 0) / 0.40, -0.25, 0.30))
        if fund.get("company_event_score") is not None: fs += float(np.clip(fund.get("company_event_score", 0), -1, 1)) * 0.30
    fs = float(np.clip(fs, -1, 1))

    us_sig = (
        np.clip(mkt_ctx.get("nasdaq_ret_1", 0.0) / 0.025, -1, 1) * 0.30 +
        np.clip(mkt_ctx.get("sp500_ret_1", 0.0) / 0.020, -1, 1) * 0.20 +
        np.clip(mkt_ctx.get("semis_ret_1", 0.0) / 0.030, -1, 1) * 0.25 +
        np.clip(mkt_ctx.get("sox_ret_1", 0.0) / 0.030, -1, 1) * 0.10 +
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
# Future-return regression + walk-forward detail
# ═══════════════════════════════════════════════════════════════════════════
def _future_return(df: pd.DataFrame, idx: int, horizon: int) -> float:
    try:
        cur = float(df["Close"].iloc[idx])
        fut = float(df["Close"].iloc[idx + horizon])
        if cur <= 0 or fut <= 0:
            return np.nan
        return float(fut / cur - 1.0)
    except Exception:
        return np.nan


def _time_segment(df: pd.DataFrame, ind: dict, idx: int) -> str:
    try:
        dt = pd.to_datetime(df.index[idx])
        close = float(df["Close"].iloc[idx])
        ma20 = float(ind["ma20"].iloc[idx])
        ma60 = float(ind["ma60"].iloc[idx])
        trend = "up" if close >= ma20 >= ma60 else ("down" if close <= ma20 <= ma60 else "side")
        ret20 = df["Close"].pct_change().rolling(20).std().iloc[idx]
        ret60 = df["Close"].pct_change().rolling(60).std().iloc[idx]
        vol = "highvol" if _safe_float(ret20) > max(_safe_float(ret60), 1e-9) * 1.15 else "normalvol"
        phase = "early" if dt.day <= 10 else ("late" if dt.day >= 21 else "mid")
        return f"{phase}_{trend}_{vol}"
    except Exception:
        return "unknown"


def segment_to_zh(segment: str) -> str:
    """Translate internal model segment key to user-facing Chinese."""
    parts = str(segment or "").split("_")
    phase_map = {"early": "月初", "mid": "月中", "late": "月末"}
    trend_map = {"up": "多頭", "down": "空頭", "side": "盤整"}
    vol_map = {"normalvol": "正常波動", "highvol": "高波動"}
    if len(parts) >= 3:
        return f"{phase_map.get(parts[0], parts[0])}｜{trend_map.get(parts[1], parts[1])}｜{vol_map.get(parts[2], parts[2])}"
    if segment == "unknown":
        return "未知分段"
    return str(segment or "未知分段")


def _build_return_samples(df, ind, sr, horizon=10, mkt_ctx=None, inst=None):
    X, y, segs, idxs = [], [], [], []
    h = max(1, int(horizon))
    for i in range(25, len(df) - h):
        vec = build_full_feature_row(df, ind, sr, i, mkt_ctx or {}, inst or {})
        if vec is None:
            continue
        ret = _future_return(df, i, h)
        if not np.isfinite(ret):
            continue
        X.append(vec)
        y.append(float(np.clip(ret, -0.35, 0.35)))
        segs.append(_time_segment(df, ind, i))
        idxs.append(df.index[i])
    if not X:
        return np.empty((0, 27)), np.empty(0), np.array([]), []
    return np.array(X, dtype=float), np.array(y, dtype=float), np.array(segs), idxs


def _fit_return_regressor(X, y):
    model = GradientBoostingRegressor(
        n_estimators=min(140, max(50, len(X) // 4)),
        learning_rate=0.055,
        max_depth=2,
        subsample=0.9,
        random_state=42,
    )
    model.fit(X, y)
    return model


def predict_future_return_model(df, ind, sr, horizon=10, mkt_ctx=None, inst=None) -> dict:
    """Predict future return, not price. Price path is derived later."""
    if not SKLEARN_OK:
        return {"error": "sklearn 未安裝", "expected_return": 0.0, "prob_up": 0.5}
    X, y, segs, _idxs = _build_return_samples(df, ind, sr, horizon, mkt_ctx, inst)
    cur = build_full_feature_row(df, ind, sr, len(df) - 1, mkt_ctx or {}, inst or {})
    if len(X) < 60 or cur is None:
        baseline = float(df["Close"].pct_change(max(1, int(horizon))).dropna().tail(120).median())
        baseline = float(np.clip(baseline if np.isfinite(baseline) else 0.0, -0.25, 0.25))
        return {
            "expected_return": baseline,
            "prob_up": float(1 / (1 + np.exp(-baseline / 0.04))),
            "segment": segment_to_zh(_time_segment(df, ind, len(df) - 1)),
            "model": "baseline",
            "residual_std": float(df["Close"].pct_change().tail(60).std() * np.sqrt(max(1, int(horizon)))),
        }
    global_model = _fit_return_regressor(X, y)
    cur = cur.reshape(1, -1)
    pred_global = float(global_model.predict(cur)[0])
    current_seg = _time_segment(df, ind, len(df) - 1)
    pred = pred_global
    model_name = "return_global"
    same = segs == current_seg
    if int(same.sum()) >= 45:
        seg_model = _fit_return_regressor(X[same], y[same])
        pred_seg = float(seg_model.predict(cur)[0])
        pred = 0.65 * pred_seg + 0.35 * pred_global
        model_name = "return_segment_blend"
    fitted = pd.Series(global_model.predict(X))
    resid_std = float((pd.Series(y) - fitted).std())
    if not np.isfinite(resid_std) or resid_std <= 0:
        resid_std = 0.04
    pred = float(np.clip(pred, -0.35, 0.35))
    prob_up = float(1 / (1 + np.exp(-pred / max(resid_std * 0.8, 0.015))))
    return {
        "expected_return": pred,
        "prob_up": prob_up,
        "segment": segment_to_zh(current_seg),
        "model": model_name,
        "residual_std": resid_std,
        "n_samples": int(len(X)),
        "segment_samples": int(same.sum()),
    }


def walk_forward_return_backtest(df, ind, sr, horizon=10, mkt_ctx=None, inst=None, max_folds=36) -> tuple:
    X, y, segs, idxs = _build_return_samples(df, ind, sr, horizon, mkt_ctx, inst)
    min_train = max(90, int(horizon) * 4)
    if len(X) <= min_train + 5 or not SKLEARN_OK:
        empty = pd.DataFrame(columns=["date", "segment", "predicted_return", "actual_return", "hit"])
        return {"n": 0, "hit_rate": 0.0, "mae": 0.0}, empty
    step = max(5, min(20, int(horizon)))
    positions = list(range(min_train, len(X), step))[-max_folds:]
    rows = []
    for pos in positions:
        try:
            Xtr, ytr = X[:pos], y[:pos]
            model = _fit_return_regressor(Xtr, ytr)
            pred_global = float(model.predict(X[pos].reshape(1, -1))[0])
            same = segs[:pos] == segs[pos]
            pred = pred_global
            if int(same.sum()) >= 45:
                seg_model = _fit_return_regressor(Xtr[same], ytr[same])
                pred_seg = float(seg_model.predict(X[pos].reshape(1, -1))[0])
                pred = 0.65 * pred_seg + 0.35 * pred_global
            actual = float(y[pos])
            rows.append({
                "date": idxs[pos],
                "segment": segment_to_zh(str(segs[pos])),
                "predicted_return": float(np.clip(pred, -0.35, 0.35)),
                "actual_return": actual,
                "hit": bool(np.sign(pred) == np.sign(actual)) if actual != 0 else False,
            })
        except Exception:
            continue
    detail = pd.DataFrame(rows)
    if detail.empty:
        return {"n": 0, "hit_rate": 0.0, "mae": 0.0}, detail
    summary = {
        "n": int(len(detail)),
        "hit_rate": float(detail["hit"].mean()),
        "hit_rate_text": f"{detail['hit'].mean()*100:.1f}%",
        "mae": float((detail["predicted_return"] - detail["actual_return"]).abs().mean()),
        "avg_predicted_return": float(detail["predicted_return"].mean()),
        "avg_actual_return": float(detail["actual_return"].mean()),
    }
    return summary, detail


def align_forecast_with_return_model(fc: dict, df: pd.DataFrame, ret_pred: dict) -> dict:
    """Adjust Monte-Carlo median/bands so endpoint reflects return-regression forecast."""
    if not fc or not ret_pred or "expected_return" not in ret_pred:
        return fc
    try:
        out = dict(fc)
        last = float(df["Close"].iloc[-1])
        target_end = last * (1.0 + float(ret_pred.get("expected_return", 0.0)))
        old_end = float(np.asarray(out["median"])[-1])
        if not np.isfinite(target_end) or not np.isfinite(old_end) or old_end <= 0:
            return fc
        ratio = float(np.clip(target_end / old_end, 0.65, 1.45))
        n = len(out["median"])
        progress = np.linspace(1.0 / n, 1.0, n)
        factors = np.power(ratio, progress)
        for key in ("median", "lower", "upper"):
            out[key] = np.asarray(out[key], dtype=float) * factors
        out["return_model_expected_return"] = float(ret_pred.get("expected_return", 0.0))
        return out
    except Exception:
        return fc


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
    """Fetch TAIEX, VIX, US indices, SMH ETF and Philadelphia Semiconductor Index (^SOX)."""
    ctx = {
        "taiex_ret_1":0.0, "taiex_ret_5":0.0, "taiex_vol":0.01,
        "vix":20.0, "usd_twd":31.5,
        "nasdaq_ret_1":0.0, "nasdaq_ret_5":0.0,
        "sp500_ret_1":0.0, "sp500_ret_5":0.0,
        "semis_ret_1":0.0, "semis_ret_5":0.0,
        "smh_ret_1":0.0, "smh_ret_5":0.0,
        "sox_ret_1":0.0, "sox_ret_5":0.0,
        "sox_last":None,
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
    smh = _yf_return("SMH", "10d")
    sox = _yf_return("^SOX", "10d")
    semi_1 = [x for x in (smh["ret_1"], sox["ret_1"]) if abs(x) > 1e-12]
    semi_5 = [x for x in (smh["ret_5"], sox["ret_5"]) if abs(x) > 1e-12]
    ctx.update({
        "nasdaq_ret_1": ndq["ret_1"], "nasdaq_ret_5": ndq["ret_5"],
        "sp500_ret_1": spx["ret_1"], "sp500_ret_5": spx["ret_5"],
        "smh_ret_1": smh["ret_1"], "smh_ret_5": smh["ret_5"],
        "sox_ret_1": sox["ret_1"], "sox_ret_5": sox["ret_5"], "sox_last": sox.get("last"),
        "semis_ret_1": float(np.mean(semi_1)) if semi_1 else smh["ret_1"],
        "semis_ret_5": float(np.mean(semi_5)) if semi_5 else smh["ret_5"],
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
    """Fetch latest TWSE/TPEX margin financing and short balance.

    Priority: TWSE official, TPEX, FinMind open data, then public Yahoo/WantGoo pages.
    The function skips false all-zero parses; if all sources fail, note explains why.
    """
    result = {
        "date":"", "margin_balance":0.0, "margin_change":0.0,
        "short_balance":0.0, "short_change":0.0, "margin_score":0.0,
        "source":"N/A", "note":"查無融資融券資料；已嘗試 TWSE/TPEX、FinMind、Yahoo股市與公開網頁；可能為非信用交易標的、資料尚未公布或網站暫時阻擋。",
    }
    if not REQUESTS_OK:
        return result
    for date in _tw_date_candidates(28):
        for sel in ("ALL", "ALLBUT0999", "MS"):
            try:
                url = f"https://www.twse.com.tw/exchangeReport/MI_MARGN?response=json&date={date}&selectType={sel}"
                r = requests.get(url, timeout=8, headers={"User-Agent":"Mozilla/5.0"}, verify=False)
                if r.status_code != 200: continue
                payload = r.json() or {}
                for rows in _json_row_sets(payload):
                    for row in rows:
                        code, margin_bal, margin_change, short_bal, short_change = _extract_twse_margin_row(row)
                        if str(code).strip() != raw_code:
                            continue
                        m = _make_margin_result(date, margin_bal, margin_change, short_bal, short_change, f"TWSE:{sel}", "")
                        if _looks_like_valid_margin(m):
                            return m
            except Exception:
                continue
    try:
        for url in ("https://www.tpex.org.tw/openapi/v1/tpex_mainboard_margin_balance", "https://www.tpex.org.tw/openapi/v1/tpex_margin_balance"):
            r = requests.get(url, timeout=8, headers={"User-Agent":"Mozilla/5.0","Accept":"application/json"}, verify=False)
            if r.status_code != 200:
                continue
            data = r.json() or []
            for item in data if isinstance(data, list) else []:
                if not isinstance(item, dict): continue
                code = str(item.get("SecuritiesCompanyCode") or item.get("Code") or item.get("股票代號") or item.get("代號") or item.get("SecuritiesCode") or "").strip()
                if code != raw_code: continue
                def find_val(*needles):
                    for k, v in item.items():
                        ks = str(k)
                        if all(n in ks for n in needles):
                            return _safe_float(v)
                    return 0.0
                margin_bal = find_val("融資", "餘額") or find_val("資", "餘額") or find_val("資餘額")
                margin_change = find_val("融資", "增減") or find_val("資", "增減") or find_val("資增減")
                short_bal = find_val("融券", "餘額") or find_val("券", "餘額") or find_val("券餘額")
                short_change = find_val("融券", "增減") or find_val("券", "增減") or find_val("券增減")
                m = _make_margin_result("latest", margin_bal, margin_change, short_bal, short_change, "TPEX", "")
                if _looks_like_valid_margin(m):
                    return m
    except Exception:
        pass
    for fetcher in (_fetch_finmind_margin, _fetch_public_web_margin):
        m = fetcher(raw_code)
        if _looks_like_valid_margin(m):
            return m
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
    cache_key = f"topvol:{limit}"
    now = time.time()
    if cache_key in _TOP_VOLUME_CACHE:
        ts, cached = _TOP_VOLUME_CACHE[cache_key]
        if now - ts < _TOP_VOLUME_CACHE_TTL:
            return cached[:limit]
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
        ("TPEX", "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"),
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
    result = codes or fallback[:limit]
    _TOP_VOLUME_CACHE[cache_key] = (time.time(), result)
    return result[:limit]

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

def _quick_recommend_candidate(code: str, lookback_years: int, forecast_days: int) -> Optional[dict]:
    """
    Fast first-stage scan for recommendations.
    It still analyzes every top-volume stock, but avoids the expensive per-stock
    news/fundamental/ML/API-heavy calls until finalists are selected.
    """
    try:
        sym, name, df = resolve_and_fetch(code, lookback_years)
        ind = compute_indicators(df)
        sr = compute_support_resistance(df)
        diag = macd_rsi_diagnosis(df, ind)
        drift = float(np.clip(diag.get("score", 0.0), -1, 1)) * 0.0015
        fc = simple_forecast(df, days=forecast_days, n_paths=80, drift_bias=drift)
        last = float(df["Close"].iloc[-1])
        med = float(fc["median"][-1])
        fc_pct = (med - last) / max(last, 1e-9)
        vol_ratio = float(df["Volume"].iloc[-1]) / max(float(ind["vol_ma20"].iloc[-1]), 1.0)
        # First-stage score: trend + forecast + liquidity confirmation.
        score = fc_pct * 3.0 + diag.get("score", 0.0) * 1.1 + np.clip((vol_ratio - 1.0) / 2.0, -0.4, 0.6)
        return {
            "code": sym.split(".")[0], "name": name, "quick_score": float(score),
            "last": last, "forecast_pct": fc_pct * 100,
            "diagnosis": diag.get("label", "中性"),
        }
    except Exception:
        return None


def recommend_landing_quick_picks(volume_limit: int = 100, top_n: int = 10,
                                  lookback_years: int = 1, forecast_days: int = 20,
                                  progress_callback=None) -> list:
    """
    Fast homepage picks using the same first-stage scoring as daily recommendations.

    The full daily recommendation panel still runs deeper analysis on finalists.
    This helper is intentionally lighter so the homepage does not feel stuck on
    first load.
    """
    codes = get_top_volume_stocks(volume_limit)
    total = len(codes)
    rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_quick_recommend_candidate, c, max(1, lookback_years), forecast_days): c for c in codes}
        done = 0
        for fut in as_completed(futs):
            done += 1
            if progress_callback:
                progress_callback(done, total, f"快速篩選 {futs[fut]}")
            item = fut.result()
            if item is None:
                continue
            forecast_pct = float(item.get("forecast_pct", 0.0) or 0.0)
            diagnosis = str(item.get("diagnosis") or "中性")
            item["score"] = float(item.get("quick_score", 0.0) or 0.0)
            item["reason"] = f"快速評分：{diagnosis}，{forecast_days}日預測 {forecast_pct:+.1f}%"
            rows.append(item)
    rows.sort(key=lambda x: x.get("score", x.get("quick_score", 0.0)), reverse=True)
    return rows[:top_n]


def recommend_top_volume_stocks(volume_limit: int = 100, top_n: int = 5,
                                lookback_years: int = 2, forecast_days: int = 20,
                                weights: dict = None, progress_callback=None,
                                fast_mode: bool = True,
                                finalist_count: int = 18) -> list:
    """
    Analyze top-volume Taiwan stocks and return recommended candidates.

    Speed strategy:
      1) scan ALL top-volume stocks with a lightweight technical/forecast pass;
      2) run full analysis only on the best finalists.
    This preserves the requirement of checking all top-volume stocks while avoiding
    100 full Yahoo/TWSE/news/model passes every time.
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    codes = get_top_volume_stocks(volume_limit)
    total = len(codes)

    if fast_mode:
        quick_rows = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(_quick_recommend_candidate, c, max(1, lookback_years), forecast_days): c for c in codes}
            done = 0
            for fut in as_completed(futs):
                done += 1
                if progress_callback:
                    progress_callback(done, total, f"快速掃描前 {volume_limit} 大成交量：{futs[fut]}")
                item = fut.result()
                if item is not None:
                    quick_rows.append(item)
        quick_rows.sort(key=lambda x: x["quick_score"], reverse=True)
        finalists = [x["code"] for x in quick_rows[:max(top_n, finalist_count)]]
    else:
        finalists = codes

    rows = []
    final_total = len(finalists)

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
            if r.get("margin", {}).get("margin_score", 0) > 0.15:
                reason.append("融資融券偏正面")
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
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(_one, c): c for c in finalists}
        done = 0
        for fut in as_completed(futs):
            done += 1
            if progress_callback:
                progress_callback(done, final_total, f"完整分析候選股：{futs[fut]}")
            item = fut.result()
            if item is not None:
                rows.append(item)
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows[:top_n]


# ═══════════════════════════════════════════════════════════════════════════
# Intraday 1-minute data + dynamic prediction
# ═══════════════════════════════════════════════════════════════════════════
def fetch_intraday_data(symbol: str, interval: str = "1m") -> tuple:
    """Return (resolved_symbol, name, intraday_df, source). Volume_lots is per-row lots."""
    if not YF_OK:
        raise RuntimeError("yfinance 未安裝")
    raw = _tw_code_key(symbol)
    s = str(symbol or "").strip().upper()
    if s.endswith((".TW", ".TWO")):
        candidates = [s]
    elif raw and re.fullmatch(r"\d{4,6}[A-Z]?", raw):
        candidates = [f"{raw}.TW", f"{raw}.TWO"]
    else:
        candidates = [s]
    errors = []
    for sym in candidates:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period="5d", interval=interval, auto_adjust=False, prepost=False)
            df = _normalize_history_df(df)
            if df.empty:
                errors.append(f"{sym}: empty")
                continue
            df.index = pd.to_datetime(df.index).floor("min")
            df = df[~df.index.duplicated(keep="last")]
            latest_date = df.index.max().date()
            start = pd.Timestamp.combine(latest_date, MARKET_OPEN)
            end = pd.Timestamp.combine(latest_date, MARKET_CLOSE)
            day = df[(df.index >= start) & (df.index <= end)].copy()
            if day.empty:
                day = df[df.index.date == latest_date].copy()
            if day.empty:
                errors.append(f"{sym}: no regular-session rows")
                continue
            day["Volume_lots"] = pd.to_numeric(day["Volume"], errors="coerce").fillna(0) / 1000.0
            code = sym.split(".")[0]
            name = resolve_tw_display_name(code, ticker=ticker, symbol=sym) if re.fullmatch(r"\d{4,6}[A-Z]?", code) else sym
            return sym, name, day, "Yahoo Finance 1m"
        except Exception as exc:
            errors.append(f"{sym}: {exc}")
    raise RuntimeError("查無盤中資料：" + "；".join(errors[-3:]))


def intraday_stats(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}
    op = float(pd.to_numeric(df["Open"], errors="coerce").dropna().iloc[0])
    latest = float(pd.to_numeric(df["Close"], errors="coerce").dropna().iloc[-1])
    return {
        "open": op,
        "latest": latest,
        "high": float(df["High"].max()),
        "low": float(df["Low"].min()),
        "change_pct": latest / op - 1.0 if op else 0.0,
        "volume_lots": float(df.get("Volume_lots", df["Volume"] / 1000.0).sum()),
        "last_time": df.index.max(),
    }


def make_intraday_prediction(intraday: pd.DataFrame, history: pd.DataFrame = None) -> pd.DataFrame:
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
    daily_mu = 0.0
    if history is not None and not history.empty:
        daily_mu = float(history["Close"].pct_change().tail(60).mean())
        if not np.isfinite(daily_mu):
            daily_mu = 0.0
    current_ret = last_price / open_price - 1.0 if open_price else 0.0
    elapsed = max((last_time - start).total_seconds(), 0)
    total = max((end - start).total_seconds(), 1)
    remain = max(0.0, 1.0 - elapsed / total)
    target_ret = current_ret + remain * (0.25 * current_ret + 0.50 * daily_mu)
    target_ret = float(np.clip(target_ret, current_ret - 0.04, current_ret + 0.04))
    target_price = open_price * (1.0 + target_ret)
    future_mask = full_index > last_time
    n_future = int(future_mask.sum())
    if n_future > 0:
        pred.loc[future_mask] = np.linspace(last_price, target_price, n_future + 1)[1:]
    return pd.DataFrame({"Time": full_index, "Predicted": pred.values, "Actual": actual.values})


def get_intraday_analysis(symbol: str, history: pd.DataFrame = None) -> dict:
    sym, name, df, source = fetch_intraday_data(symbol, "1m")
    pred = make_intraday_prediction(df, history)
    return {
        "symbol": sym,
        "name": name,
        "df": df,
        "prediction": pred,
        "stats": intraday_stats(df),
        "source": source,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main analysis pipeline (single function, no QThread needed)
# ═══════════════════════════════════════════════════════════════════════════
def run_analysis(symbol: str, lookback_years: int = 3,
                 forecast_days: int = 30, weights: dict = None,
                 train_ratio: float = 0.8,
                 progress_callback=None,
                 use_cache: bool = True,
                 light_news: bool = False,
                 api_key: str = "") -> dict:
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
        "api_key_set": bool(read_marketdata_key(api_key)),
    }, sort_keys=True, ensure_ascii=False)
    now = time.time()
    if use_cache and cache_key in _ANALYSIS_CACHE:
        ts, cached = _ANALYSIS_CACHE[cache_key]
        if now - ts < _ANALYSIS_CACHE_TTL:
            return cached

    def _prog(s, t, msg):
        if progress_callback:
            progress_callback(s, t, msg)

    TOTAL = 9
    try:
        # 1. Fetch
        _prog(1, TOTAL, f"抓取 {symbol} 資料…")
        sym, name, df = resolve_and_fetch(symbol, lookback_years, api_key=api_key)

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
                               ("earningsGrowth","earn_growth"),("totalRevenue","total_revenue"),
                               ("revenuePerShare","rev_per_share"),("profitMargins","profit_margin"),
                               ("grossMargins","gross_margin")]:
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
            fund["company_event_score"] = company_event_score(news, fund)

        mkt_ctx["_train_ratio"] = train_ratio
        mkt_ctx["_margin"] = margin

        # 4. ML
        _prog(4, TOTAL, "訓練 ML 模型（含美股/法人/融資融券特徵）…")
        horizon = min(forecast_days, 10)
        ml = MLPredictor(horizon=horizon)
        ml_stats = ml.train(df, ind, sr, mkt_ctx, inst)
        ml_pred  = ml.predict_current(df, ind, sr, mkt_ctx, inst) if "error" not in ml_stats else {}

        # 5. Future-return regression
        _prog(5, TOTAL, "訓練未來報酬率模型與分段模型…")
        return_horizon = max(1, min(int(forecast_days), 30))
        return_pred = predict_future_return_model(df, ind, sr, horizon=return_horizon, mkt_ctx=mkt_ctx, inst=inst)

        # 6. Forecast
        _prog(6, TOTAL, "Monte-Carlo 趨勢預測（由報酬率模型校準）…")
        drift = compute_drift_bias(ind, ml_pred, {}, ns, fund, weights, mkt_ctx, inst, margin)
        fc = simple_forecast(df, days=forecast_days, n_paths=180, drift_bias=drift)
        fc = align_forecast_with_return_model(fc, df, return_pred)

        # 7. Backtests
        _prog(7, TOTAL, "回測與 MACD 黃金交叉策略…")
        bt = backtest_directional(df, ind, sr, horizon=horizon)
        return_bt, return_detail = walk_forward_return_backtest(
            df, ind, sr, horizon=return_horizon, mkt_ctx=mkt_ctx, inst=inst)
        macd_bt = simulate_macd_cross_strategy(df, ind, hold_days=10, cross_type="golden")

        # 8. Reason
        _prog(8, TOTAL, "產生自動診斷文字…")
        # UI also has its own rich rendering; core provides stable text.
        reason_lines = [
            diag["text"],
            f"三大法人：合計買賣超 {inst.get('total_net',0)/1000:,.0f} 張，分數 {inst.get('inst_score',0):+.2f}。",
            f"融資融券：融資變化 {margin.get('margin_change',0):,.0f} 張、融券變化 {margin.get('short_change',0):,.0f} 張，分數 {margin.get('margin_score',0):+.2f}。",
            f"美股背景：NASDAQ {mkt_ctx.get('nasdaq_ret_1',0)*100:+.2f}%、S&P500 {mkt_ctx.get('sp500_ret_1',0)*100:+.2f}%、SMH {mkt_ctx.get('smh_ret_1',mkt_ctx.get('semis_ret_1',0))*100:+.2f}%、費半SOX {mkt_ctx.get('sox_ret_1',0)*100:+.2f}%。",
            f"新聞情緒：{ns.get('label','中性')}（正面 {ns.get('positive_ct',0)} / 負面 {ns.get('negative_ct',0)}）。",
            f"基本面/公司動能：P/E {fund.get('pe_ratio','—')}、EPS {fund.get('eps','—')}、營收成長 {fund.get('rev_growth',0)*100 if fund.get('rev_growth') is not None else 0:+.1f}%、殖利率 {fund.get('div_yield',0)*100 if fund.get('div_yield') is not None else 0:.2f}%、技術/訂單新聞分數 {fund.get('company_event_score',0):+.2f}。",
            f"報酬率模型：預測未來 {return_horizon} 日報酬 {return_pred.get('expected_return',0)*100:+.2f}%、分段={return_pred.get('segment','—')}（月內位置｜趨勢狀態｜波動狀態）、walk-forward 命中率 {return_bt.get('hit_rate_text','N/A')}。",
        ]

        # 9. Assemble
        _prog(9, TOTAL, "整理分析結果…")
        result = {
            "symbol": sym, "name": name, "df": df,
            "indicators": ind, "sr": sr,
            "forecast": fc, "forecast_days": forecast_days,
            "ml_stats": ml_stats, "ml_predict": ml_pred,
            "return_prediction": return_pred,
            "return_backtest": return_bt,
            "backtest_detail": return_detail,
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
