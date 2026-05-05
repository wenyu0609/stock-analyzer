# 台股分析器 Streamlit 版

這是桌面版台股分析器的網頁化版本，預計部署到 Streamlit Cloud。

## 檔案

- `app.py`：Streamlit 網頁介面
- `core.py`：資料抓取、中文名稱解析、技術指標、報酬率模型、walk-forward 回測
- `requirements.txt`：部署依賴
- `README.md`：使用與部署說明

## 功能

- 支援台股上市、上櫃、興櫃、ETF、主動式 ETF 代碼
- 自動解析中文名稱，來源包含 Yahoo 台股、TWSE、TPEX 官方 open data
- 歷史價格走勢，支援曲線 / K棒切換
- 技術指標：均線、RSI、MACD、布林通道、ATR、量比
- ML 模型預測「未來報酬率」，再換算成價格路徑顯示
- 分段訓練：依月內時段、趨勢、波動狀態切 segment
- Walk-forward backtest 明細：預測報酬、實際報酬、segment、是否命中
- 盤中 1 分鐘走勢、1 分鐘逐列張數、盤中預測路徑
- 盤中資料可 10 秒自動更新，也可手動清除快取更新

## 本機執行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 部署到 Streamlit Cloud

1. 到 GitHub 建立一個 repository。
2. 上傳這四個檔案：
   - `README.md`
   - `app.py`
   - `core.py`
   - `requirements.txt`
3. 到 Streamlit Cloud 新增 app。
4. Repository 選你的 GitHub repo。
5. Main file path 填：

```text
app.py
```

6. Deploy。

## API Key（選用）

若有 Fugle MarketData API key，可到 Streamlit Cloud 的 App settings -> Secrets 加入：

```toml
FUGLE_API_KEY = "你的 Fugle API key"
```

沒有 API key 也能執行，程式會退回 Yahoo Finance / Yahoo 台股 / TWSE / TPEX 公開資料源。

## 注意

公開資料源可能會延遲、限流或短暫改版。盤中 1 分鐘資料主要來自 Yahoo Finance，成交量欄位以「股」轉「張」顯示，也就是 `Volume / 1000`。

本工具僅供研究與風險評估，不構成投資建議。
