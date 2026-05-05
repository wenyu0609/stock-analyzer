# Stock Analyzer Pro — Web 版

台灣股市智能分析系統，Streamlit 網頁版。手機、平板、電腦瀏覽器均可使用。

## 功能

- 📈 互動式 K 棒/曲線圖 + MACD + RSI
- 🤖 ML 模型（GradientBoosting + XGBoost）預測上漲機率
- 📐 報酬率模型：預測未來報酬率，再換算成預測價格路徑
- 🧩 分段訓練：依月內時段、趨勢、波動狀態建立 segment
- 🧪 Walk-forward 回測明細：預測報酬、實際報酬、segment、是否命中
- 🎯 Monte Carlo 趨勢預測，並用報酬率模型校準終點
- ⏱ 盤中走向：1 分鐘實際價格、1 分鐘逐列張數、預測至收盤
- 🔁 盤中資料支援 10 秒自動更新與手動更新
- 📊 回測統計（含交易成本、Sharpe、最大回撤）
- 📰 即時新聞情緒分析
- ⚖ 分析權重設定（技術/ML/新聞/基本面佔比）
- 📋 批次分析 + CSV 匯出
- 🔎 台股中文名稱解析：Yahoo 台股、TWSE、TPEX、興櫃官方資料源

---

## 部署到 Streamlit Cloud（免費，5分鐘完成）

### 步驟 1：上傳到 GitHub

1. 去 [github.com](https://github.com) 建立帳號（如果沒有）
2. 點右上角「+」→ New repository
3. Repository name: `stock-analyzer-pro`，選 Public，點 Create
4. 點「uploading an existing file」
5. 把以下四個檔案拖進去：
   - `README.md`
   - `app.py`
   - `core.py`  
   - `requirements.txt`
6. 點「Commit changes」

### 步驟 2：部署到 Streamlit Cloud

1. 去 [share.streamlit.io](https://share.streamlit.io)
2. 點「Sign in with GitHub」用你的 GitHub 帳號登入
3. 點「New app」
4. 選你的 repo：`stock-analyzer-pro`
5. Main file path：`app.py`
6. 點「Deploy!」
7. 等 2-3 分鐘後，你會得到：
   ```
   https://你的帳號名稱-stock-analyzer-pro-app-xxxxxx.streamlit.app
   ```

### 完成！

這個網址可以在任何裝置（手機、平板、電腦）上開啟，分享給任何人。

---

## 本地端執行（測試用）

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 注意事項

### Streamlit Cloud 免費版限制
- RAM：1 GB（足夠分析大部分股票）
- CPU：共享（分析速度比本機慢 2-3 倍）
- 閒置 30 分鐘後會進入休眠（再次開啟需要 1-2 分鐘喚醒）
- LSTM 神經網路需要 PyTorch，佔用較多記憶體

### API Key（選用）

沒有 API key 也可以使用，程式會走 Yahoo Finance / Yahoo 台股 / TWSE / TPEX 公開資料。

若有 Fugle 或 Fubon MarketData API key，可在 Streamlit Cloud 的 App settings -> Secrets 加入：

```toml
FUGLE_API_KEY = "你的 Fugle API key"
```

或在側邊欄的「Fugle/Fubon API Key」欄位手動輸入。

### 若遇到記憶體不足
目前網頁版沒有強制安裝 PyTorch，LSTM 不會佔用 Streamlit Cloud 記憶體；核心使用 GB/XGB 與報酬率模型。若免費版仍覺得慢，可先移除 `xgboost>=2.0.0`，系統會自動退回 GradientBoosting。

### 若需要更好的效能
考慮付費方案或自行租用 VPS（DigitalOcean、Linode，月費約 $6 USD）。

---

## 檔案結構

```
stock-analyzer-pro/
├── app.py              # Streamlit 主程式（UI 層）
├── core.py             # 後端引擎（分析邏輯，無 GUI 依賴）
├── requirements.txt    # 套件清單
└── README.md           # 本文件
```

## 成交量說明

盤中逐列張數以 Yahoo Finance 的 1 分鐘 `Volume` 欄位除以 1000 顯示，單位為「張」。若資料源當天最後只提供到 13:24，實際走向會停在 13:24；預測走向會繼續延伸到 13:30，不會把預測點誤標成實際成交。
