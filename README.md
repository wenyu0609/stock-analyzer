# Stock Analyzer Pro — Web 版

台灣股市智能分析系統，Streamlit 網頁版。手機、平板、電腦瀏覽器均可使用。

## 功能

- 📈 互動式 K 棒/曲線圖 + MACD + RSI
- 🤖 ML 模型（GradientBoosting + XGBoost）預測上漲機率  
- 🎯 Monte Carlo 趨勢預測（受分析權重影響）
- 📊 回測統計（含交易成本、Sharpe、最大回撤）
- 📰 即時新聞情緒分析
- ⚖ 分析權重設定（技術/ML/新聞/基本面佔比）
- 📋 批次分析 + CSV 匯出

---

## 部署到 Streamlit Cloud（免費，5分鐘完成）

### 步驟 1：上傳到 GitHub

1. 去 [github.com](https://github.com) 建立帳號（如果沒有）
2. 點右上角「+」→ New repository
3. Repository name: `stock-analyzer-pro`，選 Public，點 Create
4. 點「uploading an existing file」
5. 把以下三個檔案拖進去：
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
# 安裝套件
pip install streamlit yfinance pandas numpy plotly scikit-learn requests

# 執行
streamlit run app.py

# 瀏覽器開啟
# http://localhost:8501
```

---

## 注意事項

### Streamlit Cloud 免費版限制
- RAM：1 GB（足夠分析大部分股票）
- CPU：共享（分析速度比本機慢 2-3 倍）
- 閒置 30 分鐘後會進入休眠（再次開啟需要 1-2 分鐘喚醒）
- LSTM 神經網路需要 PyTorch，佔用較多記憶體

### 若遇到記憶體不足
在 `requirements.txt` 移除 `torch>=2.2.0`，LSTM 功能會停用，其他功能不受影響。

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
