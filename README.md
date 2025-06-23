# 港股量化分析系統

## 📋 項目概述

這是一個專為港股市場設計的量化分析系統，特別專注於 2800.HK（盈富基金）的策略回測與分析。系統使用 Plotly Dash 框架提供現代化的網頁儀表板體驗，所有結果都通過 localhost 網頁應用展示。

## 🚀 快速開始

### 1. 安裝依賴
```bash
 
```

### 2. 運行系統
```bash

```

### 3. 自動運行
系統會自動執行：
- **智能RSI策略優化**: 三階段優化分析
- **網頁儀表板**: 自動啟動 (http://localhost:8050)
- **數據分析**: CSV報告和策略結果
- **不再生成 HTML 檔案**: 所有結果在網頁儀表板中查看

## 📁 文件結構

```
港股量化分析系統/
├── run_analysis.py              # 🚀 主運行文件
├── requirements.txt            # 📦 依賴清單
├── README.md                   # 📖 說明文檔
│
├── 2800HK港股分析_完全修復版.py  # 🔧 完整版分析程序
├── main_refactored.py          # 🔄 重構版主程序
│
├── 模組化組件/
│   ├── config.py              # ⚙️  配置管理
│   ├── data_handler.py        # 📊 數據處理
│   ├── strategy_manager.py    # 📈 策略管理
│   ├── risk_management.py     # 🛡️  風險管理
│   └── visualization.py       # 📊 視覺化
│
├── Dashboard 相關/
│   ├── plotly_dashboard.py    # 🌐 Plotly Dashboard
│   ├── test_dashboard.py      # 🧪 Dashboard 測試
│   └── README_Plotly_Dashboard.md  # 📖 Dashboard 說明
│
└── 輸出文件/
    ├── data_output/           # 📊 分析數據輸出
    │   ├── csv/              # 📄 CSV報告文件
    │   └── reports/          # 📋 策略結果報告
    ├── *.png                  # 📈 圖表文件
    └── 港股輸出/              # 📊 港股專用輸出
```

## 🎯 主要功能

### 📊 數據分析
- **多源數據整合**: Yahoo Finance, AKShare, 港交所數據
- **北水數據**: 南北水資金流向分析
- **技術指標**: RSI, MACD, 布林帶等完整技術分析

### 📈 策略回測
- **多策略支持**: RSI、MACD、布林帶、綜合策略
- **風險管理**: 止損止盈、追蹤止損、倉位管理
- **績效分析**: 夏普比率、最大回撤、勝率統計

### 🌐 互動式 Dashboard
- **實時圖表**: 價格走勢、技術指標、交易信號
- **策略對比**: 多策略績效比較分析
- **響應式設計**: 現代化網頁界面

### 📋 報告生成
- **Dash 網頁應用**: 即時互動式分析儀表板
- **CSV 數據**: 詳細的策略優化結果和交易記錄
- **文字報告**: 完整的策略分析報告

## ⚙️ 配置選項

### 風險管理配置
```python
# 止損止盈設置
stop_loss_pct = 0.05      # 5% 止損
take_profit_pct = 0.10    # 10% 止盈
trailing_stop_pct = 0.03  # 3% 追蹤止損

# 倉位管理
max_position_size = 0.3   # 最大單一持倉 30%
max_daily_loss = 0.02     # 日最大損失 2%
```

### 回測配置
```python
# 回測參數
initial_capital = 1000000  # 初始資金 100萬
commission_rate = 0.001    # 手續費率 0.1%
rebalance_frequency = "monthly"  # 再平衡頻率
```

## 🔧 技術架構

### 核心技術棧
- **數據處理**: Pandas, NumPy
- **網頁框架**: Dash, Plotly
- **技術分析**: TA-Lib
- **統計分析**: SciPy, Scikit-learn
- **投資組合**: PyFolio

### 系統特色
- **模組化設計**: 可擴展的組件架構
- **多線程支持**: 充分利用多核性能
- **自動化運行**: 定時數據更新和報告生成
- **錯誤處理**: 完善的異常處理機制

## 📈 使用案例

### 1. 一鍵自動分析
```bash
# 運行智能RSI策略優化
python run_analysis.py
# 系統會自動執行所有分析並啟動網頁儀表板
```

### 2. 實時監控
```bash
# 系統會自動啟動 Dash 應用
# 訪問 http://localhost:8050
# 查看即時分析結果和互動圖表
```

### 3. 策略結果
```bash
# 檢查 data_output/ 目錄
# CSV 檔案包含詳細的策略優化結果
# 文字報告包含完整的分析總結
```

## 🛠️ 故障排除

### 常見問題

1. **依賴安裝失敗**
   ```bash
   # 升級 pip
   python -m pip install --upgrade pip
   # 重新安裝依賴
   pip install -r requirements.txt
   ```

2. **TA-Lib 安裝問題**
   ```bash
   # Windows 用戶
   pip install TA-Lib-0.4.25-cp39-cp39-win_amd64.whl
   ```

3. **Dashboard 無法訪問**
   - 檢查端口 8051 是否被占用
   - 確認防火牆設置
   - 嘗試使用 127.0.0.1:8051

### 性能優化

- **多線程配置**: 根據 CPU 核心數調整 `max_workers`
- **內存管理**: 大數據集分批處理
- **緩存策略**: 啟用數據緩存加速重複分析

## 📞 技術支持

如遇到問題，請檢查：
1. Python 版本 >= 3.8
2. 所有依賴項目已正確安裝
3. 網絡連接正常（數據獲取需要）
4. 系統資源充足（內存 >= 4GB 推薦）

## 🔄 更新日誌

### v2.1.0 (最新)
- ✅ 移除 HTML 檔案生成，純網頁應用體驗
- ✅ 智能RSI策略三階段優化
- ✅ 自動化一鍵運行模式
- ✅ 優化 Dash 儀表板性能
- ✅ 簡化用戶操作流程

### v1.0.0
- 🎯 基礎量化分析功能
- 📊 多策略回測系統
- 📈 基本圖表生成

---

🏦 **港股量化分析系統** - 專業的港股投資分析工具 