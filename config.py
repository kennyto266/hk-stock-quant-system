"""
港股量化分析系統 - 配置模組
包含所有系統配置類和常數定義
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import logging
import os
from datetime import datetime, timedelta

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemConfig:
    """配置類"""
    
    def __init__(self):
        # 數據時間範圍
        self.data_start_date = "2020-01-01"  # 開始日期
        self.data_end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")  # 結束日期（昨天）
        
        # 訓練參數
        self.train_test_split = 0.7  # 訓練集比例
        self.validation_days = 252  # 驗證天數（約一年）
        
        # 技術指標參數
        self.rsi_period = 14  # RSI週期
        self.rsi_overbought = 70  # RSI超買閾值
        self.rsi_oversold = 30  # RSI超賣閾值
        
        # 風險管理參數
        self.stop_loss = 0.05  # 止損比例
        self.take_profit = 0.10  # 止盈比例
        self.max_positions = 5  # 最大持倉數量
        
        # 資金管理參數
        self.initial_capital = 1000000  # 初始資金
        self.position_size = 0.2  # 每個倉位佔總資金比例
        
        # 數據存儲路徑
        self.data_dir = "stock_data"  # 數據存儲目錄
        self.output_dir = "output"  # 輸出目錄
        
        # 創建必要的目錄
        for directory in [self.data_dir, self.output_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

@dataclass
class NonPriceDataConfig:
    """非價格數據配置"""
    # 期貨數據配置
    enable_futures_data: bool = True
    futures_update_interval: int = 300  # 5分鐘更新
    futures_lookback_days: int = 30     # 期貨數據回看天數
    
    # 市場情緒配置
    enable_sentiment_analysis: bool = True
    sentiment_sources: List[str] = field(default_factory=lambda: ['news', 'social_media', 'options'])
    
    # 宏觀數據配置
    enable_macro_data: bool = True
    macro_indicators: List[str] = field(default_factory=lambda: ['interest_rate', 'exchange_rate', 'vix'])
    
    # 數據融合配置
    correlation_threshold: float = 0.3
    feature_selection_method: str = 'mutual_info'
    max_features: int = 50
    
    # 香港期貨配置
    hk_futures_symbols: Dict[str, str] = field(default_factory=lambda: {
        'HSI_DAY': '^HSI',      # 恆指日間期貨
        'HSI_NIGHT': 'HSI2300', # 恆指夜間期貨
        'HSCEI': '^HSCE',       # 國企指數期貨
        'HSTECH': '^HSTECH'     # 恆生科技指數期貨
    })
    
    # 股票數據批量下載配置
    hk_stocks_universe: Dict[str, str] = field(default_factory=lambda: {
        '2800.HK': '盈富基金',
        '0700.HK': '騰訊控股', 
        '0941.HK': '中國移動',
        '1299.HK': '友邦保險',
        '1398.HK': '工商銀行',
        '3988.HK': '中國銀行',
        '0005.HK': '匯豐控股',
        '1109.HK': '華潤置地',
        '2388.HK': '中銀香港',
        '3968.HK': '招商銀行',
        '2318.HK': '中國平安',
        '6862.HK': '海底撈',
        '9988.HK': '阿里巴巴',
        '9618.HK': '京東集團',
        '3690.HK': '美團',
        '1024.HK': '快手',
        '2269.HK': '藥明生物',
        '1810.HK': '小米集團',
        '9999.HK': '網易'
    })
    
    # 批量下載配置
    download_delay_seconds: float = 1.0
    default_lookback_days: int = 730  # 默認2年數據

@dataclass
class RiskConfig:
    """風險管理配置"""
    stop_loss_pct: float = 0.05  # 止損百分比 5%
    take_profit_pct: float = 0.10  # 止盈百分比 10%
    trailing_stop_pct: float = 0.03  # 追蹤止損 3%
    max_position_size: float = 0.3  # 最大單一持倉 30%
    max_daily_loss: float = 0.02  # 日最大損失 2%
    var_confidence: float = 0.05  # VaR置信度 5%
    
@dataclass
class BacktestConfig:
    """回測配置"""
    initial_capital: float = 1000000  # 初始資金 100萬
    commission_rate: float = 0.001  # 手續費率 0.1%
    min_holding_days: int = 1  # 最少持有天數
    max_holding_days: int = 20  # 最大持有天數
    rebalance_frequency: str = "monthly"  # 再平衡頻率

@dataclass  
class DisplayConfig:
    """顯示配置"""
    enable_interactive: bool = True  # 啟用互動功能
    auto_refresh: bool = True  # 自動刷新
    show_signals: bool = True  # 顯示交易信號
    detailed_logs: bool = True  # 詳細日志

@dataclass
class TechnicalConfig:
    """技術指標配置"""
    # RSI 配置
    rsi_period: int = 14
    rsi_overbought: float = 70
    rsi_oversold: float = 30
    
    # MACD 配置
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # 移動平均線配置
    ma_short: int = 5
    ma_medium: int = 20
    ma_long: int = 60
    
    # 布林通道配置
    bb_period: int = 20
    bb_std: float = 2.0
    
    # KDJ 配置
    kdj_period: int = 9
    kdj_smooth_k: int = 3
    kdj_smooth_d: int = 3

# 全域配置實例
CONFIG = SystemConfig()
RISK_CONFIG = RiskConfig()
BACKTEST_CONFIG = BacktestConfig()
DISPLAY_CONFIG = DisplayConfig()
TECHNICAL_CONFIG = TechnicalConfig()

NON_PRICE_CONFIG = NonPriceDataConfig()
# 套件可用性檢查
def check_package_availability():
    """檢查套件可用性"""
    availability = {}
    
    try:
        import schedule
        availability['SCHEDULE'] = True
    except ImportError:
        availability['SCHEDULE'] = False
        logger.warning("❌ schedule未安裝，跳過自動化功能")
    
    try:
        import dash
        from dash import dcc, html, dash_table
        import dash_bootstrap_components as dbc
        import plotly.graph_objects as go
        import plotly.express as px
        availability['DASH'] = True
    except ImportError:
        availability['DASH'] = False
        logger.warning("❌ Dash 套件未安裝，將跳過 Web 界面功能")
    
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
        availability['TKINTER'] = True
    except ImportError:
        availability['TKINTER'] = False
        logger.warning("❌ TKinter未安裝，跳過GUI功能")
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        availability['PLOTLY'] = True
    except ImportError:
        availability['PLOTLY'] = False
        logger.warning("❌ Plotly未安裝，跳過互動圖表功能")
    
    try:
        import quantstats as qs
        availability['QUANTSTATS'] = True
        logger.info("✅ QuantStats已載入，將提供專業量化分析")
    except ImportError:
        availability['QUANTSTATS'] = False
        logger.warning("❌ QuantStats未安裝，將跳過專業量化分析功能")
    
    # Plotguy功能已停用
    availability['PLOTGUY'] = False
    logger.info("ℹ️ Plotguy功能已停用，專注於內建可視化功能")
    
    return availability

# 檢查套件可用性
PACKAGE_AVAILABILITY = check_package_availability() 

# 創建必要的目錄
for directory in [CONFIG.data_dir, CONFIG.output_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory) 