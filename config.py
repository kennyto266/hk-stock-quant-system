"""
港股量化分析系統 - 配置模組
包含所有系統配置類和常數定義
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('股票分析系統.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SystemConfig:
    """系統配置類"""
    # 數據獲取配置
    data_start_date: str = "2024-01-01"  # 數據開始日期（縮短時間範圍）
    data_end_date: str = "2024-06-21"    # 數據結束日期
    data_retry_attempts: int = 3         # 數據獲取重試次數
    data_retry_delay: int = 5           # 重試延遲（秒）
    
    # 止損止盈配置
    enable_stop_loss: bool = True
    stop_loss_pct: float = 0.05  # 5%止損
    enable_take_profit: bool = True
    take_profit_pct: float = 0.10  # 10%止盈
    trailing_stop: bool = True
    trailing_stop_pct: float = 0.03  # 3%追蹤止損
    
    # 風險管理配置
    max_position_size: float = 0.2  # 單策略最大倉位20%
    max_portfolio_risk: float = 0.15  # 組合最大風險15%
    correlation_threshold: float = 0.7  # 相關性閾值
    
    # 多線程配置
    max_workers: int = 8  # 最大線程數（發揮9950X3D性能）
    
    # 自動化配置
    auto_update_enabled: bool = True
    update_interval_hours: int = 4  # 4小時更新一次
    auto_report_enabled: bool = True
    report_time: str = "09:00"  # 每天9點生成報告
  # 每天9點生成報告
  # 每日9點生成報告

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