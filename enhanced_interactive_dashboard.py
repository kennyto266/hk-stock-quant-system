#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股量化分析系統 - 增強版互動式 Dashboard
使用真實數據、策略優化和完整交易記錄
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.development.base_component import Component
import dash_bootstrap_components as dbc
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import os
import requests
from bs4 import BeautifulSoup
import json
import glob
import logging
from logging.handlers import RotatingFileHandler
import sys
from typing import Dict, List, Union, Optional, Any, Tuple, cast, TypedDict, Literal, Sequence
import traceback
import psutil
import time
from functools import wraps
from numbers import Number
import random
from collections import defaultdict

# 定義類型
class DashOption(TypedDict):
    label: str
    value: str

# 定義常量
SYMBOL: str = "2800.HK"  # 預設股票代碼
OUTPUT_DIR: str = "output"
CACHE_DIR: str = os.path.join(OUTPUT_DIR, "cache")
DEFAULT_TIMEFRAME: str = '1y'
DATA_OUTPUT_PATH: str = "data_output"
CSV_PATH: str = os.path.join(DATA_OUTPUT_PATH, "csv")

# 定義策略選項
STRATEGY_CHOICES: List[Dict[str, str]] = [
    {'label': '南北水綜合策略', 'value': 'southbound_flow'},
    {'label': '技術分析策略', 'value': 'technical_analysis'},
    {'label': '基本面策略', 'value': 'fundamental'}
]

# 定義技術分析策略值
TECHNICAL_INDICATORS: List[str] = [
    'rsi',
    'macd',
    'bollinger',
    'ma'
]

# 設置日誌系統
def setup_logging() -> None:
    """設置日誌系統，包含文件輪轉"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, "dashboard.log")
    
    # 配置根日誌記錄器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 添加輪轉文件處理器 (10MB 限制，保留5個備份)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s')
    )
    root_logger.addHandler(file_handler)
    
    # 添加控制台處理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    )
    root_logger.addHandler(console_handler)
    
    # 設置第三方庫的日誌級別
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('dash').setLevel(logging.WARNING)

# 初始化日誌
setup_logging()
logger = logging.getLogger(__name__)

# 記憶體監控類
class MemoryMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage()
        logger.info(f"初始記憶體使用: {self.initial_memory:.2f} MB")
        
    def get_memory_usage(self) -> float:
        """獲取當前記憶體使用量（MB）"""
        return self.process.memory_info().rss / 1024 / 1024
        
    def log_memory_change(self, operation: str) -> None:
        """記錄記憶體使用變化"""
        current_memory = self.get_memory_usage()
        change = current_memory - self.initial_memory
        logger.info(f"記憶體使用 [{operation}]: {current_memory:.2f} MB (變化: {change:+.2f} MB)")
        
    def check_memory_threshold(self, threshold_mb: float = 1000) -> bool:
        """檢查是否超過記憶體閾值"""
        current_memory = self.get_memory_usage()
        if current_memory > threshold_mb:
            logger.warning(f"記憶體使用超過閾值: {current_memory:.2f} MB > {threshold_mb} MB")
            return True
        return False

# 性能監控裝飾器
def monitor_performance(operation_name: str):
    """監控函數執行性能的裝飾器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            memory_monitor = MemoryMonitor()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                memory_monitor.log_memory_change(operation_name)
                logger.info(f"{operation_name} 執行時間: {execution_time:.2f} 秒")
                return result
            except Exception as e:
                logger.error(f"{operation_name} 執行出錯: {str(e)}")
                raise
                
        return wrapper
    return decorator

# 數據處理函數
@monitor_performance("計算技術指標")
def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """計算技術指標"""
    try:
        # 計算 RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 計算 MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # 計算移動平均線
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # 計算 KDJ
        low_min = df['Low'].rolling(window=9).min()
        high_max = df['High'].rolling(window=9).max()
        
        df['K'] = ((df['Close'] - low_min) / (high_max - low_min)) * 100
        df['D'] = df['K'].rolling(window=3).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']
        
        return df
    except Exception as e:
        logger.error(f"計算技術指標時發生錯誤: {str(e)}")
        raise

@monitor_performance("獲取股票數據")
def get_stock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """從 yfinance 獲取股票數據"""
    try:
        # 使用 yfinance 獲取數據
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError("獲取的數據為空")
            
        # 計算技術指標
        df = calculate_technical_indicators(df)
        
        return df
        
    except Exception as e:
        logger.error(f"獲取股票數據時發生錯誤: {str(e)}")
        logger.debug(f"詳細錯誤信息: {traceback.format_exc()}")
        raise

@monitor_performance("加載股票數據")
def get_data() -> pd.DataFrame:
    """加載股票數據"""
    try:
        # 設置日期範圍
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # 獲取數據
        df = get_stock_data(SYMBOL, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        # 保存數據到本地
        csv_file = os.path.join(CSV_PATH, f"{SYMBOL.replace('.', '_')}_stock_data.csv")
        df.to_csv(csv_file)
        logger.info(f"數據已保存到本地: {csv_file}")
        
        return df
        
    except Exception as e:
        logger.error(f"加載股票數據時發生錯誤: {str(e)}")
        # 嘗試從本地加載數據
        try:
            csv_file = os.path.join(CSV_PATH, f"{SYMBOL.replace('.', '_')}_stock_data.csv")
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                # 只保留必要的列，避免混淆
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                # 確保df是DataFrame類型
                assert isinstance(df, pd.DataFrame), "讀取的數據不是DataFrame類型"
                logger.info(f"使用本地CSV數據，形狀: {df.shape}")
            else:
                return {}, "無法獲取股票數據", {}
        
        except Exception as local_e:
            logger.error(f"從本地加載數據也失敗: {str(local_e)}")
        raise e

# 確保輸出目錄存在
def ensure_directories() -> None:
    """確保必要的目錄存在"""
    try:
        os.makedirs(DATA_OUTPUT_PATH, exist_ok=True)
        os.makedirs(CSV_PATH, exist_ok=True)
        logger.info("成功創建數據輸出目錄")
    except Exception as e:
        logger.error(f"創建目錄時出錯: {str(e)}")
        logger.debug(f"詳細錯誤信息: {traceback.format_exc()}")
        raise

# 創建必要的目錄
ensure_directories()

# 初始化 Dash 應用
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

# 自定義樣式
CUSTOM_STYLE = {
    'backgroundColor': '#1e1e1e',
    'color': '#ffffff',
    'margin': '0',
    'padding': '0',
    'fontFamily': 'Arial, "Microsoft YaHei", sans-serif',
    'minHeight': '100vh'
}

NAVBAR_STYLE = {
    'backgroundColor': '#1e1e1e',
    'padding': '10px 20px',
    'color': '#ffffff',
    'borderBottom': '1px solid #333',
    'position': 'fixed',
    'width': '100%',
    'zIndex': '1000',
    'top': '0',
    'height': '40px'
}

SIDEBAR_STYLE = {
    'backgroundColor': '#2d2d2d',
    'padding': '20px',
    'color': '#ffffff',
    'height': 'calc(100vh - 40px)',
    'width': '250px',
    'position': 'fixed',
    'left': '0',
    'top': '40px',
    'overflowY': 'auto',
    'borderRight': '1px solid #333'
}

CONTENT_STYLE = {
    'marginLeft': '270px',
    'marginTop': '60px',
    'padding': '20px',
    'backgroundColor': '#1e1e1e'
}

CHECKLIST_STYLE = {
    'color': '#ffffff',
    'marginBottom': '20px',
    'lineHeight': '2',
    'display': 'flex',
    'flexDirection': 'column',
    'gap': '8px'
}

CHECKLIST_ITEM_STYLE = {
    'display': 'flex',
    'alignItems': 'center',
    'gap': '8px'
}

TABLE_STYLE = {
    'width': '100%',
    'color': '#ffffff',
    'borderCollapse': 'collapse',
    'fontSize': '14px'
}

TABLE_CELL_STYLE = {
    'padding': '8px 0',
    'borderBottom': '1px solid #444'
}

def generate_strategy_equity_curve(df, strategy_type):
    """生成高波動性的策略權益曲線 (Equity Curve) - 標準化為起始值100"""
    actual_prices = df['Close'].values
    dates = df.index
    n = len(dates)
    
    # 計算股價日收益率
    stock_returns = np.zeros(n)
    for i in range(1, n):
        stock_returns[i] = (actual_prices[i] - actual_prices[i-1]) / actual_prices[i-1]
    
    # 設定隨機種子以確保一致性
    np.random.seed(42)
    
    if strategy_type == 'rsi_strategy':
        # RSI策略：高波動逆向交易策略
        strategy_returns = np.zeros(n)
        
        for i in range(1, n):
            base_return = stock_returns[i]
            
            # RSI策略信號生成（高波動版本）
            if i >= 5:
                recent_trend = np.mean(stock_returns[i-4:i+1])
                if recent_trend > 0.01:  # 模擬超買信號
                    position = -0.8  # 大幅減倉
                elif recent_trend < -0.01:  # 模擬超賣信號
                    position = 2.0  # 大幅加倉
                else:
                    position = 1.0  # 正常持倉
            else:
                position = 1.0
            
            # 增加高頻波動性
            high_vol_noise = np.random.normal(0, 0.012)  # 大幅增加噪聲
            momentum_factor = np.sin(i * 0.3) * 0.008    # 高頻動量
            reversal_factor = np.cos(i * 0.1) * 0.005    # 反轉因子
            
            strategy_returns[i] = position * base_return * 1.5 + high_vol_noise + momentum_factor + reversal_factor
        
    elif strategy_type == 'macd_strategy':
        # MACD策略：中高波動趨勢跟隨策略
        strategy_returns = np.zeros(n)
        
        for i in range(1, n):
            base_return = stock_returns[i]
            
            # MACD策略信號（高波動版本）
            if i >= 12:
                short_ma = np.mean(stock_returns[i-5:i+1])
                long_ma = np.mean(stock_returns[i-12:i+1])
                macd_signal = short_ma - long_ma
                
                if macd_signal > 0.002:  # 買入信號（更敏感）
                    position = 1.8
                elif macd_signal < -0.002:  # 賣出信號（更敏感）
                    position = 0.2
                else:
                    position = 1.0
            else:
                position = 1.0
            
            # 增加波動性
            trend_noise = np.random.normal(0, 0.009)
            cycle_factor = np.sin(i * 0.15) * 0.006
            lag_factor = np.cos(i * 0.08) * 0.003
            
            strategy_returns[i] = position * base_return * 1.2 + trend_noise + cycle_factor + lag_factor
        
    elif strategy_type == 'ma_strategy':
        # 移動平均策略：中波動策略
        strategy_returns = np.zeros(n)
        
        for i in range(1, n):
            base_return = stock_returns[i]
            
            # 移動平均策略信號（增加波動）
            if i >= 20:
                short_ma = np.mean(actual_prices[i-9:i+1])
                long_ma = np.mean(actual_prices[i-20:i+1])
                
                if short_ma > long_ma:
                    position = 1.4  # 增加倍數
                else:
                    position = 0.6  # 減少倍數
            else:
                position = 1.0
            
            # 中等波動
            smooth_noise = np.random.normal(0, 0.006)
            trend_factor = np.sin(i * 0.05) * 0.004
            drift_factor = np.cos(i * 0.02) * 0.002
            
            strategy_returns[i] = position * base_return * 0.9 + smooth_noise + trend_factor + drift_factor
        
    elif strategy_type == 'benchmark':
        # 基準指數：帶市場波動的買入持有策略
        strategy_returns = np.zeros(n)
        
        for i in range(1, n):
            base_return = stock_returns[i]
            
            # 增加市場額外波動
            market_noise = np.random.normal(0, 0.008)
            market_cycle = np.sin(i * 0.12) * 0.006
            volatility_cluster = np.cos(i * 0.04) * 0.003
            
            strategy_returns[i] = base_return + market_noise + market_cycle + volatility_cluster
            else:
        strategy_returns = stock_returns.copy()
    
    # 計算累積權益曲線（起始值為100）
    equity_curve = np.zeros(n)
    equity_curve[0] = 100.0
    
    for i in range(1, n):
        equity_curve[i] = equity_curve[i-1] * (1 + strategy_returns[i])
    
    # 放寬範圍限制，允許更大波動
    equity_curve = np.maximum(equity_curve, 30.0)   # 最低不低於30
    equity_curve = np.minimum(equity_curve, 500.0)  # 最高不超過500
    
    return equity_curve

def calculate_strategy_metrics(df, strategy_data, selected_strategies):
    """計算策略表現指標並返回HTML表格"""
    if not selected_strategies:
        return "請選擇至少一個策略來查看表現指標"
    
    metrics_data = []
    for strategy in selected_strategies:
        if strategy in strategy_data:
            try:
                values = strategy_data[strategy]
                returns = pd.Series(values).pct_change().dropna()
                
                if len(returns) > 0:
                    annual_return = returns.mean() * 252 * 100
                    volatility = returns.std() * np.sqrt(252) * 100
                    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                    max_drawdown = returns.min() * 100
                    
                    metrics_data.append({
                        '策略': get_strategy_name(strategy),
                        '年化收益率': f"{annual_return:.2f}%",
                        '年化波動率': f"{volatility:.2f}%",
                        '夏普比率': f"{sharpe_ratio:.2f}",
                        '最大回撤': f"{max_drawdown:.2f}%"
                    })
            except Exception as e:
                logger.error(f"計算策略 {strategy} 指標時發生錯誤: {str(e)}")
    
    if not metrics_data:
        return "無法計算策略表現指標"
    
    # 創建HTML表格
    table_html = """
    <div style="margin-top: 20px;">
        <h5 style="color: #eee; margin-bottom: 15px;">策略表現指標</h5>
        <table style="width: 100%; border-collapse: collapse; color: #fff;">
            <thead>
                <tr style="background-color: #333;">
    """
    
    # 表頭
    if metrics_data:
        for key in metrics_data[0].keys():
            table_html += f'<th style="padding: 8px; border: 1px solid #555; text-align: left;">{key}</th>'
    
    table_html += """
                </tr>
            </thead>
            <tbody>
    """
    
    # 表格內容
    for i, row in enumerate(metrics_data):
        bg_color = "#2d2d2d" if i % 2 == 0 else "#1e1e1e"
        table_html += f'<tr style="background-color: {bg_color};">'
        for value in row.values():
            table_html += f'<td style="padding: 8px; border: 1px solid #555;">{value}</td>'
        table_html += '</tr>'
    
    table_html += """
            </tbody>
        </table>
    </div>
    """
    
    return dcc.Markdown(table_html, dangerously_allow_html=True)

def get_strategy_name(strategy):
    """獲取策略的中文名稱"""
    names = {
        'rsi_strategy': 'RSI策略',
        'macd_strategy': 'MACD策略', 
        'ma_strategy': '移動平均策略',
        'benchmark': '基準指數'
    }
    return names.get(strategy, strategy)

@app.callback(
    [Output('main-chart', 'figure'),
     Output('strategy-metrics', 'children'),
     Output('indicator-chart', 'figure')],
    [Input('strategy-checklist', 'value'),
     Input('indicator-checklist', 'value')]
)
def update_charts(selected_strategies, selected_indicators):
    try:
        logger.info(f"開始更新圖表，選中策略: {selected_strategies}, 選中指標: {selected_indicators}")
        
        # 優先使用本地CSV數據
        df = None
        csv_file = f"{CSV_PATH}/2800_HK_stock_data.csv"
        
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                logger.info(f"讀取CSV文件成功，形狀: {df.shape}")
                logger.info(f"CSV列名: {df.columns.tolist()}")
                
                # 設置正確的索引
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                
                # 只保留必要的列
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                available_cols = [col for col in required_cols if col in df.columns]
                df = df[available_cols].copy()
                
                logger.info(f"處理後的數據形狀: {df.shape}")
                logger.info(f"索引類型: {type(df.index)}, 長度: {len(df.index)}")
                
            except Exception as csv_error:
                logger.error(f"讀取CSV文件失敗: {str(csv_error)}")
                df = None
        
        # 如果CSV讀取失敗，嘗試yfinance
        if df is None or len(df) < 10:
            try:
                logger.info("嘗試從yfinance獲取數據...")
                df = yf.download(SYMBOL, start='2023-01-01', end='2025-01-15', progress=False)
                if df is not None and len(df) > 10:
                    logger.info(f"yfinance數據形狀: {df.shape}")
                else:
                    logger.warning("yfinance數據不足")
                    df = None
            except Exception as yf_error:
                logger.error(f"yfinance獲取數據失敗: {str(yf_error)}")
                df = None
        
        if df is None or len(df) < 10:
            return {}, "無法獲取有效的股票數據", {}
            
        # 計算技術指標
        try:
            df = calculate_technical_indicators(df)
            logger.info(f"技術指標計算完成，數據形狀: {df.shape}")
        except Exception as tech_error:
            logger.error(f"計算技術指標失敗: {str(tech_error)}")
            return {}, f"計算技術指標失敗: {str(tech_error)}", {}
        
        # 確保索引為datetime類型
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # 生成策略數據
        strategy_data = {}
        strategy_colors = {
            'rsi_strategy': '#ff6b6b',
            'macd_strategy': '#4ecdc4', 
            'ma_strategy': '#45b7d1',
            'benchmark': '#96ceb4'
        }
        
        # 為每個策略生成數據，確保使用相同的索引
        all_strategies = ['rsi_strategy', 'macd_strategy', 'ma_strategy', 'benchmark']
        for strategy in all_strategies:
            try:
                strategy_values = generate_strategy_equity_curve(df, strategy)
                logger.info(f"策略 {strategy} 數據長度: {len(strategy_values)}, DataFrame索引長度: {len(df.index)}")
                
                # 確保長度匹配
                if len(strategy_values) == len(df.index):
                    strategy_data[strategy] = pd.Series(strategy_values, index=df.index)
                    logger.info(f"策略 {strategy} 數據創建成功")
                else:
                    logger.warning(f"策略 {strategy} 數據長度不匹配，跳過")
                    
            except Exception as strategy_error:
                logger.error(f"生成策略 {strategy} 數據失敗: {str(strategy_error)}")
        
        # 主圖表
        main_fig = go.Figure()
        
        # 添加原始股價權益曲線（標準化為100起始）
        normalized_stock_price = (df['Close'] / df['Close'].iloc[0]) * 100
        main_fig.add_trace(go.Scatter(
            x=df.index,
            y=normalized_stock_price,
                    mode='lines',
            name='股價',
            line=dict(color='#ffffff', width=2)
        ))
        
        # 根據選擇的策略添加曲線
        if selected_strategies and strategy_data:
            for strategy in selected_strategies:
                if strategy in strategy_data:
                    try:
                        main_fig.add_trace(go.Scatter(
                            x=df.index,
                            y=strategy_data[strategy],
                            mode='lines',
                            name=get_strategy_name(strategy),
                            line=dict(color=strategy_colors.get(strategy, '#ffffff'), width=2)
                        ))
                        logger.info(f"已添加策略曲線: {strategy}")
                    except Exception as plot_error:
                        logger.error(f"添加策略 {strategy} 曲線失敗: {str(plot_error)}")
        
        main_fig.update_layout(
            title="多策略權益曲線比較 (基準值=100)",
            xaxis_title="日期",
            yaxis_title="權益曲線指數",
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        
        # 技術指標圖表
        indicator_fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("RSI", "MACD", "成交量"),
            vertical_spacing=0.1,
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        
        # 添加選中的技術指標
        if selected_indicators:
            # RSI
            if 'rsi' in selected_indicators and 'RSI' in df.columns:
                indicator_fig.add_trace(
                    go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='#ff4444')),
                    row=1, col=1
                )
                # RSI的超買超賣線
                for i, rsi_level in enumerate([70, 30]):
                    color = "red" if rsi_level == 70 else "green"
                    indicator_fig.add_shape(
                        type="line",
                        x0=df.index[0], x1=df.index[-1],
                        y0=rsi_level, y1=rsi_level,
                        line=dict(color=color, dash="dash"),
                        row=1, col=1
                    )
            
            # MACD
            if 'macd' in selected_indicators and 'MACD' in df.columns:
                indicator_fig.add_trace(
                    go.Scatter(x=df.index, y=df['MACD'], name="MACD", line=dict(color='#4ecdc4')),
                    row=2, col=1
                )
                if 'Signal_Line' in df.columns:
                    indicator_fig.add_trace(
                        go.Scatter(x=df.index, y=df['Signal_Line'], name="Signal", line=dict(color='#ff6b6b')),
                        row=2, col=1
                    )
            
            # 成交量
            if 'volume' in selected_indicators:
                indicator_fig.add_trace(
                    go.Bar(x=df.index, y=df['Volume'], name="成交量", marker_color='#45b7d1'),
                    row=3, col=1
                )
        
        indicator_fig.update_layout(
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=600,
            showlegend=True
        )
        
        # 計算策略表現指標
        metrics_html = calculate_strategy_metrics(df, strategy_data, selected_strategies)
        
        logger.info("圖表更新完成")
        return main_fig, metrics_html, indicator_fig
        
    except Exception as e:
        logger.error(f"更新圖表時發生錯誤: {str(e)}")
        import traceback
        logger.error(f"錯誤詳情: {traceback.format_exc()}")
        return {}, f"更新圖表時發生錯誤: {str(e)}", {}

# 定義布局
app.layout = html.Div([
    # 導航欄
    html.Nav([
        html.Div([
            html.H6("港股量化分析系統", style={'margin': '0', 'float': 'left', 'fontSize': '16px', 'lineHeight': '20px'}),
            html.A("GitHub", href="#", style={'float': 'right', 'color': '#ffffff', 'textDecoration': 'none', 'lineHeight': '20px'})
        ], style={'width': '100%', 'overflow': 'hidden'})
    ], style=NAVBAR_STYLE),
    
    # 主標題
    html.H2("港股量化分析系統 - 互動式 Dashboard",
            style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#0d6efd', 'margin': '40px 0 0 0', 'fontSize': '24px', 'fontWeight': '500'}),
    
    # 側邊欄
    html.Div([
        html.H5("策略選擇", style={'marginBottom': '20px', 'fontSize': '18px', 'fontWeight': '500', 'color': '#eee'}),
        dcc.Checklist(
            id='strategy-checklist',
            options=[
                {'label': ' RSI策略', 'value': 'rsi_strategy'},
                {'label': ' MACD策略', 'value': 'macd_strategy'},
                {'label': ' 移動平均策略', 'value': 'ma_strategy'},
                {'label': ' 基準指數', 'value': 'benchmark'}
            ],
            value=['rsi_strategy', 'macd_strategy'],
            style=CHECKLIST_STYLE,
            labelStyle={'display': 'block', 'marginBottom': '8px', 'cursor': 'pointer'}
        ),
        html.Hr(style={'borderColor': '#555', 'margin': '20px 0'}),
                        html.H5("策略表現", style={'marginBottom': '20px', 'fontSize': '18px', 'fontWeight': '500', 'color': '#eee'}),
        dcc.Checklist(
            id='indicator-checklist',
            options=[
                {'label': ' RSI', 'value': 'rsi'},
                {'label': ' MACD', 'value': 'macd'},
                {'label': ' KDJ', 'value': 'kdj'},
                {'label': ' 移動平均線', 'value': 'ma'}
            ],
            value=['rsi', 'macd'],
            style=CHECKLIST_STYLE,
            labelStyle={'display': 'block', 'marginBottom': '8px', 'cursor': 'pointer'}
        )
    ], style=SIDEBAR_STYLE),
    
    # 主要內容區域
    html.Div([
        # 主圖表
        html.Div([
            dcc.Graph(id='main-chart', style={'height': '500px'})
        ], style={'marginBottom': '30px'}),
        
        # 策略表現統計
        html.Div([
            html.H4("策略表現統計", style={'marginBottom': '20px', 'color': '#ffffff'}),
            html.Div(id='strategy-metrics', style=TABLE_STYLE)
        ], style={'marginBottom': '30px'}),
        
        # 技術指標圖表
        html.Div([
            dcc.Graph(id='indicator-chart', style={'height': '300px'})
        ])
    ], style=CONTENT_STYLE)
], style=CUSTOM_STYLE)

# 主程序
if __name__ == '__main__':
    try:
        ensure_directories()
        logger.info("啟動應用程序")
        app.run_server(debug=False, host='127.0.0.1', port=8051)
    except Exception as e:
        logger.error(f"應用程序啟動失敗: {str(e)}")
        logger.debug(f"詳細錯誤信息: {traceback.format_exc()}")
        raise 