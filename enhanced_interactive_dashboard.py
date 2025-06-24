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
from visualization import create_interactive_dashboard

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
                return df
            else:
                raise RuntimeError("無法獲取股票數據")
        except Exception as local_e:
            logger.error(f"從本地加載數據也失敗: {str(local_e)}")
            raise RuntimeError("無法獲取股票數據") from local_e

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
    """
    根據信號與價格計算真實權益曲線
    """
    try:
        # 假設信號已經在 df 中，欄位名為 f'{strategy_type}_signal'
        signal_col = f'{strategy_type}_signal'
        if signal_col not in df.columns or 'Close' not in df.columns:
            logger.error(f"缺少必要欄位: {signal_col} 或 Close")
            return np.full(len(df), 100.0)
        
        signals = df[signal_col]
        price = df['Close']
        returns = price.pct_change().fillna(0)
        strategy_returns = signals.shift(1).fillna(0) * returns
        equity_curve = (1 + strategy_returns).cumprod() * 100
        return equity_curve.values
    except Exception as e:
        logger.error(f"生成權益曲線失敗: {e}")
        return np.full(len(df), 100.0)

def calculate_strategy_metrics(df, strategy_data, selected_strategies):
    """
    用正確的策略回報序列計算績效指標
    """
    if not selected_strategies:
        return "請選擇至少一個策略來查看表現指標"
    
    metrics_data = []
    for strategy in selected_strategies:
        signal_col = f'{strategy}_signal'
        if signal_col in df.columns and 'Close' in df.columns:
            try:
                signals = df[signal_col]
                price = df['Close']
                returns = price.pct_change().fillna(0)
                strategy_returns = signals.shift(1).fillna(0) * returns
                if len(strategy_returns) > 0:
                    annual_return = (1 + strategy_returns.mean()) ** 252 - 1
                    volatility = strategy_returns.std() * np.sqrt(252)
                    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
                    cumulative = (1 + strategy_returns).cumprod()
                    rolling_max = cumulative.expanding().max()
                    drawdown = (cumulative - rolling_max) / rolling_max
                    max_drawdown = drawdown.min()
                    
                    metrics_data.append({
                        '策略': get_strategy_name(strategy),
                        '年化收益率': f"{annual_return*100:.2f}%",
                        '年化波動率': f"{volatility*100:.2f}%",
                        '夏普比率': f"{sharpe_ratio:.2f}",
                        '最大回撤': f"{max_drawdown*100:.2f}%"
                    })
            except Exception as e:
                logger.error(f"計算策略 {strategy} 指標時發生錯誤: {str(e)}")
    
    if not metrics_data:
        return "無法計算策略表現指標"
    
    # 生成HTML表格（略，與原本一致）
    table_html = """
    <div style="margin-top: 20px;">
        <h5 style="color: #eee; margin-bottom: 15px;">策略表現指標</h5>
        <table style="width: 100%; border-collapse: collapse; color: #fff;">
            <thead>
                <tr style="background-color: #333;">
    """
    if metrics_data:
        for key in metrics_data[0].keys():
            table_html += f'<th style="padding: 8px; border: 1px solid #555; text-align: left;">{key}</th>'
    table_html += """
                </tr>
            </thead>
            <tbody>
    """
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
        'north_south_rsi': '南北水RSI策略',
        'north_south_macd': '南北水MACD策略',
        'north_south_momentum': '南北水動量策略',
        'kdj_strategy': 'KDJ策略',
        'bollinger_strategy': '布林通道策略',
        'benchmark': '基準指數'
    }
    return names.get(strategy, strategy)

# 新增：自動搜尋所有策略時序績效 csv
def find_strategy_timeseries_csv():
    csv_files = glob.glob(os.path.join(CSV_PATH, 'integrated_*_2025*.csv'))
    return csv_files

# 新增：自動讀取所有策略時序績效資料
STRATEGY_DATA = {}
for csv_file in find_strategy_timeseries_csv():
    try:
        df = pd.read_csv(csv_file)
        # 自動判斷策略名稱
        base = os.path.basename(csv_file)
        name = base.replace('integrated_', '').replace('.csv', '').split('_2025')[0]
        STRATEGY_DATA[name] = df
    except Exception as e:
        logger.error(f"讀取 {csv_file} 失敗: {e}")

# 新增：自動生成策略選單
STRATEGY_CHOICES = []
for name in STRATEGY_DATA.keys():
    label = get_strategy_name(name)
    STRATEGY_CHOICES.append({'label': str(label), 'value': str(name)})
if not STRATEGY_CHOICES:
    STRATEGY_CHOICES = [
        {'label': 'RSI策略', 'value': 'rsi'},
        {'label': 'MACD策略', 'value': 'macd'},
        {'label': '布林通道策略', 'value': 'bollinger'},
        {'label': 'KDJ策略', 'value': 'kdj'},
        {'label': 'Stochastic策略', 'value': 'stochastic'}
    ]

@app.callback(
    Output('main-chart', 'figure'),
    [Input('strategy-radio', 'value')]
)
def update_main_chart(selected_strategy):
    try:
        df = STRATEGY_DATA.get(selected_strategy)
        if df is None or len(df) < 10:
            empty_fig = go.Figure()
            empty_fig.add_annotation(text="找不到該策略的時序績效資料，請確認 csv 結構！", showarrow=False)
            return empty_fig
        # 自動判斷欄位
        date_col = 'Date' if 'Date' in df.columns else df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        # 策略收益曲線
        y_col = None
        for c in ['strategy_returns', 'equity', '累積收益', '收益曲線']:
            if c in df.columns:
                y_col = c
                break
        if y_col is None:
            # fallback: 用 Close 畫價格
            y_col = 'Close' if 'Close' in df.columns else df.columns[1]
        main_fig = go.Figure()
        main_fig.add_trace(go.Scatter(
            x=df.index,
            y=df[y_col],
                mode='lines',
                name='策略收益',
                line=dict(color='#ff9800', width=2)
            ))
        main_fig.update_layout(
            title=f"{get_strategy_name(selected_strategy)} 收益曲線",
            xaxis_title="日期",
            yaxis_title="收益/權益/價格",
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=500
        )
        return main_fig
    except Exception as e:
        logger.error(f"update_main_chart 發生未預期錯誤: {str(e)}")
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="策略資料讀取錯誤，請檢查 csv 結構！", showarrow=False)
        return empty_fig

def load_signal_params():
    params_path = "signal_params.json"
    try:
        with open(params_path, 'r', encoding='utf-8') as f:
            params = json.load(f)
        return params
    except Exception:
        return {}

params = load_signal_params()

def get_strategy_param_text(selected_strategy):
    mapping = {
        'rsi_strategy': 'rsi',
        'rsi_ultimate': 'rsi_ultimate',
        'macd_strategy': 'macd',
        'bollinger_strategy': 'bollinger',
        'kdj_strategy': 'kdj',
        'stochastic_mil8': 'stochastic',
        'combined_strategy': 'combined'
    }
    key = mapping.get(selected_strategy)
    if key and key in params:
        p = params[key]
        # 支援 period, overbought, oversold, 其他參數
        param_str = ', '.join([f"{k}={v}" for k, v in p.items()])
        return f"參數：{param_str}"
    return "參數：無"

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
        dcc.RadioItems(
            id='strategy-radio',
            options=[
                {'label': '■ RSI(快速體園)', 'value': 'rsi_strategy'},
                {'label': '■ RSI(終極商洞)', 'value': 'rsi_ultimate'},
                {'label': '■ FI MACD 策略', 'value': 'macd_strategy'},
                {'label': '■ 布林帶策略', 'value': 'bollinger_strategy'},
                {'label': '■ KDJ 策略', 'value': 'kdj_strategy'},
                {'label': '■ Stochastic mil8', 'value': 'stochastic_mil8'},
                {'label': '■ 事策略統合', 'value': 'combined_strategy'}
            ],
            value='kdj_strategy',
            style=CHECKLIST_STYLE,
            labelStyle={'display': 'block', 'marginBottom': '8px', 'cursor': 'pointer'}
        )
    ], style=SIDEBAR_STYLE),
    
    # 主要內容區域
    html.Div([
        # 參數顯示區塊
        html.Div(id='strategy-param-info', style={'color': '#ffeb3b', 'marginBottom': '10px'}),
        # 主圖表
        html.Div([
            dcc.Graph(id='main-chart', style={'height': '500px'})
        ], style={'marginBottom': '30px'})
    ], style=CONTENT_STYLE)
], style=CUSTOM_STYLE)

@app.callback(
    Output('strategy-param-info', 'children'),
    [Input('strategy-radio', 'value')]
)
def update_param_info(selected_strategy):
    params = load_signal_params()
    mapping = {
        'rsi_strategy': 'rsi',
        'rsi_ultimate': 'rsi_ultimate',
        'macd_strategy': 'macd',
        'bollinger_strategy': 'bollinger',
        'kdj_strategy': 'kdj',
        'stochastic_mil8': 'stochastic',
        'combined_strategy': 'combined'
    }
    key = mapping.get(selected_strategy)
    if key and key in params:
        p = params[key]
        param_str = ', '.join([f"{k}={v}" for k, v in p.items()])
        return f"參數：{param_str}"
    return "參數：無"

if __name__ == "__main__":
    app.run(debug=True, port=8050) 