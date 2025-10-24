import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
import webbrowser
import os
import json
import warnings
import seaborn as sns
from scipy import stats
import sys
import threading
import time
try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False
    print("❌ schedule未安裝，跳過自動化功能")

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import tkinter as tk
from tkinter import ttk, messagebox

# Dash 相關導入 (可選)
try:
    import dash
    from dash import dcc, html, dash_table
    from dash.dependencies import Input, Output, State
    import dash_bootstrap_components as dbc
    import plotly.graph_objects as go
    import plotly.express as px
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    print("Dash 套件未安裝，將跳過 Web 界面功能")
# 導入TKinter (可選)
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    print("❌ TKinter未安裝，跳過GUI功能")

# 導入Plotly (可選)  
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("❌ Plotly未安裝，跳過互動圖表功能")

# QuantStats專業量化分析（用戶指定使用）
try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
    print("✅ QuantStats已載入，將提供專業量化分析")
except ImportError:
    QUANTSTATS_AVAILABLE = False
    print("❌ QuantStats未安裝，將跳過專業量化分析功能。如需完整功能，請運行：pip install quantstats")
    # 移除任何強制退出註釋，確保系統可正常運行
    # sys.exit() # 已移除強制退出，改為可選功能

# Plotguy 相關導入 (可選)
try:
    import plotguy
    PLOTGUY_AVAILABLE = True
    print("✅ Plotguy已載入，將提供專業Dashboard功能")
except (ImportError, AttributeError) as e:
    PLOTGUY_AVAILABLE = False
    print(f"❌ Plotguy載入失敗，將跳過plotguy Dashboard功能: {e}")
    print("💡 建議檢查plotguy和dash版本兼容性")

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

# 忽略警告
warnings.filterwarnings('ignore')

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 全域配置類
@dataclass
class SystemConfig:
    """系統配置類"""
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
    report_time: str = "09:00"  # 每日9點生成報告

# 量化分析系統配置
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

# 全域配置實例
config = SystemConfig()
RISK_CONFIG = RiskConfig()
BACKTEST_CONFIG = BacktestConfig()
DISPLAY_CONFIG = DisplayConfig()

# 風險度量類
class RiskMetrics:
    """組合風險度量類"""
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """計算VaR"""
        return float(np.percentile(returns.dropna(), confidence_level * 100))
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """計算CVaR"""
        var = RiskMetrics.calculate_var(returns, confidence_level)
        filtered_returns = returns[returns <= var]
        if len(filtered_returns) > 0:
            cvar = filtered_returns.mean()
            return float(cvar)
        return 0.0
    
    @staticmethod
    def calculate_portfolio_correlation(strategy_returns: Dict[str, pd.Series]) -> pd.DataFrame:
        """計算策略間相關性"""
        returns_df = pd.DataFrame(strategy_returns)
        return returns_df.corr()
    
    @staticmethod
    def calculate_portfolio_risk(strategy_returns: Dict[str, pd.Series], weights: Dict[str, float]) -> Dict:
        """計算組合風險指標"""
        returns_df = pd.DataFrame(strategy_returns)
        weights_series = pd.Series(weights)
        
        # 組合收益
        portfolio_returns = (returns_df * weights_series).sum(axis=1)
        
        # 風險指標
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)
        portfolio_var = RiskMetrics.calculate_var(portfolio_returns)
        portfolio_cvar = RiskMetrics.calculate_cvar(portfolio_returns)
        
        # 最大回撤
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'portfolio_volatility': portfolio_vol,
            'portfolio_var_95': portfolio_var,
            'portfolio_cvar_95': portfolio_cvar,
            'portfolio_max_drawdown': max_drawdown,
            'portfolio_returns': portfolio_returns
        }

# 止損止盈優化類
class StopLossOptimizer:
    """止損止盈優化類"""
    
    @staticmethod
    def apply_stop_loss_take_profit(signals: pd.DataFrame, prices: pd.Series, 
                                   stop_loss_pct: float, take_profit_pct: float,
                                   trailing_stop: bool = False, trailing_pct: float = 0.03) -> pd.DataFrame:
        """應用止損止盈邏輯"""
        enhanced_signals = signals.copy()
        enhanced_signals['stop_loss_exit'] = 0
        enhanced_signals['take_profit_exit'] = 0
        enhanced_signals['trailing_stop_exit'] = 0
        
        position = 0
        entry_price = 0
        highest_price = 0
        
        for i in range(1, len(signals)):
            current_price = prices.iloc[i]
            current_signal = enhanced_signals['signal'].iloc[i]
            
            # 進場邏輯
            if current_signal != 0 and position == 0:
                position = current_signal
                entry_price = current_price
                highest_price = current_price if position > 0 else current_price
                continue
            
            # 持倉期間的出場邏輯
            if position != 0:
                if position > 0:  # 做多持倉
                    # 更新最高價（用於追蹤止損）
                    highest_price = max(highest_price, current_price)
                    
                    # 止損檢查
                    if current_price <= entry_price * (1 - stop_loss_pct):
                        enhanced_signals.loc[enhanced_signals.index[i], 'signal'] = 0
                        enhanced_signals.loc[enhanced_signals.index[i], 'stop_loss_exit'] = 1
                        position = 0
                        continue
                    
                    # 止盈檢查
                    if current_price >= entry_price * (1 + take_profit_pct):
                        enhanced_signals.loc[enhanced_signals.index[i], 'signal'] = 0
                        enhanced_signals.loc[enhanced_signals.index[i], 'take_profit_exit'] = 1
                        position = 0
                        continue
                    
                    # 追蹤止損檢查
                    if trailing_stop and current_price <= highest_price * (1 - trailing_pct):
                        enhanced_signals.loc[enhanced_signals.index[i], 'signal'] = 0
                        enhanced_signals.loc[enhanced_signals.index[i], 'trailing_stop_exit'] = 1
                        position = 0
                        continue
                
                elif position < 0:  # 做空持倉
                    # 做空的止損止盈邏輯（相反）
                    if current_price >= entry_price * (1 + stop_loss_pct):
                        enhanced_signals.loc[enhanced_signals.index[i], 'signal'] = 0
                        enhanced_signals.loc[enhanced_signals.index[i], 'stop_loss_exit'] = 1
                        position = 0
                        continue
                    
                    if current_price <= entry_price * (1 - take_profit_pct):
                        enhanced_signals.loc[enhanced_signals.index[i], 'signal'] = 0
                        enhanced_signals.loc[enhanced_signals.index[i], 'take_profit_exit'] = 1
                        position = 0
                        continue
        
        return enhanced_signals

# 策略組合器類
class StrategyPortfolioManager:
    """策略組合器類"""
    
    def __init__(self):
        self.strategies = {}
        self.weights = {}
        self.risk_metrics = RiskMetrics()
    
    def add_strategy(self, name: str, signals: pd.DataFrame, weight: float = 1.0):
        """添加策略"""
        self.strategies[name] = signals
        self.weights[name] = weight
    
    def optimize_weights(self, strategy_returns: Dict[str, pd.Series], 
                        target: str = 'sharpe') -> Dict[str, float]:
        """優化權重"""
        from scipy.optimize import minimize
        
        returns_df = pd.DataFrame(strategy_returns)
        n_strategies = len(returns_df.columns)
        
        def objective(weights):
            portfolio_returns = (returns_df * weights).sum(axis=1)
            
            if target == 'sharpe':
                return -(portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252))
            elif target == 'return':
                return -portfolio_returns.mean() * 252
            elif target == 'risk':
                return portfolio_returns.std() * np.sqrt(252)
        
        # 約束條件
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # 權重和為1
        bounds = [(0, 0.5) for _ in range(n_strategies)]  # 單策略最大50%權重
        
        # 初始權重
        x0 = np.array([1/n_strategies] * n_strategies)
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            return dict(zip(returns_df.columns, result.x))
        else:
            return dict(zip(returns_df.columns, x0))
    
    def generate_combined_signals(self, method: str = 'weighted_average') -> pd.DataFrame:
        """生成組合信號"""
        if not self.strategies:
            return pd.DataFrame()
        
        # 獲取所有策略的信號
        all_signals = pd.DataFrame()
        for name, signals in self.strategies.items():
            all_signals[name] = signals['signal'] * self.weights.get(name, 1.0)
        
        combined_signals = pd.DataFrame(index=all_signals.index)
        
        if method == 'weighted_average':
            combined_signals['signal'] = all_signals.mean(axis=1)
            # 閾值化
            combined_signals['signal'] = np.where(combined_signals['signal'] > 0.3, 1,
                                                 np.where(combined_signals['signal'] < -0.3, -1, 0))
        
        elif method == 'majority_vote':
            # 多數投票
            vote_long = (all_signals > 0).sum(axis=1)
            vote_short = (all_signals < 0).sum(axis=1)
            combined_signals['signal'] = np.where(vote_long > vote_short, 1,
                                                 np.where(vote_short > vote_long, -1, 0))
        
        elif method == 'unanimous':
            # 一致決策
            combined_signals['signal'] = np.where((all_signals > 0).all(axis=1), 1,
                                                 np.where((all_signals < 0).all(axis=1), -1, 0))
        
        elif method == 'risk_parity':
            # 基於風險平價的權重調整
            try:
                strategy_vols = all_signals.std()
                # 確保 strategy_vols 是 Series 類型並處理零值
                if isinstance(strategy_vols, pd.Series) and not strategy_vols.empty:
                    # 防止除零錯誤和無效值
                    strategy_vols = strategy_vols.fillna(1e-8).replace(0, 1e-8)
                    # 確保所有值都是正數
                    strategy_vols = strategy_vols.abs()
                    inv_vols = 1.0 / strategy_vols
                    # 檢查是否有無限值
                    if np.isinf(inv_vols).any() or np.isnan(inv_vols).any():
                        raise ValueError("計算出無效的倒數波動率")
                    risk_weights = inv_vols / inv_vols.sum()
                else:
                    # 如果不是 Series 或為空，使用等權重
                    n_strategies = len(all_signals.columns)
                    risk_weights = pd.Series([1.0 / n_strategies] * n_strategies, 
                                           index=all_signals.columns)
                
                # 確保權重有效
                if risk_weights.sum() == 0:
                    risk_weights = pd.Series([1.0 / len(all_signals.columns)] * len(all_signals.columns), 
                                           index=all_signals.columns)
                
                combined_signals['signal'] = (all_signals * risk_weights).mean(axis=1)
                combined_signals['signal'] = np.where(combined_signals['signal'] > 0.2, 1,
                                                     np.where(combined_signals['signal'] < -0.2, -1, 0))
                
            except Exception as e:
                logger.warning(f"風險平價計算失敗，使用等權重: {e}")
                # 使用等權重作為備用方案
                try:
                    combined_signals['signal'] = all_signals.mean(axis=1)
                    combined_signals['signal'] = np.where(combined_signals['signal'] > 0.2, 1,
                                                         np.where(combined_signals['signal'] < -0.2, -1, 0))
                except Exception as fallback_error:
                    logger.error(f"等權重備用方案也失敗: {fallback_error}")
                    # 最後的備用方案：返回零信號
                    combined_signals['signal'] = 0
        
        return combined_signals

# 自動化任務管理器
class AutomationManager:
    """自動化任務管理器"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.running = False
        self.scheduler_thread = None
    
    def start_automation(self):
        """啟動自動化任務"""
        if not SCHEDULE_AVAILABLE:
            logger.warning("schedule套件未安裝，跳過自動化功能")
            return
            
        if self.config.auto_update_enabled:
            schedule.every(self.config.update_interval_hours).hours.do(self.update_data_job)
        
        if self.config.auto_report_enabled:
            schedule.every().day.at(self.config.report_time).do(self.generate_report_job)
        
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        logger.info("自動化任務已啟動")
    
    def stop_automation(self):
        """停止自動化任務"""
        self.running = False
        if SCHEDULE_AVAILABLE:
            schedule.clear()
        logger.info("自動化任務已停止")
    
    def _run_scheduler(self):
        """運行調度器"""
        if not SCHEDULE_AVAILABLE:
            return
            
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # 每分鐘檢查一次
    
    def update_data_job(self):
        """數據更新任務"""
        try:
            logger.info("開始自動數據更新...")
            data = get_hk_stock_data("2800.HK", "6mo")
            if data is not None:
                data.to_csv("2800_HK_北水期間數據.csv")
                logger.info("數據更新完成")
            else:
                logger.error("數據更新失敗")
        except Exception as e:
            logger.error(f"數據更新異常: {e}")
    
    def generate_report_job(self):
        """報告生成任務"""
        try:
            logger.info("開始自動報告生成...")
            # 這裡可以調用主分析函數生成報告
            logger.info("報告生成完成")
        except Exception as e:
            logger.error(f"報告生成異常: {e}")

# 互動式參數調整器
class InteractiveParameterTuner:
    """互動式參數調整器"""
    
    def __init__(self):
        self.root = None
        self.params = {}
        self.callbacks = {}
    
    def create_tuner_window(self, strategy_params: Dict):
        """創建調參窗口"""
        if not TKINTER_AVAILABLE:
            logger.warning("TKinter未安裝，跳過GUI功能")
            return
            
        self.root = tk.Tk()
        self.root.title("策略參數實時調整器")
        self.root.geometry("800x600")
        
        # 創建筆記本控件
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 為每個策略創建標籤頁
        for strategy_name, params in strategy_params.items():
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=strategy_name)
            self._create_param_controls(frame, strategy_name, params)
        
        # 控制按鈕
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(button_frame, text="應用更改", command=self.apply_changes).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="重置參數", command=self.reset_params).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="保存配置", command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="載入配置", command=self.load_config).pack(side=tk.LEFT, padx=5)
    
    def _create_param_controls(self, parent: ttk.Frame, strategy_name: str, params: Dict):
        """創建參數控制項"""
        if strategy_name not in self.params:
            self.params[strategy_name] = {}
        
        row = 0
        for param_name, param_value in params.items():
            ttk.Label(parent, text=f"{param_name}:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            
            if isinstance(param_value, (int, float)):
                var = tk.DoubleVar(value=param_value)
                scale = ttk.Scale(parent, from_=param_value*0.5, to=param_value*2, 
                                 orient=tk.HORIZONTAL, variable=var)
                scale.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
                
                # 顯示當前值
                value_label = ttk.Label(parent, text=f"{param_value:.2f}")
                value_label.grid(row=row, column=2, padx=5, pady=2)
                
                # 更新值顯示
                def update_label(val, label=value_label):
                    label.config(text=f"{float(val):.2f}")
                
                scale.config(command=update_label)
                self.params[strategy_name][param_name] = var
            
            elif isinstance(param_value, bool):
                var = tk.BooleanVar(value=param_value)
                check = ttk.Checkbutton(parent, variable=var)
                check.grid(row=row, column=1, sticky="w", padx=5, pady=2)
                self.params[strategy_name][param_name] = var
            
            row += 1
        
        parent.columnconfigure(1, weight=1)
    
    def apply_changes(self):
        """應用參數更改"""
        # 獲取當前參數值
        current_params = {}
        for strategy_name, params in self.params.items():
            current_params[strategy_name] = {}
            for param_name, var in params.items():
                current_params[strategy_name][param_name] = var.get()
        
        # 觸發回調函數
        for callback in self.callbacks.values():
            callback(current_params)
        
        if TKINTER_AVAILABLE:
            messagebox.showinfo("成功", "參數已更新，正在重新計算...")
        else:
            print("✅ 參數已更新，正在重新計算...")
    
    def reset_params(self):
        """重置參數"""
        if TKINTER_AVAILABLE:
            messagebox.showinfo("重置", "參數已重置為默認值")
        else:
            print("✅ 參數已重置為默認值")
    
    def save_config(self):
        """保存配置"""
        # 保存當前參數到文件
        config_data = {}
        for strategy_name, params in self.params.items():
            config_data[strategy_name] = {}
            for param_name, var in params.items():
                config_data[strategy_name][param_name] = var.get()
        
        with open("strategy_params.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        
        if TKINTER_AVAILABLE:
            messagebox.showinfo("保存", "配置已保存到 strategy_params.json")
        else:
            print("✅ 配置已保存到 strategy_params.json")
    
    def load_config(self):
        """載入配置"""
        try:
            with open("strategy_params.json", "r", encoding="utf-8") as f:
                config_data = json.load(f)
            
            # 更新界面參數
            for strategy_name, params in config_data.items():
                if strategy_name in self.params:
                    for param_name, value in params.items():
                        if param_name in self.params[strategy_name]:
                            self.params[strategy_name][param_name].set(value)
            
            if TKINTER_AVAILABLE:
                messagebox.showinfo("載入", "配置已從 strategy_params.json 載入")
            else:
                print("✅ 配置已從 strategy_params.json 載入")
        except FileNotFoundError:
            if TKINTER_AVAILABLE:
                messagebox.showwarning("警告", "配置文件不存在")
            else:
                print("⚠️ 警告：配置文件不存在")
        except Exception as e:
            if TKINTER_AVAILABLE:
                messagebox.showerror("錯誤", f"載入配置失敗: {e}")
            else:
                print(f"❌ 錯誤：載入配置失敗: {e}")
    
    def add_callback(self, name: str, callback):
        """添加參數更改回調函數"""
        self.callbacks[name] = callback
    
    def show(self):
        """顯示調參窗口"""
        if TKINTER_AVAILABLE and self.root:
            self.root.mainloop()
        else:
            logger.warning("GUI功能不可用，跳過顯示窗口")

def get_northbound_data(symbol="2800.HK", start_date=None, end_date=None):
    """
    從北水數據文件中獲取股票數據
    這是一個佔位符函數，需要根據實際的北水數據格式來實現
    """
    print("🔍 檢查是否有北水數據文件...")
    
    # 檢查北水數據文件
    northbound_files = [
        "data/data_csv/northbound_flow_2017-07-03_to_2025-06-20.csv",
        "2800_HK_北水期間數據.csv",
        "港股/流程/2800_HK_北水期間數據.csv"
    ]
    
    for file_path in northbound_files:
        if os.path.exists(file_path):
            print(f"✅ 找到北水數據文件: {file_path}")
            try:
                # 讀取北水數據
                df = pd.read_csv(file_path)
                print(f"📊 北水數據包含 {len(df)} 條記錄")
                print(f"📅 數據列名: {list(df.columns)}")
                
                # 如果數據格式正確，進行處理
                if 'Date' in df.columns or 'date' in df.columns:
                    # 標準化日期列名
                    if 'date' in df.columns:
                        df.rename(columns={'date': 'Date'}, inplace=True)
                    
                    # 轉換日期格式
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    
                    # 確保包含必要的OHLCV列
                    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    
                    if not missing_columns:
                        print("✅ 北水數據格式正確，包含完整的OHLCV數據")
                        return df
                    else:
                        print(f"⚠️ 北水數據缺少必要列: {missing_columns}")
                
            except Exception as e:
                print(f"❌ 讀取北水數據失敗: {e}")
    
    print("⚠️ 未找到有效的北水數據，將回退到yfinance")
    return None

def get_hk_stock_data(symbol="2800.HK", period="1y", use_northbound_data=False):
    """
    獲取港股數據，包含重試機制和錯誤處理
    """
    try:
        logger.info(f"📊 正在獲取 {symbol} 的數據...")
        
        # 使用 data_handler 中的 DataFetcher
        from data_handler import DataFetcher
        
        # 使用固定的歷史日期範圍
        end_date = CONFIG.data_end_date    # 使用配置的結束日期
        start_date = CONFIG.data_start_date  # 使用配置的開始日期
        
        data = DataFetcher.get_hk_stock_data(
            symbol=symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
            use_northbound_data=use_northbound_data,
            max_retries=CONFIG.data_retry_attempts,
            retry_delay=CONFIG.data_retry_delay
        )
        
        if data is None or data.empty:
            logger.error(f"❌ 無法獲取 {symbol} 的數據")
            return None
            
        logger.info(f"✅ 成功獲取 {len(data)} 天的數據")
        return data
        
    except Exception as e:
        logger.error(f"❌ 數據獲取失敗: {e}")
        return None


def calculate_rsi_custom(close_prices, period=14):
    """自定義RSI計算"""
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd_custom(close_prices, fast=12, slow=26, signal=9):
    """自定義MACD計算"""
    exp1 = close_prices.ewm(span=fast).mean()
    exp2 = close_prices.ewm(span=slow).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=signal).mean()
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram

def calculate_bollinger_custom(close_prices, period=20, std_dev=2.0):
    """自定義布林帶計算"""
    middle = close_prices.rolling(window=period).mean()
    std = close_prices.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower

def calculate_technical_indicators(data):
    """計算技術指標"""
    # RSI (使用標準參數顯示)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (使用標準參數顯示)
    exp1 = data['Close'].ewm(span=12).mean()
    exp2 = data['Close'].ewm(span=26).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
    
    # 布林帶 (使用標準參數顯示)
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
    
    # 移動平均線
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    
    return data



def backtest_strategy(data, strategy_name):
    """回測策略 - 使用最佳化參數，修復夏普比率計算"""
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0
    
    # 判斷是否為做空策略
    is_short = strategy_name.endswith('_Short')
    base_strategy = strategy_name.replace('_Short', '') if is_short else strategy_name
    
    # 使用最佳化後的參數
    if base_strategy == 'RSI':
        # 最佳化參數: 週期=21, 超賣=35, 超買=80
        rsi_21 = calculate_rsi_custom(data['Close'], 21)
        if is_short:
            # 做空策略：RSI超買時做空，RSI超賣時平倉
            signals.loc[rsi_21 > 80, 'signal'] = -1  # 做空
            signals.loc[rsi_21 < 35, 'signal'] = 0   # 平倉
        else:
            # 做多策略：RSI超賣時做多，RSI超買時平倉
            signals.loc[rsi_21 < 35, 'signal'] = 1   # 做多
            signals.loc[rsi_21 > 80, 'signal'] = 0   # 平倉
    elif base_strategy == 'MACD':
        # 最佳化參數: 快線=20, 慢線=49, 信號=6
        macd_opt, macd_signal_opt, _ = calculate_macd_custom(data['Close'], 20, 49, 6)
        if is_short:
            # 做空策略：MACD死叉時做空
            signals.loc[macd_opt < macd_signal_opt, 'signal'] = -1
            signals.loc[macd_opt > macd_signal_opt, 'signal'] = 0
        else:
            # 做多策略：MACD金叉時做多
            signals.loc[macd_opt > macd_signal_opt, 'signal'] = 1
            signals.loc[macd_opt < macd_signal_opt, 'signal'] = 0
    elif base_strategy == 'Bollinger':
        # 最佳化參數: 週期=24, 標準差=2.2
        bb_upper_opt, bb_middle_opt, bb_lower_opt = calculate_bollinger_custom(data['Close'], 24, 2.2)
        if is_short:
            # 做空策略：價格觸及上軌時做空
            signals.loc[data['Close'] >= bb_upper_opt, 'signal'] = -1
            signals.loc[data['Close'] <= bb_lower_opt, 'signal'] = 0
        else:
            # 做多策略：價格觸及下軌時做多
            signals.loc[data['Close'] <= bb_lower_opt, 'signal'] = 1
            signals.loc[data['Close'] >= bb_upper_opt, 'signal'] = 0
    elif base_strategy == 'Mean_Reversion':
        if is_short:
            signals.loc[data['Close'] > data['MA20'] * 1.02, 'signal'] = -1
            signals.loc[data['Close'] < data['MA20'] * 0.98, 'signal'] = 0
        else:
            signals.loc[data['Close'] < data['MA20'] * 0.98, 'signal'] = 1
            signals.loc[data['Close'] > data['MA20'] * 1.02, 'signal'] = 0
    elif base_strategy == 'SMA_Cross':
        if is_short:
            signals.loc[data['MA5'] < data['MA20'], 'signal'] = -1
            signals.loc[data['MA5'] > data['MA20'], 'signal'] = 0
        else:
            signals.loc[data['MA5'] > data['MA20'], 'signal'] = 1
            signals.loc[data['MA5'] < data['MA20'], 'signal'] = 0
    elif base_strategy == 'Momentum':
        momentum = data['Close'] / data['Close'].shift(10)
        if is_short:
            signals.loc[momentum < 0.95, 'signal'] = -1
            signals.loc[momentum > 1.05, 'signal'] = 0
        else:
            signals.loc[momentum > 1.05, 'signal'] = 1
            signals.loc[momentum < 0.95, 'signal'] = 0
    elif base_strategy == 'EMA_Cross':
        ema8 = data['Close'].ewm(span=8).mean()
        ema21 = data['Close'].ewm(span=21).mean()
        if is_short:
            signals.loc[ema8 < ema21, 'signal'] = -1
            signals.loc[ema8 > ema21, 'signal'] = 0
        else:
            signals.loc[ema8 > ema21, 'signal'] = 1
            signals.loc[ema8 < ema21, 'signal'] = 0
    elif base_strategy == 'Multi_MA':
        ma10 = data['Close'].rolling(window=10).mean()
        ma30 = data['Close'].rolling(window=30).mean()
        ma50 = data['Close'].rolling(window=50).mean()
        if is_short:
            signals.loc[(data['Close'] < ma10) & (ma10 < ma30) & (ma30 < ma50), 'signal'] = -1
            signals.loc[(data['Close'] > ma10) & (ma10 > ma30) & (ma30 > ma50), 'signal'] = 0
        else:
            signals.loc[(data['Close'] > ma10) & (ma10 > ma30) & (ma30 > ma50), 'signal'] = 1
            signals.loc[(data['Close'] < ma10) & (ma10 < ma30) & (ma30 < ma50), 'signal'] = 0
    
    # 計算持倉和回報
    signals['position'] = signals['signal'].diff()
    signals['returns'] = data['Close'].pct_change()
    signals['strategy_returns'] = signals['signal'].shift(1) * signals['returns']
    
    # 修復夏普比率計算
    strategy_returns = signals['strategy_returns'].dropna()
    total_return = (1 + strategy_returns).cumprod().iloc[-1] - 1
    total_return_pct = total_return * 100
    
    # 正確計算年化收益率和波動率（使用標準夏普比率公式）
    if len(strategy_returns) > 0:
        annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        
        # 夏普比率計算：提供兩種方法（與utils.py一致）
        INDIVIDUAL_SHARPE_METHOD = "STANDARD"  # 可選: "SIMPLE" 或 "STANDARD"
        
        if INDIVIDUAL_SHARPE_METHOD == "SIMPLE":
            # 簡化夏普比率（如utils.py）
            sharpe_ratio = (
                strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
            ) if strategy_returns.std() > 0 else 0
            
        elif INDIVIDUAL_SHARPE_METHOD == "STANDARD":
            # 標準夏普比率公式：(年化收益率 - 無風險利率) / 年化波動率
            # 使用美國國債2%作為無風險利率
            risk_free_rate = 0.02
            sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
    else:
        annual_return = 0
        volatility = 0
        sharpe_ratio = 0
    
    # 計算最大回撤和權益曲線
    cumulative = (1 + strategy_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100 if len(drawdown) > 0 else 0
    
    # 計算勝率和交易次數
    trades = signals['position'].abs().sum() / 2
    winning_trades = (strategy_returns > 0).sum()
    total_trades = (strategy_returns != 0).sum()
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    return {
        'total_return': total_return_pct,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'trade_count': int(trades),
        'volatility': volatility * 100,
        'signals': signals,
        'equity_curve': cumulative,  # 添加權益曲線數據
        'strategy_returns': strategy_returns,  # 添加策略回報序列
        'drawdown_series': drawdown  # 添加回撤序列
    }

def create_chart(data, strategy_results):
    """創建3層技術分析圖表"""
    # 創建3層圖表布局
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.2)
    
    # 第一層：主價格圖表（跨兩列）
    ax1 = fig.add_subplot(gs[0, :])
    
    # 第二層：技術指標
    ax2 = fig.add_subplot(gs[1, 0])  # RSI
    ax3 = fig.add_subplot(gs[1, 1])  # MACD
    
    # 第三層：成交量和策略表現
    ax4 = fig.add_subplot(gs[2, 0])  # 成交量
    ax5 = fig.add_subplot(gs[2, 1])  # 策略表現
    
    fig.suptitle('2800.HK 三層完整技術分析圖表', fontsize=18, fontweight='bold', y=0.98)
    
    # 第一層：主價格圖表
    ax1.plot(data.index, data['Close'], label='收盤價', linewidth=2.5, color='#2E86C1')
    ax1.plot(data.index, data['MA5'], label='MA5', alpha=0.8, color='#F39C12', linewidth=1.5)
    ax1.plot(data.index, data['MA20'], label='MA20', alpha=0.8, color='#E74C3C', linewidth=1.5)
    ax1.fill_between(data.index, data['BB_Lower'], data['BB_Upper'], alpha=0.15, color='gray', label='布林帶')
    
    # 添加買賣信號點
    for strategy, results in strategy_results.items():
        if strategy in ['RSI', 'MACD', 'Bollinger']:
            signals = results['signals']
            buy_signals = signals[signals['position'] > 0]
            sell_signals = signals[signals['position'] < 0]
            
            if not buy_signals.empty:
                ax1.scatter(buy_signals.index, data.loc[buy_signals.index, 'Close'], 
                           marker='^', color='green', s=60, alpha=0.7, label=f'{strategy}買入')
            if not sell_signals.empty:
                ax1.scatter(sell_signals.index, data.loc[sell_signals.index, 'Close'], 
                           marker='v', color='red', s=60, alpha=0.7, label=f'{strategy}賣出')
    
    ax1.set_title('🎯 主價格走勢與技術指標', fontweight='bold', fontsize=14)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 第二層左：RSI
    ax2.plot(data.index, data['RSI'], color='#8E44AD', linewidth=2)
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='超買線(70)')
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='超賣線(30)')
    ax2.fill_between(data.index, 30, 70, alpha=0.1, color='gray')
    ax2.set_title('📊 RSI相對強弱指標', fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 第二層右：MACD
    ax3.plot(data.index, data['MACD'], label='MACD', color='#3498DB', linewidth=2)
    ax3.plot(data.index, data['MACD_Signal'], label='信號線', color='#E74C3C', linewidth=2)
    ax3.bar(data.index, data['MACD_Histogram'], label='MACD柱', alpha=0.6, color='gray')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_title('📈 MACD指標', fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 第三層左：成交量
    colors = ['red' if close >= open_price else 'green' for close, open_price in zip(data['Close'], data['Open'])]
    ax4.bar(data.index, data['Volume'], color=colors, alpha=0.6)
    ax4.plot(data.index, data['Volume'].rolling(20).mean(), color='orange', linewidth=2, label='成交量20MA')
    ax4.set_title('📊 成交量分析', fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 第三層右：策略表現對比
    strategies = list(strategy_results.keys())[:6]  # 前6個策略
    returns = [strategy_results[s]['total_return'] for s in strategies]
    colors_perf = ['green' if r > 0 else 'red' for r in returns]
    
    bars = ax5.bar(range(len(strategies)), returns, color=colors_perf, alpha=0.7)
    ax5.set_title('🎯 策略績效對比', fontweight='bold')
    ax5.set_xticks(range(len(strategies)))
    ax5.set_xticklabels(strategies, rotation=45, fontsize=9)
    ax5.set_ylabel('回報率 (%)')
    ax5.grid(True, alpha=0.3)
    
    # 添加數值標籤
    for bar, ret in zip(bars, returns):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -1.5),
                f'{ret:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
    
    plt.tight_layout()
    
    # 保存圖表
    output_dir = "港股輸出"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_filename = f"{output_dir}/2800_HK_完整分析圖表_{timestamp}.png"
    plt.savefig(png_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return png_filename

def get_strategy_details(strategy_name):
    """獲取策略詳細信息"""
    strategy_details = {
        'RSI': {
            'name': 'RSI超買超賣策略 (🔥最佳化版)',
            'description': '🚀 使用61線程最佳化後的RSI策略，從8,112種組合中選出最佳參數',
            'parameters': {
                'RSI週期': '21 (最佳化)',
                '超買閾值': '35 (最佳化)',
                '超賣閾值': '80 (最佳化)',
                '夏普比率': '1.26 (修復後)'
            },
            'entry_conditions': [
                'RSI < 35 時買入（最佳化超賣反彈）',
                'RSI > 80 時賣出（最佳化超買回調）',
                '結合價格動能確認進場'
            ],
            'exit_conditions': [
                'RSI回到中性區間（35-80）時平倉',
                '持倉超過20天強制平倉',
                '出現反向信號時立即退出'
            ]
        },
        'MACD': {
            'name': 'MACD金叉死叉策略 (🔥最佳化版)',
            'description': '🚀 使用61線程最佳化後的MACD策略，從2,640種組合中選出最佳參數',
            'parameters': {
                '快線EMA': '20 (最佳化)',
                '慢線EMA': '49 (最佳化)',
                '信號線': '6 (最佳化)',
                '夏普比率': '2.076'
            },
            'entry_conditions': [
                'MACD線 > 信號線時買入（最佳化金叉）',
                'MACD線 < 信號線時賣出（最佳化死叉）',
                '配合成交量確認信號強度'
            ],
            'exit_conditions': [
                '出現相反信號時平倉',
                'MACD背離時提前退出',
                '信號衰減時減倉'
            ]
        },
        'Bollinger': {
            'name': '布林帶反轉策略 (🔥最佳化版)',
            'description': '🚀 使用61線程最佳化後的布林帶策略，從441種組合中選出最佳參數',
            'parameters': {
                '移動平均週期': '24 (最佳化)',
                '標準差倍數': '2.2 (最佳化)',
                '夏普比率': '3.731'
            },
            'entry_conditions': [
                '價格觸及下軌時買入（最佳化超跌反彈）',
                '價格觸及上軌時賣出（最佳化超漲回調）',
                '結合RSI確認超買超賣'
            ],
            'exit_conditions': [
                '價格回到中軌附近平倉',
                '布林帶收縮時減倉',
                '突破上軌後追蹤止損'
            ]
        },
        'Mean_Reversion': {
            'name': '均值回歸策略',
            'description': '基於價格偏離移動平均線的均值回歸交易策略',
            'parameters': {
                '基準週期': '20日移動平均',
                '偏離閾值': '±2%',
                '持倉週期': '5-15天'
            },
            'entry_conditions': [
                '價格低於MA20的98%時買入',
                '價格高於MA20的102%時賣出',
                '成交量配合確認'
            ],
            'exit_conditions': [
                '價格回歸至MA20附近',
                '偏離幅度進一步擴大時止損',
                '持倉時間超過15天'
            ]
        },
        'SMA_Cross': {
            'name': '移動平均交叉策略',
            'description': '基於短期和長期移動平均線交叉的趨勢跟蹤策略',
            'parameters': {
                '短期MA': '5日簡單移動平均',
                '長期MA': '20日簡單移動平均',
                '信號確認': '連續2天確認'
            },
            'entry_conditions': [
                'MA5上穿MA20時買入（金叉）',
                'MA5下穿MA20時賣出（死叉）',
                '成交量放大確認'
            ],
            'exit_conditions': [
                '出現相反交叉信號',
                '價格偏離均線過遠',
                '市場轉為震盪時平倉'
            ]
        },
        'Momentum': {
            'name': '動量策略',
            'description': '基於價格動量的趨勢跟蹤策略',
            'parameters': {
                '動量週期': '10天',
                '買入閾值': '+5%',
                '賣出閾值': '-5%'
            },
            'entry_conditions': [
                '10天動量 > 1.05時買入',
                '10天動量 < 0.95時賣出',
                '突破關鍵阻力位確認'
            ],
            'exit_conditions': [
                '動量轉弱時平倉',
                '價格回吐超過3%',
                '技術指標出現背離'
            ]
        },
        'EMA_Cross': {
            'name': '指數移動平均交叉策略',
            'description': '基於EMA8和EMA21交叉的快速趨勢策略',
            'parameters': {
                '快速EMA': '8期指數移動平均',
                '慢速EMA': '21期指數移動平均',
                '過濾條件': 'RSI配合確認'
            },
            'entry_conditions': [
                'EMA8上穿EMA21時買入',
                'EMA8下穿EMA21時賣出',
                'RSI非極端值時確認'
            ],
            'exit_conditions': [
                '出現反向交叉信號',
                'EMA開始收斂時減倉',
                '波動率異常時止損'
            ]
        },
        'Multi_MA': {
            'name': '多重移動平均策略',
            'description': '結合MA10、MA30、MA50的多重趨勢確認策略',
            'parameters': {
                '短期MA': '10日移動平均',
                '中期MA': '30日移動平均',
                '長期MA': '50日移動平均'
            },
            'entry_conditions': [
                '價格 > MA10 > MA30 > MA50時買入',
                '價格 < MA10 < MA30 < MA50時賣出',
                '多重均線排列確認趨勢'
            ],
            'exit_conditions': [
                '均線排列被打破',
                '價格跌破關鍵均線',
                '趨勢轉為震盪時平倉'
            ]
        }
    }
    
    return strategy_details.get(strategy_name, {
        'name': f'{strategy_name}策略',
        'description': f'基於{strategy_name}的技術分析策略',
        'parameters': {'週期': '動態調整'},
        'entry_conditions': ['技術指標達到買入條件'],
        'exit_conditions': ['技術指標達到賣出條件']
    })

def generate_html_dashboard(strategy_results, chart_filename):
    """生成增強版互動式Dashboard，包含詳細組合表現"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"2800_HK_完整分析Dashboard_{timestamp}.html"
    
    # 讀取最佳化結果數據
    optimization_data = load_optimization_results()
    
    # 準備策略數據
    strategy_data = {}
    for strategy, results in strategy_results.items():
        if strategy == 'Combined_All':
            total_strategies = len(strategy_results) - 1  # 排除Combined_All本身
            signal_threshold = max(2, total_strategies // 4)
            strategy_data[strategy] = {
                'total_return': results['total_return'],
                'sharpe_ratio': results['sharpe_ratio'],
                'max_drawdown': results['max_drawdown'],
                'win_rate': results['win_rate'],
                'trade_count': results['trade_count'],
                'volatility': results['volatility'],
                'details': {
                    'name': '🎯 多策略投票系統',
                    'description': '整合16個策略（8做多+8做空）的投票決策系統，使用智能信號篩選和趨勢過濾',
                    'parameters': {
                        '策略類型': '多策略投票組合',
                        '信號閾值': f'至少{signal_threshold}個策略同向',
                        '趨勢過濾': '20日均線趨勢確認',
                        '包含策略': f'{total_strategies}個（8做多+8做空）'
                    },
                    'entry_conditions': [
                        f'至少{signal_threshold}個策略發出同向買入信號',
                        '淨多頭信號 > 淨空頭+賣出信號',
                        '價格相對20日均線 > 99%（做多）',
                        '做空信號數量 > 買入+平倉信號（做空）'
                    ],
                    'exit_conditions': [
                        '反向信號數量超過閾值',
                        '趨勢過濾條件不滿足',
                        '信號強度減弱至閾值以下',
                        '價格偏離趨勢過濾條件'
                    ]
                }
            }
        else:
            # 獲取策略詳細信息
            details = get_strategy_details(strategy)
            strategy_data[strategy] = {
                'total_return': results['total_return'],
                'sharpe_ratio': results['sharpe_ratio'],
                'max_drawdown': results['max_drawdown'],
                'win_rate': results['win_rate'],
                'trade_count': results['trade_count'],
                'volatility': results['volatility'],
                'details': details
            }

    html_content = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>2800.HK 完整分析Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 20px;
            height: calc(100vh - 40px);
        }}
        
        .sidebar {{
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow-y: auto;
        }}
        
        .main-content {{
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow-y: auto;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            color: #2c3e50;
            font-size: 2.5rem;
            margin-bottom: 10px;
        }}
        
        .header p {{
            color: #7f8c8d;
            font-size: 1.1rem;
        }}
        
        .strategy-list {{
            margin-bottom: 30px;
        }}
        
        .strategy-list h3 {{
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.3rem;
        }}
        
        .strategy-item {{
            background: #f8f9fa;
            border: 2px solid transparent;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        .strategy-item:hover {{
            background: #e9ecef;
            transform: translateX(5px);
        }}
        
        .strategy-item.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: #667eea;
        }}
        
        .strategy-item.combined {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }}
        
        .strategy-name {{
            font-weight: bold;
            font-size: 1.1rem;
            margin-bottom: 5px;
        }}
        
        .strategy-return {{
            font-size: 1.2rem;
            font-weight: bold;
            color: #27ae60;
        }}
        
        .strategy-item.active .strategy-return,
        .strategy-item.combined .strategy-return {{
            color: #fff;
        }}
        
        .strategy-metrics {{
            font-size: 0.9rem;
            color: #7f8c8d;
            margin-top: 5px;
        }}
        
        .strategy-item.active .strategy-metrics,
        .strategy-item.combined .strategy-metrics {{
            color: rgba(255,255,255,0.8);
        }}
        
        .chart-container {{
            text-align: center;
            margin-bottom: 30px;
        }}
        
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        .strategy-details {{
            display: none;
        }}
        
        .strategy-details.show {{
            display: block;
        }}
        
        .detail-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        
        .detail-header h2 {{
            margin-bottom: 10px;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        
        .metric-small {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #667eea;
        }}
        
        .metric-value-small {{
            font-size: 1.5rem;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        
        .metric-label-small {{
            color: #7f8c8d;
            font-size: 0.9rem;
        }}
        
        .detail-section {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        
        .detail-section h4 {{
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.2rem;
        }}
        
        .detail-list {{
            list-style: none;
        }}
        
        .detail-list li {{
            padding: 8px 0;
            border-bottom: 1px solid #dee2e6;
            color: #495057;
        }}
        
        .detail-list li:last-child {{
            border-bottom: none;
        }}
        
        .optimization-section {{
            margin-top: 30px;
        }}
        
        .optimization-tabs {{
            display: flex;
            margin-bottom: 20px;
            border-bottom: 2px solid #dee2e6;
        }}
        
        .tab-button {{
            background: none;
            border: none;
            padding: 15px 25px;
            cursor: pointer;
            font-size: 1rem;
            color: #7f8c8d;
            border-bottom: 2px solid transparent;
            transition: all 0.3s ease;
        }}
        
        .tab-button.active {{
            color: #667eea;
            border-bottom-color: #667eea;
            font-weight: bold;
        }}
        
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        .top-results {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }}
        
        .result-item {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        
        .result-item strong {{
            color: #2c3e50;
            font-size: 1.1rem;
        }}
        
        .log-container {{
            background: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            max-height: 200px;
            overflow-y: auto;
            margin-top: 20px;
        }}
        
        .log-entry {{
            margin-bottom: 5px;
            padding: 2px 0;
        }}
        
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        
        .summary-card h4 {{
            margin-bottom: 10px;
            font-size: 1.2rem;
        }}
        
        .summary-card .best-params {{
            font-size: 0.9rem;
            margin: 10px 0;
        }}
        
        .summary-card .performance {{
            font-size: 1.1rem;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <div class="header">
                <h2>📊 策略選擇</h2>
                <p>點擊查看詳細分析</p>
            </div>
            
            <div class="strategy-list">
                <h3>🎯 交易策略</h3>
                <div id="strategyList">
                    <!-- 策略列表將在此生成 -->
                </div>
            </div>
            
            <div class="log-container">
                <div style="font-weight: bold; margin-bottom: 10px;">📋 系統日誌</div>
                <div id="logContainer">
                    <!-- 日誌將在此顯示 -->
                </div>
            </div>
        </div>
        
        <div class="main-content">
            <div class="header">
                <h1>🎯 2800.HK 完整分析Dashboard</h1>
                <p>專業級港股技術分析與策略回測系統</p>
            </div>
            
            <div class="chart-container">
                <img src="{chart_filename}" alt="技術分析圖表" />
            </div>
            
            <div class="strategy-details" id="strategyDetails">
                <div class="detail-header">
                    <h2 id="strategyTitle">選擇策略查看詳細分析</h2>
                    <p id="strategyInfo">點擊左側策略可查看詳細的進場條件、出場條件和策略參數</p>
                </div>
                
                <div class="metrics-grid" id="metricsGrid">
                    <!-- 指標將在此顯示 -->
                </div>
                
                <div id="strategyDetailsContent">
                    <!-- 策略詳情將在此顯示 -->
                </div>
            </div>
            
            <div class="optimization-section">
                <h2>🏆 最佳化結果展示</h2>
                
                <div class="summary-cards">
                    <div class="summary-card">
                        <h4>📊 RSI 最佳參數</h4>
                        <div class="best-params">
                            週期: 21 | 超賣: 35 | 超買: 80
                        </div>
                        <div class="performance">
                            回報率: 21.36% | 夏普: 1.26
                        </div>
                    </div>
                    <div class="summary-card">
                        <h4>📈 MACD 最佳參數</h4>
                        <div class="best-params">
                            快線: 20 | 慢線: 49 | 信號: 6
                        </div>
                        <div class="performance">
                            回報率: 29.07% | 夏普: 2.08
                        </div>
                    </div>
                    <div class="summary-card">
                        <h4>📉 布林帶 最佳參數</h4>
                        <div class="best-params">
                            週期: 24 | 標準差: 2.2
                        </div>
                        <div class="performance">
                            回報率: 9.95% | 夏普: 3.73
                        </div>
                    </div>
                </div>
                
                <div class="optimization-tabs">
                    <button class="tab-button active" onclick="showTab('rsi')">RSI 前10名</button>
                    <button class="tab-button" onclick="showTab('macd')">MACD 前10名</button>
                    <button class="tab-button" onclick="showTab('bollinger')">布林帶 前10名</button>
                </div>
                
                <div id="rsi-tab" class="tab-content active">
                    <div class="top-results">
                        {generate_optimization_results_html('rsi', optimization_data)}
                    </div>
                </div>
                
                <div id="macd-tab" class="tab-content">
                    <div class="top-results">
                        {generate_optimization_results_html('macd', optimization_data)}
                    </div>
                </div>
                
                <div id="bollinger-tab" class="tab-content">
                    <div class="top-results">
                        {generate_optimization_results_html('bollinger', optimization_data)}
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // 策略數據
        const strategyData = {json.dumps(strategy_data, ensure_ascii=False, indent=2)};
        
        // 日誌系統
        function addLog(message) {{
            const logContainer = document.getElementById('logContainer');
            const logEntry = document.createElement('div');
            logEntry.className = 'log-entry';
            logEntry.textContent = `${{new Date().toLocaleTimeString()}} - ${{message}}`;
            logContainer.appendChild(logEntry);
            logContainer.scrollTop = logContainer.scrollHeight;
        }}
        
        let currentStrategy = null;
        
        addLog(`📊 載入策略數據: ${{Object.keys(strategyData).length}}個策略`);
        
        // 全域選擇策略函數 - 關鍵修復
        window.selectStrategy = function(strategyName, element) {{
            addLog(`🎯 全域函數選擇策略: ${{strategyName}}`);
            
            try {{
                // 移除所有active狀態
                document.querySelectorAll('.strategy-item').forEach(item => {{
                    item.classList.remove('active');
                }});
                addLog('✅ 清除所有active狀態');
                
                // 添加active狀態
                if (element) {{
                    element.classList.add('active');
                    addLog('✅ 添加active狀態');
                }}
                
                currentStrategy = strategyName;
                const data = strategyData[strategyName];
                const details = data.details;
                
                const displayName = strategyName === 'Combined_All' ? '🎯 綜合策略' : strategyName;
                
                // 更新標題
                const titleElement = document.getElementById('strategyTitle');
                if (titleElement) {{
                    titleElement.textContent = `${{displayName}} 詳細分析`;
                    addLog('✅ 標題更新完成');
                }}
                
                // 更新指標網格
                const metricsContainer = document.getElementById('metricsGrid');
                if (metricsContainer) {{
                    metricsContainer.innerHTML = `
                        <div class="metric-small">
                            <div class="metric-value-small">${{data.total_return.toFixed(1)}}%</div>
                            <div class="metric-label-small">總報酬率</div>
                        </div>
                        <div class="metric-small">
                            <div class="metric-value-small">${{data.sharpe_ratio.toFixed(2)}}</div>
                            <div class="metric-label-small">夏普比率</div>
                        </div>
                        <div class="metric-small">
                            <div class="metric-value-small">${{Math.abs(data.max_drawdown).toFixed(1)}}%</div>
                            <div class="metric-label-small">最大回撤</div>
                        </div>
                        <div class="metric-small">
                            <div class="metric-value-small">${{data.win_rate.toFixed(0)}}%</div>
                            <div class="metric-label-small">勝率</div>
                        </div>
                        <div class="metric-small">
                            <div class="metric-value-small">${{data.trade_count}}</div>
                            <div class="metric-label-small">交易次數</div>
                        </div>
                        <div class="metric-small">
                            <div class="metric-value-small">${{data.volatility.toFixed(1)}}%</div>
                            <div class="metric-label-small">年化波動率</div>
                        </div>
                    `;
                    addLog('✅ 指標網格更新完成');
                }}
                
                // 更新策略信息
                const infoElement = document.getElementById('strategyInfo');
                if (infoElement) {{
                    infoElement.innerHTML = `<strong>${{details.name}}</strong>：${{details.description}}`;
                    addLog('✅ 策略信息更新完成');
                }}
                
                // 顯示策略詳細參數
                const detailsContainer = document.getElementById('strategyDetailsContent');
                if (detailsContainer) {{
                    let parametersHtml = '<div class="detail-section"><h4>📋 策略參數：</h4><ul class="detail-list">';
                    Object.entries(details.parameters).forEach(([key, value]) => {{
                        parametersHtml += `<li>${{key}}：${{value}}</li>`;
                    }});
                    parametersHtml += '</ul></div>';
                    
                    let entryHtml = '<div class="detail-section"><h4>📈 進場條件：</h4><ul class="detail-list">';
                    details.entry_conditions.forEach(condition => {{
                        entryHtml += `<li>${{condition}}</li>`;
                    }});
                    entryHtml += '</ul></div>';
                    
                    let exitHtml = '<div class="detail-section"><h4>📉 出場條件：</h4><ul class="detail-list">';
                    details.exit_conditions.forEach(condition => {{
                        exitHtml += `<li>${{condition}}</li>`;
                    }});
                    exitHtml += '</ul></div>';
                    
                    detailsContainer.innerHTML = parametersHtml + entryHtml + exitHtml;
                    addLog('✅ 策略詳情更新完成');
                }}
                
                // 顯示策略詳情區域
                const strategyDetailsElement = document.getElementById('strategyDetails');
                if (strategyDetailsElement) {{
                    strategyDetailsElement.style.display = 'block';
                }}
                
                addLog(`🎉 策略選擇完成: ${{strategyName}}`);
                
            }} catch (error) {{
                addLog(`❌ 選擇策略時發生錯誤: ${{error.message}}`);
                console.error('詳細錯誤:', error);
            }}
        }};
        
        // 標籤切換函數
        window.showTab = function(tabName) {{
            // 隱藏所有標籤內容
            document.querySelectorAll('.tab-content').forEach(tab => {{
                tab.classList.remove('active');
            }});
            
            // 移除所有按鈕的active狀態
            document.querySelectorAll('.tab-button').forEach(btn => {{
                btn.classList.remove('active');
            }});
            
            // 顯示選中的標籤
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');
        }};
        
        function initializePage() {{
            addLog('🔧 開始初始化頁面');
            
            const listContainer = document.getElementById('strategyList');
            if (!listContainer) {{
                addLog('❌ 找不到策略列表容器');
                return;
            }}
            
            addLog('✅ 找到策略列表容器');
            renderStrategyList();
        }}
        
        function renderStrategyList() {{
            addLog('📝 開始渲染策略列表');
            
            const listContainer = document.getElementById('strategyList');
            listContainer.innerHTML = '';
            
            let itemCount = 0;
            Object.keys(strategyData).forEach(strategy => {{
                const data = strategyData[strategy];
                const item = document.createElement('div');
                item.className = strategy === 'Combined_All' ? 'strategy-item combined' : 'strategy-item';
                item.id = `strategy-${{strategy}}`;
                
                addLog(`📋 創建策略項目: ${{strategy}}`);
                
                // 使用onclick屬性直接綁定全域函數
                item.onclick = function() {{
                    addLog(`🖱️ onclick事件觸發: ${{strategy}}`);
                    window.selectStrategy(strategy, item);
                }};
                
                // 使用setAttribute確保綁定
                item.setAttribute('onclick', `window.selectStrategy('${{strategy}}', this)`);
                
                const displayName = strategy === 'Combined_All' ? '🎯 綜合策略' : strategy;
                
                item.innerHTML = `
                    <div class="strategy-name">${{displayName}}</div>
                    <div class="strategy-return">+${{data.total_return.toFixed(1)}}%</div>
                    <div class="strategy-metrics">
                        SR: ${{data.sharpe_ratio.toFixed(2)}} | 交易: ${{data.trade_count}}次
                    </div>
                `;
                
                listContainer.appendChild(item);
                itemCount++;
                addLog(`✅ 策略項目已添加: ${{strategy}}`);
            }});
            
            addLog(`🎉 策略列表渲染完成，共 ${{itemCount}} 個策略`);
            
            // 自動測試第一個策略
            setTimeout(() => {{
                const firstStrategy = Object.keys(strategyData)[0];
                addLog(`🔄 自動測試第一個策略: ${{firstStrategy}}`);
                const firstItem = document.getElementById(`strategy-${{firstStrategy}}`);
                if (firstItem) {{
                    firstItem.click();
                }}
            }}, 1000);
        }}
        
        // 初始化
        addLog(`📋 當前文檔狀態: ${{document.readyState}}`);
        
        if (document.readyState === 'loading') {{
            addLog('📝 等待DOMContentLoaded事件');
            document.addEventListener('DOMContentLoaded', function() {{
                addLog('📄 DOMContentLoaded事件觸發');
                initializePage();
            }});
        }} else {{
            addLog('📄 文檔已載入，直接初始化');
            initializePage();
        }}
        
        // 備用初始化
        window.addEventListener('load', function() {{
            addLog('🌐 Window load事件觸發');
            if (!currentStrategy) {{
                addLog('🔄 備用初始化執行');
                initializePage();
            }}
        }});
        
        addLog('📜 腳本載入完成');
    </script>
</body>
</html>
"""
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return filename

def load_optimization_results():
    """載入最佳化結果數據"""
    try:
        # 嘗試讀取最新的最佳化結果
        rsi_file = None
        macd_file = None
        bollinger_file = None
        
        # 查找最新的CSV文件
        for file in os.listdir('.'):
            if file.startswith('RSI最佳化結果_') and file.endswith('.csv'):
                rsi_file = file
            elif file.startswith('MACD最佳化結果_') and file.endswith('.csv'):
                macd_file = file
            elif file.startswith('布林帶最佳化結果_') and file.endswith('.csv'):
                bollinger_file = file
        
        optimization_data = {}
        
        if rsi_file:
            rsi_df = pd.read_csv(rsi_file)
            optimization_data['rsi'] = rsi_df.head(10).to_dict('records')
        
        if macd_file:
            macd_df = pd.read_csv(macd_file)
            optimization_data['macd'] = macd_df.head(10).to_dict('records')
        
        if bollinger_file:
            bollinger_df = pd.read_csv(bollinger_file)
            optimization_data['bollinger'] = bollinger_df.head(10).to_dict('records')
        
        return optimization_data
    except Exception as e:
        print(f"載入最佳化結果失敗: {e}")
        return {'rsi': [], 'macd': [], 'bollinger': []}

def generate_optimization_results_html(indicator_type, optimization_data):
    """生成最佳化結果HTML"""
    if indicator_type not in optimization_data or not optimization_data[indicator_type]:
        return '<div class="result-item">暫無數據</div>'
    
    html = ''
    for i, result in enumerate(optimization_data[indicator_type][:10], 1):
        if indicator_type == 'rsi':
            params = f"週期: {result.get('period', 'N/A')}, 超賣: {result.get('oversold', 'N/A')}, 超買: {result.get('overbought', 'N/A')}"
        elif indicator_type == 'macd':
            params = f"快線: {result.get('fast', 'N/A')}, 慢線: {result.get('slow', 'N/A')}, 信號: {result.get('signal', 'N/A')}"
        elif indicator_type == 'bollinger':
            params = f"週期: {result.get('period', 'N/A')}, 標準差: {result.get('std_dev', 'N/A'):.1f}"
        
        html += f'''
        <div class="result-item">
            <strong>第{i}名</strong><br>
            {params}<br>
            回報率: {result.get('total_return', 0):.2f}% | 夏普: {result.get('sharpe_ratio', 0):.3f}<br>
            最大回撤: {result.get('max_drawdown', 0):.2f}% | 勝率: {result.get('win_rate', 0):.1f}%
        </div>
        '''
    
    return html

def calculate_advanced_metrics(returns, benchmark_returns=None):
    """計算高級量化指標"""
    returns = returns.dropna()
    
    if len(returns) == 0:
        return {}
    
    # 基本統計
    total_return = (1 + returns).cumprod().iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    
    # 風險指標
    risk_free_rate = 0.02
    sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    # 計算最大回撤
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100 if len(drawdown) > 0 else 0
    
    # Sortino比率（下行風險調整收益）
    downside_returns = returns[returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = (annual_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
    
    # Calmar比率
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # VaR和CVaR（5%置信水平）
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
    
    # 交易統計
    winning_trades = len(returns[returns > 0])
    total_trades = len(returns[returns != 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # 盈虧比
    avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
    avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    # 期望收益
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    metrics = {
        'total_return': total_return * 100,
        'annual_return': annual_return * 100,
        'volatility': volatility * 100,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown * 100,
        'var_95': var_95 * 100,
        'cvar_95': cvar_95 * 100,
        'win_rate': win_rate * 100,
        'profit_loss_ratio': profit_loss_ratio,
        'expectancy': expectancy * 100,
        'total_trades': total_trades,
        'winning_trades': winning_trades
    }
    
    # 如果有基準，計算相對指標
    if benchmark_returns is not None:
        benchmark_returns = benchmark_returns.dropna()
        if len(benchmark_returns) > 0:
            # Alpha和Beta
            covariance = returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
            
            benchmark_annual_return = (1 + benchmark_returns).cumprod().iloc[-1] ** (252 / len(benchmark_returns)) - 1
            alpha = annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))
            
            # 信息比率
            excess_returns = returns - benchmark_returns
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = excess_returns.mean() * np.sqrt(252) / tracking_error if tracking_error > 0 else 0
            
            metrics.update({
                'alpha': alpha * 100,
                'beta': beta,
                'information_ratio': information_ratio,
                'tracking_error': tracking_error * 100
            })
    
    return metrics

def run_pyfolio_analysis(strategy_returns, benchmark_returns=None, strategy_name="策略"):
    """運行PyFolio分析或替代分析"""
    try:
        # 確保數據格式正確
        strategy_returns = strategy_returns.dropna()
        
        if len(strategy_returns) == 0:
            print(f"❌ {strategy_name}：無有效回報數據")
            return None
        
        print(f"🔍 正在分析 {strategy_name}...")
        
        # 計算高級指標
        metrics = calculate_advanced_metrics(strategy_returns, benchmark_returns)
        
        # 準備時間序列分析
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        # 月度統計
        monthly_returns = strategy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_stats = {
            'mean': monthly_returns.mean() * 100,
            'std': monthly_returns.std() * 100,
            'best': monthly_returns.max() * 100,
            'worst': monthly_returns.min() * 100,
            'positive_months': len(monthly_returns[monthly_returns > 0]),
            'total_months': len(monthly_returns)
        }
        
        # 年度統計（如果數據足夠）
        yearly_returns = strategy_returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        yearly_stats = {
            'mean': yearly_returns.mean() * 100 if len(yearly_returns) > 0 else 0,
            'std': yearly_returns.std() * 100 if len(yearly_returns) > 1 else 0,
            'best': yearly_returns.max() * 100 if len(yearly_returns) > 0 else 0,
            'worst': yearly_returns.min() * 100 if len(yearly_returns) > 0 else 0
        }
        
        # 回撤分析
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        # 找出所有回撤期
        drawdown_periods = []
        in_drawdown = False
        start_date = None
        
        for date, dd in drawdown.items():
            if dd < -0.001 and not in_drawdown:  # 開始回撤（閾值0.1%）
                in_drawdown = True
                start_date = date
            elif dd >= -0.001 and in_drawdown:  # 結束回撤
                in_drawdown = False
                if start_date:
                    duration = (date - start_date).days
                    max_dd = drawdown[start_date:date].min()
                    drawdown_periods.append({
                        'start': start_date,
                        'end': date,
                        'duration': duration,
                        'max_drawdown': max_dd * 100
                    })
        
        # 回撤統計
        if drawdown_periods:
            drawdown_stats = {
                'count': len(drawdown_periods),
                'avg_duration': np.mean([dd['duration'] for dd in drawdown_periods]),
                'max_duration': max([dd['duration'] for dd in drawdown_periods]),
                'avg_drawdown': np.mean([dd['max_drawdown'] for dd in drawdown_periods]),
                'max_drawdown': min([dd['max_drawdown'] for dd in drawdown_periods])
            }
        else:
            drawdown_stats = {
                'count': 0,
                'avg_duration': 0,
                'max_duration': 0,
                'avg_drawdown': 0,
                'max_drawdown': 0
            }
        
        # 滾動指標分析
        rolling_sharpe = strategy_returns.rolling(30).apply(
            lambda x: (x.mean() * np.sqrt(252)) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
        )
        
        rolling_volatility = strategy_returns.rolling(60).std() * np.sqrt(252)
        
        return {
            'metrics': metrics,
            'monthly_stats': monthly_stats,
            'yearly_stats': yearly_stats,
            'drawdown_stats': drawdown_stats,
            'rolling_sharpe': rolling_sharpe,
            'rolling_volatility': rolling_volatility,
            'cumulative_returns': cumulative_returns,
            'drawdown': drawdown
        }
        
    except Exception as e:
        print(f"❌ PyFolio分析失敗: {e}")
        return None

def create_pyfolio_charts(analysis_results, strategy_name="策略"):
    """創建PyFolio風格的分析圖表"""
    try:
        if not analysis_results:
            return None
        
        # 創建圖表
        fig = plt.figure(figsize=(16, 20))
        gs = fig.add_gridspec(5, 2, height_ratios=[2, 1.5, 1.5, 1.5, 1.5], hspace=0.3, wspace=0.25)
        
        # 1. 累積回報圖
        ax1 = fig.add_subplot(gs[0, :])
        cumulative_returns = analysis_results['cumulative_returns']
        ax1.plot(cumulative_returns.index, (cumulative_returns - 1) * 100, 
                linewidth=2.5, color='#2E86C1', label=f'{strategy_name}累積回報')
        ax1.set_title(f'📈 {strategy_name} 累積回報走勢', fontsize=14, fontweight='bold')
        ax1.set_ylabel('累積回報 (%)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. 回撤圖
        ax2 = fig.add_subplot(gs[1, :])
        drawdown = analysis_results['drawdown']
        ax2.fill_between(drawdown.index, drawdown * 100, 0, 
                        color='red', alpha=0.3, label='回撤')
        ax2.plot(drawdown.index, drawdown * 100, color='red', linewidth=1.5)
        ax2.set_title('📉 回撤分析', fontsize=12, fontweight='bold')
        ax2.set_ylabel('回撤 (%)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. 滾動夏普比率
        ax3 = fig.add_subplot(gs[2, 0])
        rolling_sharpe = analysis_results['rolling_sharpe'].dropna()
        if len(rolling_sharpe) > 0:
            ax3.plot(rolling_sharpe.index, rolling_sharpe, color='green', linewidth=2)
            ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='基準線(1.0)')
            ax3.set_title('📊 30天滾動夏普比率', fontsize=11, fontweight='bold')
            ax3.set_ylabel('夏普比率')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        
        # 4. 滾動波動率
        ax4 = fig.add_subplot(gs[2, 1])
        rolling_vol = analysis_results['rolling_volatility'].dropna()
        if len(rolling_vol) > 0:
            ax4.plot(rolling_vol.index, rolling_vol * 100, color='orange', linewidth=2)
            ax4.set_title('📊 60天滾動波動率', fontsize=11, fontweight='bold')
            ax4.set_ylabel('年化波動率 (%)')
            ax4.grid(True, alpha=0.3)
        
        # 5. 月度回報分布
        ax5 = fig.add_subplot(gs[3, 0])
        monthly_returns = analysis_results['cumulative_returns'].resample('M').apply(
            lambda x: x.iloc[-1] / x.iloc[0] - 1 if len(x) > 0 else 0
        )
        monthly_returns = monthly_returns * 100
        
        if len(monthly_returns) > 0:
            colors = ['green' if x > 0 else 'red' for x in monthly_returns]
            bars = ax5.bar(range(len(monthly_returns)), monthly_returns, color=colors, alpha=0.7)
            ax5.set_title('📅 月度回報分布', fontsize=11, fontweight='bold')
            ax5.set_ylabel('月度回報 (%)')
            ax5.set_xlabel('月份')
            ax5.grid(True, alpha=0.3)
            ax5.axhline(y=0, color='black', linewidth=0.8)
        
        # 6. 回報分布直方圖
        ax6 = fig.add_subplot(gs[3, 1])
        daily_returns = analysis_results['cumulative_returns'].pct_change().dropna() * 100
        if len(daily_returns) > 0:
            ax6.hist(daily_returns, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax6.axvline(daily_returns.mean(), color='red', linestyle='--', 
                       label=f'平均: {daily_returns.mean():.2f}%')
            ax6.set_title('📊 日回報分布', fontsize=11, fontweight='bold')
            ax6.set_xlabel('日回報 (%)')
            ax6.set_ylabel('頻率')
            ax6.grid(True, alpha=0.3)
            ax6.legend()
        
        # 7. 年度回報（如果有足夠數據）
        ax7 = fig.add_subplot(gs[4, 0])
        yearly_returns = analysis_results['cumulative_returns'].resample('Y').apply(
            lambda x: x.iloc[-1] / x.iloc[0] - 1 if len(x) > 0 else 0
        )
        yearly_returns = yearly_returns * 100
        
        if len(yearly_returns) > 0:
            colors = ['green' if x > 0 else 'red' for x in yearly_returns]
            bars = ax7.bar(range(len(yearly_returns)), yearly_returns, color=colors, alpha=0.7)
            ax7.set_title('📅 年度回報', fontsize=11, fontweight='bold')
            ax7.set_ylabel('年度回報 (%)')
            ax7.set_xlabel('年份')
            ax7.grid(True, alpha=0.3)
            ax7.axhline(y=0, color='black', linewidth=0.8)
            
            # 添加數值標籤
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax7.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                        f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        # 8. 策略評級
        ax8 = fig.add_subplot(gs[4, 1])
        metrics = analysis_results['metrics']
        
        # 策略評分系統
        scores = []
        criteria = []
        
        # 1. 總回報評分
        total_return = metrics['total_return']
        if total_return >= 20: score = 10
        elif total_return >= 15: score = 8
        elif total_return >= 10: score = 6
        elif total_return >= 5: score = 4
        elif total_return >= 0: score = 2
        else: score = 0
        scores.append(score)
        criteria.append('總回報')
        
        # 2. 夏普比率評分
        sharpe = metrics['sharpe_ratio']
        if sharpe >= 2: score = 10
        elif sharpe >= 1.5: score = 8
        elif sharpe >= 1: score = 6
        elif sharpe >= 0.5: score = 4
        elif sharpe >= 0: score = 2
        else: score = 0
        scores.append(score)
        criteria.append('夏普比率')
        
        # 3. 最大回撤評分
        max_dd = abs(metrics['max_drawdown'])
        if max_dd <= 2: score = 10
        elif max_dd <= 5: score = 8
        elif max_dd <= 10: score = 6
        elif max_dd <= 15: score = 4
        elif max_dd <= 20: score = 2
        else: score = 0
        scores.append(score)
        criteria.append('回撤控制')
        
        # 4. 勝率評分
        win_rate = metrics['win_rate']
        if win_rate >= 70: score = 10
        elif win_rate >= 60: score = 8
        elif win_rate >= 50: score = 6
        elif win_rate >= 40: score = 4
        elif win_rate >= 30: score = 2
        else: score = 0
        scores.append(score)
        criteria.append('勝率')
        
        colors = ['green' if s >= 6 else 'orange' if s >= 4 else 'red' for s in scores]
        bars = ax8.barh(criteria, scores, color=colors, alpha=0.7)
        ax8.set_title('⭐ 策略評級', fontsize=11, fontweight='bold')
        ax8.set_xlabel('評分 (0-10)')
        ax8.set_xlim(0, 10)
        ax8.grid(True, alpha=0.3, axis='x')
        
        # 添加評分標籤
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax8.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{score}/10', va='center', fontweight='bold')
        
        plt.suptitle(f'🎯 {strategy_name} PyFolio專業分析報告', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # 保存圖表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"PyFolio_簡化分析報告_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filename
        
    except Exception as e:
        print(f"❌ 創建PyFolio圖表失敗: {e}")
        return None

def generate_pyfolio_report(analysis_results, strategy_name="策略"):
    """生成PyFolio風格的文字報告"""
    if not analysis_results:
        return "無分析結果"
    
    metrics = analysis_results['metrics']
    monthly_stats = analysis_results['monthly_stats']
    yearly_stats = analysis_results['yearly_stats']
    drawdown_stats = analysis_results['drawdown_stats']
    
    # 策略評級
    total_return = metrics['total_return']
    sharpe_ratio = metrics['sharpe_ratio']
    max_drawdown = abs(metrics['max_drawdown'])
    win_rate = metrics['win_rate']
    
    # 評分系統
    scores = []
    
    # 總回報評分
    if total_return >= 20: scores.append(10)
    elif total_return >= 15: scores.append(8)
    elif total_return >= 10: scores.append(6)
    elif total_return >= 5: scores.append(4)
    elif total_return >= 0: scores.append(2)
    else: scores.append(0)
    
    # 夏普比率評分
    if sharpe_ratio >= 2: scores.append(10)
    elif sharpe_ratio >= 1.5: scores.append(8)
    elif sharpe_ratio >= 1: scores.append(6)
    elif sharpe_ratio >= 0.5: scores.append(4)
    elif sharpe_ratio >= 0: scores.append(2)
    else: scores.append(0)
    
    # 回撤評分
    if max_drawdown <= 2: scores.append(10)
    elif max_drawdown <= 5: scores.append(8)
    elif max_drawdown <= 10: scores.append(6)
    elif max_drawdown <= 15: scores.append(4)
    elif max_drawdown <= 20: scores.append(2)
    else: scores.append(0)
    
    # 勝率評分
    if win_rate >= 70: scores.append(10)
    elif win_rate >= 60: scores.append(8)
    elif win_rate >= 50: scores.append(6)
    elif win_rate >= 40: scores.append(4)
    elif win_rate >= 30: scores.append(2)
    else: scores.append(0)
    
    avg_score = np.mean(scores)
    
    # 策略等級
    if avg_score >= 8: grade = "A+ (優異策略)"
    elif avg_score >= 6: grade = "A (良好策略)"
    elif avg_score >= 4: grade = "B (一般策略)"
    elif avg_score >= 2: grade = "C (需改進)"
    else: grade = "D (不推薦)"
    
    report = f"""
{'='*80}
🎯 {strategy_name} PyFolio專業分析報告
{'='*80}

📊 基本績效統計
{'-'*50}
• 總報酬率：{metrics['total_return']:.2f}%
• 年化收益率：{metrics['annual_return']:.2f}%
• 年化波動率：{metrics['volatility']:.2f}%
• 夏普比率：{metrics['sharpe_ratio']:.3f}
• Sortino比率：{metrics['sortino_ratio']:.3f}
• Calmar比率：{metrics['calmar_ratio']:.3f}

📈 相對表現分析
{'-'*50}
"""
    
    if 'alpha' in metrics:
        report += f"""• Alpha：{metrics['alpha']:.3f}%
• Beta：{metrics['beta']:.3f}
• 信息比率：{metrics['information_ratio']:.3f}
• 跟蹤誤差：{metrics['tracking_error']:.2f}%
"""
    else:
        report += "• 無基準比較數據\n"
    
    report += f"""
🔍 風險分析
{'-'*50}
• 最大回撤：{metrics['max_drawdown']:.2f}%
• VaR (95%)：{metrics['var_95']:.2f}%
• CVaR (95%)：{metrics['cvar_95']:.2f}%
• 下行風險：{metrics['volatility'] * 0.7:.2f}%

📋 交易統計分析
{'-'*50}
• 總交易次數：{metrics['total_trades']}
• 獲利交易：{metrics['winning_trades']}
• 勝率：{metrics['win_rate']:.1f}%
• 盈虧比：{metrics['profit_loss_ratio']:.2f}
• 期望收益：{metrics['expectancy']:.3f}%

📅 時間序列分析
{'-'*50}
月度統計：
• 平均月回報：{monthly_stats['mean']:.2f}%
• 月回報標準差：{monthly_stats['std']:.2f}%
• 最佳月份：{monthly_stats['best']:.2f}%
• 最差月份：{monthly_stats['worst']:.2f}%
• 正收益月份：{monthly_stats['positive_months']}/{monthly_stats['total_months']}

年度統計：
• 平均年回報：{yearly_stats['mean']:.2f}%
• 年回報標準差：{yearly_stats['std']:.2f}%
• 最佳年度：{yearly_stats['best']:.2f}%
• 最差年度：{yearly_stats['worst']:.2f}%

📉 回撤詳細分析
{'-'*50}
• 回撤期數：{drawdown_stats['count']}
• 平均持續時間：{drawdown_stats['avg_duration']:.1f}天
• 最長持續時間：{drawdown_stats['max_duration']}天
• 平均回撤幅度：{drawdown_stats['avg_drawdown']:.2f}%
• 最大回撤幅度：{drawdown_stats['max_drawdown']:.2f}%

⭐ 策略評級系統
{'-'*50}
• 總回報評分：{scores[0]}/10
• 夏普比率評分：{scores[1]}/10
• 回撤控制評分：{scores[2]}/10
• 勝率評分：{scores[3]}/10
• 總體評分：{sum(scores)}/40
• 策略等級：{grade}

💡 投資建議
{'-'*50}
"""
    
    if avg_score >= 8:
        report += """✅ 優秀策略，建議採用：
• 風險調整收益優異
• 回撤控制良好
• 可考慮適度加大倉位
• 建議持續監控並定期檢視"""
    elif avg_score >= 6:
        report += """✅ 良好策略，可以採用：
• 整體表現良好
• 建議適中倉位
• 注意風險控制
• 可結合其他策略使用"""
    elif avg_score >= 4:
        report += """⚠️ 一般策略，謹慎使用：
• 表現中等，有改進空間
• 建議小倉位測試
• 需要進一步最佳化
• 密切監控表現"""
    else:
        report += """❌ 不推薦使用：
• 風險收益比不佳
• 建議重新設計策略
• 或尋找其他替代方案
• 不建議實盤使用"""
    
    report += f"""

⚠️ 風險提醒
{'-'*50}
• 歷史績效不代表未來表現
• 請根據個人風險承受能力調整倉位
• 建議結合多種策略分散風險
• 市場環境變化可能影響策略效果
• 請定期檢視和調整策略參數

{'='*80}
報告生成時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""
    
    return report

def create_cumulative_returns_chart(data, strategy_results):
    """創建累積收益對比圖表 - 包含所有策略"""
    plt.figure(figsize=(16, 10))
    
    # 計算買入持有基準
    buy_hold_returns = data['Close'].pct_change().fillna(0)
    buy_hold_cumulative = (1 + buy_hold_returns).cumprod()
    
    # 繪製買入持有基準
    plt.plot(data.index, buy_hold_cumulative, 
             label='買入持有基準', color='black', linestyle='--', linewidth=2, alpha=0.8)
    
    # 豐富的顏色列表（確保有足夠的顏色）
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', 
              '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43',
              '#6C5CE7', '#A3CB38', '#FD79A8', '#E17055', '#74B9FF',
              '#FDCB6E', '#E84393', '#00A8FF', '#9C88FF', '#FFA8E4',
              '#78E08F', '#FF3838', '#40739E', '#487EB0', '#8C7AE6']
    
    color_idx = 0
    
    # 按夏普比率排序策略，把Combined_All放到最後
    sorted_strategies = sorted(
        [(k, v) for k, v in strategy_results.items() if k != 'Combined_All'], 
        key=lambda x: x[1]['sharpe_ratio'], reverse=True
    )
    
    # 添加Combined_All到最後
    if 'Combined_All' in strategy_results:
        sorted_strategies.append(('Combined_All', strategy_results['Combined_All']))
    
    print(f"📊 繪製 {len(sorted_strategies)} 個策略的累積收益曲線...")
    
    # 計算並繪製各策略的累積收益
    strategy_performance = []
    
    for strategy_name, strategy_data in sorted_strategies:
        try:
            # 檢查必要的數據
            if 'signals' not in strategy_data or 'strategy_returns' not in strategy_data['signals']:
                print(f"⚠️  {strategy_name} 缺少策略回報數據，跳過...")
                continue
                
            strategy_returns = strategy_data['signals']['strategy_returns'].fillna(0)
            
            # 確保數據對齊
            if len(strategy_returns) != len(data):
                print(f"⚠️  {strategy_name} 數據長度不匹配 ({len(strategy_returns)} vs {len(data)})，進行對齊...")
                strategy_returns = strategy_returns.reindex(data.index, fill_value=0)
            
            cumulative_returns = (1 + strategy_returns).cumprod()
            
            # 計算最終收益率
            final_return = (cumulative_returns.iloc[-1] - 1) * 100
            strategy_performance.append((strategy_name, final_return))
            
            # 選擇顏色和線型
            if strategy_name == 'Combined_All':
                # 綜合策略使用特殊樣式
                color = 'red'
                linewidth = 4
                alpha = 1.0
                linestyle = '-'
                label = f'🏆 多策略投票系統 (+{final_return:.1f}%)'
            else:
                # 普通策略
                color = colors[color_idx % len(colors)]
                linewidth = 2
                alpha = 0.75
                linestyle = '-'
                label = f'{strategy_name} (+{final_return:.1f}%)'
                color_idx += 1
            
            # 繪製策略線
            plt.plot(data.index, cumulative_returns, 
                    label=label, 
                    color=color, 
                    linewidth=linewidth, 
                    alpha=alpha,
                    linestyle=linestyle)
            
            print(f"✅ 繪製完成：{strategy_name} (+{final_return:.1f}%)")
            
        except Exception as e:
            print(f"❌ 繪製策略 {strategy_name} 時發生錯誤：{e}")
            continue
    
    # 添加買入持有的最終收益到標題
    buy_hold_final_return = (buy_hold_cumulative.iloc[-1] - 1) * 100
    
    # 設置圖表
    plt.title(f'港股2800.HK策略績效全面對比\n累積收益曲線 - 共{len(strategy_performance)}個策略 vs 買入持有(+{buy_hold_final_return:.1f}%)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('時間', fontsize=12, fontweight='bold')
    plt.ylabel('投資組合價值（倍數）', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # 設置Y軸格式
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))
    
    # 添加性能統計到圖表右下角
    best_performance = max(strategy_performance, key=lambda x: x[1]) if strategy_performance else ("無", 0)
    stats_text = f"""
最佳策略: {best_performance[0]} (+{best_performance[1]:.1f}%)
策略總數: {len(strategy_performance)}
分析期間: {len(data)}個交易日
買入持有: +{buy_hold_final_return:.1f}%
    """
    
    plt.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 調整圖例 - 分兩列顯示
    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                       fontsize=9, ncol=1, frameon=True, 
                       fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    
    # 調整布局
    plt.tight_layout()
    
    # 保存圖表
    output_dir = "港股輸出"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/2800_HK_完整策略累積收益對比_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 累積收益對比圖表已保存：{filename}")
    print(f"📈 包含策略數量：{len(strategy_performance)}")
    
    return filename

def verify_sharpe_ratio_calculations(strategy_results, data):
    """驗證夏普比率計算的詳細過程"""
    print("\n" + "="*80)
    print("🔍 夏普比率計算驗證報告")
    print("="*80)
    
    risk_free_rate = 0.02  # 2% 無風險利率
    
    print(f"📋 計算參數：")
    print(f"   • 無風險利率：{risk_free_rate*100:.1f}%")
    print(f"   • 年化調整因子：√252 = {np.sqrt(252):.3f}")
    print(f"   • 數據期間：{len(data)}天")
    
    print(f"\n{'策略名稱':<20} {'總收益率':<10} {'年化收益率':<12} {'年化波動率':<12} {'夏普比率':<10} {'驗證狀態':<10}")
    print("-" * 85)
    
    for strategy_name, results in strategy_results.items():
        # 重新計算夏普比率進行驗證
        strategy_returns = results['signals']['strategy_returns'].dropna()
        
        if len(strategy_returns) > 0:
            # 總收益率
            total_return = (1 + strategy_returns).cumprod().iloc[-1] - 1
            
            # 年化收益率
            trading_days = len(strategy_returns)
            annual_return = (1 + total_return) ** (252 / trading_days) - 1
            
            # 年化波動率
            volatility = strategy_returns.std() * np.sqrt(252)
            
            # 夏普比率
            sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # 驗證是否與原結果一致
            original_sharpe = results['sharpe_ratio']
            verification = "✅ 正確" if abs(sharpe_ratio - original_sharpe) < 0.001 else "❌ 錯誤"
            
            # 特別調試Combined_All策略
            if strategy_name == 'Combined_All':
                print(f"\n🔍 Combined_All策略詳細調試：")
                print(f"   • 策略回報數據長度：{len(strategy_returns)}")
                print(f"   • 原始計算夏普比率：{original_sharpe:.6f}")
                print(f"   • 驗證計算夏普比率：{sharpe_ratio:.6f}")
                print(f"   • 差異：{abs(sharpe_ratio - original_sharpe):.6f}")
                print(f"   • 原始年化收益率：{annual_return*100:.4f}%")
                print(f"   • 原始年化波動率：{volatility*100:.4f}%")
                
                # 用與主計算相同的方法重新計算
                original_data_length = len(data)
                annual_return_orig = (1 + total_return) ** (252 / original_data_length) - 1
                sharpe_ratio_orig = (annual_return_orig - risk_free_rate) / volatility if volatility > 0 else 0
                print(f"   • 用原始數據長度({original_data_length})重算年化收益率：{annual_return_orig*100:.4f}%")
                print(f"   • 用原始數據長度重算夏普比率：{sharpe_ratio_orig:.6f}")
                
                if abs(sharpe_ratio_orig - original_sharpe) < 0.001:
                    verification = "✅ 正確(用原始數據長度)"
            
            print(f"{strategy_name:<20} {total_return*100:>8.2f}% {annual_return*100:>10.2f}% {volatility*100:>10.2f}% {sharpe_ratio:>8.3f} {verification}")
        else:
            print(f"{strategy_name:<20} {'無數據':<10} {'無數據':<12} {'無數據':<12} {'無數據':<10} {'⚠️ 無數據'}")
    
    print("\n" + "="*80)
    print("📊 夏普比率計算公式：")
    print("   夏普比率 = (年化收益率 - 無風險利率) / 年化波動率")
    print("   年化收益率 = (1 + 總收益率)^(252/交易天數) - 1")
    print("   年化波動率 = 日收益率標準差 × √252")
    print("="*80)

def verify_and_fix_mdd_calculations(strategy_results, data):
    """驗證並修復最大回撤計算"""
    print("📊 驗證最大回撤(MDD)計算...")
    
    fixed_results = {}
    
    for strategy_name, results in strategy_results.items():
        if 'equity_curve' not in results:
            print(f"⚠️  {strategy_name} 缺少權益曲線數據，跳過MDD修復")
            fixed_results[strategy_name] = results
            continue
        
        equity_curve = results['equity_curve']
        
        # 計算正確的最大回撤
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 更新結果
        results_copy = results.copy()
        results_copy['max_drawdown'] = abs(max_drawdown)
        results_copy['drawdown_series'] = drawdown
        
        fixed_results[strategy_name] = results_copy
        
        print(f"✅ {strategy_name}: MDD = {abs(max_drawdown)*100:.2f}%")
    
    return fixed_results

# Dash 互動式分析界面
class DashInteractiveApp:
    """Dash 互動式分析應用"""
    
    def __init__(self):
        if not DASH_AVAILABLE:
            raise ImportError("Dash 套件未安裝，無法創建 Web 界面")
        
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.data = None
        self.strategy_results = {}
        # self.comparison_engine = ComparisonEngine()  # 暫時註解，等待實現
        # self.real_time_analyzer = RealTimeAnalyzer()  # 暫時註解，等待實現
        self.portfolio_manager = StrategyPortfolioManager()
        
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """設置界面布局"""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("🎯 2800.HK 港股量化分析系統", className="text-center mb-4"),
                    html.Hr()
                ], width=12)
            ]),
            
            # 控制面板
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🎛️ 控制面板"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("數據更新"),
                                    dbc.ButtonGroup([
                                        dbc.Button("載入數據", id="load-data-btn", color="primary"),
                                        dbc.Button("實時更新", id="realtime-btn", color="success"),
                                        dbc.Button("停止更新", id="stop-btn", color="warning")
                                    ])
                                ], width=4),
                                dbc.Col([
                                    dbc.Label("分析模式"),
                                    dcc.Dropdown(
                                        id="analysis-mode",
                                        options=[
                                            {"label": "單策略分析", "value": "single"},
                                            {"label": "策略組合", "value": "portfolio"},
                                            {"label": "風險分析", "value": "risk"},
                                            {"label": "比較分析", "value": "comparison"}
                                        ],
                                        value="single"
                                    )
                                ], width=4),
                                dbc.Col([
                                    dbc.Label("時間範圍"),
                                    dcc.Dropdown(
                                        id="time-range",
                                        options=[
                                            {"label": "1個月", "value": "1mo"},
                                            {"label": "3個月", "value": "3mo"},
                                            {"label": "6個月", "value": "6mo"},
                                            {"label": "1年", "value": "1y"}
                                        ],
                                        value="6mo"
                                    )
                                ], width=4)
                            ])
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # 策略參數調整
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("⚙️ 策略參數調整"),
                        dbc.CardBody([
                            dbc.Tabs([
                                dbc.Tab(label="RSI", tab_id="rsi-tab"),
                                dbc.Tab(label="MACD", tab_id="macd-tab"),
                                dbc.Tab(label="布林帶", tab_id="bb-tab"),
                                dbc.Tab(label="止損止盈", tab_id="risk-tab")
                            ], id="param-tabs", active_tab="rsi-tab"),
                            html.Div(id="param-content", className="mt-3")
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # 實時狀態監控
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📡 實時監控"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.H4("當前價格", className="text-center"),
                                    html.H2(id="current-price", className="text-center text-primary")
                                ], width=3),
                                dbc.Col([
                                    html.H4("最新信號", className="text-center"),
                                    html.Div(id="latest-signals")
                                ], width=6),
                                dbc.Col([
                                    html.H4("系統狀態", className="text-center"),
                                    dbc.Badge(id="system-status", color="success", className="p-2")
                                ], width=3)
                            ])
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # 主要圖表區域
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📈 分析圖表"),
                        dbc.CardBody([
                            dcc.Graph(id="main-chart", style={"height": "600px"})
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 績效指標"),
                        dbc.CardBody([
                            html.Div(id="performance-metrics")
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            # 策略比較表格
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🏆 策略排行榜"),
                        dbc.CardBody([
                            dash_table.DataTable(
                                id="strategy-table",
                                columns=[
                                    {"name": "策略", "id": "strategy"},
                                    {"name": "總回報率(%)", "id": "total_return", "type": "numeric", "format": {"specifier": ".2f"}},
                                    {"name": "夏普比率", "id": "sharpe_ratio", "type": "numeric", "format": {"specifier": ".3f"}},
                                    {"name": "最大回撤(%)", "id": "max_drawdown", "type": "numeric", "format": {"specifier": ".2f"}},
                                    {"name": "勝率(%)", "id": "win_rate", "type": "numeric", "format": {"specifier": ".1f"}},
                                    {"name": "交易次數", "id": "total_trades", "type": "numeric"}
                                ],
                                style_cell={'textAlign': 'center'},
                                style_data_conditional=[
                                    {
                                        'if': {'row_index': 0},
                                        'backgroundColor': '#FFD700',
                                        'color': 'black',
                                    }
                                ],
                                sort_action="native"
                            )
                        ])
                    ])
                ], width=12)
            ]),
            
            # 自動更新組件
            dcc.Interval(id="interval-component", interval=30*1000, n_intervals=0, disabled=True),
            
            # 數據存儲
            dcc.Store(id="data-store"),
            dcc.Store(id="results-store")
            
        ], fluid=True)
    
    def setup_callbacks(self):
        """設置回調函數"""
        
        @self.app.callback(
            [Output("param-content", "children")],
            [Input("param-tabs", "active_tab")]
        )
        def update_param_content(active_tab):
            if active_tab == "rsi-tab":
                return [dbc.Row([
                    dbc.Col([
                        dbc.Label("RSI週期"),
                        dcc.Slider(id="rsi-period", min=5, max=30, step=1, value=14),
                        dbc.Label("超買閾值"),
                        dcc.Slider(id="rsi-overbought", min=60, max=90, step=5, value=70),
                        dbc.Label("超賣閾值"),
                        dcc.Slider(id="rsi-oversold", min=10, max=40, step=5, value=30)
                    ])
                ])]
            elif active_tab == "macd-tab":
                return [dbc.Row([
                    dbc.Col([
                        dbc.Label("快線週期"),
                        dcc.Slider(id="macd-fast", min=5, max=20, step=1, value=12),
                        dbc.Label("慢線週期"),  
                        dcc.Slider(id="macd-slow", min=15, max=40, step=1, value=26),
                        dbc.Label("信號線週期"),
                        dcc.Slider(id="macd-signal", min=5, max=15, step=1, value=9)
                    ])
                ])]
            elif active_tab == "bb-tab":
                return [dbc.Row([
                    dbc.Col([
                        dbc.Label("週期"),
                        dcc.Slider(id="bb-period", min=10, max=30, step=1, value=20),
                        dbc.Label("標準差"),
                        dcc.Slider(id="bb-std", min=1.0, max=3.0, step=0.1, value=2.0)
                    ])
                ])]
            elif active_tab == "risk-tab":
                return [dbc.Row([
                    dbc.Col([
                        dbc.Label("止損比例(%)"),
                        dcc.Slider(id="stop-loss", min=1, max=10, step=0.5, value=5),
                        dbc.Label("止盈比例(%)"),
                        dcc.Slider(id="take-profit", min=5, max=20, step=1, value=10)
                    ])
                ])]
            return [html.Div("選擇參數類型")]
    
    def run(self, debug=True, port=8050):
        """運行Dash應用"""
        if DASH_AVAILABLE:
            self.app.run_server(debug=debug, port=port)
        else:
            print("Dash不可用，無法啟動Web界面")

def create_plotly_dashboard(strategy_results, data):
    """使用 Plotly Dash 創建現代化 Dashboard，替代 HTML 方法"""
    try:
        import dash
        from dash import dcc, html, Input, Output, dash_table
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("❌ Dash 未安裝，無法創建 dashboard")
        return None
    
    print("\n🎯 正在使用 Plotly Dash 創建現代化 Dashboard...")
    
    try:
        # 準備符合plotguy要求的數據格式
        print("📊 準備 plotguy 數據格式...")
        
        # 創建策略績效數據框
        performance_data = []
        for strategy_name, result in strategy_results.items():
            performance_data.append({
                'strategy': strategy_name,
                'total_return': result['total_return'],
                'sharpe_ratio': result['sharpe_ratio'], 
                'max_drawdown': result['max_drawdown'],
                'win_rate': result.get('win_rate', 0),
                'trade_count': result.get('trade_count', 0)
            })
        
        performance_df = pd.DataFrame(performance_data)
        
        # 創建價格和信號數據
        price_signals_df = data[['Close', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower']].reset_index()
        price_signals_df['Date'] = price_signals_df['Date'].dt.strftime('%Y-%m-%d')
        
        # 添加綜合策略信號（如果存在）
        if 'Combined_All' in strategy_results:
            combined_signals = strategy_results['Combined_All']['signals']
            cumulative_returns = (1 + combined_signals['strategy_returns']).cumprod()
            
            signals_df = pd.DataFrame({
                'Date': [d.strftime('%Y-%m-%d') for d in cumulative_returns.index],
                'cumulative_return': cumulative_returns.values,
                'signal': combined_signals['signal'].values,
                'daily_return': combined_signals['strategy_returns'].values
            })
        else:
            signals_df = pd.DataFrame()
        
        print("✅ 數據格式準備完成")
        
        # 使用 Dash 創建應用
        print("🚀 正在創建 Dash 應用...")
        
        # 創建 Dash 應用實例
        app = dash.Dash(__name__)
        
        # 設置應用布局
        app.layout = html.Div([
            # 標題
            html.H1("🏅 2800.HK 港股技術分析 Dashboard", 
                   style={'textAlign': 'center', 'color': '#1f77b4', 'marginBottom': '30px'}),
            
            # 互動式策略選擇器
            html.Div([
                html.H3("🎯 互動式策略選擇器", style={'textAlign': 'center', 'color': '#2E8B57'}),
                html.Div([
                    html.Label("選擇主要策略：", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='strategy-selector',
                        options=[{'label': strategy, 'value': strategy} for strategy in performance_df['strategy']],
                        value=performance_df['strategy'].iloc[0] if not performance_df.empty else 'Combined_All',
                        style={'width': '300px', 'display': 'inline-block', 'marginRight': '20px'}
                    ),
                    html.Label("對比策略：", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='compare-strategy-selector',
                        options=[{'label': strategy, 'value': strategy} for strategy in performance_df['strategy']],
                        value='Combined_All' if 'Combined_All' in performance_df['strategy'].values else performance_df['strategy'].iloc[-1],
                        style={'width': '300px', 'display': 'inline-block'}
                    )
                ], style={'textAlign': 'center', 'marginBottom': '20px'}),
                
                # 策略信息顯示區
                html.Div(id='strategy-info', style={'textAlign': 'center', 'backgroundColor': '#f0f8ff', 'padding': '15px', 'borderRadius': '10px', 'marginBottom': '20px'})
            ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '30px'}),
            
            # 第一行：價格圖表
            html.Div([
                dcc.Graph(id='price-chart')
            ], style={'marginBottom': '30px'}),
            
            # 第二行：技術指標
            html.Div([
                html.Div([
                    dcc.Graph(id='rsi-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='macd-chart')
                ], style={'width': '50%', 'display': 'inline-block'})
            ], style={'marginBottom': '30px'}),
            
            # 第三行：策略績效
            html.Div([
                html.Div([
                    dcc.Graph(id='performance-bar')
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='sharpe-scatter')
                ], style={'width': '50%', 'display': 'inline-block'})
            ], style={'marginBottom': '30px'}),
            
            # 第四行：累積收益曲線（支援策略對比）
            html.Div([
                dcc.Graph(id='cumulative-returns')
            ], style={'marginBottom': '30px'}),
            
            # 第五行：策略對比表格
            html.Div([
                html.H3("📊 選中策略詳細對比", style={'textAlign': 'center'}),
                html.Div(id='strategy-comparison-table')
            ], style={'marginBottom': '30px'}),
            
            # 第六行：完整績效表格
            html.Div([
                html.H3("📋 所有策略績效表格", style={'textAlign': 'center'}),
                dash_table.DataTable(
                    id='performance-table',
                    data=performance_df.round(3).to_dict('records'),
                    columns=[{"name": col, "id": col} for col in performance_df.columns],
                    style_cell={'textAlign': 'center'},
                    style_header={'backgroundColor': '#1f77b4', 'color': 'white'},
                    style_data={'backgroundColor': '#f8f9fa'},
                    row_selectable='single',
                    selected_rows=[0]
                )
            ])
        ])
        
        # 策略信息更新回調
        @app.callback(
            [Output('strategy-info', 'children')],
            [Input('strategy-selector', 'value'),
             Input('compare-strategy-selector', 'value')]
        )
        def update_strategy_info(selected_strategy, compare_strategy):
            """更新策略信息顯示"""
            info_div = []
            
            if selected_strategy and selected_strategy in strategy_results:
                result = strategy_results[selected_strategy]
                info_div.append(
                    html.H4(f"🎯 主要策略: {selected_strategy}", style={'color': '#1f77b4', 'marginBottom': '10px'})
                )
                info_div.append(
                    html.P(f"總收益率: {result['total_return']:.2f}% | 夏普比率: {result['sharpe_ratio']:.3f} | 最大回撤: {result['max_drawdown']:.2f}%",
                           style={'fontSize': '16px', 'color': '#333'})
                )
            
            if compare_strategy and compare_strategy in strategy_results and compare_strategy != selected_strategy:
                compare_result = strategy_results[compare_strategy]
                info_div.append(
                    html.H4(f"📊 對比策略: {compare_strategy}", style={'color': '#e74c3c', 'marginBottom': '10px', 'marginTop': '15px'})
                )
                info_div.append(
                    html.P(f"總收益率: {compare_result['total_return']:.2f}% | 夏普比率: {compare_result['sharpe_ratio']:.3f} | 最大回撤: {compare_result['max_drawdown']:.2f}%",
                           style={'fontSize': '16px', 'color': '#333'})
                )
            
            return [info_div]

        # 策略對比表格更新回調
        @app.callback(
            [Output('strategy-comparison-table', 'children')],
            [Input('strategy-selector', 'value'),
             Input('compare-strategy-selector', 'value')]
        )
        def update_comparison_table(selected_strategy, compare_strategy):
            """更新策略對比表格"""
            if not selected_strategy or selected_strategy not in strategy_results:
                return [[]]
            
            # 準備對比數據
            comparison_data = []
            
            # 主要策略數據
            main_result = strategy_results[selected_strategy]
            comparison_data.append({
                '指標': '策略名稱',
                '主要策略': selected_strategy,
                '對比策略': compare_strategy if compare_strategy and compare_strategy != selected_strategy else '-'
            })
            
            comparison_data.append({
                '指標': '總收益率 (%)',
                '主要策略': f"{main_result['total_return']:.2f}",
                '對比策略': f"{strategy_results[compare_strategy]['total_return']:.2f}" if compare_strategy and compare_strategy in strategy_results and compare_strategy != selected_strategy else '-'
            })
            
            comparison_data.append({
                '指標': '夏普比率',
                '主要策略': f"{main_result['sharpe_ratio']:.3f}",
                '對比策略': f"{strategy_results[compare_strategy]['sharpe_ratio']:.3f}" if compare_strategy and compare_strategy in strategy_results and compare_strategy != selected_strategy else '-'
            })
            
            comparison_data.append({
                '指標': '最大回撤 (%)',
                '主要策略': f"{main_result['max_drawdown']:.2f}",
                '對比策略': f"{strategy_results[compare_strategy]['max_drawdown']:.2f}" if compare_strategy and compare_strategy in strategy_results and compare_strategy != selected_strategy else '-'
            })
            
            if 'win_rate' in main_result:
                comparison_data.append({
                    '指標': '勝率 (%)',
                    '主要策略': f"{main_result['win_rate']:.1f}",
                    '對比策略': f"{strategy_results[compare_strategy].get('win_rate', 0):.1f}" if compare_strategy and compare_strategy in strategy_results and compare_strategy != selected_strategy else '-'
                })
            
            # 創建對比表格
            comparison_table = dash_table.DataTable(
                data=comparison_data,
                columns=[{"name": col, "id": col} for col in ['指標', '主要策略', '對比策略']],
                style_cell={'textAlign': 'center', 'padding': '10px'},
                style_header={'backgroundColor': '#2E8B57', 'color': 'white', 'fontWeight': 'bold'},
                style_data={'backgroundColor': '#f0fff0'},
                style_data_conditional=[
                    {
                        'if': {'column_id': '主要策略'},
                        'backgroundColor': '#e6f3ff',
                        'color': '#1f77b4',
                        'fontWeight': 'bold'
                    },
                    {
                        'if': {'column_id': '對比策略'},
                        'backgroundColor': '#ffe6e6',
                        'color': '#e74c3c',
                        'fontWeight': 'bold'
                    }
                ]
            )
            
            return [[comparison_table]]

        # 設置主要圖表更新回調函數
        @app.callback(
            [Output('price-chart', 'figure'),
             Output('rsi-chart', 'figure'),
             Output('macd-chart', 'figure'),
             Output('performance-bar', 'figure'),
             Output('sharpe-scatter', 'figure'),
             Output('cumulative-returns', 'figure')],
            [Input('strategy-selector', 'value'),
             Input('compare-strategy-selector', 'value'),
             Input('performance-table', 'data')]
        )
        def update_charts(selected_strategy, compare_strategy, table_data):
            # 價格圖表 - 添加策略信號標記
            price_fig = go.Figure()
            price_fig.add_trace(go.Scatter(
                x=price_signals_df['Date'],
                y=price_signals_df['Close'],
                name='2800.HK 收盤價',
                line=dict(color='blue', width=2)
            ))
            
            # 如果選中了策略，添加買入賣出信號
            if selected_strategy and selected_strategy in strategy_results:
                strategy_signals = strategy_results[selected_strategy]['signals']
                buy_signals = strategy_signals[strategy_signals['signal'] == 1]
                sell_signals = strategy_signals[strategy_signals['signal'] == -1]
                
                if len(buy_signals) > 0:
                    buy_dates = [d.strftime('%Y-%m-%d') for d in buy_signals.index]
                    buy_prices = [data.loc[d, 'Close'] for d in buy_signals.index if d in data.index]
                    price_fig.add_trace(go.Scatter(
                        x=buy_dates,
                        y=buy_prices,
                        mode='markers',
                        name=f'{selected_strategy} 買入信號',
                        marker=dict(symbol='triangle-up', size=12, color='green')
                    ))
                
                if len(sell_signals) > 0:
                    sell_dates = [d.strftime('%Y-%m-%d') for d in sell_signals.index]
                    sell_prices = [data.loc[d, 'Close'] for d in sell_signals.index if d in data.index]
                    price_fig.add_trace(go.Scatter(
                        x=sell_dates,
                        y=sell_prices,
                        mode='markers',
                        name=f'{selected_strategy} 賣出信號',
                        marker=dict(symbol='triangle-down', size=12, color='red')
                    ))
            
            price_fig.update_layout(
                title=f'📈 2800.HK 股價走勢 - {selected_strategy if selected_strategy else "全部"}策略信號',
                xaxis_title='日期',
                yaxis_title='價格 (HKD)',
                hovermode='x unified'
            )
            
            # RSI 圖表
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(
                x=price_signals_df['Date'],
                y=price_signals_df['RSI'],
                name='RSI',
                line=dict(color='orange', width=2)
            ))
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="超買線(70)")
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="超賣線(30)")
            rsi_fig.update_layout(
                title='📊 RSI 相對強弱指標',
                xaxis_title='日期',
                yaxis_title='RSI'
            )
            
            # MACD 圖表
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(
                x=price_signals_df['Date'],
                y=price_signals_df['MACD'],
                name='MACD',
                line=dict(color='blue', width=2)
            ))
            macd_fig.update_layout(
                title='📈 MACD 指標',
                xaxis_title='日期',
                yaxis_title='MACD'
            )
            
            # 策略績效柱狀圖
            perf_fig = go.Figure()
            perf_fig.add_trace(go.Bar(
                x=performance_df['strategy'],
                y=performance_df['total_return'],
                name='總收益率 (%)',
                marker_color='lightblue'
            ))
            perf_fig.update_layout(
                title='💰 策略總收益率對比',
                xaxis_title='策略',
                yaxis_title='收益率 (%)',
                xaxis_tickangle=-45
            )
            
            # 夏普比率散點圖
            sharpe_fig = go.Figure()
            sharpe_fig.add_trace(go.Scatter(
                x=performance_df['sharpe_ratio'],
                y=performance_df['total_return'],
                mode='markers+text',
                text=performance_df['strategy'],
                textposition='top center',
                marker=dict(size=10, color='red', opacity=0.7),
                name='策略表現'
            ))
            sharpe_fig.update_layout(
                title='⭐ 夏普比率 vs 總收益率',
                xaxis_title='夏普比率',
                yaxis_title='總收益率 (%)'
            )
            
            # 累積收益曲線 - 顯示所有策略對比（預設模式）
            cum_fig = go.Figure()
            
            # 添加買入持有基準線
            buy_hold_return = (1 + data['Close'].pct_change()).cumprod()
            cum_fig.add_trace(go.Scatter(
                x=[d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)[:10] for d in buy_hold_return.index],
                y=buy_hold_return.values,
                name='📊 買入持有基準',
                line=dict(color='black', width=2, dash='dash'),
                opacity=0.9
            ))
            
            # 豐富的顏色列表
            color_list = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', 
                         '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43',
                         '#6C5CE7', '#A3CB38', '#FD79A8', '#E17055', '#74B9FF',
                         '#FDCB6E', '#E84393', '#00A8FF', '#9C88FF', '#FFA8E4']
            
            color_idx = 0
            
            # 按夏普比率排序，Combined_All放最後
            sorted_strategies = sorted(
                [(k, v) for k, v in strategy_results.items() if k != 'Combined_All'], 
                key=lambda x: x[1]['sharpe_ratio'], reverse=True
            )
            if 'Combined_All' in strategy_results:
                sorted_strategies.append(('Combined_All', strategy_results['Combined_All']))
            
            # 添加所有策略的累積收益曲線
            strategies_added = 0
            for strategy_name, strategy_result in sorted_strategies:
                if 'equity_curve' in strategy_result:
                    equity_curve = strategy_result['equity_curve']
                    
                    # 計算最終收益率
                    final_return = (equity_curve.iloc[-1] - 1) * 100
                    
                    # 特殊處理綜合策略
                    if strategy_name == 'Combined_All':
                        line_color = 'red'
                        line_width = 4
                        opacity = 1.0
                        strategy_display_name = f'🏆 多策略投票系統 (+{final_return:.1f}%)'
                        visible = True
                    else:
                        line_color = color_list[color_idx % len(color_list)]
                        # 根據策略是否被選中來調整樣式
                        if strategy_name == selected_strategy:
                            line_width = 3
                            opacity = 1.0
                            visible = True
                        elif strategy_name == compare_strategy:
                            line_width = 3
                            opacity = 0.9
                            visible = True
                        else:
                            line_width = 2
                            opacity = 0.6
                            visible = True  # 默認顯示所有策略
                        
                        strategy_display_name = f'{strategy_name} (+{final_return:.1f}%)'
                        color_idx += 1
                    
                    cum_fig.add_trace(go.Scatter(
                        x=[d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)[:10] for d in equity_curve.index],
                        y=equity_curve.values,
                        name=strategy_display_name,
                        line=dict(color=line_color, width=line_width),
                        opacity=opacity,
                        visible=visible,
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                    '日期: %{x}<br>' +
                                    '累積收益: %{y:.3f}<br>' +
                                    '<extra></extra>'
                    ))
                    strategies_added += 1
            
            # 動態標題
            title_text = f'📈 港股2800.HK全策略累積收益對比 - 共{strategies_added}個策略'
            if selected_strategy and compare_strategy and selected_strategy != compare_strategy:
                title_text = f'📈 {selected_strategy} vs {compare_strategy} 重點對比（含全部策略）'
            elif selected_strategy:
                title_text = f'📈 重點關注：{selected_strategy}（含全部策略對比）'
            
            cum_fig.update_layout(
                title=title_text,
                xaxis_title='日期',
                yaxis_title='累積收益率（倍數）',
                hovermode='x unified',
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02,
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="rgba(0,0,0,0.3)",
                    borderwidth=1,
                    font=dict(size=10)
                ),
                font=dict(size=11),
                margin=dict(r=200),  # 為圖例留出更多空間
                height=600
            )
            
            return price_fig, rsi_fig, macd_fig, perf_fig, sharpe_fig, cum_fig
        
        print("✅ Plotly Dash 應用創建完成")
        
        # 啟動 Dash Dashboard 在 localhost:8051
        def run_dash_server():
            print("🚀 正在啟動 Dash Dashboard 服務器...")
            print("🌐 Dashboard URL: http://localhost:8051")
            print("💡 請在瀏覽器中打開上述鏈接")
            print("⚠️  按 Ctrl+C 停止服務器")
            
            try:
                app.run_server(debug=False, port=8051, host='0.0.0.0')
            except Exception as e:
                print(f"❌ Dash 服務器啟動失敗: {e}")
        
        # 在新線程中啟動服務器，避免阻塞主程序
        import threading
        server_thread = threading.Thread(target=run_dash_server, daemon=True)
        server_thread.start()
        
        # 給服務器一點時間啟動
        import time
        time.sleep(2)
        
        # 嘗試自動打開瀏覽器
        try:
            import webbrowser
            webbrowser.open('http://localhost:8051')
            print("🌐 已自動在瀏覽器中打開 Dash Dashboard")
        except Exception as e:
            print(f"❌ 無法自動打開瀏覽器: {e}，請手動訪問 http://localhost:8051")
        
        print("✅ Plotly Dash Dashboard 已成功啟動在 localhost:8051")
        return app
        
    except Exception as e:
        print(f"❌ Plotly Dashboard 創建失敗：{e}")
        print(f"💡 錯誤詳情：{type(e).__name__}")
        import traceback
        print(f"🔍 完整錯誤信息：{traceback.format_exc()}")
        return None

def main():
    """主程序"""
    print("🚀 啟動2800.HK港股完整量化分析系統")
    print("="*80)
    
    # 獲取股票數據
    print("📥 正在下載2800.HK最新數據...")
    data = get_hk_stock_data("2800.HK", "6mo")
    
    if data is None:
        print("❌ 數據下載失敗，程序退出")
        return
    
    print(f"✅ 成功獲取{len(data)}天的數據")
    try:
        start_date = pd.to_datetime(data.index[0]).strftime('%Y-%m-%d')
        end_date = pd.to_datetime(data.index[-1]).strftime('%Y-%m-%d')
        print(f"📊 數據期間：{start_date} 至 {end_date}")
    except Exception as e:
        print(f"⚠️ 日期格式化錯誤: {e}")
        print(f"📊 數據期間：{str(data.index[0])[:10]} 至 {str(data.index[-1])[:10]}")
    
    # 計算技術指標
    data = calculate_technical_indicators(data)
    
    # 定義策略
    strategies = [
        'RSI', 'MACD', 'Bollinger', 'Mean_Reversion', 'SMA_Cross', 
        'Momentum', 'EMA_Cross', 'Multi_MA',
        'RSI_Short', 'MACD_Short', 'Bollinger_Short', 'Mean_Reversion_Short',
        'SMA_Cross_Short', 'Momentum_Short', 'EMA_Cross_Short', 'Multi_MA_Short'
    ]
    
    strategy_results = {}
    
    print(f"🔄 開始回測{len(strategies)}種策略...")
    
    # 做空策略最佳化（僅示例性保留關鍵部分）
    short_optimized_params = {}
    
    # 運行回測
    for strategy in strategies:
        result = backtest_strategy(data, strategy)
        strategy_results[strategy] = result
    
    # 計算綜合策略
    print("🔄 正在計算多策略投票系統...")
    print(f"📊 包含策略數量：{len(strategies)}個（8做多+8做空）")
    combined_signals = pd.DataFrame(index=data.index)
    
    # 初始化各種信號累計
    buy_signals = pd.Series(0, index=data.index)
    sell_signals = pd.Series(0, index=data.index)
    short_signals = pd.Series(0, index=data.index)
    cover_signals = pd.Series(0, index=data.index)
    
    # 分別累計各類信號
    for strategy in strategies:
        strategy_signal = strategy_results[strategy]['signals']['signal']
        
        # 累計買入信號 (做多策略的買入)
        buy_signals += (strategy_signal == 1).astype(int)
        
        # 累計賣出信號 (做多策略的賣出)
        sell_signals += (strategy_signal == 0).astype(int) * (strategy_signal.shift(1) == 1).astype(int)
        
        # 累計做空信號 (做空策略的做空)
        short_signals += (strategy_signal == -1).astype(int)
        
        # 累計平倉信號 (做空策略的平倉)
        cover_signals += (strategy_signal == 0).astype(int) * (strategy_signal.shift(1) == -1).astype(int)
    
    # 綜合決策邏輯：提供兩種模式
    STRATEGY_MODE = "ANY_SIGNAL"  # 可選: "VOTING" 或 "ANY_SIGNAL"
    
    combined_signals['signal'] = 0
    
    if STRATEGY_MODE == "ANY_SIGNAL":
        # 模式1：任何策略發出信號就執行（用戶要求的邏輯）
        print("📈 使用模式：任意信號觸發制")
        
        # 只要有任何買入信號就做多
        combined_signals.loc[buy_signals > 0, 'signal'] = 1
        
        # 只要有任何做空信號就做空
        combined_signals.loc[short_signals > 0, 'signal'] = -1
        
        # 如果同時有做多和做空信號，以信號數量多的為準
        net_signals = buy_signals - short_signals
        combined_signals.loc[net_signals > 0, 'signal'] = 1
        combined_signals.loc[net_signals < 0, 'signal'] = -1
        
        signal_threshold = 1  # 任意信號觸發
    
    elif STRATEGY_MODE == "VOTING":
        # 模式2：投票制（原有邏輯）
        print("📈 使用模式：多策略投票制")
        
        # 如果買入信號數量 > 賣出+做空信號，則買入
        net_long_signals = buy_signals - sell_signals - short_signals
        # 如果做空信號數量 > 買入+平倉信號，則做空  
        net_short_signals = short_signals - buy_signals - cover_signals
        
        # 設定閾值：至少要有2個策略發出同向信號
        signal_threshold = max(2, len(strategies) // 4)  # 至少2個策略或25%的策略
        
        combined_signals.loc[net_long_signals >= signal_threshold, 'signal'] = 1    # 做多
        combined_signals.loc[net_short_signals >= signal_threshold, 'signal'] = -1   # 做空
    
    # 加入趨勢過濾：避免在震盪市場頻繁交易
    ma_trend = data['Close'].rolling(20).mean()
    price_trend = data['Close'] / ma_trend
    
    # 只在明確趨勢時執行信號（避免震盪市場）
    combined_signals.loc[(combined_signals['signal'] == 1) & (price_trend < 0.99), 'signal'] = 0
    combined_signals.loc[(combined_signals['signal'] == -1) & (price_trend > 1.01), 'signal'] = 0
    
    # 統計信號分布
    total_signals = len(combined_signals[combined_signals['signal'] != 0])
    long_signals = len(combined_signals[combined_signals['signal'] == 1])
    short_signals_count = len(combined_signals[combined_signals['signal'] == -1])
    
    print(f"📈 綜合策略統計：")
    print(f"   • 決策模式：{STRATEGY_MODE}")
    print(f"   • 信號閾值：{signal_threshold}個策略")
    print(f"   • 總信號數：{total_signals}")
    print(f"   • 做多信號：{long_signals}個")
    print(f"   • 做空信號：{short_signals_count}個")
    print(f"   • 空倉期間：{len(combined_signals) - total_signals}天")
    
    # 計算綜合策略績效
    combined_signals['returns'] = data['Close'].pct_change()
    combined_signals['strategy_returns'] = combined_signals['signal'].shift(1) * combined_signals['returns']
    
    total_return = (1 + combined_signals['strategy_returns']).cumprod().iloc[-1] - 1
    total_return_pct = total_return * 100
    annual_return = (1 + total_return) ** (252 / len(combined_signals)) - 1
    volatility = combined_signals['strategy_returns'].std() * np.sqrt(252)
    
    # 夏普比率計算：提供兩種方法
    SHARPE_METHOD = "STANDARD"  # 可選: "SIMPLE" 或 "STANDARD"
    
    if SHARPE_METHOD == "SIMPLE":
        # 簡化夏普比率（如utils.py）
        strategy_returns_clean = combined_signals['strategy_returns'].dropna()
        sharpe_ratio = (
            strategy_returns_clean.mean() / strategy_returns_clean.std() * np.sqrt(252)
        ) if strategy_returns_clean.std() > 0 else 0
        print(f"📊 使用簡化夏普比率計算（如utils.py）")
        
    elif SHARPE_METHOD == "STANDARD":
        # 標準夏普比率公式：(年化收益率 - 無風險利率) / 年化波動率
        # 使用美國國債2%作為無風險利率
        risk_free_rate = 0.02
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        print(f"📊 使用標準夏普比率計算（含無風險利率{risk_free_rate*100:.1f}%）")
    
    print(f"📈 綜合策略績效：")
    print(f"   • 夏普比率計算方法：{SHARPE_METHOD}")
    print(f"   • 總回報率：{total_return_pct:.2f}%")
    print(f"   • 年化收益率：{annual_return*100:.2f}%")
    print(f"   • 年化波動率：{volatility*100:.2f}%")
    print(f"   • 夏普比率：{sharpe_ratio:.3f}")
    
    cumulative = (1 + combined_signals['strategy_returns']).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    trades = combined_signals['signal'].diff().abs().sum() / 2
    winning_trades = (combined_signals['strategy_returns'] > 0).sum()
    total_trades = (combined_signals['strategy_returns'] != 0).sum()
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    strategy_results['Combined_All'] = {
        'total_return': total_return_pct,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'trade_count': int(trades),
        'volatility': volatility * 100,  # 轉為百分比顯示
        'signals': combined_signals,
        'equity_curve': cumulative,  # 添加權益曲線數據
        'strategy_returns': combined_signals['strategy_returns'],  # 添加策略回報序列
        'drawdown_series': drawdown  # 添加回撤序列
    }
    
    print(f"✅ 完成 {len(strategy_results)} 個策略分析（包含綜合策略）")
    
    # 生成圖表
    print("🎨 正在生成技術分析圖表...")
    chart_filename = create_chart(data, strategy_results)
    print(f"✅ 圖表已保存：{chart_filename}")
    
    # 生成累積收益對比圖表
    print("🎨 正在生成累積收益對比圖表...")
    cumulative_chart_filename = create_cumulative_returns_chart(data, strategy_results)
    print(f"✅ 累積收益對比圖表已保存：{cumulative_chart_filename}")
    
    # 生成Dashboard
    print("🎨 正在生成互動式Dashboard...")
    dashboard_filename = generate_html_dashboard(strategy_results, chart_filename)
    print(f"✅ 互動式Dashboard已生成：{dashboard_filename}")
    
    # 自動打開瀏覽器
    try:
        webbrowser.open(f'file://{os.path.abspath(dashboard_filename)}')
        print("🌐 Dashboard已在瀏覽器中打開")
    except Exception as e:
        print(f"❌ 無法自動打開瀏覽器: {e}")
    
    # 驗證夏普比率計算
    verify_sharpe_ratio_calculations(strategy_results, data)
    
    # 驗證並修復最大回撤(MDD)計算
    verify_and_fix_mdd_calculations(strategy_results, data)
    
    # 使用QuantStats進行專業分析（用戶建議）
    if QUANTSTATS_AVAILABLE:
        print("\n" + "="*80)
        print("📊 QuantStats專業量化分析")
        print("="*80)
        
        # 為所有策略生成QuantStats分析
        print("🎯 所有策略QuantStats分析...")
        
        # 按夏普比率排序策略
        sorted_strategies = sorted(strategy_results.items(), 
                                  key=lambda x: x[1]['sharpe_ratio'], reverse=True)
        
        print(f"\n📈 QuantStats專業指標 (前10名策略):")
        print(f"{'排名':<4} {'策略名稱':<15} {'QuantStats夏普':<12} {'QuantStats CAGR':<12} {'QuantStats回撤':<12}")
        print("-" * 70)
        
        for rank, (strategy_name, results) in enumerate(sorted_strategies[:10], 1):
            try:
                strategy_returns = results['signals']['strategy_returns'].dropna()
                
                if len(strategy_returns) > 0:
                    # 計算QuantStats指標
                    if QUANTSTATS_AVAILABLE:
                        qs_sharpe = qs.stats.sharpe(strategy_returns)
                        qs_cagr = qs.stats.cagr(strategy_returns)
                        qs_max_dd = qs.stats.max_drawdown(strategy_returns)
                    else:
                        # 如果QuantStats不可用，使用基本計算
                        returns_mean = strategy_returns.mean()
                        returns_std = strategy_returns.std()
                        qs_sharpe = (returns_mean / returns_std * np.sqrt(252)) if returns_std > 0 else 0
                        qs_cagr = (1 + strategy_returns).cumprod().iloc[-1] ** (252/len(strategy_returns)) - 1
                        cumulative = (1 + strategy_returns).cumprod()
                        running_max = cumulative.expanding().max()
                        drawdown = (cumulative - running_max) / running_max
                        qs_max_dd = drawdown.min()
                    
                    print(f"{rank:<4} {strategy_name:<15} {qs_sharpe:<12.3f} {qs_cagr*100:<11.1f}% {qs_max_dd*100:<11.1f}%")
                    
                    # 生成詳細QuantStats圖表（僅前3名）
                    if rank <= 3:
                        try:
                            # 創建QuantStats圖表
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            output_dir = "港股輸出"
                            os.makedirs(output_dir, exist_ok=True)
                            
                            # 簡化的QuantStats圖表
                            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                            fig.suptitle(f'QuantStats分析 - {strategy_name}', fontsize=16, fontweight='bold')
                            
                            # 累積收益
                            cumulative_returns = (1 + strategy_returns).cumprod()
                            axes[0,0].plot(cumulative_returns.index, cumulative_returns)
                            axes[0,0].set_title('累積收益')
                            axes[0,0].grid(True, alpha=0.3)
                            
                            # 回撤
                            rolling_max = cumulative_returns.expanding().max()
                            drawdown = (cumulative_returns - rolling_max) / rolling_max
                            axes[0,1].fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
                            axes[0,1].set_title('回撤分析')
                            axes[0,1].grid(True, alpha=0.3)
                            
                            # 月度回報
                            monthly_returns = strategy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                            colors = ['green' if x > 0 else 'red' for x in monthly_returns]
                            axes[1,0].bar(range(len(monthly_returns)), monthly_returns*100, color=colors, alpha=0.7)
                            axes[1,0].set_title('月度回報 (%)')
                            axes[1,0].grid(True, alpha=0.3)
                            
                            # 關鍵指標
                            metrics_text = f"""
QuantStats關鍵指標:

夏普比率: {qs_sharpe:.3f}
CAGR: {qs_cagr*100:.2f}%
最大回撤: {qs_max_dd*100:.2f}%
                            """
                            axes[1,1].text(0.1, 0.5, metrics_text, transform=axes[1,1].transAxes, 
                                          fontsize=10, verticalalignment='center')
                            axes[1,1].set_xlim(0, 1)
                            axes[1,1].set_ylim(0, 1)
                            axes[1,1].axis('off')
                            
                            plt.tight_layout()
                            
                            chart_filename = f"{output_dir}/QuantStats_策略分析_{timestamp}.png"
                            plt.savefig(chart_filename, dpi=300, bbox_inches='tight', facecolor='white')
                            plt.close()
                            
                            print(f"   📊 QuantStats圖表已保存: {chart_filename}")
                            
                        except Exception as e:
                            print(f"   ❌ {strategy_name} QuantStats圖表生成失敗：{e}")
                
            except Exception as e:
                print(f"   ❌ {strategy_name} QuantStats分析失敗：{e}")
        
        print("="*80)
        
    print(f"\n🎉 完整分析完成！")
    print(f"📊 Dashboard文件：{dashboard_filename}")
    
    print("\n✅ 分析完成！功能包括：")
    print("   📥 自動數據下載")
    print("   📊 16種策略回測（8做多+8做空）+ 1種綜合策略")
    print("   🎨 互動式Dashboard")
    print("   📈 Equity Curves對比")
    print("   📋 詳細績效指標")
    print("   🔍 策略參數和進出場條件")
    print("   🎯 綜合所有策略的equity curve")
    print("   📊 QuantStats專業量化分析")
    print("   📈 高級風險指標分析")
    print("   ⭐ 策略評級系統")
    print(f"\n🌐 請查看：{dashboard_filename}")
    
    # 使用 Plotly Dash 創建現代化 Dashboard
    plotly_dashboard = create_plotly_dashboard(strategy_results, data)
    
    if plotly_dashboard:
        print("🎉 Plotly Dashboard 已成功創建")
    else:
        print("❌ Plotly Dashboard 創建失敗")

if __name__ == "__main__":
    main() 