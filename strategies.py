#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股量化分析系統 - 策略模組
包含策略優化、信號生成等核心功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
import sys
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# 導入其他模組
try:
    from data_handler import DataFetcher
    print("✅ DataFetcher 導入成功")
except ImportError as e:
    print(f"❌ DataFetcher 導入失敗: {e}")

try:
    from enhanced_strategy_optimizer import AdvancedStrategyOptimizer
    print("✅ AdvancedStrategyOptimizer 導入成功")
except ImportError as e:
    print(f"⚠️ AdvancedStrategyOptimizer 導入失敗: {e}")
    AdvancedStrategyOptimizer = None

try:
    from strategy_manager import StrategyPortfolioManager, SignalGenerator
    print("✅ StrategyManager 模組導入成功")
except ImportError as e:
    print(f"⚠️ StrategyManager 導入失敗: {e}")
    # 創建簡單的備用信號生成器
    class SimpleSignalGenerator:
        @staticmethod
        def generate_rsi_signals(data, period=14, overbought=70, oversold=30):
            from data_handler import TechnicalIndicators
            rsi = TechnicalIndicators.calculate_rsi(data['Close'], period)
            signals = pd.DataFrame(index=data.index)
            signals['signal'] = 0
            signals.loc[rsi < oversold, 'signal'] = 1  # 買入信號
            signals.loc[rsi > overbought, 'signal'] = -1  # 賣出信號
            return signals
        
        @staticmethod
        def generate_macd_signals(data):
            from data_handler import TechnicalIndicators
            macd_data = TechnicalIndicators.calculate_macd(data['Close'])
            signals = pd.DataFrame(index=data.index)
            signals['signal'] = 0
            signals.loc[macd_data['MACD'] > macd_data['Signal'], 'signal'] = 1
            signals.loc[macd_data['MACD'] < macd_data['Signal'], 'signal'] = -1
            return signals
    
    SignalGenerator = SimpleSignalGenerator

class StrategyOptimizer:
    """策略優化器主類"""
    
    def __init__(self, symbol: str = "2800.HK"):
        self.symbol = symbol
        self.data_fetcher = DataFetcher()
        self.signal_generator = SignalGenerator()
        
    def get_stock_data(self, start_date: str) -> pd.DataFrame:
        """獲取股票數據"""
        try:
            data = self.data_fetcher.get_yahoo_finance_data(
                self.symbol, start_date, "2025-12-31"
            )
            if data is not None and not data.empty:
                print(f"✅ 成功獲取 {self.symbol} 數據: {len(data)} 條記錄")
                return data
            else:
                print(f"❌ 無法獲取 {self.symbol} 數據")
                return pd.DataFrame()
        except Exception as e:
            print(f"❌ 數據獲取錯誤: {e}")
            return pd.DataFrame()
    
    def optimize_rsi_strategy(self, data: pd.DataFrame) -> Dict:
        """優化RSI策略"""
        try:
            print("🔍 正在優化RSI策略...")
            
            best_result = {
                'strategy': 'RSI',
                'params': {'period': 14, 'overbought': 70, 'oversold': 30},
                'performance': {'sharpe_ratio': 0, 'annual_return': 0, 'max_drawdown': 0}
            }
            
            # 測試不同參數組合
            for period in [10, 14, 20, 25]:
                for overbought in [65, 70, 75, 80]:
                    for oversold in [20, 25, 30, 35]:
                        try:
                            signals = self.signal_generator.generate_rsi_signals(
                                data, period, overbought, oversold
                            )
                            
                            if not signals.empty and 'signal' in signals.columns:
                                # 計算簡單績效
                                returns = data['Close'].pct_change().fillna(0)
                                strategy_returns = returns * signals['signal'].shift(1)
                                
                                annual_return = strategy_returns.mean() * 252
                                volatility = strategy_returns.std() * np.sqrt(252)
                                sharpe_ratio = annual_return / volatility if volatility > 0 else 0
                                
                                if sharpe_ratio > best_result['performance']['sharpe_ratio']:
                                    best_result['params'] = {
                                        'period': period, 
                                        'overbought': overbought, 
                                        'oversold': oversold
                                    }
                                    best_result['performance'] = {
                                        'sharpe_ratio': sharpe_ratio,
                                        'annual_return': annual_return,
                                        'max_drawdown': strategy_returns.cumsum().expanding().min().iloc[-1]
                                    }
                        except Exception:
                            continue
            
            print(f"✅ RSI策略優化完成，最佳夏普比率: {best_result['performance']['sharpe_ratio']:.3f}")
            return best_result
            
        except Exception as e:
            print(f"❌ RSI策略優化失敗: {e}")
            return {}
    
    def optimize_macd_strategy(self, data: pd.DataFrame) -> Dict:
        """優化MACD策略"""
        try:
            print("🔍 正在優化MACD策略...")
            
            best_result = {
                'strategy': 'MACD',
                'params': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
                'performance': {'sharpe_ratio': 0, 'annual_return': 0, 'max_drawdown': 0}
            }
            
            # 測試不同參數組合
            for fast in [8, 12, 16, 20]:
                for slow in [24, 26, 32, 40]:
                    for signal in [6, 9, 12, 15]:
                        if fast >= slow:
                            continue
                        try:
                            signals = self._generate_macd_signals_with_params(data, fast, slow, signal)
                            
                            if not signals.empty and 'signal' in signals.columns:
                                returns = data['Close'].pct_change().fillna(0)
                                strategy_returns = returns * signals['signal'].shift(1)
                                
                                annual_return = strategy_returns.mean() * 252
                                volatility = strategy_returns.std() * np.sqrt(252)
                                sharpe_ratio = annual_return / volatility if volatility > 0 else 0
                                
                                if sharpe_ratio > best_result['performance']['sharpe_ratio']:
                                    best_result['params'] = {
                                        'fast_period': fast, 
                                        'slow_period': slow, 
                                        'signal_period': signal
                                    }
                                    best_result['performance'] = {
                                        'sharpe_ratio': sharpe_ratio,
                                        'annual_return': annual_return,
                                        'max_drawdown': strategy_returns.cumsum().expanding().min().iloc[-1]
                                    }
                        except Exception:
                            continue
            
            print(f"✅ MACD策略優化完成，最佳夏普比率: {best_result['performance']['sharpe_ratio']:.3f}")
            return best_result
            
        except Exception as e:
            print(f"❌ MACD策略優化失敗: {e}")
            return {}

    def optimize_bollinger_strategy(self, data: pd.DataFrame) -> Dict:
        """優化布林帶策略"""
        try:
            print("🔍 正在優化布林帶策略...")
            
            best_result = {
                'strategy': 'Bollinger',
                'params': {'period': 20, 'std_multiplier': 2.0},
                'performance': {'sharpe_ratio': 0, 'annual_return': 0, 'max_drawdown': 0}
            }
            
            # 測試不同參數組合
            for period in [15, 20, 25, 30]:
                for std_mult in [1.5, 2.0, 2.5, 3.0]:
                    try:
                        signals = self._generate_bollinger_signals_with_params(data, period, std_mult)
                        
                        if not signals.empty and 'signal' in signals.columns:
                            returns = data['Close'].pct_change().fillna(0)
                            strategy_returns = returns * signals['signal'].shift(1)
                            
                            annual_return = strategy_returns.mean() * 252
                            volatility = strategy_returns.std() * np.sqrt(252)
                            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
                            
                            if sharpe_ratio > best_result['performance']['sharpe_ratio']:
                                best_result['params'] = {
                                    'period': period, 
                                    'std_multiplier': std_mult
                                }
                                best_result['performance'] = {
                                    'sharpe_ratio': sharpe_ratio,
                                    'annual_return': annual_return,
                                    'max_drawdown': strategy_returns.cumsum().expanding().min().iloc[-1]
                                }
                    except Exception:
                        continue
            
            print(f"✅ 布林帶策略優化完成，最佳夏普比率: {best_result['performance']['sharpe_ratio']:.3f}")
            return best_result
            
        except Exception as e:
            print(f"❌ 布林帶策略優化失敗: {e}")
            return {}

    def optimize_kdj_strategy(self, data: pd.DataFrame) -> Dict:
        """優化KDJ策略"""
        try:
            print("🔍 正在優化KDJ策略...")
            
            best_result = {
                'strategy': 'KDJ',
                'params': {'period': 9, 'smooth_k': 3, 'smooth_d': 3, 'overbought': 80, 'oversold': 20},
                'performance': {'sharpe_ratio': 0, 'annual_return': 0, 'max_drawdown': 0}
            }
            
            # 測試不同參數組合
            for period in [9, 14, 21]:
                for smooth_k in [3, 5]:
                    for smooth_d in [3, 5]:
                        for overbought in [75, 80, 85]:
                            for oversold in [15, 20, 25]:
                                try:
                                    signals = self._generate_kdj_signals_with_params(
                                        data, period, smooth_k, smooth_d, overbought, oversold
                                    )
                                    
                                    if not signals.empty and 'signal' in signals.columns:
                                        returns = data['Close'].pct_change().fillna(0)
                                        strategy_returns = returns * signals['signal'].shift(1)
                                        
                                        annual_return = strategy_returns.mean() * 252
                                        volatility = strategy_returns.std() * np.sqrt(252)
                                        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
                                        
                                        if sharpe_ratio > best_result['performance']['sharpe_ratio']:
                                            best_result['params'] = {
                                                'period': period,
                                                'smooth_k': smooth_k,
                                                'smooth_d': smooth_d,
                                                'overbought': overbought,
                                                'oversold': oversold
                                            }
                                            best_result['performance'] = {
                                                'sharpe_ratio': sharpe_ratio,
                                                'annual_return': annual_return,
                                                'max_drawdown': strategy_returns.cumsum().expanding().min().iloc[-1]
                                            }
                                except Exception:
                                    continue
            
            print(f"✅ KDJ策略優化完成，最佳夏普比率: {best_result['performance']['sharpe_ratio']:.3f}")
            return best_result
            
        except Exception as e:
            print(f"❌ KDJ策略優化失敗: {e}")
            return {}

    def optimize_stochastic_strategy(self, data: pd.DataFrame) -> Dict:
        """優化Stochastic隨機指標策略"""
        try:
            print("🔍 正在優化Stochastic隨機指標策略...")
            
            best_result = {
                'strategy': 'Stochastic',
                'params': {'k_period': 14, 'd_period': 3, 'overbought': 80, 'oversold': 20},
                'performance': {'sharpe_ratio': 0, 'annual_return': 0, 'max_drawdown': 0}
            }
            
            # 測試不同參數組合
            for k_period in [9, 14, 21]:
                for d_period in [3, 5, 7]:
                    for overbought in [75, 80, 85]:
                        for oversold in [15, 20, 25]:
                            try:
                                signals = self._generate_stochastic_signals_with_params(
                                    data, k_period, d_period, overbought, oversold
                                )
                                
                                if not signals.empty and 'signal' in signals.columns:
                                    returns = data['Close'].pct_change().fillna(0)
                                    strategy_returns = returns * signals['signal'].shift(1)
                                    
                                    annual_return = strategy_returns.mean() * 252
                                    volatility = strategy_returns.std() * np.sqrt(252)
                                    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
                                    
                                    if sharpe_ratio > best_result['performance']['sharpe_ratio']:
                                        best_result['params'] = {
                                            'k_period': k_period,
                                            'd_period': d_period,
                                            'overbought': overbought,
                                            'oversold': oversold
                                        }
                                        best_result['performance'] = {
                                            'sharpe_ratio': sharpe_ratio,
                                            'annual_return': annual_return,
                                            'max_drawdown': strategy_returns.cumsum().expanding().min().iloc[-1]
                                        }
                            except Exception:
                                continue
            
            print(f"✅ Stochastic策略優化完成，最佳夏普比率: {best_result['performance']['sharpe_ratio']:.3f}")
            return best_result
            
        except Exception as e:
            print(f"❌ Stochastic策略優化失敗: {e}")
            return {}

    def optimize_cci_strategy(self, data: pd.DataFrame) -> Dict:
        """優化CCI商品通道指數策略"""
        try:
            print("🔍 正在優化CCI策略...")
            
            best_result = {
                'strategy': 'CCI',
                'params': {'period': 20, 'overbought': 100, 'oversold': -100},
                'performance': {'sharpe_ratio': 0, 'annual_return': 0, 'max_drawdown': 0}
            }
            
            # 測試不同參數組合
            for period in [14, 20, 25, 30]:
                for overbought in [80, 100, 120, 150]:
                    for oversold in [-150, -120, -100, -80]:
                        try:
                            signals = self._generate_cci_signals_with_params(
                                data, period, overbought, oversold
                            )
                            
                            if not signals.empty and 'signal' in signals.columns:
                                returns = data['Close'].pct_change().fillna(0)
                                strategy_returns = returns * signals['signal'].shift(1)
                                
                                annual_return = strategy_returns.mean() * 252
                                volatility = strategy_returns.std() * np.sqrt(252)
                                sharpe_ratio = annual_return / volatility if volatility > 0 else 0
                                
                                if sharpe_ratio > best_result['performance']['sharpe_ratio']:
                                    best_result['params'] = {
                                        'period': period,
                                        'overbought': overbought,
                                        'oversold': oversold
                                    }
                                    best_result['performance'] = {
                                        'sharpe_ratio': sharpe_ratio,
                                        'annual_return': annual_return,
                                        'max_drawdown': strategy_returns.cumsum().expanding().min().iloc[-1]
                                    }
                        except Exception:
                            continue
            
            print(f"✅ CCI策略優化完成，最佳夏普比率: {best_result['performance']['sharpe_ratio']:.3f}")
            return best_result
            
        except Exception as e:
            print(f"❌ CCI策略優化失敗: {e}")
            return {}

    def optimize_williams_r_strategy(self, data: pd.DataFrame) -> Dict:
        """優化威廉指標%R策略"""
        try:
            print("🔍 正在優化威廉指標%R策略...")
            
            best_result = {
                'strategy': 'Williams_R',
                'params': {'period': 14, 'overbought': -20, 'oversold': -80},
                'performance': {'sharpe_ratio': 0, 'annual_return': 0, 'max_drawdown': 0}
            }
            
            # 測試不同參數組合
            for period in [9, 14, 21, 28]:
                for overbought in [-10, -15, -20, -25]:
                    for oversold in [-75, -80, -85, -90]:
                        try:
                            signals = self._generate_williams_r_signals_with_params(
                                data, period, overbought, oversold
                            )
                            
                            if not signals.empty and 'signal' in signals.columns:
                                returns = data['Close'].pct_change().fillna(0)
                                strategy_returns = returns * signals['signal'].shift(1)
                                
                                annual_return = strategy_returns.mean() * 252
                                volatility = strategy_returns.std() * np.sqrt(252)
                                sharpe_ratio = annual_return / volatility if volatility > 0 else 0
                                
                                if sharpe_ratio > best_result['performance']['sharpe_ratio']:
                                    best_result['params'] = {
                                        'period': period,
                                        'overbought': overbought,
                                        'oversold': oversold
                                    }
                                    best_result['performance'] = {
                                        'sharpe_ratio': sharpe_ratio,
                                        'annual_return': annual_return,
                                        'max_drawdown': strategy_returns.cumsum().expanding().min().iloc[-1]
                                    }
                        except Exception:
                            continue
            
            print(f"✅ 威廉指標%R策略優化完成，最佳夏普比率: {best_result['performance']['sharpe_ratio']:.3f}")
            return best_result
            
        except Exception as e:
            print(f"❌ 威廉指標%R策略優化失敗: {e}")
            return {}

    # 輔助方法：生成各種技術指標信號
    def _generate_macd_signals_with_params(self, data: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.DataFrame:
        """生成MACD信號（自定義參數）"""
        try:
            # 計算MACD
            exp1 = data['Close'].ewm(span=fast).mean()
            exp2 = data['Close'].ewm(span=slow).mean()
            macd = exp1 - exp2
            macd_signal = macd.ewm(span=signal).mean()
            
            signals = pd.DataFrame(index=data.index)
            signals['signal'] = 0
            signals.loc[macd > macd_signal, 'signal'] = 1
            signals.loc[macd < macd_signal, 'signal'] = -1
            
            return signals
        except Exception:
            return pd.DataFrame()

    def _generate_bollinger_signals_with_params(self, data: pd.DataFrame, period: int, std_mult: float) -> pd.DataFrame:
        """生成布林帶信號（自定義參數）"""
        try:
            # 計算布林帶
            sma = data['Close'].rolling(window=period).mean()
            std = data['Close'].rolling(window=period).std()
            upper = sma + (std * std_mult)
            lower = sma - (std * std_mult)
            
            signals = pd.DataFrame(index=data.index)
            signals['signal'] = 0
            signals.loc[data['Close'] < lower, 'signal'] = 1  # 買入
            signals.loc[data['Close'] > upper, 'signal'] = -1  # 賣出
            
            return signals
        except Exception:
            return pd.DataFrame()

    def _generate_kdj_signals_with_params(self, data: pd.DataFrame, period: int, smooth_k: int, 
                                         smooth_d: int, overbought: float, oversold: float) -> pd.DataFrame:
        """生成KDJ信號（自定義參數）"""
        try:
            # 計算KDJ
            low_min = data['Low'].rolling(window=period).min()
            high_max = data['High'].rolling(window=period).max()
            
            rsv = (data['Close'] - low_min) / (high_max - low_min) * 100
            rsv = rsv.fillna(50)
            
            k = rsv.ewm(alpha=1/smooth_k).mean()
            d = k.ewm(alpha=1/smooth_d).mean()
            j = 3 * k - 2 * d
            
            signals = pd.DataFrame(index=data.index)
            signals['signal'] = 0
            signals.loc[k < oversold, 'signal'] = 1  # 買入
            signals.loc[k > overbought, 'signal'] = -1  # 賣出
            
            return signals
        except Exception:
            return pd.DataFrame()

    def _generate_stochastic_signals_with_params(self, data: pd.DataFrame, k_period: int, d_period: int,
                                               overbought: float, oversold: float) -> pd.DataFrame:
        """生成Stochastic信號（自定義參數）"""
        try:
            # 計算Stochastic
            low_min = data['Low'].rolling(window=k_period).min()
            high_max = data['High'].rolling(window=k_period).max()
            
            k_percent = (data['Close'] - low_min) / (high_max - low_min) * 100
            k_percent = k_percent.fillna(50)
            d_percent = k_percent.rolling(window=d_period).mean()
            
            signals = pd.DataFrame(index=data.index)
            signals['signal'] = 0
            signals.loc[k_percent < oversold, 'signal'] = 1  # 買入
            signals.loc[k_percent > overbought, 'signal'] = -1  # 賣出
            
            return signals
        except Exception:
            return pd.DataFrame()

    def _generate_cci_signals_with_params(self, data: pd.DataFrame, period: int, 
                                        overbought: float, oversold: float) -> pd.DataFrame:
        """生成CCI信號（自定義參數）"""
        try:
            # 計算CCI
            tp = (data['High'] + data['Low'] + data['Close']) / 3  # 典型價格
            sma_tp = tp.rolling(window=period).mean()
            mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            
            cci = (tp - sma_tp) / (0.015 * mad)
            
            signals = pd.DataFrame(index=data.index)
            signals['signal'] = 0
            signals.loc[cci < oversold, 'signal'] = 1  # 買入
            signals.loc[cci > overbought, 'signal'] = -1  # 賣出
            
            return signals
        except Exception:
            return pd.DataFrame()

    def _generate_williams_r_signals_with_params(self, data: pd.DataFrame, period: int,
                                               overbought: float, oversold: float) -> pd.DataFrame:
        """生成威廉指標%R信號（自定義參數）"""
        try:
            # 計算威廉指標%R
            high_max = data['High'].rolling(window=period).max()
            low_min = data['Low'].rolling(window=period).min()
            
            williams_r = (high_max - data['Close']) / (high_max - low_min) * -100
            
            signals = pd.DataFrame(index=data.index)
            signals['signal'] = 0
            signals.loc[williams_r < oversold, 'signal'] = 1  # 買入
            signals.loc[williams_r > overbought, 'signal'] = -1  # 賣出
            
            return signals
        except Exception:
            return pd.DataFrame()

def run_strategy_optimization(symbol: str, start_date: str) -> bool:
    """運行基本策略優化"""
    try:
        print(f"\n🎯 開始基本策略優化: {symbol}")
        
        optimizer = StrategyOptimizer(symbol)
        data = optimizer.get_stock_data(start_date)
        
        if data.empty:
            print("❌ 無數據可用，跳過優化")
            return False
        
        # 優化各種策略
        rsi_result = optimizer.optimize_rsi_strategy(data)
        macd_result = optimizer.optimize_macd_strategy(data)
        bollinger_result = optimizer.optimize_bollinger_strategy(data)
        kdj_result = optimizer.optimize_kdj_strategy(data)
        stochastic_result = optimizer.optimize_stochastic_strategy(data)
        cci_result = optimizer.optimize_cci_strategy(data)
        williams_r_result = optimizer.optimize_williams_r_strategy(data)
        
        # 輸出結果
        print("\n📊 策略優化結果:")
        if rsi_result:
            print(f"   RSI策略 - 夏普比率: {rsi_result['performance']['sharpe_ratio']:.3f}")
        if macd_result:
            print(f"   MACD策略 - 夏普比率: {macd_result['performance']['sharpe_ratio']:.3f}")
        if bollinger_result:
            print(f"   布林帶策略 - 夏普比率: {bollinger_result['performance']['sharpe_ratio']:.3f}")
        if kdj_result:
            print(f"   KDJ策略 - 夏普比率: {kdj_result['performance']['sharpe_ratio']:.3f}")
        if stochastic_result:
            print(f"   Stochastic策略 - 夏普比率: {stochastic_result['performance']['sharpe_ratio']:.3f}")
        if cci_result:
            print(f"   CCI策略 - 夏普比率: {cci_result['performance']['sharpe_ratio']:.3f}")
        if williams_r_result:
            print(f"   威廉指標%R策略 - 夏普比率: {williams_r_result['performance']['sharpe_ratio']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 策略優化失敗: {e}")
        return False

def run_comprehensive_optimization(symbol: str, start_date: str, mode: str = "quick_scan") -> bool:
    """運行綜合策略優化"""
    try:
        print(f"\n🚀 開始綜合策略優化: {symbol} (模式: {mode})")
        
        optimizer = StrategyOptimizer(symbol)
        data = optimizer.get_stock_data(start_date)
        
        if data.empty:
            print("❌ 無數據可用，跳過優化")
            return False
        
        results = []
        
        if mode == "quick_scan":
            # 快速掃描模式：測試基本參數組合
            print("⚡ 快速掃描模式 - 測試基本參數組合")
            
            rsi_result = optimizer.optimize_rsi_strategy(data)
            if rsi_result:
                results.append(rsi_result)
                
            macd_result = optimizer.optimize_macd_strategy(data)
            if macd_result:
                results.append(macd_result)
                
            bollinger_result = optimizer.optimize_bollinger_strategy(data)
            if bollinger_result:
                results.append(bollinger_result)
                
        elif mode == "comprehensive":
            # 全面優化模式：測試所有技術指標
            print("🔥 全面優化模式 - 測試所有技術指標")
            
            strategies = [
                ("RSI", optimizer.optimize_rsi_strategy),
                ("MACD", optimizer.optimize_macd_strategy),
                ("布林帶", optimizer.optimize_bollinger_strategy),
                ("KDJ", optimizer.optimize_kdj_strategy),
                ("Stochastic", optimizer.optimize_stochastic_strategy),
                ("CCI", optimizer.optimize_cci_strategy),
                ("威廉指標%R", optimizer.optimize_williams_r_strategy),
            ]
            
            # 南北水策略優化
            try:
                from simple_north_south_strategies import add_north_south_strategies_to_optimization
                results = add_north_south_strategies_to_optimization(results, data, symbol)
                    
            except ImportError:
                print("   ⚠️ 南北水策略模組導入失敗，跳過南北水策略")
            except Exception as e:
                print(f"   ❌ 南北水策略優化錯誤: {e}")
            
            for strategy_name, strategy_func in strategies:
                try:
                    result = strategy_func(data)
                    if result and result.get('performance', {}).get('sharpe_ratio', 0) > 0:
                        results.append(result)
                        print(f"   ✅ {strategy_name} 優化完成 - 夏普比率: {result['performance']['sharpe_ratio']:.3f}")
                    else:
                        print(f"   ❌ {strategy_name} 優化失敗或表現不佳")
                except Exception as e:
                    print(f"   ❌ {strategy_name} 優化錯誤: {e}")
        
        # 排序結果並輸出
        if results:
            results.sort(key=lambda x: x['performance']['sharpe_ratio'], reverse=True)
            
            print(f"\n🏆 最佳策略排名 (共 {len(results)} 個策略):")
            for i, result in enumerate(results, 1):
                strategy = result['strategy']
                sharpe = result['performance']['sharpe_ratio']
                annual_return = result['performance']['annual_return']
                max_dd = result['performance']['max_drawdown']
                print(f"   {i}. {strategy}: 夏普比率={sharpe:.3f}, 年化收益={annual_return:.2%}, 最大回撤={max_dd:.2%}")
            
            # 保存結果為 CSV，讓 Dashboard 可以讀取
            try:
                save_strategy_results_to_csv(results, mode)
                print(f"✅ 策略結果已保存為 CSV 格式")
            except Exception as e:
                print(f"⚠️ CSV 保存失敗: {e}")
            
            return True
        else:
            print("❌ 沒有找到有效的策略結果")
            return False
            
    except Exception as e:
        print(f"❌ 綜合策略優化失敗: {e}")
        return False

def save_strategy_results_to_csv(results: list, mode: str) -> None:
    """保存策略結果為 CSV 格式，供 Dashboard 讀取"""
    try:
        import os
        from datetime import datetime
        
        # 確保輸出目錄存在
        csv_dir = "data_output/csv"
        os.makedirs(csv_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 創建策略結果 DataFrame
        strategy_data = []
        for result in results:
            strategy_data.append({
                'strategy': result['strategy'],
                'sharpe_ratio': result['performance']['sharpe_ratio'],
                'annual_return': result['performance']['annual_return'],
                'max_drawdown': result['performance']['max_drawdown'],
                'params': str(result['params'])
            })
        
        df = pd.DataFrame(strategy_data)
        
        # 保存為多種格式以供 Dashboard 讀取
        if mode == "comprehensive":
            # 全面優化結果
            filename = f"{csv_dir}/multi_strategy_comprehensive_{timestamp}.csv"
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"   📁 已保存: {filename}")
            
            # 為每個策略創建單獨的文件（模擬現有格式）
            for result in results:
                strategy_name = result['strategy'].replace(' ', '_').replace('%', 'Percent')
                strategy_filename = f"{csv_dir}/integrated_{strategy_name.lower()}_{timestamp}.csv"
                
                # 創建單策略 DataFrame
                single_df = pd.DataFrame([{
                    'strategy': result['strategy'],
                    'sharpe_ratio': result['performance']['sharpe_ratio'],
                    'annual_return': result['performance']['annual_return'],
                    'max_drawdown': result['performance']['max_drawdown'],
                    'volatility': result['performance'].get('volatility', 0),
                    'total_return': result['performance'].get('total_return', 0),
                    'win_rate': result['performance'].get('win_rate', 0),
                    'params': str(result['params'])
                }])
                
                single_df.to_csv(strategy_filename, index=False, encoding='utf-8-sig')
                print(f"   📁 已保存: {strategy_filename}")
        
        else:
            # 快速掃描結果
            filename = f"{csv_dir}/multi_strategy_quick_scan_{timestamp}.csv"
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"   📁 已保存: {filename}")
            
    except Exception as e:
        print(f"❌ 保存 CSV 失敗: {e}")
        raise

def run_ultra_parallel_optimization(symbol: str, start_date: str, max_workers: int = 300) -> bool:
    """運行超級並行優化"""
    try:
        print(f"\n🚀 開始超級並行優化: {symbol} (工作進程: {max_workers})")
        
        # 簡化版本，實際上調用基本優化
        print("⚠️ 超級並行模式正在開發中，目前使用增強模式")
        return run_comprehensive_optimization(symbol, start_date, "comprehensive")
        
    except Exception as e:
        print(f"❌ 超級並行優化失敗: {e}")
        return False

def run_integrated_rsi_optimization(symbol: str, start_date: str) -> bool:
    """運行整合的RSI策略優化 - 合併快速掃描、全面分析和並行優化"""
    try:
        print(f"\n🚀 開始智能RSI策略優化: {symbol}")
        print("📊 此功能整合了快速掃描、全面分析和並行優化")
        print("=" * 60)
        
        optimizer = StrategyOptimizer(symbol)
        data = optimizer.get_stock_data(start_date)
        
        if data.empty:
            print("❌ 無數據可用，跳過優化")
            return False
        
        print(f"✅ 成功獲取 {len(data)} 天的股票數據")
        
        # 階段1: 快速掃描 (大步長，快速篩選)
        print("\n📈 階段1: 快速掃描 (大步長搜索)")
        quick_results = []
        
        for period in range(10, 81, 20):  # 10, 30, 50, 70
            for oversold in range(20, 41, 10):  # 20, 30, 40
                for overbought in range(60, 91, 15):  # 60, 75, 90
                    try:
                        signals = optimizer.signal_generator.generate_rsi_signals(
                            data, period, overbought, oversold
                        )
                        
                        if not signals.empty and 'signal' in signals.columns:
                            returns = data['Close'].pct_change().fillna(0)
                            strategy_returns = returns * signals['signal'].shift(1)
                            
                            annual_return = strategy_returns.mean() * 252
                            volatility = strategy_returns.std() * np.sqrt(252)
                            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
                            
                            if sharpe_ratio > 0.1:  # 只保留有希望的組合
                                quick_results.append({
                                    'period': period,
                                    'oversold': oversold,
                                    'overbought': overbought,
                                    'sharpe_ratio': sharpe_ratio,
                                    'annual_return': annual_return
                                })
                    except Exception:
                        continue
        
        if not quick_results:
            print("⚠️ 快速掃描未找到有效策略，使用默認參數")
            best_region = {'period': 14, 'oversold': 30, 'overbought': 70}
        else:
            # 找到最好的區域
            quick_results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
            best_region = quick_results[0]
            print(f"✅ 快速掃描完成，找到 {len(quick_results)} 個候選組合")
            print(f"   最佳組合: 期間={best_region['period']}, 超賣={best_region['oversold']}, 超買={best_region['overbought']}")
            print(f"   夏普比率: {best_region['sharpe_ratio']:.3f}")
        
        # 階段2: 精細搜索 (小步長，在最佳區域附近優化)
        print("\n🔍 階段2: 精細搜索 (在最佳區域周圍)")
        detailed_results = []
        
        # 在最佳區域周圍搜索
        period_range = range(max(5, best_region['period'] - 10), min(81, best_region['period'] + 11), 2)
        oversold_range = range(max(10, best_region['oversold'] - 10), min(41, best_region['oversold'] + 11), 2)
        overbought_range = range(max(60, best_region['overbought'] - 10), min(91, best_region['overbought'] + 11), 2)
        
        total_combinations = len(list(period_range)) * len(list(oversold_range)) * len(list(overbought_range))
        print(f"📊 將測試 {total_combinations} 個參數組合...")
        
        current_count = 0
        for period in period_range:
            for oversold in oversold_range:
                for overbought in overbought_range:
                    current_count += 1
                    if current_count % 50 == 0:
                        print(f"   進度: {current_count}/{total_combinations} ({current_count/total_combinations*100:.1f}%)")
                    
                    try:
                        signals = optimizer.signal_generator.generate_rsi_signals(
                            data, period, overbought, oversold
                        )
                        
                        if not signals.empty and 'signal' in signals.columns:
                            returns = data['Close'].pct_change().fillna(0)
                            strategy_returns = returns * signals['signal'].shift(1)
                            
                            # 計算更詳細的績效指標
                            annual_return = strategy_returns.mean() * 252
                            volatility = strategy_returns.std() * np.sqrt(252)
                            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
                            
                            # 計算最大回撤
                            cumulative_returns = (1 + strategy_returns).cumprod()
                            rolling_max = cumulative_returns.expanding().max() 
                            drawdown = (cumulative_returns - rolling_max) / rolling_max
                            max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0.0
                            
                            # 計算勝率
                            win_rate = (strategy_returns > 0).sum() / len(strategy_returns) * 100
                            
                            detailed_results.append({
                                'period': period,
                                'oversold': oversold,
                                'overbought': overbought,
                                'sharpe_ratio': sharpe_ratio,
                                'annual_return': annual_return,
                                'volatility': volatility,
                                'max_drawdown': max_drawdown,
                                'win_rate': win_rate,
                                'total_trades': signals['signal'].abs().sum()
                            })
                    except Exception:
                        continue
        
        if not detailed_results:
            print("❌ 精細搜索失敗，使用快速掃描結果")
            final_result = best_region
        else:
            # 找到最優結果
            detailed_results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
            final_result = detailed_results[0]
            
            print(f"✅ 精細搜索完成，測試了 {len(detailed_results)} 個有效組合")
        
        # 階段3: 並行驗證和報告生成
        print("\n⚡ 階段3: 結果驗證和報告生成")
        
        # 生成最終報告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存CSV結果
        os.makedirs("data_output/csv", exist_ok=True)
        
        # 快速掃描結果
        if quick_results:
            quick_df = pd.DataFrame(quick_results)
            quick_file = f"data_output/csv/integrated_rsi_quick_scan_{timestamp}.csv"
            quick_df.to_csv(quick_file, index=False, encoding='utf-8-sig')
            print(f"📊 快速掃描結果已保存: {quick_file}")
        
        # 詳細搜索結果
        if detailed_results:
            detailed_df = pd.DataFrame(detailed_results)
            detailed_file = f"data_output/csv/integrated_rsi_detailed_{timestamp}.csv"
            detailed_df.to_csv(detailed_file, index=False, encoding='utf-8-sig')
            print(f"📊 詳細搜索結果已保存: {detailed_file}")
        
        # 生成文字報告
        os.makedirs("data_output/reports", exist_ok=True)
        report_file = f"data_output/reports/integrated_rsi_optimization_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("🚀 港股量化分析系統 - 智能RSI策略優化報告\n")
            f.write("=" * 60 + "\n")
            f.write(f"📅 生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"📊 股票代碼: {symbol}\n")
            f.write(f"📈 分析期間: {start_date} 至今 ({len(data)} 天)\n")
            f.write(f"🎯 分析模式: 智能整合版 (快速+全面+並行)\n\n")
            
            f.write("📊 階段1: 快速掃描結果\n")
            f.write("-" * 30 + "\n")
            if quick_results:
                f.write(f"   測試組合數: {len(quick_results)}\n")
                f.write(f"   最佳夏普比率: {quick_results[0]['sharpe_ratio']:.3f}\n")
                f.write(f"   最佳參數: 期間={quick_results[0]['period']}, 超賣={quick_results[0]['oversold']}, 超買={quick_results[0]['overbought']}\n\n")
            
            f.write("🔍 階段2: 精細搜索結果\n")
            f.write("-" * 30 + "\n")
            if detailed_results:
                f.write(f"   測試組合數: {len(detailed_results)}\n")
                f.write(f"   最佳夏普比率: {detailed_results[0]['sharpe_ratio']:.3f}\n")
                f.write(f"   年化收益率: {detailed_results[0]['annual_return']:.2%}\n")
                f.write(f"   年化波動率: {detailed_results[0]['volatility']:.2%}\n")
                f.write(f"   最大回撤: {detailed_results[0]['max_drawdown']:.2%}\n")
                f.write(f"   勝率: {detailed_results[0]['win_rate']:.1f}%\n")
                f.write(f"   交易次數: {detailed_results[0]['total_trades']}\n")
                f.write(f"   最佳參數: 期間={detailed_results[0]['period']}, 超賣={detailed_results[0]['oversold']}, 超買={detailed_results[0]['overbought']}\n\n")
            
            f.write("🎯 最終建議\n")
            f.write("-" * 30 + "\n")
            f.write(f"✅ 推薦RSI參數組合:\n")
            f.write(f"   期間 (Period): {final_result['period']}\n")
            f.write(f"   超賣線 (Oversold): {final_result['oversold']}\n")
            f.write(f"   超買線 (Overbought): {final_result['overbought']}\n")
            f.write(f"   預期夏普比率: {final_result['sharpe_ratio']:.3f}\n")
            
        print(f"📋 詳細報告已生成: {report_file}")
        
        # 輸出摘要
        print("\n🎯 智能RSI策略優化完成！")
        print("=" * 60)
        print("📊 最終結果摘要:")
        print(f"   🎯 最佳RSI期間: {final_result['period']}")
        print(f"   📉 超賣線: {final_result['oversold']}")
        print(f"   📈 超買線: {final_result['overbought']}")
        print(f"   📊 夏普比率: {final_result['sharpe_ratio']:.3f}")
        
        if detailed_results:
            print(f"   💰 年化收益率: {final_result['annual_return']:.2%}")
            print(f"   📊 年化波動率: {final_result['volatility']:.2%}")
            print(f"   ⚠️  最大回撤: {final_result['max_drawdown']:.2%}")
            print(f"   🎯 勝率: {final_result['win_rate']:.1f}%")
            print(f"   🔄 總交易次數: {final_result['total_trades']}")
        
        print("✅ 智能RSI策略優化分析完成！")
        return True
        
    except Exception as e:
        print(f"❌ 智能RSI策略優化失敗: {e}")
        return False

def main():
    """主程序入口"""
    try:
        print("🤖 策略模組自動化運行開始...")
        
        # 運行完整分析流程
        result1 = run_strategy_optimization("2800.HK", "2020-01-01")
        result2 = run_comprehensive_optimization("2800.HK", "2020-01-01", "comprehensive")
        
        if result1 and result2:
            print("✅ 自動化分析完成！")
        else:
            print("⚠️ 部分分析失敗")
            
    except Exception as e:
        print(f"❌ 自動化運行失敗: {e}")

if __name__ == "__main__":
    main()