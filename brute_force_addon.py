#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股量化分析系統 - 暴力搜索插件
專門提供0-300範圍的暴力搜索功能，可獨立運行或整合到主系統
"""

import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import partial
import time
from datetime import datetime
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 嘗試導入主系統
try:
    from strategies import (
        test_rsi_params_static, 
        test_macd_params_static, 
        load_stock_data, 
        calculate_performance_metrics
    )
    MAIN_SYSTEM_AVAILABLE = True
    print("✅ 成功連接到主策略系統")
except ImportError:
    MAIN_SYSTEM_AVAILABLE = False
    print("⚠️  主策略系統不可用，使用獨立模式")

class BruteForceSearcher:
    """專用暴力搜索器 - 0到300範圍的全面搜索"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    def brute_force_rsi_search(self, step_size: int = 1, max_processes: int = None) -> dict:
        """RSI策略暴力搜索"""
        print("🔥 RSI暴力搜索 (0-300範圍)")
        print("=" * 60)
        
        if max_processes is None:
            max_processes = min(32, mp.cpu_count())
        
        # 生成所有參數組合
        periods = list(range(2, 301, step_size))  # RSI週期：2-300
        oversold_thresholds = list(range(10, 51, step_size))  # 超賣線：10-50
        overbought_thresholds = list(range(50, 91, step_size))  # 超買線：50-90
        
        param_combinations = []
        for period in periods:
            for oversold in oversold_thresholds:
                for overbought in overbought_thresholds:
                    if oversold < overbought:  # 確保邏輯正確
                        param_combinations.append((period, oversold, overbought))
        
        total_combinations = len(param_combinations)
        print(f"📊 總參數組合數: {total_combinations:,}")
        print(f"🚀 使用進程數: {max_processes}")
        
        start_time = time.time()
        
        # 並行測試
        if MAIN_SYSTEM_AVAILABLE:
            test_func = partial(test_rsi_params_static, self.data)
        else:
            test_func = partial(self._test_rsi_simple, self.data)
            
        with mp.Pool(max_processes) as pool:
            results = pool.map(test_func, param_combinations)
        
        # 處理結果
        valid_results = [r for r in results if r is not None and r.get('sharpe_ratio', 0) > 0]
        
        if not valid_results:
            print("❌ 沒有找到有效的參數組合")
            return {}
        
        # 找到最佳結果
        best_result = max(valid_results, key=lambda x: x.get('sharpe_ratio', 0))
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 RSI暴力搜索完成!")
        print(f"⏱️  總耗時: {total_time:.1f}秒")
        print(f"📈 最佳夏普比率: {best_result.get('sharpe_ratio', 0):.3f}")
        print(f"🎯 最佳參數: {best_result.get('params', {})}")
        
        return {
            'best_params': best_result.get('params', {}),
            'best_performance': best_result,
            'total_tested': total_combinations,
            'valid_results': len(valid_results),
            'search_time': total_time,
            'all_results': valid_results[:100]  # 保存前100個結果
        }
    
    def brute_force_macd_search(self, step_size: int = 1, max_processes: int = None) -> dict:
        """MACD策略暴力搜索"""
        print("🔥 MACD暴力搜索 (0-300範圍)")
        print("=" * 60)
        
        if max_processes is None:
            max_processes = min(32, mp.cpu_count())
        
        # 生成所有參數組合
        fast_periods = list(range(3, 101, step_size))  # 快線：3-100
        slow_periods = list(range(10, 201, step_size))  # 慢線：10-200
        signal_periods = list(range(3, 51, step_size))  # 信號線：3-50
        
        param_combinations = []
        for fast in fast_periods:
            for slow in slow_periods:
                for signal in signal_periods:
                    if fast < slow:  # 確保快線小於慢線
                        param_combinations.append((fast, slow, signal))
        
        total_combinations = len(param_combinations)
        print(f"📊 總參數組合數: {total_combinations:,}")
        print(f"🚀 使用進程數: {max_processes}")
        
        start_time = time.time()
        
        # 並行測試
        if MAIN_SYSTEM_AVAILABLE:
            test_func = partial(test_macd_params_static, self.data)
        else:
            test_func = partial(self._test_macd_simple, self.data)
            
        with mp.Pool(max_processes) as pool:
            results = pool.map(test_func, param_combinations)
        
        # 處理結果
        valid_results = [r for r in results if r is not None and r.get('sharpe_ratio', 0) > 0]
        
        if not valid_results:
            print("❌ 沒有找到有效的參數組合")
            return {}
        
        # 找到最佳結果
        best_result = max(valid_results, key=lambda x: x.get('sharpe_ratio', 0))
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 MACD暴力搜索完成!")
        print(f"⏱️  總耗時: {total_time:.1f}秒")
        print(f"📈 最佳夏普比率: {best_result.get('sharpe_ratio', 0):.3f}")
        print(f"🎯 最佳參數: {best_result.get('params', {})}")
        
        return {
            'best_params': best_result.get('params', {}),
            'best_performance': best_result,
            'total_tested': total_combinations,
            'valid_results': len(valid_results),
            'search_time': total_time,
            'all_results': valid_results[:100]
        }
    
    def _test_rsi_simple(self, data, params):
        """簡單RSI測試（當主系統不可用時）"""
        period, oversold, overbought = params
        try:
            # 計算RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # 生成信號
            buy_signals = (rsi < oversold)
            sell_signals = (rsi > overbought)
            
            # 計算收益
            returns = data['Close'].pct_change()
            strategy_returns = returns * buy_signals.shift(1)
            
            if strategy_returns.std() == 0:
                return None
                
            sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
            
            return {
                'params': {'period': period, 'oversold': oversold, 'overbought': overbought},
                'sharpe_ratio': sharpe,
                'total_return': (1 + strategy_returns).prod() - 1
            }
        except:
            return None
    
    def _test_macd_simple(self, data, params):
        """簡單MACD測試（當主系統不可用時）"""
        fast, slow, signal = params
        try:
            # 計算MACD
            exp1 = data['Close'].ewm(span=fast).mean()
            exp2 = data['Close'].ewm(span=slow).mean()
            macd = exp1 - exp2
            macd_signal = macd.ewm(span=signal).mean()
            
            # 生成信號
            buy_signals = (macd > macd_signal)
            
            # 計算收益
            returns = data['Close'].pct_change()
            strategy_returns = returns * buy_signals.shift(1)
            
            if strategy_returns.std() == 0:
                return None
                
            sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
            
            return {
                'params': {'fast': fast, 'slow': slow, 'signal': signal},
                'sharpe_ratio': sharpe,
                'total_return': (1 + strategy_returns).prod() - 1
            }
        except:
            return None

def run_brute_force_search_addon(symbol: str = "2800.HK", 
                                strategy_type: str = "RSI",
                                start_date: str = "2020-01-01",
                                step_size: int = 1,
                                max_processes: int = None) -> dict:
    """
    🔥 專用暴力搜索入口函數（插件版）
    """
    print("🚀 啟動暴力搜索插件")
    print(f"📈 股票代號: {symbol}")
    print(f"🎯 策略類型: {strategy_type}")
    print(f"📅 開始日期: {start_date}")
    print(f"⚡ 搜索步長: {step_size}")
    
    # 載入數據
    try:
        if MAIN_SYSTEM_AVAILABLE:
            data = load_stock_data(symbol, start_date)
        else:
            # 使用模擬數據
            dates = pd.date_range(start_date, periods=1000)
            np.random.seed(42)
            prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 1000)))
            data = pd.DataFrame({
                'Date': dates,
                'Close': prices,
                'Open': prices * 0.99,
                'High': prices * 1.01,
                'Low': prices * 0.98,
                'Volume': np.random.randint(1000000, 10000000, 1000)
            })
            print("⚠️  使用模擬數據進行測試")
    except Exception as e:
        print(f"❌ 數據載入失敗: {e}")
        return {}
    
    # 創建搜索器
    searcher = BruteForceSearcher(data)
    
    # 執行搜索
    if strategy_type.upper() == 'RSI':
        result = searcher.brute_force_rsi_search(step_size, max_processes)
    elif strategy_type.upper() == 'MACD':
        result = searcher.brute_force_macd_search(step_size, max_processes)
    else:
        print(f"❌ 不支援的策略類型: {strategy_type}")
        return {}
    
    # 保存結果
    if result:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"brute_force_{strategy_type.lower()}_{symbol.replace('.', '_')}_{timestamp}.json"
        
        import json
        with open(filename, 'w', encoding='utf-8') as f:
            # 轉換numpy類型為Python原生類型
            clean_result = {}
            for k, v in result.items():
                if isinstance(v, (np.int64, np.float64)):
                    clean_result[k] = float(v)
                elif isinstance(v, list):
                    clean_result[k] = v[:10]  # 只保存前10個結果
                else:
                    clean_result[k] = v
            
            json.dump(clean_result, f, ensure_ascii=False, indent=2)
        
        print(f"💾 結果已保存: {filename}")
    
    return result

def main():
    """主函數 - 交互式界面"""
    print("=" * 80)
    print("🔥 港股量化分析系統 - 暴力搜索插件")
    print("=" * 80)
    print("1. RSI暴力搜索 (步長1 - 完全暴力)")
    print("2. RSI快速搜索 (步長5)")
    print("3. MACD暴力搜索 (步長1 - 完全暴力)")
    print("4. MACD快速搜索 (步長5)")
    print("0. 退出")
    print("=" * 80)
    
    try:
        choice = input("請選擇 (0-4): ").strip()
        
        if choice == "1":
            print("啟動RSI完全暴力搜索...")
            result = run_brute_force_search_addon("2800.HK", "RSI", step_size=1)
        elif choice == "2":
            print("啟動RSI快速搜索...")
            result = run_brute_force_search_addon("2800.HK", "RSI", step_size=5)
        elif choice == "3":
            print("啟動MACD完全暴力搜索...")
            result = run_brute_force_search_addon("2800.HK", "MACD", step_size=1)
        elif choice == "4":
            print("啟動MACD快速搜索...")
            result = run_brute_force_search_addon("2800.HK", "MACD", step_size=5)
        elif choice == "0":
            print("退出")
            return
        else:
            print("無效選擇")
            return
            
        if result:
            print("\n🎉 搜索完成!")
            print(f"📈 最佳夏普比率: {result.get('best_performance', {}).get('sharpe_ratio', 0):.3f}")
            print(f"🎯 最佳參數: {result.get('best_params', {})}")
        else:
            print("❌ 搜索失敗")
            
    except KeyboardInterrupt:
        print("\n\n👋 用戶中斷，退出程序")
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")

if __name__ == "__main__":
    main()