#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股量化分析系統 - 獨立暴力搜索腳本
專門提供0-300範圍的暴力搜索功能，完全獨立運行
"""

import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import partial
import time
from datetime import datetime
import os
import json
import warnings
warnings.filterwarnings('ignore')

class StandaloneBruteForceSearcher:
    """獨立暴力搜索器 - 完全不依賴主系統"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    def rsi_brute_force_0_to_300(self, step_size: int = 1, max_processes: int = None):
        """RSI策略0-300範圍暴力搜索"""
        print("🔥 RSI暴力搜索 (0-300範圍)")
        print("=" * 80)
        print(f"📊 參數範圍: 0-300")
        print(f"⚡ 搜索步長: {step_size}")
        
        if max_processes is None:
            max_processes = min(32, mp.cpu_count())
        
        print(f"🚀 使用進程數: {max_processes}")
        print("⚠️  這是真正的暴力搜索，將測試大量參數組合！")
        print("=" * 80)
        
        # 生成所有參數組合
        periods = list(range(2, 301, step_size))  # RSI週期：2-300
        oversold_thresholds = list(range(10, 51, step_size))  # 超賣線：10-50
        overbought_thresholds = list(range(50, 91, step_size))  # 超買線：50-90
        
        param_combinations = []
        for period in periods:
            for oversold in oversold_thresholds:
                for overbought in overbought_thresholds:
                    if oversold < overbought:
                        param_combinations.append((period, oversold, overbought))
        
        total_combinations = len(param_combinations)
        print(f"📊 總參數組合數: {total_combinations:,}")
        
        start_time = time.time()
        
        # 並行測試
        test_func = partial(self._test_rsi_params, self.data)
        
        print("🚀 開始並行測試...")
        with mp.Pool(max_processes) as pool:
            results = pool.map(test_func, param_combinations)
        
        # 處理結果
        valid_results = [r for r in results if r is not None and r.get('sharpe_ratio', 0) > 0]
        
        if not valid_results:
            print("❌ 沒有找到有效的參數組合")
            return {}
        
        # 按夏普比率排序
        valid_results.sort(key=lambda x: x.get('sharpe_ratio', 0), reverse=True)
        best_result = valid_results[0]
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 RSI暴力搜索完成!")
        print(f"⏱️  總耗時: {total_time:.1f}秒 ({total_time/60:.1f}分鐘)")
        print(f"📊 測試組合數: {total_combinations:,}")
        print(f"✅ 有效結果數: {len(valid_results):,}")
        print(f"📈 最佳夏普比率: {best_result.get('sharpe_ratio', 0):.4f}")
        print(f"🎯 最佳參數: {best_result.get('params', {})}")
        print(f"💰 最佳總收益: {best_result.get('total_return', 0):.2%}")
        
        return {
            'strategy_type': 'RSI',
            'best_params': best_result.get('params', {}),
            'best_performance': best_result,
            'total_tested': total_combinations,
            'valid_results': len(valid_results),
            'search_time': total_time,
            'top_10_results': valid_results[:10]
        }
    
    def macd_brute_force_0_to_300(self, step_size: int = 1, max_processes: int = None):
        """MACD策略0-300範圍暴力搜索"""
        print("🔥 MACD暴力搜索 (0-300範圍)")
        print("=" * 80)
        print(f"📊 參數範圍: 0-300")
        print(f"⚡ 搜索步長: {step_size}")
        
        if max_processes is None:
            max_processes = min(32, mp.cpu_count())
        
        print(f"🚀 使用進程數: {max_processes}")
        print("⚠️  這是真正的暴力搜索，將測試大量參數組合！")
        print("=" * 80)
        
        # 生成所有參數組合
        fast_periods = list(range(3, 101, step_size))  # 快線：3-100
        slow_periods = list(range(10, 201, step_size))  # 慢線：10-200
        signal_periods = list(range(3, 51, step_size))  # 信號線：3-50
        
        param_combinations = []
        for fast in fast_periods:
            for slow in slow_periods:
                for signal in signal_periods:
                    if fast < slow:
                        param_combinations.append((fast, slow, signal))
        
        total_combinations = len(param_combinations)
        print(f"📊 總參數組合數: {total_combinations:,}")
        
        start_time = time.time()
        
        # 並行測試
        test_func = partial(self._test_macd_params, self.data)
        
        print("🚀 開始並行測試...")
        with mp.Pool(max_processes) as pool:
            results = pool.map(test_func, param_combinations)
        
        # 處理結果
        valid_results = [r for r in results if r is not None and r.get('sharpe_ratio', 0) > 0]
        
        if not valid_results:
            print("❌ 沒有找到有效的參數組合")
            return {}
        
        # 按夏普比率排序
        valid_results.sort(key=lambda x: x.get('sharpe_ratio', 0), reverse=True)
        best_result = valid_results[0]
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 MACD暴力搜索完成!")
        print(f"⏱️  總耗時: {total_time:.1f}秒 ({total_time/60:.1f}分鐘)")
        print(f"📊 測試組合數: {total_combinations:,}")
        print(f"✅ 有效結果數: {len(valid_results):,}")
        print(f"📈 最佳夏普比率: {best_result.get('sharpe_ratio', 0):.4f}")
        print(f"🎯 最佳參數: {best_result.get('params', {})}")
        print(f"💰 最佳總收益: {best_result.get('total_return', 0):.2%}")
        
        return {
            'strategy_type': 'MACD',
            'best_params': best_result.get('params', {}),
            'best_performance': best_result,
            'total_tested': total_combinations,
            'valid_results': len(valid_results),
            'search_time': total_time,
            'top_10_results': valid_results[:10]
        }
    
    def _test_rsi_params(self, data, params):
        """測試RSI參數組合"""
        period, oversold, overbought = params
        try:
            # 計算RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # 生成交易信號
            buy_signals = (rsi < oversold) & (rsi.shift(1) >= oversold)  # RSI從上方穿越超賣線
            sell_signals = (rsi > overbought) & (rsi.shift(1) <= overbought)  # RSI從下方穿越超買線
            
            # 計算持倉
            position = 0
            positions = []
            
            for i in range(len(data)):
                if buy_signals.iloc[i] and position == 0:
                    position = 1  # 買入
                elif sell_signals.iloc[i] and position == 1:
                    position = 0  # 賣出
                positions.append(position)
            
            positions = pd.Series(positions, index=data.index)
            
            # 計算策略收益
            returns = data['Close'].pct_change()
            strategy_returns = returns * positions.shift(1)
            
            # 去除無效值
            strategy_returns = strategy_returns.dropna()
            
            if len(strategy_returns) == 0 or strategy_returns.std() == 0:
                return None
                
            # 計算績效指標
            total_return = (1 + strategy_returns).prod() - 1
            annual_return = (1 + strategy_returns.mean()) ** 252 - 1
            annual_volatility = strategy_returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
            
            # 計算最大回撤
            cumulative_returns = (1 + strategy_returns).cumprod()
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = drawdown.min()
            
            return {
                'params': {'period': period, 'oversold': oversold, 'overbought': overbought},
                'sharpe_ratio': sharpe_ratio,
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'max_drawdown': max_drawdown,
                'trades': len(strategy_returns[strategy_returns != 0])
            }
        except Exception as e:
            return None
    
    def _test_macd_params(self, data, params):
        """測試MACD參數組合"""
        fast, slow, signal = params
        try:
            # 計算MACD
            exp1 = data['Close'].ewm(span=fast).mean()
            exp2 = data['Close'].ewm(span=slow).mean()
            macd = exp1 - exp2
            macd_signal = macd.ewm(span=signal).mean()
            macd_histogram = macd - macd_signal
            
            # 生成交易信號
            buy_signals = (macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1))
            sell_signals = (macd < macd_signal) & (macd.shift(1) >= macd_signal.shift(1))
            
            # 計算持倉
            position = 0
            positions = []
            
            for i in range(len(data)):
                if buy_signals.iloc[i] and position == 0:
                    position = 1  # 買入
                elif sell_signals.iloc[i] and position == 1:
                    position = 0  # 賣出
                positions.append(position)
            
            positions = pd.Series(positions, index=data.index)
            
            # 計算策略收益
            returns = data['Close'].pct_change()
            strategy_returns = returns * positions.shift(1)
            
            # 去除無效值
            strategy_returns = strategy_returns.dropna()
            
            if len(strategy_returns) == 0 or strategy_returns.std() == 0:
                return None
                
            # 計算績效指標
            total_return = (1 + strategy_returns).prod() - 1
            annual_return = (1 + strategy_returns.mean()) ** 252 - 1
            annual_volatility = strategy_returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
            
            # 計算最大回撤
            cumulative_returns = (1 + strategy_returns).cumprod()
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = drawdown.min()
            
            return {
                'params': {'fast': fast, 'slow': slow, 'signal': signal},
                'sharpe_ratio': sharpe_ratio,
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'max_drawdown': max_drawdown,
                'trades': len(strategy_returns[strategy_returns != 0])
            }
        except Exception as e:
            return None

def generate_sample_data(symbol: str = "2800.HK", start_date: str = "2020-01-01", periods: int = 1000):
    """生成示例股價數據"""
    print(f"📊 生成 {symbol} 的示例數據 (從 {start_date} 開始，{periods} 個交易日)")
    
    dates = pd.date_range(start_date, periods=periods, freq='D')
    
    # 使用更真實的股價模擬
    np.random.seed(42)  # 確保結果可重現
    
    # 模擬真實的股價波動
    returns = np.random.normal(0.001, 0.02, periods)  # 日收益率
    prices = [100]  # 起始價格
    
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))
    
    # 添加一些趨勢和週期性
    trend = np.linspace(0, 0.5, periods)
    cycle = 0.1 * np.sin(np.linspace(0, 4*np.pi, periods))
    prices = np.array(prices) * (1 + trend + cycle)
    
    # 計算其他OHLV數據
    data = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Open': prices * (1 + np.random.normal(0, 0.005, periods)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.01, periods))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, periods))),
        'Volume': np.random.randint(1000000, 10000000, periods)
    })
    
    # 確保High >= Close >= Low
    data['High'] = np.maximum(data[['Open', 'Close', 'High']].max(axis=1), data['High'])
    data['Low'] = np.minimum(data[['Open', 'Close', 'Low']].min(axis=1), data['Low'])
    
    print(f"✅ 數據生成完成: {len(data)} 行")
    print(f"📈 價格範圍: {data['Close'].min():.2f} - {data['Close'].max():.2f}")
    
    return data

def save_results(results: dict, filename: str = None):
    """保存搜索結果到JSON文件"""
    if not results:
        print("❌ 無結果可保存")
        return
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if filename is None:
        strategy_type = results.get('strategy_type', 'unknown')
        filename = f"brute_force_{strategy_type.lower()}_{timestamp}.json"
    
    # 清理numpy類型
    def clean_data(obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: clean_data(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_data(item) for item in obj]
        else:
            return obj
    
    clean_results = clean_data(results)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(clean_results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 結果已保存到: {filename}")
    return filename

def main():
    """主函數 - 提供交互式暴力搜索"""
    print("=" * 80)
    print("🔥 港股量化分析系統 - 獨立暴力搜索引擎")
    print("=" * 80)
    print("專門提供 0-300 參數範圍的暴力搜索功能")
    print("=" * 80)
    print()
    print("選項:")
    print("1. RSI完全暴力搜索 (步長1) - 最精確但最慢")
    print("2. RSI快速搜索 (步長5) - 平衡精確度和速度")
    print("3. MACD完全暴力搜索 (步長1) - 最精確但最慢")
    print("4. MACD快速搜索 (步長5) - 平衡精確度和速度")
    print("5. 自定義搜索")
    print("0. 退出")
    print("=" * 80)
    
    try:
        choice = input("請選擇 (0-5): ").strip()
        
        if choice == "0":
            print("👋 退出程序")
            return
        elif choice not in ["1", "2", "3", "4", "5"]:
            print("❌ 無效選擇")
            return
        
        # 生成示例數據
        print("\n📊 準備數據...")
        data = generate_sample_data("2800.HK", "2020-01-01", 1000)
        
        # 創建搜索器
        searcher = StandaloneBruteForceSearcher(data)
        
        # 執行搜索
        print("\n🚀 開始搜索...")
        start_time = time.time()
        
        if choice == "1":
            print("🔥 啟動RSI完全暴力搜索...")
            result = searcher.rsi_brute_force_0_to_300(step_size=1)
        elif choice == "2":
            print("⚡ 啟動RSI快速搜索...")
            result = searcher.rsi_brute_force_0_to_300(step_size=5)
        elif choice == "3":
            print("🔥 啟動MACD完全暴力搜索...")
            result = searcher.macd_brute_force_0_to_300(step_size=1)
        elif choice == "4":
            print("⚡ 啟動MACD快速搜索...")
            result = searcher.macd_brute_force_0_to_300(step_size=5)
        elif choice == "5":
            strategy = input("策略類型 (RSI/MACD): ").strip().upper()
            step = int(input("搜索步長 (1-10): ").strip())
            
            if strategy == "RSI":
                result = searcher.rsi_brute_force_0_to_300(step_size=step)
            elif strategy == "MACD":
                result = searcher.macd_brute_force_0_to_300(step_size=step)
            else:
                print("❌ 不支援的策略類型")
                return
        
        # 保存結果
        if result:
            print("\n💾 保存結果...")
            filename = save_results(result)
            
            print("\n" + "=" * 80)
            print("🎉 搜索完成！")
            print("=" * 80)
            print(f"📁 結果檔案: {filename}")
            print(f"📈 最佳夏普比率: {result['best_performance']['sharpe_ratio']:.4f}")
            print(f"🎯 最佳參數: {result['best_params']}")
            print(f"💰 最佳收益: {result['best_performance']['total_return']:.2%}")
            print(f"📊 測試組合: {result['total_tested']:,}")
            print(f"⏱️  總耗時: {result['search_time']:.1f}秒")
            print("=" * 80)
        else:
            print("❌ 搜索失敗")
            
    except KeyboardInterrupt:
        print("\n\n👋 用戶中斷，退出程序")
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()