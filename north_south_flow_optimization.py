#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌊 南北水策略參數優化模組
將南北水策略集成到現有的參數優化工作流中

優化策略：
1. 南北水RSI策略 - 參數優化
2. 南北水MACD策略 - 參數優化  
3. 南北水淨流入策略 - 參數優化
4. 南北水綜合策略 - 參數優化
5. 南北水均線策略 - 參數優化
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import warnings
import yfinance as yf
from multiprocessing import Pool, cpu_count
warnings.filterwarnings('ignore')
from north_south_flow_integration import NorthSouthFlowIntegration

class NorthSouthFlowOptimization:
    def __init__(self):
        self.integration = NorthSouthFlowIntegration()
        self.max_processes = min(32, cpu_count())  # 限制最大進程數
        
    def optimize_strategy_parameters(self):
        """優化南北水策略參數"""
        print("🔧 開始優化南北水策略參數...")
        
        # 加載數據
        df = self.integration.load_north_south_data()
        if df.empty:
            print("❌ 無法加載南北水數據，無法進行優化")
            return None
            
        # 按市場分組優化
        results = []
        for market in df['market'].unique():
            market_df = df[df['market'] == market].copy()
            if len(market_df) < 30:  # 需要足夠的數據進行優化
                continue
                
            print(f"\n🏢 優化 {market} 的策略參數...")
            
            # RSI參數優化
            rsi_results = self.optimize_rsi_parameters(market_df)
            if rsi_results:
                results.append({
                    'market': market,
                    'strategy': 'RSI',
                    **rsi_results
                })
            
            # MACD參數優化
            macd_results = self.optimize_macd_parameters(market_df)
            if macd_results:
                results.append({
                    'market': market,
                    'strategy': 'MACD',
                    **macd_results
                })
        
        if results:
            results_df = pd.DataFrame(results)
            print("\n✅ 策略參數優化完成！")
            self.print_optimization_results(results_df)
            return results_df
        else:
            print("❌ 無法完成策略參數優化")
            return None
            
    def _evaluate_rsi_params(self, args):
        """評估單個RSI參數組合"""
        df, period, ob, os = args
        if ob <= os:  # 超買閾值必須大於超賣閾值
            return None
            
        # 計算RSI
        prices = df['total_turnover']
        rsi = self.calculate_rsi(prices, period)
        
        # 生成交易信號
        signals = pd.Series(0, index=df.index)
        signals[rsi < os] = 1  # 買入信號
        signals[rsi > ob] = -1  # 賣出信號
        
        # 計算策略收益
        returns = self.calculate_strategy_returns(df, signals)
        if returns is not None:
            sharpe = self.calculate_sharpe_ratio(returns)
            return {
                'period': period,
                'overbought': ob,
                'oversold': os,
                'sharpe_ratio': sharpe,
                'annual_return': np.mean(returns) * 252,
                'max_drawdown': self.calculate_max_drawdown(returns)
            }
        return None
            
    def optimize_rsi_parameters(self, df):
        """使用多進程優化RSI策略參數"""
        # RSI參數範圍
        periods = range(10, 21, 1)  # RSI週期：10-20天
        overbought = range(65, 81, 1)  # 超買閾值：65-80
        oversold = range(20, 36, 1)  # 超賣閾值：20-35
        
        # 準備參數組合
        param_combinations = [(df, p, ob, os) 
                            for p in periods 
                            for ob in overbought 
                            for os in oversold]
        
        # 使用多進程進行參數優化
        with Pool(processes=self.max_processes) as pool:
            results = pool.map(self._evaluate_rsi_params, param_combinations)
        
        # 過濾有效結果並找出最佳參數
        valid_results = [r for r in results if r is not None]
        if valid_results:
            best_result = max(valid_results, key=lambda x: x['sharpe_ratio'])
            return best_result
        return None
        
    def _evaluate_macd_params(self, args):
        """評估單個MACD參數組合"""
        df, fast, slow, signal = args
        if fast >= slow:  # 快線週期必須小於慢線週期
            return None
            
        # 計算MACD
        prices = df['total_turnover']
        macd_line, signal_line, _ = self.calculate_macd(prices, fast, slow, signal)
        
        # 生成交易信號
        signals = pd.Series(0, index=df.index)
        signals[(macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))] = 1  # 金叉買入
        signals[(macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))] = -1  # 死叉賣出
        
        # 計算策略收益
        returns = self.calculate_strategy_returns(df, signals)
        if returns is not None:
            sharpe = self.calculate_sharpe_ratio(returns)
            return {
                'fast_period': fast,
                'slow_period': slow,
                'signal_period': signal,
                'sharpe_ratio': sharpe,
                'annual_return': np.mean(returns) * 252,
                'max_drawdown': self.calculate_max_drawdown(returns)
            }
        return None
        
    def optimize_macd_parameters(self, df):
        """使用多進程優化MACD策略參數"""
        # MACD參數範圍
        fast_periods = range(8, 15, 1)  # 快線：8-14天
        slow_periods = range(20, 31, 1)  # 慢線：20-30天
        signal_periods = range(7, 12, 1)  # 信號線：7-11天
        
        # 準備參數組合
        param_combinations = [(df, fast, slow, signal) 
                            for fast in fast_periods 
                            for slow in slow_periods 
                            for signal in signal_periods]
        
        # 使用多進程進行參數優化
        with Pool(processes=self.max_processes) as pool:
            results = pool.map(self._evaluate_macd_params, param_combinations)
        
        # 過濾有效結果並找出最佳參數
        valid_results = [r for r in results if r is not None]
        if valid_results:
            best_result = max(valid_results, key=lambda x: x['sharpe_ratio'])
            return best_result
        return None
    
    def calculate_rsi(self, prices, period=14):
        """計算RSI指標"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """計算MACD指標"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def calculate_strategy_returns(self, df, signals):
        """計算策略收益"""
        if len(signals) < 2:
            return None
        
        # 使用總成交額的變化率作為基準收益
        price_returns = df['total_turnover'].pct_change()
        
        # 根據信號計算策略收益
        strategy_returns = signals.shift(1) * price_returns
        strategy_returns = strategy_returns[~strategy_returns.isna()]
        
        return strategy_returns
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """計算夏普比率"""
        if len(returns) < 2:
            return -np.inf
        
        # 年化收益率和波動率
        annual_return = np.mean(returns) * 252
        annual_volatility = np.std(returns) * np.sqrt(252)
        
        if annual_volatility == 0:
            return -np.inf
            
        sharpe = (annual_return - risk_free_rate) / annual_volatility
        return sharpe
    
    def calculate_max_drawdown(self, returns):
        """計算最大回撤"""
        if len(returns) < 2:
            return 0
            
        # 計算累積收益
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        
        return abs(drawdowns.min())
    
    def print_optimization_results(self, results_df):
        """打印優化結果"""
        print("\n" + "="*60)
        print("📊 南北水策略參數優化結果")
        print("="*60)
        
        for _, row in results_df.iterrows():
            print(f"\n🏢 {row['market']} - {row['strategy']}策略:")
            if row['strategy'] == 'RSI':
                print(f"   📈 最優參數:")
                print(f"      - RSI週期: {row['period']}天")
                print(f"      - 超買閾值: {row['overbought']}")
                print(f"      - 超賣閾值: {row['oversold']}")
            else:  # MACD
                print(f"   📈 最優參數:")
                print(f"      - 快線週期: {row['fast_period']}天")
                print(f"      - 慢線週期: {row['slow_period']}天")
                print(f"      - 信號線週期: {row['signal_period']}天")
            
            print(f"   📊 策略表現:")
            print(f"      - 夏普比率: {row['sharpe_ratio']:.2f}")
            print(f"      - 年化收益: {row['annual_return']*100:.1f}%")
            print(f"      - 最大回撤: {row['max_drawdown']*100:.1f}%")
        
        print("="*60)

def run_north_south_comprehensive_optimization(symbol: str = "2800.HK", mode: str = "comprehensive") -> bool:
    """運行南北水綜合策略優化"""
    try:
        print(f"\n🌊 開始南北水綜合策略優化: {symbol} (模式: {mode})")
        
        optimizer = NorthSouthFlowOptimization()
        data = optimizer.integration.load_north_south_data()
        
        if data.empty:
            print("❌ 無南北水數據可用，跳過優化")
            return False
        
        results = []
        
        # 測試主要市場
        main_markets = ["滬股通南向", "深股通南向"]
        
        for market in main_markets:
            print(f"\n📊 正在優化 {market} 策略...")
            
            # RSI策略
            rsi_result = optimizer.optimize_rsi_parameters(data[data['market'] == market])
            if rsi_result:
                results.append({
                    'market': market,
                    'strategy': 'RSI',
                    **rsi_result
                })
            
            # MACD策略  
            macd_result = optimizer.optimize_macd_parameters(data[data['market'] == market])
            if macd_result:
                results.append({
                    'market': market,
                    'strategy': 'MACD',
                    **macd_result
                })
            
            # 淨流入策略
            flow_result = optimizer.optimize_north_south_flow_strategy(data, market)
            if flow_result and flow_result.get('performance', {}).get('sharpe_ratio', 0) > 0:
                results.append({
                    'market': market,
                    'strategy': '淨流入策略',
                    **flow_result['params']
                })
            
            # 均線策略
            sma_result = optimizer.optimize_north_south_sma_strategy(data, market)
            if sma_result and sma_result.get('performance', {}).get('sharpe_ratio', 0) > 0:
                results.append({
                    'market': market,
                    'strategy': '均線策略',
                    **sma_result['params']
                })
        
        # 排序結果並輸出
        if results:
            results_df = pd.DataFrame(results)
            results_df.sort_values(by='sharpe_ratio', ascending=False, inplace=True)
            
            print(f"\n🏆 南北水策略排名 (共 {len(results_df)} 個策略):")
            for i, (_, row) in enumerate(results_df.iterrows(), 1):
                strategy = row['strategy']
                sharpe = row['sharpe_ratio']
                annual_return = row['annual_return']
                max_dd = row['max_drawdown']
                params = row['params']
                print(f"   {i}. {strategy}")
                print(f"      夏普比率: {sharpe:.3f}, 年化收益: {annual_return:.2%}, 最大回撤: {max_dd:.2%}")
                print(f"      參數: {params}")
            
            # 保存結果為 CSV
            save_north_south_results_to_csv(results, mode)
            print(f"✅ 南北水策略結果已保存為 CSV 格式")
            
            return True
        else:
            print("❌ 沒有找到有效的南北水策略結果")
            return False
            
    except Exception as e:
        print(f"❌ 南北水綜合策略優化失敗: {e}")
        return False

def save_north_south_results_to_csv(results: list, mode: str) -> None:
    """保存南北水策略結果為 CSV 格式"""
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
                'sharpe_ratio': result['sharpe_ratio'],
                'annual_return': result['annual_return'],
                'max_drawdown': result['max_drawdown'],
                'volatility': result.get('volatility', 0),
                'total_return': result.get('total_return', 0),
                'win_rate': result.get('win_rate', 0),
                'params': str(result['params'])
            })
        
        df = pd.DataFrame(strategy_data)
        
        # 保存主要結果文件
        filename = f"{csv_dir}/north_south_strategies_optimized_{timestamp}.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"   📁 已保存: {filename}")
        
        # 為每個策略生成實際的權益曲線
        optimizer = NorthSouthFlowOptimization()
        data = optimizer.integration.load_north_south_data()
        
        if not data.empty:
            for result in results:
                strategy_name = result['strategy']
                params = result['params']
                
                # 根據策略類型生成信號和權益曲線
                if 'RSI' in strategy_name:
                    market_type = strategy_name.split('_')[-1]
                    market_column = f"{market_type}_amount"
                    rsi = optimizer.calculate_rsi(data[market_column], params['period'])
                    signals = pd.Series(0, index=data.index)
                    signals[rsi < params['oversold']] = 1
                    signals[rsi > params['overbought']] = -1
                    price_returns = data['price'].pct_change().fillna(0)
                    strategy_returns = price_returns * signals.shift(1)
                    
                elif 'MACD' in strategy_name:
                    market_type = strategy_name.split('_')[-1]
                    market_column = f"{market_type}_amount"
                    macd_data = optimizer.calculate_macd(data[market_column], 
                                                       params['fast_period'], 
                                                       params['slow_period'], 
                                                       params['signal_period'])
                    signals = pd.Series(0, index=data.index)
                    signals[macd_data[0] > macd_data[1]] = 1
                    signals[macd_data[0] < macd_data[1]] = -1
                    price_returns = data['price'].pct_change().fillna(0)
                    strategy_returns = price_returns * signals.shift(1)
                    
                elif '淨流入' in strategy_name:
                    market_type = strategy_name.split('_')[-1]
                    flow_column = f"{market_type}_amount"
                    ma_flow = optimizer.calculate_sma(data[flow_column], params['ma_period'])
                    threshold = np.percentile(data[flow_column].dropna(), params['threshold_percentile'])
                    signals = pd.Series(0, index=data.index)
                    signals[ma_flow > threshold] = 1
                    signals[ma_flow < -threshold] = -1
                    price_returns = data['price'].pct_change().fillna(0)
                    strategy_returns = price_returns * signals.shift(1)
                    
                elif '均線' in strategy_name:
                    market_type = strategy_name.split('_')[-1]
                    market_column = f"{market_type}_amount"
                    fast_sma = optimizer.calculate_sma(data[market_column], params['fast_ma'])
                    slow_sma = optimizer.calculate_sma(data[market_column], params['slow_ma'])
                    signals = pd.Series(0, index=data.index)
                    signals[fast_sma > slow_sma] = 1
                    signals[fast_sma < slow_sma] = -1
                    price_returns = data['price'].pct_change().fillna(0)
                    strategy_returns = price_returns * signals.shift(1)
                
                # 計算權益曲線
                equity_curve = (1 + strategy_returns).cumprod()
                equity_data = pd.DataFrame({
                    'Date': equity_curve.index,
                    'Equity': equity_curve.values
                })
                
                # 保存權益曲線
                strategy_name_clean = strategy_name.replace('🌊 ', '').replace(' ', '_')
                equity_filename = f"{csv_dir}/equity_curve_{strategy_name_clean}_{timestamp}.csv"
                equity_data.to_csv(equity_filename, index=False, encoding='utf-8-sig')
                print(f"   📁 已保存權益曲線: {equity_filename}")
            
    except Exception as e:
        print(f"❌ 保存南北水策略 CSV 失敗: {e}")
        raise

def main():
    """主函數"""
    optimizer = NorthSouthFlowOptimization()
    optimizer.optimize_strategy_parameters()

if __name__ == "__main__":
    main() 