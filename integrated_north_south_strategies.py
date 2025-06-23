#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌊 整合南北水策略到港股量化分析系統
將南北水技術指標策略與現有策略整合

整合策略：
1. 原有RSI、MACD、布林帶、KDJ策略
2. 新增南北水RSI策略
3. 新增南北水MACD策略
4. 新增南北水淨流入策略
5. 新增南北水綜合策略
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 導入原有策略模組
from north_south_flow_strategies import NorthSouthFlowStrategies
from data_handler import DataFetcher

class IntegratedStrategiesWithNorthSouth:
    def __init__(self, symbol="2800.HK"):
        self.symbol = symbol
        self.ns_strategies = NorthSouthFlowStrategies()
        
        print(f"🚀 整合南北水策略系統初始化 - 目標股票: {symbol}")
    
    def get_stock_data(self, start_date=None, end_date=None, period="90d"):
        """獲取股票數據"""
        print(f"📊 獲取 {self.symbol} 股票數據...")
        
        try:
            # 設置默認日期範圍
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if not start_date:
                # 從period解析天數，默認90天
                days = int(period.replace('d', '')) if 'd' in period else 90
                start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            # 使用現有的data_handler
            data = DataFetcher.get_yahoo_finance_data(self.symbol, start_date, end_date)
            
            if data is None or data.empty:
                raise ValueError("無法獲取股票數據")
            
            # 重置索引，使date成為列
            data.reset_index(inplace=True)
            if 'Date' not in data.columns:
                data['Date'] = data.index
            data['Date'] = pd.to_datetime(data['Date']).dt.date
            
            print(f"✅ 獲取股票數據：{len(data)} 條記錄")
            return data
            
        except Exception as e:
            print(f"❌ 獲取股票數據失敗：{e}")
            return pd.DataFrame()
    
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
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """計算布林帶"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    def calculate_kdj(self, high, low, close, k_period=9, d_period=3, smooth_k=3):
        """計算KDJ指標"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        rsv = 100 * (close - lowest_low) / (highest_high - lowest_low)
        k = rsv.ewm(span=smooth_k).mean()
        d = k.ewm(span=d_period).mean()
        j = 3 * k - 2 * d
        
        return k, d, j
    
    def traditional_rsi_strategy(self, stock_data):
        """傳統RSI策略"""
        data = stock_data.copy()
        data['rsi'] = self.calculate_rsi(data['Close'])
        
        data['signal'] = 0
        data.loc[data['rsi'] < 30, 'signal'] = 1  # 超賣買入
        data.loc[data['rsi'] > 70, 'signal'] = -1  # 超買賣出
        
        data['position'] = data['signal'].replace(to_replace=0, method='ffill').fillna(0)
        data['returns'] = data['Close'].pct_change()
        data['strategy_returns'] = data['position'].shift(1) * data['returns']
        
        return data[['Date', 'signal', 'position', 'returns', 'strategy_returns', 'rsi']].copy()
    
    def traditional_macd_strategy(self, stock_data):
        """傳統MACD策略"""
        data = stock_data.copy()
        macd, signal, histogram = self.calculate_macd(data['Close'])
        data['macd'] = macd
        data['macd_signal'] = signal
        data['macd_histogram'] = histogram
        
        data['signal'] = 0
        # 金叉買入
        data.loc[(data['macd'] > data['macd_signal']) & 
                 (data['macd'].shift(1) <= data['macd_signal'].shift(1)), 'signal'] = 1
        # 死叉賣出
        data.loc[(data['macd'] < data['macd_signal']) & 
                 (data['macd'].shift(1) >= data['macd_signal'].shift(1)), 'signal'] = -1
        
        data['position'] = data['signal'].replace(to_replace=0, method='ffill').fillna(0)
        data['returns'] = data['Close'].pct_change()
        data['strategy_returns'] = data['position'].shift(1) * data['returns']
        
        return data[['Date', 'signal', 'position', 'returns', 'strategy_returns', 'macd']].copy()
    
    def bollinger_bands_strategy(self, stock_data):
        """布林帶策略"""
        data = stock_data.copy()
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data['Close'])
        data['bb_upper'] = bb_upper
        data['bb_middle'] = bb_middle
        data['bb_lower'] = bb_lower
        
        data['signal'] = 0
        data.loc[data['Close'] < data['bb_lower'], 'signal'] = 1  # 價格觸及下軌買入
        data.loc[data['Close'] > data['bb_upper'], 'signal'] = -1  # 價格觸及上軌賣出
        
        data['position'] = data['signal'].replace(to_replace=0, method='ffill').fillna(0)
        data['returns'] = data['Close'].pct_change()
        data['strategy_returns'] = data['position'].shift(1) * data['returns']
        
        return data[['Date', 'signal', 'position', 'returns', 'strategy_returns', 'bb_upper', 'bb_lower']].copy()
    
    def kdj_strategy(self, stock_data):
        """KDJ策略"""
        data = stock_data.copy()
        k, d, j = self.calculate_kdj(data['High'], data['Low'], data['Close'])
        data['k'] = k
        data['d'] = d
        data['j'] = j
        
        data['signal'] = 0
        data.loc[(data['k'] < 20) & (data['d'] < 20), 'signal'] = 1  # 超賣買入
        data.loc[(data['k'] > 80) & (data['d'] > 80), 'signal'] = -1  # 超買賣出
        
        data['position'] = data['signal'].replace(to_replace=0, method='ffill').fillna(0)
        data['returns'] = data['Close'].pct_change()
        data['strategy_returns'] = data['position'].shift(1) * data['returns']
        
        return data[['Date', 'signal', 'position', 'returns', 'strategy_returns', 'k', 'd', 'j']].copy()
    
    def align_dates(self, stock_strategy, ns_strategy, date_col='Date'):
        """對齊股票策略和南北水策略的日期"""
        if stock_strategy.empty or ns_strategy.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # 確保日期格式一致
        stock_strategy[date_col] = pd.to_datetime(stock_strategy[date_col]).dt.date
        ns_strategy['date'] = pd.to_datetime(ns_strategy['date']).dt.date
        
        # 找到共同日期
        common_dates = set(stock_strategy[date_col]).intersection(set(ns_strategy['date']))
        
        if not common_dates:
            return pd.DataFrame(), pd.DataFrame()
        
        # 篩選共同日期的數據
        stock_aligned = stock_strategy[stock_strategy[date_col].isin(common_dates)].sort_values(date_col)
        ns_aligned = ns_strategy[ns_strategy['date'].isin(common_dates)].sort_values('date')
        
        return stock_aligned.reset_index(drop=True), ns_aligned.reset_index(drop=True)
    
    def create_combined_strategy(self, stock_strategy, ns_strategy, strategy_name):
        """創建股票與南北水結合的策略"""
        stock_aligned, ns_aligned = self.align_dates(stock_strategy, ns_strategy)
        
        if stock_aligned.empty or ns_aligned.empty:
            return pd.DataFrame()
        
        # 創建結合策略DataFrame
        combined = pd.DataFrame()
        combined['date'] = stock_aligned['Date']
        combined['stock_signal'] = stock_aligned['signal']
        combined['ns_signal'] = ns_aligned.get('signal', ns_aligned.get('combined_signal', ns_aligned.get('final_signal', 0)))
        
        # 結合信號邏輯：兩個信號同向才發出信號
        combined['combined_signal'] = 0
        combined.loc[(combined['stock_signal'] == 1) & (combined['ns_signal'] == 1), 'combined_signal'] = 1
        combined.loc[(combined['stock_signal'] == -1) & (combined['ns_signal'] == -1), 'combined_signal'] = -1
        
        # 也可以考慮其他結合方式
        combined['weighted_signal'] = (combined['stock_signal'] * 0.6 + combined['ns_signal'] * 0.4)
        combined['final_signal'] = 0
        combined.loc[combined['weighted_signal'] > 0.5, 'final_signal'] = 1
        combined.loc[combined['weighted_signal'] < -0.5, 'final_signal'] = -1
        
        combined['position'] = combined['final_signal'].replace(to_replace=0, method='ffill').fillna(0)
        combined['returns'] = stock_aligned['returns']
        combined['strategy_returns'] = combined['position'].shift(1) * combined['returns']
        
        combined['strategy_name'] = strategy_name
        
        return combined
    
    def run_all_integrated_strategies(self, period="90d"):
        """運行所有整合策略"""
        print("🚀 開始運行所有整合策略...")
        
        # 1. 獲取股票數據
        stock_data = self.get_stock_data(period=period)
        if stock_data.empty:
            print("❌ 無法獲取股票數據，停止運行")
            return {}
        
        # 2. 獲取南北水策略
        try:
            ns_strategies = self.ns_strategies.get_all_north_south_strategies()
        except Exception as e:
            print(f"❌ 獲取南北水策略失敗：{e}")
            ns_strategies = {}
        
        # 3. 計算傳統股票策略
        print("📊 計算傳統股票策略...")
        traditional_strategies = {}
        
        try:
            traditional_strategies['RSI策略'] = self.traditional_rsi_strategy(stock_data)
            traditional_strategies['MACD策略'] = self.traditional_macd_strategy(stock_data)
            traditional_strategies['布林帶策略'] = self.bollinger_bands_strategy(stock_data)
            traditional_strategies['KDJ策略'] = self.kdj_strategy(stock_data)
        except Exception as e:
            print(f"⚠️ 計算傳統策略時出錯：{e}")
        
        # 4. 創建整合策略
        print("🌊 創建股票與南北水整合策略...")
        integrated_strategies = {}
        
        # 將南北水策略與股票策略結合
        for trad_name, trad_strategy in traditional_strategies.items():
            for ns_name, ns_strategy in ns_strategies.items():
                if not ns_strategy.empty:
                    integrated_name = f"{trad_name}_+_{ns_name}"
                    try:
                        integrated_result = self.create_combined_strategy(trad_strategy, ns_strategy, integrated_name)
                        if not integrated_result.empty:
                            integrated_strategies[integrated_name] = integrated_result
                    except Exception as e:
                        print(f"⚠️ 創建整合策略 {integrated_name} 失敗：{e}")
        
        # 5. 合併所有策略結果
        all_strategies = {}
        all_strategies.update(traditional_strategies)
        all_strategies.update({f"南北水_{k}": v for k, v in ns_strategies.items()})
        all_strategies.update(integrated_strategies)
        
        print(f"✅ 完成所有策略計算，共 {len(all_strategies)} 個策略")
        
        return all_strategies
    
    def calculate_strategy_performance(self, strategy_df):
        """計算策略績效指標"""
        if strategy_df.empty or 'strategy_returns' not in strategy_df.columns:
            return {}
        
        strategy_returns = strategy_df['strategy_returns'].dropna()
        
        if len(strategy_returns) == 0 or strategy_returns.std() == 0:
            return {}
        
        # 基本績效指標
        total_return = (1 + strategy_returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1 if len(strategy_returns) > 0 else 0
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # 最大回撤
        cumulative_returns = (1 + strategy_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 勝率
        win_rate = (strategy_returns > 0).mean()
        
        # 交易次數
        if 'signal' in strategy_df.columns:
            trades = (strategy_df['signal'] != 0).sum()
        elif 'final_signal' in strategy_df.columns:
            trades = (strategy_df['final_signal'] != 0).sum()
        else:
            trades = len(strategy_returns)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': trades
        }
    
    def create_performance_summary(self, all_strategies):
        """創建績效總結"""
        print("📊 計算策略績效...")
        
        summary_data = []
        
        for strategy_name, strategy_df in all_strategies.items():
            performance = self.calculate_strategy_performance(strategy_df)
            
            if performance and not np.isnan(performance.get('sharpe_ratio', 0)):
                # 判斷策略類型
                if '南北水' in strategy_name and '+' in strategy_name:
                    strategy_type = "整合策略"
                elif '南北水' in strategy_name:
                    strategy_type = "南北水策略"
                else:
                    strategy_type = "傳統策略"
                
                summary_data.append({
                    'strategy_name': strategy_name,
                    'strategy_type': strategy_type,
                    'sharpe_ratio': performance['sharpe_ratio'],
                    'annual_return': performance['annual_return'],
                    'max_drawdown': performance['max_drawdown'],
                    'win_rate': performance['win_rate'],
                    'total_trades': performance['total_trades'],
                    'volatility': performance['volatility']
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        if not summary_df.empty:
            summary_df = summary_df.sort_values('sharpe_ratio', ascending=False)
        
        return summary_df
    
    def save_results(self, all_strategies, summary_df):
        """保存結果到CSV"""
        print("💾 保存結果...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存績效摘要
        summary_filename = f"data_output/csv/integrated_strategies_summary_{timestamp}.csv"
        summary_df.to_csv(summary_filename, index=False, encoding='utf-8-sig')
        
        # 保存詳細策略結果
        for strategy_name, strategy_df in all_strategies.items():
            if not strategy_df.empty:
                # 清理文件名中的特殊字符
                clean_name = strategy_name.replace('/', '_').replace('+', 'plus').replace(' ', '_')
                filename = f"data_output/csv/strategy_{clean_name}_{timestamp}.csv"
                strategy_df.to_csv(filename, index=False, encoding='utf-8-sig')
        
        print(f"✅ 結果已保存到 data_output/csv/ 目錄")
        print(f"📊 績效摘要: {summary_filename}")
        
        return summary_filename

def main():
    """主函數：演示整合南北水策略系統"""
    print("🌊 港股量化分析系統 - 南北水策略整合版")
    print("="*60)
    
    # 創建整合策略系統
    integrated_system = IntegratedStrategiesWithNorthSouth("2800.HK")
    
    # 運行所有整合策略
    all_strategies = integrated_system.run_all_integrated_strategies(period="90d")
    
    if not all_strategies:
        print("❌ 無法計算任何策略")
        return
    
    # 創建績效摘要
    summary = integrated_system.create_performance_summary(all_strategies)
    
    # 保存結果
    summary_file = integrated_system.save_results(all_strategies, summary)
    
    # 顯示結果
    print("\n🏆 策略績效排名（按夏普比率）:")
    print("="*80)
    
    for _, row in summary.head(10).iterrows():  # 顯示前10名
        print(f"📈 {row['strategy_name']}")
        print(f"   類型: {row['strategy_type']}")
        print(f"   夏普比率: {row['sharpe_ratio']:.3f}")
        print(f"   年化收益: {row['annual_return']:.2%}")
        print(f"   最大回撤: {row['max_drawdown']:.2%}")
        print(f"   勝率: {row['win_rate']:.2%}")
        print(f"   交易次數: {row['total_trades']}")
        print("-" * 60)
    
    # 統計各類型策略
    print("\n📊 策略類型統計:")
    type_stats = summary.groupby('strategy_type').agg({
        'sharpe_ratio': ['mean', 'max'],
        'strategy_name': 'count'
    }).round(3)
    print(type_stats)

if __name__ == "__main__":
    main() 