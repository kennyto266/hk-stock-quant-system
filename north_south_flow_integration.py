#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌊 南北水策略整合模組
將南北水技術指標整合到港股量化分析系統中

主要功能：
1. 從北水json數據計算技術指標
2. 生成南北水策略信號
3. 與現有策略系統無縫整合
4. 支援CSV格式輸出，適配Dashboard顯示
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class NorthSouthFlowIntegration:
    def __init__(self):
        self.north_data_dir = "../../../北水json"
        self.output_dir = "north_south_integration"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 市場定義
        self.markets = {
            0: "滬股通北向",    # SSE Northbound
            1: "滬股通南向",    # SSE Southbound  
            2: "深股通北向",    # SZSE Northbound
            3: "深股通南向"     # SZSE Southbound
        }
        
        # 技術指標參數
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        print("🌊 南北水技術指標整合系統初始化")
        print(f"📂 北水數據目錄：{self.north_data_dir}")
        print(f"📊 輸出目錄：{self.output_dir}")
    
    def load_north_south_data(self):
        """加載南北水數據"""
        print("📁 加載南北水數據...")
        
        # 檢查數據目錄是否存在
        if not os.path.exists(self.north_data_dir):
            print(f"❌ 北水數據目錄不存在: {self.north_data_dir}")
            return pd.DataFrame()
        
        # 獲取所有可用的JSON文件
        json_files = [f for f in os.listdir(self.north_data_dir) if f.endswith('.json')]
        if not json_files:
            print("❌ 未找到任何南北水數據文件")
            return pd.DataFrame()
        
        json_files.sort()  # 按日期排序
        
        # 從文件名獲取日期範圍
        try:
            first_date = datetime.strptime(json_files[0][:8], '%Y%m%d').date()
            last_date = datetime.strptime(json_files[-1][:8], '%Y%m%d').date()
            print(f"📅 使用可用數據範圍: {first_date} 到 {last_date}")
        except Exception as e:
            print(f"⚠️ 無法解析日期範圍: {e}")
            return pd.DataFrame()
        
        all_records = []
        for filename in json_files:
            try:
                file_path = os.path.join(self.north_data_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                records = self.parse_daily_data(data, filename)
                if records:
                    all_records.extend(records)
            except Exception as e:
                print(f"⚠️ 讀取 {filename} 失敗：{e}")
                continue
        
        if not all_records:
            print("❌ 沒有成功加載任何南北水數據")
            return pd.DataFrame()
        
        # 轉換為DataFrame
        df = pd.DataFrame(all_records)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by=['market', 'date']).reset_index(drop=True)
        
        print(f"✅ 加載南北水數據：{len(df)} 條記錄，{df['date'].dt.date.nunique()} 個交易日")
        return df
    
    def parse_daily_data(self, data, filename):
        """解析單日南北水數據"""
        try:
            date_str = filename[:8]  # 20250620
            date_obj = datetime.strptime(date_str, '%Y%m%d').date()
            
            records = []
            
            for market_data in data:
                market_id = market_data['id']
                market_name = self.markets.get(market_id, f"市場{market_id}")
                
                if 'content' in market_data and market_data['content']:
                    # 提取總成交額和其他數據
                    total_turnover = 0
                    buy_turnover = 0
                    sell_turnover = 0
                    trade_count = 0
                    
                    for content in market_data['content']:
                        if content.get('style') == 1:  # 總計數據
                            table = content.get('table', {})
                            tr_data = table.get('tr', [])
                            
                            if tr_data:
                                try:
                                    # 處理不同市場的數據格式
                                    if '北向' in market_name:
                                        # 北向：只有總成交額
                                        total_turnover = float(tr_data[0]['td'][0][0].replace(',', ''))
                                        if len(tr_data) > 1:
                                            trade_count = int(tr_data[1]['td'][0][0].replace(',', ''))
                                    else:
                                        # 南向：有買入和賣出
                                        total_turnover = float(tr_data[0]['td'][0][0].replace(',', ''))
                                        if len(tr_data) > 1:
                                            buy_turnover = float(tr_data[1]['td'][0][0].replace(',', ''))
                                        if len(tr_data) > 2:
                                            sell_turnover = float(tr_data[2]['td'][0][0].replace(',', ''))
                                        if len(tr_data) > 3:
                                            trade_count = int(tr_data[3]['td'][0][0].replace(',', ''))
                                            
                                except (ValueError, IndexError, KeyError) as e:
                                    print(f"⚠️ 解析 {filename} 的市場 {market_name} 數據時出錯：{e}")
                                    continue
                    
                    records.append({
                        'date': date_obj,
                        'market': market_name,
                        'market_id': market_id,
                        'total_turnover': total_turnover,
                        'buy_turnover': buy_turnover,
                        'sell_turnover': sell_turnover,
                        'net_flow': buy_turnover - sell_turnover,
                        'trade_count': trade_count
                    })
            
            return records
            
        except Exception as e:
            print(f"❌ 解析 {filename} 失敗：{e}")
            return []
    
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
    
    def calculate_technical_indicators(self, df):
        """計算南北水技術指標"""
        print("📊 計算南北水技術指標...")
        
        if df.empty:
            print("❌ 無數據可用於計算技術指標")
            return pd.DataFrame()
        
        # 按市場分組計算
        result_dfs = []
        for market in df['market'].unique():
            market_df = df[df['market'] == market].copy()
            if len(market_df) < 20:  # 需要足夠的數據計算指標
                print(f"⚠️ {market} 數據不足，跳過技術指標計算")
                continue
                
            # 按日期排序
            market_df = market_df.sort_values('date')
            
            # 計算淨流入相關指標
            market_df['net_flow'] = market_df['buy_turnover'] - market_df['sell_turnover']
            market_df['net_flow_ratio'] = market_df['net_flow'] / market_df['total_turnover']
            
            # 計算RSI
            delta = market_df['net_flow'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            market_df['ns_rsi'] = 100 - (100 / (1 + rs))
            
            # 計算MACD
            exp1 = market_df['net_flow'].ewm(span=self.macd_fast, adjust=False).mean()
            exp2 = market_df['net_flow'].ewm(span=self.macd_slow, adjust=False).mean()
            market_df['ns_macd'] = exp1 - exp2
            market_df['ns_macd_signal'] = market_df['ns_macd'].ewm(span=self.macd_signal, adjust=False).mean()
            market_df['ns_macd_hist'] = market_df['ns_macd'] - market_df['ns_macd_signal']
            
            # 計算移動平均
            market_df['ma5'] = market_df['net_flow'].rolling(window=5).mean()
            market_df['ma10'] = market_df['net_flow'].rolling(window=10).mean()
            market_df['ma20'] = market_df['net_flow'].rolling(window=20).mean()
            
            # 處理NaN值
            market_df = market_df.fillna(method='bfill')
            result_dfs.append(market_df)
        
        if not result_dfs:
            print("❌ 無法計算任何市場的技術指標")
            return pd.DataFrame()
            
        # 合併所有市場的結果
        result_df = pd.concat(result_dfs, ignore_index=True)
        print(f"✅ 技術指標計算完成：{len(result_df)} 條記錄")
        
        return result_df
    
    def generate_north_south_signals(self, df):
        """生成南北水策略信號"""
        print("🎯 生成南北水策略信號...")
        
        if df.empty:
            print("❌ 無數據可用於生成信號")
            return pd.DataFrame()
        
        signals_df = df.copy()
        
        # 初始化信號列
        signals_df['rsi_signal'] = 0  # 1: 買入, -1: 賣出, 0: 持有
        signals_df['macd_signal'] = 0
        signals_df['ma_signal'] = 0
        signals_df['combined_signal'] = 0
        
        # 按市場分組處理
        result_dfs = []
        for market in signals_df['market'].unique():
            market_df = signals_df[signals_df['market'] == market].copy()
            if len(market_df) < 20:
                print(f"⚠️ {market} 數據不足，跳過信號生成")
                continue
            
            # RSI信號
            market_df.loc[market_df['ns_rsi'] > 70, 'rsi_signal'] = -1  # 超買
            market_df.loc[market_df['ns_rsi'] < 30, 'rsi_signal'] = 1   # 超賣
            
            # MACD信號
            market_df.loc[market_df['ns_macd'] > market_df['ns_macd_signal'], 'macd_signal'] = 1
            market_df.loc[market_df['ns_macd'] < market_df['ns_macd_signal'], 'macd_signal'] = -1
            
            # 移動平均信號
            market_df.loc[market_df['ma5'] > market_df['ma20'], 'ma_signal'] = 1
            market_df.loc[market_df['ma5'] < market_df['ma20'], 'ma_signal'] = -1
            
            # 計算綜合信號
            market_df['combined_signal'] = (
                market_df['rsi_signal'] * 0.3 +
                market_df['macd_signal'] * 0.4 +
                market_df['ma_signal'] * 0.3
            )
            
            # 計算信號強度（0-100）
            market_df['signal_strength'] = abs(market_df['combined_signal']) * 100
            
            # 判斷市場趨勢
            market_df['market_trend'] = '盤整'
            market_df.loc[market_df['combined_signal'] > 0.5, 'market_trend'] = '上升'
            market_df.loc[market_df['combined_signal'] < -0.5, 'market_trend'] = '下降'
            
            # 計算策略績效指標
            market_df['strategy_returns'] = market_df['combined_signal'].shift(1) * market_df['net_flow_ratio']
            market_df['cumulative_returns'] = (1 + market_df['strategy_returns']).cumprod()
            
            # 計算最大回撤
            rolling_max = market_df['cumulative_returns'].expanding().max()
            drawdowns = market_df['cumulative_returns'] / rolling_max - 1
            market_df['max_drawdown'] = drawdowns.min()
            
            # 計算勝率
            market_df['win'] = market_df['strategy_returns'] > 0
            market_df['win_rate'] = market_df['win'].expanding().mean()
            
            result_dfs.append(market_df)
            print(f"✅ {market} 信號生成完成")
        
        if not result_dfs:
            print("❌ 無法生成任何市場的信號")
            return pd.DataFrame()
        
        # 合併所有市場的結果
        result_df = pd.concat(result_dfs, ignore_index=True)
        print(f"✅ 信號生成完成：{len(result_df)} 條記錄")
        
        return result_df
    
    def save_strategy_results(self, df):
        """保存策略結果"""
        print("💾 保存南北水策略結果...")
        
        if df.empty:
            print("❌ 無數據可保存")
            return pd.DataFrame()
        
        # 創建輸出目錄
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 保存完整數據
        signals_file = os.path.join(self.output_dir, 'north_south_signals.csv')
        df.to_csv(signals_file, index=False, encoding='utf-8-sig')
        print(f"✅ 信號數據已保存至：{signals_file}")
        
        # 生成策略摘要
        summary_data = []
        for market in df['market'].unique():
            market_df = df[df['market'] == market].copy()
            if market_df.empty:
                continue
                
            # 獲取最新數據
            latest_data = market_df.iloc[-1]
            
            # 計算策略績效
            total_returns = market_df['strategy_returns'].sum()
            win_rate = market_df['win_rate'].iloc[-1] if 'win_rate' in market_df.columns else None
            max_drawdown = market_df['max_drawdown'].iloc[-1] if 'max_drawdown' in market_df.columns else None
            
            # 統計信號
            buy_signals = len(market_df[market_df['combined_signal'] > 0.5])
            sell_signals = len(market_df[market_df['combined_signal'] < -0.5])
            hold_signals = len(market_df) - buy_signals - sell_signals
            
            # 生成市場摘要
            market_summary = {
                '市場': market,
                '最新日期': latest_data['date'].strftime('%Y-%m-%d'),
                '總成交額': latest_data['total_turnover'],
                '淨流入': latest_data['net_flow'],
                '淨流入比率': latest_data['net_flow_ratio'],
                'RSI': latest_data['ns_rsi'],
                'MACD': latest_data['ns_macd'],
                'MACD信號線': latest_data['ns_macd_signal'],
                'MACD柱狀': latest_data['ns_macd_hist'],
                '5日均線': latest_data['ma5'],
                '10日均線': latest_data['ma10'],
                '20日均線': latest_data['ma20'],
                'RSI信號': latest_data['rsi_signal'],
                'MACD信號': latest_data['macd_signal'],
                'MA信號': latest_data['ma_signal'],
                '綜合信號': latest_data['combined_signal'],
                '信號強度': latest_data['signal_strength'],
                '市場趨勢': latest_data['market_trend'],
                '買入信號數': buy_signals,
                '賣出信號數': sell_signals,
                '無信號數': hold_signals,
                '總收益率': total_returns,
                '勝率': win_rate if win_rate is not None else 'N/A',
                '最大回撤': max_drawdown if max_drawdown is not None else 'N/A'
            }
            summary_data.append(market_summary)
        
        if not summary_data:
            print("❌ 無法生成策略摘要")
            return pd.DataFrame()
        
        # 保存策略摘要
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(self.output_dir, 'strategy_summary.csv')
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        print(f"✅ 策略摘要已保存至：{summary_file}")
        
        return summary_df

    def integrate_north_south_flow(self):
        """整合南北水策略"""
        print("\n🌊 開始整合南北水策略...")
        
        # 1. 加載數據
        df = self.load_north_south_data()
        if df.empty:
            print("❌ 無法加載南北水數據，整合終止")
            return None
            
        # 2. 計算技術指標
        df = self.calculate_technical_indicators(df)
        if df.empty:
            print("❌ 計算技術指標失敗，整合終止")
            return None
            
        # 3. 生成策略信號
        signals_df = self.generate_north_south_signals(df)
        if signals_df.empty:
            print("❌ 生成策略信號失敗，整合終止")
            return None
            
        # 4. 保存策略結果
        summary_df = self.save_strategy_results(signals_df)
        if summary_df.empty:
            print("❌ 保存策略結果失敗，整合終止")
            return None
            
        print("\n📊 南北水策略整合完成")
        print("\n策略摘要：")
        for _, row in summary_df.iterrows():
            print(f"\n{row['市場']}:")
            print(f"  📅 最新日期: {row['最新日期']}")
            print(f"  💰 總成交額: {row['總成交額']:,.0f}")
            print(f"  💹 淨流入: {row['淨流入']:,.0f}")
            print(f"  📈 技術指標:")
            print(f"    - RSI: {row['RSI']:.2f}")
            print(f"    - MACD: {row['MACD']:.4f}")
            print(f"  🎯 信號統計:")
            print(f"    - 買入信號: {row['買入信號數']}")
            print(f"    - 賣出信號: {row['賣出信號數']}")
            print(f"    - 無信號: {row['無信號數']}")
            print(f"  📊 策略表現:")
            print(f"    - 勝率: {row['勝率']}")
            print(f"    - 總收益率: {row['總收益率']}")
            print(f"    - 最大回撤: {row['最大回撤']}")
            print(f"  🔮 當前狀態:")
            print(f"    - 綜合信號: {row['綜合信號']:.2f}")
            print(f"    - 信號強度: {row['信號強度']:.2f}")
            print(f"    - 市場趨勢: {row['市場趨勢']}")
        
        return summary_df

def main():
    """主函數"""
    integration = NorthSouthFlowIntegration()
    
    # 加載數據
    ns_data = integration.load_north_south_data()
    if not ns_data.empty:
        # 計算指標
        ns_indicators = integration.calculate_technical_indicators(ns_data)
        if not ns_indicators.empty:
            # 生成信號
            ns_signals = integration.generate_north_south_signals(ns_indicators)
            if not ns_signals.empty:
                # 保存結果
                integration.save_strategy_results(ns_signals)
                print("✅ 南北水策略整合完成！")
            else:
                print("❌ 無法生成策略信號")
        else:
            print("❌ 無法計算技術指標")
    else:
        print("❌ 無法加載南北水數據")

if __name__ == "__main__":
    main() 