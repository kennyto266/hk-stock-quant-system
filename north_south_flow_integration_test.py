"""
🧪 南北水策略整合測試模組
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')
from north_south_flow_integration import NorthSouthFlowIntegration

class TestNorthSouthFlowIntegration(unittest.TestCase):
    """測試南北水策略整合"""
    
    def setUp(self):
        """測試前準備"""
        self.integration = NorthSouthFlowIntegration()
        
    def test_load_north_south_data(self):
        """測試加載南北水數據"""
        print("\n🧪 測試加載南北水數據...")
        
        # 加載數據
        df = self.integration.load_north_south_data()
        
        # 驗證數據結構
        self.assertIsInstance(df, pd.DataFrame)
        if not df.empty:
            # 驗證必要列存在
            required_columns = ['date', 'market', 'market_id', 'total_turnover', 
                              'buy_turnover', 'sell_turnover', 'net_flow', 'trade_count']
            for col in required_columns:
                self.assertTrue(col in df.columns, f"缺少列：{col}")
            
            # 驗證數據類型
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['date']))
            self.assertTrue(pd.api.types.is_numeric_dtype(df['total_turnover']))
            self.assertTrue(pd.api.types.is_numeric_dtype(df['net_flow']))
            
            # 驗證數據完整性
            self.assertFalse(df['date'].isnull().any())
            self.assertFalse(df['market'].isnull().any())
            self.assertFalse(df['total_turnover'].isnull().any())
            
            # 驗證市場類型
            expected_markets = ["滬股通北向", "滬股通南向", "深股通北向", "深股通南向"]
            actual_markets = df['market'].unique()
            for market in actual_markets:
                self.assertIn(market, expected_markets)
            
            # 驗證數據排序
            self.assertTrue(df.equals(df.sort_values(['market', 'date']).reset_index(drop=True)))
            
            print(f"✅ 成功加載 {len(df)} 條記錄，{df['date'].dt.date.nunique()} 個交易日")
            print(f"📅 數據範圍：{df['date'].min()} 到 {df['date'].max()}")
        else:
            print("⚠️ 未找到南北水數據")
    
    def test_calculate_technical_indicators(self):
        """測試計算技術指標"""
        print("\n🧪 測試計算技術指標...")
        
        # 創建測試數據
        dates = pd.date_range(start='2025-01-01', end='2025-01-30', freq='D')
        test_data = pd.DataFrame({
            'date': dates,
            'market': '滬股通北向',
            'total_turnover': np.random.uniform(1000, 2000, len(dates)),
            'buy_turnover': np.random.uniform(500, 1000, len(dates)),
            'sell_turnover': np.random.uniform(500, 1000, len(dates))
        })
        test_data['net_flow'] = test_data['buy_turnover'] - test_data['sell_turnover']
        
        # 計算RSI
        rsi = self.integration.calculate_rsi(test_data['total_turnover'])
        self.assertTrue(isinstance(rsi, pd.Series))
        self.assertTrue(np.all(np.logical_and(rsi >= 0, rsi <= 100)))
        
        # 計算MACD
        macd, signal, hist = self.integration.calculate_macd(test_data['total_turnover'])
        self.assertTrue(isinstance(macd, pd.Series))
        self.assertTrue(isinstance(signal, pd.Series))
        self.assertTrue(isinstance(hist, pd.Series))
        
        print("✅ 技術指標計算正確")
    
    def test_generate_signals(self):
        """測試生成交易信號"""
        print("\n🧪 測試生成交易信號...")
        
        # 加載實際數據
        df = self.integration.load_north_south_data()
        if df.empty:
            print("⚠️ 無法加載數據，跳過信號生成測試")
            return
        
        # 計算指標
        indicators_df = self.integration.calculate_north_south_indicators(df)
        if indicators_df.empty:
            print("⚠️ 無法計算指標，跳過信號生成測試")
            return
        
        # 生成信號
        signals_df = self.integration.generate_north_south_signals(indicators_df)
        
        # 驗證信號
        self.assertIsInstance(signals_df, pd.DataFrame)
        if not signals_df.empty:
            # 驗證信號列存在
            signal_columns = [col for col in signals_df.columns if 'signal' in col]
            self.assertTrue(len(signal_columns) > 0)
            
            # 驗證信號值
            for col in signal_columns:
                unique_signals = signals_df[col].unique()
                self.assertTrue(all(signal in [-1, 0, 1] for signal in unique_signals if pd.notna(signal)))
            
            print(f"✅ 成功生成 {len(signals_df)} 條信號記錄")
            print(f"📊 包含 {len(signal_columns)} 個信號指標")
        else:
            print("⚠️ 信號生成結果為空")
    
    def test_save_results(self):
        """測試保存結果"""
        print("\n🧪 測試保存結果...")
        
        # 加載數據並生成信號
        df = self.integration.load_north_south_data()
        if df.empty:
            print("⚠️ 無法加載數據，跳過結果保存測試")
            return
        
        indicators_df = self.integration.calculate_north_south_indicators(df)
        if indicators_df.empty:
            print("⚠️ 無法計算指標，跳過結果保存測試")
            return
        
        signals_df = self.integration.generate_north_south_signals(indicators_df)
        if signals_df.empty:
            print("⚠️ 無法生成信號，跳過結果保存測試")
            return
        
        # 保存結果
        summary_df = self.integration.save_strategy_results(signals_df)
        
        # 驗證結果
        self.assertIsInstance(summary_df, pd.DataFrame)
        if not summary_df.empty:
            # 驗證輸出文件
            output_files = ['north_south_signals.csv', 'north_south_summary.csv']
            for file in output_files:
                file_path = os.path.join(self.integration.output_dir, file)
                self.assertTrue(os.path.exists(file_path))
                self.assertTrue(os.path.getsize(file_path) > 0)
            
            print("✅ 成功保存策略結果")
            print(f"📊 生成了 {len(output_files)} 個輸出文件")
        else:
            print("⚠️ 結果保存失敗")

if __name__ == '__main__':
    unittest.main() 