#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股量化分析系統 - 暴力搜索測試模組
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import unittest

# 添加父目錄到 Python 路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 導入策略模組
from unified_strategy_optimizer import UnifiedStrategyOptimizer
from strategies import RSIStrategy

class TestBruteForceSearch(unittest.TestCase):
    """暴力搜索測試類"""
    
    def test_brute_force_search(self):
        """測試暴力搜索功能"""
        print("\n🔍 開始暴力搜索測試...")
        
        # 創建測試數據
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        data = pd.DataFrame({
            'Date': dates,
            'Open': np.random.randn(len(dates)).cumsum() + 100,
            'High': np.random.randn(len(dates)).cumsum() + 102,
            'Low': np.random.randn(len(dates)).cumsum() + 98,
            'Close': np.random.randn(len(dates)).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, len(dates))
        }).set_index('Date')
        
        # 創建優化器
        optimizer = UnifiedStrategyOptimizer(data=data)
        
        # 定義參數網格
        param_grid = {
            'period': list(range(5, 301, 5)),
            'overbought': list(range(50, 91, 5)),
            'oversold': list(range(10, 51, 5))
        }
        
        # 運行優化
        results = optimizer.optimize(RSIStrategy, param_grid, mode='grid')
        
        # 驗證結果
        self.assertIsNotNone(results)
        self.assertIn('best_params', results)
        self.assertIn('performance', results)
        
        # 輸出結果
        print("\n✅ 暴力搜索測試結果:")
        print(f"   最佳參數: {results['best_params']}")
        print(f"   夏普比率: {results['performance']['sharpe_ratio']:.3f}")
        print(f"   年化收益: {results['performance']['annual_return']:.2%}")
        print(f"   最大回撤: {results['performance']['max_drawdown']:.2%}")
        print(f"   交易次數: {results['performance']['trade_count']}")

if __name__ == '__main__':
    unittest.main()