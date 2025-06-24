#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試儀表板數據加載功能
"""

import sys
import os
import pandas as pd
import yfinance as yf
from datetime import datetime

# 添加項目路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import logger

def test_dashboard_stock_loading():
    """測試儀表板股票數據加載"""
    
    print("=" * 60)
    print("測試儀表板股票數據加載功能")
    print("=" * 60)
    
    # 測試1: 檢查CSV文件加載
    print("\\n測試1: CSV文件數據加載")
    csv_path = "data_output/csv"
    csv_file = os.path.join(csv_path, "0700_HK_stock_data.csv")
    
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            print(f"CSV文件加載成功: {csv_file}")
            print(f"數據形狀: {df.shape}")
            print(f"日期範圍: {df['Date'].min()} 到 {df['Date'].max()}")
            print(f"列名: {df.columns.tolist()}")
            
            # 設置索引
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # 保留必要列
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            available_cols = [col for col in required_cols if col in df.columns]
            df_processed = df[available_cols].copy()
            
            print(f"數據處理成功，處理後形狀: {df_processed.shape}")
            print(f"股價範圍: {df_processed['Close'].min():.2f} - {df_processed['Close'].max():.2f}")
            
        except Exception as e:
            print(f"CSV文件加載失敗: {str(e)}")
            return False
    else:
        print(f"CSV文件不存在: {csv_file}")
        return False
    
    # 測試2: 檢查儀表板關鍵函數
    print("\\n測試2: 儀表板核心函數")
    try:
        # 模擬儀表板的數據處理流程
        from enhanced_interactive_dashboard import calculate_technical_indicators
        
        # 計算技術指標
        df_with_indicators = calculate_technical_indicators(df_processed)
        print(f"技術指標計算成功")
        print(f"指標數據形狀: {df_with_indicators.shape}")
        print(f"包含指標: {[col for col in df_with_indicators.columns if col not in required_cols]}")
        
    except Exception as e:
        print(f"技術指標計算失敗: {str(e)}")
        return False
    
    # 測試3: 策略權益曲線生成
    print("\\n測試3: 策略權益曲線生成")
    try:
        from enhanced_interactive_dashboard import generate_strategy_equity_curve
        
        strategies = ['rsi_strategy', 'macd_strategy', 'ma_strategy', 'benchmark']
        for strategy in strategies:
            equity_curve = generate_strategy_equity_curve(df_with_indicators, strategy)
            print(f"{strategy} 權益曲線生成成功，長度: {len(equity_curve)}")
            print(f"   起始值: {equity_curve[0]:.2f}, 結束值: {equity_curve[-1]:.2f}")
        
    except Exception as e:
        print(f"策略權益曲線生成失敗: {str(e)}")
        return False
    
    # 測試4: 模擬儀表板更新函數
    print("\\n測試4: 模擬儀表板更新")
    try:
        # 模擬選中的策略和指標
        selected_strategies = ['rsi_strategy', 'macd_strategy']
        selected_indicators = ['rsi', 'macd', 'volume']
        
        print(f"選中策略: {selected_strategies}")
        print(f"選中指標: {selected_indicators}")
        
        # 生成策略數據
        strategy_data = {}
        for strategy in selected_strategies:
            strategy_values = generate_strategy_equity_curve(df_with_indicators, strategy)
            strategy_data[strategy] = pd.Series(strategy_values, index=df_with_indicators.index)
        
        print(f"策略數據生成成功，包含 {len(strategy_data)} 個策略")
        
        # 檢查數據完整性
        for strategy, data in strategy_data.items():
            print(f"   {strategy}: 長度={len(data)}, 範圍={data.min():.2f}-{data.max():.2f}")
        
    except Exception as e:
        print(f"儀表板更新模擬失敗: {str(e)}")
        return False
    
    print("\\n" + "=" * 60)
    print("所有測試通過！儀表板股票數據加載功能正常")
    print("可以啟動儀表板進行實際測試")
    print("啟動命令: python enhanced_interactive_dashboard.py")
    print("訪問地址: http://127.0.0.1:8051")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    test_dashboard_stock_loading()