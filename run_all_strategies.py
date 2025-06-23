#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股量化分析系統 - 自動運行所有技術指標策略
一鍵執行所有策略優化，無需用戶交互
"""

import os
import sys
import time
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# 添加當前目錄到Python路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def main():
    """主運行函數 - 自動執行所有策略"""
    print("🎯 港股量化分析系統 - 多技術指標策略優化")
    print("=" * 60)
    print("🚀 自動運行模式：RSI, MACD, 布林帶, KDJ, Stochastic, CCI, 威廉指標%R")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # 導入策略模組
        from strategies import run_comprehensive_optimization
        print("✅ 策略模組載入成功")
        
        # 設定參數
        symbol = "2800.HK"
        start_date = "2020-01-01"
        
        print(f"\n📊 目標股票: {symbol}")
        print(f"📅 分析期間: {start_date} 至今")
        print(f"🔍 分析模式: 全面優化 (7個技術指標)")
        
        # 運行綜合策略優化
        print("\n🔥 開始執行全面策略優化...")
        success = run_comprehensive_optimization(symbol, start_date, "comprehensive")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if success:
            print(f"\n✅ 策略優化成功完成!")
            print(f"⏱️ 總執行時間: {execution_time:.2f}秒")
            print(f"📁 結果保存在: data_output/reports/")
            print(f"🌐 Dashboard: 將自動啟動 http://localhost:8050")
        else:
            print(f"\n❌ 策略優化失敗")
            print(f"⏱️ 執行時間: {execution_time:.2f}秒")
            
    except ImportError as e:
        print(f"❌ 模組導入失敗: {e}")
        print("💡 請確保所有依賴已正確安裝: pip install -r requirements.txt")
        
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")
        print("💡 請檢查網絡連接和數據源可用性")
        
    finally:
        print("\n" + "=" * 60)
        print("🎉 港股量化分析系統運行完成")
        print("=" * 60)

if __name__ == "__main__":
    main() 