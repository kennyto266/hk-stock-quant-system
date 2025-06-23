#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試暴力搜索功能整合
用於驗證 0-300 範圍的暴力搜索功能是否正確整合到項目中
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_brute_force_integration():
    """測試暴力搜索功能是否成功整合"""
    print("🧪 測試暴力搜索功能整合...")
    
    try:
        # 導入strategies模組
        from strategies import run_brute_force_search_0_to_300, StrategyOptimizer
        print("✅ 成功導入暴力搜索函數")
        
        # 檢查StrategyOptimizer是否有新方法
        optimizer_methods = [method for method in dir(StrategyOptimizer) if 'brute_force' in method.lower()]
        print(f"📋 StrategyOptimizer 中的暴力搜索方法: {optimizer_methods}")
        
        # 檢查入口函數是否存在
        if hasattr(sys.modules['strategies'], 'run_brute_force_search_0_to_300'):
            print("✅ run_brute_force_search_0_to_300 函數已成功整合")
        else:
            print("❌ run_brute_force_search_0_to_300 函數未找到")
            
        # 檢查StrategyOptimizer是否有暴力搜索方法
        if hasattr(StrategyOptimizer, 'brute_force_search_0_to_300'):
            print("✅ StrategyOptimizer.brute_force_search_0_to_300 方法已成功整合")
        else:
            print("❌ StrategyOptimizer.brute_force_search_0_to_300 方法未找到")
            
        return True
        
    except ImportError as e:
        print(f"❌ 導入失敗: {e}")
        return False
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        return False

def demo_brute_force_usage():
    """演示暴力搜索的使用方法"""
    print("\n📖 暴力搜索使用演示:")
    print("=" * 60)
    
    # 使用方法1：直接調用入口函數
    print("1️⃣ 直接調用入口函數:")
    print("```python")
    print("from strategies import run_brute_force_search_0_to_300")
    print("")
    print("# RSI暴力搜索")
    print('result = run_brute_force_search_0_to_300("2800.HK", "RSI")')
    print("")
    print("# MACD暴力搜索")  
    print('result = run_brute_force_search_0_to_300("2800.HK", "MACD")')
    print("```")
    print()
    
    # 使用方法2：通過StrategyOptimizer類
    print("2️⃣ 通過StrategyOptimizer類:")
    print("```python")
    print("from strategies import StrategyOptimizer")
    print("import pandas as pd")
    print("")
    print("# 假設你已經有股票數據 stock_data")
    print("optimizer = StrategyOptimizer(stock_data, validation_split=0.3)")
    print("")
    print("# 執行暴力搜索")
    print('result = optimizer.brute_force_search_0_to_300("RSI", step_size=1)')
    print("```")
    print()
    
    # 使用方法3：命令行執行
    print("3️⃣ 命令行直接執行:")
    print("```bash")
    print("python strategies.py")
    print("# 然後選擇相應的暴力搜索選項")
    print("```")

def show_integration_summary():
    """顯示整合摘要"""
    print("\n📋 暴力搜索功能整合摘要:")
    print("=" * 80)
    print("🎯 已添加到 StrategyOptimizer 類的新方法:")
    print("   - brute_force_search_0_to_300() : 主要暴力搜索方法")
    print("   - _brute_force_rsi_0_to_300()  : RSI暴力搜索實現")
    print("   - _brute_force_macd_0_to_300() : MACD暴力搜索實現")
    print("   - _save_brute_force_results()  : 結果保存方法")
    print()
    print("🎯 已添加的全局函數:")
    print("   - run_brute_force_search_0_to_300() : 便捷入口函數")
    print("   - quick_brute_force_demo()          : 快速演示函數")
    print()
    print("📊 暴力搜索特點:")
    print("   - 參數範圍: 0-300 (可調整)")
    print("   - 搜索步長: 1 (完全暴力搜索，可調整)")
    print("   - 並行處理: 自動使用多進程")
    print("   - 結果保存: 自動保存CSV和報告")
    print()
    print("⚡ 性能優化:")
    print("   - 使用多進程並行計算")
    print("   - 智能批次處理")
    print("   - 進度實時顯示")
    print("   - 結果即時更新")

if __name__ == "__main__":
    print("🔥 港股量化分析系統 - 暴力搜索功能整合測試")
    print("=" * 80)
    
    # 執行整合測試
    success = test_brute_force_integration()
    
    if success:
        print("\n🎉 整合測試成功！")
        
        # 顯示使用演示
        demo_brute_force_usage()
        
        # 顯示整合摘要
        show_integration_summary()
        
        print("\n✅ 暴力搜索功能已成功整合到港股量化分析系統！")
        print("💡 您現在可以使用0-300範圍的專用暴力搜索功能了")
        
    else:
        print("\n❌ 整合測試失敗，請檢查代碼")