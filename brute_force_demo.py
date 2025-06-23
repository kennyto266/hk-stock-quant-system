# -*- coding: utf-8 -*-
"""
港股量化分析系統 - 暴力搜索演示
展示如何使用 0-300 範圍的專用暴力搜索功能
"""

import sys
import os

def main():
    """主演示函數"""
    print("港股量化分析系統 - 暴力搜索演示")
    print("=" * 60)
    print("1. RSI暴力搜索 (0-300範圍，步長1)")
    print("2. RSI快速搜索 (0-300範圍，步長5)")  
    print("3. MACD暴力搜索 (0-300範圍，步長1)")
    print("4. MACD快速搜索 (0-300範圍，步長5)")
    print("5. 測試功能整合")
    print("0. 退出")
    print("=" * 60)
    
    choice = input("請選擇 (0-5): ").strip()
    
    if choice == "1":
        print("啟動RSI暴力搜索...")
        demo_rsi_brute_force()
    elif choice == "2":
        print("啟動RSI快速搜索...")
        demo_rsi_quick_search()
    elif choice == "3":
        print("啟動MACD暴力搜索...")
        demo_macd_brute_force()
    elif choice == "4":
        print("啟動MACD快速搜索...")
        demo_macd_quick_search()
    elif choice == "5":
        test_integration()
    elif choice == "0":
        print("退出演示")
        return
    else:
        print("無效選擇")

def demo_rsi_brute_force():
    """演示RSI暴力搜索"""
    try:
        from strategies import run_brute_force_search_0_to_300
        
        print("執行RSI暴力搜索 (0-300範圍，步長1)...")
        print("警告：這將測試大量參數組合，可能需要較長時間！")
        
        confirm = input("確認執行？(y/N): ").strip().lower()
        if confirm != 'y':
            print("取消執行")
            return
            
        result = run_brute_force_search_0_to_300(
            symbol="2800.HK", 
            strategy_type="RSI",
            step_size=1  # 完全暴力搜索
        )
        
        if result:
            print("RSI暴力搜索完成！")
            print(f"最佳參數: {result.get('best_params', {})}")
        else:
            print("RSI暴力搜索失敗")
            
    except ImportError:
        print("無法導入暴力搜索功能，請檢查strategies.py")
    except Exception as e:
        print(f"執行失敗: {e}")

def demo_rsi_quick_search():
    """演示RSI快速搜索"""
    try:
        from strategies import run_brute_force_search_0_to_300
        
        print("執行RSI快速搜索 (0-300範圍，步長5)...")
        
        result = run_brute_force_search_0_to_300(
            symbol="2800.HK", 
            strategy_type="RSI",
            step_size=5  # 快速搜索
        )
        
        if result:
            print("RSI快速搜索完成！")
            print(f"最佳參數: {result.get('best_params', {})}")
        else:
            print("RSI快速搜索失敗")
            
    except Exception as e:
        print(f"執行失敗: {e}")

def demo_macd_brute_force():
    """演示MACD暴力搜索"""
    try:
        from strategies import run_brute_force_search_0_to_300
        
        print("執行MACD暴力搜索 (0-300範圍，步長1)...")
        print("警告：這將測試大量參數組合，可能需要較長時間！")
        
        confirm = input("確認執行？(y/N): ").strip().lower()
        if confirm != 'y':
            print("取消執行")
            return
            
        result = run_brute_force_search_0_to_300(
            symbol="2800.HK", 
            strategy_type="MACD",
            step_size=1
        )
        
        if result:
            print("MACD暴力搜索完成！")
            print(f"最佳參數: {result.get('best_params', {})}")
        else:
            print("MACD暴力搜索失敗")
            
    except Exception as e:
        print(f"執行失敗: {e}")

def demo_macd_quick_search():
    """演示MACD快速搜索"""
    try:
        from strategies import run_brute_force_search_0_to_300
        
        print("執行MACD快速搜索 (0-300範圍，步長5)...")
        
        result = run_brute_force_search_0_to_300(
            symbol="2800.HK", 
            strategy_type="MACD",
            step_size=5
        )
        
        if result:
            print("MACD快速搜索完成！")
            print(f"最佳參數: {result.get('best_params', {})}")
        else:
            print("MACD快速搜索失敗")
            
    except Exception as e:
        print(f"執行失敗: {e}")

def test_integration():
    """測試功能整合"""
    print("測試暴力搜索功能整合...")
    
    try:
        # 測試導入
        from strategies import run_brute_force_search_0_to_300, StrategyOptimizer
        print("成功導入暴力搜索函數")
        
        # 測試StrategyOptimizer方法
        if hasattr(StrategyOptimizer, 'brute_force_search_0_to_300'):
            print("StrategyOptimizer.brute_force_search_0_to_300 方法存在")
        else:
            print("StrategyOptimizer.brute_force_search_0_to_300 方法不存在")
            
        print("功能整合測試完成！")
        
    except ImportError as e:
        print(f"導入失敗: {e}")
    except Exception as e:
        print(f"測試失敗: {e}")

if __name__ == "__main__":
    main()