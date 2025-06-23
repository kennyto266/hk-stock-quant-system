#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股量化分析系統 - 主運行文件
Created on 2025-06-22
Author: AI Assistant
"""

import sys
import os
import time
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 添加當前目錄到Python路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_handler import DataFetcher
    from strategies import StrategyOptimizer, run_strategy_optimization, run_comprehensive_optimization
    # 可選導入，如果不存在就跳過相關功能
    # 註釋掉 HTML 儀表板功能，只使用 Dash
    # try:
    #     from visualization import create_interactive_dashboard
    #     VISUALIZATION_AVAILABLE = True
    # except ImportError:
    VISUALIZATION_AVAILABLE = False
        
    # 風險管理模組為可選功能
    try:
        from risk_management import RiskManager
        RISK_MANAGEMENT_AVAILABLE = True
    except ImportError:
        RISK_MANAGEMENT_AVAILABLE = False
        RiskManager = None
        
except ImportError as e:
    print(f"❌ 導入錯誤: {e}")
    print("請確保所有必要的模組文件都存在")
    sys.exit(1)

def print_banner():
    """顯示系統啟動橫幅"""
    print("=" * 80)
    print("🚀 港股量化分析系統 v2.0")
    print("📊 股票代碼: 2800.HK (追蹤基金)")
    print("💻 系統作者: AI Assistant")
    print("📅 創建日期: 2025-06-22")
    print("=" * 80)

def get_system_info():
    """獲取系統信息"""
    try:
        import psutil
        import multiprocessing as mp
        
        cpu_count = mp.cpu_count()
        memory = psutil.virtual_memory()
        
        print("💻 系統配置:")
        print(f"   CPU核心數: {cpu_count}")
        print(f"   可用內存: {memory.total / (1024**3):.1f} GB")
        print(f"   內存使用率: {memory.percent}%")
        print("-" * 50)
        
    except ImportError:
        print("⚠️  psutil未安裝，無法顯示系統信息")

def main_menu():
    """主選單"""
    print("\n🎯 請選擇分析模式:")
    print("1. 📈 快速策略優化 (基本模式)")
    print("2. 🚀 智能RSI策略優化 (整合版 - 快速+全面+並行)")
    print("3. 📊 生成可視化儀表板")
    print("4. 🎲 風險管理分析")
    print("5. 🤖 完全自動化運行 (推薦)")
    print("0. 🚪 退出系統")
    print("-" * 50)

def run_basic_analysis():
    """基本策略優化分析"""
    print("\n🎯 啟動基本策略優化...")
    try:
        result = run_strategy_optimization("2800.HK", "2020-01-01")
        if result:
            print("✅ 基本分析完成！")
        else:
            print("❌ 基本分析失敗")
    except Exception as e:
        print(f"❌ 基本分析錯誤: {e}")

def run_comprehensive_analysis():
    """全面策略掃描分析"""
    print("\n🔍 啟動全面策略掃描...")
    
    print("請選擇掃描模式:")
    print("1. 快速掃描 (步長20)")
    print("2. 全面掃描 (步長5)")
    
    try:
        choice = input("請輸入選擇 (1-2): ").strip()
        
        if choice == "1":
            mode = "quick_scan"
            print("🚀 啟動快速掃描模式...")
        elif choice == "2":
            mode = "comprehensive"
            print("🔍 啟動全面掃描模式...")
        else:
            print("❌ 無效選擇，使用快速掃描")
            mode = "quick_scan"
            
        result = run_comprehensive_optimization("2800.HK", "2020-01-01", mode)
        if result:
            print("✅ 全面分析完成！")
        else:
            print("❌ 全面分析失敗")
            
    except Exception as e:
        print(f"❌ 全面分析錯誤: {e}")

def run_ultra_parallel_analysis():
    """終極並行優化分析"""
    print("\n🚀 啟動終極並行優化...")
    
    try:
        from strategies import run_ultra_parallel_optimization
        
        print("⚠️  警告: 此模式將使用大量系統資源")
        confirm = input("確定要繼續嗎? (y/N): ").strip().lower()
        
        if confirm == 'y':
            print("🔥 啟動終極並行模式...")
            result = run_ultra_parallel_optimization("2800.HK", "2020-01-01", 300)
            if result:
                print("✅ 並行優化完成！")
            else:
                print("❌ 並行優化失敗")
        else:
            print("❌ 用戶取消操作")
            
    except Exception as e:
        print(f"❌ 並行優化錯誤: {e}")

def generate_dashboard():
    """啟動 Dash 網頁應用"""
    print("\n🌐 正在啟動 Dash 網頁應用...")
    
    try:
        import subprocess
        import sys
        import os
        
        if os.path.exists("enhanced_interactive_dashboard.py"):
            print("📱 正在啟動 Dash 應用...")
            subprocess.Popen([sys.executable, "enhanced_interactive_dashboard.py"], 
                           creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0)
            print("🎉 Dash 應用已啟動！請在瀏覽器中訪問 http://127.0.0.1:8050")
            print("📊 所有分析結果都可在網頁儀表板中查看")
        else:
            print("❌ enhanced_interactive_dashboard.py 不存在")
            
    except Exception as e:
        print(f"❌ Dash 應用啟動錯誤: {e}")

def run_risk_analysis():
    """風險管理分析"""
    print("\n🎲 啟動風險管理分析...")
    
    try:
        if not RISK_MANAGEMENT_AVAILABLE:
            print("❌ 風險管理模組未安裝，無法進行風險分析")
            return
            
        # 獲取數據
        data_handler = DataFetcher()
        stock_data = data_handler.get_yahoo_finance_data("2800.HK", "2020-01-01", "2025-12-31")
        
        if stock_data is not None and not stock_data.empty:
            if RiskManager is not None:
                risk_manager = RiskManager()
                
                # 基本風險指標
                volatility = risk_manager.calculate_volatility(stock_data['Close'])
                var_95 = risk_manager.calculate_var(stock_data['Close'], confidence_level=0.95)
                
                print(f"📊 風險分析結果:")
                print(f"   年化波動率: {volatility:.2%}")
                print(f"   95% VaR: {var_95:.2%}")
                
                print("✅ 風險分析完成！")
            else:
                print("❌ RiskManager 類別無法導入")
        else:
            print("❌ 無法獲取股票數據")
            
    except Exception as e:
        print(f"❌ 風險分析錯誤: {e}")

def run_auto_mode():
    """完全自動化運行模式"""
    print("\n🤖 啟動完全自動化模式...")
    print("🚀 將自動執行所有分析功能...")
    
    try:
        # 導入主程序
        from strategies import main as strategies_main
        
        print("🎯 調用 strategies.py 主程序...")
        strategies_main()
        print("✅ 自動化分析完成！")
        
    except Exception as e:
        print(f"❌ 自動化分析錯誤: {e}")

def main_interactive():
    """互動式主程序"""
    print_banner()
    get_system_info()
    
    while True:
        main_menu()
        
        try:
            choice = input("請輸入您的選擇 (0-5): ").strip()
            
            if choice == '0':
                print("👋 感謝使用港股量化分析系統！")
                break
            elif choice == '1':
                run_basic_analysis()
            elif choice == '2':
                run_integrated_rsi_optimization()
            elif choice == '3':
                generate_dashboard()
            elif choice == '4':
                run_risk_analysis()
            elif choice == '5':
                run_auto_mode()
            else:
                print("❌ 無效選擇，請重新輸入")
                
        except KeyboardInterrupt:
            print("\n\n👋 程序被用戶中斷，再見！")
            break
        except Exception as e:
            print(f"❌ 程序錯誤: {e}")
            print("請重新選擇...")

def main_auto():
    """完全自動化運行模式 - 原main函數"""
    print_banner()
    get_system_info()
    
    print("🤖 啟動完全自動化模式...")
    print("🚀 將自動執行所有分析功能，無需用戶輸入...")
    print("="*80)
    
    try:
        # 導入主程序並直接執行
        from strategies import main as strategies_main
        
        print("🎯 調用 strategies.py 主程序...")
        strategies_main()
        print("✅ 自動化分析完成！")
        
        # 自動生成並啟動儀表板
        print("\n" + "="*80)
        print("📊 正在生成互動式儀表板...")
        generate_and_launch_dashboard()
        
    except Exception as e:
        print(f"❌ 自動化分析錯誤: {e}")

def generate_and_launch_dashboard():
    """啟動 Dash 網頁應用 (不再生成HTML)"""
    try:
        print("🌐 正在啟動 Dash 網頁應用...")
        
        import subprocess
        import sys
        import os
        
        # 檢查是否有enhanced_interactive_dashboard.py
        if os.path.exists("enhanced_interactive_dashboard.py"):
            print("📱 正在啟動 Dash 應用...")
            # 在後台啟動Dash應用
            subprocess.Popen([sys.executable, "enhanced_interactive_dashboard.py"], 
                           creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0)
            print("🎉 Dash 應用已啟動！請在瀏覽器中訪問 http://127.0.0.1:8050")
            print("📊 所有分析結果都可在網頁儀表板中查看")
        else:
            print("❌ enhanced_interactive_dashboard.py 不存在")
            
    except Exception as e:
        print(f"❌ Dash 應用啟動錯誤: {e}")

def run_integrated_rsi_optimization():
    """運行整合的RSI策略優化分析"""
    print("\n🚀 啟動智能RSI策略優化...")
    print("📊 此功能整合了快速掃描、全面分析和並行優化")
    
    try:
        from strategies import run_integrated_rsi_optimization
        
        print("⚡ 此模式將按階段優化RSI策略:")
        print("   📈 階段1: 快速掃描 (大步長)")
        print("   🔍 階段2: 精細搜索 (小步長)")
        print("   ⚡ 階段3: 結果驗證")
        
        confirm = input("\n確定要開始嗎? (y/N): ").strip().lower()
        
        if confirm == 'y':
            print("🔥 啟動智能RSI優化...")
            result = run_integrated_rsi_optimization("2800.HK", "2020-01-01")
            if result:
                print("✅ 智能RSI優化完成！")
            else:
                print("❌ 智能RSI優化失敗")
        else:
            print("❌ 用戶取消操作")
            
    except Exception as e:
        print(f"❌ 智能RSI優化錯誤: {e}")

if __name__ == "__main__":
    # 直接運行智能RSI策略優化，不需要用戶選擇
    print_banner()
    get_system_info()
    
    print("🚀 自動啟動智能RSI策略優化...")
    print("📊 整合快速掃描、全面分析和並行優化功能")
    print("="*80)
    
    try:
        from strategies import run_integrated_rsi_optimization as rsi_optimizer
        
        print("🔥 開始智能RSI優化...")
        result = rsi_optimizer("2800.HK", "2020-01-01")
        
        if result:
            print("\n" + "="*80)
            print("✅ 智能RSI優化完成！")
            print("🌐 正在啟動 Dash 網頁應用...")
            
            # 直接啟動Dash應用，不生成HTML
            try:
                import subprocess
                import sys
                import os
                
                if os.path.exists("enhanced_interactive_dashboard.py"):
                    print("📱 正在啟動 Dash 應用...")
                    subprocess.Popen([sys.executable, "enhanced_interactive_dashboard.py"], 
                                   creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0)
                    print("🎉 Dash 應用已啟動！請在瀏覽器中訪問 http://127.0.0.1:8050")
                    print("📊 所有分析結果都可在網頁儀表板中查看")
                else:
                    print("⚠️ enhanced_interactive_dashboard.py 不存在")
                    
            except Exception as dash_error:
                print(f"⚠️ Dash應用啟動失敗: {dash_error}")
                
        else:
            print("❌ 智能RSI優化失敗")
            
    except Exception as e:
        print(f"❌ 智能RSI優化錯誤: {e}") 