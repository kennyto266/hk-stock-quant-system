#!/usr/bin/env python3
"""
港股量化分析系統 - 新技術指標策略 Dashboard
快速顯示最新的多技術指標策略優化結果
"""

import os
import subprocess
import sys
import time
import webbrowser
from datetime import datetime

def check_csv_files():
    """檢查是否有最新的策略 CSV 文件"""
    csv_dir = "data_output/csv"
    if not os.path.exists(csv_dir):
        print("❌ CSV 目錄不存在，請先運行策略優化")
        return False
    
    # 檢查新技術指標的 CSV 文件
    new_strategy_files = [
        "integrated_macd_",
        "integrated_bollinger_", 
        "integrated_kdj_",
        "integrated_stochastic_",
        "integrated_cci_",
        "integrated_williams_r_",
        "multi_strategy_comprehensive_"
    ]
    
    found_files = []
    for pattern in new_strategy_files:
        files = [f for f in os.listdir(csv_dir) if f.startswith(pattern)]
        if files:
            latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(csv_dir, x)))
            found_files.append(latest_file)
    
    if found_files:
        print(f"✅ 找到 {len(found_files)} 個新技術指標策略文件:")
        for file in found_files:
            print(f"   📁 {file}")
        return True
    else:
        print("❌ 未找到新技術指標策略文件，請先運行 'python run_all_strategies.py'")
        return False

def launch_dashboard():
    """啟動 Dashboard"""
    try:
        if not os.path.exists("enhanced_interactive_dashboard.py"):
            print("❌ enhanced_interactive_dashboard.py 不存在")
            return False
        
        print("🚀 正在啟動 Dashboard...")
        
        # 啟動 Dashboard（後台運行）
        process = subprocess.Popen(
            [sys.executable, "enhanced_interactive_dashboard.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
        )
        
        # 等待 Dashboard 啟動
        print("⏳ 等待 Dashboard 啟動...")
        time.sleep(5)
        
        # 檢查 Dashboard 是否成功啟動
        dashboard_url = "http://localhost:8050"
        
        print(f"🌐 Dashboard URL: {dashboard_url}")
        print("🔍 正在檢查 Dashboard 狀態...")
        
        # 自動打開瀏覽器
        try:
            webbrowser.open(dashboard_url)
            print("✅ 已在瀏覽器中打開 Dashboard")
        except Exception as e:
            print(f"⚠️ 無法自動打開瀏覽器: {e}")
            print(f"🌐 請手動在瀏覽器中訪問: {dashboard_url}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard 啟動失敗: {e}")
        return False

def main():
    """主函數"""
    print("=" * 60)
    print("🎯 港股量化分析系統 - 新技術指標策略 Dashboard")
    print("=" * 60)
    print(f"⏰ 啟動時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 檢查 CSV 文件
    print("🔍 檢查策略結果文件...")
    if not check_csv_files():
        print("\n💡 建議運行步驟:")
        print("   1. python run_all_strategies.py  # 生成策略結果")
        print("   2. python run_dashboard_with_new_strategies.py  # 啟動 Dashboard")
        return
    
    print()
    
    # 啟動 Dashboard
    if launch_dashboard():
        print()
        print("🎉 新技術指標策略 Dashboard 啟動成功!")
        print()
        print("📊 Dashboard 功能:")
        print("   • RSI 策略優化結果")
        print("   • MACD 策略優化結果")
        print("   • 布林帶策略優化結果")
        print("   • KDJ 策略優化結果")
        print("   • Stochastic 策略優化結果")
        print("   • CCI 策略優化結果")
        print("   • 威廉指標%R 策略優化結果")
        print("   • 多策略績效對比")
        print("   • 互動式權益曲線圖表")
        print()
        print("🌐 Dashboard URL: http://localhost:8050")
        print("📝 按 Ctrl+C 可停止 Dashboard")
        print()
        print("=" * 60)
        
        # 保持腳本運行
        try:
            print("⏳ Dashboard 正在運行中... (按 Ctrl+C 停止)")
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            print("\n🛑 用戶請求停止 Dashboard")
            print("✅ Dashboard 已停止")
    
    else:
        print("❌ Dashboard 啟動失敗")

if __name__ == "__main__":
    main() 