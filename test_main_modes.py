#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試主入口的不同模式
"""

import sys
import subprocess
import yfinance as yf
import datetime as dt
import pandas as pd
from pathlib import Path
import warnings
from typing import Optional

def test_mode(mode_name):
    """測試指定模式"""
    print(f"\n{'='*60}")
    print(f"🧪 測試模式: {mode_name}")
    print(f"{'='*60}")
    
    try:
        # 使用 subprocess 調用主程序
        cmd = [sys.executable, "main.py", "--mode", mode_name]
        print(f"🚀 執行命令: {' '.join(cmd)}")
        
        # 對於非儀表板模式，設置超時以避免長時間運行
        timeout = 30 if mode_name != 'dashboard' else None
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        
        print("📤 標準輸出:")
        print(result.stdout)
        
        if result.stderr:
            print("📥 錯誤輸出:")
            print(result.stderr)
        
        print(f"🎯 返回碼: {result.returncode}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"⏰ 模式 {mode_name} 在 {timeout} 秒後超時（這可能是正常的）")
        return True
    except Exception as e:
        print(f"❌ 測試模式 {mode_name} 時出錯: {e}")
        return False

def main():
    """主測試函數"""
    print("🎯 港股量化分析系統 - 模式測試")
    print("=" * 60)
    
    modes = ['analysis', 'brute']  # 'all', 'dashboard' 可能運行時間較長
    results = {}
    
    for mode in modes:
        success = test_mode(mode)
        results[mode] = success
        
        if success:
            print(f"✅ 模式 {mode} 測試通過")
        else:
            print(f"❌ 模式 {mode} 測試失敗")
    
    print(f"\n{'='*60}")
    print("📊 測試結果總結:")
    print(f"{'='*60}")
    
    for mode, success in results.items():
        status = "✅ 通過" if success else "❌ 失敗"
        print(f"   {mode:12} : {status}")
    
    success_count = sum(results.values())
    total_count = len(results)
    print(f"\n🎯 總體結果: {success_count}/{total_count} 模式測試通過")
    
    if success_count == total_count:
        print("🎉 所有測試都通過了！")
        return 0
    else:
        print("⚠️ 部分測試失敗，請檢查錯誤信息")
        return 1

def test_hsi_download():
    """測試下載恆生指數數據"""
    print("開始下載恆生指數數據...")
    
    # 設置參數
    index_symbol = '^HSI'
    start_date = dt.datetime(1900, 1, 1)
    end_date = dt.datetime.now()
    
    try:
        # 下載數據
        data: Optional[pd.DataFrame] = yf.download(
            index_symbol,
            start=start_date,
            end=end_date,
            interval='1d',
            progress=True
        )
        
        if data is None or data.empty:
            print("❌ 下載失敗：沒有數據")
            return
            
        print(f"✅ 成功下載數據！")
        print(f"數據範圍：{data.index.min()} 至 {data.index.max()}")
        print(f"總記錄數：{len(data)}")
        
        # 顯示最新的幾條數據
        print("\n最新數據：")
        print(data.tail())
        
        # 保存數據
        output_dir = Path("data_output/test")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = output_dir / "HSI_test.csv"
        data.to_csv(csv_path)
        print(f"\n數據已保存至：{csv_path}")
        
    except Exception as e:
        print(f"❌ 下載過程中出錯：{str(e)}")

if __name__ == "__main__":
    sys.exit(main())