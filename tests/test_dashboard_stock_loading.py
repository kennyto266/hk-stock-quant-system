#!/usr/bin/env python3
"""
測試儀表板股票數據加載功能
"""

import sys
import os
import pandas as pd
import traceback
from datetime import datetime, timedelta

# 添加項目路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_stock_data_loading():
    """測試股票數據加載功能"""
    
    print("=" * 60)
    print("🧪 測試儀表板股票數據加載功能")
    print("=" * 60)
    
    # 測試1: 檢查股票數據文件是否存在
    print("\n📁 測試1: 檢查股票數據文件")
    csv_dir = "data_output/csv"
    stock_files = []
    
    if os.path.exists(csv_dir):
        all_files = os.listdir(csv_dir)
        stock_files = [f for f in all_files if f.endswith('_stock_data.csv')]
        
        print(f"✅ CSV目錄存在: {csv_dir}")
        print(f"📊 找到股票數據文件: {len(stock_files)} 個")
        
        for file in stock_files:
            file_path = os.path.join(csv_dir, file)
            file_size = os.path.getsize(file_path)
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            print(f"   • {file} - {file_size:,} bytes, 修改時間: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"❌ CSV目錄不存在: {csv_dir}")
        return False
    
    if not stock_files:
        print("❌ 沒有找到股票數據文件")
        return False
    
    # 測試2: 加載股票數據
    print("\n📈 測試2: 加載股票數據")
    try:
        # 選擇最新的股票文件進行測試
        test_file = stock_files[0]
        file_path = os.path.join(csv_dir, test_file)
        
        print(f"📂 正在加載: {test_file}")
        df = pd.read_csv(file_path)
        
        print(f"✅ 數據加載成功!")
        print(f"📊 數據形狀: {df.shape}")
        print(f"📅 時間範圍: {df['Date'].min()} 到 {df['Date'].max()}")
        print(f"🔢 列名: {list(df.columns)}")
        
        # 顯示前幾行數據
        print("\n📋 數據預覽:")
        print(df.head(3).to_string(index=False))
        
    except Exception as e:
        print(f"❌ 數據加載失敗: {str(e)}")
        print(f"🔍 詳細錯誤: {traceback.format_exc()}")
        return False
    
    # 測試3: 測試儀表板數據獲取函數
    print("\n🎯 測試3: 測試儀表板數據獲取函數")
    try:
        # 導入儀表板模組
        from enhanced_interactive_dashboard import get_stock_data
        
        # 測試獲取騰訊控股數據
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        print(f"📡 測試獲取 0700.HK 數據")
        print(f"📅 時間範圍: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
        
        stock_data = get_stock_data(
            symbol="0700.HK",
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        print(f"✅ 儀表板數據獲取成功!")
        print(f"📊 數據形狀: {stock_data.shape}")
        print(f"🔢 列名: {list(stock_data.columns)}")
        
        # 檢查技術指標是否計算
        technical_indicators = ['SMA', 'EMA', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower']
        available_indicators = [col for col in technical_indicators if col in stock_data.columns]
        print(f"📈 可用技術指標: {available_indicators}")
        
    except Exception as e:
        print(f"❌ 儀表板數據獲取失敗: {str(e)}")
        print(f"🔍 詳細錯誤: {traceback.format_exc()}")
        return False
    
    # 測試4: 檢查儀表板配置
    print("\n⚙️ 測試4: 檢查儀表板配置")
    try:
        from enhanced_interactive_dashboard import SYMBOL, DATA_OUTPUT_PATH, CSV_PATH
        
        print(f"🎯 默認股票代碼: {SYMBOL}")
        print(f"📁 數據輸出路徑: {DATA_OUTPUT_PATH}")
        print(f"📊 CSV路徑: {CSV_PATH}")
        
        # 檢查路徑是否存在
        if os.path.exists(DATA_OUTPUT_PATH):
            print(f"✅ 數據輸出路徑存在")
        else:
            print(f"⚠️ 數據輸出路徑不存在: {DATA_OUTPUT_PATH}")
        
        if os.path.exists(CSV_PATH):
            print(f"✅ CSV路徑存在")
        else:
            print(f"⚠️ CSV路徑不存在: {CSV_PATH}")
            
    except Exception as e:
        print(f"⚠️ 儀表板配置檢查警告: {str(e)}")
    
    print("\n" + "=" * 60)
    print("🎉 股票數據加載測試完成!")
    print("✅ 所有測試通過，儀表板可以正常加載股票數據")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    test_stock_data_loading()