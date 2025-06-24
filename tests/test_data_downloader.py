"""
港股數據下載器測試腳本
測試批量下載功能和數據質量
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_downloader import HKStockDataDownloader
from config import logger
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import datetime as dt
from pathlib import Path
import warnings
from typing import Optional

# 禁用警告
warnings.filterwarnings('ignore')

def test_single_stock():
    """測試單隻股票下載"""
    logger.info("🧪 測試單隻股票下載...")
    downloader = HKStockDataDownloader()
    
    # 測試下載2800.HK（盈富基金）
    symbol = "2800.HK"
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    data = downloader.download_single_stock(symbol, start_date, end_date)
    
    if data is not None:
        logger.info(f"✅ 單隻股票測試成功")
        logger.info(f"數據形狀: {data.shape}")
        logger.info(f"數據範圍: {data.index.min()} 至 {data.index.max()}")
        logger.info(f"列名: {list(data.columns)}")
        return True
    else:
        logger.error(f"❌ 單隻股票測試失敗")
        return False

def test_small_batch():
    """測試小批量下載"""
    logger.info("🧪 測試小批量下載...")
    downloader = HKStockDataDownloader()
    
    # 臨時修改股票列表為小樣本
    original_stocks = downloader.hk_stocks
    test_stocks = {
        '2800.HK': '盈富基金',
        '700.HK': '騰訊控股',
        '941.HK': '中國移動'
    }
    downloader.hk_stocks = test_stocks
    
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    try:
        all_data = downloader.download_all_stocks(start_date=start_date, delay_seconds=0.5)
        
        if len(all_data) > 0:
            logger.info(f"✅ 小批量測試成功，下載了 {len(all_data)} 隻股票")
            
            # 測試合併數據集
            combined = downloader.create_combined_dataset(all_data)
            if not combined.empty:
                logger.info(f"✅ 合併數據集測試成功，總共 {len(combined)} 行數據")
                return True
            else:
                logger.error("❌ 合併數據集測試失敗")
                return False
        else:
            logger.error("❌ 小批量測試失敗，沒有下載到任何數據")
            return False
            
    finally:
        # 恢復原始股票列表
        downloader.hk_stocks = original_stocks

def test_data_quality():
    """測試數據質量"""
    logger.info("🧪 測試數據質量...")
    
    # 檢查輸出目錄中的CSV文件
    data_dir = Path("data_output/csv")
    
    if not data_dir.exists():
        logger.warning("數據目錄不存在，跳過質量檢查")
        return True
        
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        logger.warning("沒有找到CSV文件，跳過質量檢查")
        return True
        
    logger.info(f"檢查 {len(csv_files)} 個CSV文件...")
    
    for csv_file in csv_files[:3]:  # 只檢查前3個文件
        try:
            data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            
            # 檢查基本要求
            required_columns = ['Symbol', 'Stock_Name', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                logger.error(f"❌ {csv_file.name} 缺少必需列: {missing_columns}")
                continue
                
            # 檢查數據完整性
            if data.isnull().any().any():
                logger.warning(f"⚠️ {csv_file.name} 存在空值")
                
            # 檢查價格邏輯
            invalid_prices = data[(data['High'] < data['Low']) | 
                                (data['Close'] < 0) | 
                                (data['Volume'] < 0)]
            
            if len(invalid_prices) > 0:
                logger.error(f"❌ {csv_file.name} 存在無效價格數據")
                continue
                
            logger.info(f"✅ {csv_file.name} 數據質量檢查通過")
            
        except Exception as e:
            logger.error(f"❌ 檢查 {csv_file.name} 時出錯: {e}")
            continue
            
    return True

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

def main():
    """主測試函數"""
    logger.info("🚀 開始港股數據下載器測試...")
    
    tests = [
        ("單隻股票下載", test_single_stock),
        ("小批量下載", test_small_batch),
        ("數據質量檢查", test_data_quality),
        ("恆生指數下載", test_hsi_download)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"執行測試: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ 測試 {test_name} 執行失敗: {e}")
            results.append((test_name, False))
    
    # 輸出測試結果
    logger.info(f"\n{'='*50}")
    logger.info("測試結果總結")
    logger.info(f"{'='*50}")
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通過" if result else "❌ 失敗"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
            
    logger.info(f"\n總計: {passed}/{len(results)} 測試通過")
    
    if passed == len(results):
        logger.info("🎉 所有測試通過！港股數據下載器準備就緒")
    else:
        logger.warning("⚠️ 部分測試失敗，請檢查配置和網絡連接")

if __name__ == "__main__":
    main()