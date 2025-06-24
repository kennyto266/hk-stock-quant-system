"""
測試恆生指數和盈富基金數據下載
"""

from data_downloader.stock_data_downloader import HKStockDataDownloader
import datetime as dt
import warnings
import pandas as pd
import time
from random import uniform
import logging
import yfinance as yf
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
import pytz
import json
import numpy as np

# 禁用警告
warnings.filterwarnings('ignore')

# 設置中文顯示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 定義一些常用的港股
STOCK_LIST = {
    '2800': '盈富基金',
    '0005': '匯豐控股',
    '0700': '騰訊控股',
    '9988': '阿里巴巴',
    '0941': '中國移動',
    '0388': '香港交易所',
    '1299': '友邦保險',
    '3690': '美團-W',
    '0001': '長和',
    '0016': '新鴻基地產'
}

def validate_data(data: pd.DataFrame, symbol: str) -> bool:
    """
    驗證下載的數據

    Args:
        data (pd.DataFrame): 股票數據
        symbol (str): 股票代碼

    Returns:
        bool: 數據是否有效
    """
    if data.empty:
        print(f"❌ {symbol} 數據為空")
        return False
        
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"❌ {symbol} 缺少必要列: {', '.join(missing_columns)}")
        return False
        
    if len(data) < 5:  # 至少需要5個交易日的數據
        print(f"❌ {symbol} 數據太少 (只有 {len(data)} 條記錄)")
        return False
        
    return True

def main():
    """主函數"""
    print("開始下載數據...")
    
    # 設置日誌級別
    logging.basicConfig(level=logging.INFO)
    
    # 初始化下載器
    downloader = HKStockDataDownloader(max_retries=5)  # 增加重試次數
    
    # 設置參數
    symbols = ['HSI', '2800.HK']  # 恆生指數和盈富基金
    start_date = dt.datetime(2020, 1, 1).strftime('%Y-%m-%d')  # 從2020年開始
    end_date = dt.datetime.now().strftime('%Y-%m-%d')
    
    # 下載數據
    success_count = 0
    for i, symbol in enumerate(symbols):
        print(f"\n嘗試下載 {symbol}...")
        print(f"時間範圍：{start_date} 至 {end_date}")
        
        # 如果不是第一個股票，添加延遲
        if i > 0:
            delay = uniform(2, 4)  # 2-4秒的隨機延遲
            print(f"等待 {delay:.1f} 秒後下載...")
            time.sleep(delay)
        
        data = downloader.download_stock_data(symbol, start_date, end_date)
        
        if not validate_data(data, symbol):
            continue
            
        print(f"✅ 成功下載 {symbol} 數據！")
        print(f"數據範圍：{str(data.index[0])[:10]} 至 {str(data.index[-1])[:10]}")
        print(f"數據條數：{len(data)}")
        print("\n數據預覽：")
        print(data.head())
        success_count += 1
    
    # 總結
    print(f"\n下載完成！成功率：{success_count}/{len(symbols)}")

def download_with_retry(session, url, params, max_retries=3):
    """帶重試機制的下載函數"""
    for attempt in range(max_retries):
        try:
            # 如果不是第一次嘗試，添加延遲
            if attempt > 0:
                delay = 10 * (2 ** attempt) + uniform(5, 10)
                print(f"\n等待 {delay:.1f} 秒後重試...")
                time.sleep(delay)
            
            response = session.get(url, params=params)
            
            if response.status_code == 200:
                return response
            elif response.status_code == 429:  # Too Many Requests
                print(f"\n遇到速率限制（嘗試 {attempt + 1}/{max_retries}）")
                continue
            else:
                print(f"\n請求失敗：狀態碼 {response.status_code}")
                print("錯誤信息：")
                print(response.text)
                
        except Exception as e:
            print(f"\n請求出錯：{str(e)}")
            
        if attempt < max_retries - 1:
            print("準備重試...")
        
    return None

def test_download():
    print("開始下載盈富基金（2800.HK）數據...")
    
    try:
        # 創建一個session並禁用SSL驗證
        session = requests.Session()
        session.verify = False
        
        # 添加請求頭，模擬瀏覽器行為
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # 獲取當前時間戳
        current_time = int(time.time())
        
        # 構建請求
        query_url = f"https://query1.finance.yahoo.com/v8/finance/chart/2800.HK"
        params = {
            "period1": int(datetime(2024, 1, 1).timestamp()),
            "period2": current_time,
            "interval": "1d",
            "events": "history"
        }
        
        # 使用重試機制發送請求
        response = download_with_retry(session, query_url, params)
        
        if response is not None:
            print("\n成功獲取數據！")
            print("響應內容：")
            print(response.text[:500] + "...")  # 只打印前500個字符
            
    except Exception as e:
        print(f"\n下載出錯：{str(e)}")

def get_stock_data_direct(symbol, start_date, end_date, retries=3):
    """直接從Yahoo Finance API獲取數據"""
    base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"
    
    # 轉換日期為Unix時間戳
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())
    
    # 構建URL
    url = f"{base_url}{symbol}?period1={start_timestamp}&period2={end_timestamp}&interval=1d"
    
    # 設置headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # 檢查HTTP錯誤
            
            data = response.json()
            
            # 檢查是否有數據
            if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                result = data['chart']['result'][0]
                
                # 獲取時間戳和價格數據
                timestamps = result['timestamp']
                quotes = result['indicators']['quote'][0]
                
                # 創建DataFrame
                df = pd.DataFrame({
                    'Open': quotes.get('open', []),
                    'High': quotes.get('high', []),
                    'Low': quotes.get('low', []),
                    'Close': quotes.get('close', []),
                    'Volume': quotes.get('volume', [])
                }, index=pd.to_datetime(timestamps, unit='s'))
                
                return df
                
            return None
            
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                print(f"請求失敗，正在重試... ({attempt + 1}/{retries})")
                time.sleep(2 ** attempt)  # 指數退避
            else:
                print(f"無法獲取數據: {str(e)}")
                return None
        except Exception as e:
            print(f"處理數據時出錯: {str(e)}")
            return None

def analyze_stock(symbol='2800', days=30, show_plot=True):
    """分析指定股票的數據"""
    stock_name = STOCK_LIST.get(symbol, '未知')
    print(f"\n開始分析 {symbol} ({stock_name}) 的數據...")
    
    # 計算香港時區的日期範圍
    hk_tz = pytz.timezone('Asia/Hong_Kong')
    end_date = datetime.now(hk_tz)
    start_date = end_date - timedelta(days=days)
    
    try:
        # 1. 下載數據
        print(f"下載{symbol}的數據（從 {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}）...")
        
        # 確保股票代碼格式正確
        formatted_symbol = f"{symbol}.HK" if not symbol.endswith('.HK') else symbol
        
        # 使用直接API方法獲取數據
        stock_data = get_stock_data_direct(formatted_symbol, start_date, end_date)
        
        if stock_data is None or stock_data.empty:
            print("錯誤：無法獲取數據")
            return None
            
        # 2. 顯示基本數據
        print("\n數據概覽：")
        print(f"數據行數: {len(stock_data)}")
        print("\n前5行數據：")
        print(stock_data.head())
        
        # 3. 計算每日回報率
        daily_returns = stock_data['Close'].pct_change()
        
        if show_plot:
            # 4. 創建圖表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # 4.1 繪製價格走勢
            stock_data['Close'].plot(ax=ax1)
            ax1.set_title(f'{formatted_symbol} ({stock_name}) 收盤價走勢')
            ax1.set_xlabel('日期')
            ax1.set_ylabel('價格 (HKD)')
            ax1.grid(True)
            
            # 4.2 繪製每日回報率
            daily_returns.plot(ax=ax2)
            ax2.set_title(f'{formatted_symbol} ({stock_name}) 每日回報率')
            ax2.set_xlabel('日期')
            ax2.set_ylabel('回報率')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
        
        # 5. 顯示統計信息
        print("\n統計信息：")
        stats = {
            "平均每日回報率": f"{daily_returns.mean():.4%}",
            "回報率標準差": f"{daily_returns.std():.4%}",
            "最大單日漲幅": f"{daily_returns.max():.4%}",
            "最大單日跌幅": f"{daily_returns.min():.4%}",
            "最新收盤價": f"{stock_data['Close'].iloc[-1]:.3f}",
            "交易量": f"{stock_data['Volume'].mean():,.0f}"
        }
        
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # 6. 保存數據到CSV
        csv_filename = f"{formatted_symbol.replace('.', '_')}_{datetime.now().strftime('%Y%m%d')}.csv"
        stock_data.to_csv(csv_filename)
        print(f"\n數據已保存到: {csv_filename}")
        
        return stock_data
        
    except Exception as e:
        print(f"分析過程中出錯: {str(e)}")
        print("請檢查網絡連接和股票代碼是否正確")
        return None

def analyze_multiple_stocks(symbols=None, days=30):
    """分析多隻股票並比較"""
    if symbols is None:
        symbols = list(STOCK_LIST.keys())[:5]  # 默認分析前5隻股票
    
    # 收集所有股票的收盤價數據
    all_data = pd.DataFrame()
    all_returns = pd.DataFrame()
    
    for symbol in symbols:
        stock_data = analyze_stock(symbol, days, show_plot=False)
        if stock_data is not None:
            all_data[f"{symbol} ({STOCK_LIST.get(symbol, '未知')})"] = stock_data['Close']
            all_returns[f"{symbol} ({STOCK_LIST.get(symbol, '未知')})"] = stock_data['Close'].pct_change()
    
    if not all_data.empty:
        # 繪製所有股票的價格走勢（歸一化）
        plt.figure(figsize=(15, 10))
        normalized_data = all_data / all_data.iloc[0] * 100
        normalized_data.plot()
        plt.title('股票價格走勢比較 (歸一化)')
        plt.xlabel('日期')
        plt.ylabel('價格 (基準=100)')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
        # 計算並顯示相關係數
        print("\n股票間的相關係數：")
        correlation = all_returns.corr()
        print(correlation)
        
        # 繪製相關係數熱圖
        plt.figure(figsize=(10, 8))
        plt.imshow(correlation, cmap='RdYlBu', aspect='auto')
        plt.colorbar()
        
        # 修復類型錯誤：將Index轉換為list
        column_labels = correlation.columns.tolist()
        plt.xticks(range(len(column_labels)), column_labels, rotation=45, ha='right')
        plt.yticks(range(len(column_labels)), column_labels)
        
        plt.title('股票回報率相關係數熱圖')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # 分析多隻股票
    print("開始分析多隻港股...")
    analyze_multiple_stocks(['2800', '0005', '0700', '9988', '0388'], 30) 