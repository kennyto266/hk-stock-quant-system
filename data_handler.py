"""
港股量化分析系統 - 數據處理模組
包含數據獲取、技術指標計算、數據清理等功能
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from typing import Optional, Dict, List, Tuple, Union, Any, cast
import warnings
from config import logger
import time
import logging
import ssl
import urllib3
import os

# 忽略警告
warnings.filterwarnings('ignore')
# 忽略SSL驗證警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def format_date_safely(date_value: Any) -> Optional[str]:
    """安全地格式化日期值"""
    if pd.isna(date_value):
        return None
    try:
        if isinstance(date_value, (datetime, pd.Timestamp)):
            return date_value.strftime('%Y-%m-%d')
        return None
    except:
        return None

class DataFetcher:
    """數據獲取器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        
    def get_stock_file_path(self, symbol: str) -> str:
        """獲取股票數據文件路徑"""
        return os.path.join(self.data_dir, f"{symbol}.csv")
        
    def download_stock_data(self, symbol: str, start_date: str, end_date: str) -> None:
        """下載股票數據並保存為CSV"""
        try:
            # 使用akshare下載港股數據
            data = ak.stock_hk_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date)
            
            if data.empty:
                raise ValueError(f"無法獲取 {symbol} 的數據")
                
            # 重命名列
            data = data.rename(columns={
                '日期': 'date',
                '開盤': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盤': 'close',
                '成交量': 'volume'
            })
            
            # 將日期列轉換為datetime
            data['date'] = pd.to_datetime(data['date'])
            
            # 保存數據
            file_path = self.get_stock_file_path(symbol)
            data.to_csv(file_path, index=False)
            
            self.logger.info(f"✅ 成功下載並保存 {symbol} 數據: {len(data)} 天")
            
        except Exception as e:
            self.logger.error(f"❌ 下載 {symbol} 數據失敗: {str(e)}")
            raise
            
    def read_stock_data(self, symbol: str) -> pd.DataFrame:
        """從CSV讀取股票數據"""
        try:
            file_path = self.get_stock_file_path(symbol)
            data = pd.read_csv(file_path)
            
            # 將日期列轉換為datetime
            data['date'] = pd.to_datetime(data['date'])
            
            self.logger.info(f"✅ 從CSV讀取 {symbol} 數據: {len(data)} 天")
            return data
            
        except Exception as e:
            self.logger.error(f"❌ 讀取 {symbol} 數據失敗: {str(e)}")
            raise

    @staticmethod
    def get_northbound_data(symbol: str = "2800.HK", 
                           start_date: Optional[str] = None, 
                           end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        獲取北水南下數據
        
        Args:
            symbol: 股票代碼
            start_date: 開始日期
            end_date: 結束日期
            
        Returns:
            北水數據DataFrame
        """
        try:
            logger.info(f"🌊 正在獲取 {symbol} 的北水數據...")
            
            # 設置默認日期範圍
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if not start_date:
                start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
            
            # 這裡可以實現北水數據的API調用
            # 由於實際API可能需要授權，這裡提供模擬數據結構
            
            # 創建模擬北水數據
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            northbound_data = pd.DataFrame({
                'Date': date_range,
                'Northbound_Net_Buy': np.random.normal(100000, 50000, len(date_range)),
                'Northbound_Buy': np.random.normal(500000, 100000, len(date_range)),
                'Northbound_Sell': np.random.normal(400000, 80000, len(date_range)),
                'Northbound_Holdings': np.random.normal(10000000, 1000000, len(date_range))
            })
            
            northbound_data.set_index('Date', inplace=True)
            
            # 添加 north_flow 列（使用淨買入金額除以持倉金額的比例）
            northbound_data['north_flow'] = northbound_data['Northbound_Net_Buy'] / northbound_data['Northbound_Holdings']
            
            logger.info(f"✅ 成功獲取 {len(northbound_data)} 天的北水數據")
            return northbound_data
            
        except Exception as e:
            logger.error(f"❌ 獲取北水數據失敗: {e}")
            return None
    
    @staticmethod
    def _clean_data(data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        清理數據
        
        Args:
            data: 原始數據
            
        Returns:
            清理後的數據
        """
        try:
            # 確保輸入是 DataFrame
            if not isinstance(data, pd.DataFrame):
                if isinstance(data, pd.Series):
                    data = data.to_frame()
                else:
                    return pd.DataFrame()  # 返回空的 DataFrame
            
            # 移除空值
            cleaned_data = data.dropna()
            
            # 確保數據類型正確
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in cleaned_data.columns:
                    cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
            
            # 移除異常值（價格為0或負數）
            if 'Close' in cleaned_data.columns:
                cleaned_data = cleaned_data[cleaned_data['Close'] > 0]
            
            # 排序
            cleaned_data = cleaned_data.sort_index()
            
            return cast(pd.DataFrame, cleaned_data)
            
        except Exception as e:
            logger.error(f"數據清理失敗: {e}")
            return pd.DataFrame() if not isinstance(data, pd.DataFrame) else data
    
    @staticmethod
    def _merge_northbound_data(stock_data: pd.DataFrame, 
                              northbound_data: pd.DataFrame) -> pd.DataFrame:
        """
        合併北水數據
        
        Args:
            stock_data: 股票數據
            northbound_data: 北水數據
            
        Returns:
            合併後的數據
        """
        try:
            # 確保索引格式一致
            stock_data.index = pd.to_datetime(stock_data.index)
            northbound_data.index = pd.to_datetime(northbound_data.index)
            
            # 合併數據
            merged_data = stock_data.join(northbound_data, how='left')
            
            # 填充缺失值
            northbound_columns = ['Northbound_Net_Buy', 'Northbound_Buy', 
                                'Northbound_Sell', 'Northbound_Holdings']
            for col in northbound_columns:
                if col in merged_data.columns:
                    merged_data[col] = merged_data[col].fillna(0)
            
            return merged_data
            
        except Exception as e:
            logger.error(f"數據合併失敗: {e}")
            return stock_data

    def get_yahoo_finance_data(self, symbol: str, start_date: str, end_date: str, retries: int = 3) -> pd.DataFrame:
        """
        直接從Yahoo Finance API獲取數據，與 main.py 的 get_stock_data_direct 一致
        Args:
            symbol: 股票代碼（如 2800.HK）
            start_date: 開始日期（YYYY-MM-DD）
            end_date: 結束日期（YYYY-MM-DD）
            retries: 重試次數
        Returns:
            pd.DataFrame 或 None
        """
        import requests
        import time
        import pandas as pd
        from datetime import datetime
        import pytz
        hk_tz = pytz.timezone('Asia/Hong_Kong')
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=hk_tz)
            end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=hk_tz)
        except Exception as e:
            self.logger.error(f"❌ 日期格式錯誤: {e}")
            return None
        base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"
        start_timestamp = int(start_dt.timestamp())
        end_timestamp = int(end_dt.timestamp())
        url = f"{base_url}{symbol}?period1={start_timestamp}&period2={end_timestamp}&interval=1d"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        for attempt in range(retries):
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                    result = data['chart']['result'][0]
                    timestamps = result['timestamp']
                    quotes = result['indicators']['quote'][0]
                    df = pd.DataFrame({
                        'open': quotes.get('open', []),
                        'high': quotes.get('high', []),
                        'low': quotes.get('low', []),
                        'close': quotes.get('close', []),
                        'volume': quotes.get('volume', [])
                    }, index=pd.to_datetime(timestamps, unit='s'))
                    return df
                return None
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    self.logger.warning(f"請求失敗，正在重試... ({attempt + 1}/{retries})")
                    time.sleep(2 ** attempt)
                else:
                    self.logger.error(f"無法獲取數據: {str(e)}")
                    return None
            except Exception as e:
                self.logger.error(f"處理數據時出錯: {str(e)}")
                return None
    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        通用數據獲取接口，供 UnifiedStrategyOptimizer 使用。
        內部調用 get_yahoo_finance_data，並自動標準化欄位。
        """
        df = self.get_yahoo_finance_data(symbol, start_date, end_date)
        if df is None or df.empty:
            self.logger.error(f"❌ fetch_data: 無法獲取 {symbol} 數據，日期: {start_date}~{end_date}")
            return None
        # 標準化欄位名稱
        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume',
            'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'
        })
        # 將 index 轉為 date 欄位
        if 'date' not in df.columns:
            df = df.reset_index().rename(columns={'index': 'date'})
        # 確保所有欄位小寫
        df.columns = [str(col).lower() for col in df.columns]
        # 日誌：輸出欄位、shape、前幾行
        self.logger.info(f"fetch_data: {symbol} 數據 shape={df.shape}, columns={list(df.columns)}")
        self.logger.info(f"fetch_data: {symbol} head={df.head(3)}")
        # 只檢查 close 欄位空值，允許其他欄位有小量空值
        if df['close'].isnull().sum() > 0:
            self.logger.warning(f"fetch_data: {symbol} 有 {df['close'].isnull().sum()} 個 close 空值，將移除")
            df = df.dropna(subset=['close'])
        if df.empty:
            self.logger.error(f"❌ fetch_data: {symbol} 處理後數據為空，放棄")
            return None
        return df


class TechnicalIndicators:
    """技術指標計算器"""
    
    @staticmethod
    def calculate_rsi(close_prices: Union[pd.Series, pd.DataFrame], period: int = 14) -> pd.Series:
        """
        計算RSI指標
        
        Args:
            close_prices: 收盤價序列或DataFrame
            period: 計算週期
            
        Returns:
            RSI值序列
        """
        try:
            if isinstance(close_prices, pd.DataFrame):
                close_prices = close_prices['Close']
            
            if len(close_prices) < period:
                return pd.Series(index=close_prices.index, dtype=float)
            
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return pd.Series(rsi, index=close_prices.index)
            
        except Exception as e:
            logger.error(f"RSI計算失敗: {e}")
            return pd.Series(index=close_prices.index, dtype=float)
    
    @staticmethod
    def calculate_macd(close_prices: Union[pd.Series, pd.DataFrame], fast: int = 12, 
                      slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        計算MACD指標
        
        Args:
            close_prices: 收盤價序列或DataFrame
            fast: 快線週期
            slow: 慢線週期
            signal: 信號線週期
            
        Returns:
            包含MACD線、信號線和柱狀圖的字典
        """
        try:
            if isinstance(close_prices, pd.DataFrame):
                close_prices = close_prices['Close']
            
            exp1 = close_prices.ewm(span=fast, adjust=False).mean()
            exp2 = close_prices.ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            histogram = macd - signal_line
            
            return {
                'MACD': macd,
                'Signal': signal_line,
                'Histogram': histogram
            }
            
        except Exception as e:
            logger.error(f"MACD計算失敗: {e}")
            empty_series = pd.Series(index=close_prices.index, dtype=float)
            return {
                'MACD': empty_series,
                'Signal': empty_series,
                'Histogram': empty_series
            }
    
    @staticmethod
    def calculate_bollinger_bands(close_prices: Union[pd.Series, pd.DataFrame], period: int = 20, 
                                std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        計算布林通道
        
        Args:
            close_prices: 收盤價序列或DataFrame
            period: 計算週期
            std_dev: 標準差倍數
            
        Returns:
            包含中軌、上軌和下軌的字典
        """
        try:
            if isinstance(close_prices, pd.DataFrame):
                close_prices = close_prices['Close']
            
            middle = close_prices.rolling(window=period).mean()
            std = close_prices.rolling(window=period).std()
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            
            return {
                'Middle': middle,
                'Upper': upper,
                'Lower': lower
            }
            
        except Exception as e:
            logger.error(f"布林通道計算失敗: {e}")
            empty_series = pd.Series(index=close_prices.index, dtype=float)
            return {
                'Middle': empty_series,
                'Upper': empty_series,
                'Lower': empty_series
            }
    
    @staticmethod
    def calculate_moving_averages(close_prices: Union[pd.Series, pd.DataFrame], 
                                periods: List[int] = [5, 10, 20, 50]) -> Dict[str, pd.Series]:
        """
        計算移動平均線
        
        Args:
            close_prices: 收盤價序列或DataFrame
            periods: 計算週期列表
            
        Returns:
            包含各週期移動平均線的字典
        """
        try:
            if isinstance(close_prices, pd.DataFrame):
                close_prices = close_prices['Close']
            
            ma_dict = {}
            for period in periods:
                ma = close_prices.rolling(window=period).mean()
                ma_dict[f'MA{period}'] = ma
            
            return ma_dict
            
        except Exception as e:
            logger.error(f"移動平均線計算失敗: {e}")
            empty_series = pd.Series(index=close_prices.index, dtype=float)
            return {f'MA{period}': empty_series.copy() for period in periods}
    
    @staticmethod
    def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """
        計算所有技術指標
        
        Args:
            data: 股票數據DataFrame
            
        Returns:
            添加了技術指標的DataFrame
        """
        try:
            result_data = data.copy()
            
            # 計算RSI
            result_data['RSI'] = TechnicalIndicators.calculate_rsi(result_data['Close'])
            
            # 計算MACD
            macd_data = TechnicalIndicators.calculate_macd(result_data['Close'])
            result_data['MACD'] = macd_data['MACD']
            result_data['MACD_Signal'] = macd_data['Signal']
            result_data['MACD_Histogram'] = macd_data['Histogram']
            
            # 計算布林通道
            bb_data = TechnicalIndicators.calculate_bollinger_bands(result_data['Close'])
            result_data['BB_Middle'] = bb_data['Middle']
            result_data['BB_Upper'] = bb_data['Upper']
            result_data['BB_Lower'] = bb_data['Lower']
            
            # 計算移動平均線
            ma_data = TechnicalIndicators.calculate_moving_averages(result_data['Close'])
            for ma_name, ma_values in ma_data.items():
                result_data[ma_name] = ma_values
            
            return result_data
            
        except Exception as e:
            logger.error(f"技術指標計算失敗: {e}")
            return data

class DataValidator:
    """數據驗證器"""
    
    @staticmethod
    def validate_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
        """
        驗證數據質量
        
        Args:
            data: 待驗證的DataFrame
            
        Returns:
            驗證結果字典
        """
        try:
            # 獲取日期範圍
            start_date = None
            end_date = None
            if not data.empty and isinstance(data.index, pd.DatetimeIndex):
                start_date = format_date_safely(data.index.min())
                end_date = format_date_safely(data.index.max())
            
            validation_results = {
                'missing_values': data.isnull().sum().to_dict(),
                'data_types': {str(k): str(v) for k, v in data.dtypes.to_dict().items()},
                'row_count': len(data),
                'column_count': len(data.columns),
                'date_range': {
                    'start': start_date,
                    'end': end_date
                }
            }
            
            return validation_results
            
        except Exception as e:
            logger.error(f"數據驗證失敗: {e}")
            return {
                'error': str(e),
                'status': 'failed'
            } 