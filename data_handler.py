"""
港股量化分析系統 - 數據處理模組
包含數據獲取、技術指標計算、數據清理等功能
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from typing import Optional, Dict, List, Tuple, Union, Any, cast
import warnings
from config import logger
import time

# 忽略警告
warnings.filterwarnings('ignore')

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
    
    @staticmethod
    def get_yahoo_finance_data(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        使用 Yahoo Finance API 獲取股票數據
        
        Args:
            symbol: 股票代碼
            start_date: 開始日期 (YYYY-MM-DD)
            end_date: 結束日期 (YYYY-MM-DD)
            
        Returns:
            股票數據DataFrame
        """
        try:
            # 轉換日期為時間戳
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
            
            # 構建 Yahoo Finance API URL
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                "period1": start_ts,
                "period2": end_ts,
                "interval": "1d",
                "events": "history",
                "includeAdjustedClose": True
            }
            
            # 發送請求（設置超時）
            headers = {
                'User-Agent': 'Mozilla/5.0',
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip, deflate'
            }
            response = requests.get(url, params=params, headers=headers, timeout=30)
            data = response.json()
            
            # 解析數據
            if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                result = data['chart']['result'][0]
                timestamps = result['timestamp']
                quote = result['indicators']['quote'][0]
                
                # 創建 DataFrame
                df = pd.DataFrame({
                    'Open': quote.get('open', []),
                    'High': quote.get('high', []),
                    'Low': quote.get('low', []),
                    'Close': quote.get('close', []),
                    'Volume': quote.get('volume', [])
                }, index=pd.to_datetime([datetime.fromtimestamp(x) for x in timestamps]))
                
                # 清理數據
                df = DataFetcher._clean_data(df)
                
                if not df.empty:
                    logger.info(f"✅ 成功從 Yahoo Finance API 獲取 {len(df)} 天的數據")
                    return df
                    
            logger.error("❌ Yahoo Finance API 返回的數據格式不正確")
            return None
            
        except requests.Timeout:
            logger.error("❌ Yahoo Finance API 請求超時")
            return None
        except Exception as e:
            logger.error(f"❌ 從 Yahoo Finance API 獲取數據失敗: {e}")
            return None

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