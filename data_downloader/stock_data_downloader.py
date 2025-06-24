"""
股票數據下載器
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime
import logging
from typing import Dict, Optional, List, Union
import tempfile
import time
from random import uniform
import requests
import socket

class HKStockDataDownloader:
    """港股數據下載器"""

    def __init__(self, cache_dir: str = 'data/stock_data', max_retries: int = 3):
        """
        初始化下載器

        Args:
            cache_dir (str): 緩存目錄
            max_retries (int): 最大重試次數
        """
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        os.makedirs(cache_dir, exist_ok=True)
        
        # 設置日誌
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # 禁用SSL驗證警告
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # 設置yfinance配置
        yf.set_tz_cache_location(tempfile.gettempdir())

    def _check_internet_connection(self) -> bool:
        """
        檢查網絡連接

        Returns:
            bool: 是否連接正常
        """
        try:
            # 嘗試連接到Yahoo Finance
            socket.create_connection(("query1.finance.yahoo.com", 443), timeout=5)
            return True
        except OSError:
            self.logger.error("無法連接到Yahoo Finance，請檢查網絡連接")
            return False

    def _format_symbol(self, symbol: str) -> str:
        """
        格式化股票代碼

        Args:
            symbol (str): 原始股票代碼

        Returns:
            str: 格式化後的股票代碼
        """
        # 移除所有空白字符
        symbol = symbol.strip()
        
        # 如果已經包含.HK後綴，直接返回
        if symbol.endswith('.HK'):
            return symbol
            
        # 補零並加上.HK後綴
        return f"{symbol.zfill(4)}.HK"

    def _download_with_retry(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        帶重試機制的數據下載

        Args:
            symbol (str): 股票代碼
            start_date (str): 開始日期
            end_date (str): 結束日期

        Returns:
            Optional[pd.DataFrame]: 股票數據，如果下載失敗則返回None
        """
        if not self._check_internet_connection():
            return None

        formatted_symbol = self._format_symbol(symbol)
        retry_count = 0
        base_delay = 10  # 增加基礎延遲時間到10秒
        
        while retry_count < self.max_retries:
            try:
                # 添加指數退避延遲
                if retry_count > 0:
                    delay = base_delay * (2 ** (retry_count - 1)) + uniform(5, 10)
                    self.logger.info(f"等待 {delay:.1f} 秒後重試 (第 {retry_count + 1} 次嘗試)...")
                    time.sleep(delay)
                
                self.logger.info(f"下載 {formatted_symbol} 數據...")
                
                # 創建一個新的session用於此次下載
                session = requests.Session()
                session.verify = False  # 禁用SSL驗證
                
                # 設置較長的超時時間
                data = yf.download(
                    formatted_symbol,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    timeout=60,  # 增加超時時間到60秒
                    session=session
                )
                
                if data is not None and not data.empty:
                    return data
                else:
                    self.logger.warning(f"下載的數據為空: {formatted_symbol}")
                    
            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"下載失敗: {error_msg}")
                
                # 如果是速率限制錯誤，增加等待時間
                if "Too Many Requests" in error_msg:
                    delay = base_delay * (2 ** retry_count) + uniform(10, 20)
                    self.logger.info(f"遇到速率限制，等待 {delay:.1f} 秒...")
                    time.sleep(delay)
            
            retry_count += 1
            
        return None

    def download_stock_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: Optional[str] = None, 
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        下載股票數據

        Args:
            symbol (str): 股票代碼
            start_date (str): 開始日期
            end_date (str, optional): 結束日期
            use_cache (bool): 是否使用緩存

        Returns:
            pd.DataFrame: 股票數據
        """
        try:
            # 處理結束日期
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')

            # 檢查緩存
            cache_file = os.path.join(self.cache_dir, f"{symbol}_{start_date}_{end_date}.csv")
            
            if use_cache and os.path.exists(cache_file):
                self.logger.info(f"從緩存讀取 {symbol} 數據")
                try:
                    data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    if isinstance(data, pd.DataFrame) and not data.empty:
                        return data
                except Exception as e:
                    self.logger.warning(f"讀取緩存失敗: {str(e)}")

            # 下載數據
            data = self._download_with_retry(symbol, start_date, end_date)
            
            if data is None or data.empty:
                self.logger.warning(f"無法獲取 {symbol} 數據")
                return pd.DataFrame()

            # 數據清理
            data = data.fillna(method='ffill').fillna(method='bfill')

            # 保存到緩存
            if use_cache and not data.empty:
                try:
                    data.to_csv(cache_file)
                    self.logger.info(f"已緩存 {symbol} 數據")
                except Exception as e:
                    self.logger.warning(f"保存緩存失敗: {str(e)}")

            self.logger.info(f"成功下載 {symbol} 數據 ({len(data)} 條記錄)")
            return data

        except Exception as e:
            self.logger.error(f"下載 {symbol} 數據時出錯: {str(e)}")
            return pd.DataFrame()

    def download_multiple_stocks(
        self, 
        symbols: List[str], 
        start_date: str, 
        end_date: Optional[str] = None, 
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        批量下載多個股票的數據

        Args:
            symbols (list): 股票代碼列表
            start_date (str): 開始日期
            end_date (str, optional): 結束日期
            use_cache (bool): 是否使用緩存

        Returns:
            dict: 股票數據字典
        """
        data_dict: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            data = self.download_stock_data(symbol, start_date, end_date, use_cache)
            if not data.empty:
                data_dict[symbol] = data
            # 添加延遲以避免速率限制
            if symbol != symbols[-1]:  # 如果不是最後一個股票
                delay = uniform(10, 20)  # 10-20秒的隨機延遲
                self.logger.info(f"等待 {delay:.1f} 秒後下載下一個股票...")
                time.sleep(delay)
        return data_dict

    def clear_cache(self) -> None:
        """清除緩存"""
        try:
            import shutil
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir, exist_ok=True)
                self.logger.info("緩存已清除")
        except Exception as e:
            self.logger.error(f"清除緩存時出錯: {str(e)}")

    def list_cached_data(self) -> List[str]:
        """列出緩存的數據"""
        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.csv')]
            if cache_files:
                self.logger.info(f"緩存文件 ({len(cache_files)} 個):")
                for file in sorted(cache_files):
                    self.logger.info(f"   {file}")
            else:
                self.logger.info("無緩存文件")
            return cache_files
        except Exception as e:
            self.logger.error(f"列出緩存時出錯: {str(e)}")
            return []