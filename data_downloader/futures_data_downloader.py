"""
港股期貨數據下載器
獲取恆生指數期貨、國企指數期貨等相關期貨數據
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import logger, NON_PRICE_CONFIG
from data_handler import TechnicalIndicators

class HKFuturesDataDownloader:
    """港股期貨數據下載器"""
    
    def __init__(self, data_dir: str = "data_output/futures"):
        from pathlib import Path
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 從配置獲取期貨合約
        self.futures_symbols = NON_PRICE_CONFIG.hk_futures_symbols
        
        # 擴展期貨合約映射
        self.extended_futures = {
            # 恆生指數相關
            '^HSI': '恆生指數',
            'HSI=F': '恆生指數期貨主力合約',
            
            # 國企指數相關  
            '^HSCE': '恆生國企指數',
            'HSCE=F': '國企指數期貨主力合約',
            
            # 科技指數相關
            '^HSTECH': '恆生科技指數',
            'HSTECH=F': '科技指數期貨主力合約',
            
            # 小型恆指期貨
            'MHI=F': '小型恆指期貨主力合約',
            
            # 中華120指數
            '^HSC120': '中華120指數',
            
            # 相關ETF作為替代
            '2800.HK': '盈富基金ETF',
            '2828.HK': '恆生國企ETF',
            '3067.HK': '恆生科技ETF'
        }
        
    def download_futures_data(self, 
                            symbol: str, 
                            start_date: str,
                            end_date: str,
                            interval: str = "1d") -> Optional[pd.DataFrame]:
        """下載期貨數據"""
        try:
            logger.info(f"正在下載期貨數據: {symbol}")
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                logger.warning(f"❌ {symbol} 沒有獲取到期貨數據")
                return None
                
            # 數據清理
            data = data.dropna()
            
            # 添加元數據
            data['Symbol'] = symbol
            data['Contract_Name'] = self.extended_futures.get(symbol, symbol)
            data['Data_Type'] = 'futures'
            
            # 計算額外指標
            data['Price_Change'] = data['Close'].diff()
            data['Price_Change_Pct'] = data['Close'].pct_change() * 100
            data['Volatility'] = data['Close'].rolling(window=20).std()
            
            # 重新排列列
            columns_order = ['Symbol', 'Contract_Name', 'Data_Type', 'Open', 'High', 
                           'Low', 'Close', 'Volume', 'Price_Change', 'Price_Change_Pct', 
                           'Volatility']
            
            available_columns = [col for col in columns_order if col in data.columns]
            data = data[available_columns]
            
            # 保存數據
            filename = f"{symbol.replace('=', '_').replace('^', '').replace('.', '_')}_futures.csv"
            filepath = self.data_dir / filename
            data.to_csv(filepath)
            logger.info(f"✅ {symbol} 期貨數據已保存到 {filepath}")
            
            return data
            
        except Exception as e:
            logger.error(f"❌ 下載 {symbol} 期貨數據失敗: {e}")
            return None
    
    def download_all_futures(self, 
                           start_date: str = "2023-01-01",
                           end_date: str = None,
                           interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """批量下載所有期貨數據"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        logger.info(f"開始批量下載期貨數據...")
        logger.info(f"時間範圍: {start_date} 到 {end_date}")
        
        all_futures_data = {}
        failed_downloads = []
        
        for symbol in self.extended_futures.keys():
            data = self.download_futures_data(symbol, start_date, end_date, interval)
            
            if data is not None:
                all_futures_data[symbol] = data
            else:
                failed_downloads.append(symbol)
                
            # 添加延遲
            import time
            time.sleep(NON_PRICE_CONFIG.download_delay_seconds)
        
        self._generate_futures_report(all_futures_data, failed_downloads)
        return all_futures_data
    
    def calculate_futures_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """計算期貨技術指標"""
        try:
            # 移動平均線
            data['MA5'] = data['Close'].rolling(window=5).mean()
            data['MA20'] = data['Close'].rolling(window=20).mean()
            data['MA60'] = data['Close'].rolling(window=60).mean()
            
            # 布林帶
            data['BB_Middle'] = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
            data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = data['Close'].ewm(span=12).mean()
            exp2 = data['Close'].ewm(span=26).mean()
            data['MACD'] = exp1 - exp2
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
            data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
            
            return data
            
        except Exception as e:
            logger.error(f"計算期貨技術指標失敗: {e}")
            return data
    
    def analyze_futures_correlation(self, 
                                  all_data: Dict[str, pd.DataFrame],
                                  target_stock: str = "2800.HK") -> pd.DataFrame:
        """分析期貨與目標股票的相關性"""
        try:
            correlation_results = []
            
            # 獲取目標股票數據作為基準
            if target_stock not in all_data:
                logger.warning(f"目標股票 {target_stock} 數據不存在")
                return pd.DataFrame()
                
            target_data = all_data[target_stock]['Close']
            
            for symbol, futures_data in all_data.items():
                if symbol == target_stock:
                    continue
                    
                futures_close = futures_data['Close']
                
                # 對齊數據
                aligned_data = pd.concat([target_data, futures_close], axis=1, join='inner')
                aligned_data.columns = ['target', 'futures']
                aligned_data = aligned_data.dropna()
                
                if len(aligned_data) < 20:  # 需要足夠的數據點
                    continue
                    
                # 計算相關性
                correlation = aligned_data['target'].corr(aligned_data['futures'])
                
                # 計算滾動相關性
                rolling_corr = aligned_data['target'].rolling(window=20).corr(aligned_data['futures'])
                avg_rolling_corr = rolling_corr.mean()
                
                correlation_results.append({
                    'Futures_Symbol': symbol,
                    'Contract_Name': self.extended_futures.get(symbol, symbol),
                    'Correlation': round(correlation, 4),
                    'Avg_Rolling_Correlation': round(avg_rolling_corr, 4),
                    'Data_Points': len(aligned_data),
                    'Correlation_Strength': self._categorize_correlation(correlation)
                })
            
            results_df = pd.DataFrame(correlation_results)
            results_df = results_df.sort_values('Correlation', key=abs, ascending=False)
            
            # 保存相關性分析結果
            filepath = self.data_dir / f"futures_correlation_analysis_{datetime.now().strftime('%Y%m%d')}.csv"
            results_df.to_csv(filepath, index=False)
            logger.info(f"相關性分析結果已保存到: {filepath}")
            
            return results_df
            
        except Exception as e:
            logger.error(f"期貨相關性分析失敗: {e}")
            return pd.DataFrame()
    
    def _categorize_correlation(self, correlation: float) -> str:
        """分類相關性強度"""
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            return "非常強"
        elif abs_corr >= 0.6:
            return "強"
        elif abs_corr >= 0.4:
            return "中等"
        elif abs_corr >= 0.2:
            return "弱"
        else:
            return "非常弱"
    
    def generate_futures_features(self, 
                                all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """生成期貨特徵數據集"""
        try:
            features_list = []
            
            for symbol, data in all_data.items():
                # 計算技術指標
                data_with_indicators = self.calculate_futures_indicators(data.copy())
                
                # 選擇特徵列
                feature_columns = ['Close', 'Volume', 'Volatility', 'Price_Change_Pct',
                                 'MA5', 'MA20', 'RSI', 'MACD', 'MACD_Signal']
                
                available_features = [col for col in feature_columns if col in data_with_indicators.columns]
                features_data = data_with_indicators[available_features].copy()
                
                # 添加標識
                for col in available_features:
                    features_data[f"{symbol}_{col}"] = features_data[col]
                    
                features_list.append(features_data[[f"{symbol}_{col}" for col in available_features]])
            
            # 合併所有特徵
            if features_list:
                combined_features = pd.concat(features_list, axis=1, join='outer')
                
                # 保存特徵數據
                filepath = self.data_dir / f"futures_features_{datetime.now().strftime('%Y%m%d')}.csv"
                combined_features.to_csv(filepath)
                logger.info(f"期貨特徵數據已保存到: {filepath}")
                
                return combined_features
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"生成期貨特徵失敗: {e}")
            return pd.DataFrame()
    
    def _generate_futures_report(self, 
                               successful_data: Dict[str, pd.DataFrame],
                               failed_downloads: List[str]):
        """生成期貨數據下載報告"""
        report_lines = [
            "=" * 50,
            "港股期貨數據下載報告",
            "=" * 50,
            f"總共嘗試下載: {len(self.extended_futures)} 個期貨合約",
            f"成功下載: {len(successful_data)} 個合約",
            f"下載失敗: {len(failed_downloads)} 個合約",
            ""
        ]
        
        if successful_data:
            report_lines.append("成功下載的期貨合約:")
            for symbol in successful_data.keys():
                data_points = len(successful_data[symbol])
                start_date = successful_data[symbol].index.min().strftime('%Y-%m-%d')
                end_date = successful_data[symbol].index.max().strftime('%Y-%m-%d')
                contract_name = self.extended_futures.get(symbol, symbol)
                report_lines.append(f"  {symbol} ({contract_name}): {data_points} 個交易日 ({start_date} 至 {end_date})")
                
        if failed_downloads:
            report_lines.append("\n下載失敗的合約:")
            for symbol in failed_downloads:
                contract_name = self.extended_futures.get(symbol, symbol)
                report_lines.append(f"  {symbol} ({contract_name})")
                
        report_content = "\n".join(report_lines)
        
        # 保存報告
        report_filename = f"futures_download_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report_filepath = self.data_dir.parent / "reports" / report_filename
        report_filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        logger.info(f"\n{report_content}")
        logger.info(f"期貨數據下載報告已保存到: {report_filepath}")

def main():
    """測試期貨數據下載器"""
    logger.info("🚀 測試港股期貨數據下載器...")
    
    downloader = HKFuturesDataDownloader()
    
    # 測試單個期貨合約下載
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    # 下載所有期貨數據
    all_data = downloader.download_all_futures(start_date=start_date)
    
    if all_data:
        logger.info(f"✅ 成功下載 {len(all_data)} 個期貨合約數據")
        
        # 分析相關性
        correlation_results = downloader.analyze_futures_correlation(all_data)
        if not correlation_results.empty:
            logger.info("✅ 相關性分析完成")
            logger.info(f"前5個最相關的期貨合約:")
            print(correlation_results.head())
        
        # 生成特徵數據
        features = downloader.generate_futures_features(all_data)
        if not features.empty:
            logger.info(f"✅ 期貨特徵數據生成完成，維度: {features.shape}")
    else:
        logger.error("❌ 期貨數據下載失敗")

if __name__ == "__main__":
    main()