# 港股量化分析系統 - 非價格數據整合指南

## 📊 概述

本指南詳細說明如何在現有港股量化分析系統中整合**非價格數據**（Non-Price Data），特別是**香港日夜期貨數據**，以提升策略的預測能力和Alpha生成效果。

---

## 🎯 為什麼需要非價格數據？

### 傳統價格數據的局限性
- **信息滯後**：價格反映的是已發生的市場情緒
- **噪音過多**：短期價格波動包含大量隨機成分
- **同質化**：所有參與者都看到相同的價格數據
- **有效市場假說**：價格已充分反映所有公開信息

### 非價格數據的優勢
- **信息領先性**：可能預示未來價格走勢
- **獨特性**：較少人關注，競爭優勢明顯
- **多維度分析**：提供更全面的市場視角
- **高夏普比率潛力**：創造真正的Alpha

---

## 🏗️ 系統架構擴展

### 1. 數據架構擴展

#### 📁 新增模組結構
```
港股量化分析系統/
├── 📊 non_price_data/                 # 新增：非價格數據模組
│   ├── __init__.py
│   ├── futures_data_handler.py        # 期貨數據處理器
│   ├── market_sentiment_analyzer.py   # 市場情緒分析器
│   ├── macro_data_fetcher.py          # 宏觀數據獲取器
│   ├── options_data_handler.py        # 期權數據處理器
│   ├── news_sentiment_analyzer.py     # 新聞情緒分析器
│   └── alternative_data_sources.py    # 替代數據源
├── 🔧 data_fusion/                    # 新增：數據融合模組
│   ├── __init__.py
│   ├── data_synchronizer.py           # 數據同步器
│   ├── feature_engineer.py            # 特徵工程
│   └── correlation_analyzer.py        # 相關性分析器
└── 📈 enhanced_strategies/             # 增強策略模組
    ├── __init__.py
    ├── multi_factor_strategies.py     # 多因子策略
    ├── futures_arbitrage.py           # 期現套利
    └── sentiment_momentum.py          # 情緒動量策略
```

---

## 🚀 Step-by-Step 實施計劃

### 階段0：股票數據批量下載系統 (1天)

#### 0.1 創建股票數據批量下載器

```python
# data_downloader/stock_data_downloader.py
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import os
from pathlib import Path
from config import logger
import time

class HKStockDataDownloader:
    """港股數據批量下載器"""
    
    def __init__(self, data_dir: str = "data_output/csv"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 港股標的池
        self.hk_stocks = {
            '2800.HK': '盈富基金',
            '700.HK': '騰訊控股', 
            '939.HK': '建設銀行',
            '941.HK': '中國移動',
            '1299.HK': '友邦保險',
            '1398.HK': '工商銀行',
            '3988.HK': '中國銀行',
            '5.HK': '匯豐控股',
            '1109.HK': '華潤置地',
            '2388.HK': '中銀香港',
            '3968.HK': '招商銀行',
            '2318.HK': '中國平安',
            '6862.HK': '海底撈',
            '9988.HK': '阿里巴巴',
            '9618.HK': '京東集團',
            '3690.HK': '美團',
            '1024.HK': '快手',
            '2269.HK': '藥明生物',
            '1810.HK': '小米集團',
            '9999.HK': '網易'
        }
        
    def download_single_stock(self, 
                            symbol: str, 
                            start_date: str, 
                            end_date: str,
                            save_csv: bool = True) -> Optional[pd.DataFrame]:
        """下載單隻股票數據"""
        try:
            logger.info(f"正在下載 {symbol} ({self.hk_stocks.get(symbol, '未知')}) 數據...")
            
            # 下載數據
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                logger.warning(f"❌ {symbol} 沒有獲取到數據")
                return None
                
            # 數據清理
            data = data.dropna()
            
            # 添加股票代碼列
            data['Symbol'] = symbol
            data['Stock_Name'] = self.hk_stocks.get(symbol, symbol)
            
            # 重新排列列順序
            columns_order = ['Symbol', 'Stock_Name', 'Open', 'High', 'Low', 'Close', 'Volume']
            if 'Dividends' in data.columns:
                columns_order.append('Dividends')
            if 'Stock Splits' in data.columns:
                columns_order.append('Stock Splits')
                
            data = data[columns_order]
            
            if save_csv:
                # 保存到CSV
                filename = f"{symbol.replace('.', '_')}_stock_data.csv"
                filepath = self.data_dir / filename
                data.to_csv(filepath)
                logger.info(f"✅ {symbol} 數據已保存到 {filepath}")
                
            return data
            
        except Exception as e:
            logger.error(f"❌ 下載 {symbol} 數據失敗: {e}")
            return None
            
    def download_all_stocks(self, 
                          start_date: str = "2023-01-01", 
                          end_date: str = None,
                          delay_seconds: float = 1.0) -> Dict[str, pd.DataFrame]:
        """批量下載所有股票數據"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        logger.info(f"開始批量下載 {len(self.hk_stocks)} 隻港股數據...")
        logger.info(f"時間範圍: {start_date} 到 {end_date}")
        
        all_data = {}
        failed_downloads = []
        
        for i, (symbol, name) in enumerate(self.hk_stocks.items(), 1):
            logger.info(f"進度: {i}/{len(self.hk_stocks)} - {symbol}")
            
            data = self.download_single_stock(symbol, start_date, end_date)
            
            if data is not None:
                all_data[symbol] = data
            else:
                failed_downloads.append(symbol)
                
            # 添加延遲避免被限制
            if delay_seconds > 0:
                time.sleep(delay_seconds)
                
        # 生成摘要報告
        self._generate_download_report(all_data, failed_downloads)
        
        return all_data
    
    def create_combined_dataset(self, 
                              all_data: Dict[str, pd.DataFrame],
                              save_csv: bool = True) -> pd.DataFrame:
        """創建合併數據集"""
        try:
            # 合併所有股票數據
            combined_data = pd.concat(all_data.values(), ignore_index=False)
            combined_data = combined_data.sort_index()
            
            if save_csv:
                # 保存合併數據
                filename = f"HK_Stocks_Combined_{datetime.now().strftime('%Y%m%d')}.csv"
                filepath = self.data_dir / filename
                combined_data.to_csv(filepath)
                logger.info(f"✅ 合併數據已保存到 {filepath}")
                
            return combined_data
            
        except Exception as e:
            logger.error(f"❌ 創建合併數據集失敗: {e}")
            return pd.DataFrame()
    
    def _generate_download_report(self, 
                                successful_data: Dict[str, pd.DataFrame],
                                failed_downloads: List[str]):
        """生成下載報告"""
        report_lines = [
            "=" * 50,
            "港股數據下載報告",
            "=" * 50,
            f"總共嘗試下載: {len(self.hk_stocks)} 隻股票",
            f"成功下載: {len(successful_data)} 隻股票",
            f"下載失敗: {len(failed_downloads)} 隻股票",
            ""
        ]
        
        if successful_data:
            report_lines.append("成功下載的股票:")
            for symbol in successful_data.keys():
                data_points = len(successful_data[symbol])
                start_date = successful_data[symbol].index.min().strftime('%Y-%m-%d')
                end_date = successful_data[symbol].index.max().strftime('%Y-%m-%d')
                report_lines.append(f"  {symbol}: {data_points} 個交易日 ({start_date} 至 {end_date})")
                
        if failed_downloads:
            report_lines.append("\n下載失敗的股票:")
            for symbol in failed_downloads:
                report_lines.append(f"  {symbol}")
                
        report_content = "\n".join(report_lines)
        
        # 保存報告
        report_filename = f"download_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report_filepath = self.data_dir.parent / "reports" / report_filename
        report_filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        # 同時輸出到控制台
        logger.info(f"\n{report_content}")
        logger.info(f"詳細報告已保存到: {report_filepath}")

    def update_existing_data(self, 
                           symbols: List[str] = None,
                           days_back: int = 7) -> Dict[str, pd.DataFrame]:
        """更新現有數據（增量下載）"""
        if symbols is None:
            symbols = list(self.hk_stocks.keys())
            
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        logger.info(f"增量更新最近 {days_back} 天的數據...")
        
        updated_data = {}
        for symbol in symbols:
            # 檢查現有文件
            filename = f"{symbol.replace('.', '_')}_stock_data.csv"
            filepath = self.data_dir / filename
            
            if filepath.exists():
                # 讀取現有數據
                existing_data = pd.read_csv(filepath, index_col=0, parse_dates=True)
                
                # 下載新數據
                new_data = self.download_single_stock(symbol, start_date, end_date, save_csv=False)
                
                if new_data is not None:
                    # 合併數據（去重）
                    combined = pd.concat([existing_data, new_data])
                    combined = combined[~combined.index.duplicated(keep='last')]
                    combined = combined.sort_index()
                    
                    # 保存更新後的數據
                    combined.to_csv(filepath)
                    updated_data[symbol] = combined
                    logger.info(f"✅ {symbol} 數據已更新")
            else:
                # 文件不存在，完整下載
                data = self.download_single_stock(symbol, "2023-01-01", end_date)
                if data is not None:
                    updated_data[symbol] = data
                    
        return updated_data
```

#### 0.2 創建數據下載腳本

```python
# scripts/download_hk_stocks.py
"""
港股數據批量下載腳本
使用方法: python scripts/download_hk_stocks.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_downloader.stock_data_downloader import HKStockDataDownloader
from config import logger
import argparse
from datetime import datetime, timedelta

def main():
    parser = argparse.ArgumentParser(description='港股數據批量下載工具')
    parser.add_argument('--start-date', type=str, 
                       default=(datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d'),
                       help='開始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, 
                       default=datetime.now().strftime('%Y-%m-%d'),
                       help='結束日期 (YYYY-MM-DD)')
    parser.add_argument('--symbols', nargs='+', 
                       help='指定股票代碼（可選）')
    parser.add_argument('--update-only', action='store_true',
                       help='僅更新現有數據')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='下載間隔（秒）')
    
    args = parser.parse_args()
    
    # 創建下載器
    downloader = HKStockDataDownloader()
    
    if args.update_only:
        # 增量更新模式
        logger.info("🔄 執行增量更新...")
        updated_data = downloader.update_existing_data(
            symbols=args.symbols,
            days_back=7
        )
        logger.info(f"✅ 完成！更新了 {len(updated_data)} 隻股票")
        
    else:
        # 完整下載模式
        if args.symbols:
            # 下載指定股票
            downloader.hk_stocks = {
                symbol: downloader.hk_stocks.get(symbol, symbol) 
                for symbol in args.symbols
            }
            
        logger.info("📥 執行完整數據下載...")
        all_data = downloader.download_all_stocks(
            start_date=args.start_date,
            end_date=args.end_date,
            delay_seconds=args.delay
        )
        
        if all_data:
            # 創建合併數據集
            combined_data = downloader.create_combined_dataset(all_data)
            logger.info(f"✅ 完成！下載了 {len(all_data)} 隻股票，共 {len(combined_data)} 條記錄")
        else:
            logger.error("❌ 沒有成功下載任何數據")

if __name__ == "__main__":
    main()
```

### 階段1：香港日夜期貨數據整合 (2-3天)

#### 1.1 創建期貨數據處理器

```python
# non_price_data/futures_data_handler.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import requests
from datetime import datetime, timedelta
from config import logger

class HKFuturesDataHandler:
    \"\"\"香港期貨數據處理器\"\"\"
    
    def __init__(self):
        self.futures_symbols = {
            'HSI_DAY': '^HSI',      # 恆指日間期貨
            'HSI_NIGHT': 'HSI2300', # 恆指夜間期貨
            'HSCEI': '^HSCE',       # 國企指數期貨
            'HSTECH': '^HSTECH'     # 恆生科技指數期貨
        }
        
    def fetch_futures_data(self, 
                          start_date: str, 
                          end_date: str,
                          symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
        \"\"\"獲取期貨數據\"\"\"
        if symbols is None:
            symbols = list(self.futures_symbols.keys())
            
        futures_data = {}
        
        for symbol in symbols:
            try:
                ticker = self.futures_symbols.get(symbol)
                if ticker:
                    data = yf.download(ticker, start=start_date, end=end_date)
                    futures_data[symbol] = data
                    logger.info(f\"✅ 成功獲取 {symbol} 期貨數據\")
            except Exception as e:
                logger.error(f\"❌ 獲取 {symbol} 期貨數據失敗: {e}\")
                
        return futures_data
    
    def calculate_day_night_spread(self, 
                                  day_data: pd.DataFrame, 
                                  night_data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"計算日夜期貨價差\"\"\"
        try:
            # 對齊數據時間戳
            aligned_data = pd.merge(
                day_data[['Close']].rename(columns={'Close': 'Day_Close'}),
                night_data[['Close']].rename(columns={'Close': 'Night_Close'}),
                left_index=True, right_index=True, how='inner'
            )
            
            # 計算價差指標
            aligned_data['Spread'] = aligned_data['Night_Close'] - aligned_data['Day_Close']
            aligned_data['Spread_Pct'] = aligned_data['Spread'] / aligned_data['Day_Close'] * 100
            aligned_data['Spread_MA5'] = aligned_data['Spread_Pct'].rolling(5).mean()
            aligned_data['Spread_Std5'] = aligned_data['Spread_Pct'].rolling(5).std()
            aligned_data['Spread_Zscore'] = (aligned_data['Spread_Pct'] - aligned_data['Spread_MA5']) / aligned_data['Spread_Std5']
            
            return aligned_data
            
        except Exception as e:
            logger.error(f\"計算日夜價差失敗: {e}\")
            return pd.DataFrame()
    
    def generate_futures_signals(self, spread_data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"基於期貨價差生成交易信號\"\"\"
        signals = pd.DataFrame(index=spread_data.index)
        
        # 信號1：價差均值回歸
        signals['Mean_Reversion_Signal'] = np.where(
            spread_data['Spread_Zscore'] > 2, -1,  # 做空信號
            np.where(spread_data['Spread_Zscore'] < -2, 1, 0)  # 做多信號
        )
        
        # 信號2：趨勢跟隨
        signals['Trend_Signal'] = np.where(
            spread_data['Spread_Pct'] > spread_data['Spread_MA5'], 1, -1
        )
        
        # 信號3：波動率突破
        signals['Volatility_Signal'] = np.where(
            abs(spread_data['Spread_Zscore']) > 1.5, 1, 0
        )
        
        return signals
```

#### 1.2 修改現有配置文件

```python
# config.py 新增配置
@dataclass
class NonPriceDataConfig:
    \"\"\"非價格數據配置\"\"\"
    # 期貨數據配置
    enable_futures_data: bool = True
    futures_update_interval: int = 300  # 5分鐘更新
    futures_lookback_days: int = 30     # 期貨數據回看天數
    
    # 市場情緒配置
    enable_sentiment_analysis: bool = True
    sentiment_sources: List[str] = ['news', 'social_media', 'options']
    
    # 宏觀數據配置
    enable_macro_data: bool = True
    macro_indicators: List[str] = ['interest_rate', 'exchange_rate', 'vix']
    
    # 數據融合配置
    correlation_threshold: float = 0.3
    feature_selection_method: str = 'mutual_info'
    max_features: int = 50

# 添加到全域配置
NON_PRICE_CONFIG = NonPriceDataConfig()
```

#### 1.3 擴展數據處理主類

```python
# data_handler.py 新增方法
class DataFetcher:
    # ... 現有代碼 ...
    
    @staticmethod
    def get_non_price_data(symbol: str, 
                          start_date: str, 
                          end_date: str,
                          data_types: List[str] = ['futures']) -> Dict[str, pd.DataFrame]:
        \"\"\"獲取非價格數據\"\"\"
        non_price_data = {}
        
        if 'futures' in data_types:
            from non_price_data.futures_data_handler import HKFuturesDataHandler
            futures_handler = HKFuturesDataHandler()
            futures_data = futures_handler.fetch_futures_data(start_date, end_date)
            non_price_data['futures'] = futures_data
            
        if 'sentiment' in data_types:
            # 實現情緒數據獲取
            pass
            
        if 'macro' in data_types:
            # 實現宏觀數據獲取
            pass
            
        return non_price_data
```

### 階段2：數據融合和特徵工程 (2-3天)

#### 2.1 創建數據同步器

```python
# data_fusion/data_synchronizer.py
class DataSynchronizer:
    \"\"\"多源數據同步器\"\"\"
    
    def __init__(self):
        self.time_tolerance = timedelta(minutes=30)  # 時間容差
        
    def synchronize_datasets(self, 
                           price_data: pd.DataFrame,
                           non_price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        \"\"\"同步價格和非價格數據\"\"\"
        # 以價格數據的時間戳為基準
        base_index = price_data.index
        synchronized_data = price_data.copy()
        
        for data_type, data in non_price_data.items():
            # 時間對齊
            aligned_data = self._align_timestamps(base_index, data)
            
            # 添加前綴避免列名衝突
            aligned_data.columns = [f\"{data_type}_{col}\" for col in aligned_data.columns]
            
            # 合併數據
            synchronized_data = pd.merge(
                synchronized_data, aligned_data,
                left_index=True, right_index=True,
                how='left'
            )
            
        return synchronized_data
    
    def _align_timestamps(self, 
                         target_index: pd.DatetimeIndex, 
                         source_data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"對齊時間戳\"\"\"
        # 使用前向填充和線性插值
        aligned = source_data.reindex(target_index, method='ffill')
        aligned = aligned.interpolate(method='linear', limit=5)
        return aligned
```

#### 2.2 創建特徵工程模組

```python
# data_fusion/feature_engineer.py
class FeatureEngineer:
    \"\"\"特徵工程器\"\"\"
    
    def __init__(self):
        self.feature_cache = {}
        
    def create_futures_features(self, 
                               price_data: pd.DataFrame,
                               futures_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        \"\"\"創建期貨相關特徵\"\"\"
        features = pd.DataFrame(index=price_data.index)
        
        # 特徵1：期現基差
        if 'HSI_DAY' in futures_data:
            hsi_futures = futures_data['HSI_DAY']
            features['Basis'] = hsi_futures['Close'] - price_data['Close']
            features['Basis_Pct'] = features['Basis'] / price_data['Close'] * 100
            
        # 特徵2：日夜價差
        if 'HSI_DAY' in futures_data and 'HSI_NIGHT' in futures_data:
            day_night_spread = self._calculate_day_night_spread(
                futures_data['HSI_DAY'], futures_data['HSI_NIGHT']
            )
            features = pd.merge(features, day_night_spread, 
                              left_index=True, right_index=True, how='left')
            
        # 特徵3：期貨成交量比率
        features['Futures_Volume_Ratio'] = (
            hsi_futures['Volume'] / price_data['Volume']
        ).rolling(5).mean()
        
        # 特徵4：期貨波動率
        features['Futures_Volatility'] = (
            hsi_futures['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        )
        
        return features
    
    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"創建技術指標特徵\"\"\"
        features = pd.DataFrame(index=data.index)
        
        # 價格動量特徵
        for period in [5, 10, 20]:
            features[f'Momentum_{period}'] = data['Close'].pct_change(period)
            features[f'Price_MA_Ratio_{period}'] = data['Close'] / data['Close'].rolling(period).mean()
            
        # 波動率特徵
        for period in [5, 10, 20]:
            features[f'Volatility_{period}'] = data['Close'].pct_change().rolling(period).std()
            
        # 成交量特徵
        features['Volume_MA_Ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
        features['Price_Volume_Trend'] = (data['Close'].pct_change() * data['Volume']).rolling(5).mean()
        
        return features
    
    def select_features(self, 
                       features: pd.DataFrame, 
                       target: pd.Series,
                       method: str = 'mutual_info',
                       max_features: int = 50) -> List[str]:
        \"\"\"特徵選擇\"\"\"
        from sklearn.feature_selection import mutual_info_regression, SelectKBest
        from sklearn.feature_selection import f_regression
        
        # 移除NaN值
        clean_features = features.dropna()
        clean_target = target.loc[clean_features.index]
        
        if method == 'mutual_info':
            selector = SelectKBest(
                score_func=mutual_info_regression, 
                k=min(max_features, clean_features.shape[1])
            )
        else:
            selector = SelectKBest(
                score_func=f_regression,
                k=min(max_features, clean_features.shape[1])
            )
            
        selector.fit(clean_features, clean_target)
        selected_features = clean_features.columns[selector.get_support()].tolist()
        
        return selected_features
```

### 階段3：增強策略開發 (3-4天)

#### 3.1 多因子策略

```python
# enhanced_strategies/multi_factor_strategies.py
class MultiFactorStrategy:
    """多因子策略基類"""
    
    def __init__(self, factors: List[str], weights: Dict[str, float] = None):
        self.factors = factors
        self.weights = weights or {factor: 1.0/len(factors) for factor in factors}
        self.factor_scores = {}
        
    def calculate_factor_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """計算各因子得分"""
        scores = pd.DataFrame(index=data.index)
        
        # 期貨基差因子
        if 'futures_basis' in self.factors:
            scores['futures_basis'] = self._normalize_factor(
                data.get('Basis_Pct', pd.Series(index=data.index))
            )
            
        # 日夜價差因子
        if 'day_night_spread' in self.factors:
            scores['day_night_spread'] = self._normalize_factor(
                data.get('Spread_Zscore', pd.Series(index=data.index))
            )
            
        # 動量因子
        if 'momentum' in self.factors:
            scores['momentum'] = self._normalize_factor(
                data['Close'].pct_change(20)
            )
            
        # 波動率因子
        if 'volatility' in self.factors:
            vol = data['Close'].pct_change().rolling(20).std()
            scores['volatility'] = self._normalize_factor(-vol)  # 低波動率為正
            
        return scores
    
    def _normalize_factor(self, factor_data: pd.Series) -> pd.Series:
        """標準化因子數據"""
        return (factor_data - factor_data.rolling(252).mean()) / factor_data.rolling(252).std()
    
    def generate_signals(self, factor_scores: pd.DataFrame) -> pd.Series:
        """基於因子得分生成信號"""
        composite_score = pd.Series(0, index=factor_scores.index)
        
        for factor, weight in self.weights.items():
            if factor in factor_scores.columns:
                composite_score += weight * factor_scores[factor]
                
        # 生成三分位信號
        signals = pd.Series(0, index=composite_score.index)
        
        # 滾動排名
        rolling_rank = composite_score.rolling(60).rank(pct=True)
        
        signals[rolling_rank > 0.7] = 1   # 做多信號
        signals[rolling_rank < 0.3] = -1  # 做空信號
        
        return signals
```

#### 3.2 期現套利策略

```python
# enhanced_strategies/futures_arbitrage.py
class FuturesArbitrageStrategy:
    """期現套利策略"""
    
    def __init__(self, spread_threshold: float = 2.0):
        self.spread_threshold = spread_threshold
        
    def generate_arbitrage_signals(self, 
                                 spot_data: pd.DataFrame,
                                 futures_data: pd.DataFrame) -> pd.DataFrame:
        """生成套利信號"""
        signals = pd.DataFrame(index=spot_data.index)
        
        # 計算基差
        basis = futures_data['Close'] - spot_data['Close']
        basis_zscore = (basis - basis.rolling(60).mean()) / basis.rolling(60).std()
        
        # 套利信號
        signals['Long_Spot_Short_Futures'] = np.where(basis_zscore > self.spread_threshold, 1, 0)
        signals['Short_Spot_Long_Futures'] = np.where(basis_zscore < -self.spread_threshold, 1, 0)
        
        # 平倉信號
        signals['Close_Position'] = np.where(abs(basis_zscore) < 0.5, 1, 0)
        
        return signals
```

### 階段4：系統整合和儀表板擴展 (2-3天)

#### 4.1 修改主儀表板以支援非價格數據

```python
# enhanced_interactive_dashboard.py 新增功能
def load_non_price_data():
    """載入非價格數據"""
    try:
        from non_price_data.futures_data_handler import HKFuturesDataHandler
        from data_fusion.data_synchronizer import DataSynchronizer
        from data_fusion.feature_engineer import FeatureEngineer
        
        # 載入期貨數據
        futures_handler = HKFuturesDataHandler()
        futures_data = futures_handler.fetch_futures_data(
            start_date="2024-01-01", 
            end_date="2024-12-31"
        )
        
        # 載入股票數據
        stock_data = load_stock_data_from_csv()
        
        # 數據同步
        synchronizer = DataSynchronizer()
        synchronized_data = synchronizer.synchronize_datasets(
            stock_data, {'futures': futures_data}
        )
        
        # 特徵工程
        feature_engineer = FeatureEngineer()
        features = feature_engineer.create_futures_features(stock_data, futures_data)
        
        return synchronized_data, features
        
    except Exception as e:
        logger.error(f"載入非價格數據失敗: {e}")
        return None, None

def create_non_price_charts(features: pd.DataFrame):
    """創建非價格數據圖表"""
    charts = []
    
    # 期現基差圖
    if 'Basis_Pct' in features.columns:
        basis_chart = go.Scatter(
            x=features.index,
            y=features['Basis_Pct'],
            mode='lines',
            name='期現基差 (%)',
            line=dict(color='orange')
        )
        charts.append(basis_chart)
    
    # 日夜價差圖
    if 'Spread_Zscore' in features.columns:
        spread_chart = go.Scatter(
            x=features.index,
            y=features['Spread_Zscore'],
            mode='lines',
            name='日夜價差 Z-Score',
            line=dict(color='purple')
        )
        charts.append(spread_chart)
    
    return charts

# 在主儀表板回調中添加非價格數據
@app.callback(
    Output('main-chart', 'figure'),
    [Input('strategy-checkboxes', 'value'),
     Input('show-non-price', 'value')]  # 新增控制項
)
def update_main_chart(selected_strategies, show_non_price):
    """更新主圖表，包含非價格數據"""
    # ... 現有策略邏輯 ...
    
    if show_non_price and 'non_price_features' in globals():
        non_price_charts = create_non_price_charts(non_price_features)
        for chart in non_price_charts:
            main_fig.add_trace(chart)
    
    return main_fig
```

#### 4.2 添加新的控制面板

```python
# 在儀表板佈局中添加非價格數據控制
html.Div([
    html.H5("非價格數據選項", style={'color': '#eee', 'marginBottom': '10px'}),
    dcc.Checklist(
        id='non-price-options',
        options=[
            {'label': '期現基差', 'value': 'basis'},
            {'label': '日夜價差', 'value': 'day_night_spread'},
            {'label': '期貨波動率', 'value': 'futures_volatility'},
            {'label': '成交量比率', 'value': 'volume_ratio'}
        ],
        value=[],
        style={'color': '#eee'}
    )
], style={'marginBottom': '20px'})
```

### 階段5：測試和驗證 (1-2天)

#### 5.1 數據質量檢查

```python
# tests/test_non_price_data.py
def test_futures_data_quality():
    """測試期貨數據質量"""
    from non_price_data.futures_data_handler import HKFuturesDataHandler
    
    handler = HKFuturesDataHandler()
    data = handler.fetch_futures_data("2024-01-01", "2024-12-31")
    
    assert len(data) > 0, "期貨數據為空"
    assert 'HSI_DAY' in data, "缺少恆指日間數據"
    assert not data['HSI_DAY'].empty, "恆指日間數據為空"
    
    # 檢查數據完整性
    assert data['HSI_DAY']['Close'].notna().sum() > 200, "有效數據點太少"
    
def test_feature_engineering():
    """測試特徵工程"""
    from data_fusion.feature_engineer import FeatureEngineer
    
    # 模擬數據
    price_data = pd.DataFrame({
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000, 10000, 100)
    }, index=pd.date_range('2024-01-01', periods=100))
    
    futures_data = {
        'HSI_DAY': pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 105,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=pd.date_range('2024-01-01', periods=100))
    }
    
    engineer = FeatureEngineer()
    features = engineer.create_futures_features(price_data, futures_data)
    
    assert 'Basis' in features.columns, "缺少基差特徵"
    assert features['Basis'].notna().sum() > 50, "基差特徵有太多NaN"
```

#### 5.2 策略回測驗證

```python
def backtest_multi_factor_strategy():
    """回測多因子策略"""
    from enhanced_strategies.multi_factor_strategies import MultiFactorStrategy
    
    # 載入數據
    synchronized_data, features = load_non_price_data()
    
    # 創建策略
    strategy = MultiFactorStrategy(
        factors=['futures_basis', 'momentum', 'volatility'],
        weights={'futures_basis': 0.4, 'momentum': 0.4, 'volatility': 0.2}
    )
    
    # 計算因子得分
    factor_scores = strategy.calculate_factor_scores(features)
    
    # 生成信號
    signals = strategy.generate_signals(factor_scores)
    
    # 計算回測表現
    returns = synchronized_data['Close'].pct_change() * signals.shift(1)
    
    # 評估指標
    annual_return = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    
    print(f"多因子策略表現:")
    print(f"年化收益率: {annual_return:.2%}")
    print(f"年化波動率: {volatility:.2%}")
    print(f"夏普比率: {sharpe_ratio:.2f}")
    
    return {
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio
    }
```

---

## 🎯 具體實施步驟

### 第1週：基礎建設

**Day 1-2: 環境準備**
```bash
# 安裝新依賴
pip install scikit-learn
pip install pandas-ta
pip install yfinance>=0.2.0

# 創建新目錄結構
mkdir non_price_data
mkdir data_fusion 
mkdir enhanced_strategies
mkdir tests
```

**Day 3-4: 期貨數據模組**
- 實施 `HKFuturesDataHandler` 類
- 測試期貨數據獲取功能
- 驗證日夜價差計算

**Day 5-7: 數據融合框架**
- 實施 `DataSynchronizer` 類
- 實施 `FeatureEngineer` 類
- 建立特徵選擇管道

### 第2週：策略開發

**Day 8-10: 多因子策略**
- 實施基礎多因子框架
- 開發期貨基差因子
- 開發動量和波動率因子

**Day 11-12: 期現套利策略**
- 實施套利信號生成邏輯
- 建立風險控制機制
- 回測驗證策略效果

**Day 13-14: 系統整合**
- 整合所有新模組
- 更新主儀表板
- 添加新的可視化功能

---

## 📊 效果評估指標

### 策略表現指標
- **夏普比率**：目標 > 2.0（相比純價格策略的1.5）
- **資訊比率**：衡量相對基準的超額收益穩定性
- **最大回撤**：控制在 < 5%
- **勝率**：目標 > 55%

### 數據質量指標
- **數據完整性**：> 95%的交易日有完整數據
- **更新及時性**：數據延遲 < 30分鐘
- **準確性**：與官方數據差異 < 0.1%

### 系統性能指標
- **處理速度**：特徵計算 < 5秒
- **記憶體使用**：< 2GB
- **穩定性**：7×24小時運行無中斷

---

## 🚀 未來擴展方向

### 1. 更多替代數據源
- **衛星數據**：停車場衛星圖像分析消費趨勢
- **社交媒體情緒**：Twitter、微博情緒指標
- **搜索趨勢**：Google Trends、百度指數
- **專利數據**：科技公司創新指標

### 2. 機器學習整合
- **特徵自動化**：AutoML特徵工程
- **深度學習**：LSTM預測期貨價差
- **強化學習**：動態倉位配置

### 3. 實時交易系統
- **API整合**：券商交易API
- **風險監控**：實時風險指標
- **自動執行**：信號自動下單

### 4. 跨市場分析
- **A股聯動**：滬深港通資金流
- **美股期貨**：VIX恐慌指數
- **商品期貨**：黃金、原油關聯性

---

## 💡 關鍵成功因素

### 技術層面
1. **數據同步精度**：確保不同數據源時間戳準確對齊
2. **特徵穩定性**：避免過擬合，建立robustness測試
3. **性能優化**：使用向量化操作和並行處理
4. **錯誤處理**：建立完善的異常捕捉機制

### 業務層面
1. **因子有效性**：定期驗證因子預測能力
2. **交易成本**：考慮實際交易手續費和衝擊成本
3. **市場環境**：適應不同市場階段（牛熊市）
4. **合規要求**：遵守香港證監會相關規定

### 運營層面
1. **監控告警**：建立數據異常和系統故障告警
2. **文檔維護**：詳細記錄所有修改和配置
3. **定期檢查**：每月進行系統健康檢查
4. **備份恢復**：建立數據和系統備份方案

---

## 📋 檢查清單

### 開發完成檢查
- [ ] 期貨數據獲取功能正常
- [ ] 日夜價差計算準確
- [ ] 數據同步無時間偏移
- [ ] 特徵工程管道運行順暢
- [ ] 多因子策略邏輯正確
- [ ] 期現套利策略可用
- [ ] 儀表板顯示非價格數據
- [ ] 所有單元測試通過
- [ ] 回測結果符合預期
- [ ] 系統性能滿足要求

### 部署前檢查
- [ ] 生產環境配置完成
- [ ] 數據源API權限獲取
- [ ] 監控系統部署
- [ ] 備份機制建立
- [ ] 文檔更新完整
- [ ] 團隊培訓完成

這個指南提供了一個完整的非價格數據整合框架，可以幫助您的系統從傳統的價格驅動策略進化為多因子、多數據源的現代量化交易平台，大幅提升策略的夏普比率和Alpha生成能力。

---

## 🌟 其他高級策略類型

### 1. 宏觀經濟策略 💹

#### 1.1 利率環境策略
```python
# enhanced_strategies/macro_strategies.py
class InterestRateStrategy:
    """利率環境策略"""
    
    def __init__(self):
        self.data_sources = {
            'fed_rate': 'FRED API - 美聯儲利率',
            'hkma_rate': 'HKMA - 香港基準利率', 
            'pboc_rate': '人民銀行 - 中國基準利率',
            'yield_curve': '美國國債收益率曲線',
            'credit_spreads': '信用利差數據'
        }
        
    def generate_signals(self, rate_data, stock_data):
        """基於利率變化生成交易信號"""
        signals = {}
        
        # 利率上升時的防禦性股票偏好
        if self._is_rate_rising(rate_data):
            signals['banking_stocks'] = 'BUY'  # 銀行股受益
            signals['reit_stocks'] = 'SELL'    # REITs受壓
            signals['growth_stocks'] = 'REDUCE' # 成長股估值壓縮
            
        # 利率下降時的成長股偏好  
        elif self._is_rate_falling(rate_data):
            signals['growth_stocks'] = 'BUY'   # 成長股受益
            signals['reit_stocks'] = 'BUY'     # REITs受益
            signals['banking_stocks'] = 'REDUCE' # 銀行股息差收窄
            
        return signals
```

#### 1.2 匯率套利策略
```python
class CurrencyArbitrageStrategy:
    """匯率套利策略"""
    
    def __init__(self):
        self.currency_pairs = {
            'USDCNY': '美元人民幣',
            'USDHKD': '美元港元',
            'EURCNY': '歐元人民幣',
            'JPYCNY': '日元人民幣'
        }
        
    def identify_arbitrage_opportunities(self, fx_data, stock_data):
        """識別匯率套利機會"""
        opportunities = []
        
        # AH股套利
        ah_premium = self._calculate_ah_premium(stock_data)
        usdcny_rate = fx_data['USDCNY']
        
        if ah_premium > 0.15:  # A股溢價超過15%
            opportunities.append({
                'type': 'AH_arbitrage',
                'action': 'LONG_H_SHORT_A',
                'expected_return': ah_premium * 0.3,
                'risk_level': 'MEDIUM'
            })
            
        return opportunities
```

### 2. 事件驅動策略 📈

#### 2.1 財報事件策略
```python
class EarningsEventStrategy:
    """財報事件策略"""
    
    def __init__(self):
        self.earnings_calendar = self._load_earnings_calendar()
        self.event_windows = {
            'pre_earnings': (-10, -1),   # 財報前10-1天
            'earnings_day': (0, 0),      # 財報當天
            'post_earnings': (1, 5)      # 財報後1-5天
        }
        
    def earnings_momentum_strategy(self, symbol, earnings_data):
        """財報動量策略"""
        signals = []
        
        # 財報前預期建倉
        if self._is_positive_guidance(earnings_data):
            signals.append({
                'symbol': symbol,
                'action': 'BUY',
                'window': 'pre_earnings',
                'confidence': 0.7,
                'reason': '正面業績指引'
            })
            
        # 財報後趨勢跟隨
        surprise_factor = self._calculate_earnings_surprise(earnings_data)
        if surprise_factor > 0.05:  # 業績超預期5%以上
            signals.append({
                'symbol': symbol,
                'action': 'HOLD_EXTEND',
                'window': 'post_earnings',
                'confidence': 0.8,
                'reason': f'業績超預期{surprise_factor:.1%}'
            })
            
        return signals
```

#### 2.2 監管政策策略
```python
class RegulatoryEventStrategy:
    """監管政策事件策略"""
    
    def __init__(self):
        self.policy_categories = {
            'fintech_regulation': ['1024.HK', '9988.HK', '700.HK'],
            'healthcare_policy': ['2269.HK', '1833.HK'],
            'property_policy': ['1109.HK', '1997.HK', '0016.HK'],
            'education_policy': ['1797.HK', '9901.HK']
        }
        
    def monitor_policy_impact(self, policy_events, sector_performance):
        """監控政策影響"""
        alerts = []
        
        for event in policy_events:
            affected_sectors = self._identify_affected_sectors(event)
            
            for sector in affected_sectors:
                impact_score = self._calculate_policy_impact(event, sector)
                
                if abs(impact_score) > 0.1:  # 影響超過10%
                    alerts.append({
                        'event': event['title'],
                        'sector': sector,
                        'impact_score': impact_score,
                        'recommended_action': 'REDUCE' if impact_score < 0 else 'INCREASE',
                        'affected_stocks': self.policy_categories.get(sector, [])
                    })
                    
        return alerts
```

### 3. 跨市場套利策略 🌐

#### 3.1 AH股套利策略  
```python
class AHStockArbitrageStrategy:
    """A股H股套利策略"""
    
    def __init__(self):
        self.ah_pairs = {
            '939.HK': '601939.SS',    # 建設銀行
            '1398.HK': '601398.SS',   # 工商銀行
            '3988.HK': '601988.SS',   # 中國銀行
            '2318.HK': '601318.SS',   # 中國平安
            '1109.HK': '001209.SZ'    # 華潤置地
        }
        
    def calculate_ah_premium(self, h_price, a_price, fx_rate):
        """計算AH股溢價"""
        # 轉換A股價格到港幣
        a_price_hkd = a_price * fx_rate
        premium = (h_price - a_price_hkd) / a_price_hkd
        return premium
        
    def generate_arbitrage_signals(self, price_data, fx_data):
        """生成套利信號"""
        signals = []
        
        for h_symbol, a_symbol in self.ah_pairs.items():
            premium = self.calculate_ah_premium(
                price_data[h_symbol]['close'],
                price_data[a_symbol]['close'], 
                fx_data['CNYHHK']
            )
            
            # 套利閾值
            if premium > 0.15:  # H股溢價超過15%
                signals.append({
                    'type': 'AH_ARBITRAGE',
                    'long_symbol': a_symbol,
                    'short_symbol': h_symbol,
                    'premium': premium,
                    'expected_return': premium * 0.5,
                    'risk': 'MEDIUM'
                })
                
        return signals
```

### 4. 高頻量化策略 ⚡

#### 4.1 Level-2數據策略
```python
class Level2DataStrategy:
    """基於Level-2數據的高頻策略"""
    
    def __init__(self):
        self.tick_size = 0.01  # 港股最小價格變動
        self.order_book_levels = 10
        
    def order_book_imbalance_strategy(self, order_book_data):
        """訂單簿失衡策略"""
        signals = []
        
        # 計算買賣盤失衡
        bid_volume = sum(order_book_data['bids']['volume'][:5])
        ask_volume = sum(order_book_data['asks']['volume'][:5]) 
        
        imbalance_ratio = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        
        if imbalance_ratio > 0.3:  # 買盤明顯強於賣盤
            signals.append({
                'action': 'BUY',
                'strategy': 'order_imbalance',
                'confidence': min(0.9, imbalance_ratio),
                'hold_period': '1-5min'
            })
            
        elif imbalance_ratio < -0.3:  # 賣盤明顯強於買盤
            signals.append({
                'action': 'SELL',
                'strategy': 'order_imbalance', 
                'confidence': min(0.9, abs(imbalance_ratio)),
                'hold_period': '1-5min'
            })
            
        return signals
```

### 5. 另類數據策略 🛰️

#### 5.1 衛星數據策略
```python
class SatelliteDataStrategy:
    """衛星數據策略"""
    
    def __init__(self):
        self.satellite_indicators = {
            'parking_lots': '購物中心停車場衛星圖像',
            'factory_activity': '工廠活動熱力圖',
            'port_traffic': '港口貨櫃流量',
            'construction_sites': '建築工地活動水平'
        }
        
    def retail_traffic_analysis(self, satellite_data, retail_stocks):
        """零售流量分析"""
        signals = []
        
        # 分析購物中心人流
        for location in satellite_data['shopping_centers']:
            traffic_trend = self._calculate_traffic_trend(location['images'])
            
            if traffic_trend > 0.2:  # 人流增長20%以上
                # 找到相關零售股
                related_stocks = self._map_location_to_stocks(location, retail_stocks)
                
                for stock in related_stocks:
                    signals.append({
                        'symbol': stock,
                        'action': 'BUY',
                        'data_source': 'satellite_traffic',
                        'confidence': 0.6,
                        'reason': f'{location["name"]}人流增長{traffic_trend:.1%}'
                    })
                    
        return signals
```

#### 5.2 社交媒體情緒策略
```python
class SocialSentimentStrategy:
    """社交媒體情緒策略"""
    
    def __init__(self):
        self.sentiment_sources = {
            'weibo': '微博情緒指數',
            'wechat': '微信公眾號文章情緒',
            'reddit': 'Reddit討論情緒',
            'news': '財經新聞情緒'
        }
        
    def sentiment_momentum_strategy(self, sentiment_data, stock_symbols):
        """情緒動量策略"""
        signals = []
        
        for symbol in stock_symbols:
            # 獲取該股票的情緒數據
            stock_sentiment = sentiment_data.get(symbol, {})
            
            # 計算情緒變化趨勢
            sentiment_score = self._calculate_sentiment_score(stock_sentiment)
            sentiment_momentum = self._calculate_sentiment_momentum(stock_sentiment)
            
            # 生成交易信號
            if sentiment_score > 0.7 and sentiment_momentum > 0.3:
                signals.append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'strategy': 'sentiment_momentum',
                    'confidence': sentiment_score,
                    'data_source': 'social_media',
                    'reason': f'正面情緒且動量強勁(分數:{sentiment_score:.2f})'
                })
                
        return signals
```

### 6. ESG主題策略 🌱

#### 6.1 綠色金融策略
```python
class ESGStrategy:
    """ESG主題投資策略"""
    
    def __init__(self):
        self.esg_factors = {
            'carbon_emissions': '碳排放數據',
            'renewable_energy': '可再生能源使用比例', 
            'board_diversity': '董事會多元化',
            'worker_safety': '工人安全記錄'
        }
        
    def green_finance_screening(self, esg_data, stock_universe):
        """綠色金融篩選"""
        green_portfolio = []
        
        for symbol in stock_universe:
            esg_score = esg_data.get(symbol, {})
            
            # ESG評分標準
            if (esg_score.get('environmental', 0) > 70 and
                esg_score.get('social', 0) > 60 and
                esg_score.get('governance', 0) > 65):
                
                green_portfolio.append({
                    'symbol': symbol,
                    'esg_score': esg_score,
                    'green_rank': self._calculate_green_rank(esg_score),
                    'recommended_weight': self._calculate_esg_weight(esg_score)
                })
                
        return sorted(green_portfolio, key=lambda x: x['green_rank'], reverse=True)
```

### 7. 數據獲取與整合方案 📡

#### 7.1 數據源對應表
```python
DATA_SOURCE_MAPPING = {
    # 基礎市場數據
    'stock_prices': 'Yahoo Finance / Wind / Bloomberg',
    'futures_data': 'HKEX / CME / Wind API',
    
    # 宏觀經濟數據  
    'interest_rates': 'FRED API / HKMA / PBOC',
    'fx_rates': 'Yahoo Finance / OANDA API',
    'economic_indicators': 'FRED / Wind / Choice',
    
    # 企業基本面數據
    'earnings_data': 'Wind / Bloomberg / FactSet',
    'financial_statements': 'HKEX披露易 / Wind',
    'analyst_estimates': 'I/B/E/S / FactSet',
    
    # 另類數據
    'satellite_data': 'Planet Labs / Maxar / RS Metrics',
    'social_sentiment': 'Twitter API / 微博API / 自建爬蟲',
    'patent_data': 'USPTO / CNIPA / 智慧芽',
    'supply_chain': 'FactSet供應鏈 / Wind產業鏈',
    
    # ESG數據
    'esg_scores': 'MSCI ESG / Sustainalytics / Wind ESG',
    'carbon_data': 'CDP / 企業ESG報告',
    
    # 高頻數據
    'level2_data': 'HKEX Market Data / Wind實時',
    'options_data': 'HKEX / Wind期權數據'
}
```

#### 7.2 數據更新調度
```python
# data_scheduler/update_scheduler.py
class DataUpdateScheduler:
    """數據更新調度器"""
    
    def __init__(self):
        self.update_schedule = {
            'high_frequency': {
                'interval': '1min',
                'sources': ['level2_data', 'futures_tick'],
                'active_hours': '09:30-16:00'
            },
            'daily': {
                'interval': '1day',
                'sources': ['stock_prices', 'economic_indicators'],
                'update_time': '18:00'
            },
            'weekly': {
                'interval': '1week', 
                'sources': ['satellite_data', 'social_sentiment'],
                'update_day': 'Sunday'
            },
            'monthly': {
                'interval': '1month',
                'sources': ['esg_scores', 'patent_data'],
                'update_day': 1
            }
        }
```

### 8. 策略組合優化 🎯

#### 8.1 多策略組合
```python
class StrategyPortfolio:
    """多策略組合管理"""
    
    def __init__(self):
        self.strategies = {
            'momentum': {'weight': 0.25, 'target_vol': 0.15},
            'mean_reversion': {'weight': 0.20, 'target_vol': 0.12},
            'futures_arbitrage': {'weight': 0.20, 'target_vol': 0.08},
            'sentiment_driven': {'weight': 0.15, 'target_vol': 0.18},
            'macro_rotation': {'weight': 0.20, 'target_vol': 0.14}
        }
        
    def optimize_weights(self, strategy_returns, target_sharpe=2.0):
        """優化策略權重"""
        from scipy.optimize import minimize
        
        def objective(weights):
            portfolio_return = np.sum(strategy_returns.mean() * weights) * 252
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(strategy_returns.cov() * 252, weights)))
            sharpe = portfolio_return / portfolio_vol
            return -(sharpe - target_sharpe)**2  # 最大化目標夏普比率
            
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 權重和為1
            {'type': 'ineq', 'fun': lambda x: x}  # 非負權重
        ]
        
        result = minimize(objective, 
                         x0=np.array(list(self.strategies.values())),
                         method='SLSQP',
                         constraints=constraints)
        
        return result.x
```

#### 8.2 動態倉位管理
```python
class DynamicPositionManager:
    """動態倉位管理"""
    
    def __init__(self, max_leverage=2.0, risk_budget=0.02):
        self.max_leverage = max_leverage
        self.risk_budget = risk_budget  # 每日VaR限制
        
    def calculate_position_size(self, signal_strength, volatility, correlation_matrix):
        """計算最優倉位大小"""
        # Kelly公式變形
        win_rate = self._estimate_win_rate(signal_strength)
        avg_win = self._estimate_avg_return(signal_strength, 'win')
        avg_loss = self._estimate_avg_return(signal_strength, 'loss')
        
        if avg_loss != 0:
            kelly_fraction = win_rate - (1 - win_rate) * avg_win / abs(avg_loss)
        else:
            kelly_fraction = 0
            
        # 考慮波動率調整
        vol_adjusted_size = kelly_fraction / volatility
        
        # 考慮組合相關性
        diversification_factor = self._calculate_diversification_factor(correlation_matrix)
        
        final_size = min(vol_adjusted_size * diversification_factor, self.max_leverage)
        
        return max(0, final_size)  # 確保非負
```

這個擴展的指南涵蓋了從基礎數據下載到高級策略開發的完整方案，不僅包括期貨數據，還提供了宏觀經濟、事件驅動、跨市場套利、高頻交易、另類數據和ESG等多種策略類型的具體實現方法，能夠全面提升您的量化交易系統的Alpha生成能力。