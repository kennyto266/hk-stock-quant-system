"""
市場情緒分析模組
分析新聞、期權數據等來評估市場情緒
"""

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

class MarketSentimentAnalyzer:
    """市場情緒分析器"""
    
    def __init__(self, data_dir: str = "data_output/sentiment"):
        from pathlib import Path
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def calculate_vix_sentiment(self, vix_data: pd.DataFrame) -> pd.DataFrame:
        """計算VIX恐慌指數情緒"""
        try:
            sentiment_data = vix_data.copy()
            
            # VIX情緒分級
            sentiment_data['VIX_Level'] = pd.cut(sentiment_data['Close'], 
                                               bins=[0, 15, 20, 30, 100],
                                               labels=['極度樂觀', '樂觀', '謹慎', '恐慌'])
            
            # VIX變化率
            sentiment_data['VIX_Change'] = sentiment_data['Close'].pct_change()
            
            # VIX移動平均
            sentiment_data['VIX_MA5'] = sentiment_data['Close'].rolling(5).mean()
            sentiment_data['VIX_MA20'] = sentiment_data['Close'].rolling(20).mean()
            
            # VIX相對位置
            vix_min_20d = sentiment_data['Close'].rolling(20).min()
            vix_max_20d = sentiment_data['Close'].rolling(20).max()
            sentiment_data['VIX_Relative_Position'] = (
                (sentiment_data['Close'] - vix_min_20d) / (vix_max_20d - vix_min_20d)
            )
            
            # 情緒評分 (0-100, 100為最恐慌)
            sentiment_data['Fear_Score'] = np.clip(
                (sentiment_data['Close'] - 10) / 20 * 100, 0, 100
            )
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"VIX情緒計算失敗: {e}")
            return vix_data
    
    def analyze_options_sentiment(self, options_data: pd.DataFrame = None) -> pd.DataFrame:
        """分析期權市場情緒（模擬數據）"""
        try:
            # 由於期權數據較難獲取，這裡創建模擬的市場情緒指標
            if options_data is None:
                # 創建模擬期權情緒數據
                dates = pd.date_range(start='2023-01-01', end=datetime.now(), freq='D')
                
                # 模擬看跌看漲比率
                np.random.seed(42)
                put_call_ratio = 0.8 + 0.4 * np.random.randn(len(dates))
                put_call_ratio = np.clip(put_call_ratio, 0.3, 2.0)
                
                options_data = pd.DataFrame({
                    'Put_Call_Ratio': put_call_ratio,
                    'Options_Volume': np.random.lognormal(10, 0.5, len(dates)),
                    'Implied_Volatility': 0.2 + 0.1 * np.random.randn(len(dates))
                }, index=dates)
            
            # 期權情緒分析
            sentiment_data = options_data.copy()
            
            # 看跌看漲比率情緒
            sentiment_data['PCR_Sentiment'] = pd.cut(sentiment_data['Put_Call_Ratio'],
                                                    bins=[0, 0.7, 1.0, 1.3, 10],
                                                    labels=['極度樂觀', '樂觀', '中性', '悲觀'])
            
            # 隱含波動率情緒
            iv_mean = sentiment_data['Implied_Volatility'].rolling(20).mean()
            iv_std = sentiment_data['Implied_Volatility'].rolling(20).std()
            sentiment_data['IV_ZScore'] = (sentiment_data['Implied_Volatility'] - iv_mean) / iv_std
            
            # 期權成交量情緒
            volume_ma = sentiment_data['Options_Volume'].rolling(20).mean()
            sentiment_data['Volume_Relative'] = sentiment_data['Options_Volume'] / volume_ma
            
            # 綜合期權情緒評分
            sentiment_data['Options_Sentiment_Score'] = (
                (sentiment_data['Put_Call_Ratio'] - 0.8) * 30 +  # PCR影響
                sentiment_data['IV_ZScore'] * 20 +                # IV影響  
                (sentiment_data['Volume_Relative'] - 1) * 10      # 成交量影響
            )
            
            # 標準化到0-100
            score_min = sentiment_data['Options_Sentiment_Score'].rolling(60).min()
            score_max = sentiment_data['Options_Sentiment_Score'].rolling(60).max()
            sentiment_data['Normalized_Sentiment'] = (
                (sentiment_data['Options_Sentiment_Score'] - score_min) / 
                (score_max - score_min) * 100
            )
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"期權情緒分析失敗: {e}")
            return pd.DataFrame()
    
    def calculate_technical_sentiment(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """基於技術指標計算情緒"""
        try:
            sentiment_data = stock_data.copy()
            
            # RSI情緒
            delta = sentiment_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            sentiment_data['RSI'] = 100 - (100 / (1 + rs))
            
            # RSI情緒分級
            sentiment_data['RSI_Sentiment'] = pd.cut(sentiment_data['RSI'],
                                                   bins=[0, 30, 40, 60, 70, 100],
                                                   labels=['極度超賣', '超賣', '中性', '超買', '極度超買'])
            
            # 布林帶位置情緒
            bb_middle = sentiment_data['Close'].rolling(20).mean()
            bb_std = sentiment_data['Close'].rolling(20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            
            sentiment_data['BB_Position'] = (
                (sentiment_data['Close'] - bb_lower) / (bb_upper - bb_lower)
            )
            
            # 成交量情緒
            volume_ma = sentiment_data['Volume'].rolling(20).mean()
            sentiment_data['Volume_Sentiment'] = sentiment_data['Volume'] / volume_ma
            
            # 價格動量情緒
            sentiment_data['Price_Momentum'] = sentiment_data['Close'].pct_change(5)
            
            # 綜合技術情緒評分
            # RSI貢獻 (30分)
            rsi_score = np.where(sentiment_data['RSI'] < 30, 100,
                        np.where(sentiment_data['RSI'] < 40, 75,
                        np.where(sentiment_data['RSI'] < 60, 50,
                        np.where(sentiment_data['RSI'] < 70, 25, 0))))
            
            # 布林帶位置貢獻 (25分)
            bb_score = sentiment_data['BB_Position'] * 100
            
            # 成交量貢獻 (25分)
            volume_score = np.clip((sentiment_data['Volume_Sentiment'] - 0.5) * 50 + 50, 0, 100)
            
            # 動量貢獻 (20分)
            momentum_score = np.clip((sentiment_data['Price_Momentum'] + 0.05) * 1000, 0, 100)
            
            sentiment_data['Technical_Sentiment_Score'] = (
                rsi_score * 0.3 + bb_score * 0.25 + 
                volume_score * 0.25 + momentum_score * 0.2
            )
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"技術情緒計算失敗: {e}")
            return stock_data
    
    def create_composite_sentiment(self, 
                                 stock_data: pd.DataFrame,
                                 vix_data: pd.DataFrame = None,
                                 options_data: pd.DataFrame = None) -> pd.DataFrame:
        """創建綜合情緒指標"""
        try:
            # 基礎技術情緒
            sentiment_df = self.calculate_technical_sentiment(stock_data)
            
            # 如果有VIX數據，添加恐慌指數
            if vix_data is not None and not vix_data.empty:
                vix_sentiment = self.calculate_vix_sentiment(vix_data)
                # 對齊日期並合併
                aligned_vix = vix_sentiment.reindex(sentiment_df.index, method='ffill')
                sentiment_df['Fear_Score'] = aligned_vix['Fear_Score']
            else:
                # 模擬VIX情緒
                sentiment_df['Fear_Score'] = 30 + 20 * np.random.randn(len(sentiment_df))
                sentiment_df['Fear_Score'] = np.clip(sentiment_df['Fear_Score'], 0, 100)
            
            # 如果有期權數據，添加期權情緒
            if options_data is not None and not options_data.empty:
                options_sentiment = self.analyze_options_sentiment(options_data)
                aligned_options = options_sentiment.reindex(sentiment_df.index, method='ffill')
                sentiment_df['Options_Sentiment'] = aligned_options['Normalized_Sentiment']
            else:
                # 模擬期權情緒
                options_sentiment = self.analyze_options_sentiment()
                aligned_options = options_sentiment.reindex(sentiment_df.index, method='ffill')
                sentiment_df['Options_Sentiment'] = aligned_options['Normalized_Sentiment']
            
            # 計算綜合情緒評分
            sentiment_df['Composite_Sentiment'] = (
                sentiment_df['Technical_Sentiment_Score'] * 0.4 +  # 技術指標 40%
                (100 - sentiment_df['Fear_Score']) * 0.35 +       # VIX恐慌指數 35%
                sentiment_df['Options_Sentiment'] * 0.25          # 期權情緒 25%
            )
            
            # 情緒分級
            sentiment_df['Sentiment_Level'] = pd.cut(sentiment_df['Composite_Sentiment'],
                                                   bins=[0, 20, 40, 60, 80, 100],
                                                   labels=['極度悲觀', '悲觀', '中性', '樂觀', '極度樂觀'])
            
            # 情緒趨勢
            sentiment_df['Sentiment_Trend'] = sentiment_df['Composite_Sentiment'].rolling(5).mean()
            sentiment_df['Sentiment_Change'] = sentiment_df['Composite_Sentiment'].diff()
            
            # 保存情緒數據
            filename = f"market_sentiment_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = self.data_dir / filename
            sentiment_df.to_csv(filepath)
            logger.info(f"市場情緒數據已保存到: {filepath}")
            
            return sentiment_df
            
        except Exception as e:
            logger.error(f"綜合情緒計算失敗: {e}")
            return stock_data
    
    def generate_sentiment_signals(self, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """生成基於情緒的交易信號"""
        try:
            signals_df = sentiment_data.copy()
            
            # 情緒極值信號
            signals_df['Extreme_Pessimism_Signal'] = (
                sentiment_data['Composite_Sentiment'] < 20
            ).astype(int)
            
            signals_df['Extreme_Optimism_Signal'] = (
                sentiment_data['Composite_Sentiment'] > 80
            ).astype(int)
            
            # 情緒反轉信號
            sentiment_ma5 = sentiment_data['Composite_Sentiment'].rolling(5).mean()
            sentiment_ma20 = sentiment_data['Composite_Sentiment'].rolling(20).mean()
            
            signals_df['Bullish_Sentiment_Cross'] = (
                (sentiment_ma5 > sentiment_ma20) & 
                (sentiment_ma5.shift(1) <= sentiment_ma20.shift(1))
            ).astype(int)
            
            signals_df['Bearish_Sentiment_Cross'] = (
                (sentiment_ma5 < sentiment_ma20) & 
                (sentiment_ma5.shift(1) >= sentiment_ma20.shift(1))
            ).astype(int)
            
            # 情緒動量信號
            signals_df['Sentiment_Momentum_Up'] = (
                sentiment_data['Sentiment_Change'] > 5
            ).astype(int)
            
            signals_df['Sentiment_Momentum_Down'] = (
                sentiment_data['Sentiment_Change'] < -5
            ).astype(int)
            
            # 綜合情緒交易信號
            signals_df['Buy_Signal'] = (
                signals_df['Extreme_Pessimism_Signal'] | 
                signals_df['Bullish_Sentiment_Cross'] |
                signals_df['Sentiment_Momentum_Up']
            ).astype(int)
            
            signals_df['Sell_Signal'] = (
                signals_df['Extreme_Optimism_Signal'] | 
                signals_df['Bearish_Sentiment_Cross'] |
                signals_df['Sentiment_Momentum_Down']
            ).astype(int)
            
            return signals_df
            
        except Exception as e:
            logger.error(f"情緒信號生成失敗: {e}")
            return sentiment_data
    
    def analyze_sentiment_effectiveness(self, 
                                     sentiment_signals: pd.DataFrame,
                                     forward_returns: pd.Series,
                                     holding_days: int = 5) -> Dict:
        """分析情緒信號的有效性"""
        try:
            analysis_results = {}
            
            # 買入信號分析
            buy_signals = sentiment_signals['Buy_Signal'] == 1
            if buy_signals.sum() > 0:
                buy_returns = []
                for signal_date in sentiment_signals.index[buy_signals]:
                    try:
                        future_date = signal_date + timedelta(days=holding_days)
                        if future_date in forward_returns.index:
                            ret = forward_returns.loc[future_date]
                            buy_returns.append(ret)
                    except:
                        continue
                
                if buy_returns:
                    analysis_results['buy_signal_count'] = len(buy_returns)
                    analysis_results['buy_avg_return'] = np.mean(buy_returns)
                    analysis_results['buy_win_rate'] = np.mean(np.array(buy_returns) > 0)
                    analysis_results['buy_sharpe'] = np.mean(buy_returns) / np.std(buy_returns) if np.std(buy_returns) > 0 else 0
            
            # 賣出信號分析
            sell_signals = sentiment_signals['Sell_Signal'] == 1
            if sell_signals.sum() > 0:
                sell_returns = []
                for signal_date in sentiment_signals.index[sell_signals]:
                    try:
                        future_date = signal_date + timedelta(days=holding_days)
                        if future_date in forward_returns.index:
                            ret = -forward_returns.loc[future_date]  # 做空收益
                            sell_returns.append(ret)
                    except:
                        continue
                
                if sell_returns:
                    analysis_results['sell_signal_count'] = len(sell_returns)
                    analysis_results['sell_avg_return'] = np.mean(sell_returns)
                    analysis_results['sell_win_rate'] = np.mean(np.array(sell_returns) > 0)
                    analysis_results['sell_sharpe'] = np.mean(sell_returns) / np.std(sell_returns) if np.std(sell_returns) > 0 else 0
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"情緒信號有效性分析失敗: {e}")
            return {}

def main():
    """測試市場情緒分析器"""
    logger.info("🚀 測試市場情緒分析器...")
    
    analyzer = MarketSentimentAnalyzer()
    
    # 創建模擬股票數據
    dates = pd.date_range(start='2023-01-01', end=datetime.now(), freq='D')
    np.random.seed(42)
    
    stock_data = pd.DataFrame({
        'Close': 100 * np.exp(np.cumsum(0.001 + 0.02 * np.random.randn(len(dates)))),
        'Volume': np.random.lognormal(15, 0.5, len(dates))
    }, index=dates)
    
    # 計算綜合情緒
    sentiment_data = analyzer.create_composite_sentiment(stock_data)
    
    if not sentiment_data.empty:
        logger.info(f"✅ 情緒分析完成，數據維度: {sentiment_data.shape}")
        
        # 生成交易信號
        signals = analyzer.generate_sentiment_signals(sentiment_data)
        
        buy_signals = signals['Buy_Signal'].sum()
        sell_signals = signals['Sell_Signal'].sum()
        
        logger.info(f"生成買入信號: {buy_signals} 個")
        logger.info(f"生成賣出信號: {sell_signals} 個")
        
        # 顯示最近的情緒狀態
        recent_sentiment = sentiment_data.tail(5)[['Composite_Sentiment', 'Sentiment_Level']]
        logger.info("最近5天的市場情緒:")
        print(recent_sentiment)
        
    else:
        logger.error("❌ 情緒分析失敗")

if __name__ == "__main__":
    main()