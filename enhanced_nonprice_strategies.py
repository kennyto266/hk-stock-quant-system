"""
增強的NonPrice數據策略
整合股票、期貨、情緒分析等多維度數據的量化策略
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_downloader import HKStockDataDownloader
from data_downloader.futures_data_downloader import HKFuturesDataDownloader
from data_downloader.sentiment_analyzer import MarketSentimentAnalyzer
from config import logger, NON_PRICE_CONFIG, RISK_CONFIG
from data_handler import TechnicalIndicators

class NonPriceEnhancedStrategy:
    """增強的NonPrice數據策略基類"""
    
    def __init__(self, symbol: str = "2800.HK"):
        self.symbol = symbol
        self.stock_downloader = HKStockDataDownloader()
        self.futures_downloader = HKFuturesDataDownloader()
        self.sentiment_analyzer = MarketSentimentAnalyzer()
        
        # 策略參數
        self.lookback_days = NON_PRICE_CONFIG.default_lookback_days
        self.correlation_threshold = NON_PRICE_CONFIG.correlation_threshold
        
        # 風險參數
        self.stop_loss_pct = RISK_CONFIG.stop_loss_pct
        self.take_profit_pct = RISK_CONFIG.take_profit_pct
        self.max_position_size = RISK_CONFIG.max_position_size
        
    def load_multi_asset_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """載入多資產數據"""
        try:
            logger.info(f"載入多資產數據: {start_date} 至 {end_date}")
            
            all_data = {}
            
            # 載入主要股票數據
            stock_data = self.stock_downloader.download_single_stock(
                self.symbol, start_date, end_date, save_csv=False
            )
            if stock_data is not None:
                all_data['stock'] = stock_data
                logger.info(f"✅ 股票數據載入成功: {stock_data.shape}")
            
            # 載入相關ETF數據作為期貨替代
            etf_symbols = ['2800.HK', '2828.HK', '3067.HK']  # 盈富、國企、科技ETF
            for etf_symbol in etf_symbols:
                if etf_symbol != self.symbol:
                    etf_data = self.stock_downloader.download_single_stock(
                        etf_symbol, start_date, end_date, save_csv=False
                    )
                    if etf_data is not None:
                        all_data[f'etf_{etf_symbol}'] = etf_data
            
            # 生成市場情緒數據
            if 'stock' in all_data:
                sentiment_data = self.sentiment_analyzer.create_composite_sentiment(all_data['stock'])
                if not sentiment_data.empty:
                    all_data['sentiment'] = sentiment_data
                    logger.info(f"✅ 情緒數據生成成功: {sentiment_data.shape}")
            
            logger.info(f"總共載入 {len(all_data)} 個數據源")
            return all_data
            
        except Exception as e:
            logger.error(f"載入多資產數據失敗: {e}")
            return {}
    
    def calculate_cross_asset_signals(self, all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """計算跨資產信號"""
        try:
            if 'stock' not in all_data:
                logger.error("缺少股票數據，無法計算跨資產信號")
                return pd.DataFrame()
                
            stock_data = all_data['stock'].copy()
            signals_df = pd.DataFrame(index=stock_data.index)
            
            # 基礎股票信號
            signals_df['Close'] = stock_data['Close']
            signals_df['Volume'] = stock_data['Volume']
            
            # 技術指標信號
            signals_df = self._add_technical_signals(signals_df)
            
            # 情緒信號
            if 'sentiment' in all_data:
                signals_df = self._add_sentiment_signals(signals_df, all_data['sentiment'])
            
            # 跨資產相關性信號
            signals_df = self._add_correlation_signals(signals_df, all_data)
            
            # 成交量信號
            signals_df = self._add_volume_signals(signals_df)
            
            # 綜合信號
            signals_df = self._generate_composite_signals(signals_df)
            
            return signals_df
            
        except Exception as e:
            logger.error(f"計算跨資產信號失敗: {e}")
            return pd.DataFrame()
    
    def _add_technical_signals(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """添加技術指標信號"""
        try:
            # RSI信號
            signals_df['RSI'] = TechnicalIndicators.calculate_rsi(signals_df['Close'])
            
            signals_df['RSI_Buy'] = (signals_df['RSI'] < 30).astype(int)
            signals_df['RSI_Sell'] = (signals_df['RSI'] > 70).astype(int)
            
            # 移動平均信號
            signals_df['MA5'] = TechnicalIndicators.calculate_sma(signals_df['Close'], 5)
            signals_df['MA20'] = TechnicalIndicators.calculate_sma(signals_df['Close'], 20)
            signals_df['MA60'] = TechnicalIndicators.calculate_sma(signals_df['Close'], 60)
            
            signals_df['MA_Bull'] = (
                (signals_df['MA5'] > signals_df['MA20']) & 
                (signals_df['MA20'] > signals_df['MA60'])
            ).astype(int)
            
            signals_df['MA_Bear'] = (
                (signals_df['MA5'] < signals_df['MA20']) & 
                (signals_df['MA20'] < signals_df['MA60'])
            ).astype(int)
            
            # MACD信號
            signals_df['MACD'] = TechnicalIndicators.calculate_macd(signals_df['Close'])
            signals_df['MACD_Signal'] = TechnicalIndicators.calculate_macd_signal(signals_df['Close'])
            signals_df['MACD_Histogram'] = TechnicalIndicators.calculate_macd_histogram(signals_df['MACD'], signals_df['MACD_Signal'])
            
            signals_df['MACD_Buy'] = (
                (signals_df['MACD'] > signals_df['MACD_Signal']) & 
                (signals_df['MACD'].shift(1) <= signals_df['MACD_Signal'].shift(1))
            ).astype(int)
            
            signals_df['MACD_Sell'] = (
                (signals_df['MACD'] < signals_df['MACD_Signal']) & 
                (signals_df['MACD'].shift(1) >= signals_df['MACD_Signal'].shift(1))
            ).astype(int)
            
            return signals_df
            
        except Exception as e:
            logger.error(f"添加技術信號失敗: {e}")
            return signals_df
    
    def _add_sentiment_signals(self, signals_df: pd.DataFrame, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """添加情緒信號"""
        try:
            # 對齊情緒數據
            aligned_sentiment = sentiment_data.reindex(signals_df.index, method='ffill')
            
            if 'Composite_Sentiment' in aligned_sentiment.columns:
                signals_df['Sentiment_Score'] = aligned_sentiment['Composite_Sentiment']
                
                # 情緒極值信號
                signals_df['Sentiment_Extreme_Bull'] = (signals_df['Sentiment_Score'] < 20).astype(int)
                signals_df['Sentiment_Extreme_Bear'] = (signals_df['Sentiment_Score'] > 80).astype(int)
                
                # 情緒趨勢信號
                sentiment_ma5 = signals_df['Sentiment_Score'].rolling(5).mean()
                sentiment_ma20 = signals_df['Sentiment_Score'].rolling(20).mean()
                
                signals_df['Sentiment_Trend_Bull'] = (sentiment_ma5 > sentiment_ma20).astype(int)
                signals_df['Sentiment_Trend_Bear'] = (sentiment_ma5 < sentiment_ma20).astype(int)
            
            return signals_df
            
        except Exception as e:
            logger.error(f"添加情緒信號失敗: {e}")
            return signals_df
    
    def _add_correlation_signals(self, signals_df: pd.DataFrame, all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """添加跨資產相關性信號"""
        try:
            main_close = signals_df['Close']
            correlation_signals = []
            
            # 計算與其他ETF的相關性
            for key, data in all_data.items():
                if key.startswith('etf_') and 'Close' in data.columns:
                    other_close = data['Close'].reindex(main_close.index, method='ffill')
                    
                    # 滾動相關性
                    rolling_corr = main_close.rolling(20).corr(other_close)
                    correlation_signals.append(rolling_corr)
            
            if correlation_signals:
                # 平均相關性
                avg_correlation = pd.concat(correlation_signals, axis=1).mean(axis=1)
                signals_df['Avg_Correlation'] = avg_correlation
                
                # 相關性分化信號
                signals_df['Correlation_Divergence'] = (
                    avg_correlation < self.correlation_threshold
                ).astype(int)
                
                # 相關性趨勢
                signals_df['Correlation_Trend'] = avg_correlation.rolling(10).mean()
            
            return signals_df
            
        except Exception as e:
            logger.error(f"添加相關性信號失敗: {e}")
            return signals_df
    
    def _add_volume_signals(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """添加成交量信號"""
        try:
            # 成交量移動平均
            signals_df['Volume_MA20'] = signals_df['Volume'].rolling(20).mean()
            signals_df['Volume_Ratio'] = signals_df['Volume'] / signals_df['Volume_MA20']
            
            # 異常成交量信號
            signals_df['Volume_Spike'] = (signals_df['Volume_Ratio'] > 2.0).astype(int)
            signals_df['Volume_Dry'] = (signals_df['Volume_Ratio'] < 0.5).astype(int)
            
            # 價量背離信號
            price_change = signals_df['Close'].pct_change()
            volume_change = signals_df['Volume'].pct_change()
            
            # 價漲量減（看空信號）
            signals_df['Price_Volume_Divergence_Bear'] = (
                (price_change > 0.01) & (volume_change < -0.1)
            ).astype(int)
            
            # 價跌量增（可能底部信號）
            signals_df['Price_Volume_Divergence_Bull'] = (
                (price_change < -0.01) & (volume_change > 0.1)
            ).astype(int)
            
            return signals_df
            
        except Exception as e:
            logger.error(f"添加成交量信號失敗: {e}")
            return signals_df
    
    def _generate_composite_signals(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """生成綜合交易信號"""
        try:
            # 買入信號權重
            buy_signals = [
                ('RSI_Buy', 0.20),
                ('MA_Bull', 0.25),
                ('MACD_Buy', 0.20),
                ('Sentiment_Extreme_Bull', 0.15),
                ('Sentiment_Trend_Bull', 0.10),
                ('Price_Volume_Divergence_Bull', 0.10)
            ]
            
            # 賣出信號權重
            sell_signals = [
                ('RSI_Sell', 0.20),
                ('MA_Bear', 0.25),
                ('MACD_Sell', 0.20),
                ('Sentiment_Extreme_Bear', 0.15),
                ('Sentiment_Trend_Bear', 0.10),
                ('Price_Volume_Divergence_Bear', 0.10)
            ]
            
            # 計算綜合買入評分
            buy_score = pd.Series(0, index=signals_df.index)
            for signal_name, weight in buy_signals:
                if signal_name in signals_df.columns:
                    buy_score += signals_df[signal_name] * weight
            
            # 計算綜合賣出評分
            sell_score = pd.Series(0, index=signals_df.index)
            for signal_name, weight in sell_signals:
                if signal_name in signals_df.columns:
                    sell_score += signals_df[signal_name] * weight
            
            signals_df['Buy_Score'] = buy_score
            signals_df['Sell_Score'] = sell_score
            
            # 生成最終交易信號
            signals_df['Final_Buy_Signal'] = (buy_score > 0.6).astype(int)
            signals_df['Final_Sell_Signal'] = (sell_score > 0.6).astype(int)
            
            # 強信號（需要更高閾值）
            signals_df['Strong_Buy_Signal'] = (buy_score > 0.8).astype(int)
            signals_df['Strong_Sell_Signal'] = (sell_score > 0.8).astype(int)
            
            return signals_df
            
        except Exception as e:
            logger.error(f"生成綜合信號失敗: {e}")
            return signals_df

class NonPriceBacktester:
    """NonPrice策略回測器"""
    
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position = 0
        self.position_value = 0
        self.trades = []
        
    def backtest_strategy(self, signals_df: pd.DataFrame, 
                         signal_column: str = 'Final_Buy_Signal',
                         exit_signal_column: str = 'Final_Sell_Signal') -> Dict:
        """回測NonPrice策略"""
        try:
            logger.info(f"開始回測 NonPrice 策略...")
            
            portfolio_values = []
            positions = []
            trade_log = []
            
            for i, (date, row) in enumerate(signals_df.iterrows()):
                current_price = row['Close']
                buy_signal = row.get(signal_column, 0)
                sell_signal = row.get(exit_signal_column, 0)
                
                # 處理交易信號
                if buy_signal and self.position == 0:
                    # 買入
                    shares = int(self.current_capital * RISK_CONFIG.max_position_size / current_price)
                    if shares > 0:
                        self.position = shares
                        self.position_value = shares * current_price
                        self.current_capital -= self.position_value
                        
                        trade_log.append({
                            'date': date,
                            'action': 'buy',
                            'price': current_price,
                            'shares': shares,
                            'value': self.position_value
                        })
                
                elif sell_signal and self.position > 0:
                    # 賣出
                    sell_value = self.position * current_price
                    self.current_capital += sell_value
                    
                    trade_log.append({
                        'date': date,
                        'action': 'sell',
                        'price': current_price,
                        'shares': self.position,
                        'value': sell_value
                    })
                    
                    self.position = 0
                    self.position_value = 0
                
                # 計算組合價值
                current_position_value = self.position * current_price if self.position > 0 else 0
                total_value = self.current_capital + current_position_value
                
                portfolio_values.append(total_value)
                positions.append(self.position)
            
            # 計算績效指標
            portfolio_series = pd.Series(portfolio_values, index=signals_df.index)
            returns = portfolio_series.pct_change().dropna()
            
            performance_metrics = self._calculate_performance_metrics(
                portfolio_series, returns, trade_log
            )
            
            return {
                'portfolio_values': portfolio_series,
                'trade_log': trade_log,
                'performance_metrics': performance_metrics
            }
            
        except Exception as e:
            logger.error(f"回測失敗: {e}")
            return {}
    
    def _calculate_performance_metrics(self, 
                                     portfolio_series: pd.Series, 
                                     returns: pd.Series,
                                     trade_log: List[Dict]) -> Dict:
        """計算績效指標"""
        try:
            total_return = (portfolio_series.iloc[-1] / self.initial_capital - 1) * 100
            annualized_return = (portfolio_series.iloc[-1] / self.initial_capital) ** (252 / len(portfolio_series)) - 1
            
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0
            
            # 最大回撤
            cumulative = portfolio_series / portfolio_series.expanding().max()
            max_drawdown = (1 - cumulative.min()) * 100
            
            # 交易統計
            profitable_trades = 0
            total_trades = 0
            
            for i in range(0, len(trade_log)-1, 2):
                if i+1 < len(trade_log):
                    buy_trade = trade_log[i]
                    sell_trade = trade_log[i+1]
                    
                    if buy_trade['action'] == 'buy' and sell_trade['action'] == 'sell':
                        total_trades += 1
                        if sell_trade['value'] > buy_trade['value']:
                            profitable_trades += 1
            
            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
            
            return {
                'total_return_pct': round(total_return, 2),
                'annualized_return_pct': round(annualized_return * 100, 2),
                'volatility_pct': round(volatility * 100, 2),
                'sharpe_ratio': round(sharpe_ratio, 3),
                'max_drawdown_pct': round(max_drawdown, 2),
                'total_trades': total_trades,
                'win_rate_pct': round(win_rate, 1),
                'final_capital': round(portfolio_series.iloc[-1], 0)
            }
            
        except Exception as e:
            logger.error(f"績效指標計算失敗: {e}")
            return {}

def main():
    """主函數 - 測試NonPrice增強策略"""
    logger.info("🚀 測試 NonPrice 增強策略...")
    
    # 創建策略實例
    strategy = NonPriceEnhancedStrategy(symbol="2800.HK")
    
    # 設定時間範圍
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    
    # 載入多資產數據
    all_data = strategy.load_multi_asset_data(start_date, end_date)
    
    if all_data:
        logger.info(f"✅ 成功載入 {len(all_data)} 個數據源")
        
        # 計算跨資產信號
        signals_df = strategy.calculate_cross_asset_signals(all_data)
        
        if not signals_df.empty:
            logger.info(f"✅ 信號計算完成，數據維度: {signals_df.shape}")
            
            # 顯示信號統計
            buy_signals = signals_df['Final_Buy_Signal'].sum()
            sell_signals = signals_df['Final_Sell_Signal'].sum()
            strong_buy = signals_df['Strong_Buy_Signal'].sum()
            strong_sell = signals_df['Strong_Sell_Signal'].sum()
            
            logger.info(f"買入信號: {buy_signals} 個 (強信號: {strong_buy} 個)")
            logger.info(f"賣出信號: {sell_signals} 個 (強信號: {strong_sell} 個)")
            
            # 執行回測
            backtester = NonPriceBacktester()
            backtest_results = backtester.backtest_strategy(signals_df)
            
            if backtest_results:
                metrics = backtest_results['performance_metrics']
                logger.info("📊 回測結果:")
                logger.info(f"總收益率: {metrics['total_return_pct']}%")
                logger.info(f"年化收益率: {metrics['annualized_return_pct']}%")
                logger.info(f"夏普比率: {metrics['sharpe_ratio']}")
                logger.info(f"最大回撤: {metrics['max_drawdown_pct']}%")
                logger.info(f"勝率: {metrics['win_rate_pct']}%")
                logger.info(f"總交易次數: {metrics['total_trades']}")
                
                # 保存結果
                from pathlib import Path
                output_dir = Path("data_output/backtest_results")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # 保存信號數據
                signals_file = output_dir / f"nonprice_signals_{datetime.now().strftime('%Y%m%d')}.csv"
                signals_df.to_csv(signals_file)
                logger.info(f"信號數據已保存到: {signals_file}")
                
                # 保存績效數據
                portfolio_file = output_dir / f"nonprice_portfolio_{datetime.now().strftime('%Y%m%d')}.csv"
                backtest_results['portfolio_values'].to_csv(portfolio_file)
                logger.info(f"組合數據已保存到: {portfolio_file}")
                
                return True
            else:
                logger.error("❌ 回測失敗")
                return False
        else:
            logger.error("❌ 信號計算失敗")
            return False
    else:
        logger.error("❌ 數據載入失敗")
        return False

if __name__ == "__main__":
    main()