"""
港股量化分析系統 - 策略管理模組
包含策略組合管理、權重優化、信號生成等功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize
import warnings
from config import logger
from risk_management import RiskMetrics

warnings.filterwarnings('ignore')

class StrategyPortfolioManager:
    """策略組合管理器"""
    
    def __init__(self):
        self.strategies = {}
        self.weights = {}
        
    def add_strategy(self, name: str, signals: pd.DataFrame, weight: float = 1.0):
        """
        添加策略
        
        Args:
            name: 策略名稱
            signals: 策略信號
            weight: 策略權重
        """
        try:
            if signals.empty:
                logger.warning(f"策略 {name} 的信號為空")
                return
            
            self.strategies[name] = signals
            self.weights[name] = weight
            logger.info(f"✅ 策略 {name} 已添加，權重: {weight}")
            
        except Exception as e:
            logger.error(f"添加策略 {name} 失敗: {e}")
    
    def optimize_weights(self, strategy_returns: Dict[str, pd.Series], 
                        target: str = 'sharpe') -> Dict[str, float]:
        """
        優化策略權重
        
        Args:
            strategy_returns: 策略收益率字典
            target: 優化目標 ('sharpe', 'min_var', 'max_return')
            
        Returns:
            優化後的權重字典
        """
        try:
            if not strategy_returns:
                return {}
            
            # 構建收益率矩陣
            returns_df = pd.DataFrame(strategy_returns)
            returns_df = returns_df.dropna()
            
            if returns_df.empty:
                logger.warning("所有策略收益率數據為空")
                return {}
            
            n_strategies = len(returns_df.columns)
            
            # 定義目標函數
            def objective(weights):
                portfolio_returns = (returns_df * weights).sum(axis=1)
                
                if target == 'sharpe':
                    mean_return = portfolio_returns.mean()
                    std_return = portfolio_returns.std()
                    if std_return == 0:
                        return -float('inf')
                    return -(mean_return / std_return)  # 負號因為要最大化
                
                elif target == 'min_var':
                    return portfolio_returns.var()
                
                elif target == 'max_return':
                    return -portfolio_returns.mean()  # 負號因為要最大化
                
                else:
                    return portfolio_returns.var()
            
            # 約束條件
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # 權重和為1
            bounds = tuple((0, 1) for _ in range(n_strategies))  # 權重範圍0-1
            
            # 初始猜測（等權重）
            x0 = np.array([1.0/n_strategies] * n_strategies)
            
            # 優化
            result = minimize(objective, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                optimized_weights = dict(zip(returns_df.columns, result.x))
                logger.info(f"✅ 權重優化完成，目標: {target}")
                return optimized_weights
            else:
                logger.warning(f"權重優化失敗: {result.message}")
                # 返回等權重
                return dict(zip(returns_df.columns, [1.0/n_strategies] * n_strategies))
        
        except Exception as e:
            logger.error(f"權重優化失敗: {e}")
            # 返回等權重
            if strategy_returns:
                n = len(strategy_returns)
                return dict(zip(strategy_returns.keys(), [1.0/n] * n))
            return {}
    
    def generate_combined_signals(self, method: str = 'weighted_average') -> pd.DataFrame:
        """
        生成組合信號
        
        Args:
            method: 組合方法 ('weighted_average', 'majority_vote', 'unanimous', 'risk_parity')
            
        Returns:
            組合信號DataFrame
        """
        try:
            if not self.strategies:
                logger.warning("沒有可用的策略")
                return pd.DataFrame()
            
            # 獲取所有策略的信號
            all_signals = pd.DataFrame()
            for name, signals in self.strategies.items():
                if 'signal' in signals.columns:
                    all_signals[name] = signals['signal']
            
            if all_signals.empty:
                logger.warning("所有策略信號為空")
                return pd.DataFrame()
            
            # 對齊數據
            all_signals = all_signals.dropna()
            
            # 創建組合信號DataFrame
            combined_signals = pd.DataFrame(index=all_signals.index)
            
            if method == 'weighted_average':
                # 加權平均
                weights_series = pd.Series(self.weights)
                # 確保權重和策略對齊
                aligned_weights = weights_series.reindex(all_signals.columns).fillna(1.0)
                aligned_weights = aligned_weights / aligned_weights.sum()
                
                combined_signals['signal'] = (all_signals * aligned_weights).sum(axis=1)
                combined_signals['signal'] = np.where(combined_signals['signal'] > 0.5, 1,
                                                     np.where(combined_signals['signal'] < -0.5, -1, 0))
            
            elif method == 'majority_vote':
                # 多數投票
                vote_long = (all_signals > 0).sum(axis=1)
                vote_short = (all_signals < 0).sum(axis=1)
                combined_signals['signal'] = np.where(vote_long > vote_short, 1,
                                                     np.where(vote_short > vote_long, -1, 0))
            
            elif method == 'unanimous':
                # 一致決策
                combined_signals['signal'] = np.where((all_signals > 0).all(axis=1), 1,
                                                     np.where((all_signals < 0).all(axis=1), -1, 0))
            
            elif method == 'risk_parity':
                # 基於風險平價的權重調整
                try:
                    strategy_vols = all_signals.std()
                    # 確保 strategy_vols 是 Series 類型並處理零值
                    if isinstance(strategy_vols, pd.Series):
                        # 防止除零錯誤
                        strategy_vols = strategy_vols.replace(0, 1e-8)
                        inv_vols = 1 / strategy_vols
                        risk_weights = inv_vols / inv_vols.sum()
                    else:
                        # 如果不是 Series，使用等權重
                        risk_weights = pd.Series([1.0 / len(all_signals.columns)] * len(all_signals.columns), 
                                               index=all_signals.columns)
                    
                    combined_signals['signal'] = (all_signals * risk_weights).mean(axis=1)
                    combined_signals['signal'] = np.where(combined_signals['signal'] > 0.2, 1,
                                                         np.where(combined_signals['signal'] < -0.2, -1, 0))
                except Exception as e:
                    logger.warning(f"風險平價計算失敗，使用等權重: {e}")
                    # 使用等權重作為備用方案
                    combined_signals['signal'] = all_signals.mean(axis=1)
                    combined_signals['signal'] = np.where(combined_signals['signal'] > 0.2, 1,
                                                         np.where(combined_signals['signal'] < -0.2, -1, 0))
            
            else:
                # 默認使用加權平均
                logger.warning(f"未知的組合方法: {method}，使用加權平均")
                return self.generate_combined_signals('weighted_average')
            
            logger.info(f"✅ 組合信號生成完成，方法: {method}")
            return combined_signals
        
        except Exception as e:
            logger.error(f"組合信號生成失敗: {e}")
            return pd.DataFrame()
    
    def get_strategy_performance(self, price_data: pd.Series) -> Dict[str, Dict]:
        """
        計算各策略績效
        
        Args:
            price_data: 價格數據
            
        Returns:
            策略績效字典
        """
        try:
            performance = {}
            
            for name, signals in self.strategies.items():
                if 'signal' in signals.columns and not signals.empty:
                    strategy_performance = self._calculate_single_strategy_performance(
                        signals, price_data, name
                    )
                    performance[name] = strategy_performance
            
            return performance
        
        except Exception as e:
            logger.error(f"策略績效計算失敗: {e}")
            return {}
    
    def _calculate_single_strategy_performance(self, signals: pd.DataFrame, 
                                             price_data: pd.Series, 
                                             strategy_name: str) -> Dict:
        """
        計算單一策略績效
        
        Args:
            signals: 策略信號
            price_data: 價格數據
            strategy_name: 策略名稱
            
        Returns:
            績效指標字典
        """
        try:
            # 對齊數據
            aligned_data = pd.DataFrame({
                'price': price_data,
                'signal': signals['signal']
            }).dropna()
            
            if aligned_data.empty:
                return {'error': '數據為空'}
            
            # 計算收益率
            aligned_data['returns'] = aligned_data['price'].pct_change()
            aligned_data['strategy_returns'] = aligned_data['signal'].shift(1) * aligned_data['returns']
            
            strategy_returns = aligned_data['strategy_returns'].dropna()
            
            if strategy_returns.empty:
                return {'error': '策略收益為空'}
            
            # 績效指標
            total_return = (1 + strategy_returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
            volatility = strategy_returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility != 0 else 0
            
            # 最大回撤
            cumulative_returns = (1 + strategy_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # 交易統計
            trades = aligned_data['signal'].diff().fillna(0)
            trade_count = (trades != 0).sum()
            
            # 勝率
            winning_trades = strategy_returns[strategy_returns > 0]
            losing_trades = strategy_returns[strategy_returns < 0]
            win_rate = len(winning_trades) / len(strategy_returns) if len(strategy_returns) > 0 else 0
            
            return {
                'total_return': float(total_return * 100),
                'annual_return': float(annual_return * 100),
                'volatility': float(volatility * 100),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown * 100),
                'trade_count': int(trade_count),
                'win_rate': float(win_rate * 100),
                'avg_win': float(winning_trades.mean() * 100) if len(winning_trades) > 0 else 0,
                'avg_loss': float(losing_trades.mean() * 100) if len(losing_trades) > 0 else 0,
                'returns_series': strategy_returns
            }
        
        except Exception as e:
            logger.error(f"策略 {strategy_name} 績效計算失敗: {e}")
            return {'error': str(e)}

class SignalGenerator:
    """信號生成器"""
    
    @staticmethod
    def generate_rsi_signals(data: pd.DataFrame, rsi_period: int = 14, 
                           overbought: float = 70, oversold: float = 30) -> pd.DataFrame:
        """
        生成RSI信號
        
        Args:
            data: 股票數據
            rsi_period: RSI週期
            overbought: 超買閾值
            oversold: 超賣閾值
            
        Returns:
            RSI信號DataFrame
        """
        try:
            if 'RSI' not in data.columns:
                logger.error("數據中缺少RSI指標")
                return pd.DataFrame()
            
            signals = pd.DataFrame(index=data.index)
            rsi = data['RSI']
            
            # 生成信號
            signals['signal'] = 0
            signals.loc[rsi < oversold, 'signal'] = 1  # 買入信號
            signals.loc[rsi > overbought, 'signal'] = -1  # 賣出信號
            
            # 添加額外信息
            signals['rsi_value'] = rsi
            signals['strategy_name'] = 'RSI'
            
            return signals
        
        except Exception as e:
            logger.error(f"RSI信號生成失敗: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def generate_macd_signals(data: pd.DataFrame) -> pd.DataFrame:
        """
        生成MACD信號
        
        Args:
            data: 股票數據
            
        Returns:
            MACD信號DataFrame
        """
        try:
            required_cols = ['MACD_MACD', 'MACD_Signal']
            if not all(col in data.columns for col in required_cols):
                logger.error("數據中缺少MACD指標")
                return pd.DataFrame()
            
            signals = pd.DataFrame(index=data.index)
            macd = data['MACD_MACD']
            signal_line = data['MACD_Signal']
            
            # 生成信號
            signals['signal'] = 0
            signals.loc[macd > signal_line, 'signal'] = 1  # 買入信號
            signals.loc[macd < signal_line, 'signal'] = -1  # 賣出信號
            
            # 添加額外信息
            signals['macd_value'] = macd
            signals['signal_line'] = signal_line
            signals['strategy_name'] = 'MACD'
            
            return signals
        
        except Exception as e:
            logger.error(f"MACD信號生成失敗: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def generate_bollinger_signals(data: pd.DataFrame) -> pd.DataFrame:
        """
        生成布林帶信號
        
        Args:
            data: 股票數據
            
        Returns:
            布林帶信號DataFrame
        """
        try:
            required_cols = ['Close', 'BB_Upper', 'BB_Lower', 'BB_Middle']
            if not all(col in data.columns for col in required_cols):
                logger.error("數據中缺少布林帶指標")
                return pd.DataFrame()
            
            signals = pd.DataFrame(index=data.index)
            close = data['Close']
            upper = data['BB_Upper']
            lower = data['BB_Lower']
            middle = data['BB_Middle']
            
            # 生成信號
            signals['signal'] = 0
            signals.loc[close < lower, 'signal'] = 1  # 買入信號（價格觸及下軌）
            signals.loc[close > upper, 'signal'] = -1  # 賣出信號（價格觸及上軌）
            
            # 添加額外信息
            signals['close_price'] = close
            signals['bb_upper'] = upper
            signals['bb_lower'] = lower
            signals['bb_middle'] = middle
            signals['strategy_name'] = 'Bollinger'
            
            return signals
        
        except Exception as e:
            logger.error(f"布林帶信號生成失敗: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def generate_ma_crossover_signals(data: pd.DataFrame, 
                                     fast_period: int = 5, 
                                     slow_period: int = 20) -> pd.DataFrame:
        """
        生成移動平均交叉信號
        
        Args:
            data: 股票數據
            fast_period: 快線週期
            slow_period: 慢線週期
            
        Returns:
            移動平均交叉信號DataFrame
        """
        try:
            fast_ma_col = f'MA_{fast_period}'
            slow_ma_col = f'MA_{slow_period}'
            
            if not all(col in data.columns for col in [fast_ma_col, slow_ma_col]):
                logger.error(f"數據中缺少移動平均線 {fast_ma_col} 或 {slow_ma_col}")
                return pd.DataFrame()
            
            signals = pd.DataFrame(index=data.index)
            fast_ma = data[fast_ma_col]
            slow_ma = data[slow_ma_col]
            
            # 生成信號
            signals['signal'] = 0
            signals.loc[fast_ma > slow_ma, 'signal'] = 1  # 買入信號
            signals.loc[fast_ma < slow_ma, 'signal'] = -1  # 賣出信號
            
            # 添加額外信息
            signals['fast_ma'] = fast_ma
            signals['slow_ma'] = slow_ma
            signals['strategy_name'] = f'MA_Cross_{fast_period}_{slow_period}'
            
            return signals
        
        except Exception as e:
            logger.error(f"移動平均交叉信號生成失敗: {e}")
            return pd.DataFrame() 