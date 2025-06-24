"""
港股量化分析系統 - 風險管理模組
包含風險度量、止損止盈、組合管理等功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from config import RISK_CONFIG, logger

class RiskMetrics:
    """組合風險度量類"""
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """
        計算VaR (Value at Risk)
        
        Args:
            returns: 收益率序列
            confidence_level: 置信度水平
            
        Returns:
            VaR值
        """
        try:
            if returns.empty:
                return 0.0
            return float(np.percentile(returns.dropna(), confidence_level * 100))
        except Exception as e:
            logger.error(f"VaR計算失敗: {e}")
            return 0.0
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """
        計算CVaR (Conditional Value at Risk)
        
        Args:
            returns: 收益率序列
            confidence_level: 置信度水平
            
        Returns:
            CVaR值
        """
        try:
            if returns.empty:
                return 0.0
            var = RiskMetrics.calculate_var(returns, confidence_level)
            filtered_returns = returns[returns <= var]
            if len(filtered_returns) > 0:
                cvar = filtered_returns.mean()
                return float(cvar)
            return 0.0
        except Exception as e:
            logger.error(f"CVaR計算失敗: {e}")
            return 0.0
    
    @staticmethod
    def calculate_portfolio_correlation(strategy_returns: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        計算策略間相關性
        
        Args:
            strategy_returns: 策略收益率字典
            
        Returns:
            相關性矩陣
        """
        try:
            if not strategy_returns:
                return pd.DataFrame()
            returns_df = pd.DataFrame(strategy_returns)
            return returns_df.corr()
        except Exception as e:
            logger.error(f"相關性計算失敗: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def calculate_portfolio_risk(strategy_returns: Dict[str, pd.Series], weights: Dict[str, float]) -> Dict:
        """
        計算組合風險指標
        
        Args:
            strategy_returns: 策略收益率字典
            weights: 策略權重字典
            
        Returns:
            風險指標字典
        """
        try:
            if not strategy_returns or not weights:
                return {}
            
            returns_df = pd.DataFrame(strategy_returns)
            weights_series = pd.Series(weights)
            
            # 組合收益
            portfolio_returns = (returns_df * weights_series).sum(axis=1)
            
            # 風險指標
            portfolio_vol = portfolio_returns.std() * np.sqrt(252)
            portfolio_var = RiskMetrics.calculate_var(portfolio_returns)
            portfolio_cvar = RiskMetrics.calculate_cvar(portfolio_returns)
            
            # 最大回撤
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            return {
                'portfolio_volatility': float(portfolio_vol),
                'portfolio_var': float(portfolio_var),
                'portfolio_cvar': float(portfolio_cvar),
                'max_drawdown': float(max_drawdown),
                'portfolio_returns': portfolio_returns
            }
        except Exception as e:
            logger.error(f"組合風險計算失敗: {e}")
            return {}

class StopLossOptimizer:
    """止損止盈優化器"""
    
    @staticmethod
    def apply_stop_loss_take_profit(signals: pd.DataFrame, prices: pd.Series, 
                                   stop_loss_pct: float, take_profit_pct: float,
                                   trailing_stop: bool = False, trailing_pct: float = 0.03) -> pd.DataFrame:
        """
        應用止損止盈邏輯
        
        Args:
            signals: 原始信號DataFrame
            prices: 價格序列
            stop_loss_pct: 止損百分比
            take_profit_pct: 止盈百分比
            trailing_stop: 是否使用追蹤止損
            trailing_pct: 追蹤止損百分比
            
        Returns:
            調整後的信號DataFrame
        """
        try:
            if signals.empty or prices.empty:
                return signals
            
            adjusted_signals = signals.copy()
            position = 0
            entry_price = 0
            highest_price = 0
            
            for i in range(len(signals)):
                current_price = prices.iloc[i]
                current_signal = signals.iloc[i]['signal'] if 'signal' in signals.columns else 0
                
                # 開倉邏輯
                if position == 0 and current_signal != 0:
                    position = current_signal
                    entry_price = current_price
                    highest_price = current_price if position > 0 else current_price
                    continue
                
                # 平倉邏輯
                if position != 0:
                    # 更新最高/最低價
                    if position > 0:  # 多頭
                        highest_price = max(highest_price, current_price)
                        
                        # 止損
                        if current_price <= entry_price * (1 - stop_loss_pct):
                            adjusted_signals.iloc[i, adjusted_signals.columns.get_loc('signal')] = -position
                            position = 0
                            continue
                        
                        # 止盈
                        if current_price >= entry_price * (1 + take_profit_pct):
                            adjusted_signals.iloc[i, adjusted_signals.columns.get_loc('signal')] = -position
                            position = 0
                            continue
                        
                        # 追蹤止損
                        if trailing_stop and current_price <= highest_price * (1 - trailing_pct):
                            adjusted_signals.iloc[i, adjusted_signals.columns.get_loc('signal')] = -position
                            position = 0
                            continue
                    
                    else:  # 空頭
                        highest_price = min(highest_price, current_price)
                        
                        # 止損
                        if current_price >= entry_price * (1 + stop_loss_pct):
                            adjusted_signals.iloc[i, adjusted_signals.columns.get_loc('signal')] = -position
                            position = 0
                            continue
                        
                        # 止盈
                        if current_price <= entry_price * (1 - take_profit_pct):
                            adjusted_signals.iloc[i, adjusted_signals.columns.get_loc('signal')] = -position
                            position = 0
                            continue
                        
                        # 追蹤止損
                        if trailing_stop and current_price >= highest_price * (1 + trailing_pct):
                            adjusted_signals.iloc[i, adjusted_signals.columns.get_loc('signal')] = -position
                            position = 0
                            continue
                
                # 原始信號反向平倉
                if position != 0 and current_signal == -position:
                    adjusted_signals.iloc[i, adjusted_signals.columns.get_loc('signal')] = current_signal
                    position = 0
                else:
                    # 保持原信號
                    adjusted_signals.iloc[i, adjusted_signals.columns.get_loc('signal')] = 0
            
            return adjusted_signals
        
        except Exception as e:
            logger.error(f"止損止盈應用失敗: {e}")
            return signals

class PositionSizer:
    """倉位管理器"""
    
    @staticmethod
    def calculate_position_size(account_value: float, risk_per_trade: float, 
                              entry_price: float, stop_loss_price: float) -> int:
        """
        計算倉位大小
        
        Args:
            account_value: 賬戶總價值
            risk_per_trade: 每筆交易風險百分比
            entry_price: 入場價格
            stop_loss_price: 止損價格
            
        Returns:
            建議倉位大小
        """
        try:
            if stop_loss_price == 0 or entry_price == 0:
                return 0
            
            risk_amount = account_value * risk_per_trade
            price_risk = abs(entry_price - stop_loss_price)
            
            if price_risk == 0:
                return 0
            
            position_size = int(risk_amount / price_risk)
            
            # 限制最大倉位
            max_position_value = account_value * RISK_CONFIG.max_position_size
            max_shares = int(max_position_value / entry_price)
            
            return min(position_size, max_shares)
        
        except Exception as e:
            logger.error(f"倉位計算失敗: {e}")
            return 0
    
    @staticmethod
    def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        凱利公式計算最優倉位比例
        
        Args:
            win_rate: 勝率
            avg_win: 平均盈利
            avg_loss: 平均虧損
            
        Returns:
            最優倉位比例
        """
        try:
            if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
                return 0.0
            
            b = avg_win / abs(avg_loss)  # 賠率
            p = win_rate  # 勝率
            q = 1 - win_rate  # 敗率
            
            kelly_pct = (b * p - q) / b
            
            # 限制最大倉位比例
            return max(0, min(kelly_pct, RISK_CONFIG.max_position_size))
        
        except Exception as e:
            logger.error(f"凱利公式計算失敗: {e}")
            return 0.0 