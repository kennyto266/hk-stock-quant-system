#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股量化分析系統 - 增強版策略優化器
專門針對港股市場特點進行策略優化和表現提升
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from itertools import product
import warnings
from datetime import datetime, timedelta
from scipy.optimize import minimize, differential_evolution
from scipy.stats import rankdata
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import sharpe_ratio
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class AdvancedStrategyOptimizer:
    """增強版策略優化器"""
    
    def __init__(self, data: pd.DataFrame, validation_split: float = 0.3):
        """
        初始化優化器
        
        Args:
            data: 股票數據
            validation_split: 驗證集比例
        """
        self.data = data.copy()
        self.validation_split = validation_split
        
        # 分割訓練和驗證數據
        split_idx = int(len(data) * (1 - validation_split))
        self.train_data = data.iloc[:split_idx]
        self.validation_data = data.iloc[split_idx:]
        
        self.optimization_results = {}
        self.ensemble_results = {}
        
        print(f"📊 數據分割完成: 訓練集 {len(self.train_data)} 天, 驗證集 {len(self.validation_data)} 天")
    
    def walk_forward_optimize(self, strategy_class, param_ranges: Dict, 
                             window_size: int = 252, step_size: int = 63) -> Dict:
        """
        走步優化 - 模擬真實交易環境
        
        Args:
            strategy_class: 策略類
            param_ranges: 參數範圍
            window_size: 優化窗口大小（天）
            step_size: 步進大小（天）
        """
        results = []
        param_names = list(param_ranges.keys())
        
        # 生成所有參數組合
        param_combinations = list(product(*param_ranges.values()))
        
        print(f"🚶 開始走步優化: {len(param_combinations)} 個參數組合")
        
        # 走步優化
        for start_idx in range(0, len(self.data) - window_size, step_size):
            end_idx = start_idx + window_size
            window_data = self.data.iloc[start_idx:end_idx]
            
            if len(window_data) < 50:  # 數據太少跳過
                continue
            
            best_sharpe = -np.inf
            best_params = None
            
            # 測試所有參數組合
            for params in param_combinations:
                try:
                    param_dict = dict(zip(param_names, params))
                    
                    # 創建策略實例
                    if strategy_class.__name__ == 'RSIStrategy':
                        strategy = strategy_class(param_dict.get('period', 14),
                                                param_dict.get('overbought', 70),
                                                param_dict.get('oversold', 30))
                    elif strategy_class.__name__ == 'MACDStrategy':
                        strategy = strategy_class(param_dict.get('fast_period', 12),
                                                param_dict.get('slow_period', 26),
                                                param_dict.get('signal_period', 9))
                    elif strategy_class.__name__ == 'BollingerStrategy':
                        strategy = strategy_class(param_dict.get('period', 20),
                                                param_dict.get('num_std', 2.0))
                    else:
                        continue
                    
                    # 生成信號和計算績效
                    signals = strategy.generate_signals(window_data)
                    returns_df = strategy.calculate_returns(window_data, signals)
                    metrics = strategy.calculate_performance_metrics(returns_df)
                    
                    if metrics['sharpe_ratio'] > best_sharpe:
                        best_sharpe = metrics['sharpe_ratio']
                        best_params = param_dict
                        
                except Exception as e:
                    continue
            
            if best_params:
                results.append({
                    'period_start': window_data.index[0],
                    'period_end': window_data.index[-1],
                    'best_params': best_params,
                    'best_sharpe': best_sharpe
                })
        
        print(f"✅ 走步優化完成: {len(results)} 個時期")
        return {
            'results': results,
            'strategy_name': strategy_class.__name__,
            'avg_sharpe': np.mean([r['best_sharpe'] for r in results if r['best_sharpe'] > -np.inf])
        }
    
    def genetic_algorithm_optimize(self, strategy_class, param_ranges: Dict, 
                                  generations: int = 50, population_size: int = 100) -> Dict:
        """
        遺傳算法優化
        
        Args:
            strategy_class: 策略類
            param_ranges: 參數範圍字典
            generations: 進化代數
            population_size: 種群大小
        """
        print(f"🧬 開始遺傳算法優化: {generations} 代, 種群 {population_size}")
        
        # 定義目標函數
        def objective_function(params):
            try:
                # 解析參數
                if strategy_class.__name__ == 'RSIStrategy':
                    period = int(params[0])
                    overbought = params[1]
                    oversold = params[2]
                    strategy = strategy_class(period, overbought, oversold)
                elif strategy_class.__name__ == 'MACDStrategy':
                    fast_period = int(params[0])
                    slow_period = int(params[1])
                    signal_period = int(params[2])
                    if fast_period >= slow_period:
                        return -10  # 懲罰無效參數
                    strategy = strategy_class(fast_period, slow_period, signal_period)
                elif strategy_class.__name__ == 'BollingerStrategy':
                    period = int(params[0])
                    num_std = params[1]
                    strategy = strategy_class(period, num_std)
                else:
                    return -10
                
                # 計算策略績效
                signals = strategy.generate_signals(self.train_data)
                returns_df = strategy.calculate_returns(self.train_data, signals)
                metrics = strategy.calculate_performance_metrics(returns_df)
                
                # 組合目標：夏普比率 + 收益率 - 回撤懲罰
                score = (metrics['sharpe_ratio'] * 0.4 + 
                        metrics['annual_return'] * 0.003 + 
                        -metrics['max_drawdown'] * 0.01)
                
                return -score  # 最小化，所以取負號
                
            except Exception as e:
                return 10  # 懲罰錯誤參數
        
        # 設置參數邊界
        param_names = list(param_ranges.keys())
        bounds = []
        for param_name in param_names:
            param_range = param_ranges[param_name]
            bounds.append((min(param_range), max(param_range)))
        
        # 執行差分進化算法
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=generations,
            popsize=population_size // len(bounds),
            seed=42,
            workers=1
        )
        
        # 解析最佳參數
        best_params = dict(zip(param_names, result.x))
        
        # 驗證最佳參數
        try:
            if strategy_class.__name__ == 'RSIStrategy':
                strategy = strategy_class(int(best_params['period']),
                                        best_params['overbought'],
                                        best_params['oversold'])
            elif strategy_class.__name__ == 'MACDStrategy':
                strategy = strategy_class(int(best_params['fast_period']),
                                        int(best_params['slow_period']),
                                        int(best_params['signal_period']))
            elif strategy_class.__name__ == 'BollingerStrategy':
                strategy = strategy_class(int(best_params['period']),
                                        best_params['num_std'])
            
            # 訓練集績效
            signals = strategy.generate_signals(self.train_data)
            returns_df = strategy.calculate_returns(self.train_data, signals)
            train_metrics = strategy.calculate_performance_metrics(returns_df)
            
            # 驗證集績效
            val_signals = strategy.generate_signals(self.validation_data)
            val_returns_df = strategy.calculate_returns(self.validation_data, val_signals)
            val_metrics = strategy.calculate_performance_metrics(val_returns_df)
            
        except Exception as e:
            train_metrics = {'sharpe_ratio': 0, 'annual_return': 0, 'max_drawdown': 100}
            val_metrics = {'sharpe_ratio': 0, 'annual_return': 0, 'max_drawdown': 100}
        
        print(f"✅ 遺傳算法優化完成")
        
        return {
            'strategy_name': strategy_class.__name__,
            'best_params': best_params,
            'train_metrics': train_metrics,
            'validation_metrics': val_metrics,
            'optimization_score': -result.fun,
            'convergence': result.success
        }
    
    def ensemble_optimization(self, strategies_results: List[Dict]) -> Dict:
        """
        集成優化 - 組合多個策略
        
        Args:
            strategies_results: 各策略優化結果列表
        """
        print("🎯 開始集成策略優化...")
        
        # 獲取所有策略在驗證集上的收益序列
        strategy_returns = {}
        
        for result in strategies_results:
            strategy_name = result['strategy_name']
            try:
                # 重建策略
                if strategy_name == 'RSIStrategy':
                    from strategies import RSIStrategy
                    params = result['best_params']
                    strategy = RSIStrategy(int(params['period']),
                                         params['overbought'],
                                         params['oversold'])
                elif strategy_name == 'MACDStrategy':
                    from strategies import MACDStrategy
                    params = result['best_params']
                    strategy = MACDStrategy(int(params['fast_period']),
                                          int(params['slow_period']),
                                          int(params['signal_period']))
                elif strategy_name == 'BollingerStrategy':
                    from strategies import BollingerStrategy
                    params = result['best_params']
                    strategy = BollingerStrategy(int(params['period']),
                                               params['num_std'])
                else:
                    continue
                
                # 獲取驗證集收益
                signals = strategy.generate_signals(self.validation_data)
                returns_df = strategy.calculate_returns(self.validation_data, signals)
                strategy_returns[strategy_name] = returns_df['Strategy_Return'].fillna(0)
                
            except Exception as e:
                print(f"⚠️ 策略 {strategy_name} 集成失敗: {e}")
                continue
        
        if len(strategy_returns) < 2:
            print("❌ 可用策略少於2個，無法進行集成優化")
            return {}
        
        # 轉換為DataFrame
        returns_df = pd.DataFrame(strategy_returns)
        returns_df = returns_df.dropna()
        
        if returns_df.empty:
            print("❌ 策略收益數據為空")
            return {}
        
        # 優化權重
        n_strategies = len(returns_df.columns)
        
        def portfolio_sharpe(weights):
            portfolio_returns = (returns_df * weights).sum(axis=1)
            if portfolio_returns.std() == 0:
                return -10
            return -(portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252))
        
        # 約束條件
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_strategies))
        
        # 初始猜測
        x0 = np.array([1/n_strategies] * n_strategies)
        
        # 優化
        result = minimize(portfolio_sharpe, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = dict(zip(returns_df.columns, result.x))
            
            # 計算集成策略績效
            portfolio_returns = (returns_df * result.x).sum(axis=1)
            
            # 計算績效指標
            total_return = (1 + portfolio_returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe = annual_return / volatility if volatility > 0 else 0
            
            # 計算最大回撤
            cumulative = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min())
            
            ensemble_metrics = {
                'total_return': float(total_return * 100),
                'annual_return': float(annual_return * 100),
                'volatility': float(volatility * 100),
                'sharpe_ratio': float(sharpe),
                'max_drawdown': float(max_drawdown * 100)
            }
            
            print("✅ 集成策略優化完成")
            
            return {
                'optimal_weights': optimal_weights,
                'ensemble_metrics': ensemble_metrics,
                'individual_strategies': strategies_results,
                'returns_series': portfolio_returns
            }
        else:
            print(f"❌ 集成優化失敗: {result.message}")
            return {}
    
    def risk_adjusted_optimization(self, strategy_class, param_ranges: Dict) -> Dict:
        """
        風險調整優化 - 考慮下行風險和尾部風險
        """
        print(f"⚡ 開始風險調整優化: {strategy_class.__name__}")
        
        best_score = -np.inf
        best_params = None
        best_metrics = None
        
        param_names = list(param_ranges.keys())
        param_combinations = list(product(*param_ranges.values()))
        
        for params in param_combinations[:100]:  # 限制組合數量
            try:
                param_dict = dict(zip(param_names, params))
                
                # 創建策略
                if strategy_class.__name__ == 'RSIStrategy':
                    strategy = strategy_class(param_dict.get('period', 14),
                                            param_dict.get('overbought', 70),
                                            param_dict.get('oversold', 30))
                elif strategy_class.__name__ == 'MACDStrategy':
                    if param_dict.get('fast_period', 12) >= param_dict.get('slow_period', 26):
                        continue
                    strategy = strategy_class(param_dict.get('fast_period', 12),
                                            param_dict.get('slow_period', 26),
                                            param_dict.get('signal_period', 9))
                elif strategy_class.__name__ == 'BollingerStrategy':
                    strategy = strategy_class(param_dict.get('period', 20),
                                            param_dict.get('num_std', 2.0))
                else:
                    continue
                
                # 計算策略收益
                signals = strategy.generate_signals(self.train_data)
                returns_df = strategy.calculate_returns(self.train_data, signals)
                strategy_returns = returns_df['Strategy_Return'].dropna()
                
                if len(strategy_returns) < 30:
                    continue
                
                # 計算風險調整指標
                metrics = strategy.calculate_performance_metrics(returns_df)
                
                # 額外風險指標
                negative_returns = strategy_returns[strategy_returns < 0]
                if len(negative_returns) > 0:
                    downside_deviation = negative_returns.std() * np.sqrt(252)
                    sortino_ratio = metrics['annual_return'] / (downside_deviation * 100) if downside_deviation > 0 else 0
                else:
                    sortino_ratio = metrics['sharpe_ratio']
                
                # VaR (95%)
                var_95 = np.percentile(strategy_returns, 5) * np.sqrt(252) * 100
                
                # 組合評分 (重視風險調整後收益)
                risk_score = (
                    sortino_ratio * 0.4 +
                    metrics['sharpe_ratio'] * 0.3 +
                    -metrics['max_drawdown'] * 0.01 +
                    -abs(var_95) * 0.001 +
                    metrics['win_rate'] * 0.01
                )
                
                if risk_score > best_score:
                    best_score = risk_score
                    best_params = param_dict
                    best_metrics = metrics.copy()
                    best_metrics['sortino_ratio'] = sortino_ratio
                    best_metrics['var_95'] = var_95
                    
            except Exception as e:
                continue
        
        print(f"✅ 風險調整優化完成")
        
        return {
            'strategy_name': strategy_class.__name__,
            'best_params': best_params,
            'metrics': best_metrics,
            'risk_score': best_score
        }
    
    def create_optimization_report(self, results: Dict) -> str:
        """創建優化報告"""
        report = []
        report.append("# 策略優化報告")
        report.append("=" * 50)
        report.append(f"📅 優化時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"📊 數據期間: {self.data.index[0].strftime('%Y-%m-%d')} 至 {self.data.index[-1].strftime('%Y-%m-%d')}")
        report.append(f"📈 總交易日: {len(self.data)} 天")
        report.append("")
        
        # 各策略結果
        for strategy_name, result in results.items():
            if isinstance(result, dict) and 'best_params' in result:
                report.append(f"## {strategy_name}")
                report.append(f"**最佳參數**: {result['best_params']}")
                
                if 'metrics' in result:
                    metrics = result['metrics']
                    report.append(f"- 夏普比率: {metrics.get('sharpe_ratio', 0):.3f}")
                    report.append(f"- 年化收益: {metrics.get('annual_return', 0):.2f}%")
                    report.append(f"- 最大回撤: {metrics.get('max_drawdown', 0):.2f}%")
                    report.append(f"- 勝率: {metrics.get('win_rate', 0):.2f}%")
                    
                    if 'sortino_ratio' in metrics:
                        report.append(f"- Sortino比率: {metrics['sortino_ratio']:.3f}")
                    if 'var_95' in metrics:
                        report.append(f"- VaR(95%): {metrics['var_95']:.2f}%")
                
                report.append("")
        
        # 集成策略結果
        if 'ensemble' in results:
            ensemble = results['ensemble']
            report.append("## 集成策略")
            
            if 'optimal_weights' in ensemble:
                report.append("**最優權重配置**:")
                for strategy, weight in ensemble['optimal_weights'].items():
                    report.append(f"- {strategy}: {weight:.3f}")
                
                metrics = ensemble['ensemble_metrics']
                report.append(f"\n**集成策略績效**:")
                report.append(f"- 夏普比率: {metrics['sharpe_ratio']:.3f}")
                report.append(f"- 年化收益: {metrics['annual_return']:.2f}%")
                report.append(f"- 最大回撤: {metrics['max_drawdown']:.2f}%")
                report.append("")
        
        # 優化建議
        report.append("## 📋 優化建議")
        report.append("1. **參數穩定性**: 定期重新優化參數以適應市場變化")
        report.append("2. **風險控制**: 關注最大回撤，考慮增加風險管理機制")
        report.append("3. **集成策略**: 使用多策略組合以降低單一策略風險")
        report.append("4. **實盤驗證**: 在模擬環境中測試策略穩定性")
        report.append("")
        
        return "\n".join(report)

# 使用示例函數
def run_comprehensive_optimization(data: pd.DataFrame) -> Dict:
    """
    運行綜合策略優化
    
    Args:
        data: 股票數據
        
    Returns:
        優化結果字典
    """
    print("🚀 開始綜合策略優化...")
    
    # 初始化優化器
    optimizer = AdvancedStrategyOptimizer(data, validation_split=0.3)
    
    # 導入策略類
    try:
        from strategies import RSIStrategy, MACDStrategy, BollingerStrategy
    except ImportError:
        print("❌ 無法導入策略類，請確保strategies.py可用")
        return {}
    
    results = {}
    
    # 1. RSI策略優化
    print("🔄 優化RSI策略...")
    rsi_ranges = {
        'period': [10, 12, 14, 16, 18, 20, 22],
        'overbought': [65, 70, 75, 80],
        'oversold': [20, 25, 30, 35]
    }
    
    results['RSI_genetic'] = optimizer.genetic_algorithm_optimize(
        RSIStrategy, rsi_ranges, generations=30, population_size=60
    )
    
    results['RSI_risk_adjusted'] = optimizer.risk_adjusted_optimization(
        RSIStrategy, rsi_ranges
    )
    
    # 2. MACD策略優化
    print("🔄 優化MACD策略...")
    macd_ranges = {
        'fast_period': [8, 10, 12, 14, 16],
        'slow_period': [20, 24, 26, 28, 32],
        'signal_period': [6, 8, 9, 10, 12]
    }
    
    results['MACD_genetic'] = optimizer.genetic_algorithm_optimize(
        MACDStrategy, macd_ranges, generations=30, population_size=60
    )
    
    # 3. 布林帶策略優化
    print("🔄 優化布林帶策略...")
    bb_ranges = {
        'period': [15, 18, 20, 22, 25],
        'num_std': [1.5, 1.8, 2.0, 2.2, 2.5]
    }
    
    results['Bollinger_genetic'] = optimizer.genetic_algorithm_optimize(
        BollingerStrategy, bb_ranges, generations=25, population_size=50
    )
    
    # 4. 集成策略優化
    individual_results = [
        results['RSI_genetic'],
        results['MACD_genetic'],
        results['Bollinger_genetic']
    ]
    
    valid_results = [r for r in individual_results if r and 'best_params' in r]
    
    if len(valid_results) >= 2:
        print("🔄 優化集成策略...")
        results['ensemble'] = optimizer.ensemble_optimization(valid_results)
    
    # 5. 生成報告
    report = optimizer.create_optimization_report(results)
    results['optimization_report'] = report
    
    print("✅ 綜合策略優化完成!")
    print("\n" + "="*50)
    print("📊 優化摘要:")
    
    for name, result in results.items():
        if isinstance(result, dict) and 'best_params' in result:
            if 'validation_metrics' in result:
                sharpe = result['validation_metrics'].get('sharpe_ratio', 0)
                print(f"- {name}: 驗證集夏普比率 {sharpe:.3f}")
            elif 'metrics' in result:
                sharpe = result['metrics'].get('sharpe_ratio', 0)
                print(f"- {name}: 夏普比率 {sharpe:.3f}")
    
    return results

if __name__ == "__main__":
    # 測試代碼
    print("🧪 增強版策略優化器測試")
    
    # 創建模擬數據
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    np.random.seed(42)
    
    # 模擬港股走勢（有趨勢和波動）
    returns = np.random.normal(0.0005, 0.02, len(dates))
    trend = np.linspace(0, 0.3, len(dates))
    
    price = 100 * np.exp(np.cumsum(returns) + trend)
    
    test_data = pd.DataFrame({
        'Open': price * (1 + np.random.normal(0, 0.001, len(dates))),
        'High': price * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
        'Low': price * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
        'Close': price,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    # 運行優化
    optimization_results = run_comprehensive_optimization(test_data)
    
    if optimization_results:
        print("\n📄 優化報告:")
        print(optimization_results.get('optimization_report', '報告生成失敗'))
    
    print("✅ 測試完成")