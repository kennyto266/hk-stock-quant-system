#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
統一策略優化器，支援多種優化模式（網格、隨機、遺傳、暴力、多進程等）
"""

import pandas as pd
import numpy as np
import random
from typing import Dict, List, Any, Tuple, Callable, Optional, Union
from datetime import datetime
import itertools
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import logging
import gc
import psutil
import warnings
from data_handler import DataFetcher
import strategies

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class UnifiedStrategyOptimizer:
    """
    統一策略優化器，支援多種優化模式（網格、隨機、遺傳、暴力、多進程等）
    """
    def __init__(self, symbol: str, start_date: str, end_date: Optional[str] = None):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date if end_date is not None else datetime.now().strftime('%Y-%m-%d')
        self.data: Optional[pd.DataFrame] = None
        self.train_data: Optional[pd.DataFrame] = None
        self.validation_data: Optional[pd.DataFrame] = None
        self.best_params: Dict = {}
        self.use_cross_validation = False  # 關閉交叉驗證
        self.train_ratio = 0.7  # 使用70%數據作為訓練集
        self.max_processes = min(32, cpu_count())
        self.signals: Optional[pd.Series] = None
        self._setup_logging()
        
    def _load_data(self) -> Optional[pd.DataFrame]:
        """載入數據"""
        try:
            fetcher = DataFetcher()
            data = fetcher.fetch_data(self.symbol, self.start_date, self.end_date)
            if isinstance(data, pd.DataFrame):
                logger.info(f"✅ 成功載入 {len(data)} 天的數據")
                return data
            else:
                logger.error("❌ 數據載入失敗")
                return None
        except Exception as e:
            logger.error(f"❌ 數據載入失敗: {str(e)}")
            return None
            
    def _generate_param_grid(self) -> List[Dict]:
        """生成參數網格"""
        param_grid = []
        
        # RSI 策略參數
        for period in range(5, 301, 5):  # 5-300
            for overbought in range(50, 91, 5):  # 50-90
                for oversold in range(10, 51, 5):  # 10-50
                    if oversold < overbought:
                        param_grid.append({
                            'strategy_type': 'RSI',
                            'period': period,
                            'overbought': overbought,
                            'oversold': oversold
                        })
        
        # MACD 策略參數
        for fast in range(5, 21):  # 5-20
            for slow in range(15, 41):  # 15-40
                for signal in range(5, 16):  # 5-15
                    if fast < slow:
                        param_grid.append({
                            'strategy_type': 'MACD',
                            'fast_period': fast,
                            'slow_period': slow,
                            'signal_period': signal
                        })
        
        # 布林通道策略參數
        for period in range(10, 51, 2):  # 10-50
            for std_dev in np.arange(1.5, 3.1, 0.1):  # 1.5-3.0
                param_grid.append({
                    'strategy_type': 'Bollinger',
                    'period': period,
                    'std_dev': float(std_dev)
                })
                
        return param_grid
        
    def _create_strategy(self, params: Dict) -> Any:
        """根據參數創建策略實例"""
        strategy_type = params.get('strategy_type', 'RSI')
        
        if strategy_type == 'RSI':
            return strategies.RSIStrategy(
                period=params.get('period', 14),
                overbought=params.get('overbought', 70),
                oversold=params.get('oversold', 30)
            )
        elif strategy_type == 'MACD':
            return strategies.MACDStrategy(
                fast_period=params.get('fast_period', 12),
                slow_period=params.get('slow_period', 26),
                signal_period=params.get('signal_period', 9)
            )
        elif strategy_type == 'Bollinger':
            return strategies.BollingerStrategy(
                period=params.get('period', 20),
                std_dev=params.get('std_dev', 2.0)
            )
        else:
            raise ValueError(f"不支援的策略類型: {strategy_type}")
            
    def optimize(self) -> Optional[Dict]:
        """優化策略參數"""
        logger.info("🔄 開始優化策略參數...")
        
        # 載入數據
        self.data = self._load_data()
        if not isinstance(self.data, pd.DataFrame) or self.data.empty:
            logger.error("❌ 數據載入失敗")
            return None
            
        # 分割訓練集和驗證集
        train_size = int(len(self.data) * self.train_ratio)
        self.train_data = self.data.iloc[:train_size].copy()
        self.validation_data = self.data.iloc[train_size:].copy()
        
        if isinstance(self.train_data, pd.DataFrame) and isinstance(self.validation_data, pd.DataFrame):
            logger.info(f"📈 訓練集大小: {len(self.train_data)} 天")
            logger.info(f"📊 驗證集大小: {len(self.validation_data)} 天")
        else:
            logger.error("❌ 數據分割失敗")
            return None
        
        # 生成參數網格
        param_combinations = self._generate_param_grid()
        if not param_combinations:
            logger.error("❌ 無有效參數組合")
            return None
            
        logger.info(f"🔍 總參數組合數: {len(param_combinations)}")
        
        # 使用多進程優化
        with Pool(processes=self.max_processes) as pool:
            eval_args = [(params, self.train_data, self.validation_data) 
                        for params in param_combinations]
            results = list(tqdm(pool.imap(self._evaluate_params, eval_args),
                              total=len(param_combinations),
                              desc="參數優化進度"))
        
        # 找出最佳參數
        if results:
            best_result = max(results, key=lambda x: x['metrics']['sharpe_ratio'])
            self.best_params = best_result['params']
            
            # 使用最佳參數生成信號
            strategy = self._create_strategy(self.best_params)
            strategy.fit(self.data)
            self.signals = strategy.signals
            
            logger.info(f"✅ 最佳參數: {self.best_params}")
            logger.info(f"📊 最佳夏普比率: {best_result['metrics']['sharpe_ratio']:.2f}")
            
            return self.best_params
        else:
            logger.error("❌ 無有效優化結果")
            return None

    def _evaluate_params(self, args: Tuple[Dict, pd.DataFrame, pd.DataFrame]) -> Dict:
        """評估單個參數組合"""
        params, train_data, validation_data = args
        
        if not isinstance(train_data, pd.DataFrame) or not isinstance(validation_data, pd.DataFrame):
            return {
                'params': params,
                'metrics': {'sharpe_ratio': float('-inf')}
            }
        
        try:
            # 在訓練集上訓練策略
            strategy = self._create_strategy(params)
            strategy.fit(train_data)
            
            # 在驗證集上評估
            metrics = strategy.evaluate(validation_data)
            
            return {
                'params': params,
                'metrics': metrics
            }
        except Exception as e:
            logger.error(f"❌ 參數評估失敗: {params} - {str(e)}")
            return {
                'params': params,
                'metrics': {'sharpe_ratio': float('-inf')}
            }

    def _setup_logging(self):
        """設置日誌記錄"""
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(handler)

    def evaluate_strategy(self, strategy_class: Callable, params: Dict, data: pd.DataFrame) -> Dict:
        """評估單個策略的性能，包含更全面的績效指標"""
        try:
            strategy = strategy_class(**params)
            signals = strategy.generate_signals(data)
            if signals.empty or 'signal' not in signals.columns:
                return {}
            
            # 計算交易統計
            nonzero_count = int((signals['signal'] != 0).sum())
            if nonzero_count == 0:  # 如果沒有交易信號，返回空結果
                return {}
                
            signals['position'] = signals['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
            signals['returns'] = data['Close'].pct_change().fillna(0)
            signals['strategy_returns'] = signals['position'].shift(1) * signals['returns']
            
            # 移除首個交易日的收益（因為沒有前一天的倉位）
            signals.loc[signals.index[0], 'strategy_returns'] = 0
            
            signals['equity'] = (1 + signals['strategy_returns']).cumprod()
            
            # 計算基礎績效指標
            total_return = signals['equity'].iloc[-1] - 1
            annual_return = (1 + total_return) ** (252 / len(signals)) - 1
            daily_returns = signals['strategy_returns']
            volatility = daily_returns.std() * np.sqrt(252)
            
            # 計算夏普比率（如果波動率為0，則返回0）
            risk_free_rate = 0.02  # 假設無風險利率為2%
            excess_returns = annual_return - risk_free_rate
            sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
            
            # 計算索提諾比率
            downside_returns = daily_returns[daily_returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0
            sortino_ratio = excess_returns / downside_volatility if downside_volatility > 0 else 0
            
            # 計算最大回撤及其持續時間
            rolling_max = signals['equity'].expanding().max()
            drawdowns = signals['equity'] / rolling_max - 1
            max_drawdown = drawdowns.min()
            
            # 計算最大回撤持續時間
            drawdown_periods = pd.DataFrame(index=signals.index)
            drawdown_periods['drawdown'] = drawdowns
            drawdown_periods['is_drawdown'] = drawdown_periods['drawdown'] < 0
            drawdown_periods['drawdown_group'] = (drawdown_periods['is_drawdown'] != drawdown_periods['is_drawdown'].shift()).cumsum()
            drawdown_durations = drawdown_periods[drawdown_periods['is_drawdown']].groupby('drawdown_group').size()
            max_drawdown_duration = drawdown_durations.max() if not drawdown_durations.empty else 0
            
            # 計算交易統計
            trading_days = signals[signals['strategy_returns'] != 0]
            win_rate = (trading_days['strategy_returns'] > 0).mean() * 100 if not trading_days.empty else 0
            
            # 計算平均獲利和虧損
            avg_profit = trading_days[trading_days['strategy_returns'] > 0]['strategy_returns'].mean() if not trading_days.empty else 0
            avg_loss = trading_days[trading_days['strategy_returns'] < 0]['strategy_returns'].mean() if not trading_days.empty else 0
            profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
            
            # 計算持倉時間統計
            position_changes = signals['position'].diff().fillna(0)
            trade_starts = position_changes != 0
            holding_periods = []
            current_period = 0
            
            for i in range(len(signals)):
                if trade_starts.iloc[i]:
                    if current_period > 0:
                        holding_periods.append(current_period)
                    current_period = 1
                elif signals['position'].iloc[i] != 0:
                    current_period += 1
            
            if current_period > 0:
                holding_periods.append(current_period)
            
            avg_holding_period = np.mean(holding_periods) if holding_periods else 0
            
            # 清理記憶體
            del signals, drawdown_periods, trading_days
            gc.collect()
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'annual_return': annual_return * 100,  # 轉換為百分比
                'max_drawdown': max_drawdown * 100,  # 轉換為百分比
                'max_drawdown_duration': max_drawdown_duration,
                'volatility': volatility * 100,  # 轉換為百分比
                'win_rate': win_rate,
                'profit_loss_ratio': profit_loss_ratio,
                'avg_holding_period': avg_holding_period,
                'trade_count': nonzero_count,
                'avg_profit': avg_profit * 100,  # 轉換為百分比
                'avg_loss': avg_loss * 100  # 轉換為百分比
            }
        except Exception as e:
            logger.error(f"Strategy evaluation failed: {e}")
            return {}

    def _evaluate_grid_params(self, args: Tuple[Callable, Dict, pd.DataFrame]) -> Optional[Tuple[Dict, Dict]]:
        """評估單個網格搜索參數組合"""
        strategy_class, params, data = args
        metrics = self.evaluate_strategy(strategy_class, params, data)
        if metrics and 'sharpe_ratio' in metrics and metrics['trade_count'] > 0:
            return params, metrics
        return None

    def grid_search(self, strategy_class, param_grid: Dict) -> Dict:
        """
        使用多進程進行網格搜索，包含交叉驗證
        
        Args:
            strategy_class: 策略類
            param_grid: 參數網格，格式為 {'param_name': [param_values]}
        """
        # 驗證參數範圍
        self._validate_param_grid(param_grid)
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        logger.info(f"開始網格搜索，共 {len(param_combinations)} 組參數組合")
        
        # 準備交叉驗證
        n_splits = 5  # 5折交叉驗證
        data_length = len(self.train_data)
        fold_size = data_length // n_splits
        
        all_results = []
        for fold in range(n_splits):
            logger.info(f"開始第 {fold + 1} 折交叉驗證")
            
            # 準備訓練和驗證數據
            val_start = fold * fold_size
            val_end = val_start + fold_size
            if fold == n_splits - 1:  # 最後一折使用剩餘所有數據
                val_end = data_length
            
            train_data = pd.concat([
                self.train_data.iloc[:val_start],
                self.train_data.iloc[val_end:]
            ])
            val_data = self.train_data.iloc[val_start:val_end]
            
            # 準備參數組合
            eval_args = [(strategy_class, dict(zip(param_names, params)), train_data) 
                        for params in param_combinations]
            
            # 使用多進程進行評估
            with Pool(processes=self.max_processes) as pool:
                fold_results = list(tqdm(pool.imap(self._evaluate_grid_params, eval_args), 
                                      total=len(param_combinations), 
                                      desc=f"第 {fold + 1} 折搜索進度"))
            
            # 過濾有效結果並計算驗證集績效
            valid_results = []
            for result in fold_results:
                if result is not None:
                    params, train_metrics = result
                    val_metrics = self.evaluate_strategy(strategy_class, params, val_data)
                    if val_metrics:  # 只保留在驗證集上也有效的結果
                        valid_results.append({
                            'params': params,
                            'train_metrics': train_metrics,
                            'val_metrics': val_metrics,
                            'fold': fold
                        })
            
            all_results.extend(valid_results)
            
            # 清理記憶體
            del train_data, val_data, fold_results, valid_results
            gc.collect()
        
        if all_results:
            # 根據所有折的平均績效進行排序
            def avg_score(result):
                train_sharpe = result['train_metrics']['sharpe_ratio']
                val_sharpe = result['val_metrics']['sharpe_ratio']
                train_sortino = result['train_metrics']['sortino_ratio']
                val_sortino = result['val_metrics']['sortino_ratio']
                train_trades = result['train_metrics']['trade_count']
                val_trades = result['val_metrics']['trade_count']
                
                # 綜合考慮夏普比率、索提諾比率和交易次數
                train_score = (train_sharpe + train_sortino) * (1 + np.log1p(train_trades) / 10)
                val_score = (val_sharpe + val_sortino) * (1 + np.log1p(val_trades) / 10)
                
                # 如果任一指標為負，則分數為負無窮
                if train_sharpe <= 0 or val_sharpe <= 0:
                    return float('-inf')
                
                # 返回訓練集和驗證集的平均分數
                return (train_score + val_score) / 2
            
            # 找出最佳參數組合
            best_result = max(all_results, key=avg_score)
            best_params = best_result['params']
            
            # 在完整驗證集上進行最終評估
            final_validation_metrics = self.evaluate_strategy(strategy_class, best_params, self.validation_data)
            
            # 計算參數穩定性分數
            param_stability = self._calculate_param_stability(all_results, best_params)
            
            logger.info(f"網格搜索完成，找到最佳參數：{best_params}")
            logger.info(f"參數穩定性：{param_stability}")
            logger.info(f"訓練集績效：{best_result['train_metrics']}")
            logger.info(f"交叉驗證集績效：{best_result['val_metrics']}")
            logger.info(f"最終驗證集績效：{final_validation_metrics}")
            
            return {
                'best_params': best_params,
                'param_stability': param_stability,
                'train_metrics': best_result['train_metrics'],
                'cv_metrics': best_result['val_metrics'],
                'validation_metrics': final_validation_metrics,
                'method': 'Grid Search with Cross-Validation'
            }
        
        logger.warning("網格搜索未找到有效參數組合")
        return {}
        
    def _validate_param_grid(self, param_grid: Dict) -> None:
        """驗證參數網格的有效性"""
        if not param_grid:
            raise ValueError("參數網格不能為空")
        
        for param_name, param_values in param_grid.items():
            if not param_values:
                raise ValueError(f"參數 {param_name} 的值列表不能為空")
            
            # 檢查參數類型
            if all(isinstance(x, (int, float)) for x in param_values):
                # 數值參數的範圍檢查
                if min(param_values) <= 0 and param_name.lower() in ['period', 'fast', 'slow', 'signal']:
                    raise ValueError(f"參數 {param_name} 的值必須大於0")
                if param_name.lower() in ['std_dev', 'threshold'] and min(param_values) <= 0:
                    raise ValueError(f"參數 {param_name} 的值必須大於0")
            
            # 檢查參數數量
            if len(param_values) > 100:
                logger.warning(f"參數 {param_name} 的值數量過多 ({len(param_values)})，可能導致搜索時間過長")
    
    def _calculate_param_stability(self, results: List[Dict], best_params: Dict) -> Dict:
        """計算參數穩定性分數"""
        param_scores = {}
        for param_name in best_params.keys():
            # 收集所有結果中該參數的值
            param_values = [r['params'][param_name] for r in results]
            # 計算最佳值的出現頻率
            best_value = best_params[param_name]
            frequency = param_values.count(best_value) / len(param_values)
            # 計算與最佳值的平均偏差
            if isinstance(best_value, (int, float)):
                deviations = [abs(v - best_value) / best_value for v in param_values]
                avg_deviation = np.mean(deviations)
                param_scores[param_name] = {
                    'stability': frequency,
                    'avg_deviation': avg_deviation
                }
            else:
                param_scores[param_name] = {
                    'stability': frequency
                }
        return param_scores

    def _evaluate_random_params(self, args: Tuple[Callable, Dict, pd.DataFrame]) -> Optional[Tuple[Dict, Dict]]:
        """評估單個隨機搜索參數組合"""
        strategy_class, param_ranges, data = args
        params = {name: random.choice(values) for name, values in param_ranges.items()}
        metrics = self.evaluate_strategy(strategy_class, params, data)
        if metrics and 'sharpe_ratio' in metrics and metrics['trade_count'] > 0:
            return params, metrics
        return None

    def random_search(self, strategy_class, param_grid: Dict, n_iter: int = 100) -> Dict:
        """
        使用多進程進行隨機搜索，包含交叉驗證
        
        Args:
            strategy_class: 策略類
            param_grid: 參數網格，格式為 {'param_name': [param_values]}
            n_iter: 每折的迭代次數
        """
        # 驗證參數範圍
        self._validate_param_grid(param_grid)
        
        logger.info(f"開始隨機搜索，每折迭代次數：{n_iter}")
        
        # 準備交叉驗證
        n_splits = 5  # 5折交叉驗證
        data_length = len(self.train_data)
        fold_size = data_length // n_splits
        
        all_results = []
        for fold in range(n_splits):
            logger.info(f"開始第 {fold + 1} 折交叉驗證")
            
            # 準備訓練和驗證數據
            val_start = fold * fold_size
            val_end = val_start + fold_size
            if fold == n_splits - 1:  # 最後一折使用剩餘所有數據
                val_end = data_length
            
            train_data = pd.concat([
                self.train_data.iloc[:val_start],
                self.train_data.iloc[val_end:]
            ])
            val_data = self.train_data.iloc[val_start:val_end]
            
            # 準備參數組合
            eval_args = [(strategy_class, param_grid, train_data) 
                        for _ in range(n_iter)]
            
            # 使用多進程進行評估
            with Pool(processes=self.max_processes) as pool:
                fold_results = list(tqdm(pool.imap(self._evaluate_random_params, eval_args), 
                                      total=n_iter, 
                                      desc=f"第 {fold + 1} 折搜索進度"))
            
            # 過濾有效結果並計算驗證集績效
            valid_results = []
            for result in fold_results:
                if result is not None:
                    params, train_metrics = result
                    val_metrics = self.evaluate_strategy(strategy_class, params, val_data)
                    if val_metrics:  # 只保留在驗證集上也有效的結果
                        valid_results.append({
                            'params': params,
                            'train_metrics': train_metrics,
                            'val_metrics': val_metrics,
                            'fold': fold
                        })
            
            all_results.extend(valid_results)
            
            # 清理記憶體
            del train_data, val_data, fold_results, valid_results
            gc.collect()
        
        if all_results:
            # 根據所有折的平均績效進行排序
            def avg_score(result):
                train_sharpe = result['train_metrics']['sharpe_ratio']
                val_sharpe = result['val_metrics']['sharpe_ratio']
                train_sortino = result['train_metrics']['sortino_ratio']
                val_sortino = result['val_metrics']['sortino_ratio']
                train_trades = result['train_metrics']['trade_count']
                val_trades = result['val_metrics']['trade_count']
                
                # 綜合考慮夏普比率、索提諾比率和交易次數
                train_score = (train_sharpe + train_sortino) * (1 + np.log1p(train_trades) / 10)
                val_score = (val_sharpe + val_sortino) * (1 + np.log1p(val_trades) / 10)
                
                # 如果任一指標為負，則分數為負無窮
                if train_sharpe <= 0 or val_sharpe <= 0:
                    return float('-inf')
                
                # 返回訓練集和驗證集的平均分數
                return (train_score + val_score) / 2
            
            # 找出最佳參數組合
            best_result = max(all_results, key=avg_score)
            best_params = best_result['params']
            
            # 在完整驗證集上進行最終評估
            final_validation_metrics = self.evaluate_strategy(strategy_class, best_params, self.validation_data)
            
            # 計算參數穩定性分數
            param_stability = self._calculate_param_stability(all_results, best_params)
            
            # 計算參數分佈統計
            param_stats = self._calculate_param_distribution(all_results)
            
            logger.info(f"隨機搜索完成，找到最佳參數：{best_params}")
            logger.info(f"參數穩定性：{param_stability}")
            logger.info(f"參數分佈：{param_stats}")
            logger.info(f"訓練集績效：{best_result['train_metrics']}")
            logger.info(f"交叉驗證集績效：{best_result['val_metrics']}")
            logger.info(f"最終驗證集績效：{final_validation_metrics}")
            
            return {
                'best_params': best_params,
                'param_stability': param_stability,
                'param_distribution': param_stats,
                'train_metrics': best_result['train_metrics'],
                'cv_metrics': best_result['val_metrics'],
                'validation_metrics': final_validation_metrics,
                'method': 'Random Search with Cross-Validation'
            }
        
        logger.warning("隨機搜索未找到有效參數組合")
        return {}
        
    def _calculate_param_distribution(self, results: List[Dict]) -> Dict:
        """計算參數分佈統計"""
        param_stats = {}
        if not results:
            return param_stats
            
        # 獲取所有參數名
        param_names = results[0]['params'].keys()
        
        for param_name in param_names:
            param_values = [r['params'][param_name] for r in results]
            
            if all(isinstance(x, (int, float)) for x in param_values):
                param_stats[param_name] = {
                    'mean': np.mean(param_values),
                    'std': np.std(param_values),
                    'min': min(param_values),
                    'max': max(param_values),
                    'median': np.median(param_values)
                }
            else:
                # 對於非數值參數，計算每個值的出現頻率
                value_counts = {}
                for value in param_values:
                    value_counts[str(value)] = value_counts.get(str(value), 0) + 1
                param_stats[param_name] = {
                    'value_counts': value_counts
                }
        
        return param_stats

    def brute_force(self, test_func: Callable, param_combinations: List, max_processes: Optional[int] = None) -> List:
        """
        暴力搜索所有參數組合，包含記憶體管理和進度追蹤
        
        Args:
            test_func: 測試函數
            param_combinations: 參數組合列表
            max_processes: 最大進程數
        """
        if max_processes is None or max_processes <= 0:
            max_processes = self.max_processes
        
        total_combinations = len(param_combinations)
        logger.info(f"開始暴力搜索，共 {total_combinations} 組參數組合")
        
        # 計算每批次的大小，避免記憶體溢出
        batch_size = min(1000, total_combinations)  # 每批次最多1000組參數
        num_batches = (total_combinations + batch_size - 1) // batch_size
        
        all_results = []
        total_valid = 0
        total_processed = 0
        
        try:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_combinations)
                current_batch = param_combinations[start_idx:end_idx]
                
                logger.info(f"處理第 {batch_idx + 1}/{num_batches} 批次，"
                          f"參數組合 {start_idx + 1} 到 {end_idx}")
                
                # 使用多進程處理當前批次
                with Pool(processes=max_processes) as pool:
                    batch_results = list(tqdm(
                        pool.imap(test_func, current_batch),
                        total=len(current_batch),
                        desc=f"批次 {batch_idx + 1} 進度"
                    ))
                
                # 過濾並保存有效結果
                valid_results = [r for r in batch_results if r]
                all_results.extend(valid_results)
                
                # 更新統計信息
                total_valid += len(valid_results)
                total_processed += len(current_batch)
                
                # 輸出當前進度
                success_rate = (len(valid_results) / len(current_batch)) * 100
                overall_success_rate = (total_valid / total_processed) * 100
                
                logger.info(f"批次 {batch_idx + 1} 完成:")
                logger.info(f"- 當前批次: 處理 {len(current_batch)} 組，"
                          f"有效 {len(valid_results)} 組 ({success_rate:.2f}%)")
                logger.info(f"- 總體進度: 處理 {total_processed}/{total_combinations} 組，"
                          f"有效 {total_valid} 組 ({overall_success_rate:.2f}%)")
                
                # 清理記憶體
                del batch_results, valid_results
                gc.collect()
                
        except Exception as e:
            logger.error(f"暴力搜索過程中發生錯誤: {e}")
            # 返回已收集的結果
            return all_results
        
        logger.info(f"暴力搜索完成，共找到 {len(all_results)} 個有效結果")
        return all_results

    def optimize(self, strategy_class, param_grid: Dict, mode: str = 'grid', n_iter: int = 100, 
                test_func: Optional[Callable] = None, param_combinations: Optional[List] = None, 
                max_processes: Optional[int] = None) -> Dict:
        """
        統一優化入口，包含錯誤處理和詳細日誌
        
        Args:
            strategy_class: 策略類
            param_grid: 參數網格
            mode: 優化模式 ('grid', 'random', 'brute')
            n_iter: 隨機搜索迭代次數
            test_func: 暴力搜索測試函數
            param_combinations: 暴力搜索參數組合
            max_processes: 最大進程數
        """
        start_time = datetime.now()
        logger.info(f"開始優化，模式：{mode}")
        logger.info(f"優化開始時間：{start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # 驗證輸入參數
            if not isinstance(param_grid, dict):
                raise ValueError("param_grid 必須是字典類型")
            
            if mode not in ['grid', 'random', 'brute']:
                raise ValueError(f"不支持的優化模式：{mode}")
            
            if mode == 'random' and n_iter <= 0:
                raise ValueError(f"隨機搜索的迭代次數必須大於0，當前值：{n_iter}")
            
            if mode == 'brute':
                if test_func is None:
                    raise ValueError("暴力搜索模式需要提供 test_func")
                if param_combinations is None:
                    raise ValueError("暴力搜索模式需要提供 param_combinations")
                if not callable(test_func):
                    raise ValueError("test_func 必須是可調用的函數")
                if not isinstance(param_combinations, list):
                    raise ValueError("param_combinations 必須是列表類型")
            
            # 記錄記憶體使用情況
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            logger.info(f"初始記憶體使用：{initial_memory:.2f} MB")
            
            # 執行優化
            results = {}
            if mode == 'grid':
                results = self.grid_search(strategy_class, param_grid)
            elif mode == 'random':
                results = self.random_search(strategy_class, param_grid, n_iter)
            elif mode == 'brute' and test_func is not None and param_combinations is not None:
                # 在這裡我們已經確保了 test_func 和 param_combinations 不為 None
                results = {
                    'results': self.brute_force(test_func, param_combinations, max_processes),
                    'method': 'Brute Force'
                }
            
            # 檢查結果
            if not results:
                logger.warning(f"{mode} 優化未找到有效結果")
                return {}
            
            # 記錄優化結果摘要
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_change = final_memory - initial_memory
            
            summary = {
                'optimization_info': {
                    'mode': mode,
                    'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'duration_seconds': duration,
                    'initial_memory_mb': initial_memory,
                    'final_memory_mb': final_memory,
                    'memory_change_mb': memory_change
                }
            }
            
            # 合併優化結果和摘要
            results.update(summary)
            
            # 輸出優化完成信息
            logger.info(f"優化完成，耗時：{duration:.2f} 秒")
            logger.info(f"記憶體使用變化：{memory_change:+.2f} MB")
            
            return results
            
        except Exception as e:
            logger.error(f"優化過程發生錯誤: {str(e)}")
            import traceback
            logger.error(f"錯誤詳情:\n{traceback.format_exc()}")
            return {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'optimization_info': {
                    'mode': mode,
                    'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'status': 'failed'
                }
            }
        finally:
            # 強制清理記憶體
            gc.collect()

    def create_report(self, results: Dict) -> str:
        """生成優化報告"""
        report = []
        report.append("# 策略優化報告")
        report.append(f"生成時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        if 'best_params' in results:
            report.append("## 最佳參數")
            for param, value in results['best_params'].items():
                report.append(f"- {param}: {value}")
            report.append("")
            
            if 'train_metrics' in results:
                report.append("### 訓練集績效")
                for metric, value in results['train_metrics'].items():
                    report.append(f"- {metric}: {value:.2f}")
                report.append("")
                
            if 'validation_metrics' in results:
                report.append("### 驗證集績效")
                for metric, value in results['validation_metrics'].items():
                    report.append(f"- {metric}: {value:.2f}")
                report.append("")
                
        elif 'results' in results:
            report.append("## 暴力搜索結果（前10名）")
            for r in results['results'][:10]:
                report.append(str(r))
                
        return "\n".join(report) 