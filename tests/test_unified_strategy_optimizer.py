import pytest
import pandas as pd
import numpy as np
from unified_strategy_optimizer import UnifiedStrategyOptimizer

class DummyStrategy:
    def __init__(self, param1=1, param2=2):
        self.param1 = param1
        self.param2 = param2
    def generate_signals(self, data):
        return pd.Series(np.where(data['Close'] > data['Close'].shift(1), 1, 0), index=data.index)
    def calculate_returns(self, data, signals):
        return pd.Series(np.random.normal(0, 0.01, len(data)), index=data.index)
    def calculate_performance_metrics(self, returns):
        return {'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-6)}

def brute_force_test_func(params):
    return {'params': params, 'sharpe_ratio': np.random.rand()}

def test_grid_search():
    data = pd.DataFrame({'Close': np.random.rand(100)})
    optimizer = UnifiedStrategyOptimizer(data)
    param_grid = {'param1': [1, 2], 'param2': [2, 3]}
    result = optimizer.optimize(DummyStrategy, param_grid, mode='grid')
    assert 'best_params' in result
    assert 'train_metrics' in result
    assert 'validation_metrics' in result

def test_random_search():
    data = pd.DataFrame({'Close': np.random.rand(100)})
    optimizer = UnifiedStrategyOptimizer(data)
    param_grid = {'param1': [1, 2], 'param2': [2, 3]}
    result = optimizer.optimize(DummyStrategy, param_grid, mode='random', n_iter=10)
    assert 'best_params' in result
    assert 'train_metrics' in result
    assert 'validation_metrics' in result

def test_brute_force():
    data = pd.DataFrame({'Close': np.random.rand(100)})
    optimizer = UnifiedStrategyOptimizer(data)
    param_combinations = [(1, 2), (2, 3)]
    result = optimizer.optimize(DummyStrategy, {}, mode='brute', test_func=brute_force_test_func, param_combinations=param_combinations)
    assert 'results' in result
    assert result['method'] == 'Brute Force'
