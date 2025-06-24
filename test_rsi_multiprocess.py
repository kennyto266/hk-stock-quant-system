import pandas as pd
import numpy as np
from datetime import datetime
import time
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple, Optional, Union, Any, cast
import yfinance as yf

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """計算RSI指標"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = pd.Series(100 - (100 / (1 + rs)), index=prices.index)
    return rsi

def evaluate_rsi_params(args: Tuple[pd.DataFrame, int, int, int]) -> Optional[Dict[str, Union[int, float]]]:
    """評估單個RSI參數組合"""
    df, period, ob, os = args
    if ob <= os:  # 超買閾值必須大於超賣閾值
        return None
        
    try:
        # 計算RSI
        prices = pd.Series(df['Close'].values, index=df.index)
        rsi = calculate_rsi(prices, period)
        
        # 生成交易信號
        signals = pd.Series(0, index=df.index)
        signals[rsi < os] = 1  # 買入信號
        signals[rsi > ob] = -1  # 賣出信號
        
        # 計算策略收益
        returns = signals.shift(1) * df['Close'].pct_change()
        returns = returns[~returns.isna()]
        
        if len(returns) < 2:
            return None
            
        # 計算績效指標
        annual_return = np.mean(returns) * 252
        annual_volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        
        # 計算最大回撤
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        max_drawdown = abs(drawdowns.min())
        
        return {
            'period': period,
            'overbought': ob,
            'oversold': os,
            'sharpe_ratio': float(sharpe_ratio),
            'annual_return': float(annual_return),
            'max_drawdown': float(max_drawdown),
            'total_trades': int(len(signals[signals != 0]))
        }
    except Exception as e:
        print(f"評估參數組合時出錯: period={period}, ob={ob}, os={os}, error={str(e)}")
        return None

def main():
    # 生成測試數據
    print("生成測試數據...")
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    prices = np.random.randn(len(dates)).cumsum() + 100
    df = pd.DataFrame({
        'Close': prices,
        'Open': prices + np.random.randn(len(dates)) * 0.1,
        'High': prices + np.abs(np.random.randn(len(dates)) * 0.2),
        'Low': prices - np.abs(np.random.randn(len(dates)) * 0.2),
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    if df.empty:
        print("無法生成數據")
        return
        
    df = cast(pd.DataFrame, df)  # 確保型別檢查器知道df是DataFrame
    
    # 設置參數範圍
    periods = range(5, 301, 5)  # RSI週期：5-300，步距5
    overbought = range(50, 91, 5)  # 超買閾值：50-90，步距5
    oversold = range(10, 51, 5)  # 超賣閾值：10-50，步距5
    
    # 準備參數組合
    param_combinations = [(df, p, ob, os) 
                         for p in periods 
                         for ob in overbought 
                         for os in oversold]
    
    print(f"總參數組合數: {len(param_combinations)}")
    print(f"可用CPU核心數: {cpu_count()}")
    max_processes = min(32, cpu_count())
    print(f"使用進程數: {max_processes}")
    
    # 使用多進程進行參數優化
    print("\n開始多進程優化...")
    start_time = time.time()
    
    with Pool(processes=max_processes) as pool:
        results = pool.map(evaluate_rsi_params, param_combinations)
    
    # 過濾有效結果並排序
    valid_results = [r for r in results if r is not None]
    valid_results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 輸出結果
    print(f"\n優化完成！用時: {total_time:.2f} 秒")
    print(f"有效參數組合數: {len(valid_results)}")
    
    if valid_results:
        print("\n前10個最佳參數組合:")
        for i, result in enumerate(valid_results[:10], 1):
            print(f"\n{i}. 參數組合:")
            print(f"   - RSI週期: {result['period']}天")
            print(f"   - 超買閾值: {result['overbought']}")
            print(f"   - 超賣閾值: {result['oversold']}")
            print(f"   績效指標:")
            print(f"   - 夏普比率: {result['sharpe_ratio']:.2f}")
            print(f"   - 年化收益: {result['annual_return']*100:.2f}%")
            print(f"   - 最大回撤: {result['max_drawdown']*100:.2f}%")
            print(f"   - 總交易次數: {result['total_trades']}")

if __name__ == "__main__":
    main() 