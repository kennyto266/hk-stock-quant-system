"""
港股量化分析系統 - 策略模組（自動重建）
提供最小可用的回測入口，確保 main.py 能正確執行
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from multiprocessing import Pool, cpu_count

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def evaluate_rsi_params(args):
    df, period, ob, os_ = args
    if ob <= os_:
        return None
    try:
        prices = df['Close'] if 'Close' in df.columns else df['close']
        rsi = calculate_rsi(prices, period)
        signals = pd.Series(0, index=df.index)
        signals[rsi < os_] = 1
        signals[rsi > ob] = -1
        returns = signals.shift(1) * prices.pct_change()
        returns = returns[~returns.isna()]
        if len(returns) < 2:
            return None
        annual_return = np.mean(returns) * 252
        annual_volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        max_drawdown = abs(drawdowns.min())
        return {
            'period': period,
            'overbought': ob,
            'oversold': os_,
            'sharpe_ratio': float(sharpe_ratio),
            'annual_return': float(annual_return),
            'max_drawdown': float(max_drawdown),
            'total_trades': int(len(signals[signals != 0]))
        }
    except Exception as e:
        print(f"評估參數組合時出錯: period={period}, ob={ob}, os={os_}, error={str(e)}")
        return None

def run_analysis():
    print("[策略] 單一標的快速掃描回測（範例）")
    # 這裡可放入真實回測邏輯
    return True
        
def run_brute_force(stock_list=None):
    print("[策略] RSI暴力優化回測（多股票自動化）")
    if stock_list is None:
        stock_list = ["9988.HK"]
    csv_dir = os.path.join("data_output", "csv")
    for symbol in stock_list:
        csv_path = os.path.join(csv_dir, f"{symbol.replace('.', '_')}_2020-01-01_{datetime.now().strftime('%Y-%m-%d')}.csv")
        param_path = os.path.join(csv_dir, f"{symbol.replace('.', '_')}_rsi_params.txt")
        if not os.path.exists(csv_path):
            print(f"找不到 {csv_path}，請先下載數據")
            continue
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        period_range = list(range(0, 301, 5))
        ob_range = list(range(50, 91, 10))
        os_range = list(range(10, 51, 10))
        param_combinations = [(df, p, ob, os_) for p in period_range for ob in ob_range for os_ in os_range]
        max_processes = min(16, cpu_count())
        print(f"{symbol} 使用進程數: {max_processes}")
        with Pool(processes=max_processes) as pool:
            results = pool.map(evaluate_rsi_params, param_combinations)
        valid_results = [r for r in results if r is not None]
        valid_results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
        print(f"{symbol} 有效參數組合數: {len(valid_results)}")
        if valid_results:
            best = valid_results[0]
            print(f"{symbol} 最佳參數: {best}")
            param_text = (
                f"回測標的：{symbol}\n"
                f"最佳RSI參數：週期(period)={best['period']}，超買(overbought)={best['overbought']}，超賣(oversold)={best['oversold']}\n"
                f"夏普比率={best['sharpe_ratio']:.2f}，年化收益={best['annual_return']*100:.2f}% ，最大回撤={best['max_drawdown']*100:.2f}% ，總交易次數={best['total_trades']}\n"
                f"優化範圍：週期 {period_range}，超買 {ob_range}，超賣 {os_range} (步距分別為5, 10, 10)\n"
                f"多進程優化，進程數={max_processes}"
            )
            with open(param_path, "w", encoding="utf-8") as f:
                f.write(param_text)
            period, ob, os_ = best['period'], best['overbought'], best['oversold']
            prices = df['Close'] if 'Close' in df.columns else df['close']
            rsi = calculate_rsi(prices, period)
            signals = pd.Series(0, index=df.index)
            signals[rsi < os_] = 1
            signals[rsi > ob] = -1
            strategy_return = signals.shift(1) * prices.pct_change()
            stock_return = prices.pct_change()
            df['cum_strategy'] = (1 + strategy_return.fillna(0)).cumprod()
            df['cum_stock'] = (1 + stock_return.fillna(0)).cumprod()
            out = df[['cum_strategy', 'cum_stock']].dropna()
            out_path = os.path.join(csv_dir, f"{symbol.replace('.', '_')}_rsi_vs_stock.csv")
            out.to_csv(out_path)
            print(f"{symbol} 策略與股票累積收益已輸出: {out_path}")
            print(f"{symbol} 最佳參數已寫入: {param_path}")
        else:
            print(f"{symbol} 無有效參數組合")

def run_all_strategies():
    print("[策略] 多策略整合優化（範例）")
    # 這裡可放入真實回測邏輯
    return True
        
def run_integrated_strategies():
    print("[策略] 所有策略網格優化（範例）")
    # 這裡可放入真實回測邏輯
    return True
        
def run_macd_strategy(stock_list=None):
    print("[策略] MACD 策略回測（多股票自動化）")
    if stock_list is None:
        stock_list = ["9988.HK"]
    csv_dir = os.path.join("data_output", "csv")
    for symbol in stock_list:
        csv_path = os.path.join(csv_dir, f"{symbol.replace('.', '_')}_2020-01-01_{datetime.now().strftime('%Y-%m-%d')}.csv")
        out_path = os.path.join(csv_dir, f"{symbol.replace('.', '_')}_macd_vs_stock.csv")
        if not os.path.exists(csv_path):
            print(f"找不到 {csv_path}，請先下載數據")
            continue
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        close = df['Close'] if 'Close' in df.columns else df['close']
        short_ema = close.ewm(span=12, adjust=False).mean()
        long_ema = close.ewm(span=26, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=9, adjust=False).mean()
        df['macd_signal'] = 0
        df.loc[macd > signal, 'macd_signal'] = 1
        df.loc[macd < signal, 'macd_signal'] = -1
        strategy_return = df['macd_signal'].shift(1) * close.pct_change()
        stock_return = close.pct_change()
        df['cum_macd'] = (1 + strategy_return.fillna(0)).cumprod()
        df['cum_stock'] = (1 + stock_return.fillna(0)).cumprod()
        out = df[['cum_macd', 'cum_stock']].dropna()
        out.to_csv(out_path)
        print(f"{symbol} MACD 策略與股票累積收益已輸出: {out_path}")

def run_sma_ema_strategy(stock_list=None):
    print("[策略] SMA/EMA 均線交叉策略回測（多股票自動化）")
    if stock_list is None:
        stock_list = ["9988.HK"]
    csv_dir = os.path.join("data_output", "csv")
    for symbol in stock_list:
        csv_path = os.path.join(csv_dir, f"{symbol.replace('.', '_')}_2020-01-01_{datetime.now().strftime('%Y-%m-%d')}.csv")
        out_path = os.path.join(csv_dir, f"{symbol.replace('.', '_')}_smaema_vs_stock.csv")
        if not os.path.exists(csv_path):
            print(f"找不到 {csv_path}，請先下載數據")
            continue
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        close = df['Close'] if 'Close' in df.columns else df['close']
        sma = close.rolling(window=20).mean()
        ema = close.ewm(span=50, adjust=False).mean()
        df['smaema_signal'] = 0
        df.loc[sma > ema, 'smaema_signal'] = 1
        df.loc[sma < ema, 'smaema_signal'] = -1
        strategy_return = df['smaema_signal'].shift(1) * close.pct_change()
        stock_return = close.pct_change()
        df['cum_smaema'] = (1 + strategy_return.fillna(0)).cumprod()
        df['cum_stock'] = (1 + stock_return.fillna(0)).cumprod()
        out = df[['cum_smaema', 'cum_stock']].dropna()
        out.to_csv(out_path)
        print(f"{symbol} SMA/EMA 策略與股票累積收益已輸出: {out_path}")

def run_bollinger_strategy(stock_list=None):
    print("[策略] 布林通道策略回測（多股票自動化）")
    if stock_list is None:
        stock_list = ["9988.HK"]
    csv_dir = os.path.join("data_output", "csv")
    for symbol in stock_list:
        csv_path = os.path.join(csv_dir, f"{symbol.replace('.', '_')}_2020-01-01_{datetime.now().strftime('%Y-%m-%d')}.csv")
        out_path = os.path.join(csv_dir, f"{symbol.replace('.', '_')}_boll_vs_stock.csv")
        if not os.path.exists(csv_path):
            print(f"找不到 {csv_path}，請先下載數據")
            continue
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        close = df['Close'] if 'Close' in df.columns else df['close']
        ma = close.rolling(window=20).mean()
        std = close.rolling(window=20).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        df['boll_signal'] = 0
        df.loc[close < lower, 'boll_signal'] = 1
        df.loc[close > upper, 'boll_signal'] = -1
        strategy_return = df['boll_signal'].shift(1) * close.pct_change()
        stock_return = close.pct_change()
        df['cum_boll'] = (1 + strategy_return.fillna(0)).cumprod()
        df['cum_stock'] = (1 + stock_return.fillna(0)).cumprod()
        out = df[['cum_boll', 'cum_stock']].dropna()
        out.to_csv(out_path)
        print(f"{symbol} 布林通道策略與股票累積收益已輸出: {out_path}")

def run_kd_strategy(stock_list=None):
    print("[策略] KD 隨機指標策略回測（多股票自動化）")
    if stock_list is None:
        stock_list = ["9988.HK"]
    csv_dir = os.path.join("data_output", "csv")
    for symbol in stock_list:
        csv_path = os.path.join(csv_dir, f"{symbol.replace('.', '_')}_2020-01-01_{datetime.now().strftime('%Y-%m-%d')}.csv")
        out_path = os.path.join(csv_dir, f"{symbol.replace('.', '_')}_kd_vs_stock.csv")
        if not os.path.exists(csv_path):
            print(f"找不到 {csv_path}，請先下載數據")
            continue
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        low = df['Low'] if 'Low' in df.columns else df['low']
        high = df['High'] if 'High' in df.columns else df['high']
        close = df['Close'] if 'Close' in df.columns else df['close']
        low_min = low.rolling(window=9).min()
        high_max = high.rolling(window=9).max()
        rsv = (close - low_min) / (high_max - low_min) * 100
        k = rsv.ewm(com=2).mean()
        d = k.ewm(com=2).mean()
        df['kd_signal'] = 0
        df.loc[k < 20, 'kd_signal'] = 1
        df.loc[k > 80, 'kd_signal'] = -1
        strategy_return = df['kd_signal'].shift(1) * close.pct_change()
        stock_return = close.pct_change()
        df['cum_kd'] = (1 + strategy_return.fillna(0)).cumprod()
        df['cum_stock'] = (1 + stock_return.fillna(0)).cumprod()
        out = df[['cum_kd', 'cum_stock']].dropna()
        out.to_csv(out_path)
        print(f"{symbol} KD 隨機指標策略與股票累積收益已輸出: {out_path}")

def run_momentum_strategy(stock_list=None):
    print("[策略] 動能策略回測（多股票自動化）")
    if stock_list is None:
        stock_list = ["9988.HK"]
    csv_dir = os.path.join("data_output", "csv")
    for symbol in stock_list:
        csv_path = os.path.join(csv_dir, f"{symbol.replace('.', '_')}_2020-01-01_{datetime.now().strftime('%Y-%m-%d')}.csv")
        out_path = os.path.join(csv_dir, f"{symbol.replace('.', '_')}_momentum_vs_stock.csv")
        if not os.path.exists(csv_path):
            print(f"找不到 {csv_path}，請先下載數據")
            continue
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        close = df['Close'] if 'Close' in df.columns else df['close']
        momentum = close - close.shift(10)
        df['momentum_signal'] = 0
        df.loc[momentum > 0, 'momentum_signal'] = 1
        df.loc[momentum < 0, 'momentum_signal'] = -1
        strategy_return = df['momentum_signal'].shift(1) * close.pct_change()
        stock_return = close.pct_change()
        df['cum_momentum'] = (1 + strategy_return.fillna(0)).cumprod()
        df['cum_stock'] = (1 + stock_return.fillna(0)).cumprod()
        out = df[['cum_momentum', 'cum_stock']].dropna()
        out.to_csv(out_path)
        print(f"{symbol} 動能策略與股票累積收益已輸出: {out_path}")
