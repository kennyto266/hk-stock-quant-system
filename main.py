"""
港股量化分析系統 - 主程式
自動重建：數據下載、策略回測、圖表輸出、Dashboard 啟動
"""

import os
import sys
from datetime import datetime
from data_handler import DataFetcher
from strategies import run_analysis, run_brute_force, run_all_strategies, run_integrated_strategies, \
    run_macd_strategy, run_sma_ema_strategy, run_bollinger_strategy, run_kd_strategy, run_momentum_strategy
import subprocess

# 自動檢查並安裝必要依賴
REQUIRED_PACKAGES = [
    'pandas', 'dash', 'plotly', 'dash-bootstrap-components'
]
for pkg in REQUIRED_PACKAGES:
    try:
        __import__(pkg.replace('-', '_'))
    except ImportError:
        print(f'未發現 {pkg}，自動安裝中...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])

# 數據下載流程
STOCK_LIST = [
    "2800.HK", "0005.HK", "0700.HK", "9988.HK", "0941.HK", "0388.HK", "1299.HK", "3690.HK", "0001.HK", "0016.HK"
]
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime('%Y-%m-%d')

def download_stock_data():
    fetcher = DataFetcher()
    for symbol in STOCK_LIST:
        csv_dir = os.path.join("data_output", "csv")
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, f"{symbol.replace('.', '_')}_{START_DATE}_{END_DATE}.csv")
        if os.path.exists(csv_path):
            print(f"{symbol} 已有本地csv，直接讀取 {csv_path}")
            continue
        print(f"下載 {symbol} 數據...")
        df = fetcher.get_yahoo_finance_data(symbol, START_DATE, END_DATE)
        if df is not None and not df.empty:
            df.to_csv(csv_path)
            print(f"{symbol} 數據下載完成，已保存到 {csv_path}")
        else:
            print(f"{symbol} 數據為空或下載失敗")

# 策略回測與優化

def run_all_backtests(stock_list=None):
    from strategies import run_analysis, run_brute_force, run_macd_strategy, run_sma_ema_strategy, run_bollinger_strategy, run_kd_strategy, run_momentum_strategy, run_all_strategies, run_integrated_strategies
    print("單一標的快速掃描回測...")
    run_analysis()
    print("RSI暴力優化回測...")
    run_brute_force(stock_list)
    print("MACD策略回測...")
    run_macd_strategy(stock_list)
    print("SMA/EMA策略回測...")
    run_sma_ema_strategy(stock_list)
    print("布林通道策略回測...")
    run_bollinger_strategy(stock_list)
    print("KD策略回測...")
    run_kd_strategy(stock_list)
    print("動能策略回測...")
    run_momentum_strategy(stock_list)
    print("多策略整合優化...")
    run_all_strategies()
    print("所有策略網格優化...")
    run_integrated_strategies()

# 啟動 Dashboard

# 載入多策略對比曲線並繪製
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html

def launch_rsi_dashboard():
    csv_dir = os.path.join("data_output", "csv")
    symbol = "9988.HK"
    strategies = [
        ("RSI", f"{symbol.replace('.', '_')}_rsi_vs_stock.csv", "cum_strategy", f"{symbol.replace('.', '_')}_rsi_params.txt"),
        ("MACD", f"{symbol.replace('.', '_')}_macd_vs_stock.csv", "cum_macd", None),
        ("SMAEMA", f"{symbol.replace('.', '_')}_smaema_vs_stock.csv", "cum_smaema", None),
        ("Bollinger", f"{symbol.replace('.', '_')}_boll_vs_stock.csv", "cum_boll", None),
        ("KD", f"{symbol.replace('.', '_')}_kd_vs_stock.csv", "cum_kd", None),
        ("Momentum", f"{symbol.replace('.', '_')}_momentum_vs_stock.csv", "cum_momentum", None),
    ]
    # 收集每個策略的 SR/MDD
    param_texts = []
    traces = []
    stock_trace = None
    for strat_name, csv_file, strat_col, param_file in strategies:
        csv_path = os.path.join(csv_dir, csv_file)
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        # 讀取 SR/MDD
        sr, mdd = None, None
        if param_file and os.path.exists(os.path.join(csv_dir, param_file)):
            with open(os.path.join(csv_dir, param_file), "r", encoding="utf-8") as f:
                for line in f:
                    if "夏普比率" in line:
                        sr = line.split("夏普比率=")[-1].split("，")[0].strip()
                    if "最大回撤" in line:
                        mdd = line.split("最大回撤=")[-1].split("%")[-2].replace("，", "").strip()
        # 其他策略SR/MDD直接計算
        if strat_col in df.columns:
            returns = df[strat_col].pct_change().dropna()
            if not returns.empty:
                sr = f"{(returns.mean()/returns.std()*252**0.5):.2f}" if returns.std() != 0 else "0"
                mdd = f"{(1 - df[strat_col].div(df[strat_col].cummax()).min()):.2%}"
            param_texts.append(f"{strat_name}：SR={sr}，MDD={mdd}")
            traces.append(go.Scatter(x=df.index, y=df[strat_col], mode='lines', name=f'{strat_name}策略累積收益'))
        if stock_trace is None and 'cum_stock' in df.columns:
            stock_trace = go.Scatter(x=df.index, y=df['cum_stock'], mode='lines', name='股票累積收益')
    if stock_trace:
        traces.append(stock_trace)
    print("Dashboard 已啟動，請於瀏覽器開啟：http://localhost:8050/")
    app = Dash(__name__)
    app.layout = html.Div([
        html.H1(f'{symbol} 多策略收益對比'),
        html.P(" | ".join(param_texts), style={"color": "#444", "fontSize": "18px"}),
        dcc.Graph(
            figure={
                'data': traces,
                'layout': go.Layout(title=f'{symbol} 多策略收益對比', xaxis={'title': '日期'}, yaxis={'title': '累積收益'})
            }
        )
    ])
    app.run(debug=True, port=8050)

def launch_dashboard():
    print("啟動新版 Modern Dashboard...")
    subprocess.Popen([sys.executable, "modern_dashboard.py"])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 有股票代號參數，僅回測該股票，不啟動 dashboard
        stock_code = sys.argv[1]
        STOCK_LIST = [stock_code]
        download_stock_data()
        run_all_backtests(STOCK_LIST)
    else:
        print("=== 港股量化分析系統 啟動 ===")
        download_stock_data()
        run_all_backtests(STOCK_LIST)
        launch_dashboard()

