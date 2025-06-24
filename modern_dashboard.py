import sys
import subprocess
import os
import glob
import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go

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

CSV_PATH = "data_output/csv"

STRATEGIES = [
    ("RSI", "cum_strategy"),
    ("MACD", "cum_macd"),
    ("SMAEMA", "cum_smaema"),
    ("Bollinger", "cum_boll"),
    ("KD", "cum_kd"),
    ("Momentum", "cum_momentum"),
]

def scan_strategy_files():
    files = glob.glob(os.path.join(CSV_PATH, "*_vs_stock.csv"))
    strategies = {}
    stock_file = None
    for f in files:
        fname = os.path.basename(f)
        if "rsi" in fname.lower():
            strategies["RSI"] = f
        elif "macd" in fname.lower():
            strategies["MACD"] = f
        elif "smaema" in fname.lower():
            strategies["SMAEMA"] = f
        elif "boll" in fname.lower():
            strategies["Bollinger"] = f
        elif "kd" in fname.lower():
            strategies["KD"] = f
        elif "momentum" in fname.lower():
            strategies["Momentum"] = f
        # 儲存一個有 cum_stock 欄位的檔案作為股票基準
        if stock_file is None:
            try:
                df = pd.read_csv(f, nrows=2)
                if "cum_stock" in df.columns:
                    stock_file = f
            except Exception:
                pass
    return strategies, stock_file

def read_csv_compat(filepath):
    df = pd.read_csv(filepath, index_col=0)
    if 'Date' not in df.columns:
        df = df.reset_index().rename(columns={df.index.name or 'index': 'Date'})
    return df

def calc_metrics(df):
    sr = df['strategy_returns'].mean() / df['strategy_returns'].std() * (252 ** 0.5) if df['strategy_returns'].std() > 0 else 0
    mdd = ((df['equity'] - df['equity'].cummax()) / df['equity'].cummax()).min()
    ann_return = (df['equity'].iloc[-1] ** (252 / len(df)) - 1) if len(df) > 0 else 0
    return {
        "Sharpe Ratio": f"{sr:.3f}",
        "Max Drawdown": f"{mdd:.2%}",
        "Annual Return": f"{ann_return:.2%}"
    }

strategies, stock_file = scan_strategy_files()
default_strategy = list(strategies.keys())[0] if strategies else None

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "港股量化分析系統 - Modern Dashboard"

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("策略選擇", style={"color": "#fff"}),
            html.Div([
                dcc.Input(id="stock-input", type="text", placeholder="輸入股票代號 (如 9988.HK)", debounce=True, style={"width": "80%", "marginBottom": "10px"}),
                html.Button("回測", id="run-backtest-btn", n_clicks=0, style={"marginLeft": "10px"}),
            ], style={"marginBottom": "20px"}),
            dbc.Checklist(
                id="strategy-select",
                options=[{"label": s[0], "value": s[1]} for s in STRATEGIES],
                value=[s[1] for s in STRATEGIES],  # 預設全選
                inline=False,
                switch=True,
                style={"color": "#fff"}
            ),
            html.Hr(),
            html.H4("策略表現", style={"color": "#fff"}),
            html.Div(id="strategy-metrics"),
        ], width=3, style={"background": "#222", "padding": "20px", "minHeight": "100vh"}),
        dbc.Col([
            html.Div([
                html.H2("港股量化分析系統 - 互動式 Dashboard", style={"background": "#1976d2", "color": "white", "padding": "10px", "borderRadius": "8px"}),
            ], style={"marginBottom": "20px"}),
            dcc.Graph(id="main-graph"),
        ], width=9)
    ])
], fluid=True)

# 股票代號狀態儲存
from dash import ctx
import time

@app.callback(
    Output("main-graph", "figure"),
    Output("strategy-metrics", "children"),
    Output("stock-input", "value"),
    Input("strategy-select", "value"),
    Input("run-backtest-btn", "n_clicks"),
    State("stock-input", "value"),
    prevent_initial_call=True
)
def update_graph(selected_strategies, n_clicks, stock_code):
    # 若有按下回測按鈕，觸發回測
    triggered = ctx.triggered_id
    if triggered == "run-backtest-btn" and stock_code:
        import subprocess
        import sys
        # 執行 main.py 並傳入股票代號，只回測不啟動 dashboard
        try:
            subprocess.run([sys.executable, "main.py", stock_code], check=True)
            time.sleep(2)  # 等待回測產生 csv
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"回測失敗: {e}", showarrow=False)
            return fig, [html.Div(f"回測失敗: {e}", style={"color": "#f88"})], stock_code
    # 根據股票代號切換 csv
    code = stock_code if stock_code else "9988.HK"
    # 重新掃描該股票所有策略 csv
    def scan_csv_for_code(code):
        files = glob.glob(os.path.join(CSV_PATH, f"{code.replace('.', '_')}_*_vs_stock.csv"))
        strategies = {}
        stock_file = None
        for f in files:
            fname = os.path.basename(f)
            if "rsi" in fname.lower():
                strategies["RSI"] = f
            elif "macd" in fname.lower():
                strategies["MACD"] = f
            elif "smaema" in fname.lower():
                strategies["SMAEMA"] = f
            elif "boll" in fname.lower():
                strategies["Bollinger"] = f
            elif "kd" in fname.lower():
                strategies["KD"] = f
            elif "momentum" in fname.lower():
                strategies["Momentum"] = f
            if stock_file is None:
                try:
                    df = pd.read_csv(f, nrows=2)
                    if "cum_stock" in df.columns:
                        stock_file = f
                except Exception:
                    pass
        return strategies, stock_file
    strategies, stock_file = scan_csv_for_code(code)
    traces = []
    metrics = []
    for strat_name, strat_col in STRATEGIES:
        if strat_name in strategies and strat_col in selected_strategies:
            try:
                df = read_csv_compat(strategies[strat_name])
                if strat_col in df.columns and "Date" in df.columns:
                    traces.append(go.Scatter(x=df['Date'], y=df[strat_col], mode='lines', name=f"{strat_name}"))
                    returns = df[strat_col].pct_change().dropna()
                    sr = f"{(returns.mean()/returns.std()*252**0.5):.2f}" if returns.std() != 0 else "0"
                    mdd = f"{(1 - df[strat_col].div(df[strat_col].cummax()).min()):.2%}"
                    metrics.append(html.Div(f"{strat_name}：SR={sr}，MDD={mdd}", style={"color": "#fff", "marginBottom": "8px"}))
            except Exception as e:
                metrics.append(html.Div(f"{strat_name} 資料讀取錯誤: {e}", style={"color": "#f88"}))
    if stock_file:
        try:
            df_stock = read_csv_compat(stock_file)
            if "cum_stock" in df_stock.columns and "Date" in df_stock.columns:
                traces.append(go.Scatter(x=df_stock['Date'], y=df_stock["cum_stock"], mode='lines', name="股票累積收益"))
        except Exception as e:
            metrics.append(html.Div(f"股票基準資料讀取錯誤: {e}", style={"color": "#f88"}))
    if not traces:
        fig = go.Figure()
        fig.add_annotation(text="找不到任何策略資料，請確認 csv 檔案！", showarrow=False)
    else:
        fig = go.Figure(data=traces)
        fig.update_layout(title=f"{code} 多策略收益對比", xaxis_title="日期", yaxis_title="累積收益", plot_bgcolor="#222", paper_bgcolor="#222", font_color="#fff")
    return fig, metrics, code

if __name__ == "__main__":
    app.run(debug=True, port=8050) 