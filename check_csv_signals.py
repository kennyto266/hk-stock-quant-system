import pandas as pd
import numpy as np
import json
import os

csv_path = 'data_output/csv/2800_HK_stock_data.csv'
params_path = 'signal_params.json'

# 預設參數
DEFAULT_PARAMS = {
    "rsi": {"period": 14, "overbought": 70, "oversold": 30},
    "rsi_ultimate": {"period": 21, "overbought": 75, "oversold": 25}
}

# 讀取參數
if os.path.exists(params_path):
    with open(params_path, 'r', encoding='utf-8') as f:
        params = json.load(f)
else:
    params = DEFAULT_PARAMS
    print(f"【警告】找不到 {params_path}，使用預設參數 {DEFAULT_PARAMS}")

# 讀取資料
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"找不到 {csv_path}")
df = pd.read_csv(csv_path)

# --- RSI 快速體園 ---
rsi_period = params['rsi'].get('period', 14)
rsi_overbought = params['rsi'].get('overbought', 70)
rsi_oversold = params['rsi'].get('oversold', 30)
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
rs = gain / (loss + 1e-9)
rsi = 100 - (100 / (1 + rs))
df['rsi'] = rsi
# 持續持有信號
rsi_signal = np.zeros(len(df))
position = 0
rsi_arr = np.asarray(rsi)
for i in range(1, len(df)):
    val = rsi_arr[i]
    if not np.isnan(val) and val < rsi_oversold:
        position = 1
    elif not np.isnan(val) and val > rsi_overbought:
        position = -1
    rsi_signal[i] = position
df['rsi_strategy_signal'] = rsi_signal

# --- RSI 終極商洞 ---
rsi_ult_period = params['rsi_ultimate'].get('period', 21)
rsi_ult_overbought = params['rsi_ultimate'].get('overbought', 75)
rsi_ult_oversold = params['rsi_ultimate'].get('oversold', 25)
delta_ult = df['Close'].diff()
gain_ult = (delta_ult.where(delta_ult > 0, 0)).rolling(window=rsi_ult_period).mean()
loss_ult = (-delta_ult.where(delta_ult < 0, 0)).rolling(window=rsi_ult_period).mean()
rs_ult = gain_ult / (loss_ult + 1e-9)
rsi_ult = 100 - (100 / (1 + rs_ult))
df['rsi_ultimate'] = rsi_ult
# 持續持有信號
rsi_ult_signal = np.zeros(len(df))
position = 0
rsi_ult_arr = np.asarray(rsi_ult)
for i in range(1, len(df)):
    val = rsi_ult_arr[i]
    if not np.isnan(val) and val < rsi_ult_oversold:
        position = 1
    elif not np.isnan(val) and val > rsi_ult_overbought:
        position = -1
    rsi_ult_signal[i] = position
df['rsi_ultimate_signal'] = rsi_ult_signal

# --- MACD ---
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
macd = ema12 - ema26
macd_signal = macd.ewm(span=9, adjust=False).mean()
df['macd'] = macd
df['macd_signal'] = macd_signal
df['macd_strategy_signal'] = np.where(macd > macd_signal, 1, -1)

# --- 布林帶 ---
ma20 = df['Close'].rolling(window=20).mean()
std20 = df['Close'].rolling(window=20).std()
boll_upper = ma20 + 2 * std20
boll_lower = ma20 - 2 * std20
df['bollinger_upper'] = boll_upper
df['bollinger_lower'] = boll_lower
df['bollinger_strategy_signal'] = np.where(df['Close'] > boll_upper, -1, np.where(df['Close'] < boll_lower, 1, 0))

# --- KDJ ---
low_min = df['Low'].rolling(window=9).min()
high_max = df['High'].rolling(window=9).max()
rsv = (df['Close'] - low_min) / (high_max - low_min + 1e-9) * 100
df['kdj_k'] = rsv.ewm(com=2).mean()
df['kdj_d'] = df['kdj_k'].ewm(com=2).mean()
df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
df['kdj_strategy_signal'] = np.where(df['kdj_j'] > 80, -1, np.where(df['kdj_j'] < 20, 1, 0))

# --- Stochastic mil8 ---
low14 = df['Low'].rolling(window=14).min()
high14 = df['High'].rolling(window=14).max()
stoch_k = 100 * (df['Close'] - low14) / (high14 - low14 + 1e-9)
stoch_d = stoch_k.rolling(window=3).mean()
df['stochastic_k'] = stoch_k
df['stochastic_d'] = stoch_d
df['stochastic_mil8_signal'] = np.where(stoch_k > 80, -1, np.where(stoch_k < 20, 1, 0))

# --- 綜合策略（簡單加總所有信號）---
signal_cols = [
    'rsi_strategy_signal',
    'rsi_ultimate_signal',
    'macd_strategy_signal',
    'bollinger_strategy_signal',
    'kdj_strategy_signal',
    'stochastic_mil8_signal'
]
df['combined_strategy_signal'] = np.sign(df[signal_cols].sum(axis=1))

# --- 寫回 csv ---
print('【寫入所有 *_strategy_signal 欄位到 csv】')
df.to_csv(csv_path, index=False)

# 檢查所有 *_signal 欄位
print('【所有欄位名稱】')
print(list(df.columns))
print('\n【前 5 行數據】')
print(df.head())

signal_cols = [col for col in df.columns if col.endswith('_signal')]
if not signal_cols:
    print('\n【警告】找不到任何 *_signal 欄位！')
else:
    print(f'\n【發現 {len(signal_cols)} 個 *_signal 欄位】')
    for col in signal_cols:
        all_zero = (df[col].fillna(0) == 0).all()
        all_nan = df[col].isna().all()
        print(f'欄位: {col} | 全為 0: {all_zero} | 全為 NaN: {all_nan} | 非零數量: {(df[col].fillna(0) != 0).sum()}') 