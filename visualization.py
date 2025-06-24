"""
港股量化分析系統 - 可視化模組
包含圖表生成、儀表板、報告輸出等功能
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import warnings
from datetime import datetime
import logging
try:
    from config import logger
except ImportError:
    logger = logging.getLogger(__name__)

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

class PlotlyCharts:
    """Plotly圖表生成器"""
    
    @staticmethod
    def create_price_chart(data: pd.DataFrame, title: str = "股價走勢圖") -> go.Figure:
        """
        創建股價走勢圖
        
        Args:
            data: 股票數據
            title: 圖表標題
            
        Returns:
            Plotly圖表對象
        """
        try:
            fig = go.Figure()
            
            # 添加收盤價線
            if 'Close' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name='收盤價',
                    line=dict(color='#1f77b4', width=2)
                ))
            
            # 添加移動平均線
            ma_columns = [col for col in data.columns if col.startswith('MA_')]
            colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            for i, ma_col in enumerate(ma_columns[:4]):
                if ma_col in data.columns:
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data[ma_col],
                        mode='lines',
                        name=ma_col,
                        line=dict(color=colors[i % len(colors)], width=1)
                    ))
            
            fig.update_layout(
                title=title,
                xaxis_title="日期",
                yaxis_title="價格 (HKD)",
                hovermode='x unified',
                template='plotly_white',
                height=600
            )
            
            return fig
        
        except Exception as e:
            logger.error(f"股價走勢圖創建失敗: {e}")
            return go.Figure()
    
    @staticmethod
    def create_technical_indicators_chart(data: pd.DataFrame) -> go.Figure:
        """
        創建技術指標圖表
        
        Args:
            data: 包含技術指標的數據
            
        Returns:
            Plotly子圖對象
        """
        try:
            # 創建子圖
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=('股價與布林帶', 'RSI', 'MACD', '成交量'),
                vertical_spacing=0.08,
                row_heights=[0.4, 0.2, 0.2, 0.2]
            )
            
            # 1. 股價與布林帶
            if 'Close' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index, y=data['Close'],
                    mode='lines', name='收盤價',
                    line=dict(color='blue')
                ), row=1, col=1)
            
            if all(col in data.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
                fig.add_trace(go.Scatter(
                    x=data.index, y=data['BB_Upper'],
                    mode='lines', name='布林帶上軌',
                    line=dict(color='red', dash='dash')
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=data.index, y=data['BB_Middle'],
                    mode='lines', name='布林帶中軌',
                    line=dict(color='orange')
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=data.index, y=data['BB_Lower'],
                    mode='lines', name='布林帶下軌',
                    line=dict(color='red', dash='dash')
                ), row=1, col=1)
            
            # 2. RSI
            if 'RSI' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index, y=data['RSI'],
                    mode='lines', name='RSI',
                    line=dict(color='purple')
                ), row=2, col=1)
                
                # 添加RSI超買超賣線
                fig.add_hline(y=70, line_dash="dash", line_color="red", 
                             annotation_text="超買", row="2", col="1")
                fig.add_hline(y=30, line_dash="dash", line_color="green", 
                             annotation_text="超賣", row="2", col="1")
            
            # 3. MACD
            if all(col in data.columns for col in ['MACD_MACD', 'MACD_Signal', 'MACD_Histogram']):
                fig.add_trace(go.Scatter(
                    x=data.index, y=data['MACD_MACD'],
                    mode='lines', name='MACD',
                    line=dict(color='blue')
                ), row=3, col=1)
                
                fig.add_trace(go.Scatter(
                    x=data.index, y=data['MACD_Signal'],
                    mode='lines', name='信號線',
                    line=dict(color='red')
                ), row=3, col=1)
                
                fig.add_trace(go.Bar(
                    x=data.index, y=data['MACD_Histogram'],
                    name='MACD柱狀圖',
                    marker_color='green'
                ), row=3, col=1)
            
            # 4. 成交量
            if 'Volume' in data.columns:
                fig.add_trace(go.Bar(
                    x=data.index, y=data['Volume'],
                    name='成交量',
                    marker_color='lightblue'
                ), row=4, col=1)
            
            fig.update_layout(
                title="技術指標綜合圖表",
                height=800,
                template='plotly_white',
                showlegend=True
            )
            
            return fig
        
        except Exception as e:
            logger.error(f"技術指標圖表創建失敗: {e}")
            return go.Figure()
    
    @staticmethod
    def create_strategy_performance_chart(performance_data: Dict[str, Dict]) -> go.Figure:
        """
        創建策略績效比較圖
        
        Args:
            performance_data: 策略績效數據
            
        Returns:
            Plotly圖表對象
        """
        try:
            if not performance_data:
                return go.Figure()
            
            # 提取績效指標
            strategies = list(performance_data.keys())
            metrics = ['total_return', 'annual_return', 'volatility', 'sharpe_ratio', 'max_drawdown']
            
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=('總收益率 (%)', '年化收益率 (%)', '波動率 (%)', 
                              '夏普比率', '最大回撤 (%)', '勝率 (%)'),
                specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
            )
            
            positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
            metric_names = ['total_return', 'annual_return', 'volatility', 
                           'sharpe_ratio', 'max_drawdown', 'win_rate']
            
            for i, metric in enumerate(metric_names):
                if i < len(positions):
                    values = []
                    for strategy in strategies:
                        if metric in performance_data[strategy] and performance_data[strategy][metric] is not None:
                            values.append(performance_data[strategy][metric])
                        else:
                            values.append(0)
                    
                    fig.add_trace(go.Bar(
                        x=strategies,
                        y=values,
                        name=metric,
                        showlegend=False
                    ), row=positions[i][0], col=positions[i][1])
            
            fig.update_layout(
                title="策略績效比較",
                height=600,
                template='plotly_white'
            )
            
            return fig
        
        except Exception as e:
            logger.error(f"策略績效圖表創建失敗: {e}")
            return go.Figure()
    
    @staticmethod
    def create_signals_chart(data: pd.DataFrame, signals: pd.DataFrame, 
                           strategy_name: str = "策略信號") -> go.Figure:
        """
        創建交易信號圖表
        
        Args:
            data: 股票數據
            signals: 交易信號
            strategy_name: 策略名稱
            
        Returns:
            Plotly圖表對象
        """
        try:
            fig = go.Figure()
            
            # 添加股價
            if 'Close' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name='收盤價',
                    line=dict(color='blue', width=2)
                ))
            
            # 添加買入信號
            buy_signals = signals[signals['signal'] == 1]
            if not buy_signals.empty:
                buy_prices = data.loc[buy_signals.index, 'Close'] if 'Close' in data.columns else []
                if len(buy_prices) > 0:
                    fig.add_trace(go.Scatter(
                        x=buy_signals.index,
                        y=buy_prices,
                        mode='markers',
                        name='買入信號',
                        marker=dict(color='green', size=10, symbol='triangle-up')
                    ))
            
            # 添加賣出信號
            sell_signals = signals[signals['signal'] == -1]
            if not sell_signals.empty:
                sell_prices = data.loc[sell_signals.index, 'Close'] if 'Close' in data.columns else []
                if len(sell_prices) > 0:
                    fig.add_trace(go.Scatter(
                        x=sell_signals.index,
                        y=sell_prices,
                        mode='markers',
                        name='賣出信號',
                        marker=dict(color='red', size=10, symbol='triangle-down')
                    ))
            
            fig.update_layout(
                title=f"{strategy_name} 交易信號",
                xaxis_title="日期",
                yaxis_title="價格 (HKD)",
                hovermode='x unified',
                template='plotly_white',
                height=600
            )
            
            return fig
        
        except Exception as e:
            logger.error(f"交易信號圖表創建失敗: {e}")
            return go.Figure()

class MatplotlibCharts:
    """Matplotlib圖表生成器"""
    
    @staticmethod
    def create_correlation_heatmap(data: pd.DataFrame, title: str = "相關性熱力圖"):
        """
        創建相關性熱力圖
        
        Args:
            data: 數據DataFrame
            title: 圖表標題
            
        Returns:
            Matplotlib圖表對象
        """
        try:
            # 只選擇數值列
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            numeric_df = data[numeric_cols]
            if not numeric_df.empty:
                correlation_data = numeric_df.corr()
            else:
                correlation_data = pd.DataFrame()
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # 創建熱力圖
            sns.heatmap(correlation_data, 
                       annot=True, 
                       cmap='coolwarm', 
                       center=0,
                       square=True,
                       ax=ax,
                       cbar_kws={'shrink': 0.8})
            
            ax.set_title(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            return fig
        
        except Exception as e:
            logger.error(f"相關性熱力圖創建失敗: {e}")
            return plt.figure()
    
    @staticmethod
    def create_returns_distribution(returns: pd.Series, title: str = "收益率分佈"):
        """
        創建收益率分佈圖
        
        Args:
            returns: 收益率序列
            title: 圖表標題
            
        Returns:
            Matplotlib圖表對象
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 直方圖
            ax1.hist(returns.dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title(f"{title} - 直方圖")
            ax1.set_xlabel("收益率")
            ax1.set_ylabel("頻率")
            ax1.grid(True, alpha=0.3)
            
            # Q-Q圖
            from scipy import stats
            try:
                stats.probplot(returns.dropna(), dist="norm", plot=ax2)
            except Exception:
                # 如果 probplot 失敗，創建簡單的散點圖
                ax2.scatter(range(len(returns.dropna())), sorted(returns.dropna()))
                ax2.set_title(f"{title} - 數據分佈")
            ax2.set_title(f"{title} - Q-Q圖")
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
        
        except Exception as e:
            logger.error(f"收益率分佈圖創建失敗: {e}")
            return plt.figure()

class ReportGenerator:
    """報告生成器"""
    
    @staticmethod
    def generate_strategy_report(performance_data: Dict[str, Dict], 
                               output_path: str = "策略分析報告.html") -> bool:
        """
        生成策略分析報告
        
        Args:
            performance_data: 策略績效數據
            output_path: 輸出路徑
            
        Returns:
            是否成功生成報告
        """
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html lang="zh-TW">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>港股量化策略分析報告</title>
                <style>
                    body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                    .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
                    h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 20px; }}
                    h2 {{ color: #34495e; border-left: 4px solid #3498db; padding-left: 15px; }}
                    .metric-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; margin: 10px; border-radius: 10px; text-align: center; }}
                    .metric-value {{ font-size: 2em; font-weight: bold; }}
                    .metric-label {{ font-size: 0.9em; opacity: 0.9; }}
                    table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                    th, td {{ padding: 12px; text-align: center; border: 1px solid #ddd; }}
                    th {{ background-color: #3498db; color: white; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    .positive {{ color: #27ae60; font-weight: bold; }}
                    .negative {{ color: #e74c3c; font-weight: bold; }}
                    .summary {{ background-color: #ecf0f1; padding: 20px; border-radius: 10px; margin: 20px 0; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🏦 港股量化策略分析報告</h1>
                    <div class="summary">
                        <h2>📊 報告摘要</h2>
                        <p>本報告分析了 {len(performance_data)} 個量化交易策略在港股市場的表現。</p>
                        <p>分析時間：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
            """
            
            # 添加策略績效表格
            html_content += """
                    <h2>📈 策略績效總覽</h2>
                    <table>
                        <tr>
                            <th>策略名稱</th>
                            <th>總收益率 (%)</th>
                            <th>年化收益率 (%)</th>
                            <th>波動率 (%)</th>
                            <th>夏普比率</th>
                            <th>最大回撤 (%)</th>
                            <th>勝率 (%)</th>
                            <th>交易次數</th>
                        </tr>
            """
            
            for strategy_name, metrics in performance_data.items():
                if 'error' not in metrics:
                    total_return_class = 'positive' if metrics.get('total_return', 0) > 0 else 'negative'
                    html_content += f"""
                        <tr>
                            <td><strong>{strategy_name}</strong></td>
                            <td class="{total_return_class}">{metrics.get('total_return', 0):.2f}</td>
                            <td class="{total_return_class}">{metrics.get('annual_return', 0):.2f}</td>
                            <td>{metrics.get('volatility', 0):.2f}</td>
                            <td>{metrics.get('sharpe_ratio', 0):.2f}</td>
                            <td class="negative">{metrics.get('max_drawdown', 0):.2f}</td>
                            <td>{metrics.get('win_rate', 0):.2f}</td>
                            <td>{metrics.get('trade_count', 0)}</td>
                        </tr>
                    """
            
            html_content += """
                    </table>
                    
                    <h2>🎯 策略建議</h2>
                    <div class="summary">
                        <ul>
                            <li><strong>最佳收益策略：</strong>選擇總收益率最高的策略作為主要配置</li>
                            <li><strong>風險控制：</strong>關注最大回撤較小的策略，降低組合風險</li>
                            <li><strong>夏普比率：</strong>優先考慮夏普比率大於1的策略</li>
                            <li><strong>組合配置：</strong>建議採用多策略組合，分散風險</li>
                        </ul>
                    </div>
                    
                    <div style="text-align: center; margin-top: 40px; color: #7f8c8d;">
                        <p>報告由港股量化分析系統自動生成</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # 寫入文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"✅ 策略分析報告已生成：{output_path}")
            return True
        
        except Exception as e:
            logger.error(f"報告生成失敗: {e}")
            return False
    
    @staticmethod
    def export_to_csv(data: pd.DataFrame, filename: str) -> bool:
        """
        導出數據到CSV文件
        
        Args:
            data: 要導出的數據
            filename: 文件名
            
        Returns:
            是否成功導出
        """
        try:
            data.to_csv(filename, encoding='utf-8-sig', index=True)
            logger.info(f"✅ 數據已導出到CSV：{filename}")
            return True
        
        except Exception as e:
            logger.error(f"CSV導出失敗: {e}")
            return False

def create_interactive_dashboard(stock_data: pd.DataFrame, symbol: str) -> str:
    """
    創建交互式儀表板
    
    Args:
        stock_data: 股票數據
        symbol: 股票代碼
        
    Returns:
        生成的儀表板文件路徑
    """
    try:
        import os
        
        # 生成儀表板HTML
        charts = PlotlyCharts()
        
        # 創建價格圖表
        price_fig = charts.create_price_chart(stock_data, f"{symbol} 股價走勢")
        
        # 創建技術指標圖表  
        tech_fig = charts.create_technical_indicators_chart(stock_data)
        
        # 生成HTML內容
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_完整分析Dashboard_{timestamp}.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{symbol} 港股分析儀表板</title>
            <meta charset="utf-8">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1 style="text-align: center;">{symbol} 港股量化分析儀表板</h1>
            <div id="price-chart">{price_fig.to_html(include_plotlyjs=False, div_id="price-chart")}</div>
            <div id="tech-chart">{tech_fig.to_html(include_plotlyjs=False, div_id="tech-chart")}</div>
        </body>
        </html>
        """
        
        # 寫入文件
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        logger.info(f"儀表板已生成: {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"儀表板生成失敗: {e}")
        return ""

def plot_results(
    data: pd.DataFrame,
    signals: pd.Series,
    params: Dict[str, Any],
    save_path: Optional[str] = None
) -> None:
    """繪製策略結果圖表"""
    try:
        # 設置繪圖樣式
        sns.set_theme(style="darkgrid")
        
        # 創建圖表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
        fig.suptitle(f"策略回測結果 - {params['strategy_type']}", fontsize=16)
        
        # 繪製價格和信號
        ax1.plot(data['date'], data['close'], label='收盤價', color='blue', alpha=0.7)
        
        # 標記買入和賣出點
        buy_signals = signals == 1
        sell_signals = signals == -1
        
        if buy_signals.any():
            ax1.scatter(
                data.loc[buy_signals, 'date'],
                data.loc[buy_signals, 'close'],
                marker='^',
                color='green',
                s=100,
                label='買入信號'
            )
            
        if sell_signals.any():
            ax1.scatter(
                data.loc[sell_signals, 'date'],
                data.loc[sell_signals, 'close'],
                marker='v',
                color='red',
                s=100,
                label='賣出信號'
            )
            
        # 設置第一個子圖的標籤和格式
        ax1.set_title('價格和交易信號')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('價格')
        ax1.legend()
        ax1.grid(True)
        
        # 繪製技術指標
        if params['strategy_type'] == 'RSI':
            # 計算RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=params['period']).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=params['period']).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # 繪製RSI
            ax2.plot(data['date'], rsi, label='RSI', color='purple', alpha=0.7)
            ax2.axhline(y=params['overbought'], color='r', linestyle='--', alpha=0.5)
            ax2.axhline(y=params['oversold'], color='g', linestyle='--', alpha=0.5)
            ax2.fill_between(
                data['date'],
                params['overbought'],
                params['oversold'],
                alpha=0.1,
                color='gray'
            )
            
            # 設置第二個子圖的標籤和格式
            ax2.set_title('RSI 指標')
            ax2.set_xlabel('日期')
            ax2.set_ylabel('RSI')
            ax2.set_ylim(0, 100)
            ax2.grid(True)
            
        # 調整布局
        plt.tight_layout()
        
        # 保存或顯示圖表
        if save_path:
            plt.savefig(save_path)
            logger.info(f"✅ 圖表已保存至: {save_path}")
        else:
            plt.show()
            
    except Exception as e:
        logger.error(f"❌ 繪圖失敗: {str(e)}")
        raise 