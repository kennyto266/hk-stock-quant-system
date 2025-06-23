#!/usr/bin/env python3
"""
港股量化分析系統 - 重構版主程式
整合所有模組，提供統一的分析介面
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from typing import Dict, List, Optional

# 添加模組路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 導入自定義模組
try:
    from config import SystemConfig, TECHNICAL_CONFIG, RISK_CONFIG, logger
    from data_handler import DataFetcher, TechnicalIndicators
    from strategy_manager import StrategyPortfolioManager, SignalGenerator
    from risk_management import RiskMetrics, PortfolioManager
    from visualization import PlotlyCharts, ReportGenerator
except ImportError as e:
    print(f"模組導入失敗: {e}")
    print("請確保所有模組文件都在正確位置")
    sys.exit(1)

warnings.filterwarnings('ignore')

class HKStockAnalysisSystem:
    """港股量化分析系統主類"""
    
    def __init__(self, symbol: str = "2800.HK"):
        """
        初始化分析系統
        
        Args:
            symbol: 股票代碼
        """
        self.symbol = symbol
        self.config = SystemConfig()
        self.data_fetcher = DataFetcher()
        self.strategy_manager = StrategyPortfolioManager()
        self.risk_manager = PortfolioManager()
        self.signal_generator = SignalGenerator()
        self.charts = PlotlyCharts()
        self.report_generator = ReportGenerator()
        
        # 數據存儲
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.strategies_data: Dict[str, pd.DataFrame] = {}
        self.performance_results: Dict[str, Dict] = {}
        
        logger.info(f"🚀 港股量化分析系統已初始化，目標股票: {symbol}")
    
    def fetch_and_process_data(self, period: str = "6mo", 
                              use_northbound: bool = False) -> bool:
        """
        獲取並處理股票數據
        
        Args:
            period: 數據週期
            use_northbound: 是否使用北水數據
            
        Returns:
            是否成功獲取數據
        """
        try:
            logger.info("📊 開始獲取股票數據...")
            
            # 獲取原始數據
            self.raw_data = self.data_fetcher.get_hk_stock_data(
                symbol=self.symbol,
                period=period,
                use_northbound_data=use_northbound
            )
            
            if self.raw_data is None or self.raw_data.empty:
                logger.error("❌ 股票數據獲取失敗")
                return False
            
            logger.info(f"✅ 成功獲取 {len(self.raw_data)} 條數據記錄")
            
            # 計算技術指標
            logger.info("🔧 計算技術指標...")
            self.processed_data = TechnicalIndicators.add_all_indicators(
                self.raw_data.copy()
            )
            
            if self.processed_data is None or self.processed_data.empty:
                logger.error("❌ 技術指標計算失敗")
                return False
            
            logger.info("✅ 技術指標計算完成")
            return True
        
        except Exception as e:
            logger.error(f"❌ 數據獲取處理失敗: {e}")
            return False
    
    def generate_all_strategies(self) -> bool:
        """
        生成所有策略信號
        
        Returns:
            是否成功生成策略
        """
        try:
            if self.processed_data is None:
                logger.error("❌ 請先獲取並處理數據")
                return False
            
            logger.info("🎯 開始生成策略信號...")
            
            # 1. RSI策略
            rsi_signals = self.signal_generator.generate_rsi_signals(
                self.processed_data,
                rsi_period=TECHNICAL_CONFIG['rsi_period'],
                overbought=TECHNICAL_CONFIG['rsi_overbought'],
                oversold=TECHNICAL_CONFIG['rsi_oversold']
            )
            
            if not rsi_signals.empty:
                self.strategies_data['RSI'] = rsi_signals
                self.strategy_manager.add_strategy('RSI', rsi_signals, weight=0.25)
                logger.info("✅ RSI策略信號生成完成")
            
            # 2. MACD策略
            macd_signals = self.signal_generator.generate_macd_signals(
                self.processed_data
            )
            
            if not macd_signals.empty:
                self.strategies_data['MACD'] = macd_signals
                self.strategy_manager.add_strategy('MACD', macd_signals, weight=0.25)
                logger.info("✅ MACD策略信號生成完成")
            
            # 3. 布林帶策略
            bb_signals = self.signal_generator.generate_bollinger_signals(
                self.processed_data
            )
            
            if not bb_signals.empty:
                self.strategies_data['Bollinger'] = bb_signals
                self.strategy_manager.add_strategy('Bollinger', bb_signals, weight=0.25)
                logger.info("✅ 布林帶策略信號生成完成")
            
            # 4. 移動平均交叉策略
            ma_signals = self.signal_generator.generate_ma_crossover_signals(
                self.processed_data,
                fast_period=TECHNICAL_CONFIG['ma_fast'],
                slow_period=TECHNICAL_CONFIG['ma_slow']
            )
            
            if not ma_signals.empty:
                self.strategies_data['MA_Cross'] = ma_signals
                self.strategy_manager.add_strategy('MA_Cross', ma_signals, weight=0.25)
                logger.info("✅ 移動平均交叉策略信號生成完成")
            
            logger.info(f"🎉 共生成 {len(self.strategies_data)} 個策略")
            return True
        
        except Exception as e:
            logger.error(f"❌ 策略生成失敗: {e}")
            return False
    
    def calculate_performance(self) -> bool:
        """
        計算策略績效
        
        Returns:
            是否成功計算績效
        """
        try:
            if self.processed_data is None or not self.strategies_data:
                logger.error("❌ 請先生成策略信號")
                return False
            
            logger.info("📈 開始計算策略績效...")
            
            # 獲取價格數據
            price_data = self.processed_data['Close']
            
            # 計算各策略績效
            self.performance_results = self.strategy_manager.get_strategy_performance(
                price_data
            )
            
            # 生成組合策略
            combined_signals = self.strategy_manager.generate_combined_signals(
                method='weighted_average'
            )
            
            if not combined_signals.empty:
                # 添加組合策略到策略管理器
                self.strategy_manager.add_strategy('組合策略', combined_signals, weight=1.0)
                
                # 計算組合策略績效
                combined_performance = self.strategy_manager._calculate_single_strategy_performance(
                    combined_signals, price_data, '組合策略'
                )
                self.performance_results['組合策略'] = combined_performance
                logger.info("✅ 組合策略績效計算完成")
            
            # 輸出績效摘要
            self._print_performance_summary()
            
            logger.info("✅ 所有策略績效計算完成")
            return True
        
        except Exception as e:
            logger.error(f"❌ 績效計算失敗: {e}")
            return False
    
    def _print_performance_summary(self):
        """打印績效摘要"""
        try:
            print("\n" + "="*80)
            print("📊 策略績效摘要")
            print("="*80)
            
            print(f"{'策略名稱':<15} {'總收益率':<10} {'年化收益率':<12} {'夏普比率':<10} {'最大回撤':<10} {'勝率':<8}")
            print("-" * 80)
            
            for strategy_name, metrics in self.performance_results.items():
                if 'error' not in metrics:
                    print(f"{strategy_name:<15} "
                          f"{metrics.get('total_return', 0):>8.2f}% "
                          f"{metrics.get('annual_return', 0):>10.2f}% "
                          f"{metrics.get('sharpe_ratio', 0):>8.2f} "
                          f"{metrics.get('max_drawdown', 0):>8.2f}% "
                          f"{metrics.get('win_rate', 0):>6.1f}%")
                else:
                    print(f"{strategy_name:<15} {'錯誤':<50}")
            
            print("="*80)
        
        except Exception as e:
            logger.error(f"績效摘要輸出失敗: {e}")
    
    def generate_visualizations(self, save_charts: bool = True) -> bool:
        """
        生成可視化圖表
        
        Args:
            save_charts: 是否保存圖表
            
        Returns:
            是否成功生成圖表
        """
        try:
            if self.processed_data is None:
                logger.error("❌ 請先處理數據")
                return False
            
            logger.info("📊 開始生成圖表...")
            
            # 1. 股價走勢圖
            price_chart = self.charts.create_price_chart(
                self.processed_data,
                title=f"{self.symbol} 股價走勢圖"
            )
            
            if save_charts:
                price_chart.write_html(f"{self.symbol}_價格走勢.html")
                logger.info("✅ 股價走勢圖已保存")
            
            # 2. 技術指標圖
            tech_chart = self.charts.create_technical_indicators_chart(
                self.processed_data
            )
            
            if save_charts:
                tech_chart.write_html(f"{self.symbol}_技術指標.html")
                logger.info("✅ 技術指標圖已保存")
            
            # 3. 策略績效圖
            if self.performance_results:
                perf_chart = self.charts.create_strategy_performance_chart(
                    self.performance_results
                )
                
                if save_charts:
                    perf_chart.write_html(f"{self.symbol}_策略績效.html")
                    logger.info("✅ 策略績效圖已保存")
            
            # 4. 交易信號圖
            for strategy_name, signals in self.strategies_data.items():
                signal_chart = self.charts.create_signals_chart(
                    self.processed_data,
                    signals,
                    strategy_name=f"{strategy_name} 策略"
                )
                
                if save_charts:
                    signal_chart.write_html(f"{self.symbol}_{strategy_name}_信號.html")
            
            logger.info("✅ 所有圖表生成完成")
            return True
        
        except Exception as e:
            logger.error(f"❌ 圖表生成失敗: {e}")
            return False
    
    def generate_report(self, output_path: str = None) -> bool:
        """
        生成分析報告
        
        Args:
            output_path: 輸出路徑
            
        Returns:
            是否成功生成報告
        """
        try:
            if not self.performance_results:
                logger.error("❌ 請先計算策略績效")
                return False
            
            if output_path is None:
                output_path = f"{self.symbol}_策略分析報告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            
            logger.info("📄 開始生成分析報告...")
            
            success = self.report_generator.generate_strategy_report(
                self.performance_results,
                output_path
            )
            
            if success:
                logger.info(f"✅ 分析報告已生成: {output_path}")
            else:
                logger.error("❌ 報告生成失敗")
            
            return success
        
        except Exception as e:
            logger.error(f"❌ 報告生成失敗: {e}")
            return False
    
    def export_data(self, export_signals: bool = True, 
                   export_performance: bool = True) -> bool:
        """
        導出數據到CSV
        
        Args:
            export_signals: 是否導出信號數據
            export_performance: 是否導出績效數據
            
        Returns:
            是否成功導出
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 導出處理後的數據
            if self.processed_data is not None:
                filename = f"{self.symbol}_數據_{timestamp}.csv"
                self.report_generator.export_to_csv(self.processed_data, filename)
            
            # 導出策略信號
            if export_signals and self.strategies_data:
                for strategy_name, signals in self.strategies_data.items():
                    filename = f"{self.symbol}_{strategy_name}_信號_{timestamp}.csv"
                    self.report_generator.export_to_csv(signals, filename)
            
            # 導出績效數據
            if export_performance and self.performance_results:
                perf_df = pd.DataFrame(self.performance_results).T
                filename = f"{self.symbol}_策略績效_{timestamp}.csv"
                self.report_generator.export_to_csv(perf_df, filename)
            
            logger.info("✅ 數據導出完成")
            return True
        
        except Exception as e:
            logger.error(f"❌ 數據導出失敗: {e}")
            return False
    
    def run_full_analysis(self, period: str = "6mo", 
                         use_northbound: bool = False,
                         generate_charts: bool = True,
                         generate_report: bool = True,
                         export_data: bool = True) -> bool:
        """
        運行完整分析流程
        
        Args:
            period: 數據週期
            use_northbound: 是否使用北水數據
            generate_charts: 是否生成圖表
            generate_report: 是否生成報告
            export_data: 是否導出數據
            
        Returns:
            是否成功完成分析
        """
        try:
            logger.info("🚀 開始完整分析流程...")
            
            # 1. 獲取數據
            if not self.fetch_and_process_data(period, use_northbound):
                return False
            
            # 2. 生成策略
            if not self.generate_all_strategies():
                return False
            
            # 3. 計算績效
            if not self.calculate_performance():
                return False
            
            # 4. 生成圖表
            if generate_charts:
                self.generate_visualizations(save_charts=True)
            
            # 5. 生成報告
            if generate_report:
                self.generate_report()
            
            # 6. 導出數據
            if export_data:
                self.export_data()
            
            logger.info("🎉 完整分析流程執行完成！")
            return True
        
        except Exception as e:
            logger.error(f"❌ 完整分析流程失敗: {e}")
            return False


def main():
    """主函數"""
    try:
        print("🏦 港股量化分析系統 - 重構版")
        print("="*50)
        
        # 創建分析系統
        analyzer = HKStockAnalysisSystem("2800.HK")
        
        # 運行完整分析
        success = analyzer.run_full_analysis(
            period="6mo",
            use_northbound=False,
            generate_charts=True,
            generate_report=True,
            export_data=True
        )
        
        if success:
            print("\n🎉 分析完成！請查看生成的報告和圖表。")
        else:
            print("\n❌ 分析過程中出現錯誤，請檢查日誌。")
    
    except KeyboardInterrupt:
        print("\n⚠️ 用戶中斷操作")
    except Exception as e:
        print(f"\n❌ 程序執行錯誤: {e}")
        logger.error(f"主程序執行錯誤: {e}")


if __name__ == "__main__":
    main() 