#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股量化分析系統 - 多策略優化運行腳本
測試多種技術指標策略的優化效果
"""

import os
import sys
import time
import warnings
from datetime import datetime
import multiprocessing as mp

warnings.filterwarnings('ignore')

# 添加當前目錄到Python路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 導入模組
try:
    from strategies import StrategyOptimizer, run_strategy_optimization, run_comprehensive_optimization
    print("✅ 策略模組導入成功")
except ImportError as e:
    print(f"❌ 策略模組導入失敗: {e}")
    sys.exit(1)

class MultiStrategyOptimizer:
    """多策略優化器"""
    
    def __init__(self, symbol: str = "2800.HK", start_date: str = "2020-01-01"):
        self.symbol = symbol
        self.start_date = start_date
        self.results = {}
        
    def run_single_strategy_test(self, strategy_name: str) -> dict:
        """運行單個策略測試"""
        try:
            print(f"\n🚀 開始測試 {strategy_name} 策略...")
            
            optimizer = StrategyOptimizer(self.symbol)
            data = optimizer.get_stock_data(self.start_date)
            
            if data.empty:
                print(f"❌ {strategy_name} 策略：無法獲取數據")
                return {}
            
            # 根據策略名稱調用相應的優化方法
            if strategy_name == "RSI":
                result = optimizer.optimize_rsi_strategy(data)
            elif strategy_name == "MACD":
                result = optimizer.optimize_macd_strategy(data)
            elif strategy_name == "Bollinger":
                result = optimizer.optimize_bollinger_strategy(data)
            elif strategy_name == "KDJ":
                result = optimizer.optimize_kdj_strategy(data)
            elif strategy_name == "Stochastic":
                result = optimizer.optimize_stochastic_strategy(data)
            elif strategy_name == "CCI":
                result = optimizer.optimize_cci_strategy(data)
            elif strategy_name == "Williams_R":
                result = optimizer.optimize_williams_r_strategy(data)
            else:
                print(f"❌ 未知策略: {strategy_name}")
                return {}
            
            if result and result.get('performance', {}).get('sharpe_ratio', 0) > 0:
                print(f"✅ {strategy_name} 策略優化成功 - 夏普比率: {result['performance']['sharpe_ratio']:.3f}")
                return result
            else:
                print(f"❌ {strategy_name} 策略優化失敗或表現不佳")
                return {}
                
        except Exception as e:
            print(f"❌ {strategy_name} 策略測試錯誤: {e}")
            return {}
    
    def run_parallel_optimization(self, max_workers: int = None) -> dict:
        """並行運行所有策略優化"""
        try:
            if max_workers is None:
                max_workers = min(8, mp.cpu_count())  # 限制最大進程數
            
            print(f"\n🔥 啟動並行多策略優化 (使用 {max_workers} 個進程)")
            
            # 定義所有策略
            strategies = [
                "RSI", "MACD", "Bollinger", "KDJ", 
                "Stochastic", "CCI", "Williams_R"
            ]
            
            # 並行執行策略優化
            with mp.Pool(processes=max_workers) as pool:
                results = pool.map(self.run_single_strategy_test, strategies)
            
            # 整理結果
            strategy_results = {}
            for i, strategy in enumerate(strategies):
                if results[i]:
                    strategy_results[strategy] = results[i]
            
            return strategy_results
            
        except Exception as e:
            print(f"❌ 並行優化失敗: {e}")
            return {}
    
    def run_sequential_optimization(self) -> dict:
        """順序運行所有策略優化"""
        try:
            print("\n📈 順序執行多策略優化...")
            
            strategies = [
                "RSI", "MACD", "Bollinger", "KDJ", 
                "Stochastic", "CCI", "Williams_R"
            ]
            
            strategy_results = {}
            
            for strategy in strategies:
                start_time = time.time()
                result = self.run_single_strategy_test(strategy)
                end_time = time.time()
                
                if result:
                    strategy_results[strategy] = result
                    print(f"   ⏱️ {strategy} 用時: {end_time - start_time:.2f}秒")
                else:
                    print(f"   ❌ {strategy} 失敗")
            
            return strategy_results
            
        except Exception as e:
            print(f"❌ 順序優化失敗: {e}")
            return {}
    
    def analyze_results(self, results: dict) -> None:
        """分析和展示結果"""
        try:
            if not results:
                print("❌ 沒有可分析的結果")
                return
            
            print(f"\n📊 多策略優化結果分析 (共 {len(results)} 個策略)")
            print("=" * 80)
            
            # 按夏普比率排序
            sorted_results = sorted(
                results.items(), 
                key=lambda x: x[1]['performance']['sharpe_ratio'], 
                reverse=True
            )
            
            print(f"🏆 策略表現排名:")
            print("-" * 80)
            print(f"{'排名':<4} {'策略':<12} {'夏普比率':<10} {'年化收益':<12} {'最大回撤':<12} {'最佳參數'}")
            print("-" * 80)
            
            for i, (strategy, result) in enumerate(sorted_results, 1):
                sharpe = result['performance']['sharpe_ratio']
                annual_return = result['performance']['annual_return'] * 100
                max_drawdown = abs(result['performance']['max_drawdown']) * 100
                params = str(result['params'])[:30] + "..." if len(str(result['params'])) > 30 else str(result['params'])
                
                print(f"{i:<4} {strategy:<12} {sharpe:<10.3f} {annual_return:<12.2f}% {max_drawdown:<12.2f}% {params}")
            
            # 策略統計分析
            print(f"\n📈 統計分析:")
            print("-" * 50)
            
            sharpe_ratios = [r['performance']['sharpe_ratio'] for r in results.values()]
            annual_returns = [r['performance']['annual_return'] * 100 for r in results.values()]
            max_drawdowns = [abs(r['performance']['max_drawdown']) * 100 for r in results.values()]
            
            print(f"   平均夏普比率: {sum(sharpe_ratios) / len(sharpe_ratios):.3f}")
            print(f"   最高夏普比率: {max(sharpe_ratios):.3f}")
            print(f"   平均年化收益: {sum(annual_returns) / len(annual_returns):.2f}%")
            print(f"   平均最大回撤: {sum(max_drawdowns) / len(max_drawdowns):.2f}%")
            
            # 推薦策略
            print(f"\n💡 策略推薦:")
            print("-" * 30)
            
            best_strategy = sorted_results[0]
            print(f"   🥇 最佳策略: {best_strategy[0]} (夏普比率: {best_strategy[1]['performance']['sharpe_ratio']:.3f})")
            
            positive_strategies = [s for s, r in results.items() if r['performance']['sharpe_ratio'] > 0.5]
            if positive_strategies:
                print(f"   ⭐ 表現良好策略: {', '.join(positive_strategies)}")
            
            # 保存結果到文件
            self.save_results_to_file(results)
            
        except Exception as e:
            print(f"❌ 結果分析失敗: {e}")
    
    def save_results_to_file(self, results: dict) -> None:
        """保存結果到文件"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data_output/reports/multi_strategy_optimization_{timestamp}.txt"
            
            # 確保目錄存在
            os.makedirs("data_output/reports", exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("港股多策略優化報告\n")
                f.write("=" * 50 + "\n")
                f.write(f"生成時間: {datetime.now()}\n")
                f.write(f"股票代碼: {self.symbol}\n")
                f.write(f"分析期間: {self.start_date} 至今\n")
                f.write(f"策略數量: {len(results)}\n\n")
                
                # 詳細結果
                for strategy, result in results.items():
                    f.write(f"\n## {strategy} 策略\n")
                    f.write(f"最佳參數: {result['params']}\n")
                    f.write(f"夏普比率: {result['performance']['sharpe_ratio']:.3f}\n")
                    f.write(f"年化收益: {result['performance']['annual_return']*100:.2f}%\n")
                    f.write(f"最大回撤: {abs(result['performance']['max_drawdown'])*100:.2f}%\n")
                    f.write("-" * 40 + "\n")
            
            print(f"✅ 結果已保存到: {filename}")
            
        except Exception as e:
            print(f"❌ 保存結果失敗: {e}")

def main():
    """主運行函數"""
    print("🎯 港股多策略優化系統")
    print("=" * 50)
    
    # 初始化優化器
    symbol = "2800.HK"
    start_date = "2020-01-01"
    
    multi_optimizer = MultiStrategyOptimizer(symbol, start_date)
    
    print("\n🚀 自動運行所有策略優化...")
    print("   包含: RSI, MACD, 布林帶, KDJ, Stochastic, CCI, 威廉指標%R")
    
    try:
        # 自動運行順序執行模式（最穩定）
        print("\n📈 開始順序執行多策略優化...")
        start_time = time.time()
        results = multi_optimizer.run_sequential_optimization()
        end_time = time.time()
        
        print(f"\n⏱️ 總執行時間: {end_time - start_time:.2f}秒")
        
        # 分析結果
        if results:
            multi_optimizer.analyze_results(results)
            print("\n✅ 所有策略優化完成！")
        else:
            print("\n❌ 策略優化失敗，請檢查數據連接")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ 用戶中斷執行")
    except Exception as e:
        print(f"\n❌ 執行錯誤: {e}")
        # 如果順序執行失敗，嘗試基本策略測試
        print("\n🔄 嘗試運行基本策略測試...")
        try:
            success = run_strategy_optimization(symbol, start_date)
            if success:
                print("✅ 基本策略測試完成")
            else:
                print("❌ 基本策略測試也失敗")
        except Exception as fallback_error:
            print(f"❌ 基本策略測試錯誤: {fallback_error}")
    
    print("\n🎉 多策略優化系統運行完成！")

if __name__ == "__main__":
    main() 