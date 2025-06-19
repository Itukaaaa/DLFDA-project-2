import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import logging
import datetime
import time
import argparse
from find_best_threshold import analyze_thresholds, plot_threshold_results
from test import process_data , calculate_10min_returns

# 设置日志配置
log_dir = "backtest/logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 创建专用的logger
logger = logging.getLogger('multi_threshold')
logger.setLevel(logging.INFO)

# 检查是否已有处理器，避免重复添加
if not logger.handlers:
    # 创建文件处理器
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(os.path.join(log_dir, f"threshold_backtest.log"), mode='a')
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 防止日志传递到根日志记录器
    logger.propagate = False

# 创建图片保存目录
threshold_img_dir = "backtest/threshold_images/multi"
if not os.path.exists(threshold_img_dir):
    os.makedirs(threshold_img_dir)

# 创建结果保存目录
results_dir = "backtest/results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

def voting_analysis(file1, file2, 
                  start_threshold_1=0.85, 
                  end_threshold_1=0.995, 
                  step_1=0.005, 
                  start_threshold_2=0.85, 
                  end_threshold_2=0.995, 
                  step_2=0.005, 
                  min_daily_trades=3):
    """
    对两个预测文件进行投票分析
    
    参数:
    file1: 第一个预测结果文件
    file2: 第二个预测结果文件
    start_threshold_1: 起始阈值
    end_threshold_1: 结束阈值
    step_1: 步长
    min_daily_trades: 每日最小交易次数要求
    
    返回:
    包含不同阈值结果的DataFrame
    """
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*50}")
    print(f"开始两文件投票分析")
    print(f"文件1: {os.path.basename(file1)}")
    print(f"文件2: {os.path.basename(file2)}")
    print(f"阈值范围: {start_threshold_1} - {end_threshold_1}, 步长: {step_1}")
    print(f"{'='*50}\n")
    
    logger.info(f"Start to vote: {os.path.basename(file1)} 和 {os.path.basename(file2)}")

    print(f"\n开始阈值分析, 范围: {start_threshold_1} - {end_threshold_1}, 步长: {step_1}")
    results = []
    current_threshold_1 = start_threshold_1
    
    logger.info(f"Starting threshold analysis, range: {start_threshold_1} - {end_threshold_1}, step: {step_1}")
    print(f"Starting threshold analysis, range: {start_threshold_1} - {end_threshold_1}, step: {step_1}")

    while current_threshold_1 <= end_threshold_1:
        logger.info(f"Testing threshold: {current_threshold_1}")
        print(f"Testing threshold: {current_threshold_1:.3f}")
        
        df1 = process_data(filename=file1, 
                            use_threshold=True,
                            long_threshold=current_threshold_1, 
                            short_threshold=current_threshold_1)
        
        df2 = process_data(filename=file2, 
                            use_threshold=True,
                            long_threshold=current_threshold_1, 
                            short_threshold=current_threshold_1)
    
        return_df = calculate_10min_returns(df1['close'])
        
        total_trades = 0.0
        total_return = 0.0
        
        for i in range(0, len(df1), 10):
            if df1['predicted_label'].iloc[i] == 0 and df2['predicted_label'].iloc[i] == 0:
                total_trades += 1
                total_return += -return_df['10min_return'][i]
            elif df1['predicted_label'].iloc[i] == 2 and df2['predicted_label'].iloc[i] == 2:
                total_trades += 1
                total_return += return_df['10min_return'][i]
                
        avg_return = total_return * 10000 / total_trades if total_trades > 0 else 0
        total_pnl = total_return * 10000
        
        # 计算每日平均交易次数
        trading_days = len(df1['close'])/1440
        daily_avg_trades = total_trades / trading_days
        
        # 记录结果 - 增加每日平均PNL
        results.append({
            'threshold_1': current_threshold_1,
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'avg_pnl': avg_return,
            'daily_avg_pnl': total_pnl / trading_days, 
            'daily_avg_trades': daily_avg_trades
        })
        
        # 修改日志输出，显示每日平均PNL, 增加精度到3位小数
        logger.info(f"Threshold_1 {current_threshold_1:.3f}: trades={total_trades}, total PNL={total_pnl:.2f}, avg PNL={avg_return:.2f}, daily avg trades={daily_avg_trades:.2f}, daily avg PNL={total_pnl/trading_days:.2f}")
        print(f"Threshold {current_threshold_1:.3f}: trades={total_trades}, total PNL={total_pnl:.2f}, avg PNL={avg_return:.2f}, daily avg trades={daily_avg_trades:.2f}, daily avg PNL={total_pnl/trading_days:.2f}")
        
        # 修改提前结束的条件输出
        if daily_avg_trades < min_daily_trades and current_threshold_1 > 0.7:
            logger.info(f"Daily average trades {daily_avg_trades:.2f} is below minimum requirement {min_daily_trades}, and threshold already reached {current_threshold_1:.3f}, stopping early")
            print(f"Daily average trades {daily_avg_trades:.2f} is below minimum requirement {min_daily_trades}, and threshold already reached {current_threshold_1:.3f}, stopping early")
            break
            
        current_threshold_1 += step_1
    
    results_df = pd.DataFrame(results)
    
    # 寻找最佳阈值, 增加精度显示
    if not results_df.empty:
        # 根据平均PNL寻找最佳阈值
        best_avg_pnl_idx = results_df['avg_pnl'].argmax()
        best_avg_pnl_threshold = results_df.iloc[best_avg_pnl_idx]['threshold_1']
        best_avg_pnl = results_df.iloc[best_avg_pnl_idx]['avg_pnl']
        best_avg_pnl_trades = results_df.iloc[best_avg_pnl_idx]['total_trades']
        
        # 根据总PNL寻找最佳阈值
        best_total_pnl_idx = results_df['total_pnl'].argmax()
        best_total_pnl_threshold = results_df.iloc[best_total_pnl_idx]['threshold_1']
        best_total_pnl = results_df.iloc[best_total_pnl_idx]['total_pnl']
        best_total_pnl_trades = results_df.iloc[best_total_pnl_idx]['total_trades']
        
        # 根据每日平均PNL寻找最佳阈值
        best_daily_pnl_idx = results_df['daily_avg_pnl'].argmax()
        best_daily_pnl_threshold = results_df.iloc[best_daily_pnl_idx]['threshold_1']
        best_daily_pnl = results_df.iloc[best_daily_pnl_idx]['daily_avg_pnl']
        best_daily_pnl_trades = results_df.iloc[best_daily_pnl_idx]['total_trades']
        
        logger.info("=" * 50)
        logger.info(f"Best avg PNL threshold: {best_avg_pnl_threshold:.3f}, avg PNL: {best_avg_pnl:.2f}, trades: {best_avg_pnl_trades}, daily avg trades: {best_avg_pnl_trades/trading_days:.2f}")
        logger.info(f"Best total PNL threshold: {best_total_pnl_threshold:.3f}, total PNL: {best_total_pnl:.2f}, trades: {best_total_pnl_trades}, daily avg trades: {best_total_pnl_trades/trading_days:.2f}")
        logger.info(f"Best daily PNL threshold: {best_daily_pnl_threshold:.3f}, daily avg PNL: {best_daily_pnl:.2f}, trades: {best_daily_pnl_trades}, daily avg trades: {best_daily_pnl_trades/trading_days:.2f}")
        
        print("=" * 50)
        print(f"Best avg PNL threshold: {best_avg_pnl_threshold:.3f}, avg PNL: {best_avg_pnl:.2f}, trades: {best_avg_pnl_trades}, daily avg trades: {best_avg_pnl_trades/trading_days:.2f}")
        print(f"Best total PNL threshold: {best_total_pnl_threshold:.3f}, total PNL: {best_total_pnl:.2f}, trades: {best_total_pnl_trades}, daily avg trades: {best_total_pnl_trades/trading_days:.2f}")
        print(f"Best daily PNL threshold: {best_daily_pnl_threshold:.3f}, daily avg PNL: {best_daily_pnl:.2f}, trades: {best_daily_pnl_trades}, daily avg trades: {best_daily_pnl_trades/trading_days:.2f}")
    
    # plot_threshold_results(results_df, 
    #                        save_path=os.path.join(threshold_img_dir, f"threshold_analysis_{timestamp}.png"),
    #                        trading_days=trading_days)
    
    return results_df
    

def main():
    parser = argparse.ArgumentParser(description="两个预测文件的投票分析工具")
    parser.add_argument("--file1",  default="infer_result/bigbal_105230.csv", help="第一个预测文件路径")
    parser.add_argument("--file2",  default="infer_result/bigbal_134843.csv", help="第二个预测文件路径")
    parser.add_argument("--start_1", type=float, default=0.75, help="起始阈值")
    parser.add_argument("--end_1", type=float, default=0.8, help="结束阈值")
    parser.add_argument("--step_1", type=float, default=0.005, help="阈值步长")
    parser.add_argument("--min_trades", type=int, default=3, help="每日最小交易次数")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.file1):
        print(f"错误: 文件1不存在: {args.file1}")
        return
        
    if not os.path.exists(args.file2):
        print(f"错误: 文件2不存在: {args.file2}")
        return
    
    # 执行投票分析
    voting_analysis(
        file1=args.file1,
        file2=args.file2,
        start_threshold_1=args.start_1,
        end_threshold_1=args.end_1,
        step_1=args.step_1,
        start_threshold_2=args.start_1,
        end_threshold_2=args.end_1,
        step_2=args.step_1,
        min_daily_trades=args.min_trades
    )

if __name__ == "__main__":
    main()
