import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import datetime
import sys
from test import process_data, calculate_10min_returns

# 设置日志配置
log_dir = "backtest/logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 创建专用的logger
logger = logging.getLogger('threshold')
logger.setLevel(logging.INFO)

# 检查是否已有处理器，避免重复添加
if not logger.handlers:
    # 创建文件处理器
    file_handler = logging.FileHandler(os.path.join(log_dir, "threshold.log"), mode='a')
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 防止日志传递到根日志记录器
    logger.propagate = False

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 创建图片保存目录
threshold_img_dir = "backtest/threshold_images"
if not os.path.exists(threshold_img_dir):
    os.makedirs(threshold_img_dir)

def analyze_thresholds(filename="infer_result/example.csv", 
                       start_threshold=0.5, 
                       end_threshold=0.99, 
                       step=0.02, 
                       min_daily_trades=50,
                       trading_days=1):
    """
    在给定的阈值范围内遍历，寻找最佳交易阈值
    
    参数:
    filename: 推理结果文件
    start_threshold: 起始阈值
    end_threshold: 结束阈值
    step: 步长
    min_daily_trades: 每日最小交易次数要求
    trading_days: 数据集包含的交易天数
    
    返回:
    包含不同阈值结果的DataFrame
    """
    results = []
    current_threshold = start_threshold
    
    logger.info(f"Starting threshold analysis, range: {start_threshold} - {end_threshold}, step: {step}")
    print(f"Starting threshold analysis, range: {start_threshold} - {end_threshold}, step: {step}")
    
    while current_threshold <= end_threshold:
        logger.info(f"Testing threshold: {current_threshold}")
        print(f"Testing threshold: {current_threshold:.2f}")
        
        # 使用相同的阈值进行long和short
        df = process_data(filename=filename, 
                          use_threshold=True,
                          long_threshold=current_threshold, 
                          short_threshold=current_threshold)
        
        return_df = calculate_10min_returns(df['close'])
        
        # 计算交易表现
        total_return = 0.0
        total_trades = 0
        
        for i in range(0, len(df) - 10):
            if df['predicted_label'][i] == 0:  # short
                total_return += -return_df['10min_return'][i]
                total_trades += 1
            elif df['predicted_label'][i] == 2:  # long
                total_return += return_df['10min_return'][i]
                total_trades += 1
        
        # 计算平均收益
        avg_return = total_return * 10000 / total_trades if total_trades > 0 else 0
        total_pnl = total_return * 10000
        
        # 计算准确率
        accuracy = np.mean(df['label'].values == df['predicted_label'].values)
        
        # 计算每日平均交易次数
        daily_avg_trades = total_trades / trading_days
        
        # 记录结果 - 增加每日平均PNL
        results.append({
            'threshold': current_threshold,
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'avg_pnl': avg_return,
            'daily_avg_pnl': total_pnl / trading_days,  # 新增每日平均PNL
            'accuracy': accuracy,
            'daily_avg_trades': daily_avg_trades
        })
        
        # 修改日志输出，显示每日平均PNL
        logger.info(f"Threshold {current_threshold:.2f}: trades={total_trades}, total PNL={total_pnl:.2f}, avg PNL={avg_return:.2f}, daily avg trades={daily_avg_trades:.2f}, daily avg PNL={total_pnl/trading_days:.2f}, accuracy={accuracy:.4f}")
        print(f"Threshold {current_threshold:.2f}: trades={total_trades}, total PNL={total_pnl:.2f}, avg PNL={avg_return:.2f}, daily avg trades={daily_avg_trades:.2f}, daily avg PNL={total_pnl/trading_days:.2f}, accuracy={accuracy:.4f}")
        
        # 修改提前结束的条件为每日平均交易次数
        if daily_avg_trades < min_daily_trades and current_threshold > 0.7:
            logger.info(f"Daily average trades {daily_avg_trades:.2f} is below minimum requirement {min_daily_trades}, and threshold already reached {current_threshold:.2f}, stopping early")
            print(f"Daily average trades {daily_avg_trades:.2f} is below minimum requirement {min_daily_trades}, and threshold already reached {current_threshold:.2f}, stopping early")
            break
            
        current_threshold += step
    
    # 创建DataFrame
    results_df = pd.DataFrame(results)
    
    # 寻找最佳阈值
    if not results_df.empty:
        # 根据平均PNL寻找最佳阈值
        best_avg_pnl_idx = results_df['avg_pnl'].argmax()
        best_avg_pnl_threshold = results_df.iloc[best_avg_pnl_idx]['threshold']
        best_avg_pnl = results_df.iloc[best_avg_pnl_idx]['avg_pnl']
        best_avg_pnl_trades = results_df.iloc[best_avg_pnl_idx]['total_trades']
        
        # 根据总PNL寻找最佳阈值
        best_total_pnl_idx = results_df['total_pnl'].argmax()
        best_total_pnl_threshold = results_df.iloc[best_total_pnl_idx]['threshold']
        best_total_pnl = results_df.iloc[best_total_pnl_idx]['total_pnl']
        best_total_pnl_trades = results_df.iloc[best_total_pnl_idx]['total_trades']
        
        # 根据每日平均PNL寻找最佳阈值
        best_daily_pnl_idx = results_df['daily_avg_pnl'].argmax()
        best_daily_pnl_threshold = results_df.iloc[best_daily_pnl_idx]['threshold']
        best_daily_pnl = results_df.iloc[best_daily_pnl_idx]['daily_avg_pnl']
        best_daily_pnl_trades = results_df.iloc[best_daily_pnl_idx]['total_trades']
        
        logger.info("=" * 50)
        logger.info(f"Best avg PNL threshold: {best_avg_pnl_threshold:.2f}, avg PNL: {best_avg_pnl:.2f}, trades: {best_avg_pnl_trades}, daily avg trades: {best_avg_pnl_trades/trading_days:.2f}")
        logger.info(f"Best total PNL threshold: {best_total_pnl_threshold:.2f}, total PNL: {best_total_pnl:.2f}, trades: {best_total_pnl_trades}, daily avg trades: {best_total_pnl_trades/trading_days:.2f}")
        logger.info(f"Best daily PNL threshold: {best_daily_pnl_threshold:.2f}, daily avg PNL: {best_daily_pnl:.2f}, trades: {best_daily_pnl_trades}, daily avg trades: {best_daily_pnl_trades/trading_days:.2f}")
        
        print("=" * 50)
        print(f"Best avg PNL threshold: {best_avg_pnl_threshold:.2f}, avg PNL: {best_avg_pnl:.2f}, trades: {best_avg_pnl_trades}, daily avg trades: {best_avg_pnl_trades/trading_days:.2f}")
        print(f"Best total PNL threshold: {best_total_pnl_threshold:.2f}, total PNL: {best_total_pnl:.2f}, trades: {best_total_pnl_trades}, daily avg trades: {best_total_pnl_trades/trading_days:.2f}")
        print(f"Best daily PNL threshold: {best_daily_pnl_threshold:.2f}, daily avg PNL: {best_daily_pnl:.2f}, trades: {best_daily_pnl_trades}, daily avg trades: {best_daily_pnl_trades/trading_days:.2f}")
        
    return results_df

def plot_threshold_results(results_df, save_path, trading_days=1):
    """绘制阈值分析结果图表"""
    if results_df.empty:
        logger.error("Results are empty, cannot create plot")
        return
        
    # 创建单个图表，两个Y轴
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()
    
    # 绘制平均单笔PNL (左Y轴)
    line1 = ax1.plot(results_df['threshold'], results_df['avg_pnl'], 'b-', marker='o', label='Average PNL per Trade')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Average PNL per Trade', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # 绘制每日平均PNL (右Y轴) - 修改这部分以显示每日平均PNL
    line2 = ax2.plot(results_df['threshold'], results_df['daily_avg_pnl'], 'g-', marker='s', label='Average Daily PNL')
    ax2.set_ylabel('Average Daily PNL', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
    # 标记最佳平均单笔PNL点
    best_avg_idx = results_df['avg_pnl'].argmax()
    best_avg_threshold = results_df.iloc[best_avg_idx]['threshold']
    best_avg_pnl = results_df.iloc[best_avg_idx]['avg_pnl']
    ax1.plot(best_avg_threshold, best_avg_pnl, 'bo', markersize=10)
    
    # 标记最佳每日平均PNL点
    best_daily_idx = results_df['daily_avg_pnl'].argmax()
    best_daily_threshold = results_df.iloc[best_daily_idx]['threshold']
    best_daily_pnl = results_df.iloc[best_daily_idx]['daily_avg_pnl']
    ax2.plot(best_daily_threshold, best_daily_pnl, 'go', markersize=10)
    
    # 添加图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    plt.legend(lines, labels, loc='upper right')
    
    plt.title('Threshold vs. PNL')
    plt.grid(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path)
    logger.info(f"PNL analysis chart saved to: {save_path}")
    
    # 绘制交易次数与阈值的关系图
    plt.figure(figsize=(10, 6))
    
    # 创建两个y轴
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    # 绘制总交易次数
    line1 = ax1.plot(results_df['threshold'], results_df['total_trades'], 'r-', marker='o', label='Total Trades')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Total Trades')
    
    # 绘制每日平均交易次数
    daily_avg_trades = results_df['total_trades'] / trading_days
    line2 = ax2.plot(results_df['threshold'], daily_avg_trades, 'b-', marker='s', label='Daily Average Trades')
    ax2.set_ylabel('Daily Average Trades')
    
    # 添加图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    plt.title('Threshold vs. Trades')
    plt.grid(True)
    
    # 保存图片
    trades_path = save_path.replace('.png', '_trades.png')
    plt.savefig(trades_path)
    logger.info(f"Trades count chart saved to: {trades_path}")
    
    return

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Starting threshold optimization analysis")
    
    # 获取输入文件名，如果命令行参数提供则使用，否则使用默认值
    input_file = "infer_result/big_214403.csv"
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        
    # 设置阈值范围
    start_threshold = 0.9
    end_threshold = 0.995
    step_size = 0.005
    min_daily_trades_required = 3  # 每日最小交易次数
    
    # 设置交易天数计算参数
    minutes_per_day = 1440  # 默认每天1440分钟 (24小时)
    
    # 读取文件行数来计算交易天数
    try:
        data = pd.read_csv(input_file)
        total_minutes = len(data)
        trading_days = total_minutes / minutes_per_day
        logger.info(f"File contains {total_minutes} minute data rows, estimated trading days: {trading_days:.2f}")
        print(f"File contains {total_minutes} minute data rows, estimated trading days: {trading_days:.2f}")
    except Exception as e:
        logger.error(f"Error reading file to calculate trading days: {e}")
        trading_days = 1  # 出错默认为1天
        print(f"Could not read file to calculate trading days, using default value of 1 day")
    
    # 运行阈值分析
    results = analyze_thresholds(
        filename=input_file,
        start_threshold=start_threshold,
        end_threshold=end_threshold,
        step=step_size,
        min_daily_trades=min_daily_trades_required,  # 修改参数名
        trading_days=trading_days
    )
    
    # 保存结果到CSV
    csv_path = os.path.join(threshold_img_dir, f"threshold_results_{timestamp}.csv")
    results.to_csv(csv_path, index=False)
    logger.info(f"Complete results saved to: {csv_path}")
    
    # 绘制并保存结果图表
    plot_path = os.path.join(threshold_img_dir, f"threshold_analysis_{timestamp}.png")
    plot_threshold_results(results, plot_path, trading_days)
    
    logger.info("Threshold optimization analysis completed")
    print("Threshold optimization analysis completed")
