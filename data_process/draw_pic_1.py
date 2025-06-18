# 本文件用来根据之前计算出的标签分布数据绘制图表
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

# 确保matplotlib使用英文
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_label_data(file_path):
    """加载标签分布数据"""
    with open(file_path, 'r') as f:
        data = json.load(f)

    # 提取日期和标签比例
    dates = []
    label_0_values = []
    label_2_values = []

    # 按时间排序处理数据
    for date_str in sorted(data.keys()):
        # 解析日期时间字符串
        date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        dates.append(date_obj)
        
        # 提取标签比例
        label_0 = float(data[date_str]['0'])
        label_2 = float(data[date_str]['2'])
        label_0_values.append(label_0)
        label_2_values.append(label_2)
    
    return dates, label_0_values, label_2_values

def plot_label_proportions(dates, label_0_values, label_2_values, output_path=None):
    """绘制标签0和标签2比例随时间的变化图"""
    plt.figure(figsize=(14, 7))

    # 绘制两条线
    plt.plot(dates, label_0_values, 'b-o', label='Label 0', linewidth=2, markersize=5)
    plt.plot(dates, label_2_values, 'r-s', label='Label 2', linewidth=2, markersize=5)

    # 设置x轴格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=4))  # 每4个月标记一次
    plt.gcf().autofmt_xdate()  # 自动格式化x轴日期标签

    # 设置标题和标签
    plt.title('Proportion of Label 0 and Label 2 Over Time', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Proportion', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 设置y轴范围，确保比例变化更加明显
    plt.ylim(0, max(max(label_0_values), max(label_2_values)) * 1.1)

    # 保存图表
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Label proportions chart has been saved to {output_path}")

    plt.tight_layout()
    
    return plt.gcf()  # 返回图形对象

def plot_label_difference(dates, label_0_values, label_2_values, output_path=None):
    """绘制标签2减去标签0的差值随时间的变化图"""
    # 计算差值
    label_diff_values = [l2 - l0 for l2, l0 in zip(label_2_values, label_0_values)]
    
    plt.figure(figsize=(14, 7))

    # 绘制差值线
    plt.plot(dates, label_diff_values, 'g-^', label='Label 2 - Label 0', linewidth=2, markersize=5)

    # 添加零线参考
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # 设置x轴格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=4))  # 每4个月标记一次
    plt.gcf().autofmt_xdate()  # 自动格式化x轴日期标签

    # 设置标题和标签
    plt.title('Difference Between Label 2 and Label 0 Over Time', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Difference (Label 2 - Label 0)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 计算适当的Y轴范围，确保0点在适当位置
    diff_max = max(label_diff_values)
    diff_min = min(label_diff_values)
    y_margin = max(abs(diff_max), abs(diff_min)) * 0.2
    plt.ylim(diff_min - y_margin, diff_max + y_margin)

    # 保存图表
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Label difference chart has been saved to {output_path}")

    plt.tight_layout()
    
    return plt.gcf()  # 返回图形对象

def plot_combined_charts(dates, label_0_values, label_2_values, output_path=None):
    """绘制上下排列的标签比例图和差值图"""
    # 创建一个具有2行1列的子图布局，高度比例为2:1
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # 在上方子图绘制标签比例
    ax1.plot(dates, label_0_values, 'b-o', label='Label 0', linewidth=2, markersize=5)
    ax1.plot(dates, label_2_values, 'r-s', label='Label 2', linewidth=2, markersize=5)
    
    # 设置上方子图的Y轴范围和标签
    ax1.set_ylim(0, max(max(label_0_values), max(label_2_values)) * 1.1)
    ax1.set_title('Proportion of Label 0 and Label 2 Over Time', fontsize=16)
    ax1.set_ylabel('Proportion', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 计算差值
    label_diff_values = [l2 - l0 for l2, l0 in zip(label_2_values, label_0_values)]
    
    # 在下方子图绘制差值
    ax2.plot(dates, label_diff_values, 'g-^', label='Label 2 - Label 0', linewidth=2, markersize=5)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # 设置下方子图的Y轴范围和标签
    diff_max = max(label_diff_values)
    diff_min = min(label_diff_values)
    y_margin = max(abs(diff_max), abs(diff_min)) * 0.2
    ax2.set_ylim(diff_min - y_margin, diff_max + y_margin)
    ax2.set_xlabel('Time', fontsize=14)
    ax2.set_ylabel('Difference\n(Label 2 - Label 0)', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 为两个子图设置相同的x轴格式
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    
    # 隐藏上方图的x轴标签，保持整洁
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    # 调整布局
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.1)  # 减小子图之间的垂直间距
    
    # 保存图表
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Combined chart has been saved to {output_path}")
    
    return fig

def main():
    # 读取JSON文件
    file_path = 'result/label_distribution.json'
    dates, label_0_values, label_2_values = load_label_data(file_path)
    
    # 绘制组合图表
    combined_output_path = 'result/combined_label_charts.png'
    plot_combined_charts(dates, label_0_values, label_2_values, combined_output_path)
    
    # 如果仍然需要单独的图表，保留原来的绘图函数调用，但默认注释掉
    # prop_output_path = 'result/label_proportion_over_time.png'
    # plot_label_proportions(dates, label_0_values, label_2_values, prop_output_path)
    
    # diff_output_path = 'result/label_difference_over_time.png'
    # plot_label_difference(dates, label_0_values, label_2_values, diff_output_path)
    
    # 显示图表
    plt.show()

# 执行主函数
if __name__ == "__main__":
    main()




