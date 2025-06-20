import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, linregress

def load_data(json_file):
    """加载JSON数据"""
    with open(json_file, 'r') as f:
        return json.load(f)

def interpret_correlation(r, p):
    """解释相关系数的意义"""
    # 判断相关强度
    abs_r = abs(r)
    if abs_r < 0.3:
        strength = "弱相关"
    elif abs_r < 0.5:
        strength = "中等相关"
    elif abs_r < 0.7:
        strength = "较强相关"
    else:
        strength = "强相关"
    
    # 判断方向
    direction = "正" if r > 0 else "负"
    
    # 判断显著性
    significance = "统计显著" if p < 0.05 else "统计不显著"
    
    return f"{strength} ({direction}相关), {significance} (p<0.05)"

def plot_scatter(data, output_file="scatter_plot.png"):
    """绘制散点图并添加统计分析"""
    # 提取数据
    focus_acc = np.array([item['focus_acc'] for item in data])
    avg_conf = np.array([item['avg_conf'] for item in data])
    
    # 计算描述性统计
    focus_acc_stats = {
        'mean': np.mean(focus_acc),
        'median': np.median(focus_acc),
        'std': np.std(focus_acc),
        'min': np.min(focus_acc),
        'max': np.max(focus_acc)
    }
    
    avg_conf_stats = {
        'mean': np.mean(avg_conf),
        'median': np.median(avg_conf),
        'std': np.std(avg_conf),
        'min': np.min(avg_conf),
        'max': np.max(avg_conf)
    }
    
    # 计算相关系数和回归分析
    correlation, p_value = pearsonr(focus_acc, avg_conf)
    slope, intercept, r_value, p_value_reg, std_err = linregress(focus_acc, avg_conf)
    
    # 创建图表
    plt.figure(figsize=(12, 9))
    plt.scatter(focus_acc, avg_conf, alpha=0.7, s=80, c='blue', edgecolors='w')
    
    # 添加标题和标签
    plt.title('Focus Accuracy vs Average Confidence', fontsize=18)
    plt.xlabel('Focus Accuracy', fontsize=15)
    plt.ylabel('Average Confidence', fontsize=15)
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 添加趋势线
    x_range = np.linspace(min(focus_acc), max(focus_acc), 100)
    plt.plot(x_range, slope * x_range + intercept, "r--", linewidth=2.5, 
             label=f'Trend: y = {slope:.4f}x + {intercept:.4f}')
    
    # 添加统计信息框
    stats_text = (
        f'Pearson Correlation (r): {correlation:.4f}\n'
        f'P-value: {p_value:.4e}\n'
        f'R-squared: {r_value**2:.4f}\n'
    )
    plt.text(0.05, 0.95, stats_text, 
             transform=plt.gca().transAxes, fontsize=13, 
             bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))
    
    # 添加描述性统计
    desc_text = (
        f'Focus Accuracy Stats:\n'
        f'  Mean: {focus_acc_stats["mean"]:.4f}, Median: {focus_acc_stats["median"]:.4f}\n'
        f'  Std: {focus_acc_stats["std"]:.4f}, Range: [{focus_acc_stats["min"]:.4f}, {focus_acc_stats["max"]:.4f}]\n\n'
        f'Avg Confidence Stats:\n'
        f'  Mean: {avg_conf_stats["mean"]:.4f}, Median: {avg_conf_stats["median"]:.4f}\n'
        f'  Std: {avg_conf_stats["std"]:.4f}, Range: [{avg_conf_stats["min"]:.4f}, {avg_conf_stats["max"]:.4f}]'
    )
    plt.text(0.05, 0.05, desc_text, 
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(facecolor='whitesmoke', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # 设置坐标轴范围
    plt.xlim(min(focus_acc) - 0.01, max(focus_acc) + 0.01)
    plt.ylim(min(avg_conf) - 0.05, max(avg_conf) + 0.05)
    
    # 添加图例
    plt.legend(fontsize=13, loc='lower right')
    
    # 保存和显示图表
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    # 控制台输出详细统计信息
    print("\n" + "="*60)
    print("焦点准确度 (Focus Accuracy) 统计摘要:")
    print(f"  平均值: {focus_acc_stats['mean']:.6f}")
    print(f"  中位数: {focus_acc_stats['median']:.6f}")
    print(f"  标准差: {focus_acc_stats['std']:.6f}")
    print(f"  最小值: {focus_acc_stats['min']:.6f}")
    print(f"  最大值: {focus_acc_stats['max']:.6f}")
    
    print("\n平均置信度 (Average Confidence) 统计摘要:")
    print(f"  平均值: {avg_conf_stats['mean']:.6f}")
    print(f"  中位数: {avg_conf_stats['median']:.6f}")
    print(f"  标准差: {avg_conf_stats['std']:.6f}")
    print(f"  最小值: {avg_conf_stats['min']:.6f}")
    print(f"  最大值: {avg_conf_stats['max']:.6f}")
    
    print("\n" + "="*60)
    print("相关性分析:")
    print(f"  皮尔逊相关系数 (r): {correlation:.6f}")
    print(f"  P值: {p_value:.6e}")
    print(f"  R平方: {r_value**2:.6f}")
    print(f"  回归方程: y = {slope:.6f}x + {intercept:.6f}")
    print(f"  解释: {interpret_correlation(correlation, p_value)}")
    print("="*60)
    print(f"图表已保存至: {output_file}")

# 主程序
if __name__ == "__main__":
    # 替换为你的JSON文件路径
    json_file = "infer_result/infer_bal.json"
    output_image = "Denhance/balpic1.png"
    
    # 加载数据并绘图
    data = load_data(json_file)
    plot_scatter(data, output_image)