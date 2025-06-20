import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr  # 用于计算相关系数

def load_data(json_file):
    """加载JSON数据"""
    with open(json_file, 'r') as f:
        return json.load(f)

def plot_focus_acc_over_time(json_file1, json_file2, output_file="focus_acc_over_time.png"):
    """绘制两个JSON文件中focus_acc随时间的变化曲线"""
    # 加载数据
    data1 = load_data(json_file1)
    data2 = load_data(json_file2)
    
    # 提取focus_acc数据
    focus_acc1 = np.array([item['focus_acc'] for item in data1])
    focus_acc2 = np.array([item['focus_acc'] for item in data2])
    
    # 计算相关系数
    correlation = np.corrcoef(focus_acc1, focus_acc2)[0, 1]
    pearson_r, pearson_p = pearsonr(focus_acc1, focus_acc2)
    
    # 创建时间序列（项目索引）
    time_points = np.arange(1, len(data1) + 1)  # 从1开始编号
    
    # 创建图表
    plt.figure(figsize=(12, 7))
    
    # 绘制两条曲线
    plt.plot(time_points, focus_acc1, 'o-', linewidth=2.5, markersize=8, 
             color='#1f77b4', label='File 1', alpha=0.9)
    plt.plot(time_points, focus_acc2, 's--', linewidth=2.5, markersize=8, 
             color='#ff7f0e', label='File 2', alpha=0.9)
    
    # 添加标题和标签
    plt.title('Focus Accuracy Over Time', fontsize=16)
    plt.xlabel('Time (Item Index)', fontsize=14)
    plt.ylabel('Focus Accuracy', fontsize=14)
    
    # 设置坐标轴范围
    all_focus = np.concatenate([focus_acc1, focus_acc2])
    plt.ylim(min(all_focus) - 0.01, max(all_focus) + 0.01)
    plt.xlim(0.5, len(time_points) + 0.5)
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加图例
    plt.legend(fontsize=12, loc='best')
    
    # 添加数据点标签（可选）
    for i, (t, acc1, acc2) in enumerate(zip(time_points, focus_acc1, focus_acc2)):
        if i % 4 == 0:  # 每隔4个点标注一次
            plt.text(t, acc1 + 0.005, f'{acc1:.3f}', 
                     ha='center', va='bottom', fontsize=9, color='#1f77b4')
            plt.text(t, acc2 - 0.005, f'{acc2:.3f}', 
                     ha='center', va='top', fontsize=9, color='#ff7f0e')
    
    # 添加整体统计信息
    avg1 = np.mean(focus_acc1)
    avg2 = np.mean(focus_acc2)
    plt.axhline(y=avg1, color='#1f77b4', linestyle=':', alpha=0.5, 
                label=f'File 1 Avg: {avg1:.4f}')
    plt.axhline(y=avg2, color='#ff7f0e', linestyle=':', alpha=0.5, 
                label=f'File 2 Avg: {avg2:.4f}')
    
    # 添加差异分析
    diff = focus_acc1 - focus_acc2
    avg_diff = np.mean(diff)
    plt.text(0.98, 0.02, f'Avg Difference: {avg_diff:.4f}', 
             transform=plt.gca().transAxes, ha='right', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))
    
    # 添加相关系数信息
    stats_text = (
        f'Pearson Correlation: {correlation:.4f}\n'
        f'P-value: {pearson_p:.4e}'
    )
    plt.text(0.02, 0.98, stats_text, 
             transform=plt.gca().transAxes, ha='left', va='top', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))
    
    # 保存和显示图表
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    # 控制台输出相关系数信息
    print("\n相关系数分析:")
    print(f"Pearson相关系数 (r): {correlation:.6f}")
    print(f"P值: {pearson_p:.6e}")
    print(f"解释: {interpret_correlation(correlation, pearson_p)}")
    print(f"图表已保存至: {output_file}")
    
    # 返回相关系数以供进一步分析
    return correlation, pearson_p

def interpret_correlation(r, p):
    """解释相关系数的意义"""
    strength = ""
    if abs(r) < 0.3:
        strength = "弱相关"
    elif abs(r) < 0.7:
        strength = "中等相关"
    else:
        strength = "强相关"
    
    direction = "正" if r > 0 else "负"
    
    significance = "统计显著" if p < 0.05 else "统计不显著"
    
    return f"{strength} ({direction}相关), {significance} (p<0.05)"

# 主程序
if __name__ == "__main__":
    # 替换为你的JSON文件路径
    json_file1 = "infer_result/infer_bal.json"
    json_file2 = "infer_result/infer_strict.json"
    output_image = "Denhance/focus_acc_over_time.png"
    
    # 绘图并获取相关系数
    correlation, p_value = plot_focus_acc_over_time(json_file1, json_file2, output_image)