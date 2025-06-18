# 本文件用于分析标签分布的统计特性和相关性

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_label_data(file_path):
    """加载标签分布数据"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 创建一个字典来存储数据，后续转换为DataFrame
    processed_data = {
        'datetime': [],
        'label_0': [],
        'label_1': [],
        'label_2': []
    }
    
    # 按时间排序处理数据
    for date_str in sorted(data.keys()):
        # 解析日期时间字符串
        date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        processed_data['datetime'].append(date_obj)
        
        # 提取标签比例
        processed_data['label_0'].append(float(data[date_str]['0']))
        processed_data['label_1'].append(float(data[date_str]['1']))
        processed_data['label_2'].append(float(data[date_str]['2']))
    
    # 转换为DataFrame
    df = pd.DataFrame(processed_data)
    
    # 添加衍生指标
    df['label_0+2'] = df['label_0'] + df['label_2']
    df['label_2-0'] = df['label_2'] - df['label_0']
    
    return df

def calculate_descriptive_stats(df):
    """计算描述性统计指标"""
    # 选择感兴趣的列进行统计
    columns_of_interest = ['label_0', 'label_2', 'label_0+2', 'label_2-0']
    
    # 计算描述性统计量
    desc_stats = df[columns_of_interest].describe()
    
    # 添加额外的统计量
    desc_stats.loc['range'] = desc_stats.loc['max'] - desc_stats.loc['min']
    desc_stats.loc['var'] = df[columns_of_interest].var()
    desc_stats.loc['skew'] = df[columns_of_interest].skew()
    desc_stats.loc['kurtosis'] = df[columns_of_interest].kurtosis()
    
    return desc_stats

def calculate_correlation(df):
    """计算相关系数矩阵"""
    columns_of_interest = ['label_0', 'label_2', 'label_0+2', 'label_2-0']
    return df[columns_of_interest].corr()

def plot_correlation_heatmap(corr_matrix, output_path=None):
    """绘制相关性热力图"""
    plt.figure(figsize=(10, 8))
    
    # 使用seaborn绘制热力图 - 移除掩码，显示完整对称热力图
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", 
                vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    
    plt.title('Correlation Matrix of Label Proportions', fontsize=16)
    plt.tight_layout()
    
    # 保存图表
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Correlation heatmap has been saved to {output_path}")

def main():
    # 读取JSON文件
    file_path = 'result/label_distribution.json'
    df = load_label_data(file_path)
    
    # 计算描述性统计量
    desc_stats = calculate_descriptive_stats(df)
    
    # 计算相关系数矩阵
    corr_matrix = calculate_correlation(df)
    
    # 打印结果
    print("\n===== Descriptive Statistics =====")
    print(desc_stats)
    
    print("\n===== Correlation Matrix =====")
    print(corr_matrix)
    
    # 将结果保存到CSV文件
    desc_stats.to_csv('result/label_descriptive_stats.csv')
    corr_matrix.to_csv('result/label_correlation_matrix.csv')
    
    # 绘制相关性热力图
    plot_correlation_heatmap(corr_matrix, 'result/label_correlation_heatmap.png')
    
    print("\nAnalysis complete. Results saved to CSV files and images.")

if __name__ == "__main__":
    main()
