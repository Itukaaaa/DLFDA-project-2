import pandas as pd
import os
import json
import numpy as np

def calculate_label_distribution(file_path, chunk_size=100000):
    """
    计算CSV文件中按时间顺序的标签分布
    
    参数:
    file_path: CSV文件路径
    chunk_size: 每个块的行数，默认1000
    
    返回:
    包含标签分布的字典，键为时间戳，值为各标签的比例
    """
    try:
        # 读取CSV文件
        print(f"正在读取文件: {file_path}")
        df = pd.read_csv(file_path)
        
        # 只保留需要的列
        df = df[['trade_time', 'label']]
        
        # 初始化结果字典
        label_distribution = {}
        
        # 按每chunk_size行分块处理
        total_rows = len(df)
        num_chunks = (total_rows + chunk_size - 1) // chunk_size
        
        print(f"总行数: {total_rows}, 将分为 {num_chunks} 个块进行处理")
        
        for i in range(num_chunks):
            # 获取当前块的数据
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_rows)
            chunk = df.iloc[start_idx:end_idx]
            
            # 获取块的初始时间作为key
            chunk_start_time = chunk['trade_time'].iloc[0]
            
            # 计算label分布
            label_counts = chunk['label'].value_counts().to_dict()
            total_count = sum(label_counts.values())
            
            # 计算各label的比例并确保所有label都有值
            label_distribution[chunk_start_time] = {
                0: label_counts.get(0, 0) / total_count if total_count > 0 else 0,
                1: label_counts.get(1, 0) / total_count if total_count > 0 else 0,
                2: label_counts.get(2, 0) / total_count if total_count > 0 else 0
            }
            
            if (i + 1) % 10 == 0 or i == num_chunks - 1:
                print(f"已处理 {i+1}/{num_chunks} 个块")
        
        return label_distribution
        
    except Exception as e:
        print(f"处理文件时发生错误: {str(e)}")
        return {}

def save_distribution(distribution, output_path):
    """保存分布结果到JSON文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(distribution, f, ensure_ascii=False, indent=4)
    print(f"结果已保存到: {output_path}")

def analyze_label_statistics(distribution):
    """
    分析标签分布，计算平均值和标准差
    
    参数:
    distribution: 包含标签分布的字典
    
    返回:
    包含统计结果的字典
    """
    # 提取每个标签的概率列表
    label_0_probs = [chunk_dist[0] for chunk_dist in distribution.values()]
    label_1_probs = [chunk_dist[1] for chunk_dist in distribution.values()]
    label_2_probs = [chunk_dist[2] for chunk_dist in distribution.values()]
    
    # 计算平均值和标准差
    stats = {
        'mean': {
            '0': np.mean(label_0_probs),
            '1': np.mean(label_1_probs),
            '2': np.mean(label_2_probs)
        },
        'std': {
            '0': np.std(label_0_probs),
            '1': np.std(label_1_probs),
            '2': np.std(label_2_probs)
        }
    }
    
    return stats

def sort_and_save_probabilities(distribution, output_dir):
    """
    对标签0和标签2的概率进行排序并保存为CSV
    
    参数:
    distribution: 包含标签分布的字典
    output_dir: 输出目录
    """
    # 创建包含时间戳和标签概率的数据框
    data = []
    for timestamp in distribution.keys():
        data.append({
            'timestamp': timestamp,
            'label_0': distribution[timestamp][0],
            'label_1': distribution[timestamp][1],    
            'label_2': distribution[timestamp][2]
        })
    
    df = pd.DataFrame(data)
    
    # 保存统计信息
    stats = analyze_label_statistics(distribution)
    stats_df = pd.DataFrame({
        'label': ['0', '1', '2'],
        'mean': [stats['mean']['0'], stats['mean']['1'], stats['mean']['2']],
        'std': [stats['std']['0'], stats['std']['1'], stats['std']['2']]
    })
    stats_path = os.path.join(output_dir, "label_statistics.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"\n标签统计信息已保存到: {stats_path}")
    print("标签统计信息:")
    print(stats_df)
    
    # 按标签0概率排序并保存
    sorted_0_df = df.sort_values(by='label_0', ascending=False)
    sorted_0_path = os.path.join(output_dir, "label_0_sorted.csv")
    sorted_0_df.to_csv(sorted_0_path, index=False)
    print(f"\n按标签0概率排序的结果已保存到: {sorted_0_path}")
    
    # 按标签2概率排序并保存
    sorted_2_df = df.sort_values(by='label_2', ascending=False)
    sorted_2_path = os.path.join(output_dir, "label_2_sorted.csv")
    sorted_2_df.to_csv(sorted_2_path, index=False)
    print(f"按标签2概率排序的结果已保存到: {sorted_2_path}")
    
    return {
        'statistics': stats_df,
        'sorted_by_label_0': sorted_0_df,
        'sorted_by_label_2': sorted_2_df
    }

def main():
    # 项目路径
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 数据文件路径
    data_path = os.path.join(base_path, "data", "BTCUSDT_feature_derived.csv")
    
    # 输出目录
    results_dir = os.path.join(base_path, "result")
    os.makedirs(results_dir, exist_ok=True)
    
    # 输出文件路径
    output_path = os.path.join(results_dir, "label_distribution.json")
    
    # 计算标签分布
    distribution = calculate_label_distribution(data_path)
    
    if distribution:
        # 打印一些结果示例
        print("\n标签分布示例:")
        count = 0
        for time, dist in distribution.items():
            print(f"时间: {time}")
            print(f"标签 0: {dist[0]:.2%}")
            print(f"标签 1: {dist[1]:.2%}")
            print(f"标签 2: {dist[2]:.2%}")
            print("-" * 40)
            count += 1
            if count >= 5:  # 只显示前5个结果
                break
        
        # 保存结果
        save_distribution(distribution, output_path)
        
        # 分析标签统计信息并排序保存
        print("\n开始分析标签分布统计信息...")
        sort_and_save_probabilities(distribution, results_dir)

if __name__ == "__main__":
    main()
