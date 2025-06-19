import pandas as pd
import os
import json
import numpy as np

def calculate_label_distribution(file_path, chunk_size=100000):
    """
    计算CSV文件中按时间顺序的标签分布（不使用时间字段）

    参数:
    file_path: CSV文件路径
    chunk_size: 每个块的行数，默认100000

    返回:
    包含标签分布的字典，键为块编号字符串，值为各标签的比例
    """
    try:
        print(f"正在读取文件: {file_path}")
        df = pd.read_csv(file_path)

        # 只保留标签列
        df = df[['label']]

        label_distribution = {}

        total_rows = len(df)
        num_chunks = (total_rows + chunk_size - 1) // chunk_size
        print(f"总行数: {total_rows}, 将分为 {num_chunks} 个块进行处理")

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_rows)
            chunk = df.iloc[start_idx:end_idx]

            label_counts = chunk['label'].value_counts().to_dict()
            total_count = sum(label_counts.values())

            label_distribution[f"chunk_{i}"] = {
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
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(distribution, f, ensure_ascii=False, indent=4)
    print(f"结果已保存到: {output_path}")

def analyze_label_statistics(distribution):
    label_0_probs = [chunk_dist[0] for chunk_dist in distribution.values()]
    label_1_probs = [chunk_dist[1] for chunk_dist in distribution.values()]
    label_2_probs = [chunk_dist[2] for chunk_dist in distribution.values()]

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
    data = []
    for chunk_id in distribution.keys():
        data.append({
            'chunk': chunk_id,
            'label_0': distribution[chunk_id][0],
            'label_1': distribution[chunk_id][1],
            'label_2': distribution[chunk_id][2]
        })

    df = pd.DataFrame(data)

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

    sorted_0_df = df.sort_values(by='label_0', ascending=False)
    sorted_0_path = os.path.join(output_dir, "label_0_sorted.csv")
    sorted_0_df.to_csv(sorted_0_path, index=False)
    print(f"\n按标签0概率排序的结果已保存到: {sorted_0_path}")

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
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, "data", "BTCUSDT-feature-label.csv")
    results_dir = os.path.join(base_path, "result")
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "label_distribution.json")

    distribution = calculate_label_distribution(data_path)

    if distribution:
        print("\n标签分布示例:")
        for i, (chunk_id, dist) in enumerate(distribution.items()):
            print(f"{chunk_id}:")
            print(f"标签 0: {dist[0]:.2%}")
            print(f"标签 1: {dist[1]:.2%}")
            print(f"标签 2: {dist[2]:.2%}")
            print("-" * 40)
            if i >= 4: break

        save_distribution(distribution, output_path)
        print("\n开始分析标签分布统计信息...")
        sort_and_save_probabilities(distribution, results_dir)

if __name__ == "__main__":
    main()
