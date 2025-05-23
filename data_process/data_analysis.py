import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import logging

# 设置日志记录
log_dir = "result/analysis_output"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=f"{log_dir}/analysis_log.txt",
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 确保输出目录存在
output_dir = "analysis_output"
os.makedirs(output_dir, exist_ok=True)

csv_file = "data/BTCUSDT_feature_derived.csv"

# 读取CSV文件
try:
    df = pd.read_csv(csv_file)
    logging.info(f"Data loaded successfully, {len(df)} rows in total")
except Exception as e:
    logging.error(f"Error loading file: {e}")
    exit(1)

# 显示数据的基本信息
logging.info("\nData Basic Information:")
buffer = df.info(buf=None)
logging.info(df.info(buf=None))

logging.info("\nData Statistical Summary:")
logging.info("\n" + df.describe().to_string())

volume_col = 'volume' 

# 1. 成交量分析
logging.info(f"\n==== Volume ({volume_col}) Analysis ====")
volume_stats = {
    'Mean': df[volume_col].mean(),
    'Std Dev': df[volume_col].std(),
    'Min': df[volume_col].min(),
    '25th Percentile': df[volume_col].quantile(0.25),
    'Median': df[volume_col].median(),
    '75th Percentile': df[volume_col].quantile(0.75),
    'Max': df[volume_col].max(),
    'Total Count': len(df)
}

for key, value in volume_stats.items():
    logging.info(f"{key}: {value:.4f}")

# 绘制成交量分布图
plt.figure(figsize=(12, 6))

# 绘制直方图
plt.subplot(121)
sns.histplot(df[volume_col], kde=True)
plt.title(f'{volume_col} Distribution Histogram')
plt.xlabel(volume_col)
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# 绘制箱线图
plt.subplot(122)
sns.boxplot(y=df[volume_col])
plt.title(f'{volume_col} Boxplot')
plt.ylabel(volume_col)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}/volume_distribution.png")
logging.info(f"Volume distribution plot saved to {output_dir}/volume_distribution.png")
plt.close()

# 2. feature1 分析
feature1 = 'feature1'  # 替换为实际的feature1列名

logging.info(f"\n==== {feature1} Analysis ====")
feature1_stats = {
    'Mean': df[feature1].mean(),
    'Std Dev': df[feature1].std(),
    'Min': df[feature1].min(),
    '25th Percentile': df[feature1].quantile(0.25),
    'Median': df[feature1].median(),
    '75th Percentile': df[feature1].quantile(0.75),
    'Max': df[feature1].max(),
    'Count < -0.001': (df[feature1] < -0.001).sum(),
    'Percentage < -0.001': (df[feature1] < -0.001).mean() * 100,
    'Count < -0.0005': (df[feature1] < -0.0005).sum(),
    'Percentage < -0.0005': (df[feature1] < -0.0005).mean() * 100,
    'Count < 0': (df[feature1] < 0).sum(),
    'Percentage < 0': (df[feature1] < 0).mean() * 100
}

for key, value in feature1_stats.items():
    if 'Percentage' in key:
        logging.info(f"{key}: {value:.2f}%")
    else:
        logging.info(f"{key}: {value}")

# 绘制feature1分布图
plt.figure(figsize=(12, 6))

plt.subplot(121)
sns.histplot(df[feature1], kde=True)
plt.title(f'{feature1} Distribution Histogram')
plt.xlabel(feature1)
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.subplot(122)
sns.boxplot(y=df[feature1])
plt.title(f'{feature1} Boxplot')
plt.ylabel(feature1)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}/feature1_distribution.png")
logging.info(f"{feature1} distribution plot saved to {output_dir}/feature1_distribution.png")
plt.close()

# 3. feature2 分析
feature2 = 'feature2'  # 替换为实际的feature2列名

logging.info(f"\n==== {feature2} Analysis ====")
feature2_stats = {
    'Mean': df[feature2].mean(),
    'Std Dev': df[feature2].std(),
    'Min': df[feature2].min(),
    '25th Percentile': df[feature2].quantile(0.25),
    'Median': df[feature2].median(),
    '75th Percentile': df[feature2].quantile(0.75),
    'Max': df[feature2].max(),
    'Count > 0.001': (df[feature2] > 0.001).sum(),
    'Percentage > 0.001': (df[feature2] > 0.001).mean() * 100,
    'Count > 0.0005': (df[feature2] > 0.0005).sum(),
    'Percentage > 0.0005': (df[feature2] > 0.0005).mean() * 100,
    'Count > 0': (df[feature2] > 0).sum(),
    'Percentage > 0': (df[feature2] > 0).mean() * 100
}

for key, value in feature2_stats.items():
    if 'Percentage' in key:
        logging.info(f"{key}: {value:.2f}%")
    else:
        logging.info(f"{key}: {value}")

# 绘制feature2分布图
plt.figure(figsize=(12, 6))

plt.subplot(121)
sns.histplot(df[feature2], kde=True)
plt.title(f'{feature2} Distribution Histogram')
plt.xlabel(feature2)
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.subplot(122)
sns.boxplot(y=df[feature2])
plt.title(f'{feature2} Boxplot')
plt.ylabel(feature2)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}/feature2_distribution.png")
logging.info(f"{feature2} distribution plot saved to {output_dir}/feature2_distribution.png")
plt.close()

# 统计数据保存到日志而不是CSV
logging.info("\n\n==== Summary Statistics ====")
logging.info("\nVolume Statistics:")
for key, value in volume_stats.items():
    logging.info(f"{key}: {value:.4f}")

logging.info(f"\n{feature1} Statistics:")
for key, value in feature1_stats.items():
    if 'Percentage' in key:
        logging.info(f"{key}: {value:.2f}%")
    else:
        logging.info(f"{key}: {value}")

logging.info(f"\n{feature2} Statistics:")
for key, value in feature2_stats.items():
    if 'Percentage' in key:
        logging.info(f"{key}: {value:.2f}%")
    else:
        logging.info(f"{key}: {value}")

# 绘制feature1和feature2的散点图来分析关系
plt.figure(figsize=(10, 8))
sns.scatterplot(x=feature1, y=feature2, data=df, alpha=0.5)
plt.title(f'{feature1} vs {feature2} Scatter Plot')
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.grid(True, alpha=0.3)
plt.savefig(f"{output_dir}/feature1_feature2_scatter.png")
logging.info(f"{feature1} and {feature2} scatter plot saved to {output_dir}/feature1_feature2_scatter.png")
plt.close()

logging.info("\nAnalysis completed! All results saved to output directory.")
print("Analysis completed! Results saved to output directory and log file.")
