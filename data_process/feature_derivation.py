import pandas as pd
import numpy as np

# 读取数据
print("开始读取数据...")
df = pd.read_csv('data/BTCUSDT.csv')
print("数据读取完成")

# 计算特征1: (十分钟后的最高价 - 一分钟后最低价) / 当前最低价
df['feature1'] = (df['high'].shift(-10) - df['low'].shift(-1)) / df['low']

# 计算特征2: (十分钟后的最低价 - 一分钟后最高价) / 当前最高价
df['feature2'] = (df['low'].shift(-10) - df['high'].shift(-1)) / df['high']

# 计算特征3: (十分钟后的中间价 - 一分钟后中间价) / 当前中间价
df['mid'] = (df['high'] + df['low']) / 2
df['feature3'] = (df['mid'].shift(-10) - df['mid'].shift(-1)) / df['mid']

# 创建标签
def create_label(row):
    if row['feature1'] < 0:
        return 0
    elif row['feature2'] > 0:
        return 2
    else:
        return 1

df['label'] = df.apply(create_label, axis=1)

# 处理因shift操作产生的NaN值
df = df.dropna().reset_index(drop=True)

# 保存结果
df.to_csv('data/BTCUSDT_feature_new.csv', index=False)

print(f"特征衍生完成，共处理 {len(df)} 条数据")
print(f"标签分布:")
for i in range(9):
    print(f"标签 {i}: {len(df[df['label'] == i])} 条")
