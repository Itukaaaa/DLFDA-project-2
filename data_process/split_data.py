import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 设置文件路径
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
input_file = os.path.join(data_dir, 'BTCUSDT_feature_derived.csv')

# 读取数据
print(f"正在读取数据: {input_file}")
df = pd.read_csv(input_file)
print(f"数据形状: {df.shape}")

df['trade_time'] = pd.to_datetime(df['trade_time'])
df = df.sort_values('trade_time').reset_index(drop=True)

total = len(df)
train_end = int(total * 0.7)    # 前70%训练集
val_end = train_end + int(total * 0.1)  # 接下来10%验证集

train_data = df.iloc[:train_end]
val_data = df.iloc[train_end:val_end]
test_data = df.iloc[val_end:]   # 最后20%测试集

# 重置每个子集的索引（可选）
train_data = train_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

# 打印分割后的数据集大小
print(f"训练集大小: {train_data.shape} ({len(train_data)/len(df)*100:.1f}%)")
print(f"验证集大小: {val_data.shape} ({len(val_data)/len(df)*100:.1f}%)")
print(f"测试集大小: {test_data.shape} ({len(test_data)/len(df)*100:.1f}%)")

# 保存分割后的数据集
train_file = os.path.join(data_dir, 'train.csv')
val_file = os.path.join(data_dir, 'val.csv')
test_file = os.path.join(data_dir, 'test.csv')

train_data.to_csv(train_file, index=False)
val_data.to_csv(val_file, index=False)
test_data.to_csv(test_file, index=False)

print(f"数据已成功分割并保存到:")
print(f"训练集: {train_file}")
print(f"验证集: {val_file}")
print(f"测试集: {test_file}")
