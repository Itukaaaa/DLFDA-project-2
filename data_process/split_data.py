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

# 第一次分割：将数据分为训练集(70%)和临时集(30%)
train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42)

# 第二次分割：将临时集分为验证集(10%)和测试集(20%)
val_data, test_data = train_test_split(temp_data, test_size=2/3, random_state=42)

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
