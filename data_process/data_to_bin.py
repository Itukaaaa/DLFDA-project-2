# 分类波动幅度 小为1（三分类中1），大为0（三分类中0、2）

import pandas as pd
from sklearn.model_selection import train_test_split
import os

input_file = "data/BTCUSDT-feature-label.csv"
df = pd.read_csv(input_file)
total_len = len(df)
df.loc[df['label'] == 0, 'label'] = 1
df.loc[df['label'] == 2, 'label'] = 0

df.to_csv("data/BTCUSDT_2_01.csv")