import pandas as pd
from sklearn.model_selection import train_test_split
import os

input_file = "data/BTCUSDT_2_01.csv"
df = pd.read_csv(input_file)
total_len = len(df)
interval = int(total_len * 0.1)
df1 = df.iloc[total_len - 7 * interval: total_len - 5 * interval].reset_index(drop=True)
df2 = df.iloc[total_len - interval: total_len].reset_index(drop=True)
for col in ['high', 'low', 'close']:
    df2[col] = (df[col] - df['open']) / df['open']
df2.drop(columns=['open'], inplace=True)

df1.to_csv("data/cut_2_01.csv")
# df2.to_csv("splits/inference_1_02.csv")