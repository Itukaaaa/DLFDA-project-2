import pandas as pd
from sklearn.model_selection import train_test_split
import os

input_file = "data/BTCUSDT-feature-label-balanced.csv"
df = pd.read_csv(input_file)
total_len = len(df)
interval = int(total_len * 0.1)
# df1 = df.iloc[total_len - 7 * interval: total_len - 5 * interval].reset_index(drop=True)
# df2 = df.iloc[1500000: 1700000].reset_index(drop=True)
# for col in ['high', 'low', 'close']:
#     df2[col] = (df[col] - df['open']) / df['open']
# df2.drop(columns=['open'], inplace=True)

# df1.to_csv("data/new_strict.csv")
# df2.to_csv("splits/inference.csv")

train = df.iloc[200000:500000].reset_index(drop=True)
val = df.iloc[1400000:1500000].reset_index(drop=True)
test = df.iloc[1700000:1800000].reset_index(drop=True)

# for col in ['high', 'low', 'close']:
#     train[col] = (train[col] - train['open']) / train['open']
#     val[col] = (val[col] - val['open']) / val['open']
#     test[col] = (test[col] - test['open']) / test['open']

# train.drop(columns=['open'], inplace=True)
# val.drop(columns=['open'], inplace=True)
# test.drop(columns=['open'], inplace=True)

train.to_csv("splits/train.csv")
# val.to_csv("splits/val.csv")
# test.to_csv("splits/test.csv")

pd.concat([train, val, test]).reset_index(drop=True).to_csv("data/integrate.csv")