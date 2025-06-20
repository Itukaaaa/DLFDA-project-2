import argparse
import pandas as pd
import numpy as np
import json

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()

p = argparse.ArgumentParser()
p.add_argument("--csv", required=True, help="完整推理文件")
p.add_argument("--json", default="infer_result/infer_bal.json")

args = p.parse_args()

df = pd.read_csv(args.csv)
total_len = len(df)
interval = 100000

otpt = []

for start in range(0, total_len, interval):
    end = min(start + interval, total_len)
    chunk = df.iloc[start:end]

    cur = {}

    cm = np.zeros((3, 3), dtype=int)
    conf = 0
    
    for i in range(len(chunk)):
        row = chunk.iloc[i]
        true_label = int(row['label'])
        
        # 获取预测概率并确定预测标签
        pred_values = np.array([row['pred0'], row['pred1'], row['pred2']])
        pred_probs = softmax(pred_values)
        pred_label = np.argmax(pred_probs)
        conf += pred_probs[pred_label]
        
        # 更新混淆矩阵
        cm[true_label][pred_label] += 1

    total_rows = len(chunk)
    focus_acc = (cm[0][0] + cm[2][2] - cm[0][2] - cm[2][0]) / (cm[0][0] + cm[1][0] + cm[2][0] + cm[0][2] + cm[1][2] + cm[2][2])
    avg_conf = conf / total_rows

    # 转换为原生 Python float 类型
    cur["focus_acc"] = float(focus_acc)
    cur["avg_conf"] = float(avg_conf)

    print(f"{start // interval + 1} epoches accoplished:")
    print(cur)

    otpt.append(cur)

with open(args.json, "w") as f:
    json.dump(otpt, f)