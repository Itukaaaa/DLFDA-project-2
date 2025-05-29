import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import test

def multi_test(file1='infer_result/example.csv',file2='infer_result/example1.csv',long_threshold=0.52, short_threshold=0.49):
    df1 = test.process_data(file1,use_threshold=True,long_threshold=long_threshold, short_threshold=short_threshold)
    df2 = test.process_data(file2,use_threshold=True,long_threshold=long_threshold, short_threshold=short_threshold)
    true_labels1 = df1['label'].copy()
    pred_labels1 = df1['predicted_label'].copy()
    true_labels2 = df2['label'].copy()
    pred_labels2 = df2['predicted_label'].copy()
    print("File 1 Classification Report:")
    print(confusion_matrix(true_labels1, pred_labels1))
    print("File 2 Classification Report:")
    print(confusion_matrix(true_labels2, pred_labels2))
    
    for i in range(len(true_labels1)):
        if pred_labels1[i] == 2 and pred_labels2[i] == 2:
            pred_labels2[i] = 2
        elif pred_labels1[i] == 0 and pred_labels2[i] == 0:
            pred_labels2[i] = 0
        else:
            pred_labels2[i] = 1
    
    print("Combined Classification Report:")
    print(confusion_matrix(true_labels2, pred_labels2))
    return df2,pred_labels2
            
if __name__ == "__main__":
    df2,pred_labels2 = multi_test(file1='infer_result/example1.csv',file2='infer_result/example2.csv',long_threshold=0.5, short_threshold=0.45)
    return_df = test.calculate_10min_returns(df2['close'])
    
    total_return = 0.0
    totl_trade = 0
    for i in range(0,len(df2)-10):
        if pred_labels2[i] == 0:
            total_return += -return_df['10min_return'][i]
            totl_trade += 1
        elif pred_labels2[i] == 2:
            total_return += return_df['10min_return'][i]
            totl_trade += 1
        else:
            continue
    print(f"总的时间节点共：{len(df2)-10}, 总交易次数: {totl_trade}")
    print(f"总收益率: 万分之{total_return*10000}, 平均收益率: 万分之{total_return*10000/totl_trade if totl_trade > 0 else 0}")

    



