import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import test
import logging
import os
import datetime

# 设置日志配置
log_dir = "backtest/logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 创建专用的logger
logger = logging.getLogger('multi_test')
logger.setLevel(logging.INFO)

# 检查是否已有处理器，避免重复添加
if not logger.handlers:
    # 创建文件处理器
    file_handler = logging.FileHandler(os.path.join(log_dir, "multi_backtest.log"), mode='a')
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 防止日志传递到根日志记录器
    logger.propagate = False

# 创建图片保存目录
img_dir = "backtest/multi_images"
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

def process_data(filename="infer_result/example.csv",use_threshold = True,long_threshold = 0.5, short_threshold = 0.5):
    logger.info(f"{filename} threshold settings: \nuse_threshold={use_threshold}, long_threshold={long_threshold}, short_threshold={short_threshold}")
    # Read the CSV file
    df = pd.read_csv(filename)
    
    # 先进行softmax操作
    pred_columns = ['pred0', 'pred1', 'pred2']
    df['total_exp'] = np.exp(df[pred_columns]).sum(axis=1)
    for col in pred_columns:
        df[col] = np.exp(df[col]) / df['total_exp']
    
    # 根据三列中的最大值创建预测标签
    df['predicted_label'] = df[pred_columns].idxmax(axis=1)       
    
    # 将'predX'转换为数字X
    df['predicted_label'] = df['predicted_label'].str.replace('pred', '').astype(int)
    
    # 应用阈值规则 - 使用向量化操作
    if use_threshold:
        mask_0 = (df['predicted_label'] == 0) & (df['pred0'] > short_threshold)
        mask_2 = (df['predicted_label'] == 2) & (df['pred2'] > long_threshold)
        mask_default = ~(mask_0 | mask_2)
        df.loc[mask_default, 'predicted_label'] = 1
        
    logger.info(f"Data processing completed, {len(df)} rows processed")
    return df

def process_data_bin(filename="infer_result/example.csv"):
    logger.info(f"{filename}")
    df = pd.read_csv(filename)
    pred_columns = ['pred0', 'pred1']
    df['total_exp'] = np.exp(df[pred_columns]).sum(axis=1)
    for col in pred_columns:
        df[col] = np.exp(df[col]) / df['total_exp']
    df['predicted_label'] = df[pred_columns].idxmax(axis=1)
    df['predicted_label'] = df['predicted_label'].str.replace('pred', '').astype(int)
    logging.info(f"Data processing completed, {len(df)} rows processed")
    return df

def multi_test(file1='infer_result/example.csv',file2='infer_result/example1.csv',long_threshold=0.5, short_threshold=0.5):
    logger.info(f"Multi test started with files: {file1} and {file2}")
    logger.info(f"Threshold settings: long_threshold={long_threshold}, short_threshold={short_threshold}")
    
    df1 = process_data(file1,use_threshold=True,long_threshold=long_threshold, short_threshold=short_threshold)
    df2 = process_data(file2,use_threshold=True,long_threshold=long_threshold, short_threshold=short_threshold)
    true_labels1 = df1['label'].copy()
    pred_labels1 = df1['predicted_label'].copy()
    true_labels2 = df2['label'].copy()
    pred_labels2 = df2['predicted_label'].copy()
    
    cm1 = confusion_matrix(true_labels1, pred_labels1)
    logger.info(f"File 1 Confusion matrix:\n{cm1}")
    print("File 1 Classification Report:")
    print(confusion_matrix(true_labels1, pred_labels1))
    
    cm2 = confusion_matrix(true_labels2, pred_labels2)
    logger.info(f"File 2 Confusion matrix:\n{cm2}")
    print("File 2 Classification Report:")
    print(confusion_matrix(true_labels2, pred_labels2))
    
    for i in range(len(true_labels1)):
        if pred_labels1[i] == 2 and pred_labels2[i] == 2:
            pred_labels2[i] = 2
        elif pred_labels1[i] == 0 and pred_labels2[i] == 0:
            pred_labels2[i] = 0
        else:
            pred_labels2[i] = 1
    
    cm_combined = confusion_matrix(true_labels2, pred_labels2)
    logger.info(f"Combined Confusion matrix:\n{cm_combined}")
    print("Combined Classification Report:")
    print(confusion_matrix(true_labels2, pred_labels2))
    
    # 计算并记录准确率
    accuracy = np.mean(true_labels2 == pred_labels2)
    logger.info(f"Combined accuracy: {accuracy:.4f}")
    
    # 记录分类报告
    report = classification_report(true_labels2, pred_labels2)
    logger.info(f"Combined classification report:\n{report}")
    
    # 可视化混淆矩阵
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_combined, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['0', '1', '2'], 
                yticklabels=['0', '1', '2'])
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.title('Combined Confusion Matrix')
    plt.tight_layout()
    
    # 使用时间戳保存图片
    cm_img_path = os.path.join(img_dir, f"combined_confusion_matrix_{timestamp}.png")
    plt.savefig(cm_img_path)
    logger.info(f"Combined confusion matrix saved to: {cm_img_path}")
    
    return df2,pred_labels2

def multi_test_bin(file1='infer_result/example1.csv',file2='infer_result/example_bin1.csv',long_threshold=0.5, short_threshold=0.5):
    # 请注意一般这里第一个用来传三分类，第二个用来传二分类
    logger.info(f"Multi test BIN started with files: {file1} and {file2}")
    logger.info(f"Threshold settings: long_threshold={long_threshold}, short_threshold={short_threshold}")
    
    df1 = process_data(file1,use_threshold=True,long_threshold=long_threshold, short_threshold=short_threshold)
    df2 = process_data_bin(file2)
    true_labels1 = df1['label'].copy()
    pred_labels1 = df1['predicted_label'].copy()
    true_labels2 = df2['label'].copy()
    pred_labels2 = df2['predicted_label'].copy()
    
    cm1 = confusion_matrix(true_labels1, pred_labels1)
    logger.info(f"File 1 Confusion matrix:\n{cm1}")
    print("File 1 Classification Report:")
    print(confusion_matrix(true_labels1, pred_labels1))
    
    for i in range(len(true_labels1)):
        if pred_labels1[i] == 2 and pred_labels2[i] == 0:
            pred_labels1[i] = 2
        elif pred_labels1[i] == 0 and pred_labels2[i] == 0:
            pred_labels1[i] = 0
        else:
            pred_labels1[i] = 1
    
    cm_combined = confusion_matrix(true_labels1, pred_labels1)
    logger.info(f"Combined Confusion matrix:\n{cm_combined}")
    print("Combined Classification Report:")
    print(confusion_matrix(true_labels1, pred_labels1))
    
    # 计算并记录准确率
    accuracy = np.mean(true_labels1 == pred_labels1)
    logger.info(f"Combined accuracy: {accuracy:.4f}")
    
    # 记录分类报告
    report = classification_report(true_labels1, pred_labels1)
    logger.info(f"Combined classification report:\n{report}")
    
    # 可视化混淆矩阵
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_combined, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['0', '1', '2'], 
                yticklabels=['0', '1', '2'])
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.title('Combined Confusion Matrix')
    plt.tight_layout()
    
    # 使用时间戳保存图片
    cm_img_path = os.path.join(img_dir, f"combined_confusion_matrix_{timestamp}.png")
    plt.savefig(cm_img_path)
    logger.info(f"Combined confusion matrix saved to: {cm_img_path}")
    
    return df1,pred_labels1
            
if __name__ == "__main__":
    logger.info("=====================================")
    logger.info("Starting multi-model backtest")
    
    df2,pred_labels2 = multi_test(file1='infer_result/big_204822.csv',file2='infer_result/big_214403.csv',long_threshold=0.55, short_threshold=0.54)
    # df2,pred_labels2 = multi_test_bin(file1='infer_result/big_230741.csv',file2='infer_result/big_1-02_105510.csv',long_threshold=0.6, short_threshold=0.56)
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
            
    result_msg = f"Total minutes:{len(df2)-10}, Total trade: {totl_trade}"
    print(result_msg)
    logger.info(result_msg)
    
    return_msg = f"Total pnl:{total_return*10000:.4f}, avergae pnl:{total_return*10000/totl_trade if totl_trade > 0 else 0:.4f}"
    print(return_msg)
    logger.info(return_msg)
    
    logger.info("Multi-model backtest finished\n")
    print("Finished\n")





