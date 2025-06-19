import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import datetime

# 设置日志配置
log_dir = "backtest/logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    filename=os.path.join(log_dir, "backtest.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"  # 追加模式
)

# 创建图片保存目录
img_dir = "backtest/images"
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

def process_data(filename="infer_result/example.csv",use_threshold = True,long_threshold = 0.5, short_threshold = 0.5):
    logging.info(f"{filename} threshold settings: \nuse_threshold={use_threshold}, long_threshold={long_threshold}, short_threshold={short_threshold}")
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
        
    logging.info(f"Data processing completed, {len(df)} rows processed")
    return df

    
def calculate_10min_returns(relative_returns, initial_price=7000):
    """
    根据相对收益率计算绝对价格序列和10分钟收益率
    
    参数:
    relative_returns: 包含每分钟相对收益率的DataFrame或Series
    initial_price: 初始价格，默认为7000美元
    
    返回:
    包含绝对价格和10分钟收益率的DataFrame
    """
    
    # 确保数据是DataFrame格式
    if isinstance(relative_returns, pd.Series):
        relative_returns = relative_returns.to_frame(name='relative_return')
    
    # 复制数据，避免修改原始数据
    df = relative_returns.copy()
    
    # 计算绝对价格序列
    absolute_prices = [initial_price]
    
    for i in range(len(df)):
        # 当前价格 = 前一分钟价格 * (1 + 相对收益率)
        current_price = absolute_prices[-1] * (1 + df.iloc[i]['relative_return'])
        absolute_prices.append(current_price)
    
    # 去掉第一个初始价格（因为它不对应任何时间点的相对收益率）
    absolute_prices = absolute_prices[1:]
    
    # 将绝对价格添加到DataFrame中
    df['absolute_price'] = absolute_prices
    
    # 计算10分钟收益率：10分钟之后的收盘价减去1分钟之后的收盘价，计算相对收益率
    df['10min_return'] = (df['absolute_price'].shift(-10) - df['absolute_price'].shift(-1)) / df['absolute_price'].shift(-1)
    
    return df

if __name__ == "__main__":
    logging.info("=====================================")
    
    # 调用测试函数
    df = process_data(filename = "infer_result/middle_204822.csv",use_threshold=True,
                        long_threshold=0.52, short_threshold=0.5)
    return_df = calculate_10min_returns(df['close'])
    
    true_labels = df['label'].values
    predicted_labels = df['predicted_label'].values
    
    accuracy = np.mean(true_labels == predicted_labels)
    logging.info(f"Overall accuracy: {accuracy:.4f}")
    print("整体准确率:", accuracy)
    
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, predicted_labels)
    
    logging.info(f"Confusion matrix:\n{cm}")
    print("混淆矩阵:\n", cm)
    
    # 打印分类报告
    report = classification_report(true_labels, predicted_labels)
    logging.info(f"Classification report:\n{report}")
    print(report)
    
    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['0', '1', '2'], 
                yticklabels=['0', '1', '2'])
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.title('confusion matrix')
    plt.tight_layout()
    
    # 使用时间戳保存图片
    cm_img_path = os.path.join(img_dir, f"confusion_matrix_{timestamp}.png")
    plt.savefig(cm_img_path)
    logging.info(f"Confusion matrix saved to: {cm_img_path}")
    plt.show()
    
    total_return = 0.0
    totl_trade = 0
    for i in range(0,len(df)-10):
        if df['predicted_label'][i] == 0:
            total_return += -return_df['10min_return'][i]
            totl_trade += 1
        elif df['predicted_label'][i] == 2:
            total_return += return_df['10min_return'][i]
            totl_trade += 1
        else:
            continue
    
    result_msg = f"Total minutes:{len(df)-10}, total trade: {totl_trade}"
    print(result_msg)
    logging.info(result_msg)
    
    return_msg = f"Total pnl:{total_return*10000:.4f}, average pnl:{total_return*10000/totl_trade if totl_trade > 0 else 0:.4f}"
    print(return_msg)
    logging.info(return_msg)
    
    logging.info("Finished\n")
    print("Finished\n")
