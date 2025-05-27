import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def describe(filename="infer_result/example.csv"):
    # Read the CSV file
    df = pd.read_csv(filename)
    
    # 对pred0, pred1, pred2三列进行统计性描述
    pred_columns = ['pred0', 'pred1', 'pred2']
    stats_description = df[pred_columns].describe()
    print("预测列的统计描述:")
    print(stats_description)
    
    # 根据三列中的最大值创建预测标签
    df['predicted_label'] = df[pred_columns].idxmax(axis=1)       
        
    # 将'predX'转换为数字X
    df['predicted_label'] = df['predicted_label'].str.replace('pred', '').astype(int)
    
    # 应用阈值规则 - 使用向量化操作
    mask_0 = (df['predicted_label'] == 0) & (df['pred0'] > 0.1)
    mask_2 = (df['predicted_label'] == 2) & (df['pred2'] > 0.1)
    mask_default = ~(mask_0 | mask_2)

    # 应用条件
    df.loc[mask_default, 'predicted_label'] = 1
    
    print(f"已创建预测标签")
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
    # 调用测试函数
    df = describe()
    return_df = calculate_10min_returns(df['close'])
    
    true_labels = df['label'].values
    predicted_labels = df['predicted_label'].values
    
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # 打印分类报告
    print(classification_report(true_labels, predicted_labels))
    
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['0', '1', '2'], 
                yticklabels=['0', '1', '2'])
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.title('confusion matrix')
    plt.tight_layout()
    plt.savefig('backtest/confusion_matrix.png')
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
    print(f"总的时间节点共：{len(df)-10}, 总交易次数: {totl_trade}")
    print(f"总收益率: 万分之{total_return*10000}, 平均收益率: 万分之{total_return*10000/totl_trade if totl_trade > 0 else 0}")

    print("测试完成。")
