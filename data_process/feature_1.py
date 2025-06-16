import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns


def generate_features(df, windows=[5, 10, 20 , 30, 60, 100]):
    """生成基于OHLCV的技术指标因子，避免价格暴露"""
    # 复制一份数据以避免修改原始数据
    data = df.copy()
    
    # 1. 波动性相关因子
    # 真实波动幅度 (True Range)
    data['tr'] = np.maximum(
        np.maximum(
            data['high'] - data['low'],
            np.abs(data['high'] - data['close'].shift(1))
        ),
        np.abs(data['low'] - data['close'].shift(1))
    )
    
    # 各个窗口的平均真实波动幅度 (ATR)
    for window in windows:
        data[f'atr_{window}'] = data['tr'].rolling(window=window).mean()
    
    # 波动率比率 (当前波动率/历史波动率)
    for window in windows:
        data[f'volatility_ratio_{window}'] = data['tr'] / data[f'atr_{window}']
    
    # 2. 价格形态因子
    # 上影线比例 = (high-max(open,close))/tr
    data['upper_shadow_ratio'] = (data['high'] - np.maximum(data['open'], data['close'])) / data['tr']
    
    # 下影线比例 = (min(open,close)-low)/tr
    data['lower_shadow_ratio'] = (np.minimum(data['open'], data['close']) - data['low']) / data['tr']
    
    # 实体比例 = abs(open-close)/tr
    data['body_ratio'] = np.abs(data['open'] - data['close']) / data['tr']
    
    # 价格变化方向 (1=上涨, -1=下跌)
    data['direction'] = np.where(data['close'] > data['open'], 1, -1)
    
    # 3. 相对强弱指标 (RSI)
    for window in windows:
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        data[f'rsi_{window}'] = 100 - (100 / (1 + rs))
    
    # 4. 成交量相关因子
    # 相对成交量 (当前成交量/历史平均成交量)
    for window in windows:
        data[f'rel_volume_{window}'] = data['volume'] / data['volume'].rolling(window=window).mean()
    
    # 成交量变化率
    data['volume_change'] = data['volume'].pct_change()
    
    # 成交量震荡指标 (OBV-On Balance Volume变化率)
    data['obv'] = np.where(
        data['close'] > data['close'].shift(1),
        data['volume'],
        np.where(
            data['close'] < data['close'].shift(1),
            -data['volume'],
            0
        )
    ).cumsum()
    data['obv_change'] = data['obv'].pct_change()
    
    # 5. 趋势因子
    # 价格动量 (当前价格/历史价格 - 1)
    for window in windows:
        data[f'momentum_{window}'] = data['close'] / data['close'].shift(window) - 1
    
    # 移动平均偏离度
    for window in windows:
        data[f'ma_{window}'] = data['close'].rolling(window=window).mean()
        data[f'ma_deviation_{window}'] = (data['close'] - data[f'ma_{window}']) / data[f'ma_{window}']
    
    # 删除包含NaN的行
    data = data.dropna()
    
    return data

def evaluate_features(df, features, label_column='label', method='spearman', top_n=20):
    """
    评估特征与标签的相关性
    
    参数:
    df: 包含特征和标签的DataFrame
    features: 要评估的特征列表
    label_column: 标签列名称
    method: 相关性计算方法 ('pearson'或'spearman')
    top_n: 展示相关性最高的前N个特征
    
    返回:
    相关性排序结果
    """
    correlations = {}
    
    if method == 'pearson':
        for feature in features:
            correlations[feature] = np.abs(df[feature].corr(df[label_column]))
    else:  # 默认使用spearman
        for feature in features:
            correlations[feature] = np.abs(spearmanr(df[feature], df[label_column])[0])
    
    # 按相关性排序
    corr_df = pd.DataFrame({
        'Feature': list(correlations.keys()),
        'Correlation': list(correlations.values())
    }).sort_values('Correlation', ascending=False)
    
    # 可视化前N个特征的相关性
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Correlation', y='Feature', data=corr_df.head(top_n))
    plt.title(f'Top {top_n} Features by {method.capitalize()} Correlation')
    plt.tight_layout()
    plt.savefig(f'result/feature_correlation_{method}.png')
    plt.show()
    
    return corr_df

# 使用示例
if __name__ == "__main__":    
    # 生成特征
    df = pd.read_csv('data/BTCUSDT_feature_derived.csv')
    print("数据读取完成")
    
    feature_df = generate_features(df)
    print("特征生成完成")
    
    feature_df['label'] = df['label']  # 假设标签列名为'label'
    
    # 获取生成的特征列名(排除原始数据和标签)
    original_columns = ['open', 'high', 'low', 'close', 'volume', 'feature1','feature2','feature3','label']
    feature_columns = [col for col in feature_df.columns if col not in original_columns]
    
    # 评估特征
    correlation_results = evaluate_features(feature_df, feature_columns)
    print(correlation_results)
    
    # 获取前20个高相关性因子
    top_features = correlation_results['Feature'].head(20).tolist()
    
    # 构建包含前20个高相关性因子和标签的数据框
    training_data = feature_df[['open','close','high','low','volume','label'] + top_features]
    
    # 保存用于训练的数据
    training_data.to_csv('data/BTCUSDT-feature-label.csv', index=False)
    print("已保存前20个高相关性因子到CSV文件，可用于模型训练")
