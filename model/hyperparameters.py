# 训练参数
INPUT_DIM = 5  # 特征数量: open, high, low, close, volume
HIDDEN_DIM = 256 # LSTM隐藏层维度
NUM_LAYERS = 2  # LSTM层数
OUTPUT_DIM = 3  # 输出维度修改为3（对应3个类别）
DROPOUT = 0.4  # Dropout比率
BATCH_SIZE = 128  # 批次大小
NUM_EPOCHS = 30  # 训练轮数
SEQ_LENGTH = 30  # 序列长度
PATIENCE = 5  # 早停耐心值：连续几轮验证准确率未提升就停止训练
LEANRING_RATE = 0.0001  # 学习率