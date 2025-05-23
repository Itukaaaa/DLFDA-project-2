import torch
import torch.nn as nn
import os
import hyperparameters as hp

class LSTMModel(nn.Module):
    def __init__(self, input_dim=hp.INPUT_DIM, hidden_dim=hp.HIDDEN_DIM, num_layers=hp.NUM_LAYERS, output_dim=hp.OUTPUT_DIM, dropout=hp.DROPOUT):
        """
        初始化LSTM模型
        
        参数:
            input_dim (int): 输入特征的维度，默认为5
            hidden_dim (int): LSTM隐藏层的维度，默认为128
            num_layers (int): LSTM的层数，默认为2
            output_dim (int): 输出维度，默认为9（对应9个类别的one-hot编码）
            dropout (float): dropout比率，默认为0.2
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 定义LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # 输入形状为 (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 定义全连接层
        self.fc1 = nn.Linear(hidden_dim, output_dim*4)
        self.fc2 = nn.Linear(output_dim*4, output_dim)
        
        # 应用自定义权重初始化
        self._initialize_weights()
        
    def _initialize_weights(self, scale=0.2):
        """
        自定义权重初始化，给参数赋予比较大的随机初始化值
        
        参数:
            scale (float): 放大随机初始化值的比例因子，默认为0.1
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                # 对于权重参数，使用较大的均匀分布初始化
                nn.init.uniform_(param, -scale, scale)
                # print(param)
            elif 'bias' in name:
                # 对于偏置参数，初始化为小的常数
                nn.init.constant_(param, 0.05)
                # print(param)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入数据，形状为 [batch_size, seq_len, input_dim]
            
        返回:
            torch.Tensor: 预测结果，形状为 [batch_size, output_dim]，
                          代表每个类别的得分（未归一化的概率）
        """
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # 通过LSTM层
        # out形状: (batch_size, seq_len, hidden_dim)
        out, _ = self.lstm(x, (h0, c0))
        
        # 只取序列的最后一个时间步的输出
        # out[:, -1, :] 形状: (batch_size, hidden_dim)
        out = self.fc1(out[:, -1, :])
        out = torch.relu(out)
        
        # 通过第二个全连接层
        out = self.fc2(out)
        
        # 注意：这里不添加softmax，因为PyTorch的交叉熵损失函数会自动应用softmax
        # 如果需要概率分布，可以在评估时手动应用softmax：
        # probabilities = torch.softmax(out, dim=1)
        
        return out
    
    def save_model(self, path):
        """保存模型"""
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.state_dict(), path)
        print(f"模型已保存至: {path}")
    
    def load_model(self, path):
        """加载模型"""
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
            print(f"模型已从 {path} 加载")
            return True
        else:
            print(f"找不到模型: {path}")
            return False


class TransformerModel(nn.Module):
    """
    Transformer 版本（接口与 LSTMModel 保持一致）
      ‣ 输入:  [batch, seq_len, input_dim]
      ‣ 输出:  [batch, output_dim]（未 softmax）
    """
    def __init__(self,
                 input_dim=hp.INPUT_DIM,
                 hidden_dim=hp.HIDDEN_DIM,
                 num_layers=hp.NUM_LAYERS,
                 output_dim=hp.OUTPUT_DIM,
                 dropout=hp.DROPOUT,
                 nhead=4,
                 max_len=512):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_len = max_len

        # 1) 逐时间步特征映射到 hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 2) LayerNorm（方案 B）—— 在送入 Transformer 之前做归一化
        self.pre_ln = nn.LayerNorm(hidden_dim)

        # 3) 可学习位置编码，std 较小（0.02）
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.01)

        # 4) Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu"          # gelu 更平滑
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 5) 分类头
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, output_dim * 4)
        self.fc2 = nn.Linear(output_dim * 4, output_dim)

        # （可选）仅对 fc 层做轻量初始化
        self._init_fc()

    # --------------------------------------------------
    def _init_fc(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)        
        for n, p in self.named_parameters():
            if 'fc' in n and 'weight' in n:
                nn.init.xavier_uniform_(p)
            elif 'fc' in n and 'bias' in n:
                nn.init.zeros_(p)


    # --------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, input_dim]
        """
        B, L, _ = x.size()

        # 安全检查
        if L > self.max_len:
            raise ValueError(f"seq_len={L} 超过 max_len={self.max_len}，请调高 max_len")

        # 1) 投影
        x = self.input_proj(x)              # [B, L, hidden]

        # 2) LayerNorm
        x = self.pre_ln(x)

        # 3) 加位置编码
        x = x + self.pos_embedding[:, :L, :]

        # 4) Transformer
        x = self.transformer(x)             # 仍 [B, L, hidden]

        # 5) 取最后一个时间步
        last_token = x[:, -1, :]            # [B, hidden]

        # 6) 分类头
        out = self.dropout(last_token)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)                 # [B, output_dim]
        return out

    # --------------------------------------------------
    # 保存 / 加载接口与 LSTMModel 一致
    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"模型已保存至: {path}")

    def load_model(self, path):
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
            print(f"模型已从 {path} 加载")
            return True
        else:
            print(f"找不到模型: {path}")
            return False

