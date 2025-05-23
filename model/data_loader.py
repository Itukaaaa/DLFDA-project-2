import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

class FinancialDataset(Dataset):
    def __init__(self, csv_file, seq_length=300, num_classes=9, resample=False, target_samples_per_class=1000000):
        """
        Args:
            csv_file (string): 数据集CSV文件的路径
            seq_length (int): 输入序列的长度（天数）
            num_classes (int): 标签的类别数量
            resample (bool): 是否进行重采样均衡类别
            target_samples_per_class (int): 重采样时每个类别的目标样本数量
        """
        # 读取CSV文件
        self.data_frame = pd.read_csv(csv_file)
        
        # 提取特征和标签
        self.features = self.data_frame[['open', 'high', 'low', 'close', 'volume']].values
        self.labels = self.data_frame['label'].values
        
        # 序列长度
        self.seq_length = seq_length
        
        # 类别数量
        self.num_classes = num_classes
        
        # 计算有效的样本数量
        self.valid_idx = len(self.features) - self.seq_length
        
        # 创建有效样本的索引
        self.indices = list(range(self.valid_idx))
        
        # 如果需要重采样，创建重采样索引
        if resample and self.valid_idx > 0:
            self.indices = self._get_resampled_indices(target_samples_per_class)
        
        self.normalize_features()
            
    def _get_resampled_indices(self, target_samples_per_class):
        """获取重采样后的索引列表，确保每个类别样本数量相同
        
        Args:
            target_samples_per_class (int): 每个类别的目标样本数量
            
        Returns:
            list: 重采样后的索引列表
        """
        # 按类别划分索引
        class_indices = [[] for _ in range(self.num_classes)]
        for idx in range(self.valid_idx):
            label = int(self.labels[idx + self.seq_length - 1])
            class_indices[label].append(idx)
        
        # 重采样
        resampled_indices = []
        for label, indices in enumerate(class_indices):
            n_samples = len(indices)
            
            if n_samples == 0:
                print(f"警告: 类别 {label} 没有样本")
                continue  # 跳过没有样本的类别
            
            # 计算需要重复的次数
            repeats = target_samples_per_class // n_samples
            remainder = target_samples_per_class % n_samples
            
            # 添加完整重复的部分
            for _ in range(repeats):
                resampled_indices.extend(indices)
            
            # 添加余数部分（随机抽样）
            if remainder > 0:
                extra_indices = np.random.choice(indices, remainder, replace=False)
                resampled_indices.extend(extra_indices)
        
        # 随机打乱
        np.random.shuffle(resampled_indices)
        
        return resampled_indices
        
    def normalize_features(self):
        """对特征进行归一化处理"""
        # 对每个特征分别进行归一化
        for i in range(self.features.shape[1]):
            mean = np.mean(self.features[:, i])
            std = np.std(self.features[:, i])
            if std != 0:
                self.features[:, i] = (self.features[:, i] - mean) / std
            
    def __len__(self):
        """返回数据集中样本的数量"""
        return len(self.indices)
    
    def __getitem__(self, idx):
        """获取指定索引的样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            tuple: (x, y) 其中 x 是输入特征序列,y 是one-hot编码的目标标签
        """
        if idx >= len(self.indices):
            raise IndexError("索引超出范围")
            
        # 获取实际的样本索引
        sample_idx = self.indices[idx]
        
        # 提取序列数据
        x = self.features[sample_idx:sample_idx + self.seq_length]
        # 使用序列最后一天对应的标签
        label = int(self.labels[sample_idx + self.seq_length - 1])
        
        x = torch.FloatTensor(x)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

def get_data_loaders(batch_size=128, seq_length=300, shuffle=True, num_workers=4, resample_train=False, target_samples_per_class=350000, verbose=True):
    """创建训练集、验证集和测试集的DataLoader
    
    Args:
        batch_size (int): 批次大小
        seq_length (int): 输入序列长度
        shuffle (bool): 是否打乱训练数据的顺序（True表示打乱，False表示保持原始顺序）
        num_workers (int): 数据加载的并行工作进程数
        resample_train (bool): 是否对训练集进行重采样
        target_samples_per_class (int): 每个类别的目标样本数量
        verbose (bool): 是否打印详细信息
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if verbose:
        print("========== 开始加载数据集 ==========")
    
    # 创建数据集
    print("加载训练集...") if verbose else None
    train_dataset = FinancialDataset(
        os.path.join('data', 'train.csv'), 
        seq_length=seq_length,
        resample=resample_train,
        target_samples_per_class=target_samples_per_class
    )
    
    print("加载验证集...") if verbose else None
    val_dataset = FinancialDataset(os.path.join('data', 'val.csv'), seq_length)
    
    print("加载测试集...") if verbose else None
    test_dataset = FinancialDataset(os.path.join('data', 'test.csv'), seq_length)
    
    if verbose:
        print("\n数据集信息:")
        print(f"训练集样本数: {len(train_dataset)}")
        print(f"验证集样本数: {len(val_dataset)}")
        print(f"测试集样本数: {len(test_dataset)}")
        
        if resample_train:
            print(f"已对训练集进行重采样，目标每类样本数量: {target_samples_per_class}")
        
    # 创建DataLoader
    print("\n创建数据加载器...") if verbose else None
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    if verbose:
        print("\n数据加载器信息:")
        print(f"训练集批次数: {len(train_loader)}")
        print(f"验证集批次数: {len(val_loader)}")
        print(f"测试集批次数: {len(test_loader)}")
        
        # 打印第一个批次的形状
        try:
            x_batch, y_batch = next(iter(train_loader))
            print(f"\n单个批次形状示例:")
            print(f"输入特征 (X): {x_batch.shape} - [批次大小, 序列长度, 特征维度]")
            print(f"目标标签 (y): {y_batch.shape} - [批次大小]")
        except StopIteration:
            print("警告: 训练集为空，无法显示批次形状")
        
        print("========== 数据加载完成 ==========")
    
    return train_loader, val_loader, test_loader

def load_single_dataset(dataset_type, batch_size=32, seq_length=300, shuffle=False, num_workers=4, resample=False, target_samples_per_class=350000, verbose=True):
    """加载单个数据集
    
    Args:
        dataset_type (str): 'train', 'val' 或 'test'
        batch_size (int): 批次大小
        seq_length (int): 输入序列长度
        shuffle (bool): 是否打乱数据
        num_workers (int): 数据加载的并行工作进程数
        resample (bool): 是否进行重采样（仅适用于训练集）
        target_samples_per_class (int): 每个类别的目标样本数量
        verbose (bool): 是否打印详细信息
        
    Returns:
        DataLoader: 指定类型的数据加载器
    """
    if dataset_type not in ['train', 'val', 'test']:
        raise ValueError("dataset_type必须是'train', 'val'或'test'")
    
    if verbose:
        print(f"加载{dataset_type}数据集...")
    
    # 仅对训练集进行重采样
    do_resample = resample and dataset_type == 'train'
    
    dataset = FinancialDataset(
        os.path.join('data', f'{dataset_type}.csv'), 
        seq_length=seq_length,
        resample=do_resample,
        target_samples_per_class=target_samples_per_class
    )
    
    if verbose:
        print(f"{dataset_type}数据集样本数: {len(dataset)}")
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if dataset_type == 'train' else False,
        num_workers=num_workers
    )
    
    if verbose:
        print(f"{dataset_type}数据集批次数: {len(loader)}")
        
        # 打印第一个批次的形状
        try:
            x_batch, y_batch = next(iter(loader))
            print(f"单个批次形状: 特征={x_batch.shape}, 标签={y_batch.shape}")
        except StopIteration:
            print(f"警告: {dataset_type}数据集为空，无法显示批次形状")
    
    return loader

if __name__ == "__main__":
    # 测试数据加载器
    # train_loader, val_loader, test_loader = get_data_loaders(resample_train=True, target_samples_per_class=1000000)
    test_loader = load_single_dataset('test')
    for x,y in test_loader:
        print(y)
        break