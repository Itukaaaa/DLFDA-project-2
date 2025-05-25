import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List
import os

class FinDataset(Dataset):
    def __init__(self,csv_file:str,seq_len:int,feat_cols:List[str],label_col:str,log = -1):
        self.df=pd.read_csv(csv_file)
        assert label_col in self.df.columns, f"{csv_file} 缺少列 {label_col}"
        self.labels=self.df[label_col].values.astype(int)
        if self.labels.min()==1: self.labels-=1  # 1-based -> 0-based
        self.features=self.df[feat_cols].values.astype(np.float32)
        self.features=(self.features-self.features.mean(0))/(self.features.std(0)+1e-9)
        self.seq_len=seq_len
        self.valid=len(self.df)-seq_len
        if log != -1:
            log(f"{csv_file.name}: samples={self.valid}  class_dist={np.bincount(self.labels)[:4]}")
    def __len__(self):return self.valid
    def __getitem__(self,idx):
        x=self.features[idx:idx+self.seq_len]
        y=self.labels[idx+self.seq_len-1]
        return torch.from_numpy(x), torch.tensor(y)

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