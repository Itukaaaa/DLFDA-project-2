import pandas as pd, numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch

class FinDataset(Dataset):
    def __init__(self, csv_file, seq_len, feat_cols, label_col, log):
        self.df = pd.read_csv(csv_file)
        self.labels = self.df[label_col].values.astype(int)
        if self.labels.min() == 1: self.labels -= 1
        self.features = self.df[feat_cols].values.astype(np.float32)
        self.features = (self.features - self.features.mean(0)) / (self.features.std(0) + 1e-9)
        self.seq_len = seq_len
        self.valid = len(self.df) - seq_len
        log(f"{Path(csv_file).name}: samples={self.valid}  class_dist={np.bincount(self.labels)[:4]}")
    def __len__(self): return self.valid
    def __getitem__(self, idx):
        x = self.features[idx:idx+self.seq_len]
        y = self.labels[idx+self.seq_len-1]
        return torch.from_numpy(x), torch.tensor(y)

def chronological_split(cfg, log):
    df = pd.read_csv(cfg.csv)
    for col in ['high', 'low', 'close']:
        df[col] = (df[col] - df['open']) / df['open']
    df.drop(columns=['open'], inplace=True)

    total_len = len(df)
    subset_end = int(total_len * 0.3)
    df = df.iloc[:subset_end].reset_index(drop=True)
    log(f"✔ Using only first 30% of data: {subset_end} / {total_len} rows")

    tr_end = int(len(df) * (1 - cfg.test_pct - cfg.val_pct))
    val_end = tr_end + int(len(df) * cfg.val_pct)

    out = Path('splits'); out.mkdir(exist_ok=True)
    df.iloc[:tr_end].to_csv(out/'train.csv', index=False)
    df.iloc[tr_end:val_end].to_csv(out/'val.csv', index=False)
    df.iloc[val_end:].to_csv(out/'test.csv', index=False)
    log(f"✔ Split → {tr_end}/{val_end-tr_end}/{len(df)-val_end}")
    return {'train': out/'train.csv', 'val': out/'val.csv', 'test': out/'test.csv'}

def make_loaders(cfg, paths, log):
    base = ['high','low','close','volume']
    feats = base + (cfg.extra_feats or [])
    tr = FinDataset(paths['train'], cfg.seq_len, feats, cfg.label_col, log)
    val = FinDataset(paths['val'], cfg.seq_len, feats, cfg.label_col, log)
    te = FinDataset(paths['test'], cfg.seq_len, feats, cfg.label_col, log)
    return (
        DataLoader(tr, cfg.batch, True, num_workers=4, pin_memory=True),
        DataLoader(val, cfg.batch, False, num_workers=4, pin_memory=True),
        DataLoader(te, cfg.batch, False, num_workers=4, pin_memory=True),
        len(np.unique(tr.labels))
    )
