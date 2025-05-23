# crypto_ts_classification.py  (v2 with training log)
"""
End-to-end cryptocurrency time-series classification WITH persistent training log.
Run e.g.:
    python crypto_ts_classification.py --csv /mnt/d/path/BTCUSDT.csv
It now:
• Splits data chronologically
• Builds DataLoaders
• Trains LSTM/Transformer
• Writes best checkpoint & confusion-matrix PNG
• **Saves detailed log** to logs/train_YYYYMMDD_HHMMSS.log
"""
from __future__ import annotations
import os, argparse, math, random, time, json, logging, datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
# ────────────────────────────────────────────────────────────────────────────
# 0.  utils: reproducibility + logging helper
# ────────────────────────────────────────────────────────────────────────────

def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True

def init_logger():
    """Return log() function that prints & persists to timestamped file."""
    log_dir = Path("logs"); log_dir.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir/f"train_{ts}.log"
    logging.basicConfig(filename=log_file,
                        level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%H:%M:%S')
    def _log(msg:str):
        print(msg);
        logging.info(msg)
    _log(f"Log file: {log_file}")
    return _log
# ────────────────────────────────────────────────────────────────────────────
# 1.  Config dataclass
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class CFG:
    csv:str
    label_col:str="label"
    seed:int=1337
    test_pct:float=0.2
    val_pct:float=0.1
    seq_len:int=120
    batch:int=256
    lr:float=1e-5
    epochs:int=50
    patience:int=7
    model:str="transformer"
    hidden:int=256
    layers:int=2
    dropout:float=0.4
    nhead:int=4
    extra_feats:List[str]|None=None
    device:str="cuda" if torch.cuda.is_available() else "cpu"
# ────────────────────────────────────────────────────────────────────────────
# 2.  Data split
# ────────────────────────────────────────────────────────────────────────────

def chronological_split(cfg:CFG, log):
    df=pd.read_csv(cfg.csv)
    df['trade_time']=pd.to_datetime(df['trade_time'])
    df=df.sort_values('trade_time').reset_index(drop=True)
    tot=len(df)
    tr_end=int(tot*(1-cfg.test_pct-cfg.val_pct))
    val_end=tr_end+int(tot*cfg.val_pct)
    out=Path('splits'); out.mkdir(exist_ok=True)
    paths={
        'train':out/'train.csv',
        'val':out/'val.csv',
        'test':out/'test.csv'
    }
    df.iloc[:tr_end].to_csv(paths['train'],index=False)
    df.iloc[tr_end:val_end].to_csv(paths['val'],index=False)
    df.iloc[val_end:].to_csv(paths['test'],index=False)
    log(f"✔ Data split: {tr_end}/{val_end-tr_end}/{tot-val_end} rows ➜ {out}")
    return paths
# ────────────────────────────────────────────────────────────────────────────
# 3.  Dataset & DataLoader
# ────────────────────────────────────────────────────────────────────────────
class FinDataset(Dataset):
    def __init__(self,csv_file:str|Path,seq_len:int,feat_cols:List[str],label_col:str,log):
        self.df=pd.read_csv(csv_file)
        assert label_col in self.df.columns, f"{csv_file} 缺少列 {label_col}"
        self.labels=self.df[label_col].values.astype(int)
        if self.labels.min()==1: self.labels-=1  # 1-based -> 0-based
        self.features=self.df[feat_cols].values.astype(np.float32)
        self.features=(self.features-self.features.mean(0))/(self.features.std(0)+1e-9)
        self.seq_len=seq_len
        self.valid=len(self.df)-seq_len
        log(f"{csv_file.name}: samples={self.valid}  class_dist={np.bincount(self.labels)[:4]}")
    def __len__(self):return self.valid
    def __getitem__(self,idx):
        x=self.features[idx:idx+self.seq_len]
        y=self.labels[idx+self.seq_len-1]
        return torch.from_numpy(x), torch.tensor(y)

def make_loaders(cfg:CFG,paths:dict,log):
    base=['open','high','low','close','volume']
    feats=base+(cfg.extra_feats or [])
    tr=FinDataset(paths['train'],cfg.seq_len,feats,cfg.label_col,log)
    val=FinDataset(paths['val'],cfg.seq_len,feats,cfg.label_col,log)
    te=FinDataset(paths['test'],cfg.seq_len,feats,cfg.label_col,log)
    tr_loader=DataLoader(tr,cfg.batch,True,num_workers=4,pin_memory=True)
    val_loader=DataLoader(val,cfg.batch,False,num_workers=4,pin_memory=True)
    test_loader=DataLoader(te,cfg.batch,False,num_workers=4,pin_memory=True)
    return tr_loader,val_loader,test_loader,len(np.unique(tr.labels))
# ────────────────────────────────────────────────────────────────────────────
# 4.  Models (unchanged)
# ────────────────────────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self,in_dim,hid,layers,n_cls,drop):
        super().__init__()
        self.lstm=nn.LSTM(in_dim,hid,layers,batch_first=True,dropout=drop if layers>1 else 0)
        self.head=nn.Sequential(nn.LayerNorm(hid),nn.Linear(hid,hid*2),nn.GELU(),nn.Dropout(drop),nn.Linear(hid*2,n_cls))
    def forward(self,x):_,(h,_) = self.lstm(x); return self.head(h[-1])
class TransEncoder(nn.Module):
    def __init__(self,in_dim,hid,layers,n_cls,nhead,drop,max_len=1024):
        super().__init__(); self.proj=nn.Linear(in_dim,hid); self.pos=nn.Parameter(torch.randn(1,max_len,hid)*0.02)
        enc=nn.TransformerEncoderLayer(hid,nhead,hid*4,drop,batch_first=True,activation='gelu')
        self.enc=nn.TransformerEncoder(enc,layers); self.norm=nn.LayerNorm(hid); self.cls=nn.Sequential(nn.Linear(hid,hid*2),nn.GELU(),nn.Dropout(drop),nn.Linear(hid*2,n_cls))
    def forward(self,x):B,L,_=x.shape; x=self.proj(x)+self.pos[:,:L]; x=self.enc(x); x=self.norm(x[:,-1]); return self.cls(x)

def build_model(cfg:CFG,in_dim:int,n_class:int):
    return (LSTMClassifier if cfg.model=='lstm' else TransEncoder)(in_dim,cfg.hidden,cfg.layers,n_class,cfg.dropout) if cfg.model=='lstm' else TransEncoder(in_dim,cfg.hidden,cfg.layers,n_class,cfg.nhead,cfg.dropout)
# ────────────────────────────────────────────────────────────────────────────
# 5.  Train / Evaluate helpers
# ────────────────────────────────────────────────────────────────────────────

def acc(logits,y):return (logits.argmax(1)==y).float().mean().item()

def run_epoch(model, loader, criterion, optimizer, device, log, train=True):
    """
    单个 epoch 的训练或评估。
    ──────────────────────────────
    * train=True  →  反向传播 + 更新参数
    * 每 200 个 batch（或最后一个 batch）打印一次累积 loss / acc
    """
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_acc  = 0.0

    for i, (x, y) in enumerate(loader, start=1):      # i 从 1 开始便于取模
        x, y = x.to(device), y.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            outputs = model(x)
            loss = criterion(outputs, y)

            if train:
                loss.backward()
                optimizer.step()

        # 累计统计
        running_loss += loss.item()
        running_acc  += (outputs.argmax(1) == y).float().mean().item()

        # 每 200 个 batch 记录一次，或在最后一个 batch 记录一次
        if i % 200 == 0 or i == len(loader):
            log(f"[{'train' if train else 'eval '}]"           # 标签
                f" batch {i}/{len(loader)} | "
                f"loss {running_loss / i:.4f} | "
                f"acc {running_acc / i:.4f}")

    # 返回 epoch 平均指标
    epoch_loss = running_loss / len(loader)
    epoch_acc  = running_acc  / len(loader)
    return epoch_loss, epoch_acc


# ────────────────────────────────────────────────────────────────────────────
# 6.  Main
# ────────────────────────────────────────────────────────────────────────────

def main(cfg:CFG):
    set_seed(cfg.seed)
    log=init_logger()
    paths=chronological_split(cfg,log)
    tr_loader,val_loader,te_loader,n_cls=make_loaders(cfg,paths,log)
    in_dim=next(iter(tr_loader))[0].shape[-1]
    model=build_model(cfg,in_dim,n_cls).to(cfg.device)
    labels_for_weights = tr_loader.dataset.labels[cfg.seq_len-1:]
    cls_w = compute_class_weight('balanced',classes=np.arange(n_cls),y=labels_for_weights)
    crit=nn.CrossEntropyLoss(weight=torch.tensor(cls_w,dtype=torch.float32).to(cfg.device))
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=3,factor=0.5)
    best_acc=no_improve=0
    ckpt=Path('checkpoints'); ckpt.mkdir(exist_ok=True)
    for epoch in range(1,cfg.epochs+1):
        tr_loss,tr_acc=run_epoch(model,tr_loader,crit,opt,cfg.device,log,True)
        val_loss,val_acc=run_epoch(model,val_loader,crit,opt,cfg.device,log,False)
        sched.step(val_loss)
        log(f"Epoch {epoch:02d} | train {tr_loss:.4f}/{tr_acc:.4f} | val {val_loss:.4f}/{val_acc:.4f} | lr {opt.param_groups[0]['lr']:.2e}")
        if val_acc>best_acc:
            best_acc=val_acc; no_improve=0
            torch.save(model.state_dict(),ckpt/'best.pt'); log("✓ save best")
        else:
            no_improve+=1
            if no_improve>=cfg.patience: log("Early-stop"); break
    # test
    model.load_state_dict(torch.load(ckpt/'best.pt'))
    te_loss,te_acc=run_epoch(model,te_loader,crit,opt,cfg.device,log,False)
    log(f"TEST loss/acc {te_loss:.4f}/{te_acc:.4f}")
    # confusion matrix
    all_p,all_y=[],[]
    model.eval()
    with torch.no_grad():
        for x,y in te_loader:
            x=x.to(cfg.device); out=model(x); all_p+=out.argmax(1).cpu().tolist(); all_y+=y.tolist()
    cm=confusion_matrix(all_y,all_p)
    plt.imshow(cm,cmap='Blues'); plt.colorbar(); plt.title('Confusion')
    plt.xlabel('pred'); plt.ylabel('true')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j,i,int(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    plt.tight_layout(); plt.savefig('confusion_matrix.png'); plt.close()
    log('Confusion matrix saved ➜ confusion_matrix.png')
# ────────────────────────────────────────────────────────────────────────────
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--csv',required=True)
    p.add_argument('--labelcol',default='label')
    p.add_argument('--model',choices=['lstm','transformer'],default='transformer')
    p.add_argument('--seq',type=int,default=120)
    p.add_argument('--batch',type=int,default=256)
    p.add_argument('--hidden',type=int,default=128)
    p.add_argument('--layers',type=int,default=2)
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--extra',nargs='*',default=None)
    args=p.parse_args()
    cfg=CFG(csv=args.csv,label_col=args.labelcol,model=args.model,seq_len=args.seq,batch=args.batch,hidden=args.hidden,layers=args.layers,epochs=args.epochs,lr=args.lr,extra_feats=args.extra)
    main(cfg)
