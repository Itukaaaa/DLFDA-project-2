import torch.nn as nn
import torch
import torch.nn as nn
from config import CFG

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

def build_model(cfg, in_dim, n_class):
    if cfg.model == 'lstm':
        return LSTMClassifier(in_dim, cfg.hidden, cfg.layers, n_class, cfg.dropout)
    return TransEncoder(in_dim, cfg.hidden, cfg.layers, n_class, cfg.nhead, cfg.dropout)
