import torch
import torch.nn as nn
import os
import hyperparameters as hp

class LSTMClassifier(nn.Module):
    def __init__(self,in_dim,hid,layers,n_cls,drop):
        super().__init__()
        self.lstm = nn.LSTM(in_dim,hid,layers,
                            batch_first=True,dropout=drop if layers>1 else 0)
        self.head = nn.Sequential(nn.LayerNorm(hid),nn.Linear(hid,hid*2),
                                  nn.GELU(),nn.Dropout(drop),nn.Linear(hid*2,n_cls))
    def forward(self,x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])

class TransEncoder(nn.Module):
    def __init__(self,in_dim,hid,layers,n_cls,nhead,drop,max_len=1024):
        super().__init__()
        self.proj = nn.Linear(in_dim,hid)
        self.pos = nn.Parameter(torch.randn(1,max_len,hid)*0.02)
        enc = nn.TransformerEncoderLayer(
            hid,nhead,hid*4,drop,batch_first=True,activation='gelu')
        self.enc=nn.TransformerEncoder(enc,layers)
        self.norm=nn.LayerNorm(hid)
        self.cls=nn.Sequential(nn.Linear(hid,hid*2),nn.GELU(),
                               nn.Dropout(drop),nn.Linear(hid*2,n_cls))
    def forward(self,x):
        B,L,_=x.shape
        x=self.proj(x)+self.pos[:,:L]
        x=self.enc(x)
        x=self.norm(x[:,-1])
        return self.cls(x)