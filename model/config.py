from dataclasses import dataclass
import torch
from typing import List, Union
from dataclasses import dataclass, field

@dataclass
class CFG:
    csv: str
    label_col: str = "label"
    seed: int = 1337
    test_pct: float = 0.2
    val_pct: float = 0.1
    seq_len: int = 120
    batch: int = 256
    lr: float = 1e-5
    epochs: int = 50
    patience: int = 2
    model: str = "transformer"
    hidden: int = 512
    layers: int = 3
    dropout: float = 0.4
    nhead: int = 4
    resample: bool = True
    extra_feats: List[str] = field(default_factory=['ma_deviation_60','ma_deviation_30','rsi_20','rsi_30','ma_deviation_100','momentum_20','ma_deviation_20','rsi_60','momentum_30','momentum_60','rsi_100','rsi_10','ma_deviation_10','momentum_10','momentum_100','rsi_5','momentum_5','ma_deviation_5','obv_change','direction'])
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
