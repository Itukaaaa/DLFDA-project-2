from dataclasses import dataclass
import torch
from typing import List, Union

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
    patience: int = 3
    model: str = "transformer"
    hidden: int = 256
    layers: int = 2
    dropout: float = 0.4
    nhead: int = 4
    resample: bool = True
    extra_feats: Union[List[str], None] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
