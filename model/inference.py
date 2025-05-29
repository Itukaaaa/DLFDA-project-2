"""
run_inference.py
────────────────
用法示例:
    python run_inference.py \
        --csv splits/test.csv \
        --ckpt checkpoints/best.pt \
        --model transformer \
        --seq 300 \
        --outfile preds_test.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from typing import List
from torch.utils.data import Dataset, DataLoader

from model import LSTMClassifier, TransEncoder
from data import FinDataset

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
    hidden: int = 512
    layers: int = 3
    dropout: float = 0.4
    nhead: int = 4
    resample: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",      required=True,  help="待推断的 csv 文件")
    p.add_argument("--ckpt",     required=True,  help="训练好的 best.pt")
    p.add_argument("--model",    choices=["transformer", "lstm"], default="transformer")
    p.add_argument("--seq",      type=int, default=CFG.seq_len, help="滑动窗口长度")
    p.add_argument("--batch",    type=int, default=CFG.batch)
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--outfile",  default="infer_result/example.csv", help="保存预测结果的文件名")
    return p.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
def build_model(model_type: str, input_dim: int, num_classes: int, ckpt: str, device: str):
    if model_type == "transformer":
        model = TransEncoder(input_dim,CFG.hidden,CFG.layers,num_classes,CFG.nhead,CFG.dropout)
    else:
        model = LSTMClassifier(input_dim,CFG.hidden,CFG.layers,num_classes,CFG.dropout)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model

# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def infer(model, loader, device):
    preds = []
    for x, _ in loader:                       # x:[B, L, F]
        x = x.to(device)
        out = model(x)                    # logits
        pred = out
        preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)

# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    csv_path  = Path(args.csv)
    ckpt_path = Path(args.ckpt)

    # 1) 选定与训练完全一致的特征列
    base=['high','low','close','volume']
    extra_f=['ma_deviation_60','ma_deviation_30','rsi_20','rsi_30','ma_deviation_100','momentum_20','ma_deviation_20','rsi_60','momentum_30','momentum_60','rsi_100','rsi_10','ma_deviation_10','momentum_10','momentum_100','rsi_5','momentum_5','ma_deviation_5','obv_change','direction']
    feats=base+extra_f
    n_feat = len(feats)
    num_classes = 3   # 若训练时自动推断，请相应调整

    # 2) 构建数据集
    ds = FinDataset(csv_path, args.seq, feats, CFG.label_col)
    # with open("assist.txt", "w") as file:
    #     for i in range(5):
    #         x, y = ds[i]
    #         file.write(f"Features: {x.tolist()}, Label: {y.item()}\n")
    # input("press enter to continue")
    dl = DataLoader(ds, batch_size=CFG.batch, shuffle=False, num_workers=4, pin_memory=True)

    # 3) 构建并载入模型
    model = build_model(args.model, n_feat, num_classes, ckpt_path, args.device)

    # 4) 推断
    predictions = infer(model, dl, args.device)

    # 5) 将预测写回 csv（与最后一个时间步对齐）
    out_df = pd.read_csv(csv_path)
    out_df = out_df.iloc[args.seq:]        # 对齐滑动窗口最后一步
    out_df["pred0"] = predictions[:,0]
    out_df["pred1"] = predictions[:,1]
    out_df["pred2"] = predictions[:,2]
    for feat in extra_f:
        out_df.drop(columns=[feat], inplace=True)
    out_df.drop(columns=['volume'], inplace=True)
    out_df.to_csv(args.outfile, index=False)
    print(f"✅  推断完成，已保存到 {args.outfile}\n"
          f"    rows={len(out_df)}, seq_len={args.seq}")

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
