import argparse
from config import CFG
from utils import set_seed, init_logger
from data import chronological_split, make_loaders
from model import build_model
from train import train_model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--labelcol', default=CFG.label_col)
    p.add_argument('--model', choices=['lstm', 'transformer'], default=CFG.model)
    p.add_argument('--seq', type=int, default=CFG.seq_len)
    p.add_argument('--batch', type=int, default=CFG.batch)
    p.add_argument('--hidden', type=int, default=CFG.hidden)
    p.add_argument('--layers', type=int, default=CFG.layers)
    p.add_argument('--epochs', type=int, default=CFG.epochs)
    p.add_argument('--lr', type=float, default=CFG.lr)
    p.add_argument("--extra", nargs='+', default=['ma_deviation_60','ma_deviation_30','rsi_20','rsi_30','ma_deviation_100','momentum_20','ma_deviation_20','rsi_60','momentum_30','momentum_60','rsi_100','rsi_10','ma_deviation_10','momentum_10','momentum_100','rsi_5','momentum_5','ma_deviation_5','obv_change','direction'], help="Extra features")
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    cfg = CFG(csv=args.csv, label_col=args.labelcol, model=args.model,
              seq_len=args.seq, batch=args.batch, hidden=args.hidden,
              layers=args.layers, epochs=args.epochs, lr=args.lr, extra_feats=args.extra)

    set_seed(cfg.seed)
    log = init_logger()
    paths = chronological_split(cfg, log)
    loaders = make_loaders(cfg, paths, log)
    model = build_model(cfg, in_dim=next(iter(loaders[0]))[0].shape[-1], n_class=loaders[3]).to(cfg.device)
    train_model(cfg, model, loaders, log)

