import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
import os, argparse, math, random, time, json, logging, datetime
from pathlib import Path
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def acc(logits,y):return (logits.argmax(1)==y).float().mean().item()

def validate(model, loader, criterion, device, log,
             save_dir="confusion_val", epoch=None):
    """
    • 评估模式 (no grad)
    • 计算 val_loss / val_acc
    • 生成并保存混淆矩阵图片
      └ 文件名:  confusion_val/epoch_XX.png  (若 epoch 为 None → val_cm.png)
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    tot_loss = tot_acc = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            tot_loss += loss.item()
            tot_acc  += (out.argmax(1) == y).float().mean().item()

            all_preds.extend(out.argmax(1).cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    # —— 保存混淆矩阵 ————————————————
    cm = confusion_matrix(all_targets, all_preds)
    fig_name = f"epoch_{epoch:02d}.png" if epoch is not None else "val_cm.png"
    fig_path = os.path.join(save_dir, fig_name)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Validation Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    log(f"✔ saved validation confusion matrix → {fig_path}")

    avg_loss = tot_loss / len(loader)
    avg_acc  = tot_acc  / len(loader)

    return avg_loss, avg_acc

    # return avg_loss, avg_acc

def run_epoch(model, loader, criterion, optimizer, device, log, train=True):
    """
    单个 epoch 的训练或评估。
    ──────────────────────────────
    * train=True  →  反向传播 + 更新参数
    * 每 200 个 batch(或最后一个 batch)打印一次累积 loss / acc
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

def train_model(cfg, model, loaders, log):
    train_loader, val_loader, test_loader, n_cls = loaders

    # 类别权重
    labels_for_weights = train_loader.dataset.labels[cfg.seq_len - 1:]
    cls_w = compute_class_weight('balanced', classes=torch.arange(n_cls).numpy(), y=labels_for_weights)
    cls_w = torch.tensor(cls_w, dtype=torch.float32).to(cfg.device)
    crit = nn.CrossEntropyLoss(weight=cls_w)

    # 优化器与学习率调度器
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    ckpt = Path("checkpoints") / timestamp
    ckpt.mkdir(parents=True, exist_ok=True)
    cm_dir = Path("confusion_val") / timestamp
    cm_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, crit, opt, cfg.device, log, train=True)
        val_loss, val_acc = validate(model, val_loader, crit, cfg.device, log,
                                                    save_dir=str(cm_dir), epoch=epoch)
        sched.step(val_loss)

        log(f"Epoch {epoch:02d} | train {tr_loss:.4f}/{tr_acc:.4f} | "
            f"val {val_loss:.4f}/{val_acc:.4f} | lr {opt.param_groups[0]['lr']:.2e}")

        if val_loss - best_loss < 5e-5:
            best_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), ckpt / 'best.pt')
            log("✓ Best model saved")
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                log("Early stopping")
                break

    # 加载最佳模型
    model.load_state_dict(torch.load(ckpt / 'best.pt'))
    te_loss, te_acc = run_epoch(model, test_loader, crit, opt, cfg.device, log, train=False)
    log(f"[TEST] loss/acc = {te_loss:.4f} / {te_acc:.4f}")

    # 画混淆矩阵
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            out = model(x)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap='Blues')
    plt.title('Test Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(str(cm_dir) + '/confusion_matrix.png')
    plt.close()
    log("✓ Test confusion matrix saved to confusion_matrix.png")