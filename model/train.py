import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from multiprocessing import freeze_support
from data_loader import get_data_loaders , load_single_dataset
from model import LSTMModel,TransformerModel  # 可替换为 TransformerModel
import hyperparameters as hp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight


def log_message(message, file):
    with open(file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
    print(message)

def calculate_accuracy(outputs, targets):
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)

def train_one_epoch(model, data_loader, criterion, optimizer, device, log_file=None):
    model.train()
    total_loss, total_acc = 0, 0

    for batch_idx, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        acc = calculate_accuracy(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc

        if (batch_idx + 1) % 200 == 0:
            msg = f"[训练] Batch {batch_idx+1}/{len(data_loader)} | Loss: {loss.item():.4f} | Acc: {acc:.4f}"
            log_message(msg, log_file)


    return total_loss / len(data_loader), total_acc / len(data_loader)


def validate(model, data_loader, criterion, device, log_file, show_confusion_matrix=True):
    model.eval()
    total_loss, total_acc = 0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            acc = calculate_accuracy(output, y)
            total_loss += loss.item()
            total_acc += acc
            all_preds.extend(torch.argmax(output, dim=1).cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    if show_confusion_matrix:
        cm = confusion_matrix(all_targets, all_preds)
        log_message(f"混淆矩阵:\n{cm}", log_file)
        # 创建图形
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('confusion matrix')
        plt.colorbar()
        # 设置坐标轴
        classes = np.unique(all_targets)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        # 添加数值标签
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('true label')
        plt.xlabel('predicted label')
        plt.tight_layout()
        # 保存图片
        cm_path = os.path.join('log', f'confusion_matrix_{time.strftime("%Y%m%d-%H%M%S")}.png')
        plt.savefig(cm_path)
        log_message(f"混淆矩阵已保存到: {cm_path}", log_file)
        plt.close()
    return total_loss / len(data_loader), total_acc / len(data_loader)

def back_test(model,data_loader,device,log_file,seq_length):
    model.eval()
    test_preds , test_targets = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            preds = torch.argmax(output, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(y.cpu().numpy())
    test_df = pd.read_csv('data/test.csv')
    test_df = test_df.iloc[seq_length-1:]
    if len(test_df) != len(test_preds):
        log_message("测试数据长度不匹配", log_file)
        return
    test_df['predictions'] = test_preds
    total_gain = 0
    total_trades = 0
    for i in range(len(test_df) - 10):
        if test_df['predictions'][i] == 0:
            gain = (test_df['close'][i + 1] - test_df['close'][i + 10]) / test_df['close'][i + 1]
            total_gain += gain*10000
            total_trades += 1
        elif test_df['predictions'][i] == 1:
            continue
        elif test_df['predictions'][i] == 2:
            gain = (test_df['close'][i + 10] - test_df['close'][i + 1]) / test_df['close'][i + 1]
            total_gain += gain*10000
            total_trades += 1
    if total_trades > 0:
        average_gain = total_gain / total_trades
    else:
        average_gain = 0
    log_message(f"总收益: {total_gain:.2f} | 平均收益: {average_gain:.2f}", log_file)
    log_message(f"总交易次数: {total_trades}", log_file)
    

def main():
    freeze_support()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = f'log/train_log_{timestamp}.txt'
    os.makedirs('log/models', exist_ok=True)

    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=hp.BATCH_SIZE,
        seq_length=hp.SEQ_LENGTH,
        shuffle=True,
        resample_train=False,
        target_samples_per_class=1000000
    )

    test_loader_not_shuffle = load_single_dataset('test')
    
    model = TransformerModel(
        input_dim=hp.INPUT_DIM,
        hidden_dim=hp.HIDDEN_DIM,
        num_layers=hp.NUM_LAYERS,
        output_dim=hp.OUTPUT_DIM,
        dropout=hp.DROPOUT
    ).to(device)

    # label_list = [int(train_loader.dataset.labels[i + hp.SEQ_LENGTH - 1]) for i in train_loader.dataset.indices]
    # class_weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=label_list)
    # weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hp.LEANRING_RATE)

    best_acc, best_epoch = 0, 0
    no_improvement = 0

    for epoch in range(hp.NUM_EPOCHS):
        log_message(f"Epoch {epoch+1}/{hp.NUM_EPOCHS}", log_file)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device,log_file)
        val_loss, val_acc = validate(model, val_loader, criterion, device, log_file)
        # back_test(model,test_loader_not_shuffle,device,log_file,hp.SEQ_LENGTH)

        log_message(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}", log_file)
        log_message(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}", log_file)

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), f'log/models/best_model_{timestamp}.pth')
            log_message("[保存] 最佳模型已保存", log_file)
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= hp.PATIENCE:
                log_message("[早停] 验证准确率未提升，训练终止", log_file)
                break

    model.load_state_dict(torch.load(f'log/models/best_model_{timestamp}.pth'))
    test_loss, test_acc = validate(model, test_loader, criterion, device, log_file)
    # back_test(model,test_loader_not_shuffle,device,log_file,hp.SEQ_LENGTH)
    log_message(f"[测试] Loss: {test_loss:.4f}, Acc: {test_acc:.4f}", log_file)

if __name__ == '__main__':
    main()
