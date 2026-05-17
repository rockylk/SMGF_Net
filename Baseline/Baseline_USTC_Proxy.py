# -*- coding: utf-8 -*-
# =========================================================
# Baseline: USTC-TFC2016 Proxy (Traffic as Image CNN)
# 核心定位: 严格复现原文献中的 28x28 灰度图 + LeNet-5 改良版 CNN
# 对齐指标: Clean Acc, Macro-F1, Robust Acc, Robust F1, ASR, Rel-Drop, ECE
# =========================================================

import os
import time
import random
import warnings
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from scapy.all import rdpcap, IP, TCP, UDP, Raw
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

# 强制要求并使用 CUDA
if not torch.cuda.is_available():
    raise RuntimeError("【错误】未检测到可用 GPU！")
DEVICE = torch.device('cuda')

# =========================================================
# 0. 全局配置
# =========================================================
RESULT_DIR = r"./RESULT_BASELINE"
os.makedirs(RESULT_DIR, exist_ok=True)
LOG_FILE = os.path.join(RESULT_DIR, f"USTC_Baseline_Log_{int(time.time())}.txt")

BATCH_SIZE = 512
MAX_BYTES = 784  # 原论文设定：取前 784 个字节转为 28x28 图像

class Logger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.log("=" * 80)
        self.log("=== Baseline: USTC-TFC2016 Original Proxy (Traffic as Image) ===")
        self.log(f"Hardware: {torch.cuda.get_device_name(0)}")
        self.log(f"Config: BATCH={BATCH_SIZE}, BYTES_LIMIT={MAX_BYTES}")
        self.log("=" * 80)

    def log(self, message):
        print(message)
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(str(message) + '\n')

# =========================================================
# 1. 载荷特征提取 (严格按照 USTC 原文献)
# =========================================================
def extract_ustc_payload_bytes(pcap_path, max_len=MAX_BYTES):
    """提取 PCAP 中的前 784 个载荷字节，取值 0-255"""
    try:
        packets = rdpcap(pcap_path)
    except:
        return None
    
    byte_sequence = []
    for pkt in packets:
        if Raw in pkt:
            payload = bytes(pkt[Raw].load)
            byte_sequence.extend(list(payload))
            if len(byte_sequence) >= max_len:
                break
                
    if len(byte_sequence) == 0:
        return None
        
    # 截断或填充补零 (Padding 0x00)
    if len(byte_sequence) >= max_len:
        byte_sequence = byte_sequence[:max_len]
    else:
        byte_sequence = byte_sequence + [0] * (max_len - len(byte_sequence))
        
    return np.array(byte_sequence, dtype=np.float32)

def load_ustc_dataset(root_dir, logger):
    subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {name: i for i, name in enumerate(subdirs)}
    logger.log(f"Loading USTC Dataset: Total {len(subdirs)} classes.")
    
    X_list, Y_list = [], []
    for class_name in subdirs:
        class_dir = os.path.join(root_dir, class_name)
        label_int = class_to_idx[class_name]
        for f in [f for f in os.listdir(class_dir) if f.endswith((".pcap", ".pcapng"))]:
            seq = extract_ustc_payload_bytes(os.path.join(class_dir, f))
            if seq is not None:
                X_list.append(seq)
                Y_list.append(label_int)
                
    return np.array(X_list), np.array(Y_list), len(subdirs)

# =========================================================
# 2. 数据集类与预处理
# =========================================================
class USTCDataset(Dataset):
    def __init__(self, X_raw, Y):
        # 归一化到 [0, 1] 并重塑为 (1, 28, 28) 的图像格式
        X_normalized = X_raw / 255.0
        self.X = torch.tensor(X_normalized, dtype=torch.float32).view(-1, 1, 28, 28)
        self.Y = torch.tensor(Y, dtype=torch.long)
    def __len__(self):
        return len(self.Y)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def split_dataset_stratified(X, Y, logger, train_ratio=0.8):
    class_buckets = defaultdict(list)
    for x, y in zip(X, Y):
        class_buckets[y].append(x)

    X_train, Y_train, X_test, Y_test = [], [], [], []
    for y, items in class_buckets.items():
        random.shuffle(items)
        n = len(items)
        if n == 1:
            X_train.extend(items); Y_train.append(y)
            continue
        n_train = max(1, int(n * train_ratio))
        if n_train >= n: n_train = n - 1
        X_train.extend(items[:n_train]); Y_train.extend([y] * n_train)
        X_test.extend(items[n_train:]); Y_test.extend([y] * (n - n_train))

    logger.log(f"Dataset Split -> Train: {len(Y_train)}, Test: {len(Y_test)}")
    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)

def compute_ece(probs, labels, n_bins=15):
    if len(probs) == 0: return 0.0
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float32)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if np.sum(mask) > 0:
            ece += (np.sum(mask) / len(labels)) * abs(np.mean(accuracies[mask]) - np.mean(confidences[mask]))
    return float(ece) * 100.0

# =========================================================
# 3. USTC 原始模型架构 (完全复现论文结构)
# =========================================================
class USTC_Original_CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 论文 C1: 32 kernels of 5x5, padding=2 to keep 28x28
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2) # P1: output 14x14
        
        # 论文 C2: 64 kernels of 5x5, padding=2 to keep 14x14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2) # P2: output 7x7
        
        # 论文 Fully Connection: 1024
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# =========================================================
# 4. 对抗攻击模块 (基于字节流注入)
# =========================================================
def apply_payload_attack(X, attack_name, strength=0.5):
    """由于 USTC 模型基于纯载荷，攻击体现为向字节流中注入噪声，导致像素错位"""
    if attack_name == 'clean': return X.copy()
    
    X_attacked = X.copy()
    N, L = X_attacked.shape  # L = 784
    
    # 模拟 Padding 或 Dummy 注入
    if attack_name in ['dummy', 'padding', 'mixed']:
        num_attack = int(N * min(1.0, 0.3 + 0.5 * strength))
        idx = np.random.choice(N, num_attack, replace=False)
        
        for i in idx:
            seq = X_attacked[i].copy()
            valid_len = np.sum(seq > 0)
            if valid_len < 10: continue
            
            # 注入随机噪声字节 (模拟加密数据)
            noise_len = int(L * 0.1 * strength)
            noise_bytes = np.random.randint(1, 256, size=noise_len)
            
            # 随机位置插入导致空间平移错位
            insert_pos = np.random.randint(1, valid_len)
            new_seq = np.insert(seq[:valid_len], insert_pos, noise_bytes)
            
            if len(new_seq) > L: new_seq = new_seq[:L]
            else: new_seq = np.pad(new_seq, (0, L - len(new_seq)), constant_values=0)
            X_attacked[i] = new_seq
            
    # Adaptive: 强行替换部分内容以模拟极限混淆
    elif attack_name == 'adaptive':
        num_attack = int(N * min(1.0, 0.4 + 0.6 * strength))
        idx = np.random.choice(N, num_attack, replace=False)
        for i in idx:
            seq = X_attacked[i].copy()
            valid_len = int(np.sum(seq > 0))
            replace_len = int(valid_len * 0.15 * strength)
            if replace_len > 0:
                start_p = np.random.randint(0, max(1, valid_len - replace_len + 1))
                seq[start_p : start_p + replace_len] = np.random.randint(1, 256, size=replace_len)
            X_attacked[i] = seq
            
    return X_attacked

# =========================================================
# 5. 评估流程
# =========================================================
def predict_model(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(DEVICE))
            probs = F.softmax(logits, dim=1)
            all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def evaluate_attack(model, X_test, Y_test, logger, attack_name, strength, clean_acc=None, clean_preds=None):
    # 应用攻击
    X_atk = apply_payload_attack(X_test, attack_name, strength)
    # 打包为 Dataset (内部会自动做 /255.0 和 Reshape)
    loader = DataLoader(USTCDataset(X_atk, Y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    preds, labels, probs = predict_model(model, loader)
    rob_acc = accuracy_score(labels, preds) * 100.0
    rob_f1 = f1_score(labels, preds, average='macro') * 100.0
    ece = compute_ece(probs, labels)
    
    if clean_preds is not None and clean_acc is not None:
        asr = (np.sum((clean_preds == labels) & (preds != labels)) / max(np.sum(clean_preds == labels), 1)) * 100.0
        drop = ((clean_acc - rob_acc) / max(clean_acc, 1e-8)) * 100.0
        logger.log(f"[USTC-CNN | {attack_name:8s} | Str: {strength:.1f}] Rob_Acc: {rob_acc:.2f}% | Rob_F1: {rob_f1:.2f}% | ASR: {asr:.2f}% | Drop: {drop:.2f}% | ECE: {ece:.2f}%")
        return rob_acc, rob_f1, asr, drop, ece
    else:
        logger.log(f"[USTC-CNN | Clean Baseline ] Acc: {rob_acc:.2f}% | Macro-F1: {rob_f1:.2f}% | ECE: {ece:.2f}%")
        return rob_acc, rob_f1, ece, preds

def save_csv(filepath, rows, header):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(','.join(header) + '\n')
        for r in rows: f.write(','.join(map(str, r)) + '\n')

# =========================================================
# 6. 主函数
# =========================================================
if __name__ == "__main__":
    logger = Logger(LOG_FILE)
    pcap_dir = r"/home/adminl/桌面/lkfiles/aes-128-gcm/aes-128-gcm"
    
    # 1. 提取载荷
    X_all_raw, Y_all, num_classes = load_ustc_dataset(pcap_dir, logger)
    X_train_raw, Y_train, X_test_raw, Y_test = split_dataset_stratified(X_all_raw, Y_all, logger, train_ratio=0.8)
    
    train_loader = DataLoader(USTCDataset(X_train_raw, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. 初始化与训练
    model = USTC_Original_CNN(num_classes=num_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    logger.log("\n>>> Training USTC-TFC2016 Original CNN (LeNet-5 Variant)...")
    EPOCHS = 30
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x.to(DEVICE))
            loss = criterion(logits, y.to(DEVICE))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            logger.log(f"[USTC] Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")
            
    # 3. 核心对抗评估
    logger.log("\n" + "="*60 + "\nUSTC-TFC2016 CNN BASELINE EVALUATION\n" + "="*60)
    
    # Clean 评估
    c_acc, c_f1, c_ece, clean_preds = evaluate_attack(model, X_test_raw, Y_test, logger, 'clean', 0.0)
    
    # 对抗攻击
    results = []
    attacks = ['dummy', 'mixed', 'adaptive']
    strengths = [0.4, 0.7, 1.0]
    
    for atk in attacks:
        for s in strengths:
            r_acc, r_f1, asr, drop, ece = evaluate_attack(model, X_test_raw, Y_test, logger, atk, s, c_acc, clean_preds)
            results.append([atk, s, f"{r_acc:.2f}", f"{r_f1:.2f}", f"{asr:.2f}", f"{drop:.2f}", f"{ece:.2f}"])
            
    # 保存结果
    csv_path = os.path.join(RESULT_DIR, "baseline_ustc_attacks.csv")
    save_csv(csv_path, results, ["Attack", "Strength", "Robust_Acc", "Robust_F1", "ASR", "Relative_Drop", "ECE"])
    logger.log(f"\n[Done] Baseline USTC results saved to {csv_path}")