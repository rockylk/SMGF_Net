# -*- coding: utf-8 -*-
# =========================================================
# Baseline: FS-Net Proxy (Bi-GRU Sequence Model)
# 核心定位: 模拟以 FS-Net 为代表的端到端序列流量分类模型
# 对齐指标: Clean Acc, Macro-F1, Robust Acc, Robust F1, ASR, Rel-Drop, ECE
# =========================================================

import os
import time
import copy
import random
import warnings
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from scapy.all import rdpcap, IP, TCP, UDP
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

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
LOG_FILE = os.path.join(RESULT_DIR, f"FSNet_Baseline_Log_{int(time.time())}.txt")

BATCH_SIZE = 512
MAX_SEQ_LEN = 200  # RNN 处理长序列较慢且容易梯度消失，200 是一个很好的平衡点

class Logger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.log("=" * 80)
        self.log("=== Baseline: FS-Net Proxy (Bi-GRU Sequence Model) ===")
        self.log(f"Hardware: {torch.cuda.get_device_name(0)}")
        self.log(f"Config: BATCH={BATCH_SIZE}, SEQ_LEN={MAX_SEQ_LEN}")
        self.log("=" * 80)

    def log(self, message):
        print(message)
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(str(message) + '\n')

# =========================================================
# 1. 核心特征工程: 构造一维包序列
# =========================================================
def extract_raw_x_from_pcap(pcap_path, max_pkts=MAX_SEQ_LEN*2):
    """提取基础的一维流信息 (Dir, Size, IAT)"""
    try:
        packets = rdpcap(pcap_path)
        packets = [p for p in packets if IP in p and (TCP in p or UDP in p)]
        if len(packets) < 5: return None
        client_ip = packets[0][IP].src
    except: return None

    raw_x = []
    prev_time = float(packets[0].time)
    for pkt in packets[:max_pkts]:
        curr_time = float(pkt.time)
        direction = 0 if pkt[IP].src == client_ip else 1
        size = len(pkt)
        iat = curr_time - prev_time
        raw_x.append([direction, size, iat])
        prev_time = curr_time
        
    return np.array(raw_x, dtype=np.float32)

def load_seq_dataset(root_dir, logger):
    subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {name: i for i, name in enumerate(subdirs)}
    logger.log(f"Loading FS-Net Dataset: Total {len(subdirs)} classes.")
    
    RAW_X_list, Y_list = [], []
    for class_name in subdirs:
        class_dir = os.path.join(root_dir, class_name)
        label_int = class_to_idx[class_name]
        for f in [f for f in os.listdir(class_dir) if f.endswith((".pcap", ".pcapng"))]:
            raw_x = extract_raw_x_from_pcap(os.path.join(class_dir, f))
            if raw_x is not None:
                RAW_X_list.append(raw_x)
                Y_list.append(label_int)
                
    return RAW_X_list, np.array(Y_list), len(subdirs)

# =========================================================
# 2. 数据集类、分层抽样与序列预处理
# =========================================================
class SeqDataset(Dataset):
    def __init__(self, X_padded, Y):
        self.X = torch.tensor(X_padded, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)
    def __len__(self): return len(self.Y)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

def split_dataset_stratified(RAW_X, Y, logger, train_ratio=0.8):
    class_buckets = defaultdict(list)
    for x, y in zip(RAW_X, Y):
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
    return X_train, np.array(Y_train), X_test, np.array(Y_test)

def fit_scaler_and_pad(X_raw_list, scaler=None, is_train=True, max_len=MAX_SEQ_LEN):
    """对序列特征进行标准化，并填充/截断为固定长度 (Batch, Seq_Len, 3)"""
    if is_train:
        all_tokens = np.vstack(X_raw_list)
        scaler = StandardScaler().fit(all_tokens)
        
    X_padded = []
    for x in X_raw_list:
        x_scaled = scaler.transform(x)
        if len(x_scaled) >= max_len:
            X_padded.append(x_scaled[:max_len])
        else:
            pad_width = max_len - len(x_scaled)
            X_padded.append(np.pad(x_scaled, ((0, pad_width), (0, 0)), 'constant', constant_values=0))
            
    return np.array(X_padded, dtype=np.float32), scaler

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
# 3. FS-Net 代理模型架构 (Bi-GRU + Attention)
# =========================================================
class FS_Net_Proxy(nn.Module):
    def __init__(self, num_classes, input_dim=3, hidden_dim=128, num_layers=2):
        super().__init__()
        # 核心 Bi-GRU 提取时序特征
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, 
                          num_layers=num_layers, batch_first=True, bidirectional=True)
        
        # 序列注意力机制 (替代简单的最后一步输出，增强基线性能)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len, 3)
        gru_out, _ = self.gru(x) # gru_out shape: (Batch, Seq_Len, hidden_dim * 2)
        
        # 序列注意力池化
        attn_weights = F.softmax(self.attention(gru_out), dim=1) # (Batch, Seq_Len, 1)
        context_vector = torch.sum(attn_weights * gru_out, dim=1) # (Batch, hidden_dim * 2)
        
        return self.classifier(context_vector)

# =========================================================
# 4. 物理攻击模块
# =========================================================
def sanitize_raw_x(raw_x):
    raw_x = raw_x.copy()
    raw_x[:, 0] = (raw_x[:, 0] >= 0.5).astype(np.float32)
    raw_x[:, 1] = np.clip(raw_x[:, 1], 1.0, None)
    raw_x[:, 2] = np.clip(raw_x[:, 2], 1e-6, None)
    return raw_x

def attack_padding_raw(raw_x, strength=0.5):
    raw_x = sanitize_raw_x(raw_x)
    idx = np.random.choice(len(raw_x), size=min(max(1, int(len(raw_x) * (0.10 + 0.20 * strength))), len(raw_x)), replace=False)
    raw_x[idx, 1] += np.random.randint(20, max(int(64 + 512 * strength), 21), size=len(idx))
    return raw_x

def attack_jitter_raw(raw_x, strength=0.5):
    raw_x = sanitize_raw_x(raw_x)
    idx = np.random.choice(len(raw_x), size=min(max(1, int(len(raw_x) * (0.20 + 0.30 * strength))), len(raw_x)), replace=False)
    raw_x[idx, 2] *= np.random.uniform(1.0 - (0.10 + 0.70 * strength), 1.0 + (0.10 + 0.70 * strength), size=len(idx))
    return raw_x

def attack_dummy_raw(raw_x, strength=0.5):
    raw_x = sanitize_raw_x(raw_x)
    median_iat = max(float(np.median(raw_x[:, 2])), 1e-6)
    new_list, offset = raw_x.tolist(), 0
    for p in sorted(np.random.randint(1, max(2, len(raw_x)), size=max(1, int(len(raw_x) * (0.03 + 0.10 * strength)))).tolist()):
        base = new_list[min(max(p - 1 + offset, 0), len(new_list) - 1)]
        new_list.insert(min(p + offset, len(new_list)), [base[0] if random.random() < 0.7 else (1 - base[0]), random.uniform(40, 200 + 400 * strength), max(median_iat * random.uniform(0.2, 1.0), 1e-6)])
        offset += 1
    return np.array(new_list, dtype=np.float32)

def apply_physical_attack_to_raw(X_test_raw, attack_name, strength=0.5):
    X_atk_raw = []
    for x in X_test_raw:
        if attack_name == 'clean': atk_x = x.copy()
        elif attack_name == 'padding': atk_x = attack_padding_raw(x, strength)
        elif attack_name == 'iat_jitter': atk_x = attack_jitter_raw(x, strength)
        elif attack_name == 'dummy': atk_x = attack_dummy_raw(x, strength)
        elif attack_name in ['mixed', 'adaptive']: 
            atk_x = attack_dummy_raw(attack_jitter_raw(attack_padding_raw(x, strength), strength), strength)
        X_atk_raw.append(atk_x)
    return X_atk_raw

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

def evaluate_attack(model, X_test_raw, Y_test, scaler, logger, attack_name, strength, clean_acc=None, clean_preds=None):
    # 1. 对原始流应用物理层攻击 (打乱包顺序/修改长度)
    X_atk_raw = apply_physical_attack_to_raw(X_test_raw, attack_name, strength)
    
    # 2. 对攻击后的序列重新进行标准化与 Pad (因为 Dummy 攻击可能增加了序列长度)
    X_atk_padded, _ = fit_scaler_and_pad(X_atk_raw, scaler=scaler, is_train=False)
    
    loader = DataLoader(SeqDataset(X_atk_padded, Y_test), batch_size=BATCH_SIZE, shuffle=False)
    preds, labels, probs = predict_model(model, loader)
    
    rob_acc = accuracy_score(labels, preds) * 100.0
    rob_f1 = f1_score(labels, preds, average='macro') * 100.0
    ece = compute_ece(probs, labels)
    
    if clean_preds is not None and clean_acc is not None:
        asr = (np.sum((clean_preds == labels) & (preds != labels)) / max(np.sum(clean_preds == labels), 1)) * 100.0
        drop = ((clean_acc - rob_acc) / max(clean_acc, 1e-8)) * 100.0
        logger.log(f"[FS-Net | {attack_name:8s} | Str: {strength:.1f}] Rob_Acc: {rob_acc:.2f}% | Rob_F1: {rob_f1:.2f}% | ASR: {asr:.2f}% | Drop: {drop:.2f}% | ECE: {ece:.2f}%")
        return rob_acc, rob_f1, asr, drop, ece
    else:
        logger.log(f"[FS-Net | Clean Baseline ] Acc: {rob_acc:.2f}% | Macro-F1: {rob_f1:.2f}% | ECE: {ece:.2f}%")
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
    
    # 1. 数据加载
    X_all_raw, Y_all, num_classes = load_seq_dataset(pcap_dir, logger)
    X_train_raw, Y_train, X_test_raw, Y_test = split_dataset_stratified(X_all_raw, Y_all, logger, train_ratio=0.8)
    
    # 2. 训练集特征预处理
    X_train_padded, scaler = fit_scaler_and_pad(X_train_raw, is_train=True)
    train_loader = DataLoader(SeqDataset(X_train_padded, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. 模型训练
    model = FS_Net_Proxy(num_classes=num_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    logger.log("\n>>> Training FS-Net (Bi-GRU Sequence) Proxy...")
    EPOCHS = 40
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
        if (epoch + 1) % 10 == 0:
            logger.log(f"[FS-Net] Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")
            
    # 4. 核心评估环节
    logger.log("\n" + "="*60 + "\nFS-Net BASELINE EVALUATION\n" + "="*60)
    
    # Clean 评估
    c_acc, c_f1, c_ece, clean_preds = evaluate_attack(model, X_test_raw, Y_test, scaler, logger, 'clean', 0.0)
    
    # 攻击评估
    results = []
    attacks = ['padding', 'dummy', 'mixed', 'adaptive']
    strengths = [0.4, 0.7, 1.0]
    
    for atk in attacks:
        for s in strengths:
            r_acc, r_f1, asr, drop, ece = evaluate_attack(model, X_test_raw, Y_test, scaler, logger, atk, s, c_acc, clean_preds)
            results.append([atk, s, f"{r_acc:.2f}", f"{r_f1:.2f}", f"{asr:.2f}", f"{drop:.2f}", f"{ece:.2f}"])
            
    csv_path = os.path.join(RESULT_DIR, "baseline_fsnet_attacks.csv")
    save_csv(csv_path, results, ["Attack", "Strength", "Robust_Acc", "Robust_F1", "ASR", "Relative_Drop", "ECE"])
    logger.log(f"\n[Done] Baseline FS-Net results saved to {csv_path}")