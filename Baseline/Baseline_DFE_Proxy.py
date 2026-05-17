# -*- coding: utf-8 -*-
# =========================================================
# Baseline: DFE (Deep Flow Embedding) Proxy
# 核心定位: 模拟主流的基于宏观统计量的伪图像 CNN 流量分类模型
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
import torch.nn.functional as F
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, Linear, Sequential, Dropout
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
LOG_FILE = os.path.join(RESULT_DIR, f"DFE_Baseline_Log_{int(time.time())}.txt")

BATCH_SIZE = 2048  # CNN 占用显存极小，可与 V14 保持一致

class Logger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.log("=" * 80)
        self.log("=== Baseline: DFE Proxy (9x9 Pseudo-image CNN) ===")
        self.log(f"Hardware: {torch.cuda.get_device_name(0)}")
        self.log("=" * 80)

    def log(self, message):
        print(message)
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(str(message) + '\n')

# =========================================================
# 1. 核心特征工程: 构造 DFE 81维特征 (9x9 伪图像)
# =========================================================
def extract_raw_x_from_pcap(pcap_path, max_pkts=400):
    """提取基础的一维流信息 (Dir, Size, IAT) 供后续攻击和转换使用"""
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

def raw_x_to_dfe_features(raw_x):
    """
    将 raw_x 转化为 DFE 论文中典型的 81 维统计量
    构成: 前24个包的Size(24) + IAT(24) + Dir(24) + Size统计(4) + IAT统计(4) + Duration(1) = 81
    """
    N = len(raw_x)
    dirs = raw_x[:, 0]
    sizes = np.clip(raw_x[:, 1], 1.0, None)
    iats = np.clip(raw_x[:, 2], 1e-6, None)
    
    # 截取前 24 个包特征 (不足补 0)
    seq_len = 24
    f_sizes = sizes[:seq_len] if N >= seq_len else np.pad(sizes, (0, seq_len - N), 'constant')
    f_iats = iats[:seq_len] if N >= seq_len else np.pad(iats, (0, seq_len - N), 'constant')
    f_dirs = dirs[:seq_len] if N >= seq_len else np.pad(dirs, (0, seq_len - N), 'constant')
    
    # 全局统计量 (4 + 4 + 1 = 9)
    stats = [
        np.mean(sizes), np.std(sizes), np.max(sizes), np.min(sizes),
        np.mean(iats), np.std(iats), np.max(iats), np.min(iats),
        np.sum(iats) # Duration
    ]
    
    # 拼接为 81 维
    features_81 = np.concatenate([f_sizes, f_iats, f_dirs, stats])
    return features_81.astype(np.float32)

def load_dfe_dataset(root_dir, logger):
    subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {name: i for i, name in enumerate(subdirs)}
    logger.log(f"Loading DFE Dataset: Total {len(subdirs)} classes.")
    
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
# 2. 数据集类与分层抽样
# =========================================================
class DFEDataset(Dataset):
    def __init__(self, dfe_features, Y):
        # 传入的 dfe_features 是已经标准化并 Reshape 为 (N, 1, 9, 9) 的张量
        self.X = torch.tensor(dfe_features, dtype=torch.float32)
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

def fit_and_transform_dfe(X_train_raw, X_test_raw):
    """将 Raw_X 转化为 81 维特征，进行 StandardScaler，再 Reshape 为 9x9"""
    X_train_81 = np.array([raw_x_to_dfe_features(x) for x in X_train_raw])
    scaler = StandardScaler().fit(X_train_81)
    
    # Transform
    X_train_scaled = scaler.transform(X_train_81).reshape(-1, 1, 9, 9)
    
    # 我们不在这里直接转换 X_test_raw，因为测试时需要先应用物理攻击再转换
    return X_train_scaled, scaler

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
# 3. DFE 2D-CNN 模型架构
# =========================================================
class DFE_CNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Input: 1 x 9 x 9
        self.features = Sequential(
            Conv2d(1, 16, kernel_size=3, padding=1),
            BatchNorm2d(16),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2), # Output: 16 x 4 x 4
            
            Conv2d(16, 32, kernel_size=3, padding=1),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2)  # Output: 32 x 2 x 2
        )
        self.classifier = Sequential(
            Linear(32 * 2 * 2, 64),
            ReLU(),
            Dropout(0.5),
            Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# =========================================================
# 4. 对抗攻击模块 (严格对齐 V14，物理篡改 Raw_X)
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
            # DFE 只看统计量，没有语义(Entropy)概念。因此对它来说，Adaptive == Mixed
            atk_x = attack_dummy_raw(attack_jitter_raw(attack_padding_raw(x, strength), strength), strength)
        X_atk_raw.append(atk_x)
    return X_atk_raw

# =========================================================
# 5. 训练与评估流程
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
    # 1. 对原始流应用物理层攻击
    X_atk_raw = apply_physical_attack_to_raw(X_test_raw, attack_name, strength)
    # 2. 提取 DFE 特征
    X_atk_81 = np.array([raw_x_to_dfe_features(x) for x in X_atk_raw])
    # 3. 标准化并重塑伪图像
    X_atk_scaled = scaler.transform(X_atk_81).reshape(-1, 1, 9, 9)
    
    loader = DataLoader(DFEDataset(X_atk_scaled, Y_test), batch_size=BATCH_SIZE, shuffle=False)
    preds, labels, probs = predict_model(model, loader)
    
    rob_acc = accuracy_score(labels, preds) * 100.0
    rob_f1 = f1_score(labels, preds, average='macro') * 100.0
    ece = compute_ece(probs, labels)
    
    if clean_preds is not None and clean_acc is not None:
        asr = (np.sum((clean_preds == labels) & (preds != labels)) / max(np.sum(clean_preds == labels), 1)) * 100.0
        drop = ((clean_acc - rob_acc) / max(clean_acc, 1e-8)) * 100.0
        logger.log(f"[DFE | {attack_name:8s} | Str: {strength:.1f}] Rob_Acc: {rob_acc:.2f}% | Rob_F1: {rob_f1:.2f}% | ASR: {asr:.2f}% | Drop: {drop:.2f}% | ECE: {ece:.2f}%")
        return rob_acc, rob_f1, asr, drop, ece
    else:
        logger.log(f"[DFE | Clean Baseline ] Acc: {rob_acc:.2f}% | Macro-F1: {rob_f1:.2f}% | ECE: {ece:.2f}%")
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
    pcap_dir = r"/home/adminl/桌面/lkfiles/aes-128-gcm/aes-128-gcm" # 请根据实际路径修改
    
    # 1. 数据加载
    X_all_raw, Y_all, num_classes = load_dfe_dataset(pcap_dir, logger)
    X_train_raw, Y_train, X_test_raw, Y_test = split_dataset_stratified(X_all_raw, Y_all, logger, train_ratio=0.8)
    
    # 2. 训练集提取特征与标准化 (生成 9x9 图像)
    X_train_scaled, scaler = fit_and_transform_dfe(X_train_raw, X_test_raw)
    train_loader = DataLoader(DFEDataset(X_train_scaled, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. 模型训练
    model = DFE_CNN(num_classes=num_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    logger.log("\n>>> Training DFE (2D-CNN Pseudo-image) Proxy...")
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
            logger.log(f"[DFE] Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")
            
    # 4. 核心评估环节
    logger.log("\n" + "="*60 + "\nDFE BASELINE EVALUATION\n" + "="*60)
    
    # Clean 评估
    c_acc, c_f1, c_ece, clean_preds = evaluate_attack(model, X_test_raw, Y_test, scaler, logger, 'clean', 0.0)
    
    # 攻击评估
    results = []
    attacks = ['dummy', 'iat_jitter', 'mixed', 'adaptive']
    strengths = [0.4, 0.7, 1.0]
    
    for atk in attacks:
        for s in strengths:
            r_acc, r_f1, asr, drop, ece = evaluate_attack(model, X_test_raw, Y_test, scaler, logger, atk, s, c_acc, clean_preds)
            results.append([atk, s, f"{r_acc:.2f}", f"{r_f1:.2f}", f"{asr:.2f}", f"{drop:.2f}", f"{ece:.2f}"])
            
    # 保存结果
    csv_path = os.path.join(RESULT_DIR, "baseline_dfe_attacks.csv")
    save_csv(csv_path, results, ["Attack", "Strength", "Robust_Acc", "Robust_F1", "ASR", "Relative_Drop", "ECE"])
    logger.log(f"\n[Done] Baseline DFE results saved to {csv_path}")