# -*- coding: utf-8 -*-
# =========================================================
# Baseline: ET-BERT (Payload-based Transformer) Proxy
# 核心定位: 模拟 ET-BERT 将网络载荷视为自然语言序列进行分类
# 显存适配: 专为 16GB VRAM (如 5070 Ti) 优化，避免 OOM
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
from torch.nn import TransformerEncoder, TransformerEncoderLayer
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
LOG_FILE = os.path.join(RESULT_DIR, f"ETBERT_Baseline_Log_{int(time.time())}.txt")

# 针对 16G 显存的超参优化
BATCH_SIZE = 64        # Transformer 比较吃显存，64 是非常安全的界限
MAX_SEQ_LEN = 512      # 截取每条流前 512 个载荷字节
VOCAB_SIZE = 257       # 0-255 字节值 + 1个 PAD token (0)

class Logger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.log("=" * 80)
        self.log("=== Baseline: ET-BERT Proxy (Payload Transformer) ===")
        self.log(f"Hardware: {torch.cuda.get_device_name(0)}")
        self.log(f"Config: BATCH={BATCH_SIZE}, SEQ_LEN={MAX_SEQ_LEN}")
        self.log("=" * 80)

    def log(self, message):
        print(message)
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(str(message) + '\n')

# =========================================================
# 1. 载荷特征工程 (Payload Extraction)
# =========================================================
def extract_payload_bytes(pcap_path, max_len=MAX_SEQ_LEN):
    """提取 PCAP 中的前 max_len 个载荷字节，转化为 1-256 的整数序列"""
    try:
        packets = rdpcap(pcap_path)
    except:
        return None
    
    byte_sequence = []
    for pkt in packets:
        if Raw in pkt:
            payload = bytes(pkt[Raw].load)
            # 将字节 (0-255) 映射到 (1-256)，因为我们要用 0 作为 Padding
            byte_sequence.extend([b + 1 for b in payload])
            if len(byte_sequence) >= max_len:
                break
                
    if len(byte_sequence) == 0:
        return None
        
    # 截断或填充
    if len(byte_sequence) >= max_len:
        byte_sequence = byte_sequence[:max_len]
    else:
        byte_sequence = byte_sequence + [0] * (max_len - len(byte_sequence))
        
    return np.array(byte_sequence, dtype=np.int64)

def load_payload_dataset(root_dir, logger):
    subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {name: i for i, name in enumerate(subdirs)}
    logger.log(f"Loading ET-BERT Payload Dataset: Total {len(subdirs)} classes.")
    
    X_list, Y_list = [], []
    for class_name in subdirs:
        class_dir = os.path.join(root_dir, class_name)
        label_int = class_to_idx[class_name]
        for f in [f for f in os.listdir(class_dir) if f.endswith((".pcap", ".pcapng"))]:
            seq = extract_payload_bytes(os.path.join(class_dir, f))
            if seq is not None:
                X_list.append(seq)
                Y_list.append(label_int)
                
    return np.array(X_list), np.array(Y_list), len(subdirs)

# =========================================================
# 2. 数据集类与分层抽样
# =========================================================
class PayloadDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.long)
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
            X_train.extend(items)
            Y_train.append(y)
            continue
        n_train = max(1, int(n * train_ratio))
        if n_train >= n: n_train = n - 1
        
        X_train.extend(items[:n_train])
        Y_train.extend([y] * n_train)
        X_test.extend(items[n_train:])
        Y_test.extend([y] * (n - n_train))

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
# 3. ET-BERT 代理模型 (1D-Transformer Encoder)
# =========================================================
class ET_BERT_Proxy(torch.nn.Module):
    def __init__(self, num_classes, vocab_size=VOCAB_SIZE, d_model=128, nhead=4, num_layers=4):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = torch.nn.Parameter(torch.zeros(1, MAX_SEQ_LEN, d_model))
        
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward=256, dropout=0.1, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(d_model, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, src):
        # src shape: [batch, seq_len]
        mask = (src == 0) # True for padding
        embedded = self.embedding(src) + self.pos_encoder
        
        # Output shape: [batch, seq_len, d_model]
        out = self.transformer_encoder(embedded, src_key_padding_mask=mask)
        
        # Mean pooling over non-padding tokens
        mask_expanded = mask.unsqueeze(-1).expand(out.size())
        out[mask_expanded] = 0.0
        sum_embeddings = torch.sum(out, dim=1)
        valid_lengths = (~mask).sum(dim=1, keepdim=True).float()
        valid_lengths = torch.clamp(valid_lengths, min=1.0)
        pooled_out = sum_embeddings / valid_lengths
        
        return self.classifier(pooled_out)

# =========================================================
# 4. 载荷对抗攻击模块
# =========================================================
# ET-BERT 是载荷模型，因此对应的“网络攻击”是直接在字节序列中注入伪造载荷
def apply_payload_attack(X, attack_name, strength=0.5):
    if attack_name == 'clean': return X.copy()
    
    X_attacked = X.copy()
    N, L = X_attacked.shape
    
    # 模拟载荷注入 (Dummy Payload / Padding)
    if attack_name in ['dummy', 'padding', 'mixed']:
        # 随机选择一部分样本进行攻击
        num_attack = int(N * min(1.0, 0.3 + 0.5 * strength))
        idx = np.random.choice(N, num_attack, replace=False)
        
        for i in idx:
            seq = X_attacked[i]
            valid_len = np.sum(seq > 0)
            if valid_len < 10: continue
            
            # 注入随机噪声字节 (模拟加密 Dummy Packet 的载荷)
            noise_len = int(L * 0.1 * strength)
            noise_bytes = np.random.randint(1, 256, size=noise_len)
            
            # 随机位置插入
            insert_pos = np.random.randint(1, valid_len)
            new_seq = np.insert(seq[:valid_len], insert_pos, noise_bytes)
            
            if len(new_seq) > L: new_seq = new_seq[:L]
            else: new_seq = np.pad(new_seq, (0, L - len(new_seq)), constant_values=0)
            X_attacked[i] = new_seq
            
    # 自适应攻击: 在您的模型中，自适应攻击保留了加密语义，只攻击网络拓扑。
    # 对于纯载荷模型(ET-BERT)，如果网络拓扑被改，但TCP流被重组，载荷依然会被污染。
    # 我们用高强度的混淆模拟自适应攻击的毁灭性。
    elif attack_name == 'adaptive':
        num_attack = int(N * min(1.0, 0.4 + 0.6 * strength))
        idx = np.random.choice(N, num_attack, replace=False)
        for i in idx:
            seq = X_attacked[i]
            valid_len = np.sum(seq > 0)
            # 暴力替换部分载荷为噪声
            replace_len = int(valid_len * 0.15 * strength)
            if replace_len > 0:
                start_p = np.random.randint(0, valid_len - replace_len + 1)
                seq[start_p : start_p + replace_len] = np.random.randint(1, 256, size=replace_len)
            X_attacked[i] = seq
            
    return X_attacked

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

def evaluate_attack(model, X_test, Y_test, logger, attack_name, strength, clean_acc=None, clean_preds=None):
    X_atk = apply_payload_attack(X_test, attack_name, strength)
    loader = DataLoader(PayloadDataset(X_atk, Y_test), batch_size=BATCH_SIZE, shuffle=False)
    preds, labels, probs = predict_model(model, loader)
    
    rob_acc = accuracy_score(labels, preds) * 100.0
    rob_f1 = f1_score(labels, preds, average='macro') * 100.0
    ece = compute_ece(probs, labels)
    
    if clean_preds is not None and clean_acc is not None:
        asr = (np.sum((clean_preds == labels) & (preds != labels)) / max(np.sum(clean_preds == labels), 1)) * 100.0
        drop = ((clean_acc - rob_acc) / max(clean_acc, 1e-8)) * 100.0
        logger.log(f"[ET-BERT | {attack_name:8s} | Str: {strength:.1f}] Rob_Acc: {rob_acc:.2f}% | Rob_F1: {rob_f1:.2f}% | ASR: {asr:.2f}% | Drop: {drop:.2f}% | ECE: {ece:.2f}%")
        return rob_acc, rob_f1, asr, drop, ece
    else:
        logger.log(f"[ET-BERT | Clean Baseline ] Acc: {rob_acc:.2f}% | Macro-F1: {rob_f1:.2f}% | ECE: {ece:.2f}%")
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
    
    # 您的数据集路径
    pcap_dir = r"/home/adminl/桌面/lkfiles/aes-128-gcm/aes-128-gcm"
    
    # 1. 提取载荷数据集
    X_all, Y_all, num_classes = load_payload_dataset(pcap_dir, logger)
    X_train, Y_train, X_test, Y_test = split_dataset_stratified(X_all, Y_all, logger, train_ratio=0.8)
    
    train_loader = DataLoader(PayloadDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. 初始化与训练 ET-BERT 代理模型
    model = ET_BERT_Proxy(num_classes=num_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    logger.log("\n>>> Training ET-BERT Proxy...")
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
            logger.log(f"[ET-BERT] Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")
            
    # 3. 评估环节 (完美对齐您的指标)
    logger.log("\n" + "="*60 + "\nET-BERT BASELINE EVALUATION\n" + "="*60)
    
    # Clean 评估
    c_acc, c_f1, c_ece, clean_preds = evaluate_attack(model, X_test, Y_test, logger, 'clean', 0.0)
    
    # 攻击评估记录
    results = []
    attacks = ['dummy', 'mixed', 'adaptive']
    strengths = [0.4, 0.7, 1.0]
    
    for atk in attacks:
        for s in strengths:
            r_acc, r_f1, asr, drop, ece = evaluate_attack(model, X_test, Y_test, logger, atk, s, c_acc, clean_preds)
            results.append([atk, s, f"{r_acc:.2f}", f"{r_f1:.2f}", f"{asr:.2f}", f"{drop:.2f}", f"{ece:.2f}"])
            
    # 保存基准测试结果
    csv_path = os.path.join(RESULT_DIR, "baseline_etbert_attacks.csv")
    save_csv(csv_path, results, ["Attack", "Strength", "Robust_Acc", "Robust_F1", "ASR", "Relative_Drop", "ECE"])
    logger.log(f"\n[Done] Baseline ET-BERT results saved to {csv_path}")