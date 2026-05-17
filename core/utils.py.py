# -*- coding: utf-8 -*-
import os
import time
import random
import torch
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

class Logger:
    """Standard Logger for experiment tracking."""
    def __init__(self, filepath, batch_size):
        self.filepath = filepath
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self.log("=" * 80)
        self.log("=== SMGF-Net: Adversarially Robust Encrypted Traffic Framework ===")
        self.log(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        self.log(f"Hardware: GPU CUDA Activated ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
        self.log(f"Global Config: BATCH_SIZE = {batch_size}")
        self.log("=" * 80)

    def log(self, message):
        print(message)
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(str(message) + '\n')

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def split_dataset_stratified(dataset, logger, train_ratio=0.8):
    """Stratified sampling to handle imbalanced datasets (e.g., long-tail malware)."""
    class_buckets = defaultdict(list)
    for d in dataset:
        class_buckets[int(d.y.item())].append(d)

    train_set, test_set = [], []
    for y, items in class_buckets.items():
        random.shuffle(items)
        n = len(items)
        if n == 1:
            train_set.extend(items)
            continue
        n_train = max(1, int(n * train_ratio))
        if n_train >= n: n_train = n - 1
        train_set.extend(items[:n_train])
        test_set.extend(items[n_train:])

    random.shuffle(train_set)
    random.shuffle(test_set)
    logger.log(f"Dataset Split (Stratified) -> Train: {len(train_set)}, Test: {len(test_set)}")
    return train_set, test_set

def fit_and_apply_scalers(train_set, test_set, logger):
    """Normalize macroscopic features to prevent gradient explosion."""
    if not train_set: return None, None
    node_scaler = StandardScaler().fit(np.concatenate([d.x.numpy() for d in train_set], axis=0))
    stats_scaler = StandardScaler().fit(np.concatenate([d.stats_attr.numpy() for d in train_set], axis=0))

    for ds in [train_set, test_set]:
        for data in ds:
            data.x = torch.tensor(node_scaler.transform(data.x.numpy()), dtype=torch.float)
            data.stats_attr = torch.tensor(stats_scaler.transform(data.stats_attr.numpy()), dtype=torch.float)
    logger.log("[Scaler] Fitted on train_set and applied to train/test.")
    return node_scaler, stats_scaler

def compute_ece(probs, labels, n_bins=15):
    """
    Calculate Expected Calibration Error (ECE).
    Ref: Equation (16) in the paper. Evaluates overconfidence under adversarial evasion.
    """
    if len(probs) == 0: return 0.0
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float32)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if np.sum(mask) > 0:
            acc_bin = np.mean(accuracies[mask])
            conf_bin = np.mean(confidences[mask])
            ece += (np.sum(mask) / len(labels)) * abs(acc_bin - conf_bin)
    return float(ece) * 100.0

def save_simple_csv(filepath, rows, header):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(','.join(header) + '\n')
        for row in rows: f.write(','.join(map(str, row)) + '\n')