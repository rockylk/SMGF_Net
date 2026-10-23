# -*- coding: utf-8 -*-
"""
V16 Final Paper Experiment Runner
=================================

Purpose
-------
This is the paper-grade V16 experiment suite. It preserves the corrected
Phase-3 data/attack pipeline and the successful Phase-4 perturbation-aware
fine-tuning, then restores the broad V15 evaluation structure with three
training seeds, multiple perturbation types/strengths, strict component
ablations, representation/sensitivity/gating figures, and deployment metrics.

The FINAL/FULL method is:
  corrected pipeline + learned semantic gate + contrastive pretraining
  + robust fine-tuning (clean CE + attacked CE + consistency KL).

Ablations are strict and interpretable:
  STRICT_NO_GATE       : same GatedFusion module, gate multiplier forced to 1
  NO_CONTRASTIVE       : no contrastive pretraining
  NO_ROBUST_FT         : clean supervised fine-tuning only
  NO_CONSISTENCY_KL    : robust fine-tuning without the KL term

Key design rules
----------------
1. Dataset file enumeration is sorted and a dataset manifest fingerprint is logged.
2. One canonical stratified Train/Validation/Test split is created ONCE.
3. Scalers are fitted ONCE on Train only and reused everywhere.
4. V16 FINAL uses the same corrected split/scaler/attack definitions established in Phase-3/4.
5. FULL is trained once per training seed and REUSED by main experiments + ablation.
6. Attack sets are generated ONCE with dedicated deterministic seeds and reused by all models.
7. Main and ablation endpoints use the same strengths:
      Mixed    = 1.0
      Adaptive = 1.0
8. Multiple training seeds are run in one invocation and results are reported as mean +/- SD.
9. The test set is never used for early stopping; validation loss selects the best fine-tuning checkpoint.
10. Primary metrics are written to one log and a paper-oriented Markdown report; no CSV is produced.
11. Model-only deployment latency is clearly labeled as model-only latency.

The final method intentionally retains the Phase-4 consistency regularization
because Phase-4 demonstrated a large robustness gain under the corrected pipeline.
No ECE-specific loss or label smoothing is introduced.
"""

import os

# Must be set before CUDA initialization for stronger reproducibility where supported.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
import time
import copy
import json
import math
import random
import hashlib
import platform
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.nn import (
    Linear, Sequential, BatchNorm1d, ReLU,
    Dropout, Sigmoid, Conv1d, MaxPool1d
)

import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, AttentionalAggregation
from torch_geometric.utils import subgraph

from scapy.all import rdpcap, IP, TCP, UDP, Raw
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import sklearn

warnings.filterwarnings("ignore")


# =========================================================
# 0. EXPERIMENT CONFIGURATION
# =========================================================

# Dataset: can be overridden without editing the file:
#   Linux/macOS:  export SMGF_PCAP_DIR=/path/to/aes-128-gcm
#   Windows PS:   $env:SMGF_PCAP_DIR='D:\\dataset\\aes-128-gcm'
PCAP_DIR = os.environ.get(
    "SMGF_PCAP_DIR",
    r"/home/admin1/dataset/aes-128-gcm/aes-128-gcm"
)
DATASET_NAME = os.environ.get("SMGF_DATASET_NAME", "CipherSpectrum")

# Output root. A timestamped run folder is created inside this directory.
RESULT_ROOT = os.environ.get("SMGF_RESULT_DIR", str(Path.cwd() / "RESULT"))

# Reproducibility design:
SPLIT_SEED = 42
ATTACK_BASE_SEED = 20260919
TRAIN_SEEDS = [42, 43, 44]

# Canonical split: approximately 72% / 8% / 20% per class.
TRAIN_RATIO = 0.72
VAL_RATIO = 0.08
TEST_RATIO = 0.20

# Training protocol.
BATCH_SIZE = 64
PRETRAIN_EPOCHS = 30
FINETUNE_EPOCHS = 50
EARLY_STOP_PATIENCE = 5
PRETRAIN_LR = 0.001
FINETUNE_LR = 0.002
WEIGHT_DECAY = 1e-4
CONTRASTIVE_TEMPERATURE = 0.1

# Final robust fine-tuning protocol inherited from Phase-4.
ROBUST_BATCH_RATIO = 0.25
ROBUST_STRENGTH_MIN = 0.3
ROBUST_STRENGTH_MAX = 0.9
ATTACK_CE_WEIGHT = 1.0
CONSISTENCY_KL_WEIGHT = 0.2

# Fixed Phase-3 seed-42 reference, used only for reporting deltas.
PHASE3_BASELINE = {
    "clean_acc": 86.73,
    "mixed_robust_acc": 48.35,
    "adaptive_robust_acc": 72.18,
}

# Evaluation protocol. These restore the V15 multi-attack structure while
# retaining the Phase-4 primary endpoints for direct continuity.
ECE_BINS = 15
MULTI_ATTACKS = ["padding", "iat_jitter", "dummy", "mixed"]
MULTI_STRENGTHS = [0.3, 0.6, 0.9, 1.0]
ADAPTIVE_STRENGTHS = [0.4, 0.7, 1.0]
PRIMARY_MIXED_STRENGTH = 1.0
PRIMARY_ADAPTIVE_STRENGTH = 1.0

# Optional artifacts.
SAVE_CHECKPOINTS = True
RUN_TSNE = True
RUN_MODEL_ONLY_LATENCY = True
RUN_SENSITIVITY_FIGURE = True
RUN_GATE_DIAGNOSTICS = True
TSNE_MAX_POINTS = 400
LATENCY_MAX_SAMPLES = 500
GATE_DIAG_MAX_SAMPLES = 1000

# Strict paper ablations evaluated only at Clean / primary Mixed / primary Adaptive.
ABLATION_VARIANTS = [
    "FULL",
    "STRICT_NO_GATE",
    "NO_CONTRASTIVE",
    "NO_ROBUST_FT",
    "NO_CONSISTENCY_KL",
]


# =========================================================
# 1. RUNTIME / LOGGING / REPRODUCIBILITY UTILITIES
# =========================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "No CUDA GPU detected. This script intentionally keeps the original GPU-only execution policy."
    )
DEVICE = torch.device("cuda")

RUN_STAMP = time.strftime("%Y%m%d_%H%M%S", time.localtime())
RUN_DIR = Path(RESULT_ROOT) / f"SMGF_V16_FINAL_{RUN_STAMP}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = RUN_DIR / "experiment.log"
RESULT_MD = RUN_DIR / "results.md"
SPLIT_MANIFEST_FILE = RUN_DIR / "split_manifest.json"
CONFIG_FILE = RUN_DIR / "run_config.json"
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
FIGURE_DIR = RUN_DIR / "figures"
if SAVE_CHECKPOINTS:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
if RUN_TSNE or RUN_SENSITIVITY_FIGURE or RUN_GATE_DIAGNOSTICS:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


class Logger:
    def __init__(self, filepath: Path):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message=""):
        msg = str(message)
        print(msg)
        with self.filepath.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")


LOGGER = Logger(LOG_FILE)


def set_seed(seed: int, deterministic: bool = True):
    """Seed Python / NumPy / PyTorch and request deterministic algorithms where possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyG/CUDA scatter kernels may not be bitwise deterministic on all versions.
        # warn_only keeps the experiment running but surfaces unsupported operations.
        torch.use_deterministic_algorithms(True, warn_only=True)


def stable_seed(*parts, modulo: int = 2**31 - 1) -> int:
    text = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo


def log_runtime_environment(logger: Logger):
    logger.log("=" * 96)
    logger.log("SMGF-Net V16 FINAL - Paper-grade multi-seed experiment suite")
    logger.log(f"Run directory: {RUN_DIR}")
    logger.log(f"Dataset label: {DATASET_NAME}")
    logger.log(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    logger.log(f"Python: {sys.version.split()[0]}")
    logger.log(f"Platform: {platform.platform()}")
    logger.log(f"PyTorch: {torch.__version__}")
    logger.log(f"PyTorch Geometric: {torch_geometric.__version__}")
    logger.log(f"scikit-learn: {sklearn.__version__}")
    logger.log(f"CUDA available: {torch.cuda.is_available()}")
    logger.log(f"CUDA runtime: {torch.version.cuda}")
    logger.log(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.log(f"SPLIT_SEED={SPLIT_SEED}")
    logger.log(f"TRAIN_SEEDS={TRAIN_SEEDS}")
    logger.log(f"ATTACK_BASE_SEED={ATTACK_BASE_SEED}")
    logger.log(
        f"Split ratios train/val/test = {TRAIN_RATIO:.2f}/{VAL_RATIO:.2f}/{TEST_RATIO:.2f}"
    )
    logger.log(
        f"Primary endpoints: Mixed={PRIMARY_MIXED_STRENGTH:.1f}, "
        f"Adaptive={PRIMARY_ADAPTIVE_STRENGTH:.1f}"
    )
    logger.log("Deterministic algorithms requested: True (warn_only for unsupported CUDA/PyG ops)")
    logger.log("FINAL MODE: corrected pipeline + learned gate + contrastive pretraining + robust fine-tuning.")
    logger.log(
        f"Robust batches={ROBUST_BATCH_RATIO:.0%}; Mixed strength="
        f"[{ROBUST_STRENGTH_MIN:.1f}, {ROBUST_STRENGTH_MAX:.1f}]; "
        f"attack_CE_weight={ATTACK_CE_WEIGHT}; consistency_KL_weight={CONSISTENCY_KL_WEIGHT}."
    )
    logger.log(f"TRAIN_SEEDS={TRAIN_SEEDS}; full multi-attack evaluation + strict ablations enabled.")
    logger.log(
        "PIPELINE FIXES: unified 400-packet observation window, discrete direction restoration, "
        "zero-strength no-op, and attacked-window truncation."
    )
    logger.log("=" * 96)


# =========================================================
# 2. FEATURE ENGINEERING (kept from V14, with stable sample indexing)
# =========================================================

def get_direction(pkt, client_ip):
    try:
        return 0 if pkt[IP].src == client_ip else 1
    except Exception:
        return 0


def calculate_entropy(data):
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = counts / len(data)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def extract_features_v13(packets, client_ip, max_nodes=400):
    node_features, sizes, times = [], [], []
    observed_packets = packets[:max_nodes]
    start_time = float(observed_packets[0].time)
    total_bytes, large_pkt_count, uplink_count = 0, 0, 0
    # Phase-1 correction carried into Phase-3:
    # all statistics use the SAME observation window as graph nodes.
    pkt_count = len(observed_packets)
    duration = float(observed_packets[-1].time) - start_time

    handshake_len = 20
    raw_fingerprint = []
    entropy_seq_len = 5
    entropy_seq = []

    prev_time = start_time
    for pkt in observed_packets:
        curr_time = float(pkt.time)
        size = len(pkt)
        direction = get_direction(pkt, client_ip)
        iat = curr_time - prev_time

        signed_size = size if direction == 0 else -size
        if len(raw_fingerprint) < handshake_len:
            raw_fingerprint.append(signed_size)

        if len(entropy_seq) < entropy_seq_len and Raw in pkt:
            payload = pkt[Raw].load
            if len(payload) > 0:
                ent = calculate_entropy(payload[:128])
                entropy_seq.append(ent if direction == 0 else -ent)

        total_bytes += size
        if size > 1200:
            large_pkt_count += 1
        if direction == 0:
            uplink_count += 1

        sizes.append(size)
        times.append(iat)
        node_features.append([
            direction,
            size,
            iat,
            np.mean(sizes[-5:]),
            np.mean(times[-5:]),
        ])
        prev_time = curr_time

    while len(raw_fingerprint) < handshake_len:
        raw_fingerprint.append(0)
    while len(entropy_seq) < entropy_seq_len:
        entropy_seq.append(0.0)

    avg_size = np.mean(sizes) if sizes else 0
    avg_iat = np.mean(times) if times else 0
    std_size = np.std(sizes) if sizes else 0
    std_iat = np.std(times) if times else 0
    large_ratio = large_pkt_count / pkt_count if pkt_count > 0 else 0
    uplink_ratio = uplink_count / pkt_count if pkt_count > 0 else 0

    stats_features = [
        np.log1p(pkt_count),
        np.log1p(duration),
        np.log1p(total_bytes),
        avg_size,
        avg_iat,
        std_size,
        std_iat,
        large_ratio,
        uplink_ratio,
    ]

    return (
        np.array(node_features, dtype=np.float32),
        np.array(stats_features, dtype=np.float32),
        np.array(raw_fingerprint, dtype=np.float32),
        np.array(entropy_seq, dtype=np.float32),
    )


def pcap_to_graph_v13(pcap_path, label_int, sample_index):
    try:
        packets = rdpcap(pcap_path)
        packets = [p for p in packets if IP in p and (TCP in p or UDP in p)]
        if len(packets) < 5:
            return None
        client_ip = packets[0][IP].src
    except Exception:
        return None

    node_feats, stats_feats, finger_feats, entropy_feats = extract_features_v13(packets, client_ip)

    x = torch.tensor(node_feats, dtype=torch.float)
    stats_x = torch.tensor(stats_feats, dtype=torch.float).unsqueeze(0)
    finger_x = torch.tensor(finger_feats, dtype=torch.float).unsqueeze(0)
    entropy_x = torch.tensor(entropy_feats, dtype=torch.float).unsqueeze(0)

    num_nodes = x.size(0)
    edge_index, edge_attr = [], []
    for i in range(num_nodes - 1):
        edge_index.extend([[i, i + 1], [i + 1, i]])
        edge_attr.extend([[x[i + 1, 2]], [x[i + 1, 2]]])

    if not edge_index:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr = torch.tensor([[0.0]], dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([label_int], dtype=torch.long),
        stats_attr=stats_x,
        finger_attr=finger_x,
        entropy_attr=entropy_x,
        num_nodes=num_nodes,
        sample_index=torch.tensor([sample_index], dtype=torch.long),
    )


def load_dataset_closedset(root_dir: str, logger: Logger):
    root = Path(root_dir)
    subdirs = sorted([p.name for p in root.iterdir() if p.is_dir()])
    class_to_idx = {name: i for i, name in enumerate(subdirs)}

    logger.log(f"Loading dataset from: {root.resolve()}")
    logger.log(f"Detected classes: {len(subdirs)}")

    dataset = []
    metadata = []
    skipped = 0

    # IMPORTANT: file enumeration is sorted. This removes one source of split drift.
    for class_name in subdirs:
        class_dir = root / class_name
        label_int = class_to_idx[class_name]
        files = sorted([
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".pcap", ".pcapng"}
        ], key=lambda p: p.name)

        for p in files:
            sample_index = len(metadata)
            data = pcap_to_graph_v13(str(p), label_int, sample_index)
            if data is not None and data.x.size(0) > 0:
                dataset.append(data)
                metadata.append({
                    "sample_index": sample_index,
                    "relative_path": str(p.relative_to(root)).replace("\\", "/"),
                    "class_name": class_name,
                    "label": label_int,
                    "file_size": p.stat().st_size,
                })
            else:
                skipped += 1

    # sample_index must be contiguous for valid samples only. Reindex if any file was skipped.
    for new_idx, (data, meta) in enumerate(zip(dataset, metadata)):
        data.sample_index = torch.tensor([new_idx], dtype=torch.long)
        meta["sample_index"] = new_idx

    idx_to_class = {i: n for n, i in class_to_idx.items()}
    logger.log(f"Loaded valid samples: {len(dataset)} | skipped: {skipped}")
    return dataset, len(subdirs), idx_to_class, metadata


def dataset_manifest_hash(metadata: List[dict]) -> str:
    h = hashlib.sha256()
    for row in sorted(metadata, key=lambda r: r["sample_index"]):
        token = (
            f"{row['sample_index']}|{row['relative_path']}|{row['class_name']}|"
            f"{row['label']}|{row['file_size']}\n"
        )
        h.update(token.encode("utf-8"))
    return h.hexdigest()


def stratified_split_indices(
    dataset: List[Data],
    split_seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
):
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, abs_tol=1e-9):
        raise ValueError("train/val/test ratios must sum to 1.0")

    labels = np.array([int(d.y.item()) for d in dataset], dtype=np.int64)
    rng = np.random.default_rng(split_seed)

    train_idx, val_idx, test_idx = [], [], []
    for cls in sorted(np.unique(labels)):
        cls_idx = np.where(labels == cls)[0].copy()
        cls_idx.sort()
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        if n < 3:
            raise RuntimeError(
                f"Class {cls} has only {n} samples; at least 3 are required for train/val/test."
            )

        n_test = max(1, int(round(n * test_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_test + n_val >= n:
            # Preserve at least one training sample.
            overflow = n_test + n_val - (n - 1)
            while overflow > 0 and n_test > 1:
                n_test -= 1
                overflow -= 1
            while overflow > 0 and n_val > 1:
                n_val -= 1
                overflow -= 1
            if overflow > 0:
                raise RuntimeError(f"Cannot split class {cls} with n={n} into all three partitions.")

        test_idx.extend(cls_idx[:n_test].tolist())
        val_idx.extend(cls_idx[n_test:n_test + n_val].tolist())
        train_idx.extend(cls_idx[n_test + n_val:].tolist())

    # Keep each partition in deterministic sample-index order.
    train_idx = sorted(train_idx)
    val_idx = sorted(val_idx)
    test_idx = sorted(test_idx)
    return train_idx, val_idx, test_idx


def materialize_split(dataset, indices):
    return [dataset[i] for i in indices]


def save_split_manifest(metadata, train_idx, val_idx, test_idx, manifest_hash):
    by_idx = {m["sample_index"]: m for m in metadata}

    def rows(indices):
        return [by_idx[i] for i in indices]

    payload = {
        "dataset_manifest_sha256": manifest_hash,
        "split_seed": SPLIT_SEED,
        "ratios": {
            "train": TRAIN_RATIO,
            "validation": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "counts": {
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        "train": rows(train_idx),
        "validation": rows(val_idx),
        "test": rows(test_idx),
    }
    with SPLIT_MANIFEST_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def fit_and_apply_scalers(train_set, val_set, test_set, logger: Logger):
    all_node_features = np.concatenate([d.x.numpy() for d in train_set], axis=0)
    node_scaler = StandardScaler().fit(all_node_features)

    all_stats = np.concatenate([d.stats_attr.numpy() for d in train_set], axis=0)
    stats_scaler = StandardScaler().fit(all_stats)

    # IMPORTANT: scaler fitted only on training data and then reused everywhere.
    for ds_name, ds in [("train", train_set), ("validation", val_set), ("test", test_set)]:
        for data in ds:
            data.x = torch.tensor(node_scaler.transform(data.x.numpy()), dtype=torch.float)
            data.stats_attr = torch.tensor(
                stats_scaler.transform(data.stats_attr.numpy()), dtype=torch.float
            )
        logger.log(f"Scaler applied to {ds_name}: {len(ds)} samples")

    logger.log("[Scaler] Fitted ONCE on train only; reused for validation/test/attacks/all models.")
    return node_scaler, stats_scaler


# =========================================================
# 3. MODEL DEFINITION (kept from V14)
# =========================================================

def augment_graph_view(data, crop_ratio_range=(0.5, 0.9), mask_prob=0.15):
    device = data.x.device
    num_nodes = data.num_nodes if not isinstance(data.num_nodes, torch.Tensor) else data.num_nodes.item()
    min_crop = max(5, int(num_nodes * crop_ratio_range[0]))
    max_crop = int(num_nodes * crop_ratio_range[1])

    if num_nodes > min_crop:
        crop_len = random.randint(min_crop, min(max_crop, num_nodes))
        start_idx = random.randint(0, num_nodes - crop_len)
        subset = torch.arange(start_idx, start_idx + crop_len, device=device)
        edge_index, edge_attr = subgraph(
            subset, data.edge_index, data.edge_attr, relabel_nodes=True
        )
        x = data.x[subset]
    else:
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

    mask = torch.rand(x.shape, device=device) > mask_prob
    x = x * mask.float()
    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
    return x, edge_index, edge_attr, batch


class FingerprintCNN(torch.nn.Module):
    def __init__(self, seq_len=20, out_dim=32):
        super().__init__()
        self.conv1 = Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn1 = BatchNorm1d(16)
        self.pool1 = MaxPool1d(2)
        self.conv2 = Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = BatchNorm1d(32)
        self.pool2 = MaxPool1d(2)
        self.fc = Linear(32 * (seq_len // 4), out_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(self.bn1(self.conv1(x)).relu())
        x = self.pool2(self.bn2(self.conv2(x)).relu())
        return self.fc(x.view(x.size(0), -1)).relu()


class EntropyCNN(torch.nn.Module):
    def __init__(self, seq_len=5, out_dim=16):
        super().__init__()
        self.conv1 = Conv1d(1, 8, kernel_size=3, padding=1)
        self.bn1 = BatchNorm1d(8)
        self.fc = Linear(8 * seq_len, out_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.bn1(self.conv1(x)).relu()
        return self.fc(x.view(x.size(0), -1)).relu()


class GatedFusionV13(torch.nn.Module):
    """
    Identical fusion structure for both Phase-3 variants.

    gate_mode="learned": original sigmoid multiplicative gate.
    gate_mode="identity": strict no-gating ablation; gate multiplier is exactly 1.

    The global_gate module is instantiated in BOTH modes so parameter structure
    and initialization remain identical. In identity mode its output is simply
    bypassed in forward().
    """
    def __init__(self, graph_dim, stats_dim, finger_dim, entropy_dim, gate_mode="learned"):
        super().__init__()
        if gate_mode not in {"learned", "identity"}:
            raise ValueError(f"Unsupported gate_mode: {gate_mode}")
        self.gate_mode = gate_mode
        global_ctx_dim = stats_dim + finger_dim + entropy_dim
        self.global_gate = Sequential(
            Linear(global_ctx_dim, graph_dim),
            ReLU(),
            Linear(graph_dim, graph_dim),
            Sigmoid(),
        )
        self.global_proj = Linear(global_ctx_dim, 64)

    def forward(self, h_graph, h_stats, h_finger, h_entropy):
        h_global_ctx = torch.cat([h_stats, h_finger, h_entropy], dim=1)
        if self.gate_mode == "learned":
            gate = self.global_gate(h_global_ctx)
        else:
            gate = torch.ones_like(h_graph)
        return torch.cat([h_graph * gate, self.global_proj(h_global_ctx)], dim=1)


class SimpleFusionV13(torch.nn.Module):
    """Ablation fusion without semantics-guided gating."""
    def __init__(self, graph_dim, stats_dim, finger_dim, entropy_dim):
        super().__init__()
        in_dim = graph_dim + stats_dim + finger_dim + entropy_dim
        self.proj = Sequential(
            Linear(in_dim, graph_dim + 64),
            ReLU(),
        )

    def forward(self, h_graph, h_stats, h_finger, h_entropy):
        h = torch.cat([h_graph, h_stats, h_finger, h_entropy], dim=1)
        return self.proj(h)


def info_nce_loss(z1, z2, temperature=0.2):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    cos_sim = torch.mm(z1, z2.t()) / temperature
    labels = torch.arange(z1.shape[0], device=z1.device)
    return F.cross_entropy(cos_sim, labels)


class ContrastiveGNN_V13(torch.nn.Module):
    def __init__(
        self,
        node_in_channels,
        stats_dim,
        finger_seq_len,
        entropy_seq_len,
        hidden_channels,
        out_channels,
        gate_mode="learned",
    ):
        super().__init__()
        self.gate_mode = gate_mode

        self.conv1 = GATv2Conv(
            node_in_channels, hidden_channels, heads=4, edge_dim=1, concat=True
        )
        self.bn1 = BatchNorm1d(hidden_channels * 4)
        self.conv2 = GATv2Conv(
            hidden_channels * 4, hidden_channels, heads=4, edge_dim=1, concat=True
        )
        self.bn2 = BatchNorm1d(hidden_channels * 4)
        self.lin_skip2 = Linear(hidden_channels * 4, hidden_channels * 4)
        self.conv3 = GATv2Conv(
            hidden_channels * 4, hidden_channels, heads=4, edge_dim=1, concat=True
        )
        self.bn3 = BatchNorm1d(hidden_channels * 4)
        self.lin_skip3 = Linear(hidden_channels * 4, hidden_channels * 4)
        self.pool = AttentionalAggregation(gate_nn=Linear(hidden_channels * 4, 1))

        self.graph_dim = hidden_channels * 4
        self.finger_cnn = FingerprintCNN(seq_len=finger_seq_len, out_dim=32)
        self.entropy_cnn = EntropyCNN(seq_len=entropy_seq_len, out_dim=16)

        # Phase-3 deliberately uses the SAME fusion module for both variants.
        self.fusion_layer = GatedFusionV13(
            self.graph_dim, stats_dim, 32, 16, gate_mode=gate_mode
        )

        self.projection_head = Sequential(
            Linear(self.graph_dim, 256), ReLU(), Linear(256, 128)
        )
        self.final_dim = self.graph_dim + 64
        self.classifier_head = Sequential(
            Linear(self.final_dim, 128),
            ReLU(),
            Dropout(0.5),
            Linear(128, out_channels),
        )

    def encode_graph(self, x, edge_index, edge_attr, batch):
        x1 = self.bn1(self.conv1(x, edge_index, edge_attr=edge_attr)).relu()
        x2 = (
            self.bn2(self.conv2(x1, edge_index, edge_attr=edge_attr)).relu()
            + self.lin_skip2(x1)
        )
        x3 = (
            self.bn3(self.conv3(x2, edge_index, edge_attr=edge_attr)).relu()
            + self.lin_skip3(x2)
        )
        return self.pool(x3, batch)

    def forward_contrastive(self, x, edge_index, edge_attr, batch):
        return self.projection_head(self.encode_graph(x, edge_index, edge_attr, batch))

    def extract_fused_features(
        self, x, edge_index, edge_attr, batch, stats_attr, finger_attr, entropy_attr
    ):
        h_graph = self.encode_graph(x, edge_index, edge_attr, batch)
        h_finger = self.finger_cnn(finger_attr)
        h_entropy = self.entropy_cnn(entropy_attr)
        return self.fusion_layer(h_graph, stats_attr, h_finger, h_entropy)

    def forward_classifier(
        self, x, edge_index, edge_attr, batch, stats_attr, finger_attr, entropy_attr
    ):
        h_fused = self.extract_fused_features(
            x, edge_index, edge_attr, batch, stats_attr, finger_attr, entropy_attr
        )
        return self.classifier_head(h_fused)


# =========================================================
# 4. ATTACK MODULE - deterministic attack realization
# =========================================================

def clone_data(data):
    return copy.deepcopy(data)


def rebuild_sequential_edges_from_raw_x(raw_x):
    num_nodes = raw_x.shape[0]
    edge_index, edge_attr = [], []
    for i in range(num_nodes - 1):
        # Match clean graph construction exactly; do not inject reconstruction jitter.
        iat_val = float(max(raw_x[i + 1, 2], 0.0))
        edge_index.extend([[i, i + 1], [i + 1, i]])
        edge_attr.extend([[iat_val], [iat_val]])

    if not edge_index:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr = torch.tensor([[0.0]], dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return edge_index, edge_attr


def update_temporal_columns(raw_x):
    raw_x = raw_x.copy()
    sizes = np.clip(raw_x[:, 1], 1.0, None)
    iats = np.clip(raw_x[:, 2], 0.0, None)
    for i in range(len(raw_x)):
        s0 = max(0, i - 4)
        raw_x[i, 3] = float(np.mean(sizes[s0:i + 1]))
        raw_x[i, 4] = float(np.mean(iats[s0:i + 1]))
    raw_x[:, 1] = sizes
    raw_x[:, 2] = iats
    return raw_x


def recompute_stats_from_raw_x(raw_x):
    sizes = np.clip(raw_x[:, 1], 1.0, None)
    iats = np.clip(raw_x[:, 2], 0.0, None)
    dirs = np.rint(np.clip(raw_x[:, 0], 0.0, 1.0))
    return np.array([
        np.log1p(len(raw_x)),
        np.log1p(float(np.sum(iats))),
        np.log1p(float(np.sum(sizes))),
        float(np.mean(sizes)),
        float(np.mean(iats)),
        float(np.std(sizes)),
        float(np.std(iats)),
        float(np.mean(sizes > 1200)),
        float(np.mean(dirs == 0)),
    ], dtype=np.float32)


def build_fingerprint_from_raw_x(raw_x, seq_len=20):
    dirs = raw_x[:, 0]
    sizes = np.clip(raw_x[:, 1], 1.0, None)
    signed = np.where(dirs == 0, sizes, -sizes)
    if len(signed) >= seq_len:
        return signed[:seq_len].astype(np.float32)
    return np.pad(
        signed,
        (0, seq_len - len(signed)),
        constant_values=0.0,
    ).astype(np.float32)


def inverse_raw_features(data, node_scaler, stats_scaler):
    raw_x = node_scaler.inverse_transform(data.x.cpu().numpy())
    # Direction is a discrete raw attribute; eliminate inverse-scaling float drift.
    raw_x[:, 0] = np.rint(np.clip(raw_x[:, 0], 0.0, 1.0))
    raw_stats = stats_scaler.inverse_transform(data.stats_attr.cpu().numpy())[0]
    return raw_x, raw_stats


def _phase1_feature_reconstruction_check(original_raw_x, rebuilt_raw_x, name=""):
    """
    Phase-1 diagnostic:
    attack reconstruction should not introduce unexpected changes beyond
    the actual perturbation pipeline.
    """
    if original_raw_x.shape == rebuilt_raw_x.shape:
        diff = np.abs(original_raw_x - rebuilt_raw_x)
        return {
            "name": name,
            "shape_equal": True,
            "mean_abs_diff": float(diff.mean()),
            "max_abs_diff": float(diff.max()),
        }
    return {
        "name": name,
        "shape_equal": False,
        "mean_abs_diff": None,
        "max_abs_diff": None,
    }


def finalize_attacked_data(
    data,
    raw_x,
    node_scaler,
    stats_scaler,
    new_finger=None,
    new_entropy=None,
):
    # Keep the same maximum observation window used by clean extraction.
    raw_x = np.asarray(raw_x, dtype=np.float32)[:400].copy()
    raw_x[:, 0] = np.rint(np.clip(raw_x[:, 0], 0.0, 1.0))
    raw_x = update_temporal_columns(raw_x)
    new_stats = recompute_stats_from_raw_x(raw_x)
    edge_index, edge_attr = rebuild_sequential_edges_from_raw_x(raw_x)

    attacked = clone_data(data)
    attacked.x = torch.tensor(node_scaler.transform(raw_x), dtype=torch.float)
    attacked.stats_attr = torch.tensor(
        stats_scaler.transform(new_stats.reshape(1, -1)), dtype=torch.float
    )
    attacked.edge_index = edge_index
    attacked.edge_attr = edge_attr
    attacked.num_nodes = raw_x.shape[0]

    if new_finger is not None:
        attacked.finger_attr = torch.tensor(new_finger.reshape(1, -1), dtype=torch.float)
    if new_entropy is not None:
        attacked.entropy_attr = torch.tensor(new_entropy.reshape(1, -1), dtype=torch.float)
    return attacked


def attack_packet_length_padding(
    data,
    node_scaler,
    stats_scaler,
    strength,
    np_rng,
    py_rng,
    preserve_semantic=False,
):
    raw_x, _ = inverse_raw_features(data, node_scaler, stats_scaler)
    n = len(raw_x)
    k = max(1, int(n * (0.10 + 0.20 * strength)))
    idx = np_rng.choice(n, size=min(k, n), replace=False)
    pads = np_rng.integers(20, max(int(64 + 512 * strength), 21), size=len(idx))
    raw_x[idx, 1] = np.clip(raw_x[idx, 1] + pads, 1.0, None)

    new_finger = (
        data.finger_attr.cpu().numpy()[0]
        if preserve_semantic
        else build_fingerprint_from_raw_x(raw_x)
    )
    return finalize_attacked_data(
        data,
        raw_x,
        node_scaler,
        stats_scaler,
        new_finger,
        data.entropy_attr.cpu().numpy()[0],
    )


def attack_iat_jitter(
    data,
    node_scaler,
    stats_scaler,
    strength,
    np_rng,
    py_rng,
    preserve_semantic=False,
):
    raw_x, _ = inverse_raw_features(data, node_scaler, stats_scaler)
    n = len(raw_x)
    k = max(1, int(n * (0.20 + 0.30 * strength)))
    idx = np_rng.choice(n, size=min(k, n), replace=False)
    amp = 0.10 + 0.70 * strength
    jitter = np_rng.uniform(1.0 - amp, 1.0 + amp, size=len(idx))
    raw_x[idx, 2] = np.clip(raw_x[idx, 2] * jitter, 0.0, None)
    return finalize_attacked_data(
        data,
        raw_x,
        node_scaler,
        stats_scaler,
        data.finger_attr.cpu().numpy()[0],
        data.entropy_attr.cpu().numpy()[0],
    )


def attack_dummy_injection(
    data,
    node_scaler,
    stats_scaler,
    strength,
    np_rng,
    py_rng,
    preserve_semantic=False,
):
    raw_x, _ = inverse_raw_features(data, node_scaler, stats_scaler)
    n = len(raw_x)
    insert_num = max(1, int(n * (0.03 + 0.10 * strength)))
    median_iat = max(float(np.median(np.clip(raw_x[:, 2], 1e-6, None))), 1e-6)
    positions = sorted(np_rng.integers(1, max(2, n), size=insert_num).tolist())
    new_list = raw_x.tolist()

    offset = 0
    for p in positions:
        base_idx = min(max(p - 1 + offset, 0), len(new_list) - 1)
        base = np.array(new_list[base_idx], dtype=np.float32)
        dummy_dir = base[0] if py_rng.random() < 0.7 else (1 - base[0])
        dummy_size = py_rng.uniform(40, 200 + 400 * strength)
        dummy_iat = max(median_iat * py_rng.uniform(0.2, 1.0), 1e-6)
        new_list.insert(
            min(p + offset, len(new_list)),
            [dummy_dir, dummy_size, dummy_iat, dummy_size, dummy_iat],
        )
        offset += 1

    raw_x = np.array(new_list, dtype=np.float32)
    new_finger = (
        data.finger_attr.cpu().numpy()[0]
        if preserve_semantic
        else build_fingerprint_from_raw_x(raw_x)
    )
    return finalize_attacked_data(
        data,
        raw_x,
        node_scaler,
        stats_scaler,
        new_finger,
        data.entropy_attr.cpu().numpy()[0],
    )


def attack_mixed(
    data,
    node_scaler,
    stats_scaler,
    strength,
    np_rng,
    py_rng,
    preserve_semantic=False,
):
    out = attack_packet_length_padding(
        data, node_scaler, stats_scaler, strength, np_rng, py_rng, preserve_semantic
    )
    out = attack_iat_jitter(
        out, node_scaler, stats_scaler, strength, np_rng, py_rng, preserve_semantic
    )
    out = attack_dummy_injection(
        out, node_scaler, stats_scaler, strength, np_rng, py_rng, preserve_semantic
    )
    return out


def apply_attack_deterministic(
    data,
    attack_name,
    node_scaler,
    stats_scaler,
    strength,
    np_rng,
    py_rng,
):
    if attack_name is None or attack_name.lower() == "clean":
        return clone_data(data)
    if float(strength) <= 0.0:
        # A true zero-strength attack must be an exact no-op.
        return clone_data(data)

    name = attack_name.lower()
    if name == "padding":
        return attack_packet_length_padding(
            data, node_scaler, stats_scaler, strength, np_rng, py_rng, False
        )
    if name == "iat_jitter":
        return attack_iat_jitter(
            data, node_scaler, stats_scaler, strength, np_rng, py_rng, False
        )
    if name == "dummy":
        return attack_dummy_injection(
            data, node_scaler, stats_scaler, strength, np_rng, py_rng, False
        )
    if name == "mixed":
        return attack_mixed(
            data, node_scaler, stats_scaler, strength, np_rng, py_rng, False
        )
    if name == "adaptive":
        # Preserve the existing V14 threat model: semantic anchors are held fixed.
        return attack_mixed(
            data, node_scaler, stats_scaler, strength, np_rng, py_rng, True
        )
    raise ValueError(f"Unsupported attack: {name}")


def build_attacked_dataset_deterministic(
    dataset,
    attack_name,
    node_scaler,
    stats_scaler,
    strength,
    attack_seed,
):
    np_rng = np.random.default_rng(attack_seed)
    py_rng = random.Random(attack_seed)
    return [
        apply_attack_deterministic(
            d,
            attack_name,
            node_scaler,
            stats_scaler,
            strength,
            np_rng,
            py_rng,
        )
        for d in dataset
    ]


def validate_paired_dataset(clean_set, attacked_set, name):
    if len(clean_set) != len(attacked_set):
        raise AssertionError(f"{name}: attacked dataset length differs from clean test set")
    for c, a in zip(clean_set, attacked_set):
        if int(c.y.item()) != int(a.y.item()):
            raise AssertionError(f"{name}: label order changed")
        if int(c.sample_index.item()) != int(a.sample_index.item()):
            raise AssertionError(f"{name}: sample order changed")


def validate_adaptive_semantics(clean_set, attacked_set, atol=1e-7):
    for c, a in zip(clean_set, attacked_set):
        if not torch.allclose(c.finger_attr, a.finger_attr, atol=atol, rtol=0):
            raise AssertionError("Adaptive attack changed finger_attr under the V14 preserved-semantics rule")
        if not torch.allclose(c.entropy_attr, a.entropy_attr, atol=atol, rtol=0):
            raise AssertionError("Adaptive attack changed entropy_attr under the V14 preserved-semantics rule")


def validate_reconstruction_identity_fast(
    dataset, node_scaler, stats_scaler, logger: Logger, max_samples=16
):
    """
    Fast Phase-3 sanity check.

    Inverse -> rebuild -> normalize WITHOUT perturbation should reproduce the
    clean model inputs up to small floating-point error. This catches feature
    definition drift before the expensive training stage.
    """
    n = min(len(dataset), max_samples)
    max_x_diff = 0.0
    max_stats_diff = 0.0
    max_edge_diff = 0.0

    for data in dataset[:n]:
        raw_x, _ = inverse_raw_features(data, node_scaler, stats_scaler)
        rebuilt = finalize_attacked_data(
            data,
            raw_x,
            node_scaler,
            stats_scaler,
            data.finger_attr.cpu().numpy()[0],
            data.entropy_attr.cpu().numpy()[0],
        )

        if not torch.equal(data.edge_index, rebuilt.edge_index):
            raise AssertionError("Reconstruction identity failed: edge_index changed")

        max_x_diff = max(
            max_x_diff, float(torch.max(torch.abs(data.x - rebuilt.x)).item())
        )
        max_stats_diff = max(
            max_stats_diff,
            float(torch.max(torch.abs(data.stats_attr - rebuilt.stats_attr)).item()),
        )
        max_edge_diff = max(
            max_edge_diff,
            float(torch.max(torch.abs(data.edge_attr - rebuilt.edge_attr)).item()),
        )

    # StandardScaler round trips through float32 can introduce tiny numerical noise.
    if max_x_diff > 1e-3 or max_stats_diff > 1e-3 or max_edge_diff > 1e-4:
        raise AssertionError(
            "Reconstruction identity failed: "
            f"max_x_diff={max_x_diff:.3e}, "
            f"max_stats_diff={max_stats_diff:.3e}, "
            f"max_edge_diff={max_edge_diff:.3e}"
        )

    logger.log(
        "[Reconstruction identity] PASS | "
        f"n={n} | max_x_diff={max_x_diff:.3e} | "
        f"max_stats_diff={max_stats_diff:.3e} | "
        f"max_edge_diff={max_edge_diff:.3e}"
    )


def precompute_attack_cache(test_set, node_scaler, stats_scaler, logger: Logger):
    logger.log("\n" + "=" * 96)
    logger.log("Precomputing ONE canonical attacked test set per attack/strength")
    logger.log("These test attacks are evaluation-only and are never used for optimization.")
    logger.log("=" * 96)

    cache = {}
    specs = []
    for attack in MULTI_ATTACKS:
        for strength in MULTI_STRENGTHS:
            specs.append((attack, strength))
    for strength in ADAPTIVE_STRENGTHS:
        specs.append(("adaptive", strength))

    for attack, strength in specs:
        seed = stable_seed(ATTACK_BASE_SEED, attack, f"{strength:.3f}")
        attacked = build_attacked_dataset_deterministic(
            test_set,
            attack,
            node_scaler,
            stats_scaler,
            strength,
            seed,
        )
        validate_paired_dataset(test_set, attacked, f"{attack}@{strength}")
        if attack == "adaptive":
            validate_adaptive_semantics(test_set, attacked)
        cache[(attack, float(strength))] = attacked
        logger.log(
            f"[Attack cache] {attack:10s} strength={strength:.1f} "
            f"seed={seed} n={len(attacked)}"
        )

    logger.log("Attack cache validation passed: paired ordering/labels preserved.")
    logger.log("Adaptive cache validation passed: finger_attr and entropy_attr unchanged.")
    return cache


# =========================================================
# 5. TRAINING - paired protocol across model variants
# =========================================================

def make_loader(dataset, batch_size, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )


def compute_supervised_loss(model, dataset, batch_size=64):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model.forward_classifier(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
                batch.stats_attr,
                batch.finger_attr,
                batch.entropy_attr,
            )
            loss = criterion(logits, batch.y.view(-1))
            n = batch.y.numel()
            total_loss += loss.item() * n
            total_n += n
    return total_loss / max(total_n, 1)


def train_experiment(
    train_set,
    val_set,
    num_classes,
    logger: Logger,
    node_scaler,
    stats_scaler,
    tag="FULL",
    gate_mode="learned",
    train_seed=42,
    initial_state=None,
    run_pretrain=True,
    use_contrastive=True,
    use_robust_ft=True,
    attack_ce_weight=ATTACK_CE_WEIGHT,
    consistency_kl_weight=CONSISTENCY_KL_WEIGHT,
):
    logger.log("\n" + "-" * 96)
    logger.log(
        f"TRAIN {tag} | seed={train_seed} | gate_mode={gate_mode} | "
        f"contrastive={use_contrastive} | robust_ft={use_robust_ft} | "
        f"pretrain={'fresh' if run_pretrain and use_contrastive else ('reused' if initial_state is not None else 'skipped')}"
    )
    logger.log("-" * 96)

    # V16 FINAL keeps the validated learned-gate architecture unchanged.
    set_seed(train_seed)
    model = ContrastiveGNN_V13(
        node_in_channels=5,
        stats_dim=9,
        finger_seq_len=20,
        entropy_seq_len=5,
        hidden_channels=32,
        out_channels=num_classes,
        gate_mode=gate_mode,
    ).to(DEVICE)

    if initial_state is not None:
        model.load_state_dict(initial_state, strict=True)

    # Stage 1 remains identical to the validated Phase-4 FULL when enabled.
    if use_contrastive and run_pretrain:
        set_seed(train_seed + 10_000)
        train_loader_pre = make_loader(
            train_set, BATCH_SIZE, shuffle=True, seed=train_seed + 11_000
        )
        optimizer_pre = torch.optim.AdamW(
            model.parameters(), lr=PRETRAIN_LR, weight_decay=WEIGHT_DECAY
        )
        logger.log(
            f"[{tag}][seed={train_seed}] Stage 1 contrastive pretraining: "
            f"epochs={PRETRAIN_EPOCHS}"
        )

        for epoch in range(PRETRAIN_EPOCHS):
            model.train()
            total_loss = 0.0
            batches = 0
            for batch_data in train_loader_pre:
                data_list = batch_data.to_data_list()
                view1_list, view2_list = [], []
                for item in data_list:
                    item = item.to(DEVICE)
                    x1, ei1, ea1, _ = augment_graph_view(item)
                    x2, ei2, ea2, _ = augment_graph_view(item)
                    view1_list.append(Data(x=x1, edge_index=ei1, edge_attr=ea1))
                    view2_list.append(Data(x=x2, edge_index=ei2, edge_attr=ea2))

                b1 = Batch.from_data_list(view1_list).to(DEVICE)
                b2 = Batch.from_data_list(view2_list).to(DEVICE)

                optimizer_pre.zero_grad()
                z1 = model.forward_contrastive(
                    b1.x, b1.edge_index, b1.edge_attr, b1.batch
                )
                z2 = model.forward_contrastive(
                    b2.x, b2.edge_index, b2.edge_attr, b2.batch
                )
                loss = info_nce_loss(
                    z1, z2, temperature=CONTRASTIVE_TEMPERATURE
                )
                loss.backward()
                optimizer_pre.step()
                total_loss += loss.item()
                batches += 1

            if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == PRETRAIN_EPOCHS:
                logger.log(
                    f"[{tag}][seed={train_seed}] Contrastive epoch {epoch+1:02d}/"
                    f"{PRETRAIN_EPOCHS} | loss={total_loss / max(batches, 1):.6f}"
                )
    elif use_contrastive and initial_state is not None:
        logger.log(f"[{tag}][seed={train_seed}] Stage 1 reused from shared pretrained state.")
    else:
        logger.log(f"[{tag}][seed={train_seed}] Stage 1 contrastive pretraining skipped.")

    # Retained for checkpoint/debug provenance.
    pretrained_state = {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
    }

    # Stage 2: clean CE on every batch; deterministic robust augmentation on
    # approximately ROBUST_BATCH_RATIO of training batches only.
    set_seed(train_seed + 20_000)
    train_loader_fine = make_loader(
        train_set, BATCH_SIZE, shuffle=True, seed=train_seed + 21_000
    )
    optimizer_fine = torch.optim.AdamW(
        model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    no_improve = 0

    logger.log(
        f"[{tag}][seed={train_seed}] Stage 2 supervised fine-tuning: "
        f"max_epochs={FINETUNE_EPOCHS}, early_stop_patience={EARLY_STOP_PATIENCE}, "
        f"robust_ft={use_robust_ft}, robust_batch_ratio={ROBUST_BATCH_RATIO:.2f}, "
        f"mixed_strength=[{ROBUST_STRENGTH_MIN:.1f},{ROBUST_STRENGTH_MAX:.1f}], "
        f"attack_ce_w={attack_ce_weight}, kl_w={consistency_kl_weight}"
    )

    for epoch in range(FINETUNE_EPOCHS):
        model.train()
        total_loss = 0.0
        total_clean_ce = 0.0
        total_attack_ce = 0.0
        total_kl = 0.0
        batches = 0
        robust_batches = 0

        for batch_idx, batch_cpu in enumerate(train_loader_fine):
            # Decide robust-batch membership deterministically from seed/epoch/index.
            selector = stable_seed(
                "phase4_robust_batch", train_seed, epoch, batch_idx, modulo=1_000_000
            ) / 1_000_000.0
            use_robust_batch = use_robust_ft and (selector < ROBUST_BATCH_RATIO)

            attacked_batch = None
            attack_strength = None
            if use_robust_batch:
                robust_seed = stable_seed(
                    "phase4_train_attack", train_seed, epoch, batch_idx
                )
                strength_rng = np.random.default_rng(
                    stable_seed("phase4_strength", train_seed, epoch, batch_idx)
                )
                attack_strength = float(
                    strength_rng.uniform(ROBUST_STRENGTH_MIN, ROBUST_STRENGTH_MAX)
                )
                np_rng = np.random.default_rng(robust_seed)
                py_rng = random.Random(robust_seed)
                attacked_list = [
                    apply_attack_deterministic(
                        d,
                        "mixed",
                        node_scaler,
                        stats_scaler,
                        attack_strength,
                        np_rng,
                        py_rng,
                    )
                    for d in batch_cpu.to_data_list()
                ]
                attacked_batch = Batch.from_data_list(attacked_list).to(DEVICE)

            batch = batch_cpu.to(DEVICE)
            optimizer_fine.zero_grad()
            logits = model.forward_classifier(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
                batch.stats_attr,
                batch.finger_attr,
                batch.entropy_attr,
            )
            clean_ce = criterion(logits, batch.y.view(-1))
            loss = clean_ce
            attack_ce_value = 0.0
            kl_value = 0.0

            if attacked_batch is not None:
                attacked_logits = model.forward_classifier(
                    attacked_batch.x,
                    attacked_batch.edge_index,
                    attacked_batch.edge_attr,
                    attacked_batch.batch,
                    attacked_batch.stats_attr,
                    attacked_batch.finger_attr,
                    attacked_batch.entropy_attr,
                )
                attack_ce = criterion(
                    attacked_logits, attacked_batch.y.view(-1)
                )
                consistency_kl = F.kl_div(
                    F.log_softmax(attacked_logits, dim=1),
                    F.softmax(logits.detach(), dim=1),
                    reduction="batchmean",
                )
                loss = (
                    clean_ce
                    + attack_ce_weight * attack_ce
                    + consistency_kl_weight * consistency_kl
                )
                attack_ce_value = float(attack_ce.item())
                kl_value = float(consistency_kl.item())
                robust_batches += 1

            loss.backward()
            optimizer_fine.step()
            total_loss += loss.item()
            total_clean_ce += clean_ce.item()
            total_attack_ce += attack_ce_value
            total_kl += kl_value
            batches += 1

        val_loss = compute_supervised_loss(model, val_set, batch_size=BATCH_SIZE)
        train_loss = total_loss / max(batches, 1)

        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 0 or (epoch + 1) % 5 == 0 or no_improve == 0:
            logger.log(
                f"[{tag}][seed={train_seed}] Fine epoch {epoch+1:02d}/"
                f"{FINETUNE_EPOCHS} | train_loss={train_loss:.6f} | "
                f"clean_ce={total_clean_ce / max(batches, 1):.6f} | "
                f"attack_ce={total_attack_ce / max(robust_batches, 1):.6f} | "
                f"kl={total_kl / max(robust_batches, 1):.6f} | "
                f"robust_batches={robust_batches}/{batches} | "
                f"val_loss={val_loss:.6f} | best_epoch={best_epoch}"
            )

        if no_improve >= EARLY_STOP_PATIENCE:
            logger.log(
                f"[{tag}][seed={train_seed}] Early stopping at epoch {epoch+1}; "
                f"best epoch={best_epoch}, best val loss={best_val_loss:.6f}"
            )
            break

    if best_state is None:
        raise RuntimeError(f"{tag} seed={train_seed}: no valid checkpoint was produced")
    model.load_state_dict(best_state)

    if SAVE_CHECKPOINTS:
        ckpt_path = CHECKPOINT_DIR / f"{tag}_seed{train_seed}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "train_seed": train_seed,
            "tag": tag,
            "gate_mode": gate_mode,
            "robust_finetuning": use_robust_ft,
            "weight_decay": WEIGHT_DECAY,
            "robust_batch_ratio": ROBUST_BATCH_RATIO,
            "robust_strength_range": [ROBUST_STRENGTH_MIN, ROBUST_STRENGTH_MAX],
            "use_contrastive": use_contrastive,
            "use_robust_ft": use_robust_ft,
            "attack_ce_weight": attack_ce_weight,
            "consistency_kl_weight": consistency_kl_weight,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        }, ckpt_path)
        logger.log(f"[{tag}][seed={train_seed}] checkpoint saved: {ckpt_path}")

    return {
        "model": model,
        "tag": tag,
        "train_seed": train_seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "pretrained_state": pretrained_state,
    }


# =========================================================
# 6. EVALUATION METRICS
# =========================================================

def predict_closed_set_with_probs(model, dataset, batch_size=64):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model.forward_classifier(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
                batch.stats_attr,
                batch.finger_attr,
                batch.entropy_attr,
            )
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.view(-1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        np.asarray(all_preds, dtype=np.int64),
        np.asarray(all_labels, dtype=np.int64),
        np.asarray(all_probs, dtype=np.float64),
    )


def expected_calibration_error(labels, probs, n_bins=15):
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correctness = (predictions == labels).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences > lo) & (confidences <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        acc_bin = correctness[mask].mean()
        conf_bin = confidences[mask].mean()
        ece += (count / max(n, 1)) * abs(acc_bin - conf_bin)
    return float(ece * 100.0)


def evaluate_clean(model, test_set):
    preds, labels, probs = predict_closed_set_with_probs(model, test_set, BATCH_SIZE)
    return {
        "acc": accuracy_score(labels, preds) * 100.0,
        "macro_f1": f1_score(labels, preds, average="macro") * 100.0,
        "ece": expected_calibration_error(labels, probs, ECE_BINS),
        "preds": preds,
        "labels": labels,
        "probs": probs,
    }


def evaluate_attack(model, attacked_set, clean_eval):
    atk_preds, atk_labels, atk_probs = predict_closed_set_with_probs(
        model, attacked_set, BATCH_SIZE
    )

    robust_acc = accuracy_score(atk_labels, atk_preds) * 100.0
    robust_f1 = f1_score(atk_labels, atk_preds, average="macro") * 100.0
    clean_preds = clean_eval["preds"]
    clean_labels = clean_eval["labels"]

    if not np.array_equal(clean_labels, atk_labels):
        raise AssertionError("Attack evaluation label order differs from clean test order")

    clean_correct = (clean_preds == clean_labels)
    attack_success = np.sum(clean_correct & (atk_preds != atk_labels))
    asr = attack_success / max(np.sum(clean_correct), 1) * 100.0
    rel_drop = (
        (clean_eval["acc"] - robust_acc) / max(clean_eval["acc"], 1e-12) * 100.0
    )
    abs_drop = clean_eval["acc"] - robust_acc
    ece = expected_calibration_error(atk_labels, atk_probs, ECE_BINS)

    return {
        "robust_acc": robust_acc,
        "robust_f1": robust_f1,
        "asr": asr,
        "rel_drop": rel_drop,
        "abs_drop_pp": abs_drop,
        "ece": ece,
    }


def metric_summary(records: List[dict], key: str):
    vals = np.asarray([r[key] for r in records], dtype=np.float64)
    mean = float(vals.mean())
    sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return mean, sd


def fmt_mean_sd(records: List[dict], key: str, digits=2):
    mean, sd = metric_summary(records, key)
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def log_clean_result(logger, tag, seed, result):
    logger.log(
        f"[RESULT][{tag}][seed={seed}] CLEAN | "
        f"Acc={result['acc']:.2f}% | MacroF1={result['macro_f1']:.2f}% | "
        f"ECE={result['ece']:.2f}%"
    )


def log_attack_result(logger, tag, seed, attack, strength, result):
    logger.log(
        f"[RESULT][{tag}][seed={seed}] {attack.upper()}@{strength:.1f} | "
        f"RobAcc={result['robust_acc']:.2f}% | RobF1={result['robust_f1']:.2f}% | "
        f"ASR={result['asr']:.2f}% | RelDrop={result['rel_drop']:.2f}% | "
        f"AbsDrop={result['abs_drop_pp']:.2f}pp | ECE={result['ece']:.2f}%"
    )


def evaluate_primary_variant(model, tag, seed, test_set, attack_cache, logger):
    """Evaluate one ablation only on Clean + primary Mixed + primary Adaptive."""
    clean = evaluate_clean(model, test_set)
    mixed = evaluate_attack(
        model,
        attack_cache[("mixed", float(PRIMARY_MIXED_STRENGTH))],
        clean,
    )
    adapt = evaluate_attack(
        model,
        attack_cache[("adaptive", float(PRIMARY_ADAPTIVE_STRENGTH))],
        clean,
    )
    log_clean_result(logger, tag, seed, clean)
    log_attack_result(logger, tag, seed, "mixed", PRIMARY_MIXED_STRENGTH, mixed)
    log_attack_result(logger, tag, seed, "adaptive", PRIMARY_ADAPTIVE_STRENGTH, adapt)
    return {
        "clean_acc": clean["acc"],
        "clean_f1": clean["macro_f1"],
        "mixed_robust_acc": mixed["robust_acc"],
        "mixed_robust_f1": mixed["robust_f1"],
        "mixed_asr": mixed["asr"],
        "mixed_rel_drop": mixed["rel_drop"],
        "mixed_ece": mixed["ece"],
        "adaptive_robust_acc": adapt["robust_acc"],
        "adaptive_robust_f1": adapt["robust_f1"],
        "adaptive_asr": adapt["asr"],
        "adaptive_rel_drop": adapt["rel_drop"],
        "adaptive_ece": adapt["ece"],
    }


# =========================================================
# 7. REPRESENTATION / DEPLOYMENT AUXILIARY EXPERIMENTS
# =========================================================

def collect_embeddings(model, dataset, indices):
    subset = [dataset[i] for i in indices]
    loader = DataLoader(subset, batch_size=64, shuffle=False, num_workers=0)
    model.eval()
    all_feats, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            feats = model.extract_fused_features(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
                batch.stats_attr,
                batch.finger_attr,
                batch.entropy_attr,
            )
            all_feats.extend(feats.cpu().numpy())
            all_labels.extend(batch.y.view(-1).cpu().numpy())

    return np.asarray(all_feats, dtype=np.float32), np.asarray(all_labels, dtype=np.int64)


def run_representation_visualization(model, clean_set, attacked_set, logger, seed):
    if not RUN_TSNE:
        return None

    rng = np.random.default_rng(stable_seed("tsne_subset", seed))
    n = min(len(clean_set), TSNE_MAX_POINTS)
    if n < len(clean_set):
        indices = np.sort(rng.choice(len(clean_set), n, replace=False))
    else:
        indices = np.arange(len(clean_set))

    clean_feats, clean_labels = collect_embeddings(model, clean_set, indices)
    atk_feats, atk_labels = collect_embeddings(model, attacked_set, indices)

    X = np.vstack([clean_feats, atk_feats])
    y = np.concatenate([clean_labels, atk_labels])
    domain = np.array(["clean"] * len(clean_feats) + ["attack"] * len(atk_feats))

    logger.log(f"[t-SNE] representative FULL seed={seed}, paired n={n} clean/attack samples")
    tsne = TSNE(
        n_components=2,
        random_state=SPLIT_SEED,
        perplexity=min(30, max(5, (len(X) - 1) // 3)),
        init="pca",
    )
    Z = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    unique_classes = np.unique(y)
    cmap = plt.cm.get_cmap("tab20", len(unique_classes))
    for i, cls in enumerate(unique_classes):
        mask_clean = (y == cls) & (domain == "clean")
        mask_atk = (y == cls) & (domain == "attack")
        plt.scatter(
            Z[mask_clean, 0], Z[mask_clean, 1],
            s=18, alpha=0.65, color=cmap(i), marker="o"
        )
        plt.scatter(
            Z[mask_atk, 0], Z[mask_atk, 1],
            s=24, alpha=0.85, color=cmap(i), marker="x"
        )

    plt.title(
        f"SMGF-Net FULL seed={seed}: Clean vs Mixed@{PRIMARY_MIXED_STRENGTH:.1f}"
    )
    plt.tight_layout()
    fig_path = FIGURE_DIR / f"tsne_full_seed{seed}_mixed_{PRIMARY_MIXED_STRENGTH:.1f}.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.log(f"[t-SNE] saved: {fig_path}")
    return str(fig_path)


def run_deployment_overhead(model, test_set, logger, seed):
    if not RUN_MODEL_ONLY_LATENCY:
        return None

    dataset = test_set[:min(len(test_set), LATENCY_MAX_SAMPLES)]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 50:
                break
            batch = batch.to(DEVICE)
            _ = model.forward_classifier(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
                batch.stats_attr,
                batch.finger_attr,
                batch.entropy_attr,
            )

    torch.cuda.synchronize()
    total_t = 0.0
    sample_count = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model.forward_classifier(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
                batch.stats_attr,
                batch.finger_attr,
                batch.entropy_attr,
            )
            torch.cuda.synchronize()
            total_t += time.perf_counter() - t0
            sample_count += batch.y.size(0)

    avg_latency_ms = total_t / max(sample_count, 1) * 1000.0
    throughput = sample_count / max(total_t, 1e-12)
    params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    result = {
        "seed": seed,
        "samples": sample_count,
        "model_only_latency_ms": avg_latency_ms,
        "model_only_throughput_sps": throughput,
        "trainable_parameters": params_count,
    }
    logger.log(
        f"[DEPLOYMENT][FULL][seed={seed}] MODEL-ONLY | n={sample_count} | "
        f"latency={avg_latency_ms:.4f} ms/sample | throughput={throughput:.2f} samples/s | "
        f"params={params_count:,}"
    )
    return result


def summarize_gate_values(model, dataset, max_samples=1000):
    """Summarize learned gate activations over a bounded evaluation subset."""
    if model.fusion_layer.gate_mode != "learned":
        raise ValueError("Gate diagnostics require gate_mode='learned'")

    subset = dataset[:min(len(dataset), max_samples)]
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    values = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            h_finger = model.finger_cnn(batch.finger_attr)
            h_entropy = model.entropy_cnn(batch.entropy_attr)
            h_ctx = torch.cat(
                [batch.stats_attr, h_finger, h_entropy], dim=1
            )
            gate = model.fusion_layer.global_gate(h_ctx)
            values.append(gate.detach().cpu().numpy().reshape(-1))

    arr = np.concatenate(values) if values else np.asarray([], dtype=np.float32)
    if arr.size == 0:
        raise RuntimeError("No gate activations collected")
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "sat_low_pct": float(np.mean(arr < 0.10) * 100.0),
        "sat_high_pct": float(np.mean(arr > 0.90) * 100.0),
        "n_values": int(arr.size),
    }


def run_gate_diagnostics(model, clean_set, attack_cache, logger, seed):
    """
    Figure for the paper's gating mechanism claim.
    Reports activation distributions instead of assuming a monotonic effect.
    """
    if not RUN_GATE_DIAGNOSTICS:
        return None, None

    records = []
    clean_stats = summarize_gate_values(model, clean_set, GATE_DIAG_MAX_SAMPLES)
    records.append({"attack": "clean", "strength": 0.0, **clean_stats})

    for attack, strengths in [
        ("mixed", MULTI_STRENGTHS),
        ("adaptive", ADAPTIVE_STRENGTHS),
    ]:
        for strength in strengths:
            stats = summarize_gate_values(
                model,
                attack_cache[(attack, float(strength))],
                GATE_DIAG_MAX_SAMPLES,
            )
            records.append({"attack": attack, "strength": float(strength), **stats})
            logger.log(
                f"[GATE][seed={seed}] {attack}@{strength:.1f} | "
                f"mean={stats['mean']:.4f} std={stats['std']:.4f} "
                f"p10={stats['p10']:.4f} p50={stats['p50']:.4f} p90={stats['p90']:.4f} "
                f"sat<0.1={stats['sat_low_pct']:.2f}% sat>0.9={stats['sat_high_pct']:.2f}%"
            )

    json_path = FIGURE_DIR / f"gate_diagnostics_seed{seed}.json"
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    plt.figure(figsize=(8, 6))
    for attack in ["mixed", "adaptive"]:
        xs = [0.0]
        ys = [clean_stats["mean"]]
        yerr = [clean_stats["std"]]
        for rec in records:
            if rec["attack"] == attack:
                xs.append(rec["strength"])
                ys.append(rec["mean"])
                yerr.append(rec["std"])
        plt.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=attack)
    plt.xlabel("Attack strength")
    plt.ylabel("Mean gate activation")
    plt.title(f"Gate activation sensitivity (FULL seed={seed})")
    plt.legend()
    plt.tight_layout()
    fig_path = FIGURE_DIR / f"gate_activation_seed{seed}.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.log(f"[GATE] saved: {fig_path}")
    return str(fig_path), str(json_path)


def run_sensitivity_figure(full_attack_records, logger):
    """Aggregate three-seed robustness trajectories for the paper sensitivity figure."""
    if not RUN_SENSITIVITY_FIGURE:
        return None

    plt.figure(figsize=(8, 6))
    for attack, strengths in [
        ("mixed", MULTI_STRENGTHS),
        ("adaptive", ADAPTIVE_STRENGTHS),
    ]:
        means, sds = [], []
        for strength in strengths:
            mean, sd = metric_summary(
                full_attack_records[(attack, float(strength))], "robust_acc"
            )
            means.append(mean)
            sds.append(sd)
        plt.errorbar(
            strengths, means, yerr=sds, marker="o", capsize=3, label=attack
        )

    plt.xlabel("Attack strength")
    plt.ylabel("Robust accuracy (%)")
    plt.title("SMGF-Net robustness sensitivity")
    plt.legend()
    plt.tight_layout()
    fig_path = FIGURE_DIR / "sensitivity_mixed_adaptive.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.log(f"[SENSITIVITY] saved: {fig_path}")
    return str(fig_path)


# =========================================================
# 8. MARKDOWN REPORT
# =========================================================

def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def write_results_markdown(
    dataset_info,
    full_clean_records,
    full_attack_records,
    ablation_records,
    deployment_result,
    tsne_path,
    sensitivity_path,
    gate_figure_path,
    gate_json_path,
):
    lines = []
    is_ustc = "ustc" in DATASET_NAME.lower()
    mixed_table_no = "IV" if is_ustc else "III"
    adaptive_table_no = "VI" if is_ustc else "V"
    lines.append("# SMGF-Net V16 Final Paper Experiment Results")
    lines.append("")
    lines.append("> Generated automatically by `V16-SMGF-Net_final_paper_experiments.py`.")
    lines.append("> All reported aggregate values are mean ± SD across the configured training seeds.")
    lines.append("")

    lines.append("## 1. Experimental protocol")
    lines.append("")
    lines.append(f"- Dataset label: **{DATASET_NAME}**")
    lines.append(f"- Dataset path: `{PCAP_DIR}`")
    lines.append(f"- Dataset manifest SHA-256: `{dataset_info['manifest_hash']}`")
    lines.append(f"- Classes: {dataset_info['num_classes']}")
    lines.append(f"- Valid samples: {dataset_info['num_samples']}")
    lines.append(
        f"- Canonical split: Train={dataset_info['train_n']}, Validation={dataset_info['val_n']}, "
        f"Test={dataset_info['test_n']}"
    )
    lines.append(f"- Split seed: {SPLIT_SEED}")
    lines.append(f"- Training seeds: {TRAIN_SEEDS}")
    lines.append(f"- ECE bins: {ECE_BINS}")
    lines.append(
        f"- Primary robustness endpoints: Mixed={PRIMARY_MIXED_STRENGTH:.1f}, "
        f"Adaptive={PRIMARY_ADAPTIVE_STRENGTH:.1f}"
    )
    lines.append("- Scalers: fitted once on Train only, reused by all models and attacks.")
    lines.append("- Attacked test sets: generated once with deterministic attack seeds and reused by all models.")
    lines.append("- FINAL/FULL: corrected pipeline + learned gate + contrastive pretraining + robust fine-tuning.")
    lines.append(
        f"- Robust fine-tuning: {ROBUST_BATCH_RATIO:.0%} of train batches receive paired Mixed attacks "
        f"with strength in [{ROBUST_STRENGTH_MIN:.1f}, {ROBUST_STRENGTH_MAX:.1f}]."
    )
    lines.append(
        f"- Loss: clean CE + {ATTACK_CE_WEIGHT:.1f} * attacked CE + "
        f"{CONSISTENCY_KL_WEIGHT:.1f} * clean-to-attacked KL on robust batches."
    )
    lines.append("- Training attacks are generated only from the training split; test attacks are evaluation-only.")
    lines.append(f"- Fine-tuning AdamW uses explicit weight_decay={WEIGHT_DECAY}.")
    lines.append("")

    lines.append("## 2. Clean classification performance (paper Table II style)")
    lines.append("")
    clean_rows = [[
        "SMGF-Net (FULL)",
        fmt_mean_sd(full_clean_records, "acc"),
        fmt_mean_sd(full_clean_records, "macro_f1"),
    ]]
    lines.append(markdown_table(
        ["Method", "Accuracy (%)", "Macro-F1 (%)"], clean_rows
    ))
    lines.append("")

    lines.append("## 3. Multiple perturbation robustness")
    lines.append("")
    multi_rows = []
    for attack in MULTI_ATTACKS:
        for strength in MULTI_STRENGTHS:
            recs = full_attack_records[(attack, float(strength))]
            multi_rows.append([
                attack,
                f"{strength:.1f}",
                fmt_mean_sd(recs, "robust_acc"),
                fmt_mean_sd(recs, "robust_f1"),
                fmt_mean_sd(recs, "asr"),
                fmt_mean_sd(recs, "rel_drop"),
                fmt_mean_sd(recs, "ece"),
            ])
    lines.append(markdown_table(
        ["Attack", "Strength", "Rob_Acc (%)", "Rob_F1 (%)", "ASR (%)", "Rel_Drop (%)", "ECE (%)"],
        multi_rows,
    ))
    lines.append("")

    lines.append(
        f"### Primary Mixed endpoint (paper Table {mixed_table_no} row, strength={PRIMARY_MIXED_STRENGTH:.1f})"
    )
    lines.append("")
    mixed_recs = full_attack_records[("mixed", float(PRIMARY_MIXED_STRENGTH))]
    lines.append(markdown_table(
        ["Method", "Rob_Acc (%)", "Rob_F1 (%)", "ASR (%)", "Rel_Drop (%)", "ECE (%)"],
        [[
            "SMGF-Net (FULL)",
            fmt_mean_sd(mixed_recs, "robust_acc"),
            fmt_mean_sd(mixed_recs, "robust_f1"),
            fmt_mean_sd(mixed_recs, "asr"),
            fmt_mean_sd(mixed_recs, "rel_drop"),
            fmt_mean_sd(mixed_recs, "ece"),
        ]],
    ))
    lines.append("")

    lines.append("## 4. Adaptive attack robustness")
    lines.append("")
    adaptive_rows = []
    for strength in ADAPTIVE_STRENGTHS:
        recs = full_attack_records[("adaptive", float(strength))]
        adaptive_rows.append([
            f"{strength:.1f}",
            fmt_mean_sd(recs, "robust_acc"),
            fmt_mean_sd(recs, "robust_f1"),
            fmt_mean_sd(recs, "asr"),
            fmt_mean_sd(recs, "rel_drop"),
            fmt_mean_sd(recs, "ece"),
        ])
    lines.append(markdown_table(
        ["Strength", "Rob_Acc (%)", "Rob_F1 (%)", "ASR (%)", "Rel_Drop (%)", "ECE (%)"],
        adaptive_rows,
    ))
    lines.append("")

    lines.append(
        f"### Primary Adaptive endpoint (paper Table {adaptive_table_no} row, strength={PRIMARY_ADAPTIVE_STRENGTH:.1f})"
    )
    lines.append("")
    adapt_recs = full_attack_records[("adaptive", float(PRIMARY_ADAPTIVE_STRENGTH))]
    lines.append(markdown_table(
        ["Method", "Rob_Acc (%)", "Rob_F1 (%)", "ASR (%)", "Rel_Drop (%)", "ECE (%)"],
        [[
            "SMGF-Net (FULL)",
            fmt_mean_sd(adapt_recs, "robust_acc"),
            fmt_mean_sd(adapt_recs, "robust_f1"),
            fmt_mean_sd(adapt_recs, "asr"),
            fmt_mean_sd(adapt_recs, "rel_drop"),
            fmt_mean_sd(adapt_recs, "ece"),
        ]],
    ))
    lines.append("")

    lines.append("## 5. Ablation study (paper Table VIII style)")
    lines.append("")
    ab_rows = []
    for tag in ABLATION_VARIANTS:
        recs = ablation_records[tag]
        ab_rows.append([
            tag,
            fmt_mean_sd(recs, "clean_acc"),
            fmt_mean_sd(recs, "mixed_robust_acc"),
            fmt_mean_sd(recs, "mixed_rel_drop"),
            fmt_mean_sd(recs, "adaptive_robust_acc"),
            fmt_mean_sd(recs, "adaptive_rel_drop"),
        ])
    lines.append(markdown_table(
        [
            "Setting",
            "Clean_Acc (%)",
            "Mixed_Rob_Acc (%)",
            "Mixed_RelDrop (%)",
            "Adapt_Rob_Acc (%)",
            "Adapt_RelDrop (%)",
        ],
        ab_rows,
    ))
    lines.append("")
    lines.append(
        "STRICT_NO_GATE is a strict identity-gate ablation using the same GatedFusion module; "
        "it does not substitute the old SimpleFusion architecture."
    )
    lines.append("")

    lines.append("## 6. Protocol checks")
    lines.append("")
    lines.append("- Phase-3 corrected feature/attack pipeline is retained.")
    lines.append("- Reconstruction identity check is run before training.")
    lines.append("- Robust perturbations are generated from train batches only.")
    lines.append("- Canonical attacked test caches are generated once and never used for optimization.")
    lines.append("- FULL and strict ablations are evaluated on identical primary attack caches.")
    lines.append("- Contrastive-enabled ablations reuse the same per-seed pretrained state as FULL.")
    lines.append("- Fine-tuning explicitly uses the configured AdamW weight decay.")
    lines.append("")

    if deployment_result is not None:
        lines.append("## 7. Model-only deployment overhead")
        lines.append("")
        lines.append(
            "This timing covers the neural-network forward pass only. It does not include PCAP parsing, feature extraction, graph construction, or host-to-device transfer."
        )
        lines.append("")
        lines.append(markdown_table(
            ["Representative seed", "Samples", "Latency (ms/sample)", "Throughput (samples/s)", "Trainable params"],
            [[
                deployment_result["seed"],
                deployment_result["samples"],
                f"{deployment_result['model_only_latency_ms']:.4f}",
                f"{deployment_result['model_only_throughput_sps']:.2f}",
                f"{deployment_result['trainable_parameters']:,}",
            ]],
        ))
        lines.append("")

    if tsne_path:
        lines.append("## 8. Representation visualization")
        lines.append("")
        lines.append(f"- Paired t-SNE figure: `{tsne_path}`")
        lines.append("")

    if sensitivity_path or gate_figure_path:
        lines.append("## 9. Robustness and gate mechanism figures")
        lines.append("")
        if sensitivity_path:
            lines.append(f"- Robustness sensitivity figure: `{sensitivity_path}`")
        if gate_figure_path:
            lines.append(f"- Gate activation figure: `{gate_figure_path}`")
        if gate_json_path:
            lines.append(f"- Gate diagnostics data: `{gate_json_path}`")
        lines.append("")

    lines.append("## 10. Reproducibility artifacts")
    lines.append("")
    lines.append(f"- Main log: `{LOG_FILE}`")
    lines.append(f"- Split manifest: `{SPLIT_MANIFEST_FILE}`")
    lines.append(f"- Run configuration: `{CONFIG_FILE}`")
    if SAVE_CHECKPOINTS:
        lines.append(f"- Checkpoints: `{CHECKPOINT_DIR}`")
    lines.append("")

    RESULT_MD.write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# 9. CONFIG / CONSISTENCY HELPERS
# =========================================================

def save_run_config(dataset_info):
    payload = {
        "dataset_name": DATASET_NAME,
        "pcap_dir": PCAP_DIR,
        "result_root": RESULT_ROOT,
        "run_dir": str(RUN_DIR),
        "dataset": dataset_info,
        "split_seed": SPLIT_SEED,
        "attack_base_seed": ATTACK_BASE_SEED,
        "train_seeds": TRAIN_SEEDS,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "pretrain_epochs": PRETRAIN_EPOCHS,
            "finetune_epochs": FINETUNE_EPOCHS,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "pretrain_lr": PRETRAIN_LR,
            "finetune_lr": FINETUNE_LR,
            "weight_decay": WEIGHT_DECAY,
            "contrastive_temperature": CONTRASTIVE_TEMPERATURE,
        },
        "v16_final": {
            "model": "corrected pipeline + learned gate + contrastive pretraining + robust fine-tuning",
            "robust_batch_ratio": ROBUST_BATCH_RATIO,
            "robust_attack": "mixed",
            "robust_strength_min": ROBUST_STRENGTH_MIN,
            "robust_strength_max": ROBUST_STRENGTH_MAX,
            "attack_ce_weight": ATTACK_CE_WEIGHT,
            "consistency_kl_weight": CONSISTENCY_KL_WEIGHT,
            "training_attack_scope": "train split only",
            "phase3_baseline": PHASE3_BASELINE,
            "ablation_variants": ABLATION_VARIANTS,
            "run_tsne": RUN_TSNE,
            "run_gate_diagnostics": RUN_GATE_DIAGNOSTICS,
            "run_sensitivity_figure": RUN_SENSITIVITY_FIGURE,
            "run_model_only_latency": RUN_MODEL_ONLY_LATENCY,
            "pipeline_fixes": [
                "clean/attack statistics use the same <=400 packet observation window",
                "direction restored as discrete 0/1 after inverse scaling",
                "zero-strength attack is exact no-op",
                "attacked graph is truncated to the same 400-node observation window",
            ],
        },
        "evaluation": {
            "ece_bins": ECE_BINS,
            "multi_attacks": MULTI_ATTACKS,
            "multi_strengths": MULTI_STRENGTHS,
            "adaptive_strengths": ADAPTIVE_STRENGTHS,
            "primary_mixed_strength": PRIMARY_MIXED_STRENGTH,
            "primary_adaptive_strength": PRIMARY_ADAPTIVE_STRENGTH,
        },
        "versions": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "sklearn": sklearn.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    with CONFIG_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def assert_full_ablation_matches_main(full_clean_records, full_attack_records, ablation_records):
    """Hard fail if the FULL ablation row is not literally built from the main FULL evaluations."""
    if len(full_clean_records) != len(ablation_records["FULL"]):
        raise AssertionError("FULL ablation replicate count differs from main FULL replicate count")

    mixed_main = full_attack_records[("mixed", float(PRIMARY_MIXED_STRENGTH))]
    adapt_main = full_attack_records[("adaptive", float(PRIMARY_ADAPTIVE_STRENGTH))]

    for i, ab in enumerate(ablation_records["FULL"]):
        if not math.isclose(ab["clean_acc"], full_clean_records[i]["acc"], abs_tol=1e-12):
            raise AssertionError("FULL clean mismatch between main and ablation")
        if not math.isclose(ab["mixed_robust_acc"], mixed_main[i]["robust_acc"], abs_tol=1e-12):
            raise AssertionError("FULL Mixed mismatch between main and ablation")
        if not math.isclose(ab["adaptive_robust_acc"], adapt_main[i]["robust_acc"], abs_tol=1e-12):
            raise AssertionError("FULL Adaptive mismatch between main and ablation")


# =========================================================
# 10. MAIN: ONE INVOCATION -> ALL RESULTS
# =========================================================

def main():
    log_runtime_environment(LOGGER)

    if not os.path.exists(PCAP_DIR):
        raise FileNotFoundError(
            f"Dataset path does not exist: {PCAP_DIR}\n"
            "Set environment variable SMGF_PCAP_DIR to the correct dataset directory."
        )

    # -----------------------------------------------------
    # A. Load dataset in a stable order and create ONE split
    # -----------------------------------------------------
    dataset, num_classes, idx_to_class, metadata = load_dataset_closedset(PCAP_DIR, LOGGER)
    if len(dataset) == 0:
        raise RuntimeError("No valid samples were loaded.")

    manifest_hash = dataset_manifest_hash(metadata)
    LOGGER.log(f"Dataset manifest SHA-256: {manifest_hash}")

    train_idx, val_idx, test_idx = stratified_split_indices(
        dataset,
        split_seed=SPLIT_SEED,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
    )
    train_set = materialize_split(dataset, train_idx)
    val_set = materialize_split(dataset, val_idx)
    test_set = materialize_split(dataset, test_idx)

    LOGGER.log(
        f"Canonical split -> Train: {len(train_set)} | Validation: {len(val_set)} | Test: {len(test_set)}"
    )
    save_split_manifest(metadata, train_idx, val_idx, test_idx, manifest_hash)

    # One scaler fit only.
    node_scaler, stats_scaler = fit_and_apply_scalers(
        train_set, val_set, test_set, LOGGER
    )
    validate_reconstruction_identity_fast(
        test_set, node_scaler, stats_scaler, LOGGER, max_samples=16
    )

    dataset_info = {
        "manifest_hash": manifest_hash,
        "num_classes": num_classes,
        "num_samples": len(dataset),
        "train_n": len(train_set),
        "val_n": len(val_set),
        "test_n": len(test_set),
    }
    save_run_config(dataset_info)

    # -----------------------------------------------------
    # B. Precompute ONE deterministic attack cache
    # -----------------------------------------------------
    attack_cache = precompute_attack_cache(
        test_set, node_scaler, stats_scaler, LOGGER
    )

    # Result containers across training seeds.
    full_clean_records = []
    full_attack_records = {
        (attack, float(strength)): []
        for attack in MULTI_ATTACKS
        for strength in MULTI_STRENGTHS
    }
    for strength in ADAPTIVE_STRENGTHS:
        full_attack_records[("adaptive", float(strength))] = []

    ablation_records = {tag: [] for tag in ABLATION_VARIANTS}

    deployment_result = None
    tsne_path = None
    gate_figure_path = None
    gate_json_path = None
    sensitivity_path = None

    # -----------------------------------------------------
    # C. Paired multi-seed training/evaluation
    # -----------------------------------------------------
    for seed_i, train_seed in enumerate(TRAIN_SEEDS):
        LOGGER.log("\n" + "#" * 96)
        LOGGER.log(f"TRAINING REPLICATE {seed_i + 1}/{len(TRAIN_SEEDS)} | seed={train_seed}")
        LOGGER.log("#" * 96)

        # C1. FINAL FULL: paper method.
        full_bundle = train_experiment(
            train_set,
            val_set,
            num_classes,
            LOGGER,
            node_scaler,
            stats_scaler,
            tag="FULL",
            gate_mode="learned",
            train_seed=train_seed,
            initial_state=None,
            run_pretrain=True,
            use_contrastive=True,
            use_robust_ft=True,
            attack_ce_weight=ATTACK_CE_WEIGHT,
            consistency_kl_weight=CONSISTENCY_KL_WEIGHT,
        )
        full_model = full_bundle["model"]
        shared_pretrained_state = full_bundle["pretrained_state"]

        full_clean = evaluate_clean(full_model, test_set)
        log_clean_result(LOGGER, "FULL", train_seed, full_clean)
        full_clean_records.append(full_clean)

        # Evaluate all main attack settings on the precomputed canonical attacked sets.
        per_seed_attack_results = {}
        for attack in MULTI_ATTACKS:
            for strength in MULTI_STRENGTHS:
                result = evaluate_attack(
                    full_model,
                    attack_cache[(attack, float(strength))],
                    full_clean,
                )
                full_attack_records[(attack, float(strength))].append(result)
                per_seed_attack_results[(attack, float(strength))] = result
                log_attack_result(LOGGER, "FULL", train_seed, attack, strength, result)

        for strength in ADAPTIVE_STRENGTHS:
            result = evaluate_attack(
                full_model,
                attack_cache[("adaptive", float(strength))],
                full_clean,
            )
            full_attack_records[("adaptive", float(strength))].append(result)
            per_seed_attack_results[("adaptive", float(strength))] = result
            log_attack_result(LOGGER, "FULL", train_seed, "adaptive", strength, result)

        # The FULL ablation row is REUSED, not retrained.
        primary_mixed = per_seed_attack_results[("mixed", float(PRIMARY_MIXED_STRENGTH))]
        primary_adapt = per_seed_attack_results[("adaptive", float(PRIMARY_ADAPTIVE_STRENGTH))]
        ablation_records["FULL"].append({
            "clean_acc": full_clean["acc"],
            "clean_f1": full_clean["macro_f1"],
            "mixed_robust_acc": primary_mixed["robust_acc"],
            "mixed_robust_f1": primary_mixed["robust_f1"],
            "mixed_asr": primary_mixed["asr"],
            "mixed_rel_drop": primary_mixed["rel_drop"],
            "mixed_ece": primary_mixed["ece"],
            "adaptive_robust_acc": primary_adapt["robust_acc"],
            "adaptive_robust_f1": primary_adapt["robust_f1"],
            "adaptive_asr": primary_adapt["asr"],
            "adaptive_rel_drop": primary_adapt["rel_drop"],
            "adaptive_ece": primary_adapt["ece"],
        })

        # Representative paper figures / deployment diagnostics from the first FULL seed.
        if seed_i == 0:
            tsne_path = run_representation_visualization(
                full_model,
                test_set,
                attack_cache[("mixed", float(PRIMARY_MIXED_STRENGTH))],
                LOGGER,
                train_seed,
            )
            deployment_result = run_deployment_overhead(
                full_model, test_set, LOGGER, train_seed
            )
            gate_figure_path, gate_json_path = run_gate_diagnostics(
                full_model, test_set, attack_cache, LOGGER, train_seed
            )

        # FULL fine-tuned model is no longer needed; keep its shared contrastive state on CPU.
        del full_model
        del full_bundle
        torch.cuda.empty_cache()

        # C2-C5. Strict paper ablations. Every contrastive ablation except
        # NO_CONTRASTIVE reuses the exact same pretrained state from FULL.
        variant_specs = [
            {
                "tag": "STRICT_NO_GATE",
                "gate_mode": "identity",
                "initial_state": shared_pretrained_state,
                "run_pretrain": False,
                "use_contrastive": True,
                "use_robust_ft": True,
                "attack_ce_weight": ATTACK_CE_WEIGHT,
                "consistency_kl_weight": CONSISTENCY_KL_WEIGHT,
            },
            {
                "tag": "NO_CONTRASTIVE",
                "gate_mode": "learned",
                "initial_state": None,
                "run_pretrain": False,
                "use_contrastive": False,
                "use_robust_ft": True,
                "attack_ce_weight": ATTACK_CE_WEIGHT,
                "consistency_kl_weight": CONSISTENCY_KL_WEIGHT,
            },
            {
                "tag": "NO_ROBUST_FT",
                "gate_mode": "learned",
                "initial_state": shared_pretrained_state,
                "run_pretrain": False,
                "use_contrastive": True,
                "use_robust_ft": False,
                "attack_ce_weight": 0.0,
                "consistency_kl_weight": 0.0,
            },
            {
                "tag": "NO_CONSISTENCY_KL",
                "gate_mode": "learned",
                "initial_state": shared_pretrained_state,
                "run_pretrain": False,
                "use_contrastive": True,
                "use_robust_ft": True,
                "attack_ce_weight": ATTACK_CE_WEIGHT,
                "consistency_kl_weight": 0.0,
            },
        ]

        for spec in variant_specs:
            tag = spec["tag"]
            bundle = train_experiment(
                train_set,
                val_set,
                num_classes,
                LOGGER,
                node_scaler,
                stats_scaler,
                tag=tag,
                gate_mode=spec["gate_mode"],
                train_seed=train_seed,
                initial_state=spec["initial_state"],
                run_pretrain=spec["run_pretrain"],
                use_contrastive=spec["use_contrastive"],
                use_robust_ft=spec["use_robust_ft"],
                attack_ce_weight=spec["attack_ce_weight"],
                consistency_kl_weight=spec["consistency_kl_weight"],
            )
            model = bundle["model"]
            ablation_records[tag].append(
                evaluate_primary_variant(
                    model, tag, train_seed, test_set, attack_cache, LOGGER
                )
            )
            del model
            del bundle
            torch.cuda.empty_cache()

        del shared_pretrained_state
        torch.cuda.empty_cache()

    sensitivity_path = run_sensitivity_figure(full_attack_records, LOGGER)

    # -----------------------------------------------------
    # D. Hard consistency checks and aggregate reporting
    # -----------------------------------------------------
    assert_full_ablation_matches_main(
        full_clean_records, full_attack_records, ablation_records
    )
    LOGGER.log("\n[CONSISTENCY CHECK] PASS")
    LOGGER.log("V16 FINAL FULL uses the corrected pipeline, learned gate, contrastive pretraining, and robust fine-tuning.")
    LOGGER.log("Training attacks are generated only from train-split batches; test cache remains evaluation-only.")
    LOGGER.log(
        f"FULL Mixed main/ablation values share the exact same Mixed@{PRIMARY_MIXED_STRENGTH:.1f} cache."
    )
    LOGGER.log(
        f"FULL Adaptive main/ablation values share the exact same Adaptive@{PRIMARY_ADAPTIVE_STRENGTH:.1f} cache."
    )

    # Aggregate summary in log.
    LOGGER.log("\n" + "=" * 96)
    LOGGER.log("AGGREGATED PRIMARY RESULTS (mean +/- SD across training seeds)")
    LOGGER.log("=" * 96)
    LOGGER.log(
        "FULL CLEAN | "
        f"Acc={fmt_mean_sd(full_clean_records, 'acc')}% | "
        f"MacroF1={fmt_mean_sd(full_clean_records, 'macro_f1')}% | "
        f"ECE={fmt_mean_sd(full_clean_records, 'ece')}%"
    )

    mixed_primary_records = full_attack_records[("mixed", float(PRIMARY_MIXED_STRENGTH))]
    adaptive_primary_records = full_attack_records[("adaptive", float(PRIMARY_ADAPTIVE_STRENGTH))]
    LOGGER.log(
        f"FULL MIXED@{PRIMARY_MIXED_STRENGTH:.1f} | "
        f"RobAcc={fmt_mean_sd(mixed_primary_records, 'robust_acc')}% | "
        f"RobF1={fmt_mean_sd(mixed_primary_records, 'robust_f1')}% | "
        f"ASR={fmt_mean_sd(mixed_primary_records, 'asr')}% | "
        f"RelDrop={fmt_mean_sd(mixed_primary_records, 'rel_drop')}% | "
        f"ECE={fmt_mean_sd(mixed_primary_records, 'ece')}%"
    )
    LOGGER.log(
        f"FULL ADAPTIVE@{PRIMARY_ADAPTIVE_STRENGTH:.1f} | "
        f"RobAcc={fmt_mean_sd(adaptive_primary_records, 'robust_acc')}% | "
        f"RobF1={fmt_mean_sd(adaptive_primary_records, 'robust_f1')}% | "
        f"ASR={fmt_mean_sd(adaptive_primary_records, 'asr')}% | "
        f"RelDrop={fmt_mean_sd(adaptive_primary_records, 'rel_drop')}% | "
        f"ECE={fmt_mean_sd(adaptive_primary_records, 'ece')}%"
    )

    for tag in ABLATION_VARIANTS:
        recs = ablation_records[tag]
        LOGGER.log(
            f"ABLATION {tag} | "
            f"Clean={fmt_mean_sd(recs, 'clean_acc')}% | "
            f"Mixed={fmt_mean_sd(recs, 'mixed_robust_acc')}% | "
            f"MixedDrop={fmt_mean_sd(recs, 'mixed_rel_drop')}% | "
            f"Adaptive={fmt_mean_sd(recs, 'adaptive_robust_acc')}% | "
            f"AdaptDrop={fmt_mean_sd(recs, 'adaptive_rel_drop')}%"
        )

    write_results_markdown(
        dataset_info,
        full_clean_records,
        full_attack_records,
        ablation_records,
        deployment_result,
        tsne_path,
        sensitivity_path,
        gate_figure_path,
        gate_json_path,
    )

    LOGGER.log("\n" + "=" * 96)
    LOGGER.log("ALL EXPERIMENTS COMPLETED")
    LOGGER.log(f"Main log      : {LOG_FILE}")
    LOGGER.log(f"Markdown      : {RESULT_MD}")
    LOGGER.log(f"Split manifest: {SPLIT_MANIFEST_FILE}")
    LOGGER.log(f"Run config    : {CONFIG_FILE}")
    if SAVE_CHECKPOINTS:
        LOGGER.log(f"Checkpoints   : {CHECKPOINT_DIR}")
    if RUN_TSNE:
        LOGGER.log(f"Figures       : {FIGURE_DIR}")
    LOGGER.log("=" * 96)


if __name__ == "__main__":
    main()
