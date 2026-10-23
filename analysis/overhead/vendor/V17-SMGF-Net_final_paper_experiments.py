# -*- coding: utf-8 -*-
"""
SMGF-Net V17 FINAL: reliability-aware gated fusion + perturbation-consistent training.

Paper method:
    learned context-conditioned Gate
  + perturbation-aware attacked-label CE
  + asymmetric clean-to-perturbed KL consistency

There is no representation-pretraining stage in this experiment.

Reviewer-proof protocol:
  * fixed, sorted dataset enumeration and manifest SHA-256
  * one stratified split shared by all models and seeds
  * train-only scaler fit once, then reused everywhere
  * corrected <=400-packet observation window inherited from validated V16 core
  * reconstruction-identity check before training
  * deterministic train-only perturbations
  * one canonical evaluation cache shared by every model and seed
  * train seeds 42/43/44
  * strict identity-gate ablation in the same GatedFusion module
  * component-isolating ablations for attacked CE and KL
  * clean validation only for early stopping
  * ECE is evaluation-only and never appears in the optimization objective
  * per-seed checkpoints, per-seed JSON, mean±SD reporting

Primary ablations:
  FINAL              : Gate + attacked CE + KL
  STRICT_NO_GATE     : identity Gate + attacked CE + KL
  NO_ATTACK_CE       : Gate + KL, attacked-label CE weight = 0
  NO_KL              : Gate + attacked CE, KL weight = 0
  CLEAN_ONLY         : Gate + clean CE only, no train perturbation
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


# =============================================================================
# 0. IMMUTABLE PAPER PROTOCOL
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
CORE_SOURCE = SCRIPT_DIR / "V16-SMGF-Net_final_paper_experiments.py"

EXPECTED_CORE_SHA256 = (
    "3ecaf0054a1a190b444fe6f3c19cf6203273ad32707041ec53b09318ca20a4e2"
)
EXPECTED_DATASET_MANIFEST_SHA256 = (
    "821bee04771f0c8dce3680a4301164d8ab48e03ec0d1ed57c3930015ad200a61"
)

DATASET_NAME = os.environ.get("SMGF_DATASET_NAME", "CipherSpectrum")
PCAP_DIR = os.environ.get(
    "SMGF_PCAP_DIR",
    "/home/admin1/dataset/aes-128-gcm/aes-128-gcm",
)
RESULT_ROOT = Path(
    os.environ.get("SMGF_RESULT_DIR", str(Path.cwd() / "RESULT"))
).resolve()

SPLIT_SEED = 42
ATTACK_BASE_SEED = 20260919
TRAIN_SEEDS = [42, 43, 44]
TRAIN_RATIO = 0.72
VAL_RATIO = 0.08
TEST_RATIO = 0.20

BATCH_SIZE = 64
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 5
LEARNING_RATE = 0.002
WEIGHT_DECAY = 1e-4

ROBUST_BATCH_RATIO = 0.25
ROBUST_STRENGTH_MIN = 0.3
ROBUST_STRENGTH_MAX = 0.9
ATTACK_CE_WEIGHT = 1.0
CONSISTENCY_KL_WEIGHT = 0.2

MULTI_ATTACKS = ["padding", "iat_jitter", "dummy", "mixed"]
MULTI_STRENGTHS = [0.3, 0.6, 0.9, 1.0]
ADAPTIVE_STRENGTHS = [0.4, 0.7, 1.0]
PRIMARY_MIXED_STRENGTH = 1.0
PRIMARY_ADAPTIVE_STRENGTH = 1.0
ECE_BINS = 15

EXPECTED_NUM_CLASSES = 41
EXPECTED_NUM_SAMPLES = 41000
EXPECTED_TRAIN_N = 29520
EXPECTED_VAL_N = 3280
EXPECTED_TEST_N = 8200

SAVE_CHECKPOINTS = True
RUN_TSNE = True
RUN_MODEL_ONLY_LATENCY = True
RUN_SENSITIVITY_FIGURE = True
RUN_CHANNEL_GATE_AUDIT = True
GATE_AUDIT_MAX_SAMPLES = 1000
TSNE_MAX_POINTS = 400
LATENCY_MAX_SAMPLES = 500

ABLATION_SPECS = [
    {
        "tag": "FINAL",
        "gate_mode": "learned",
        "use_robust_batches": True,
        "attack_ce_weight": ATTACK_CE_WEIGHT,
        "kl_weight": CONSISTENCY_KL_WEIGHT,
    },
    {
        "tag": "STRICT_NO_GATE",
        "gate_mode": "identity",
        "use_robust_batches": True,
        "attack_ce_weight": ATTACK_CE_WEIGHT,
        "kl_weight": CONSISTENCY_KL_WEIGHT,
    },
    {
        "tag": "NO_ATTACK_CE",
        "gate_mode": "learned",
        "use_robust_batches": True,
        "attack_ce_weight": 0.0,
        "kl_weight": CONSISTENCY_KL_WEIGHT,
    },
    {
        "tag": "NO_KL",
        "gate_mode": "learned",
        "use_robust_batches": True,
        "attack_ce_weight": ATTACK_CE_WEIGHT,
        "kl_weight": 0.0,
    },
    {
        "tag": "CLEAN_ONLY",
        "gate_mode": "learned",
        "use_robust_batches": False,
        "attack_ce_weight": 0.0,
        "kl_weight": 0.0,
    },
]
ABLATION_TAGS = [x["tag"] for x in ABLATION_SPECS]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def model_state_sha256(model) -> str:
    h = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        h.update(key.encode("utf-8"))
        h.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def import_validated_core():
    if not CORE_SOURCE.is_file():
        raise FileNotFoundError(CORE_SOURCE)
    actual = sha256_file(CORE_SOURCE)
    if actual != EXPECTED_CORE_SHA256:
        raise RuntimeError(
            "Validated V16 core changed. Refusing to run silently. "
            f"expected={EXPECTED_CORE_SHA256}, actual={actual}"
        )

    tmp_root = Path(tempfile.mkdtemp(prefix="smgf_v17_core_import_"))
    old_result = os.environ.get("SMGF_RESULT_DIR")
    old_pcap = os.environ.get("SMGF_PCAP_DIR")
    os.environ["SMGF_RESULT_DIR"] = str(tmp_root)
    os.environ["SMGF_PCAP_DIR"] = PCAP_DIR
    try:
        spec = importlib.util.spec_from_file_location(
            "smgf_v16_validated_core_for_v17", CORE_SOURCE
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import validated core: {CORE_SOURCE}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        if old_result is None:
            os.environ.pop("SMGF_RESULT_DIR", None)
        else:
            os.environ["SMGF_RESULT_DIR"] = old_result
        if old_pcap is None:
            os.environ.pop("SMGF_PCAP_DIR", None)
        else:
            os.environ["SMGF_PCAP_DIR"] = old_pcap
        shutil.rmtree(tmp_root, ignore_errors=True)
    return module


CORE = import_validated_core()
torch = CORE.torch
F = CORE.F
Batch = CORE.Batch
DataLoader = CORE.DataLoader

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required for the formal paper experiment.")
DEVICE = torch.device("cuda")

RUN_STAMP = time.strftime("%Y%m%d_%H%M%S", time.localtime())
RUN_DIR = RESULT_ROOT / f"SMGF_V17_FINAL_{RUN_STAMP}"
LOG_FILE = RUN_DIR / "experiment.log"
RESULT_FILE = RUN_DIR / "results.md"
CONFIG_FILE = RUN_DIR / "run_config.json"
SPLIT_FILE = RUN_DIR / "split_manifest.json"
PER_SEED_FILE = RUN_DIR / "per_seed_results.json"
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
FIGURE_DIR = RUN_DIR / "figures"
GATE_AUDIT_FILE = FIGURE_DIR / "gate_channel_audit_seed42.json"

RUN_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Rewire validated-core auxiliary outputs into the V17 run folder.
CORE.RUN_DIR = RUN_DIR
CORE.LOG_FILE = LOG_FILE
CORE.RESULT_MD = RESULT_FILE
CORE.SPLIT_MANIFEST_FILE = SPLIT_FILE
CORE.CONFIG_FILE = CONFIG_FILE
CORE.CHECKPOINT_DIR = CHECKPOINT_DIR
CORE.FIGURE_DIR = FIGURE_DIR
CORE.RUN_TSNE = RUN_TSNE
CORE.RUN_MODEL_ONLY_LATENCY = RUN_MODEL_ONLY_LATENCY
CORE.RUN_SENSITIVITY_FIGURE = RUN_SENSITIVITY_FIGURE
CORE.RUN_GATE_DIAGNOSTICS = False
CORE.TSNE_MAX_POINTS = TSNE_MAX_POINTS
CORE.LATENCY_MAX_SAMPLES = LATENCY_MAX_SAMPLES


class Logger:
    def __init__(self, path: Path):
        self.path = Path(path)

    def log(self, msg=""):
        msg = str(msg)
        print(msg)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")


LOGGER = Logger(LOG_FILE)


# =============================================================================
# 1. FAIL-CLOSED PROTOCOL CHECKS
# =============================================================================

def assert_close(name, actual, expected, atol=1e-12):
    if not math.isclose(
        float(actual), float(expected), rel_tol=0.0, abs_tol=atol
    ):
        raise AssertionError(
            f"{name} mismatch: actual={actual}, expected={expected}"
        )


def validate_protocol_constants():
    expected_pairs = [
        ("SPLIT_SEED", CORE.SPLIT_SEED, SPLIT_SEED),
        ("ATTACK_BASE_SEED", CORE.ATTACK_BASE_SEED, ATTACK_BASE_SEED),
        ("BATCH_SIZE", CORE.BATCH_SIZE, BATCH_SIZE),
        ("FINETUNE_EPOCHS", CORE.FINETUNE_EPOCHS, MAX_EPOCHS),
        (
            "EARLY_STOP_PATIENCE",
            CORE.EARLY_STOP_PATIENCE,
            EARLY_STOP_PATIENCE,
        ),
        ("FINETUNE_LR", CORE.FINETUNE_LR, LEARNING_RATE),
        ("WEIGHT_DECAY", CORE.WEIGHT_DECAY, WEIGHT_DECAY),
        (
            "ROBUST_BATCH_RATIO",
            CORE.ROBUST_BATCH_RATIO,
            ROBUST_BATCH_RATIO,
        ),
        (
            "ROBUST_STRENGTH_MIN",
            CORE.ROBUST_STRENGTH_MIN,
            ROBUST_STRENGTH_MIN,
        ),
        (
            "ROBUST_STRENGTH_MAX",
            CORE.ROBUST_STRENGTH_MAX,
            ROBUST_STRENGTH_MAX,
        ),
    ]
    for name, actual, expected in expected_pairs:
        assert_close(name, actual, expected)

    if list(CORE.MULTI_ATTACKS) != MULTI_ATTACKS:
        raise AssertionError("MULTI_ATTACKS differs from validated protocol")
    if list(CORE.MULTI_STRENGTHS) != MULTI_STRENGTHS:
        raise AssertionError("MULTI_STRENGTHS differs from validated protocol")
    if list(CORE.ADAPTIVE_STRENGTHS) != ADAPTIVE_STRENGTHS:
        raise AssertionError("ADAPTIVE_STRENGTHS differs from validated protocol")

    LOGGER.log("[protocol] PASS: all fixed constants match validated V16 core")
    LOGGER.log(
        "[objective] clean CE + attacked CE + asymmetric clean-to-perturbed KL"
    )
    LOGGER.log("[objective] ECE is evaluation-only; it is not an optimization term")


def log_runtime():
    LOGGER.log("=" * 104)
    LOGGER.log("SMGF-Net V17 FINAL - paper-grade three-seed experiment")
    LOGGER.log(f"Run directory: {RUN_DIR}")
    LOGGER.log(f"Python: {platform.python_version()}")
    LOGGER.log(f"PyTorch: {torch.__version__}")
    LOGGER.log(f"CUDA runtime: {torch.version.cuda}")
    LOGGER.log(f"GPU: {torch.cuda.get_device_name(0)}")
    LOGGER.log(f"TRAIN_SEEDS={TRAIN_SEEDS}")
    LOGGER.log(f"SPLIT_SEED={SPLIT_SEED}")
    LOGGER.log(f"ATTACK_BASE_SEED={ATTACK_BASE_SEED}")
    LOGGER.log(
        f"split={TRAIN_RATIO:.2f}/{VAL_RATIO:.2f}/{TEST_RATIO:.2f} | "
        f"robust_batches={ROBUST_BATCH_RATIO:.2f} | "
        f"mixed_train_strength=[{ROBUST_STRENGTH_MIN:.1f},"
        f"{ROBUST_STRENGTH_MAX:.1f}]"
    )
    LOGGER.log("=" * 104)


# =============================================================================
# 2. FINAL MODEL - NO REPRESENTATION-PRETRAINING HEAD
# =============================================================================

class SMGFNetV17(torch.nn.Module):
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

        self.conv1 = CORE.GATv2Conv(
            node_in_channels,
            hidden_channels,
            heads=4,
            edge_dim=1,
            concat=True,
        )
        self.bn1 = CORE.BatchNorm1d(hidden_channels * 4)
        self.conv2 = CORE.GATv2Conv(
            hidden_channels * 4,
            hidden_channels,
            heads=4,
            edge_dim=1,
            concat=True,
        )
        self.bn2 = CORE.BatchNorm1d(hidden_channels * 4)
        self.lin_skip2 = CORE.Linear(
            hidden_channels * 4, hidden_channels * 4
        )
        self.conv3 = CORE.GATv2Conv(
            hidden_channels * 4,
            hidden_channels,
            heads=4,
            edge_dim=1,
            concat=True,
        )
        self.bn3 = CORE.BatchNorm1d(hidden_channels * 4)
        self.lin_skip3 = CORE.Linear(
            hidden_channels * 4, hidden_channels * 4
        )
        self.pool = CORE.AttentionalAggregation(
            gate_nn=CORE.Linear(hidden_channels * 4, 1)
        )

        self.graph_dim = hidden_channels * 4
        self.finger_cnn = CORE.FingerprintCNN(
            seq_len=finger_seq_len, out_dim=32
        )
        self.entropy_cnn = CORE.EntropyCNN(
            seq_len=entropy_seq_len, out_dim=16
        )
        self.fusion_layer = CORE.GatedFusionV13(
            self.graph_dim,
            stats_dim,
            32,
            16,
            gate_mode=gate_mode,
        )

        self.final_dim = self.graph_dim + 64
        self.classifier_head = CORE.Sequential(
            CORE.Linear(self.final_dim, 128),
            CORE.ReLU(),
            CORE.Dropout(0.5),
            CORE.Linear(128, out_channels),
        )

    def encode_graph(self, x, edge_index, edge_attr, batch):
        x1 = self.bn1(
            self.conv1(x, edge_index, edge_attr=edge_attr)
        ).relu()
        x2 = (
            self.bn2(
                self.conv2(x1, edge_index, edge_attr=edge_attr)
            ).relu()
            + self.lin_skip2(x1)
        )
        x3 = (
            self.bn3(
                self.conv3(x2, edge_index, edge_attr=edge_attr)
            ).relu()
            + self.lin_skip3(x2)
        )
        return self.pool(x3, batch)

    def extract_fused_features(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        stats_attr,
        finger_attr,
        entropy_attr,
    ):
        h_graph = self.encode_graph(x, edge_index, edge_attr, batch)
        h_finger = self.finger_cnn(finger_attr)
        h_entropy = self.entropy_cnn(entropy_attr)
        return self.fusion_layer(
            h_graph,
            stats_attr,
            h_finger,
            h_entropy,
        )

    def forward_classifier(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        stats_attr,
        finger_attr,
        entropy_attr,
    ):
        h = self.extract_fused_features(
            x,
            edge_index,
            edge_attr,
            batch,
            stats_attr,
            finger_attr,
            entropy_attr,
        )
        return self.classifier_head(h)


# =============================================================================
# 3. DATA / SPLIT / ATTACK CACHE
# =============================================================================

def save_split_manifest(metadata, train_idx, val_idx, test_idx, manifest_hash):
    def rows(indices):
        return [
            {
                "sample_index": int(metadata[i]["sample_index"]),
                "relative_path": metadata[i]["relative_path"],
                "label": int(metadata[i]["label"]),
            }
            for i in indices
        ]

    payload = {
        "dataset_name": DATASET_NAME,
        "dataset_manifest_sha256": manifest_hash,
        "split_seed": SPLIT_SEED,
        "ratios": {
            "train": TRAIN_RATIO,
            "validation": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "train": rows(train_idx),
        "validation": rows(val_idx),
        "test": rows(test_idx),
    }
    SPLIT_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_and_prepare_data():
    dataset, num_classes, idx_to_class, metadata = (
        CORE.load_dataset_closedset(PCAP_DIR, LOGGER)
    )
    if len(dataset) != EXPECTED_NUM_SAMPLES:
        raise AssertionError(
            f"Expected {EXPECTED_NUM_SAMPLES} samples, got {len(dataset)}"
        )
    if num_classes != EXPECTED_NUM_CLASSES:
        raise AssertionError(
            f"Expected {EXPECTED_NUM_CLASSES} classes, got {num_classes}"
        )

    manifest_hash = CORE.dataset_manifest_hash(metadata)
    LOGGER.log(f"Dataset manifest SHA-256: {manifest_hash}")
    if manifest_hash != EXPECTED_DATASET_MANIFEST_SHA256:
        raise AssertionError(
            "Dataset manifest changed. Refusing protocol drift. "
            f"expected={EXPECTED_DATASET_MANIFEST_SHA256}, "
            f"actual={manifest_hash}"
        )

    train_idx, val_idx, test_idx = CORE.stratified_split_indices(
        dataset,
        split_seed=SPLIT_SEED,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
    )
    train_set = CORE.materialize_split(dataset, train_idx)
    val_set = CORE.materialize_split(dataset, val_idx)
    test_set = CORE.materialize_split(dataset, test_idx)

    train_ids = set(int(i) for i in train_idx)
    val_ids = set(int(i) for i in val_idx)
    test_ids = set(int(i) for i in test_idx)
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise AssertionError("Train/validation/test split overlap detected")

    counts = (len(train_set), len(val_set), len(test_set))
    expected = (EXPECTED_TRAIN_N, EXPECTED_VAL_N, EXPECTED_TEST_N)
    if counts != expected:
        raise AssertionError(f"Split counts changed: {counts} != {expected}")

    LOGGER.log(
        f"Canonical split -> Train={len(train_set)} | "
        f"Validation={len(val_set)} | Test={len(test_set)}"
    )
    save_split_manifest(
        metadata,
        train_idx,
        val_idx,
        test_idx,
        manifest_hash,
    )

    node_scaler, stats_scaler = CORE.fit_and_apply_scalers(
        train_set,
        val_set,
        test_set,
        LOGGER,
    )
    CORE.validate_reconstruction_identity_fast(
        test_set,
        node_scaler,
        stats_scaler,
        LOGGER,
        max_samples=16,
    )

    return (
        train_set,
        val_set,
        test_set,
        node_scaler,
        stats_scaler,
        num_classes,
        idx_to_class,
        manifest_hash,
    )


def validate_zero_strength_noop(
    test_set,
    node_scaler,
    stats_scaler,
    max_samples=16,
):
    attrs = [
        "x",
        "edge_index",
        "edge_attr",
        "stats_attr",
        "finger_attr",
        "entropy_attr",
        "y",
    ]
    subset = test_set[: min(max_samples, len(test_set))]
    for attack in ["padding", "iat_jitter", "dummy", "mixed", "adaptive"]:
        for i, clean in enumerate(subset):
            seed = CORE.stable_seed("v17_zero_noop", attack, i)
            attacked = CORE.apply_attack_deterministic(
                clean,
                attack,
                node_scaler,
                stats_scaler,
                0.0,
                np.random.default_rng(seed),
                random.Random(seed),
            )
            for attr in attrs:
                a = getattr(clean, attr)
                b = getattr(attacked, attr)
                if not torch.equal(a, b):
                    raise AssertionError(
                        f"Zero-strength no-op failed: attack={attack}, "
                        f"sample={i}, attr={attr}"
                    )
    LOGGER.log(
        "[zero-strength identity] PASS: padding/iat/dummy/mixed/adaptive "
        "are exact no-ops at strength=0"
    )


def precompute_canonical_eval_cache(
    test_set,
    node_scaler,
    stats_scaler,
):
    cache = {}
    LOGGER.log("=" * 104)
    LOGGER.log("Precomputing ONE canonical evaluation cache")
    LOGGER.log("These attacked test sets are evaluation-only.")
    LOGGER.log("=" * 104)

    specs = []
    for attack in MULTI_ATTACKS:
        for strength in MULTI_STRENGTHS:
            specs.append((attack, float(strength)))
    for strength in ADAPTIVE_STRENGTHS:
        specs.append(("adaptive", float(strength)))

    for attack, strength in specs:
        attack_seed = CORE.stable_seed(
            ATTACK_BASE_SEED,
            attack,
            f"{strength:.3f}",
        )
        attacked = CORE.build_attacked_dataset_deterministic(
            test_set,
            attack,
            node_scaler,
            stats_scaler,
            strength,
            attack_seed,
        )
        CORE.validate_paired_dataset(
            test_set,
            attacked,
            f"{attack}@{strength}",
        )
        if attack == "adaptive":
            CORE.validate_adaptive_semantics(test_set, attacked)
        cache[(attack, strength)] = attacked
        LOGGER.log(
            f"[eval cache] {attack:10s} strength={strength:.1f} "
            f"seed={attack_seed} n={len(attacked)}"
        )

    LOGGER.log("[eval cache] PASS: paired ordering and labels preserved")
    LOGGER.log(
        "[eval cache] PASS: cache is never passed to the training function"
    )
    return cache


# =============================================================================
# 4. TRAINING
# =============================================================================

def forward_logits(model, batch):
    return model.forward_classifier(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.stats_attr,
        batch.finger_attr,
        batch.entropy_attr,
    )


def train_model(
    train_set,
    val_set,
    num_classes,
    node_scaler,
    stats_scaler,
    tag,
    gate_mode,
    train_seed,
    use_robust_batches,
    attack_ce_weight,
    kl_weight,
):
    LOGGER.log("")
    LOGGER.log("-" * 104)
    LOGGER.log(
        f"TRAIN {tag} | seed={train_seed} | gate_mode={gate_mode} | "
        f"robust_batches={use_robust_batches} | "
        f"attack_ce_w={attack_ce_weight} | kl_w={kl_weight}"
    )
    LOGGER.log("-" * 104)

    CORE.set_seed(train_seed)
    model = SMGFNetV17(
        node_in_channels=5,
        stats_dim=9,
        finger_seq_len=20,
        entropy_seq_len=5,
        hidden_channels=32,
        out_channels=num_classes,
        gate_mode=gate_mode,
    ).to(DEVICE)
    init_sha256 = model_state_sha256(model)
    LOGGER.log(
        f"[{tag}][seed={train_seed}] initial_state_sha256={init_sha256}"
    )

    # Every variant starts from the same seed-controlled initialization.
    train_loader = CORE.make_loader(
        train_set,
        BATCH_SIZE,
        shuffle=True,
        seed=train_seed + 21_000,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    no_improve = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss = 0.0
        total_clean_ce = 0.0
        total_attack_ce = 0.0
        total_kl = 0.0
        batches = 0
        robust_batches = 0

        for batch_idx, batch_cpu in enumerate(train_loader):
            selector = CORE.stable_seed(
                "v17_robust_batch",
                train_seed,
                epoch,
                batch_idx,
                modulo=1_000_000,
            ) / 1_000_000.0
            use_robust_batch = (
                use_robust_batches and selector < ROBUST_BATCH_RATIO
            )

            attacked_batch = None
            if use_robust_batch:
                robust_seed = CORE.stable_seed(
                    "v17_train_attack",
                    train_seed,
                    epoch,
                    batch_idx,
                )
                strength_rng = np.random.default_rng(
                    CORE.stable_seed(
                        "v17_train_strength",
                        train_seed,
                        epoch,
                        batch_idx,
                    )
                )
                strength = float(
                    strength_rng.uniform(
                        ROBUST_STRENGTH_MIN,
                        ROBUST_STRENGTH_MAX,
                    )
                )
                np_rng = np.random.default_rng(robust_seed)
                py_rng = random.Random(robust_seed)
                attacked_list = [
                    CORE.apply_attack_deterministic(
                        d,
                        "mixed",
                        node_scaler,
                        stats_scaler,
                        strength,
                        np_rng,
                        py_rng,
                    )
                    for d in batch_cpu.to_data_list()
                ]
                attacked_batch = Batch.from_data_list(
                    attacked_list
                ).to(DEVICE)

            batch = batch_cpu.to(DEVICE)
            optimizer.zero_grad()
            clean_logits = forward_logits(model, batch)
            clean_ce = criterion(clean_logits, batch.y.view(-1))
            loss = clean_ce
            attack_ce_value = 0.0
            kl_value = 0.0

            if attacked_batch is not None:
                attacked_logits = forward_logits(model, attacked_batch)
                attack_ce = criterion(
                    attacked_logits,
                    attacked_batch.y.view(-1),
                )
                consistency_kl = F.kl_div(
                    F.log_softmax(attacked_logits, dim=1),
                    F.softmax(clean_logits.detach(), dim=1),
                    reduction="batchmean",
                )
                loss = (
                    clean_ce
                    + attack_ce_weight * attack_ce
                    + kl_weight * consistency_kl
                )
                attack_ce_value = float(attack_ce.item())
                kl_value = float(consistency_kl.item())
                robust_batches += 1

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_clean_ce += float(clean_ce.item())
            total_attack_ce += attack_ce_value
            total_kl += kl_value
            batches += 1

        val_loss = CORE.compute_supervised_loss(
            model,
            val_set,
            batch_size=BATCH_SIZE,
        )
        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 0 or (epoch + 1) % 5 == 0 or no_improve == 0:
            LOGGER.log(
                f"[{tag}][seed={train_seed}] epoch {epoch+1:02d}/"
                f"{MAX_EPOCHS} | "
                f"loss={total_loss/max(batches,1):.6f} | "
                f"clean_ce={total_clean_ce/max(batches,1):.6f} | "
                f"attack_ce={total_attack_ce/max(robust_batches,1):.6f} | "
                f"kl={total_kl/max(robust_batches,1):.6f} | "
                f"robust_batches={robust_batches}/{batches} | "
                f"val_loss={val_loss:.6f} | best_epoch={best_epoch}"
            )

        if no_improve >= EARLY_STOP_PATIENCE:
            LOGGER.log(
                f"[{tag}][seed={train_seed}] early stop epoch={epoch+1}; "
                f"best_epoch={best_epoch}; best_val_loss={best_val_loss:.6f}"
            )
            break

    if best_state is None:
        raise RuntimeError(f"{tag} seed={train_seed}: no checkpoint")
    model.load_state_dict(best_state)

    if SAVE_CHECKPOINTS:
        ckpt = CHECKPOINT_DIR / f"{tag}_seed{train_seed}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "tag": tag,
                "train_seed": train_seed,
                "gate_mode": gate_mode,
                "use_robust_batches": use_robust_batches,
                "attack_ce_weight": attack_ce_weight,
                "kl_weight": kl_weight,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "weight_decay": WEIGHT_DECAY,
                "robust_batch_ratio": ROBUST_BATCH_RATIO,
                "robust_strength_range": [
                    ROBUST_STRENGTH_MIN,
                    ROBUST_STRENGTH_MAX,
                ],
                "ece_is_evaluation_only": True,
                "initial_state_sha256": init_sha256,
            },
            ckpt,
        )
        LOGGER.log(f"[{tag}][seed={train_seed}] checkpoint: {ckpt}")

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "initial_state_sha256": init_sha256,
    }


# =============================================================================
# 5. EVALUATION
# =============================================================================

def evaluate_primary(model, tag, seed, test_set, attack_cache):
    clean = CORE.evaluate_clean(model, test_set)
    mixed = CORE.evaluate_attack(
        model,
        attack_cache[("mixed", PRIMARY_MIXED_STRENGTH)],
        clean,
    )
    adaptive = CORE.evaluate_attack(
        model,
        attack_cache[("adaptive", PRIMARY_ADAPTIVE_STRENGTH)],
        clean,
    )
    CORE.log_clean_result(LOGGER, tag, seed, clean)
    CORE.log_attack_result(
        LOGGER,
        tag,
        seed,
        "mixed",
        PRIMARY_MIXED_STRENGTH,
        mixed,
    )
    CORE.log_attack_result(
        LOGGER,
        tag,
        seed,
        "adaptive",
        PRIMARY_ADAPTIVE_STRENGTH,
        adaptive,
    )
    return {
        "seed": seed,
        "clean_acc": clean["acc"],
        "clean_f1": clean["macro_f1"],
        "clean_ece": clean["ece"],
        "mixed_robust_acc": mixed["robust_acc"],
        "mixed_robust_f1": mixed["robust_f1"],
        "mixed_asr": mixed["asr"],
        "mixed_rel_drop": mixed["rel_drop"],
        "mixed_ece": mixed["ece"],
        "adaptive_robust_acc": adaptive["robust_acc"],
        "adaptive_robust_f1": adaptive["robust_f1"],
        "adaptive_asr": adaptive["asr"],
        "adaptive_rel_drop": adaptive["rel_drop"],
        "adaptive_ece": adaptive["ece"],
    }


def mean_sd(records, key):
    vals = np.asarray([float(x[key]) for x in records], dtype=np.float64)
    return float(vals.mean()), float(vals.std(ddof=1))


def fmt_mean_sd(records, key):
    m, s = mean_sd(records, key)
    return f"{m:.2f} ± {s:.2f}"


# =============================================================================
# 6. CHANNEL-WISE GATE AUDIT
# =============================================================================

def balanced_test_indices(dataset, max_n, seed):
    by_class = defaultdict(list)
    for i, data in enumerate(dataset):
        by_class[int(data.y.view(-1)[0].item())].append(i)

    rng = np.random.default_rng(seed)
    classes = sorted(by_class)
    per_class = max(1, max_n // len(classes))
    selected = []
    leftovers = []
    for cls in classes:
        idx = np.asarray(by_class[cls], dtype=np.int64)
        rng.shuffle(idx)
        take = min(per_class, len(idx))
        selected.extend(idx[:take].tolist())
        leftovers.extend(idx[take:].tolist())

    if len(selected) < max_n and leftovers:
        leftovers = np.asarray(leftovers, dtype=np.int64)
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max_n - len(selected)].tolist())
    return np.asarray(sorted(selected[:max_n]), dtype=np.int64)


def collect_gate_matrix(model, dataset, indices):
    subset = [dataset[int(i)] for i in indices]
    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            h_finger = model.finger_cnn(batch.finger_attr)
            h_entropy = model.entropy_cnn(batch.entropy_attr)
            h_ctx = torch.cat(
                [batch.stats_attr, h_finger, h_entropy],
                dim=1,
            )
            gate = model.fusion_layer.global_gate(h_ctx)
            rows.append(gate.detach().cpu().numpy())
    return np.vstack(rows)


def rankdata_average(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman_small(x, y):
    rx = rankdata_average(x)
    ry = rankdata_average(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def gate_attack_summary(clean_gates, attack_gates_by_strength, strengths):
    channel_means = []
    for strength in strengths:
        channel_means.append(
            attack_gates_by_strength[float(strength)].mean(axis=0)
        )
    channel_means = np.stack(channel_means, axis=0)

    rho = np.asarray(
        [
            spearman_small(strengths, channel_means[:, j])
            for j in range(channel_means.shape[1])
        ],
        dtype=np.float64,
    )
    final_gates = attack_gates_by_strength[float(strengths[-1])]
    delta = final_gates - clean_gates
    mean_shift = delta.mean(axis=0)

    return {
        "median_channel_rho": float(np.median(rho)),
        "mean_abs_channel_rho": float(np.mean(np.abs(rho))),
        "strong_suppressor_channels": int(np.sum(rho <= -0.8)),
        "strong_enhancer_channels": int(np.sum(rho >= 0.8)),
        "final_mean_signed_gate_delta": float(delta.mean()),
        "final_mean_abs_paired_gate_delta": float(np.abs(delta).mean()),
        "channels_final_negative_shift": int(np.sum(mean_shift < 0)),
        "channels_final_positive_shift": int(np.sum(mean_shift > 0)),
        "channel_rho": rho.tolist(),
        "final_channel_mean_shift": mean_shift.tolist(),
    }


def run_channel_gate_audit(model, clean_set, attack_cache, seed):
    if not RUN_CHANNEL_GATE_AUDIT:
        return None
    if model.fusion_layer.gate_mode != "learned":
        raise AssertionError("Channel gate audit requires learned gate")

    n = min(GATE_AUDIT_MAX_SAMPLES, len(clean_set))
    indices = balanced_test_indices(
        clean_set,
        n,
        CORE.stable_seed("v17_gate_audit_subset", seed),
    )
    clean_gates = collect_gate_matrix(model, clean_set, indices)

    payload = {
        "seed": seed,
        "sample_count": int(len(indices)),
        "sampling": "deterministic class-balanced test subset",
        "interpretation": (
            "channel-wise modulation; global monotonic suppression is not "
            "assumed or required"
        ),
        "clean_gate_mean": float(clean_gates.mean()),
        "clean_gate_median": float(np.median(clean_gates)),
    }

    for attack, strengths in [
        ("mixed", MULTI_STRENGTHS),
        ("adaptive", ADAPTIVE_STRENGTHS),
    ]:
        attacked = {}
        for strength in strengths:
            attacked[float(strength)] = collect_gate_matrix(
                model,
                attack_cache[(attack, float(strength))],
                indices,
            )
        payload[attack] = gate_attack_summary(
            clean_gates,
            attacked,
            strengths,
        )

    GATE_AUDIT_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    LOGGER.log(f"[gate audit] channel-wise audit saved: {GATE_AUDIT_FILE}")
    LOGGER.log(
        "[gate audit] no claim of global monotonic gate decay is made"
    )
    return str(GATE_AUDIT_FILE)


# =============================================================================
# 7. REPORTING
# =============================================================================

def save_run_config(dataset_info):
    config = {
        "paper_model": (
            "learned Gate + perturbation-aware attacked CE + "
            "asymmetric clean-to-perturbed KL"
        ),
        "dataset_name": DATASET_NAME,
        "pcap_dir": PCAP_DIR,
        "dataset": dataset_info,
        "expected_dataset_manifest_sha256": (
            EXPECTED_DATASET_MANIFEST_SHA256
        ),
        "validated_core_source": str(CORE_SOURCE),
        "validated_core_sha256": EXPECTED_CORE_SHA256,
        "script_sha256": sha256_file(Path(__file__).resolve()),
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
            "max_epochs": MAX_EPOCHS,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "early_stop_metric": "clean validation cross-entropy only",
        },
        "robust_training": {
            "robust_batch_ratio": ROBUST_BATCH_RATIO,
            "train_attack": "mixed",
            "strength_range": [
                ROBUST_STRENGTH_MIN,
                ROBUST_STRENGTH_MAX,
            ],
            "attack_ce_weight": ATTACK_CE_WEIGHT,
            "consistency_kl_weight": CONSISTENCY_KL_WEIGHT,
            "kl_direction": "clean stop-gradient -> attacked prediction",
            "ece_is_evaluation_only": True,
            "training_attack_scope": "train split batches only",
        },
        "evaluation": {
            "multi_attacks": MULTI_ATTACKS,
            "multi_strengths": MULTI_STRENGTHS,
            "adaptive_strengths": ADAPTIVE_STRENGTHS,
            "primary_mixed_strength": PRIMARY_MIXED_STRENGTH,
            "primary_adaptive_strength": PRIMARY_ADAPTIVE_STRENGTH,
            "canonical_eval_cache_shared_across_all_models_and_seeds": True,
            "evaluation_cache_never_used_for_optimization": True,
        },
        "ablations": ABLATION_SPECS,
        "claims_boundary": {
            "adaptive_attack": (
                "deterministic semantics-preserving structural/mixed "
                "perturbation; not gradient-based white-box adaptation"
            ),
            "gate": (
                "channel-wise context-conditioned modulation; no global "
                "monotonic-decay claim"
            ),
            "theory": (
                "perturbation-propagation upper bound and KL-based sufficient "
                "prediction-stability condition; not universal robustness proof"
            ),
        },
    }
    CONFIG_FILE.write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_results(
    dataset_info,
    final_clean_records,
    final_attack_records,
    ablation_records,
    per_seed,
    deployment_result,
    tsne_path,
    sensitivity_path,
    gate_audit_path,
):
    lines = [
        "# SMGF-Net V17 FINAL Results",
        "",
        "## 1. Protocol",
        "",
        "- Final model: learned Gate + attacked CE + asymmetric KL consistency.",
        "- No representation-pretraining stage.",
        "- Train seeds: 42, 43, 44.",
        "- Split seed: 42.",
        "- Scaler fitted once on train only.",
        "- Canonical evaluation attacks shared by all models and seeds.",
        "- Test attacks are evaluation-only.",
        "- ECE is evaluation-only and never optimized.",
        "- Strict no-gate keeps the same GatedFusion module and uses an identity multiplier.",
        "",
        "## 2. Dataset",
        "",
        f"- Manifest SHA-256: {dataset_info['manifest_hash']}",
        f"- Classes: {dataset_info['num_classes']}",
        f"- Samples: {dataset_info['num_samples']}",
        f"- Train/Val/Test: {dataset_info['train_n']}/"
        f"{dataset_info['val_n']}/{dataset_info['test_n']}",
        "",
        "## 3. Final model clean performance",
        "",
        "| Metric | mean ± SD |",
        "| --- | ---: |",
        f"| Accuracy (%) | {fmt_mean_sd(final_clean_records, 'acc')} |",
        f"| Macro-F1 (%) | {fmt_mean_sd(final_clean_records, 'macro_f1')} |",
        f"| ECE (%) | {fmt_mean_sd(final_clean_records, 'ece')} |",
        "",
        "## 4. Final model robustness",
        "",
        "| Attack | Strength | RobAcc | RobF1 | ASR | RelDrop | ECE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for attack in MULTI_ATTACKS:
        for strength in MULTI_STRENGTHS:
            recs = final_attack_records[(attack, float(strength))]
            lines.append(
                f"| {attack} | {strength:.1f} | "
                f"{fmt_mean_sd(recs, 'robust_acc')} | "
                f"{fmt_mean_sd(recs, 'robust_f1')} | "
                f"{fmt_mean_sd(recs, 'asr')} | "
                f"{fmt_mean_sd(recs, 'rel_drop')} | "
                f"{fmt_mean_sd(recs, 'ece')} |"
            )
    for strength in ADAPTIVE_STRENGTHS:
        recs = final_attack_records[("adaptive", float(strength))]
        lines.append(
            f"| adaptive | {strength:.1f} | "
            f"{fmt_mean_sd(recs, 'robust_acc')} | "
            f"{fmt_mean_sd(recs, 'robust_f1')} | "
            f"{fmt_mean_sd(recs, 'asr')} | "
            f"{fmt_mean_sd(recs, 'rel_drop')} | "
            f"{fmt_mean_sd(recs, 'ece')} |"
        )

    lines.extend(
        [
            "",
            "## 5. Strict component ablation",
            "",
            "| Setting | Gate | Attack CE | KL | Clean Acc | Clean ECE | "
            "Mixed RobAcc | Mixed RelDrop | Mixed ECE | Adaptive RobAcc | "
            "Adaptive RelDrop | Adaptive ECE |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: |",
        ]
    )

    flags = {
        "FINAL": ("learned", "on", "on"),
        "STRICT_NO_GATE": ("identity", "on", "on"),
        "NO_ATTACK_CE": ("learned", "off", "on"),
        "NO_KL": ("learned", "on", "off"),
        "CLEAN_ONLY": ("learned", "off", "off"),
    }
    for tag in ABLATION_TAGS:
        recs = ablation_records[tag]
        g, a, k = flags[tag]
        lines.append(
            f"| {tag} | {g} | {a} | {k} | "
            f"{fmt_mean_sd(recs, 'clean_acc')} | "
            f"{fmt_mean_sd(recs, 'clean_ece')} | "
            f"{fmt_mean_sd(recs, 'mixed_robust_acc')} | "
            f"{fmt_mean_sd(recs, 'mixed_rel_drop')} | "
            f"{fmt_mean_sd(recs, 'mixed_ece')} | "
            f"{fmt_mean_sd(recs, 'adaptive_robust_acc')} | "
            f"{fmt_mean_sd(recs, 'adaptive_rel_drop')} | "
            f"{fmt_mean_sd(recs, 'adaptive_ece')} |"
        )

    lines.extend(
        [
            "",
            "## 6. Reproducibility and interpretation checks",
            "",
            "- Stable dataset manifest and fixed split verified before training.",
            "- Reconstruction identity test executed before attack generation.",
            "- Training perturbations are generated from train batches only.",
            "- Canonical attacked test sets are generated once and never optimized on.",
            "- All variants use the same seed-controlled initialization protocol.",
            "- Strict no-gate uses the same GatedFusion module.",
            "- Channel-gate audit does not assume global monotonic suppression.",
            "- ECE is an evaluation metric only.",
            "- The adaptive endpoint is not described as gradient-based white-box adaptation.",
            "",
            "## 7. Auxiliary artifacts",
            "",
            f"- Per-seed JSON: {PER_SEED_FILE}",
            f"- Gate channel audit: {gate_audit_path}",
            f"- t-SNE: {tsne_path}",
            f"- Sensitivity figure: {sensitivity_path}",
        ]
    )
    if deployment_result is not None:
        lines.extend(
            [
                "- Deployment timing is neural forward only; it excludes PCAP "
                "parsing, feature extraction, graph construction, and H2D.",
                f"- Model-only latency: "
                f"{deployment_result['model_only_latency_ms']:.4f} ms/sample.",
                f"- Model-only throughput: "
                f"{deployment_result['model_only_throughput_sps']:.2f} samples/s.",
                f"- Trainable parameters: "
                f"{deployment_result['trainable_parameters']:,}.",
            ]
        )

    RESULT_FILE.write_text("\n".join(lines), encoding="utf-8")
    PER_SEED_FILE.write_text(
        json.dumps(per_seed, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# =============================================================================
# 8. MAIN
# =============================================================================

def main():
    log_runtime()
    validate_protocol_constants()

    if not Path(PCAP_DIR).exists():
        raise FileNotFoundError(
            f"Dataset path does not exist: {PCAP_DIR}. "
            "Set SMGF_PCAP_DIR."
        )

    (
        train_set,
        val_set,
        test_set,
        node_scaler,
        stats_scaler,
        num_classes,
        _idx_to_class,
        manifest_hash,
    ) = load_and_prepare_data()

    dataset_info = {
        "manifest_hash": manifest_hash,
        "num_classes": num_classes,
        "num_samples": EXPECTED_NUM_SAMPLES,
        "train_n": len(train_set),
        "val_n": len(val_set),
        "test_n": len(test_set),
    }
    save_run_config(dataset_info)

    validate_zero_strength_noop(
        test_set,
        node_scaler,
        stats_scaler,
        max_samples=16,
    )

    attack_cache = precompute_canonical_eval_cache(
        test_set,
        node_scaler,
        stats_scaler,
    )

    final_clean_records = []
    final_attack_records = {
        (attack, float(strength)): []
        for attack in MULTI_ATTACKS
        for strength in MULTI_STRENGTHS
    }
    for strength in ADAPTIVE_STRENGTHS:
        final_attack_records[("adaptive", float(strength))] = []

    ablation_records = {tag: [] for tag in ABLATION_TAGS}
    per_seed = {tag: [] for tag in ABLATION_TAGS}

    deployment_result = None
    tsne_path = None
    gate_audit_path = None

    for seed_i, train_seed in enumerate(TRAIN_SEEDS):
        LOGGER.log("")
        LOGGER.log("#" * 104)
        LOGGER.log(
            f"TRAINING REPLICATE {seed_i+1}/{len(TRAIN_SEEDS)} "
            f"| seed={train_seed}"
        )
        LOGGER.log("#" * 104)

        # FINAL paper method.
        spec = ABLATION_SPECS[0]
        bundle = train_model(
            train_set,
            val_set,
            num_classes,
            node_scaler,
            stats_scaler,
            tag=spec["tag"],
            gate_mode=spec["gate_mode"],
            train_seed=train_seed,
            use_robust_batches=spec["use_robust_batches"],
            attack_ce_weight=spec["attack_ce_weight"],
            kl_weight=spec["kl_weight"],
        )
        final_model = bundle["model"]
        expected_init_sha256 = bundle["initial_state_sha256"]

        final_clean = CORE.evaluate_clean(final_model, test_set)
        CORE.log_clean_result(LOGGER, "FINAL", train_seed, final_clean)
        final_clean_records.append(final_clean)

        per_seed_attacks = {}
        for attack in MULTI_ATTACKS:
            for strength in MULTI_STRENGTHS:
                result = CORE.evaluate_attack(
                    final_model,
                    attack_cache[(attack, float(strength))],
                    final_clean,
                )
                final_attack_records[(attack, float(strength))].append(
                    result
                )
                per_seed_attacks[(attack, float(strength))] = result
                CORE.log_attack_result(
                    LOGGER,
                    "FINAL",
                    train_seed,
                    attack,
                    strength,
                    result,
                )
        for strength in ADAPTIVE_STRENGTHS:
            result = CORE.evaluate_attack(
                final_model,
                attack_cache[("adaptive", float(strength))],
                final_clean,
            )
            final_attack_records[("adaptive", float(strength))].append(
                result
            )
            per_seed_attacks[("adaptive", float(strength))] = result
            CORE.log_attack_result(
                LOGGER,
                "FINAL",
                train_seed,
                "adaptive",
                strength,
                result,
            )

        primary_mixed = per_seed_attacks[
            ("mixed", PRIMARY_MIXED_STRENGTH)
        ]
        primary_adaptive = per_seed_attacks[
            ("adaptive", PRIMARY_ADAPTIVE_STRENGTH)
        ]
        final_primary = {
            "seed": train_seed,
            "clean_acc": final_clean["acc"],
            "clean_f1": final_clean["macro_f1"],
            "clean_ece": final_clean["ece"],
            "mixed_robust_acc": primary_mixed["robust_acc"],
            "mixed_robust_f1": primary_mixed["robust_f1"],
            "mixed_asr": primary_mixed["asr"],
            "mixed_rel_drop": primary_mixed["rel_drop"],
            "mixed_ece": primary_mixed["ece"],
            "adaptive_robust_acc": primary_adaptive["robust_acc"],
            "adaptive_robust_f1": primary_adaptive["robust_f1"],
            "adaptive_asr": primary_adaptive["asr"],
            "adaptive_rel_drop": primary_adaptive["rel_drop"],
            "adaptive_ece": primary_adaptive["ece"],
        }
        final_primary["best_epoch"] = bundle["best_epoch"]
        final_primary["best_val_loss"] = bundle["best_val_loss"]
        final_primary["initial_state_sha256"] = expected_init_sha256
        ablation_records["FINAL"].append(final_primary)
        per_seed["FINAL"].append(final_primary)

        if seed_i == 0:
            tsne_path = CORE.run_representation_visualization(
                final_model,
                test_set,
                attack_cache[("mixed", PRIMARY_MIXED_STRENGTH)],
                LOGGER,
                train_seed,
            )
            deployment_result = CORE.run_deployment_overhead(
                final_model,
                test_set,
                LOGGER,
                train_seed,
            )
            gate_audit_path = run_channel_gate_audit(
                final_model,
                test_set,
                attack_cache,
                train_seed,
            )

        del final_model
        del bundle
        torch.cuda.empty_cache()

        # Strict component ablations.
        for spec in ABLATION_SPECS[1:]:
            bundle = train_model(
                train_set,
                val_set,
                num_classes,
                node_scaler,
                stats_scaler,
                tag=spec["tag"],
                gate_mode=spec["gate_mode"],
                train_seed=train_seed,
                use_robust_batches=spec["use_robust_batches"],
                attack_ce_weight=spec["attack_ce_weight"],
                kl_weight=spec["kl_weight"],
            )
            model = bundle["model"]
            if bundle["initial_state_sha256"] != expected_init_sha256:
                raise AssertionError(
                    f"Initialization mismatch for seed={train_seed}: "
                    f"FINAL={expected_init_sha256}, "
                    f"{spec['tag']}={bundle['initial_state_sha256']}"
                )
            rec = evaluate_primary(
                model,
                spec["tag"],
                train_seed,
                test_set,
                attack_cache,
            )
            rec["best_epoch"] = bundle["best_epoch"]
            rec["best_val_loss"] = bundle["best_val_loss"]
            rec["initial_state_sha256"] = bundle["initial_state_sha256"]
            ablation_records[spec["tag"]].append(rec)
            per_seed[spec["tag"]].append(rec)
            del model
            del bundle
            torch.cuda.empty_cache()

        # Incremental persistence: completed seeds survive later interruption.
        PER_SEED_FILE.write_text(
            json.dumps(per_seed, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        LOGGER.log(
            f"[initialization] PASS: all variants share the exact same "
            f"initial state for seed={train_seed}"
        )

    # FINAL row must be the exact same trained models used in the full matrix.
    for i, rec in enumerate(ablation_records["FINAL"]):
        clean = final_clean_records[i]
        mixed = final_attack_records[
            ("mixed", PRIMARY_MIXED_STRENGTH)
        ][i]
        adaptive = final_attack_records[
            ("adaptive", PRIMARY_ADAPTIVE_STRENGTH)
        ][i]
        assert_close("FINAL clean reuse", rec["clean_acc"], clean["acc"], 1e-9)
        assert_close(
            "FINAL mixed reuse",
            rec["mixed_robust_acc"],
            mixed["robust_acc"],
            1e-9,
        )
        assert_close(
            "FINAL adaptive reuse",
            rec["adaptive_robust_acc"],
            adaptive["robust_acc"],
            1e-9,
        )
    LOGGER.log("[consistency] PASS: FINAL ablation row reuses exact FINAL models")

    sensitivity_path = CORE.run_sensitivity_figure(
        final_attack_records,
        LOGGER,
    )

    write_results(
        dataset_info,
        final_clean_records,
        final_attack_records,
        ablation_records,
        per_seed,
        deployment_result,
        tsne_path,
        sensitivity_path,
        gate_audit_path,
    )

    LOGGER.log("")
    LOGGER.log("=" * 104)
    LOGGER.log("ALL V17 FINAL EXPERIMENTS COMPLETED")
    LOGGER.log(f"Results: {RESULT_FILE}")
    LOGGER.log(f"Per-seed: {PER_SEED_FILE}")
    LOGGER.log(f"Config: {CONFIG_FILE}")
    LOGGER.log(f"Split: {SPLIT_FILE}")
    LOGGER.log(f"Checkpoints: {CHECKPOINT_DIR}")
    LOGGER.log(f"Figures: {FIGURE_DIR}")
    LOGGER.log("=" * 104)


if __name__ == "__main__":
    main()
