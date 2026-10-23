# -*- coding: utf-8 -*-
"""
CipherSpectrum matched 2x2 ablation, fixed training seed 42.

This is the FINAL matched ablation companion to the frozen V17 main experiment.
It is intentionally separate from the historical V17 ablation code.

Design
------
All variants use the exact same CipherSpectrum dataset, split, scaler fitting,
optimization schedule, checkpoint rule, deterministic attack generation and
training seed. Only two factors are manipulated:

1) Semantics-guided Gate
   - learned: learned context-conditioned channel-wise gate
   - identity: strict identity gate in the SAME GatedFusionV13 module

2) Robust attacked-label training
   - on : 20% robust batches; Mixed strength Uniform[0.3,0.9];
          loss = clean CE + attacked-label CE
   - off: no attacked batches are generated or forwarded;
          loss = clean CE only

The resulting 2x2 factorial design is:

    FULL                learned gate   robust training ON
    STRICT_NO_GATE      identity gate  robust training ON
    CLEAN_ONLY          learned gate   robust training OFF
    NO_GATE_CLEAN_ONLY  identity gate  robust training OFF

FULL is NOT retrained. The formal seed42 checkpoint from the already completed
CipherSpectrum frozen three-seed experiment is reused and re-evaluated on the
same canonical Mixed@1.0 and Adaptive@1.0 attacked test sets.

All three new ablation variants are trained with seed42.

Frozen optimization protocol
----------------------------
- 100 epochs; no early stopping
- epochs 1-70: learning rate 0.002
- epochs 71-100: learning rate 0.0005
- AdamW weight_decay = 1e-4
- batch_size = 64
- checkpoint = minimum clean-validation CE
- robust batch ratio = 0.20 when robust training is ON
- training attack = Mixed
- training strength = Uniform[0.3,0.9]
- attacked-label CE weight = 1.0
- KL weight = 0.0

Primary matched evaluation
--------------------------
- Clean
- Mixed@1.0
- Adaptive@1.0

The attacked test sets are built exactly once and shared by every variant.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


# =============================================================================
# 0. IMMUTABLE MATCHED-ABLATION PROTOCOL
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SOURCE = SCRIPT_DIR / "V17-SMGF-Net_final_paper_experiments.py"
CORE_SOURCE = SCRIPT_DIR / "V16-SMGF-Net_final_paper_experiments.py"

EXPECTED_BASE_SHA256 = (
    "17c88de543207e0241c666a4aa7aae4d36c023dda37ac131d7531957ede2cfa9"
)
EXPECTED_CORE_SHA256 = (
    "3ecaf0054a1a190b444fe6f3c19cf6203273ad32707041ec53b09318ca20a4e2"
)
EXPECTED_DATASET_MANIFEST_SHA256 = (
    "821bee04771f0c8dce3680a4301164d8ab48e03ec0d1ed57c3930015ad200a61"
)
EXPECTED_FORMAL_SPLIT_SHA256 = (
    "01e3b572e8514a238b3733e224161acd1112b79b56f038635be9339b23cfad6b"
)
EXPECTED_FORMAL_FULL_SEED42_CKPT_SHA256 = (
    "a0002cb6cb692b9965b29ec187b7aefd902a475361c1cdc0450077397c4ee7d9"
)
EXPECTED_INITIAL_STATE_SHA256 = (
    "8bdaa29a7aaca7276c4384a7a834491e6b9b723eeb494320485cf868b404a452"
)

# The dataset path used by the already frozen CipherSpectrum experiment.
CIPHER_ROOT = Path("/home/admin1/dataset/aes-128-gcm/aes-128-gcm")

DATASET_NAME = "CipherSpectrum"
EXPECTED_NUM_CLASSES = 41
EXPECTED_NUM_SAMPLES = 41000
EXPECTED_TRAIN_N = 29520
EXPECTED_VAL_N = 3280
EXPECTED_TEST_N = 8200

TRAIN_SEED = 42
SPLIT_SEED = 42
ATTACK_BASE_SEED = 20260919

EPOCHS = 100
BATCH_SIZE = 64
LR_HIGH = 0.002
LR_LOW = 0.0005
LR_SWITCH_EPOCH = 71
WEIGHT_DECAY = 1e-4

ROBUST_BATCH_RATIO = 0.20
ROBUST_STRENGTH_MIN = 0.3
ROBUST_STRENGTH_MAX = 0.9
ATTACK_CE_WEIGHT = 1.0
KL_WEIGHT = 0.0

PRIMARY_MIXED_STRENGTH = 1.0
PRIMARY_ADAPTIVE_STRENGTH = 1.0

RESULT_ROOT = Path(
    os.environ.get(
        "SMGF_RESULT_DIR",
        str(SCRIPT_DIR / "RESULT"),
    )
).resolve()

STAMP = time.strftime("%Y%m%d_%H%M%S", time.localtime())
RUN_DIR = (
    RESULT_ROOT
    / f"SMGF_V17_CIPHER_MATCHED_ABLATION_SEED42_{STAMP}"
)
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
HISTORY_DIR = RUN_DIR / "history"
LOG_FILE = RUN_DIR / "experiment.log"
CONFIG_FILE = RUN_DIR / "run_config.json"
SPLIT_FILE = RUN_DIR / "split_manifest.json"
PER_VARIANT_FILE = RUN_DIR / "per_variant_results.json"
PAPER_CSV = RUN_DIR / "PAPER_READY_CIPHER_MATCHED_ABLATION.csv"
PAPER_MD = RUN_DIR / "PAPER_READY_CIPHER_MATCHED_ABLATION.md"
FACTORIAL_FILE = RUN_DIR / "factorial_effects.json"
MANIFEST_FILE = RUN_DIR / "run_manifest.json"

VARIANT_SPECS = [
    {
        "tag": "FULL",
        "gate_mode": "learned",
        "use_robust_batches": True,
        "reuse_formal_checkpoint": True,
    },
    {
        "tag": "STRICT_NO_GATE",
        "gate_mode": "identity",
        "use_robust_batches": True,
        "reuse_formal_checkpoint": False,
    },
    {
        "tag": "CLEAN_ONLY",
        "gate_mode": "learned",
        "use_robust_batches": False,
        "reuse_formal_checkpoint": False,
    },
    {
        "tag": "NO_GATE_CLEAN_ONLY",
        "gate_mode": "identity",
        "use_robust_batches": False,
        "reuse_formal_checkpoint": False,
    },
]

# Formal seed42 primary values from the archived frozen 3-seed run.
# Re-evaluation must reproduce these values, otherwise the ablation is not
# matched to the already published/frozen main-model protocol.
EXPECTED_FULL_SEED42 = {
    "clean_acc": 91.01219512195122,
    "clean_f1": 90.8058161407013,
    "clean_ece": 1.1148182996889444,
    "mixed_robust_acc": 79.45121951219512,
    "mixed_robust_f1": 79.33260723314804,
    "mixed_asr": 14.819777569342088,
    "mixed_rel_drop": 12.702666488007505,
    "mixed_ece": 7.536278725124713,
    "adaptive_robust_acc": 87.39024390243902,
    "adaptive_robust_f1": 87.3292446778646,
    "adaptive_asr": 6.016347313412837,
    "adaptive_rel_drop": 3.9796328554200753,
    "adaptive_ece": 2.969984473069994,
}


# =============================================================================
# 1. SOURCE IMPORT / HASH LOCKS
# =============================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_source_locks() -> None:
    if not BASE_SOURCE.is_file():
        raise FileNotFoundError(BASE_SOURCE)
    if not CORE_SOURCE.is_file():
        raise FileNotFoundError(CORE_SOURCE)

    base_sha = sha256_file(BASE_SOURCE)
    core_sha = sha256_file(CORE_SOURCE)

    if base_sha != EXPECTED_BASE_SHA256:
        raise RuntimeError(
            "V17 base SHA mismatch; refusing matched ablation drift. "
            f"expected={EXPECTED_BASE_SHA256}, actual={base_sha}"
        )
    if core_sha != EXPECTED_CORE_SHA256:
        raise RuntimeError(
            "V16 core SHA mismatch; refusing matched ablation drift. "
            f"expected={EXPECTED_CORE_SHA256}, actual={core_sha}"
        )


def import_base():
    assert_source_locks()

    tmp_root = Path(
        tempfile.mkdtemp(prefix="smgf_cipher_matched_ablation_import_")
    )
    old_result = os.environ.get("SMGF_RESULT_DIR")
    old_pcap = os.environ.get("SMGF_PCAP_DIR")

    os.environ["SMGF_RESULT_DIR"] = str(tmp_root)
    os.environ["SMGF_PCAP_DIR"] = str(CIPHER_ROOT)

    try:
        spec = importlib.util.spec_from_file_location(
            "smgf_v17_cipher_matched_ablation_base",
            BASE_SOURCE,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Cannot import V17 base source: {BASE_SOURCE}"
            )
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


BASE = import_base()
CORE = BASE.CORE
torch = BASE.torch
Batch = BASE.Batch

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required by the frozen CipherSpectrum protocol."
    )

DEVICE = torch.device("cuda")


class Logger:
    def __init__(self, path: Path):
        self.path = Path(path)

    def log(self, message=""):
        msg = str(message)
        print(msg, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")


def rewire_outputs(logger: Logger) -> None:
    BASE.LOGGER = logger
    BASE.RUN_DIR = RUN_DIR
    BASE.LOG_FILE = LOG_FILE
    BASE.RESULT_FILE = PAPER_MD
    BASE.CONFIG_FILE = CONFIG_FILE
    BASE.SPLIT_FILE = SPLIT_FILE
    BASE.PER_SEED_FILE = PER_VARIANT_FILE
    BASE.CHECKPOINT_DIR = CHECKPOINT_DIR

    CORE.RUN_DIR = RUN_DIR
    CORE.LOG_FILE = LOG_FILE
    CORE.RESULT_MD = PAPER_MD
    CORE.SPLIT_MANIFEST_FILE = SPLIT_FILE
    CORE.CONFIG_FILE = CONFIG_FILE
    CORE.CHECKPOINT_DIR = CHECKPOINT_DIR


# =============================================================================
# 2. FORMAL FULL SEED42 CHECKPOINT DISCOVERY / VALIDATION
# =============================================================================

def discover_formal_full_seed42_checkpoint() -> Path:
    explicit = os.environ.get("SMGF_CIPHER_FULL_SEED42_CKPT")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    roots = []
    for root in [
        RESULT_ROOT,
        SCRIPT_DIR,
        Path("/home/admin1/lkfiles/RESULT"),
    ]:
        root = root.expanduser().resolve()
        if root not in roots:
            roots.append(root)

    found = []
    pattern = (
        "SMGF_V17_CIPHERSPECTRUM_FINAL_3SEED_*/"
        "checkpoints/FINAL_seed42_REUSED.pt"
    )
    for root in roots:
        if root.exists():
            found.extend(root.glob(pattern))

    found = [
        p for p in found
        if p.is_file()
        and sha256_file(p)
        == EXPECTED_FORMAL_FULL_SEED42_CKPT_SHA256
    ]
    if not found:
        raise FileNotFoundError(
            "Cannot find the formal CipherSpectrum seed42 FULL checkpoint "
            f"with SHA={EXPECTED_FORMAL_FULL_SEED42_CKPT_SHA256}. "
            "Set SMGF_CIPHER_FULL_SEED42_CKPT explicitly if needed."
        )

    return max(found, key=lambda p: p.stat().st_mtime)


def validate_full_checkpoint(path: Path, logger: Logger) -> dict:
    actual_sha = sha256_file(path)
    if actual_sha != EXPECTED_FORMAL_FULL_SEED42_CKPT_SHA256:
        raise RuntimeError(
            "Formal FULL seed42 checkpoint SHA mismatch. "
            f"expected={EXPECTED_FORMAL_FULL_SEED42_CKPT_SHA256}, "
            f"actual={actual_sha}"
        )

    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    if "model_state_dict" not in payload:
        raise RuntimeError(
            "Formal FULL seed42 checkpoint lacks model_state_dict."
        )

    seed = int(
        payload.get(
            "seed",
            payload.get("train_seed", -1),
        )
    )
    if seed != TRAIN_SEED:
        raise RuntimeError(
            f"Formal FULL checkpoint seed mismatch: {seed}"
        )

    if str(payload.get("gate_mode", "")) != "learned":
        raise RuntimeError(
            "Formal FULL checkpoint is not learned-gate."
        )
    if abs(
        float(payload.get("robust_batch_ratio", -1))
        - ROBUST_BATCH_RATIO
    ) > 1e-12:
        raise RuntimeError(
            "Formal FULL checkpoint robust ratio mismatch."
        )
    if abs(
        float(payload.get("weight_decay", -1))
        - WEIGHT_DECAY
    ) > 1e-12:
        raise RuntimeError(
            "Formal FULL checkpoint weight decay mismatch."
        )
    if abs(
        float(payload.get("attack_ce_weight", -1))
        - ATTACK_CE_WEIGHT
    ) > 1e-12:
        raise RuntimeError(
            "Formal FULL checkpoint attacked CE weight mismatch."
        )
    if abs(
        float(payload.get("kl_weight", -1))
        - KL_WEIGHT
    ) > 1e-12:
        raise RuntimeError(
            "Formal FULL checkpoint KL weight mismatch."
        )

    epoch_budget = int(
        payload.get(
            "full_epochs",
            payload.get("epochs", -1),
        )
    )
    if epoch_budget != EPOCHS:
        raise RuntimeError(
            f"Formal FULL epoch budget mismatch: {epoch_budget}"
        )

    observed_schedule = [
        list(x)
        for x in payload.get("lr_schedule", [])
    ]
    expected_schedule = [
        [1, 70, LR_HIGH],
        [71, 100, LR_LOW],
    ]
    if observed_schedule != expected_schedule:
        raise RuntimeError(
            "Formal FULL LR schedule mismatch: "
            f"{observed_schedule}"
        )

    logger.log(
        "[FULL checkpoint] PASS | "
        f"path={path} | SHA={actual_sha}"
    )
    return payload


# =============================================================================
# 3. DATA / SPLIT / SHARED PRIMARY ATTACK CACHE
# =============================================================================

def load_and_validate_data(logger: Logger):
    if not CIPHER_ROOT.is_dir():
        raise FileNotFoundError(
            f"CipherSpectrum path not found: {CIPHER_ROOT}"
        )

    (
        train_set,
        val_set,
        test_set,
        node_scaler,
        stats_scaler,
        num_classes,
        idx_to_class,
        manifest_hash,
    ) = BASE.load_and_prepare_data()

    if manifest_hash != EXPECTED_DATASET_MANIFEST_SHA256:
        raise RuntimeError(
            "CipherSpectrum dataset manifest mismatch. "
            f"expected={EXPECTED_DATASET_MANIFEST_SHA256}, "
            f"actual={manifest_hash}"
        )
    if num_classes != EXPECTED_NUM_CLASSES:
        raise RuntimeError(
            f"Class count drift: {num_classes} != {EXPECTED_NUM_CLASSES}"
        )

    counts = (
        len(train_set),
        len(val_set),
        len(test_set),
    )
    expected_counts = (
        EXPECTED_TRAIN_N,
        EXPECTED_VAL_N,
        EXPECTED_TEST_N,
    )
    if counts != expected_counts:
        raise RuntimeError(
            f"Split count drift: {counts} != {expected_counts}"
        )

    if not SPLIT_FILE.is_file():
        raise RuntimeError(
            "Matched ablation split manifest was not created."
        )
    split_sha = sha256_file(SPLIT_FILE)
    if split_sha != EXPECTED_FORMAL_SPLIT_SHA256:
        raise RuntimeError(
            "Split manifest differs from formal CipherSpectrum run. "
            f"expected={EXPECTED_FORMAL_SPLIT_SHA256}, "
            f"actual={split_sha}"
        )

    BASE.validate_zero_strength_noop(
        test_set,
        node_scaler,
        stats_scaler,
        max_samples=16,
    )

    logger.log(
        "[data] PASS | "
        f"manifest={manifest_hash} | split_sha={split_sha} | "
        f"Train/Val/Test={counts}"
    )

    return {
        "train_set": train_set,
        "val_set": val_set,
        "test_set": test_set,
        "node_scaler": node_scaler,
        "stats_scaler": stats_scaler,
        "num_classes": num_classes,
        "idx_to_class": idx_to_class,
        "manifest_hash": manifest_hash,
        "split_sha256": split_sha,
    }


def build_primary_attack_cache(data_info: dict, logger: Logger):
    test_set = data_info["test_set"]
    node_scaler = data_info["node_scaler"]
    stats_scaler = data_info["stats_scaler"]

    cache = {}
    specs = [
        ("mixed", PRIMARY_MIXED_STRENGTH),
        ("adaptive", PRIMARY_ADAPTIVE_STRENGTH),
    ]

    logger.log("")
    logger.log("=" * 104)
    logger.log(
        "BUILD ONE SHARED MATCHED EVALUATION CACHE "
        "(Mixed@1.0 + Adaptive@1.0)"
    )
    logger.log("=" * 104)

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
            f"MATCHED_ABLATION:{attack}@{strength}",
        )
        if attack == "adaptive":
            CORE.validate_adaptive_semantics(
                test_set,
                attacked,
            )

        cache[(attack, float(strength))] = attacked
        logger.log(
            f"[eval cache] {attack}@{strength:.1f} | "
            f"seed={attack_seed} | n={len(attacked)}"
        )

    logger.log(
        "[eval cache] PASS: exact same attacked test objects "
        "will be shared by all four variants"
    )
    return cache


# =============================================================================
# 4. MATCHED MODEL / TRAINING HELPERS
# =============================================================================

def lr_at(epoch: int) -> float:
    return LR_HIGH if epoch < LR_SWITCH_EPOCH else LR_LOW


def cpu_model_state(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def cpu_optimizer_state(optimizer):
    state = copy.deepcopy(optimizer.state_dict())
    for value in state.get("state", {}).values():
        for key, item in list(value.items()):
            if torch.is_tensor(item):
                value[key] = item.detach().cpu().clone()
    return state


def new_model(num_classes: int, gate_mode: str):
    return BASE.SMGFNetV17(
        node_in_channels=5,
        stats_dim=9,
        finger_seq_len=20,
        entropy_seq_len=5,
        hidden_channels=32,
        out_channels=num_classes,
        gate_mode=gate_mode,
    ).to(DEVICE)


def build_seed42_initial_state_reference(
    num_classes: int,
    formal_payload: dict,
    logger: Logger,
):
    CORE.set_seed(TRAIN_SEED)
    model = new_model(
        num_classes,
        gate_mode="learned",
    )
    init_sha = BASE.model_state_sha256(model)
    del model
    torch.cuda.empty_cache()

    if init_sha != EXPECTED_INITIAL_STATE_SHA256:
        raise RuntimeError(
            "Seed42 initialization no longer reproduces the frozen "
            "CipherSpectrum reference initialization. "
            f"expected={EXPECTED_INITIAL_STATE_SHA256}, actual={init_sha}"
        )

    archived_sha = formal_payload.get("initial_state_sha256")
    if archived_sha and archived_sha != EXPECTED_INITIAL_STATE_SHA256:
        raise RuntimeError(
            "Formal FULL checkpoint initial-state SHA differs from the "
            "frozen reference. "
            f"checkpoint={archived_sha}, expected={EXPECTED_INITIAL_STATE_SHA256}"
        )

    logger.log(
        "[initialization] PASS: all matched variants must start from "
        f"SHA={EXPECTED_INITIAL_STATE_SHA256}"
    )
    return init_sha


def train_ablation_variant(
    data_info: dict,
    tag: str,
    gate_mode: str,
    use_robust_batches: bool,
    expected_init_sha: str,
    logger: Logger,
):
    train_set = data_info["train_set"]
    val_set = data_info["val_set"]
    node_scaler = data_info["node_scaler"]
    stats_scaler = data_info["stats_scaler"]
    num_classes = data_info["num_classes"]

    CORE.set_seed(TRAIN_SEED)
    model = new_model(
        num_classes,
        gate_mode=gate_mode,
    )
    init_sha = BASE.model_state_sha256(model)

    # GatedFusionV13 instantiates identical parameters in learned and identity
    # modes, so all four conditions must begin from the same parameter state.
    if init_sha != expected_init_sha:
        raise RuntimeError(
            f"{tag}: initial state is not matched. "
            f"expected={expected_init_sha}, actual={init_sha}"
        )

    loader = CORE.make_loader(
        train_set,
        BATCH_SIZE,
        shuffle=True,
        seed=TRAIN_SEED + 21_000,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR_HIGH,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    best_opt = None
    history = []

    logger.log("")
    logger.log("=" * 104)
    logger.log(
        f"TRAIN MATCHED ABLATION: {tag} | "
        f"seed={TRAIN_SEED} | gate={gate_mode} | "
        f"robust_training={use_robust_batches}"
    )
    logger.log(
        "100 epochs | LR 0.002 -> 0.0005 @ epoch71 | "
        "AdamW WD=1e-4 | checkpoint=min clean-val CE"
    )
    if use_robust_batches:
        logger.log(
            "robust batches=0.20 | Mixed U[0.3,0.9] | "
            "loss=clean CE + attacked-label CE | NO_KL"
        )
    else:
        logger.log(
            "robust batches=OFF | no attacked training samples are "
            "generated or forwarded | loss=clean CE only"
        )
    logger.log("=" * 104)

    for epoch_zero in range(EPOCHS):
        epoch = epoch_zero + 1
        lr = lr_at(epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr

        model.train()
        total_loss = 0.0
        clean_total = 0.0
        attack_total = 0.0
        batches = 0
        robust_batches = 0

        for batch_idx, batch_cpu in enumerate(loader):
            attacked_batch = None

            if use_robust_batches:
                selector = CORE.stable_seed(
                    "v17_robust_batch",
                    TRAIN_SEED,
                    epoch_zero,
                    batch_idx,
                    modulo=1_000_000,
                ) / 1_000_000.0

                if selector < ROBUST_BATCH_RATIO:
                    attack_seed = CORE.stable_seed(
                        "v17_train_attack",
                        TRAIN_SEED,
                        epoch_zero,
                        batch_idx,
                    )
                    strength_rng = np.random.default_rng(
                        CORE.stable_seed(
                            "v17_train_strength",
                            TRAIN_SEED,
                            epoch_zero,
                            batch_idx,
                        )
                    )
                    strength = float(
                        strength_rng.uniform(
                            ROBUST_STRENGTH_MIN,
                            ROBUST_STRENGTH_MAX,
                        )
                    )
                    np_rng = np.random.default_rng(
                        attack_seed
                    )
                    py_rng = random.Random(
                        attack_seed
                    )

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
                    attacked_batch = (
                        Batch.from_data_list(
                            attacked_list
                        ).to(DEVICE)
                    )

            batch = batch_cpu.to(DEVICE)
            optimizer.zero_grad()

            clean_logits = BASE.forward_logits(
                model,
                batch,
            )
            clean_ce = criterion(
                clean_logits,
                batch.y.view(-1),
            )
            loss = clean_ce
            attack_ce_value = 0.0

            if attacked_batch is not None:
                attacked_logits = BASE.forward_logits(
                    model,
                    attacked_batch,
                )
                attack_ce = criterion(
                    attacked_logits,
                    attacked_batch.y.view(-1),
                )
                loss = (
                    clean_ce
                    + ATTACK_CE_WEIGHT * attack_ce
                )
                attack_ce_value = float(
                    attack_ce.item()
                )
                robust_batches += 1

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            clean_total += float(clean_ce.item())
            attack_total += attack_ce_value
            batches += 1

        val_loss = CORE.compute_supervised_loss(
            model,
            val_set,
            batch_size=BATCH_SIZE,
        )
        is_new_best = (
            val_loss < best_val - 1e-8
        )

        if is_new_best:
            best_val = float(val_loss)
            best_epoch = epoch
            best_state = cpu_model_state(model)
            best_opt = cpu_optimizer_state(
                optimizer
            )

        row = {
            "epoch": epoch,
            "lr": lr,
            "loss": total_loss / max(batches, 1),
            "clean_ce": (
                clean_total
                / max(batches, 1)
            ),
            "attack_ce": (
                attack_total
                / max(robust_batches, 1)
                if use_robust_batches
                else 0.0
            ),
            "robust_batches": robust_batches,
            "total_batches": batches,
            "val_loss": float(val_loss),
            "is_new_best": bool(is_new_best),
            "best_epoch_so_far": best_epoch,
            "best_val_loss_so_far": best_val,
        }
        history.append(row)

        logger.log(
            f"[{tag}][seed=42] "
            f"epoch {epoch:03d}/100 | "
            f"lr={lr:.7f} | "
            f"loss={row['loss']:.6f} | "
            f"clean_ce={row['clean_ce']:.6f} | "
            f"attack_ce={row['attack_ce']:.6f} | "
            f"robust_batches={robust_batches}/{batches} | "
            f"val={val_loss:.6f} | "
            f"best_ep={best_epoch} | "
            f"best={best_val:.6f}"
        )

    if best_state is None:
        raise RuntimeError(
            f"{tag}: no best checkpoint captured."
        )

    history_path = (
        HISTORY_DIR
        / f"{tag}_seed42.csv"
    )
    with history_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                history[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(history)

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"{tag}_seed42.pt"
    )
    torch.save(
        {
            "model_state_dict": best_state,
            "optimizer_state_dict": best_opt,
            "tag": tag,
            "dataset": DATASET_NAME,
            "train_seed": TRAIN_SEED,
            "split_seed": SPLIT_SEED,
            "epochs": EPOCHS,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "lr_schedule": [
                [1, 70, LR_HIGH],
                [71, 100, LR_LOW],
            ],
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "gate_mode": gate_mode,
            "use_robust_batches":
                use_robust_batches,
            "robust_batch_ratio": (
                ROBUST_BATCH_RATIO
                if use_robust_batches
                else 0.0
            ),
            "robust_strength_range": (
                [
                    ROBUST_STRENGTH_MIN,
                    ROBUST_STRENGTH_MAX,
                ]
                if use_robust_batches
                else None
            ),
            "train_attack": (
                "mixed"
                if use_robust_batches
                else None
            ),
            "attack_ce_weight": (
                ATTACK_CE_WEIGHT
                if use_robust_batches
                else 0.0
            ),
            "kl_weight": KL_WEIGHT,
            "checkpoint_rule":
                "minimum clean-validation CE",
            "initial_state_sha256":
                init_sha,
            "base_source_sha256":
                EXPECTED_BASE_SHA256,
            "core_source_sha256":
                EXPECTED_CORE_SHA256,
            "dataset_manifest_sha256":
                data_info["manifest_hash"],
            "split_manifest_sha256":
                data_info["split_sha256"],
        },
        checkpoint_path,
    )

    logger.log(
        f"[{tag}] TRAIN COMPLETE | "
        f"best_epoch={best_epoch} | "
        f"best_val={best_val:.12f} | "
        f"checkpoint={checkpoint_path}"
    )

    del model
    torch.cuda.empty_cache()

    return {
        "tag": tag,
        "checkpoint": str(
            checkpoint_path
        ),
        "checkpoint_sha256":
            sha256_file(checkpoint_path),
        "history": str(history_path),
        "history_sha256":
            sha256_file(history_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "initial_state_sha256":
            init_sha,
        "gate_mode": gate_mode,
        "use_robust_batches":
            use_robust_batches,
    }


# =============================================================================
# 5. SHARED MATCHED EVALUATION
# =============================================================================

def load_model_from_checkpoint(
    path: Path,
    num_classes: int,
    gate_mode: str,
):
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(
            path,
            map_location="cpu",
        )

    model = new_model(
        num_classes,
        gate_mode=gate_mode,
    )
    model.load_state_dict(
        payload["model_state_dict"],
        strict=True,
    )
    return model, payload


def scalar_clean(result):
    return {
        "acc": float(result["acc"]),
        "macro_f1": float(
            result["macro_f1"]
        ),
        "ece": float(result["ece"]),
    }


def scalar_attack(result):
    return {
        "robust_acc": float(
            result["robust_acc"]
        ),
        "robust_f1": float(
            result["robust_f1"]
        ),
        "asr": float(result["asr"]),
        "rel_drop": float(
            result["rel_drop"]
        ),
        "ece": float(result["ece"]),
    }


def evaluate_variant(
    tag: str,
    checkpoint_path: Path,
    gate_mode: str,
    data_info: dict,
    attack_cache: dict,
    logger: Logger,
):
    model, payload = (
        load_model_from_checkpoint(
            checkpoint_path,
            data_info["num_classes"],
            gate_mode,
        )
    )

    clean_raw = CORE.evaluate_clean(
        model,
        data_info["test_set"],
    )
    CORE.log_clean_result(
        logger,
        tag,
        TRAIN_SEED,
        clean_raw,
    )

    mixed_raw = CORE.evaluate_attack(
        model,
        attack_cache[
            (
                "mixed",
                PRIMARY_MIXED_STRENGTH,
            )
        ],
        clean_raw,
    )
    CORE.log_attack_result(
        logger,
        tag,
        TRAIN_SEED,
        "mixed",
        PRIMARY_MIXED_STRENGTH,
        mixed_raw,
    )

    adaptive_raw = CORE.evaluate_attack(
        model,
        attack_cache[
            (
                "adaptive",
                PRIMARY_ADAPTIVE_STRENGTH,
            )
        ],
        clean_raw,
    )
    CORE.log_attack_result(
        logger,
        tag,
        TRAIN_SEED,
        "adaptive",
        PRIMARY_ADAPTIVE_STRENGTH,
        adaptive_raw,
    )

    result = {
        "tag": tag,
        "train_seed": TRAIN_SEED,
        "gate_mode": gate_mode,
        "best_epoch": int(
            payload.get(
                "best_epoch",
                -1,
            )
        ),
        "best_val_loss": float(
            payload.get(
                "best_val_loss",
                float("nan"),
            )
        ),
        "checkpoint":
            str(checkpoint_path),
        "checkpoint_sha256":
            sha256_file(
                checkpoint_path
            ),
        "clean": scalar_clean(
            clean_raw
        ),
        "mixed@1.0": scalar_attack(
            mixed_raw
        ),
        "adaptive@1.0": scalar_attack(
            adaptive_raw
        ),
    }

    del model
    torch.cuda.empty_cache()
    return result


def assert_full_reproduces_formal(
    full_result: dict,
    logger: Logger,
):
    observed = {
        "clean_acc":
            full_result["clean"]["acc"],
        "clean_f1":
            full_result["clean"]["macro_f1"],
        "clean_ece":
            full_result["clean"]["ece"],
        "mixed_robust_acc":
            full_result[
                "mixed@1.0"
            ]["robust_acc"],
        "mixed_robust_f1":
            full_result[
                "mixed@1.0"
            ]["robust_f1"],
        "mixed_asr":
            full_result[
                "mixed@1.0"
            ]["asr"],
        "mixed_rel_drop":
            full_result[
                "mixed@1.0"
            ]["rel_drop"],
        "mixed_ece":
            full_result[
                "mixed@1.0"
            ]["ece"],
        "adaptive_robust_acc":
            full_result[
                "adaptive@1.0"
            ]["robust_acc"],
        "adaptive_robust_f1":
            full_result[
                "adaptive@1.0"
            ]["robust_f1"],
        "adaptive_asr":
            full_result[
                "adaptive@1.0"
            ]["asr"],
        "adaptive_rel_drop":
            full_result[
                "adaptive@1.0"
            ]["rel_drop"],
        "adaptive_ece":
            full_result[
                "adaptive@1.0"
            ]["ece"],
    }

    for key, expected in (
        EXPECTED_FULL_SEED42.items()
    ):
        actual = observed[key]
        if not math.isclose(
            actual,
            expected,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise RuntimeError(
                "FULL seed42 does not reproduce "
                "the formal archived result: "
                f"{key}: expected={expected}, "
                f"actual={actual}"
            )

    logger.log(
        "[FULL matched reproduction] PASS: "
        "all Clean/Mixed@1/Adaptive@1 "
        "metrics reproduce the formal seed42 result"
    )


# =============================================================================
# 6. FACTORIAL EFFECTS / PAPER OUTPUT
# =============================================================================

def primary_metric_map(result: dict):
    return {
        "clean_acc":
            result["clean"]["acc"],
        "clean_f1":
            result["clean"]["macro_f1"],
        "mixed_robust_acc":
            result[
                "mixed@1.0"
            ]["robust_acc"],
        "mixed_robust_f1":
            result[
                "mixed@1.0"
            ]["robust_f1"],
        "adaptive_robust_acc":
            result[
                "adaptive@1.0"
            ]["robust_acc"],
        "adaptive_robust_f1":
            result[
                "adaptive@1.0"
            ]["robust_f1"],
    }


def subtract_metric_maps(
    a: dict,
    b: dict,
):
    return {
        key: float(a[key] - b[key])
        for key in a
    }


def compute_factorial_effects(
    results_by_tag: dict,
):
    full = primary_metric_map(
        results_by_tag["FULL"]
    )
    no_gate = primary_metric_map(
        results_by_tag[
            "STRICT_NO_GATE"
        ]
    )
    clean_only = primary_metric_map(
        results_by_tag["CLEAN_ONLY"]
    )
    no_gate_clean = primary_metric_map(
        results_by_tag[
            "NO_GATE_CLEAN_ONLY"
        ]
    )

    gate_under_robust = (
        subtract_metric_maps(
            full,
            no_gate,
        )
    )
    gate_without_robust = (
        subtract_metric_maps(
            clean_only,
            no_gate_clean,
        )
    )
    robust_with_gate = (
        subtract_metric_maps(
            full,
            clean_only,
        )
    )
    robust_without_gate = (
        subtract_metric_maps(
            no_gate,
            no_gate_clean,
        )
    )
    interaction = {
        key: (
            gate_under_robust[key]
            - gate_without_robust[key]
        )
        for key in gate_under_robust
    }

    return {
        "definition": (
            "All effects are percentage-point "
            "differences at fixed training seed42. "
            "Positive means the named component "
            "increases the corresponding metric."
        ),
        "gate_effect_under_robust_training":
            gate_under_robust,
        "gate_effect_without_robust_training":
            gate_without_robust,
        "robust_training_effect_with_gate":
            robust_with_gate,
        "robust_training_effect_without_gate":
            robust_without_gate,
        "gate_x_robust_interaction":
            interaction,
    }


def write_paper_csv(
    results_by_tag: dict,
):
    fieldnames = [
        "Variant",
        "Gate",
        "RobustTraining",
        "TrainSeed",
        "BestEpoch",
        "BestValCE",
        "CleanAcc",
        "CleanF1",
        "MixedRobAcc",
        "MixedRobF1",
        "AdaptiveRobAcc",
        "AdaptiveRobF1",
        "CleanECE",
        "MixedASR",
        "MixedRelDrop",
        "MixedECE",
        "AdaptiveASR",
        "AdaptiveRelDrop",
        "AdaptiveECE",
    ]

    with PAPER_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for spec in VARIANT_SPECS:
            result = results_by_tag[
                spec["tag"]
            ]
            writer.writerow(
                {
                    "Variant":
                        spec["tag"],
                    "Gate":
                        spec["gate_mode"],
                    "RobustTraining":
                        (
                            "ON"
                            if spec[
                                "use_robust_batches"
                            ]
                            else "OFF"
                        ),
                    "TrainSeed":
                        TRAIN_SEED,
                    "BestEpoch":
                        result[
                            "best_epoch"
                        ],
                    "BestValCE":
                        result[
                            "best_val_loss"
                        ],
                    "CleanAcc":
                        result[
                            "clean"
                        ]["acc"],
                    "CleanF1":
                        result[
                            "clean"
                        ]["macro_f1"],
                    "MixedRobAcc":
                        result[
                            "mixed@1.0"
                        ]["robust_acc"],
                    "MixedRobF1":
                        result[
                            "mixed@1.0"
                        ]["robust_f1"],
                    "AdaptiveRobAcc":
                        result[
                            "adaptive@1.0"
                        ]["robust_acc"],
                    "AdaptiveRobF1":
                        result[
                            "adaptive@1.0"
                        ]["robust_f1"],
                    "CleanECE":
                        result[
                            "clean"
                        ]["ece"],
                    "MixedASR":
                        result[
                            "mixed@1.0"
                        ]["asr"],
                    "MixedRelDrop":
                        result[
                            "mixed@1.0"
                        ]["rel_drop"],
                    "MixedECE":
                        result[
                            "mixed@1.0"
                        ]["ece"],
                    "AdaptiveASR":
                        result[
                            "adaptive@1.0"
                        ]["asr"],
                    "AdaptiveRelDrop":
                        result[
                            "adaptive@1.0"
                        ]["rel_drop"],
                    "AdaptiveECE":
                        result[
                            "adaptive@1.0"
                        ]["ece"],
                }
            )


def write_paper_md(
    results_by_tag: dict,
    factorial_effects: dict,
):
    lines = [
        "# CipherSpectrum V17 Matched Ablation — Seed42",
        "",
        "This is a fixed-seed component-diagnostic experiment, not a "
        "replacement for the formal three-seed main-model results.",
        "",
        "Main results use seeds 42/43/44. All ablation rows here use "
        "training seed42 so that component effects are compared under "
        "matched stochastic conditions.",
        "",
        "## Matched protocol",
        "",
        "- Dataset: CipherSpectrum, 41 classes / 41,000 samples.",
        "- Split seed: 42.",
        "- Train/Validation/Test: 29,520 / 3,280 / 8,200.",
        "- Training seed: 42 for every row.",
        "- 100 epochs; no early stopping.",
        "- LR: 0.002 for epochs 1-70; 0.0005 for 71-100.",
        "- AdamW weight decay: 1e-4.",
        "- Checkpoint: minimum clean-validation CE.",
        "- Robust condition: 20% Mixed attacked batches, "
        "strength Uniform[0.3,0.9], attacked-label CE weight=1.",
        "- Clean-only condition: no attacked samples are generated or "
        "forwarded during training.",
        "- KL weight=0 in every condition.",
        "- One shared deterministic Mixed@1.0 and Adaptive@1.0 test "
        "cache is used by all rows.",
        "",
        "## Paper-facing compact ablation table",
        "",
        "| Variant | Gate | Robust train | Clean Acc | Clean F1 | "
        "Mixed RobAcc | Mixed RobF1 | Adaptive RobAcc | Adaptive RobF1 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for spec in VARIANT_SPECS:
        result = results_by_tag[
            spec["tag"]
        ]
        lines.append(
            "| {tag} | {gate} | {robust} | "
            "{ca:.2f} | {cf:.2f} | "
            "{ma:.2f} | {mf:.2f} | "
            "{aa:.2f} | {af:.2f} |".format(
                tag=spec["tag"],
                gate=(
                    "Learned"
                    if spec[
                        "gate_mode"
                    ] == "learned"
                    else "Identity"
                ),
                robust=(
                    "Yes"
                    if spec[
                        "use_robust_batches"
                    ]
                    else "No"
                ),
                ca=result[
                    "clean"
                ]["acc"],
                cf=result[
                    "clean"
                ]["macro_f1"],
                ma=result[
                    "mixed@1.0"
                ]["robust_acc"],
                mf=result[
                    "mixed@1.0"
                ]["robust_f1"],
                aa=result[
                    "adaptive@1.0"
                ]["robust_acc"],
                af=result[
                    "adaptive@1.0"
                ]["robust_f1"],
            )
        )

    lines.extend(
        [
            "",
            "## Factorial diagnostic effects",
            "",
            "Effects below are percentage-point differences at fixed "
            "seed42; they are diagnostic contrasts, not uncertainty "
            "estimates.",
            "",
        ]
    )

    effect_order = [
        (
            "Gate effect under robust training",
            "gate_effect_under_robust_training",
        ),
        (
            "Gate effect without robust training",
            "gate_effect_without_robust_training",
        ),
        (
            "Robust-training effect with learned Gate",
            "robust_training_effect_with_gate",
        ),
        (
            "Robust-training effect without Gate",
            "robust_training_effect_without_gate",
        ),
        (
            "Gate × robust-training interaction",
            "gate_x_robust_interaction",
        ),
    ]

    for title, key in effect_order:
        effect = factorial_effects[key]
        lines.append(
            "- {title}: CleanAcc {ca:+.2f} pp; "
            "MixedRobAcc {ma:+.2f} pp; "
            "AdaptiveRobAcc {aa:+.2f} pp.".format(
                title=title,
                ca=effect["clean_acc"],
                ma=effect[
                    "mixed_robust_acc"
                ],
                aa=effect[
                    "adaptive_robust_acc"
                ],
            )
        )

    lines.extend(
        [
            "",
            "## Reporting note",
            "",
            "Recommended manuscript statement:",
            "",
            "Main performance is reported over three training seeds "
            "(42, 43, and 44). To isolate architectural and training-"
            "component effects under matched stochastic conditions, "
            "all ablation experiments use the fixed training seed 42 "
            "while retaining the identical data split, optimization "
            "schedule, checkpoint criterion, and deterministic attack "
            "realizations.",
            "",
            "If a component contrast is smaller than the seed-to-seed "
            "dispersion already observed for the FULL main model, avoid "
            "claiming that the difference is stable; selectively add "
            "seeds43/44 for that ambiguous contrast if needed.",
        ]
    )

    PAPER_MD.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_config(
    data_info: dict,
    full_source: Path,
    full_payload: dict,
    expected_init_sha: str,
):
    config = {
        "experiment":
            "CipherSpectrum matched 2x2 ablation",
        "dataset": DATASET_NAME,
        "dataset_root": str(
            CIPHER_ROOT
        ),
        "dataset_manifest_sha256":
            data_info["manifest_hash"],
        "split_manifest_sha256":
            data_info["split_sha256"],
        "split_seed": SPLIT_SEED,
        "train_seed": TRAIN_SEED,
        "split_counts": {
            "train": EXPECTED_TRAIN_N,
            "validation":
                EXPECTED_VAL_N,
            "test": EXPECTED_TEST_N,
        },
        "source_sha256": {
            "V17_base":
                EXPECTED_BASE_SHA256,
            "V16_core":
                EXPECTED_CORE_SHA256,
        },
        "formal_full_seed42": {
            "source_checkpoint":
                str(full_source),
            "checkpoint_sha256":
                EXPECTED_FORMAL_FULL_SEED42_CKPT_SHA256,
            "best_epoch":
                int(
                    full_payload.get(
                        "best_epoch",
                        -1,
                    )
                ),
            "best_val_loss":
                float(
                    full_payload.get(
                        "best_val_loss",
                        float("nan"),
                    )
                ),
        },
        "matched_initial_state_sha256":
            expected_init_sha,
        "frozen_protocol": {
            "epochs": EPOCHS,
            "early_stopping": False,
            "lr_schedule": [
                [1, 70, LR_HIGH],
                [71, 100, LR_LOW],
            ],
            "optimizer": "AdamW",
            "weight_decay":
                WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "robust_batch_ratio":
                ROBUST_BATCH_RATIO,
            "train_attack": "mixed",
            "robust_strength_distribution":
                "Uniform[0.3,0.9]",
            "attack_ce_weight":
                ATTACK_CE_WEIGHT,
            "kl_weight": KL_WEIGHT,
            "checkpoint_rule":
                "minimum clean-validation CE",
        },
        "variants": VARIANT_SPECS,
        "primary_evaluation": [
            "Clean",
            "Mixed@1.0",
            "Adaptive@1.0",
        ],
    }

    CONFIG_FILE.write_text(
        json.dumps(
            config,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# =============================================================================
# 7. MAIN
# =============================================================================

def main():
    RUN_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = Logger(LOG_FILE)
    rewire_outputs(logger)

    logger.log("=" * 104)
    logger.log(
        "CipherSpectrum V17 MATCHED 2x2 ABLATION | seed42"
    )
    logger.log(
        "FULL / STRICT_NO_GATE / CLEAN_ONLY / NO_GATE_CLEAN_ONLY"
    )
    logger.log(
        "Only Gate and robust attacked-CE training are manipulated."
    )
    logger.log(
        "No LR/WD/ratio/epoch/checkpoint-rule tuning."
    )
    logger.log(f"Run directory: {RUN_DIR}")
    logger.log(f"Dataset root: {CIPHER_ROOT}")
    logger.log(
        f"V17 SHA: {sha256_file(BASE_SOURCE)}"
    )
    logger.log(
        f"V16 SHA: {sha256_file(CORE_SOURCE)}"
    )
    logger.log("=" * 104)

    full_source = (
        discover_formal_full_seed42_checkpoint()
    )
    full_payload = (
        validate_full_checkpoint(
            full_source,
            logger,
        )
    )

    data_info = (
        load_and_validate_data(
            logger
        )
    )
    attack_cache = (
        build_primary_attack_cache(
            data_info,
            logger,
        )
    )

    expected_init_sha = (
        build_seed42_initial_state_reference(
            data_info["num_classes"],
            full_payload,
            logger,
        )
    )

    # Copy the already formalized FULL seed42 checkpoint into this
    # ablation archive. FULL is deliberately NOT retrained.
    full_copy = (
        CHECKPOINT_DIR
        / "FULL_seed42_REUSED.pt"
    )
    shutil.copy2(
        full_source,
        full_copy,
    )
    if (
        sha256_file(full_copy)
        != EXPECTED_FORMAL_FULL_SEED42_CKPT_SHA256
    ):
        raise RuntimeError(
            "Copied FULL seed42 checkpoint SHA changed."
        )

    write_config(
        data_info,
        full_source,
        full_payload,
        expected_init_sha,
    )

    train_records = {
        "FULL": {
            "tag": "FULL",
            "reused": True,
            "checkpoint":
                str(full_copy),
            "checkpoint_sha256":
                sha256_file(full_copy),
            "best_epoch":
                int(
                    full_payload.get(
                        "best_epoch",
                        -1,
                    )
                ),
            "best_val_loss":
                float(
                    full_payload.get(
                        "best_val_loss",
                        float("nan"),
                    )
                ),
            "initial_state_sha256":
                expected_init_sha,
            "gate_mode": "learned",
            "use_robust_batches":
                True,
        }
    }

    # Train only the three new matched variants.
    for spec in VARIANT_SPECS:
        if spec[
            "reuse_formal_checkpoint"
        ]:
            continue

        record = train_ablation_variant(
            data_info=data_info,
            tag=spec["tag"],
            gate_mode=
                spec["gate_mode"],
            use_robust_batches=
                spec[
                    "use_robust_batches"
                ],
            expected_init_sha=
                expected_init_sha,
            logger=logger,
        )
        train_records[
            spec["tag"]
        ] = record

    # Evaluate ALL four rows using the exact same clean test set and
    # the exact same two attacked test datasets.
    results_by_tag = {}

    for spec in VARIANT_SPECS:
        tag = spec["tag"]
        checkpoint_path = Path(
            train_records[
                tag
            ]["checkpoint"]
        )

        logger.log("")
        logger.log(
            f"FORMAL MATCHED EVALUATION: {tag}"
        )

        result = evaluate_variant(
            tag=tag,
            checkpoint_path=
                checkpoint_path,
            gate_mode=
                spec["gate_mode"],
            data_info=data_info,
            attack_cache=
                attack_cache,
            logger=logger,
        )
        results_by_tag[
            tag
        ] = result

        if tag == "FULL":
            assert_full_reproduces_formal(
                result,
                logger,
            )

    factorial_effects = (
        compute_factorial_effects(
            results_by_tag
        )
    )

    PER_VARIANT_FILE.write_text(
        json.dumps(
            {
                "dataset":
                    DATASET_NAME,
                "train_seed":
                    TRAIN_SEED,
                "split_seed":
                    SPLIT_SEED,
                "dataset_manifest_sha256":
                    data_info[
                        "manifest_hash"
                    ],
                "split_manifest_sha256":
                    data_info[
                        "split_sha256"
                    ],
                "formal_full_source":
                    str(full_source),
                "training":
                    train_records,
                "results":
                    results_by_tag,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    FACTORIAL_FILE.write_text(
        json.dumps(
            factorial_effects,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    write_paper_csv(
        results_by_tag
    )
    write_paper_md(
        results_by_tag,
        factorial_effects,
    )

    run_manifest = {
        "experiment":
            "CipherSpectrum matched 2x2 ablation",
        "paper_role":
            "fixed-seed component diagnostic",
        "train_seed": TRAIN_SEED,
        "split_seed": SPLIT_SEED,
        "dataset_manifest_sha256":
            data_info[
                "manifest_hash"
            ],
        "split_manifest_sha256":
            data_info[
                "split_sha256"
            ],
        "source_sha256": {
            "this_script":
                sha256_file(
                    Path(__file__).resolve()
                ),
            "V17_base":
                EXPECTED_BASE_SHA256,
            "V16_core":
                EXPECTED_CORE_SHA256,
        },
        "matched_initial_state_sha256":
            expected_init_sha,
        "formal_full_checkpoint_sha256":
            EXPECTED_FORMAL_FULL_SEED42_CKPT_SHA256,
        "artifacts": {
            "config":
                str(CONFIG_FILE),
            "per_variant_results":
                str(PER_VARIANT_FILE),
            "paper_csv":
                str(PAPER_CSV),
            "paper_md":
                str(PAPER_MD),
            "factorial_effects":
                str(FACTORIAL_FILE),
        },
        "checkpoints": {
            tag: {
                "path":
                    record[
                        "checkpoint"
                    ],
                "sha256":
                    record[
                        "checkpoint_sha256"
                    ],
            }
            for tag, record
            in train_records.items()
        },
    }

    MANIFEST_FILE.write_text(
        json.dumps(
            run_manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    logger.log("")
    logger.log("=" * 104)
    logger.log(
        "CIPHERSPECTRUM MATCHED ABLATION COMPLETE"
    )
    logger.log(
        f"Paper table: {PAPER_CSV}"
    )
    logger.log(
        f"Paper summary: {PAPER_MD}"
    )
    logger.log(
        f"Per-variant JSON: {PER_VARIANT_FILE}"
    )
    logger.log(
        f"Factorial effects: {FACTORIAL_FILE}"
    )
    logger.log("=" * 104)


if __name__ == "__main__":
    main()
