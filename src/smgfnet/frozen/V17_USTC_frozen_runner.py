# -*- coding: utf-8 -*-
"""
SMGF-Net V17 on USTC-TFC2016 -- frozen-protocol runner.

Purpose
-------
Run the already-frozen V17 protocol on USTC-TFC2016 without any
dataset-specific hyperparameter tuning.

Default behavior is a deterministic 10% per-class seed42 pilot.
Set SMGF_USTC_PILOT_FRACTION=1.0 for a full-dataset run.
Set SMGF_USTC_TRAIN_SEED=43 or 44 later to reuse exactly the same code.

Important invariants
--------------------
* Dataset root is fixed to /home/admin1/dataset/USTC-TFC2016-Split.
* This file MUST live in the same directory as:
    V17-SMGF-Net_final_paper_experiments.py
    V16-SMGF-Net_final_paper_experiments.py
* Model/training protocol is frozen from the final CipherSpectrum run:
    V17 + learned Gate + attacked-label CE + NO_KL
    100 epochs, no early stopping
    epochs 1-70: lr=0.002
    epochs 71-100: lr=0.0005
    AdamW weight_decay=1e-4
    robust batch ratio=0.20
    Mixed train strength Uniform[0.3,0.9]
    attacked CE weight=1.0
    checkpoint=min clean-validation CE
* Stratified split is always 72/8/20 with split seed 42.
* Pilot subset selection is independent of training seed.
* Raw graph extraction is cached locally so later reruns do not re-parse PCAPs.
* USTC reconstruction validation uses a documented 1e-3 edge tolerance.
  The original V16 threshold (1e-4) is too strict for USTC float32 IAT
  round-trip reconstruction; x/stats tolerances remain 1e-3.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


# =============================================================================
# 0. IMMUTABLE SOURCES / PATHS / FROZEN PROTOCOL
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

# Confirmed server dataset location. No path guessing is used.
USTC_ROOT = Path("/home/admin1/dataset/USTC-TFC2016-Split")

# Code, cache and RESULT are deliberately colocated under the same run directory.
RESULT_ROOT = SCRIPT_DIR / "RESULT"
CACHE_ROOT = SCRIPT_DIR / "USTC_CACHE"

DATASET_NAME = "USTC-TFC2016"
BENCHMARK_VARIANT = "USTC-TFC2016 archived 12-class / 21,256-flow subset"
EXPECTED_CLASS_COUNTS = {
    "Cridex": 2000,
    "Geodo": 2000,
    "Gmail": 1409,
    "Htbot": 2000,
    "Miuref": 2000,
    "Neris": 2000,
    "Nsis": 2000,
    "Shifu": 1518,
    "Tinba": 329,
    "Virut": 2000,
    "WorldOfWarcraft": 2000,
    "Zeus": 2000,
}
EXPECTED_TOTAL_FLOWS = 21256
TRAIN_SEED = int(os.environ.get("SMGF_USTC_TRAIN_SEED", "42"))
if TRAIN_SEED not in {42, 43, 44}:
    raise ValueError("SMGF_USTC_TRAIN_SEED must be one of 42, 43, 44.")

SPLIT_SEED = 42
PILOT_SELECTION_SEED = 42
ATTACK_BASE_SEED = 20260919

TRAIN_RATIO = 0.72
VAL_RATIO = 0.08
TEST_RATIO = 0.20

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
GATE_MODE = "learned"

# Default pilot is deliberately small. It is not paper-valid.
PILOT_FRACTION = float(os.environ.get("SMGF_USTC_PILOT_FRACTION", "0.10"))
if not (0.0 < PILOT_FRACTION <= 1.0):
    raise ValueError("SMGF_USTC_PILOT_FRACTION must be in (0, 1].")

RUN_KIND = "FULL" if abs(PILOT_FRACTION - 1.0) < 1e-12 else "PILOT"
FRACTION_TAG = f"{int(round(PILOT_FRACTION * 10000)):05d}"
STAMP = time.strftime("%Y%m%d_%H%M%S", time.localtime())

RUN_DIR = RESULT_ROOT / (
    f"SMGF_V17_USTC_SEED{TRAIN_SEED}_{RUN_KIND}_"
    f"F{FRACTION_TAG}_{STAMP}"
)
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
HISTORY_DIR = RUN_DIR / "history"
LOG_FILE = RUN_DIR / "experiment.log"
CONFIG_FILE = RUN_DIR / "run_config.json"
SPLIT_FILE = RUN_DIR / "split_manifest.json"
INVENTORY_FILE = RUN_DIR / "dataset_inventory.json"
SUMMARY_FILE = RUN_DIR / "summary.json"
RESULT_FILE = RUN_DIR / "results.md"
FAILED_PCAP_FILE = RUN_DIR / "failed_pcaps.json"

CACHE_FILE = CACHE_ROOT / f"ustc_raw_graphs_f{FRACTION_TAG}.pt"
CACHE_META_FILE = CACHE_ROOT / f"ustc_raw_graphs_f{FRACTION_TAG}.json"

PARSE_PROGRESS_EVERY = 25

# Dataset-specific numeric validation threshold. This does NOT alter model inputs.
RECON_X_ATOL = 1e-3
RECON_STATS_ATOL = 1e-3
RECON_EDGE_ATOL = 1e-3

OLD_MANUSCRIPT_REFERENCE = {
    "clean_acc": 93.46,
    "clean_f1": 93.66,
    "mixed1_acc": 55.71,
    "mixed1_f1": 56.08,
    "adaptive1_acc": 73.89,
    "adaptive1_f1": 74.35,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_source_layout() -> None:
    missing = [p for p in [BASE_SOURCE, CORE_SOURCE] if not p.is_file()]
    if missing:
        msg = (
            "USTC runner must be in the SAME directory as V17/V16 sources. "
            "Missing: " + ", ".join(str(p) for p in missing)
        )
        raise FileNotFoundError(msg)

    base_sha = sha256_file(BASE_SOURCE)
    core_sha = sha256_file(CORE_SOURCE)
    if base_sha != EXPECTED_BASE_SHA256:
        raise RuntimeError(
            "V17 base SHA mismatch; refusing protocol drift. "
            f"expected={EXPECTED_BASE_SHA256}, actual={base_sha}"
        )
    if core_sha != EXPECTED_CORE_SHA256:
        raise RuntimeError(
            "V16 core SHA mismatch; refusing protocol drift. "
            f"expected={EXPECTED_CORE_SHA256}, actual={core_sha}"
        )


def import_base_v17():
    assert_source_layout()

    tmp_root = Path(tempfile.mkdtemp(prefix="smgf_ustc_import_"))
    old_result = os.environ.get("SMGF_RESULT_DIR")
    os.environ["SMGF_RESULT_DIR"] = str(tmp_root)
    try:
        spec = importlib.util.spec_from_file_location(
            "smgf_v17_base_for_ustc_frozen", BASE_SOURCE
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import V17 base: {BASE_SOURCE}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        if old_result is None:
            os.environ.pop("SMGF_RESULT_DIR", None)
        else:
            os.environ["SMGF_RESULT_DIR"] = old_result
        shutil.rmtree(tmp_root, ignore_errors=True)
    return module


BASE = import_base_v17()
CORE = BASE.CORE
torch = BASE.torch
Batch = BASE.Batch
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
    BASE.RESULT_FILE = RESULT_FILE
    BASE.CONFIG_FILE = CONFIG_FILE
    BASE.SPLIT_FILE = SPLIT_FILE

    CORE.RUN_DIR = RUN_DIR
    CORE.LOG_FILE = LOG_FILE
    CORE.RESULT_MD = RESULT_FILE
    CORE.SPLIT_MANIFEST_FILE = SPLIT_FILE
    CORE.CONFIG_FILE = CONFIG_FILE


# =============================================================================
# 1. DATA INVENTORY / DETERMINISTIC SELECTION / CACHE
# =============================================================================

def class_pcap_files(class_dir: Path):
    return sorted(
        [
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".pcap", ".pcapng"}
        ],
        key=lambda p: p.name,
    )


def validate_dataset_root() -> list[Path]:
    if not USTC_ROOT.is_dir():
        raise FileNotFoundError(
            f"Confirmed USTC path does not exist: {USTC_ROOT}"
        )

    class_dirs = sorted(
        [p for p in USTC_ROOT.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )
    if len(class_dirs) < 2:
        raise RuntimeError(
            f"Expected class subdirectories under {USTC_ROOT}; found {len(class_dirs)}."
        )

    observed_counts = {
        d.name: len(class_pcap_files(d))
        for d in class_dirs
    }
    empty = [name for name, n in observed_counts.items() if n == 0]
    if empty:
        raise RuntimeError(
            "Some top-level class directories contain no direct PCAP files: "
            + ", ".join(empty[:20])
        )

    # Reproduce the archived USTC protocol that generated the historical
    # SMGF-Net 93.46/93.66 result: 12 classes and 21,256 direct PCAP flows.
    # Fail closed on a different corpus or a partially copied directory.
    if observed_counts != EXPECTED_CLASS_COUNTS:
        raise RuntimeError(
            "USTC benchmark inventory mismatch. This frozen runner expects "
            f"{BENCHMARK_VARIANT}.\n"
            f"Expected={EXPECTED_CLASS_COUNTS}\n"
            f"Observed={observed_counts}"
        )
    if sum(observed_counts.values()) != EXPECTED_TOTAL_FLOWS:
        raise RuntimeError(
            "USTC benchmark total-flow mismatch: "
            f"expected={EXPECTED_TOTAL_FLOWS}, "
            f"observed={sum(observed_counts.values())}"
        )
    return class_dirs


def deterministic_select(files: list[Path], class_name: str) -> list[Path]:
    if PILOT_FRACTION >= 1.0:
        return list(files)

    n = len(files)
    n_select = min(n, max(3, int(round(n * PILOT_FRACTION))))
    rng = np.random.default_rng(
        CORE.stable_seed(
            "ustc_subset",
            PILOT_SELECTION_SEED,
            class_name,
        )
    )
    selected_idx = np.sort(rng.choice(n, size=n_select, replace=False))
    return [files[int(i)] for i in selected_idx]


def build_selection_plan(class_dirs: list[Path]):
    plan = []
    selected_files = []

    for class_dir in class_dirs:
        available = class_pcap_files(class_dir)
        selected = deterministic_select(available, class_dir.name)
        plan.append(
            {
                "class_name": class_dir.name,
                "available": len(available),
                "selected": len(selected),
                "fraction": len(selected) / max(len(available), 1),
            }
        )
        selected_files.extend((class_dir.name, p) for p in selected)

    return plan, selected_files


def source_selection_hash(selected_files) -> str:
    h = hashlib.sha256()
    for class_name, p in selected_files:
        rel = str(p.relative_to(USTC_ROOT)).replace("\\", "/")
        stat = p.stat()
        token = f"{class_name}|{rel}|{stat.st_size}\n"
        h.update(token.encode("utf-8"))
    return h.hexdigest()


def expected_cache_meta(plan, selected_files):
    return {
        "dataset": DATASET_NAME,
        "dataset_root": str(USTC_ROOT),
        "pilot_fraction": PILOT_FRACTION,
        "pilot_selection_seed": PILOT_SELECTION_SEED,
        "source_selection_sha256": source_selection_hash(selected_files),
        "expected_v17_base_sha256": EXPECTED_BASE_SHA256,
        "expected_v16_core_sha256": EXPECTED_CORE_SHA256,
        "class_plan": plan,
        "selected_file_count": len(selected_files),
    }


def cache_meta_matches(actual: dict, expected: dict) -> bool:
    keys = [
        "dataset",
        "dataset_root",
        "pilot_fraction",
        "pilot_selection_seed",
        "source_selection_sha256",
        "expected_v17_base_sha256",
        "expected_v16_core_sha256",
        "selected_file_count",
    ]
    return all(actual.get(k) == expected.get(k) for k in keys)


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_cache_if_valid(expected_meta: dict, logger: Logger):
    if not CACHE_FILE.is_file() or not CACHE_META_FILE.is_file():
        return None

    try:
        actual_meta = json.loads(CACHE_META_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.log(f"[CACHE] metadata unreadable; rebuilding: {exc}")
        return None

    if not cache_meta_matches(actual_meta, expected_meta):
        logger.log("[CACHE] cache metadata does not match current source selection; rebuilding.")
        return None

    logger.log(f"[CACHE] loading raw graphs: {CACHE_FILE}")
    payload = safe_torch_load(CACHE_FILE)
    required = {"dataset", "metadata", "selection", "skipped"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        logger.log("[CACHE] cache payload incomplete; rebuilding.")
        return None

    dataset = payload["dataset"]
    metadata = payload["metadata"]
    if len(dataset) != len(metadata):
        logger.log("[CACHE] dataset/metadata length mismatch; rebuilding.")
        return None

    manifest_hash = CORE.dataset_manifest_hash(metadata)
    cached_manifest = actual_meta.get("parsed_dataset_manifest_sha256")
    if cached_manifest and manifest_hash != cached_manifest:
        logger.log("[CACHE] parsed manifest mismatch; rebuilding.")
        return None

    logger.log(
        f"[CACHE] HIT | valid_samples={len(dataset)} | "
        f"manifest={manifest_hash}"
    )
    return (
        dataset,
        metadata,
        payload["selection"],
        int(payload["skipped"]),
        payload.get("failed_pcaps", []),
        manifest_hash,
    )


def parse_and_cache(
    class_dirs: list[Path],
    plan: list[dict],
    selected_files,
    expected_meta: dict,
    logger: Logger,
):
    class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}
    by_class = {d.name: [] for d in class_dirs}
    for class_name, p in selected_files:
        by_class[class_name].append(p)

    dataset = []
    metadata = []
    skipped = 0
    failed_pcaps = []
    processed = 0
    total_selected = len(selected_files)
    t0 = time.time()

    logger.log(
        f"[PARSE] start | selected_files={total_selected} | "
        f"classes={len(class_dirs)}"
    )

    for class_dir in class_dirs:
        class_name = class_dir.name
        label = class_to_idx[class_name]
        files = by_class[class_name]
        logger.log(
            f"[PARSE class] {class_name:24s} files={len(files)}"
        )

        class_valid = 0
        class_skipped = 0

        for p in files:
            processed += 1
            sample_index = len(metadata)

            try:
                data = CORE.pcap_to_graph_v13(
                    str(p),
                    label,
                    sample_index,
                )
            except Exception as exc:
                data = None
                failed_pcaps.append(
                    {
                        "relative_path": str(p.relative_to(USTC_ROOT)).replace("\\", "/"),
                        "class_name": class_name,
                        "error": repr(exc),
                    }
                )

            if data is not None and data.x.size(0) > 0:
                dataset.append(data)
                metadata.append(
                    {
                        "sample_index": sample_index,
                        "relative_path": str(p.relative_to(USTC_ROOT)).replace("\\", "/"),
                        "class_name": class_name,
                        "label": label,
                        "file_size": p.stat().st_size,
                    }
                )
                class_valid += 1
            else:
                skipped += 1
                class_skipped += 1
                if not any(
                    x.get("relative_path")
                    == str(p.relative_to(USTC_ROOT)).replace("\\", "/")
                    for x in failed_pcaps[-1:]
                ):
                    failed_pcaps.append(
                        {
                            "relative_path": str(p.relative_to(USTC_ROOT)).replace("\\", "/"),
                            "class_name": class_name,
                            "error": "pcap_to_graph_v13 returned None/empty",
                        }
                    )

            if (
                processed % PARSE_PROGRESS_EVERY == 0
                or processed == total_selected
            ):
                elapsed = time.time() - t0
                rate = processed / max(elapsed, 1e-9)
                remain = (total_selected - processed) / max(rate, 1e-9)
                logger.log(
                    f"[PARSE progress] {processed}/{total_selected} "
                    f"({100.0 * processed / total_selected:.1f}%) | "
                    f"valid={len(dataset)} | skipped={skipped} | "
                    f"elapsed={elapsed/60:.1f}m | eta={remain/60:.1f}m"
                )

        logger.log(
            f"[PARSE class done] {class_name:24s} "
            f"valid={class_valid} skipped={class_skipped}"
        )

    # Reindex valid samples contiguously after skipped/invalid files.
    for new_idx, (data, meta) in enumerate(zip(dataset, metadata)):
        data.sample_index = torch.tensor([new_idx], dtype=torch.long)
        meta["sample_index"] = new_idx

    if not dataset:
        raise RuntimeError("No valid USTC samples were parsed.")

    # Fail closed if any class loses so much data that a 3-way split is impossible.
    per_class_valid = {}
    for meta in metadata:
        per_class_valid[meta["class_name"]] = (
            per_class_valid.get(meta["class_name"], 0) + 1
        )
    too_small = {k: v for k, v in per_class_valid.items() if v < 3}
    if too_small:
        raise RuntimeError(
            f"Classes with <3 valid samples after parsing: {too_small}"
        )

    manifest_hash = CORE.dataset_manifest_hash(metadata)

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    tmp_cache = CACHE_FILE.with_suffix(CACHE_FILE.suffix + ".tmp")
    torch.save(
        {
            "dataset": dataset,
            "metadata": metadata,
            "selection": plan,
            "skipped": skipped,
            "failed_pcaps": failed_pcaps,
        },
        tmp_cache,
    )
    os.replace(tmp_cache, CACHE_FILE)

    final_meta = dict(expected_meta)
    final_meta.update(
        {
            "valid_sample_count": len(dataset),
            "skipped_sample_count": skipped,
            "parsed_dataset_manifest_sha256": manifest_hash,
            "cache_file": str(CACHE_FILE),
            "cache_created_at": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime()
            ),
        }
    )
    tmp_meta = CACHE_META_FILE.with_suffix(".json.tmp")
    tmp_meta.write_text(
        json.dumps(final_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp_meta, CACHE_META_FILE)

    logger.log(
        f"[CACHE] saved | {CACHE_FILE} | "
        f"valid={len(dataset)} skipped={skipped} | manifest={manifest_hash}"
    )
    return (
        dataset,
        metadata,
        plan,
        skipped,
        failed_pcaps,
        manifest_hash,
    )


def load_or_build_raw_dataset(logger: Logger):
    class_dirs = validate_dataset_root()
    plan, selected_files = build_selection_plan(class_dirs)

    logger.log(f"USTC root: {USTC_ROOT}")
    logger.log(f"Detected classes: {len(class_dirs)}")
    logger.log(
        f"Run scope: {RUN_KIND} | per-class fraction={PILOT_FRACTION:.4f}"
    )
    for row in plan:
        logger.log(
            f"[inventory/select] {row['class_name']:24s} "
            f"available={row['available']:6d} selected={row['selected']:6d}"
        )

    expected_meta = expected_cache_meta(plan, selected_files)
    cached = load_cache_if_valid(expected_meta, logger)
    if cached is not None:
        return class_dirs, cached, expected_meta

    built = parse_and_cache(
        class_dirs,
        plan,
        selected_files,
        expected_meta,
        logger,
    )
    return class_dirs, built, expected_meta


# =============================================================================
# 2. SPLIT / SCALING / NUMERIC SANITY CHECKS
# =============================================================================

def validate_reconstruction_identity_ustc(
    dataset,
    node_scaler,
    stats_scaler,
    logger: Logger,
    max_samples=16,
):
    """
    USTC-specific reconstruction sanity check.

    The check remains strict on graph topology and semantic anchors. Numeric
    tolerances only cover float32 inverse/rebuild round-trip drift.
    """
    n = min(len(dataset), max_samples)
    if n <= 0:
        raise RuntimeError("No samples available for reconstruction validation.")

    max_x_diff = 0.0
    max_stats_diff = 0.0
    max_edge_diff = 0.0

    for idx, data in enumerate(dataset[:n]):
        raw_x, _ = CORE.inverse_raw_features(
            data,
            node_scaler,
            stats_scaler,
        )
        rebuilt = CORE.finalize_attacked_data(
            data,
            raw_x,
            node_scaler,
            stats_scaler,
            data.finger_attr.cpu().numpy()[0],
            data.entropy_attr.cpu().numpy()[0],
        )

        if not torch.equal(data.edge_index, rebuilt.edge_index):
            raise AssertionError(
                f"USTC reconstruction failed: edge_index changed at sample {idx}"
            )
        if not torch.equal(data.y, rebuilt.y):
            raise AssertionError(
                f"USTC reconstruction failed: label changed at sample {idx}"
            )
        if not torch.equal(data.finger_attr, rebuilt.finger_attr):
            raise AssertionError(
                f"USTC reconstruction failed: finger_attr changed at sample {idx}"
            )
        if not torch.equal(data.entropy_attr, rebuilt.entropy_attr):
            raise AssertionError(
                f"USTC reconstruction failed: entropy_attr changed at sample {idx}"
            )

        max_x_diff = max(
            max_x_diff,
            float(torch.max(torch.abs(data.x - rebuilt.x)).item()),
        )
        max_stats_diff = max(
            max_stats_diff,
            float(
                torch.max(
                    torch.abs(data.stats_attr - rebuilt.stats_attr)
                ).item()
            ),
        )
        max_edge_diff = max(
            max_edge_diff,
            float(
                torch.max(
                    torch.abs(data.edge_attr - rebuilt.edge_attr)
                ).item()
            ),
        )

    logger.log(
        "[USTC reconstruction diagnostic] "
        f"n={n} | max_x_diff={max_x_diff:.3e} | "
        f"max_stats_diff={max_stats_diff:.3e} | "
        f"max_edge_diff={max_edge_diff:.3e} | "
        f"tolerances=({RECON_X_ATOL:.1e},"
        f"{RECON_STATS_ATOL:.1e},{RECON_EDGE_ATOL:.1e})"
    )

    if max_x_diff > RECON_X_ATOL:
        raise AssertionError(
            f"USTC reconstruction x drift too large: {max_x_diff:.3e}"
        )
    if max_stats_diff > RECON_STATS_ATOL:
        raise AssertionError(
            f"USTC reconstruction stats drift too large: {max_stats_diff:.3e}"
        )
    if max_edge_diff > RECON_EDGE_ATOL:
        raise AssertionError(
            f"USTC reconstruction edge drift too large: {max_edge_diff:.3e}"
        )

    logger.log("[USTC reconstruction identity] PASS")


def save_split_manifest(
    metadata,
    train_idx,
    val_idx,
    test_idx,
    manifest_hash,
):
    by_idx = {m["sample_index"]: m for m in metadata}

    payload = {
        "dataset": DATASET_NAME,
        "dataset_root": str(USTC_ROOT),
        "run_kind": RUN_KIND,
        "pilot_fraction": PILOT_FRACTION,
        "dataset_manifest_sha256": manifest_hash,
        "split_seed": SPLIT_SEED,
        "ratios": {
            "train": TRAIN_RATIO,
            "validation": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "train": [by_idx[i] for i in train_idx],
        "validation": [by_idx[i] for i in val_idx],
        "test": [by_idx[i] for i in test_idx],
    }
    SPLIT_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def prepare_data(logger: Logger):
    class_dirs, raw_bundle, expected_meta = load_or_build_raw_dataset(logger)
    (
        dataset,
        metadata,
        selection,
        skipped,
        failed_pcaps,
        manifest_hash,
    ) = raw_bundle

    train_idx, val_idx, test_idx = CORE.stratified_split_indices(
        dataset,
        split_seed=SPLIT_SEED,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
    )

    if set(train_idx) & set(val_idx):
        raise AssertionError("Train/validation overlap detected.")
    if set(train_idx) & set(test_idx):
        raise AssertionError("Train/test overlap detected.")
    if set(val_idx) & set(test_idx):
        raise AssertionError("Validation/test overlap detected.")

    train_set = CORE.materialize_split(dataset, train_idx)
    val_set = CORE.materialize_split(dataset, val_idx)
    test_set = CORE.materialize_split(dataset, test_idx)

    logger.log(
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

    INVENTORY_FILE.write_text(
        json.dumps(
            {
                "dataset": DATASET_NAME,
                "dataset_root": str(USTC_ROOT),
                "num_classes": len(class_dirs),
                "selection": selection,
                "valid_samples": len(dataset),
                "skipped_samples": skipped,
                "failed_pcap_records": len(failed_pcaps),
                "manifest_sha256": manifest_hash,
                "cache_file": str(CACHE_FILE),
                "cache_meta_file": str(CACHE_META_FILE),
                "source_selection_sha256":
                    expected_meta["source_selection_sha256"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    FAILED_PCAP_FILE.write_text(
        json.dumps(failed_pcaps, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    node_scaler, stats_scaler = CORE.fit_and_apply_scalers(
        train_set,
        val_set,
        test_set,
        logger,
    )

    validate_reconstruction_identity_ustc(
        test_set,
        node_scaler,
        stats_scaler,
        logger,
        max_samples=16,
    )
    BASE.validate_zero_strength_noop(
        test_set,
        node_scaler,
        stats_scaler,
        max_samples=16,
    )

    return {
        "train_set": train_set,
        "val_set": val_set,
        "test_set": test_set,
        "node_scaler": node_scaler,
        "stats_scaler": stats_scaler,
        "num_classes": len(class_dirs),
        "manifest_hash": manifest_hash,
        "selection": selection,
        "skipped": skipped,
        "failed_pcaps": failed_pcaps,
        "cache_file": str(CACHE_FILE),
    }


# =============================================================================
# 3. FROZEN TRAINING -- EXACTLY MATCH FINAL CIPHERSPECTRUM LOGIC
# =============================================================================

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


def lr_at(epoch: int) -> float:
    return LR_HIGH if epoch < LR_SWITCH_EPOCH else LR_LOW


def new_model(num_classes: int):
    return BASE.SMGFNetV17(
        node_in_channels=5,
        stats_dim=9,
        finger_seq_len=20,
        entropy_seq_len=5,
        hidden_channels=32,
        out_channels=num_classes,
        gate_mode=GATE_MODE,
    ).to(DEVICE)


def train_frozen_seed(data_info: dict, logger: Logger):
    train_set = data_info["train_set"]
    val_set = data_info["val_set"]
    node_scaler = data_info["node_scaler"]
    stats_scaler = data_info["stats_scaler"]
    num_classes = data_info["num_classes"]

    CORE.set_seed(TRAIN_SEED)
    model = new_model(num_classes)
    init_sha = BASE.model_state_sha256(model)

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
    logger.log(f"TRAIN USTC FROZEN seed={TRAIN_SEED}")
    logger.log(
        "100 epochs | lr 0.002 -> 0.0005 @ epoch71 | "
        "wd=1e-4 | robust_ratio=0.20 | attacked CE=1.0 | NO_KL"
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
            selector = CORE.stable_seed(
                "v17_robust_batch",
                TRAIN_SEED,
                epoch_zero,
                batch_idx,
                modulo=1_000_000,
            ) / 1_000_000.0

            attacked_batch = None
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
                np_rng = np.random.default_rng(attack_seed)
                py_rng = random.Random(attack_seed)

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

            clean_logits = BASE.forward_logits(model, batch)
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
                loss = clean_ce + ATTACK_CE_WEIGHT * attack_ce
                attack_ce_value = float(attack_ce.item())
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
        is_new_best = val_loss < best_val - 1e-8

        if is_new_best:
            best_val = float(val_loss)
            best_epoch = epoch
            best_state = cpu_model_state(model)
            best_opt = cpu_optimizer_state(optimizer)

        row = {
            "epoch": epoch,
            "lr": lr,
            "loss": total_loss / max(batches, 1),
            "clean_ce": clean_total / max(batches, 1),
            "attack_ce": attack_total / max(robust_batches, 1),
            "robust_batches": robust_batches,
            "total_batches": batches,
            "val_loss": float(val_loss),
            "is_new_best": bool(is_new_best),
            "best_epoch_so_far": best_epoch,
            "best_val_loss_so_far": best_val,
        }
        history.append(row)

        logger.log(
            f"[seed={TRAIN_SEED}] epoch {epoch:03d}/100 | "
            f"lr={lr:.7f} | loss={row['loss']:.6f} | "
            f"clean_ce={row['clean_ce']:.6f} | "
            f"attack_ce={row['attack_ce']:.6f} | "
            f"robust_batches={robust_batches}/{batches} | "
            f"val={val_loss:.6f} | best_ep={best_epoch} | "
            f"best={best_val:.6f}"
        )

    if best_state is None:
        raise RuntimeError(
            f"seed={TRAIN_SEED}: no best checkpoint captured."
        )

    history_path = HISTORY_DIR / f"seed{TRAIN_SEED}.csv"
    with history_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(history[0].keys()),
        )
        writer.writeheader()
        writer.writerows(history)

    checkpoint_path = (
        CHECKPOINT_DIR / f"USTC_FINAL_seed{TRAIN_SEED}.pt"
    )
    torch.save(
        {
            "model_state_dict": best_state,
            "optimizer_state_dict": best_opt,
            "tag": "USTC_FROZEN",
            "dataset": DATASET_NAME,
            "run_kind": RUN_KIND,
            "pilot_fraction": PILOT_FRACTION,
            "seed": TRAIN_SEED,
            "epochs": EPOCHS,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "lr_schedule": [
                [1, 70, LR_HIGH],
                [71, 100, LR_LOW],
            ],
            "weight_decay": WEIGHT_DECAY,
            "robust_batch_ratio": ROBUST_BATCH_RATIO,
            "robust_strength_range": [
                ROBUST_STRENGTH_MIN,
                ROBUST_STRENGTH_MAX,
            ],
            "attack_ce_weight": ATTACK_CE_WEIGHT,
            "kl_weight": KL_WEIGHT,
            "gate_mode": GATE_MODE,
            "initial_state_sha256": init_sha,
            "base_source_sha256": EXPECTED_BASE_SHA256,
            "core_source_sha256": EXPECTED_CORE_SHA256,
            "dataset_manifest_sha256":
                data_info["manifest_hash"],
            "cache_file": data_info["cache_file"],
        },
        checkpoint_path,
    )

    logger.log(
        f"[seed={TRAIN_SEED}] TRAIN COMPLETE | "
        f"best_epoch={best_epoch} | best_val={best_val:.12f}"
    )
    logger.log(f"[seed={TRAIN_SEED}] checkpoint={checkpoint_path}")

    return {
        "model": model,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "initial_state_sha256": init_sha,
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
    }


# =============================================================================
# 4. PRIMARY EVALUATION
# =============================================================================

def scalar_clean(result):
    return {
        "acc": float(result["acc"]),
        "macro_f1": float(result["macro_f1"]),
        "ece": float(result["ece"]),
    }


def scalar_attack(result):
    return {
        "robust_acc": float(result["robust_acc"]),
        "robust_f1": float(result["robust_f1"]),
        "asr": float(result["asr"]),
        "rel_drop": float(result["rel_drop"]),
        "ece": float(result["ece"]),
    }


def evaluate_primary(
    data_info: dict,
    train_info: dict,
    logger: Logger,
):
    model = train_info["model"]
    model.load_state_dict(train_info["best_state"])

    test_set = data_info["test_set"]
    node_scaler = data_info["node_scaler"]
    stats_scaler = data_info["stats_scaler"]

    clean_raw = CORE.evaluate_clean(model, test_set)
    CORE.log_clean_result(
        logger,
        "USTC_FROZEN",
        TRAIN_SEED,
        clean_raw,
    )
    clean = scalar_clean(clean_raw)

    attacks = {}
    for attack in ["mixed", "adaptive"]:
        strength = 1.0
        attack_seed = CORE.stable_seed(
            ATTACK_BASE_SEED,
            attack,
            f"{strength:.3f}",
        )
        logger.log(
            f"[EVAL cache] building {attack}@1.0 "
            f"for test n={len(test_set)} seed={attack_seed}"
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
            f"USTC:{attack}@{strength}",
        )
        if attack == "adaptive":
            CORE.validate_adaptive_semantics(
                test_set,
                attacked,
            )

        raw = CORE.evaluate_attack(
            model,
            attacked,
            clean_raw,
        )
        CORE.log_attack_result(
            logger,
            "USTC_FROZEN",
            TRAIN_SEED,
            attack,
            strength,
            raw,
        )
        attacks[f"{attack}@1.0"] = scalar_attack(raw)
        del attacked

    return clean, attacks


# =============================================================================
# 5. REPORTING
# =============================================================================

def write_config_and_results(
    data_info: dict,
    train_info: dict,
    clean: dict,
    attacks: dict,
):
    config = {
        "dataset": DATASET_NAME,
        "dataset_root": str(USTC_ROOT),
        "script_dir": str(SCRIPT_DIR),
        "result_root": str(RESULT_ROOT),
        "cache_root": str(CACHE_ROOT),
        "run_kind": RUN_KIND,
        "pilot_fraction": PILOT_FRACTION,
        "paper_valid": RUN_KIND == "FULL",
        "train_seed": TRAIN_SEED,
        "split_seed": SPLIT_SEED,
        "pilot_selection_seed": PILOT_SELECTION_SEED,
        "dataset_manifest_sha256":
            data_info["manifest_hash"],
        "num_classes": data_info["num_classes"],
        "split_counts": {
            "train": len(data_info["train_set"]),
            "validation": len(data_info["val_set"]),
            "test": len(data_info["test_set"]),
        },
        "frozen_protocol": {
            "epochs": EPOCHS,
            "early_stopping": False,
            "lr_schedule": [
                [1, 70, LR_HIGH],
                [71, 100, LR_LOW],
            ],
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "robust_batch_ratio":
                ROBUST_BATCH_RATIO,
            "train_attack": "mixed",
            "robust_strength_distribution":
                "Uniform[0.3,0.9]",
            "attack_ce_weight":
                ATTACK_CE_WEIGHT,
            "kl_weight": KL_WEIGHT,
            "gate_mode": GATE_MODE,
            "checkpoint_rule":
                "minimum clean-validation CE",
        },
        "reconstruction_tolerances": {
            "x": RECON_X_ATOL,
            "stats": RECON_STATS_ATOL,
            "edge": RECON_EDGE_ATOL,
        },
        "source_sha256": {
            "V17_base": EXPECTED_BASE_SHA256,
            "V16_core": EXPECTED_CORE_SHA256,
        },
        "cache_file": data_info["cache_file"],
        "skipped_samples": data_info["skipped"],
        "failed_pcap_records":
            len(data_info["failed_pcaps"]),
    }
    CONFIG_FILE.write_text(
        json.dumps(
            config,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = {
        "dataset": DATASET_NAME,
        "run_kind": RUN_KIND,
        "pilot_fraction": PILOT_FRACTION,
        "paper_valid": RUN_KIND == "FULL",
        "warning": (
            "PILOT subset results must not be inserted into the paper."
            if RUN_KIND == "PILOT"
            else
            "Full-dataset seed result under the frozen protocol."
        ),
        "train_seed": TRAIN_SEED,
        "best_epoch": train_info["best_epoch"],
        "best_val_loss":
            train_info["best_val_loss"],
        "initial_state_sha256":
            train_info["initial_state_sha256"],
        "checkpoint":
            train_info["checkpoint"],
        "clean": clean,
        "attacks": attacks,
        "historical_manuscript_reference_context_only":
            OLD_MANUSCRIPT_REFERENCE,
    }
    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    mixed = attacks["mixed@1.0"]
    adaptive = attacks["adaptive@1.0"]
    warning = (
        f"PILOT ONLY: deterministic {PILOT_FRACTION:.1%} "
        "per-class subset. Do not use these values in the paper."
        if RUN_KIND == "PILOT"
        else
        "FULL-DATASET run under the frozen protocol."
    )

    lines = [
        "# USTC-TFC2016 V17 Frozen-Protocol Result",
        "",
        f"**{warning}**",
        "",
        "No USTC-specific hyperparameter tuning was performed.",
        "",
        "## Data",
        "",
        f"- Dataset root: {USTC_ROOT}",
        f"- Classes: {data_info['num_classes']}",
        f"- Dataset manifest SHA-256: {data_info['manifest_hash']}",
        f"- Train/Val/Test: "
        f"{len(data_info['train_set'])}/"
        f"{len(data_info['val_set'])}/"
        f"{len(data_info['test_set'])}",
        f"- Raw graph cache: {data_info['cache_file']}",
        f"- Skipped PCAPs: {data_info['skipped']}",
        "",
        "## Frozen protocol",
        "",
        "- V17 + learned Gate + attacked-label CE + NO_KL",
        "- 100 epochs; no early stopping",
        "- LR 0.002 for epochs 1-70",
        "- LR 0.0005 for epochs 71-100",
        "- AdamW weight decay = 1e-4",
        "- robust batch ratio = 0.20",
        "- Mixed train strength = Uniform[0.3,0.9]",
        "- attacked CE weight = 1.0",
        "- checkpoint = minimum clean-validation CE",
        "",
        "## Primary result",
        "",
        f"- Train seed: {TRAIN_SEED}",
        f"- Best epoch: {train_info['best_epoch']}",
        f"- Best clean-val CE: "
        f"{train_info['best_val_loss']:.12f}",
        "",
        "| Condition | Acc/RobAcc | F1/RobF1 | ASR | RelDrop | ECE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Clean | {clean['acc']:.2f} | "
        f"{clean['macro_f1']:.2f} | - | - | "
        f"{clean['ece']:.2f} |",
        f"| Mixed@1.0 | {mixed['robust_acc']:.2f} | "
        f"{mixed['robust_f1']:.2f} | "
        f"{mixed['asr']:.2f} | "
        f"{mixed['rel_drop']:.2f} | "
        f"{mixed['ece']:.2f} |",
        f"| Adaptive@1.0 | {adaptive['robust_acc']:.2f} | "
        f"{adaptive['robust_f1']:.2f} | "
        f"{adaptive['asr']:.2f} | "
        f"{adaptive['rel_drop']:.2f} | "
        f"{adaptive['ece']:.2f} |",
        "",
        "## Historical manuscript values -- context only",
        "",
        "- Clean Acc/F1 = 93.46 / 93.66",
        "- Mixed@1 Acc/F1 = 55.71 / 56.08",
        "- Adaptive@1 Acc/F1 = 73.89 / 74.35",
        "",
        "Historical values came from an older model/protocol and must not "
        "be used to tune the frozen USTC experiment.",
    ]
    RESULT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =============================================================================
# 6. MAIN
# =============================================================================

def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required by the frozen experimental protocol."
        )

    RUN_DIR.mkdir(parents=True, exist_ok=False)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    logger = Logger(LOG_FILE)
    rewire_outputs(logger)

    logger.log("=" * 104)
    logger.log(
        f"USTC-TFC2016 | V17 frozen protocol | seed={TRAIN_SEED}"
    )
    logger.log(
        f"RUN_KIND={RUN_KIND} | "
        f"PILOT_FRACTION={PILOT_FRACTION:.4f}"
    )
    logger.log(
        "Frozen: 100ep | Step-LR | WD=1e-4 | "
        "robust_ratio=0.20 | attacked CE=1.0 | NO_KL"
    )
    logger.log(f"Script directory: {SCRIPT_DIR}")
    logger.log(f"Dataset root: {USTC_ROOT}")
    logger.log(f"Result directory: {RUN_DIR}")
    logger.log(f"Cache root: {CACHE_ROOT}")
    logger.log(f"V17 SHA-256: {sha256_file(BASE_SOURCE)}")
    logger.log(f"V16 SHA-256: {sha256_file(CORE_SOURCE)}")
    logger.log("=" * 104)

    data_info = prepare_data(logger)
    train_info = train_frozen_seed(
        data_info,
        logger,
    )
    clean, attacks = evaluate_primary(
        data_info,
        train_info,
        logger,
    )
    write_config_and_results(
        data_info,
        train_info,
        clean,
        attacks,
    )

    logger.log("")
    logger.log("=" * 104)
    logger.log("USTC FROZEN RUN COMPLETE")
    logger.log(f"Results: {RESULT_FILE}")
    logger.log(f"Summary: {SUMMARY_FILE}")
    logger.log(f"Config: {CONFIG_FILE}")
    logger.log(f"Split: {SPLIT_FILE}")
    logger.log(f"Inventory: {INVENTORY_FILE}")
    logger.log(f"Failed PCAPs: {FAILED_PCAP_FILE}")
    logger.log(f"Checkpoint: {train_info['checkpoint']}")
    logger.log("=" * 104)


if __name__ == "__main__":
    main()
