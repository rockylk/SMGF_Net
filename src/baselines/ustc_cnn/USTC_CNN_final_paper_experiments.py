# -*- coding: utf-8 -*-
"""
USTC-CNN [31] final paper baseline runner under the frozen SMGF-Net V17
evaluation protocol.

Why this runner exists
----------------------
The historical local baseline scripts predate the frozen V17 protocol and
use ad-hoc random 80/20 splits or test-driven model selection. This runner
reproduces the lightweight two-convolution-layer USTC-CNN architecture while
locking the *evaluation* to the same paper-facing data protocol used by V17.

Method fidelity
---------------
Architecture follows Wang et al., ICOIN 2017 / DeepTraffic traffic_cnn.py:
    28x28 single-channel byte image
    Conv5x5(1->32) + ReLU + MaxPool2
    Conv5x5(32->64) + ReLU + MaxPool2
    FC(7*7*64 -> 1024) + ReLU + Dropout
    FC(1024 -> C)
Training retains the original source's clean-only SGD setup:
    batch size 50
    learning rate 1e-4
    TRAIN_ROUND=20,000 in the authors' shell script; the original Python loop
    uses range(TRAIN_ROUND+1), hence 20,001 actual optimization updates
    summed cross-entropy during optimization
The only checkpointing adaptation is paper hygiene: minimum CLEAN validation
mean CE selects the checkpoint; test data never selects a checkpoint.

Frozen evaluation protocol
--------------------------
* exact final CipherSpectrum split/manifest
* exact archived 12-class USTC-TFC2016 split/manifest
* train seeds 42/43/44
* clean / Mixed@1.0 / Adaptive@1.0 reporting
* deterministic packet-padding + IAT-jitter + dummy-injection attack family
* ECE=15 bins; ASR is conditional on clean-correct samples
* mean +/- sample SD across the three training seeds
* prediction probabilities are persisted for final ROC regeneration

Strict input adaptation
-----------------------
The original USTC-CNN consumes a 784-byte session image. To match this paper's
strict no-routing-identifier preprocessing, each packet is serialized and the
first 38 bytes are removed before byte concatenation. This is the same
38-byte offset used by the official ET-BERT paper rerun (hex offset 76).
The remaining bytes are concatenated, truncated/padded to 784 bytes, scaled
to [0,1], and reshaped to 1x28x28.

IAT note
--------
USTC-CNN's byte-image representation has no timestamp channel. We still apply
the full physical Mixed/Adaptive perturbation family to packet sequences, but
pure IAT changes are representation-invisible to this baseline. The run
manifest states this explicitly; this is a property of the model input, not a
silent omission.

This is an architecture-faithful, protocol-adapted reproduction, not a claim
of bitwise identity with the authors' original dataset preprocessing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from scapy.all import Raw, rdpcap


# =============================================================================
# 0. FROZEN PAPER PROTOCOL
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
WORK_ROOT = Path(os.environ.get("SMGF_WORK_ROOT", str(Path.cwd()))).expanduser().resolve()
RESULT_ROOT = Path(
    os.environ.get("USTCCNN_RESULT_ROOT", str(WORK_ROOT / "RESULT"))
).expanduser().resolve()
RUN_DIR = Path(
    os.environ.get(
        "USTCCNN_FINAL_RUN_DIR",
        str(RESULT_ROOT / "USTC_CNN_FINAL_PAPER_BASELINE_V1"),
    )
).expanduser().resolve()

SEEDS = [42, 43, 44]
SPLIT_SEED = 42
ATTACK_BASE_SEED = 20260919

# Original-source training settings.
SOURCE_TRAIN_ROUND = 20_000
TRAIN_STEPS = SOURCE_TRAIN_ROUND + 1  # official code uses range(TRAIN_ROUND + 1)
BATCH_SIZE = 50
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 200
DROPOUT_P = 0.5

IMAGE_SIDE = 28
IMAGE_BYTES = IMAGE_SIDE * IMAGE_SIDE
PACKET_PREFIX_STRIP_BYTES = 38

ECE_BINS = 15
PRIMARY_MIXED_STRENGTH = 1.0
PRIMARY_ADAPTIVE_STRENGTH = 1.0

EXPECTED_SPLIT_SHA = {
    "CipherSpectrum":
        "01e3b572e8514a238b3733e224161acd1112b79b56f038635be9339b23cfad6b",
    "USTC-TFC2016":
        "a34754daa9e202f154484a5b63a8840eb829e56aeaaff4549bac6cb711127f3b",
}
EXPECTED_DATASET_MANIFEST = {
    "CipherSpectrum":
        "821bee04771f0c8dce3680a4301164d8ab48e03ec0d1ed57c3930015ad200a61",
    "USTC-TFC2016":
        "612c69a851dad04658619fd6c79ef60f34d08dd7dd1a0e14262fa01428cb2190",
}
EXPECTED_CLASSES = {
    "CipherSpectrum": 41,
    "USTC-TFC2016": 12,
}
EXPECTED_COUNTS = {
    "CipherSpectrum": {
        "train": 29520,
        "validation": 3280,
        "test": 8200,
    },
    "USTC-TFC2016": {
        "train": 15304,
        "validation": 1700,
        "test": 4252,
    },
}

BASELINE_CITATION = (
    "Wang W, Zhu M, Zeng X, Ye X, Sheng Y. Malware traffic classification "
    "using convolutional neural network for representation learning. "
    "ICOIN 2017:712-717."
)
ARCHITECTURE_SOURCE = (
    "https://github.com/echowei/DeepTraffic/"
    "tree/master/1.malware_traffic_classification/4.TrainAndTest"
)

RUN_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = RUN_DIR / "experiment.log"
RUN_MANIFEST = RUN_DIR / "run_manifest.json"
PER_SEED_CSV = RUN_DIR / "USTC_CNN_per_seed_primary.csv"
AGGREGATE_CSV = RUN_DIR / "USTC_CNN_aggregate_primary.csv"
SUMMARY_JSON = RUN_DIR / "summary.json"
PAPER_MD = RUN_DIR / "PAPER_READY_USTC_CNN_BASELINE.md"


# =============================================================================
# 1. UTILITIES
# =============================================================================

class Logger:
    def __init__(self, path: Path):
        self.path = path

    def log(self, message=""):
        msg = str(message)
        print(msg, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")


LOGGER = Logger(LOG_FILE)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(*parts, modulo: int = 2**31 - 1) -> int:
    text = "|".join(str(x) for x in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def mean_sd(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    return {
        "mean": statistics.mean(vals),
        "sample_sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
    }


def fmt_ms(stats: Dict[str, float], digits: int = 2) -> str:
    return f"{stats['mean']:.{digits}f} ± {stats['sample_sd']:.{digits}f}"


# =============================================================================
# 2. DATASET / SPLIT DISCOVERY
# =============================================================================

@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    root: Path
    split_manifest: Path
    num_classes: int
    expected_dataset_manifest: str
    expected_split_sha: str
    expected_counts: Dict[str, int]


def discover_split_manifest(
    dataset_name: str,
    env_name: str,
    glob_patterns: Sequence[str],
) -> Path:
    explicit = os.environ.get(env_name)
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p

    roots = [
        RESULT_ROOT,
        WORK_ROOT / "RESULT",
        WORK_ROOT,
        SCRIPT_DIR,
        SCRIPT_DIR.parent,
        Path("/home/admin1/lkfiles/RESULT"),
        Path("/home/admin1/lkfiles"),
    ]
    wanted = EXPECTED_SPLIT_SHA[dataset_name]
    seen = set()
    for root in roots:
        try:
            root = root.expanduser().resolve()
        except Exception:
            continue
        if str(root) in seen or not root.exists():
            continue
        seen.add(str(root))
        for pattern in glob_patterns:
            for p in root.glob(pattern):
                if p.is_file() and sha256_file(p) == wanted:
                    return p.resolve()
    raise FileNotFoundError(
        f"Cannot locate frozen split manifest for {dataset_name}. Set {env_name}."
    )


def build_dataset_specs(selection: str) -> List[DatasetSpec]:
    cipher_root = Path(
        os.environ.get(
            "USTCCNN_CIPHER_ROOT",
            "/home/admin1/dataset/aes-128-gcm/aes-128-gcm",
        )
    ).expanduser().resolve()
    ustc_root = Path(
        os.environ.get(
            "USTCCNN_USTC_ROOT",
            "/home/admin1/dataset/USTC-TFC2016-Split",
        )
    ).expanduser().resolve()

    specs: List[DatasetSpec] = []
    if selection in {"both", "cipher"}:
        split = discover_split_manifest(
            "CipherSpectrum",
            "USTCCNN_CIPHER_SPLIT_MANIFEST",
            [
                "SMGF_V17_CIPHERSPECTRUM_FINAL_3SEED_*/split_manifest.json",
                "**/SMGF_V17_CIPHERSPECTRUM_FINAL_3SEED_*/split_manifest.json",
            ],
        )
        specs.append(
            DatasetSpec(
                key="cipher",
                name="CipherSpectrum",
                root=cipher_root,
                split_manifest=split,
                num_classes=EXPECTED_CLASSES["CipherSpectrum"],
                expected_dataset_manifest=EXPECTED_DATASET_MANIFEST["CipherSpectrum"],
                expected_split_sha=EXPECTED_SPLIT_SHA["CipherSpectrum"],
                expected_counts=EXPECTED_COUNTS["CipherSpectrum"],
            )
        )

    if selection in {"both", "ustc"}:
        split = discover_split_manifest(
            "USTC-TFC2016",
            "USTCCNN_USTC_SPLIT_MANIFEST",
            [
                "SMGF_V17_USTC_SEED42_FULL_F10000_*/split_manifest.json",
                "**/SMGF_V17_USTC_SEED42_FULL_F10000_*/split_manifest.json",
                "SMGF_V17_USTC_FINAL_3SEED_*/split_manifest.json",
                "**/SMGF_V17_USTC_FINAL_3SEED_*/split_manifest.json",
            ],
        )
        specs.append(
            DatasetSpec(
                key="ustc",
                name="USTC-TFC2016",
                root=ustc_root,
                split_manifest=split,
                num_classes=EXPECTED_CLASSES["USTC-TFC2016"],
                expected_dataset_manifest=EXPECTED_DATASET_MANIFEST["USTC-TFC2016"],
                expected_split_sha=EXPECTED_SPLIT_SHA["USTC-TFC2016"],
                expected_counts=EXPECTED_COUNTS["USTC-TFC2016"],
            )
        )
    return specs


def validate_dataset_spec(spec: DatasetSpec) -> dict:
    if not spec.root.is_dir():
        raise FileNotFoundError(spec.root)
    if sha256_file(spec.split_manifest) != spec.expected_split_sha:
        raise RuntimeError(f"{spec.name}: split manifest SHA mismatch")

    split = load_json(spec.split_manifest)
    if split.get("dataset_manifest_sha256") != spec.expected_dataset_manifest:
        raise RuntimeError(
            f"{spec.name}: dataset manifest mismatch: "
            f"{split.get('dataset_manifest_sha256')}"
        )
    if int(split.get("split_seed", SPLIT_SEED)) != SPLIT_SEED:
        raise RuntimeError(f"{spec.name}: split seed is not {SPLIT_SEED}")

    counts = {
        "train": len(split["train"]),
        "validation": len(split["validation"]),
        "test": len(split["test"]),
    }
    if counts != spec.expected_counts:
        raise RuntimeError(f"{spec.name}: split counts drift: {counts}")

    labels = sorted({
        int(row["label"])
        for part in ("train", "validation", "test")
        for row in split[part]
    })
    if labels != list(range(spec.num_classes)):
        raise RuntimeError(f"{spec.name}: unexpected labels: {labels}")

    for part in ("train", "validation", "test"):
        for row in split[part]:
            p = spec.root / row["relative_path"]
            if not p.is_file():
                raise FileNotFoundError(p)
            if "file_size" in row and int(row["file_size"]) != int(p.stat().st_size):
                raise RuntimeError(f"{spec.name}: file-size drift: {p}")
    return split


# =============================================================================
# 3. RAW-PACKET PHYSICAL PERTURBATIONS
# =============================================================================

def _raw_payload_bytes(pkt) -> bytes:
    return bytes(pkt[Raw].load) if Raw in pkt else b""


def _replace_raw_payload(pkt, payload: bytes):
    q = pkt.copy()
    if Raw in q:
        q[Raw].load = payload
        return q
    if payload:
        q = q / Raw(load=payload)
    return q


def _filler_bytes(
    length: int,
    rng: np.random.Generator,
    semantics_preserving: bool,
) -> bytes:
    if length <= 0:
        return b""
    if semantics_preserving:
        return b"\x00" * int(length)
    return rng.integers(
        0, 256, size=int(length), dtype=np.uint8
    ).tobytes()


def attack_packet_length_padding_raw(
    packets,
    strength: float,
    np_rng: np.random.Generator,
    semantics_preserving: bool,
):
    out = [p.copy() for p in packets]
    n = len(out)
    if n == 0 or strength <= 0:
        return out

    k = max(1, int(n * (0.10 + 0.20 * strength)))
    idx = np_rng.choice(n, size=min(k, n), replace=False)
    max_pad = max(int(64 + 512 * strength), 21)
    pads = np_rng.integers(20, max_pad, size=len(idx))

    for pos, pad_len in zip(idx.tolist(), pads.tolist()):
        q = out[pos]
        payload = _raw_payload_bytes(q)
        filler = _filler_bytes(
            int(pad_len), np_rng, semantics_preserving
        )
        out[pos] = _replace_raw_payload(q, payload + filler)
    return out


def _packet_times(packets) -> np.ndarray:
    values = []
    for i, pkt in enumerate(packets):
        try:
            values.append(float(pkt.time))
        except Exception:
            values.append(float(i))
    return np.asarray(values, dtype=np.float64)


def attack_iat_jitter_raw(
    packets,
    strength: float,
    np_rng: np.random.Generator,
):
    out = [p.copy() for p in packets]
    n = len(out)
    if n <= 1 or strength <= 0:
        return out

    times = _packet_times(out)
    iats = np.diff(times, prepend=times[0])
    iats[0] = 0.0
    iats = np.clip(iats, 0.0, None)

    k = max(1, int(n * (0.20 + 0.30 * strength)))
    idx = np_rng.choice(n, size=min(k, n), replace=False)
    amp = 0.10 + 0.70 * strength
    multipliers = np_rng.uniform(
        1.0 - amp, 1.0 + amp, size=len(idx)
    )
    for pos, mult in zip(idx.tolist(), multipliers.tolist()):
        if pos != 0:
            iats[pos] = max(iats[pos] * float(mult), 0.0)

    rebuilt = np.cumsum(iats)
    base_time = times[0]
    for i, pkt in enumerate(out):
        pkt.time = base_time + float(rebuilt[i])
    return out


def _dummy_from_base(
    base_pkt,
    target_total_size: int,
    np_rng: np.random.Generator,
    semantics_preserving: bool,
):
    q = _replace_raw_payload(base_pkt.copy(), b"")
    header_size = len(bytes(q))
    payload_len = max(0, int(target_total_size) - header_size)
    if payload_len:
        q = _replace_raw_payload(
            q,
            _filler_bytes(
                payload_len,
                np_rng,
                semantics_preserving,
            ),
        )
    return q


def attack_dummy_injection_raw(
    packets,
    strength: float,
    np_rng: np.random.Generator,
    py_rng: random.Random,
    semantics_preserving: bool,
):
    out = [p.copy() for p in packets]
    n = len(out)
    if n == 0 or strength <= 0:
        return out

    insert_num = max(1, int(n * (0.03 + 0.10 * strength)))
    times = _packet_times(out)
    if len(times) > 1:
        positive = np.diff(times)
        positive = positive[positive > 0]
        median_iat = float(np.median(positive)) if len(positive) else 1e-6
    else:
        median_iat = 1e-6

    positions = sorted(
        np_rng.integers(1, max(2, n), size=insert_num).tolist()
    )
    offset = 0
    for p in positions:
        base_idx = min(max(p - 1 + offset, 0), len(out) - 1)
        base = out[base_idx]
        target_size = int(py_rng.uniform(40, 200 + 400 * strength))
        dummy = _dummy_from_base(
            base,
            target_size,
            np_rng,
            semantics_preserving,
        )
        base_time = float(getattr(base, "time", 0.0))
        dummy_gap = max(
            median_iat * py_rng.uniform(0.2, 1.0),
            1e-6,
        )
        dummy.time = base_time + dummy_gap
        insert_at = min(p + offset, len(out))
        out.insert(insert_at, dummy)
        offset += 1

    out.sort(key=lambda pkt: float(getattr(pkt, "time", 0.0)))
    return out


def attack_mixed_raw(
    packets,
    strength: float,
    np_rng: np.random.Generator,
    py_rng: random.Random,
    semantics_preserving: bool,
):
    out = attack_packet_length_padding_raw(
        packets,
        strength,
        np_rng,
        semantics_preserving,
    )
    out = attack_iat_jitter_raw(out, strength, np_rng)
    out = attack_dummy_injection_raw(
        out,
        strength,
        np_rng,
        py_rng,
        semantics_preserving,
    )
    return out


def apply_attack_raw(
    packets,
    attack_name: str,
    strength: float,
    attack_seed: int,
):
    if strength <= 0:
        return [p.copy() for p in packets]

    np_rng = np.random.default_rng(int(attack_seed))
    py_rng = random.Random(int(attack_seed))
    name = attack_name.lower()

    if name == "mixed":
        return attack_mixed_raw(
            packets,
            strength,
            np_rng,
            py_rng,
            semantics_preserving=False,
        )
    if name == "adaptive":
        return attack_mixed_raw(
            packets,
            strength,
            np_rng,
            py_rng,
            semantics_preserving=True,
        )
    raise ValueError(f"Unsupported attack: {attack_name}")


# =============================================================================
# 4. STRICT 784-BYTE IMAGE REPRESENTATION / CACHE
# =============================================================================

def packets_to_784_bytes(packets) -> np.ndarray:
    """
    Remove 38 leading serialized bytes from EACH packet, concatenate remaining
    bytes, then truncate/zero-pad to exactly 784 uint8 values.
    """
    out = bytearray()
    for pkt in packets:
        raw = bytes(pkt)
        if len(raw) > PACKET_PREFIX_STRIP_BYTES:
            out.extend(raw[PACKET_PREFIX_STRIP_BYTES:])
        if len(out) >= IMAGE_BYTES:
            break

    arr = np.zeros(IMAGE_BYTES, dtype=np.uint8)
    if out:
        n = min(len(out), IMAGE_BYTES)
        arr[:n] = np.frombuffer(bytes(out[:n]), dtype=np.uint8)
    return arr


def dataset_dir(spec: DatasetSpec) -> Path:
    return RUN_DIR / spec.key


def prepared_dir(spec: DatasetSpec) -> Path:
    return dataset_dir(spec) / "prepared"


def prepared_npz(spec: DatasetSpec, condition: str) -> Path:
    return prepared_dir(spec) / f"{condition}.npz"


def preparation_meta(spec: DatasetSpec) -> Path:
    return dataset_dir(spec) / "dataset_preparation.json"


def condition_rows(
    spec: DatasetSpec,
    rows: Sequence[dict],
    condition: str,
    attack_name: Optional[str] = None,
    strength: Optional[float] = None,
):
    xs = np.zeros((len(rows), IMAGE_BYTES), dtype=np.uint8)
    ys = np.zeros(len(rows), dtype=np.int64)
    sample_indices = np.zeros(len(rows), dtype=np.int64)

    for pos, row in enumerate(rows):
        p = spec.root / row["relative_path"]
        packets = rdpcap(str(p))
        sample_idx = int(row.get("sample_index", pos))

        if attack_name is not None:
            seed = stable_seed(
                ATTACK_BASE_SEED,
                spec.name,
                attack_name,
                f"{float(strength):.3f}",
                sample_idx,
            )
            packets = apply_attack_raw(
                packets,
                attack_name,
                float(strength),
                seed,
            )

        xs[pos] = packets_to_784_bytes(packets)
        ys[pos] = int(row["label"])
        sample_indices[pos] = sample_idx

        if (pos + 1) % 500 == 0 or pos + 1 == len(rows):
            LOGGER.log(
                f"[prepare][{spec.name}][{condition}] "
                f"{pos + 1}/{len(rows)}"
            )

    return xs, ys, sample_indices


def save_condition(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    sample_index: np.ndarray,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=x,
        y=y,
        sample_index=sample_index,
    )


def condition_valid(
    path: Path,
    expected_n: int,
    expected_classes: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            x = z["x"]
            y = z["y"]
            idx = z["sample_index"]
            if x.shape != (expected_n, IMAGE_BYTES):
                return False
            if y.shape != (expected_n,) or idx.shape != (expected_n,):
                return False
            if x.dtype != np.uint8:
                return False
            labels = sorted(np.unique(y).tolist())
            if labels != list(range(expected_classes)):
                return False
    except Exception:
        return False
    return True


def preparation_reuse_allowed(spec: DatasetSpec) -> bool:
    meta_path = preparation_meta(spec)
    if not meta_path.is_file():
        return False
    try:
        meta = load_json(meta_path)
    except Exception:
        return False

    return all([
        meta.get("script_sha256") == sha256_file(SCRIPT_PATH),
        meta.get("dataset_manifest_sha256") == spec.expected_dataset_manifest,
        meta.get("split_sha256") == spec.expected_split_sha,
        int(meta.get("split_seed", -1)) == SPLIT_SEED,
        meta.get("representation", {}).get("packet_prefix_strip_bytes")
            == PACKET_PREFIX_STRIP_BYTES,
        meta.get("attack_protocol", {}).get("attack_base_seed")
            == ATTACK_BASE_SEED,
        meta.get("attack_protocol", {}).get("mixed_strength")
            == PRIMARY_MIXED_STRENGTH,
        meta.get("attack_protocol", {}).get("adaptive_strength")
            == PRIMARY_ADAPTIVE_STRENGTH,
    ])


def prepare_dataset(spec: DatasetSpec) -> dict:
    split = validate_dataset_spec(spec)
    reuse_allowed = preparation_reuse_allowed(spec)
    expected = {
        "train_clean": (split["train"], None, None),
        "validation_clean": (split["validation"], None, None),
        "test_clean": (split["test"], None, None),
        "test_mixed_1.0": (
            split["test"], "mixed", PRIMARY_MIXED_STRENGTH
        ),
        "test_adaptive_1.0": (
            split["test"], "adaptive", PRIMARY_ADAPTIVE_STRENGTH
        ),
    }

    paths = {}
    for condition, (rows, attack, strength) in expected.items():
        path = prepared_npz(spec, condition)
        paths[condition] = path
        if reuse_allowed and condition_valid(path, len(rows), spec.num_classes):
            LOGGER.log(
                f"[prepare][{spec.name}] reuse valid cache: {condition}"
            )
            continue

        LOGGER.log(
            f"[prepare][{spec.name}] building {condition} n={len(rows)}"
        )
        x, y, sample_idx = condition_rows(
            spec,
            rows,
            condition,
            attack_name=attack,
            strength=strength,
        )
        save_condition(path, x, y, sample_idx)

    meta = {
        "dataset": spec.name,
        "script_sha256": sha256_file(SCRIPT_PATH),
        "dataset_root": str(spec.root),
        "dataset_manifest_sha256": spec.expected_dataset_manifest,
        "split_manifest": str(spec.split_manifest),
        "split_sha256": spec.expected_split_sha,
        "split_seed": SPLIT_SEED,
        "num_classes": spec.num_classes,
        "counts": spec.expected_counts,
        "representation": {
            "type": "route-stripped 784-byte session image",
            "packet_prefix_strip_bytes": PACKET_PREFIX_STRIP_BYTES,
            "image_shape": [1, IMAGE_SIDE, IMAGE_SIDE],
            "normalization": "uint8 / 255.0 at model input",
            "iat_visible_to_model": False,
            "note": (
                "IAT perturbation is applied at packet level but this byte-image "
                "baseline has no timestamp channel."
            ),
        },
        "attack_protocol": {
            "attack_base_seed": ATTACK_BASE_SEED,
            "mixed_strength": PRIMARY_MIXED_STRENGTH,
            "adaptive_strength": PRIMARY_ADAPTIVE_STRENGTH,
            "mixed": (
                "padding + IAT jitter + dummy insertion; added bytes use "
                "deterministic pseudo-random filler"
            ),
            "adaptive": (
                "same structural family; genuine payload is retained and added "
                "padding/dummy bytes use zero filler"
            ),
            "not_gradient_white_box": True,
        },
        "cache_sha256": {
            key: sha256_file(path) for key, path in paths.items()
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    dump_json(preparation_meta(spec), meta)
    return meta


# =============================================================================
# 5. USTC-CNN MODEL
# =============================================================================

class USTCCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.dropout = nn.Dropout(p=DROPOUT_P)
        self.fc2 = nn.Linear(1024, num_classes)
        self.reset_parameters_like_source()

    def reset_parameters_like_source(self):
        for module in (self.conv1, self.conv2, self.fc1, self.fc2):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.1, a=-0.2, b=0.2)
            nn.init.constant_(module.bias, 0.1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class ByteImageDataset(Dataset):
    def __init__(self, npz_path: Path):
        with np.load(npz_path, allow_pickle=False) as z:
            self.x = np.asarray(z["x"], dtype=np.uint8)
            self.y = np.asarray(z["y"], dtype=np.int64)

    def __len__(self):
        return int(len(self.y))

    def __getitem__(self, idx):
        x = torch.from_numpy(self.x[idx].copy()).float().div_(255.0)
        x = x.view(1, IMAGE_SIDE, IMAGE_SIDE)
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    drop_last: bool = False,
):
    g = torch.Generator()
    g.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=g,
    )


def model_parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# =============================================================================
# 6. TRAINING / CHECKPOINT SELECTION
# =============================================================================

def model_dir(spec: DatasetSpec, seed: int) -> Path:
    return dataset_dir(spec) / "models" / f"seed{seed}"


def checkpoint_path(spec: DatasetSpec, seed: int) -> Path:
    return model_dir(spec, seed) / "best.pt"


def training_summary_path(spec: DatasetSpec, seed: int) -> Path:
    return model_dir(spec, seed) / "training_summary.json"


def evaluate_validation_loss(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
) -> float:
    loader = make_loader(
        dataset,
        batch_size=256,
        shuffle=False,
        seed=0,
    )
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            total += float(criterion(logits, y).item())
            n += int(y.numel())
    return total / max(n, 1)


def existing_training_valid(
    spec: DatasetSpec,
    seed: int,
) -> bool:
    ckpt = checkpoint_path(spec, seed)
    summary_file = training_summary_path(spec, seed)
    if not ckpt.is_file() or not summary_file.is_file():
        return False
    try:
        summary = load_json(summary_file)
    except Exception:
        return False

    expected = {
        "dataset": spec.name,
        "seed": seed,
        "train_steps": TRAIN_STEPS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": "SGD",
        "training_regime": "clean-only baseline",
        "checkpoint_rule": "minimum clean-validation mean CE",
        "split_sha256": spec.expected_split_sha,
        "dataset_manifest_sha256": spec.expected_dataset_manifest,
        "script_sha256": sha256_file(SCRIPT_PATH),
    }
    if any(summary.get(k) != v for k, v in expected.items()):
        return False
    recorded_ckpt_sha = summary.get("checkpoint_sha256")
    return bool(
        recorded_ckpt_sha
        and recorded_ckpt_sha == sha256_file(ckpt)
    )


def train_one(
    spec: DatasetSpec,
    seed: int,
    device: torch.device,
    force: bool = False,
):
    ckpt = checkpoint_path(spec, seed)
    summary_file = training_summary_path(spec, seed)
    if not force and existing_training_valid(spec, seed):
        LOGGER.log(
            f"[train][{spec.name}][seed={seed}] reuse protocol-matched checkpoint"
        )
        return load_json(summary_file)
    if (ckpt.is_file() or summary_file.is_file()) and not force:
        LOGGER.log(
            f"[train][{spec.name}][seed={seed}] stale checkpoint metadata detected; retraining"
        )

    set_seed(seed)
    train_ds = ByteImageDataset(prepared_npz(spec, "train_clean"))
    val_ds = ByteImageDataset(prepared_npz(spec, "validation_clean"))
    loader = make_loader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=seed + 31_000,
        drop_last=True,
    )
    iterator = iter(loader)

    model = USTCCNN(spec.num_classes).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    # Original TensorFlow source used reduce_sum cross entropy.
    criterion_train = nn.CrossEntropyLoss(reduction="sum")

    best_val_ce = float("inf")
    best_step = -1
    best_state = None
    history = []
    running_loss = 0.0
    running_n = 0
    start = time.time()

    LOGGER.log("=" * 96)
    LOGGER.log(
        f"TRAIN USTC-CNN | dataset={spec.name} | seed={seed} | "
        f"steps={TRAIN_STEPS} | batch={BATCH_SIZE} | SGD lr={LEARNING_RATE}"
    )
    LOGGER.log(
        "Training is clean-only to preserve the baseline method. "
        "Checkpoint=min clean-validation mean CE."
    )
    LOGGER.log("=" * 96)

    for step in range(1, TRAIN_STEPS + 1):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)

        model.train()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion_train(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        running_n += int(y.numel())

        should_validate = (
            step == 1
            or step % VALIDATE_EVERY == 0
            or step == TRAIN_STEPS
        )
        if should_validate:
            val_ce = evaluate_validation_loss(model, val_ds, device)
            improved = val_ce < best_val_ce - 1e-12
            if improved:
                best_val_ce = float(val_ce)
                best_step = int(step)
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

            row = {
                "step": int(step),
                "train_sum_ce_per_sample_since_last_log":
                    running_loss / max(running_n, 1),
                "validation_mean_ce": float(val_ce),
                "new_best": bool(improved),
                "best_step_so_far": int(best_step),
                "best_validation_mean_ce_so_far": float(best_val_ce),
            }
            history.append(row)
            LOGGER.log(
                f"[train][{spec.name}][seed={seed}] step={step:05d}/"
                f"{TRAIN_STEPS} | trainCE/sample={row['train_sum_ce_per_sample_since_last_log']:.6f} | "
                f"valCE={val_ce:.6f} | best_step={best_step}"
            )
            running_loss = 0.0
            running_n = 0

    if best_state is None:
        raise RuntimeError(
            f"{spec.name} seed={seed}: no validation checkpoint was selected"
        )

    model.load_state_dict(best_state)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "dataset": spec.name,
        "seed": seed,
        "architecture": "USTC-CNN two-layer 5x5 CNN",
        "num_classes": spec.num_classes,
        "train_steps": TRAIN_STEPS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": "SGD",
        "train_loss": "summed cross entropy",
        "training_regime": "clean-only baseline",
        "checkpoint_rule": "minimum clean-validation mean CE",
        "best_step": best_step,
        "best_validation_mean_ce": best_val_ce,
        "split_sha256": spec.expected_split_sha,
        "dataset_manifest_sha256": spec.expected_dataset_manifest,
        "script_sha256": sha256_file(SCRIPT_PATH),
    }
    torch.save(payload, ckpt)

    summary = {
        k: v for k, v in payload.items()
        if k != "model_state_dict"
    }
    summary["train_seconds"] = time.time() - start
    summary["trainable_parameters"] = model_parameter_count(model)
    summary["history"] = history
    summary["checkpoint"] = str(ckpt)
    summary["checkpoint_sha256"] = sha256_file(ckpt)
    dump_json(summary_file, summary)
    return summary


# =============================================================================
# 7. METRICS / EVALUATION
# =============================================================================

def expected_calibration_error(
    labels: np.ndarray,
    probs: np.ndarray,
    n_bins: int = ECE_BINS,
) -> float:
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correctness = (preds == labels).astype(np.float64)
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
        if count:
            ece += (
                count / max(n, 1)
            ) * abs(
                correctness[mask].mean() - confidences[mask].mean()
            )
    return float(ece * 100.0)


def multiclass_macro_auroc(
    labels: np.ndarray,
    probs: np.ndarray,
    num_classes: int,
) -> Optional[float]:
    try:
        y_bin = label_binarize(
            labels,
            classes=list(range(num_classes)),
        )
        if num_classes == 2:
            value = roc_auc_score(y_bin[:, 0], probs[:, 1])
        else:
            value = roc_auc_score(
                y_bin,
                probs,
                average="macro",
                multi_class="ovr",
            )
        return float(value * 100.0)
    except Exception:
        return None


def predict_condition(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    num_classes: int,
):
    loader = make_loader(
        dataset,
        batch_size=256,
        shuffle=False,
        seed=0,
    )
    labels, probs = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device, non_blocking=True))
            p = torch.softmax(logits, dim=1).cpu().numpy()
            labels.append(y.numpy())
            probs.append(p)

    y_true = np.concatenate(labels).astype(np.int64)
    probability = np.concatenate(probs).astype(np.float64)
    if probability.shape != (len(y_true), num_classes):
        raise RuntimeError(
            f"Probability shape mismatch: {probability.shape}"
        )
    preds = probability.argmax(axis=1).astype(np.int64)
    return {
        "acc": float(accuracy_score(y_true, preds) * 100.0),
        "macro_f1": float(
            f1_score(
                y_true,
                preds,
                average="macro",
                zero_division=0,
            ) * 100.0
        ),
        "ece": expected_calibration_error(
            y_true, probability, ECE_BINS
        ),
        "macro_auroc": multiclass_macro_auroc(
            y_true, probability, num_classes
        ),
        "labels": y_true,
        "preds": preds,
        "probs": probability,
    }


def attack_metrics(clean_eval, attacked_eval):
    if not np.array_equal(
        clean_eval["labels"], attacked_eval["labels"]
    ):
        raise RuntimeError(
            "Attack label order differs from clean test order"
        )

    labels = clean_eval["labels"]
    clean_correct = clean_eval["preds"] == labels
    attacked_wrong = attacked_eval["preds"] != labels
    attack_success = int(np.sum(clean_correct & attacked_wrong))
    clean_correct_n = int(np.sum(clean_correct))
    robust_acc = float(attacked_eval["acc"])
    rel_drop = (
        (float(clean_eval["acc"]) - robust_acc)
        / max(float(clean_eval["acc"]), 1e-12)
        * 100.0
    )
    return {
        "robust_acc": robust_acc,
        "robust_f1": float(attacked_eval["macro_f1"]),
        "asr": attack_success / max(clean_correct_n, 1) * 100.0,
        "rel_drop": float(rel_drop),
        "abs_drop_pp": float(clean_eval["acc"]) - robust_acc,
        "ece": float(attacked_eval["ece"]),
        "macro_auroc": attacked_eval["macro_auroc"],
    }


def compact_eval(obj):
    return {
        k: v for k, v in obj.items()
        if k not in {"labels", "preds", "probs"}
    }


def prediction_dir(spec: DatasetSpec, seed: int) -> Path:
    return dataset_dir(spec) / "predictions" / f"seed{seed}"


def seed_result_path(spec: DatasetSpec, seed: int) -> Path:
    return dataset_dir(spec) / "results" / f"seed{seed}.json"


def evaluate_seed(
    spec: DatasetSpec,
    seed: int,
    device: torch.device,
):
    ckpt = checkpoint_path(spec, seed)
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint; train first: {ckpt}"
        )
    payload = torch.load(ckpt, map_location="cpu")
    model = USTCCNN(spec.num_classes).to(device)
    model.load_state_dict(payload["model_state_dict"])

    conditions = {
        "clean": "test_clean",
        "mixed_1.0": "test_mixed_1.0",
        "adaptive_1.0": "test_adaptive_1.0",
    }
    evals = {}
    pred_dir = prediction_dir(spec, seed)
    pred_dir.mkdir(parents=True, exist_ok=True)

    for name, prepared_name in conditions.items():
        ds = ByteImageDataset(prepared_npz(spec, prepared_name))
        result = predict_condition(
            model,
            ds,
            device,
            spec.num_classes,
        )
        np.savez_compressed(
            pred_dir / f"{name}.npz",
            labels=result["labels"],
            preds=result["preds"],
            probs=result["probs"].astype(np.float32),
        )
        evals[name] = result

    clean = evals["clean"]
    mixed = attack_metrics(clean, evals["mixed_1.0"])
    adaptive = attack_metrics(clean, evals["adaptive_1.0"])

    result = {
        "dataset": spec.name,
        "seed": seed,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
        "clean": compact_eval(clean),
        "mixed_1.0": mixed,
        "adaptive_1.0": adaptive,
        "primary": {
            "clean_acc": clean["acc"],
            "clean_f1": clean["macro_f1"],
            "clean_ece": clean["ece"],
            "clean_auroc": clean["macro_auroc"],
            "mixed_robust_acc": mixed["robust_acc"],
            "mixed_robust_f1": mixed["robust_f1"],
            "mixed_asr": mixed["asr"],
            "mixed_rel_drop": mixed["rel_drop"],
            "mixed_ece": mixed["ece"],
            "mixed_auroc": mixed["macro_auroc"],
            "adaptive_robust_acc": adaptive["robust_acc"],
            "adaptive_robust_f1": adaptive["robust_f1"],
            "adaptive_asr": adaptive["asr"],
            "adaptive_rel_drop": adaptive["rel_drop"],
            "adaptive_ece": adaptive["ece"],
            "adaptive_auroc": adaptive["macro_auroc"],
        },
    }
    out = seed_result_path(spec, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump_json(out, result)

    LOGGER.log(
        f"[RESULT][{spec.name}][seed={seed}] CLEAN "
        f"Acc={clean['acc']:.2f} F1={clean['macro_f1']:.2f} "
        f"ECE={clean['ece']:.2f}"
    )
    LOGGER.log(
        f"[RESULT][{spec.name}][seed={seed}] MIXED@1.0 "
        f"RobAcc={mixed['robust_acc']:.2f} RobF1={mixed['robust_f1']:.2f} "
        f"ASR={mixed['asr']:.2f} RelDrop={mixed['rel_drop']:.2f} "
        f"ECE={mixed['ece']:.2f}"
    )
    LOGGER.log(
        f"[RESULT][{spec.name}][seed={seed}] ADAPTIVE@1.0 "
        f"RobAcc={adaptive['robust_acc']:.2f} "
        f"RobF1={adaptive['robust_f1']:.2f} "
        f"ASR={adaptive['asr']:.2f} "
        f"RelDrop={adaptive['rel_drop']:.2f} "
        f"ECE={adaptive['ece']:.2f}"
    )
    return result


# =============================================================================
# 8. AGGREGATION
# =============================================================================

PRIMARY_KEYS = [
    "clean_acc",
    "clean_f1",
    "clean_ece",
    "clean_auroc",
    "mixed_robust_acc",
    "mixed_robust_f1",
    "mixed_asr",
    "mixed_rel_drop",
    "mixed_ece",
    "mixed_auroc",
    "adaptive_robust_acc",
    "adaptive_robust_f1",
    "adaptive_asr",
    "adaptive_rel_drop",
    "adaptive_ece",
    "adaptive_auroc",
]


def aggregate(specs: Sequence[DatasetSpec]):
    all_seed_rows = []
    aggregate_rows = []
    summary = {}

    for spec in specs:
        records = []
        for seed in SEEDS:
            p = seed_result_path(spec, seed)
            if not p.is_file():
                raise FileNotFoundError(
                    f"Missing seed result; evaluate first: {p}"
                )
            obj = load_json(p)
            row = {
                "dataset": spec.name,
                "seed": seed,
                **obj["primary"],
            }
            records.append(row)
            all_seed_rows.append(row)

        agg = {"dataset": spec.name}
        for key in PRIMARY_KEYS:
            vals = [
                r[key] for r in records
                if r.get(key) is not None
            ]
            agg[key] = mean_sd(vals) if vals else None
        aggregate_rows.append(agg)
        summary[spec.name] = {
            "per_seed": records,
            "aggregate": agg,
        }

    PER_SEED_CSV.parent.mkdir(parents=True, exist_ok=True)
    with PER_SEED_CSV.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        fields = ["dataset", "seed"] + PRIMARY_KEYS
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_seed_rows)

    with AGGREGATE_CSV.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        fields = ["dataset"]
        for key in PRIMARY_KEYS:
            fields.extend([f"{key}_mean", f"{key}_sample_sd"])
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for agg in aggregate_rows:
            row = {"dataset": agg["dataset"]}
            for key in PRIMARY_KEYS:
                if agg[key] is None:
                    row[f"{key}_mean"] = ""
                    row[f"{key}_sample_sd"] = ""
                else:
                    row[f"{key}_mean"] = agg[key]["mean"]
                    row[f"{key}_sample_sd"] = agg[key]["sample_sd"]
            w.writerow(row)

    dump_json(SUMMARY_JSON, summary)

    lines = [
        "# USTC-CNN final baseline under frozen V17 evaluation protocol",
        "",
        "Architecture-faithful, protocol-adapted reproduction of USTC-CNN [31].",
        "",
        "Important: training is CLEAN-ONLY and follows the original baseline "
        "optimizer/update budget. V17 robust training is NOT added to the baseline.",
        "",
        "| Dataset | Clean Acc | Clean Macro-F1 | Clean ECE | Mixed@1 RobAcc | "
        "Mixed@1 RobF1 | Mixed ASR | Mixed RelDrop | Mixed ECE | "
        "Adaptive@1 RobAcc | Adaptive@1 RobF1 | Adaptive ASR | "
        "Adaptive RelDrop | Adaptive ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for agg in aggregate_rows:
        def fm(k):
            return fmt_ms(agg[k]) if agg[k] is not None else "NA"

        lines.append(
            f"| {agg['dataset']} | "
            f"{fm('clean_acc')} | {fm('clean_f1')} | {fm('clean_ece')} | "
            f"{fm('mixed_robust_acc')} | {fm('mixed_robust_f1')} | "
            f"{fm('mixed_asr')} | {fm('mixed_rel_drop')} | {fm('mixed_ece')} | "
            f"{fm('adaptive_robust_acc')} | {fm('adaptive_robust_f1')} | "
            f"{fm('adaptive_asr')} | {fm('adaptive_rel_drop')} | "
            f"{fm('adaptive_ece')} |"
        )

    lines += [
        "",
        "## Protocol notes",
        "",
        f"- train seeds: {SEEDS}",
        f"- split seed: {SPLIT_SEED}",
        f"- attack base seed: {ATTACK_BASE_SEED}",
        f"- optimizer: SGD, lr={LEARNING_RATE}, batch={BATCH_SIZE}",
        f"- source TRAIN_ROUND: {SOURCE_TRAIN_ROUND}; actual updates: {TRAIN_STEPS}",
        "- checkpoint rule: minimum clean-validation mean CE; no test-driven selection",
        f"- route-prefix stripping: first {PACKET_PREFIX_STRIP_BYTES} bytes of each packet",
        "- IAT is physically perturbed but is representation-invisible to USTC-CNN",
        "- Adaptive is deterministic semantics-preserving structural perturbation, "
        "not a gradient-based white-box attack",
        "- prediction probabilities are saved for provenance-matched ROC regeneration",
        "",
        f"Architecture source: {ARCHITECTURE_SOURCE}",
    ]
    PAPER_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


# =============================================================================
# 9. MANIFEST / CLI
# =============================================================================

def write_run_manifest(
    specs: Sequence[DatasetSpec],
    args,
    device: torch.device,
):
    manifest = {
        "baseline": "USTC-CNN [31]",
        "citation": BASELINE_CITATION,
        "architecture_source": ARCHITECTURE_SOURCE,
        "reproduction_scope": (
            "architecture-faithful, protocol-adapted strict-input reproduction"
        ),
        "script": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "device": str(device),
        "torch_version": torch.__version__,
        "seeds": SEEDS,
        "training": {
            "regime": "clean-only baseline",
            "optimizer": "SGD",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "source_train_round": SOURCE_TRAIN_ROUND,
            "train_steps": TRAIN_STEPS,
            "training_loss": "summed cross entropy",
            "checkpoint_rule": "minimum clean-validation mean CE",
            "validation_every_steps": VALIDATE_EVERY,
            "robust_training_added": False,
        },
        "input": {
            "bytes": IMAGE_BYTES,
            "shape": [1, IMAGE_SIDE, IMAGE_SIDE],
            "strip_first_bytes_per_packet": PACKET_PREFIX_STRIP_BYTES,
            "iat_visible_to_model": False,
        },
        "evaluation": {
            "split_seed": SPLIT_SEED,
            "attack_base_seed": ATTACK_BASE_SEED,
            "primary_mixed_strength": PRIMARY_MIXED_STRENGTH,
            "primary_adaptive_strength": PRIMARY_ADAPTIVE_STRENGTH,
            "ece_bins": ECE_BINS,
            "asr_definition": (
                "fraction of clean-correct samples misclassified after attack"
            ),
            "rel_drop_definition": (
                "(clean_accuracy - robust_accuracy) / clean_accuracy * 100"
            ),
            "adaptive_is_gradient_white_box": False,
        },
        "datasets": [
            {
                "name": s.name,
                "root": str(s.root),
                "classes": s.num_classes,
                "counts": s.expected_counts,
                "dataset_manifest_sha256": s.expected_dataset_manifest,
                "split_manifest": str(s.split_manifest),
                "split_sha256": s.expected_split_sha,
            }
            for s in specs
        ],
        "cli": vars(args),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    dump_json(RUN_MANIFEST, manifest)


def parse_args():
    p = argparse.ArgumentParser(
        description="USTC-CNN formal baseline under frozen V17 evaluation protocol"
    )
    p.add_argument(
        "--mode",
        choices=["prepare", "train", "evaluate", "aggregate", "all"],
        default=os.environ.get("USTCCNN_FINAL_MODE", "all"),
    )
    p.add_argument(
        "--datasets",
        choices=["both", "cipher", "ustc"],
        default=os.environ.get("USTCCNN_FINAL_DATASETS", "both"),
    )
    p.add_argument(
        "--force-train",
        action="store_true",
        help="retrain even if a seed checkpoint already exists",
    )
    return p.parse_args()


def main():
    args = parse_args()
    specs = build_dataset_specs(args.datasets)

    if not torch.cuda.is_available():
        LOGGER.log(
            "[warning] CUDA not available; CPU execution is allowed for "
            "functional testing but formal timing comparisons must use the paper GPU."
        )
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    LOGGER.log("=" * 104)
    LOGGER.log("USTC-CNN [31] FINAL PAPER BASELINE")
    LOGGER.log(f"run_dir={RUN_DIR}")
    LOGGER.log(f"mode={args.mode} datasets={args.datasets} device={device}")
    LOGGER.log(
        "Method fidelity: clean-only SGD baseline; V17 robust training is NOT added."
    )
    LOGGER.log("=" * 104)

    for spec in specs:
        validate_dataset_spec(spec)
    write_run_manifest(specs, args, device)

    if args.mode in {"prepare", "all"}:
        for spec in specs:
            prepare_dataset(spec)

    if args.mode in {"train", "all"}:
        for spec in specs:
            # Preparation must already exist if train-only mode is used.
            required = [
                prepared_npz(spec, "train_clean"),
                prepared_npz(spec, "validation_clean"),
            ]
            if not all(p.is_file() for p in required):
                raise FileNotFoundError(
                    f"{spec.name}: prepared train/validation cache missing; run --mode prepare"
                )
            for seed in SEEDS:
                train_one(
                    spec,
                    seed,
                    device,
                    force=args.force_train,
                )

    if args.mode in {"evaluate", "all"}:
        for spec in specs:
            required = [
                prepared_npz(spec, "test_clean"),
                prepared_npz(spec, "test_mixed_1.0"),
                prepared_npz(spec, "test_adaptive_1.0"),
            ]
            if not all(p.is_file() for p in required):
                raise FileNotFoundError(
                    f"{spec.name}: prepared test cache missing; run --mode prepare"
                )
            for seed in SEEDS:
                evaluate_seed(spec, seed, device)

    if args.mode in {"aggregate", "all"}:
        aggregate(specs)

    LOGGER.log("[done] USTC-CNN formal baseline workflow completed.")


if __name__ == "__main__":
    main()
