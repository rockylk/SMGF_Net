# -*- coding: utf-8 -*-
"""
Official ET-BERT final-paper baseline runner for SMGF-Net comparison.

Purpose
-------
Recompute the ET-BERT rows used in the SMGF-Net paper under a fully traceable
protocol for BOTH datasets:
  1) CipherSpectrum
  2) USTC-TFC2016

The runner uses:
- the official ET-BERT repository and official pretrained checkpoint;
- the exact frozen SMGF-Net train/validation/test split manifests;
- train seeds 42 / 43 / 44;
- clean evaluation;
- Mixed attack at strength = 1.0;
- Adaptive attack at strength = 1.0;
- probability outputs so that Accuracy, Macro-F1, ASR, relative drop, ECE,
  and macro one-vs-rest AUROC can all be recomputed.

IMPORTANT SCIENTIFIC NOTE
-------------------------
SMGF-Net's perturbation generator is defined in feature space. ET-BERT consumes
raw packet bytes. Therefore an exact tensor-by-tensor reuse of the SMGF attack
cache is impossible without changing ET-BERT's input representation.

This runner translates the SAME PHYSICAL PERTURBATION FAMILY to raw packet
sequences:
- packet-length padding;
- IAT jitter;
- dummy packet insertion.

Mixed:
- padding/dummy bytes use deterministic pseudo-random filler;
- timing jitter is applied;
- therefore byte-level content can be disturbed in addition to structure.

Adaptive:
- the same structural perturbation family is used;
- the payload bytes of genuine packets are preserved;
- padding and dummy payload use zero filler;
- timing changes remain semantics-preserving.

This preserves the intended distinction:
Mixed = brute-force compounded structural/byte corruption.
Adaptive = semantics-preserving structural manipulation.

Because ET-BERT does not expose SMGF-Net's separate fingerprint/entropy anchors,
"Adaptive" cannot literally freeze those feature tensors. The runner records
this translation explicitly in run_manifest.json and the paper-ready report.

Default output
--------------
RESULT/ETBERT_FINAL_PAPER_BASELINE_V1/

Expected paper-facing outputs
-----------------------------
- clean Acc / Macro-F1 / ECE / macro AUROC
- Mixed@1.0 RobAcc / RobF1 / ASR / RelDrop / ECE / AUROC
- Adaptive@1.0 RobAcc / RobF1 / ASR / RelDrop / ECE / AUROC
- three-seed mean ± sample SD
- per-seed checkpoints, predictions, probabilities, manifests and logs

Run modes
---------
ETBERT_FINAL_MODE=prepare
ETBERT_FINAL_MODE=train
ETBERT_FINAL_MODE=evaluate
ETBERT_FINAL_MODE=aggregate
ETBERT_FINAL_MODE=all

Dataset selection
-----------------
ETBERT_FINAL_DATASETS=both       # default
ETBERT_FINAL_DATASETS=cipher
ETBERT_FINAL_DATASETS=ustc

Typical server setup
--------------------
cd /home/admin1/lkfiles
export PYTHONPATH=/home/admin1/lkfiles/ET-BERT
export MKL_THREADING_LAYER=GNU
unset MKL_SERVICE_FORCE_INTEL

nohup env ETBERT_FINAL_MODE=all ETBERT_FINAL_DATASETS=both \
    python -u ET-BERT_final_paper_experiments.py \
    > etbert_final_baseline.log 2>&1 &

Required assets
---------------
ETBERT_ROOT=/home/admin1/lkfiles/ET-BERT
ETBERT_PRETRAINED=/home/admin1/lkfiles/ET-BERT/models/pre-trained_model.bin

Optional data-root overrides
----------------------------
ETBERT_CIPHER_ROOT=/home/admin1/dataset/aes-128-gcm/aes-128-gcm
ETBERT_USTC_ROOT=/home/admin1/dataset/USTC-TFC2016-Split
ETBERT_RESULT_ROOT=/home/admin1/lkfiles/RESULT
ETBERT_FINAL_RUN_DIR=/custom/output/path
ETBERT_CIPHER_SPLIT_MANIFEST=/path/to/split_manifest.json
ETBERT_USTC_SPLIT_MANIFEST=/path/to/split_manifest.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from scapy.all import IP, TCP, UDP, Raw, rdpcap


# =============================================================================
# 0. GLOBAL CONFIGURATION
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
WORK_ROOT = Path(
    os.environ.get("SMGF_WORK_ROOT", str(Path.cwd()))
).expanduser().resolve()

RESULT_ROOT = Path(
    os.environ.get(
        "ETBERT_RESULT_ROOT",
        str(WORK_ROOT / "RESULT"),
    )
).expanduser().resolve()

RUN_DIR = Path(
    os.environ.get(
        "ETBERT_FINAL_RUN_DIR",
        str(RESULT_ROOT / "ETBERT_FINAL_PAPER_BASELINE_V1"),
    )
).expanduser().resolve()

MODE = os.environ.get("ETBERT_FINAL_MODE", "all").strip().lower()
if MODE not in {"prepare", "train", "evaluate", "aggregate", "all"}:
    raise ValueError(
        "ETBERT_FINAL_MODE must be prepare/train/evaluate/aggregate/all"
    )

DATASET_SELECTION = os.environ.get(
    "ETBERT_FINAL_DATASETS", "both"
).strip().lower()
if DATASET_SELECTION not in {"both", "cipher", "ustc"}:
    raise ValueError(
        "ETBERT_FINAL_DATASETS must be both/cipher/ustc"
    )

SEEDS = [42, 43, 44]
EPOCHS = 10
BATCH_SIZE = 32
INFER_BATCH_SIZE = 64
LEARNING_RATE = 2e-5
SEQ_LENGTH = 128
FLOW_PACKET_LIMIT = 5
BIGRAM_TOKENS_PER_PACKET = 128

ECE_BINS = 15
PRIMARY_MIXED_STRENGTH = 1.0
PRIMARY_ADAPTIVE_STRENGTH = 1.0
ATTACK_BASE_SEED = 20260923

# Optional full sweep for future sensitivity analysis.
FULL_SWEEP = os.environ.get("ETBERT_FINAL_FULL_SWEEP", "0") == "1"
if FULL_SWEEP:
    MIXED_STRENGTHS = [0.3, 0.6, 0.9, 1.0]
    ADAPTIVE_STRENGTHS = [0.4, 0.7, 1.0]
else:
    MIXED_STRENGTHS = [PRIMARY_MIXED_STRENGTH]
    ADAPTIVE_STRENGTHS = [PRIMARY_ADAPTIVE_STRENGTH]

LOG_FILE = RUN_DIR / "experiment.log"
RUN_MANIFEST = RUN_DIR / "run_manifest.json"
SUMMARY_JSON = RUN_DIR / "summary.json"
PAPER_MD = RUN_DIR / "PAPER_READY_ETBERT_BASELINE.md"
PER_SEED_CSV = RUN_DIR / "ETBERT_per_seed_primary.csv"
AGGREGATE_CSV = RUN_DIR / "ETBERT_aggregate_primary.csv"

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


# =============================================================================
# 1. LOGGING / HASH / UTILITY
# =============================================================================

class Logger:
    def __init__(self, path: Path):
        self.path = path

    def log(self, message=""):
        msg = str(message)
        print(msg, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")


LOGGER: Optional[Logger] = None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def stable_seed(*parts, modulo: int = 2**31 - 1) -> int:
    text = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo


def git_identity(repo: Path):
    result = {"head": None, "remote": None}
    try:
        result["head"] = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        result["remote"] = subprocess.check_output(
            ["git", "-C", str(repo), "config", "--get", "remote.origin.url"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        pass
    return result


def mean_sd(values: Sequence[float]):
    vals = [float(v) for v in values]
    return {
        "mean": statistics.mean(vals),
        "sample_sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
    }


def fmt_ms(stats, digits=2):
    return (
        f"{stats['mean']:.{digits}f} ± "
        f"{stats['sample_sd']:.{digits}f}"
    )


# =============================================================================
# 2. OFFICIAL ET-BERT DISCOVERY
# =============================================================================

def discover_etbert_root() -> Path:
    explicit = os.environ.get("ETBERT_ROOT")
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(p)
        return p

    candidates = [
        WORK_ROOT / "ET-BERT",
        SCRIPT_DIR / "ET-BERT",
        SCRIPT_DIR.parent / "ET-BERT",
        Path("/home/admin1/lkfiles/ET-BERT"),
    ]
    for p in candidates:
        if p.is_dir():
            return p.resolve()

    raise FileNotFoundError(
        "Official ET-BERT repository not found. Set ETBERT_ROOT."
    )


ETBERT_ROOT = discover_etbert_root()
ETBERT_PYTHON = os.environ.get("ETBERT_PYTHON", sys.executable)

OFFICIAL_TRAIN_SCRIPT = (
    ETBERT_ROOT / "fine-tuning" / "run_classifier.py"
)
OFFICIAL_INFER_SCRIPT = (
    ETBERT_ROOT / "inference" / "run_classifier_infer.py"
)
OFFICIAL_PREPROCESS_SCRIPT = (
    ETBERT_ROOT / "data_process" / "dataset_generation.py"
)


def discover_file(
    env_name: str,
    candidates: Sequence[Path],
    description: str,
):
    explicit = os.environ.get(env_name)
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p

    for p in candidates:
        if p.is_file():
            return p.resolve()

    raise FileNotFoundError(
        f"{description} not found. Set {env_name}."
    )


VOCAB_PATH = discover_file(
    "ETBERT_VOCAB",
    [
        ETBERT_ROOT / "models" / "encryptd_vocab.txt",
        ETBERT_ROOT / "models" / "encrypted_vocab.txt",
    ],
    "Official ET-BERT encrypted vocabulary",
)

CONFIG_PATH = discover_file(
    "ETBERT_CONFIG",
    [
        ETBERT_ROOT / "models" / "bert" / "base_config.json",
        ETBERT_ROOT / "bert_base_config.json",
    ],
    "Official ET-BERT base config",
)

PRETRAINED_PATH = discover_file(
    "ETBERT_PRETRAINED",
    [
        ETBERT_ROOT / "models" / "pre-trained_model.bin",
        ETBERT_ROOT / "models" / "pretrained_model.bin",
        ETBERT_ROOT / "pre-trained_model.bin",
        ETBERT_ROOT / "pretrained_model.bin",
    ],
    "Official ET-BERT pretrained checkpoint",
)


def validate_official_etbert():
    for p in [
        OFFICIAL_TRAIN_SCRIPT,
        OFFICIAL_INFER_SCRIPT,
        OFFICIAL_PREPROCESS_SCRIPT,
        VOCAB_PATH,
        CONFIG_PATH,
        PRETRAINED_PATH,
    ]:
        if not p.is_file():
            raise FileNotFoundError(p)

    train_text = OFFICIAL_TRAIN_SCRIPT.read_text(
        encoding="utf-8",
        errors="ignore",
    )
    markers = [
        "class Classifier",
        "pretrained_model_path",
        "str2embedding",
        "str2encoder",
    ]
    if not all(x in train_text for x in markers):
        raise RuntimeError(
            "ETBERT_ROOT does not match the official ET-BERT code layout."
        )


def prepare_patched_train_script():
    """
    Preserve official code semantics but remove the author's machine-specific
    /data2/... confusion-matrix output path.

    We patch a private copy inside RUN_DIR; the official repository is not
    modified by this runner.
    """
    src = OFFICIAL_TRAIN_SCRIPT.read_text(
        encoding="utf-8",
        errors="strict",
    )

    old = (
        'with open("/data2/lxj/pre-train/results/'
        'confusion_matrix",\'w\') as f:'
    )
    new = (
        'with open(args.output_model_path + '
        '".confusion_matrix", "w") as f:'
    )

    patched = src.replace(old, new)

    # The user's local official checkout may already have been manually fixed.
    if "/data2/lxj/pre-train/results/confusion_matrix" in patched:
        raise RuntimeError(
            "Could not safely patch the hard-coded confusion-matrix path."
        )

    path = RUN_DIR / "_patched_official_run_classifier.py"
    path.write_text(patched, encoding="utf-8")
    return path


# =============================================================================
# 3. DATASET / SPLIT CONFIG
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
):
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

    wanted_sha = EXPECTED_SPLIT_SHA[dataset_name]

    for root in roots:
        if not root.exists():
            continue
        for pattern in glob_patterns:
            for p in root.glob(pattern):
                if (
                    p.is_file()
                    and sha256_file(p) == wanted_sha
                ):
                    return p.resolve()

    raise FileNotFoundError(
        f"Cannot locate frozen split manifest for {dataset_name}. "
        f"Set {env_name}."
    )


def build_dataset_specs():
    cipher_root = Path(
        os.environ.get(
            "ETBERT_CIPHER_ROOT",
            "/home/admin1/dataset/aes-128-gcm/aes-128-gcm",
        )
    ).expanduser().resolve()

    ustc_root = Path(
        os.environ.get(
            "ETBERT_USTC_ROOT",
            "/home/admin1/dataset/USTC-TFC2016-Split",
        )
    ).expanduser().resolve()

    specs = []

    if DATASET_SELECTION in {"both", "cipher"}:
        cipher_split = discover_split_manifest(
            "CipherSpectrum",
            "ETBERT_CIPHER_SPLIT_MANIFEST",
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
                split_manifest=cipher_split,
                num_classes=EXPECTED_CLASSES["CipherSpectrum"],
                expected_dataset_manifest=EXPECTED_DATASET_MANIFEST[
                    "CipherSpectrum"
                ],
                expected_split_sha=EXPECTED_SPLIT_SHA[
                    "CipherSpectrum"
                ],
                expected_counts=EXPECTED_COUNTS["CipherSpectrum"],
            )
        )

    if DATASET_SELECTION in {"both", "ustc"}:
        ustc_split = discover_split_manifest(
            "USTC-TFC2016",
            "ETBERT_USTC_SPLIT_MANIFEST",
            [
                "SMGF_V17_USTC_SEED42_FULL_F10000_*/split_manifest.json",
                "**/SMGF_V17_USTC_SEED42_FULL_F10000_*/split_manifest.json",
            ],
        )
        specs.append(
            DatasetSpec(
                key="ustc",
                name="USTC-TFC2016",
                root=ustc_root,
                split_manifest=ustc_split,
                num_classes=EXPECTED_CLASSES["USTC-TFC2016"],
                expected_dataset_manifest=EXPECTED_DATASET_MANIFEST[
                    "USTC-TFC2016"
                ],
                expected_split_sha=EXPECTED_SPLIT_SHA[
                    "USTC-TFC2016"
                ],
                expected_counts=EXPECTED_COUNTS["USTC-TFC2016"],
            )
        )

    return specs


DATASETS = build_dataset_specs()


def validate_dataset_spec(spec: DatasetSpec):
    if not spec.root.is_dir():
        raise FileNotFoundError(spec.root)

    if sha256_file(spec.split_manifest) != spec.expected_split_sha:
        raise RuntimeError(
            f"{spec.name}: split manifest SHA mismatch."
        )

    split = load_json(spec.split_manifest)
    manifest_value = split.get("dataset_manifest_sha256")
    if manifest_value != spec.expected_dataset_manifest:
        raise RuntimeError(
            f"{spec.name}: dataset manifest mismatch: "
            f"{manifest_value}"
        )

    counts = {
        "train": len(split["train"]),
        "validation": len(split["validation"]),
        "test": len(split["test"]),
    }
    if counts != spec.expected_counts:
        raise RuntimeError(
            f"{spec.name}: split counts drift: {counts}"
        )

    labels = sorted(
        {
            int(row["label"])
            for part in ("train", "validation", "test")
            for row in split[part]
        }
    )
    if labels != list(range(spec.num_classes)):
        raise RuntimeError(
            f"{spec.name}: unexpected labels {labels}"
        )

    for part in ("train", "validation", "test"):
        for row in split[part]:
            p = spec.root / row["relative_path"]
            if not p.is_file():
                raise FileNotFoundError(p)
            if "file_size" in row:
                if int(p.stat().st_size) != int(row["file_size"]):
                    raise RuntimeError(
                        f"{spec.name}: file-size drift: {p}"
                    )


# =============================================================================
# 4. ET-BERT TOKENIZATION
# =============================================================================

def official_bigram_generation(
    hex_string: str,
    packet_len: int = BIGRAM_TOKENS_PER_PACKET,
):
    tokens = []
    for i in range(max(0, len(hex_string) - 1)):
        if len(tokens) >= packet_len:
            break
        tokens.append(hex_string[i] + hex_string[i + 1])
    return tokens


def packets_to_etbert_text(packets) -> str:
    """
    Reproduce the official flow-level feature generation:
    first 5 packets -> bytes(packet).hex()[76:] -> overlapping bigrams.
    """
    tokens: List[str] = []
    for i, pkt in enumerate(packets):
        if i >= FLOW_PACKET_LIMIT:
            break
        packet_hex = bytes(pkt).hex()[76:]
        tokens.extend(
            official_bigram_generation(
                packet_hex,
                BIGRAM_TOKENS_PER_PACKET,
            )
        )
    return " ".join(tokens)


# =============================================================================
# 5. RAW-PACKET ATTACK TRANSLATION
# =============================================================================

def _raw_payload_bytes(pkt):
    if Raw in pkt:
        return bytes(pkt[Raw].load)
    return b""


def _replace_raw_payload(pkt, payload: bytes):
    q = pkt.copy()

    if Raw in q:
        q[Raw].load = payload
        return q

    # If no Raw layer exists, add one after the existing protocol stack.
    # For ET-BERT evaluation this is a deterministic byte-sequence mutation;
    # packets are not transmitted.
    if payload:
        q = q / Raw(load=payload)
    return q


def _filler_bytes(
    length: int,
    rng: np.random.Generator,
    semantics_preserving: bool,
):
    if length <= 0:
        return b""
    if semantics_preserving:
        return b"\x00" * length
    arr = rng.integers(
        0,
        256,
        size=length,
        dtype=np.uint8,
    )
    return arr.tobytes()


def attack_packet_length_padding_raw(
    packets,
    strength: float,
    np_rng: np.random.Generator,
    semantics_preserving: bool,
):
    out = [p.copy() for p in packets]
    n = len(out)
    if n == 0:
        return out

    k = max(1, int(n * (0.10 + 0.20 * strength)))
    idx = np_rng.choice(
        n,
        size=min(k, n),
        replace=False,
    )
    max_pad = max(int(64 + 512 * strength), 21)
    pads = np_rng.integers(
        20,
        max_pad,
        size=len(idx),
    )

    for pos, pad_len in zip(idx.tolist(), pads.tolist()):
        q = out[pos]
        payload = _raw_payload_bytes(q)
        filler = _filler_bytes(
            int(pad_len),
            np_rng,
            semantics_preserving,
        )
        out[pos] = _replace_raw_payload(
            q,
            payload + filler,
        )

    return out


def _packet_times(packets):
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
    if n <= 1:
        return out

    times = _packet_times(out)
    iats = np.diff(times, prepend=times[0])
    iats[0] = 0.0
    iats = np.clip(iats, 0.0, None)

    k = max(1, int(n * (0.20 + 0.30 * strength)))
    idx = np_rng.choice(
        n,
        size=min(k, n),
        replace=False,
    )

    amp = 0.10 + 0.70 * strength
    multipliers = np_rng.uniform(
        1.0 - amp,
        1.0 + amp,
        size=len(idx),
    )

    for pos, mult in zip(idx.tolist(), multipliers.tolist()):
        if pos == 0:
            continue
        iats[pos] = max(iats[pos] * float(mult), 0.0)

    rebuilt = np.cumsum(iats)
    base_time = times[0]
    for i, pkt in enumerate(out):
        pkt.time = base_time + float(rebuilt[i])

    return out


def _zero_dummy_from_base(
    base_pkt,
    target_total_size: int,
):
    """
    Create a deterministic dummy packet by copying a neighboring packet,
    zeroing its Raw payload, and adjusting zero-payload length toward the
    requested total serialized packet size.
    """
    q = base_pkt.copy()
    q = _replace_raw_payload(q, b"")

    header_size = len(bytes(q))
    payload_len = max(0, int(target_total_size) - header_size)
    if payload_len > 0:
        q = _replace_raw_payload(
            q,
            b"\x00" * payload_len,
        )
    return q


def _random_dummy_from_base(
    base_pkt,
    target_total_size: int,
    np_rng: np.random.Generator,
):
    q = base_pkt.copy()
    q = _replace_raw_payload(q, b"")
    header_size = len(bytes(q))
    payload_len = max(0, int(target_total_size) - header_size)
    if payload_len > 0:
        q = _replace_raw_payload(
            q,
            _filler_bytes(
                payload_len,
                np_rng,
                semantics_preserving=False,
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
    if n == 0:
        return out

    insert_num = max(
        1,
        int(n * (0.03 + 0.10 * strength)),
    )

    times = _packet_times(out)
    if len(times) > 1:
        iats = np.diff(times)
        positive = iats[iats > 0]
        median_iat = (
            float(np.median(positive))
            if len(positive)
            else 1e-6
        )
    else:
        median_iat = 1e-6

    positions = sorted(
        np_rng.integers(
            1,
            max(2, n),
            size=insert_num,
        ).tolist()
    )

    offset = 0
    for p in positions:
        base_idx = min(
            max(p - 1 + offset, 0),
            len(out) - 1,
        )
        base = out[base_idx]
        target_size = py_rng.uniform(
            40,
            200 + 400 * strength,
        )

        if semantics_preserving:
            dummy = _zero_dummy_from_base(
                base,
                int(target_size),
            )
        else:
            dummy = _random_dummy_from_base(
                base,
                int(target_size),
                np_rng,
            )

        base_time = float(getattr(base, "time", 0.0))
        dummy_gap = max(
            median_iat * py_rng.uniform(0.2, 1.0),
            1e-6,
        )
        dummy.time = base_time + dummy_gap

        insert_at = min(
            p + offset,
            len(out),
        )
        out.insert(insert_at, dummy)
        offset += 1

    # Enforce monotonic packet order after insertion.
    out.sort(
        key=lambda pkt: float(
            getattr(pkt, "time", 0.0)
        )
    )
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
    out = attack_iat_jitter_raw(
        out,
        strength,
        np_rng,
    )
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

    np_rng = np.random.default_rng(attack_seed)
    py_rng = random.Random(attack_seed)

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
# 6. PREPARE CLEAN / ATTACK TSV FILES
# =============================================================================

def dataset_dir(spec: DatasetSpec):
    return RUN_DIR / spec.key


def tsv_dir(spec: DatasetSpec):
    return dataset_dir(spec) / "datasets"


def model_dir(spec: DatasetSpec, seed: int):
    return dataset_dir(spec) / "models" / f"seed{seed}"


def prediction_dir(spec: DatasetSpec, seed: int):
    return dataset_dir(spec) / "predictions" / f"seed{seed}"


def prep_meta_path(spec: DatasetSpec):
    return dataset_dir(spec) / "dataset_preparation.json"


def clean_tsv(spec: DatasetSpec, part: str):
    return tsv_dir(spec) / f"{part}_clean.tsv"


def attack_tsv(
    spec: DatasetSpec,
    attack: str,
    strength: float,
):
    return (
        tsv_dir(spec)
        / f"test_{attack}_{strength:.1f}.tsv"
    )


def write_tsv(path: Path, rows: Iterable[Tuple[int, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        w = csv.writer(
            f,
            delimiter="\t",
            lineterminator="\n",
        )
        w.writerow(["label", "text_a"])
        for label, text in rows:
            if not text:
                raise RuntimeError(
                    f"Empty ET-BERT text representation for {path}"
                )
            w.writerow([int(label), text])
            count += 1
    return count


def clean_rows_from_split(
    spec: DatasetSpec,
    split_rows,
):
    for pos, row in enumerate(split_rows, start=1):
        p = spec.root / row["relative_path"]
        packets = rdpcap(str(p))
        text = packets_to_etbert_text(packets)
        yield int(row["label"]), text

        if (
            LOGGER is not None
            and (pos % 500 == 0 or pos == len(split_rows))
        ):
            LOGGER.log(
                f"[{spec.name} clean] "
                f"{pos}/{len(split_rows)}"
            )


def attacked_rows_from_test(
    spec: DatasetSpec,
    test_rows,
    attack: str,
    strength: float,
):
    for pos, row in enumerate(test_rows, start=1):
        p = spec.root / row["relative_path"]
        packets = rdpcap(str(p))

        seed = stable_seed(
            ATTACK_BASE_SEED,
            spec.name,
            attack,
            f"{strength:.3f}",
            int(row["sample_index"]),
        )
        attacked = apply_attack_raw(
            packets,
            attack,
            strength,
            seed,
        )
        text = packets_to_etbert_text(attacked)
        yield int(row["label"]), text

        if (
            LOGGER is not None
            and (pos % 500 == 0 or pos == len(test_rows))
        ):
            LOGGER.log(
                f"[{spec.name} {attack}@{strength:.1f}] "
                f"{pos}/{len(test_rows)}"
            )


def preparation_valid(spec: DatasetSpec):
    meta_path = prep_meta_path(spec)
    if not meta_path.is_file():
        return False

    try:
        meta = load_json(meta_path)
    except Exception:
        return False

    if meta.get("split_sha256") != spec.expected_split_sha:
        return False
    if (
        meta.get("dataset_manifest_sha256")
        != spec.expected_dataset_manifest
    ):
        return False

    required = {
        "train_clean": clean_tsv(spec, "train"),
        "validation_clean": clean_tsv(spec, "validation"),
        "test_clean": clean_tsv(spec, "test"),
    }
    for s in MIXED_STRENGTHS:
        required[f"test_mixed_{s:.1f}"] = attack_tsv(
            spec, "mixed", s
        )
    for s in ADAPTIVE_STRENGTHS:
        required[f"test_adaptive_{s:.1f}"] = attack_tsv(
            spec, "adaptive", s
        )

    recorded = meta.get("tsv_sha256", {})
    for key, path in required.items():
        if not path.is_file():
            return False
        if recorded.get(key) != sha256_file(path):
            return False

    return True


def prepare_dataset(spec: DatasetSpec):
    assert LOGGER is not None

    if preparation_valid(spec):
        LOGGER.log(
            f"[{spec.name}] valid prepared TSV cache found; reuse."
        )
        return load_json(prep_meta_path(spec))

    validate_dataset_spec(spec)
    split = load_json(spec.split_manifest)
    tsv_dir(spec).mkdir(parents=True, exist_ok=True)

    LOGGER.log("=" * 104)
    LOGGER.log(
        f"PREPARE {spec.name}: clean + canonical attacked test TSVs"
    )
    LOGGER.log("=" * 104)

    counts = {}

    for part in ("train", "validation", "test"):
        path = clean_tsv(spec, part)
        counts[f"{part}_clean"] = write_tsv(
            path,
            clean_rows_from_split(
                spec,
                split[part],
            ),
        )

    for strength in MIXED_STRENGTHS:
        path = attack_tsv(
            spec,
            "mixed",
            strength,
        )
        counts[f"test_mixed_{strength:.1f}"] = write_tsv(
            path,
            attacked_rows_from_test(
                spec,
                split["test"],
                "mixed",
                strength,
            ),
        )

    for strength in ADAPTIVE_STRENGTHS:
        path = attack_tsv(
            spec,
            "adaptive",
            strength,
        )
        counts[f"test_adaptive_{strength:.1f}"] = write_tsv(
            path,
            attacked_rows_from_test(
                spec,
                split["test"],
                "adaptive",
                strength,
            ),
        )

    paths = {
        "train_clean": clean_tsv(spec, "train"),
        "validation_clean": clean_tsv(spec, "validation"),
        "test_clean": clean_tsv(spec, "test"),
    }
    for s in MIXED_STRENGTHS:
        paths[f"test_mixed_{s:.1f}"] = attack_tsv(
            spec, "mixed", s
        )
    for s in ADAPTIVE_STRENGTHS:
        paths[f"test_adaptive_{s:.1f}"] = attack_tsv(
            spec, "adaptive", s
        )

    meta = {
        "dataset": spec.name,
        "dataset_root": str(spec.root),
        "dataset_manifest_sha256":
            spec.expected_dataset_manifest,
        "split_manifest": str(spec.split_manifest),
        "split_sha256": spec.expected_split_sha,
        "num_classes": spec.num_classes,
        "counts": counts,
        "etbert_preprocess": {
            "flow_packet_limit": FLOW_PACKET_LIMIT,
            "hex_offset": 76,
            "bigram_tokens_per_packet":
                BIGRAM_TOKENS_PER_PACKET,
            "seq_length": SEQ_LENGTH,
        },
        "attack_translation": {
            "mixed": (
                "packet-length padding + IAT jitter + dummy insertion; "
                "padding/dummy filler bytes are deterministic random bytes"
            ),
            "adaptive": (
                "same structural perturbation family; genuine payload bytes "
                "remain unchanged; zero filler is used for padding/dummies"
            ),
            "attack_base_seed": ATTACK_BASE_SEED,
            "mixed_strengths": MIXED_STRENGTHS,
            "adaptive_strengths": ADAPTIVE_STRENGTHS,
        },
        "tsv_sha256": {
            key: sha256_file(path)
            for key, path in paths.items()
        },
        "created_at": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }
    dump_json(prep_meta_path(spec), meta)
    return meta


# =============================================================================
# 7. TRAIN OFFICIAL ET-BERT
# =============================================================================

def child_env():
    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"
    env.pop("MKL_SERVICE_FORCE_INTEL", None)

    existing = env.get("PYTHONPATH", "")
    parts = [
        str(ETBERT_ROOT),
        str(ETBERT_ROOT / "fine-tuning"),
    ]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def run_command(
    argv: Sequence[str],
    log_path: Path,
):
    assert LOGGER is not None
    LOGGER.log(
        "[CMD] " + " ".join(str(x) for x in argv)
    )

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        proc = subprocess.Popen(
            [str(x) for x in argv],
            cwd=str(ETBERT_ROOT),
            env=child_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None

        for line in proc.stdout:
            print(line, end="", flush=True)
            f.write(line)
            f.flush()

        code = proc.wait()

    if code != 0:
        raise RuntimeError(
            f"Command failed with exit code {code}. "
            f"See {log_path}"
        )


def train_command(
    patched_train_script: Path,
    spec: DatasetSpec,
    seed: int,
    output_model: Path,
):
    return [
        ETBERT_PYTHON,
        str(patched_train_script),
        "--pretrained_model_path",
        str(PRETRAINED_PATH),
        "--vocab_path",
        str(VOCAB_PATH),
        "--config_path",
        str(CONFIG_PATH),
        "--train_path",
        str(clean_tsv(spec, "train")),
        "--dev_path",
        str(clean_tsv(spec, "validation")),
        "--test_path",
        str(clean_tsv(spec, "test")),
        "--output_model_path",
        str(output_model),
        "--epochs_num",
        str(EPOCHS),
        "--batch_size",
        str(BATCH_SIZE),
        "--embedding",
        "word_pos_seg",
        "--encoder",
        "transformer",
        "--mask",
        "fully_visible",
        "--seq_length",
        str(SEQ_LENGTH),
        "--learning_rate",
        str(LEARNING_RATE),
        "--seed",
        str(seed),
        "--tokenizer",
        "bert",
    ]


def train_summary_path(spec: DatasetSpec, seed: int):
    return model_dir(spec, seed) / "train_summary.json"


def valid_trained_model(spec: DatasetSpec, seed: int):
    summary = train_summary_path(spec, seed)
    model = model_dir(spec, seed) / "finetuned.bin"
    if not summary.is_file() or not model.is_file():
        return False

    try:
        obj = load_json(summary)
    except Exception:
        return False

    if obj.get("seed") != seed:
        return False
    if obj.get("split_sha256") != spec.expected_split_sha:
        return False
    if obj.get("checkpoint_sha256") != sha256_file(model):
        return False

    return True


def train_one(
    spec: DatasetSpec,
    seed: int,
    patched_train_script: Path,
):
    assert LOGGER is not None

    if valid_trained_model(spec, seed):
        LOGGER.log(
            f"[{spec.name} seed={seed}] valid checkpoint exists; reuse."
        )
        return load_json(
            train_summary_path(spec, seed)
        )

    out = model_dir(spec, seed)
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / "finetuned.bin"
    run_command(
        train_command(
            patched_train_script,
            spec,
            seed,
            model_path,
        ),
        out / "train.log",
    )

    if not model_path.is_file():
        raise RuntimeError(
            f"Expected checkpoint not created: {model_path}"
        )

    summary = {
        "dataset": spec.name,
        "seed": seed,
        "split_sha256": spec.expected_split_sha,
        "checkpoint": str(model_path),
        "checkpoint_sha256": sha256_file(model_path),
        "pretrained_sha256": sha256_file(
            PRETRAINED_PATH
        ),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "seq_length": SEQ_LENGTH,
        "created_at": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }
    dump_json(
        train_summary_path(spec, seed),
        summary,
    )
    return summary


def train_all(
    specs: Sequence[DatasetSpec],
    patched_train_script: Path,
):
    for spec in specs:
        for seed in SEEDS:
            train_one(
                spec,
                seed,
                patched_train_script,
            )


# =============================================================================
# 8. OFFICIAL INFERENCE + METRICS
# =============================================================================

def prediction_file(
    spec: DatasetSpec,
    seed: int,
    condition: str,
):
    return prediction_dir(spec, seed) / f"{condition}.tsv"


def inference_command(
    spec: DatasetSpec,
    seed: int,
    test_path: Path,
    pred_path: Path,
):
    model_path = model_dir(spec, seed) / "finetuned.bin"

    return [
        ETBERT_PYTHON,
        str(OFFICIAL_INFER_SCRIPT),
        "--load_model_path",
        str(model_path),
        "--vocab_path",
        str(VOCAB_PATH),
        "--config_path",
        str(CONFIG_PATH),
        "--test_path",
        str(test_path),
        "--prediction_path",
        str(pred_path),
        "--labels_num",
        str(spec.num_classes),
        "--batch_size",
        str(INFER_BATCH_SIZE),
        "--seq_length",
        str(SEQ_LENGTH),
        "--embedding",
        "word_pos_seg",
        "--encoder",
        "transformer",
        "--mask",
        "fully_visible",
        "--tokenizer",
        "bert",
        "--output_prob",
    ]


def read_true_labels(path: Path):
    y = []
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for row in csv.DictReader(
            f,
            delimiter="\t",
        ):
            y.append(int(row["label"]))
    return np.asarray(y, dtype=np.int64)


def read_prediction_prob(path: Path):
    preds = []
    probs = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for row in csv.DictReader(
            f,
            delimiter="\t",
        ):
            preds.append(int(row["label"]))
            if "prob" not in row:
                raise RuntimeError(
                    f"Prediction file lacks probabilities: {path}"
                )
            probs.append(
                [float(v) for v in row["prob"].split()]
            )

    return (
        np.asarray(preds, dtype=np.int64),
        np.asarray(probs, dtype=np.float64),
    )


def expected_calibration_error(
    labels: np.ndarray,
    probs: np.ndarray,
    n_bins: int = ECE_BINS,
):
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correctness = (
        preds == labels
    ).astype(np.float64)

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )
    ece = 0.0
    n = len(labels)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            mask = (
                (confidences >= lo)
                & (confidences <= hi)
            )
        else:
            mask = (
                (confidences > lo)
                & (confidences <= hi)
            )

        count = int(mask.sum())
        if count == 0:
            continue

        acc_bin = correctness[mask].mean()
        conf_bin = confidences[mask].mean()
        ece += (
            count / max(n, 1)
        ) * abs(acc_bin - conf_bin)

    return float(ece * 100.0)


def multiclass_macro_auroc(
    labels: np.ndarray,
    probs: np.ndarray,
    num_classes: int,
):
    """
    Macro one-vs-rest AUROC in percent.
    Returns None if sklearn cannot compute a valid multiclass score.
    """
    try:
        y_bin = label_binarize(
            labels,
            classes=list(range(num_classes)),
        )
        if num_classes == 2:
            value = roc_auc_score(
                y_bin[:, 0],
                probs[:, 1],
            )
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


def evaluate_prediction(
    test_tsv: Path,
    pred_tsv: Path,
    num_classes: int,
):
    labels = read_true_labels(test_tsv)
    preds, probs = read_prediction_prob(pred_tsv)

    if len(labels) != len(preds):
        raise RuntimeError(
            f"Prediction length mismatch: "
            f"{len(labels)} vs {len(preds)}"
        )
    if probs.shape != (
        len(labels),
        num_classes,
    ):
        raise RuntimeError(
            f"Probability shape mismatch: {probs.shape}"
        )

    return {
        "acc": float(
            accuracy_score(labels, preds) * 100.0
        ),
        "macro_f1": float(
            f1_score(
                labels,
                preds,
                average="macro",
                zero_division=0,
            ) * 100.0
        ),
        "ece": expected_calibration_error(
            labels,
            probs,
            ECE_BINS,
        ),
        "macro_auroc": multiclass_macro_auroc(
            labels,
            probs,
            num_classes,
        ),
        "n": int(len(labels)),
        "labels": labels,
        "preds": preds,
        "probs": probs,
    }


def attack_metrics(
    clean_eval,
    attacked_eval,
):
    clean_labels = clean_eval["labels"]
    attacked_labels = attacked_eval["labels"]

    if not np.array_equal(
        clean_labels,
        attacked_labels,
    ):
        raise RuntimeError(
            "Attack label order differs from clean test order."
        )

    clean_correct = (
        clean_eval["preds"] == clean_labels
    )
    attack_failed = (
        attacked_eval["preds"] != attacked_labels
    )

    attack_success = int(
        np.sum(
            clean_correct & attack_failed
        )
    )
    clean_correct_count = int(
        np.sum(clean_correct)
    )

    robust_acc = attacked_eval["acc"]
    rel_drop = (
        (clean_eval["acc"] - robust_acc)
        / max(clean_eval["acc"], 1e-12)
        * 100.0
    )

    return {
        "robust_acc": robust_acc,
        "robust_f1": attacked_eval["macro_f1"],
        "asr": (
            attack_success
            / max(clean_correct_count, 1)
            * 100.0
        ),
        "rel_drop": rel_drop,
        "abs_drop_pp":
            clean_eval["acc"] - robust_acc,
        "ece": attacked_eval["ece"],
        "macro_auroc":
            attacked_eval["macro_auroc"],
    }


def compact_clean(eval_obj):
    return {
        k: v
        for k, v in eval_obj.items()
        if k not in {"labels", "preds", "probs"}
    }


def eval_summary_path(spec: DatasetSpec, seed: int):
    return (
        prediction_dir(spec, seed)
        / "evaluation_summary.json"
    )


def inference_condition(
    spec: DatasetSpec,
    seed: int,
    condition: str,
    input_tsv: Path,
):
    pred_path = prediction_file(
        spec,
        seed,
        condition,
    )
    pred_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_command(
        inference_command(
            spec,
            seed,
            input_tsv,
            pred_path,
        ),
        pred_path.with_suffix(".log"),
    )
    return pred_path


def evaluate_seed(
    spec: DatasetSpec,
    seed: int,
):
    assert LOGGER is not None

    if not valid_trained_model(spec, seed):
        raise RuntimeError(
            f"Missing valid trained model: "
            f"{spec.name} seed={seed}"
        )

    clean_pred = inference_condition(
        spec,
        seed,
        "clean",
        clean_tsv(spec, "test"),
    )
    clean_eval = evaluate_prediction(
        clean_tsv(spec, "test"),
        clean_pred,
        spec.num_classes,
    )

    attacks = {}

    for strength in MIXED_STRENGTHS:
        condition = f"mixed_{strength:.1f}"
        input_tsv = attack_tsv(
            spec,
            "mixed",
            strength,
        )
        pred = inference_condition(
            spec,
            seed,
            condition,
            input_tsv,
        )
        ev = evaluate_prediction(
            input_tsv,
            pred,
            spec.num_classes,
        )
        attacks[condition] = attack_metrics(
            clean_eval,
            ev,
        )

    for strength in ADAPTIVE_STRENGTHS:
        condition = f"adaptive_{strength:.1f}"
        input_tsv = attack_tsv(
            spec,
            "adaptive",
            strength,
        )
        pred = inference_condition(
            spec,
            seed,
            condition,
            input_tsv,
        )
        ev = evaluate_prediction(
            input_tsv,
            pred,
            spec.num_classes,
        )
        attacks[condition] = attack_metrics(
            clean_eval,
            ev,
        )

    summary = {
        "dataset": spec.name,
        "seed": seed,
        "split_sha256": spec.expected_split_sha,
        "checkpoint_sha256": sha256_file(
            model_dir(spec, seed) / "finetuned.bin"
        ),
        "clean": compact_clean(clean_eval),
        "attacks": attacks,
    }
    dump_json(
        eval_summary_path(spec, seed),
        summary,
    )

    primary_mixed = attacks[
        f"mixed_{PRIMARY_MIXED_STRENGTH:.1f}"
    ]
    primary_adaptive = attacks[
        f"adaptive_{PRIMARY_ADAPTIVE_STRENGTH:.1f}"
    ]

    LOGGER.log(
        f"[RESULT][{spec.name}][seed={seed}] CLEAN | "
        f"Acc={clean_eval['acc']:.2f} | "
        f"F1={clean_eval['macro_f1']:.2f} | "
        f"ECE={clean_eval['ece']:.2f}"
    )
    LOGGER.log(
        f"[RESULT][{spec.name}][seed={seed}] MIXED@1.0 | "
        f"RobAcc={primary_mixed['robust_acc']:.2f} | "
        f"RobF1={primary_mixed['robust_f1']:.2f} | "
        f"ASR={primary_mixed['asr']:.2f} | "
        f"RelDrop={primary_mixed['rel_drop']:.2f} | "
        f"ECE={primary_mixed['ece']:.2f}"
    )
    LOGGER.log(
        f"[RESULT][{spec.name}][seed={seed}] ADAPTIVE@1.0 | "
        f"RobAcc={primary_adaptive['robust_acc']:.2f} | "
        f"RobF1={primary_adaptive['robust_f1']:.2f} | "
        f"ASR={primary_adaptive['asr']:.2f} | "
        f"RelDrop={primary_adaptive['rel_drop']:.2f} | "
        f"ECE={primary_adaptive['ece']:.2f}"
    )

    return summary


def evaluate_all(specs):
    for spec in specs:
        for seed in SEEDS:
            evaluate_seed(
                spec,
                seed,
            )


# =============================================================================
# 9. AGGREGATION / PAPER READY OUTPUT
# =============================================================================

PRIMARY_METRICS = [
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


def flatten_primary(summary):
    clean = summary["clean"]
    mixed = summary["attacks"][
        f"mixed_{PRIMARY_MIXED_STRENGTH:.1f}"
    ]
    adaptive = summary["attacks"][
        f"adaptive_{PRIMARY_ADAPTIVE_STRENGTH:.1f}"
    ]

    return {
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
        "adaptive_robust_acc":
            adaptive["robust_acc"],
        "adaptive_robust_f1":
            adaptive["robust_f1"],
        "adaptive_asr": adaptive["asr"],
        "adaptive_rel_drop":
            adaptive["rel_drop"],
        "adaptive_ece": adaptive["ece"],
        "adaptive_auroc":
            adaptive["macro_auroc"],
    }


def aggregate_all(specs):
    assert LOGGER is not None

    per_seed_rows = []
    aggregate = {}

    for spec in specs:
        seed_rows = []

        for seed in SEEDS:
            path = eval_summary_path(spec, seed)
            if not path.is_file():
                raise FileNotFoundError(path)

            summary = load_json(path)
            flat = flatten_primary(summary)

            row = {
                "dataset": spec.name,
                "seed": seed,
                **flat,
            }
            per_seed_rows.append(row)
            seed_rows.append(row)

        aggregate[spec.name] = {}
        for metric in PRIMARY_METRICS:
            values = [
                r[metric]
                for r in seed_rows
                if r[metric] is not None
            ]
            aggregate[spec.name][metric] = mean_sd(values)

    with PER_SEED_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        fields = [
            "dataset",
            "seed",
            *PRIMARY_METRICS,
        ]
        w = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        w.writeheader()
        w.writerows(per_seed_rows)

    with AGGREGATE_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        fields = [
            "dataset",
            "metric",
            "mean",
            "sample_sd",
        ]
        w = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        w.writeheader()

        for dataset_name, metrics in aggregate.items():
            for metric, stats in metrics.items():
                w.writerow({
                    "dataset": dataset_name,
                    "metric": metric,
                    "mean": stats["mean"],
                    "sample_sd":
                        stats["sample_sd"],
                })

    final = {
        "experiment":
            "Official ET-BERT final-paper baseline rerun",
        "datasets": [
            s.name for s in specs
        ],
        "seeds": SEEDS,
        "official_etbert": {
            "root": str(ETBERT_ROOT),
            "git": git_identity(ETBERT_ROOT),
            "pretrained": str(PRETRAINED_PATH),
            "pretrained_sha256":
                sha256_file(PRETRAINED_PATH),
            "vocab_sha256":
                sha256_file(VOCAB_PATH),
            "config_sha256":
                sha256_file(CONFIG_PATH),
        },
        "protocol": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "seq_length": SEQ_LENGTH,
            "checkpoint_rule":
                "official ET-BERT best validation accuracy",
            "ece_bins": ECE_BINS,
            "primary_mixed_strength":
                PRIMARY_MIXED_STRENGTH,
            "primary_adaptive_strength":
                PRIMARY_ADAPTIVE_STRENGTH,
            "attack_translation":
                "raw-packet structural translation; see run_manifest.json",
        },
        "aggregate_mean_sample_sd": aggregate,
        "script_sha256": sha256_file(
            SCRIPT_PATH
        ),
    }
    dump_json(
        SUMMARY_JSON,
        final,
    )

    lines = []
    lines.append(
        "# Official ET-BERT final-paper baseline rerun"
    )
    lines.append("")
    lines.append(
        "Three independent fine-tuning seeds: 42, 43, 44."
    )
    lines.append("")
    lines.append(
        "The clean train/validation/test split is exactly the frozen "
        "SMGF-Net split for each dataset."
    )
    lines.append("")
    lines.append(
        "## Primary paper-facing results"
    )
    lines.append("")
    lines.append(
        "| Dataset | Clean Acc | Clean F1 | Clean ECE | "
        "Mixed RobAcc | Mixed RobF1 | Mixed ASR | Mixed RelDrop | "
        "Mixed ECE | Adaptive RobAcc | Adaptive RobF1 | Adaptive ASR | "
        "Adaptive RelDrop | Adaptive ECE |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: | ---: |"
    )

    for spec in specs:
        a = aggregate[spec.name]
        lines.append(
            f"| {spec.name} | "
            f"{fmt_ms(a['clean_acc'])} | "
            f"{fmt_ms(a['clean_f1'])} | "
            f"{fmt_ms(a['clean_ece'])} | "
            f"{fmt_ms(a['mixed_robust_acc'])} | "
            f"{fmt_ms(a['mixed_robust_f1'])} | "
            f"{fmt_ms(a['mixed_asr'])} | "
            f"{fmt_ms(a['mixed_rel_drop'])} | "
            f"{fmt_ms(a['mixed_ece'])} | "
            f"{fmt_ms(a['adaptive_robust_acc'])} | "
            f"{fmt_ms(a['adaptive_robust_f1'])} | "
            f"{fmt_ms(a['adaptive_asr'])} | "
            f"{fmt_ms(a['adaptive_rel_drop'])} | "
            f"{fmt_ms(a['adaptive_ece'])} |"
        )

    lines.append("")
    lines.append(
        "## Interpretation boundary"
    )
    lines.append("")
    lines.append(
        "SMGF-Net's original perturbation engine acts on extracted feature "
        "tensors. ET-BERT consumes raw bytes, so the runner translates the "
        "same packet-length, timing, and dummy-injection perturbation family "
        "to packet sequences before official ET-BERT preprocessing. The "
        "Adaptive variant preserves genuine packet payload bytes and uses "
        "zero filler for added padding/dummies. Therefore these values are "
        "a faithful architecture-level physical-perturbation comparison, "
        "not a claim that the internal tensors are identical across models."
    )
    lines.append("")
    lines.append(
        "Probability TSV files are retained for later ROC/Fig.5 regeneration."
    )

    PAPER_MD.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    LOGGER.log("=" * 104)
    LOGGER.log("ET-BERT FINAL BASELINE AGGREGATION COMPLETE")
    for spec in specs:
        a = aggregate[spec.name]
        LOGGER.log(
            f"{spec.name}: "
            f"Clean={fmt_ms(a['clean_acc'])} | "
            f"Mixed={fmt_ms(a['mixed_robust_acc'])} | "
            f"Adaptive={fmt_ms(a['adaptive_robust_acc'])}"
        )
    LOGGER.log(
        f"Paper-ready: {PAPER_MD}"
    )
    LOGGER.log("=" * 104)

    return final


# =============================================================================
# 10. RUN MANIFEST
# =============================================================================

def write_run_manifest(
    specs,
    patched_train_script: Path,
):
    manifest = {
        "experiment":
            "Official ET-BERT final-paper baseline rerun",
        "created_at": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "script": str(SCRIPT_PATH),
        "script_sha256":
            sha256_file(SCRIPT_PATH),
        "official_etbert": {
            "root": str(ETBERT_ROOT),
            "git": git_identity(ETBERT_ROOT),
            "official_train_script":
                str(OFFICIAL_TRAIN_SCRIPT),
            "official_train_sha256":
                sha256_file(OFFICIAL_TRAIN_SCRIPT),
            "patched_private_train_script":
                str(patched_train_script),
            "patched_train_sha256":
                sha256_file(patched_train_script),
            "inference_script":
                str(OFFICIAL_INFER_SCRIPT),
            "inference_sha256":
                sha256_file(OFFICIAL_INFER_SCRIPT),
            "preprocess_script":
                str(OFFICIAL_PREPROCESS_SCRIPT),
            "preprocess_sha256":
                sha256_file(OFFICIAL_PREPROCESS_SCRIPT),
            "pretrained":
                str(PRETRAINED_PATH),
            "pretrained_sha256":
                sha256_file(PRETRAINED_PATH),
            "vocab":
                str(VOCAB_PATH),
            "vocab_sha256":
                sha256_file(VOCAB_PATH),
            "config":
                str(CONFIG_PATH),
            "config_sha256":
                sha256_file(CONFIG_PATH),
        },
        "datasets": {
            spec.name: {
                "root": str(spec.root),
                "split_manifest":
                    str(spec.split_manifest),
                "split_sha256":
                    spec.expected_split_sha,
                "dataset_manifest_sha256":
                    spec.expected_dataset_manifest,
                "num_classes":
                    spec.num_classes,
                "split_counts":
                    spec.expected_counts,
            }
            for spec in specs
        },
        "training": {
            "seeds": SEEDS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "seq_length": SEQ_LENGTH,
            "checkpoint_rule":
                "official ET-BERT best validation accuracy",
        },
        "evaluation": {
            "ece_bins": ECE_BINS,
            "mixed_strengths":
                MIXED_STRENGTHS,
            "adaptive_strengths":
                ADAPTIVE_STRENGTHS,
            "primary_mixed_strength":
                PRIMARY_MIXED_STRENGTH,
            "primary_adaptive_strength":
                PRIMARY_ADAPTIVE_STRENGTH,
            "attack_base_seed":
                ATTACK_BASE_SEED,
            "asr_definition":
                "fraction of clean-correct samples misclassified after attack",
            "rel_drop_definition":
                "(CleanAcc-RobAcc)/CleanAcc*100",
            "auroc":
                "macro one-vs-rest",
        },
        "attack_translation": {
            "scientific_constraint": (
                "SMGF attacks are defined in feature space while ET-BERT "
                "uses raw bytes; exact internal-tensor cache reuse is impossible."
            ),
            "mixed": (
                "packet-length padding + IAT jitter + dummy insertion; "
                "deterministic random filler for newly added bytes"
            ),
            "adaptive": (
                "same structural perturbation family; payload bytes of genuine "
                "packets are preserved; added bytes are zero filler"
            ),
        },
    }
    dump_json(
        RUN_MANIFEST,
        manifest,
    )


# =============================================================================
# 11. MAIN
# =============================================================================

def main():
    global LOGGER

    RUN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    LOGGER = Logger(LOG_FILE)

    validate_official_etbert()

    for spec in DATASETS:
        validate_dataset_spec(spec)

    patched_train_script = (
        prepare_patched_train_script()
    )
    write_run_manifest(
        DATASETS,
        patched_train_script,
    )

    LOGGER.log("=" * 104)
    LOGGER.log(
        "OFFICIAL ET-BERT FINAL-PAPER BASELINE RERUN"
    )
    LOGGER.log("=" * 104)
    LOGGER.log(f"mode={MODE}")
    LOGGER.log(
        "datasets="
        + ",".join(s.name for s in DATASETS)
    )
    LOGGER.log(
        f"etbert_root={ETBERT_ROOT}"
    )
    LOGGER.log(
        f"pretrained={PRETRAINED_PATH}"
    )
    LOGGER.log(
        f"run_dir={RUN_DIR}"
    )
    LOGGER.log(
        f"repo_identity={json.dumps(git_identity(ETBERT_ROOT), ensure_ascii=False)}"
    )

    if MODE in {"prepare", "all"}:
        for spec in DATASETS:
            prepare_dataset(spec)

    if MODE in {"train", "all"}:
        for spec in DATASETS:
            if not preparation_valid(spec):
                prepare_dataset(spec)
        train_all(
            DATASETS,
            patched_train_script,
        )

    if MODE in {"evaluate", "all"}:
        for spec in DATASETS:
            if not preparation_valid(spec):
                prepare_dataset(spec)
        evaluate_all(DATASETS)

    if MODE in {"aggregate", "all"}:
        aggregate_all(DATASETS)

    LOGGER.log("DONE")


if __name__ == "__main__":
    main()
