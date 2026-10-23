# -*- coding: utf-8 -*-
"""
Shared frozen-protocol utilities for paper-facing non-ET-BERT baselines.

This module does NOT define a classifier or training objective. It only
standardizes the experimental boundary around each baseline:
  * exact final CipherSpectrum / USTC split manifests
  * exact sample identities and labels
  * deterministic Dir/Size/IAT extraction
  * V17-compatible deterministic structural attacks
  * common evaluation metrics and reproducibility helpers

The baseline runners preserve their own model architecture, optimizer,
training budget and clean-only objective.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from scapy.all import IP, TCP, UDP, rdpcap
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent

SEEDS = [42, 43, 44]
SPLIT_SEED = 42
ATTACK_BASE_SEED = 20260919

MULTI_ATTACKS = ["padding", "iat_jitter", "dummy", "mixed"]
MULTI_STRENGTHS = [0.3, 0.6, 0.9, 1.0]
ADAPTIVE_STRENGTHS = [0.4, 0.7, 1.0]
PRIMARY_MIXED_STRENGTH = 1.0
PRIMARY_ADAPTIVE_STRENGTH = 1.0
ECE_BINS = 15

RAW_MAX_PACKETS = 400
RAW_CACHE_PROTOCOL = "ip-tcpudp-dir-size-iat-first400-no-sample-drop-v1"

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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(*parts, modulo: int = 2**31 - 1) -> int:
    text = "|".join(str(p) for p in parts)
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


def mean_sd(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    return {
        "mean": statistics.mean(vals),
        "sample_sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
    }


def fmt_ms(stats: Dict[str, float], digits: int = 2) -> str:
    return (
        f"{stats['mean']:.{digits}f} ± "
        f"{stats['sample_sd']:.{digits}f}"
    )


def discover_split_manifest(
    dataset_name: str,
    env_name: str,
    glob_patterns: Sequence[str],
    work_root: Path,
    result_root: Path,
) -> Path:
    explicit = os.environ.get(env_name)
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p

    roots = [
        result_root,
        work_root / "RESULT",
        work_root,
        SCRIPT_DIR,
        SCRIPT_DIR.parent,
        Path("/home/admin1/lkfiles/RESULT"),
        Path("/home/admin1/lkfiles"),
    ]
    wanted = EXPECTED_SPLIT_SHA[dataset_name]
    seen = set()
    for root in roots:
        try:
            root = Path(root).expanduser().resolve()
        except Exception:
            continue
        token = str(root)
        if token in seen or not root.exists():
            continue
        seen.add(token)
        for pattern in glob_patterns:
            for p in root.glob(pattern):
                if p.is_file() and sha256_file(p) == wanted:
                    return p.resolve()

    raise FileNotFoundError(
        f"Cannot locate frozen split manifest for {dataset_name}. "
        f"Set {env_name} explicitly."
    )


def build_dataset_specs(
    selection: str,
    work_root: Path,
    result_root: Path,
    env_prefix: str,
) -> List[DatasetSpec]:
    if selection not in {"both", "cipher", "ustc"}:
        raise ValueError("selection must be both/cipher/ustc")

    cipher_root = Path(
        os.environ.get(
            f"{env_prefix}_CIPHER_ROOT",
            "/home/admin1/dataset/aes-128-gcm/aes-128-gcm",
        )
    ).expanduser().resolve()
    ustc_root = Path(
        os.environ.get(
            f"{env_prefix}_USTC_ROOT",
            "/home/admin1/dataset/USTC-TFC2016-Split",
        )
    ).expanduser().resolve()

    specs: List[DatasetSpec] = []
    if selection in {"both", "cipher"}:
        split = discover_split_manifest(
            "CipherSpectrum",
            f"{env_prefix}_CIPHER_SPLIT_MANIFEST",
            [
                "SMGF_V17_CIPHERSPECTRUM_FINAL_3SEED_*/split_manifest.json",
                "**/SMGF_V17_CIPHERSPECTRUM_FINAL_3SEED_*/split_manifest.json",
            ],
            work_root,
            result_root,
        )
        specs.append(
            DatasetSpec(
                key="cipher",
                name="CipherSpectrum",
                root=cipher_root,
                split_manifest=split,
                num_classes=EXPECTED_CLASSES["CipherSpectrum"],
                expected_dataset_manifest=EXPECTED_DATASET_MANIFEST[
                    "CipherSpectrum"
                ],
                expected_split_sha=EXPECTED_SPLIT_SHA["CipherSpectrum"],
                expected_counts=EXPECTED_COUNTS["CipherSpectrum"],
            )
        )

    if selection in {"both", "ustc"}:
        split = discover_split_manifest(
            "USTC-TFC2016",
            f"{env_prefix}_USTC_SPLIT_MANIFEST",
            [
                "SMGF_V17_USTC_SEED42_FULL_F10000_*/split_manifest.json",
                "**/SMGF_V17_USTC_SEED42_FULL_F10000_*/split_manifest.json",
                "SMGF_V17_USTC_FINAL_3SEED_*/split_manifest.json",
                "**/SMGF_V17_USTC_FINAL_3SEED_*/split_manifest.json",
            ],
            work_root,
            result_root,
        )
        specs.append(
            DatasetSpec(
                key="ustc",
                name="USTC-TFC2016",
                root=ustc_root,
                split_manifest=split,
                num_classes=EXPECTED_CLASSES["USTC-TFC2016"],
                expected_dataset_manifest=EXPECTED_DATASET_MANIFEST[
                    "USTC-TFC2016"
                ],
                expected_split_sha=EXPECTED_SPLIT_SHA["USTC-TFC2016"],
                expected_counts=EXPECTED_COUNTS["USTC-TFC2016"],
            )
        )
    return specs


def validate_dataset_spec(spec: DatasetSpec) -> dict:
    if not spec.root.is_dir():
        raise FileNotFoundError(spec.root)
    if sha256_file(spec.split_manifest) != spec.expected_split_sha:
        raise RuntimeError(f"{spec.name}: frozen split SHA mismatch")

    split = load_json(spec.split_manifest)
    observed_manifest = split.get("dataset_manifest_sha256")
    if observed_manifest != spec.expected_dataset_manifest:
        raise RuntimeError(
            f"{spec.name}: dataset manifest mismatch: {observed_manifest}"
        )
    if int(split.get("split_seed", SPLIT_SEED)) != SPLIT_SEED:
        raise RuntimeError(
            f"{spec.name}: split seed is not frozen seed {SPLIT_SEED}"
        )

    counts = {
        "train": len(split["train"]),
        "validation": len(split["validation"]),
        "test": len(split["test"]),
    }
    if counts != spec.expected_counts:
        raise RuntimeError(
            f"{spec.name}: frozen split counts drift: {counts}"
        )

    labels = sorted({
        int(row["label"])
        for part in ("train", "validation", "test")
        for row in split[part]
    })
    if labels != list(range(spec.num_classes)):
        raise RuntimeError(
            f"{spec.name}: unexpected label set {labels}"
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
    return split


def extract_raw_x_from_pcap(
    pcap_path: Path,
    max_pkts: int = RAW_MAX_PACKETS,
):
    """
    Server-proxy representation: packet direction, serialized packet size and IAT.

    Unlike the historical proxy, the formal runner does not silently drop flows
    with fewer than five valid packets because the frozen V17 split fixes sample
    identity. A nonempty short flow is retained and its short-flow status is
    recorded in preparation metadata. A flow with zero valid IP/TCP-or-UDP
    packets is a hard protocol error.
    """
    packets = rdpcap(str(pcap_path))
    packets = [
        p for p in packets
        if IP in p and (TCP in p or UDP in p)
    ]
    if not packets:
        raise RuntimeError(
            f"No valid IP/TCP-or-UDP packets in frozen sample: {pcap_path}"
        )

    packets = packets[:max_pkts]
    client_ip = packets[0][IP].src
    prev_time = float(packets[0].time)
    raw_x = []
    for pkt in packets:
        curr_time = float(pkt.time)
        direction = 0.0 if pkt[IP].src == client_ip else 1.0
        size = float(len(pkt))
        iat = max(curr_time - prev_time, 0.0)
        raw_x.append([direction, size, iat])
        prev_time = curr_time
    return np.asarray(raw_x, dtype=np.float32)


def raw_cache_dir(cache_root: Path, spec: DatasetSpec) -> Path:
    return Path(cache_root) / spec.key


def raw_cache_path(
    cache_root: Path,
    spec: DatasetSpec,
    part: str,
) -> Path:
    return raw_cache_dir(cache_root, spec) / f"{part}_raw.npz"


def raw_cache_meta_path(
    cache_root: Path,
    spec: DatasetSpec,
) -> Path:
    return raw_cache_dir(cache_root, spec) / "raw_cache_manifest.json"


def pack_sequences(
    sequences: Sequence[np.ndarray],
):
    offsets = np.zeros(len(sequences) + 1, dtype=np.int64)
    total = 0
    for i, seq in enumerate(sequences):
        total += int(len(seq))
        offsets[i + 1] = total
    if total:
        values = np.concatenate(sequences, axis=0).astype(np.float32)
    else:
        values = np.empty((0, 3), dtype=np.float32)
    return values, offsets


def unpack_sequences(
    values: np.ndarray,
    offsets: np.ndarray,
) -> List[np.ndarray]:
    out = []
    for i in range(len(offsets) - 1):
        a = int(offsets[i])
        b = int(offsets[i + 1])
        out.append(
            np.asarray(values[a:b], dtype=np.float32).copy()
        )
    return out


def save_raw_partition(
    path: Path,
    sequences: Sequence[np.ndarray],
    labels: np.ndarray,
    sample_indices: np.ndarray,
):
    values, offsets = pack_sequences(sequences)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        values=values,
        offsets=offsets,
        labels=np.asarray(labels, dtype=np.int64),
        sample_indices=np.asarray(sample_indices, dtype=np.int64),
    )


def load_raw_partition(path: Path):
    with np.load(path, allow_pickle=False) as z:
        values = np.asarray(z["values"], dtype=np.float32)
        offsets = np.asarray(z["offsets"], dtype=np.int64)
        labels = np.asarray(z["labels"], dtype=np.int64)
        sample_indices = np.asarray(
            z["sample_indices"], dtype=np.int64
        )
    seqs = unpack_sequences(values, offsets)
    if len(seqs) != len(labels):
        raise RuntimeError(
            f"Raw cache sequence/label mismatch: {path}"
        )
    return seqs, labels, sample_indices


def raw_cache_valid(
    cache_root: Path,
    spec: DatasetSpec,
) -> bool:
    meta_path = raw_cache_meta_path(cache_root, spec)
    if not meta_path.is_file():
        return False
    try:
        meta = load_json(meta_path)
    except Exception:
        return False

    expected = {
        "common_script_sha256": sha256_file(SCRIPT_PATH),
        "raw_cache_protocol": RAW_CACHE_PROTOCOL,
        "dataset_manifest_sha256":
            spec.expected_dataset_manifest,
        "split_sha256": spec.expected_split_sha,
        "split_seed": SPLIT_SEED,
        "raw_max_packets": RAW_MAX_PACKETS,
    }
    if any(meta.get(k) != v for k, v in expected.items()):
        return False

    for part, n in spec.expected_counts.items():
        path = raw_cache_path(cache_root, spec, part)
        if not path.is_file():
            return False
        if meta.get("cache_sha256", {}).get(part) != sha256_file(path):
            return False
        try:
            seqs, labels, sample_indices = load_raw_partition(path)
        except Exception:
            return False
        if len(seqs) != n:
            return False
        if labels.shape != (n,) or sample_indices.shape != (n,):
            return False
    return True


def prepare_raw_cache(
    spec: DatasetSpec,
    cache_root: Path,
    logger,
):
    split = validate_dataset_spec(spec)
    if raw_cache_valid(cache_root, spec):
        logger.log(
            f"[raw-cache][{spec.name}] reuse protocol-matched cache"
        )
        return load_json(raw_cache_meta_path(cache_root, spec))

    short_counts = {}
    cache_hashes = {}
    for part in ("train", "validation", "test"):
        rows = split[part]
        seqs = []
        labels = np.zeros(len(rows), dtype=np.int64)
        sample_indices = np.zeros(len(rows), dtype=np.int64)
        short = 0

        for pos, row in enumerate(rows):
            p = spec.root / row["relative_path"]
            seq = extract_raw_x_from_pcap(p)
            if len(seq) < 5:
                short += 1
            seqs.append(seq)
            labels[pos] = int(row["label"])
            sample_indices[pos] = int(
                row.get("sample_index", pos)
            )

            if (pos + 1) % 500 == 0 or pos + 1 == len(rows):
                logger.log(
                    f"[raw-cache][{spec.name}][{part}] "
                    f"{pos + 1}/{len(rows)}"
                )

        path = raw_cache_path(cache_root, spec, part)
        save_raw_partition(
            path, seqs, labels, sample_indices
        )
        short_counts[part] = int(short)
        cache_hashes[part] = sha256_file(path)

    meta = {
        "dataset": spec.name,
        "common_script_sha256": sha256_file(SCRIPT_PATH),
        "raw_cache_protocol": RAW_CACHE_PROTOCOL,
        "raw_max_packets": RAW_MAX_PACKETS,
        "dataset_manifest_sha256":
            spec.expected_dataset_manifest,
        "split_sha256": spec.expected_split_sha,
        "split_seed": SPLIT_SEED,
        "counts": spec.expected_counts,
        "short_flow_lt5_counts": short_counts,
        "historical_proxy_min_packets": 5,
        "formal_policy": (
            "No frozen sample is silently dropped. Nonempty flows with "
            "<5 valid packets are retained to preserve exact split identity."
        ),
        "cache_sha256": cache_hashes,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    dump_json(raw_cache_meta_path(cache_root, spec), meta)
    return meta


def sanitize_raw_x(raw_x: np.ndarray) -> np.ndarray:
    out = np.asarray(raw_x, dtype=np.float32).copy()
    if out.ndim != 2 or out.shape[1] != 3:
        raise ValueError(
            f"Expected raw_x shape [N,3], got {out.shape}"
        )
    out[:, 0] = np.rint(np.clip(out[:, 0], 0.0, 1.0))
    out[:, 1] = np.clip(out[:, 1], 1.0, None)
    out[:, 2] = np.clip(out[:, 2], 0.0, None)
    return out


def attack_packet_length_padding(
    raw_x: np.ndarray,
    strength: float,
    np_rng: np.random.Generator,
    py_rng: random.Random,
):
    out = sanitize_raw_x(raw_x)
    if strength <= 0:
        return out
    n = len(out)
    k = max(1, int(n * (0.10 + 0.20 * strength)))
    idx = np_rng.choice(n, size=min(k, n), replace=False)
    pads = np_rng.integers(
        20,
        max(int(64 + 512 * strength), 21),
        size=len(idx),
    )
    out[idx, 1] = np.clip(
        out[idx, 1] + pads, 1.0, None
    )
    return out


def attack_iat_jitter(
    raw_x: np.ndarray,
    strength: float,
    np_rng: np.random.Generator,
    py_rng: random.Random,
):
    out = sanitize_raw_x(raw_x)
    if strength <= 0:
        return out
    n = len(out)
    k = max(1, int(n * (0.20 + 0.30 * strength)))
    idx = np_rng.choice(n, size=min(k, n), replace=False)
    amp = 0.10 + 0.70 * strength
    jitter = np_rng.uniform(
        1.0 - amp, 1.0 + amp, size=len(idx)
    )
    out[idx, 2] = np.clip(
        out[idx, 2] * jitter, 0.0, None
    )
    return out


def attack_dummy_injection(
    raw_x: np.ndarray,
    strength: float,
    np_rng: np.random.Generator,
    py_rng: random.Random,
):
    out = sanitize_raw_x(raw_x)
    if strength <= 0:
        return out

    n = len(out)
    insert_num = max(1, int(n * (0.03 + 0.10 * strength)))
    median_iat = max(
        float(np.median(np.clip(out[:, 2], 1e-6, None))),
        1e-6,
    )
    positions = sorted(
        np_rng.integers(
            1, max(2, n), size=insert_num
        ).tolist()
    )
    new_list = out.tolist()
    offset = 0
    for p in positions:
        base_idx = min(
            max(p - 1 + offset, 0),
            len(new_list) - 1,
        )
        base = np.asarray(
            new_list[base_idx], dtype=np.float32
        )
        dummy_dir = (
            float(base[0])
            if py_rng.random() < 0.7
            else float(1.0 - base[0])
        )
        dummy_size = py_rng.uniform(
            40, 200 + 400 * strength
        )
        dummy_iat = max(
            median_iat * py_rng.uniform(0.2, 1.0),
            1e-6,
        )
        new_list.insert(
            min(p + offset, len(new_list)),
            [dummy_dir, dummy_size, dummy_iat],
        )
        offset += 1

    # Match the frozen V17 core exactly: attacked traces are
    # re-truncated to the same 400-packet observation window used by clean
    # extraction. Without this step dummy/mixed/adaptive attacks can expose
    # FS-Net to >400 packets even though SMGF-Net never sees beyond 400.
    return np.asarray(
        new_list[:RAW_MAX_PACKETS],
        dtype=np.float32,
    )


def attack_mixed(
    raw_x: np.ndarray,
    strength: float,
    np_rng: np.random.Generator,
    py_rng: random.Random,
):
    out = attack_packet_length_padding(
        raw_x, strength, np_rng, py_rng
    )
    out = attack_iat_jitter(
        out, strength, np_rng, py_rng
    )
    out = attack_dummy_injection(
        out, strength, np_rng, py_rng
    )
    return out


def apply_attack_deterministic(
    raw_x: np.ndarray,
    attack_name: str,
    strength: float,
    np_rng: np.random.Generator,
    py_rng: random.Random,
):
    if attack_name is None or attack_name.lower() == "clean":
        return sanitize_raw_x(raw_x)
    if float(strength) <= 0.0:
        return sanitize_raw_x(raw_x)

    name = attack_name.lower()
    if name == "padding":
        return attack_packet_length_padding(
            raw_x, strength, np_rng, py_rng
        )
    if name == "iat_jitter":
        return attack_iat_jitter(
            raw_x, strength, np_rng, py_rng
        )
    if name == "dummy":
        return attack_dummy_injection(
            raw_x, strength, np_rng, py_rng
        )
    if name in {"mixed", "adaptive"}:
        # V17 Adaptive preserves SMGF semantic anchors while perturbing the
        # same structural variables. FS-Net/DFE proxy representations have
        # no semantic-anchor branch, so the structural transformation is the
        # same family. The canonical condition seed remains attack-name-specific.
        return attack_mixed(
            raw_x, strength, np_rng, py_rng
        )
    raise ValueError(f"Unsupported attack: {attack_name}")


def canonical_attack_seed(
    attack_name: str,
    strength: float,
) -> int:
    return stable_seed(
        ATTACK_BASE_SEED,
        attack_name,
        f"{float(strength):.3f}",
    )


def build_attacked_raw_dataset(
    raw_sequences: Sequence[np.ndarray],
    attack_name: str,
    strength: float,
) -> List[np.ndarray]:
    seed = canonical_attack_seed(
        attack_name, strength
    )
    np_rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    return [
        apply_attack_deterministic(
            x,
            attack_name,
            strength,
            np_rng,
            py_rng,
        )
        for x in raw_sequences
    ]


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
        if count:
            ece += (
                count / max(n, 1)
            ) * abs(
                correctness[mask].mean()
                - confidences[mask].mean()
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
            value = roc_auc_score(
                y_bin[:, 0], probs[:, 1]
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


def classification_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    num_classes: int,
):
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    if probs.shape != (len(labels), num_classes):
        raise RuntimeError(
            f"Probability shape mismatch: {probs.shape}"
        )
    preds = probs.argmax(axis=1).astype(np.int64)
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
            labels, probs, ECE_BINS
        ),
        "macro_auroc": multiclass_macro_auroc(
            labels, probs, num_classes
        ),
        "labels": labels,
        "preds": preds,
        "probs": probs,
    }


def attack_metrics(clean_eval, attacked_eval):
    clean_labels = clean_eval["labels"]
    attacked_labels = attacked_eval["labels"]
    if not np.array_equal(
        clean_labels, attacked_labels
    ):
        raise RuntimeError(
            "Attack label order differs from clean order"
        )

    clean_correct = (
        clean_eval["preds"] == clean_labels
    )
    attacked_wrong = (
        attacked_eval["preds"] != attacked_labels
    )
    attack_success = int(
        np.sum(clean_correct & attacked_wrong)
    )
    clean_correct_n = int(np.sum(clean_correct))
    robust_acc = float(attacked_eval["acc"])
    rel_drop = (
        (float(clean_eval["acc"]) - robust_acc)
        / max(float(clean_eval["acc"]), 1e-12)
        * 100.0
    )
    return {
        "robust_acc": robust_acc,
        "robust_f1": float(
            attacked_eval["macro_f1"]
        ),
        "asr": (
            attack_success
            / max(clean_correct_n, 1)
            * 100.0
        ),
        "rel_drop": float(rel_drop),
        "abs_drop_pp": (
            float(clean_eval["acc"]) - robust_acc
        ),
        "ece": float(attacked_eval["ece"]),
        "macro_auroc":
            attacked_eval["macro_auroc"],
    }


def compact_eval(obj):
    return {
        k: v for k, v in obj.items()
        if k not in {"labels", "preds", "probs"}
    }


def attack_conditions(full_sweep: bool):
    if full_sweep:
        out = []
        for attack in MULTI_ATTACKS:
            for s in MULTI_STRENGTHS:
                out.append((attack, float(s)))
        for s in ADAPTIVE_STRENGTHS:
            out.append(("adaptive", float(s)))
        return out
    return [
        ("mixed", PRIMARY_MIXED_STRENGTH),
        ("adaptive", PRIMARY_ADAPTIVE_STRENGTH),
    ]
