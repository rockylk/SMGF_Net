# -*- coding: utf-8 -*-
"""
Figure 10 - CipherSpectrum channel-wise Gate audit for final SMGF-Net V17.

The script uses the formal frozen CipherSpectrum 3-seed archive and the exact
canonical deterministic attack seeds. It does not assume a global monotonic
Gate decay. The mechanism claim tested is heterogeneous, context-conditioned
channel-wise reweighting.

Figure design:
(a) Mixed attack: seed42 channel-wise mean Gate shift vs. clean.
(b) Adaptive attack: seed42 channel-wise mean Gate shift vs. clean.
(c) Cross-seed mean absolute paired Gate shift (seeds 42/43/44, mean +/- SD).

Outputs are written to:
<formal_run>/figures/figure10_gate_audit/
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SOURCE = SCRIPT_DIR / "V17-SMGF-Net_final_paper_experiments.py"
CORE_SOURCE = SCRIPT_DIR / "V16-SMGF-Net_final_paper_experiments.py"
EXPECTED_BASE_SHA256 = "17c88de543207e0241c666a4aa7aae4d36c023dda37ac131d7531957ede2cfa9"
EXPECTED_CORE_SHA256 = "3ecaf0054a1a190b444fe6f3c19cf6203273ad32707041ec53b09318ca20a4e2"
EXPECTED_DATASET_MANIFEST_SHA256 = "821bee04771f0c8dce3680a4301164d8ab48e03ec0d1ed57c3930015ad200a61"
EXPECTED_SPLIT_MANIFEST_SHA256 = "01e3b572e8514a238b3733e224161acd1112b79b56f038635be9339b23cfad6b"
EXPECTED_NUM_CLASSES = 41
EXPECTED_NUM_SAMPLES = 41000
EXPECTED_TRAIN_N = 29520
EXPECTED_VAL_N = 3280
EXPECTED_TEST_N = 8200
FORMAL_SEEDS = [42, 43, 44]
REPRESENTATIVE_SEED = 42
EXPECTED_CKPT_SHA256 = {
    42: "a0002cb6cb692b9965b29ec187b7aefd902a475361c1cdc0450077397c4ee7d9",
    43: "a8b1e4d140faaa7f9a3c2effbc23bf281387fb0780f384769d6e103b8d7a469c",
    44: "99a84aba032660fb2de178708b95b802d27ce690037dca1739e97e02ef6e7673",
}
EXPECTED_CKPT_FILENAMES = {
    42: "FINAL_seed42_REUSED.pt",
    43: "FINAL_seed43.pt",
    44: "FINAL_seed44.pt",
}
DEFAULT_FORMAL_RUN_NAME = "SMGF_V17_CIPHERSPECTRUM_FINAL_3SEED_20260923_135542"
PCAP_DIR = Path(os.environ.get(
    "SMGF_PCAP_DIR", "/home/admin1/dataset/aes-128-gcm/aes-128-gcm"
)).expanduser().resolve()
ATTACK_BASE_SEED = 20260919
BATCH_SIZE = 64
AUDIT_MAX_SAMPLES = int(os.environ.get("SMGF_GATE_AUDIT_MAX_SAMPLES", "1000"))
MIXED_STRENGTHS = [0.3, 0.6, 0.9, 1.0]
ADAPTIVE_STRENGTHS = [0.4, 0.7, 1.0]
EXPECTED_GATE_CHANNELS = 128


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_source_locks() -> None:
    if sha256_file(BASE_SOURCE) != EXPECTED_BASE_SHA256:
        raise RuntimeError("V17 source SHA mismatch; refusing Figure 10 protocol drift.")
    if sha256_file(CORE_SOURCE) != EXPECTED_CORE_SHA256:
        raise RuntimeError("V16 core SHA mismatch; refusing Figure 10 protocol drift.")


def discover_formal_run() -> Path:
    explicit = os.environ.get("SMGF_CIPHER_FORMAL_RUN")
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(p)
        return p
    candidates = [
        SCRIPT_DIR / DEFAULT_FORMAL_RUN_NAME,
        Path.cwd() / DEFAULT_FORMAL_RUN_NAME,
        Path.cwd() / "RESULT" / DEFAULT_FORMAL_RUN_NAME,
        Path("/home/admin1/lkfiles/RESULT") / DEFAULT_FORMAL_RUN_NAME,
    ]
    for p in candidates:
        p = p.expanduser().resolve()
        if p.is_dir():
            return p
    roots = [SCRIPT_DIR, Path.cwd(), Path.cwd() / "RESULT", Path("/home/admin1/lkfiles/RESULT")]
    found = []
    for root in roots:
        root = root.expanduser().resolve()
        if not root.exists():
            continue
        for p in root.glob("SMGF_V17_CIPHERSPECTRUM_FINAL_3SEED_*"):
            ckpt = p / "checkpoints" / EXPECTED_CKPT_FILENAMES[42]
            if ckpt.is_file() and sha256_file(ckpt) == EXPECTED_CKPT_SHA256[42]:
                found.append(p)
    if not found:
        raise FileNotFoundError(
            "Cannot locate the formal CipherSpectrum three-seed archive. "
            "Set SMGF_CIPHER_FORMAL_RUN explicitly."
        )
    return max(found, key=lambda p: p.stat().st_mtime)


FORMAL_RUN = discover_formal_run()
OUT_DIR = FORMAL_RUN / "figures" / "figure10_gate_audit"
PNG_PATH = OUT_DIR / "Figure10_GateAudit.png"
PDF_PATH = OUT_DIR / "Figure10_GateAudit.pdf"
SVG_PATH = OUT_DIR / "Figure10_GateAudit.svg"
CHANNEL_CSV = OUT_DIR / "Figure10_GateAudit_seed42_channel_matrix.csv"
STRENGTH_CSV = OUT_DIR / "Figure10_GateAudit_strength_summary.csv"
SEED_CSV = OUT_DIR / "Figure10_GateAudit_seed_summary.csv"
JSON_PATH = OUT_DIR / "Figure10_GateAudit.json"
CAPTION_PATH = OUT_DIR / "Figure10_GateAudit_caption.md"
LOG_PATH = OUT_DIR / "experiment.log"
REPRO_SPLIT_PATH = OUT_DIR / "reproduced_split_manifest.json"


class Logger:
    def __init__(self, path: Path):
        self.path = Path(path)
    def log(self, msg=""):
        msg = str(msg)
        print(msg, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")


LOGGER = Logger(LOG_PATH)


def import_final_v17():
    assert_source_locks()
    tmp_root = Path(tempfile.mkdtemp(prefix="smgf_fig10_v17_import_"))
    old_result = os.environ.get("SMGF_RESULT_DIR")
    old_pcap = os.environ.get("SMGF_PCAP_DIR")
    os.environ["SMGF_RESULT_DIR"] = str(tmp_root)
    os.environ["SMGF_PCAP_DIR"] = str(PCAP_DIR)
    try:
        spec = importlib.util.spec_from_file_location("smgf_v17_figure10_gate_audit_base", BASE_SOURCE)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import {BASE_SOURCE}")
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


BASE = import_final_v17()
CORE = BASE.CORE
torch = BASE.torch
DataLoader = CORE.DataLoader
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required for the final V17 Gate audit.")
DEVICE = torch.device("cuda")


def rewire_outputs() -> None:
    BASE.RUN_DIR = OUT_DIR
    BASE.LOG_FILE = LOG_PATH
    BASE.SPLIT_FILE = REPRO_SPLIT_PATH
    BASE.CHECKPOINT_DIR = OUT_DIR
    BASE.FIGURE_DIR = OUT_DIR
    BASE.LOGGER = LOGGER
    CORE.RUN_DIR = OUT_DIR
    CORE.LOG_FILE = LOG_PATH
    CORE.SPLIT_MANIFEST_FILE = REPRO_SPLIT_PATH
    CORE.CHECKPOINT_DIR = OUT_DIR
    CORE.FIGURE_DIR = OUT_DIR


def validate_formal_archive() -> dict[int, Path]:
    checkpoints = {}
    formal_split = FORMAL_RUN / "split_manifest.json"
    if not formal_split.is_file():
        raise FileNotFoundError(formal_split)
    if sha256_file(formal_split) != EXPECTED_SPLIT_MANIFEST_SHA256:
        raise RuntimeError("Formal split manifest SHA mismatch.")
    for seed in FORMAL_SEEDS:
        ckpt = FORMAL_RUN / "checkpoints" / EXPECTED_CKPT_FILENAMES[seed]
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        actual = sha256_file(ckpt)
        if actual != EXPECTED_CKPT_SHA256[seed]:
            raise RuntimeError(f"seed{seed} checkpoint SHA mismatch: {actual}")
        checkpoints[seed] = ckpt
        LOGGER.log(f"[checkpoint] seed={seed} PASS | {ckpt.name} | SHA={actual}")
    per_seed = FORMAL_RUN / "per_seed_results.json"
    if not per_seed.is_file():
        raise FileNotFoundError(per_seed)
    payload = json.loads(per_seed.read_text(encoding="utf-8"))
    seeds = [int(r["seed"]) for r in payload.get("test_results", [])]
    if seeds != FORMAL_SEEDS:
        raise RuntimeError(f"Formal per-seed result order mismatch: {seeds}")
    LOGGER.log("[formal archive] PASS")
    return checkpoints


def load_formal_dataset():
    if not PCAP_DIR.is_dir():
        raise FileNotFoundError(f"CipherSpectrum dataset path not found: {PCAP_DIR}")
    (
        train_set, val_set, test_set, node_scaler, stats_scaler,
        num_classes, idx_to_class, manifest_hash,
    ) = BASE.load_and_prepare_data()
    if manifest_hash != EXPECTED_DATASET_MANIFEST_SHA256:
        raise RuntimeError("CipherSpectrum dataset manifest mismatch.")
    observed = (
        num_classes, len(train_set) + len(val_set) + len(test_set),
        len(train_set), len(val_set), len(test_set),
    )
    expected = (
        EXPECTED_NUM_CLASSES, EXPECTED_NUM_SAMPLES,
        EXPECTED_TRAIN_N, EXPECTED_VAL_N, EXPECTED_TEST_N,
    )
    if observed != expected:
        raise RuntimeError(f"Dataset/split count drift: {observed} != {expected}")
    if not REPRO_SPLIT_PATH.is_file():
        raise RuntimeError("Reproduced split manifest was not written.")
    split_sha = sha256_file(REPRO_SPLIT_PATH)
    if split_sha != EXPECTED_SPLIT_MANIFEST_SHA256:
        raise RuntimeError(f"Reproduced split SHA mismatch: {split_sha}")
    BASE.validate_zero_strength_noop(test_set, node_scaler, stats_scaler, max_samples=16)
    LOGGER.log(
        f"[dataset] PASS | manifest={manifest_hash} | "
        f"Train/Val/Test={len(train_set)}/{len(val_set)}/{len(test_set)}"
    )
    return {
        "train_set": train_set, "val_set": val_set, "test_set": test_set,
        "node_scaler": node_scaler, "stats_scaler": stats_scaler,
        "num_classes": num_classes, "idx_to_class": idx_to_class,
        "manifest_hash": manifest_hash, "split_sha256": split_sha,
    }


def balanced_test_indices(dataset, max_n: int, seed: int) -> np.ndarray:
    by_class = defaultdict(list)
    for i, data in enumerate(dataset):
        by_class[int(data.y.view(-1)[0].item())].append(i)
    rng = np.random.default_rng(seed)
    classes = sorted(by_class)
    per_class = max(1, max_n // len(classes))
    selected, leftovers = [], []
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
    selected = np.asarray(sorted(selected[:max_n]), dtype=np.int64)
    observed_classes = {int(dataset[int(i)].y.view(-1)[0].item()) for i in selected}
    if len(observed_classes) != EXPECTED_NUM_CLASSES:
        raise RuntimeError(
            f"Balanced audit subset covers {len(observed_classes)} / {EXPECTED_NUM_CLASSES} classes"
        )
    return selected


def build_gate_audit_attack_cache(data_info: dict):
    test_set = data_info["test_set"]
    node_scaler = data_info["node_scaler"]
    stats_scaler = data_info["stats_scaler"]
    cache = {}
    specs = [
        *[("mixed", float(s)) for s in MIXED_STRENGTHS],
        *[("adaptive", float(s)) for s in ADAPTIVE_STRENGTHS],
    ]
    LOGGER.log("=" * 96)
    LOGGER.log("REBUILD CANONICAL FIGURE-10 ATTACK CACHE")
    LOGGER.log("=" * 96)
    for attack, strength in specs:
        attack_seed = CORE.stable_seed(ATTACK_BASE_SEED, attack, f"{strength:.3f}")
        attacked = CORE.build_attacked_dataset_deterministic(
            test_set, attack, node_scaler, stats_scaler, strength, attack_seed
        )
        CORE.validate_paired_dataset(test_set, attacked, f"Figure10:{attack}@{strength:.1f}")
        if attack == "adaptive":
            CORE.validate_adaptive_semantics(test_set, attacked)
        cache[(attack, strength)] = attacked
        LOGGER.log(
            f"[attack cache] {attack:8s}@{strength:.1f} | seed={attack_seed} | n={len(attacked)}"
        )
    LOGGER.log("[attack cache] PASS")
    return cache


def new_model(num_classes: int):
    return BASE.SMGFNetV17(
        node_in_channels=5, stats_dim=9, finger_seq_len=20,
        entropy_seq_len=5, hidden_channels=32, out_channels=num_classes,
        gate_mode="learned",
    ).to(DEVICE)


def load_model(checkpoint: Path, num_classes: int):
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    if "model_state_dict" not in payload:
        raise RuntimeError(f"Checkpoint lacks model_state_dict: {checkpoint}")
    model = new_model(num_classes)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if model.fusion_layer.gate_mode != "learned":
        raise RuntimeError("Figure 10 requires learned Gate mode.")
    if int(model.graph_dim) != EXPECTED_GATE_CHANNELS:
        raise RuntimeError(f"Unexpected Gate channel count: {model.graph_dim}")
    return model, payload


def collect_gate_matrix(model, dataset, indices: np.ndarray) -> np.ndarray:
    subset = [dataset[int(i)] for i in indices]
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            h_finger = model.finger_cnn(batch.finger_attr)
            h_entropy = model.entropy_cnn(batch.entropy_attr)
            h_ctx = torch.cat([batch.stats_attr, h_finger, h_entropy], dim=1)
            gate = model.fusion_layer.global_gate(h_ctx)
            rows.append(gate.detach().cpu().numpy())
    matrix = np.vstack(rows).astype(np.float64, copy=False)
    if matrix.shape != (len(indices), EXPECTED_GATE_CHANNELS):
        raise RuntimeError(f"Gate matrix shape mismatch: {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise RuntimeError("Non-finite Gate activation detected.")
    if matrix.min() < -1e-7 or matrix.max() > 1.0 + 1e-7:
        raise RuntimeError("Gate activation escaped sigmoid range [0,1].")
    return matrix


def rankdata_average(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def spearman_small(x, y) -> float:
    rx, ry = rankdata_average(x), rankdata_average(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize_delta(clean_gates: np.ndarray, attacked_gates: np.ndarray) -> dict:
    delta = attacked_gates - clean_gates
    channel_mean_delta = delta.mean(axis=0)
    return {
        "mean_signed_paired_delta": float(delta.mean()),
        "mean_abs_paired_delta": float(np.abs(delta).mean()),
        "median_abs_paired_delta": float(np.median(np.abs(delta))),
        "mean_abs_channel_shift": float(np.mean(np.abs(channel_mean_delta))),
        "channels_mean_negative": int(np.sum(channel_mean_delta < 0)),
        "channels_mean_positive": int(np.sum(channel_mean_delta > 0)),
        "channels_mean_near_zero": int(np.sum(np.isclose(channel_mean_delta, 0.0, atol=1e-10))),
        "element_fraction_negative": float(np.mean(delta < 0)),
        "element_fraction_positive": float(np.mean(delta > 0)),
        "channel_mean_delta": channel_mean_delta.tolist(),
    }


def summarize_attack_trend(clean_gates, attacked_by_strength, strengths) -> dict:
    channel_means = np.stack(
        [attacked_by_strength[float(s)].mean(axis=0) for s in strengths], axis=0
    )
    rho = np.asarray([
        spearman_small(strengths, channel_means[:, j])
        for j in range(channel_means.shape[1])
    ], dtype=np.float64)
    final_delta = attacked_by_strength[float(strengths[-1])] - clean_gates
    final_channel_shift = final_delta.mean(axis=0)
    return {
        "median_channel_rho": float(np.median(rho)),
        "mean_abs_channel_rho": float(np.mean(np.abs(rho))),
        "strong_suppressor_channels_rho_le_neg_0_8": int(np.sum(rho <= -0.8)),
        "strong_enhancer_channels_rho_ge_pos_0_8": int(np.sum(rho >= 0.8)),
        "final_mean_signed_gate_delta": float(final_delta.mean()),
        "final_mean_abs_paired_gate_delta": float(np.abs(final_delta).mean()),
        "final_channels_negative_shift": int(np.sum(final_channel_shift < 0)),
        "final_channels_positive_shift": int(np.sum(final_channel_shift > 0)),
        "channel_rho": rho.tolist(),
        "final_channel_mean_shift": final_channel_shift.tolist(),
    }


def run_seed_audit(seed, checkpoint, data_info, attack_cache, audit_indices):
    LOGGER.log("-" * 96)
    LOGGER.log(f"GATE AUDIT seed={seed}")
    LOGGER.log("-" * 96)
    model, payload = load_model(checkpoint, data_info["num_classes"])
    clean = collect_gate_matrix(model, data_info["test_set"], audit_indices)
    result = {
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "best_epoch": int(payload.get("best_epoch", payload.get("epoch", -1))),
        "sample_count": int(len(audit_indices)),
        "gate_channels": int(clean.shape[1]),
        "clean_gate_mean": float(clean.mean()),
        "clean_gate_sd": float(clean.std(ddof=1)),
        "clean_gate_median": float(np.median(clean)),
        "conditions": {},
        "attack_trends": {},
    }
    raw = {"clean": clean, "mixed": {}, "adaptive": {}}
    for attack, strengths in (("mixed", MIXED_STRENGTHS), ("adaptive", ADAPTIVE_STRENGTHS)):
        attacked_by_strength = {}
        for strength in strengths:
            attacked = collect_gate_matrix(
                model, attack_cache[(attack, float(strength))], audit_indices
            )
            attacked_by_strength[float(strength)] = attacked
            raw[attack][float(strength)] = attacked
            stats = summarize_delta(clean, attacked)
            result["conditions"][f"{attack}@{strength:.1f}"] = stats
            LOGGER.log(
                f"[seed={seed}] {attack}@{strength:.1f} | "
                f"mean|dg|={stats['mean_abs_paired_delta']:.6f} | "
                f"mean_dg={stats['mean_signed_paired_delta']:+.6f} | "
                f"channels(-/+)={stats['channels_mean_negative']}/{stats['channels_mean_positive']}"
            )
        result["attack_trends"][attack] = summarize_attack_trend(
            clean, attacked_by_strength, strengths
        )
    del model
    torch.cuda.empty_cache()
    return result, raw


def aggregate_strength_summary(per_seed: dict[int, dict]) -> list[dict]:
    rows = []
    for attack, strengths in (
        ("mixed", [0.0] + MIXED_STRENGTHS),
        ("adaptive", [0.0] + ADAPTIVE_STRENGTHS),
    ):
        for strength in strengths:
            if strength == 0.0:
                vals = np.zeros(len(FORMAL_SEEDS), dtype=np.float64)
                signed = np.zeros(len(FORMAL_SEEDS), dtype=np.float64)
            else:
                key = f"{attack}@{strength:.1f}"
                vals = np.asarray([
                    per_seed[s]["conditions"][key]["mean_abs_paired_delta"]
                    for s in FORMAL_SEEDS
                ])
                signed = np.asarray([
                    per_seed[s]["conditions"][key]["mean_signed_paired_delta"]
                    for s in FORMAL_SEEDS
                ])
            rows.append({
                "attack": attack,
                "strength": float(strength),
                "mean_abs_gate_shift_mean": float(vals.mean()),
                "mean_abs_gate_shift_sd": float(vals.std(ddof=1)),
                "mean_signed_gate_shift_mean": float(signed.mean()),
                "mean_signed_gate_shift_sd": float(signed.std(ddof=1)),
                **{f"seed{s}_mean_abs_gate_shift": float(v) for s, v in zip(FORMAL_SEEDS, vals)},
            })
    return rows


def aggregate_seed_summary(per_seed: dict[int, dict]) -> list[dict]:
    rows = []
    for seed in FORMAL_SEEDS:
        rec = per_seed[seed]
        for attack in ("mixed", "adaptive"):
            trend = rec["attack_trends"][attack]
            rows.append({
                "seed": seed,
                "attack": attack,
                "clean_gate_mean": rec["clean_gate_mean"],
                "clean_gate_sd": rec["clean_gate_sd"],
                "median_channel_rho": trend["median_channel_rho"],
                "mean_abs_channel_rho": trend["mean_abs_channel_rho"],
                "strong_suppressor_channels": trend["strong_suppressor_channels_rho_le_neg_0_8"],
                "strong_enhancer_channels": trend["strong_enhancer_channels_rho_ge_pos_0_8"],
                "final_mean_signed_gate_delta": trend["final_mean_signed_gate_delta"],
                "final_mean_abs_paired_gate_delta": trend["final_mean_abs_paired_gate_delta"],
                "final_channels_negative_shift": trend["final_channels_negative_shift"],
                "final_channels_positive_shift": trend["final_channels_positive_shift"],
            })
    return rows


def representative_channel_matrices(raw_seed42: dict):
    clean = raw_seed42["clean"]
    mixed = np.stack([
        (raw_seed42["mixed"][float(s)] - clean).mean(axis=0)
        for s in MIXED_STRENGTHS
    ], axis=1)
    adaptive = np.stack([
        (raw_seed42["adaptive"][float(s)] - clean).mean(axis=0)
        for s in ADAPTIVE_STRENGTHS
    ], axis=1)
    combined_final = 0.5 * (mixed[:, -1] + adaptive[:, -1])
    order = np.argsort(combined_final, kind="mergesort")
    return mixed[order, :], adaptive[order, :], order


def write_csvs(strength_rows, seed_rows, mixed_matrix, adaptive_matrix, channel_order):
    with STRENGTH_CSV.open("w", encoding="utf-8", newline="") as f:
        fields = list(strength_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(strength_rows)
    with SEED_CSV.open("w", encoding="utf-8", newline="") as f:
        fields = list(seed_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(seed_rows)
    rows = []
    for rank_idx, original_channel in enumerate(channel_order, start=1):
        row = {"rank": rank_idx, "original_channel": int(original_channel)}
        for col, s in enumerate(MIXED_STRENGTHS):
            row[f"mixed_{s:.1f}_mean_delta"] = float(mixed_matrix[rank_idx - 1, col])
        for col, s in enumerate(ADAPTIVE_STRENGTHS):
            row[f"adaptive_{s:.1f}_mean_delta"] = float(adaptive_matrix[rank_idx - 1, col])
        rows.append(row)
    with CHANNEL_CSV.open("w", encoding="utf-8", newline="") as f:
        fields = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def choose_serif_font(font_manager) -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"):
        if name in available:
            return name
    return "DejaVu Serif"


def style_line_axis(ax) -> None:
    ax.set_facecolor("#ffffff")
    ax.minorticks_off()
    ax.grid(True, which="major", linestyle="--", linewidth=0.40, color="#c9c9c9", alpha=0.65)
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.60)
        ax.spines[spine].set_alpha(0.55)
    ax.tick_params(axis="both", which="major", direction="in", top=True, right=True,
                   length=3.5, width=0.55, pad=2.5)
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_alpha(0.50); tick.tick2line.set_alpha(0.50)
    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_alpha(0.50); tick.tick2line.set_alpha(0.50)


def style_heatmap_axis(ax) -> None:
    ax.set_facecolor("#ffffff")
    ax.minorticks_off()
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.60)
        ax.spines[spine].set_alpha(0.55)
    ax.tick_params(axis="both", which="major", direction="out", length=3.0, width=0.55, pad=2.0)
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_alpha(0.50); tick.tick2line.set_alpha(0.50)
    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_alpha(0.50); tick.tick2line.set_alpha(0.50)


def create_figure(mixed_matrix, adaptive_matrix, strength_rows) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors, font_manager
    font_name = choose_serif_font(font_manager)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [font_name], "font.size": 8.4,
        "axes.labelsize": 8.8, "axes.titlesize": 9.2,
        "xtick.labelsize": 7.7, "ytick.labelsize": 7.7, "legend.fontsize": 7.8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
    })
    fig = plt.figure(figsize=(7.2, 2.85), facecolor="#ffffff")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 0.86, 1.20],
                          left=0.065, right=0.985, top=0.90, bottom=0.24, wspace=0.34)
    ax_m, ax_a, ax_s = (fig.add_subplot(gs[0, i]) for i in range(3))
    mixed_plot = np.column_stack([np.zeros(mixed_matrix.shape[0]), mixed_matrix])
    adaptive_plot = np.column_stack([np.zeros(adaptive_matrix.shape[0]), adaptive_matrix])
    all_abs = np.concatenate([np.abs(mixed_plot).ravel(), np.abs(adaptive_plot).ravel()])
    vlim = max(float(np.quantile(all_abs, 0.995)), 1e-4)
    norm = colors.TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    im = ax_m.imshow(mixed_plot, aspect="auto", origin="lower", interpolation="nearest",
                     cmap="RdBu_r", norm=norm)
    ax_a.imshow(adaptive_plot, aspect="auto", origin="lower", interpolation="nearest",
                cmap="RdBu_r", norm=norm)
    mx, axx = [0.0] + MIXED_STRENGTHS, [0.0] + ADAPTIVE_STRENGTHS
    ax_m.set_xticks(np.arange(len(mx))); ax_m.set_xticklabels([f"{x:.1f}" for x in mx])
    ax_a.set_xticks(np.arange(len(axx))); ax_a.set_xticklabels([f"{x:.1f}" for x in axx])
    yt = np.asarray([0, 31, 63, 95, 127], dtype=int)
    yl = [str(int(y + 1)) for y in yt]
    ax_m.set_yticks(yt); ax_m.set_yticklabels(yl)
    ax_a.set_yticks(yt); ax_a.set_yticklabels(yl)
    for ax in (ax_m, ax_a):
        ax.set_xlabel("Perturbation Strength")
        ax.set_ylabel("Channel Rank")
        style_heatmap_axis(ax)
    ax_m.set_title("(a) Mixed Attack", pad=4.0)
    ax_a.set_title("(b) Adaptive Attack", pad=4.0)
    cbar = fig.colorbar(im, ax=[ax_m, ax_a], orientation="horizontal",
                        fraction=0.065, pad=0.18, aspect=34)
    cbar.set_label("Mean Gate Shift vs. Clean  ($\\Delta g$)", labelpad=2.5)
    cbar.outline.set_linewidth(0.55); cbar.outline.set_alpha(0.55)
    cbar.ax.tick_params(which="major", length=2.8, width=0.5, labelsize=7.2)
    mrows = [r for r in strength_rows if r["attack"] == "mixed"]
    arows = [r for r in strength_rows if r["attack"] == "adaptive"]
    def arrays(rows):
        return (
            np.asarray([r["strength"] for r in rows]),
            np.asarray([r["mean_abs_gate_shift_mean"] for r in rows]),
            np.asarray([r["mean_abs_gate_shift_sd"] for r in rows]),
        )
    xm, ym, sm = arrays(mrows); xa, ya, sa = arrays(arows)
    ax_s.fill_between(xm, np.maximum(0, ym-sm), ym+sm, alpha=0.13, linewidth=0, color="#3b6ea8")
    ax_s.plot(xm, ym, linewidth=1.15, marker="o", markersize=3.8,
              markerfacecolor="white", markeredgewidth=0.75, color="#3b6ea8", label="Mixed")
    ax_s.fill_between(xa, np.maximum(0, ya-sa), ya+sa, alpha=0.13, linewidth=0, color="#b25745")
    ax_s.plot(xa, ya, linewidth=1.15, marker="s", markersize=3.6,
              markerfacecolor="white", markeredgewidth=0.75, color="#b25745", label="Adaptive")
    ax_s.set_xlim(-0.02, 1.02); ax_s.set_xticks([0.0, 0.3, 0.6, 0.9, 1.0])
    ax_s.set_xlabel("Perturbation Strength"); ax_s.set_ylabel(r"Mean $|\Delta g|$")
    ax_s.set_title("(c) Cross-seed Modulation Magnitude", pad=4.0)
    style_line_axis(ax_s)
    leg = ax_s.legend(loc="upper left", frameon=True, fancybox=False,
                      borderpad=0.35, handlelength=1.8)
    leg.get_frame().set_facecolor("#ffffff"); leg.get_frame().set_alpha(0.85); leg.get_frame().set_linewidth(0.50)
    for path, kwargs in (
        (PNG_PATH, {"dpi": 600}), (PDF_PATH, {}), (SVG_PATH, {})
    ):
        fig.savefig(path, bbox_inches="tight", pad_inches=0.035, facecolor="#ffffff", **kwargs)
    plt.close(fig)


def write_caption(representative, strength_rows, audit_indices) -> None:
    def lookup(attack, strength):
        return next(r for r in strength_rows if r["attack"] == attack and abs(r["strength"]-strength) < 1e-12)
    mf, af = lookup("mixed", 1.0), lookup("adaptive", 1.0)
    mt, at = representative["attack_trends"]["mixed"], representative["attack_trends"]["adaptive"]
    text = f"""# Figure 10 caption and reporting notes

**Figure 10. Channel-wise Gate audit of the final SMGF-Net on CipherSpectrum.**
Panels (a) and (b) show, for the prespecified representative seed-42 model, the mean change in each graph-channel gate relative to clean input under increasing Mixed and Adaptive perturbation strengths. Channels use one common ordering determined by their combined signed response at maximum attack strength. Blue and red therefore denote channel suppression and enhancement, not different feature branches. Panel (c) shows the mean absolute paired Gate shift across the three formal models (seeds 42, 43, and 44); shaded regions denote +/-1 sample SD. All conditions use the same deterministic class-balanced subset of {len(audit_indices)} test flows and the exact canonical attack seeds used by the final robustness evaluation.

## Interpretation boundary
The final Gate is a 128-dimensional context-conditioned multiplier on the graph representation. The supported mechanism claim is heterogeneous channel-wise reweighting under perturbation. A global monotonic decrease of the average Gate is neither assumed nor required.

## Representative seed42 diagnostics
Mixed: median channel rho={mt['median_channel_rho']:.4f}; mean |rho|={mt['mean_abs_channel_rho']:.4f}; strong suppressor/enhancer channels={mt['strong_suppressor_channels_rho_le_neg_0_8']}/{mt['strong_enhancer_channels_rho_ge_pos_0_8']}; final negative/positive channel shifts={mt['final_channels_negative_shift']}/{mt['final_channels_positive_shift']}.

Adaptive: median channel rho={at['median_channel_rho']:.4f}; mean |rho|={at['mean_abs_channel_rho']:.4f}; strong suppressor/enhancer channels={at['strong_suppressor_channels_rho_le_neg_0_8']}/{at['strong_enhancer_channels_rho_ge_pos_0_8']}; final negative/positive channel shifts={at['final_channels_negative_shift']}/{at['final_channels_positive_shift']}.

Cross-seed final-strength mean |Delta Gate|: Mixed@1.0={mf['mean_abs_gate_shift_mean']:.6f} +/- {mf['mean_abs_gate_shift_sd']:.6f}; Adaptive@1.0={af['mean_abs_gate_shift_mean']:.6f} +/- {af['mean_abs_gate_shift_sd']:.6f}.
"""
    CAPTION_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rewire_outputs()
    LOGGER.log("=" * 96)
    LOGGER.log("FIGURE 10 - V17 CIPHERSPECTRUM CHANNEL-WISE GATE AUDIT")
    LOGGER.log("=" * 96)
    LOGGER.log(f"formal_run={FORMAL_RUN}")
    LOGGER.log(f"pcap_dir={PCAP_DIR}")
    LOGGER.log(f"audit_max_samples={AUDIT_MAX_SAMPLES}")
    LOGGER.log(f"V17_SHA={sha256_file(BASE_SOURCE)}")
    LOGGER.log(f"V16_SHA={sha256_file(CORE_SOURCE)}")
    checkpoints = validate_formal_archive()
    data_info = load_formal_dataset()
    audit_subset_seed = CORE.stable_seed("figure10_gate_audit_subset", 42)
    audit_indices = balanced_test_indices(
        data_info["test_set"], min(AUDIT_MAX_SAMPLES, len(data_info["test_set"])), audit_subset_seed
    )
    LOGGER.log(
        f"[audit subset] n={len(audit_indices)} | seed={audit_subset_seed} | classes={EXPECTED_NUM_CLASSES}"
    )
    attack_cache = build_gate_audit_attack_cache(data_info)
    per_seed, raw_seed42 = {}, None
    for seed in FORMAL_SEEDS:
        result, raw = run_seed_audit(
            seed, checkpoints[seed], data_info, attack_cache, audit_indices
        )
        per_seed[seed] = result
        if seed == REPRESENTATIVE_SEED:
            raw_seed42 = raw
    if raw_seed42 is None:
        raise RuntimeError("Representative seed42 raw Gate matrices were not captured.")
    strength_rows = aggregate_strength_summary(per_seed)
    seed_rows = aggregate_seed_summary(per_seed)
    mixed_matrix, adaptive_matrix, channel_order = representative_channel_matrices(raw_seed42)
    write_csvs(strength_rows, seed_rows, mixed_matrix, adaptive_matrix, channel_order)
    create_figure(mixed_matrix, adaptive_matrix, strength_rows)
    write_caption(per_seed[REPRESENTATIVE_SEED], strength_rows, audit_indices)
    payload = {
        "experiment": "Figure 10 channel-wise Gate audit",
        "interpretation": "heterogeneous channel-wise context-conditioned reweighting; no global monotonic-decay claim",
        "formal_run": str(FORMAL_RUN),
        "source_sha256": {
            "this_script": sha256_file(Path(__file__).resolve()),
            "V17_base": EXPECTED_BASE_SHA256,
            "V16_core": EXPECTED_CORE_SHA256,
        },
        "dataset": {
            "name": "CipherSpectrum", "manifest_sha256": data_info["manifest_hash"],
            "split_manifest_sha256": data_info["split_sha256"], "num_classes": data_info["num_classes"],
            "train_n": len(data_info["train_set"]), "val_n": len(data_info["val_set"]),
            "test_n": len(data_info["test_set"]),
        },
        "audit_subset": {
            "sample_count": int(len(audit_indices)),
            "selection": "deterministic class-balanced test subset",
            "selection_seed": int(audit_subset_seed),
            "indices": audit_indices.tolist(),
        },
        "attack_protocol": {
            "attack_base_seed": ATTACK_BASE_SEED,
            "mixed_strengths": MIXED_STRENGTHS,
            "adaptive_strengths": ADAPTIVE_STRENGTHS,
        },
        "representative_heatmap_seed": REPRESENTATIVE_SEED,
        "channel_order_original_indices": channel_order.tolist(),
        "per_seed": per_seed,
        "cross_seed_strength_summary": strength_rows,
        "artifacts": {
            "figure_png": str(PNG_PATH), "figure_pdf": str(PDF_PATH), "figure_svg": str(SVG_PATH),
            "channel_csv": str(CHANNEL_CSV), "strength_csv": str(STRENGTH_CSV),
            "seed_csv": str(SEED_CSV), "caption": str(CAPTION_PATH), "log": str(LOG_PATH),
        },
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.log("=" * 96)
    LOGGER.log("FIGURE 10 GATE AUDIT COMPLETE")
    LOGGER.log(f"PNG={PNG_PATH}")
    LOGGER.log(f"PDF={PDF_PATH}")
    LOGGER.log(f"SVG={SVG_PATH}")
    LOGGER.log(f"JSON={JSON_PATH}")
    LOGGER.log(f"CAPTION={CAPTION_PATH}")
    LOGGER.log("=" * 96)


if __name__ == "__main__":
    main()
