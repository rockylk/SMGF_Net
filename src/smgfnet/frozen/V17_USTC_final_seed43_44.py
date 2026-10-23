# -*- coding: utf-8 -*-
"""
USTC-TFC2016 V17 FINAL: train full seed43/44 and aggregate seed42/43/44.

This file does not duplicate the training implementation. It invokes the
validated V17_USTC_frozen_runner.py with FULL data for seeds 43 and 44.
That guarantees the model, LR schedule, weight decay, robust ratio, attack
training, checkpoint rule, data split and cache logic stay identical.

Formal frozen protocol:
- 12-class / 21,256-flow archived USTC experimental subset
- split seed 42, 72/8/20
- V17 + learned Gate + attacked-label CE + NO_KL
- 100 epochs, no early stopping
- LR 0.002 for epochs 1-70, 0.0005 for epochs 71-100
- AdamW weight_decay=1e-4
- batch_size=64
- robust batch ratio=0.20
- Mixed train attack, strength Uniform[0.3,0.9]
- attacked-label CE weight=1.0
- checkpoint=min clean-validation CE

Seed42 is reused from the completed formal run.
Seeds43/44 must use the same full cache and identical split_manifest bytes.
The script fails closed on any source, cache, protocol or split drift.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = SCRIPT_DIR / "RESULT"

RUNNER = SCRIPT_DIR / "V17_USTC_frozen_runner.py"
V17_SOURCE = SCRIPT_DIR / "V17-SMGF-Net_final_paper_experiments.py"
V16_SOURCE = SCRIPT_DIR / "V16-SMGF-Net_final_paper_experiments.py"

EXPECTED_RUNNER_SHA256 = "5f048e69f18271b7be5634f136d73d6776bce579a2741fb681b529ae3bf3e22b"
EXPECTED_V17_SHA256 = "17c88de543207e0241c666a4aa7aae4d36c023dda37ac131d7531957ede2cfa9"
EXPECTED_V16_SHA256 = "3ecaf0054a1a190b444fe6f3c19cf6203273ad32707041ec53b09318ca20a4e2"

DATASET_NAME = "USTC-TFC2016"
EXPECTED_DATASET_MANIFEST_SHA256 = "612c69a851dad04658619fd6c79ef60f34d08dd7dd1a0e14262fa01428cb2190"
EXPECTED_SPLIT_MANIFEST_SHA256 = "a34754daa9e202f154484a5b63a8840eb829e56aeaaff4549bac6cb711127f3b"

EXPECTED_SPLIT_COUNTS = {
    "train": 15304,
    "validation": 1700,
    "test": 4252,
}
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

EXPECTED_PROTOCOL = {
    "epochs": 100,
    "early_stopping": False,
    "lr_schedule": [[1, 70, 0.002], [71, 100, 0.0005]],
    "weight_decay": 0.0001,
    "batch_size": 64,
    "robust_batch_ratio": 0.2,
    "train_attack": "mixed",
    "robust_strength_distribution": "Uniform[0.3,0.9]",
    "attack_ce_weight": 1.0,
    "kl_weight": 0.0,
    "gate_mode": "learned",
    "checkpoint_rule": "minimum clean-validation CE",
}

EXPECTED_CACHE = SCRIPT_DIR / "USTC_CACHE" / "ustc_raw_graphs_f10000.pt"
EXPECTED_CACHE_META = SCRIPT_DIR / "USTC_CACHE" / "ustc_raw_graphs_f10000.json"
EXPECTED_SERVER_CACHE_STR = "/home/admin1/lkfiles/USTC_CACHE/ustc_raw_graphs_f10000.pt"

SEED42_RESULT = RESULT_ROOT / "SMGF_V17_USTC_SEED42_FULL_F10000_20260924_082814"
TARGET_SEEDS = [43, 44]
ALL_SEEDS = [42, 43, 44]

STAMP = time.strftime("%Y%m%d_%H%M%S", time.localtime())
FINAL_DIR = RESULT_ROOT / ("SMGF_V17_USTC_FINAL_3SEED_" + STAMP)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def same_float(a, b, atol=1e-12) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=atol)


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(label + " not found: " + str(path))


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(label + " not found: " + str(path))


def tail_text(path: Path, n=80) -> str:
    if not path.is_file():
        return "<missing log>"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def validate_source_locks() -> None:
    expected = {
        RUNNER: EXPECTED_RUNNER_SHA256,
        V17_SOURCE: EXPECTED_V17_SHA256,
        V16_SOURCE: EXPECTED_V16_SHA256,
    }
    for path, expected_sha in expected.items():
        require_file(path, "source")
        actual = sha256_file(path)
        if actual != expected_sha:
            raise RuntimeError(
                "Source SHA mismatch for "
                + str(path)
                + "\nexpected="
                + expected_sha
                + "\nactual="
                + actual
            )


def validate_cache() -> dict:
    require_file(EXPECTED_CACHE, "full raw graph cache")
    require_file(EXPECTED_CACHE_META, "full cache metadata")
    meta = load_json(EXPECTED_CACHE_META)

    if meta.get("dataset") != DATASET_NAME:
        raise RuntimeError("Cache dataset mismatch.")
    if not same_float(meta.get("pilot_fraction"), 1.0):
        raise RuntimeError("Cache is not the full dataset.")
    if int(meta.get("selected_file_count", -1)) != EXPECTED_TOTAL_FLOWS:
        raise RuntimeError("Cache selected count mismatch.")
    if int(meta.get("valid_sample_count", -1)) != EXPECTED_TOTAL_FLOWS:
        raise RuntimeError("Cache valid count mismatch.")
    if int(meta.get("skipped_sample_count", -1)) != 0:
        raise RuntimeError("Cache has skipped samples.")
    if meta.get("parsed_dataset_manifest_sha256") != EXPECTED_DATASET_MANIFEST_SHA256:
        raise RuntimeError("Cache dataset manifest mismatch.")
    if meta.get("expected_v17_base_sha256") != EXPECTED_V17_SHA256:
        raise RuntimeError("Cache V17 SHA mismatch.")
    if meta.get("expected_v16_core_sha256") != EXPECTED_V16_SHA256:
        raise RuntimeError("Cache V16 SHA mismatch.")

    actual_counts = {
        row["class_name"]: int(row["selected"])
        for row in meta.get("class_plan", [])
    }
    if actual_counts != EXPECTED_CLASS_COUNTS:
        raise RuntimeError(
            "Cache class inventory mismatch.\nexpected="
            + repr(EXPECTED_CLASS_COUNTS)
            + "\nactual="
            + repr(actual_counts)
        )
    return meta


def validate_protocol(config: dict, seed: int) -> None:
    if config.get("dataset") != DATASET_NAME:
        raise RuntimeError("seed%d dataset mismatch" % seed)
    if config.get("run_kind") != "FULL":
        raise RuntimeError("seed%d is not FULL" % seed)
    if not same_float(config.get("pilot_fraction"), 1.0):
        raise RuntimeError("seed%d pilot_fraction drift" % seed)
    if config.get("paper_valid") is not True:
        raise RuntimeError("seed%d not paper-valid" % seed)
    if int(config.get("train_seed", -1)) != seed:
        raise RuntimeError("seed%d train_seed mismatch" % seed)
    if int(config.get("split_seed", -1)) != 42:
        raise RuntimeError("seed%d split_seed drift" % seed)
    if int(config.get("num_classes", -1)) != 12:
        raise RuntimeError("seed%d class count drift" % seed)
    if config.get("dataset_manifest_sha256") != EXPECTED_DATASET_MANIFEST_SHA256:
        raise RuntimeError("seed%d dataset manifest drift" % seed)
    if config.get("split_counts") != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError("seed%d split counts drift" % seed)
    if config.get("frozen_protocol") != EXPECTED_PROTOCOL:
        raise RuntimeError(
            "seed%d protocol drift\nexpected=%r\nactual=%r"
            % (seed, EXPECTED_PROTOCOL, config.get("frozen_protocol"))
        )

    source_sha = config.get("source_sha256", {})
    if source_sha.get("V17_base") != EXPECTED_V17_SHA256:
        raise RuntimeError("seed%d V17 SHA drift" % seed)
    if source_sha.get("V16_core") != EXPECTED_V16_SHA256:
        raise RuntimeError("seed%d V16 SHA drift" % seed)

    actual_cache = str(config.get("cache_file", "")).replace("\\", "/")
    if actual_cache != EXPECTED_SERVER_CACHE_STR:
        raise RuntimeError(
            "seed%d cache path drift: %s" % (seed, actual_cache)
        )


def validate_inventory(inventory: dict, seed: int) -> None:
    if int(inventory.get("num_classes", -1)) != 12:
        raise RuntimeError("seed%d inventory class count drift" % seed)
    if int(inventory.get("valid_samples", -1)) != EXPECTED_TOTAL_FLOWS:
        raise RuntimeError("seed%d valid sample count drift" % seed)
    if int(inventory.get("skipped_samples", -1)) != 0:
        raise RuntimeError("seed%d has skipped samples" % seed)
    if int(inventory.get("failed_pcap_records", -1)) != 0:
        raise RuntimeError("seed%d has failed PCAPs" % seed)
    if inventory.get("manifest_sha256") != EXPECTED_DATASET_MANIFEST_SHA256:
        raise RuntimeError("seed%d inventory manifest drift" % seed)

    actual_counts = {
        row["class_name"]: int(row["selected"])
        for row in inventory.get("selection", [])
    }
    if actual_counts != EXPECTED_CLASS_COUNTS:
        raise RuntimeError("seed%d class inventory drift" % seed)


def validate_result_dir(path: Path, seed: int) -> dict:
    require_dir(path, "seed%d result directory" % seed)

    summary_path = path / "summary.json"
    config_path = path / "run_config.json"
    inventory_path = path / "dataset_inventory.json"
    split_path = path / "split_manifest.json"
    history_path = path / "history" / ("seed%d.csv" % seed)
    checkpoint_path = path / "checkpoints" / ("USTC_FINAL_seed%d.pt" % seed)
    log_path = path / "experiment.log"

    for p in [
        summary_path,
        config_path,
        inventory_path,
        split_path,
        history_path,
        checkpoint_path,
        log_path,
    ]:
        require_file(p, "seed%d artifact" % seed)

    summary = load_json(summary_path)
    config = load_json(config_path)
    inventory = load_json(inventory_path)

    validate_protocol(config, seed)
    validate_inventory(inventory, seed)

    split_sha = sha256_file(split_path)
    if split_sha != EXPECTED_SPLIT_MANIFEST_SHA256:
        raise RuntimeError(
            "seed%d split manifest differs from formal seed42.\nexpected=%s\nactual=%s"
            % (seed, EXPECTED_SPLIT_MANIFEST_SHA256, split_sha)
        )

    if summary.get("dataset") != DATASET_NAME:
        raise RuntimeError("seed%d summary dataset mismatch" % seed)
    if summary.get("run_kind") != "FULL":
        raise RuntimeError("seed%d summary is not FULL" % seed)
    if summary.get("paper_valid") is not True:
        raise RuntimeError("seed%d summary not paper-valid" % seed)
    if int(summary.get("train_seed", -1)) != seed:
        raise RuntimeError("seed%d summary train_seed mismatch" % seed)

    clean = summary.get("clean", {})
    attacks = summary.get("attacks", {})
    for key in ["acc", "macro_f1", "ece"]:
        if key not in clean:
            raise RuntimeError("seed%d missing clean metric %s" % (seed, key))
    for condition in ["mixed@1.0", "adaptive@1.0"]:
        if condition not in attacks:
            raise RuntimeError("seed%d missing %s" % (seed, condition))
        for key in ["robust_acc", "robust_f1", "asr", "rel_drop", "ece"]:
            if key not in attacks[condition]:
                raise RuntimeError(
                    "seed%d missing %s metric %s" % (seed, condition, key)
                )

    return {
        "seed": seed,
        "path": str(path.resolve()),
        "summary": summary,
        "config": config,
        "split_sha256": split_sha,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "history_sha256": sha256_file(history_path),
    }


def validate_seed42() -> dict:
    result = validate_result_dir(SEED42_RESULT, 42)
    if int(result["summary"].get("best_epoch", -1)) != 62:
        raise RuntimeError("Formal seed42 best_epoch drift.")
    if not same_float(
        result["summary"].get("best_val_loss"),
        0.15106678774687354,
        atol=1e-14,
    ):
        raise RuntimeError("Formal seed42 best_val_loss drift.")
    return result


def candidate_dirs(seed: int) -> list[Path]:
    pattern = "SMGF_V17_USTC_SEED%d_FULL_F10000_*" % seed
    return sorted(
        [p for p in RESULT_ROOT.glob(pattern) if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def find_existing_valid_result(seed: int):
    for path in candidate_dirs(seed):
        try:
            return validate_result_dir(path, seed)
        except Exception:
            continue
    return None


def launch_seed(seed: int) -> dict:
    existing = find_existing_valid_result(seed)
    if existing is not None:
        print(
            "[seed=%d] valid FULL result already exists; reuse %s"
            % (seed, existing["path"]),
            flush=True,
        )
        return existing

    before = {p.resolve() for p in candidate_dirs(seed)}
    env = os.environ.copy()
    env["SMGF_USTC_PILOT_FRACTION"] = "1.0"
    env["SMGF_USTC_TRAIN_SEED"] = str(seed)

    log_path = SCRIPT_DIR / ("ustc_seed%d_full.log" % seed)
    print("=" * 100, flush=True)
    print("Launching USTC FULL seed=%d" % seed, flush=True)
    print("Shared cache: %s" % EXPECTED_CACHE, flush=True)
    print("Log: %s" % log_path, flush=True)
    print("=" * 100, flush=True)

    with log_path.open("w", encoding="utf-8", buffering=1) as logf:
        proc = subprocess.run(
            [sys.executable, "-u", str(RUNNER)],
            cwd=str(SCRIPT_DIR),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if proc.returncode != 0:
        raise RuntimeError(
            "seed%d failed, returncode=%d\n%s"
            % (seed, proc.returncode, tail_text(log_path, 100))
        )

    new_dirs = [
        p
        for p in candidate_dirs(seed)
        if p.resolve() not in before
    ]
    if not new_dirs:
        raise RuntimeError(
            "seed%d exited successfully but no new result directory was found"
            % seed
        )

    result = validate_result_dir(new_dirs[0], seed)
    print(
        "[seed=%d] complete: %s" % (seed, result["path"]),
        flush=True,
    )
    return result


def flatten_primary(result: dict) -> dict:
    s = result["summary"]
    return {
        "seed": int(s["train_seed"]),
        "best_epoch": int(s["best_epoch"]),
        "best_val_loss": float(s["best_val_loss"]),
        "clean_acc": float(s["clean"]["acc"]),
        "clean_macro_f1": float(s["clean"]["macro_f1"]),
        "clean_ece": float(s["clean"]["ece"]),
        "mixed1_robust_acc": float(s["attacks"]["mixed@1.0"]["robust_acc"]),
        "mixed1_robust_f1": float(s["attacks"]["mixed@1.0"]["robust_f1"]),
        "mixed1_asr": float(s["attacks"]["mixed@1.0"]["asr"]),
        "mixed1_rel_drop": float(s["attacks"]["mixed@1.0"]["rel_drop"]),
        "mixed1_ece": float(s["attacks"]["mixed@1.0"]["ece"]),
        "adaptive1_robust_acc": float(s["attacks"]["adaptive@1.0"]["robust_acc"]),
        "adaptive1_robust_f1": float(s["attacks"]["adaptive@1.0"]["robust_f1"]),
        "adaptive1_asr": float(s["attacks"]["adaptive@1.0"]["asr"]),
        "adaptive1_rel_drop": float(s["attacks"]["adaptive@1.0"]["rel_drop"]),
        "adaptive1_ece": float(s["attacks"]["adaptive@1.0"]["ece"]),
    }


def aggregate(rows: list[dict]) -> dict:
    out = {}
    metric_names = [
        k
        for k in rows[0]
        if k not in {"seed", "best_epoch"}
    ]
    for metric in metric_names:
        values = [float(row[metric]) for row in rows]
        out[metric] = {
            "mean": float(statistics.mean(values)),
            "sample_sd": float(statistics.stdev(values)),
            "per_seed": {
                str(int(row["seed"])): float(row[metric])
                for row in rows
            },
        }
    out["best_epoch"] = {
        "per_seed": {
            str(int(row["seed"])): int(row["best_epoch"])
            for row in rows
        }
    }
    return out


def fmt(agg: dict, key: str) -> str:
    return "%.2f ± %.2f" % (
        agg[key]["mean"],
        agg[key]["sample_sd"],
    )


def write_archive(results: dict[int, dict]) -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=False)

    rows = [flatten_primary(results[s]) for s in ALL_SEEDS]
    rows.sort(key=lambda row: row["seed"])
    agg = aggregate(rows)

    per_seed_csv = FINAL_DIR / "paper_ready_primary_per_seed.csv"
    with per_seed_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    aggregate_csv = FINAL_DIR / "paper_ready_aggregate_metrics.csv"
    aggregate_rows = []
    for metric, item in agg.items():
        if metric == "best_epoch":
            continue
        aggregate_rows.append(
            {
                "metric": metric,
                "mean": item["mean"],
                "sample_sd": item["sample_sd"],
                "seed42": item["per_seed"]["42"],
                "seed43": item["per_seed"]["43"],
                "seed44": item["per_seed"]["44"],
            }
        )
    with aggregate_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "mean",
                "sample_sd",
                "seed42",
                "seed43",
                "seed44",
            ],
        )
        writer.writeheader()
        writer.writerows(aggregate_rows)

    aggregate_json = {
        "dataset": DATASET_NAME,
        "paper_valid": True,
        "train_seeds": ALL_SEEDS,
        "dataset_manifest_sha256": EXPECTED_DATASET_MANIFEST_SHA256,
        "split_manifest_sha256": EXPECTED_SPLIT_MANIFEST_SHA256,
        "per_seed": rows,
        "aggregate_mean_sample_sd": agg,
        "result_directories": {
            str(seed): results[seed]["path"]
            for seed in ALL_SEEDS
        },
    }
    (FINAL_DIR / "aggregate_summary.json").write_text(
        json.dumps(aggregate_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    run_manifest = {
        "dataset": DATASET_NAME,
        "experimental_subset": {
            "classes": 12,
            "flows": EXPECTED_TOTAL_FLOWS,
            "class_counts": EXPECTED_CLASS_COUNTS,
        },
        "dataset_manifest_sha256": EXPECTED_DATASET_MANIFEST_SHA256,
        "split_manifest_sha256": EXPECTED_SPLIT_MANIFEST_SHA256,
        "split_seed": 42,
        "split_counts": EXPECTED_SPLIT_COUNTS,
        "shared_cache": str(EXPECTED_CACHE.resolve()),
        "source_sha256": {
            "orchestrator": sha256_file(Path(__file__).resolve()),
            "runner": EXPECTED_RUNNER_SHA256,
            "v17": EXPECTED_V17_SHA256,
            "v16": EXPECTED_V16_SHA256,
        },
        "protocol": EXPECTED_PROTOCOL,
        "seeds": {
            str(seed): {
                "result_dir": results[seed]["path"],
                "split_sha256": results[seed]["split_sha256"],
                "checkpoint_sha256": results[seed]["checkpoint_sha256"],
                "history_sha256": results[seed]["history_sha256"],
            }
            for seed in ALL_SEEDS
        },
    }
    (FINAL_DIR / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# USTC-TFC2016 V17 Final 3-Seed Result",
        "",
        "Formal full-data result under the frozen protocol.",
        "",
        "Dataset subset: 12 classes / 21,256 flows",
        "Split seed: 42",
        "Train/Validation/Test: 15,304 / 1,700 / 4,252",
        "",
        "## Aggregate primary results (mean ± sample SD, n=3)",
        "",
        "| Condition | Acc/RobAcc | F1/RobF1 | ASR | RelDrop | ECE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        "| Clean | %s | %s | - | - | %s |"
        % (
            fmt(agg, "clean_acc"),
            fmt(agg, "clean_macro_f1"),
            fmt(agg, "clean_ece"),
        ),
        "| Mixed@1.0 | %s | %s | %s | %s | %s |"
        % (
            fmt(agg, "mixed1_robust_acc"),
            fmt(agg, "mixed1_robust_f1"),
            fmt(agg, "mixed1_asr"),
            fmt(agg, "mixed1_rel_drop"),
            fmt(agg, "mixed1_ece"),
        ),
        "| Adaptive@1.0 | %s | %s | %s | %s | %s |"
        % (
            fmt(agg, "adaptive1_robust_acc"),
            fmt(agg, "adaptive1_robust_f1"),
            fmt(agg, "adaptive1_asr"),
            fmt(agg, "adaptive1_rel_drop"),
            fmt(agg, "adaptive1_ece"),
        ),
        "",
        "## Per-seed checkpoints",
        "",
    ]
    for row in rows:
        seed = int(row["seed"])
        lines.append(
            "- seed%d: best_epoch=%d, best_val_CE=%.12f"
            % (seed, row["best_epoch"], row["best_val_loss"])
        )
        lines.append("  result=" + results[seed]["path"])
        lines.append(
            "  checkpoint_sha256=" + results[seed]["checkpoint_sha256"]
        )

    (FINAL_DIR / "results_3seed.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    readme = "\n".join(
        [
            "# README FIRST - USTC V17 final 3-seed archive",
            "",
            "Seeds 42/43/44 use the same full cache, dataset manifest,",
            "split seed, split manifest, model source and frozen protocol.",
            "Only the training seed changes.",
            "",
            "Dataset: archived 12-class / 21,256-flow USTC subset.",
            "This is not the full original 20-class / 215,256-flow corpus.",
            "",
            "Files:",
            "- aggregate_summary.json",
            "- paper_ready_primary_per_seed.csv",
            "- paper_ready_aggregate_metrics.csv",
            "- results_3seed.md",
            "- run_manifest.json",
            "",
            "Do not treat the old manuscript Mixed 55.71 value as a",
            "matched strength-1.0 result: the archived old V14 log recorded",
            "55.71 at Mixed strength 0.9 although the manuscript labeled it 1.0.",
            "",
        ]
    )
    (FINAL_DIR / "README_FIRST_最终实验归档说明.md").write_text(
        readme,
        encoding="utf-8",
    )

    print("", flush=True)
    print("=" * 100, flush=True)
    print("USTC V17 FINAL 3-SEED COMPLETE", flush=True)
    print("Archive: %s" % FINAL_DIR, flush=True)
    print("Clean Acc      : %s" % fmt(agg, "clean_acc"), flush=True)
    print("Clean Macro-F1 : %s" % fmt(agg, "clean_macro_f1"), flush=True)
    print("Mixed@1 Acc    : %s" % fmt(agg, "mixed1_robust_acc"), flush=True)
    print("Mixed@1 F1     : %s" % fmt(agg, "mixed1_robust_f1"), flush=True)
    print("Adaptive@1 Acc : %s" % fmt(agg, "adaptive1_robust_acc"), flush=True)
    print("Adaptive@1 F1  : %s" % fmt(agg, "adaptive1_robust_f1"), flush=True)
    print("=" * 100, flush=True)


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 100, flush=True)
    print("USTC-TFC2016 | V17 FINAL | full seed43 + seed44", flush=True)
    print("No model/LR/WD/robust-ratio/checkpoint-rule changes", flush=True)
    print("=" * 100, flush=True)

    validate_source_locks()
    validate_cache()
    seed42 = validate_seed42()

    print("[LOCK] sources verified", flush=True)
    print(
        "[LOCK] dataset manifest: " + EXPECTED_DATASET_MANIFEST_SHA256,
        flush=True,
    )
    print(
        "[LOCK] split manifest: " + EXPECTED_SPLIT_MANIFEST_SHA256,
        flush=True,
    )
    print("[seed=42] reuse: " + seed42["path"], flush=True)

    results = {42: seed42}

    for seed in TARGET_SEEDS:
        results[seed] = launch_seed(seed)
        if results[seed]["split_sha256"] != seed42["split_sha256"]:
            raise RuntimeError("seed%d split differs from seed42" % seed)
        if (
            results[seed]["config"]["dataset_manifest_sha256"]
            != seed42["config"]["dataset_manifest_sha256"]
        ):
            raise RuntimeError(
                "seed%d dataset manifest differs from seed42" % seed
            )

    write_archive(results)


if __name__ == "__main__":
    main()
