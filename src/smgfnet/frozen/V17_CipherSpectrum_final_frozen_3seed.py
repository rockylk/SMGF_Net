# -*- coding: utf-8 -*-
"""
CipherSpectrum final frozen-protocol three-seed runner.

The model/training protocol is frozen:
V17 learned Gate + attacked-label CE + NO_KL;
100 epochs; lr=0.002 for epochs 1-70 and 0.0005 for 71-100;
AdamW weight_decay=1e-4; robust batch ratio=0.20;
Mixed train strength U(0.3,0.9); attacked CE weight=1.0.

Seed42 is reused from the selected RATIO_020 checkpoint.
Only seeds43 and 44 are newly trained.
After both trainings finish, one deterministic canonical test attack cache is
shared across all three seeds for the full CipherSpectrum evaluation matrix.
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

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SOURCE = SCRIPT_DIR / "V17-SMGF-Net_final_paper_experiments.py"
EXPECTED_BASE_SHA256 = (
    "17c88de543207e0241c666a4aa7aae4d36c023dda37ac131d7531957ede2cfa9"
)
EXPECTED_MANIFEST = (
    "821bee04771f0c8dce3680a4301164d8ab48e03ec0d1ed57c3930015ad200a61"
)

SEEDS = [42, 43, 44]
NEW_SEEDS = [43, 44]
EPOCHS = 100
LR1 = 0.002
LR2 = 0.0005
LR_SWITCH = 71
WEIGHT_DECAY = 1e-4
ROBUST_RATIO = 0.20
ROBUST_MIN = 0.3
ROBUST_MAX = 0.9
ATTACK_CE_WEIGHT = 1.0
KL_WEIGHT = 0.0
BATCH_SIZE = 64

MULTI_ATTACKS = ["padding", "iat_jitter", "dummy", "mixed"]
MULTI_STRENGTHS = [0.3, 0.6, 0.9, 1.0]
ADAPTIVE_STRENGTHS = [0.4, 0.7, 1.0]

RESULT_ROOT = Path(
    os.environ.get("SMGF_RESULT_DIR", str(Path.cwd() / "RESULT"))
).resolve()
STAMP = time.strftime("%Y%m%d_%H%M%S", time.localtime())
RUN_DIR = RESULT_ROOT / f"SMGF_V17_CIPHERSPECTRUM_FINAL_3SEED_{STAMP}"
CKPT_DIR = RUN_DIR / "checkpoints"
HISTORY_DIR = RUN_DIR / "history"
FIGURE_DIR = RUN_DIR / "figures"
LOG_FILE = RUN_DIR / "experiment.log"
RESULT_FILE = RUN_DIR / "results.md"
SUMMARY_FILE = RUN_DIR / "summary.json"
PER_SEED_FILE = RUN_DIR / "per_seed_results.json"
CONFIG_FILE = RUN_DIR / "run_config.json"
SPLIT_FILE = RUN_DIR / "split_manifest.json"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def import_base():
    if sha256_file(BASE_SOURCE) != EXPECTED_BASE_SHA256:
        raise RuntimeError("V17 base source SHA mismatch; refusing protocol drift.")
    tmp = Path(tempfile.mkdtemp(prefix="smgf_cipher_final_import_"))
    old = os.environ.get("SMGF_RESULT_DIR")
    os.environ["SMGF_RESULT_DIR"] = str(tmp)
    try:
        spec = importlib.util.spec_from_file_location(
            "smgf_v17_base_cipher_final", BASE_SOURCE
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        if old is None:
            os.environ.pop("SMGF_RESULT_DIR", None)
        else:
            os.environ["SMGF_RESULT_DIR"] = old
        shutil.rmtree(tmp, ignore_errors=True)
    return module


BASE = import_base()
CORE = BASE.CORE
torch = BASE.torch
Batch = BASE.Batch
DEVICE = torch.device("cuda")


class Logger:
    def __init__(self, path):
        self.path = Path(path)

    def log(self, msg=""):
        print(msg, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")


def rewire(logger):
    BASE.LOGGER = logger
    BASE.RUN_DIR = RUN_DIR
    BASE.LOG_FILE = LOG_FILE
    BASE.RESULT_FILE = RESULT_FILE
    BASE.CONFIG_FILE = CONFIG_FILE
    BASE.SPLIT_FILE = SPLIT_FILE
    BASE.PER_SEED_FILE = PER_SEED_FILE
    BASE.CHECKPOINT_DIR = CKPT_DIR
    BASE.FIGURE_DIR = FIGURE_DIR

    CORE.RUN_DIR = RUN_DIR
    CORE.LOG_FILE = LOG_FILE
    CORE.RESULT_MD = RESULT_FILE
    CORE.SPLIT_MANIFEST_FILE = SPLIT_FILE
    CORE.CONFIG_FILE = CONFIG_FILE
    CORE.CHECKPOINT_DIR = CKPT_DIR
    CORE.FIGURE_DIR = FIGURE_DIR


def cpu_model_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def cpu_optimizer_state(optimizer):
    state = copy.deepcopy(optimizer.state_dict())
    for value in state.get("state", {}).values():
        for k, item in list(value.items()):
            if torch.is_tensor(item):
                value[k] = item.detach().cpu().clone()
    return state


def lr_at(epoch):
    return LR1 if epoch < LR_SWITCH else LR2


def discover_seed42():
    explicit = os.environ.get("SMGF_SEED42_CKPT")
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p

    roots = []
    for root in [
        RESULT_ROOT,
        Path.cwd() / "RESULT",
        Path("/home/admin1/lkfiles/RESULT"),
    ]:
        root = root.expanduser().resolve()
        if root not in roots:
            roots.append(root)

    found = []
    pattern = (
        "SMGF_V17_ROBUST_RATIO_TUNING_SEED42_*/"
        "RATIO_020/checkpoints/best.pt"
    )
    for root in roots:
        if root.exists():
            found.extend(root.glob(pattern))
    found = [p for p in found if p.is_file()]
    if not found:
        raise FileNotFoundError(
            "Cannot find seed42 RATIO_020 checkpoint. "
            "Set SMGF_SEED42_CKPT explicitly."
        )
    return max(found, key=lambda p: p.stat().st_mtime)


def validate_seed42(path, logger):
    payload = torch.load(path, map_location="cpu")
    if "model_state_dict" not in payload:
        raise RuntimeError("seed42 checkpoint lacks model_state_dict")
    checks = {
        "tag": payload.get("tag"),
        "seed": int(payload.get("seed", payload.get("train_seed", -1))),
        "ratio": float(payload.get("robust_batch_ratio", -1)),
        "wd": float(payload.get("weight_decay", -1)),
        "attack_ce": float(payload.get("attack_ce_weight", -1)),
        "kl": float(payload.get("kl_weight", -1)),
    }
    if checks["tag"] != "RATIO_020":
        raise AssertionError(f"seed42 tag mismatch: {checks}")
    if checks["seed"] != 42:
        raise AssertionError(f"seed42 seed mismatch: {checks}")
    if abs(checks["ratio"] - ROBUST_RATIO) > 1e-12:
        raise AssertionError(f"seed42 ratio mismatch: {checks}")
    if abs(checks["wd"] - WEIGHT_DECAY) > 1e-12:
        raise AssertionError(f"seed42 WD mismatch: {checks}")
    if abs(checks["attack_ce"] - ATTACK_CE_WEIGHT) > 1e-12:
        raise AssertionError(f"seed42 attack CE mismatch: {checks}")
    if abs(checks["kl"] - KL_WEIGHT) > 1e-12:
        raise AssertionError(f"seed42 KL mismatch: {checks}")
    if str(payload.get("gate_mode", "")) != "learned":
        raise AssertionError("seed42 gate mode is not learned")
    if int(payload.get("full_epochs", -1)) != EPOCHS:
        raise AssertionError(
            f"seed42 epoch budget mismatch: {payload.get('full_epochs')}"
        )
    expected_schedule = [[1, 70, LR1], [71, 100, LR2]]
    observed_schedule = [list(x) for x in payload.get("lr_schedule", [])]
    if observed_schedule != expected_schedule:
        raise AssertionError(
            f"seed42 LR schedule mismatch: {observed_schedule}"
        )
    logger.log(f"[seed42] validated frozen checkpoint: {path}")
    return payload


def train_seed(seed, train_set, val_set, num_classes, node_scaler, stats_scaler, logger):
    CORE.set_seed(seed)
    model = BASE.SMGFNetV17(
        node_in_channels=5,
        stats_dim=9,
        finger_seq_len=20,
        entropy_seq_len=5,
        hidden_channels=32,
        out_channels=num_classes,
        gate_mode="learned",
    ).to(DEVICE)
    init_sha = BASE.model_state_sha256(model)
    loader = CORE.make_loader(
        train_set, BATCH_SIZE, shuffle=True, seed=seed + 21_000
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR1, weight_decay=WEIGHT_DECAY
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    best_opt = None
    history = []

    logger.log("")
    logger.log("=" * 96)
    logger.log(f"TRAIN FINAL FROZEN seed={seed}")
    logger.log(
        f"epochs=100 | lr 0.002->0.0005@71 | wd=1e-4 | "
        f"robust_ratio={ROBUST_RATIO:.2f} | NO_KL"
    )
    logger.log("=" * 96)

    for epoch_zero in range(EPOCHS):
        epoch = epoch_zero + 1
        lr = lr_at(epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr

        model.train()
        total = clean_total = attack_total = 0.0
        batches = robust_batches = 0

        for batch_idx, batch_cpu in enumerate(loader):
            selector = CORE.stable_seed(
                "v17_robust_batch", seed, epoch_zero, batch_idx, modulo=1_000_000
            ) / 1_000_000.0
            attacked_batch = None

            if selector < ROBUST_RATIO:
                attack_seed = CORE.stable_seed(
                    "v17_train_attack", seed, epoch_zero, batch_idx
                )
                strength_rng = np.random.default_rng(
                    CORE.stable_seed(
                        "v17_train_strength", seed, epoch_zero, batch_idx
                    )
                )
                strength = float(strength_rng.uniform(ROBUST_MIN, ROBUST_MAX))
                np_rng = np.random.default_rng(attack_seed)
                py_rng = random.Random(attack_seed)
                attacked = [
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
                attacked_batch = Batch.from_data_list(attacked).to(DEVICE)

            batch = batch_cpu.to(DEVICE)
            optimizer.zero_grad()
            clean_logits = BASE.forward_logits(model, batch)
            clean_ce = criterion(clean_logits, batch.y.view(-1))
            loss = clean_ce
            attack_ce_value = 0.0

            if attacked_batch is not None:
                attacked_logits = BASE.forward_logits(model, attacked_batch)
                attack_ce = criterion(
                    attacked_logits, attacked_batch.y.view(-1)
                )
                loss = clean_ce + ATTACK_CE_WEIGHT * attack_ce
                attack_ce_value = float(attack_ce.item())
                robust_batches += 1

            loss.backward()
            optimizer.step()

            total += float(loss.item())
            clean_total += float(clean_ce.item())
            attack_total += attack_ce_value
            batches += 1

        val_loss = CORE.compute_supervised_loss(
            model, val_set, batch_size=BATCH_SIZE
        )
        new_best = val_loss < best_val - 1e-8
        if new_best:
            best_val = float(val_loss)
            best_epoch = epoch
            best_state = cpu_model_state(model)
            best_opt = cpu_optimizer_state(optimizer)

        row = {
            "epoch": epoch,
            "lr": lr,
            "loss": total / max(batches, 1),
            "clean_ce": clean_total / max(batches, 1),
            "attack_ce": attack_total / max(robust_batches, 1),
            "robust_batches": robust_batches,
            "total_batches": batches,
            "val_loss": float(val_loss),
            "is_new_best": bool(new_best),
            "best_epoch_so_far": best_epoch,
            "best_val_loss_so_far": best_val,
        }
        history.append(row)
        logger.log(
            f"[seed={seed}] epoch {epoch:03d}/100 | lr={lr:.7f} | "
            f"loss={row['loss']:.6f} | clean_ce={row['clean_ce']:.6f} | "
            f"attack_ce={row['attack_ce']:.6f} | "
            f"robust_batches={robust_batches}/{batches} | "
            f"val={val_loss:.6f} | best_ep={best_epoch} | best={best_val:.6f}"
        )

    if best_state is None:
        raise RuntimeError(f"seed={seed}: no best checkpoint")

    hist_path = HISTORY_DIR / f"seed{seed}.csv"
    with hist_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    ckpt_path = CKPT_DIR / f"FINAL_seed{seed}.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "optimizer_state_dict": best_opt,
            "tag": "FINAL",
            "seed": seed,
            "dataset": "CipherSpectrum",
            "epochs": EPOCHS,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "lr_schedule": [[1, 70, LR1], [71, 100, LR2]],
            "weight_decay": WEIGHT_DECAY,
            "robust_batch_ratio": ROBUST_RATIO,
            "robust_strength_range": [ROBUST_MIN, ROBUST_MAX],
            "attack_ce_weight": ATTACK_CE_WEIGHT,
            "kl_weight": KL_WEIGHT,
            "gate_mode": "learned",
            "initial_state_sha256": init_sha,
            "base_source_sha256": EXPECTED_BASE_SHA256,
            "dataset_manifest_sha256": EXPECTED_MANIFEST,
        },
        ckpt_path,
    )
    logger.log(
        f"[seed={seed}] COMPLETE | best_epoch={best_epoch} | "
        f"best_val={best_val:.12f} | {ckpt_path}"
    )
    del model
    torch.cuda.empty_cache()
    return {
        "seed": seed,
        "checkpoint": str(ckpt_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "initial_state_sha256": init_sha,
        "history": str(hist_path),
    }


def new_model(num_classes):
    return BASE.SMGFNetV17(
        node_in_channels=5,
        stats_dim=9,
        finger_seq_len=20,
        entropy_seq_len=5,
        hidden_channels=32,
        out_channels=num_classes,
        gate_mode="learned",
    ).to(DEVICE)


def evaluate_seed(seed, ckpt_path, num_classes, test_set, attack_cache, logger):
    payload = torch.load(ckpt_path, map_location="cpu")
    model = new_model(num_classes)
    model.load_state_dict(payload["model_state_dict"])

    clean = CORE.evaluate_clean(model, test_set)
    CORE.log_clean_result(logger, "FINAL", seed, clean)
    attacks = {}

    for attack in MULTI_ATTACKS:
        for strength in MULTI_STRENGTHS:
            key = f"{attack}@{strength:.1f}"
            result = CORE.evaluate_attack(
                model, attack_cache[(attack, float(strength))], clean
            )
            attacks[key] = result
            CORE.log_attack_result(
                logger, "FINAL", seed, attack, strength, result
            )

    for strength in ADAPTIVE_STRENGTHS:
        key = f"adaptive@{strength:.1f}"
        result = CORE.evaluate_attack(
            model, attack_cache[("adaptive", float(strength))], clean
        )
        attacks[key] = result
        CORE.log_attack_result(
            logger, "FINAL", seed, "adaptive", strength, result
        )

    clean_scalar = {
        "acc": float(clean["acc"]),
        "macro_f1": float(clean["macro_f1"]),
        "ece": float(clean["ece"]),
    }
    rec = {
        "seed": seed,
        "checkpoint": str(ckpt_path),
        "best_epoch": int(payload.get("best_epoch", payload.get("epoch", -1))),
        "best_val_loss": float(
            payload.get("best_val_loss", payload.get("val_loss", float("nan")))
        ),
        "clean": clean_scalar,
        "attacks": attacks,
    }
    del model
    torch.cuda.empty_cache()
    return rec


def mean_sd(vals):
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1))


def fmt(vals):
    m, s = mean_sd(vals)
    return f"{m:.2f} ± {s:.2f}"


def aggregate(per_seed):
    out = {"clean": {}, "attacks": {}}
    for key in ["acc", "macro_f1", "ece"]:
        vals = [float(r["clean"][key]) for r in per_seed]
        m, s = mean_sd(vals)
        out["clean"][key] = {"mean": m, "sd": s, "values": vals}
    for name in per_seed[0]["attacks"]:
        out["attacks"][name] = {}
        for key in ["robust_acc", "robust_f1", "asr", "rel_drop", "ece"]:
            vals = [float(r["attacks"][name][key]) for r in per_seed]
            m, s = mean_sd(vals)
            out["attacks"][name][key] = {
                "mean": m, "sd": s, "values": vals
            }
    return out


def final_attack_records(per_seed):
    out = {}
    for attack in MULTI_ATTACKS:
        for strength in MULTI_STRENGTHS:
            out[(attack, float(strength))] = [
                r["attacks"][f"{attack}@{strength:.1f}"] for r in per_seed
            ]
    for strength in ADAPTIVE_STRENGTHS:
        out[("adaptive", float(strength))] = [
            r["attacks"][f"adaptive@{strength:.1f}"] for r in per_seed
        ]
    return out


def write_results(dataset_info, per_seed, agg, train_records, seed42_source, sensitivity):
    primary_m = "mixed@1.0"
    primary_a = "adaptive@1.0"
    lines = [
        "# SMGF-Net V17 CipherSpectrum Final Frozen-Protocol Results",
        "",
        "## Protocol",
        "",
        "- Model: V17 + learned Gate + attacked-label CE + NO_KL.",
        "- Frozen protocol; no further hyperparameter tuning.",
        "- Formal seeds: 42, 43, 44.",
        "- Seed42 reuses the validation-selected r=0.20 checkpoint.",
        "- Seeds43/44 are newly trained under the same frozen protocol.",
        "- 100 epochs; no early stopping.",
        "- LR: 0.002 for epochs 1-70; 0.0005 for epochs 71-100.",
        "- AdamW weight decay: 1e-4.",
        "- Robust batch ratio: 0.20.",
        "- Mixed train strength: Uniform[0.3,0.9].",
        "- Attacked CE weight: 1.0; KL weight: 0.",
        "- Checkpoint: minimum clean-validation CE.",
        "- One deterministic test attack cache is shared across all three seeds.",
        "",
        "## Primary three-seed results",
        "",
        "| Seed | Best epoch | Clean Acc | Clean F1 | Mixed@1 Acc | Mixed@1 F1 | Adaptive@1 Acc | Adaptive@1 F1 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in per_seed:
        lines.append(
            f"| {r['seed']} | {r['best_epoch']} | "
            f"{r['clean']['acc']:.2f} | {r['clean']['macro_f1']:.2f} | "
            f"{r['attacks'][primary_m]['robust_acc']:.2f} | "
            f"{r['attacks'][primary_m]['robust_f1']:.2f} | "
            f"{r['attacks'][primary_a]['robust_acc']:.2f} | "
            f"{r['attacks'][primary_a]['robust_f1']:.2f} |"
        )
    lines.append(
        f"| mean ± SD | — | "
        f"{fmt([r['clean']['acc'] for r in per_seed])} | "
        f"{fmt([r['clean']['macro_f1'] for r in per_seed])} | "
        f"{fmt([r['attacks'][primary_m]['robust_acc'] for r in per_seed])} | "
        f"{fmt([r['attacks'][primary_m]['robust_f1'] for r in per_seed])} | "
        f"{fmt([r['attacks'][primary_a]['robust_acc'] for r in per_seed])} | "
        f"{fmt([r['attacks'][primary_a]['robust_f1'] for r in per_seed])} |"
    )
    lines.extend([
        "",
        "## Full robustness matrix",
        "",
        "| Attack | Strength | RobAcc | RobF1 | ASR | RelDrop | ECE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for attack in MULTI_ATTACKS:
        for strength in MULTI_STRENGTHS:
            name = f"{attack}@{strength:.1f}"
            xs = [r["attacks"][name] for r in per_seed]
            lines.append(
                f"| {attack} | {strength:.1f} | "
                f"{fmt([x['robust_acc'] for x in xs])} | "
                f"{fmt([x['robust_f1'] for x in xs])} | "
                f"{fmt([x['asr'] for x in xs])} | "
                f"{fmt([x['rel_drop'] for x in xs])} | "
                f"{fmt([x['ece'] for x in xs])} |"
            )
    for strength in ADAPTIVE_STRENGTHS:
        name = f"adaptive@{strength:.1f}"
        xs = [r["attacks"][name] for r in per_seed]
        lines.append(
            f"| adaptive | {strength:.1f} | "
            f"{fmt([x['robust_acc'] for x in xs])} | "
            f"{fmt([x['robust_f1'] for x in xs])} | "
            f"{fmt([x['asr'] for x in xs])} | "
            f"{fmt([x['rel_drop'] for x in xs])} | "
            f"{fmt([x['ece'] for x in xs])} |"
        )

    RESULT_FILE.write_text("\n".join(lines), encoding="utf-8")

    PER_SEED_FILE.write_text(
        json.dumps(
            {
                "dataset": "CipherSpectrum",
                "seed42_source": seed42_source,
                "new_training": train_records,
                "test_results": per_seed,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    SUMMARY_FILE.write_text(
        json.dumps(
            {
                "dataset": "CipherSpectrum",
                "dataset_info": dataset_info,
                "protocol_frozen": True,
                "seeds": SEEDS,
                "aggregate": agg,
                "sensitivity_figure": sensitivity,
                "script_sha256": sha256_file(Path(__file__).resolve()),
                "base_source_sha256": sha256_file(BASE_SOURCE),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    logger = Logger(LOG_FILE)
    rewire(logger)

    logger.log("=" * 104)
    logger.log("CipherSpectrum FINAL frozen-protocol three-seed experiment")
    logger.log("No more hyperparameter search.")
    logger.log(
        "seed42 reused; train only seed43/44; formal test after both trainings."
    )
    logger.log("=" * 104)

    seed42_path = discover_seed42()
    seed42_payload = validate_seed42(seed42_path, logger)
    seed42_copy = CKPT_DIR / "FINAL_seed42_REUSED.pt"
    shutil.copy2(seed42_path, seed42_copy)

    (
        train_set,
        val_set,
        test_set,
        node_scaler,
        stats_scaler,
        num_classes,
        _idx_to_class,
        manifest_hash,
    ) = BASE.load_and_prepare_data()
    if manifest_hash != EXPECTED_MANIFEST:
        raise AssertionError("CipherSpectrum manifest mismatch")

    dataset_info = {
        "manifest_hash": manifest_hash,
        "num_classes": num_classes,
        "num_samples": len(train_set) + len(val_set) + len(test_set),
        "train_n": len(train_set),
        "val_n": len(val_set),
        "test_n": len(test_set),
    }

    CONFIG_FILE.write_text(
        json.dumps(
            {
                "dataset": "CipherSpectrum",
                "protocol_frozen": True,
                "seeds": SEEDS,
                "seed42_reused": str(seed42_path),
                "new_train_seeds": NEW_SEEDS,
                "epochs": EPOCHS,
                "lr_schedule": [[1,70,LR1],[71,100,LR2]],
                "weight_decay": WEIGHT_DECAY,
                "robust_batch_ratio": ROBUST_RATIO,
                "robust_strength_range": [ROBUST_MIN, ROBUST_MAX],
                "attack_ce_weight": ATTACK_CE_WEIGHT,
                "kl_weight": KL_WEIGHT,
                "dataset_manifest": manifest_hash,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    BASE.validate_zero_strength_noop(
        test_set, node_scaler, stats_scaler, max_samples=16
    )

    train_records = []
    for seed in NEW_SEEDS:
        train_records.append(
            train_seed(
                seed,
                train_set,
                val_set,
                num_classes,
                node_scaler,
                stats_scaler,
                logger,
            )
        )

    logger.log("")
    logger.log("Training seed43/44 complete. Building formal test attack cache.")
    attack_cache = BASE.precompute_canonical_eval_cache(
        test_set, node_scaler, stats_scaler
    )

    ckpts = {
        42: seed42_copy,
        43: Path(train_records[0]["checkpoint"]),
        44: Path(train_records[1]["checkpoint"]),
    }
    per_seed = []
    for seed in SEEDS:
        logger.log("")
        logger.log(f"FORMAL TEST seed={seed}")
        per_seed.append(
            evaluate_seed(
                seed, ckpts[seed], num_classes, test_set, attack_cache, logger
            )
        )
        PER_SEED_FILE.write_text(
            json.dumps(
                {"completed_test_seeds": per_seed},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    agg = aggregate(per_seed)
    records = final_attack_records(per_seed)
    sensitivity = CORE.run_sensitivity_figure(records, logger)
    sensitivity = str(sensitivity) if sensitivity else None

    write_results(
        dataset_info,
        per_seed,
        agg,
        train_records,
        str(seed42_path),
        sensitivity,
    )

    logger.log("")
    logger.log("=" * 104)
    logger.log("CIPHERSPECTRUM FINAL THREE-SEED EXPERIMENT COMPLETE")
    logger.log(f"Results: {RESULT_FILE}")
    logger.log(f"Summary: {SUMMARY_FILE}")
    logger.log(f"Per-seed: {PER_SEED_FILE}")
    logger.log("=" * 104)


if __name__ == "__main__":
    main()
