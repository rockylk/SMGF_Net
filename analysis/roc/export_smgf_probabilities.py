# -*- coding: utf-8 -*-
"""Export formal SMGF test probabilities using frozen checkpoints, no training.

Run on the original GPU server. Original result files are read-only inputs.
Requires the original training environment, datasets, cache and frozen sources.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import numpy as np

HASHES = {
    "V17-SMGF-Net_final_paper_experiments.py": "17c88de543207e0241c666a4aa7aae4d36c023dda37ac131d7531957ede2cfa9",
    "V16-SMGF-Net_final_paper_experiments.py": "3ecaf0054a1a190b444fe6f3c19cf6203273ad32707041ec53b09318ca20a4e2",
    "V17_USTC_frozen_runner.py": "5f048e69f18271b7be5634f136d73d6776bce579a2741fb681b529ae3bf3e22b",
}
CIPHER = "SMGF_V17_CIPHERSPECTRUM_FINAL_3SEED_20260923_135542"
USTC_AGG = "SMGF_V17_USTC_FINAL_3SEED_20260924_144614"
USTC_RUNS = {
    42: "SMGF_V17_USTC_SEED42_FULL_F10000_20260924_082814",
    43: "SMGF_V17_USTC_SEED43_FULL_F10000_20260924_144614",
    44: "SMGF_V17_USTC_SEED44_FULL_F10000_20260924_152046",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def check_split(formal, regenerated):
    original, current = read_json(formal), read_json(regenerated)
    if original["dataset_manifest_sha256"] != current["dataset_manifest_sha256"]:
        raise RuntimeError("Dataset manifest changed")
    for partition in ("train", "validation", "test"):
        fields = ("sample_index", "relative_path", "label")
        a = [tuple(r[k] for k in fields) for r in original[partition]]
        b = [tuple(r[k] for k in fields) for r in current[partition]]
        if a != b:
            raise RuntimeError(f"Frozen {partition} split/order changed")
    return original


def run_dataset(args, dataset, scratch):
    os.environ["SMGF_RESULT_DIR"] = str(scratch)
    os.environ["SMGF_PCAP_DIR"] = str(args.cipher_pcap_root)
    os.environ["SMGF_USTC_PILOT_FRACTION"] = "1.0"
    if dataset == "cipher":
        native = import_file(args.source_dir / "V17-SMGF-Net_final_paper_experiments.py", "fig5_frozen_cipher")
        core, torch = native.CORE, native.torch
        native.SPLIT_FILE = scratch / "cipher_split.json"
        native.LOGGER = native.Logger(scratch / "cipher_inference.log")
        _, _, test, node_scaler, stats_scaler, classes, _, _ = native.load_and_prepare_data()
        formal_split = args.result_root / CIPHER / "split_manifest.json"
        split = check_split(formal_split, native.SPLIT_FILE)
        records = read_json(args.result_root / CIPHER / "per_seed_results.json")["test_results"]
        records = {int(r["seed"]): r for r in records}
        def model_factory():
            return native.SMGFNetV17(node_in_channels=5, stats_dim=9, finger_seq_len=20,
                                     entropy_seq_len=5, hidden_channels=32,
                                     out_channels=classes, gate_mode="learned").to(native.DEVICE)
        checkpoints = {seed: args.result_root / CIPHER / "checkpoints" /
                       ("FINAL_seed42_REUSED.pt" if seed == 42 else f"FINAL_seed{seed}.pt")
                       for seed in (42, 43, 44)}
        expected_hashes = None
    else:
        native = import_file(args.source_dir / "V17_USTC_frozen_runner.py", "fig5_frozen_ustc")
        core, torch = native.CORE, native.torch
        native.RUN_DIR = scratch
        native.SPLIT_FILE = scratch / "ustc_split.json"
        native.INVENTORY_FILE = scratch / "ustc_inventory.json"
        native.FAILED_PCAP_FILE = scratch / "ustc_failed_pcaps.json"
        native.USTC_ROOT = args.ustc_pcap_root
        logger = native.Logger(scratch / "ustc_inference.log")
        native.rewire_outputs(logger)
        data = native.prepare_data(logger)
        test, node_scaler, stats_scaler, classes = (
            data["test_set"], data["node_scaler"], data["stats_scaler"], data["num_classes"])
        formal_split = args.result_root / USTC_RUNS[42] / "split_manifest.json"
        split = check_split(formal_split, native.SPLIT_FILE)
        records = {seed: read_json(args.result_root / folder / "summary.json") for seed, folder in USTC_RUNS.items()}
        checkpoints = {seed: args.result_root / folder / "checkpoints" / f"USTC_FINAL_seed{seed}.pt"
                       for seed, folder in USTC_RUNS.items()}
        expected_hashes = read_json(args.result_root / USTC_AGG / "run_manifest.json")["seeds"]
        model_factory = lambda: native.new_model(classes)
    expected_n, expected_c = (8200, 41) if dataset == "cipher" else (4252, 12)
    if len(test) != expected_n or classes != expected_c:
        raise RuntimeError("Unexpected frozen test population")
    conditions = {"clean": test}
    for attack in ("mixed", "adaptive"):
        attack_seed = core.stable_seed(20260919, attack, "1.000")
        attacked = core.build_attacked_dataset_deterministic(test, attack, node_scaler, stats_scaler, 1.0, attack_seed)
        core.validate_paired_dataset(test, attacked, f"Fig5:{dataset}:{attack}@1.0")
        if attack == "adaptive":
            core.validate_adaptive_semantics(test, attacked)
        conditions[attack + "_1.0"] = attacked
    canonical_y = np.array([r["label"] for r in split["test"]])
    sample_ids = np.array([r["sample_index"] for r in split["test"]])
    checkpoint_hashes = {seed: sha(path) for seed, path in checkpoints.items()}
    if len(set(checkpoint_hashes.values())) != 3:
        raise RuntimeError("Distinct formal checkpoints are required")
    for seed, checkpoint in checkpoints.items():
        if expected_hashes and checkpoint_hashes[seed] != expected_hashes[str(seed)]["checkpoint_sha256"]:
            raise RuntimeError("Formal USTC checkpoint SHA mismatch")
        if dataset == "cipher" and Path(records[seed]["checkpoint"]).name != checkpoint.name:
            raise RuntimeError("Formal Cipher checkpoint name mismatch")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if int(payload.get("seed", -1)) != seed:
            raise RuntimeError("Checkpoint seed mismatch")
        model = model_factory()
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        for condition, data in conditions.items():
            target = args.output / "SMGF-Net" / dataset / f"seed{seed}" / condition
            target.parent.mkdir(parents=True, exist_ok=True)
            reference = records[seed]["clean"] if condition == "clean" else records[seed]["attacks"][condition.replace("_1.0", "@1.0")]
            ref_acc = reference.get("acc", reference.get("robust_acc"))
            if Path(str(target) + ".npz").exists() and Path(str(target) + ".json").exists():
                old = read_json(Path(str(target) + ".json"))
                if old["probability_sha256"] == sha(Path(str(target) + ".npz")) and old["checkpoint_sha256"] == checkpoint_hashes[seed]:
                    print("SKIP_VERIFIED", dataset, seed, condition, flush=True)
                    continue
                raise RuntimeError("Existing export provenance mismatch")
            predictions, labels, probabilities = core.predict_closed_set_with_probs(model, data, batch_size=64)
            if not np.array_equal(labels, canonical_y) or probabilities.shape != (expected_n, expected_c):
                raise RuntimeError("Prediction population/order mismatch")
            accuracy = float(np.mean(predictions == labels) * 100)
            if abs(accuracy - ref_acc) > 1e-8:
                raise RuntimeError(f"Formal accuracy not reproduced: {dataset}/{seed}/{condition}: {accuracy} vs {ref_acc}")
            if not np.isfinite(probabilities).all() or abs(probabilities.sum(axis=1)-1).max() > 1e-4:
                raise RuntimeError("Invalid probabilities")
            np.savez_compressed(Path(str(target) + ".npz"), labels=labels, preds=predictions,
                                probs=probabilities, sample_indices=sample_ids)
            evidence = dict(dataset=dataset, seed=seed, condition=condition,
                            checkpoint_sha256=checkpoint_hashes[seed], split_sha256=sha(formal_split),
                            probability_sha256=sha(Path(str(target) + ".npz")),
                            reference_accuracy_pct=ref_acc, recomputed_accuracy_pct=accuracy,
                            n=expected_n, classes=expected_c, source_sha256=HASHES,
                            training_performed=False, checkpoint_selection_changed=False,
                            attack_base_seed=20260919)
            Path(str(target) + ".json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
            print("EXPORTED", dataset, seed, condition, "accuracy", accuracy, flush=True)
        del model, payload
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cipher-pcap-root", type=Path, default=Path("/home/admin1/dataset/aes-128-gcm/aes-128-gcm"))
    parser.add_argument("--ustc-pcap-root", type=Path, default=Path("/home/admin1/dataset/USTC-TFC2016-Split"))
    parser.add_argument("--datasets", choices=("both", "cipher", "ustc"), default="both")
    args = parser.parse_args()
    if "云端" in str(args.result_root):
        parser.error("Cloud-labelled experiment directories are excluded")
    args.output = args.output.resolve()
    for filename, expected in HASHES.items():
        if sha(args.source_dir / filename) != expected:
            raise RuntimeError("Frozen source SHA mismatch: " + filename)
    if args.output == args.result_root.resolve() or args.result_root.resolve() in args.output.parents:
        raise RuntimeError("Write exports outside the immutable formal RESULT directory")
    args.output.mkdir(parents=True, exist_ok=True)
    for dataset in (("cipher", "ustc") if args.datasets == "both" else (args.datasets,)):
        with tempfile.TemporaryDirectory(prefix="fig5_inference_") as temporary:
            run_dataset(args, dataset, Path(temporary))


if __name__ == "__main__":
    main()
