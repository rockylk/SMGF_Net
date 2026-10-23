#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only frozen V17 source/checkpoint inventory. No model deserialization."""
from __future__ import annotations
import argparse, csv, hashlib, json, platform, sys, os
from pathlib import Path

ROOT = Path(os.environ.get('S56_CODE_ROOT', '/home/admin1/lkfiles')).resolve()
RESULT = Path(os.environ.get('S56_RESULT_ROOT', '/home/admin1/lkfiles/RESULT')).resolve()
HERE = Path(__file__).resolve().parent
SEEDS = (42, 43, 44)
MODEL_DIRS = {
    "SMGF-Net": "SMGF_V17_CIPHERSPECTRUM_FINAL_3SEED_20260923_135542",
    "USTC-CNN": "USTC_CNN_FINAL_PAPER_BASELINE_V1",
    "ET-BERT": "ETBERT_FINAL_PAPER_BASELINE_V1",
    "PEAN-NHP": "PEAN_NHP_FINAL_PAPER_BASELINE_V1_100EP",
    "IBGC": "IBGC_V17_V1_100EP",
}
SOURCES = {
    "SMGF-Net": "V17-SMGF-Net_final_paper_experiments.py",
    "USTC-CNN": "BaseLine/USTC_CNN_final_paper_experiments.py",
    "ET-BERT": "BaseLine/ET-BERT_final_paper_experiments.py",
    "PEAN-NHP": "BaseLine/PEAN_NHP_V17_model.py",
    "IBGC": "BaseLine/ibgc_model.py",
}
ROBUST = {
    "SMGF-Net": (78.92, 86.88),
    "USTC-CNN": (9.05, 9.55),
    "ET-BERT": (22.94, 25.20),
    "PEAN-NHP": (28.51, 28.88),
    "IBGC": (17.62, 17.37),
}
def _candidate_dirs(*folders: str):
    """Visible GPU top-level sibling folders; inspect exact scoped locations.

    MODEL CODE may be alongside RESULT, but model weights remain ONLY
    RESULT/<frozen official run>. Folder names alone never authorize a
    different historical checkpoint.
    """
    seen=set()
    for folder in folders:
        root=ROOT/folder if folder else ROOT
        for p in (root,root/"src",root/"BaseLine",root/"baseline"):
            if p in seen:continue
            seen.add(p)
            if p.is_dir():yield p

def _verified_vendor_relative(name: str, locations):
    """Use only same-SHA copy of the frozen official model code.

    Anything in PEAN_FINAL_V17 / PEAN_NHP_V1_100EP could be older:
    name-matching alone is NOT sufficient for scientific reproducibility.
    """
    frozen=HERE/"vendor"/name
    if not frozen.is_file():
        raise FileNotFoundError("Package missing required frozen source "+str(frozen))
    frozen_sha=sha256(frozen)
    for root in locations:
        candidate=root/name
        if candidate.is_file() and sha256(candidate)==frozen_sha:
            return candidate.resolve()
    return frozen.resolve()

def base_dir() -> Path:
    """IBGC's *three* source modules must be a co-located, matched set."""
    names=("ibgc_graph.py","ibgc_model.py","baseline_v17_common.py")
    frozen=HERE/"vendor"
    if not all((frozen/n).is_file() for n in names):
        raise FileNotFoundError("IBGC frozen code module missing in "+str(frozen))
    digests={n:sha256(frozen/n) for n in names}
    candidates=list(_candidate_dirs("IBGC_V17_V1_100EP","", "BaseLine"))
    for folder in candidates:
        if all((folder/n).is_file() and sha256(folder/n)==digests[n] for n in names):
            return folder.resolve()
    return frozen.resolve()

def source_path(model: str) -> Path:
    """Exact model-specific server folder topology, with hash-attested fallback."""
    if model=="IBGC":return base_dir()/"ibgc_model.py"
    if model=="PEAN-NHP":
        candidates=list(_candidate_dirs(
            "PEAN_NHP_V1_100EP","PEAN_FINAL_V17",
            "PEAN_FINAL_V2_100EP","", "BaseLine"))
        frozen=HERE/"vendor"/"PEAN_NHP_V17_model.py"
        if not frozen.is_file():raise FileNotFoundError(frozen)
        digest=sha256(frozen)
        trf=HERE/"vendor"/"official_sources"/"PEAN"/"TRF.py"
        if not trf.is_file():raise FileNotFoundError(trf)
        trf_digest=sha256(trf)
        for folder in candidates:
            main=folder/"PEAN_NHP_V17_model.py"
            dep=folder/"official_sources"/"PEAN"/"TRF.py"
            if (main.is_file() and dep.is_file()
                and sha256(main)==digest and sha256(dep)==trf_digest):
                return main.resolve()
        return frozen.resolve()
    if model=="USTC-CNN":
        return _verified_vendor_relative(
            "USTC_CNN_final_paper_experiments.py",
            _candidate_dirs("USTC_CNN_V17_V1_100EP","USTC_CNN_FINAL", "", "BaseLine"))
    if model=="SMGF-Net":
        # V17 imports an exact V16 core from its own directory; use a MATCHED pair.
        # Never substitute V15/V16 as the classifier when V17 is requested.
        names=("V17-SMGF-Net_final_paper_experiments.py",
               "V16-SMGF-Net_final_paper_experiments.py")
        vendor=HERE/"vendor"
        for name in names:
            if not (vendor/name).is_file():
                raise FileNotFoundError("Missing bundled V17 source/core "+name)
        expected={name:sha256(vendor/name) for name in names}
        for folder in _candidate_dirs("","SMGF-Net","SMGF_V17"):
            if all((folder/n).is_file() and sha256(folder/n)==expected[n] for n in names):
                return (folder/names[0]).resolve()
        return (vendor/names[0]).resolve()
    if model=="ET-BERT":
        return ROOT/"ET-BERT"/"inference"/"run_classifier_infer.py"
    raise ValueError(f"Unknown official model: {model}")

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def checkpoint_path(model: str, seed: int, dataset: str = "cipher") -> Path:
    root = RESULT / MODEL_DIRS[model]
    if model == "SMGF-Net":
        if dataset != "cipher":
            raise ValueError("SMGF USTC requires separately identified USTC snapshot; no fallback")
        file = "FINAL_seed42_REUSED.pt" if seed == 42 else f"FINAL_seed{seed}.pt"
        return root / "checkpoints" / file
    if model == "ET-BERT":
        return root / dataset / "models" / f"seed{seed}" / "finetuned.bin"
    if model == "USTC-CNN":
        return root / dataset / "models" / f"seed{seed}" / "best.pt"
    return root / dataset / "models" / f"seed{seed}" / "best_validation.pt"

def inventory(dataset: str = "cipher", root: Path = ROOT) -> dict:
    # Current deployment is CipherSpectrum-only; USTC uses separate SMGF directory.
    rows = []
    for model, folder in MODEL_DIRS.items():
        src = source_path(model)
        for seed in SEEDS:
            if model == "SMGF-Net" and dataset != "cipher":
                ckpt = RESULT / "SMGF_V17_USTC_FINAL_3SEED_20260924_144614" / "checkpoints" / (
                    "FINAL_seed42_REUSED.pt" if seed == 42 else f"FINAL_seed{seed}.pt")
            else:
                ckpt = RESULT / folder / dataset / "models" / f"seed{seed}" / (
                    "finetuned.bin" if model == "ET-BERT" else
                    "best.pt" if model == "USTC-CNN" else "best_validation.pt"
                ) if model != "SMGF-Net" else checkpoint_path(model, seed, dataset)
            rows.append({
                "model": model, "dataset": dataset, "seed": seed,
                "source_path": str(src), "source_sha256": sha256(src) if src.is_file() else "",
                "source_status": "PRESENT" if src.is_file() else "MISSING",
                "checkpoint_path": str(ckpt),
                "checkpoint_sha256": sha256(ckpt) if ckpt.is_file() else "",
                "checkpoint_size_bytes": ckpt.stat().st_size if ckpt.is_file() else "",
                "checkpoint_status": "PRESENT" if ckpt.is_file() else "MISSING",
            })
    return {
        "schema": "section56-v17-inventory-v1", "dataset": dataset, "seeds": list(SEEDS),
        "scope": "read-only checkpoint/file identity; existence is NOT proof of model validity",
        "python": sys.version, "platform": platform.platform(), "rows": rows,
    }

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cipher", "ustc"], default="cipher")
    p.add_argument("--output-dir", type=Path, default=HERE / "results" / "inventory")
    args = p.parse_args()
    data = inventory(args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.dataset}_inventory.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / f"{args.dataset}_inventory.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data["rows"][0]))
        writer.writeheader()
        writer.writerows(data["rows"])
    print(f"DATASET={args.dataset} COUNT={len(data['rows'])}")
    for model in MODEL_DIRS:
        subset = [x for x in data["rows"] if x["model"] == model]
        print(f"{model}: checkpoints={sum(x['checkpoint_status']=='PRESENT' for x in subset)}/{len(subset)}")
    print(f"OUTPUT={args.output_dir.resolve()}")

if __name__ == "__main__":
    main()