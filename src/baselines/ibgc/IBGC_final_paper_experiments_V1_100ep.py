#!/usr/bin/env python3
"""Standalone V17 IBGC experiment entry point; see README before running."""
import argparse
import csv
import os
import platform
import statistics
import sys
from pathlib import Path
import numpy as np
import torch
import torch_geometric
import sklearn
import baseline_v17_common as C
from ibgc_data import prepare, load_partition, verify_common, CONDITIONS
from ibgc_engine import train, evaluate, EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY
from ibgc_io import Logger, sha256, read_json, write_json, bind_manifest

ROOT = Path(__file__).resolve().parent
RUN_NAME = "IBGC_V17_V1_100EP"


def code_identity():
    return {p.name: sha256(p) for p in sorted(ROOT.glob("*.py"))}


def aggregate(run_dir, specs):
    rows = []
    for spec in specs:
        prep = read_json(run_dir / spec.key / "preparation.json")
        for seed in C.SEEDS:
            e = read_json(run_dir / spec.key / "evaluations" / ("seed" + str(seed)) / "evaluation.json")
            identity = e["identity"]
            stored = read_json(run_dir / spec.key / "models" / ("seed" + str(seed)) / "identity.json")
            if (identity != stored or e["completed_epochs"] != 100 or identity["seed"] != seed
                    or identity["preparation"] != prep or identity["code"] != code_identity()):
                raise RuntimeError("Evaluation provenance mismatch during aggregation")
            for condition, metrics in e["metrics"].items():
                for key, value in metrics.items():
                    if key in ("probabilities_sha256", "n", "cross_entropy"):
                        continue
                    rows.append(dict(dataset=spec.name, seed=seed, condition=condition,
                                     metric=key, value=value))
    groups = {}
    for r in rows:
        groups.setdefault((r["dataset"], r["condition"], r["metric"]), []).append(r["value"])
    aggregated = []
    for (dataset, condition, metric), values in groups.items():
        valid = all(v is not None and np.isfinite(v) for v in values)
        if len(values) != 3:
            raise RuntimeError("Aggregation requires all three seeds")
        aggregated.append(dict(dataset=dataset, condition=condition, metric=metric,
                               mean=statistics.mean(values) if valid else None,
                               sample_sd=statistics.stdev(values) if valid else None,
                               n_seeds=3, status="ok" if valid else "undefined"))
    for filename, records in (("per_seed.csv", rows), ("aggregate.csv", aggregated)):
        with (run_dir / filename).open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0]))
            w.writeheader()
            w.writerows(records)
    write_json(run_dir / "summary.json", dict(seeds=C.SEEDS, epochs=100,
               datasets=[s.name for s in specs], aggregate=aggregated))
    lines = ["# IBGC V17 results", "", "All metrics in percent; absolute drop in percentage points.",
             "Three seeds; sample SD (ddof=1). Negative drops are retained.", "",
             "| Dataset | Condition | Metric | Mean ± sample SD |", "|---|---|---|---:|"]
    for r in aggregated:
        cell = "undefined" if r["mean"] is None else "{:.4f} ± {:.4f}".format(r["mean"], r["sample_sd"])
        lines.append("| {} | {} | {} | {} |".format(r["dataset"], r["condition"], r["metric"], cell))
    (run_dir / "PAPER_READY_IBGC.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["prepare", "train", "evaluate", "aggregate", "all"], default="all")
    p.add_argument("--datasets", choices=["both", "cipher", "ustc"], default="both")
    p.add_argument("--seeds", default="42,43,44", choices=["42,43,44"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--work-root", type=Path, default=Path.cwd())
    p.add_argument("--result-root", type=Path, default=Path("/home/admin1/lkfiles/RESULT"))
    p.add_argument("--raw-cache-root", type=Path)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    verify_common()
    run_dir = (args.run_dir or args.result_root / RUN_NAME).resolve()
    cache_root = (args.raw_cache_root or args.result_root / "BASELINE_V17_SHARED_RAW_CACHE").resolve()
    specs = C.build_dataset_specs(args.datasets, args.work_root, args.result_root, "IBGC")
    device = torch.device(args.device)
    if args.mode in ("train", "evaluate", "all") and device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose --device cpu explicitly")
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_dir / ".running.lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError("Run is locked. If prior process has exited, remove only " + str(lock))
    os.write(fd, (str(os.getpid()) + "\n").encode())
    os.close(fd)
    log = Logger(run_dir / "experiment.log")
    try:
        manifest = dict(baseline=RUN_NAME, code=code_identity(), epochs=EPOCHS,
                        seeds=C.SEEDS, batch_size=BATCH_SIZE, optimizer="Adam",
                        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                        scheduler=None, selection="minimum clean validation CE; earliest tie",
                        attack_base_seed=C.ATTACK_BASE_SEED, train_conditions=["clean"],
                        evaluation_conditions=["clean", "mixed@1", "adaptive@1"],
                        normalization="none (upstream graph features)", early_stopping=False)
        bind_manifest(run_dir / "run_manifest.json", manifest)
        environment = dict(python=platform.python_version(), torch=torch.__version__,
                           torch_geometric=torch_geometric.__version__, numpy=np.__version__,
                           sklearn=sklearn.__version__, device=str(device), threads=args.threads,
                           cuda_version=torch.version.cuda)
        log.log("[start] mode={} datasets={} environment={}".format(args.mode, args.datasets, environment))
        if args.mode == "aggregate":
            aggregate(run_dir, specs)
            return
        for spec in specs:
            out = run_dir / spec.key
            prep = prepare(spec, cache_root, out, log)
            if args.mode == "prepare":
                continue
            # Test graphs are built during preparation but are never passed to train.
            train_data = validation_data = None
            if args.mode in ("all", "train"):
                train_data = load_partition(out, prep, "train_clean")
                validation_data = load_partition(out, prep, "validation_clean")
            for seed in C.SEEDS:
                identity = dict(run=manifest, preparation=prep, code=code_identity(), seed=seed,
                                environment=environment)
                model_dir = out / "models" / ("seed" + str(seed))
                if args.mode in ("all", "train"):
                    train(train_data, validation_data, spec.num_classes, seed, device,
                          model_dir, identity, log, resume=args.resume)
                if args.mode in ("all", "evaluate"):
                    # Evaluation may move between devices; validate against stored
                    # training identity, while retaining current code/prep checks.
                    stored = read_json(model_dir / "identity.json")
                    for field in ("run", "preparation", "code", "seed"):
                        if stored[field] != identity[field]:
                            raise RuntimeError("Training/evaluation identity mismatch: " + field)
                    test_sets = {c: load_partition(out, prep, "test_" + c)
                                 for c in ("clean", "mixed_1.0", "adaptive_1.0")}
                    evaluate(test_sets, spec.num_classes, device, model_dir,
                             out / "evaluations" / ("seed" + str(seed)), stored, log)
                    del test_sets
            del train_data, validation_data
        if args.mode in ("all", "evaluate"):
            aggregate(run_dir, specs)
        log.log("[done] " + str(run_dir))
    except Exception:
        import traceback
        log.log(traceback.format_exc())
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
