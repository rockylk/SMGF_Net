# Reproducing the archived experiments

## What can run without original datasets

```bash
python scripts/validate_archive.py
python scripts/summarize_results.py --check
python scripts/summarize_results.py
python scripts/run_experiment.py --model smgfnet --dataset cipher --dry-run
```

The first three commands verify/rebuild the archived result documents using Python's standard library. They do not repeat training, inference, or profiling. Dry-run displays the selected entry point settings without importing PyTorch or modifying outputs.

To redraw the original ROC figures, install NumPy, Matplotlib, and Pillow, then run:

```bash
python analysis/roc/calculate_and_plot.py --plot-only --include-clean
```

This reads the colocated AUROC summary and compact curve archive. It writes the figure products into `analysis/roc/`; original vector snapshots in `figures/roc/` are retained separately.

## Configure original assets

Copy `configs/paths.example.json` to `configs/paths.local.json`. Replace the two dataset paths and, when needed, the ET-BERT checkout and seed42 checkpoint paths. All relative config paths are resolved against the repository root.

- Dataset directories must contain the exact original relative PCAP paths in `data/splits/cipher.json` and `data/splits/ustc.json`.
- Keep both split files byte-for-byte. Hashes, population counts, labels, and ordering are checked by the frozen runners.
- Use the original CUDA-capable environment for formal SMGF experiments. Install compatible PyTorch/PyG wheels rather than assuming one CUDA wheel fits every machine.
- ET-BERT requires the pinned upstream source and pretrained assets described in [THIRD_PARTY.md](THIRD_PARTY.md).
- No raw PCAP, trained weight, or full individual probability file is included in this source archive.

## Baseline training and evaluation

```bash
python scripts/run_experiment.py --model ustc_cnn --dataset both --mode all --config configs/paths.local.json
python scripts/run_experiment.py --model pean_nhp --dataset both --mode all --config configs/paths.local.json
python scripts/run_experiment.py --model ibgc --dataset both --mode all --config configs/paths.local.json
python scripts/run_experiment.py --model etbert --dataset both --mode all --config configs/paths.local.json
```

`--dataset cipher` or `--dataset ustc` selects one dataset. The baseline protocols use all three formal training seeds. The optional `prepare`, `train`, `evaluate`, and `aggregate` modes retain their original meanings; later stages require the outputs of earlier stages in the same result directory. Use `scripts/summarize_results.py` to inspect the bundled CSV snapshots, not a native training runner's aggregate mode.

IBGC uses a shared raw direction/size/IAT cache. In `all` or `prepare` mode the adapter generates that cache using the exact frozen common utilities before calling the original IBGC runner. An existing protocol-matched cache can be reused under `<cache_root>/baseline_raw/`. No FS-Net model is needed to prepare it.

New outputs default to `outputs/RESULT/<original-run-name>/`; preprocessing caches default to `outputs/cache/`. Original dataset-specific schemas and filenames are retained so downstream inference and provenance tools remain compatible. The adapter rejects a result directory inside the archived `src`, `data`, `results`, `analysis`, `figures`, or `manifests` trees.

## SMGF-Net USTC

```bash
python scripts/run_experiment.py --model smgfnet --dataset ustc --seeds 42,43,44 --config configs/paths.local.json
```

Runs the original USTC frozen runner once per seed with full data (`SMGF_USTC_PILOT_FRACTION=1.0`). The adapter relocates dataset/output/cache path globals after loading the original source, and does not modify source bytes, frozen hyperparameters, or model code. Each seed gets its original timestamped output layout. These newly trained models are new runs, not replacements for the bundled historical result observations.

## SMGF-Net CipherSpectrum

```bash
python scripts/run_experiment.py --model smgfnet --dataset cipher --config configs/paths.local.json
```

This is the **original archival orchestration**: it reuses the selected `RATIO_020` seed42 checkpoint and trains seeds43/44, then evaluates all three on the canonical attack cache. Supply the original seed42 checkpoint through `smgf_seed42_checkpoint` in the config or `--smgf-seed42-checkpoint`.

The selected checkpoint itself is not included. Consequently, this entry point is not a from-scratch, checkpoint-free three-seed training command. The archive intentionally does not silently retrain seed42 with a different random stream or substitute another checkpoint. The original selected source, final seed42 result, and model checksum are preserved in the result snapshots.

## Matched 2×2 ablation

```bash
python scripts/run_experiment.py --model ablation --dataset cipher --seeds 42 --config configs/paths.local.json
```

Uses the original full seed42 checkpoint plus three newly trained ablation variants. The checkpoint's exact expected SHA-256 is `a0002cb6cb692b9965b29ec187b7aefd902a475361c1cdc0450077397c4ee7d9`. The original runner checks initialization, data split, canonical attacks, and reproduction of the Full result. This strict experiment requires the original checkpoint lineage and environment, not arbitrary compatible weights.

## Probability export and macro ROC

Export SMGF probabilities using the original frozen checkpoints:

```bash
python analysis/roc/export_smgf_probabilities.py --source-dir src/smgfnet/frozen --result-root /path/to/formal-results --output outputs/probability_exports
```

The exporter checks source hashes, original checkpoint identity, split/order, population sizes, and agreement with archived accuracies. It does not train. Formal result directories must retain their original names and expected per-seed files.

Recompute the full five-model analysis from restored probability files:

```bash
python analysis/roc/calculate_and_plot.py --data-root /path/to/formal-results --exports outputs/probability_exports --output outputs/roc --include-clean
```

The complete calculation requires all 90 model/dataset/seed/condition probability outputs. The archive includes their original summary/provenance records and compact display curves, not the full prediction arrays.

## Gate audit, strength sweeps, and profiling

- Gate audit source: `src/smgfnet/frozen/V17_CipherSpectrum_Figure10_Gate_Audit.py`. In the original CUDA environment, set `SMGF_CIPHER_FORMAL_RUN` to the formal CipherSpectrum run, `SMGF_PCAP_DIR` to the dataset, and `SMGF_RESULT_DIR` to a new output directory before executing the file. This audit verifies the exact three checkpoints and data split.
- Strength sweeps: `analysis/sensitivity/native/` contains the original three baseline GPU evaluators and collector. Their native layout expects the sibling baseline modules and external formal result directories. They are preserved as archival sources, not advertised as standalone portable commands. The complete portable observed-value table is regenerated by `scripts/summarize_results.py` into `results/SENSITIVITY.md`.
- The original cross-model figure script in `analysis/sensitivity/` likewise expects the original experiment layout. Its source and the 45 measured intermediate observations are retained; see the preceding result-regeneration command for layout-independent access.
- Profiling: `analysis/overhead/server_oneclick.py --code-root /path/to/original-code-root --result-root /path/to/formal-results` retains the original server layout discovery and checksum verification. It requires the five model checkpoints, exact prepared inputs/raw cache, upstream ET-BERT, and CUDA. The included vendor sources are the original fallback sources. `rebuild_fig7_only.py` reads existing measurement JSONs without rerunning GPU work. Historical efficiency Figure7 filenames are unrelated to the current manuscript's Gate Fig. 7; Table VIII is the adopted overhead presentation.

No new training, SSH access, source download, or GPU experiment is started by creating this archive. Current packaging verification is reported separately in `manifests/validation_report.json`.
