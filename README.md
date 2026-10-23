# SMGF-Net

**Semantics-Guided Gated Fusion with Perturbation-Aware Training for Robust Encrypted Traffic Classification**

This repository organizes the final SMGF-Net V17 experiments and the four paper baselines: USTC-CNN, ET-BERT, PEAN-NHP, and IBGC. It includes training/evaluation runners, frozen data splits, analysis code, and archived experimental results.

The final method uses a learned context-conditioned channel-wise Gate and single-stage clean plus attacked-label cross-entropy. **The final experiments use no KL or contrastive objective.** The historical V16/V17 filenames identify checksum-preserved dependencies; their old default entry points are not the final training configuration. Use `scripts/run_experiment.py` or the dataset-specific frozen runners described below.

[中文说明](docs/README_zh.md) · [Experimental results](results/RESULTS.md) · [Reproduction](docs/REPRODUCING.md) · [Protocol](docs/PROTOCOL.md) · [Third-party provenance](docs/THIRD_PARTY.md)

## Repository structure

```text
github-src/
├── README.md
├── requirements.txt
├── requirements-etbert.txt
├── configs/
│   └── paths.example.json       # User-supplied datasets, assets, and output paths
├── src/
│   ├── smgfnet/frozen/          # Final dataset runners, model/core dependencies, ablation, Gate audit
│   └── baselines/
│       ├── common/             # Shared frozen population, metrics, and attack utilities
│       ├── ustc_cnn/
│       ├── etbert/             # Official ET-BERT orchestration; upstream assets are external
│       ├── pean_nhp/           # Constrained-input adapter and pinned PEAN source mirror
│       └── ibgc/               # Graph/model reproduction and frozen-cache adapter
├── scripts/
│   ├── run_experiment.py       # Portable path adapter; preserves training/attack settings
│   ├── summarize_results.py    # Regenerate Markdown/CSV from archived observations
│   └── validate_archive.py     # Offline archive, source, split, result, and CLI checks
├── data/splits/                # Exact CipherSpectrum and USTC split manifests
├── analysis/
│   ├── roc/                   # Macro OvR ROC analysis and compact redraw data
│   ├── sensitivity/           # Strength-sweep evaluation and figure code
│   └── overhead/              # Frozen five-model GPU profiling toolchain
├── results/
│   ├── RESULTS.md             # Clean, Mixed, Adaptive, AUROC, ablation, Gate, overhead
│   ├── SENSITIVITY.md
│   ├── tables/                # Normalized per-seed and aggregate machine-readable results
│   └── raw/                   # Original selected CSV/JSON snapshots
├── figures/                   # Selected existing vector figures
├── manifests/                 # Original source paths and SHA-256 checksums
└── docs/
```

Frozen files are deliberately retained byte-for-byte because the experimental runners verify their hashes. The directory organization and path adapter are new; the trained architecture, objectives, attacks, and archived results are unchanged. This is an organized release of the final experiment implementation, not a new implementation claiming independently reproduced GPU results.

## Results at a glance

Robust accuracy (%), mean ± sample SD over seeds 42, 43, 44:

| Dataset | Model | Mixed@1.0 | Adaptive@1.0 |
| --- | --- | ---: | ---: |
| CipherSpectrum | USTC-CNN | 9.05 ± 0.29 | 9.55 ± 0.42 |
| CipherSpectrum | ET-BERT | 22.94 ± 3.95 | 25.20 ± 3.67 |
| CipherSpectrum | PEAN-NHP | 28.51 ± 2.07 | 28.88 ± 2.01 |
| CipherSpectrum | IBGC | 17.62 ± 2.03 | 17.37 ± 2.19 |
| CipherSpectrum | SMGF-Net | **78.92 ± 0.46** | **86.88 ± 0.55** |
| USTC-TFC2016 | USTC-CNN | 28.63 ± 1.63 | 37.87 ± 0.76 |
| USTC-TFC2016 | ET-BERT | 75.10 ± 2.32 | 63.88 ± 3.78 |
| USTC-TFC2016 | PEAN-NHP | 69.02 ± 1.50 | 69.12 ± 1.84 |
| USTC-TFC2016 | IBGC | 26.03 ± 1.14 | 26.62 ± 1.43 |
| USTC-TFC2016 | SMGF-Net | **86.99 ± 2.71** | **89.28 ± 2.02** |

CipherSpectrum has 41 classes and 8,200 test flows. USTC-TFC2016 denotes the archived **12-class, 21,256-flow subset**, with 4,252 test flows. See [RESULTS.md](results/RESULTS.md) for full metrics and statistical boundaries.

## Quick start: inspect and regenerate results

These commands use only the Python standard library and require no GPU, PCAPs, or checkpoints:

```bash
python scripts/validate_archive.py
python scripts/summarize_results.py --check
python scripts/summarize_results.py
```

## Training and evaluation

Use Python 3.10 or newer. Install a CUDA-compatible PyTorch build and matching PyTorch Geometric dependencies for your GPU, followed by:

```bash
python -m pip install -r requirements.txt
```

The requirements list declares dependencies; it is not a reconstructed lockfile for all historical environments. Observed environment details are recorded in the original manifests. ET-BERT should use its pinned upstream dependencies in a separate environment.

Copy `configs/paths.example.json` to `configs/paths.local.json` and supply local dataset and asset paths. Relative paths are resolved from the repository root, not the current working directory. No paths or credentials are loaded from the original workstation.

Inspect a launch plan without importing ML packages or starting training:

```bash
python scripts/run_experiment.py --model ustc_cnn --dataset both --config configs/paths.local.json --dry-run
```

Run the complete final baseline protocols:

```bash
python scripts/run_experiment.py --model ustc_cnn --dataset both --config configs/paths.local.json
python scripts/run_experiment.py --model pean_nhp --dataset both --config configs/paths.local.json
python scripts/run_experiment.py --model ibgc --dataset both --config configs/paths.local.json
python scripts/run_experiment.py --model etbert --dataset both --config configs/paths.local.json
```

The IBGC `all`/`prepare` path first prepares the exact common direction/size/IAT cache from PCAPs. All baselines retain their own clean-only training settings; SMGF-Net robust training is not added to them.

Run the frozen SMGF-Net protocol:

```bash
python scripts/run_experiment.py --model smgfnet --dataset ustc --seeds 42,43,44 --config configs/paths.local.json
python scripts/run_experiment.py --model smgfnet --dataset cipher --config configs/paths.local.json
python scripts/run_experiment.py --model ablation --dataset cipher --seeds 42 --config configs/paths.local.json
```

**CipherSpectrum archive orchestration reuses the selected seed-42 checkpoint and trains seeds 43/44.** Supply `smgf_seed42_checkpoint` pointing to that original `RATIO_020` checkpoint. The matched ablation additionally requires its exact frozen checkpoint hash. This repository does not bundle that weight file or replace it with a newly trained model. USTC uses the full dataset for all three seeds; the adapter explicitly disables the historical 10% pilot default.

Detailed commands, dependencies, output layout, and inference-only export are in [REPRODUCING.md](docs/REPRODUCING.md).

## ROC analysis and plotting

The compact original ROC curve data are included. To redraw the archived main and clean reference figures without reading individual probability outputs:

```bash
python analysis/roc/calculate_and_plot.py --plot-only --include-clean
```

To recompute from original probabilities, see [REPRODUCING.md](docs/REPRODUCING.md). Legend AUROC uses original seed-level class AUCs, not the area under an interpolated mean curve. Bands show cross-model sample SD, not a 95% CI.

## Data, weights, and provenance

Raw PCAPs, checkpoints, full per-flow probability dumps, caches, credentials, and intermediate training logs are not included. Exact split manifests, compact analysis data, final summaries, and source checksums are included. Obtain datasets and upstream pretrained assets through their original authorized distribution channels. There is no invented download URL for unavailable project assets.

This archive includes the accepted later ET-BERT official rerun. Its documented preprocessing, attack seed, and checkpoint-selection differences are retained. PEAN-NHP is a constrained-input PEAN variant, not native-input official PEAN. See [PROTOCOL.md](docs/PROTOCOL.md).

Third-party attribution and license status are recorded in [THIRD_PARTY.md](docs/THIRD_PARTY.md). No new license grant is assigned to upstream code. The project owner has not selected a license for the project-authored code in this archive.

## Manuscript

The accompanying manuscript is entitled *Semantics-Guided Gated Fusion with Perturbation-Aware Training for Robust Encrypted Traffic Classification*. No publication DOI or acceptance status is asserted here.
