# Experimental results

Generated from the archived final per-seed observations. All performance metrics are percentages unless noted. Values are mean ± sample SD over independently trained models (seeds 42, 43, 44), with ddof=1. No new training or GPU measurements are performed by this script.

## Evaluation population

| Dataset | Classes | Train | Validation | Test |
| --- | --- | --- | --- | --- |
| CipherSpectrum | 41 | 29520 | 3280 | 8200 |
| USTC-TFC2016 archived subset | 12 | 15304 | 1700 | 4252 |

USTC refers to the archived 12-class, 21,256-flow subset, not the complete original corpus.

## CipherSpectrum

### Clean

| Model | Accuracy (%) | Macro-F1 (%) | ECE (%) |
| --- | --- | --- | --- |
| USTC-CNN | 30.26 ± 0.60 | 29.93 ± 0.75 | 3.00 ± 0.26 |
| ET-BERT | 33.58 ± 1.43 | 26.59 ± 1.05 | 3.46 ± 0.72 |
| PEAN-NHP | 86.88 ± 0.15 | 86.91 ± 0.15 | 1.18 ± 0.24 |
| IBGC | 82.17 ± 0.80 | 81.98 ± 0.97 | 1.54 ± 0.51 |
| SMGF-Net | 91.30 ± 0.25 | 90.96 ± 0.14 | 1.19 ± 0.13 |

### Mixed@1.0

| Model | Robust accuracy (%) | Robust Macro-F1 (%) | ASR (%) | Relative drop (%) | ECE (%) |
| --- | --- | --- | --- | --- | --- |
| USTC-CNN | 9.05 ± 0.29 | 10.14 ± 0.34 | 76.17 ± 0.11 | 70.11 ± 0.39 | 3.29 ± 0.26 |
| ET-BERT | 22.94 ± 3.95 | 18.94 ± 2.87 | 35.44 ± 9.65 | 31.89 ± 9.46 | 4.13 ± 1.15 |
| PEAN-NHP | 28.51 ± 2.07 | 29.68 ± 1.38 | 68.43 ± 2.65 | 67.18 ± 2.44 | 34.67 ± 2.30 |
| IBGC | 17.62 ± 2.03 | 16.31 ± 2.25 | 80.49 ± 2.36 | 78.57 ± 2.27 | 62.08 ± 2.07 |
| SMGF-Net | 78.92 ± 0.46 | 78.64 ± 0.60 | 15.21 ± 0.40 | 13.56 ± 0.74 | 8.15 ± 0.54 |

### Adaptive@1.0

| Model | Robust accuracy (%) | Robust Macro-F1 (%) | ASR (%) | Relative drop (%) | ECE (%) |
| --- | --- | --- | --- | --- | --- |
| USTC-CNN | 9.55 ± 0.42 | 10.87 ± 0.31 | 75.24 ± 1.12 | 68.43 ± 1.48 | 17.54 ± 5.94 |
| ET-BERT | 25.20 ± 3.67 | 20.68 ± 2.52 | 28.29 ± 8.94 | 25.15 ± 8.40 | 4.29 ± 1.03 |
| PEAN-NHP | 28.88 ± 2.01 | 30.02 ± 1.41 | 68.03 ± 2.38 | 66.75 ± 2.37 | 34.44 ± 2.46 |
| IBGC | 17.37 ± 2.19 | 16.17 ± 2.36 | 80.65 ± 2.26 | 78.87 ± 2.47 | 62.39 ± 1.87 |
| SMGF-Net | 86.88 ± 0.55 | 86.57 ± 0.72 | 6.32 ± 0.58 | 4.84 ± 0.82 | 3.62 ± 0.59 |

## USTC-TFC2016

### Clean

| Model | Accuracy (%) | Macro-F1 (%) | ECE (%) |
| --- | --- | --- | --- |
| USTC-CNN | 93.90 ± 0.07 | 94.33 ± 0.07 | 0.76 ± 0.10 |
| ET-BERT | 86.34 ± 0.69 | 87.01 ± 0.57 | 1.53 ± 0.40 |
| PEAN-NHP | 93.68 ± 0.04 | 94.06 ± 0.04 | 0.70 ± 0.14 |
| IBGC | 92.04 ± 0.13 | 92.33 ± 0.12 | 2.21 ± 0.54 |
| SMGF-Net | 93.69 ± 0.54 | 93.89 ± 0.61 | 1.23 ± 0.17 |

### Mixed@1.0

| Model | Robust accuracy (%) | Robust Macro-F1 (%) | ASR (%) | Relative drop (%) | ECE (%) |
| --- | --- | --- | --- | --- | --- |
| USTC-CNN | 28.63 ± 1.63 | 25.16 ± 1.98 | 70.09 ± 1.62 | 69.51 ± 1.74 | 53.02 ± 3.23 |
| ET-BERT | 75.10 ± 2.32 | 76.74 ± 1.92 | 14.12 ± 3.74 | 13.00 ± 3.38 | 10.81 ± 3.56 |
| PEAN-NHP | 69.02 ± 1.50 | 70.39 ± 1.66 | 26.99 ± 1.64 | 26.33 ± 1.58 | 16.91 ± 1.26 |
| IBGC | 26.03 ± 1.14 | 21.80 ± 1.21 | 72.36 ± 1.34 | 71.71 ± 1.25 | 63.37 ± 1.23 |
| SMGF-Net | 86.99 ± 2.71 | 87.09 ± 2.42 | 7.73 ± 2.88 | 7.15 ± 2.52 | 2.86 ± 0.56 |

### Adaptive@1.0

| Model | Robust accuracy (%) | Robust Macro-F1 (%) | ASR (%) | Relative drop (%) | ECE (%) |
| --- | --- | --- | --- | --- | --- |
| USTC-CNN | 37.87 ± 0.76 | 42.11 ± 0.77 | 63.57 ± 0.82 | 59.67 ± 0.84 | 37.97 ± 1.02 |
| ET-BERT | 63.88 ± 3.78 | 67.62 ± 2.53 | 27.02 ± 4.33 | 26.00 ± 4.97 | 22.86 ± 6.65 |
| PEAN-NHP | 69.12 ± 1.84 | 70.62 ± 1.95 | 27.02 ± 2.13 | 26.22 ± 1.93 | 16.50 ± 1.88 |
| IBGC | 26.62 ± 1.43 | 21.99 ± 1.93 | 71.79 ± 1.67 | 71.08 ± 1.57 | 62.49 ± 1.71 |
| SMGF-Net | 89.28 ± 2.02 | 89.45 ± 1.72 | 5.07 ± 2.04 | 4.70 ± 1.94 | 2.06 ± 0.21 |

## Macro one-versus-rest AUROC

| Dataset | Model | Clean (%) | Mixed@1.0 (%) | Adaptive@1.0 (%) |
| --- | --- | --- | --- | --- |
| CipherSpectrum | USTC-CNN | 85.15 ± 0.18 | 60.92 ± 0.20 | 59.54 ± 0.90 |
| CipherSpectrum | ET-BERT | 80.76 ± 0.35 | 73.06 ± 3.38 | 74.70 ± 2.57 |
| CipherSpectrum | PEAN-NHP | 99.67 ± 0.01 | 83.78 ± 1.36 | 83.77 ± 1.12 |
| CipherSpectrum | IBGC | 99.42 ± 0.07 | 77.32 ± 2.24 | 77.38 ± 2.06 |
| CipherSpectrum | SMGF-Net | 99.85 ± 0.01 | 99.34 ± 0.03 | 99.74 ± 0.03 |
| USTC-TFC2016 | USTC-CNN | 99.57 ± 0.00 | 80.50 ± 1.24 | 81.58 ± 0.92 |
| USTC-TFC2016 | ET-BERT | 99.02 ± 0.04 | 97.45 ± 0.60 | 95.69 ± 1.17 |
| USTC-TFC2016 | PEAN-NHP | 99.53 ± 0.01 | 96.66 ± 0.43 | 96.61 ± 0.49 |
| USTC-TFC2016 | IBGC | 99.43 ± 0.06 | 74.93 ± 1.40 | 75.37 ± 1.61 |
| USTC-TFC2016 | SMGF-Net | 99.56 ± 0.03 | 99.20 ± 0.10 | 99.32 ± 0.04 |

For each seed, compute an OvR AUC for every class and average all classes equally; then summarize the three seed-level AUCs. Seeds are not pooled, probabilities are not averaged between models, and displayed interpolated-curve areas do not replace original AUROC. ROC bands show cross-model sample SD, not 95% confidence intervals.

## Matched component ablation (Table VII)

| Setting | Clean Acc (%) | Mixed RobAcc (%) | Mixed RelDrop (%) | Adaptive RobAcc (%) | Adaptive RelDrop (%) |
| --- | --- | --- | --- | --- | --- |
| Full model | 91.01 | 79.45 | 12.70 | 87.39 | 3.98 |
| w/o Learned Gating | 91.04 | 77.01 | 15.41 | 83.70 | 8.06 |
| w/o Robust Training | 92.23 | 41.27 | 55.26 | 83.70 | 9.26 |
| w/o Learned Gating & Robust Training | 90.52 | 37.22 | 58.88 | 74.40 | 17.81 |

CipherSpectrum; fixed training seed 42; identical split, initialization seed, optimization schedule, clean-validation checkpoint criterion, and deterministic attacks. Full reuses the frozen main-experiment seed-42 checkpoint. These are single-seed diagnostic comparisons, not three-model means.

| Contrast (percentage points) | Mixed | Adaptive |
| --- | --- | --- |
| Gate effect with robust training | 2.44 | 3.70 |
| Gate effect without robust training | 4.05 | 9.29 |
| Training effect with learned Gate | 38.18 | 3.70 |
| Training effect with identity Gate | 39.79 | 9.29 |
| Gate × training interaction | -1.61 | -5.60 |

Interactions are subadditive in this fixed-seed comparison; they are not evidence of statistical significance or synergy.

## Computational overhead (Table VIII)

| Model | Parameters (M) | Profiled FLOPs subtotal (M) | GPU forward latency (ms/flow) |
| --- | --- | --- | --- |
| USTC-CNN | 3.306 | 27.831 | 0.275 ± 0.002 |
| ET-BERT | 132.157 | 22,370.062 | 10.772 ± 0.333 |
| PEAN-NHP | 1.774 | 13.470 | 1.343 ± 0.003 |
| IBGC | 0.015 | 1.765 | 1.760 ± 0.011 |
| SMGF-Net | 0.168 | 9.138 | 2.361 ± 0.043 |

CipherSpectrum, batch size 1, eight held-out flows per seed, 30 warm-up iterations and 100 timed forward passes per flow. Latencies are averaged within each seed and summarized across seeds. These are archived model-forward measurements, excluding PCAP parsing, preprocessing, and transfer overhead. FLOPs are the profiler-supported operator subtotal, not a complete model-wide count; the original measurement status remains MEASURED_FLOPS_UNVERIFIED. The source CSV lost the ± character as “?”; only its display is restored here.

## Channel-level Gate audit

| Attack | Strength | Mean absolute paired Gate shift ± sample SD |
| --- | --- | --- |
| mixed | 0.3 | 0.02695 ± 0.00057 |
| mixed | 0.6 | 0.04223 ± 0.00054 |
| mixed | 0.9 | 0.06146 ± 0.00179 |
| mixed | 1.0 | 0.07012 ± 0.00320 |
| adaptive | 0.4 | 0.01405 ± 0.00033 |
| adaptive | 0.7 | 0.02091 ± 0.00033 |
| adaptive | 1.0 | 0.02925 ± 0.00119 |

The audit uses a deterministic class-balanced subset of 1,000 CipherSpectrum test flows and the three frozen models. Channel shifts are bidirectional. Their magnitude describes modulation, not a causal measure of classification benefit. Historical archive filenames call this Figure10; the current manuscript uses Fig. 7.

## Strength sensitivity

See [SENSITIVITY.md](SENSITIVITY.md) for all measured Mixed and Adaptive strength points for SMGF-Net, ET-BERT, PEAN-NHP, and IBGC on CipherSpectrum. Intermediate baseline values are the 45 verified GPU observations; no missing values are interpolated.

## Reproduction and source records

- [Normalized per-seed observations](tables/per_seed_metrics.csv) and [aggregate metrics](tables/aggregate_metrics.csv).
- [Original result snapshots](raw/) and [source checksum manifest](../manifests/source_manifest.json).
- [Protocol and model adaptations](../docs/PROTOCOL.md).
- [Reproduction instructions](../docs/REPRODUCING.md).
