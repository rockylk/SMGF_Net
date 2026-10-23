# Official ET-BERT final-paper baseline rerun

Three independent fine-tuning seeds: 42, 43, 44.

The clean train/validation/test split is exactly the frozen SMGF-Net split for each dataset.

## Primary paper-facing results

| Dataset | Clean Acc | Clean F1 | Clean ECE | Mixed RobAcc | Mixed RobF1 | Mixed ASR | Mixed RelDrop | Mixed ECE | Adaptive RobAcc | Adaptive RobF1 | Adaptive ASR | Adaptive RelDrop | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CipherSpectrum | 33.58 ± 1.43 | 26.59 ± 1.05 | 3.46 ± 0.72 | 22.94 ± 3.95 | 18.94 ± 2.87 | 35.44 ± 9.65 | 31.89 ± 9.46 | 4.13 ± 1.15 | 25.20 ± 3.67 | 20.68 ± 2.52 | 28.29 ± 8.94 | 25.15 ± 8.40 | 4.29 ± 1.03 |
| USTC-TFC2016 | 86.34 ± 0.69 | 87.01 ± 0.57 | 1.53 ± 0.40 | 75.10 ± 2.32 | 76.74 ± 1.92 | 14.12 ± 3.74 | 13.00 ± 3.38 | 10.81 ± 3.56 | 63.88 ± 3.78 | 67.62 ± 2.53 | 27.02 ± 4.33 | 26.00 ± 4.97 | 22.86 ± 6.65 |

## Interpretation boundary

SMGF-Net's original perturbation engine acts on extracted feature tensors. ET-BERT consumes raw bytes, so the runner translates the same packet-length, timing, and dummy-injection perturbation family to packet sequences before official ET-BERT preprocessing. The Adaptive variant preserves genuine packet payload bytes and uses zero filler for added padding/dummies. Therefore these values are a faithful architecture-level physical-perturbation comparison, not a claim that the internal tensors are identical across models.

Probability TSV files are retained for later ROC/Fig.5 regeneration.
