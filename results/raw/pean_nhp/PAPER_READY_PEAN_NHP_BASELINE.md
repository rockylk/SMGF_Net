# PEAN-NHP V1 Final Baseline

Three seeds; values are mean ± sample SD. Clean and primary attacks only.

| Dataset | Clean Acc | Clean Macro-F1 | Mixed@1 RobAcc | Mixed ASR | Adaptive@1 RobAcc | Adaptive ASR |
|---|---:|---:|---:|---:|---:|---:|
| CipherSpectrum | 86.88 ± 0.15 | 86.91 ± 0.15 | 28.51 ± 2.07 | 68.43 ± 2.65 | 28.88 ± 2.01 | 68.03 ± 2.38 |
| USTC-TFC2016 | 93.68 ± 0.04 | 94.06 ± 0.04 | 69.02 ± 1.50 | 26.99 ± 1.64 | 69.12 ± 1.84 | 27.02 ± 2.13 |

PEAN-NHP retains PEAN network dimensions and training settings, but the byte branch is traffic-independent [CLS,SEP,PAD...] and exposes no TCP/UDP header or payload bytes.
Checkpoint selected only on frozen validation fused CE.
Mixed/Adaptive modify the physical packet sequence; only the length view can change.
