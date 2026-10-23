# USTC-CNN: archived results

Percentages; mean ± sample SD over seeds 42, 43, 44.

| Dataset | Condition | Accuracy | Macro-F1 | ASR | Relative drop | ECE | Macro OvR AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CipherSpectrum | clean | 30.26 ± 0.60 | 29.93 ± 0.75 | N/A | N/A | 3.00 ± 0.26 | 85.15 ± 0.18 |
| CipherSpectrum | mixed_1.0 | 9.05 ± 0.29 | 10.14 ± 0.34 | 76.17 ± 0.11 | 70.11 ± 0.39 | 3.29 ± 0.26 | 60.92 ± 0.20 |
| CipherSpectrum | adaptive_1.0 | 9.55 ± 0.42 | 10.87 ± 0.31 | 75.24 ± 1.12 | 68.43 ± 1.48 | 17.54 ± 5.94 | 59.54 ± 0.90 |
| USTC-TFC2016 | clean | 93.90 ± 0.07 | 94.33 ± 0.07 | N/A | N/A | 0.76 ± 0.10 | 99.57 ± 0.00 |
| USTC-TFC2016 | mixed_1.0 | 28.63 ± 1.63 | 25.16 ± 1.98 | 70.09 ± 1.62 | 69.51 ± 1.74 | 53.02 ± 3.23 | 80.50 ± 1.24 |
| USTC-TFC2016 | adaptive_1.0 | 37.87 ± 0.76 | 42.11 ± 0.77 | 63.57 ± 0.82 | 59.67 ± 0.84 | 37.97 ± 1.02 | 81.58 ± 0.92 |

See [RESULTS.md](RESULTS.md) and [PROTOCOL.md](../docs/PROTOCOL.md) for population, aggregation, and input adaptations.
