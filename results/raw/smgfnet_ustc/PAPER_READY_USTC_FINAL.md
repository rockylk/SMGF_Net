# USTC-TFC2016 final paper-ready values

These values are independently re-aggregated from the formal full-data training
seeds 42, 43 and 44. All reported dispersions are **sample standard deviations**
across the three training seeds (ddof=1).

## Experimental subset and split

- Dataset: USTC-TFC2016 prepared experimental subset
- Classes: 12
- Flows: 21,256
- Train / validation / test: 15,304 / 1,700 / 4,252
- Split seed: 42
- Training seeds: 42, 43, 44
- Dataset manifest: `612c69a851dad04658619fd6c79ef60f34d08dd7dd1a0e14262fa01428cb2190`
- Split manifest SHA-256: `a34754daa9e202f154484a5b63a8840eb829e56aeaaff4549bac6cb711127f3b`

## Main paper values

### Clean classification row

Use this row to replace the old SMGF-Net USTC clean result:

| Method | Accuracy (%) | Macro-F1 (%) |
| --- | ---: | ---: |
| SMGF-Net (Ours) | **93.69 ± 0.54** | **93.89 ± 0.61** |

Clean ECE, if reported: **1.23 ± 0.17%**.

### Mixed attack at strength = 1.0

| Method | Rob_Acc (%) | Rob_F1 (%) | ASR (%) | Rel_Drop (%) | ECE (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| SMGF-Net (Ours) | **86.99 ± 2.71** | **87.09 ± 2.42** | **7.73 ± 2.88** | **7.15 ± 2.52** | **2.86 ± 0.56** |

### Adaptive semantics-preserving attack at strength = 1.0

| Method | Rob_Acc (%) | Rob_F1 (%) | ASR (%) | Rel_Drop (%) | ECE (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| SMGF-Net (Ours) | **89.28 ± 2.02** | **89.45 ± 1.72** | **5.07 ± 2.04** | **4.70 ± 1.94** | **2.06 ± 0.21** |

## Exact aggregate values before rounding

| Metric | Mean | Sample SD |
| --- | ---: | ---: |
| Clean Accuracy | 93.6892442772 | 0.5416031812 |
| Clean Macro-F1 | 93.8860147677 | 0.6128848413 |
| Clean ECE | 1.2276319075 | 0.1746739951 |
| Mixed@1 RobAcc | 86.9943555974 | 2.7058363613 |
| Mixed@1 RobF1 | 87.0876728746 | 2.4158518427 |
| Mixed@1 ASR | 7.7307854065 | 2.8780071148 |
| Mixed@1 RelDrop | 7.1522103244 | 2.5155865879 |
| Mixed@1 ECE | 2.8611101256 | 0.5587631714 |
| Adaptive@1 RobAcc | 89.2834744434 | 2.0237167444 |
| Adaptive@1 RobF1 | 89.4488084955 | 1.7154486865 |
| Adaptive@1 ASR | 5.0734884897 | 2.0365605197 |
| Adaptive@1 RelDrop | 4.7047183725 | 1.9380176039 |
| Adaptive@1 ECE | 2.0601280421 | 0.2101084994 |

## Per-seed audit values

| Seed | Best epoch | Clean Acc | Clean F1 | Mixed RobAcc | Mixed RobF1 | Adaptive RobAcc | Adaptive RobF1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 62 | 94.2145 | 94.4805 | 87.8645 | 87.7161 | 89.2051 | 89.2294 |
| 43 | 13 | 93.1326 | 93.2563 | 83.9605 | 84.4197 | 87.3001 | 87.8536 |
| 44 | 71 | 93.7206 | 93.9213 | 89.1580 | 89.1272 | 91.3452 | 91.2634 |

## Manuscript replacement map

1. **Clean-classification table:** replace the old USTC SMGF-Net
   `93.46 ± 0.22 / 93.66 ± 0.18` with
   `93.69 ± 0.54 / 93.89 ± 0.61`.

2. **USTC Mixed robustness table:** replace the old SMGF-Net row with the new
   strength-1.0 three-seed row:
   `86.99 ± 2.71 / 87.09 ± 2.42 / 7.73 ± 2.88 / 7.15 ± 2.52 / 2.86 ± 0.56`.

3. **USTC Adaptive robustness table:** replace the old SMGF-Net row with:
   `89.28 ± 2.02 / 89.45 ± 1.72 / 5.07 ± 2.04 / 4.70 ± 1.94 / 2.06 ± 0.21`.

## Mandatory manuscript consistency corrections

- The experiments represented here use the **prepared 12-class / 21,256-flow
  subset**, not all 20 classes / 215,256 flows of the original USTC-TFC2016
  corpus. The dataset description must distinguish the source corpus from the
  actual experimental subset.
- Do not use the old manuscript's Mixed value 55.71 as a matched strength-1.0
  comparator. The archived old V14 log recorded that value at Mixed strength
  **0.9**, while the manuscript labeled it as 1.0.
- Do not mix the 10% pilot values with these formal full-data values.
- These three-seed values are now the canonical USTC main-model results.
