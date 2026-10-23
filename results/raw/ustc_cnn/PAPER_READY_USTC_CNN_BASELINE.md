# USTC-CNN final baseline under frozen V17 evaluation protocol

Architecture-faithful, protocol-adapted reproduction of USTC-CNN [31].

Important: training is CLEAN-ONLY and follows the original baseline optimizer/update budget. V17 robust training is NOT added to the baseline.

| Dataset | Clean Acc | Clean Macro-F1 | Clean ECE | Mixed@1 RobAcc | Mixed@1 RobF1 | Mixed ASR | Mixed RelDrop | Mixed ECE | Adaptive@1 RobAcc | Adaptive@1 RobF1 | Adaptive ASR | Adaptive RelDrop | Adaptive ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CipherSpectrum | 30.26 ± 0.60 | 29.93 ± 0.75 | 3.00 ± 0.26 | 9.05 ± 0.29 | 10.14 ± 0.34 | 76.17 ± 0.11 | 70.11 ± 0.39 | 3.29 ± 0.26 | 9.55 ± 0.42 | 10.87 ± 0.31 | 75.24 ± 1.12 | 68.43 ± 1.48 | 17.54 ± 5.94 |
| USTC-TFC2016 | 93.90 ± 0.07 | 94.33 ± 0.07 | 0.76 ± 0.10 | 28.63 ± 1.63 | 25.16 ± 1.98 | 70.09 ± 1.62 | 69.51 ± 1.74 | 53.02 ± 3.23 | 37.87 ± 0.76 | 42.11 ± 0.77 | 63.57 ± 0.82 | 59.67 ± 0.84 | 37.97 ± 1.02 |

## Protocol notes

- train seeds: [42, 43, 44]
- split seed: 42
- attack base seed: 20260919
- optimizer: SGD, lr=0.0001, batch=50
- source TRAIN_ROUND: 20000; actual updates: 20001
- checkpoint rule: minimum clean-validation mean CE; no test-driven selection
- route-prefix stripping: first 38 bytes of each packet
- IAT is physically perturbed but is representation-invisible to USTC-CNN
- Adaptive is deterministic semantics-preserving structural perturbation, not a gradient-based white-box attack
- prediction probabilities are saved for provenance-matched ROC regeneration

Architecture source: https://github.com/echowei/DeepTraffic/tree/master/1.malware_traffic_classification/4.TrainAndTest
