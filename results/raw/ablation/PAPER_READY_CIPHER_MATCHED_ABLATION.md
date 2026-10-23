# CipherSpectrum V17 Matched Ablation — Seed42

This is a fixed-seed component-diagnostic experiment, not a replacement for the formal three-seed main-model results.

Main results use seeds 42/43/44. All ablation rows here use training seed42 so that component effects are compared under matched stochastic conditions.

## Matched protocol

- Dataset: CipherSpectrum, 41 classes / 41,000 samples.
- Split seed: 42.
- Train/Validation/Test: 29,520 / 3,280 / 8,200.
- Training seed: 42 for every row.
- 100 epochs; no early stopping.
- LR: 0.002 for epochs 1-70; 0.0005 for 71-100.
- AdamW weight decay: 1e-4.
- Checkpoint: minimum clean-validation CE.
- Robust condition: 20% Mixed attacked batches, strength Uniform[0.3,0.9], attacked-label CE weight=1.
- Clean-only condition: no attacked samples are generated or forwarded during training.
- KL weight=0 in every condition.
- One shared deterministic Mixed@1.0 and Adaptive@1.0 test cache is used by all rows.

## Paper-facing compact ablation table

| Variant | Gate | Robust train | Clean Acc | Clean F1 | Mixed RobAcc | Mixed RobF1 | Adaptive RobAcc | Adaptive RobF1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FULL | Learned | Yes | 91.01 | 90.81 | 79.45 | 79.33 | 87.39 | 87.33 |
| STRICT_NO_GATE | Identity | Yes | 91.04 | 90.76 | 77.01 | 76.64 | 83.70 | 83.57 |
| CLEAN_ONLY | Learned | No | 92.23 | 92.03 | 41.27 | 40.79 | 83.70 | 83.45 |
| NO_GATE_CLEAN_ONLY | Identity | No | 90.52 | 90.36 | 37.22 | 36.36 | 74.40 | 74.18 |

## Factorial diagnostic effects

Effects below are percentage-point differences at fixed seed42; they are diagnostic contrasts, not uncertainty estimates.

- Gate effect under robust training: CleanAcc -0.02 pp; MixedRobAcc +2.44 pp; AdaptiveRobAcc +3.70 pp.
- Gate effect without robust training: CleanAcc +1.71 pp; MixedRobAcc +4.05 pp; AdaptiveRobAcc +9.29 pp.
- Robust-training effect with learned Gate: CleanAcc -1.22 pp; MixedRobAcc +38.18 pp; AdaptiveRobAcc +3.70 pp.
- Robust-training effect without Gate: CleanAcc +0.51 pp; MixedRobAcc +39.79 pp; AdaptiveRobAcc +9.29 pp.
- Gate × robust-training interaction: CleanAcc -1.73 pp; MixedRobAcc -1.61 pp; AdaptiveRobAcc -5.60 pp.

## Reporting note

Recommended manuscript statement:

Main performance is reported over three training seeds (42, 43, and 44). To isolate architectural and training-component effects under matched stochastic conditions, all ablation experiments use the fixed training seed 42 while retaining the identical data split, optimization schedule, checkpoint criterion, and deterministic attack realizations.

If a component contrast is smaller than the seed-to-seed dispersion already observed for the FULL main model, avoid claiming that the difference is stable; selectively add seeds43/44 for that ambiguous contrast if needed.
