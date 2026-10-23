# Experimental protocol

## Populations and split

| Dataset | Classes | Total flows | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: | ---: |
| CipherSpectrum | 41 | 41,000 | 29,520 | 3,280 | 8,200 |
| USTC-TFC2016 archived subset | 12 | 21,256 | 15,304 | 1,700 | 4,252 |

The split is stratified with split seed 42. Training seeds are 42, 43, and 44. Split manifests contain the exact sample indices, relative PCAP paths, labels, and class mapping. Train-only scalers are fitted by the original SMGF loaders. Test data are not used to select SMGF checkpoints.

| Artifact | SHA-256 |
| --- | --- |
| `data/splits/cipher.json` | `01e3b572e8514a238b3733e224161acd1112b79b56f038635be9339b23cfad6b` |
| `data/splits/ustc.json` | `a34754daa9e202f154484a5b63a8840eb829e56aeaaff4549bac6cb711127f3b` |
| CipherSpectrum dataset enumeration | `821bee04771f0c8dce3680a4301164d8ab48e03ec0d1ed57c3930015ad200a61` |
| USTC dataset enumeration | `612c69a851dad04658619fd6c79ef60f34d08dd7dd1a0e14262fa01428cb2190` |

The USTC population is not the full original 20-class corpus. Its selected class counts are preserved in the original USTC run manifest.

## Final SMGF-Net configuration

- Learned context-conditioned channel-wise Gate; graph trajectory representation, flow statistics, initial signed-size fingerprint, and payload-entropy features.
- Single-stage loss: `clean CE + 1.0 * attacked-label CE` on robust batches; clean CE on the remaining batches.
- KL weight 0. No contrastive or representation-pretraining stage in the final protocol.
- 100 epochs; batch size 64; AdamW; weight decay `1e-4`.
- Learning rate `0.002` for epochs 1–70 and `0.0005` for epochs 71–100.
- Robust batch ratio 0.20; training attacks are Mixed only, with strength sampled uniformly from `[0.3, 0.9]`.
- No early stopping; checkpoint selected by minimum clean-validation CE.
- CipherSpectrum seed42 was selected in an earlier ratio experiment, then frozen and reused. The formal three-seed runner trains seeds43/44. This lineage is retained in the archived run configuration.

The V16 core and V17 base have historical descriptions and default objectives from earlier stages. They remain dependencies because the final runners import their validated extraction, graph, model, loss, and attack functions and override the relevant final configuration. Running the old base `main()` directly is not equivalent to running the final frozen experiments.

## Baseline-specific adaptations

| Baseline | Input and training boundary | Checkpoint criterion |
| --- | --- | --- |
| USTC-CNN | Two-layer byte-image CNN; 784 bytes after a 38-byte packet prefix strip; clean SGD, lr `1e-4`, batch 50, 20,001 updates | Minimum clean-validation mean CE |
| ET-BERT | Official pinned upstream implementation and pretrained model; first-five-packet byte-token representation; fine-tuning 10 epochs, batch 32, lr `2e-5`, sequence length 128 | Official best validation accuracy |
| PEAN-NHP | PEAN dimensions and training objective retained, constant traffic-independent byte tensor, first-ten packet lengths carry sample-dependent input; 100 clean epochs, Adam lr `1e-3`, weight decay `1e-3`, gamma `0.992`, batch 64 | Minimum independent validation fused CE |
| IBGC | Authors' graph-feature equations and GCN architecture reproduced on the frozen direction/size/IAT cache; IAT is not an IBGC model feature; 100 clean epochs, Adam lr `1e-3`, weight decay 0, batch 64 | Minimum clean-validation CE, earliest tie |

PEAN-NHP must not be described as native-input official PEAN. IBGC uses serialized packet sizes, retains nonempty short and unidirectional flows, and does not claim exact upstream preprocessing/training identity. Details are in its preserved `PROVENANCE.md`.

## Attack families and archived differences

- Padding, IAT jitter, and dummy insertion are combined in Mixed. Adaptive is the documented anchor-preserving structural transformation, not a claim of gradient-based white-box optimization.
- Primary reporting uses Mixed@1.0 and Adaptive@1.0; clean is a reference condition.
- SMGF and common structural-baseline attack base seed: `20260919`.
- **Accepted official ET-BERT later rerun attack base seed: `20260923`.** This is a documented protocol difference, not a damaged or invalid archive.
- ET-BERT and byte-image inputs require packet/byte-level translation of the structural attack family. Equal physical perturbation families do not imply bitwise-identical internal feature tensors across models.
- Pure IAT changes are not directly visible to byte-only USTC-CNN or to the packet-length-only traffic-dependent input of PEAN-NHP. IBGC does not add IAT as a feature or edge criterion.
- The frozen archive and ROC analysis exclude folders whose names contain `云端`.

## Metrics and replication

- Accuracy, Macro-F1, ASR, relative drop, ECE, and AUROC are reported in percent.
- ASR is the fraction of samples correctly classified on clean input that become misclassified after attack.
- Relative drop is `(clean accuracy - robust accuracy) / clean accuracy * 100`, computed for each seed before aggregation.
- ECE uses 15 bins and is evaluation-only.
- Macro OvR AUROC is computed class by class within each seed, with equal class weights, then summarized across the three seed-level scores.
- Sample SD uses `ddof=1`. No probability averaging across models and no pooling of their test predictions is used.
- ROC display curves use a common 4,001-point FPR grid. Interpolated display areas are separate from the original seed-level AUCs used in legends. Bands represent SD across trained models.

## Matched ablation and Gate audit

The matched ablation is a 2×2 design on CipherSpectrum using training seed42 throughout. The Full checkpoint is reused; the other three conditions are trained under matched optimization and initialization-seed conditions. Identity gating leaves the fusion architecture in place. Removing robust training bypasses attacked-sample generation/forwarding but still trains with clean CE. Interaction values are fixed-seed diagnostic effects, not three-seed statistical inference.

The Gate audit uses three formal frozen checkpoints and 1,000 deterministically selected class-balanced test flows. Signed and absolute shifts describe bidirectional channel reweighting. The magnitude of a Gate shift is not established as the cause of its classification benefit. Historical Figure10 filenames correspond to the Gate audit called Fig. 7 in the current manuscript.

## GPU overhead

Table VIII records model-forward latency on CipherSpectrum at batch size1, eight held-out flows per seed, 30 warm-up iterations, and 100 timed passes per flow. Per-flow observations are averaged within a seed; the table then summarizes the three models. FLOPs are the supported-operator subtotal, with original status `MEASURED_FLOPS_UNVERIFIED` retained. Neither forward latency nor that subtotal demonstrates end-to-end preprocessing cost or line-rate throughput.

The manuscript specifies an Intel Core i5-14600KF CPU (3.50 GHz) and NVIDIA GeForce RTX 5070 Ti GPU (16 GB). Some baseline manifests record Python 3.10.20, PyTorch `2.12.0+cu132`, and NumPy `2.2.5`. These historical records do not constitute a complete verified lockfile for every run. Packaging does not rerun or newly attest GPU measurements.
