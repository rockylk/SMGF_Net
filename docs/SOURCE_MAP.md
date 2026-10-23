# Source map and archive selection

The package is arranged by model and experimental task, while checksum-locked source bodies retain their original filenames and bytes.

| Location | Responsibility |
| --- | --- |
| `scripts/run_experiment.py` | New portable dataset/output path adapter and native-runner dispatch |
| `src/smgfnet/frozen/V16-SMGF-Net_final_paper_experiments.py` | Validated PCAP extraction, graph construction, feature preprocessing, base encoders, attack/reconstruction utilities |
| `src/smgfnet/frozen/V17-SMGF-Net_final_paper_experiments.py` | Final model class `SMGFNetV17`, gating and shared evaluation/training helpers; historical standalone defaults must not be used as final settings |
| `V17_CipherSpectrum_final_frozen_3seed.py` | Frozen CipherSpectrum training, seed42 reuse, canonical attack evaluation and aggregation |
| `V17_USTC_frozen_runner.py` | Frozen full/pilot USTC parser/cache, training and evaluation; portable adapter explicitly selects full data |
| `V17_USTC_final_seed43_44.py` | Original USTC continuation/aggregation orchestration retained for provenance |
| `V17_CipherSpectrum_MATCHED_ABLATION_SEED42.py` | Final matched 2×2 Gate/training ablation |
| `V17_CipherSpectrum_Figure10_Gate_Audit.py` | Frozen-checkpoint channel response audit |
| `src/baselines/common/` | Exact shared structural protocol and metrics |
| `src/baselines/ustc_cnn/` | Formal byte-image CNN baseline |
| `src/baselines/etbert/` | Accepted official later-rerun orchestration |
| `src/baselines/pean_nhp/` | NHP data constraint, model, training and required pinned PEAN files |
| `src/baselines/ibgc/` | Graph equations/model, immutable raw-cache input adapter, training/evaluation, provenance tests |
| `analysis/roc/` | Original ROC calculation, seed/class AUC records, probability export and redraw code |
| `analysis/sensitivity/` | Frozen-model strength sweep evaluation and original figure source |
| `analysis/overhead/` | Final server profiling workflow with original fallback vendor sources |
| `scripts/summarize_results.py` | Portable result normalization, independent summary checks, and Markdown/CSV generation |

## Final archive lineage

- SMGF CipherSpectrum: `SMGF_V17_CIPHERSPECTRUM_FINAL_3SEED_20260923_135542`.
- SMGF USTC aggregate: `SMGF_V17_USTC_FINAL_3SEED_20260924_144614`, with the three corresponding full-data seed runs.
- USTC-CNN: `USTC_CNN_FINAL_PAPER_BASELINE_V1`.
- ET-BERT: `ETBERT_FINAL_PAPER_BASELINE_V1`.
- PEAN-NHP: `PEAN_NHP_FINAL_PAPER_BASELINE_V1_100EP`.
- IBGC: `IBGC_V17_V1_100EP`.
- Matched ablation: `SMGF_V17_CIPHER_MATCHED_ABLATION_SEED42_20260924_170742`.
- AUROC/ROC: the completed `Fig5_macro_OvR` analysis with 90 valid outputs and 30 complete three-seed groups.
- Strength sweep: the verified 45-observation supplement from `FIG6_CIPHER_FROZEN_GPU_SWEEP_Home`, plus formal main-run endpoints and SMGF measurements.
- Overhead table: adopted `BaseLine/TableVIII_V17_status.csv` and final `Section56_SERVER_ONECLICK_20261009` toolchain.

## Deliberately excluded history

V14/V15 standalone experiments, superseded V16 explorations, KL/robust-ratio searches, native-input PEAN results, FS-Net/DFE/RF/ODE-Flow candidates, proxy baselines, cloud-labelled duplicates, manuscript editing utilities, SSH helpers, and raw handoff transcripts are not part of the five-model final paper release. V16 core is retained only because the frozen final model imports it. Internal source comments do not override the final run configurations.

The per-file original relative path, checksum, byte count, and category are recorded in `manifests/source_manifest.json`. New wrapper/documentation files are separate and are not claimed to be original experiment sources. `.gitattributes` disables newline conversion for byte-sensitive archived files.
