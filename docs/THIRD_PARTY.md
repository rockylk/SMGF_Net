# Baseline sources, adaptations, and license status

This file records locally archived provenance. It does not grant rights to third-party materials or assert a new upstream verification.

## USTC-CNN

- Architecture source: <https://github.com/echowei/DeepTraffic/tree/master/1.malware_traffic_classification/4.TrainAndTest>.
- The included runner is the project's PyTorch reproduction with frozen split, prefix-strip, clean-validation selection, and deterministic perturbation adaptations.
- No full upstream repository or upstream weight file is bundled.

## ET-BERT

- Upstream: <https://github.com/linwhitehat/ET-BERT>.
- Archived commit: `d59470624acd11405f9e88da650cf3bf65521e9a`.
- Included: the project's formal rerun orchestration and original run manifest.
- External requirements: official `fine-tuning/run_classifier.py`, `inference/run_classifier_infer.py`, preprocessing utilities, pretrained model, byte vocabulary, and base configuration. The manifest records their hashes.
- The later rerun timestamp and its attack seed are retained as valid experiment provenance.

Prepare the exact upstream checkout separately, for example:

```bash
git clone https://github.com/linwhitehat/ET-BERT.git external/ET-BERT
git -C external/ET-BERT checkout d59470624acd11405f9e88da650cf3bf65521e9a
```

Obtain the pretrained assets according to the upstream instructions and verify them against `results/raw/etbert/run_manifest.json`. Cloning alone does not provide the pretrained weights. The project adapter does not download weights or access any private server automatically.

## PEAN-NHP

- Upstream PEAN: <https://github.com/Lin-Dada/PEAN>.
- Archived commit: `b159ec48f8e3a699d863d57a6e2dc331aa83e7cc`.
- `src/baselines/pean_nhp/official_sources/PEAN/` preserves the eight already archived source/vocabulary files required by the runner's provenance gate.
- `PEAN_NHP_V17_model.py` and the data/training adapters implement the explicit No Header/Payload constraint. All byte slots contain `[CLS, SEP, PAD, ...]` independently of the flow and condition; sample-dependent information enters through packet lengths.
- The upstream mirror available in this workspace contains no LICENSE file. Its authors retain their rights. No MIT/Apache license is assigned to it here; the archive does not represent that missing license metadata as an open-source license grant.

## IBGC

- Paper: *Interaction matters: Encrypted traffic classification via status-based interactive behavior graph*. Applied Soft Computing155 (2024),111423. DOI: <https://doi.org/10.1016/j.asoc.2024.111423>.
- Upstream: <https://github.com/martinal-lee/ibgc>.
- Archived inspected commit: `72fd198aec6472eb4111e84b3490e721ac69c6df`.
- Included: the project's mathematical reproduction, frozen-cache adapter, comparison tests, checksum manifest, and optional source fetcher. The full upstream repository and upstream pretrained weights are not included.
- The preserved [IBGC provenance record](../src/baselines/ibgc/PROVENANCE.md) details graph equations, short-flow retention, serialized size adaptation, training choices, and source-license status.

## Project code

The project owner has not specified a distribution license for the SMGF-Net code and new packaging scripts. No license is fabricated in this release. Third-party notices and provenance remain separate from any future owner-selected license.
