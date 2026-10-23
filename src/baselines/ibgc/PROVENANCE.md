# Provenance and methodological boundaries

## Primary sources

- Paper: Yao Li et al., *Interaction matters: Encrypted traffic classification via status-based interactive behavior graph*, Applied Soft Computing 155 (2024), 111423. DOI: https://doi.org/10.1016/j.asoc.2024.111423
- Authors' repository: https://github.com/martinal-lee/ibgc
- Inspected commit: **72fd198aec6472eb4111e84b3490e721ac69c6df** (retrieved 2026-10-01).
- `02_classification/GraphNet.py`: GCN and GraphSAGE models; this bundle chooses GCN.
- `02_classification/NewGraphDataset.py`: `GraphMitDataset.node_features` and `generate_new_graph`.
- `01_feature_extract/mit/NfstreamPlugin.py`: upstream direction and `packet.ip_size` extraction.
- `01_feature_extract/mit/BehaviorExtractor.py`: source-specific PCAP filtering; deliberately not used here.

Full SHA-256 values for inspected files are in `official_source_manifest.json`.
No LICENSE file appeared in the pinned repository tree. The bundle does not redistribute the full upstream
source or pretrained `.pt` files; it includes a mathematical reimplementation and a checksum-pinned
optional source fetcher. No claim of an upstream open-source license grant is made.

## Exact model and graph correspondence

Each packet is a node. A slice is a maximal consecutive run of one direction.
Within a slice, consecutive nodes are connected in both directions. Across each pair of adjacent slices,
every packet in one slice connects to every packet in the other, in both directions.
No edges connect nonadjacent slices. Graph construction adds no explicit self-loops;
PyG GCNConv adds its normal self-loops and uses symmetric GCN normalization.
There is no node padding, dense adjacency allocation, edge attention, or IAT threshold.

For a packet with size s in a slice with count n and total bytes B, adjacent slice counts n_prev/n_next
and totals B_prev/B_next, the feature order is:

1. signed size: −s for direction 0, +s for direction 1;
2. n;
3. n_prev;
4. n_next;
5. n_prev/n;
6. n_next/n;
7. B;
8. B_prev;
9. B_next;
10. s/B;
11. B/n;
12. B/B_prev, or 0 if no previous slice;
13. B/B_next, or 0 if no next slice.

The GCN dimensions, activations, mean readout, dropout 0.025, linear layer and log-softmax match
the authors' `GraphNet`. No edge attributes are consumed by that model.
`test_ibgc.py` compares both features and sorted edges against the pinned source on variable-length
inputs, and compares model outputs after loading identical weights.

## Deliberate V17 adaptations (must be stated in a paper)

1. **Frozen sample population.** Upstream processing filters length<5, filters unidirectional flows
   in most label branches, and truncates to caller-provided `node_counts` (no unique run setting supplied).
   We retain every nonempty frozen sample, including length 1 and unidirectional traces, and use its
   complete raw-cache observation window, up to 400 packets. The upstream graph equations naturally
   extend to these cases. No input sample is split into separate directional examples.
2. **Size definition.** Upstream extracts `packet.ip_size`. V17 stores serialized packet size `len(pkt)`.
   We use the latter without guessing/subtracting a link-header length. This is not a byte-for-byte
   reproduction of the upstream PCAP representation.
3. **Fractional attack sizes.** Frozen V17 dummy injection can emit fractional sizes. The graph uses
   truncation toward zero, consistent with the upstream integer feature arithmetic; the attack itself
   is left unchanged. The count of affected samples is reported. Raw caches are not modified.
4. **No IAT feature addition.** IAT remains available to the frozen structural attack functions, but
   is not a model feature or edge criterion. Pure IAT jitter leaves an IBGC graph unchanged.
5. **Training protocol.** Fixed 100 clean epochs, validation-only checkpointing and three seeds are
   experimental harmonization. Adam(1e-3, weight_decay=0), batch64, no scheduler are declared choices;
   the inspected official repository does not establish these as the authors' hyperparameters.
6. **No extra filtering or parsing.** The runner does not execute upstream NFStream/port filters,
   split regeneration, or any PCAP parser. It only reads the existing three-column cache.

Suggested method description: “IBGC was implemented using the authors' graph-feature equations and
GCN architecture, adapted to our frozen flow-level splits and serialized packet-size cache. All
nonempty short and unidirectional flows were retained without padding. Each model was trained on
clean data for exactly 100 epochs, with the checkpoint selected solely by independent validation
cross-entropy.” Do not describe this as exact original preprocessing or original training reproduction.

## Frozen project source

The user supplied a local project path containing `BaseLine/baseline_v17_common.py`.
The file was copied **byte-for-byte**, with SHA-256:

`5cac4313cc4991b3de5fd4e1025b940c01249148b67fb2c3d9cb8218650a99cf`

This matches the hash recorded in the referenced conversation. It supplies attacks and metrics;
its PCAP extraction functions are not called. Scapy is a dependency only because this unchanged
module imports it. No payload, entropy, ports, addresses, flags, sequence/acknowledgment fields or
header bytes are passed to the graph/model. The already-derived 0/1 direction is retained.

Frozen split SHA-256:

- CipherSpectrum: `01e3b572e8514a238b3733e224161acd1112b79b56f038635be9339b23cfad6b`
- USTC 12-class: `a34754daa9e202f154484a5b63a8840eb829e56aeaaff4549bac6cb711127f3b`

Dataset manifest SHA-256:

- CipherSpectrum: `821bee04771f0c8dce3680a4301164d8ab48e03ec0d1ed57c3930015ad200a61`
- USTC: `612c69a851dad04658619fd6c79ef60f34d08dd7dd1a0e14262fa01428cb2190`

These values are enforced at runtime. Raw file hashes, labels and ordered sample IDs are also checked.
The original split files were not readable locally because they were cloud-sync placeholders;
the real-server manifest/cache integration remains to be run. No data hash or dataset test result
has been invented.

## Attack semantics

`ATTACK_BASE_SEED=20260919`; condition seed = existing `canonical_attack_seed(name,1.0)`.
Each condition uses its own NumPy/Python RNG stream, initialized once and consumed sequentially
over the complete frozen test order. Training seed does not affect attack generation.
The transformation order is size padding → IAT jitter → dummy insertion, then at most 400 packets.
Reordering, per-sample RNG resetting or independently sharding the stream would change the protocol
and is therefore not done.

For this metadata-only representation, the existing common module maps both Mixed and Adaptive
to the same structural attack family, with different condition seeds. **Adaptive is the V17
semantic-anchor-preserving protocol's structural projection, not an IBGC-targeted white-box
gradient attack.** No new adaptive optimizer or payload synthesis is introduced.

## Reproducibility and limitations

Every run binds Python source hashes, graph cache hashes and protocol constants. Checkpoints bind
dataset identity, seed, software versions and device. Incompatible reuse fails explicitly.
Checkpointing is atomic per file; latest.pt stores the authoritative best state so resume can repair
an interrupted best/latest pair. No test metric is used to choose models or hyperparameters.
Prepared test graphs may be generated before training for audit, but the training function receives
only clean train and clean validation datasets.

CPU uninterrupted/resumed equivalence is tested. CUDA PyG reductions can be nondeterministic;
the runner enables deterministic settings with warnings rather than falsely promising bitwise
CUDA identity. Real-data accuracy, GPU runtime and memory have not been measured in this session.
