"""Read-only raw-cache adapter. No PCAP parsing or payload/header extraction."""
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
import baseline_v17_common as C
from ibgc_graph import build_graph, validate_raw, REPRESENTATION
from ibgc_io import sha256, read_json, write_json, save_npz

COMMON_SHA = "5cac4313cc4991b3de5fd4e1025b940c01249148b67fb2c3d9cb8218650a99cf"
PARTS = ("train", "validation", "test")
CONDITIONS = ("train_clean", "validation_clean", "test_clean", "test_mixed_1.0",
              "test_adaptive_1.0")


def verify_common():
    if sha256(C.__file__) != COMMON_SHA:
        raise RuntimeError("Frozen baseline_v17_common.py SHA mismatch")
    if C.ATTACK_BASE_SEED != 20260919 or C.SEEDS != [42, 43, 44]:
        raise RuntimeError("Frozen protocol constants mismatch")


def validate_split(spec):
    # C.validate_dataset_spec also opens every PCAP. Deliberately validate only
    # the frozen manifest and cache here, so raw-cache-only deployment works.
    if sha256(spec.split_manifest) != spec.expected_split_sha:
        raise RuntimeError("Frozen split file hash mismatch")
    split = read_json(spec.split_manifest)
    if split.get("dataset_manifest_sha256") != spec.expected_dataset_manifest:
        raise RuntimeError("Dataset manifest hash mismatch")
    if split.get("split_seed", 42) != 42:
        raise RuntimeError("Split seed mismatch")
    seen_ids, seen_paths = set(), set()
    for part in PARTS:
        rows = split[part]
        if len(rows) != spec.expected_counts[part]:
            raise RuntimeError("Frozen partition count mismatch: " + part)
        for row in rows:
            sid, path, label = row["sample_index"], row["relative_path"], row["label"]
            if sid in seen_ids or path in seen_paths:
                raise RuntimeError("Duplicate sample identity/path across frozen splits")
            if not isinstance(label, int) or not 0 <= label < spec.num_classes:
                raise RuntimeError("Invalid label in frozen split")
            seen_ids.add(sid)
            seen_paths.add(path)
    labels = {r["label"] for part in PARTS for r in split[part]}
    if labels != set(range(spec.num_classes)):
        raise RuntimeError("Frozen class mapping mismatch")
    return split


def read_raw(path, rows):
    with np.load(path, allow_pickle=False) as z:
        values, offsets = z["values"], z["offsets"]
        labels, ids = z["labels"], z["sample_indices"]
    n = len(rows)
    if offsets.shape != (n + 1,) or offsets.dtype.kind not in "iu":
        raise RuntimeError("Invalid raw offsets")
    if offsets[0] != 0 or offsets[-1] != len(values) or (np.diff(offsets) <= 0).any():
        raise RuntimeError("Empty sample or corrupt raw offsets")
    if labels.dtype.kind not in "iu" or ids.dtype.kind not in "iu":
        raise RuntimeError("Non-integer labels/IDs")
    if not np.array_equal(labels, [r["label"] for r in rows]):
        raise RuntimeError("Raw labels/order differ from frozen split")
    if not np.array_equal(ids, [r["sample_index"] for r in rows]):
        raise RuntimeError("Raw sample IDs/order differ from frozen split")
    if values.ndim != 2 or values.shape[1] != 3:
        raise RuntimeError("Raw cache must contain exactly direction,size,IAT")
    seqs = [values[a:b].copy() for a, b in zip(offsets[:-1], offsets[1:])]
    for raw in seqs:
        validate_raw(raw)
    return seqs, labels.astype(np.int64), ids.astype(np.int64)


def verify_raw_meta(spec, cache_root):
    path = C.raw_cache_meta_path(cache_root, spec)
    meta = read_json(path)
    expected = dict(common_script_sha256=COMMON_SHA, raw_cache_protocol=C.RAW_CACHE_PROTOCOL,
                    dataset_manifest_sha256=spec.expected_dataset_manifest,
                    split_sha256=spec.expected_split_sha, split_seed=42, raw_max_packets=400)
    for k, v in expected.items():
        if meta.get(k) != v:
            raise RuntimeError("Raw cache metadata mismatch: " + k)
    for part in PARTS:
        if sha256(C.raw_cache_path(cache_root, spec, part)) != meta["cache_sha256"][part]:
            raise RuntimeError("Raw cache content hash mismatch: " + part)
    return meta


def write_graph_partition(path, seqs, labels, ids, logger=None):
    xs, es, node_offsets, edge_offsets = [], [], [0], [0]
    short = unidirectional = fractional = 0
    for i, seq in enumerate(seqs):
        x, e = build_graph(seq)
        xs.append(x)
        es.append(e)  # local packet/node indices retained per graph
        node_offsets.append(node_offsets[-1] + len(x))
        edge_offsets.append(edge_offsets[-1] + e.shape[1])
        short += len(seq) < 5
        unidirectional += len(np.unique(seq[:, 0])) == 1
        fractional += bool((seq[:, 1] != np.trunc(seq[:, 1])).any())
        if logger and ((i + 1) % 1000 == 0 or i + 1 == len(seqs)):
            logger.log("[graph][{}] {}/{}".format(Path(path).stem, i + 1, len(seqs)))
    save_npz(path, x=np.concatenate(xs), edge_index=np.concatenate(es, axis=1),
             node_offsets=np.array(node_offsets, np.int64),
             edge_offsets=np.array(edge_offsets, np.int64), labels=labels, sample_indices=ids)
    return dict(n=len(seqs), short_lt5=int(short), unidirectional=int(unidirectional),
                fractional_size_samples=int(fractional), nodes=node_offsets[-1],
                directed_edges=edge_offsets[-1], max_nodes=int(np.diff(node_offsets).max()),
                max_directed_edges=int(np.diff(edge_offsets).max()), dropped=0, padded=0)


def prepare(spec, cache_root, out, logger):
    verify_common()
    split = validate_split(spec)
    meta = verify_raw_meta(spec, cache_root)
    identity = dict(dataset=spec.name, num_classes=spec.num_classes,
                    split_sha256=spec.expected_split_sha,
                    dataset_manifest_sha256=spec.expected_dataset_manifest,
                    raw_cache_manifest_sha256=sha256(C.raw_cache_meta_path(cache_root, spec)),
                    raw_cache_sha256=meta["cache_sha256"], common_sha256=COMMON_SHA,
                    representation=REPRESENTATION, graph_code_sha256=sha256(Path(__file__).with_name("ibgc_graph.py")),
                    data_code_sha256=sha256(__file__), attack_base_seed=20260919)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    prep = out / "preparation.json"
    previous = read_json(prep) if prep.exists() else {"identity": identity, "partitions": {}}
    if previous["identity"] != identity:
        raise RuntimeError("Preparation provenance changed; use a new run directory")
    # Always validate identities and values, including on graph-cache reuse.
    for part in PARTS:
        seqs, labels, ids = read_raw(C.raw_cache_path(cache_root, spec, part), split[part])
        names = [part + "_clean"]
        if part == "test":
            names += ["test_mixed_1.0", "test_adaptive_1.0"]
        for name in names:
            path = out / "prepared" / (name + ".npz")
            old = previous["partitions"].get(name)
            if old and path.exists() and sha256(path) == old["sha256"]:
                logger.log("[prepare][{}] verified reuse {}".format(spec.key, name))
                continue
            # One RNG stream per condition, in original manifest order.
            # Never shard/reorder these attacks or seed by training seed/sample.
            raw = seqs if name.endswith("clean") else C.build_attacked_raw_dataset(
                seqs, name.split("_")[1], 1.0)
            audit = write_graph_partition(path, raw, labels, ids, logger)
            if not name.endswith("clean"):
                audit["canonical_attack_seed"] = C.canonical_attack_seed(name.split("_")[1], 1.0)
                audit["changed_raw_samples"] = sum(not np.array_equal(a, b) for a, b in zip(seqs, raw))
            previous["partitions"][name] = dict(sha256=sha256(path), audit=audit)
            write_json(prep, previous)
    return previous


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, path, expected_hash=None):
        if expected_hash is not None and sha256(path) != expected_hash:
            raise RuntimeError("Prepared graph cache hash mismatch")
        with np.load(path, allow_pickle=False) as z:
            self.x = torch.from_numpy(z["x"].copy())
            self.edge = torch.from_numpy(z["edge_index"].copy())
            self.no, self.eo = z["node_offsets"].copy(), z["edge_offsets"].copy()
            self.labels, self.ids = z["labels"].copy(), z["sample_indices"].copy()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return Data(x=self.x[self.no[i]:self.no[i + 1]],
                    edge_index=self.edge[:, self.eo[i]:self.eo[i + 1]],
                    y=torch.tensor([int(self.labels[i])]),
                    sample_id=torch.tensor([int(self.ids[i])]))


def load_partition(out, prep, name):
    return GraphDataset(Path(out) / "prepared" / (name + ".npz"),
                        prep["partitions"][name]["sha256"])
