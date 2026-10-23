"""Run: python -m unittest -v test_ibgc (CPU, synthetic data only)."""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Batch, Data
import baseline_v17_common as C
from ibgc_graph import build_graph
from ibgc_model import IBGC
from ibgc_data import (read_raw, validate_split, verify_common, prepare, load_partition,
                       write_graph_partition)
from ibgc_engine import train, predict, evaluate
from ibgc_io import Logger, sha256, read_json, write_json, load_checkpoint
from IBGC_final_paper_experiments_V1_100ep import parse_args

torch.set_num_threads(1)


def raw(d=(0, 0, 1, 0), size=None):
    d = np.array(d)
    return np.column_stack([d, np.arange(len(d)) + 10 if size is None else size,
                            np.arange(len(d)) * .001]).astype(np.float32)


def fixture(root):
    """Small manufactured data; never presented as real experiment evidence."""
    split = dict(dataset_manifest_sha256="synthetic", split_seed=42)
    cache = root / "raw"
    hashes = {}
    counts = dict(train=6, validation=4, test=4)
    sid = 0
    for part, n in counts.items():
        rows, seqs = [], []
        for i in range(n):
            rows.append(dict(sample_index=sid, relative_path=str(sid) + ".pcap", label=i % 2))
            seqs.append(raw((0, 1, 0) if i % 2 else (0, 0)))
            sid += 1
        split[part] = rows
        path = cache / "synthetic" / (part + "_raw.npz")
        C.save_raw_partition(path, seqs, np.array([r["label"] for r in rows]),
                             np.array([r["sample_index"] for r in rows]))
        hashes[part] = sha256(path)
    sp = root / "split.json"
    write_json(sp, split)
    spec = C.DatasetSpec("synthetic", "synthetic", root / "no-pcaps-needed", sp,
                         2, "synthetic", sha256(sp), counts)
    write_json(cache / "synthetic" / "raw_cache_manifest.json",
               dict(common_script_sha256=sha256(C.__file__), raw_cache_protocol=C.RAW_CACHE_PROTOCOL,
                    dataset_manifest_sha256="synthetic", split_sha256=sha256(sp), split_seed=42,
                    raw_max_packets=400, cache_sha256=hashes))
    return spec, cache


class GraphTests(unittest.TestCase):
    def test_hand_computed_features_and_edges(self):
        x, e = build_graph(raw((0, 0, 1), [100, 200, 50]))
        np.testing.assert_allclose(x[0], [-100, 2, 0, 1, 0, .5, 300, 0, 50,
                                           1 / 3, 150, 0, 6])
        np.testing.assert_allclose(x[2], [50, 1, 2, 0, 2, 0, 50, 300, 0, 1, 50, 1 / 6, 0])
        self.assertEqual(set(map(tuple, e.T)), {(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)})

    def test_one_packet_and_single_direction_retained(self):
        for n in (1, 2, 4, 6, 400):
            x, e = build_graph(raw([0] * n))
            self.assertEqual(x.shape, (n, 13))
            self.assertEqual(e.shape, (2, 2 * (n - 1)))
            self.assertTrue(np.isfinite(x).all())

    def test_all_adjacent_slice_connections(self):
        _, e = build_graph(raw([0, 0, 1, 1, 0]))
        self.assertEqual(e.shape[1], 2 * (1 + 1 + 2 * 2 + 2 * 1))
        self.assertNotIn((0, 4), set(map(tuple, e.T)))

    def test_iat_invariance_and_fractional_size_rule(self):
        a = raw(size=[10.9, 11.2, 12.8, 13.1])
        b = a.copy(); b[:, 2] += 999
        np.testing.assert_array_equal(build_graph(a)[0], build_graph(b)[0])
        np.testing.assert_array_equal(np.abs(build_graph(a)[0][:, 0]), [10, 11, 12, 13])

    def test_invalid_inputs_fail(self):
        bad = [np.empty((0, 3)), np.zeros((401, 3)), np.zeros((3, 4))]
        for col, v in ((0, .5), (1, 0), (2, -1), (1, np.nan), (2, np.inf)):
            a = raw(); a[0, col] = v; bad.append(a)
        for a in bad:
            with self.assertRaises(ValueError): build_graph(a)

    def test_batch_invariance_and_backward(self):
        ds = []
        for a in (raw(), raw([0]), raw([0, 0, 1, 1, 1])):
            x, e = build_graph(a)
            ds.append(Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(e)))
        torch.manual_seed(7)
        model = IBGC(3).eval()
        batch = Batch.from_data_list(ds)
        together = model(batch.x, batch.edge_index, batch.batch)
        for i, d in enumerate(ds):
            b = Batch.from_data_list([d])
            torch.testing.assert_close(together[i], model(b.x, b.edge_index, b.batch)[0], atol=2e-5, rtol=2e-5)
        (-together[:, 0].mean()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


class FrozenProtocolTests(unittest.TestCase):
    def test_constants_hash_and_no_epoch_override(self):
        verify_common()
        self.assertEqual(C.SEEDS, [42, 43, 44])
        self.assertEqual(C.ATTACK_BASE_SEED, 20260919)
        with self.assertRaises(SystemExit): parse_args(["--epochs", "2"])
        with self.assertRaises(SystemExit): parse_args(["--seeds", "42"])

    def test_attacks_reproducible_nonmutating_and_capped(self):
        xs = [raw(), raw([0, 1] * 200)]
        copies = [x.copy() for x in xs]
        for name in ("mixed", "adaptive"):
            a = C.build_attacked_raw_dataset(xs, name, 1.0)
            b = C.build_attacked_raw_dataset(xs, name, 1.0)
            for aa, bb in zip(a, b):
                np.testing.assert_array_equal(aa, bb)
                self.assertLessEqual(len(aa), 400)
            for x, old in zip(xs, copies): np.testing.assert_array_equal(x, old)
        self.assertNotEqual(C.canonical_attack_seed("mixed", 1), C.canonical_attack_seed("adaptive", 1))

    def test_metric_definitions(self):
        y = np.array([0, 1, 2, 0])
        cp = np.array([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8],[.1,.8,.1]])
        ap = np.array([[.1,.8,.1],[.1,.8,.1],[.1,.1,.8],[.8,.1,.1]])
        clean = C.classification_metrics(y, cp, 3)
        attacked = C.classification_metrics(y, ap, 3)
        m = C.attack_metrics(clean, attacked)
        self.assertEqual(clean["acc"], 75)
        self.assertAlmostEqual(clean["ece"], 5)
        self.assertAlmostEqual(m["asr"], 100 / 3)
        self.assertEqual(m["rel_drop"], 0)
        self.assertEqual(m["abs_drop_pp"], 0)
        from sklearn.metrics import roc_auc_score
        self.assertAlmostEqual(clean["macro_auroc"], roc_auc_score(y, cp, multi_class="ovr", average="macro") * 100)

    def test_prepare_and_reuse_and_cache_tamper(self):
        with tempfile.TemporaryDirectory() as t:
            r = Path(t); spec, cache = fixture(r); out = r / "out"
            logger = Logger(r / "test.log")
            a = prepare(spec, cache, out, logger)
            b = prepare(spec, cache, out, logger)
            self.assertEqual(a, b)
            clean = load_partition(out, a, "test_clean")
            attacked = load_partition(out, a, "test_mixed_1.0")
            np.testing.assert_array_equal(clean.ids, attacked.ids)
            np.testing.assert_array_equal(clean.labels, attacked.labels)
            p = cache / "synthetic" / "train_raw.npz"
            with p.open("ab") as f: f.write(b"tamper")
            with self.assertRaises(RuntimeError): prepare(spec, cache, out, logger)

    def test_mismatched_raw_ids_and_bad_offsets(self):
        with tempfile.TemporaryDirectory() as t:
            r = Path(t); spec, cache = fixture(r)
            p = cache / "synthetic" / "test_raw.npz"
            rows = read_json(spec.split_manifest)["test"]
            swapped = rows[::-1]
            with self.assertRaises(RuntimeError): read_raw(p, swapped)
            with np.load(p) as z: arrays = {k: z[k] for k in z.files}
            arrays["offsets"][1] = 0
            np.savez(p, **arrays)
            with self.assertRaises(RuntimeError): read_raw(p, rows)

    def test_split_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            from dataclasses import replace
            r = Path(t); spec, _ = fixture(r)
            s = read_json(spec.split_manifest)
            s["test"][0]["sample_index"] = s["train"][0]["sample_index"]
            write_json(spec.split_manifest, s)
            with self.assertRaises(RuntimeError):
                validate_split(replace(spec, expected_split_sha=sha256(spec.split_manifest)))


class TrainingTests(unittest.TestCase):
    def test_formal_entrypoint_three_seeds_and_aggregate(self):
        from unittest.mock import patch
        from IBGC_final_paper_experiments_V1_100ep import main
        with tempfile.TemporaryDirectory() as t:
            r = Path(t); spec, cache = fixture(r)
            args = ["--mode", "all", "--device", "cpu", "--threads", "1",
                    "--raw-cache-root", str(cache), "--run-dir", str(r / "run")]
            with patch("baseline_v17_common.build_dataset_specs", return_value=[spec]):
                main(args)
                main(args + ["--resume"])
            summary = read_json(r / "run/summary.json")
            self.assertEqual(summary["seeds"], [42, 43, 44])
            self.assertTrue(all(x["n_seeds"] == 3 for x in summary["aggregate"]))
            self.assertFalse((r / "run/.running.lock").exists())
            for seed in (42, 43, 44):
                self.assertEqual(read_json(r / "run/synthetic/models" / ("seed" + str(seed)) /
                                           "training_summary.json")["completed_epochs"], 100)

    def test_resume_exact_and_validation_best(self):
        with tempfile.TemporaryDirectory() as t:
            r = Path(t); spec, cache = fixture(r); log = Logger(r / "test.log")
            prep = prepare(spec, cache, r / "p", log)
            tr = load_partition(r / "p", prep, "train_clean")
            va = load_partition(r / "p", prep, "validation_clean")
            identity = {"synthetic": True, "seed": 42, "epochs": 4}
            train(tr, va, 2, 42, torch.device("cpu"), r / "full", identity, log, epochs=4)
            train(tr, va, 2, 42, torch.device("cpu"), r / "resume", identity, log, epochs=4, stop_after=2)
            train(tr, va, 2, 42, torch.device("cpu"), r / "resume", identity, log, epochs=4, resume=True)
            a = load_checkpoint(r / "full/latest.pt", identity)
            b = load_checkpoint(r / "resume/latest.pt", identity)
            for k in a["model_state_dict"]:
                self.assertTrue(torch.equal(a["model_state_dict"][k], b["model_state_dict"][k]), k)
            self.assertEqual(a["best_epoch"], min(a["history"], key=lambda x:x["validation_ce"])["epoch"])
            with self.assertRaises(RuntimeError): load_checkpoint(r / "full/latest.pt", {"wrong": True})
            with self.assertRaises(RuntimeError): evaluate({}, 2, torch.device("cpu"), r / "full", r / "e", identity, log)

    def test_100_epochs_then_all_three_conditions(self):
        with tempfile.TemporaryDirectory() as t:
            r = Path(t); spec, cache = fixture(r); log = Logger(r / "test.log")
            prep = prepare(spec, cache, r / "p", log)
            identity = {"synthetic": True, "seed": 42, "epochs": 100}
            tr = load_partition(r / "p", prep, "train_clean")
            va = load_partition(r / "p", prep, "validation_clean")
            summary = train(tr, va, 2, 42, torch.device("cpu"), r / "model", identity, log)
            self.assertEqual(summary["completed_epochs"], 100)
            self.assertEqual(len(read_json(r / "model/history.json")), 100)
            ds = {c: load_partition(r / "p", prep, "test_" + c)
                  for c in ("clean", "mixed_1.0", "adaptive_1.0")}
            report = evaluate(ds, 2, torch.device("cpu"), r / "model", r / "e", identity, log)
            self.assertEqual(len(report["metrics"]), 3)
            self.assertTrue((r / "e/adaptive_1.0_probabilities.npz").exists())


class OfficialParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.environ.get("IBGC_OFFICIAL_ROOT")
        if not root:
            raise unittest.SkipTest("Optional upstream parity: set IBGC_OFFICIAL_ROOT after fetch_official.py")
        root = Path(root)
        manifest = read_json(Path(__file__).with_name("official_source_manifest.json"))
        modules = []
        for name in ("NewGraphDataset", "GraphNet"):
            rel = "02_classification/" + name + ".py"
            path = root / rel
            if sha256(path) != manifest["files"][rel]:
                raise RuntimeError("Upstream source hash mismatch")
            spec = importlib.util.spec_from_file_location("official_" + name, path)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            modules.append(mod)
        cls.graph_module, cls.model_module = modules

    def test_graph_features_and_edges_match_official(self):
        ref = object.__new__(self.graph_module.GraphMitDataset)
        rng = np.random.default_rng(45)
        for n in (1, 2, 4, 5, 6, 40, 400):
            for _ in range(5):
                d = rng.integers(0, 2, size=n)
                s = rng.integers(1, 1501, size=n)
                x, e = build_graph(raw(d, s))
                re, _, rx = ref.generate_new_graph(d.astype(str).tolist(), s.astype(str).tolist(), 0)
                np.testing.assert_allclose(x, np.array(rx, np.float32), rtol=0, atol=0)
                np.testing.assert_array_equal(e, np.array(re, np.int64).reshape(-1, 2).T)

    def test_model_matches_official_logits_and_parameters(self):
        ours = IBGC(12).eval()
        ref = self.model_module.GraphNet(12).eval()
        ref.load_state_dict(ours.state_dict(), strict=True)
        x, e = build_graph(raw())
        x, e = torch.from_numpy(x), torch.from_numpy(e)
        b = torch.zeros(len(x), dtype=torch.long)
        torch.testing.assert_close(ours(x,e,b), ref(x,e,b), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
