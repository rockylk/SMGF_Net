#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Load EXACT frozen model sources/weights. Never retrain or edit official files.

The returned input batch is an actual frozen clean-test sample when the native
prepared NPZ is valid (CNN and PEAN). Graph models require a separately
attested test-batch file exported from the formal test loader on the GPU.
ET-BERT requires a Python factory implementing the official UER model.
"""
from __future__ import annotations
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Any
from inventory import ROOT, RESULT, SOURCES, MODEL_DIRS, checkpoint_path, sha256, source_path, base_dir

@dataclass
class Loaded:
    model: Any
    inputs: tuple
    checkpoint: Path
    source: Path
    evidence_path: Path
    data_origin: str
    forward: Any
    shape: list
    sample_id: int
    # Non-SMGF graph models may use nontrivial batch input and dynamic sparsity.

def _import_file(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import official module {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

def _ckpt_state(torch, path: Path, extra_safe=None):
    kw = dict(map_location="cpu", weights_only=True)
    try:
        with torch.serialization.safe_globals(extra_safe or []):
            obj = torch.load(path, **kw)
    except Exception as e:
        raise RuntimeError(f"Cannot safely load frozen checkpoint {path}: {e}") from e
    if not isinstance(obj, dict):
        raise RuntimeError(f"Unexpected frozen checkpoint object: {path}")
    for name in ("model_state_dict", "state_dict", "model"):
        if name in obj and isinstance(obj[name], dict):
            return obj[name], obj
    # Raw torch state_dict is allowed only when ALL keys are tensors.
    if obj and all(torch.is_tensor(x) for x in obj.values()):
        return obj, obj
    raise RuntimeError(f"Missing recognized state_dict in {path}; no guessing")

def _verified_checkpoint(model_name, dataset, seed, override, evidence_path):
    if override is None:
        if model_name == "SMGF-Net" and dataset == "ustc":
            base = RESULT / "SMGF_V17_USTC_FINAL_3SEED_20260924_144614"
            file = "FINAL_seed42_REUSED.pt" if seed == 42 else f"FINAL_seed{seed}.pt"
            p = base / "checkpoints" / file
        else:
            p = checkpoint_path(model_name, seed, dataset)
    else:
        p = Path(override).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Missing frozen checkpoint: {p}")
    # Per-model original summaries are used to cross-check SHA, if present.
    if model_name == "USTC-CNN":
        import json
        summary=p.parent/"training_summary.json"
        if not summary.is_file():
            raise FileNotFoundError("USTC-CNN frozen training summary missing")
        meta=json.loads(summary.read_text(encoding="utf-8"))
        if meta.get("seed") != seed or meta.get("checkpoint_sha256") != sha256(p):
            raise RuntimeError("USTC-CNN official checkpoint/seed SHA mismatch")
    if model_name == "SMGF-Net":
        import json
        report=RESULT/"SMGF_V17_CIPHERSPECTRUM_FINAL_3SEED_20260923_135542"/"per_seed_results.json"
        if not report.is_file():
            raise FileNotFoundError("Original SMGF V17 final evaluation source missing")
        data=json.loads(report.read_text(encoding="utf-8"))
        matches=[r for r in data.get("test_results",[]) if int(r.get("seed",-1))==seed]
        if dataset=="cipher" and (len(matches)!=1 or Path(matches[0].get("checkpoint","")).name != p.name):
            raise RuntimeError("SMGF checkpoint does not match frozen final test record")
    if model_name == "PEAN-NHP":
        import json
        s = p.parent / "training_summary.json"
        if not s.is_file():
            raise FileNotFoundError(f"PEAN official training summary missing: {s}")
        meta = json.loads(s.read_text(encoding="utf-8"))
        if meta.get("best_checkpoint_sha256") != sha256(p):
            raise RuntimeError("PEAN official training-summary checkpoint SHA mismatch")
    if model_name == "IBGC":
        import json
        s = p.parent / "training_summary.json"
        if not s.is_file():
            raise FileNotFoundError("IBGC formal training summary missing")
        meta = json.loads(s.read_text(encoding="utf-8"))
        if meta.get("best_sha256") != sha256(p):
            raise RuntimeError("IBGC official checkpoint SHA mismatch")
    if model_name == "ET-BERT":
        import json
        s = p.parent / "train_summary.json"
        if not s.is_file():
            raise FileNotFoundError("ET-BERT official train_summary missing")
        meta = json.loads(s.read_text(encoding="utf-8"))
        if meta.get("checkpoint_sha256") != sha256(p):
            raise RuntimeError("ET-BERT checkpoint SHA mismatch")
    return p

def _external_graph_batch(torch, model, input_path, device, typ, sample_index):
    if not input_path:
        raise FileNotFoundError(
            f"{typ}: no frozen PyG test-batch artifact; export a real clean-test batch "
            "with the original V17 pipeline and pass --input-path. "
            "A synthetic graph would invalidate FLOPs and latency comparisons.")
    src = Path(input_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(src)
    meta_file = src.with_suffix(".provenance.json")
    if not meta_file.is_file():
        raise FileNotFoundError(f"Graph test-input provenance absent: {meta_file}")
    import json
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    if meta.get("index") != sample_index or meta.get("model") != typ or meta.get("output_sha256") != sha256(src) or "sample_id" not in meta:
        raise ValueError("Graph input provenance/index/hash mismatch")
    inp = torch.load(src, map_location="cpu", weights_only=True)
    # Saved dictionary rather than pickled PyG object: safe and portable.
    if not isinstance(inp, dict):
        raise ValueError("Graph input artifact must be a dict of torch tensors")
    # Preserve ORIGINAL forward signatures exactly:
    # SMGF: (x, edge_index, edge_attr, batch, stats, finger, entropy)
    # IBGC: (x, edge_index, batch)
    keys = (("x", "edge_index", "edge_attr", "batch",
             "stats_attr", "finger_attr", "entropy_attr") if typ == "SMGF-Net"
            else ("x", "edge_index", "batch"))
    if any(k not in inp or not torch.is_tensor(inp[k]) for k in keys):
        raise ValueError(f"Frozen PyG tensor dictionary missing keys: {keys}")
    tensors = [inp[k].to(device) for k in keys]
    if not all(t.isfinite().all() for t in tensors if t.dtype.is_floating_point):
        raise ValueError("Graph artifact has nonfinite float tensors")
    if tensors[0].ndim != 2 or tensors[1].shape[0] != 2:
        raise ValueError("Invalid graph x/edge_index dimensions")
    if inp["batch"].max().item()!=0 or inp["batch"].min().item()!=0:
        raise ValueError("Each graph profile must represent exactly one flow")
    return tuple(tensors), src, int(meta["sample_id"])

def load(model_name: str, dataset: str, seed: int, device: str,
         checkpoint_override: str | None = None, input_path: str | None = None,
         external_factory: str | None = None, sample_index: int = 0):
    import torch
    if not 0 <= sample_index < 8200:
        raise ValueError('Requested sample index outside frozen test set')
    if model_name not in MODEL_DIRS:
        raise ValueError(model_name)
    ckpt = _verified_checkpoint(model_name, dataset, seed, checkpoint_override, input_path)
    src = source_path(model_name)
    if not src.is_file():
        raise FileNotFoundError(f"Missing frozen implementation: {src}")
    classes = 41 if dataset == "cipher" else 12
    safe = []
    try:
        from torch.torch_version import TorchVersion
        safe.append(TorchVersion)
    except ImportError:
        pass
    evidence = None
    if model_name == "USTC-CNN":
        # Import of original baseline creates RUN_DIR/experiment.log; direct it
        # to disposable scratch, never the immutable official RESULT directory.
        with tempfile.TemporaryDirectory(prefix="s56_cnn_import_") as td:
            old_run=os.environ.get("USTCCNN_FINAL_RUN_DIR")
            os.environ["USTCCNN_FINAL_RUN_DIR"]=td
            try:
                native = _import_file(src, "v17_ustc_cnn_profiler")
            finally:
                if old_run is None:os.environ.pop("USTCCNN_FINAL_RUN_DIR",None)
                else:os.environ["USTCCNN_FINAL_RUN_DIR"]=old_run
        model = native.USTCCNN(classes)
        state, _ = _ckpt_state(torch, ckpt, safe)
        model.load_state_dict(state, strict=True)
        prep = Path(input_path) if input_path else RESULT / MODEL_DIRS[model_name] / dataset / "prepared" / "test_clean.npz"
        import numpy as np
        with np.load(prep, allow_pickle=False) as z:
            data = np.asarray(z["x"][sample_index], dtype=np.uint8)
            sample_id=int(z["sample_index"][sample_index])
            assert data.shape == (784,), f"Invalid native CNN frozen image: {data.shape}"
        inputs = (torch.from_numpy(data.copy()).float().div(255).view(1,1,28,28).to(device),)
        evidence = prep
        forward = model
    elif model_name == "PEAN-NHP":
        base = src.parent
        # PEAN V17 imports the official TRF from its sibling
        # official_sources/PEAN/TRF.py (bundled if missing on GPU).
        if not (base/"official_sources"/"PEAN"/"TRF.py").is_file():
            raise FileNotFoundError("PEAN official TRF dependency missing at "+str(base))
        sys.path.insert(0, str(base))
        native = _import_file(src, "v17_pean_nhp_profiler")
        model = native.PEANNHPV17(classes, device=device)
        state, payload = _ckpt_state(torch, ckpt, safe)
        if payload.get("checkpoint_schema_version") != 2:
            raise RuntimeError("PEAN-NHP unexpected checkpoint schema")
        model.load_state_dict(state, strict=True)
        prep = Path(input_path) if input_path else RESULT / MODEL_DIRS[model_name] / dataset / "prepared" / "test_clean.npz"
        import numpy as np
        with np.load(prep,allow_pickle=False) as z:
            bs = np.asarray(z["bytes"][sample_index],dtype=np.int64)
            lengths = np.asarray(z["lengths"][sample_index],dtype=np.int64)
            sample_id=int(z["sample_indices"][sample_index])
            assert bs.shape == (10,400) and lengths.shape == (10,), "Invalid native PEAN-NHP frozen sample"
        inputs = (torch.from_numpy(bs.copy()).long()[None].to(device),
                  torch.from_numpy(lengths.copy()).long()[None].to(device))
        evidence = prep
        forward = model
    elif model_name == "IBGC":
        sys.path.insert(0, str(base_dir()))
        native = _import_file(src,"v17_ibgc_profiler")
        model = native.IBGC(classes)
        state, _ = _ckpt_state(torch,ckpt,safe)
        model.load_state_dict(state,strict=True)
        inputs, evidence, sample_id = _external_graph_batch(torch, model, input_path, device, model_name, sample_index)
        forward=model
    elif model_name == "SMGF-Net":
        # Formal V17 import can create a run folder; isolate its side effect.
        with tempfile.TemporaryDirectory(prefix="s56_smgf_import_") as td:
            old = os.environ.get("SMGF_RESULT_DIR")
            os.environ["SMGF_RESULT_DIR"] = td
            try:
                native = _import_file(src,"v17_smgf_final_profiler")
            finally:
                if old is None:os.environ.pop("SMGF_RESULT_DIR",None)
                else:os.environ["SMGF_RESULT_DIR"]=old
            model = native.SMGFNetV17(
                node_in_channels=5, stats_dim=9, finger_seq_len=20,
                entropy_seq_len=5, hidden_channels=32,
                out_channels=classes, gate_mode="learned")
        state,_ = _ckpt_state(torch,ckpt,safe)
        model.load_state_dict(state,strict=True)
        inputs, evidence, sample_id = _external_graph_batch(torch, model, input_path, device, model_name, sample_index)
        forward=model.forward_classifier
    elif model_name == "ET-BERT":
        if not external_factory or ":" not in external_factory:
            raise RuntimeError(
                "ET-BERT official finetuned.bin requires upstream UER model/tokenizer loader. "
                "Pass --external-factory /absolute/file.py:factory after adapting it to "
                "load the exact official checkpoint and frozen test token sample. "
                "No substituting another BERT or generating synthetic tokens.")
        source_name, method = external_factory.split(":", 1)
        factory_file = Path(source_name).resolve()
        ext = _import_file(factory_file,"v17_etbert_official_factory")
        model, inputs, evidence, sample_id = getattr(ext, method)(str(ckpt), str(input_path or ""), dataset, seed, device, sample_index)
        if not isinstance(model,torch.nn.Module):
            raise RuntimeError("ET-BERT factory did not produce nn.Module")
        if not isinstance(inputs,(tuple,list)) or not inputs or not all(torch.is_tensor(x) for x in inputs):
            raise RuntimeError("ET-BERT factory must return torch tensor inputs")
        inputs = tuple(x.to(device) for x in inputs)
        evidence = Path(evidence).resolve()
        if not evidence.is_file():
            raise FileNotFoundError("ET-BERT factory missing frozen test-input evidence")
        forward=model
    model = model.to(device).eval()
    # Note: Checkpoint hashes and strict state restore are authoritative.
    # Graph test-batch artifacts MUST themselves be split-audited externally.
    shapes = [list(x.shape) for x in inputs]
    return Loaded(model, inputs, ckpt, src, Path(evidence), "frozen_clean_test", forward, shapes, int(sample_id))