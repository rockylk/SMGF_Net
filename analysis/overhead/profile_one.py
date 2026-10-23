#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen V17 model-only inference profile (no training, no attack generation).

FLOPs computed by torch.profiler are a SUPPORTED-OP SUBTOTAL, not audited full
FLOPs. GATv2/PyG scatter and sparse graph kernels need coverage attestation.
Outputs are evidence records, never silently marked publication-ready.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, platform, statistics, sys, time
from datetime import datetime, timezone
from pathlib import Path
from inventory import ROOT, sha256
from adapters import load

HERE = Path(__file__).resolve().parent
MODELS = ("SMGF-Net", "USTC-CNN", "ET-BERT", "PEAN-NHP", "IBGC")
SEEDS = (42,43,44)

def benchmark(torch, callable_fn, inp, warmup: int, repeats: int, device: str):
    with torch.inference_mode():
        for _ in range(warmup):
            out = callable_fn(*inp)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        timings = []
        for _ in range(repeats):
            if device.startswith("cuda"):
                st = torch.cuda.Event(enable_timing=True)
                en = torch.cuda.Event(enable_timing=True)
                st.record()
                out = callable_fn(*inp)
                en.record()
                torch.cuda.synchronize()
                timings.append(float(st.elapsed_time(en)))
            else:
                t = time.perf_counter_ns()
                out = callable_fn(*inp)
                timings.append((time.perf_counter_ns()-t)/1e6)
        if isinstance(out,(tuple,list)):
            out=out[0]
        if not isinstance(out,torch.Tensor) or not torch.isfinite(out).all():
            raise ValueError("Nonfinite or incompatible model forward output")
    return {"median_ms":statistics.median(timings),
            "mean_ms":statistics.mean(timings),
            "std_ms":statistics.stdev(timings) if len(timings)>1 else 0.,
            "min_ms":min(timings),"max_ms":max(timings),
            "samples_ms":timings}

def flops_subtotal(torch, func, inputs, device):
    acts=[torch.profiler.ProfilerActivity.CPU]
    if device.startswith("cuda"):
        acts.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.inference_mode():
        with torch.profiler.profile(activities=acts, with_flops=True, record_shapes=True) as prof:
            func(*inputs)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
    events=prof.key_averages()
    accounted=[{"operator":r.key,"flops":int(r.flops),"count":int(r.count)}
               for r in events if int(r.flops or 0)>0]
    subtotal=sum(r["flops"] for r in accounted)
    ops=[{"operator":r.key,"count":int(r.count)} for r in events
         if int(r.flops or 0)==0 and r.key.startswith(("aten::","torch_scatter::","torch_sparse::","pyg::"))]
    graph_risk=[r for r in ops if any(k in r["operator"].lower() for k in (
        "scatter","index_add","segment","spmm","sparse","gather","embedding","lstm","softmax","conv","pool"))]
    return {"supported_op_flops":subtotal,
            "supported_op_mflops":subtotal/1e6,
            "coverage":"UNVERIFIED_SUBTOTAL_NOT_TOTAL",
            "counted_ops":accounted,
            "unaccounted_ops":ops,
            "graph_and_sequence_coverage_risk":graph_risk,
            "explanation":"PyTorch FLOP profiler counts only instrumented ops; GAT/PyG, RNN, embeddings, normalization, sparse/scatter may be omitted. Not a full-model FLOP count."}

def run(args):
    import torch
    if args.seed not in SEEDS:raise ValueError("Seed must be one of 42,43,44")
    if not 0 <= args.sample_index < 8200:raise ValueError("Index outside held-out test")
    if args.model not in MODELS:raise ValueError(args.model)
    if args.warmup<1 or args.repeat<5:raise ValueError("At least one warmup and five measured runs")
    if args.dataset != "cipher":
        raise ValueError("Section 5.6 main table profiles CipherSpectrum only; USTC may use a separately audited run")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. No GPU latency evidence generated.")
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark=False
    device=args.device
    torch.cuda.init() if device=="cuda" else None
    item=load(args.model,args.dataset,args.seed,device,
              checkpoint_override=args.checkpoint,input_path=args.input_path,
              external_factory=args.external_factory, sample_index=args.sample_index)
    nparams=sum(p.numel() for p in item.model.parameters())
    ntrain=sum(p.numel() for p in item.model.parameters() if p.requires_grad)
    if args.model=="ET-BERT":
        # External loader is responsible for checking the exact native state
        # against checksum; checkpoint SHA is recorded in output.
        pass
    if not all(torch.is_tensor(x) for x in item.inputs):
        raise TypeError("Model input contains non-tensors")
    torch.backends.cudnn.deterministic=True
    time_stats=benchmark(torch,item.forward,item.inputs,args.warmup,args.repeat,device)
    ops=flops_subtotal(torch,item.forward,item.inputs,device)
    provenance=dict(
        schema="section56-v17-modelonly-v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        model=args.model, dataset=args.dataset, seed=args.seed,
        sample_index=args.sample_index, sample_id=item.sample_id,
        device=device, torch_version=torch.__version__,
        cuda_version=torch.version.cuda, gpu_name=torch.cuda.get_device_name() if device=="cuda" else "",
        platform=platform.platform(), source=str(item.source),
        source_sha256=sha256(item.source),
        checkpoint=str(item.checkpoint),
        checkpoint_sha256=sha256(item.checkpoint),
        input_path=str(item.evidence_path),
        input_sha256=sha256(item.evidence_path),
        actual_input_shape=item.shape,
        input_origin=item.data_origin,
        batch_size=1, single_example_per_seed=True,
        warmup=args.warmup, repeats=args.repeat,
        precision="fp32 (native model default; no autocast)",
        includes="model forward only on already-prepared single frozen clean-test flow",
        excludes=["PCAP parsing","fingerprint/entropy extraction","graph building",
                  "CPU preprocessing","CPU-GPU transfer","packet tokenization"],
        trainable_parameters=ntrain, total_parameters=nparams,
        parameters_m=nparams/1e6,
        latency=time_stats,
        flops=ops,
        robust_acc_source="independent frozen manuscript table; not recomputed by this script",
        paper_eligible=False,
        blocking_reasons=[
            "Full FLOP coverage has not been attested",
            "Representative end-to-end latency not assessed",
            "Measured one frozen flow per seed; multi-flow latency distribution not yet attested",
        ],
    )
    stem=f"{args.model.replace('-','_')}_cipher_seed{args.seed}_sample{args.sample_index}"
    out=Path(args.out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    file=out/(stem+".json")
    # Never overwrite a previous completed measurement silently.
    if file.exists() and not args.overwrite:
        raise FileExistsError(f"Existing result; explicitly pass --overwrite to replace: {file}")
    file.write_text(json.dumps(provenance,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    print(f"MODEL={args.model} SEED={args.seed} PARAMS={nparams} "
          f"LATENCY_MEDIAN_MS={time_stats['median_ms']:.6f} "
          f"FLOPS_SUBTOTAL={ops['supported_op_flops']} FLOPS_STATUS=UNVERIFIED")
    print(f"OUTPUT={file}")
    return provenance

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model",choices=MODELS,required=True)
    ap.add_argument("--dataset",default="cipher",choices=["cipher"])
    ap.add_argument("--seed",type=int,choices=SEEDS,required=True)
    ap.add_argument("--sample-index",type=int,default=0,help="Deterministic real held-out test-flow index (0..8199)")
    ap.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    ap.add_argument("--warmup",type=int,default=50)
    ap.add_argument("--repeat",type=int,default=200)
    ap.add_argument("--checkpoint",help="Explicit frozen checkpoint override (ET-BERT remote is missing locally)")
    ap.add_argument("--input-path",help="Native frozen prepared npz or attested PyG torch tensor dictionary")
    ap.add_argument("--external-factory",help="ET-BERT official UER factory /abs/path.py:factory")
    ap.add_argument("--out",type=Path,default=HERE/"results"/"measurements")
    ap.add_argument("--overwrite",action="store_true")
    run(ap.parse_args())
if __name__=="__main__":
    main()