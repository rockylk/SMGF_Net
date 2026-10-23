#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate 45 genuine frozen-GPU evaluations and export Figure6 supplement.

Strict: 3 models x 5 intermediate strengths x 3 seeds = 45; duplicate/missing
or unknown model/attack/strength rejected. Does not impute missing results.
Only produces CSV when all 45 records pass provenance/existence tests.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

EXPECTED={
    ("PEAN-NHP","mixed"):(.3,.6,.9),
    ("PEAN-NHP","adaptive"):(.4,.7),
    ("IBGC","mixed"):(.3,.6,.9),
    ("IBGC","adaptive"):(.4,.7),
    ("ET-BERT","mixed"):(.3,.6,.9),
    ("ET-BERT","adaptive"):(.4,.7)}
SEEDS=(42,43,44)
SPLIT="01e3b572e8514a238b3733e224161acd1112b79b56f038635be9339b23cfad6b"
FIELDS=("model","attack","strength","seed","robust_acc",
        "checkpoint_sha256","attack_manifest_sha256","split_sha256",
        "evaluation_artifact")
CHECKPOINT_IDS={"PEAN-NHP":{}, "IBGC":{}, "ET-BERT":{}}
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for piece in iter(lambda:f.read(1048576),b""):h.update(piece)
    return h.hexdigest()
def valid_sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value.lower())
def load_table(path):
    if not path.is_file():raise FileNotFoundError(f"Missing baseline output: {path}")
    with path.open("r",newline="",encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    return rows
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pean",type=Path,required=True)
    p.add_argument("--ibgc",type=Path,required=True)
    p.add_argument("--etbert",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--allow-missing-artifacts",action="store_true",
                   help="Only for importing remotely archived outputs; do not use for GPU provenance validation")
    args=p.parse_args()
    sources={"PEAN-NHP":args.pean,"IBGC":args.ibgc,"ET-BERT":args.etbert}
    rows=[];seen=set()
    for model,source in sources.items():
        rs=load_table(source)
        for item in rs:
            if any(not str(item.get(f,"")).strip() for f in FIELDS):
                raise ValueError(f"{source}: missing mandatory field in record {item}")
            model_value=item["model"]
            attack=item["attack"]
            strength=float(item["strength"]);seed=int(item["seed"])
            if model_value!=model or (model,attack) not in EXPECTED or strength not in EXPECTED[(model,attack)] or seed not in SEEDS:
                raise ValueError(f"{source}: unexpected model/attack/strength/seed {item}")
            key=model,attack,strength,seed
            if key in seen:raise ValueError(f"Duplicate measured record {key}")
            seen.add(key)
            if item["split_sha256"].lower()!=SPLIT:
                raise ValueError(f"Frozen split mismatch {key}")
            acc=float(item["robust_acc"])
            if not 0.<=acc<=100.:
                raise ValueError(f"Invalid robust accuracy {key}: {acc}")
            for k in ("checkpoint_sha256","attack_manifest_sha256"):
                if not valid_sha(item[k]):raise ValueError(f"Invalid provenance SHA {key}/{k}")
            ckpts=CHECKPOINT_IDS[model]
            if seed in ckpts and ckpts[seed]!=item["checkpoint_sha256"]:
                raise ValueError(f"Different checkpoint reused within {model}/seed{seed}")
            ckpts[seed]=item["checkpoint_sha256"]
            artifact=Path(item["evaluation_artifact"])
            if not args.allow_missing_artifacts and not artifact.is_file():
                raise ValueError(f"Missing evaluation source artifact {key}: {artifact}")
            if artifact.is_file() and artifact.suffix==".json":
                evidence=json.loads(artifact.read_text(encoding="utf-8"))
                for k in ("model","attack","seed","checkpoint_sha256","split_sha256"):
                    if str(evidence.get(k))!=str(item[k]):
                        raise ValueError(f"Source JSON mismatch {key}/{k}")
            rows.append({k:item[k] for k in FIELDS})
    expected={
        (model,attack,s,seed)
        for (model,attack),strengths in EXPECTED.items()
        for s in strengths for seed in SEEDS
    }
    if seen!=expected:
        missing=sorted(expected-seen);extra=sorted(seen-expected)
        raise RuntimeError(f"Incomplete Figure6 GPU observations: {len(missing)} missing "
                           f"(first 10 {missing[:10]}), extra={extra[:10]}")
    assert len(rows)==45
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Output already exists; refusing to overwrite previously validated supplement: "+str(args.output))
    with args.output.open("x",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows,key=lambda v:(v["model"],v["attack"],float(v["strength"]),int(v["seed"]))))
    report=args.output.with_suffix(".provenance.json")
    manifest=dict(dataset="CipherSpectrum",split_sha256=SPLIT,
                  seeds=SEEDS,count=45,sources={m:dict(path=str(path),sha256=sha(path))
                                                for m,path in sources.items()},
                  supplement_sha256=sha(args.output),
                  checkpoint_sha256_by_model_seed=CHECKPOINT_IDS,
                  missing_artifacts_allowed=bool(args.allow_missing_artifacts),
                  note="Official attack protocol/checkpoint consistency must also be confirmed "
                       "by each baseline's own evaluator and frozen endpoint checks.")
    report.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),
                      encoding="utf-8")
    print("VERIFIED_ALL_45_OBSERVATIONS",args.output)
    print("SHA256",sha(args.output))
    print("PROVENANCE",report)
if __name__=="__main__":main()