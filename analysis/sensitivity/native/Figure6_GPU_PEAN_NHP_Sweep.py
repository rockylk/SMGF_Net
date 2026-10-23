#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen PEAN-NHP CipherSpectrum intermediate-strength GPU evaluation.

Inferences ONLY. Reuses PEAN_NHP_final_data_V1_100ep.py's frozen physical
packet attack implementation and the 100-epoch validation-selected checkpoints.
Never trains, never overwrites original official evaluation/prepared results.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import PEAN_NHP_final_data_V1_100ep as D
import PEAN_NHP_final_paper_experiments_V1_100ep as P
import baseline_v17_common as C

SEEDS = (42, 43, 44)
ATTACKS = {"mixed": (.3, .6, .9), "adaptive": (.4, .7)}
SHA_SPLIT = "01e3b572e8514a238b3733e224161acd1112b79b56f038635be9339b23cfad6b"

def check(cond, msg):
    if not cond: raise RuntimeError(msg)

def sha(path):
    return D.sha256_file(path)

def atomic_npz(path, *, bytes_, lengths, labels, sample_indices):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), suffix=".npz",
                                     delete=False) as tmp:
        name = Path(tmp.name)
    try:
        np.savez_compressed(name, bytes=bytes_, lengths=lengths,
                            labels=labels, sample_indices=sample_indices)
        os.replace(name, path)
    finally:
        name.unlink(missing_ok=True)

def prepare_all(spec, formal, out, *, check_endpoints=True):
    from scapy.all import rdpcap
    preparation=P.load_prep(formal, spec)
    check(spec.expected_split_sha == SHA_SPLIT, "Split hash changed")
    checks=json.loads(spec.split_manifest.read_text(encoding="utf-8"))
    rows=checks["test"]
    check(len(rows) == 8200, "Unexpected CipherSpectrum test size")
    clean_x,clean_l,clean_y,clean_idx = D.load_condition(formal,spec,preparation,"test")
    conditions=[(a,s) for a,ss in ATTACKS.items() for s in ss]
    if check_endpoints: conditions += [("mixed",1.),("adaptive",1.)]
    cache_dir=out/"prepared"
    buffers={}
    for attack,strength in conditions:
        key=f"{attack}@{strength:.1f}"
        buffers[key]=dict(bytes_=np.empty_like(clean_x),
                          lengths=np.empty_like(clean_l))
    print(f"[PEAN-NHP] preparing {len(conditions)} attacked conditions "
          "from the frozen 8200 flow test set",flush=True)
    for i,r in enumerate(rows):
        check(int(r["sample_index"])==int(clean_idx[i]) and
              int(r["label"])==int(clean_y[i]),
              f"Frozen sample order/label mismatch at index {i}")
        p=spec.root/r["relative_path"]
        check(p.is_file(),f"Missing raw PCAP: {p}")
        meta=D.packet_metadata(rdpcap(str(p)))
        bx,bl,_=D.encode_metadata(meta)
        check(np.array_equal(bx,clean_x[i]) and np.array_equal(bl,clean_l[i]),
              f"Frozen PEAN-NHP clean representation mismatch index={i}")
        for attack,strength in conditions:
            key=f"{attack}@{strength:.1f}"
            seed=D.stable_seed(D.ATTACK_BASE_SEED,spec.name,attack,
                               f"{strength:.3f}",int(r["sample_index"]))
            altered=D.apply_attack_metadata(meta,attack,strength,seed)
            x,l,_=D.encode_metadata(altered)
            check(np.array_equal(x,bx),f"NHP byte-view changed {key}/{i}")
            buffers[key]["bytes_"][i]=x
            buffers[key]["lengths"][i]=l
        if (i+1)%1000==0: print(f"[PEAN-NHP][prepare] {i+1}/8200",flush=True)
    for key,b in buffers.items():
        path=cache_dir/f"test_{key.replace('@','_')}.npz"
        if path.is_file():
            with np.load(path,allow_pickle=False) as z:
                check(np.array_equal(z["bytes"],b["bytes_"]) and
                      np.array_equal(z["lengths"],b["lengths"]) and
                      np.array_equal(z["labels"],clean_y) and
                      np.array_equal(z["sample_indices"],clean_idx),
                      f"Existing sensitivity cache disagrees with newly derived {key}")
        else:
            atomic_npz(path,bytes_=b["bytes_"],lengths=b["lengths"],
                       labels=clean_y,sample_indices=clean_idx)
        print(f"[PEAN-NHP][prepared] {key} SHA256={sha(path)}")
        if key in ("mixed@1.0","adaptive@1.0"):
            official=(formal/"cipher"/"prepared"/f"test_{key.replace('@','_')}.npz")
            check(official.is_file(),"Missing archived reference "+str(official))
            with np.load(official,allow_pickle=False) as z:
                check(np.array_equal(z["bytes"],b["bytes_"]) and
                      np.array_equal(z["lengths"],b["lengths"]) and
                      np.array_equal(z["labels"],clean_y) and
                      np.array_equal(z["sample_indices"],clean_idx),
                      f"ENDPOINT INPUT MISMATCH for {key}. Abort to protect comparison.")
            print(f"[PEAN-NHP][canonical attack @1.0] {key} BITWISE MATCH PASS")
    return preparation,[(a,s) for a,ss in ATTACKS.items() for s in ss]

def infer_cache(torch, model, device, path, ncls, batch_size):
    from torch.utils.data import Dataset,DataLoader
    with np.load(path,allow_pickle=False) as z:
        x=z["bytes"].copy();l=z["lengths"].copy()
        y=z["labels"].astype(np.int64)
        idx=z["sample_indices"].astype(np.int64)
    class Arrays(Dataset):
        def __len__(self):return len(y)
        def __getitem__(self,i):
            return (torch.from_numpy(x[i].astype(np.int32)),
                    torch.from_numpy(l[i].astype(np.int32)),
                    int(y[i]),int(idx[i]))
    loader=DataLoader(Arrays(),batch_size=batch_size,shuffle=False,num_workers=0,
                      pin_memory=(device.type=="cuda"))
    labels,probs,ids=P.infer_condition(
        torch,model,loader,device,8200,ncls,str(path))
    check(np.array_equal(labels,y) and np.array_equal(ids,idx),
          "Model inference reordered samples/labels")
    return C.classification_metrics(labels,probs,ncls)

def run(args):
    import torch
    P.source_provenance()
    CWD=Path(args.work_root).resolve()
    result_root=Path(args.result_root).resolve()
    formal=Path(args.pean_run_dir).resolve()
    out=Path(args.output).resolve()
    check(out != formal and formal not in out.parents,
          "Output must be outside frozen official PEAN run directory")
    stub=argparse.Namespace(datasets="cipher")
    spec=P.build_specs(stub,CWD,result_root)[0]
    check(spec.name=="CipherSpectrum","Dataset configuration mismatch")
    check(spec.expected_split_sha==SHA_SPLIT,"Frozen Cipher split mismatch")
    check(torch.cuda.is_available() or args.device=="cpu",
          "GPU requested but CUDA is unavailable")
    device=P.choose_device(torch,args.device)
    prep=P.load_prep(formal,spec)
    for seed in SEEDS:
        summary_path=P.model_dir(formal,spec,seed)/"training_summary.json"
        check(summary_path.is_file(),f"Missing formal train summary: {summary_path}")
        train=json.loads(summary_path.read_text(encoding="utf-8"))
        ident=P.training_identity(spec,prep,seed)
        check(train["identity"]==ident,"Training identity mismatch")
        ckpt=P.model_dir(formal,spec,seed)/"best_validation.pt"
        check(ckpt.is_file() and sha(ckpt)==train["best_checkpoint_sha256"],
              f"Missing/SHA changed PEAN frozen checkpoint: {ckpt}")
    out.mkdir(parents=True, exist_ok=True)
    if args.mode in ("prepare","all"):
        prep,conditions=prepare_all(spec,formal,out,check_endpoints=True)
        if args.mode=="prepare":return
    conditions=[(a,s) for a,ss in ATTACKS.items() for s in ss]
    check(args.mode in ("evaluate","all"),"Unexpected mode")
    rows=[]
    for seed in SEEDS:
        P.set_seed(torch,seed)
        summary=json.loads((P.model_dir(formal,spec,seed)/
                            "training_summary.json").read_text(encoding="utf-8"))
        ckpt=P.model_dir(formal,spec,seed)/"best_validation.pt"
        payload=torch.load(ckpt,map_location="cpu",weights_only=True)
        check(payload.get("checkpoint_schema_version")==P.CHECKPOINT_SCHEMA_VERSION,
              "Checkpoint schema changed")
        check(payload.get("identity")==summary["identity"],
              "Checkpoint identity mismatch")
        from PEAN_NHP_V17_model import PEANNHPV17
        model=PEANNHPV17(spec.num_classes,device).to(device)
        model.load_state_dict(payload["model_state_dict"],strict=True)
        model.eval()
        for attack,strength in conditions:
            key=f"{attack}@{strength:.1f}"
            path=out/"prepared"/f"test_{attack}_{strength:.1f}.npz"
            check(path.is_file(),f"Prepared strength cache missing: {path}")
            metrics=infer_cache(torch,model,device,path,spec.num_classes,args.batch_size)
            result={"model":"PEAN-NHP","attack":attack,"strength":strength,
                    "seed":seed,"robust_acc":metrics["acc"],
                    "robust_f1":metrics["macro_f1"],
                    "checkpoint_sha256":sha(ckpt),
                    "attack_manifest_sha256":sha(path),
                    "split_sha256":SHA_SPLIT,
                    "evaluation_artifact":str(out/"measurements"/f"PEAN_NHP_seed{seed}_{attack}_{strength:.1f}.json")}
            path_json=out/"measurements"/f"PEAN_NHP_seed{seed}_{attack}_{strength:.1f}.json"
            path_json.parent.mkdir(parents=True,exist_ok=True)
            if path_json.exists():
                old=json.loads(path_json.read_text(encoding="utf-8"))
                check(old==result,f"Resume mismatch: {path_json}")
            else:
                temp=path_json.with_suffix(".json.tmp")
                temp.write_text(json.dumps(result,indent=2),encoding="utf-8")
                os.replace(temp,path_json)
            rows.append(result)
            print(f"[PEAN-NHP] seed{seed} {key} RobAcc={metrics['acc']:.6f}")
        del model
        if device.type=="cuda":torch.cuda.empty_cache()
    target=out/"PEAN_NHP_Figure6_observed.csv"
    cols=("model","attack","strength","seed","robust_acc","robust_f1",
          "checkpoint_sha256","attack_manifest_sha256",
          "split_sha256","evaluation_artifact")
    with target.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
    print("SAVED",target,"ROWS",len(rows))
    check(len(rows)==15,"Expected 15 seed/strength records")

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--work-root",required=True,help="SMGF_WORK_ROOT on GPU server")
    a.add_argument("--result-root",required=True,help="RESULT root on GPU server")
    a.add_argument("--pean-run-dir",required=True,help="Frozen PEAN_NHP_FINAL_PAPER_BASELINE_V1_100EP")
    a.add_argument("--output",required=True,help="New result folder, NOT inside official PEAN run")
    a.add_argument("--device",default="cuda")
    a.add_argument("--mode",choices=("prepare","evaluate","all"),default="all")
    a.add_argument("--batch-size",type=int,default=64)
    args=a.parse_args()
    run(args)
if __name__=="__main__":main()