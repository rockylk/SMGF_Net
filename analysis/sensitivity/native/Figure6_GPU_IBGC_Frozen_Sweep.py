#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen IBGC V17 inference-only CipherSpectrum strength sweep (NO training).

Source: ibgc_model.py, ibgc_graph.py (exact original hashes), and
baseline_v17_common.py (original attacks), checked against run_manifest.json.
Prediction loop follows ibgc_engine.predict (original source hash checked).
Stops unless Clean, Mixed@1 and Adaptive@1 reproduce the frozen per-seed
accuracy AND macro-F1 values. Missing raw/test source or corrupt synced files
cause hard failure. This program never touches the formal run directory.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import baseline_v17_common as C
import ibgc_graph as G

SEEDS=(42,43,44)
STRENGTHS={"mixed":(.3,.6,.9,1.0),"adaptive":(.4,.7,1.0)}
INTERMEDIATE={"mixed":(.3,.6,.9),"adaptive":(.4,.7)}
SPLIT="01e3b572e8514a238b3733e224161acd1112b79b56f038635be9339b23cfad6b"
CHECKED_SEEDS={"mixed":1309999712,"adaptive":1687021782}
REP="IBGC-V17-v1-13features-fullRaw400-intSize-noFilter-noPad-noIAT"
FILES=("ibgc_model.py","ibgc_engine.py","ibgc_graph.py","baseline_v17_common.py")
EXPECTED_CODE={
 "ibgc_model.py":"cdb0da54a6c8ce8729d7745529907a25a46893e900b2592dd9e67fcda9815275",
 "ibgc_engine.py":"a3722a17998d02f8b0004e1a8c00a4ff29420441c310e00bc906fe46dcd7811b",
 "ibgc_graph.py":"794ed2f8b4d01812f39f689670ee76b0c7586a924187aaa5c446faea5ce657b7",
 "baseline_v17_common.py":"5cac4313cc4991b3de5fd4e1025b940c01249148b67fb2c3d9cb8218650a99cf",
}
def check(ok,msg):
 if not ok:raise RuntimeError(msg)
def sha(p):return C.sha256_file(p)
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def atomic_json(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(mode="w",suffix=".tmp",dir=str(p.parent),
                                  encoding="utf-8",delete=False) as f:
  json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)
  tmp=Path(f.name)
 os.replace(tmp,p)

def identity_checks(formal):
 manifest=read(formal/"run_manifest.json")
 check(manifest["baseline"]=="IBGC_V17_V1_100EP","Wrong baseline run")
 check(manifest["seeds"]==list(SEEDS) and manifest["attack_base_seed"]==20260919,
       "Frozen seeds/attack base seed drift")
 prep=read(formal/"cipher"/"preparation.json")
 identity=prep["identity"]
 check(identity["dataset"]=="CipherSpectrum" and identity["num_classes"]==41,
       "Frozen dataset/class mismatch")
 check(identity["split_sha256"]==SPLIT and identity["representation"]==REP,
       "Frozen split/representation mismatch")
 check(identity["common_sha256"]==EXPECTED_CODE["baseline_v17_common.py"],
       "Frozen common code mismatch")
 for fn in FILES:
  p=HERE/fn
  check(p.is_file() and sha(p)==EXPECTED_CODE[fn],
        f"Original IBGC source missing/modified: {p}")
  check(manifest["code"][fn]==EXPECTED_CODE[fn],f"Manifest code drift: {fn}")
 check(identity["graph_code_sha256"]==EXPECTED_CODE["ibgc_graph.py"],
       "Graph builder not attested by preparation metadata")
 check(C.ATTACK_BASE_SEED==20260919 and G.REPRESENTATION==REP,
       "Loaded attack or representation implementation changed")
 for attack,seed in CHECKED_SEEDS.items():
  check(C.canonical_attack_seed(attack,1.0)==seed,
        f"Frozen attack seed mismatch for {attack}")
 for seed in SEEDS:
  mdir=formal/"cipher"/"models"/f"seed{seed}"
  summ=read(mdir/"training_summary.json")
  ckpt=mdir/"best_validation.pt"
  check(summ["completed_epochs"]==100,"Training budget drift")
  check(ckpt.is_file() and sha(ckpt)==summ["best_sha256"],
        f"Missing/altered best checkpoint seed{seed}")
  orig=read(formal/"cipher"/"evaluations"/f"seed{seed}"/"evaluation.json")
  check(orig["checkpoint_sha256"]==sha(ckpt),
        f"Published evaluation checkpoint mismatch seed{seed}")
  check(orig["completed_epochs"]==100 and orig["identity"]["seed"]==seed,
        "Frozen evaluation manifest mismatch")
 return prep

def raw_test(args,prep):
 expected=prep["identity"]["raw_cache_sha256"]["test"]
 if args.raw_test_cache:
  cache=args.raw_test_cache.resolve()
  check(cache.is_file() and sha(cache)==expected,
        f"Original frozen test raw cache absent or SHA drift: {cache}")
  seqs,y,ids=C.load_raw_partition(cache)
  origin={"source":"frozen_raw_cache","path":str(cache),"sha256":expected}
 else:
  check(args.dataset_root and args.split_manifest,
        "Provide --raw-test-cache (preferable) OR both --dataset-root "
        "and --split-manifest to rebuild raw input from frozen PCAPs")
  split=args.split_manifest.resolve()
  check(split.is_file() and sha(split)==SPLIT,"Split manifest SHA mismatch")
  source=read(split)
  rows=source["test"]
  check(len(rows)==8200,"Frozen test population changed")
  seqs=[];y=[];ids=[]
  for i,r in enumerate(rows):
   pc=args.dataset_root.resolve()/r["relative_path"]
   check(pc.is_file(),f"Missing raw frozen PCAP {pc}")
   if "file_size" in r:
    check(pc.stat().st_size==int(r["file_size"]),
          f"Frozen PCAP file-size mismatch {pc}")
   seqs.append(C.extract_raw_x_from_pcap(pc))
   y.append(int(r["label"]))
   ids.append(int(r["sample_index"]))
   if (i+1)%1000==0:print(f"[IBGC] PCAP extract {i+1}/8200",flush=True)
  y=np.array(y,dtype=np.int64);ids=np.array(ids,dtype=np.int64)
  origin={"source":"frozen_pcaps","path":str(args.dataset_root.resolve()),
          "split_manifest":str(split),"split_sha256":SPLIT}
 check(len(seqs)==8200 and y.shape==(8200,) and ids.shape==(8200,),
       "Raw test cache population/shape mismatch")
 check(len(set(int(v) for v in ids))==8200,"Nonunique frozen sample IDs")
 check(np.all((0<=y)&(y<41)),"Invalid frozen class labels")
 for a in seqs:G.validate_raw(a)
 return seqs,y,ids,origin

def graph_dataset(raw,labels,ids):
 import torch
 from torch_geometric.data import Data
 class GraphSamples:
  def __len__(self):return len(raw)
  def __getitem__(self,i):
   x,e=G.build_graph(raw[i])
   return Data(x=torch.from_numpy(x),edge_index=torch.from_numpy(e),
               y=torch.tensor(int(labels[i]),dtype=torch.long),
               sample_id=torch.tensor(int(ids[i]),dtype=torch.long))
 return GraphSamples()

def graph_audit(seqs,expected=None):
 nodes=edges=0
 for i,x in enumerate(seqs):
  features,e=G.build_graph(x)
  nodes+=len(features);edges+=e.shape[1]
 if expected is not None:
  check(nodes==expected["nodes"] and edges==expected["directed_edges"],
        f"Graph audit != original frozen graph: nodes={nodes} edges={edges}, "
        f"expected={expected['nodes']}/{expected['directed_edges']}")
 return dict(n=len(seqs),nodes=nodes,directed_edges=edges)

def prediction(model,torch,data,device):
 # Exact eval forward/data loader/exp loop as the uploaded frozen
 # ibgc_engine.predict(), without importing unavailable ibgc_io.py.
 from torch_geometric.loader import DataLoader
 model.eval();ys=[];probs=[];idx=[]
 with torch.inference_mode():
  for b in DataLoader(data,batch_size=64,shuffle=False,num_workers=0,drop_last=False):
   b=b.to(device)
   logp=model(b.x,b.edge_index,b.batch)
   check(bool(torch.isfinite(logp).all()),"Nonfinite model outputs")
   probs.append(logp.exp().cpu().numpy())
   ys.append(b.y.cpu().numpy());idx.append(b.sample_id.cpu().numpy())
 y=np.concatenate(ys);p=np.concatenate(probs);ids=np.concatenate(idx)
 check(p.shape==(8200,41) and np.isfinite(p).all() and
       np.allclose(p.sum(1),1.,atol=1e-5),"IBGC probability shape/calibration invalid")
 return y,p,ids

def prepare(args,formal,prep):
 seqs,y,ids,origin=raw_test(args,prep)
 target=args.output/"prepared_raw"
 target.mkdir(parents=True,exist_ok=True)
 base=[("clean",0.0)]+[(a,s) for a,x in STRENGTHS.items() for s in x]
 manifest={"dataset":"CipherSpectrum","source":origin,"split_sha256":SPLIT,
           "frozen_code_sha256":EXPECTED_CODE,
           "conditions":{}}
 for attack,strength in base:
  cond="clean" if attack=="clean" else f"{attack}_{strength:.1f}"
  file=target/(f"test_{cond}.npz")
  if attack=="clean":seq=seqs
  else:seq=C.build_attacked_raw_dataset(seqs,attack,strength)
  if strength==1.0 or attack=="clean":
   key="test_clean" if attack=="clean" else f"test_{cond}"
   audit=graph_audit(seq,prep["partitions"][key]["audit"])
  else:audit=graph_audit(seq)
  if not file.exists():
   stage=file.with_name(file.stem+".tmp.npz")
   C.save_raw_partition(stage,seq,y,ids)
   os.replace(stage,file)
  reread,yl,il=C.load_raw_partition(file)
  check(np.array_equal(y,yl) and np.array_equal(ids,il)
        and len(reread)==8200,"Saved raw test cache invalid")
  check(all(np.array_equal(a,b) for a,b in zip(seq,reread)),
        "Cached attack differs from original frozen attack implementation")
  manifest["conditions"][cond]={"path":str(file),"sha256":sha(file),
           "strength":strength,"attack":attack,"audit":audit,
           "canonical_attack_seed":None if attack=="clean" else
                  C.canonical_attack_seed(attack,strength)}
  print(f"[IBGC][prepared] {cond} nodes={audit['nodes']} edges={audit['directed_edges']}",flush=True)
 atomic_json(args.output/"prepared_raw_manifest.json",manifest)
 return manifest


CSV_COLUMNS=("model","attack","strength","seed","robust_acc","robust_f1",
             "checkpoint_sha256","attack_manifest_sha256","split_sha256",
             "evaluation_artifact")

def export_observations(args,measurements):
 expected={(a,s,seed) for a,levels in INTERMEDIATE.items() for s in levels for seed in SEEDS}
 observed={(r["attack"],float(r["strength"]),int(r["seed"])) for r in measurements}
 check(len(measurements)==15 and observed==expected,
       "Refusing incomplete or duplicate IBGC intermediate results")
 output=args.output/"IBGC_Figure6_observed.csv"
 with tempfile.NamedTemporaryFile(mode="w",suffix=".tmp",dir=str(args.output),
                                  encoding="utf-8",newline="",delete=False) as f:
  writer=csv.DictWriter(f,fieldnames=CSV_COLUMNS)
  writer.writeheader()
  writer.writerows({col:r[col] for col in CSV_COLUMNS} for r in measurements)
  tmp=Path(f.name)
 try:
  if output.exists():
   with output.open("r",encoding="utf-8",newline="") as f:
    old=list(csv.reader(f))
   with tmp.open("r",encoding="utf-8",newline="") as f:
    new=list(csv.reader(f))
   if old==new:
    print("IBGC_FIG6_MEASUREMENTS_COMPLETE (verified existing CSV)",
          output,sha(output),flush=True)
    return
   check(old==[list(CSV_COLUMNS)],
         "Existing CSV has data or unexpected header; refusing overwrite")
  os.replace(tmp,output)
  print("IBGC_FIG6_MEASUREMENTS_COMPLETE",output,sha(output),flush=True)
 finally:
  if tmp.exists():tmp.unlink()

def export_saved_evidence(args,formal):
 """Recover completed inference from archived per-condition evidence only."""
 manifest=read(args.output/"prepared_raw_manifest.json")
 check(manifest["frozen_code_sha256"]==EXPECTED_CODE and
       manifest["split_sha256"]==SPLIT,"Prepared identity mismatch")
 dirs=manifest["conditions"]
 conditions=[("clean",0.)]+[(a,s) for a,v in STRENGTHS.items() for s in v]
 check(set(dirs)=={"clean" if a=="clean" else f"{a}_{s:.1f}"
                  for a,s in conditions},"Prepared conditions mismatch")
 result=[]
 for seed in SEEDS:
  ckpt=formal/"cipher"/"models"/f"seed{seed}"/"best_validation.pt"
  checkpoint_hash=sha(ckpt)
  original=read(formal/"cipher"/"evaluations"/f"seed{seed}"/"evaluation.json")
  check(original["checkpoint_sha256"]==checkpoint_hash,
        f"Frozen checkpoint mismatch seed{seed}")
  for attack,strength in conditions:
   cond="clean" if attack=="clean" else f"{attack}_{strength:.1f}"
   evidence=args.output/"measurements"/f"IBGC_seed{seed}_{cond}.json"
   check(evidence.is_file(),f"Missing GPU result evidence: {evidence}")
   r=read(evidence)
   check(r["model"]=="IBGC" and r["attack"]==attack
         and float(r["strength"])==strength and int(r["seed"])==seed
         and r["n"]==8200 and r["split_sha256"]==SPLIT
         and r["checkpoint_sha256"]==checkpoint_hash
         and r["attack_manifest_sha256"]==dirs[cond]["sha256"]
         and r["evaluation_artifact"]==str(evidence)
         and r["source_files_sha256"]==EXPECTED_CODE
         and r["native_model_attested"] is True
         and r["verified_against_frozen"] is (strength in (0.,1.)),
         f"Invalid archived GPU measurement: {evidence}")
   check(np.isfinite(float(r["robust_acc"])) and
         np.isfinite(float(r["robust_f1"])) and
         0<=float(r["robust_acc"])<=100 and
         0<=float(r["robust_f1"])<=100,
         f"Invalid archived score: {evidence}")
   if strength in (0.,1.):
    prior=original["metrics"][cond]
    acc=prior["acc"] if attack=="clean" else prior["robust_acc"]
    f1=prior["macro_f1"] if attack=="clean" else prior["robust_f1"]
    check(abs(float(r["robust_acc"])-acc)<1e-7 and
          abs(float(r["robust_f1"])-f1)<1e-4,
          f"Frozen endpoint mismatch seed{seed}/{cond}")
   elif strength in INTERMEDIATE[attack]:result.append(r)
 print("IBGC_RECOVERY_EVIDENCE_OK: 24 GPU JSON, 9 frozen endpoints, "
       "15 intermediate measurements",flush=True)
 export_observations(args,result)

def run(args,formal,prep):
 import torch
 from torch.torch_version import TorchVersion
 from ibgc_model import IBGC
 check(torch.cuda.is_available() or args.device=="cpu",
       "CUDA not available: use --device cpu for local testing")
 device=torch.device(args.device)
 manifestfile=args.output/"prepared_raw_manifest.json"
 check(manifestfile.is_file(),"Run --mode prepare first")
 manifest=read(manifestfile)
 check(manifest["frozen_code_sha256"]==EXPECTED_CODE,
       "Prepared source hashes changed")
 dirs=manifest["conditions"]
 orig={s:read(formal/"cipher"/"evaluations"/f"seed{s}"/"evaluation.json")
       for s in SEEDS}
 measurements=[]
 for seed in SEEDS:
  ckpt=formal/"cipher"/"models"/f"seed{seed}"/"best_validation.pt"
  # Frozen checkpoint includes trusted TorchVersion metadata.
  # Keep weights_only=True and allowlist only this specific class.
  with torch.serialization.safe_globals([TorchVersion]):
   payload=torch.load(ckpt,map_location="cpu",weights_only=True)
  check(payload["schema_version"]==1 and payload["epoch"]==orig[seed]["best_epoch"],
        "Selected best checkpoint version/epoch mismatch")
  check(payload["identity"]==orig[seed]["identity"],
        "Frozen checkpoint identity differs from published evaluation")
  model=IBGC(41).to(device)
  check(sum(v.numel() for v in model.parameters())==14921,
        "Not frozen IBGC architecture/parameter count")
  model.load_state_dict(payload["model_state_dict"],strict=True)
  model.eval()
  for attack,strength in [("clean",0.)]+[(a,s) for a,v in STRENGTHS.items() for s in v]:
   cond="clean" if attack=="clean" else f"{attack}_{strength:.1f}"
   info=dirs[cond]; path=Path(info["path"])
   check(path.is_file() and sha(path)==info["sha256"],
         f"Attacked raw test provenance mismatch {cond}")
   raw,y,ids=C.load_raw_partition(path)
   data=graph_dataset(raw,y,ids)
   yy,p,ii=prediction(model,torch,data,device)
   check(np.array_equal(yy,y) and np.array_equal(ii,ids),
         f"Predictions reordered frozen samples {cond}")
   metrics=C.classification_metrics(yy,p,41)
   tag="clean" if attack=="clean" else cond
   if strength in (0.,1.):
    previous=orig[seed]["metrics"][tag]
    target_acc=previous["acc"] if attack=="clean" else previous["robust_acc"]
    target_f1=previous["macro_f1"] if attack=="clean" else previous["robust_f1"]
    check(abs(metrics["acc"]-target_acc)<1e-7 and
          abs(metrics["macro_f1"]-target_f1)<1e-4,
          f"FAIL original frozen endpoint {seed}/{tag}: "
          f"got {metrics['acc']:.6f}/{metrics['macro_f1']:.6f}, "
          f"expected {target_acc:.6f}/{target_f1:.6f}. "
          "No intermediate results exported.")
   print(f"[IBGC] seed{seed} {cond} Acc={metrics['acc']:.6f} "
         f"MacroF1={metrics['macro_f1']:.6f}",flush=True)
   evidence=args.output/"measurements"/f"IBGC_seed{seed}_{cond}.json"
   record=dict(model="IBGC",attack=attack,strength=strength,seed=seed,
       robust_acc=metrics["acc"],robust_f1=metrics["macro_f1"],
       checkpoint_sha256=sha(ckpt),attack_manifest_sha256=info["sha256"],
       split_sha256=SPLIT,evaluation_artifact=str(evidence),
       n=8200,native_model_attested=True,verified_against_frozen=(strength in (0.,1.)),
       source_files_sha256=EXPECTED_CODE)
   if evidence.exists():
    check(read(evidence)==record,f"Resume evidence mismatch: {evidence}")
   else:atomic_json(evidence,record)
   if attack!="clean" and strength in INTERMEDIATE[attack]:
    measurements.append(record)
  del model
  if device.type=="cuda":torch.cuda.empty_cache()
 export_observations(args,measurements)

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument("--run-dir",type=Path,required=True)
 parser.add_argument("--output",type=Path,required=True)
 parser.add_argument("--raw-test-cache",type=Path)
 parser.add_argument("--dataset-root",type=Path)
 parser.add_argument("--split-manifest",type=Path)
 parser.add_argument("--device",default="cuda")
 parser.add_argument("--mode",choices=("preflight","prepare","run","all","export"),
                     default="all")
 args=parser.parse_args()
 formal=args.run_dir.resolve();args.output=args.output.resolve()
 check(args.output!=formal and formal not in args.output.parents,
       "Never write into original frozen baseline directory")
 prep=identity_checks(formal)
 if args.mode=="preflight":
  print("IBGC_PREFLIGHT_OK: three uploaded original files and frozen weights verified")
  return
 args.output.mkdir(parents=True,exist_ok=True)
 if args.mode=="export":
  export_saved_evidence(args,formal)
  return
 if args.mode in ("prepare","all"):prepare(args,formal,prep)
 if args.mode in ("run","all"):run(args,formal,prep)
if __name__=="__main__":main()