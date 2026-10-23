# -*- coding: utf-8 -*-
"""PEAN-NHP V1 formal baseline under the frozen V17 protocol.

Datasets: CipherSpectrum (41 classes), archived 12-class USTC-TFC2016.
Seeds: 42,43,44.
Training: PEAN network/training setup under the project's frozen V17 comparison
budget with a No Header/Payload input constraint. The byte branch is constant
and traffic-independent; only the packet-length branch carries sample-dependent
traffic input. Architecture/training remain ensemble + TRF + --imploss True;
batch64; Adam lr1e-3 wd1e-3; gamma .992 at epoch start; EXACTLY 100 epochs;
NO early stopping. Checkpoint selection uses independent frozen validation ONLY.

Evaluation: Clean, Mixed@1.0, Adaptive@1.0. The byte view remains constant
under every condition; the length view is regenerated after the same physical
packet perturbation. Outputs probabilities
for Acc, Macro-F1, ECE, macro OvR AUROC, ASR, relative/absolute drop.

This script deliberately does NOT call upstream main.py because upstream uses
the held-out fold simultaneously as dev and test. Architecture dimensions remain aligned with upstream; official TRF.py is
imported unchanged via PEAN_NHP_V17_model.py. This is a constrained-input
PEAN variant, not native-input official PEAN.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

import PEAN_NHP_final_data_V1_100ep as D

SCRIPT=Path(__file__).resolve()
BASELINE_DIR=SCRIPT.parent
UPSTREAM=BASELINE_DIR/"official_sources"/"PEAN"
UPSTREAM_COMMIT="b159ec48f8e3a699d863d57a6e2dc331aa83e7cc"
SEEDS=(42,43,44)
MAX_EPOCHS=100
BATCH_SIZE=64
EVAL_BATCH_SIZE=64
LR=1e-3
WEIGHT_DECAY=1e-3
GAMMA=.992
REQUIRE_IMPROVEMENT=None  # V2 fixed 100-epoch protocol: early stopping disabled.
VALIDATE_EVERY_BATCHES=100
PRIMARY_STRENGTH=1.0
CHECKPOINT_SCHEMA_VERSION=2
ACTIVE_LOGGER=None

class Logger:
    def __init__(self,path:Path):
        self.path=path
        path.parent.mkdir(parents=True,exist_ok=True)
    def log(self,msg=""):
        msg=str(msg);print(msg,flush=True)
        with self.path.open("a",encoding="utf-8") as f:f.write(msg+"\n")

def require(cond,msg):
    if not cond:raise RuntimeError(msg)

def dump_json(path:Path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n",
                   encoding="utf-8")
    os.replace(tmp,path)

def sha256_file(path:Path):return D.sha256_file(path)

def mean_sd(vals):
    vals=[float(v) for v in vals]
    return dict(mean=statistics.mean(vals),
                sample_sd=statistics.stdev(vals) if len(vals)>1 else 0.0)

def source_provenance():
    files=("README.md","PEAN_model.py","TRF.py","main.py","config.py",
           "train_eval.py","utils.py","Config/vocab.txt")
    expected={
      "PEAN_model.py":"f752c445bf8b69064f7a740f2a9e80b0740c1672e3e9d0aa3bb8e75a94716886",
      "TRF.py":"927c864c0bfcf7033b04c0689f5d8df6c7a08693593b3a12ec5e17d9fa286e98",
      "Config/vocab.txt":"e829aff44464de461ce92775b092c47b1be8484da9db4cf20acdb3a1b1303384"}
    require(all((UPSTREAM/f).is_file() for f in files),"Incomplete official PEAN mirror")
    got={f:sha256_file(UPSTREAM/f) for f in files}
    for f,h in expected.items():require(got[f]==h,f"Official PEAN source drift: {f}")
    D.verify_vocabulary(UPSTREAM/"Config/vocab.txt")
    return dict(url="https://github.com/Lin-Dada/PEAN",commit=UPSTREAM_COMMIT,
                files_sha256=got)

def set_seed(torch,seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False

def choose_device(torch,text):
    if text=="auto":text="cuda" if torch.cuda.is_available() else "cpu"
    if text.startswith("cuda"):require(torch.cuda.is_available(),"CUDA unavailable")
    return torch.device(text)

def build_specs(args,work_root,result_root):
    import baseline_v17_common as C
    specs=C.build_dataset_specs(args.datasets,work_root,result_root,"PEAN_NHP_FINAL")
    for s in specs:C.validate_dataset_spec(s)
    return specs

def dataset_dir(run_dir,spec):return run_dir/spec.key
def model_dir(run_dir,spec,seed):return dataset_dir(run_dir,spec)/"models"/f"seed{seed}"
def pred_dir(run_dir,spec,seed):return dataset_dir(run_dir,spec)/"predictions"/f"seed{seed}"
def eval_file(run_dir,spec,seed):return dataset_dir(run_dir,spec)/"evaluations"/f"seed{seed}.json"
def prep_file(run_dir,spec):return D.prep_meta_path(run_dir,spec)

def load_prep(run_dir,spec):
    p=prep_file(run_dir,spec);require(p.is_file(),f"Missing preparation {p}")
    meta=json.loads(p.read_text(encoding="utf-8"))
    require(meta["split_sha256"]==spec.expected_split_sha
            and meta["dataset_manifest_sha256"]==spec.expected_dataset_manifest
            and meta["counts"]==spec.expected_counts
            and meta["num_classes"]==spec.num_classes
            and meta["protocol"]==D.PROTOCOL
            and meta["source_sha256"]==sha256_file(BASELINE_DIR/"PEAN_NHP_final_data_V1_100ep.py")
            and meta["vocab_sha256"]==sha256_file(UPSTREAM/"Config/vocab.txt"),
            f"{spec.name}: prepared metadata/source drift")
    return meta

def make_loader(torch,run_dir,spec,meta,condition,batch_size,shuffle,workers,
                epoch_seed=None):
    from torch.utils.data import Dataset,DataLoader
    x,l,y,idx=D.load_condition(run_dir,spec,meta,condition)
    class Arrays(Dataset):
        def __len__(self):return len(y)
        def __getitem__(self,i):
            return (torch.from_numpy(x[i].astype(np.int32)),
                    torch.from_numpy(l[i].astype(np.int32)),
                    int(y[i]),int(idx[i]))
    g=torch.Generator()
    g.manual_seed(int(epoch_seed if epoch_seed is not None else 0))
    return DataLoader(Arrays(),batch_size=batch_size,shuffle=shuffle,
                      num_workers=workers,pin_memory=torch.cuda.is_available(),
                      drop_last=False,generator=g)

def to_device(torch,batch,device):
    x,l,y,idx=batch
    return (x.to(device,non_blocking=True).long(),
            l.to(device,non_blocking=True).long(),
            y.to(device,non_blocking=True).long(),
            idx.cpu().numpy())

def finite_outputs(torch,outputs,where):
    for i,z in enumerate(outputs):
        require(bool(torch.isfinite(z).all().item()),
                f"{where}: nonfinite PEAN-NHP head {i}")

def build_model(torch,spec,device):
    from PEAN_NHP_V17_model import PEANNHPV17
    model=PEANNHPV17(spec.num_classes,device).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    return model,optimizer

def val_eval(torch,model,loader,device,expected_n):
    import torch.nn.functional as F
    model.eval();loss_sum=0.;correct=0;n=0
    with torch.inference_mode():
        for batch in loader:
            x,l,y,_=to_device(torch,batch,device)
            out=model(x,l);finite_outputs(torch,out,"validation")
            loss=F.cross_entropy(out[0],y)
            require(bool(torch.isfinite(loss).item()),"Validation CE nonfinite")
            loss_sum+=float(loss.item())*len(y)
            correct+=int((out[0].argmax(1)==y).sum().item());n+=len(y)
    require(n==expected_n,"Validation sample count drift")
    return loss_sum/n,100*correct/n

def training_identity(spec,meta,seed):
    return dict(dataset=spec.name,seed=seed,split_sha256=spec.expected_split_sha,
                dataset_manifest_sha256=spec.expected_dataset_manifest,
                train_cache_sha256=meta["cache_sha256"]["train_clean"],
                validation_cache_sha256=meta["cache_sha256"]["validation_clean"],
                runner_sha256=sha256_file(SCRIPT),
                data_source_sha256=sha256_file(BASELINE_DIR/"PEAN_NHP_final_data_V1_100ep.py"),
                model_source_sha256=sha256_file(BASELINE_DIR/"PEAN_NHP_V17_model.py"),
                upstream_trf_sha256=sha256_file(UPSTREAM/"TRF.py"))

def atomic_torch_save(torch,path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    torch.save(obj,tmp);os.replace(tmp,path)

def save_best(torch,path,model,identity,epoch,total_batch,val_ce,val_acc):
    payload=dict(checkpoint_schema_version=CHECKPOINT_SCHEMA_VERSION,
                 identity=identity,model_state_dict=model.state_dict(),epoch=epoch,
                 total_batch=total_batch,validation_fused_ce=val_ce,
                 validation_accuracy_pct=val_acc)
    atomic_torch_save(torch,path,payload)
    return sha256_file(path)

def save_latest(torch,path,model,opt,identity,state):
    payload=dict(checkpoint_schema_version=CHECKPOINT_SCHEMA_VERSION,
                 identity=identity,model_state_dict=model.state_dict(),
                 optimizer_state_dict=opt.state_dict(),**state)
    atomic_torch_save(torch,path,payload)

def train_one(torch,run_dir,spec,meta,seed,args,device,logger):
    from PEAN_NHP_V17_model import PEANNHPV17
    mdir=model_dir(run_dir,spec,seed);mdir.mkdir(parents=True,exist_ok=True)
    best_path=mdir/"best_validation.pt";latest_path=mdir/"latest_epoch_state.pt"
    summary_path=mdir/"training_summary.json"
    if summary_path.is_file():
        old=json.loads(summary_path.read_text(encoding="utf-8"))
        if old.get("status")=="training_completed" and old.get("identity")==training_identity(spec,meta,seed):
            require(best_path.is_file() and old["best_checkpoint_sha256"]==sha256_file(best_path),
                    f"{spec.name} seed{seed}: completed checkpoint SHA mismatch")
            logger.log(f"[PEAN-NHP][train][{spec.name}][seed={seed}] reuse completed")
            return old
        raise RuntimeError(f"{spec.name} seed{seed}: stale/incompatible training summary")

    set_seed(torch,seed)
    model,opt=build_model(torch,spec,device)
    identity=training_identity(spec,meta,seed)
    state=dict(completed_epoch=0,total_batch=0,last_improve=0,
               best_val=float("inf"),best_epoch=0,best_total_batch=0,
               validation_history=[],epoch_history=[],train_seconds_accum=0.0)
    if latest_path.is_file() and args.resume:
        payload=torch.load(latest_path,map_location="cpu",weights_only=True)
        require(payload.get("checkpoint_schema_version")==CHECKPOINT_SCHEMA_VERSION,
                f"{spec.name} seed{seed}: unsupported resume checkpoint schema")
        require(payload.get("identity")==identity,
                f"{spec.name} seed{seed}: resume provenance mismatch")
        model.load_state_dict(payload["model_state_dict"],strict=True)
        opt.load_state_dict(payload["optimizer_state_dict"])
        for k in state:state[k]=payload[k]
        logger.log(f"[PEAN-NHP][train][{spec.name}][seed={seed}] resume after epoch "
                   f"{state['completed_epoch']} total_batch={state['total_batch']}")
    elif latest_path.exists() or best_path.exists():
        raise RuntimeError(f"{spec.name} seed{seed}: existing checkpoint; use --resume or new run dir")

    if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
    wall_start=time.perf_counter()
    stopped_early=False
    max_epochs=args.max_epochs
    param_count=sum(p.numel() for p in model.parameters())
    logger.log(f"[PEAN-NHP][train][{spec.name}][seed={seed}] params={param_count} "
               f"fixedEpochs={max_epochs} batch={args.batch_size} "
               f"validationEvery={VALIDATE_EVERY_BATCHES} earlyStopping=DISABLED")

    for epoch in range(state["completed_epoch"]+1,max_epochs+1):
        ep_start=time.perf_counter()
        if opt.param_groups[0]["lr"]>1e-5:
            opt.param_groups[0]["lr"]*=GAMMA
        lr=float(opt.param_groups[0]["lr"])
        train_loader=make_loader(torch,run_dir,spec,meta,"train",args.batch_size,
                                 True,args.workers,epoch_seed=seed*100000+epoch)
        val_loader=make_loader(torch,run_dir,spec,meta,"validation",
                               args.eval_batch_size,False,args.workers,epoch_seed=0)
        model.train();loss_sum=0.;correct=0;n=0
        for local_step,batch in enumerate(train_loader):
            x,l,y,_=to_device(torch,batch,device)
            opt.zero_grad(set_to_none=True)
            outputs=model(x,l);finite_outputs(torch,outputs,"training")
            loss=PEANNHPV17.supervised_loss(outputs,y)
            require(bool(torch.isfinite(loss).item()),
                    f"{spec.name} seed{seed}: nonfinite train loss")
            loss.backward()
            for name,p in model.named_parameters():
                if p.grad is not None:
                    require(bool(torch.isfinite(p.grad).all().item()),
                            f"{spec.name} seed{seed}: nonfinite gradient {name}")
            opt.step()
            loss_sum+=float(loss.item())*len(y)
            correct+=int((outputs[0].argmax(1)==y).sum().item());n+=len(y)
            tb=int(state["total_batch"])
            if tb%VALIDATE_EVERY_BATCHES==0:
                val_ce,val_acc=val_eval(torch,model,val_loader,device,
                                        spec.expected_counts["validation"])
                event=dict(epoch=epoch,total_batch=tb,
                           train_batch_loss=float(loss.item()),
                           validation_fused_ce=val_ce,
                           validation_accuracy_pct=val_acc,lr=lr)
                improved=val_ce<float(state["best_val"])
                event["improved"]=bool(improved)
                if improved:
                    state["best_val"]=val_ce;state["best_epoch"]=epoch
                    state["best_total_batch"]=tb;state["last_improve"]=tb
                    event["checkpoint_sha256"]=save_best(
                        torch,best_path,model,identity,epoch,tb,val_ce,val_acc)
                state["validation_history"].append(event)
                logger.log(f"[PEAN-NHP][val][{spec.name}][seed={seed}] epoch={epoch} "
                           f"batch={tb} CE={val_ce:.5f} Acc={val_acc:.2f}% "
                           f"{'*' if improved else ''}")
                model.train()
            state["total_batch"]=tb+1
        epoch_seconds=time.perf_counter()-ep_start
        require(n>0,"Empty PEAN train loader")
        state["completed_epoch"]=epoch
        state["train_seconds_accum"]+=epoch_seconds
        state["epoch_history"].append(dict(
            epoch=epoch,lr=lr,train_loss_total3ce=loss_sum/n,
            train_accuracy_pct=100*correct/n,epoch_seconds=epoch_seconds,
            examples_seen=n,examples_per_second=n/epoch_seconds,
            total_batch=state["total_batch"],best_validation_fused_ce=state["best_val"],
            best_epoch=state["best_epoch"],best_total_batch=state["best_total_batch"],
            early_stop_triggered=stopped_early))
        save_latest(torch,latest_path,model,opt,identity,state)
        dump_json(mdir/"training_progress.json",
                  dict(status="training",identity=identity,**state,
                       peak_gpu_allocated_mb=(torch.cuda.max_memory_allocated(device)/1048576
                                              if device.type=="cuda" else None)))
        logger.log(f"[PEAN-NHP][epoch][{spec.name}][seed={seed}] {epoch}/{max_epochs} "
                   f"train3CE={loss_sum/n:.5f} trainAcc={100*correct/n:.2f}% "
                   f"time={epoch_seconds:.2f}s bestValCE={state['best_val']:.5f} "
                   f"bestEpoch={state['best_epoch']} totalBatch={state['total_batch']}")

    require(best_path.is_file() and state["best_epoch"]>0,"No validation-selected checkpoint")
    # Include wall segment separately; accumulated epoch time is resume-safe.
    peak=(torch.cuda.max_memory_allocated(device)/1048576 if device.type=="cuda" else None)
    result=dict(status="training_completed",identity=identity,
                pean_network_training_config="ensemble/trf/random/imploss=True",
                input_variant="PEAN-NHP: constant no-header/no-payload byte branch + packet-length branch",
                max_epochs=max_epochs,completed_epoch=state["completed_epoch"],
                early_stopped=False,total_batch=state["total_batch"],
                best_epoch=state["best_epoch"],best_total_batch=state["best_total_batch"],
                best_validation_fused_ce=state["best_val"],
                best_checkpoint_sha256=sha256_file(best_path),
                train_seconds_accum=state["train_seconds_accum"],
                current_process_wall_seconds=time.perf_counter()-wall_start,
                trainable_parameters=param_count,peak_gpu_allocated_mb=peak,
                validation_events=len(state["validation_history"]),
                config=dict(batch_size=args.batch_size,eval_batch_size=args.eval_batch_size,
                            lr=LR,weight_decay=WEIGHT_DECAY,gamma=GAMMA,
                            validation_every_batches=VALIDATE_EVERY_BATCHES,
                            early_stopping=False,
                            fixed_epoch_budget=MAX_EPOCHS,
                            training_order="deterministic epoch-shuffled frozen train only",
                            checkpoint_selection="minimum fused validation CE"))
    dump_json(summary_path,result)
    # Compact CSVs for training audit.
    if state["epoch_history"]:
        with (mdir/"epoch_metrics.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(state["epoch_history"][0]))
            w.writeheader();w.writerows(state["epoch_history"])
    if state["validation_history"]:
        fields=[]
        for e in state["validation_history"]:
            for k in e:
                if k not in fields:fields.append(k)
        with (mdir/"validation_events.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
            w.writeheader();w.writerows(state["validation_history"])
    logger.log(f"[PEAN-NHP][train][{spec.name}][seed={seed}] DONE bestEpoch="
               f"{result['best_epoch']} bestBatch={result['best_total_batch']} "
               f"trainSec={result['train_seconds_accum']:.1f} peakMB={peak}")
    return result

def infer_condition(torch,model,loader,device,expected_n,num_classes,where):
    ys=[];ps=[];ids=[]
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            x,l,y,idx=to_device(torch,batch,device)
            out=model(x,l);finite_outputs(torch,out,where)
            prob=torch.softmax(out[0],1).cpu().numpy()
            require(np.isfinite(prob).all(),f"{where}: nonfinite probabilities")
            ys.append(y.cpu().numpy());ps.append(prob);ids.append(idx)
    y=np.concatenate(ys).astype(np.int64)
    p=np.concatenate(ps).astype(np.float64)
    idx=np.concatenate(ids).astype(np.int64)
    require(len(y)==expected_n and p.shape==(expected_n,num_classes),
            f"{where}: inference population/shape drift")
    require(np.allclose(p.sum(1),1.,atol=2e-5),f"{where}: softmax normalization drift")
    return y,p,idx

def evaluate_seed(torch,run_dir,spec,meta,seed,args,device,logger):
    import baseline_v17_common as C
    from PEAN_NHP_V17_model import PEANNHPV17
    training=json.loads((model_dir(run_dir,spec,seed)/"training_summary.json").read_text(encoding="utf-8"))
    identity=training_identity(spec,meta,seed)
    require(training["identity"]==identity,"Training/evaluation identity mismatch")
    ckpt=model_dir(run_dir,spec,seed)/"best_validation.pt"
    require(ckpt.is_file() and sha256_file(ckpt)==training["best_checkpoint_sha256"],
            "Best checkpoint SHA mismatch")
    payload=torch.load(ckpt,map_location="cpu",weights_only=True)
    require(payload.get("checkpoint_schema_version")==CHECKPOINT_SCHEMA_VERSION,
            "Unsupported PEAN-NHP checkpoint schema; evaluation aborted")
    require(payload.get("identity")==identity,"Checkpoint provenance mismatch")
    model=PEANNHPV17(spec.num_classes,device).to(device)
    for name,t in payload["model_state_dict"].items():
        require(bool(torch.isfinite(t).all().item()),f"Checkpoint nonfinite: {name}")
    model.load_state_dict(payload["model_state_dict"],strict=True)

    conditions=("test","mixed_1.0","adaptive_1.0")
    evaluations={};predictions={}
    reference_ids=None;reference_y=None
    for condition in conditions:
        loader=make_loader(torch,run_dir,spec,meta,condition,args.eval_batch_size,
                           False,args.workers,epoch_seed=0)
        started=time.perf_counter()
        y,p,idx=infer_condition(torch,model,loader,device,
                                spec.expected_counts["test"],spec.num_classes,
                                f"{spec.name}/seed{seed}/{condition}")
        if reference_ids is None:reference_ids=idx;reference_y=y
        else:
            require(np.array_equal(reference_ids,idx) and np.array_equal(reference_y,y),
                    "Attack sample/label order differs from clean")
        metrics=C.classification_metrics(y,p,spec.num_classes)
        require(np.isfinite(p).all(),"Metrics received nonfinite PEAN-NHP probabilities")
        key="clean" if condition=="test" else condition
        evaluations[key]=C.compact_eval(metrics)
        evaluations[key]["inference_seconds"]=time.perf_counter()-started
        predictions[key]=metrics
        pdir=pred_dir(run_dir,spec,seed);pdir.mkdir(parents=True,exist_ok=True)
        npz=pdir/f"{key}.npz"
        np.savez_compressed(npz,sample_indices=idx,labels=y,
                            preds=metrics["preds"],probs=p.astype(np.float32))
        evaluations[key]["prediction_sha256"]=sha256_file(npz)
        logger.log(f"[PEAN-NHP][eval][{spec.name}][seed={seed}][{key}] "
                   f"Acc={metrics['acc']:.2f} F1={metrics['macro_f1']:.2f} "
                   f"ECE={metrics['ece']:.2f} AUROC={metrics['macro_auroc']}")
    for key in ("mixed_1.0","adaptive_1.0"):
        evaluations[key].update(C.attack_metrics(predictions["clean"],predictions[key]))
    result=dict(dataset=spec.name,seed=seed,status="evaluation_completed",
                identity=identity,best_epoch=training["best_epoch"],
                best_total_batch=training["best_total_batch"],
                checkpoint_sha256=training["best_checkpoint_sha256"],
                clean=evaluations["clean"],mixed_1_0=evaluations["mixed_1.0"],
                adaptive_1_0=evaluations["adaptive_1.0"])
    dump_json(eval_file(run_dir,spec,seed),result)
    return result

def flatten_eval(obj):
    return dict(
        dataset=obj["dataset"],seed=obj["seed"],best_epoch=obj["best_epoch"],
        best_total_batch=obj["best_total_batch"],
        clean_acc=obj["clean"]["acc"],clean_macro_f1=obj["clean"]["macro_f1"],
        clean_ece=obj["clean"]["ece"],clean_macro_auroc=obj["clean"]["macro_auroc"],
        mixed_robust_acc=obj["mixed_1_0"]["robust_acc"],
        mixed_robust_f1=obj["mixed_1_0"]["robust_f1"],
        mixed_asr=obj["mixed_1_0"]["asr"],mixed_rel_drop=obj["mixed_1_0"]["rel_drop"],
        mixed_abs_drop_pp=obj["mixed_1_0"]["abs_drop_pp"],
        mixed_ece=obj["mixed_1_0"]["ece"],
        mixed_macro_auroc=obj["mixed_1_0"]["macro_auroc"],
        adaptive_robust_acc=obj["adaptive_1_0"]["robust_acc"],
        adaptive_robust_f1=obj["adaptive_1_0"]["robust_f1"],
        adaptive_asr=obj["adaptive_1_0"]["asr"],
        adaptive_rel_drop=obj["adaptive_1_0"]["rel_drop"],
        adaptive_abs_drop_pp=obj["adaptive_1_0"]["abs_drop_pp"],
        adaptive_ece=obj["adaptive_1_0"]["ece"],
        adaptive_macro_auroc=obj["adaptive_1_0"]["macro_auroc"])

def aggregate(run_dir,specs,seeds,logger):
    per=[]
    summary={}
    for spec in specs:
        rows=[]
        for seed in seeds:
            p=eval_file(run_dir,spec,seed)
            require(p.is_file(),f"Missing evaluation {p}")
            obj=json.loads(p.read_text(encoding="utf-8"))
            require(obj.get("status")=="evaluation_completed","Invalid evaluation status")
            row=flatten_eval(obj);per.append(row);rows.append(row)
        metric_names=[k for k in rows[0] if k not in ("dataset","seed")]
        # best_epoch/batch are diagnostic, still aggregate but clearly marked.
        agg={m:mean_sd([r[m] for r in rows]) for m in metric_names
             if all(r[m] is not None for r in rows)}
        summary[spec.key]=dict(dataset=spec.name,seeds=list(seeds),per_seed=rows,
                               aggregate=agg)
    per_path=run_dir/"PEAN_NHP_per_seed_primary.csv"
    with per_path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(per[0]));w.writeheader();w.writerows(per)
    agg_rows=[]
    for key,obj in summary.items():
        for metric,stat in obj["aggregate"].items():
            agg_rows.append(dict(dataset=obj["dataset"],metric=metric,
                                 mean=stat["mean"],sample_sd=stat["sample_sd"]))
    with (run_dir/"PEAN_NHP_aggregate_primary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["dataset","metric","mean","sample_sd"])
        w.writeheader();w.writerows(agg_rows)
    dump_json(run_dir/"summary.json",summary)
    lines=["# PEAN-NHP V1 Final Baseline","",
           "Three seeds; values are mean ± sample SD. Clean and primary attacks only.","",
           "| Dataset | Clean Acc | Clean Macro-F1 | Mixed@1 RobAcc | Mixed ASR | Adaptive@1 RobAcc | Adaptive ASR |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for obj in summary.values():
        a=obj["aggregate"]
        fmt=lambda m:f"{a[m]['mean']:.2f} ± {a[m]['sample_sd']:.2f}"
        lines.append(f"| {obj['dataset']} | {fmt('clean_acc')} | {fmt('clean_macro_f1')} | "
                     f"{fmt('mixed_robust_acc')} | {fmt('mixed_asr')} | "
                     f"{fmt('adaptive_robust_acc')} | {fmt('adaptive_asr')} |")
    lines+=["","PEAN-NHP retains PEAN network dimensions and training settings, but the byte branch is "
             "traffic-independent [CLS,SEP,PAD...] and exposes no TCP/UDP header or payload bytes.",
             "Checkpoint selected only on frozen validation fused CE.",
             "Mixed/Adaptive modify the physical packet sequence; only the length view can change."]
    (run_dir/"PAPER_READY_PEAN_NHP_BASELINE.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    logger.log("[PEAN-NHP][aggregate] wrote summary/CSVs/PAPER_READY_PEAN_NHP_BASELINE.md")
    return summary

def write_manifest(run_dir,args,specs,provenance):
    import torch
    obj=dict(
        baseline="PEAN-NHP V1 constrained-input PEAN (No Header/Payload)",
        upstream=provenance,runner_sha256=sha256_file(SCRIPT),
        data_source_sha256=sha256_file(BASELINE_DIR/"PEAN_NHP_final_data_V1_100ep.py"),
        model_source_sha256=sha256_file(BASELINE_DIR/"PEAN_NHP_V17_model.py"),
        common_sha256=sha256_file(BASELINE_DIR/"baseline_v17_common.py"),
        seeds=list(args.seed_values),datasets=[
            dict(name=s.name,key=s.key,num_classes=s.num_classes,
                 root=str(s.root),split_manifest=str(s.split_manifest),
                 split_sha256=s.expected_split_sha,
                 dataset_manifest_sha256=s.expected_dataset_manifest,
                 counts=s.expected_counts) for s in specs],
        training=dict(max_epochs=args.max_epochs,batch_size=args.batch_size,
                      eval_batch_size=args.eval_batch_size,lr=LR,
                      weight_decay=WEIGHT_DECAY,gamma=GAMMA,
                      validation_every_train_batches=VALIDATE_EVERY_BATCHES,
                      early_stopping=False,
                      fixed_epoch_budget=MAX_EPOCHS,
                      checkpoint_selection="minimum independent validation fused CE",
                      objective="CE_fused + CE_byte + CE_length",
                      configuration="ensemble/trf/random/imploss=True",
                      input_constraint="byte branch constant and traffic-independent; packet lengths only"),
        evaluation=dict(conditions=["clean","mixed@1.0","adaptive@1.0"],
                        metrics=["Accuracy","Macro-F1","ECE15","macro-OvR-AUROC",
                                 "ASR","relative drop","absolute drop pp"]),
        attack=dict(base_seed=D.ATTACK_BASE_SEED,
                    physical_family="padding+IAT jitter+dummy insertion",
                    note="PEAN-NHP exposes no model-visible header/payload bytes. Timing jitter "
                         "does not directly enter the model; byte view is constant and only "
                         "the packet-length view can change under physical attacks."),
        versions=dict(python=os.sys.version,torch=torch.__version__,numpy=np.__version__),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    path=run_dir/"run_manifest.json"
    if path.is_file():
        old=json.loads(path.read_text(encoding="utf-8"))
        # Stagewise prepare/train/evaluate/aggregate may re-enter this function.
        # Do not silently rewrite a formal run with a different scope/config.
        immutable=("baseline","upstream","runner_sha256","data_source_sha256",
                   "model_source_sha256","common_sha256","seeds","datasets",
                   "training","evaluation","attack")
        require(all(old.get(k)==obj.get(k) for k in immutable),
                "Existing PEAN-NHP run_manifest differs from requested formal run; "
                "use a NEW --run-dir")
        return
    dump_json(path,obj)

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("prepare","train","evaluate","aggregate","all"),
                   default="all")
    p.add_argument("--datasets",choices=("cipher","ustc","both"),default="both")
    p.add_argument("--seeds",default="42,43,44")
    p.add_argument("--result-root",type=Path,default=Path(os.environ.get(
        "PEAN_NHP_RESULT_ROOT",str(Path.cwd()/"RESULT"))))
    p.add_argument("--run-dir",type=Path,default=None)
    p.add_argument("--device",default="auto")
    p.add_argument("--max-epochs",type=int,default=MAX_EPOCHS)
    p.add_argument("--batch-size",type=int,default=BATCH_SIZE)
    p.add_argument("--eval-batch-size",type=int,default=EVAL_BATCH_SIZE)
    p.add_argument("--workers",type=int,default=0)
    p.add_argument("--resume",action=argparse.BooleanOptionalAction,default=True)
    return p.parse_args()

def main():
    args=parse_args()
    seeds=tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())
    require(seeds and set(seeds).issubset(SEEDS) and len(set(seeds))==len(seeds),
            "Formal seed values must be unique subset of 42,43,44")
    args.seed_values=seeds
    require(args.max_epochs==MAX_EPOCHS,
            "Formal PEAN-NHP fixes the project comparison budget at exactly 100 epochs")
    require(args.batch_size==BATCH_SIZE,
            "Formal PEAN-NHP run fixes PEAN batch=64")
    work_root=Path(os.environ.get("SMGF_WORK_ROOT",str(Path.cwd()))).resolve()
    result_root=args.result_root.expanduser().resolve()
    run_dir=(args.run_dir.expanduser().resolve() if args.run_dir else
             result_root/"PEAN_NHP_FINAL_PAPER_BASELINE_V1_100EP")
    run_dir.mkdir(parents=True,exist_ok=True)
    logger=Logger(run_dir/"experiment.log")
    global ACTIVE_LOGGER
    ACTIVE_LOGGER=logger
    import torch
    device=choose_device(torch,args.device)
    provenance=source_provenance()
    specs=build_specs(args,work_root,result_root)
    write_manifest(run_dir,args,specs,provenance)
    logger.log(f"[PEAN-NHP] mode={args.mode} datasets={args.datasets} seeds={seeds} "
               f"run={run_dir} device={device}")
    if args.mode in ("prepare","all"):
        for spec in specs:
            D.prepare_dataset(spec,run_dir,UPSTREAM/"Config/vocab.txt",
                              sha256_file(BASELINE_DIR/"PEAN_NHP_final_data_V1_100ep.py"),
                              logger.log,clean_import=None)
        if args.mode=="prepare":return
    metas={s.key:load_prep(run_dir,s) for s in specs}
    if args.mode in ("train","all"):
        for spec in specs:
            for seed in seeds:
                train_one(torch,run_dir,spec,metas[spec.key],seed,args,device,logger)
        if args.mode=="train":return
    if args.mode in ("evaluate","all"):
        for spec in specs:
            for seed in seeds:
                evaluate_seed(torch,run_dir,spec,metas[spec.key],seed,args,device,logger)
        if args.mode=="evaluate":return
    if args.mode in ("aggregate","all"):
        aggregate(run_dir,specs,seeds,logger)

if __name__=="__main__":
    try:
        main()
    except Exception:
        text=traceback.format_exc()
        if ACTIVE_LOGGER is not None:
            try:
                ACTIVE_LOGGER.log("[PEAN-NHP][FATAL]\n"+text)
            except Exception:
                pass
        traceback.print_exc()
        raise
