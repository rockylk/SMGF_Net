#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CipherSpectrum official ET-BERT V17 intermediate-strength EVALUATION ONLY.

Uses official ET-BERT final runner for raw physical attacks and inference.
Does not call train_one/prepare_dataset/main(), nor modify formal results.
Requires all seed42/43/44 finetuned.bin files and their SHA-matched summaries.
Set --work-root, --run-dir, --dataset-root, --split-manifest, --etbert-root.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

SEEDS=(42,43,44)
STRENGTHS={"mixed":(.3,.6,.9),"adaptive":(.4,.7)}
SPLIT="01e3b572e8514a238b3733e224161acd1112b79b56f038635be9339b23cfad6b"

def check(flag,why):
    if not flag:raise RuntimeError(why)

def import_original(args):
    os.environ["SMGF_WORK_ROOT"]=str(args.work_root.resolve())
    os.environ["ETBERT_FINAL_RUN_DIR"]=str(args.run_dir.resolve())
    os.environ["ETBERT_RESULT_ROOT"]=str(args.run_dir.resolve().parent)
    os.environ["ETBERT_FINAL_DATASETS"]="cipher"
    os.environ["ETBERT_FINAL_MODE"]="evaluate"
    os.environ["ETBERT_FINAL_FULL_SWEEP"]="0" # Formal @1 cache must remain valid.
    os.environ["ETBERT_CIPHER_ROOT"]=str(args.dataset_root.resolve())
    os.environ["ETBERT_CIPHER_SPLIT_MANIFEST"]=str(args.split_manifest.resolve())
    os.environ["ETBERT_ROOT"]=str(args.etbert_root.resolve())
    if args.etbert_python:os.environ["ETBERT_PYTHON"]=args.etbert_python
    filename=Path(__file__).parent/"ET-BERT_final_paper_experiments.py"
    check(filename.is_file(),f"Official frozen runner absent: {filename}")
    spec=importlib.util.spec_from_file_location("frozen_final_etbert",filename)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod

def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding="utf-8")
    os.replace(tmp,path)

def run(args):
    mod=import_original(args)
    mod.LOGGER=mod.Logger(args.output/"etbert_inference.log")
    mod.validate_official_etbert()
    check(len(mod.DATASETS)==1 and mod.DATASETS[0].name=="CipherSpectrum",
          "Expected CipherSpectrum alone")
    spec=mod.DATASETS[0]
    mod.validate_dataset_spec(spec)
    check(spec.expected_split_sha==SPLIT,"Split SHA mismatch")
    check(mod.ATTACK_BASE_SEED==20260923,"ETBERT attack seed drift")
    check(mod.preparation_valid(spec),
          "Formal ET-BERT clean and @1 test data or hashes absent: "
          "original run must be restored; do NOT retrain here.")
    for seed in SEEDS:
        check(mod.valid_trained_model(spec,seed),
              "Frozen official ET-BERT finetuned.bin missing or SHA-invalid: "
              f"seed{seed} @ {mod.model_dir(spec,seed)}. "
              "Restore original GPU server checkpoints; never substitute pretrained weights.")
        official=mod.eval_summary_path(spec,seed)
        check(official.is_file(),f"Frozen evaluation reference absent {official}")
        previous=mod.load_json(official)
        check(previous["dataset"]=="CipherSpectrum" and previous["seed"]==seed,
              "Reference evaluation seed mismatch")
    args.output.mkdir(parents=True,exist_ok=True)
    cache=args.output/"prepared_tsv"
    cache.mkdir(parents=True,exist_ok=True)
    split=mod.load_json(spec.split_manifest)
    check(len(split["test"])==8200,"Expected 8200 frozen test rows")
    for attack,strengths in STRENGTHS.items():
        for strength in strengths:
            tsv=cache/f"test_{attack}_{strength:.1f}.tsv"
            if not tsv.exists():
                n=mod.write_tsv(tsv,mod.attacked_rows_from_test(
                    spec,split["test"],attack,strength))
                check(n==8200,f"Physical attack test cardinality mismatch {n}")
            check(len(mod.read_true_labels(tsv))==8200,
                  f"Attack TSV wrong test population: {tsv}")
            print(f"[ET-BERT][prepared] {tsv.name} sha256={mod.sha256_file(tsv)}",flush=True)
    if args.mode=="prepare":return
    # All new predictions are redirected into the NEW sensitivity folder.
    # The canonical original model_dir still points to formal-run weights.
    mod.prediction_file=lambda dataset_spec,seed,condition: (
        args.output/"predictions"/f"seed{seed}"/f"{condition}.tsv")
    records=[]
    for seed in SEEDS:
        for attack,strengths in STRENGTHS.items():
            for strength in strengths:
                condition=f"{attack}_{strength:.1f}"
                tsv=cache/f"test_{condition}.tsv"
                res_path=args.output/"measurements"/f"ETBERT_seed{seed}_{condition}.json"
                pred_path=mod.prediction_file(spec,seed,condition)
                checkpoint=mod.model_dir(spec,seed)/"finetuned.bin"
                evidence={
                    "checkpoint_sha256":mod.sha256_file(checkpoint),
                    "attack_manifest_sha256":mod.sha256_file(tsv),
                    "split_sha256":SPLIT,
                    "evaluation_artifact":str(res_path)}
                if res_path.is_file():
                    result=json.loads(res_path.read_text(encoding="utf-8"))
                    check(all(result.get(k)==v for k,v in evidence.items()),
                          f"Resume provenance differs: {res_path}")
                    check(result.get("model")=="ET-BERT" and
                          result.get("seed")==seed and
                          result.get("attack")==attack and
                          float(result.get("strength"))==strength and
                          pred_path.is_file(),
                          f"Resume artifacts incomplete: {res_path}")
                else:
                    output_pred=mod.inference_condition(spec,seed,condition,tsv)
                    eva=mod.evaluate_prediction(tsv,output_pred,spec.num_classes)
                    check(eva["n"]==8200,"ET-BERT evaluation population mismatch")
                    result={
                        "model":"ET-BERT", "seed":seed, "attack":attack,
                        "strength":strength,"robust_acc":eva["acc"],
                        "robust_f1":eva["macro_f1"],
                        **evidence}
                    atomic_json(res_path,result)
                records.append(result)
                print(f"[ET-BERT] seed{seed} {attack}@{strength:.1f}: "
                      f"RobAcc={result['robust_acc']:.5f}",flush=True)
    path=args.output/"ETBERT_Figure6_observed.csv"
    with path.open("w",newline="",encoding="utf-8") as f:
        cols=("model","attack","strength","seed","robust_acc","robust_f1",
              "checkpoint_sha256","attack_manifest_sha256","split_sha256",
              "evaluation_artifact")
        writer=csv.DictWriter(f,fieldnames=cols)
        writer.writeheader();writer.writerows(records)
    check(len(records)==15,"Expected 15 ET-BERT intermediate-seed records")
    print("SAVED",path,"RECORDS",len(records))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg,desc in (
        ("--work-root","Existing SMGF_WORK_ROOT"),
        ("--run-dir","Frozen original ETBERT_FINAL_PAPER_BASELINE_V1 output"),
        ("--dataset-root","Original CipherSpectrum PCAP root"),
        ("--split-manifest","Frozen CipherSpectrum split_manifest.json"),
        ("--etbert-root","Official ET-BERT repository root"),
        ("--output","NEW separate sensitivity output folder")):
        p.add_argument(arg,type=Path,required=True,help=desc)
    p.add_argument("--etbert-python",default=None,help="Official inference Python interpreter")
    p.add_argument("--mode",choices=("prepare","all"),default="all")
    args=p.parse_args()
    run(args)
if __name__=="__main__":main()