#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only diagnosis of the PRIOR and CURRENT GPU run. No torch needed.

This answers precisely what was finished: model/seed/sample triple counts,
the last failing stage, errors, and report artifacts. Preserve old JSONs.
"""
from __future__ import annotations
import argparse, collections, csv, json, os, re
from pathlib import Path
MODELS=("USTC-CNN","ET-BERT","PEAN-NHP","IBGC","SMGF-Net")
SEEDS=(42,43,44)
SAMPLES=(0,100,500,1000,1500,2000,4000,6000)

def diagnose(base: Path):
    root=base/"results"
    meas=root/"measurements"
    counter={name:{seed:set() for seed in SEEDS} for name in MODELS}
    warnings=[];found=[]
    for f in sorted(meas.glob("*.json")):
        if f.name=="orchestration_status.json":continue
        try:
            r=json.loads(f.read_text(encoding="utf-8"))
            m,s,i=r.get("model"),r.get("seed"),r.get("sample_index")
            if (r.get("schema")=="section56-v17-modelonly-v1"
                and m in counter and s in SEEDS and i in SAMPLES
                and r.get("device")=="cuda"
                and float(r.get("latency",{}).get("median_ms",0))>0):
                key=(m,s,i)
                if i in counter[m][s]:warnings.append("duplicate "+str(key))
                counter[m][s].add(i)
                found.append(dict(model=m,seed=s,index=i,path=str(f),params=r.get("trainable_parameters"),
                                  latency_ms=r["latency"]["median_ms"],checkpoint=r.get("checkpoint")))
            else:warnings.append("INCOMPLETE_OR_UNKNOWN "+f.name)
        except Exception as e:warnings.append("INVALID_JSON "+f.name+": "+str(e))
    status=root/"run_status.json"
    original=None
    if status.is_file():
        try:original=json.loads(status.read_text(encoding="utf-8"))
        except Exception as e:warnings.append("INVALID_run_status "+str(e))
    logs=root/"logs"
    errors=[]
    for f in sorted(logs.glob("*.log")):
        try:
            data=f.read_text(encoding="utf-8",errors="replace")
            if re.search(r"(?im)(Traceback \(most recent call last\)|ModuleNotFoundError:|FileNotFoundError:|RuntimeError:|ImportError:|AssertionError:|MODEL_FAILED|ERROR:|CUDA out of memory)",data):
                lines=data.splitlines()
                hits=[i for i,v in enumerate(lines) if re.search(r"Traceback \(most recent call last\)|(?:ModuleNotFoundError|FileNotFoundError|RuntimeError|ImportError|AssertionError):|MODEL_FAILED|CUDA out of memory",v)]
                context=[]
                for i in hits[-3:]:
                    context.extend(lines[max(0,i-2):min(len(lines),i+2)])
                errors.append(dict(file=str(f),last_errors=context[-12:]))
        except OSError as e:warnings.append("LOG_UNREADABLE "+str(f)+": "+str(e))
    files=[]
    for folder in (root/"report",root/"server_tmp"):
        if folder.is_dir():
            files +=[str(f) for f in sorted(folder.glob("*")) if f.is_file()]
    return dict(
        schema="section56-prior-run-diagnosis-v1",
        root=str(root),measurements_count=len(found),
        complete_models=[m for m in MODELS if all(len(counter[m][s])==len(SAMPLES) for s in SEEDS)],
        per_model={m:dict(total=sum(len(v) for v in counter[m].values()),
                          seeds={str(s):dict(done=len(counter[m][s]),
                                  indices=sorted(counter[m][s]),
                                  missing=sorted(set(SAMPLES)-counter[m][s])) for s in SEEDS})
                    for m in MODELS},
        original_run_status=original,
        warnings=warnings,
        logfile_errors=errors,
        output_files=files,
        individual_measurements=found)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument("--output",type=Path,default=None)
    args=ap.parse_args()
    report=diagnose(args.root.resolve())
    print("==== Section 5.6 Existing GPU Results (read-only) ====",flush=True)
    print("ROOT:",report["root"])
    print("MEASUREMENT_JSONS:",report["measurements_count"],"/120")
    for m,v in report["per_model"].items():
        print(f"{m:12s} {v['total']:2d}/24  "
              +" ".join(f"seed{s}={v['seeds'][str(s)]['done']}/8" for s in SEEDS))
    print("COMPLETE_MODELS:",",".join(report["complete_models"]) if report["complete_models"] else "NONE")
    if report["original_run_status"]:
        st=report["original_run_status"]
        print("ORIGINAL_STATUS:",st.get("status","unknown"))
        for k,v in st.get("model_status",{}).items():print("  MODEL_STATUS",k,v)
        for e in st.get("failed",[]):print("  REPORTED_FAILURE:",e)
    print("LOGFILES_WITH_ERRORS:",len(report["logfile_errors"]))
    for item in report["logfile_errors"][-10:]:
        print("LOG_ERROR:",item["file"])
        for line in item["last_errors"][-4:]:print("  ",line[:350])
    print("OTHER_WARNING_COUNT:",len(report["warnings"]))
    for msg in report["warnings"][:10]:print("WARN:",msg)
    print("OTHER_OUTPUT_FILES:",len(report["output_files"]))
    for f in report["output_files"][:25]:print("OUT:",f)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        print("DIAGNOSIS_JSON:",args.output)
if __name__=="__main__":main()
