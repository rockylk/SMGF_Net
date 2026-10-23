#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit frozen V17 multi-flow profiles; export Table VIII and Fig.7.

Hard rules: only actual GPU inference records, same five models, three official
seeds and deterministic held-out flow indices, checkpoint/input SHA match.
PyTorch profiler FLOPs always marked SUBTOTAL until independently audited.
"""
from __future__ import annotations
import argparse,csv,json,math,statistics
from pathlib import Path
from inventory import SEEDS,ROBUST,sha256
HERE=Path(__file__).resolve().parent
ORDER=("USTC-CNN","ET-BERT","PEAN-NHP","IBGC","SMGF-Net")
FIELDS=("Model","Parameters (M)","Supported-Op FLOPs Subtotal (M)",
        "Inference Latency (ms/flow)","Mixed RobAcc (%)","Adaptive RobAcc (%)","Status")

def sample_stats(xs):
    return statistics.mean(xs),statistics.stdev(xs) if len(xs)>1 else 0.0

def audit(files,min_samples=8):
    by={m:{} for m in ORDER}
    errs=[]
    hardware=set()
    for file in files:
        rec=json.loads(Path(file).read_text(encoding="utf-8"))
        name=rec.get("model")
        if name not in by or rec.get("schema")!="section56-v17-modelonly-v1":
            raise ValueError(f"Non-V17 or unsupported profile {file}")
        seed=rec.get("seed");idx=rec.get("sample_index")
        physical_id=rec.get("sample_id")
        if not isinstance(physical_id,int) or physical_id<0:
            raise ValueError(f"Missing actual frozen test sample identity in {file}")
        if rec.get("dataset")!="cipher" or seed not in SEEDS or not isinstance(idx,int) or not 0<=idx<8200:
            raise ValueError(f"Invalid model/seed/sample_index in {file}")
        for field,sha_field in (("source","source_sha256"),("checkpoint","checkpoint_sha256"),
                                ("input_path","input_sha256")):
            p=Path(rec[field]);expected=rec[sha_field]
            if not p.is_file() or sha256(p)!=expected:
                raise ValueError(f"Source/checkpoint/input changed since measurement: {p}")
        if rec.get("batch_size")!=1 or rec.get("flops",{}).get("coverage")!="UNVERIFIED_SUBTOTAL_NOT_TOTAL":
            raise ValueError(f"Profiler scope mismatch {file}")
        t=float(rec["latency"]["median_ms"])
        if not 0<t<1e6 or not math.isfinite(t):raise ValueError(f"Invalid latency in {file}")
        if not int(rec["trainable_parameters"])>0:raise ValueError("Invalid parameters")
        if rec.get("device")!="cuda":errs.append(f"{name} includes CPU measurements")
        hw=(rec["device"],rec["gpu_name"],str(rec["torch_version"]),
            rec["precision"],rec["warmup"],rec["repeats"])
        hardware.add(hw)
        if idx in by[name].setdefault(seed,{}):
            raise ValueError(f"Duplicate (model,seed,index) {name}/{seed}/{idx}")
        by[name][seed][idx]=rec
    if len(hardware)>1:errs.append("GPU/PyTorch/precision/warmup/repeats not matched")
    expected_idx=None
    # Ensure same test sample indices map to the SAME physical original flow
    # even if preprocessing output order differs among different baselines.
    identity_by_index={}
    for name,seed_map in by.items():
        for seed,profile_map in seed_map.items():
            for idx,record in profile_map.items():
                sid=record["sample_id"]
                if idx in identity_by_index and identity_by_index[idx]!=sid:
                    errs.append(f"Sample identity mismatch at index={idx}: {name}, seed{seed}")
                identity_by_index.setdefault(idx,sid)
    summary=[]
    for name in ORDER:
        seeds=by[name]
        if set(seeds)!=set(SEEDS):errs.append(f"{name}: missing seed(s) {sorted(set(SEEDS)-set(seeds))}")
        allrec=[r for si in seeds.values() for r in si.values()]
        if not allrec:
            summary.append(dict(zip(FIELDS,[name,"NA","NA","NA",
                 f"{ROBUST[name][0]:.2f}",f"{ROBUST[name][1]:.2f}","MISSING"])))
            continue
        params={r["trainable_parameters"] for r in allrec}
        if len(params)!=1:raise ValueError(f"Different trainable params across {name} checkpoints")
        if any(len(seeds.get(s,{}))<min_samples for s in SEEDS):
            errs.append(f"{name}: fewer than {min_samples} independent test flows for ≥1 seed")
        for seed in SEEDS:
            inds=set(seeds.get(seed,{}))
            if not inds:continue
            if expected_idx is None:expected_idx=inds
            elif inds!=expected_idx:
                errs.append(f"Frozen test sample indices do not match: {name} seed{seed}")
        # Equal weight per seed, then equal weight per measured clean flow.
        seed_lat=[];seed_flops=[]
        for seed in SEEDS:
            if seed not in seeds:continue
            rr=list(seeds[seed].values())
            seed_lat.append(statistics.mean(r["latency"]["median_ms"] for r in rr))
            seed_flops.append(statistics.mean(r["flops"]["supported_op_mflops"] for r in rr))
        avg,sd=sample_stats(seed_lat)
        f,fsd=sample_stats(seed_flops)
        complete=len(seeds)==3 and all(len(seeds[s])>=min_samples for s in SEEDS)
        summary.append(dict(zip(FIELDS,[
            name,f"{next(iter(params))/1e6:.5f}",f"{f:.3f} ± {fsd:.3f} [SUBTOTAL]",
            f"{avg:.5f} ± {sd:.5f}" if complete else f"{avg:.5f} [INCOMPLETE]",
            f"{ROBUST[name][0]:.2f}",f"{ROBUST[name][1]:.2f}",
            "MEASURED_FLOPS_UNVERIFIED" if complete else "INCOMPLETE"
        ])))
    can_plot=(not errs and len(hardware)==1 and all(len(by[m])==3 for m in ORDER))
    return summary,by,errs,hardware,can_plot

def plot(rows,stem,partial):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors={"SMGF-Net":"#D33C3C","USTC-CNN":"#777777","ET-BERT":"#2776B6",
            "PEAN-NHP":"#21955A","IBGC":"#AD7730"}
    pts=[]
    for r in rows:
        raw=r["Inference Latency (ms/flow)"]
        if raw=="NA":continue
        x=float(raw.split()[0])
        pts.append((r["Model"],x,float(r["Mixed RobAcc (%)"]),float(r["Adaptive RobAcc (%)"])))
    if not pts:return False
    plt.rcParams.update({"font.family":"serif","font.size":6.2,"axes.labelsize":6.2,
                        "axes.titlesize":7.1,"xtick.labelsize":5.7,
                        "ytick.labelsize":5.7,"pdf.fonttype":42,"svg.fonttype":"none"})
    fig,axes=plt.subplots(1,2,figsize=(8.38/2.54,5.0/2.54),sharey=True)
    fig.subplots_adjust(left=.15,right=.98,top=.84,bottom=.25,wspace=.34)
    for i,ax in enumerate(axes):
        for name,x,y1,y2 in pts:
            y=(y1,y2)[i]
            ax.scatter([x],[y],s=16,marker="o" if name=="SMGF-Net" else "s",
                       color=colors[name],edgecolors="white",linewidths=.4,zorder=4)
            ax.annotate(name,(x,y),xytext=(2,3),textcoords="offset points",fontsize=4.9)
        ax.set_xscale("log")
        ax.set_title(("(a) Mixed attack","(b) Adaptive attack")[i],pad=2)
        ax.set_xlabel("Model-only latency (ms/flow)")
        ax.set_ylim(0,101)
        ax.grid(True,linestyle=(0,(2,2)),lw=.4,color="#C7C7C7",alpha=.8)
        for spine in ax.spines.values():spine.set_linewidth(.52)
    axes[0].set_ylabel("Robust accuracy (%)")
    if partial:
        fig.text(.50,.038,"PARTIAL — NOT FOR PUBLICATION",
                 ha="center",size=6.2,color="#BD3030")
    for ext in ("png","pdf","svg"):
        fig.savefig(str(stem)+"."+ext,dpi=600 if ext=="png" else None,facecolor="white")
    plt.close(fig)
    return True

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measurements",type=Path,default=HERE/"results"/"measurements")
    ap.add_argument("--out",type=Path,default=HERE/"results"/"report")
    ap.add_argument("--min-samples",type=int,default=8)
    ap.add_argument("--partial-chart",action="store_true")
    args=ap.parse_args()
    if args.min_samples<2:raise ValueError("min-samples must be >=2")
    files=sorted(args.measurements.glob("*_cipher_seed*_sample*.json"))
    rows,models,errs,hw,chart_ready=audit(files,args.min_samples)
    args.out.mkdir(parents=True,exist_ok=True)
    table=args.out/"TableVIII_V17_status.csv"
    with table.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    report=dict(schema="section56-v17-audit-v2",profile_records=len(files),
                models_with_3_seeds=[m for m in ORDER if len(models[m])==3],
                hardware=[list(x) for x in hw],profile_indices_required=args.min_samples,
                latency_chart_complete=chart_ready,
                table_viii_publication_ready=False,
                reasons=errs+[
                  "torch.profiler with_flops accounts only supported operator FLOPs (not full model complexity)",
                  "GPU latency excludes preprocessing, packet parsing, graph construction and CPU-GPU transfer"],
                robust_accuracy_from="CipherSpectrum Tables III and V frozen official endpoints",
                snapshot_files_unchanged=True)
    (args.out/"validation_status.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    if chart_ready or (args.partial_chart and files):
        stem=args.out/("Fig7_V17_latency_robustness" if chart_ready else "Fig7_PARTIAL_NOT_FOR_PUBLICATION")
        plot(rows,stem,not chart_ready)
        print("FIG7="+str(stem))
    print("TABLE="+str(table))
    print("PROFILES="+str(len(files)))
    print("LATENCY_CHART_COMPLETE="+str(chart_ready))
    print("TABLE_PUBLICATION_READY=False (FLOPs coverage not attested)")
    for e in report["reasons"]:print("ISSUE="+str(e))
if __name__=="__main__":main()