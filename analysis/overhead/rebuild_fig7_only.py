#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild verified Fig.7 directly from existing GPU JSONs; NO GPU work.

Official figure is generated ONLY if the five frozen formal models EACH have
the exact 3 seeds x 8 original test-flow indexes under matching GPU/protocol.
Never claims unsupported-op FLOP subtotal is whole-model FLOPs.
"""
from __future__ import annotations
import csv,json,math,sys
from pathlib import Path
from build_report import audit, ORDER, FIELDS
from inventory import SEEDS

HERE=Path(__file__).resolve().parent
IDX={0,100,500,1000,1500,2000,4000,6000}

def make_figure(rows,stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    series=[]
    for row in rows:
        name=row["Model"]
        raw=row["Inference Latency (ms/flow)"]
        if raw=="NA" or "[INCOMPLETE]" in raw:
            raise ValueError("No complete latency for "+name)
        latency=float(raw.split()[0])
        if not math.isfinite(latency) or latency<=0:
            raise ValueError("Invalid plotted latency "+name)
        mixed=float(row["Mixed RobAcc (%)"])
        adaptive=float(row["Adaptive RobAcc (%)"])
        series.append((name,latency,mixed,adaptive))
    if len(series)!=5 or {r[0] for r in series}!=set(ORDER):
        raise AssertionError("Refusing Figure7: not exactly five formal models")
    settings={
        "USTC-CNN":("#6F7784","s"),
        "ET-BERT":("#467EB7","D"),
        "PEAN-NHP":("#3A9672","^"),
        "IBGC":("#C28E45","v"),
        "SMGF-Net":("#CF4446","o"),
    }
    plt.rcParams.update({
        "font.family":"serif","font.size":6.5,"axes.labelsize":6.4,
        "axes.titlesize":7,"xtick.labelsize":6,"ytick.labelsize":6,
        "pdf.fonttype":42,"svg.fonttype":"none"})
    fig,axes=plt.subplots(1,2,sharey=True,figsize=(8.38/2.54,5.3/2.54))
    fig.subplots_adjust(left=.145,right=.985,bottom=.315,top=.88,wspace=.28)
    handles=[]
    for name in ORDER:
        color,marker=settings[name]
        handles.append(Line2D([0],[0],marker=marker,linestyle="None",
            markerfacecolor=color,markeredgecolor="white",
            markeredgewidth=.35,markersize=5,label=name))
    for axisno,ax in enumerate(axes):
        for name,x,mixed,adaptive in series:
            color,marker=settings[name]
            y=(mixed,adaptive)[axisno]
            ax.scatter(x,y,s=27,marker=marker,color=color,
                edgecolor="white",linewidth=.35,zorder=4)
        ax.set_xscale("log")
        ax.set_xlabel("Model-only latency (ms/flow)",labelpad=3)
        ax.set_title(("(a) Mixed attack","(b) Adaptive attack")[axisno])
        ax.set_ylim(0,101)
        ax.grid(True,color="#bbbbbb",linestyle=":",linewidth=.5,alpha=.65)
        for edge in ax.spines.values():edge.set_linewidth(.6)
    axes[0].set_ylabel("Robust accuracy (%)")
    fig.legend(handles=handles,ncol=3,loc="lower center",
               bbox_to_anchor=(.50,.015),frameon=False,
               handletextpad=.25,columnspacing=.7,fontsize=6)
    for ext in ("png","pdf","svg"):
        fig.savefig(str(stem)+"."+ext,dpi=600 if ext=="png" else None,
                    facecolor="white")
    plt.close(fig)
    return series

def main():
    measurements=HERE/"results"/"measurements"
    output=HERE/"results"/"report"
    files=sorted(measurements.glob("*_cipher_seed*_sample*.json"))
    print("=== READ-ONLY GPU PROFILE AUDIT (NO MODEL RUN) ===",flush=True)
    print("FOUND_JSONS",len(files),"EXPECTED",120)
    rows,by,reasons,hardware,prior_ok=audit(files,8)
    totals={}
    for m in ORDER:
        by_seed={str(seed):len(by[m].get(seed,{})) for seed in SEEDS}
        totals[m]=dict(by_seed=by_seed,total=sum(by_seed.values()))
        print("MODEL",m,totals[m],flush=True)
        for seed in SEEDS:
            indices=set(by[m].get(seed,{}))
            if indices!=IDX:
                reasons.append(m+" seed"+str(seed)+" missing="+
                    str(sorted(IDX-indices))+" extra="+str(sorted(indices-IDX)))
    ready=(len(files)==120 and len(hardware)==1 and prior_ok and not reasons
           and all(totals[m]["total"]==24 for m in ORDER))
    audit_record=dict(
        schema="section56-verified-plot-v1",
        complete=ready,expected_models=list(ORDER),
        expected_records=120,observed_records=len(files),
        record_counts=totals,observed_hardware=[list(x) for x in hardware],
        issues=list(dict.fromkeys(reasons)),
        notes=[
            "Model-only GPU latency; excludes preprocessing and transfers.",
            "FLOPs are operator subtotal, NOT audited full model FLOPs.",
            "Robust accuracies are frozen CipherSpectrum Tables III and V endpoints.",
        ])
    output.mkdir(parents=True,exist_ok=True)
    with (output/"TableVIII_5Model_audit.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if ready:
        stem=output/"Fig7_VERIFIED_5_MODELS"
        plotted=make_figure(rows,stem)
        audit_record["figure"]={ "stem":str(stem),
                                 "plotted_model_count":len(plotted),
                                 "points_per_panel":len(plotted),
                                 "points":[dict(model=m,latency_ms=x,
                                    mixed_robacc_pct=y,adaptive_robacc_pct=z)
                                    for m,x,y,z in plotted]}
        print("SUCCESS: FIGURE HAS EXACTLY 5 POINTS PER PANEL")
        print("FIGURE",str(stem)+".png/.pdf/.svg")
    else:
        audit_record["figure"]=None
        print("BLOCKED: NO OFFICIAL FIGURE GENERATED; data are incomplete or noncomparable")
        for reason in audit_record["issues"]:print("ISSUE",reason)
    out=output/"Fig7_verified_5model_status.json"
    out.write_text(json.dumps(audit_record,ensure_ascii=False,indent=2),encoding="utf-8")
    print("AUDIT_JSON",out)
    print("TABLE_CSV",output/"TableVIII_5Model_audit.csv")
    if not ready:raise SystemExit(2)
if __name__=="__main__":main()
