# -*- coding: utf-8 -*-
"""Cross-model CipherSpectrum sensitivity from genuine V17 three-seed results.

Publication figure: two panels, total exported width 8.38 cm.
Original clean / strength-1 endpoints are immutable; baseline intermediate
strengths use 45 measured GPU observations and are never interpolated.
The locally downloaded verified CSV is auto-selected if available.

Run with: uv run --no-project --python 3.12 --with numpy --with matplotlib
          python Figure6_CipherSpectrum_CrossModel_Sensitivity.py --require-complete
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
FINAL = ROOT / "SMGF_V17_CIPHERSPECTRUM_FINAL_3SEED_20260923_135542"
OUT = FINAL / "figures" / "figure6_cross_model_sensitivity"
DEFAULT_SUPPLEMENT = ROOT / "FIG6_CIPHER_FROZEN_GPU_SWEEP_Home" / "Figure6_CrossModel_45_verified.csv"
PATHS = {
    "SMGF-Net": FINAL / "per_seed_results.json",
    "PEAN-NHP": ROOT / "PEAN_NHP_FINAL_PAPER_BASELINE_V1_100EP/PEAN_NHP_per_seed_primary.csv",
    "IBGC": ROOT / "IBGC_V17_V1_100EP/per_seed.csv",
    "ET-BERT": ROOT / "ETBERT_FINAL_PAPER_BASELINE_V1/ETBERT_per_seed_primary.csv",
}
MODELS = ("SMGF-Net", "PEAN-NHP", "IBGC", "ET-BERT")
STRENGTHS = {"mixed": (0., .3, .6, .9, 1.),
             "adaptive": (0., .4, .7, 1.)}
SEEDS = (42, 43, 44)
# Two-decimal values frozen in the final manuscript Tables II/III/V.
ENDPOINTS = {"SMGF-Net": (91.30, 78.92, 86.88),
             "PEAN-NHP": (86.88, 28.51, 28.88),
             "IBGC": (82.17, 17.62, 17.37),
             "ET-BERT": (33.58, 22.94, 25.20)}
# Different markers and linestyles remain discernible in grayscale.
COLORS = {"SMGF-Net": "#D33C3C", "PEAN-NHP": "#25945B",
          "IBGC": "#696969", "ET-BERT": "#267AB4"}
MARKERS = {"SMGF-Net": "o", "PEAN-NHP": "s", "IBGC": "^", "ET-BERT": "D"}
LINESTYLES = {"SMGF-Net": "-", "PEAN-NHP": "--",
              "IBGC": ":", "ET-BERT": "-."}
SUP_COLS = ("model", "attack", "strength", "seed", "robust_acc",
            "checkpoint_sha256", "attack_manifest_sha256", "split_sha256",
            "evaluation_artifact")

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1048576), b""):
            h.update(part)
    return h.hexdigest()

def csvrows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def add(db, model, attack, strength, seed, acc, source):
    if model not in MODELS or attack not in STRENGTHS:
        raise ValueError(f"Invalid model/attack {model}/{attack}")
    strength, seed, acc = float(strength), int(seed), float(acc)
    if strength not in STRENGTHS[attack] or seed not in SEEDS:
        raise ValueError(f"Invalid strength/seed {strength}/{seed}")
    if not math.isfinite(acc) or not 0 <= acc <= 100:
        raise ValueError(f"Out-of-range robust accuracy {acc}")
    key = (model, attack, strength, seed)
    if key in db:
        raise ValueError(f"Duplicate observation: {key}")
    db[key] = dict(model=model, attack=attack, strength=strength, seed=seed,
                   robust_acc=acc, source=source)

def frozen_data():
    db = {}
    p = PATHS["SMGF-Net"]
    report = json.loads(p.read_text(encoding="utf-8"))
    if report.get("dataset") != "CipherSpectrum":
        raise ValueError("SMGF dataset mismatch")
    if [int(r["seed"]) for r in report["test_results"]] != list(SEEDS):
        raise ValueError("SMGF seed set mismatch")
    for r in report["test_results"]:
        seed = int(r["seed"])
        for attack, strengths in STRENGTHS.items():
            add(db, "SMGF-Net", attack, 0., seed, r["clean"]["acc"], str(p))
            for s in strengths[1:]:
                add(db, "SMGF-Net", attack, s, seed,
                    r["attacks"][f"{attack}@{s:.1f}"]["robust_acc"], str(p))
    for model in ("PEAN-NHP", "ET-BERT"):
        p = PATHS[model]
        rows = [r for r in csvrows(p) if r["dataset"] == "CipherSpectrum"]
        if len(rows) != 3 or {int(r["seed"]) for r in rows} != set(SEEDS):
            raise ValueError(f"{model}: invalid seed list")
        for r in rows:
            seed = int(r["seed"])
            for attack in STRENGTHS:
                add(db, model, attack, 0., seed, r["clean_acc"], str(p))
                add(db, model, attack, 1., seed,
                    r[f"{attack}_robust_acc"], str(p))
    p = PATHS["IBGC"]
    identity = defaultdict(dict)
    for r in csvrows(p):
        if r["dataset"] == "CipherSpectrum":
            identity[(int(r["seed"]), r["condition"])][r["metric"]] = float(r["value"])
    for seed in SEEDS:
        for attack in STRENGTHS:
            add(db, "IBGC", attack, 0., seed,
                identity[(seed, "clean")]["acc"], str(p))
            add(db, "IBGC", attack, 1., seed,
                identity[(seed, f"{attack}_1.0")]["robust_acc"], str(p))
    for model in MODELS:
        for attack, pos in (("mixed", 1), ("adaptive", 2)):
            for s, paper in ((0., ENDPOINTS[model][0]), (1., ENDPOINTS[model][pos])):
                value = mean(db[(model, attack, s, seed)]["robust_acc"] for seed in SEEDS)
                if abs(value - paper) > .0051:
                    raise ValueError(f"Paper endpoint mismatch: {model}/{attack}@{s}: {value:.4f}, vs {paper:.2f}")
    return db

def supplement(db, path):
    # These fields are required for auditability, not automatic proof that an
    # independent re-evaluation used the same frozen physical-attack protocol.
    rows = csvrows(path)
    for r in rows:
        if any(not str(r.get(k, "")).strip() for k in SUP_COLS):
            raise ValueError(f"Missing supplement values/provenance: {r}")
        for k in ("checkpoint_sha256", "attack_manifest_sha256", "split_sha256"):
            s = r[k].lower()
            if len(s) != 64 or any(ch not in "0123456789abcdef" for ch in s):
                raise ValueError(f"Malformed provenance hash: {k}")
        if r["model"] == "SMGF-Net":
            raise ValueError("SMGF-Net archived curves cannot be overwritten")
        if float(r["strength"]) in (0., 1.):
            raise ValueError("Cannot overwrite an official baseline endpoint")
        add(db, r["model"], r["attack"], r["strength"], r["seed"],
            r["robust_acc"], "rerun:" + r["evaluation_artifact"])
    return len(rows)

def summarize(db):
    summary, missing = [], []
    for attack, strengths in STRENGTHS.items():
        for model in MODELS:
            for strength in strengths:
                values = [db[(model, attack, strength, seed)]["robust_acc"]
                          for seed in SEEDS if (model, attack, strength, seed) in db]
                if len(values) not in (0, 3):
                    raise ValueError(f"Partial seed triplet: {model}/{attack}@{strength}")
                row = dict(model=model, attack=attack, strength=strength,
                           mean=mean(values) if values else "",
                           sd=stdev(values) if values else "",
                           n_seeds=len(values),
                           status="observed" if values else "MISSING_NOT_INTERPOLATED")
                summary.append(row)
                if not values: missing.append(row)
    return summary, missing

def serif():
    available = {x.name for x in font_manager.fontManager.ttflist}
    return next((f for f in ("Times New Roman", "Times", "Nimbus Roman",
                            "DejaVu Serif") if f in available), "DejaVu Serif")

def plot(db, summary, complete):
    """Two-panel publication figure with a strict 8.38 cm output width."""
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [serif()],
        "font.size": 6.4, "axes.labelsize": 6.5, "axes.titlesize": 7.0,
        "xtick.labelsize": 6.0, "ytick.labelsize": 6.0,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 2, figsize=(8.38 / 2.54, 4.65 / 2.54),
                             sharey=True)
    fig.subplots_adjust(left=.154, right=.981, top=.865,
                        bottom=.245 if complete else .28, wspace=.30)

    for ax, (attack, strengths) in zip(axes, STRENGTHS.items()):
        for model in MODELS:
            found = sorted(
                (r for r in summary if r["attack"] == attack
                 and r["model"] == model and r["status"] == "observed"),
                key=lambda r: r["strength"]
            )
            x = [r["strength"] for r in found]
            y = [r["mean"] for r in found]
            sd = [r["sd"] for r in found]
            color, marker = COLORS[model], MARKERS[model]
            if len(x) == len(strengths):
                ax.fill_between(
                    x, [max(0, v-z) for v, z in zip(y, sd)],
                    [min(100, v+z) for v, z in zip(y, sd)],
                    color=color, alpha=.09, linewidth=0, zorder=2
                )
                ax.plot(
                    x, y, color=color, linestyle=LINESTYLES[model],
                    linewidth=0.9, marker=marker, markersize=1.65,
                    markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=.72, label=model, zorder=4
                )
            else:
                # No line is drawn across unmeasured intermediate strengths.
                ax.scatter(x, y, marker=marker, s=13., facecolors="white",
                           edgecolors=color, linewidths=.8, zorder=4,
                           label=model + " (endpoints)")
        ax.set_xlim(-.027, 1.027)
        ax.set_ylim(0, 100)
        ax.set_xticks(strengths)
        ax.set_xticklabels([f"{v:.1f}" for v in strengths])
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(axis="both", which="major", direction="out",
                       length=2.0, width=.52, pad=1.7, labelleft=True)
        ax.set_axisbelow(True)
        ax.grid(True, which="major", axis="both", color="#C9C9C9",
                linestyle=(0, (2, 2)), linewidth=.39, alpha=.78)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(.54)
            spine.set_color("#383838")
        ax.set_title("(a) Mixed attack" if attack == "mixed" else
                     "(b) Adaptive attack", pad=3.0)
        ax.set_xlabel("Perturbation strength", labelpad=2.0)

    axes[0].set_ylabel("Robust accuracy (%)", labelpad=2.0)
    # In-panel legend uses the gap below SMGF and above the baselines,
    # not the upper-left, which would obscure high clean accuracy points.
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper right",
                   bbox_to_anchor=(.985, .77), frameon=False, ncol=1,
                   fontsize=5.9, handlelength=1.8, handletextpad=.36,
                   labelspacing=.18, borderaxespad=.1, borderpad=0)
    if not complete:
        fig.text(.50, .055, "PARTIAL - missing baseline measurements",
                 ha="center", fontsize=6.2, fontweight="bold")

    OUT.mkdir(parents=True, exist_ok=True)
    stem = "Figure6_CipherSpectrum_CrossModel_Sensitivity"
    if not complete:
        stem += "_PARTIAL"
    # Do not use bbox_inches="tight": it can exceed 8.5 cm in PDF.
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}",
                    dpi=600 if ext == "png" else None, facecolor="white")
    plt.close(fig)
    return OUT / (stem + ".png")


def save_csv(path, data, names):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(data)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--supplement", type=Path,
                   default=DEFAULT_SUPPLEMENT if DEFAULT_SUPPLEMENT.is_file() else None,
                   help="Observed, provenance-annotated baseline strength-rerun CSV")
    p.add_argument("--require-complete", action="store_true",
                   help="Fail unless all baselines have real intermediate-strength results")
    args = p.parse_args()
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    db = frozen_data()
    extra = supplement(db, args.supplement) if args.supplement else 0
    summary, missing = summarize(db)
    if args.require_complete and missing:
        raise RuntimeError(f"Incomplete cross-model curves: {len(missing)} unmeasured "
                           "strength/model points. No interpolation is allowed.")
    OUT.mkdir(parents=True, exist_ok=True)
    save_csv(OUT / "Figure6_CrossModel_observed_seed_values.csv",
             sorted(db.values(), key=lambda x:(x["model"],x["attack"],x["strength"],x["seed"])),
             ("model","attack","strength","seed","robust_acc","source"))
    save_csv(OUT / "Figure6_CrossModel_summary.csv", summary,
             ("model","attack","strength","mean","sd","n_seeds","status"))
    png = plot(db, summary, complete=not missing)
    provenance = dict(dataset="CipherSpectrum", seeds=SEEDS, strengths=STRENGTHS,
                      models=MODELS, manuscript_ready=not missing, figure=str(png),
                      official_sources={k:dict(path=str(v), sha256=sha256(v))
                                        for k,v in PATHS.items()},
                      supplementary_rows=extra,
                      supplement_sha256=sha256(args.supplement) if args.supplement else None,
                      missing_model_attack_strengths=missing)
    (OUT / "Figure6_CrossModel_provenance.json").write_text(
        json.dumps(provenance,ensure_ascii=False,indent=2),encoding="utf-8")
    status = ("COMPLETE" if not missing else "PARTIAL - NOT MANUSCRIPT READY")
    (OUT / "README.md").write_text(
        "# Cross-model sensitivity on CipherSpectrum\n\n"
        f"Status: {status}\n\n"
        f"Unmeasured baseline attack/strength groups: {len(missing)}.\n"
        f"Supplement observations loaded: {extra}.\n"
        "Official V17 endpoints are checked against Tables II/III/V.\n"
        "No missing strengths are interpolated. Baselines retain their "
        "original RNG seeds, so equal nominal attack strengths do not imply "
        "identical perturbation instances.\n"
        "Publication layout: 8.38 cm wide, full fine axes, light grids, "
        "distinct lines/markers, mean with sample SD, PNG/PDF/SVG.\n",
        encoding="utf-8")
    print("OUTPUT="+str(png))
    print("COMPLETE="+str(not missing))
    print("MISSING="+str(len(missing)))
    print("SUMMARY="+str(OUT / "Figure6_CrossModel_summary.csv"))
    print("PROVENANCE="+str(OUT / "Figure6_CrossModel_provenance.json"))

if __name__=="__main__":
    main()