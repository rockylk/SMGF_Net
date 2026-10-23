# -*- coding: utf-8 -*-
"""Five-model macro OvR ROC with seed-level AUROC and sample SD.

Run after export: python calculate_and_plot.py --data-root <formal-results>
Use --exports <export-root> for separately exported/restored probabilities.
The default export is blocked if any of the five models lacks three seeds.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from audit_panel_alignment import require_matplotlib_panel_alignment

MODELS = ("USTC-CNN", "ET-BERT", "PEAN-NHP", "IBGC", "SMGF-Net")
DATASETS = {"cipher": ("CipherSpectrum", 8200, 41), "ustc": ("USTC-TFC2016", 4252, 12)}
SEEDS = (42, 43, 44)
CONDITIONS = ("clean", "mixed_1.0", "adaptive_1.0")
RUNS = {
    "USTC-CNN": "USTC_CNN_FINAL_PAPER_BASELINE_V1",
    "ET-BERT": "ETBERT_FINAL_PAPER_BASELINE_V1",
    "PEAN-NHP": "PEAN_NHP_FINAL_PAPER_BASELINE_V1_100EP",
    "IBGC": "IBGC_V17_V1_100EP",
}
SPLITS = {
    "cipher": "SMGF_V17_CIPHERSPECTRUM_FINAL_3SEED_20260923_135542/split_manifest.json",
    "ustc": "SMGF_V17_USTC_SEED42_FULL_F10000_20260924_082814/split_manifest.json",
}
COLORS = {"USTC-CNN": "#80718B", "ET-BERT": "#3976A0", "PEAN-NHP": "#418C83",
          "IBGC": "#6A7075", "SMGF-Net": "#C45A35"}
STYLES = {"USTC-CNN": ":", "ET-BERT": "--", "PEAN-NHP": "-.", "IBGC": (0, (5, 2, 1, 2)), "SMGF-Net": "-"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        keys = fields or (list(rows[0]) if rows else [])
        if keys:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


def binary_roc_auc(positive, scores):
    """Keep every score threshold and average-rank ties; never infer scores."""
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        raise ValueError("An OvR class lacks positive or negative examples")
    order = np.argsort(scores, kind="stable")
    ascending, truth = scores[order], positive[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ascending)) + 1]
    stops = np.r_[starts[1:], len(scores)]
    ranks = np.repeat((starts + 1 + stops) / 2, stops - starts)
    rank_auc = (ranks[truth].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    y, s = truth[::-1], ascending[::-1]
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[ends]
    fp = ends + 1 - tp
    tpr, fpr = np.r_[0., tp / n_pos], np.r_[0., fp / n_neg]
    auc = float(np.trapezoid(tpr, fpr))
    if abs(auc - rank_auc) > 1e-12:
        raise AssertionError("ROC integral and tie-aware rank AUC disagree")
    return fpr, tpr, auc


def interpolate_roc(fpr, tpr, grid):
    """Interpolate along ROC segments without turning vertical jumps diagonal."""
    x, first, counts = np.unique(fpr, return_index=True, return_counts=True)
    if not np.all(np.diff(x) > 0):
        raise AssertionError("Unique FPR coordinates must be strictly increasing")
    low, high = tpr[first], tpr[first + counts - 1]
    left = np.clip(np.searchsorted(x, grid, side="right") - 1, 0, len(x)-1)
    right = np.minimum(left + 1, len(x)-1)
    widths = x[right] - x[left]
    fraction = np.divide(grid-x[left], widths, out=np.zeros_like(grid), where=widths > 0)
    # Between FPR coordinates, connect the previous vertical segment's upper
    # endpoint to the next segment's lower endpoint; at a jump use its upper end.
    return high[left] + fraction * (low[right] - high[left])


def macro_curve(labels, probabilities, grid):
    curves, aucs = [], []
    for k in range(probabilities.shape[1]):
        fpr, tpr, auc = binary_roc_auc(labels == k, probabilities[:, k])
        curves.append(interpolate_roc(fpr, tpr, grid))
        aucs.append(auc)
    return np.mean(curves, axis=0), float(np.mean(aucs)), aucs


def load_result(root, exports, model, dataset, seed, condition, split):
    overlay = exports / model / dataset / f"seed{seed}" / condition
    if model == "SMGF-Net":
        path = Path(str(overlay) + ".npz")
        provenance = read_json(Path(str(overlay) + ".json"))
        if provenance["probability_sha256"] != sha(path):
            raise ValueError("SMGF probability provenance hash mismatch")
        if provenance["split_sha256"] != sha(root / SPLITS[dataset]):
            raise ValueError("SMGF formal split hash mismatch")
        checkpoint = provenance["checkpoint_sha256"]
        expected_auc = None
        expected_acc = provenance["reference_accuracy_pct"]
        source = "New inference using the frozen formal checkpoint"
    else:
        base = root / RUNS[model] / dataset
        if model == "ET-BERT":
            path = Path(str(overlay) + ".tsv")
            if not path.exists():
                path = base / "predictions" / f"seed{seed}" / f"{condition}.tsv"
            record = read_json(base / "predictions" / f"seed{seed}" / "evaluation_summary.json")
            metrics = record["clean"] if condition == "clean" else record["attacks"][condition]
            if record["split_sha256"] != sha(root / SPLITS[dataset]):
                raise ValueError("ET-BERT formal split hash mismatch")
        elif model == "IBGC":
            path = base / "evaluations" / f"seed{seed}" / f"{condition}_probabilities.npz"
            record = read_json(path.parent / "evaluation.json")
            metrics = record["metrics"][condition]
        else:
            path = base / "predictions" / f"seed{seed}" / f"{condition}.npz"
            category = "results" if model == "USTC-CNN" else "evaluations"
            record = read_json(base / category / f"seed{seed}.json")
            metrics = record[condition.replace(".", "_") if model == "PEAN-NHP" else condition]
        checkpoint = record["checkpoint_sha256"]
        expected_auc = metrics["macro_auroc"] / 100
        expected_acc = metrics.get("acc", metrics.get("robust_acc"))
        source = "Formal archived probability output"
    canonical_labels = np.array([r["label"] for r in split["test"]])
    canonical_ids = np.array([r["sample_index"] for r in split["test"]])
    ids = None
    if path.suffix == ".tsv":
        if b"\x00" in path.read_bytes():
            raise ValueError("Local TSV read contains NUL bytes; complete probability rows unavailable")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        probabilities = np.array([np.fromstring(row["prob"], sep=" ") for row in rows])
        predictions = np.array([int(row["label"]) for row in rows])
        labels = canonical_labels
    else:
        with np.load(path, allow_pickle=False) as archive:
            probabilities, labels = archive["probs"], archive["labels"]
            predictions = archive["preds"]
            for field in ("sample_indices", "sample_index"):
                if field in archive.files:
                    ids = archive[field]
                    break
        if ids is None and model == "USTC-CNN":
            with np.load(root / RUNS[model] / dataset / "prepared" / f"test_{condition}.npz", allow_pickle=False) as archive:
                ids = archive["sample_index"]
    name, n, classes = DATASETS[dataset]
    if probabilities.shape != (n, classes) or not np.array_equal(labels, canonical_labels):
        raise ValueError("Formal test population, class count or label order mismatch")
    if ids is not None and not np.array_equal(ids, canonical_ids):
        raise ValueError("Formal sample indices differ")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Invalid probability values")
    if abs(probabilities.sum(axis=1) - 1).max() > 1e-4:
        raise ValueError("Probability row sums differ from one")
    if not np.array_equal(probabilities.argmax(axis=1), predictions):
        raise ValueError("Prediction differs from probability argmax")
    accuracy = float(np.mean(labels == predictions) * 100)
    if abs(accuracy - expected_acc) > 1e-8:
        raise ValueError(f"Original formal accuracy not reproduced: {accuracy} vs {expected_acc}")
    return labels, probabilities, checkpoint, expected_auc, dict(
        model=model, dataset=name, seed=seed, condition=condition, n=n, classes=classes,
        probability_file=str(path.resolve()), probability_sha256=sha(path),
        checkpoint_sha256=checkpoint, accuracy_pct=accuracy, provenance=source)


def analyze(root, exports, output):
    grid = np.linspace(0, 1, 4001)
    samples, classes, missing = [], [], []
    curves, checkpoint_ids = {}, {}
    for model in MODELS:
        for dataset in DATASETS:
            split = read_json(root / SPLITS[dataset])
            for condition in CONDITIONS:
                for seed in SEEDS:
                    try:
                        y, p, checkpoint, expected_auc, source = load_result(root, exports, model, dataset, seed, condition, split)
                        tpr, auc, per_class = macro_curve(y, p, grid)
                        if expected_auc is not None and abs(auc - expected_auc) > 1e-10:
                            raise ValueError("Original macro OvR AUROC not reproduced")
                    except (OSError, ValueError, KeyError, json.JSONDecodeError, EOFError) as exc:
                        missing.append(dict(model=model, dataset=DATASETS[dataset][0], seed=seed,
                                            condition=condition, reason=str(exc)))
                        continue
                    key = (model, dataset, condition, seed)
                    curves[key] = tpr
                    checkpoint_ids[key] = checkpoint
                    samples.append(dict(source, macro_ovr_auroc=auc,
                                        display_interpolation_area=float(np.trapezoid(tpr, grid))))
                    classes.extend(dict(model=model, dataset=DATASETS[dataset][0], seed=seed,
                                        condition=condition, class_id=k, auroc=a) for k, a in enumerate(per_class))
    aggregates = []
    groups = defaultdict(list)
    for sample in samples:
        groups[(sample["model"], sample["dataset"], sample["condition"])].append(sample)
    arrays = {"fpr": grid}
    for model in MODELS:
        for dataset in DATASETS:
            for condition in CONDITIONS:
                keys = [(model, dataset, condition, seed) for seed in SEEDS]
                if not all(key in curves for key in keys):
                    continue
                if len({checkpoint_ids[key] for key in keys}) != 3:
                    raise ValueError("Three distinct formal checkpoints are required")
                values = np.array([s["macro_ovr_auroc"] for s in groups[(model, DATASETS[dataset][0], condition)]])
                stack = np.stack([curves[key] for key in keys])
                mean, sd = stack.mean(axis=0), stack.std(axis=0, ddof=1)
                group_key = f"{model}__{dataset}__{condition}"
                arrays[group_key + "__mean"] = mean
                arrays[group_key + "__sample_sd"] = sd
                arrays[group_key + "__seed_curves"] = stack
                aggregates.append(dict(model=model, dataset=DATASETS[dataset][0], dataset_key=dataset,
                                       condition=condition, n_models=3,
                                       mean_auroc=float(values.mean()), sample_sd_auroc=float(values.std(ddof=1))))
    write_csv(output / "per_seed_auroc_and_provenance.csv", samples)
    write_csv(output / "per_class_auroc.csv", classes)
    write_csv(output / "auroc_mean_sample_sd.csv", aggregates)
    write_csv(output / "missing_probability_inputs.csv", missing,
              ["model", "dataset", "seed", "condition", "reason"])
    np.savez_compressed(output / "macro_roc_curves.npz", **arrays)
    report = dict(expected_probability_outputs=90, valid_probability_outputs=len(samples),
                  unavailable_probability_outputs=len(missing), complete_three_seed_groups=len(aggregates),
                  fpr_grid_points=len(grid), class_averaging="equal-weight OvR",
                  seed_sd_ddof=1, pooled_seeds=False, averaged_probabilities=False,
                  auroc_source="Mean of original per-class threshold ROC integrals for each seed",
                  display_area_used_for_legend=False,
                  etbert_official_later_rerun_accepted=True,
                  excluded_path_substring="云端")
    (output / "analysis_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return grid, arrays, aggregates, missing


def draw_figure(grid, arrays, stats, output, conditions, preview=False):
    wanted = {(m, d, c) for m in MODELS for d in DATASETS for c in conditions}
    found = {(r["model"], r["dataset_key"], r["condition"]) for r in stats}
    missing_groups = wanted - found
    if missing_groups and not preview:
        raise RuntimeError("Final five-model figure blocked: incomplete three-seed groups. See missing_probability_inputs.csv")
    # 【修改1、8】保留新罗马字体，所有基础文字字号统一扩大至此前的 1.3 倍。
    # 【修改1】四边封闭外框以及左右下刻度线统一改为 0.8 pt；刻度仍然仅在左、下且朝外。
    font_scale = 1.3
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 7 * font_scale, "axes.labelsize": 7 * font_scale,
        "xtick.labelsize": 6.5 * font_scale, "ytick.labelsize": 6.5 * font_scale,
        "legend.fontsize": 6.5 * font_scale, "axes.linewidth": 0.8,
        "axes.spines.top": True, "axes.spines.right": True,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.top": False, "ytick.right": False,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.width": 0.8, "ytick.minor.width": 0.8,
        "legend.frameon": False, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(len(conditions), 2, squeeze=False,
                             figsize=(7.2047244094, 8.0708661417 if len(conditions) == 2 else 4.5275590551))
    # 【修改4、5】先保留双列布局，再把每个子图绘图区的宽、高都精确缩至原来的 0.8 倍。
    # 【修改4】第二排 C、D 额外下移 0.045 个画布高度，避免与上排图例/面板标注重叠。
    # 【修改5】左右两列向画布中心微调，使缩小后的子图仍保持紧凑间距。
    fig.subplots_adjust(left=0.12, right=0.97, top=0.90,
                        bottom=0.23 if len(conditions)==2 else 0.32,
                        wspace=0.14, hspace=0.48)
    for panel_row in range(len(conditions)):
        for panel_col in range(2):
            panel = axes[panel_row, panel_col]
            original_position = panel.get_position()
            center_x = original_position.x0 + original_position.width / 2
            center_y = original_position.y0 + original_position.height / 2
            center_x += 0.030 if panel_col == 0 else -0.045
            if len(conditions) == 2 and panel_row == 1:
                center_y -= 0.045
            new_width = original_position.width * 0.8
            new_height = original_position.height * 0.8
            panel.set_position([center_x - new_width / 2, center_y - new_height / 2,
                                new_width, new_height])
    lookup = {(r["model"], r["dataset_key"], r["condition"]): r for r in stats}
    for row, condition in enumerate(conditions):
        for column, dataset in enumerate(DATASETS):
            ax = axes[row, column]
            letter = "abcd"[row * 2 + column]
            ax.set_gid(letter)
            # 【修改8】子图编号 a–d 的字号同样放大 1.3 倍，并保留紧邻各自子图的相对位置。
            ax.text(0.02, 1.035, letter, transform=ax.transAxes,
                    fontsize=9 * font_scale, fontweight="bold")
            ax.plot([0, 1], [0, 1], linestyle=(0, (3, 3)), linewidth=0.65, color="#BBC1C5", zorder=0)
            handles = []
            for model in MODELS:
                key = (model, dataset, condition)
                if key not in lookup:
                    handles.append(Line2D([], [], color="#A2A7AB", linewidth=0,
                                           label=f"{model}: awaiting complete probabilities"))
                    continue
                group_key = f"{model}__{dataset}__{condition}"
                mean, sd = arrays[group_key + "__mean"], arrays[group_key + "__sample_sd"]
                ax.fill_between(grid, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1),
                                color=COLORS[model], alpha=0.12, linewidth=0, zorder=1)
                line, = ax.plot(np.r_[0, grid], np.r_[0, mean], color=COLORS[model],
                                linestyle=STYLES[model], linewidth=1.7 if model == "SMGF-Net" else 1.15,
                                zorder=4 if model == "SMGF-Net" else 2)
                stat = lookup[key]
                precision = 3 if condition == "clean" else 2
                label = (f"{model}  {100*stat['mean_auroc']:.{precision}f} ± "
                         f"{100*stat['sample_sd_auroc']:.{precision}f}%")
                line.set_label(label)
                handles.append(line)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.015)
            ax.set_xticks(np.linspace(0, 1, 6))
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.set_xlabel("False positive rate", labelpad=4)
            ax.set_ylabel("True positive rate", labelpad=4)
            # 【修改1】上、下、左、右外框全部封闭，外框与刻度统一 0.8 pt。
            # 【修改1】仅左侧、下侧显示朝外刻度；上侧、右侧不绘制刻度。
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
            ax.tick_params(axis="both", which="both", direction="out",
                           bottom=True, left=True, top=False, right=False,
                           length=3.0, width=0.8)
            # 【修改2】子图内部辅助网格线改为浅灰色虚线，数据线型保持原样。
            ax.grid(color="#EEF0F2", linewidth=0.45, linestyle=(0, (3, 3)))
            ax.set_axisbelow(True)
            # 【修改7】图例按子图中心水平对齐，且从原位置上移，保留横轴标题的间距。
            ax.legend(handles=handles, loc="upper center",
                      bbox_to_anchor=(0.5, -0.25),
                      borderaxespad=0, handlelength=2.5, handletextpad=0.7,
                      labelspacing=0.35)
        position = axes[row, 0].get_position()
        label = "Mixed@1.0" if condition == "mixed_1.0" else "Adaptive@1.0" if condition == "adaptive_1.0" else "Clean"
        # 【修改4、8】行标题随子图实际位置同步移动，字号扩大 1.3 倍。
        fig.text(0.025, (position.y0 + position.y1)/2, label, rotation=90,
                 rotation_mode="anchor", ha="center", va="center",
                 fontsize=8 * font_scale, fontweight="bold")
    for column, dataset in enumerate(DATASETS):
        position = axes[0, column].get_position()
        # 【修改6、8】CipherSpectrum 与 USTC-TFC2016 两个列标题下移，字号扩大 1.3 倍。
        fig.text((position.x0 + position.x1)/2, 0.92, DATASETS[dataset][0],
                 ha="center", fontsize=8.5 * font_scale, fontweight="bold")
    # 【修改3】按要求删除图底部的 Lines and bands / AUROC mean ± SD 说明行。
    if missing_groups:
        # 【修改8】不完整数据预览警告的字号同步扩大 1.3 倍。
        fig.text(0.55, 0.985, "INCOMPLETE PREVIEW — five-model comparison pending probability export",
                 ha="center", fontsize=7.5 * font_scale, fontweight="bold", color="#AD4832")
    name = "Fig5_macro_OvR_INCOMPLETE_PREVIEW" if missing_groups else (
        "Fig5_macro_OvR_ROC" if len(conditions) == 2 else "Supplementary_clean_macro_OvR_ROC")
    prefix = output / name
    require_matplotlib_panel_alignment(fig, json_out=output/"qa"/f"{name}.alignment.json",
                                      require_panel_labels=True, tolerance_pt=1.5,
                                      gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(prefix.with_suffix(".pdf"))
    fig.savefig(prefix.with_suffix(".svg"))
    fig.savefig(prefix.with_suffix(".png"), dpi=600)
    fig.savefig(prefix.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(output / f"{name}_preview.png", dpi=300)
    plt.close(fig)
    print("EXPORTED " + str(prefix), flush=True)


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=folder.parent / "新一体化实验chiper")
    parser.add_argument("--exports", type=Path, default=folder / "probability_exports")
    parser.add_argument("--output", type=Path, default=folder)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true",
                        help="Redraw from the exported AUROC CSV and curve arrays without re-reading raw probabilities")
    parser.add_argument("--incomplete-preview", action="store_true")
    parser.add_argument("--include-clean", action="store_true")
    args = parser.parse_args()
    if "云端" in str(args.data_root) or "云端" in str(args.exports):
        parser.error("Directories containing 云端 are excluded from this comparison")
    (args.output / "qa").mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        with (args.output / "auroc_mean_sample_sd.csv").open(encoding="utf-8-sig", newline="") as handle:
            stats = list(csv.DictReader(handle))
        for record in stats:
            record["mean_auroc"] = float(record["mean_auroc"])
            record["sample_sd_auroc"] = float(record["sample_sd_auroc"])
        with np.load(args.output / "macro_roc_curves.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        grid, missing = arrays["fpr"], []
    else:
        grid, arrays, stats, missing = analyze(args.data_root, args.exports, args.output)
    if not args.analyze_only:
        draw_figure(grid, arrays, stats, args.output, ("mixed_1.0", "adaptive_1.0"), args.incomplete_preview)
        if args.include_clean:
            draw_figure(grid, arrays, stats, args.output, ("clean",), args.incomplete_preview)


if __name__ == "__main__":
    main()
