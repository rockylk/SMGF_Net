# -*- coding: utf-8 -*-
import os
import time
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from core.utils import compute_ece, save_simple_csv
from core.attacks import build_attacked_dataset

def predict_closed_set(model, dataset, device, batch_size):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model.forward_classifier(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                batch.stats_attr, batch.finger_attr, batch.entropy_attr
            )
            probs = F.softmax(logits, dim=1)
            all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
            all_labels.extend(batch.y.view(-1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.asarray(all_preds), np.asarray(all_labels), np.asarray(all_probs)

def evaluate_clean_classification(bundle, logger, device, batch_size):
    logger.log("\n" + "=" * 60 + "\nExperiment 1: Clean Classification (No Attack)\n" + "=" * 60)
    preds, labels, probs = predict_closed_set(bundle['model'], bundle['test_set'], device, batch_size)
    acc = accuracy_score(labels, preds) * 100.0
    macro_f1 = f1_score(labels, preds, average='macro') * 100.0
    ece = compute_ece(probs, labels)
    logger.log(f"[Clean] Accuracy: {acc:.2f}% | Macro-F1: {macro_f1:.2f}% | ECE: {ece:.2f}%")
    return acc, macro_f1, ece

def evaluate_attack_once(bundle, logger, device, batch_size, attack_name, strength):
    clean_preds, clean_labels, _ = predict_closed_set(bundle['model'], bundle['test_set'], device, batch_size)
    clean_acc = accuracy_score(clean_labels, clean_preds) * 100.0
    
    attacked_set = build_attacked_dataset(bundle['test_set'], attack_name, bundle['node_scaler'], bundle['stats_scaler'], strength)
    atk_preds, atk_labels, atk_probs = predict_closed_set(bundle['model'], attacked_set, device, batch_size)
    
    robust_acc = accuracy_score(atk_labels, atk_preds) * 100.0
    robust_f1 = f1_score(atk_labels, atk_preds, average='macro') * 100.0
    ece = compute_ece(atk_probs, atk_labels)
    
    clean_correct = (clean_preds == clean_labels)
    asr = (np.sum(clean_correct & (atk_preds != atk_labels)) / max(np.sum(clean_correct), 1)) * 100.0
    rel_drop = ((clean_acc - robust_acc) / max(clean_acc, 1e-8)) * 100.0
    
    logger.log(f"[Attack: {attack_name:8s} | Str: {strength:.1f}] Rob_Acc: {robust_acc:.2f}% | Rob_F1: {robust_f1:.2f}% | ASR: {asr:.2f}% | Drop: {rel_drop:.2f}% | ECE: {ece:.2f}%")
    return robust_acc, robust_f1, asr, rel_drop, ece

def run_multi_attack_benchmark(bundle, logger, device, batch_size, result_dir):
    logger.log("\n" + "=" * 60 + "\nExperiment 2: Multiple Perturbation Robustness\n" + "=" * 60)
    attacks, strengths, rows = ['padding', 'iat_jitter', 'dummy', 'mixed'], [0.3, 0.6, 0.9], []
    for atk in attacks:
        for s in strengths:
            r_acc, r_f1, asr, drop, ece = evaluate_attack_once(bundle, logger, device, batch_size, atk, s)
            rows.append([atk, s, f"{r_acc:.2f}", f"{r_f1:.2f}", f"{asr:.2f}", f"{drop:.2f}", f"{ece:.2f}"])
    save_simple_csv(os.path.join(result_dir, "exp2_multi_attack.csv"), rows, ["Attack", "Strength", "Robust_Acc", "Robust_F1", "ASR", "Relative_Drop", "ECE"])

def run_adaptive_attack_benchmark(bundle, logger, device, batch_size, result_dir):
    logger.log("\n" + "=" * 60 + "\nExperiment 3: Adaptive Attack Robustness (Semantic Preserved)\n" + "=" * 60)
    strengths, rows = [0.4, 0.7, 1.0], []
    for s in strengths:
        r_acc, r_f1, asr, drop, ece = evaluate_attack_once(bundle, logger, device, batch_size, "adaptive", s)
        rows.append(["adaptive", s, f"{r_acc:.2f}", f"{r_f1:.2f}", f"{asr:.2f}", f"{drop:.2f}", f"{ece:.2f}"])
    save_simple_csv(os.path.join(result_dir, "exp3_adaptive_attack.csv"), rows, ["Attack", "Strength", "Robust_Acc", "Robust_F1", "ASR", "Relative_Drop", "ECE"])

def run_representation_visualization(bundle, logger, device, batch_size, result_dir):
    logger.log("\n" + "=" * 60 + "\nExperiment 5: Representation Visualization (t-SNE)\n" + "=" * 60)
    def collect_embeddings(dataset, max_points=400):
        bundle['model'].eval()
        idx = np.random.choice(len(dataset), min(max_points, len(dataset)), replace=False)
        loader = DataLoader([dataset[i] for i in idx], batch_size=batch_size, shuffle=False)
        all_feats, all_labels = [], []
        with torch.no_grad():
            for b in loader:
                b = b.to(device)
                all_feats.extend(bundle['model'].extract_fused_features(b.x, b.edge_index, b.edge_attr, b.batch, b.stats_attr, b.finger_attr, b.entropy_attr).cpu().numpy())
                all_labels.extend(b.y.view(-1).cpu().numpy())
        return np.asarray(all_feats, dtype=np.float32), np.asarray(all_labels, dtype=np.int64)

    c_feats, c_labels = collect_embeddings(bundle['test_set'])
    atk_set = build_attacked_dataset(bundle['test_set'], 'mixed', bundle['node_scaler'], bundle['stats_scaler'], 0.7)
    a_feats, a_labels = collect_embeddings(atk_set)

    logger.log("[Vis] Processing t-SNE reduction...")
    Z = TSNE(n_components=2, random_state=42, perplexity=30, init='pca').fit_transform(np.vstack([c_feats, a_feats]))
    y, domain = np.concatenate([c_labels, a_labels]), np.array(['clean'] * len(c_feats) + ['attack'] * len(a_feats))

    plt.figure(figsize=(10, 8))
    cmap = plt.cm.get_cmap('tab10', len(np.unique(y)))
    for i, cls in enumerate(np.unique(y)):
        plt.scatter(Z[(y == cls) & (domain == 'clean'), 0], Z[(y == cls) & (domain == 'clean'), 1], s=20, alpha=0.7, color=cmap(i), marker='o', label=f'C-{cls} Clean')
        plt.scatter(Z[(y == cls) & (domain == 'attack'), 0], Z[(y == cls) & (domain == 'attack'), 1], s=30, alpha=0.9, color=cmap(i), marker='x', label=f'C-{cls} Attacked')

    plt.title("t-SNE: Clean vs Mixed Attack Representations")
    plt.tight_layout()
    fig_path = os.path.join(result_dir, "exp5_tsne_robustness.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.log(f"[Saved Image] {fig_path}")

def run_deployment_overhead(bundle, logger, device):
    logger.log("\n" + "=" * 60 + "\nExperiment 6: Deployment Overhead (Model-only, GPU bound)\n" + "=" * 60)
    model, dataset = bundle['model'], bundle['test_set'][:500]
    loader = DataLoader(dataset, batch_size=1, shuffle=False) # BS=1 to simulate streaming line-rate processing

    model.eval()
    with torch.no_grad():
        for i, b in enumerate(loader):
            if i >= 50: break
            _ = model.forward_classifier(b.x.to(device), b.edge_index.to(device), b.edge_attr.to(device), b.batch.to(device), b.stats_attr.to(device), b.finger_attr.to(device), b.entropy_attr.to(device))

    torch.cuda.synchronize()
    total_t, sample_count = 0.0, 0
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model.forward_classifier(b.x, b.edge_index, b.edge_attr, b.batch, b.stats_attr, b.finger_attr, b.entropy_attr)
            torch.cuda.synchronize()
            total_t += time.perf_counter() - t0
            sample_count += b.y.size(0)

    logger.log(f"  * Trainable Parameters  : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logger.log(f"  * Model-only Latency    : {(total_t / max(sample_count, 1)) * 1000.0:.3f} ms / sample")
    logger.log(f"  * Model-only Throughput : {sample_count / max(total_t, 1e-8):.2f} samples / second")