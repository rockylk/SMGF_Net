# -*- coding: utf-8 -*-
"""
SMGF-Net Main Execution Script
Author: Kai Liang et al.
"""
import os
import time
import torch

from core.utils import Logger
from core.features import load_dataset_closedset
from engine.trainer import train_experiment
from engine.evaluator import (
    evaluate_clean_classification, run_multi_attack_benchmark, 
    run_adaptive_attack_benchmark, run_representation_visualization, run_deployment_overhead
)

# ----------------- CONFIGURATION -----------------
BATCH_SIZE = 2048
PCAP_DIR = r"C:\Desktop\GNN\aes-128-gcm\aes-128-gcm"
RESULT_DIR = r"C:\GNN\RESULT"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# -------------------------------------------------

def run_ablation_study(dataset, num_classes, logger):
    logger.log("\n" + "=" * 60 + "\nExperiment 4: Ablation Study (Gating & Contrastive)\n" + "=" * 60)
    configs = [
        {"tag": "FULL", "use_gating": True, "use_contrastive": True},
        {"tag": "NO_GATING", "use_gating": False, "use_contrastive": True},
        {"tag": "NO_CONTRASTIVE", "use_gating": True, "use_contrastive": False},
    ]
    rows = []
    from engine.evaluator import evaluate_attack_once
    from core.utils import save_simple_csv

    for cfg in configs:
        bundle = train_experiment(dataset, num_classes, logger, DEVICE, BATCH_SIZE, tag=cfg['tag'], use_gating=cfg['use_gating'], use_contrastive=cfg['use_contrastive'])
        c_acc, _, c_ece = evaluate_clean_classification(bundle, logger, DEVICE, BATCH_SIZE)
        m_acc, _, m_asr, m_drop, _ = evaluate_attack_once(bundle, logger, DEVICE, BATCH_SIZE, "mixed", 0.6)
        a_acc, _, a_asr, a_drop, _ = evaluate_attack_once(bundle, logger, DEVICE, BATCH_SIZE, "adaptive", 0.7)
        rows.append([cfg['tag'], f"{c_acc:.2f}", f"{c_ece:.2f}", f"{m_acc:.2f}", f"{m_asr:.2f}", f"{m_drop:.2f}", f"{a_acc:.2f}", f"{a_asr:.2f}", f"{a_drop:.2f}"])

    save_simple_csv(os.path.join(RESULT_DIR, "exp4_ablation_study.csv"), rows, [
        "Setting", "Clean_Acc", "Clean_ECE", "Mixed_Rob_Acc", "Mixed_ASR", "Mixed_Drop", "Adapt_Rob_Acc", "Adapt_ASR", "Adapt_Drop"
    ])


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is strictly required for this framework.")

    os.makedirs(RESULT_DIR, exist_ok=True)
    LOG_FILE = os.path.join(RESULT_DIR, f"V14_Robust_Exp_Log_{int(time.time())}.txt")
    logger = Logger(LOG_FILE, BATCH_SIZE)

    if not os.path.exists(PCAP_DIR):
        raise FileNotFoundError(f"Dataset path not found: {PCAP_DIR}")

    # 1. Load Dataset
    dataset, num_classes, _ = load_dataset_closedset(PCAP_DIR, logger)
    if len(dataset) == 0: raise RuntimeError("Failed to load valid samples!")

    # 2. Train Full Baseline (TPAO + Dynamic Gating)
    full_bundle = train_experiment(dataset, num_classes, logger, DEVICE, BATCH_SIZE, tag="FULL_BASELINE")

    # 3. Execute Benchmarks
    evaluate_clean_classification(full_bundle, logger, DEVICE, BATCH_SIZE)
    run_multi_attack_benchmark(full_bundle, logger, DEVICE, BATCH_SIZE, RESULT_DIR)
    run_adaptive_attack_benchmark(full_bundle, logger, DEVICE, BATCH_SIZE, RESULT_DIR)
    run_ablation_study(dataset, num_classes, logger)
    run_representation_visualization(full_bundle, logger, DEVICE, BATCH_SIZE, RESULT_DIR)
    run_deployment_overhead(full_bundle, logger, DEVICE)

    logger.log("\n>>> All robustness evaluations and metrics successfully recorded!")