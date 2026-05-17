# -*- coding: utf-8 -*-
import torch
import copy
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from core.models import ContrastiveGNN_V13, info_nce_loss
from core.features import augment_graph_view
from core.utils import set_seed, split_dataset_stratified, fit_and_apply_scalers

def train_experiment(dataset, num_classes, logger, device, batch_size, tag="FULL", use_gating=True, use_contrastive=True):
    """
    Two-Stage Perturbation-Aware Optimization (TPAO).
    Stage 1: Graph contrastive pre-training for latent space compactness.
    Stage 2: Cross-entropy fine-tuning for decision boundary calibration.
    """
    set_seed(42)
    train_set, test_set = split_dataset_stratified(copy.deepcopy(dataset), logger, train_ratio=0.8)
    node_scaler, stats_scaler = fit_and_apply_scalers(train_set, test_set, logger)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    model = ContrastiveGNN_V13(
        node_in_channels=5, stats_dim=9, finger_seq_len=20, entropy_seq_len=5,
        hidden_channels=32, out_channels=num_classes, use_gating=use_gating
    ).to(device)

    # Stage 1: Contrastive Pre-training
    if use_contrastive:
        logger.log(f"\n>>> [{tag}] Stage 1: Contrastive Pre-training")
        optimizer_pre = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        for epoch in range(30):
            model.train()
            total_loss = 0.0
            for batch_data in train_loader:
                data_list = batch_data.to_data_list()
                view1_list, view2_list = [], []
                for item in data_list:
                    item = item.to(device)
                    x1, ei1, ea1, _ = augment_graph_view(item)
                    x2, ei2, ea2, _ = augment_graph_view(item)
                    view1_list.append(torch.utils.data.Data(x=x1, edge_index=ei1, edge_attr=ea1) if hasattr(torch.utils.data, 'Data') else type(item)(x=x1, edge_index=ei1, edge_attr=ea1))
                    view2_list.append(type(item)(x=x2, edge_index=ei2, edge_attr=ea2))

                b1 = Batch.from_data_list(view1_list).to(device)
                b2 = Batch.from_data_list(view2_list).to(device)
                
                optimizer_pre.zero_grad()
                z1 = model.forward_contrastive(b1.x, b1.edge_index, b1.edge_attr, b1.batch)
                z2 = model.forward_contrastive(b2.x, b2.edge_index, b2.edge_attr, b2.batch)
                loss = info_nce_loss(z1, z2, temperature=0.1)
                loss.backward()
                optimizer_pre.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0: 
                logger.log(f"[{tag}] Contrastive Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f}")

    # Stage 2: Classifier Fine-tuning
    logger.log(f"\n>>> [{tag}] Stage 2: Classifier Fine-tuning")
    optimizer_fine = torch.optim.AdamW(model.parameters(), lr=0.002)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(50):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer_fine.zero_grad()
            logits = model.forward_classifier(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch, 
                batch.stats_attr, batch.finger_attr, batch.entropy_attr
            )
            loss = criterion(logits, batch.y.view(-1))
            loss.backward()
            optimizer_fine.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0: 
            logger.log(f"[{tag}] CE Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f}")

    return {'model': model, 'test_set': test_set, 'node_scaler': node_scaler, 'stats_scaler': stats_scaler}