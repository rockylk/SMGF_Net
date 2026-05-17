# -*- coding: utf-8 -*-
import copy
import random
import torch
import numpy as np

def clone_data(data): 
    return copy.deepcopy(data)

def sanitize_raw_x(raw_x):
    """Fix inverse scaling precision issues to ensure strict physical boundaries."""
    raw_x = raw_x.copy()
    raw_x[:, 0] = (raw_x[:, 0] >= 0.5).astype(np.float32)   # direction to binary
    raw_x[:, 1] = np.clip(raw_x[:, 1], 1.0, None)           # size > 0
    raw_x[:, 2] = np.clip(raw_x[:, 2], 1e-6, None)          # iat > 0
    return raw_x

def rebuild_sequential_edges_from_raw_x(raw_x):
    num_nodes = raw_x.shape[0]
    edge_index, edge_attr = [], []
    for i in range(num_nodes - 1):
        edge_index.extend([[i, i + 1], [i + 1, i]])
        edge_attr.extend([[float(max(raw_x[i + 1, 2], 1e-6))] * 2])
    
    if not edge_index: return torch.tensor([[0], [0]], dtype=torch.long), torch.tensor([[0.0]], dtype=torch.float)
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous(), torch.tensor(edge_attr, dtype=torch.float)

def update_temporal_columns(raw_x):
    raw_x = raw_x.copy()
    sizes, iats = np.clip(raw_x[:, 1], 1.0, None), np.clip(raw_x[:, 2], 1e-6, None)
    for i in range(len(raw_x)):
        raw_x[i, 3], raw_x[i, 4] = float(np.mean(sizes[max(0, i - 4):i + 1])), float(np.mean(iats[max(0, i - 4):i + 1]))
    raw_x[:, 1], raw_x[:, 2] = sizes, iats
    return raw_x

def recompute_stats_from_raw_x(raw_x):
    raw_x = sanitize_raw_x(raw_x)
    sizes, iats, dirs = raw_x[:, 1], raw_x[:, 2], raw_x[:, 0]
    return np.array([
        np.log1p(len(raw_x)), np.log1p(float(np.sum(iats))), np.log1p(float(np.sum(sizes))),
        float(np.mean(sizes)), float(np.mean(iats)), float(np.std(sizes)), float(np.std(iats)),
        float(np.mean(sizes > 1200)), float(np.mean(dirs < 0.5))
    ], dtype=np.float32)

def build_fingerprint_from_raw_x(raw_x, seq_len=20):
    raw_x = sanitize_raw_x(raw_x)
    dirs, sizes = raw_x[:, 0], np.clip(raw_x[:, 1], 1.0, None)
    signed = np.where(dirs < 0.5, sizes, -sizes).astype(np.float32)
    if len(signed) >= seq_len: return signed[:seq_len].astype(np.float32)
    return np.pad(signed, (0, seq_len - len(signed)), constant_values=0.0).astype(np.float32)

def inverse_raw_features(data, node_scaler, stats_scaler):
    raw_x = sanitize_raw_x(node_scaler.inverse_transform(data.x.cpu().numpy()))
    raw_stats = stats_scaler.inverse_transform(data.stats_attr.cpu().numpy())[0]
    return raw_x, raw_stats

def finalize_attacked_data(data, raw_x, node_scaler, stats_scaler, new_finger=None, new_entropy=None):
    raw_x = update_temporal_columns(raw_x)
    attacked = clone_data(data)
    attacked.x = torch.tensor(node_scaler.transform(raw_x), dtype=torch.float)
    attacked.stats_attr = torch.tensor(stats_scaler.transform(recompute_stats_from_raw_x(raw_x).reshape(1, -1)), dtype=torch.float)
    attacked.edge_index, attacked.edge_attr = rebuild_sequential_edges_from_raw_x(raw_x)
    attacked.num_nodes = raw_x.shape[0]
    if new_finger is not None: attacked.finger_attr = torch.tensor(new_finger.reshape(1, -1), dtype=torch.float)
    if new_entropy is not None: attacked.entropy_attr = torch.tensor(new_entropy.reshape(1, -1), dtype=torch.float)
    return attacked

def attack_packet_length_padding(data, node_scaler, stats_scaler, strength=0.5, preserve_semantic=False):
    raw_x, _ = inverse_raw_features(data, node_scaler, stats_scaler)
    idx = np.random.choice(len(raw_x), size=min(max(1, int(len(raw_x) * (0.10 + 0.20 * strength))), len(raw_x)), replace=False)
    raw_x[idx, 1] = np.clip(raw_x[idx, 1] + np.random.randint(20, max(int(64 + 512 * strength), 21), size=len(idx)), 1.0, None)
    return finalize_attacked_data(data, raw_x, node_scaler, stats_scaler, data.finger_attr.cpu().numpy()[0] if preserve_semantic else build_fingerprint_from_raw_x(raw_x), data.entropy_attr.cpu().numpy()[0])

def attack_iat_jitter(data, node_scaler, stats_scaler, strength=0.5, preserve_semantic=False):
    raw_x, _ = inverse_raw_features(data, node_scaler, stats_scaler)
    idx = np.random.choice(len(raw_x), size=min(max(1, int(len(raw_x) * (0.20 + 0.30 * strength))), len(raw_x)), replace=False)
    raw_x[idx, 2] = np.clip(raw_x[idx, 2] * np.random.uniform(1.0 - (0.10 + 0.70 * strength), 1.0 + (0.10 + 0.70 * strength), size=len(idx)), 1e-6, None)
    return finalize_attacked_data(data, raw_x, node_scaler, stats_scaler, data.finger_attr.cpu().numpy()[0], data.entropy_attr.cpu().numpy()[0])

def attack_dummy_injection(data, node_scaler, stats_scaler, strength=0.5, preserve_semantic=False):
    raw_x, _ = inverse_raw_features(data, node_scaler, stats_scaler)
    median_iat = max(float(np.median(np.clip(raw_x[:, 2], 1e-6, None))), 1e-6)
    new_list, offset = raw_x.tolist(), 0
    for p in sorted(np.random.randint(1, max(2, len(raw_x)), size=max(1, int(len(raw_x) * (0.03 + 0.10 * strength)))).tolist()):
        base = np.array(new_list[min(max(p - 1 + offset, 0), len(new_list) - 1)], dtype=np.float32)
        new_list.insert(min(p + offset, len(new_list)), [base[0] if random.random() < 0.7 else (1 - base[0]), random.uniform(40, 200 + 400 * strength), max(median_iat * random.uniform(0.2, 1.0), 1e-6)] * 2)
        offset += 1
    return finalize_attacked_data(data, np.array(new_list, dtype=np.float32), node_scaler, stats_scaler, data.finger_attr.cpu().numpy()[0] if preserve_semantic else build_fingerprint_from_raw_x(np.array(new_list, dtype=np.float32)), data.entropy_attr.cpu().numpy()[0])

def attack_mixed(data, node_scaler, stats_scaler, strength=0.5, preserve_semantic=False):
    d1 = attack_packet_length_padding(data, node_scaler, stats_scaler, strength, preserve_semantic)
    d2 = attack_iat_jitter(d1, node_scaler, stats_scaler, strength, preserve_semantic)
    return attack_dummy_injection(d2, node_scaler, stats_scaler, strength, preserve_semantic)

def apply_attack(data, attack_name, node_scaler, stats_scaler, strength=0.5):
    if attack_name is None or attack_name.lower() == 'clean': return clone_data(data)
    name = attack_name.lower()
    if name == 'padding': return attack_packet_length_padding(data, node_scaler, stats_scaler, strength)
    elif name == 'iat_jitter': return attack_iat_jitter(data, node_scaler, stats_scaler, strength)
    elif name == 'dummy': return attack_dummy_injection(data, node_scaler, stats_scaler, strength)
    elif name == 'mixed': return attack_mixed(data, node_scaler, stats_scaler, strength, preserve_semantic=False)
    elif name == 'adaptive': return attack_mixed(data, node_scaler, stats_scaler, strength, preserve_semantic=True)
    raise ValueError(f"Unsupported attack: {name}")

def build_attacked_dataset(dataset, attack_name, node_scaler, stats_scaler, strength=0.5):
    return [apply_attack(d, attack_name, node_scaler, stats_scaler, strength) for d in dataset]