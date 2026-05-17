# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from scapy.all import rdpcap, IP, TCP, UDP, Raw

def get_direction(pkt, client_ip):
    try: return 0 if pkt[IP].src == client_ip else 1
    except: return 0

def calculate_entropy(data):
    """Calculate Shannon entropy for payload to serve as a micro-semantic anchor."""
    if not data: return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = counts / len(data)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def extract_features_v13(packets, client_ip, max_nodes=400):
    """Extract multi-modal features: node sequence, global stats, TLS fingerprint, and entropy."""
    node_features, sizes, times = [], [], []
    start_time = float(packets[0].time)
    total_bytes, large_pkt_count, uplink_count = 0, 0, 0
    pkt_count = len(packets)
    duration = float(packets[-1].time) - start_time

    handshake_len = 20
    raw_fingerprint, entropy_seq = [], []

    prev_time = start_time
    for pkt in packets[:max_nodes]:
        curr_time = float(pkt.time)
        size = len(pkt)
        direction = get_direction(pkt, client_ip)
        iat = curr_time - prev_time

        # Cryptographic semantic anchor 1: Handshake fingerprint
        signed_size = size if direction == 0 else -size
        if len(raw_fingerprint) < handshake_len:
            raw_fingerprint.append(signed_size)

        # Cryptographic semantic anchor 2: Payload entropy
        if len(entropy_seq) < 5 and Raw in pkt:
            payload = pkt[Raw].load
            if len(payload) > 0:
                ent = calculate_entropy(payload[:128])
                entropy_seq.append(ent if direction == 0 else -ent)

        total_bytes += size
        if size > 1200: large_pkt_count += 1
        if direction == 0: uplink_count += 1

        sizes.append(size)
        times.append(iat)
        node_features.append([direction, size, iat, np.mean(sizes[-5:]), np.mean(times[-5:])])
        prev_time = curr_time

    while len(raw_fingerprint) < handshake_len: raw_fingerprint.append(0)
    while len(entropy_seq) < 5: entropy_seq.append(0.0)

    stats_features = [
        np.log1p(pkt_count), np.log1p(duration), np.log1p(total_bytes),
        np.mean(sizes) if sizes else 0, np.mean(times) if times else 0, 
        np.std(sizes) if sizes else 0, np.std(times) if times else 0, 
        (large_pkt_count / pkt_count) if pkt_count > 0 else 0, 
        (uplink_count / pkt_count) if pkt_count > 0 else 0
    ]

    return (
        np.array(node_features, dtype=np.float32),
        np.array(stats_features, dtype=np.float32),
        np.array(raw_fingerprint, dtype=np.float32),
        np.array(entropy_seq, dtype=np.float32)
    )

def pcap_to_graph_v13(pcap_path, label_int):
    """Convert a PCAP file into a PyTorch Geometric Data object."""
    try:
        packets = rdpcap(pcap_path)
        packets = [p for p in packets if IP in p and (TCP in p or UDP in p)]
        if len(packets) < 5: return None
        client_ip = packets[0][IP].src
    except: return None

    node_feats, stats_feats, finger_feats, entropy_feats = extract_features_v13(packets, client_ip)

    x = torch.tensor(node_feats, dtype=torch.float)
    stats_x = torch.tensor(stats_feats, dtype=torch.float).unsqueeze(0)
    finger_x = torch.tensor(finger_feats, dtype=torch.float).unsqueeze(0)
    entropy_x = torch.tensor(entropy_feats, dtype=torch.float).unsqueeze(0)

    num_nodes = x.size(0)
    edge_index, edge_attr = [], []
    for i in range(num_nodes - 1):
        edge_index.extend([[i, i + 1], [i + 1, i]])
        edge_attr.extend([[x[i + 1, 2]], [x[i + 1, 2]]])

    if not edge_index:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr = torch.tensor([[0.0]], dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr,
        y=torch.tensor([label_int], dtype=torch.long),
        stats_attr=stats_x, finger_attr=finger_x,
        entropy_attr=entropy_x, num_nodes=num_nodes
    )

def augment_graph_view(data, crop_ratio_range=(0.5, 0.9), mask_prob=0.15):
    """Generate structural masked views for contrastive learning."""
    device = data.x.device
    num_nodes = data.num_nodes if not isinstance(data.num_nodes, torch.Tensor) else data.num_nodes.item()
    min_crop = max(5, int(num_nodes * crop_ratio_range[0]))
    max_crop = int(num_nodes * crop_ratio_range[1])

    if num_nodes > min_crop:
        crop_len = random.randint(min_crop, min(max_crop, num_nodes))
        start_idx = random.randint(0, num_nodes - crop_len)
        subset = torch.arange(start_idx, start_idx + crop_len, device=device)
        edge_index, edge_attr = subgraph(subset, data.edge_index, data.edge_attr, relabel_nodes=True)
        x = data.x[subset]
    else:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

    mask = torch.rand(x.shape, device=device) > mask_prob
    x = x * mask.float()
    return x, edge_index, edge_attr, torch.zeros(x.size(0), dtype=torch.long, device=device)

def load_dataset_closedset(root_dir, logger):
    """Load and parse PCAP files from directory."""
    subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {name: i for i, name in enumerate(subdirs)}
    logger.log(f"Loading Dataset: Total {len(subdirs)} classes. (Closed-set formulation)")
    
    dataset = []
    for class_name in subdirs:
        class_dir = os.path.join(root_dir, class_name)
        label_int = class_to_idx[class_name]
        for f in [f for f in os.listdir(class_dir) if f.endswith((".pcap", ".pcapng"))]:
            data = pcap_to_graph_v13(os.path.join(class_dir, f), label_int)
            if data is not None and data.x.size(0) > 0: dataset.append(data)
    return dataset, len(subdirs), {i: n for n, i in class_to_idx.items()}