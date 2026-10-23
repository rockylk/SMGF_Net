# -*- coding: utf-8 -*-
"""PEAN-NHP architecture: PEAN network with No Header/Payload input.

The network dimensions, Transformer, BiLSTM, fusion head, auxiliary heads,
dropout and three-CE objective are intentionally identical to PEAN V17.
The data pipeline enforces a traffic-independent constant byte tensor, so
only the native packet-length branch carries sample-dependent information.
This is a constrained-input PEAN variant, not native-input official PEAN.
"""
from __future__ import annotations

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

UPSTREAM_DIR = Path(__file__).resolve().parent / "official_sources" / "PEAN"
if not (UPSTREAM_DIR / "TRF.py").is_file():
    raise FileNotFoundError(f"Official PEAN TRF module missing: {UPSTREAM_DIR}")
if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))
import TRF  # noqa: E402

class PEANNHPV17(nn.Module):
    """PEAN architecture retained for the NHP constrained-input variant.

    PEAN Github PEAN_model.py:
    bytes -> Embedding(261,128) -> mean over all 400 positions (including PAD)
          -> original TRF positional encoding + 2 self-attention encoders
          -> flatten [10,128]=1280
    lengths -> Embedding(2000,32) -> 2-layer bidirectional LSTM(128) -> last
          -> 256
    fused FC([1280,256], 41), auxiliary FC(1280,41), FC(256,41).
    """
    def __init__(self, num_classes=41, device="cpu"):
        super().__init__()
        self.emb = nn.Embedding(261, 128, padding_idx=0)
        self.length_embedding = nn.Embedding(2000, 32, padding_idx=0)
        self.lenlstm = nn.LSTM(32, 128, num_layers=2, bidirectional=True,
                              batch_first=True, dropout=0.5)
        class TrfConfig:
            embedding_size=128
            pad_num=10
            trf_heads=8
            trf_layers=2
            dropout=0.5
        config=TrfConfig()
        config.device=torch.device(device)
        self.TRF = TRF.Model(config)
        self.fc = nn.Linear(128*10+128*2, num_classes)
        self.fc01 = nn.Linear(128*10, num_classes)
        self.fc02 = nn.Linear(128*2, num_classes)
        self.num_classes = num_classes

    def forward(self, packet_bytes, length_seq):
        if packet_bytes.ndim != 3 or packet_bytes.shape[1:] != (10,400):
            raise RuntimeError(f"PEAN-NHP expects byte IDs [B,10,400], got {packet_bytes.shape}")
        if length_seq.ndim != 2 or length_seq.shape[1] != 10:
            raise RuntimeError(f"PEAN-NHP expects lengths [B,10], got {length_seq.shape}")
        expected = torch.zeros_like(packet_bytes)
        expected[:, :, 0] = 1
        expected[:, :, 1] = 2
        if not torch.equal(packet_bytes, expected):
            raise RuntimeError(
                "PEAN-NHP fairness violation: byte view is not constant "
                "[CLS,SEP,PAD...]"
            )
        # Vectorized exact equivalent of author's per-packet mean over [B,400,128].
        packet_embed = self.emb(packet_bytes).mean(dim=2)  # B,10,128
        out1 = self.TRF(packet_embed)                      # B,1280
        out2, _ = self.lenlstm(self.length_embedding(length_seq))
        out2 = out2[:, -1, :]                              # B,256
        fused = self.fc(torch.cat((out1,out2),dim=1))
        byte_head = self.fc01(out1)
        length_head = self.fc02(out2)
        return fused, byte_head, length_head

    @staticmethod
    def supervised_loss(outputs, labels):
        # Upstream train_eval.py --imploss True uses an UNWEIGHTED SUM (not mean).
        return sum(F.cross_entropy(logits,labels) for logits in outputs)
