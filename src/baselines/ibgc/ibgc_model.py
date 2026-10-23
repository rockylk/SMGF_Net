"""GCN architecture aligned with authors' GraphNet at pinned upstream commit."""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class IBGC(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = GCNConv(13, 16)
        self.conv2 = GCNConv(16, 64)
        self.conv3 = GCNConv(64, 128)
        self.lin = nn.Linear(128, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return F.log_softmax(self.lin(F.dropout(x, 0.025, self.training)), dim=1)
