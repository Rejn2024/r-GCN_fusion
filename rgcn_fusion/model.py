"""Minimal PyTorch r-GCN for evidential mass prediction."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class RGCNLayer(nn.Module):
    """Relational graph convolution with per-relation weight matrices."""

    def __init__(self, in_features: int, out_features: int, num_relations: int):
        super().__init__()
        self.relation_weights = nn.Parameter(torch.empty(num_relations, in_features, out_features))
        self.self_loop = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.relation_weights)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        out = self.self_loop(x)
        if edge_index.numel() == 0:
            return out + self.bias
        source, target = edge_index
        degree = torch.bincount(target, minlength=x.shape[0]).clamp_min(1).to(x.dtype).unsqueeze(-1)
        messages = torch.zeros(x.shape[0], self.relation_weights.shape[-1], device=x.device, dtype=x.dtype)
        for relation in range(self.relation_weights.shape[0]):
            mask = edge_type == relation
            if torch.any(mask):
                rel_source = source[mask]
                rel_target = target[mask]
                rel_msg = x[rel_source] @ self.relation_weights[relation]
                messages.index_add_(0, rel_target, rel_msg)
        return out + messages / degree + self.bias


class RGCNEvidenceModel(nn.Module):
    """r-GCN that emits normalized Dempster-Shafer masses for each node."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_relations: int,
        num_hypotheses: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_hypotheses < 1:
            raise ValueError("num_hypotheses must be positive")
        self.num_hypotheses = num_hypotheses
        self.num_masses = 2**num_hypotheses - 1
        self.conv1 = RGCNLayer(in_features, hidden_features, max(num_relations, 1))
        self.conv2 = RGCNLayer(hidden_features, hidden_features, max(num_relations, 1))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_features, self.num_masses)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index, edge_type))
        logits = self.head(x)
        return F.softmax(logits, dim=-1)
