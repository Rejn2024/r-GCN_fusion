"""Configurable PyTorch r-GCN for evidential mass prediction and node classification."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .dempster_shafer import MAX_FULL_SUBSET_HYPOTHESES, subset_masks


class RGCNLayer(nn.Module):
    """Relational graph convolution with optional basis weights and relation gates."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_relations: int,
        *,
        num_bases: int | None = None,
        relation_gates: bool = False,
    ):
        super().__init__()
        self.num_relations = max(num_relations, 1)
        self.num_bases = min(num_bases or self.num_relations, self.num_relations)
        if self.num_bases < 1:
            raise ValueError("num_bases must be positive")
        self.basis_weights = nn.Parameter(torch.empty(self.num_bases, in_features, out_features))
        self.basis_coefficients = nn.Parameter(torch.empty(self.num_relations, self.num_bases))
        self.relation_gate_logits = nn.Parameter(torch.zeros(self.num_relations)) if relation_gates else None
        self.self_loop = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.basis_weights)
        nn.init.xavier_uniform_(self.basis_coefficients)

    @property
    def relation_weights(self) -> torch.Tensor:
        """Return materialized per-relation weights."""
        return torch.einsum("rb,bio->rio", self.basis_coefficients, self.basis_weights)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        out = self.self_loop(x)
        if edge_index.numel() == 0:
            return out + self.bias
        source, target = edge_index
        degree = torch.bincount(target, minlength=x.shape[0]).clamp_min(1).to(x.dtype).unsqueeze(-1)
        weights = self.relation_weights
        gates = torch.sigmoid(self.relation_gate_logits).to(dtype=x.dtype) if self.relation_gate_logits is not None else None
        messages = torch.zeros(x.shape[0], weights.shape[-1], device=x.device, dtype=x.dtype)
        for relation in range(self.num_relations):
            mask = edge_type == relation
            if torch.any(mask):
                rel_source = source[mask]
                rel_target = target[mask]
                rel_msg = x[rel_source] @ weights[relation]
                if gates is not None:
                    rel_msg = rel_msg * gates[relation]
                messages.index_add_(0, rel_target, rel_msg)
        return out + messages / degree + self.bias


class RGCNBlock(nn.Module):
    """Residual r-GCN block with activation, normalization, and dropout."""

    def __init__(
        self,
        hidden_features: int,
        num_relations: int,
        *,
        num_bases: int | None = None,
        dropout: float = 0.1,
        residual: bool = True,
        normalization: str | None = "layernorm",
        relation_gates: bool = False,
    ):
        super().__init__()
        self.conv = RGCNLayer(
            hidden_features,
            hidden_features,
            num_relations,
            num_bases=num_bases,
            relation_gates=relation_gates,
        )
        self.norm = nn.LayerNorm(hidden_features) if normalization == "layernorm" else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.residual = residual

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        updated = self.conv(x, edge_index, edge_type)
        updated = F.gelu(updated)
        updated = self.norm(updated)
        updated = self.dropout(updated)
        return x + updated if self.residual else updated


class RGCNEvidenceModel(nn.Module):
    """r-GCN with an evidential mass head and optional node classification heads.

    Classification tasks share the same graph encoder as evidential prediction.
    Tasks whose class count matches the configured hypotheses are scored from
    the midpoint of each singleton hypothesis' belief-plausibility interval:
    ``(belief + plausibility) / 2``. Tasks with a different class count use a
    lightweight linear classifier over the shared r-GCN node embedding, allowing
    metadata targets such as radar type or aircraft variant to have their own
    vocabularies without constraining them to the hypothesis set.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_relations: int,
        num_hypotheses: int,
        dropout: float = 0.1,
        classification_tasks: dict[str, int] | None = None,
        num_layers: int = 2,
        num_bases: int | None = None,
        residual: bool = True,
        normalization: str | None = "layernorm",
        relation_gates: bool = False,
        task_head_hidden_features: int | None = None,
        mass_head_type: str = "softmax",
    ):
        super().__init__()
        if num_hypotheses < 1:
            raise ValueError("num_hypotheses must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if normalization not in {None, "none", "layernorm"}:
            raise ValueError("normalization must be one of None, 'none', or 'layernorm'")
        if mass_head_type not in {"softmax", "dirichlet"}:
            raise ValueError("mass_head_type must be 'softmax' or 'dirichlet'")
        self.num_hypotheses = num_hypotheses
        mass_masks = subset_masks(num_hypotheses)
        self.num_masses = len(mass_masks)
        self.mass_head_type = mass_head_type
        normalization = None if normalization == "none" else normalization
        self.input_projection = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features) if normalization == "layernorm" else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            [
                RGCNBlock(
                    hidden_features,
                    max(num_relations, 1),
                    num_bases=num_bases,
                    dropout=dropout,
                    residual=residual,
                    normalization=normalization,
                    relation_gates=relation_gates,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.head = self._make_head(hidden_features, self.num_masses, task_head_hidden_features, dropout)

        classification_tasks = classification_tasks or {}
        invalid_tasks = [name for name, num_classes in classification_tasks.items() if num_classes < 2]
        if invalid_tasks:
            raise ValueError(f"classification tasks must have at least two classes: {invalid_tasks}")
        if any("." in name for name in classification_tasks):
            raise ValueError("classification task names must not contain '.'")
        self.classification_tasks = tuple(classification_tasks)
        self.ds_classification_tasks = tuple(
            name for name, num_classes in classification_tasks.items() if num_classes == num_hypotheses
        )
        self.classification_heads = nn.ModuleDict(
            {
                name: self._make_head(hidden_features, int(num_classes), task_head_hidden_features, dropout)
                for name, num_classes in classification_tasks.items()
                if num_classes != num_hypotheses
            }
        )
        if num_hypotheses <= MAX_FULL_SUBSET_HYPOTHESES:
            masks = torch.tensor(mass_masks, dtype=torch.long)
            singleton_masks = 1 << torch.arange(num_hypotheses, dtype=torch.long)
            singleton_indices = torch.tensor([mass_masks.index(int(mask)) for mask in singleton_masks], dtype=torch.long)
            plausibility_mask = (masks.unsqueeze(0) & singleton_masks.unsqueeze(1)) != 0
        else:
            singleton_indices = torch.arange(num_hypotheses, dtype=torch.long)
            plausibility_mask = torch.cat(
                [torch.eye(num_hypotheses, dtype=torch.bool), torch.ones(num_hypotheses, 1, dtype=torch.bool)],
                dim=1,
            )
        self.register_buffer("_singleton_indices", singleton_indices, persistent=False)
        self.register_buffer("_plausibility_mask", plausibility_mask, persistent=False)

    @staticmethod
    def _make_head(
        hidden_features: int,
        out_features: int,
        head_hidden_features: int | None,
        dropout: float,
    ) -> nn.Module:
        if head_hidden_features is None or head_hidden_features <= 0:
            return nn.Linear(hidden_features, out_features)
        return nn.Sequential(
            nn.Linear(hidden_features, head_hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_features, out_features),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """Return shared r-GCN node embeddings."""
        x = self.input_projection(x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_type)
        return x

    def interval_midpoints(self, masses: torch.Tensor) -> torch.Tensor:
        """Return singleton belief-plausibility midpoint scores from mass predictions."""
        belief = masses.index_select(dim=-1, index=self._singleton_indices)
        plausibility = masses @ self._plausibility_mask.to(dtype=masses.dtype, device=masses.device).T
        return (belief + plausibility) / 2.0

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        embeddings = self.encode(x, edge_index, edge_type)
        logits = self.head(embeddings)
        if self.mass_head_type == "dirichlet":
            evidence = F.softplus(logits)
            alpha = evidence + 1.0
            masses = alpha / alpha.sum(dim=-1, keepdim=True)
            uncertainty = self.num_masses / alpha.sum(dim=-1, keepdim=True)
        else:
            evidence = None
            alpha = None
            uncertainty = None
            masses = F.softmax(logits, dim=-1)
        midpoint_scores = self.interval_midpoints(masses)
        classification_scores = {
            name: midpoint_scores if name in self.ds_classification_tasks else self.classification_heads[name](embeddings)
            for name in self.classification_tasks
        }
        return {
            "masses": masses,
            "mass_logits": logits,
            "dirichlet_alpha": alpha,
            "dirichlet_evidence": evidence,
            "uncertainty": uncertainty,
            "classification_logits": classification_scores,
        }
