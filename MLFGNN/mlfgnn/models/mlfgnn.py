"""The MLF-GNN model, held fixed across all three graph-construction arms.

Architecture follows Peng et al. Section 3.3:

    feature fusion   V_PETCT = [V_PET | V_CT] -> GNN -> GNN -> MLP -> (+clinical) -> Cox
    network fusion   V_PET  -> GNN_1 ----.
                     V_CT   -> GNN_1 ----+-> fuse -> GNN_2 -> MLP -> (+clinical) -> Cox

Network fusion is the configuration that reached C-index 0.788 in the original
paper and is therefore the one carried into the graph-construction comparison.

A note on faithfulness. The paper describes network fusion as "the features and
the edges ... were separately concatenated and averaged", which is ambiguous
about which operation applies to which. This implementation concatenates the
two branches' node features (preserving both modalities' information, which
"averaged" would destroy) and averages their edge weights (a single fused edge
set is required for the second layer). Both choices are exposed as
``feature_fusion_op`` / ``edge_fusion_op`` so the alternative reading can be
run as an ablation rather than argued about.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layers import make_layer

FUSIONS = ("pet", "ct", "feature", "network")


class CoxHead(nn.Module):
    """MLP + Cox layer, with the clinical covariates injected before the Cox layer.

    Peng et al. integrate age, T_stage, N_stage and site immediately before the
    Cox layer -- i.e. the clinical variables bypass message passing entirely and
    are never smoothed over the population graph. That placement is preserved:
    it is also the reason the clinical signal survives a bad graph, and hence
    part of why the reported drop under domain shift is smaller than the graph
    quality alone would suggest.
    """

    def __init__(self, in_dim: int, clinical_dim: int = 0, hidden: int = 32,
                 dropout: float = 0.5) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cox = nn.Linear(hidden + clinical_dim, 1, bias=False)

    def forward(self, h: Tensor, clinical: Tensor | None = None) -> Tensor:
        h = self.mlp(h)
        if clinical is not None and clinical.size(1) > 0:
            h = torch.cat([h, clinical], dim=1)
        return self.cox(h).squeeze(-1)


class MLFGNN(nn.Module):
    """Multi-level fusion GNN for PFS risk prediction."""

    def __init__(
        self,
        pet_dim: int,
        ct_dim: int,
        clinical_dim: int = 0,
        hidden_dim: int = 32,
        layer_type: str = "GAT",
        fusion: str = "network",
        heads: int = 4,
        dropout: float = 0.5,
        feature_fusion_op: str = "concat",
        edge_fusion_op: str = "mean",
    ) -> None:
        super().__init__()
        if fusion not in FUSIONS:
            raise ValueError(f"unknown fusion {fusion!r}; expected one of {FUSIONS}")
        self.fusion = fusion
        self.dropout = dropout
        self.feature_fusion_op = feature_fusion_op
        self.edge_fusion_op = edge_fusion_op

        def layer(i: int, o: int):
            return make_layer(layer_type, i, o, heads=heads, dropout=dropout)

        if fusion in ("pet", "ct"):
            in_dim = pet_dim if fusion == "pet" else ct_dim
            self.gnn1 = layer(in_dim, hidden_dim)
            self.gnn2 = layer(hidden_dim, hidden_dim)
            head_in = hidden_dim
        elif fusion == "feature":
            self.gnn1 = layer(pet_dim + ct_dim, hidden_dim)
            self.gnn2 = layer(hidden_dim, hidden_dim)
            head_in = hidden_dim
        else:  # network
            self.gnn_pet = layer(pet_dim, hidden_dim)
            self.gnn_ct = layer(ct_dim, hidden_dim)
            fused_dim = hidden_dim * 2 if feature_fusion_op == "concat" else hidden_dim
            self.gnn2 = layer(fused_dim, hidden_dim)
            head_in = hidden_dim

        self.head = CoxHead(head_in, clinical_dim, hidden_dim, dropout)

    # -- forward -------------------------------------------------------------
    def forward(
        self,
        x_pet: Tensor,
        x_ct: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        clinical: Tensor | None = None,
    ) -> Tensor:
        """Return one log-hazard score per node."""
        if self.fusion in ("pet", "ct", "feature"):
            if self.fusion == "pet":
                x = x_pet
            elif self.fusion == "ct":
                x = x_ct
            else:
                x = torch.cat([x_pet, x_ct], dim=1)

            h, w = self.gnn1(x, edge_index, edge_weight)
            h = F.dropout(F.relu(h), p=self.dropout, training=self.training)
            h, _ = self.gnn2(h, edge_index, w)
            h = F.relu(h)
        else:
            h_pet, w_pet = self.gnn_pet(x_pet, edge_index, edge_weight)
            h_ct, w_ct = self.gnn_ct(x_ct, edge_index, edge_weight)
            h_pet = F.dropout(F.relu(h_pet), p=self.dropout, training=self.training)
            h_ct = F.dropout(F.relu(h_ct), p=self.dropout, training=self.training)

            h = (
                torch.cat([h_pet, h_ct], dim=1)
                if self.feature_fusion_op == "concat"
                else (h_pet + h_ct) / 2.0
            )
            w = (w_pet + w_ct) / 2.0 if self.edge_fusion_op == "mean" else torch.maximum(w_pet, w_ct)

            h, _ = self.gnn2(h, edge_index, w)
            h = F.relu(h)

        return self.head(h, clinical)

    # -- convenience ---------------------------------------------------------
    @torch.no_grad()
    def risk_scores(self, *args, **kwargs) -> Tensor:
        self.eval()
        return self.forward(*args, **kwargs)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
