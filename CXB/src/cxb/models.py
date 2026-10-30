"""Models for the benchmark: an MLP on edge values and a dense GCN.

Both take the raw Fisher-z matrix (batch, R, R) as their only input, so every explainer
returns one attribution per matrix entry, i.e. per edge. Standardisation (training-set
mean and SD per entry) happens inside the model, so gradients are with respect to the
raw connectivity values.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class _Standardise(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(np.where(std > 1e-6, std, 1.0), dtype=torch.float32))

    def forward(self, fc):
        return (fc - self.mean) / self.std


class MLP(nn.Module):
    """Upper-triangle edge values -> hidden layer -> 2 logits. Ignores the graph."""

    def __init__(self, n_rois, mean, std, hidden=64, dropout=0.5):
        super().__init__()
        self.norm = _Standardise(mean, std)
        iu = torch.triu_indices(n_rois, n_rois, 1)
        self.register_buffer("iu0", iu[0])
        self.register_buffer("iu1", iu[1])
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(iu.shape[1], hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, fc):
        return self.net(self.norm(fc)[:, self.iu0, self.iu1])


class GCN(nn.Module):
    """Dense two-layer GCN as commonly used for brain networks.

    Node features: each region's row of the standardised connectivity matrix.
    Graph: the strongest `density` share of |connections| per subject, weighted by the
    positive connectivity value, with self-loops and symmetric normalisation. The choice
    of kept edges is made without gradient; the weights carry gradient.
    Readout: "concat" joins all region embeddings in a fixed order, so the model knows
    which region is which (as in BrainGB); "meanmax" pools over regions (mean and max).

    Defaults (root=True, readout="concat", hidden=16) were chosen on validation loss in
    gate 1 (30 Oct 2026): the plain variant (no root weight, mean/max readout) reached
    only 0.68 validation accuracy on the x4 planted effect; with both changes, 1.00."""

    def __init__(self, n_rois, mean, std, hidden=16, density=0.2, dropout=0.5, readout="concat",
                 root=True):
        super().__init__()
        # root=True: each region also gets its own features through a separate weight
        # (GraphConv-style), so neighbour averaging cannot wash out a region's own signal.
        self.root1 = nn.Linear(n_rois, hidden, bias=False) if root else None
        self.root2 = nn.Linear(hidden, hidden, bias=False) if root else None
        self.norm = _Standardise(mean, std)
        self.density = density
        self.readout = readout
        self.lin1 = nn.Linear(n_rois, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(n_rois * hidden if readout == "concat" else 2 * hidden, 2)
        self.dropout = dropout
        self.register_buffer("eye", torch.eye(n_rois))

    def adjacency(self, fc):
        with torch.no_grad():
            a = fc.abs() * (1 - self.eye)
            k = int(round(self.density * a.shape[1] * (a.shape[1] - 1)))
            thr = a.flatten(1).topk(k, dim=1).values[:, -1]
            keep = (a >= thr[:, None, None]).float()
            keep = torch.maximum(keep, keep.transpose(1, 2))
        w = keep * F.relu(fc) + self.eye
        d = w.sum(-1).clamp(min=1e-6).rsqrt()
        return d[:, :, None] * w * d[:, None, :]

    def forward(self, fc):
        A = self.adjacency(fc)
        h = F.dropout(self.norm(fc), self.dropout, self.training)
        h = F.relu(A @ self.lin1(h) + (self.root1(h) if self.root1 is not None else 0))
        h = F.dropout(h, self.dropout, self.training)
        h = F.relu(A @ self.lin2(h) + (self.root2(h) if self.root2 is not None else 0))
        if self.readout == "concat":
            return self.out(F.dropout(h.flatten(1), self.dropout, self.training))
        return self.out(torch.cat([h.mean(1), h.amax(1)], dim=1))


MODELS = {"mlp": MLP, "gcn": GCN}


def train_model(model, fc_tr, y_tr, fc_va, y_va, epochs=150, lr=1e-3, weight_decay=1e-3,
                batch_size=32, seed=0):
    """Train with Adam; keep the epoch with the lowest validation loss (never test data).

    Returns (model with best weights, best epoch, list of validation losses)."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_epoch, best_state, curve = float("inf"), 0, None, []
    n = len(fc_tr)
    for epoch in range(1, epochs + 1):
        model.train()
        for idx in torch.randperm(n, generator=g).split(batch_size):
            idx = idx.to(fc_tr.device)
            loss = F.cross_entropy(model(fc_tr[idx]), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = F.cross_entropy(model(fc_va), y_va).item()
        curve.append(vl)
        if vl < best:
            best, best_epoch, best_state = vl, epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, curve
