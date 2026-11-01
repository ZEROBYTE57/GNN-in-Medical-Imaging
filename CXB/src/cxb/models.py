"""Models for the benchmark: an MLP on edge values, a dense GCN and GAT, and three published
brain-network models reimplemented from their papers (BrainNetCNN, BrainGNN, BNT).

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
                batch_size=32, seed=0, min_epochs=30):
    """Train with Adam; keep the epoch with the lowest validation loss (never test data),
    looking only from epoch `min_epochs` on (warm-up: the loss minimum often falls in the
    first few epochs, before the model has learned anything; see docs/decisions.md).

    Returns (model with best weights, best epoch, list of validation losses)."""
    g = torch.Generator().manual_seed(seed)
    lr = getattr(model, "train_lr", lr)            # tuning variants set their own rate
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_epoch, best_state, curve = float("inf"), 0, None, []
    n = len(fc_tr)
    for epoch in range(1, epochs + 1):
        model.train()
        for idx in torch.randperm(n, generator=g).split(batch_size):
            idx = idx.to(fc_tr.device)
            loss = F.cross_entropy(model(fc_tr[idx]), y_tr[idx]) + getattr(model, "aux", 0.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = F.cross_entropy(model(fc_va), y_va).item()
        curve.append(vl)
        if epoch >= min(min_epochs, epochs) and vl < best:
            best, best_epoch, best_state = vl, epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, curve


# ---------------------------------------------------------------------------------------
# Published brain-network models, reimplemented from their papers (1 Nov 2026).
# Same interface as MLP/GCN: raw Fisher-z matrix (batch, R, R) in, 2 logits out,
# standardisation inside the model. Differences from the papers are listed per class.
# ---------------------------------------------------------------------------------------


def _topk_graph(fc, eye, density):
    """Per-subject graph: strongest `density` share of |r| (chosen without gradient),
    weighted by the positive connectivity, with self-loops removed. Shared by BrainGNN."""
    with torch.no_grad():
        a = fc.abs() * (1 - eye)
        k = int(round(density * a.shape[1] * (a.shape[1] - 1)))
        thr = a.flatten(1).topk(k, dim=1).values[:, -1]
        keep = (a >= thr[:, None, None]).float()
        keep = torch.maximum(keep, keep.transpose(1, 2))
    return keep * F.relu(fc) * (1 - eye)


class BrainNetCNN(nn.Module):
    """BrainNetCNN (Kawahara et al., NeuroImage 2017), E2E-E2E-E2N-N2G architecture.

    Edge-to-edge filters combine each edge with all edges sharing its row or column;
    then edge-to-node, node-to-graph, and two fully connected layers; leaky ReLU (0.33)
    and dropout 0.5 as in the paper. Channels as in the paper's E2E model (32, 32, 64,
    256, 128, 30). Differences: 2-class output with cross-entropy (the paper regressed
    neurodevelopmental scores); our shared optimiser and epoch rule."""

    def __init__(self, n_rois, mean, std, e2e=32, e2n=64, n2g=256, dropout=0.5):
        super().__init__()
        self.norm = _Standardise(mean, std)
        R = n_rois
        self.e2e1_r, self.e2e1_c = nn.Conv2d(1, e2e, (1, R)), nn.Conv2d(1, e2e, (R, 1))
        self.e2e2_r, self.e2e2_c = nn.Conv2d(e2e, e2e, (1, R)), nn.Conv2d(e2e, e2e, (R, 1))
        self.e2n = nn.Conv2d(e2e, e2n, (1, R))
        self.n2g = nn.Conv2d(e2n, n2g, (R, 1))
        self.fc1, self.fc2, self.out = nn.Linear(n2g, 128), nn.Linear(128, 30), nn.Linear(30, 2)
        self.dropout = dropout

    @staticmethod
    def _e2e(x, conv_r, conv_c):
        return conv_r(x) + conv_c(x)            # (B,C,R,1) + (B,C,1,R) -> (B,C,R,R)

    def forward(self, fc):
        act = lambda t: F.leaky_relu(t, 0.33)   # noqa: E731
        x = self.norm(fc)[:, None]
        x = act(self._e2e(x, self.e2e1_r, self.e2e1_c))
        x = act(self._e2e(x, self.e2e2_r, self.e2e2_c))
        x = act(self.e2n(x))                     # (B,64,R,1)
        x = F.dropout(act(self.n2g(x)), self.dropout, self.training).flatten(1)
        x = F.dropout(act(self.fc1(x)), self.dropout, self.training)
        x = F.dropout(act(self.fc2(x)), self.dropout, self.training)
        return self.out(x)


class _RaGConv(nn.Module):
    """ROI-aware graph convolution (BrainGNN): each region i has its own weight matrix
    W_i = sum_k relu(alpha_ik) B_k, built from K shared bases ("communities")."""

    def __init__(self, n_rois, d_in, d_out, k=8):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(n_rois, k) * 0.1 + 1.0 / k)
        self.basis = nn.Parameter(torch.randn(k, d_in, d_out) / np.sqrt(d_in))
        self.bias = nn.Parameter(torch.zeros(d_out))

    def forward(self, h, a, idx):
        """h (B,n,d_in), a (B,n,n) weighted adjacency, idx (B,n) original ROI ids."""
        w = torch.einsum("rk,kio->rio", F.relu(self.alpha), self.basis)[idx]   # (B,n,d_in,d_out)
        m = h + a @ h                                    # own features + weighted neighbours
        return torch.einsum("bni,bnio->bno", m, w) + self.bias


class BrainGNN(nn.Module):
    """BrainGNN (Li et al., Medical Image Analysis 2021): two ROI-aware graph convolution
    layers, each followed by ROI-selection (top-k) pooling, mean||max readout after each
    block, then an MLP. Node features = connectivity rows; graph = strongest 10% of
    positive connections. Training adds the paper's unit loss (pooling vector norm 1) and
    top-k separation loss (weight 0.1 each) through `aux`. Differences: Pearson instead of
    partial correlation for edges; the group-level consistency loss is left out; our shared
    optimiser and epoch rule. Built-in explanation: `node_scores` (first pooling scores)."""

    def __init__(self, n_rois, mean, std, hidden=32, k=8, ratio=0.5, density=0.1, dropout=0.5,
                 aux_weight=0.1):
        super().__init__()
        self.norm = _Standardise(mean, std)
        self.conv1, self.conv2 = _RaGConv(n_rois, n_rois, hidden, k), _RaGConv(n_rois, hidden, hidden, k)
        self.p1, self.p2 = nn.Parameter(torch.randn(hidden)), nn.Parameter(torch.randn(hidden))
        self.mlp = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 2))
        self.ratio, self.density, self.dropout, self.aux_weight = ratio, density, dropout, aux_weight
        self.register_buffer("eye", torch.eye(n_rois))
        self.aux, self.node_scores = 0.0, None

    def _pool(self, h, a, idx, p):
        s = (h @ p) / p.norm()                           # (B,n) pooling scores
        n_keep = max(1, int(round(self.ratio * h.shape[1])))
        top = s.topk(n_keep, dim=1).indices
        rest = torch.ones_like(s, dtype=torch.bool).scatter(1, top, False)
        sig = torch.sigmoid(s)
        eps = 1e-7
        tpk = -(torch.log(sig.gather(1, top) + eps).mean()
                + torch.log(1 - sig.masked_select(rest) + eps).mean())
        gate = sig.gather(1, top)[..., None]
        h2 = h.gather(1, top[..., None].expand(-1, -1, h.shape[2])) * gate
        a2 = a.gather(1, top[..., None].expand(-1, -1, a.shape[2])).gather(2, top[:, None, :].expand(-1, n_keep, -1))
        return h2, a2, idx.gather(1, top), s, tpk + (p.norm() - 1) ** 2

    def forward(self, fc):
        B, R = fc.shape[0], fc.shape[1]
        a = _topk_graph(fc, self.eye, self.density)
        a = a / a.sum(-1, keepdim=True).clamp(min=1e-6)  # row-normalised weights
        h = F.dropout(self.norm(fc), self.dropout, self.training)
        idx = torch.arange(R, device=fc.device).expand(B, R)
        h = F.relu(self.conv1(h, a, idx))
        h, a, idx, s1, l1 = self._pool(h, a, idx, self.p1)
        z1 = torch.cat([h.mean(1), h.amax(1)], 1)
        h = F.relu(self.conv2(h, a, idx))
        h, a, idx, _, l2 = self._pool(h, a, idx, self.p2)
        z2 = torch.cat([h.mean(1), h.amax(1)], 1)
        self.node_scores = s1.detach()
        self.aux = self.aux_weight * (l1 + l2) if self.training else 0.0
        return self.mlp(torch.cat([z1, z2], 1))


class BNT(nn.Module):
    """Brain Network Transformer (Kan et al., NeurIPS 2022): node features = connectivity
    rows; multi-head self-attention encoder over regions; orthonormal clustering readout
    (soft assignment of regions to K learnable, orthonormally initialised cluster centres);
    flatten; MLP. Differences: 2 encoder layers of width R with 4 heads and K = 10 clusters
    (sized for 116 AAL regions); our shared optimiser and epoch rule. Explained with the
    same post-hoc explainers as every other model."""

    def __init__(self, n_rois, mean, std, heads=4, layers=2, ff=512, clusters=10, hidden=256, dropout=0.5):
        super().__init__()
        self.norm = _Standardise(mean, std)
        d = n_rois
        if d % heads:
            heads = 1
        self.enc = nn.ModuleList([nn.TransformerEncoderLayer(d, heads, ff, dropout=0.1, batch_first=True)
                                  for _ in range(layers)])
        centres = torch.linalg.qr(torch.randn(d, clusters))[0].T          # orthonormal rows
        self.centres = nn.Parameter(centres.contiguous())
        self.head = nn.Sequential(nn.Linear(clusters * d, hidden), nn.LeakyReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 32), nn.LeakyReLU(), nn.Linear(32, 2))

    def forward(self, fc):
        z = self.norm(fc)
        for layer in self.enc:
            z = layer(z)
        p = torch.softmax(z @ self.centres.T, dim=-1)    # (B,R,K) soft assignment
        zc = p.transpose(1, 2) @ z                       # (B,K,R) cluster embeddings
        return self.head(zc.flatten(1))



class GAT(nn.Module):
    """Dense two-layer graph attention network (Velickovic et al., 2018) in the BrainGB
    set-up (Cui et al., 2022): same node features, graph, root weights and concat readout
    as the GCN above; neighbour weights are learned attention (one head per layer) over
    the kept edges plus self-loops, instead of fixed normalised connectivity."""

    def __init__(self, n_rois, mean, std, hidden=16, density=0.2, dropout=0.5):
        super().__init__()
        self.norm = _Standardise(mean, std)
        self.density, self.dropout = density, dropout
        self.lin1, self.lin2 = nn.Linear(n_rois, hidden), nn.Linear(hidden, hidden)
        self.root1 = nn.Linear(n_rois, hidden, bias=False)
        self.root2 = nn.Linear(hidden, hidden, bias=False)
        self.att1, self.att2 = nn.Linear(hidden, 2, bias=False), nn.Linear(hidden, 2, bias=False)
        self.out = nn.Linear(n_rois * hidden, 2)
        self.register_buffer("eye", torch.eye(n_rois))

    def mask(self, fc):
        """Same kept edges as the GCN (strongest `density` share of |r|), plus self-loops."""
        with torch.no_grad():
            a = fc.abs() * (1 - self.eye)
            k = int(round(self.density * a.shape[1] * (a.shape[1] - 1)))
            thr = a.flatten(1).topk(k, dim=1).values[:, -1]
            keep = (a >= thr[:, None, None]).float()
            keep = torch.maximum(torch.maximum(keep, keep.transpose(1, 2)), self.eye.expand_as(keep))
            return keep > 0

    def _layer(self, h, lin, root, att, mask):
        wh = lin(h)                                                   # (B, R, H)
        st = att(wh)                                                  # source and target scores
        e = F.leaky_relu(st[..., :1] + st[..., 1:].transpose(1, 2), 0.2)
        a = torch.softmax(e.masked_fill(~mask, float("-inf")), dim=-1)
        a = F.dropout(a, self.dropout, self.training)
        return F.relu(a @ wh + root(h))

    def forward(self, fc):
        mask = self.mask(fc)
        h = F.dropout(self.norm(fc), self.dropout, self.training)
        h = self._layer(h, self.lin1, self.root1, self.att1, mask)
        h = F.dropout(h, self.dropout, self.training)
        h = self._layer(h, self.lin2, self.root2, self.att2, mask)
        return self.out(F.dropout(h.flatten(1), self.dropout, self.training))


MODELS.update({"brainnetcnn": BrainNetCNN, "braingnn": BrainGNN, "bnt": BNT, "gat": GAT})

# Tuning variants (decisions.md, 1 Nov): "name-half" halves the width, "name-lr0.0003" sets the
# learning rate (decimal), e.g. "bnt-half-lr0.0003". The rate is read by the training loops.
_HALF = {"braingnn": {"hidden": 16}, "bnt": {"ff": 256, "hidden": 128},
         "brainnetcnn": {"e2e": 16, "e2n": 32, "n2g": 128}, "gcn": {"hidden": 8}, "gat": {"hidden": 8}, "mlp": {"hidden": 32}}


class _ModelRegistry(dict):
    def __missing__(self, key):
        base, *opts = key.split("-")
        if base not in self or not opts:
            raise KeyError(key)
        kwargs, lr = {}, None
        for o in opts:
            if o == "half":
                kwargs.update(_HALF[base])
            elif o.startswith("lr"):
                lr = float(o[2:])
            else:
                raise KeyError(key)
        cls = self[base]

        def make(*a, **k):
            m = cls(*a, **{**kwargs, **k})
            if lr is not None:
                m.train_lr = lr
            return m
        return make


MODELS = _ModelRegistry(MODELS)
