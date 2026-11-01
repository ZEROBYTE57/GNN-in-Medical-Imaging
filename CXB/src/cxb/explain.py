"""Gradient explainers and scoring against planted ground truth.

All explainers explain the "patient" margin (logit 1 - logit 0) and return the absolute
attribution per matrix entry (batch, R, R). Edge (i, j) gets the sum of entries (i, j)
and (j, i), since a connectivity matrix stores each edge twice.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score


def _margin(model, fc):
    logits = model(fc)
    return logits[:, 1] - logits[:, 0]


def saliency(model, fc):
    x = fc.detach().clone().requires_grad_(True)
    (g,) = torch.autograd.grad(_margin(model, x).sum(), x)
    return g.abs()


def integrated_gradients(model, fc, steps=32, baseline=None):
    """IG from a baseline (default: all-zero connectivity) along a straight path,
    using the midpoint rule."""
    x = fc.detach()
    b = torch.zeros_like(x) if baseline is None else baseline.expand_as(x)
    total = torch.zeros_like(x)
    for a in (torch.arange(steps, dtype=x.dtype) + 0.5) / steps:
        z = (b + a * (x - b)).requires_grad_(True)
        (g,) = torch.autograd.grad(_margin(model, z).sum(), z)
        total += g
    return ((x - b) * total / steps).abs()


def mask_explainer(model, fc, steps=150, lr=0.05, size_coef=0.005, ent_coef=0.1, seed=0):
    """GNNExplainer-style soft mask on the connectivity entries (Ying et al., 2019),
    model-agnostic so it works for the MLP and the GCN alike.

    Per subject, a symmetric mask m in (0, 1) is learned so that the masked input
    fc * m keeps the model's predicted class, while a size penalty keeps the mask small
    and an entropy penalty pushes it towards 0/1. Returns the mask (batch, R, R)."""
    x = fc.detach()
    with torch.no_grad():
        target = model(x).argmax(1)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    logits = (0.1 * torch.randn(x.shape, generator=gen)).to(x.device).requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=lr)
    off = 1 - torch.eye(x.shape[-1], device=x.device)
    for _ in range(steps):
        m = torch.sigmoid((logits + logits.transpose(-1, -2)) / 2) * off
        out = model(x * m)
        ent = -(m * torch.log(m + 1e-8) + (1 - m) * torch.log(1 - m + 1e-8))
        loss = torch.nn.functional.cross_entropy(out, target) + size_coef * m.sum((1, 2)).mean() / x.shape[-1] \
            + ent_coef * ent.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return torch.sigmoid((logits + logits.transpose(-1, -2)) / 2) * off


EXPLAINERS = {"saliency": saliency, "ig": integrated_gradients}
ALL_EXPLAINERS = {**EXPLAINERS, "gnnexplainer": mask_explainer}


def edge_scores(attr):
    """(batch, R, R) attributions -> (batch, n_edges) upper-triangle edge scores."""
    a = attr + attr.transpose(-1, -2)
    iu = np.triu_indices(a.shape[-1], 1)
    return a[..., iu[0], iu[1]]


def auprc(scores, inside, incident):
    """AUPRC of edge scores against the planted truth.

    inside, incident: (R, R) boolean masks. Incident edges are left out ("don't care")."""
    iu = np.triu_indices(inside.shape[0], 1)
    pos, dc = inside[iu], incident[iu]
    return float(average_precision_score(pos[~dc], np.asarray(scores)[~dc]))


def t_scores(fc, y):
    """|Welch t| per upper-triangle edge between label groups: a univariate reference for
    recovery (not an upper bound: it uses training subjects only, so a model can exceed it)."""
    fc = np.asarray(fc, dtype=np.float64)
    a, b = fc[y == 1], fc[y == 0]
    se = np.sqrt(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.abs(np.where(se > 0, (a.mean(0) - b.mean(0)) / se, 0.0))
    iu = np.triu_indices(fc.shape[1], 1)
    return t[iu]
