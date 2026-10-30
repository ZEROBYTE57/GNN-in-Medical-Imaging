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


EXPLAINERS = {"saliency": saliency, "ig": integrated_gradients}


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
    """|Welch t| per upper-triangle edge between label groups (the recovery ceiling)."""
    fc = np.asarray(fc, dtype=np.float64)
    a, b = fc[y == 1], fc[y == 0]
    se = np.sqrt(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.abs(np.where(se > 0, (a.mean(0) - b.mean(0)) / se, 0.0))
    iu = np.triu_indices(fc.shape[1], 1)
    return t[iu]
