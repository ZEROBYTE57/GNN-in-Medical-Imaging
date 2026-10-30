"""SiteSwap: train so that the explanation does not depend on the site.

For every training subject, ComBat (fitted on the training data, label protected) makes a
site-swapped copy: the same brain as if scanned on the other scanner. Training uses
  loss = CE(original) + CE(swapped copy)                      (augmentation)
       + lam * mean((S(x) - S(x_swap))^2)                    (explanation consistency)
where S is the saliency map (|gradient of the patient margin|), each map scaled to mean 1.
lam = 0 gives the augmentation-only version (ablation / simplest fallback).
The epoch is chosen on validation loss of the original validation data, as for all models.
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F


def _saliency_graph(model, x):
    x = x.detach().requires_grad_(True)
    logits = model(x)
    margin = logits[:, 1] - logits[:, 0]
    (g,) = torch.autograd.grad(margin.sum(), x, create_graph=True)
    s = g.abs()
    return logits, s / s.mean(dim=(1, 2), keepdim=True).clamp(min=1e-8)


def train_siteswap(model, fc_tr, fc_tr_swap, y_tr, fc_va, y_va, lam=1.0, epochs=150, lr=1e-3,
                   weight_decay=1e-3, batch_size=32, seed=0):
    """Returns (model with best weights, best epoch, validation-loss curve)."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_epoch, best_state, curve = float("inf"), 0, None, []
    n = len(fc_tr)
    for epoch in range(1, epochs + 1):
        model.train()
        for idx in torch.randperm(n, generator=g).split(batch_size):
            idx = idx.to(fc_tr.device)
            x, xs, y = fc_tr[idx], fc_tr_swap[idx], y_tr[idx]
            loss = F.cross_entropy(model(x), y) + F.cross_entropy(model(xs), y)
            if lam > 0:
                # Saliency maps compared without dropout, so the penalty measures the site
                # swap, not random dropout noise.
                model.eval()
                _, so = _saliency_graph(model, x)
                _, ss = _saliency_graph(model, xs)
                model.train()
                loss = loss + lam * ((so - ss) ** 2).mean()
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
