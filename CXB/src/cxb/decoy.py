"""Scanner decoy: a fake, label-linked scanner effect at a known place (gate 2).

Design (docs/decisions.md, 30 Oct 2026):
- Place: the m regions outside the planted disease set where real ABIDE sites differ most
  (between-site share of variance, eta squared, among healthy controls; no labels used).
- Size: the typical real difference between two hospitals on those connections (median
  over site pairs of the mean |difference of site means|), in Fisher z.
- Mechanism: as for the disease signal, one shared signal is added to the decoy regions'
  time series ("scanner B") and connectivity is recomputed, so every matrix stays a
  valid correlation matrix. Like the disease effect, it only raises connectivity.
- Every pool subject gets two versions: without the decoy (scanner A, the planted data)
  and with it (scanner B). Which subjects are "scanner B" is decided per fold, so the
  shortcut can be linked to the label in training and removed in test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .data import connectivity
from .planting import calibrate_amplitude, plant_timeseries, region_masks


def site_effect_eta2(fc, site, idx, min_n=20):
    """Between-site share of variance per edge (eta squared), using subjects `idx` from
    sites with at least min_n of them. Returns (eta2 (R, R), sites used, site means)."""
    fc, site = np.asarray(fc, dtype=np.float64)[idx], np.asarray(site)[idx]
    sites = [s for s in np.unique(site) if (site == s).sum() >= min_n]
    keep = np.isin(site, sites)
    X, s = fc[keep], site[keep]
    grand = X.mean(0)
    means = np.stack([X[s == u].mean(0) for u in sites])
    ns = np.array([(s == u).sum() for u in sites])
    between = (ns[:, None, None] * (means - grand) ** 2).sum(0)
    total = ((X - grand) ** 2).sum(0)
    eta = np.where(total > 0, between / np.where(total > 0, total, 1.0), 0.0)
    np.fill_diagonal(eta, 0.0)
    return eta, sites, means


def select_decoy_regions(eta, exclude, m):
    """The m regions (not in `exclude`) with the largest mean eta squared over their edges."""
    score = eta.sum(1) / (eta.shape[0] - 1)
    score[np.asarray(exclude)] = -np.inf
    return np.argsort(-score, kind="stable")[:m]


def typical_site_difference(means, inside):
    """Median over site pairs of the mean |difference of site means| on `inside` edges."""
    return float(np.median([np.abs(means[a] - means[b])[inside].mean()
                            for a, b in combinations(range(len(means)), 2)]))


def assign_scanner(y, site, frac, rng):
    """Boolean 'scanner B' flags: within each site, round(frac x patients) and
    round((1 - frac) x controls) get scanner B. frac = 0.5 means no link to the label."""
    y, site = np.asarray(y), np.asarray(site)
    b = np.zeros(len(y), dtype=bool)
    for s in np.unique(site):
        for label, share in ((1, frac), (0, 1.0 - frac)):
            idx = np.where((site == s) & (y == label))[0]
            n = int(round(share * len(idx)))
            b[rng.choice(idx, size=n, replace=False)] = True
    return b


@dataclass
class Decoy:
    fc_b: np.ndarray        # (n_pool, R, R) planted data plus the scanner-B fingerprint
    regions: np.ndarray     # decoy regions
    inside: np.ndarray      # (R, R) decoy edges
    incident: np.ndarray    # (R, R) edges from decoy regions to the rest
    meta: dict = field(default_factory=dict)


def build_decoy(p, load_ts, fc_all, dx_all, site_all, m=10, seed=0, control_label=2,
                min_site_n=20, pipeline_kwargs=None):
    """Scanner-B version of every subject in planted dataset `p` (built with keep_ts=True)."""
    ctrl = np.where(np.asarray(dx_all) == control_label)[0]
    eta, sites, means = site_effect_eta2(fc_all, site_all, ctrl, min_site_n)
    regions = select_decoy_regions(eta, p.regions, m)
    inside, incident = region_masks(eta.shape[0], regions)
    target = typical_site_difference(means, inside)
    r0 = float(np.tanh(p.fc[:, inside].mean()))
    amplitude = calibrate_amplitude(r0, target)
    rng = np.random.default_rng([seed, 7])
    fc_b = np.empty_like(p.fc)
    for row, i in enumerate(p.pool_idx):
        X = p.ts[row] if row in p.ts else load_ts(i)
        Z, _ = plant_timeseries(X, regions, amplitude, rng)
        fc_b[row], _ = connectivity(Z, **(pipeline_kwargs or {}))
    meta = {"m": m, "target_shift": target, "amplitude": amplitude, "sites_used": sites,
            "mean_eta2_inside": float(eta[inside].mean()),
            "achieved_shift": float((fc_b - p.fc)[:, inside].mean())}
    return Decoy(fc_b, regions, inside, incident, meta)


def top_share(scores, mask, k):
    """Share of the k highest-scoring upper-triangle edges that fall inside `mask`."""
    iu = np.triu_indices(mask.shape[0], 1)
    top = np.argsort(-np.asarray(scores), kind="stable")[:k]
    return float(mask[iu][top].mean())
