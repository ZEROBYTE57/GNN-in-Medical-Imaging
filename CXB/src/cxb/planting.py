"""Planted ground truth: add known connectivity effects to real control connectomes.

Shared design (docs/decisions.md, 29 Oct 2026):
- A fixed, site-stratified 20% of controls is set aside only to rank edges and regions;
  it never enters the planting pool, so no subject is used twice.
- Effects are ranked by |Cohen's d| between all autism subjects and the ranking controls.
- The planting pool is the remaining controls without zero-signal regions. Each split
  gives half of every site the fake label 1 ("patient"); balancing within site keeps the
  label from being predictable from site.

Main design: region-level planting in the time series (build_regions)
- The m regions most involved in real autism differences (mean |d| over their edges) form
  the planted set. One shared signal, orthogonal to the subject's ROI time series, is added
  to every region in the set; connectivity is then recomputed with the same pipeline, so
  every planted matrix is a valid correlation matrix.
- Truth = all edges inside the set ("inside"). Edges from the set to other regions are
  diluted as a side effect ("incident"; scored as don't-care, also reported). All other
  edges are unchanged up to rounding (when the signal can be fully orthogonalised).
- Amplitude is calibrated so the mean inside shift equals level x median(|d| x SD) over
  the inside edges.

Control design: exact edge-level planting (build): top-k edges shifted by level x d x SD
in Fisher-z units; exact ground truth but the matrices are no longer valid correlation
matrices (measured: 100% of planted subjects; docs/decisions.md).

All functions are NumPy so they can be tested on synthetic data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import connectivity

LEVELS = {"x0.5": 0.5, "x1": 1.0, "x2": 2.0, "x4": 4.0}   # x4 only for the gate-1 check
REGION_COUNTS = (7, 10, 20)                                # 21, 45, 190 inside edges
MULTIPLIERS = {"weak": 1.0, "moderate": 2.0, "strong": 4.0}  # edge-level control (old levels)
EDGE_COUNTS = (20, 50, 200)


def ranking_split(dx, site, n_dead_rois, control_label=2, frac=0.2, seed=0):
    """Split controls into a ranking set (site-stratified `frac`) and a planting pool.

    Ranking controls are drawn from all controls; the pool is the remaining controls
    with no zero-signal region. Returns (rank_idx, pool_idx), sorted index arrays."""
    rng = np.random.default_rng(seed)
    dx, site, dead = np.asarray(dx), np.asarray(site), np.asarray(n_dead_rois)
    rank = []
    for s in np.unique(site):
        ctrl = np.where((dx == control_label) & (site == s))[0]
        n = int(round(frac * len(ctrl)))
        rank.extend(rng.choice(ctrl, size=n, replace=False).tolist())
    rank = np.sort(np.array(rank, dtype=int))
    pool = np.where((dx == control_label) & (dead == 0))[0]
    pool = np.setdiff1d(pool, rank)
    return rank, pool


def edge_effect_sizes(fc, idx_a, idx_b):
    """Signed Cohen's d (mean_a - mean_b) / pooled SD, and the pooled SD, per edge.

    fc: (n, R, R) Fisher-z matrices. Edges with zero pooled SD get d = 0."""
    a, b = fc[idx_a].astype(np.float64), fc[idx_b].astype(np.float64)
    na, nb = len(a), len(b)
    sd = np.sqrt(((na - 1) * a.var(0, ddof=1) + (nb - 1) * b.var(0, ddof=1)) / (na + nb - 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(sd > 0, (a.mean(0) - b.mean(0)) / sd, 0.0)
    np.fill_diagonal(d, 0.0)
    return d, sd


def top_edges(d, k):
    """Upper-triangle (i, j) of the k largest |d|, strongest first."""
    iu = np.triu_indices(d.shape[0], 1)
    order = np.argsort(-np.abs(d[iu]), kind="stable")[:k]
    return iu[0][order], iu[1][order]


def fake_labels(site, seed):
    """Fake labels for the planting pool: half of each site gets 1 (odd sites: the extra
    subject's label is random). Returns int array of 0/1."""
    rng = np.random.default_rng(seed)
    site = np.asarray(site)
    y = np.zeros(len(site), dtype=int)
    for s in np.unique(site):
        idx = rng.permutation(np.where(site == s)[0])
        n1 = len(idx) // 2 + (rng.integers(2) if len(idx) % 2 else 0)
        y[idx[:n1]] = 1
    return y


def plant(fc, y, edges, d, sd, multiplier):
    """Copy of fc with label-1 subjects shifted by multiplier * d * sd on `edges`."""
    out = fc.copy()
    i, j = edges
    shift = (multiplier * d[i, j] * sd[i, j]).astype(fc.dtype)
    p = np.where(np.asarray(y) == 1)[0][:, None]
    out[p, i, j] += shift
    out[p, j, i] += shift
    return out


def truth_mask(n_rois, edges):
    """Boolean (R, R) ground-truth mask of planted edges, symmetric."""
    m = np.zeros((n_rois, n_rois), dtype=bool)
    i, j = edges
    m[i, j] = m[j, i] = True
    return m


def invalid_fraction(fc, tol=1e-5):
    """Share of matrices that are not valid correlation matrices: tanh(z) with unit
    diagonal must be positive semidefinite (min eigenvalue >= -tol).

    tol = 1e-5: real ABIDE matrices reach about -2e-7 from float32 rounding (many are
    singular, with fewer timepoints than regions); edge-level planting reaches -0.18 to
    -1.0. Measured 29 Oct 2026."""
    r = np.tanh(fc.astype(np.float64))
    idx = np.arange(r.shape[1])
    r[:, idx, idx] = 1.0
    return float(np.mean(np.linalg.eigvalsh(r)[:, 0] < -tol))


@dataclass
class Planted:
    fc: np.ndarray          # (n_pool, R, R) planted Fisher-z matrices
    y: np.ndarray           # (n_pool,) fake labels, 1 = planted
    site: np.ndarray        # (n_pool,) site of each pool subject
    pool_idx: np.ndarray    # indices into the full dataset
    edges: tuple            # (i, j) planted edges, upper triangle
    truth: np.ndarray       # (R, R) bool ground-truth mask
    meta: dict = field(default_factory=dict)


def build(fc, dx, site, n_dead_rois, k, level, split_seed, rank_seed=0, asd_label=1, control_label=2):
    """One planted dataset: edge count k, effect level ('weak'|'moderate'|'strong')."""
    dx, site = np.asarray(dx), np.asarray(site)
    rank, pool = ranking_split(dx, site, n_dead_rois, control_label, seed=rank_seed)
    d, sd = edge_effect_sizes(fc, np.where(dx == asd_label)[0], rank)
    edges = top_edges(d, k)
    y = fake_labels(site[pool], seed=split_seed)
    mult = LEVELS[level] if level in LEVELS else MULTIPLIERS[level]
    planted = plant(fc[pool], y, edges, d, sd, mult)
    meta = {"k": k, "level": level, "multiplier": mult, "split_seed": split_seed,
            "rank_seed": rank_seed, "n_rank": len(rank), "n_pool": len(pool),
            "d_planted": d[edges].tolist()}
    return Planted(planted, y, site[pool], pool, edges, truth_mask(fc.shape[1], edges), meta)


# ----------------------------------------------------------------------------- region level (main)

def region_scores(d):
    """Mean |d| over each region's edges (diagonal excluded)."""
    a = np.abs(d).astype(np.float64)
    np.fill_diagonal(a, 0.0)
    return a.sum(1) / (a.shape[0] - 1)


def select_regions(d, m):
    """The m regions with the largest mean |d|, strongest first."""
    return np.argsort(-region_scores(d), kind="stable")[:m]


def region_masks(n_rois, regions):
    """(inside, incident) boolean (R, R) masks: edges within the set, and edges from the
    set to any other region. Diagonal is False in both."""
    s = np.zeros(n_rois, dtype=bool)
    s[np.asarray(regions)] = True
    inside = np.outer(s, s)
    incident = np.logical_xor.outer(s, s)
    np.fill_diagonal(inside, False)
    return inside, incident


def standardise(X):
    X = np.asarray(X, dtype=np.float64)
    sd = X.std(0)
    return (X - X.mean(0)) / np.where(sd > 0, sd, 1.0)


def shared_signal(X, regions, rng, min_residual=1e-3):
    """Unit-variance white signal, orthogonal to all ROI time series in X when possible.

    With fewer timepoints than regions the full orthogonal complement can be (almost)
    empty; then the signal is made orthogonal to the planted regions only, so the inside
    shift stays calibrated and other edges get only small, label-independent noise.
    Returns (signal, fully_orthogonal)."""
    s = rng.standard_normal(X.shape[0])
    s = s - s.mean()
    for cols, full in ((X, True), (X[:, regions], False)):
        b, *_ = np.linalg.lstsq(cols, s, rcond=None)
        res = s - cols @ b
        res = res - res.mean()
        if res.std() > min_residual * s.std():
            return res / res.std(), full
    raise ValueError("cannot build a signal orthogonal to the planted regions")


def plant_timeseries(X, regions, amplitude, rng):
    """Standardised copy of X with amplitude x shared signal added to `regions`."""
    Z = standardise(X)
    s, full = shared_signal(Z, regions, rng)
    Z[:, regions] += amplitude * s[:, None]
    return Z, full


def calibrate_amplitude(r0, target_shift):
    """Amplitude a such that a correlation r0 between two planted regions becomes
    tanh(arctanh(r0) + target_shift): with an orthogonal unit signal added to both,
    r' = (r0 + a^2) / (1 + a^2)."""
    rt = np.tanh(np.arctanh(r0) + target_shift)
    return float(np.sqrt((rt - r0) / (1.0 - rt)))


@dataclass
class PlantedRegions:
    fc: np.ndarray          # (n_pool, R, R) Fisher z; patients recomputed from planted time series
    y: np.ndarray           # (n_pool,) fake labels, 1 = planted
    site: np.ndarray
    pool_idx: np.ndarray
    regions: np.ndarray     # planted regions, strongest first
    inside: np.ndarray      # (R, R) truth: edges inside the set
    incident: np.ndarray    # (R, R) diluted side-effect edges (don't-care in scoring)
    meta: dict = field(default_factory=dict)
    ts: dict = field(default_factory=dict)  # pool row -> planted time series (keep_ts=True only)


def build_regions(fc, dx, site, n_dead_rois, load_ts, m, level, split_seed,
                  rank_seed=0, asd_label=1, control_label=2, pipeline_kwargs=None, keep_ts=False):
    """One region-level planted dataset.

    load_ts(i) must return the (timepoints, regions) ROI time series of dataset subject i,
    from which fc[i] was computed. Controls keep their original matrices; fake patients get
    connectivity recomputed from planted time series with the same pipeline.
    keep_ts=True stores the patients' planted time series (needed to add a scanner decoy)."""
    dx, site = np.asarray(dx), np.asarray(site)
    rank, pool = ranking_split(dx, site, n_dead_rois, control_label, seed=rank_seed)
    d, sd = edge_effect_sizes(fc, np.where(dx == asd_label)[0], rank)
    regions = select_regions(d, m)
    inside, incident = region_masks(fc.shape[1], regions)
    target = LEVELS[level] * float(np.median(np.abs(d[inside]) * sd[inside]))
    y = fake_labels(site[pool], seed=split_seed)
    patients = pool[y == 1]
    r0 = float(np.tanh(fc[patients][:, inside].mean()))
    amplitude = calibrate_amplitude(r0, target)

    rng = np.random.default_rng([split_seed, m, int(LEVELS[level] * 100)])
    out = fc[pool].copy()
    n_full, kept = 0, {}
    for row, i in enumerate(pool):
        if y[row] == 1:
            Z, full = plant_timeseries(load_ts(i), regions, amplitude, rng)
            out[row], _ = connectivity(Z, **(pipeline_kwargs or {}))
            n_full += full
            if keep_ts:
                kept[row] = Z
    meta = {"m": m, "n_inside_edges": int(inside.sum() // 2), "level": level,
            "multiplier": LEVELS[level], "target_shift": target, "amplitude": amplitude,
            "split_seed": split_seed, "rank_seed": rank_seed, "n_rank": len(rank),
            "n_pool": len(pool), "n_patients": int(y.sum()),
            "share_fully_orthogonal": n_full / max(int(y.sum()), 1)}
    return PlantedRegions(out, y, site[pool], pool, regions, inside, incident, meta, kept)
