"""Inter-patient similarity and Similarity Network Fusion.

Implements Wang et al.'s SNF (Nat. Methods 11, 2014) directly rather than
depending on ``snfpy``: the algorithm is ~60 lines, and pinning it here removes
a dependency whose defaults have changed between releases -- which matters when
the whole study is a claim about graph construction.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def scaled_exponential_kernel(
    x: np.ndarray,
    y: np.ndarray | None = None,
    k: int = 20,
    mu: float = 0.5,
    metric: str = "sqeuclidean",
) -> np.ndarray:
    """Wang et al.'s locally scaled affinity.

        W(i,j) = exp( -d(i,j)^2 / (mu * eps(i,j)) )
        eps(i,j) = ( mean_k d(i, N_k(i)) + mean_k d(j, N_k(j)) + d(i,j) ) / 3

    The local bandwidth ``eps`` is what makes the kernel scale-free -- a fixed
    global bandwidth is one of the ways hand-crafted population graphs go wrong.

    When ``y`` is given the result is the rectangular affinity of ``x`` rows
    against ``y`` rows, with the local scale of ``y`` taken from ``y`` itself.
    This is the form needed to attach held-out patients to a training graph.
    """
    dist_xx = cdist(x, x, metric=metric)
    scale_x = _local_scale(dist_xx, k)

    if y is None:
        dist = dist_xx
        scale_y = scale_x
    else:
        dist_yy = cdist(y, y, metric=metric)
        scale_y = _local_scale(dist_yy, k)
        dist = cdist(x, y, metric=metric)

    eps = (scale_x[:, None] + scale_y[None, :] + dist) / 3.0
    eps = np.maximum(eps, np.finfo(float).eps)
    w = np.exp(-dist / (mu * eps))
    if y is None:
        np.fill_diagonal(w, 1.0)
    return w


def _local_scale(dist: np.ndarray, k: int) -> np.ndarray:
    """Mean distance to the k nearest neighbours, excluding self."""
    n = dist.shape[0]
    k = int(np.clip(k, 1, max(1, n - 1)))
    d = dist.copy()
    np.fill_diagonal(d, np.inf)
    nearest = np.sort(d, axis=1)[:, :k]
    return nearest.mean(axis=1)


def affinity_matrix(x: np.ndarray, k: int = 20, mu: float = 0.5) -> np.ndarray:
    """Symmetric, unit-diagonal affinity for one modality."""
    w = scaled_exponential_kernel(x, k=k, mu=mu)
    return (w + w.T) / 2.0


def _full_kernel(w: np.ndarray) -> np.ndarray:
    """Row-normalised affinity with the diagonal held at 1/2 (Wang et al. Eq. 3)."""
    p = w.copy()
    np.fill_diagonal(p, 0.0)
    row = p.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    p = p / (2.0 * row)
    np.fill_diagonal(p, 0.5)
    return p


def _sparse_kernel(w: np.ndarray, k: int) -> np.ndarray:
    """Row-normalised affinity restricted to each row's k nearest neighbours."""
    n = w.shape[0]
    k = int(np.clip(k, 1, max(1, n - 1)))
    s = np.zeros_like(w)
    masked = w.copy()
    np.fill_diagonal(masked, -np.inf)
    idx = np.argsort(-masked, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    s[rows, idx.ravel()] = w[rows, idx.ravel()]
    row = s.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    return s / row


def snf(
    affinities: list[np.ndarray],
    k: int = 20,
    n_iters: int = 20,
) -> np.ndarray:
    """Fuse per-modality affinities into one network.

    Each view's full kernel is repeatedly diffused through the *other* views'
    sparse kernels, so a pair of patients keeps a strong edge only if several
    modalities agree about their local neighbourhood.
    """
    if len(affinities) == 1:
        return _normalise01(affinities[0])

    views = [_full_kernel(w) for w in affinities]
    sparse = [_sparse_kernel(w, k) for w in affinities]
    n_views = len(views)

    for _ in range(n_iters):
        updated = []
        for v in range(n_views):
            others = sum(views[u] for u in range(n_views) if u != v) / (n_views - 1)
            p = sparse[v] @ others @ sparse[v].T
            updated.append(_full_kernel(p))
        views = updated

    fused = sum(views) / n_views
    fused = (fused + fused.T) / 2.0
    return _normalise01(fused)


def _normalise01(w: np.ndarray) -> np.ndarray:
    """Min-max scale the off-diagonal entries into [0, 1].

    Peng et al. threshold the fused similarity at a fixed 0.2, which is only
    meaningful on a fixed scale. SNF outputs are unnormalised diffusion weights
    whose magnitude depends on cohort size, so the scaling has to be explicit
    -- and it is one of the places where "the same rule" silently means
    different things on different cohorts.
    """
    out = w.copy()
    n = out.shape[0]
    off = ~np.eye(n, dtype=bool)
    values = out[off]
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < np.finfo(float).eps:
        out[off] = 0.0
    else:
        out[off] = (values - lo) / (hi - lo)
    np.fill_diagonal(out, 1.0)
    return out
