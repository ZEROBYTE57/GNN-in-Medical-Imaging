"""Paired bootstrap on C-index differences.

The comparison of interest is *within* a fold: the same held-out patients scored
by two graph rules. Comparing two independent confidence intervals throws away
that pairing and is badly under-powered on a 41-patient centre like HMR, so the
statistic here is resampled on the *difference*.

Resampling is stratified by fold: each bootstrap replicate draws patients with
replacement inside each held-out centre, so a large centre cannot dominate the
replicate and the fold structure of the protocol is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..seed import rng
from .metrics import harrell_cindex


@dataclass
class BootstrapResult:
    """Paired difference in C-index, model A minus model B."""

    observed: float
    ci_low: float
    ci_high: float
    p_value: float
    n_iters: int
    mean: float
    std: float

    @property
    def significant(self) -> bool:
        return not (self.ci_low <= 0.0 <= self.ci_high)

    def format(self, alpha: float = 0.05) -> str:
        pct = int(round((1 - alpha) * 100))
        return (
            f"dC = {self.observed:+.3f} "
            f"({pct}% CI {self.ci_low:+.3f} to {self.ci_high:+.3f}, p = {self.p_value:.4f})"
        )


def paired_bootstrap_cindex(
    risk_a: np.ndarray,
    risk_b: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    fold: np.ndarray | None = None,
    n_iters: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapResult:
    """Bootstrap the paired difference ``C(risk_a) - C(risk_b)``.

    Parameters
    ----------
    fold:
        Stratification label per patient -- pass the held-out centre so that
        resampling happens inside each LOCO fold.

    Notes
    -----
    The p-value is the standard two-sided bootstrap value: twice the smaller
    tail mass of the replicate distribution around zero, with the usual
    ``(count + 1) / (n + 1)`` correction so it can never be exactly 0.
    """
    risk_a, risk_b = np.asarray(risk_a), np.asarray(risk_b)
    time, event = np.asarray(time), np.asarray(event)
    n = len(time)
    fold = np.zeros(n, dtype=int) if fold is None else np.asarray(fold)

    observed = harrell_cindex(risk_a, time, event) - harrell_cindex(risk_b, time, event)

    groups = [np.flatnonzero(fold == f) for f in np.unique(fold)]
    g = rng(seed)
    diffs = np.empty(n_iters)
    diffs[:] = np.nan

    for it in range(n_iters):
        idx = np.concatenate([g.choice(grp, size=len(grp), replace=True) for grp in groups])
        if event[idx].sum() < 2:
            continue                       # degenerate replicate; left as NaN
        ca = harrell_cindex(risk_a[idx], time[idx], event[idx])
        cb = harrell_cindex(risk_b[idx], time[idx], event[idx])
        diffs[it] = ca - cb

    diffs = diffs[~np.isnan(diffs)]
    if diffs.size == 0:
        return BootstrapResult(observed, np.nan, np.nan, np.nan, 0, np.nan, np.nan)

    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    n_eff = diffs.size
    tail = min((diffs <= 0).sum(), (diffs >= 0).sum())
    p = min(1.0, 2.0 * (tail + 1) / (n_eff + 1))

    return BootstrapResult(
        observed=float(observed),
        ci_low=float(lo),
        ci_high=float(hi),
        p_value=float(p),
        n_iters=int(n_eff),
        mean=float(diffs.mean()),
        std=float(diffs.std()),
    )


def bootstrap_cindex_ci(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    n_iters: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile CI for a single model's C-index (point, low, high)."""
    risk, time, event = np.asarray(risk), np.asarray(time), np.asarray(event)
    point = harrell_cindex(risk, time, event)
    g = rng(seed)
    n = len(time)
    vals = []
    for _ in range(n_iters):
        idx = g.integers(0, n, n)
        if event[idx].sum() < 2:
            continue
        vals.append(harrell_cindex(risk[idx], time[idx], event[idx]))
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)
