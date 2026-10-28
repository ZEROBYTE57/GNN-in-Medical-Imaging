"""Feature selection cascade -- reproduced from Peng et al. Section 3.1.

    z-score  ->  variance < 0.005 removed
             ->  univariate Cox, 50 repeats x 3-fold CV, rank by mean C-index,
                 keep top 50
             ->  Spearman |rho| > 0.8: drop the member of the pair with the
                 lower C-index
             ->  LASSO-Cox, keep at most `max_features` per modality

The critical difference from the original paper is *where* this runs. Peng et
al. fit selection once on a fixed training set. Under leave-one-centre-out the
whole cascade -- including the z-score statistics -- is refit inside every fold
on the training centres only. Selecting features on data that includes the
held-out centre would leak site information into the very thing this study is
trying to measure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import rankdata

logger = logging.getLogger(__name__)

_WARNED_LASSO = False


def _warn_once_lasso() -> None:
    """Announce the LASSO fallback once, not once per fold x modality x arm."""
    global _WARNED_LASSO
    if not _WARNED_LASSO:
        _WARNED_LASSO = True
        logger.warning(
            "scikit-survival not installed -- narrowing by C-index rank instead "
            "of LASSO-Cox. Install scikit-survival for the protocol-faithful path."
        )


# ---------------------------------------------------------------------------
# Fast univariate concordance
# ---------------------------------------------------------------------------
def comparable_pairs(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indices ``(i, j)`` of usable pairs, where ``i`` has the earlier event.

    A pair is comparable when the earlier of the two is an observed event
    (Harrell's definition). Ties in time are dropped.
    """
    order = np.argsort(time, kind="mergesort")
    t, e = time[order], event[order]
    n = len(t)
    ii, jj = np.triu_indices(n, k=1)
    usable = (e[ii] == 1) & (t[ii] < t[jj])
    return order[ii[usable]], order[jj[usable]]


def concordance_from_pairs(
    scores: np.ndarray, i_idx: np.ndarray, j_idx: np.ndarray
) -> np.ndarray:
    """Harrell's C for many score vectors at once.

    ``scores`` is ``(n_samples,)`` or ``(n_samples, n_features)``; higher score
    must mean higher risk. Ties count as half, as in the standard definition.
    Returns one C-index per column.
    """
    scores = np.atleast_2d(scores.T).T if scores.ndim == 1 else scores
    if len(i_idx) == 0:
        return np.full(scores.shape[1], 0.5)
    diff = scores[i_idx] - scores[j_idx]         # (n_pairs, n_features)
    concordant = (diff > 0).sum(axis=0)
    tied = (diff == 0).sum(axis=0)
    return (concordant + 0.5 * tied) / len(i_idx)


def univariate_cindex(
    x: np.ndarray, time: np.ndarray, event: np.ndarray, chunk: int = 256
) -> np.ndarray:
    """C-index of every column, oriented so that C >= 0.5.

    A univariate Cox model's C-index is a rank statistic in the covariate, so
    fitting the model is unnecessary: the C-index of the fitted risk score is
    ``max(C(x), 1 - C(x))``, reached at the sign of the coefficient. This makes
    the 150-fit-per-feature protocol tractable on 3122 raw features.
    """
    i_idx, j_idx = comparable_pairs(time, event)
    out = np.empty(x.shape[1])
    for start in range(0, x.shape[1], chunk):
        block = x[:, start : start + chunk]
        c = concordance_from_pairs(block, i_idx, j_idx)
        out[start : start + chunk] = np.maximum(c, 1.0 - c)
    return out


# ---------------------------------------------------------------------------
# The cascade
# ---------------------------------------------------------------------------
@dataclass
class SelectionResult:
    """Everything needed to apply the same selection to new patients."""

    columns: list[str]
    mean_c_index: dict[str, float]
    mu: np.ndarray
    sigma: np.ndarray
    stage_counts: dict[str, int] = field(default_factory=dict)

    def transform(self, x: np.ndarray, all_columns: list[str]) -> np.ndarray:
        """Z-score with the TRAINING statistics, then take the kept columns."""
        index = {c: i for i, c in enumerate(all_columns)}
        cols = [index[c] for c in self.columns]
        z = (x - self.mu) / self.sigma
        return z[:, cols]


class FeatureSelector:
    """Stateful cascade; fit on training rows, apply to everything."""

    def __init__(
        self,
        variance_threshold: float = 0.005,
        univariate_repeats: int = 50,
        univariate_splits: int = 3,
        top_k_univariate: int = 50,
        spearman_threshold: float = 0.8,
        lasso_cv_folds: int = 5,
        max_features: int = 12,
        seed: int = 0,
    ) -> None:
        self.variance_threshold = variance_threshold
        self.univariate_repeats = univariate_repeats
        self.univariate_splits = univariate_splits
        self.top_k_univariate = top_k_univariate
        self.spearman_threshold = spearman_threshold
        self.lasso_cv_folds = lasso_cv_folds
        self.max_features = max_features
        self.seed = seed

    # -- stages -------------------------------------------------------------
    def _standardise(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu = np.nanmean(x, axis=0)
        sigma = np.nanstd(x, axis=0)
        sigma[sigma < 1e-12] = 1.0
        z = (np.nan_to_num(x, nan=0.0) - mu) / sigma
        return z, mu, sigma

    def _repeated_univariate(
        self, z: np.ndarray, time: np.ndarray, event: np.ndarray
    ) -> np.ndarray:
        """Mean C-index over ``repeats x splits`` held-out folds."""
        from ..seed import rng

        g = rng(self.seed)
        n = len(time)
        acc = np.zeros(z.shape[1])
        n_folds = 0
        for _ in range(self.univariate_repeats):
            perm = g.permutation(n)
            folds = np.array_split(perm, self.univariate_splits)
            for fold in folds:
                if len(fold) < 4 or event[fold].sum() < 2:
                    continue
                acc += univariate_cindex(z[fold], time[fold], event[fold])
                n_folds += 1
        if n_folds == 0:
            raise ValueError("no evaluable folds -- too few events in the training set")
        return acc / n_folds

    def _prune_correlated(
        self, z: np.ndarray, columns: list[str], scores: np.ndarray
    ) -> list[int]:
        """Greedy: walk features best-first, drop any that correlate with a keeper."""
        ranks = np.apply_along_axis(rankdata, 0, z)
        order = np.argsort(-scores)
        keep: list[int] = []
        for idx in order:
            if not keep:
                keep.append(idx)
                continue
            block = ranks[:, keep]
            rho = _corr_vector(ranks[:, idx], block)
            if np.max(np.abs(rho)) <= self.spearman_threshold:
                keep.append(idx)
        logger.debug("spearman pruning: %d -> %d", len(columns), len(keep))
        return keep

    def _lasso(
        self, z: np.ndarray, time: np.ndarray, event: np.ndarray, scores: np.ndarray
    ) -> np.ndarray:
        """LASSO-Cox narrowing; falls back to the C-index ranking if unavailable.

        Returns indices into the columns of ``z``, at most ``max_features``.
        """
        n_target = min(self.max_features, z.shape[1])
        try:
            from sksurv.linear_model import CoxnetSurvivalAnalysis
            from sksurv.util import Surv
        except ImportError:
            _warn_once_lasso()
            return np.argsort(-scores)[:n_target]

        y = Surv.from_arrays(event=event.astype(bool), time=time)
        try:
            model = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.01, n_alphas=100)
            model.fit(z, y)
        except Exception as exc:  # noqa: BLE001 - coxnet is fragile on wide data
            logger.warning("LASSO-Cox failed (%s); falling back to C-index rank", exc)
            return np.argsort(-scores)[:n_target]

        # Walk the regularisation path from strong to weak penalty and stop at
        # the first alpha admitting >= n_target non-zero coefficients. This is
        # the "smallest model of the target size" rather than a CV-tuned alpha,
        # which keeps the feature count comparable to the 12 per modality that
        # Peng et al. report.
        coefs = model.coef_                       # (n_features, n_alphas)
        for k in range(coefs.shape[1]):
            nz = np.flatnonzero(coefs[:, k])
            if len(nz) >= n_target:
                strength = np.abs(coefs[nz, k])
                return nz[np.argsort(-strength)][:n_target]
        nz = np.flatnonzero(coefs[:, -1])
        if len(nz) == 0:
            return np.argsort(-scores)[:n_target]
        return nz[: n_target]

    # -- public API ---------------------------------------------------------
    def fit(
        self,
        x: np.ndarray,
        columns: list[str],
        time: np.ndarray,
        event: np.ndarray,
    ) -> SelectionResult:
        """Run the cascade on training rows only."""
        stages: dict[str, int] = {"input": x.shape[1]}

        z_all, mu, sigma = self._standardise(x)

        # 1. variance filter (on the standardised matrix the variance of a
        #    constant column is 0, so this also removes degenerate features)
        var = np.var(np.nan_to_num(x, nan=0.0), axis=0)
        keep = np.flatnonzero(var >= self.variance_threshold)
        if len(keep) == 0:
            raise ValueError("variance filter removed every feature")
        stages["after_variance"] = len(keep)

        z = z_all[:, keep]
        cols = [columns[i] for i in keep]

        # 2. repeated univariate Cox ranking -> top 50
        scores = self._repeated_univariate(z, time, event)
        top = np.argsort(-scores)[: self.top_k_univariate]
        z, cols, scores = z[:, top], [cols[i] for i in top], scores[top]
        keep = keep[top]
        stages["after_univariate"] = len(cols)

        # 3. Spearman redundancy pruning
        kept = self._prune_correlated(z, cols, scores)
        z, cols, scores = z[:, kept], [cols[i] for i in kept], scores[kept]
        keep = keep[kept]
        stages["after_spearman"] = len(cols)

        # 4. LASSO-Cox
        final = self._lasso(z, time, event, scores)
        cols = [cols[i] for i in final]
        scores = scores[final]
        stages["final"] = len(cols)

        logger.info("selection: %s", stages)
        return SelectionResult(
            columns=cols,
            mean_c_index=dict(zip(cols, scores.tolist(), strict=True)),
            mu=mu,
            sigma=sigma,
            stage_counts=stages,
        )


def _corr_vector(v: np.ndarray, block: np.ndarray) -> np.ndarray:
    """Pearson correlation of ``v`` against each column of ``block``.

    Applied to rank-transformed inputs this is Spearman's rho.
    """
    v = v - v.mean()
    b = block - block.mean(axis=0)
    denom = np.linalg.norm(v) * np.linalg.norm(b, axis=0)
    denom[denom == 0] = 1.0
    return (v @ b) / denom

