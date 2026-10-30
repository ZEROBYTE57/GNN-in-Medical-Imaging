"""ComBat-style site harmonisation, fitted on training data only.

Per matrix entry: x = alpha + X beta + sigma * (gamma_b + delta_b * eps), with b the site
(here: scanner A or B) and X optional covariates (here: the label, "protected" so that
its effect is not removed). Location (gamma) and scale (delta) are estimated per site
from training residuals. Empirical-Bayes shrinkage is omitted: with two sites of 100+
subjects each it changes estimates very little.

At test time labels are unknown, so the covariate effect is replaced by its training
average (the usual way to harmonise new subjects).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ComBatFit:
    alpha: np.ndarray       # (...,) grand mean
    beta: np.ndarray        # (k, ...) covariate effects (k may be 0)
    cov_mean: np.ndarray    # (k,) training mean of each covariate
    sigma: np.ndarray       # (...,) pooled residual SD
    gamma: np.ndarray       # (n_sites, ...) standardised location shift per site
    delta: np.ndarray       # (n_sites, ...) standardised scale per site


def fit(x, batch, covariates=None):
    """x: (n, ...) data; batch: (n,) integer site ids 0..S-1; covariates: (n, k) or None."""
    x = np.asarray(x, dtype=np.float64)
    n, shape = x.shape[0], x.shape[1:]
    flat = x.reshape(n, -1)
    batch = np.asarray(batch)
    S = int(batch.max()) + 1
    B = np.eye(S)[batch]
    C = np.zeros((n, 0)) if covariates is None else np.asarray(covariates, dtype=np.float64).reshape(n, -1)
    Cc = C - C.mean(0)
    coef, *_ = np.linalg.lstsq(np.hstack([B, Cc]), flat, rcond=None)
    site_means, beta = coef[:S], coef[S:]
    w = B.mean(0)
    alpha = w @ site_means
    resid = flat - B @ site_means - Cc @ beta
    sigma = np.sqrt((resid ** 2).mean(0))
    sig = np.where(sigma > 1e-8, sigma, 1.0)
    z = (flat - alpha - Cc @ beta) / sig
    gamma = np.stack([z[batch == s].mean(0) for s in range(S)])
    delta = np.stack([z[batch == s].std(0) for s in range(S)])
    delta = np.where(delta > 1e-8, delta, 1.0)
    return ComBatFit(alpha.reshape(shape), beta.reshape((C.shape[1],) + shape), C.mean(0),
                     sigma.reshape(shape), gamma.reshape((S,) + shape), delta.reshape((S,) + shape))


def _centre(f, covariates, n):
    """alpha + X beta, with X centred on the training mean; zero covariate effect if unknown."""
    if covariates is None or f.beta.shape[0] == 0:
        return np.broadcast_to(f.alpha, (n,) + f.alpha.shape)
    C = np.asarray(covariates, dtype=np.float64).reshape(n, -1) - f.cov_mean
    return f.alpha + np.tensordot(C, f.beta, axes=1)


def harmonise(f, x, batch, covariates=None):
    """Remove the site effect: map every subject to the pooled reference."""
    x = np.asarray(x, dtype=np.float64)
    c = _centre(f, covariates, len(x))
    sig = np.where(f.sigma > 1e-8, f.sigma, 1.0)
    z = (x - c) / sig
    return c + sig * (z - f.gamma[batch]) / f.delta[batch]


def swap(f, x, batch, new_batch, covariates=None):
    """Move subjects from site `batch` to site `new_batch` (keeps everything else)."""
    x = np.asarray(x, dtype=np.float64)
    c = _centre(f, covariates, len(x))
    sig = np.where(f.sigma > 1e-8, f.sigma, 1.0)
    z = (x - c) / sig
    z = (z - f.gamma[batch]) / f.delta[batch] * f.delta[new_batch] + f.gamma[new_batch]
    return c + sig * z
