"""ComBat-style site harmonisation, fitted on training data only.

Per matrix entry: x = alpha + X beta + sigma * (gamma_b + delta_b * eps), with b the site
(here: scanner A or B) and X optional covariates (here: the label, "protected" so that
its effect is not removed). Location (gamma) and scale (delta) are estimated per site
from training residuals. Empirical-Bayes shrinkage (Johnson et al. 2007) is optional
(eb=True); by default it is omitted, since with two sites of 100+ subjects each it changes
the estimates very little (checked in scripts/12_combat_eb_check.py).

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


def _eb_shrink(z_site, g_hat, d2_hat, iters=100, tol=1e-6):
    """Parametric empirical-Bayes estimates for one site (Johnson et al. 2007): normal prior
    on the location, inverse-gamma prior on the variance, both fitted across features."""
    n = z_site.shape[0]
    g_bar, t2 = g_hat.mean(), g_hat.var(ddof=1)
    m, v = d2_hat.mean(), d2_hat.var(ddof=1)
    a, b = (2 * v + m ** 2) / v, (m * v + m ** 3) / v
    g, d2 = g_hat.copy(), d2_hat.copy()
    for _ in range(iters):
        g_new = (n * t2 * g_hat + d2 * g_bar) / (n * t2 + d2)
        d2_new = (b + 0.5 * ((z_site - g_new) ** 2).sum(0)) / (n / 2 + a - 1)
        done = max(np.abs(g_new - g).max(), np.abs(d2_new - d2).max()) < tol
        g, d2 = g_new, d2_new
        if done:
            break
    return g, np.sqrt(d2)


def fit(x, batch, covariates=None, eb=False):
    """x: (n, ...) data; batch: (n,) integer site ids 0..S-1; covariates: (n, k) or None.
    eb=True applies empirical-Bayes shrinkage to the site estimates (standard ComBat)."""
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
    if eb:
        ok = sigma > 1e-8
        for b_ in range(S):
            g, dl = _eb_shrink(z[batch == b_][:, ok], gamma[b_, ok], delta[b_, ok] ** 2)
            gamma[b_, ok], delta[b_, ok] = g, dl
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


def add_site(f, x_new):
    """Add an unseen site (e.g. the test hospital), estimated from its own data without
    labels (covariate effect set to its training average). Returns (new fit, site id)."""
    x = np.asarray(x_new, dtype=np.float64)
    sig = np.where(f.sigma > 1e-8, f.sigma, 1.0)
    z = (x - f.alpha) / sig
    g, dl = z.mean(0), z.std(0)
    dl = np.where(dl > 1e-8, dl, 1.0)
    new = ComBatFit(f.alpha, f.beta, f.cov_mean, f.sigma,
                    np.concatenate([f.gamma, g[None]]), np.concatenate([f.delta, dl[None]]))
    return new, len(f.gamma)
