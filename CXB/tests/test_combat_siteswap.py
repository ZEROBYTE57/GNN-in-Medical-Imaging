import numpy as np
import torch

from cxb import combat
from cxb.models import MLP
from cxb.siteswap import train_siteswap


def _data(n=400, p=6, seed=0, link=0.9):
    """Feature 0 carries the label; feature 5 carries a site shift; site linked to label."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    b = np.where(rng.random(n) < link, y, 1 - y)
    x = rng.normal(0, 1, (n, p))
    x[:, 0] += 1.0 * y
    x[:, 5] += 2.0 * b
    return x, y, b


def test_unprotected_combat_removes_site_and_part_of_label():
    x, y, b = _data()
    h = combat.harmonise(combat.fit(x, b), x, b)
    assert abs(h[b == 1, 5].mean() - h[b == 0, 5].mean()) < 0.05            # site gone
    assert h[y == 1, 0].mean() - h[y == 0, 0].mean() < 0.6                    # label effect damaged


def test_protected_combat_keeps_label_and_removes_site():
    x, y, b = _data()
    f = combat.fit(x, b, covariates=y[:, None])
    h = combat.harmonise(f, x, b)                                             # test time: no labels
    assert abs(h[b == 1, 5].mean() - h[b == 0, 5].mean()) < 0.3
    assert h[y == 1, 0].mean() - h[y == 0, 0].mean() > 0.8


def test_swap_moves_site_and_round_trips():
    x, y, b = _data()
    f = combat.fit(x, b, covariates=y[:, None])
    s = combat.swap(f, x, b, 1 - b, covariates=y[:, None])
    assert abs((s[b == 0, 5] - x[b == 0, 5]).mean() - 2.0) < 0.3
    back = combat.swap(f, s, 1 - b, b, covariates=y[:, None])
    assert np.allclose(back, x)


def test_siteswap_training_runs_and_learns():
    rng = np.random.default_rng(0)
    n, R = 160, 6
    fc = rng.normal(0, 0.3, (n, R, R))
    fc = (fc + fc.transpose(0, 2, 1)) / 2
    y = np.arange(n) % 2
    fc[y == 1, 0, 1] += 1.0
    fc[y == 1, 1, 0] += 1.0
    t = torch.tensor(fc, dtype=torch.float32)
    yt = torch.tensor(y)
    model = MLP(R, fc.mean(0), fc.std(0), dropout=0.1)
    model, _, curve = train_siteswap(model, t[:120], t[:120], yt[:120], t[120:], yt[120:], lam=1.0, epochs=20)
    assert (model(t[120:]).argmax(1) == yt[120:]).float().mean() > 0.9
