"""Sanity tests for retinal_gnn.metrics on synthetic vessel masks.

Run from the project root:  python -m pytest tests -q
"""
import numpy as np
import pytest

from retinal_gnn import metrics as M


def make_tree(size=128):
    """A simple 3-pixel-wide 'vessel' tree: one trunk and two branches."""
    g = np.zeros((size, size), bool)
    g[10:118, 62:65] = True          # trunk
    for i in range(40):              # left branch (diagonal)
        g[30 + i, 62 - i: 65 - i] = True
    for i in range(40):              # right branch (diagonal)
        g[50 + i, 62 + i: 65 + i] = True
    return g


def test_perfect_prediction():
    g = make_tree()
    r = M.evaluate_image(g.astype(float), g)
    assert r["dice"] == pytest.approx(1.0, abs=1e-6)
    assert r["cldice"] == pytest.approx(1.0, abs=1e-6)
    assert r["auc"] == pytest.approx(1.0, abs=1e-6)
    assert r["betti0_err"] == 0 and r["betti1_err"] == 0


def test_empty_prediction():
    g = make_tree()
    r = M.evaluate_image(np.zeros_like(g, float), g)
    assert r["dice"] < 1e-6 and r["cldice"] == 0.0


def test_gap_hurts_topology_more_than_overlap():
    """Cutting the trunk barely changes Dice but adds a component.

    This is the failure mode motivating topology metrics.
    """
    g = make_tree()
    p = g.copy()
    p[60:63, :] = False              # 3-pixel gap across the whole trunk row band
    r = M.evaluate_image(p.astype(float), g)
    assert r["dice"] > 0.95
    assert r["betti0_err"] >= 1
    assert r["cldice"] < 1.0


def test_false_bridge_creates_hole():
    """A spurious bridge between two branches creates a loop (b1 error)."""
    g = make_tree()
    p = g.copy()
    p[85:88, 22:103] = True          # horizontal bridge joining both branch ends
    b0, b1 = M.betti_numbers(p)
    gb0, gb1 = M.betti_numbers(g)
    assert b1 > gb1


def test_fov_is_respected():
    g = make_tree()
    fov = np.zeros_like(g)
    fov[:, :64] = True               # only left half counts
    p = g.copy()
    p[:, 64:] = ~g[:, 64:]           # wreck the right half (outside FOV)
    r = M.pixel_metrics(p, g, fov)
    assert r["dice"] == pytest.approx(1.0, abs=1e-6)


def test_dice_equals_f1():
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(0)
    g = rng.random((64, 64)) > 0.8
    p = rng.random((64, 64)) > 0.8
    d = M.pixel_metrics(p, g)["dice"]
    assert d == pytest.approx(f1_score(g.ravel(), p.ravel()), abs=1e-6)


def test_aggregate():
    a = M.aggregate([dict(x=1.0), dict(x=3.0)])
    assert a["x"][0] == pytest.approx(2.0)
