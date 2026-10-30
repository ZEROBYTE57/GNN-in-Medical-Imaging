import numpy as np

from cxb.planting import (build, edge_effect_sizes, fake_labels, invalid_fraction, plant,
                          ranking_split, top_edges, truth_mask)


def _toy(n=120, R=8, seed=0):
    rng = np.random.default_rng(seed)
    fc = rng.normal(0.3, 0.2, (n, R, R)).astype(np.float32)
    fc = (fc + fc.transpose(0, 2, 1)) / 2
    fc[:, np.arange(R), np.arange(R)] = 0
    dx = np.array([1] * 40 + [2] * 80)
    site = np.array(["A", "B"] * 60)
    dead = np.zeros(n, dtype=int)
    fc[dx == 1, 1, 5] += 0.5  # a real "autism" effect on edge (1, 5)
    fc[dx == 1, 5, 1] += 0.5
    return fc, dx, site, dead


def test_ranking_split_disjoint_stratified_and_excludes_dead():
    fc, dx, site, dead = _toy()
    dead[np.where(dx == 2)[0][:3]] = 1
    rank, pool = ranking_split(dx, site, dead, seed=0)
    assert not set(rank) & set(pool)
    assert all(dx[rank] == 2) and all(dx[pool] == 2) and all(dead[pool] == 0)
    for s in "AB":
        assert (site[rank] == s).sum() == round(0.2 * ((dx == 2) & (site == s)).sum())
    r2, _ = ranking_split(dx, site, dead, seed=0)
    assert np.array_equal(rank, r2)


def test_effect_size_finds_real_edge():
    fc, dx, site, dead = _toy()
    d, sd = edge_effect_sizes(fc, np.where(dx == 1)[0], np.where(dx == 2)[0])
    assert np.allclose(d, d.T)
    i, j = top_edges(d, 1)
    assert {int(i[0]), int(j[0])} == {1, 5} and d[1, 5] > 1


def test_fake_labels_balanced_within_site():
    site = np.array(["A"] * 10 + ["B"] * 7)
    y = fake_labels(site, seed=3)
    assert y[site == "A"].sum() == 5 and y[site == "B"].sum() in (3, 4)
    assert np.array_equal(y, fake_labels(site, seed=3))


def test_plant_changes_only_patients_on_planted_edges_symmetrically():
    fc, dx, site, dead = _toy()
    d, sd = edge_effect_sizes(fc, np.where(dx == 1)[0], np.where(dx == 2)[0])
    edges = top_edges(d, 3)
    y = np.array([1, 0] * 60)
    out = plant(fc, y, edges, d, sd, 2.0)
    diff = out - fc
    assert np.allclose(diff[y == 0], 0)
    mask = truth_mask(fc.shape[1], edges)
    assert np.allclose(diff[:, ~mask], 0)
    assert np.allclose(diff, diff.transpose(0, 2, 1))
    i, j = edges
    assert np.allclose(diff[y == 1][:, i, j], 2.0 * d[i, j] * sd[i, j], atol=1e-6)


def test_invalid_fraction_detects_broken_matrix():
    good = np.zeros((2, 4, 4), dtype=np.float32)
    bad = good.copy()
    bad[1, 0, 1] = bad[1, 1, 0] = 3.0   # r about 0.995 ...
    bad[1, 0, 2] = bad[1, 2, 0] = 3.0
    bad[1, 1, 2] = bad[1, 2, 1] = -3.0  # ... inconsistent with this one
    assert invalid_fraction(good) == 0.0 and invalid_fraction(bad) == 0.5


def test_build_end_to_end():
    fc, dx, site, dead = _toy()
    p = build(fc, dx, site, dead, k=2, level="strong", split_seed=1)
    assert p.fc.shape[0] == len(p.y) == len(p.pool_idx) == len(p.site)
    assert p.truth.sum() == 4 and p.meta["multiplier"] == 4.0
    assert set(np.unique(p.y)) == {0, 1}


# ----------------------------------------------------------------------------- region level

from cxb.data import connectivity  # noqa: E402
from cxb.planting import (build_regions, calibrate_amplitude, plant_timeseries,  # noqa: E402
                          region_masks, select_regions)


def _toy_ts(n=120, R=10, T=200, seed=0):
    """Time series with real correlation structure; autism subjects share an extra signal
    on regions 2 and 7."""
    rng = np.random.default_rng(seed)
    mix = rng.normal(0, 0.5, (R, R))
    ts = [rng.standard_normal((T, R)) @ mix + rng.standard_normal((T, R)) for _ in range(n)]
    dx = np.array([1] * 40 + [2] * 80)
    for i in np.where(dx == 1)[0]:
        ts[i][:, [2, 7]] += rng.standard_normal((T, 1))
    fc = np.stack([connectivity(t)[0] for t in ts])
    return ts, fc, dx, np.array(["A", "B"] * (n // 2)), np.zeros(n, dtype=int)


def test_region_masks():
    inside, incident = region_masks(5, [1, 3])
    assert inside.sum() == 2 and inside[1, 3] and inside[3, 1]
    assert incident.sum() == 2 * 2 * 3 and incident[1, 0] and not incident[0, 2]
    assert not (inside & incident).any() and not inside.diagonal().any()


def test_select_regions_finds_real_effect():
    ts, fc, dx, site, dead = _toy_ts()
    from cxb.planting import edge_effect_sizes
    d, _ = edge_effect_sizes(fc, np.where(dx == 1)[0], np.where(dx == 2)[0])
    assert set(select_regions(d, 2).tolist()) == {2, 7}


def test_calibration_formula():
    a = calibrate_amplitude(0.3, 0.2)
    assert np.isclose((0.3 + a ** 2) / (1 + a ** 2), np.tanh(np.arctanh(0.3) + 0.2))


def test_plant_timeseries_changes_only_set_edges_and_stays_valid():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 8)) @ rng.normal(0, 0.5, (8, 8)) + rng.standard_normal((300, 8))
    Z, full = plant_timeseries(X, np.array([0, 4, 5]), 0.6, rng)
    assert full
    before, _ = connectivity(X, fisher_z=False)
    after, _ = connectivity(Z, fisher_z=False)
    inside, incident = region_masks(8, [0, 4, 5])
    other = ~inside & ~incident
    np.fill_diagonal(other, False)
    assert np.allclose(after[other], before[other], atol=1e-5)          # untouched
    assert (after[inside] > before[inside]).all()                         # raised
    r = after.astype(np.float64)
    np.fill_diagonal(r, 1.0)
    assert np.linalg.eigvalsh(r)[0] > -1e-6                               # valid matrix


def test_build_regions_end_to_end():
    ts, fc, dx, site, dead = _toy_ts()
    p = build_regions(fc, dx, site, dead, lambda i: ts[i], m=3, level="x2", split_seed=0)
    pat, ctl = p.y == 1, p.y == 0
    assert np.array_equal(p.fc[ctl], fc[p.pool_idx[ctl]])                # controls untouched
    shift = (p.fc[pat] - fc[p.pool_idx[pat]])[:, p.inside].mean()
    assert abs(shift - p.meta["target_shift"]) < 0.3 * p.meta["target_shift"]
    other = ~p.inside & ~p.incident
    np.fill_diagonal(other, False)
    assert np.abs(p.fc[pat] - fc[p.pool_idx[pat]])[:, other].max() < 1e-4
    assert p.meta["share_fully_orthogonal"] == 1.0
