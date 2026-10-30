import numpy as np

from cxb.decoy import assign_scanner, build_decoy, select_decoy_regions, site_effect_eta2, top_share
from cxb.planting import build_regions, region_masks

from test_planting import _toy_ts


def test_assign_scanner_links_label_within_site():
    rng = np.random.default_rng(0)
    y = np.array([1, 0] * 100)
    site = np.array(["A"] * 100 + ["B"] * 100)
    b = assign_scanner(y, site, 0.8, rng)
    for s in ("A", "B"):
        assert b[(site == s) & (y == 1)].mean() == 0.8
        assert b[(site == s) & (y == 0)].mean() == 0.2
    b = assign_scanner(y, site, 0.5, rng)
    assert b[y == 1].mean() == b[y == 0].mean() == 0.5


def test_site_effect_found_and_planted_regions_excluded():
    rng = np.random.default_rng(0)
    fc = rng.normal(0, 0.1, (60, 6, 6))
    site = np.array(["A", "B"] * 30)
    fc[site == "B", 4, 5] += 1.0
    fc[site == "B", 5, 4] += 1.0
    eta, sites, means = site_effect_eta2(fc, site, np.arange(60), min_n=10)
    assert eta[4, 5] > 0.9 and sites == ["A", "B"]
    assert set(select_decoy_regions(eta, [0], 2).tolist()) == {4, 5}
    assert 4 not in select_decoy_regions(eta, [4], 2)


def test_build_decoy_changes_only_decoy_edges():
    ts, fc, dx, site, dead = _toy_ts()
    p = build_regions(fc, dx, site, dead, lambda i: ts[i], m=3, level="x2", split_seed=0, keep_ts=True)
    dec = build_decoy(p, lambda i: ts[i], fc, dx, site, m=3, min_site_n=10)
    assert not set(dec.regions) & set(p.regions)
    delta = dec.fc_b - p.fc
    assert delta[:, dec.inside].mean() > 0
    other = ~dec.inside & ~dec.incident
    np.fill_diagonal(other, False)
    assert np.abs(delta[:, other]).max() < 1e-4                    # disease edges untouched
    assert abs(dec.meta["achieved_shift"] - dec.meta["target_shift"]) < 0.3 * dec.meta["target_shift"]


def test_top_share():
    inside, _ = region_masks(5, [0, 1])
    iu = np.triu_indices(5, 1)
    scores = inside[iu].astype(float)
    assert top_share(scores, inside, 1) == 1.0
