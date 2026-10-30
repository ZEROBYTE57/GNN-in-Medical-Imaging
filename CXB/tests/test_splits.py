import numpy as np

from cxb.splits import BRAINOOD_FOLDS, site_fold


def test_folds_are_disjoint_and_cover_all():
    sites = np.array(["NYU"] * 5 + [s for pair in BRAINOOD_FOLDS for s in pair])
    for fold in range(len(BRAINOOD_FOLDS)):
        tr, va, te = site_fold(sites, fold)
        assert len(te) and len(va)
        assert not set(tr) & set(va) and not set(tr) & set(te) and not set(va) & set(te)
        assert len(tr) + len(va) + len(te) == len(sites)


def test_dropped_site_is_left_out():
    sites = np.array(["NYU"] * 3 + ["CMU"] * 2 + [s for pair in BRAINOOD_FOLDS for s in pair])
    tr, va, te = site_fold(sites, 0)
    assert not np.isin(np.concatenate([tr, va, te]), np.where(sites == "CMU")[0]).any()
    assert len(tr) + len(va) + len(te) == len(sites) - 2


def test_ten_folds_with_distinct_test_sites():
    tests = [t for t, _ in BRAINOOD_FOLDS]
    assert len(tests) == 10 and len(set(tests)) == 10
    assert all(t != v for t, v in BRAINOOD_FOLDS)
