import numpy as np
import pandas as pd

from cxb.data import connectivity, roi_file_url, select_subjects


def test_connectivity_matches_numpy_and_is_symmetric():
    rng = np.random.default_rng(0)
    ts = rng.standard_normal((150, 10))
    fc, dead = connectivity(ts, fisher_z=False)
    ref = np.corrcoef(ts.T)
    np.fill_diagonal(ref, 0)
    assert dead == 0
    np.testing.assert_allclose(fc, ref, atol=1e-5)
    np.testing.assert_allclose(fc, fc.T)


def test_fisher_z_finite_for_identical_signals():
    x = np.random.default_rng(1).standard_normal((120, 1))
    fc, _ = connectivity(np.hstack([x, x, -x]), fisher_z=True)
    assert np.isfinite(fc).all()
    assert fc[0, 1] > 5 and fc[0, 2] < -5


def test_dead_roi_gets_zero_not_nan():
    ts = np.random.default_rng(2).standard_normal((100, 4))
    ts[:, 2] = 3.0
    fc, dead = connectivity(ts)
    assert dead == 1
    assert np.all(fc[2] == 0) and np.all(fc[:, 2] == 0)
    assert np.isfinite(fc).all()


def test_select_subjects_drops_missing_and_failed():
    pheno = pd.DataFrame({
        "FILE_ID": ["a", "no_filename", "c", "d"],
        "qc_rater_1": ["OK", "OK", "fail", "OK"],
        "qc_func_rater_2": ["OK", "OK", "OK", "OK"],
    })
    assert list(select_subjects(pheno)["FILE_ID"]) == ["a", "d"]
    assert list(select_subjects(pheno, quality_checked=False)["FILE_ID"]) == ["a", "c", "d"]


def test_url_layout():
    u = roi_file_url("B", "cpac", "filt_noglobal", "rois_aal", "Pitt_0050003")
    assert u == "B/Outputs/cpac/filt_noglobal/rois_aal/Pitt_0050003_rois_aal.1D"
