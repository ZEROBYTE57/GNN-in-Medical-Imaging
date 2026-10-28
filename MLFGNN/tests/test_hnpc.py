"""Real-cohort loaders, run against the public TCIA spreadsheets when present."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLINICAL = ROOT / "data/raw/INFOclinical_HN_Version2_30may2018.xlsx"
CONTOURS = ROOT / "data/raw/INFO_GTVcontours_HN.xlsx"

pytestmark = pytest.mark.skipif(
    not (CLINICAL.exists() and CONTOURS.exists()),
    reason="TCIA spreadsheets not downloaded (see scripts/01a_download_hnpc.py)",
)


@pytest.fixture(scope="module")
def clinical():
    from mlfgnn.data.hnpc import load_hnpc_clinical

    return load_hnpc_clinical(CLINICAL)


def test_cohort_matches_the_published_collection(clinical):
    """298 patients across CHUS / HGJ / CHUM / HMR, as on the TCIA page."""
    assert len(clinical) == 298
    assert set(clinical["centre"]) == {"CHUS", "HGJ", "CHUM", "HMR"}
    assert (clinical["centre"] == clinical["sheet_centre"]).all()


def test_pfs_is_first_of_component_events(clinical):
    """A patient with any of LR / DM / death is a PFS event, and PFS <= OS."""
    any_component = (clinical[["LR_event", "DM_event", "OS_event"]].sum(axis=1) > 0).astype(int)
    assert (clinical["PFS_event"] == any_component).all()
    assert (clinical["PFS_time"] <= clinical["OS_time"] + 1e-9).all()
    assert (clinical["PFS_time"] > 0).all()


def test_times_are_in_months(clinical):
    # median follow-up in Vallieres et al. is ~43 months; days would be ~1300
    assert 20 < clinical["PFS_time"].median() < 80


def test_clinical_covariates_are_categorical_at_pengs_granularity(clinical):
    assert set(clinical["T_stage"]) <= {"T1", "T2", "T3", "T4", "TX", "Unknown"}
    assert set(clinical["N_stage"]) <= {"N0", "N1", "N2", "N3", "NX", "Unknown"}
    assert clinical["age"].between(18, 100).all()


def test_gtv_names_cover_every_patient(clinical):
    from mlfgnn.data.hnpc import load_gtv_names

    names = load_gtv_names(CONTOURS)
    missing = set(clinical["patient_id"]) - set(names)
    assert not missing, f"no GTV names for {sorted(missing)[:5]}"
    assert all(v["primary"] for v in names.values()), "a patient has no primary GTV name"


def test_encoding_uses_training_vocabulary_only(clinical):
    from mlfgnn.data.clinical import encode_clinical

    train = clinical[clinical["centre"] != "HMR"]
    x, names = encode_clinical(clinical, ["age", "T_stage", "N_stage", "site"], reference=train)
    assert x.shape[0] == len(clinical)
    assert "age" in names and any(n.startswith("site=") for n in names)
    assert np.isfinite(x).all()
