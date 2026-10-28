"""ABIDE PCP loading and connectivity-matrix construction.

Kept free of network code so it can be unit-tested on synthetic signals.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# PCP quality-control columns; nilearn's quality_checked filter drops any "fail".
QC_COLUMNS = [
    "qc_rater_1",
    "qc_anat_rater_2",
    "qc_func_rater_2",
    "qc_anat_rater_3",
    "qc_func_rater_3",
]


def select_subjects(pheno: pd.DataFrame, quality_checked: bool = True) -> pd.DataFrame:
    """Subjects with preprocessed files, optionally without any QC fail."""
    keep = pheno["FILE_ID"].astype(str) != "no_filename"
    if quality_checked:
        for col in QC_COLUMNS:
            if col in pheno.columns:
                keep &= pheno[col].astype(str).str.lower() != "fail"
    return pheno.loc[keep].reset_index(drop=True)


def roi_file_url(base_url: str, pipeline: str, strategy: str, atlas: str, file_id: str) -> str:
    return f"{base_url}/Outputs/{pipeline}/{strategy}/{atlas}/{file_id}_{atlas}.1D"


def load_roi_timeseries(path: Path) -> np.ndarray:
    """PCP .1D file -> array (timepoints, rois). The first line is a '#' header."""
    return np.loadtxt(path, comments="#", ndmin=2)


def connectivity(ts: np.ndarray, fisher_z: bool = True, clip_r: float = 0.999999) -> tuple[np.ndarray, int]:
    """Pearson correlation between ROI time series.

    ROIs with zero variance (often outside the field of view) get zero
    connectivity instead of NaN; their count is returned so it can be reported.
    Diagonal is set to 0.
    """
    ts = np.asarray(ts, dtype=np.float64)
    centred = ts - ts.mean(axis=0, keepdims=True)
    sd = centred.std(axis=0)
    dead = sd < 1e-8
    z = np.zeros_like(centred)
    z[:, ~dead] = centred[:, ~dead] / sd[~dead]
    r = (z.T @ z) / ts.shape[0]
    r = np.clip(r, -clip_r, clip_r)
    np.fill_diagonal(r, 0.0)
    out = np.arctanh(r) if fisher_z else r
    out[dead, :] = 0.0
    out[:, dead] = 0.0
    return out.astype(np.float32), int(dead.sum())
