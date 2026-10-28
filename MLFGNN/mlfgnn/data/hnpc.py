"""Head-Neck-PET-CT (Vallieres et al., TCIA) -- the real cohort.

The collection ships two spreadsheets, both public on the collection page:

* ``INFOclinical_HN_Version2_30may2018.xlsx`` -- one sheet per centre
  (HGJ, CHUS, HMR, CHUM) plus an ``Excluded`` sheet. Outcome is coded as three
  event flags (Locoregional, Distant, Death) with matching
  "Time - diagnosis to X (days)" columns and a last-follow-up time.
* ``INFO_GTVcontours_HN.xlsx`` -- the ROI names of the GTV-primary and
  GTV-node contours in each patient's RTSTRUCT, which are not standardised
  across centres.

This module turns the first into the four columns the pipeline needs
(``patient_id``, ``centre``, ``PFS_time``, ``PFS_event``) plus the clinical
covariates, and the second into a per-patient ROI lookup for the mask step.

PFS follows Peng et al. Section 4.1: time to the first of locoregional
recurrence, distant metastasis or death; censored at last follow-up. Times are
converted from days to months so the 12/24/60-month evaluation horizons in the
experiment config apply unchanged.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .centres import centre_of

logger = logging.getLogger(__name__)

CENTRE_SHEETS = ("HGJ", "CHUS", "HMR", "CHUM")
DAYS_PER_MONTH = 365.25 / 12


def _norm(name: object) -> str:
    """Collapse the spreadsheet's mojibake / en-dash headers to a comparable key."""
    s = str(name).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()


def _find(columns, *needles: str) -> str | None:
    """First column whose normalised name contains every needle."""
    for c in columns:
        key = _norm(c)
        if all(n in key for n in needles):
            return c
    return None


def _site_group(value: object) -> str:
    text = str(value).lower()
    for group, keys in (
        ("Nasopharynx", ("nasoph",)),
        ("Oropharynx", ("oroph", "tonsil", "base of tongue")),
        ("Hypopharynx", ("hypoph", "piriform", "pyriform")),
        ("Larynx", ("laryn", "glott")),
        ("Oral cavity", ("oral", "tongue", "floor", "buccal", "gum", "palate")),
    ):
        if any(k in text for k in keys):
            return group
    return "Unknown"


def _stage_group(value: object, prefix: str) -> str:
    """Collapse sub-stages (T4a/T4b, N2a/N2b/N2c) to the level Peng et al. use."""
    s = str(value).strip().upper()
    m = re.match(prefix + r"\s*(\d|X|IS)", s)
    return f"{prefix}{m.group(1)}" if m else "Unknown"


def _num(series: pd.Series | None) -> pd.Series | None:
    return None if series is None else pd.to_numeric(series, errors="coerce")


def load_hnpc_clinical(path: str | Path) -> pd.DataFrame:
    """Read all four centre sheets into one tidy survival table."""
    path = Path(path)
    book = pd.ExcelFile(path)
    frames = []

    for sheet in CENTRE_SHEETS:
        if sheet not in book.sheet_names:
            logger.warning("sheet %s missing from %s", sheet, path.name)
            continue
        raw = book.parse(sheet)
        cols = list(raw.columns)

        pid_col = _find(cols, "patient")
        fu_col = _find(cols, "last follow")
        lr_t = _find(cols, "to lr")
        dm_t = _find(cols, "to dm")
        death_t = _find(cols, "to death")
        lr_e = _find(cols, "locoregional")
        dm_e = _find(cols, "distant")
        death_e = _find(cols, "death")
        missing = [
            n for n, c in (
                ("patient", pid_col), ("last follow-up", fu_col),
                ("Locoregional", lr_e), ("Distant", dm_e), ("Death", death_e),
            ) if c is None
        ]
        if missing:
            raise KeyError(f"sheet {sheet}: cannot locate columns {missing}; headers were {cols}")

        ids = raw[pid_col].astype(str).str.strip()
        keep = ids.str.match(r"^HN-", na=False)
        raw = raw.loc[keep].reset_index(drop=True)
        ids = ids.loc[keep].reset_index(drop=True)

        df = pd.DataFrame({"patient_id": ids})
        df["centre"] = [centre_of(p) for p in df["patient_id"]]
        df["sheet_centre"] = sheet

        df["age"] = _num(raw[_find(cols, "age")])
        df["sex"] = raw[_find(cols, "sex")].astype(str).str.strip().str.upper().str[0]
        df["site"] = raw[_find(cols, "primary site")].map(_site_group)
        df["T_stage"] = raw[_find(cols, "t stage")].map(lambda v: _stage_group(v, "T"))
        df["N_stage"] = raw[_find(cols, "n stage")].map(lambda v: _stage_group(v, "N"))
        df["M_stage"] = raw[_find(cols, "m stage")].map(lambda v: _stage_group(v, "M"))
        tnm = _find(cols, "tnm")
        df["TNM_stage"] = (
            raw[tnm].astype(str).str.replace("stage", "", case=False).str.strip() if tnm else "Unknown"
        )
        hpv = _find(cols, "hpv")
        df["HPV"] = raw[hpv].astype(str).str.strip() if hpv else "NA"
        therapy = _find(cols, "therapy")
        df["therapy"] = raw[therapy].astype(str).str.strip() if therapy else "Unknown"
        surg = _find(cols, "surgery")
        df["surgery"] = _num(raw[surg]) if surg else np.nan

        # --- outcomes (days) --------------------------------------------------
        fu = _num(raw[fu_col])
        flags = {
            "LR": (_num(raw[lr_e]), _num(raw[lr_t]) if lr_t else None),
            "DM": (_num(raw[dm_e]), _num(raw[dm_t]) if dm_t else None),
            "Death": (_num(raw[death_e]), _num(raw[death_t]) if death_t else None),
        }

        first = pd.Series(np.nan, index=raw.index, dtype=float)
        any_event = pd.Series(False, index=raw.index)
        for _name, (flag, t) in flags.items():
            hit = flag.fillna(0) == 1
            any_event = any_event | hit
            t_hit = (t if t is not None else fu).where(hit)
            # A flagged event with no time recorded is placed at last follow-up.
            t_hit = t_hit.fillna(fu.where(hit))
            first = pd.concat([first, t_hit], axis=1).min(axis=1)

        df["PFS_event"] = any_event.astype(int).to_numpy()
        df["PFS_time_days"] = np.where(any_event, first, fu)

        death_flag, death_time = flags["Death"]
        df["OS_event"] = death_flag.fillna(0).astype(int).to_numpy()
        os_t = (death_time if death_time is not None else fu).fillna(fu)
        df["OS_time_days"] = np.where(df["OS_event"] == 1, os_t, fu)
        df["LR_event"] = flags["LR"][0].fillna(0).astype(int).to_numpy()
        df["DM_event"] = flags["DM"][0].fillna(0).astype(int).to_numpy()
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["PFS_time"] = out["PFS_time_days"] / DAYS_PER_MONTH
    out["OS_time"] = out["OS_time_days"] / DAYS_PER_MONTH

    n0 = len(out)
    out = out[out["PFS_time"] > 0].dropna(subset=["PFS_time"]).reset_index(drop=True)
    if len(out) < n0:
        logger.warning("dropped %d patients with missing/non-positive follow-up", n0 - len(out))

    mismatch = out[out["centre"] != out["sheet_centre"]]
    if len(mismatch):
        logger.warning("%d patients whose ID prefix disagrees with their sheet", len(mismatch))

    if "Excluded" in book.sheet_names:
        exc = book.parse("Excluded")
        logger.info("Excluded sheet lists %d rows (kept out by the data providers)", len(exc))

    logger.info(
        "Head-Neck-PET-CT clinical: %d patients, %d centres, PFS events %.0f%%, "
        "median follow-up %.0f months",
        len(out), out["centre"].nunique(), 100 * out["PFS_event"].mean(), out["PFS_time"].median(),
    )
    return out


def _split_names(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    parts = re.split(r"[,;/]", str(value))
    return [s.strip() for s in parts if s.strip() and s.strip().lower() != "nan"]


def load_gtv_names(path: str | Path) -> dict[str, dict[str, list[str]]]:
    """``{patient_id: {"primary": [...], "nodes": [...]}}`` ROI names per patient."""
    book = pd.ExcelFile(Path(path))
    lookup: dict[str, dict[str, list[str]]] = {}
    for sheet in book.sheet_names:
        df = book.parse(sheet)
        cols = list(df.columns)
        pid_col = _find(cols, "patient")
        prim = _find(cols, "primary")
        nodes = _find(cols, "lymph") or _find(cols, "node")
        if pid_col is None:
            continue
        for _, row in df.iterrows():
            pid = str(row[pid_col]).strip()
            if not pid.startswith("HN-"):
                continue
            lookup[pid] = {
                "primary": _split_names(row[prim]) if prim else [],
                "nodes": _split_names(row[nodes]) if nodes else [],
            }
    logger.info("GTV contour names for %d patients", len(lookup))
    return lookup
