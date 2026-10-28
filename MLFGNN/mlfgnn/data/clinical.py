"""Clinical / outcome spreadsheet -> tidy survival table.

Head-Neck-PET-CT ships one spreadsheet ("INFOclinical_HN_Version2_30may2018")
whose column names are not machine-friendly and whose outcome coding is split
across several columns. This module normalises it to the four columns the rest
of the pipeline expects -- ``patient_id``, ``centre``, ``PFS_time``,
``PFS_event`` -- plus the four clinical covariates Peng et al. feed to the Cox
head (age, T_stage, N_stage, site).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .centres import centre_of

logger = logging.getLogger(__name__)

#: Canonical name -> candidate spreadsheet headers (matched case-insensitively,
#: whitespace/punctuation-insensitive). Extend rather than rename.
COLUMN_ALIASES: dict[str, list[str]] = {
    "patient_id": ["patientid", "patient", "id", "subjectid", "name"],
    "age": ["age", "ageatdiagnosis", "agediag"],
    "sex": ["sex", "gender"],
    "T_stage": ["tstage", "t", "ajcct"],
    "N_stage": ["nstage", "n", "ajccn"],
    "M_stage": ["mstage", "m"],
    "TNM_stage": ["tnmstage", "stage", "ajccstage", "overallstage"],
    "site": ["site", "primarysite", "tumoursite", "tumorsite", "location"],
    "therapy": ["therapy", "treatment", "treatmentmodality"],
    # outcome
    "OS_time": ["timetodeath", "ostime", "survivaltime", "followupdays"],
    "OS_event": ["death", "deceased", "osevent", "vitalstatus"],
    "PFS_time": ["timetopfs", "pfstime", "progressionfreesurvival", "timetoevent"],
    "PFS_event": ["progression", "pfsevent", "event"],
    "locoregional_time": ["timetolocoregional", "timetolocalrecurrence"],
    "locoregional_event": ["locoregional", "localrecurrence", "recurrence"],
    "distant_time": ["timetodistant", "timetometastasis"],
    "distant_event": ["distant", "metastasis", "distantmetastasis"],
}

#: Site labels are free text across centres; collapse to the six groups used in
#: Peng et al. Table 1 so the categorical covariate is comparable.
SITE_GROUPS: dict[str, list[str]] = {
    "Nasopharynx": ["nasoph", "npc"],
    "Oropharynx": ["oroph", "tonsil", "base of tongue", "bot", "opc"],
    "Hypopharynx": ["hypoph", "piriform", "pyriform"],
    "Larynx": ["laryn", "glottic", "supraglottic"],
    "Oral cavity": ["oral", "tongue", "floor of mouth", "buccal"],
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map canonical names to the actual columns present."""
    lookup = {_norm(c): c for c in df.columns}
    resolved: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in [_norm(canonical), *aliases]:
            if alias in lookup:
                resolved[canonical] = lookup[alias]
                break
    return resolved


def _group_site(value: object) -> str:
    text = str(value).lower()
    for group, keys in SITE_GROUPS.items():
        if any(k in text for k in keys):
            return group
    return "Unknown"


def _coerce_event(series: pd.Series) -> pd.Series:
    """Map yes/no/alive/dead/0/1 style columns to {0, 1} integers."""
    truthy = {"1", "yes", "y", "true", "dead", "deceased", "event", "progression"}
    falsy = {"0", "no", "n", "false", "alive", "censored", "ned"}

    def _one(v: object) -> float:
        s = str(v).strip().lower()
        if s in truthy:
            return 1.0
        if s in falsy:
            return 0.0
        try:
            return float(s)
        except ValueError:
            return np.nan

    return series.map(_one)


def load_clinical(
    path: str | Path,
    sheet: str | int = 0,
    time_unit: str = "months",
) -> pd.DataFrame:
    """Read the outcome spreadsheet and return the tidy survival table.

    Parameters
    ----------
    time_unit:
        Unit of the *source* time columns. Everything downstream (including the
        ``eval_times`` in the experiment config) is in months, so ``"days"``
        triggers a /30.44 conversion.
    """
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=sheet)
    else:
        df = pd.read_csv(path)

    cols = _resolve_columns(df)
    if "patient_id" not in cols:
        raise KeyError(
            f"could not find a patient-ID column in {path.name}; "
            f"columns were {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["patient_id"] = df[cols["patient_id"]].astype(str).str.strip()
    out["centre"] = [centre_of(p, strict=False) for p in out["patient_id"]]

    for key in ("age", "sex", "T_stage", "N_stage", "M_stage", "TNM_stage", "therapy"):
        if key in cols:
            out[key] = df[cols[key]]

    out["site"] = (
        df[cols["site"]].map(_group_site) if "site" in cols else "Unknown"
    )

    # --- outcomes -----------------------------------------------------------
    scale = 1.0 / 30.4375 if time_unit == "days" else 1.0
    for endpoint in ("PFS", "OS"):
        t_key, e_key = f"{endpoint}_time", f"{endpoint}_event"
        if t_key in cols:
            out[t_key] = pd.to_numeric(df[cols[t_key]], errors="coerce") * scale
        if e_key in cols:
            out[e_key] = _coerce_event(df[cols[e_key]])

    if "PFS_time" not in out.columns:
        out = _derive_pfs(df, cols, out, scale)

    n_before = len(out)
    out = out.dropna(subset=["PFS_time", "PFS_event"])
    out = out[out["PFS_time"] > 0]
    if len(out) < n_before:
        logger.warning(
            "dropped %d/%d patients with missing or non-positive PFS",
            n_before - len(out),
            n_before,
        )
    out["PFS_event"] = out["PFS_event"].astype(int)
    return out.reset_index(drop=True)


def _derive_pfs(
    df: pd.DataFrame, cols: dict[str, str], out: pd.DataFrame, scale: float
) -> pd.DataFrame:
    """Build PFS from component endpoints when no PFS column exists.

    PFS = time to the first of {locoregional recurrence, distant metastasis,
    death}; censored at last follow-up otherwise. This is the definition in
    Peng et al. Section 4.1.
    """
    parts: list[tuple[pd.Series, pd.Series]] = []
    for t_key, e_key in (
        ("locoregional_time", "locoregional_event"),
        ("distant_time", "distant_event"),
        ("OS_time", "OS_event"),
    ):
        if t_key in cols and e_key in cols:
            t = pd.to_numeric(df[cols[t_key]], errors="coerce") * scale
            e = _coerce_event(df[cols[e_key]])
            parts.append((t, e))

    if not parts:
        raise KeyError(
            "no PFS column and no component endpoints (locoregional / distant / "
            "OS) to derive it from -- check the spreadsheet."
        )

    times = pd.concat([t for t, _ in parts], axis=1)
    events = pd.concat([e for _, e in parts], axis=1)

    event_times = times.where(events == 1)
    first_event = event_times.min(axis=1)
    last_followup = times.max(axis=1)

    out["PFS_event"] = first_event.notna().astype(float)
    out["PFS_time"] = first_event.fillna(last_followup)
    logger.info("derived PFS from %d component endpoints", len(parts))
    return out


def encode_clinical(
    df: pd.DataFrame,
    clinical_cols: list[str],
    reference: pd.DataFrame | None = None,
) -> tuple[np.ndarray, list[str]]:
    """One-hot / numeric encoding of the clinical covariates.

    ``reference`` supplies the category vocabulary and the standardisation
    statistics. Pass the TRAINING rows there so that a held-out centre with an
    unseen category (or a shifted age distribution) cannot leak into the
    encoding -- an unseen category simply lands in the all-zeros row.
    """
    ref = df if reference is None else reference
    blocks, names = [], []

    for col in clinical_cols:
        if col not in df.columns:
            logger.warning("clinical column %r absent -- skipped", col)
            continue
        s, s_ref = df[col], ref[col]
        if pd.api.types.is_numeric_dtype(s_ref):
            mu = float(np.nanmean(pd.to_numeric(s_ref, errors="coerce")))
            sd = float(np.nanstd(pd.to_numeric(s_ref, errors="coerce"))) or 1.0
            v = pd.to_numeric(s, errors="coerce").fillna(mu).to_numpy(float)
            blocks.append(((v - mu) / sd)[:, None])
            names.append(col)
        else:
            cats = sorted(pd.Series(s_ref.astype(str)).dropna().unique())
            oh = np.zeros((len(s), len(cats)), dtype=float)
            index = {c: i for i, c in enumerate(cats)}
            for row, value in enumerate(s.astype(str)):
                j = index.get(value)
                if j is not None:
                    oh[row, j] = 1.0
            blocks.append(oh)
            names += [f"{col}={c}" for c in cats]

    if not blocks:
        return np.zeros((len(df), 0)), []
    return np.hstack(blocks), names
