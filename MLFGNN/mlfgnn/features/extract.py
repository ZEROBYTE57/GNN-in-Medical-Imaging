"""Stage 2: PyRadiomics extraction driven by the pinned config.

Produces one wide table with ``PET_*`` and ``CT_*`` columns. The digest of
``configs/pyradiomics.yaml`` and the PyRadiomics version are written into
``df.attrs`` and into a sidecar JSON, so a feature table can never be silently
mixed with one extracted under different settings.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from ..config import file_digest

logger = logging.getLogger(__name__)


def _extractor(config_path: Path):
    try:
        import radiomics
        from radiomics import featureextractor
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyRadiomics is required for feature extraction -- run this stage under "
            ".venv-radiomics (see requirements-radiomics.txt)."
        ) from exc
    # PyRadiomics logs one "GLCM is symmetrical" line per image per filter; at
    # 298 patients x 14 image types that is thousands of lines hiding real
    # warnings. Errors still surface.
    radiomics.logger.setLevel(logging.ERROR)
    return featureextractor.RadiomicsFeatureExtractor(str(config_path))


def extract_one(
    extractor,
    image_path: Path,
    mask_path: Path,
    prefix: str,
) -> dict[str, float]:
    """Extract features for one image/mask pair, prefixed by modality."""
    result = extractor.execute(str(image_path), str(mask_path))
    out: dict[str, float] = {}
    for key, value in result.items():
        if key.startswith("diagnostics_"):
            continue
        try:
            out[f"{prefix}{key}"] = float(value)
        except (TypeError, ValueError):
            continue  # array-valued or non-numeric diagnostics
    return out


def extract_cohort(
    volumes_dir: str | Path,
    clinical: pd.DataFrame,
    config_path: str | Path = "configs/pyradiomics.yaml",
    pet_name: str = "PET_SUV.nii.gz",
    ct_name: str = "CT.nii.gz",
    pet_mask_name: str = "GTV_on_PET.nii.gz",
    ct_mask_name: str = "GTV_on_CT.nii.gz",
    out_path: str | Path | None = None,
) -> pd.DataFrame:
    """Extract PET and CT radiomics for every patient in ``clinical``.

    Patients whose volumes are missing or whose extraction raises are logged and
    dropped -- extraction failures are common on TCIA (empty ROIs, single-slice
    contours) and must not abort a multi-hour run.
    """
    volumes_dir, config_path = Path(volumes_dir), Path(config_path)
    extractor = _extractor(config_path)

    rows, failures = [], []
    for _, patient in clinical.iterrows():
        pid = patient["patient_id"]
        pdir = volumes_dir / pid
        try:
            row = {"patient_id": pid}
            row.update(extract_one(extractor, pdir / pet_name, pdir / pet_mask_name, "PET_"))
            row.update(extract_one(extractor, pdir / ct_name, pdir / ct_mask_name, "CT_"))
            rows.append(row)
        except Exception as exc:  # noqa: BLE001 - per-patient isolation is the point
            logger.error("extraction failed for %s: %s", pid, exc)
            failures.append({"patient_id": pid, "error": str(exc)})

    features = pd.DataFrame(rows)
    merged = clinical.merge(features, on="patient_id", how="inner")

    n_pet = sum(c.startswith("PET_") for c in merged.columns)
    n_ct = sum(c.startswith("CT_") for c in merged.columns)
    logger.info(
        "extracted %d PET + %d CT features for %d/%d patients (%d failures)",
        n_pet, n_ct, len(merged), len(clinical), len(failures),
    )

    provenance = {
        "pyradiomics_config": str(config_path),
        "pyradiomics_config_sha256_12": file_digest(config_path),
        "pyradiomics_version": _radiomics_version(),
        "n_pet_features": n_pet,
        "n_ct_features": n_ct,
        "n_patients": len(merged),
        "failures": failures,
    }
    merged.attrs.update(provenance)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_table(merged, out_path)
        with out_path.with_suffix(".provenance.json").open("w", encoding="utf-8") as fh:
            json.dump(provenance, fh, indent=2)
    return merged


def _radiomics_version() -> str:
    try:
        import radiomics

        return getattr(radiomics, "__version__", "unknown")
    except ImportError:
        return "not-installed"


def _write_table(df: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return
        except (ImportError, ValueError) as exc:
            logger.warning("parquet unavailable (%s); writing CSV instead", exc)
            path = path.with_suffix(".csv")
    df.to_csv(path, index=False)
