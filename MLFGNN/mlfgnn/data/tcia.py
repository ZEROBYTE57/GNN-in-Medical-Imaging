"""Stage 1: Head-Neck-PET-CT DICOM -> analysis-ready NIfTI.

Runs under the Python 3.9 radiomics environment (``.venv-radiomics``): that is
the last interpreter with a PyRadiomics wheel, and SimpleITK / pydicom /
rt-utils live there too. The main environment never imports this module.

What the collection actually looks like after download
(``scripts/01a_download_hnpc.py`` lays series out as
``<patient>/<Modality>/<SeriesInstanceUID>/*.dcm``):

* every patient has one PT series;
* 206 of 298 patients have TWO CT series -- the low-dose CT of the PET/CT
  exam and the radiotherapy planning CT ("CT-sim");
* patients have two to ten RTSTRUCTs: the full RT plan structure set on the
  planning CT, and the GTV-only set used for radiomics. Which CT a structure
  set refers to is recorded in its ReferencedFrameOfReferenceSequence.

Peng et al. extract features from the PET and the CT *of the PET/CT exam*, with
the GTV (primary + involved nodes) as the VOI. So the selection rule is:

    PT  ->  the CT sharing its FrameOfReferenceUID (fallback: same study)
        ->  the RTSTRUCT referencing that CT and containing the GTV ROI names
            from INFO_GTVcontours_HN.xlsx

If the only GTV structure set sits on the planning CT, the DICOM REG object
(rigid 4x4 matrix) is used to carry the mask onto the PET/CT frame; patients
where that is impossible are reported, not silently dropped.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DICOM header inventory
# ---------------------------------------------------------------------------
@dataclass
class SeriesInfo:
    modality: str
    series_uid: str
    study_uid: str
    frame_of_ref: str
    path: Path
    n_files: int
    description: str = ""
    referenced_series: list[str] = field(default_factory=list)
    referenced_for: list[str] = field(default_factory=list)
    roi_names: list[str] = field(default_factory=list)


def _dicom_files(folder: Path) -> list[Path]:
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".dcm", ""})
    return [f for f in files if f.name != ".done"]


def _read_header(path: Path):
    import pydicom

    return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)


def inventory(patient_dir: Path) -> list[SeriesInfo]:
    """One SeriesInfo per series folder under a patient directory."""
    out: list[SeriesInfo] = []
    for mod_dir in sorted(p for p in patient_dir.iterdir() if p.is_dir()):
        for series_dir in sorted(p for p in mod_dir.iterdir() if p.is_dir()):
            files = _dicom_files(series_dir)
            if not files:
                continue
            ds = _read_header(files[0])
            info = SeriesInfo(
                modality=str(getattr(ds, "Modality", mod_dir.name)),
                series_uid=str(getattr(ds, "SeriesInstanceUID", series_dir.name)),
                study_uid=str(getattr(ds, "StudyInstanceUID", "")),
                frame_of_ref=str(getattr(ds, "FrameOfReferenceUID", "")),
                path=series_dir,
                n_files=len(files),
                description=str(getattr(ds, "SeriesDescription", "")),
            )
            if info.modality == "RTSTRUCT":
                _fill_rtstruct_refs(ds, info)
            out.append(info)
    return out


def _fill_rtstruct_refs(ds, info: SeriesInfo) -> None:
    try:
        for ref_for in ds.ReferencedFrameOfReferenceSequence:
            info.referenced_for.append(str(ref_for.FrameOfReferenceUID))
            for study in getattr(ref_for, "RTReferencedStudySequence", []):
                for series in getattr(study, "RTReferencedSeriesSequence", []):
                    info.referenced_series.append(str(series.SeriesInstanceUID))
    except AttributeError:
        pass
    try:
        info.roi_names = [str(r.ROIName) for r in ds.StructureSetROISequence]
    except AttributeError:
        pass


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
@dataclass
class Selection:
    patient_id: str
    pet: SeriesInfo
    ct: SeriesInfo
    rtstruct: SeriesInfo
    primary_rois: list[str]
    node_rois: list[str]
    contours_on: str            # "petct" | "planning"
    reg: SeriesInfo | None = None
    notes: list[str] = field(default_factory=list)


def _match_roi(names: list[str], wanted: list[str]) -> list[str]:
    """ROI names in ``names`` matching any of ``wanted`` (case/space-insensitive)."""
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())   # noqa: E731
    wanted_n = {norm(w) for w in wanted}
    return [n for n in names if norm(n) in wanted_n]


def select_series(patient_dir: Path, gtv_names: dict[str, list[str]] | None) -> Selection:
    """Apply the PET -> CT -> RTSTRUCT rule described in the module docstring."""
    pid = patient_dir.name
    series = inventory(patient_dir)
    pets = [s for s in series if s.modality == "PT"]
    cts = [s for s in series if s.modality == "CT"]
    rts = [s for s in series if s.modality == "RTSTRUCT"]
    regs = [s for s in series if s.modality == "REG"]
    if not pets or not cts or not rts:
        raise FileNotFoundError(f"{pid}: need PT, CT and RTSTRUCT; found "
                                f"{[s.modality for s in series]}")

    pet = max(pets, key=lambda s: s.n_files)
    notes: list[str] = []

    # --- CT of the PET/CT exam --------------------------------------------------
    same_for = [c for c in cts if c.frame_of_ref == pet.frame_of_ref]
    same_study = [c for c in cts if c.study_uid == pet.study_uid]
    if same_for:
        ct = max(same_for, key=lambda s: s.n_files)
    elif same_study:
        ct = max(same_study, key=lambda s: s.n_files)
        notes.append("CT chosen by study, not frame of reference")
    else:
        ct = max(cts, key=lambda s: s.n_files)
        notes.append("no CT shares PET study/frame -- took the largest CT")

    # --- RTSTRUCT with the GTV -----------------------------------------------------
    wanted_primary = (gtv_names or {}).get("primary", [])
    wanted_nodes = (gtv_names or {}).get("nodes", [])
    fallback_patterns = ["GTV", "GTVp", "GTV-1", "GTV_p", "GTV primary"]

    def score(rt: SeriesInfo) -> tuple[int, int, int]:
        on_ct = int(ct.series_uid in rt.referenced_series or ct.frame_of_ref in rt.referenced_for)
        prim = len(_match_roi(rt.roi_names, wanted_primary or fallback_patterns))
        nodes = len(_match_roi(rt.roi_names, wanted_nodes))
        return (on_ct, prim, nodes)

    ranked = sorted(rts, key=score, reverse=True)
    rt = ranked[0]
    on_ct, n_prim, _ = score(rt)
    if n_prim == 0:
        raise ValueError(f"{pid}: no RTSTRUCT contains a GTV ROI (wanted {wanted_primary or fallback_patterns}); "
                         f"available: {[r.roi_names for r in rts]}")

    contours_on = "petct" if on_ct else "planning"
    reg = None
    if not on_ct:
        # contours live on the planning CT: find a REG mapping its frame onto PET's
        for r in regs:
            reg = r
            break
        notes.append("GTV contours are on the planning CT" + (" -- REG available" if reg else " -- NO REG object"))

    return Selection(
        patient_id=pid, pet=pet, ct=ct, rtstruct=rt,
        primary_rois=_match_roi(rt.roi_names, wanted_primary or fallback_patterns),
        node_rois=_match_roi(rt.roi_names, wanted_nodes),
        contours_on=contours_on, reg=reg, notes=notes,
    )


# ---------------------------------------------------------------------------
# Image conversion
# ---------------------------------------------------------------------------
def _require_sitk():
    try:
        import SimpleITK as sitk
    except ImportError as exc:  # pragma: no cover
        raise ImportError("run stage 1 under .venv-radiomics (SimpleITK, pydicom, rt-utils)") from exc
    return sitk


def read_series(series_dir: Path):
    sitk = _require_sitk()
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(str(series_dir))
    if not files:
        raise FileNotFoundError(f"no readable DICOM series in {series_dir}")
    reader.SetFileNames(files)
    return reader.Execute()


def pet_to_suv_bw(series_dir: Path):
    """Body-weight SUV: activity [Bq/mL] x weight [g] / decay-corrected dose [Bq]."""
    sitk = _require_sitk()
    img = read_series(series_dir)
    ds = _read_header(_dicom_files(series_dir)[0])

    try:
        units = str(getattr(ds, "Units", "")).upper()
        weight_g = float(ds.PatientWeight) * 1000.0
        radio = ds.RadiopharmaceuticalInformationSequence[0]
        dose = float(radio.RadionuclideTotalDose)
        half_life = float(radio.RadionuclideHalfLife)
        start = str(getattr(radio, "RadiopharmaceuticalStartTime", None) or radio.RadiopharmaceuticalStartDateTime[8:])
        acq = str(getattr(ds, "SeriesTime", None) or ds.AcquisitionTime)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        logger.error("%s: PET SUV tags missing (%s) -- returning RAW intensities", series_dir.parent.parent.name, exc)
        return img, False

    if units == "GML":            # already SUV (g/mL)
        return img, True

    def _seconds(t: str) -> float:
        t = t.split("+")[0].split("-")[0].strip()
        return float(t[0:2]) * 3600 + float(t[2:4]) * 60 + float(t[4:] or 0)

    delay = _seconds(acq) - _seconds(start)
    if delay < 0:
        delay += 24 * 3600
    decayed = dose * (2.0 ** (-delay / half_life))
    arr = sitk.GetArrayFromImage(img).astype(np.float32) * (weight_g / decayed)
    suv = sitk.GetImageFromArray(arr)
    suv.CopyInformation(img)
    return suv, True


def rtstruct_mask(rt_dir: Path, ct_dir: Path, roi_names: list[str]) -> np.ndarray:
    """Union of the named ROIs as a (z, y, x) boolean array on the CT grid."""
    from rt_utils import RTStructBuilder

    rt_file = _dicom_files(rt_dir)[0]
    rt = RTStructBuilder.create_from(dicom_series_path=str(ct_dir), rt_struct_path=str(rt_file))
    mask = None
    for name in roi_names:
        m = rt.get_roi_mask_by_name(name)          # (y, x, z)
        mask = m if mask is None else (mask | m)
    if mask is None:
        raise ValueError(f"none of {roi_names} found in {rt_dir}")
    return np.transpose(mask, (2, 0, 1))


def reg_transform(reg_dir: Path | None):
    """Rigid transform from a DICOM REG object, or identity when absent."""
    sitk = _require_sitk()
    if reg_dir is None:
        return sitk.Transform(3, sitk.sitkIdentity)
    ds = _read_header(_dicom_files(reg_dir)[0])
    try:
        for reg in ds.RegistrationSequence:
            for mreg in reg.MatrixRegistrationSequence:
                for m in mreg.MatrixSequence:
                    mat = np.array([float(v) for v in m.FrameOfReferenceTransformationMatrix]).reshape(4, 4)
                    if not np.allclose(mat, np.eye(4)):
                        t = sitk.AffineTransform(3)
                        t.SetMatrix(mat[:3, :3].ravel().tolist())
                        t.SetTranslation(mat[:3, 3].tolist())
                        return t
    except AttributeError:
        pass
    logger.warning("REG %s had no usable matrix; using identity", reg_dir)
    return sitk.Transform(3, sitk.sitkIdentity)


def resample_mask(mask_img, reference, transform=None):
    sitk = _require_sitk()
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(reference)
    rs.SetInterpolator(sitk.sitkNearestNeighbor)
    rs.SetDefaultPixelValue(0)
    if transform is not None:
        rs.SetTransform(transform)
    return rs.Execute(mask_img)


@dataclass
class PatientVolumes:
    patient_id: str
    centre: str
    ct: Path
    pet: Path
    mask_ct: Path
    mask_pet: Path
    contours_on: str
    suv_ok: bool
    rois: list[str]


def convert_patient(patient_dir: Path, out_dir: Path, gtv_names: dict | None = None) -> PatientVolumes:
    """Select the PET/CT series, convert to NIfTI, write masks on both grids."""
    from .centres import centre_of

    sitk = _require_sitk()
    sel = select_series(patient_dir, gtv_names)
    for note in sel.notes:
        logger.info("%s: %s", sel.patient_id, note)

    ct = read_series(sel.ct.path)
    pet, suv_ok = pet_to_suv_bw(sel.pet.path)

    rois = sel.primary_rois + sel.node_rois
    if sel.contours_on == "petct":
        mask_arr = rtstruct_mask(sel.rtstruct.path, sel.ct.path, rois)
        mask_ct = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
        mask_ct.CopyInformation(ct)
    else:
        planning = next(c for c in inventory(patient_dir)
                        if c.modality == "CT" and (c.series_uid in sel.rtstruct.referenced_series
                                                   or c.frame_of_ref in sel.rtstruct.referenced_for))
        planning_img = read_series(planning.path)
        mask_arr = rtstruct_mask(sel.rtstruct.path, planning.path, rois)
        mask_plan = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
        mask_plan.CopyInformation(planning_img)
        mask_ct = resample_mask(mask_plan, ct, reg_transform(sel.reg.path if sel.reg else None))

    mask_pet = resample_mask(mask_ct, pet)
    if sitk.GetArrayViewFromImage(mask_pet).sum() == 0:
        raise ValueError(f"{sel.patient_id}: GTV mask is empty on the PET grid "
                         f"(contours on {sel.contours_on}); check registration")

    out = Path(out_dir) / sel.patient_id
    out.mkdir(parents=True, exist_ok=True)
    paths = PatientVolumes(
        patient_id=sel.patient_id, centre=centre_of(sel.patient_id),
        ct=out / "CT.nii.gz", pet=out / "PET_SUV.nii.gz",
        mask_ct=out / "GTV_on_CT.nii.gz", mask_pet=out / "GTV_on_PET.nii.gz",
        contours_on=sel.contours_on, suv_ok=suv_ok, rois=rois,
    )
    sitk.WriteImage(ct, str(paths.ct), True)
    sitk.WriteImage(pet, str(paths.pet), True)
    sitk.WriteImage(mask_ct, str(paths.mask_ct), True)
    sitk.WriteImage(mask_pet, str(paths.mask_pet), True)
    return paths
