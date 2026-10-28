"""Stage 1+2: Head-Neck-PET-CT DICOM -> NIfTI -> radiomics feature table.

Run under the Python 3.9 radiomics environment -- PyRadiomics has no wheel for
newer interpreters:

    .venv-radiomics\\Scripts\\python.exe scripts\\01_extract_features.py --inspect
    .venv-radiomics\\Scripts\\python.exe scripts\\01_extract_features.py --convert-only
    .venv-radiomics\\Scripts\\python.exe scripts\\01_extract_features.py

Inputs (all produced by earlier stages or shipped with the collection):

    data/raw/Head-Neck-PET-CT/<patient>/<Modality>/<SeriesUID>/   01a_download_hnpc.py
    data/raw/INFOclinical_HN_Version2_30may2018.xlsx               collection page
    data/raw/INFO_GTVcontours_HN.xlsx                              collection page

``--inspect`` applies the PET -> CT -> RTSTRUCT selection rule to every
downloaded patient and writes a table of what would be used, without
converting anything. Run it first: it is the fastest way to find the patients
whose contours live on the planning CT or whose PET lacks SUV tags.

Extraction is CPU-bound (hours for 298 patients with LoG + wavelet filters).
Per-patient failures are logged and skipped, never fatal.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlfgnn.data.hnpc import load_gtv_names, load_hnpc_clinical  # noqa: E402
from mlfgnn.data.tcia import convert_patient, select_series  # noqa: E402
from mlfgnn.features.extract import extract_cohort  # noqa: E402

logger = logging.getLogger("extract")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dicom-root", default="data/raw/Head-Neck-PET-CT")
    p.add_argument("--clinical", default="data/raw/INFOclinical_HN_Version2_30may2018.xlsx")
    p.add_argument("--contours", default="data/raw/INFO_GTVcontours_HN.xlsx")
    p.add_argument("--volumes-dir", default="data/interim/volumes")
    p.add_argument("--out", default="data/processed/features.csv")
    p.add_argument("--config", default="configs/pyradiomics.yaml")
    p.add_argument("--patients", nargs="*", default=None, help="restrict to these IDs")
    p.add_argument("--inspect", action="store_true", help="report series selection only")
    p.add_argument("--convert-only", action="store_true")
    p.add_argument("--skip-convert", action="store_true", help="volumes already converted")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")

    clinical = load_hnpc_clinical(args.clinical)
    gtv = load_gtv_names(args.contours)
    if args.patients:
        clinical = clinical[clinical["patient_id"].isin(args.patients)].reset_index(drop=True)

    root = Path(args.dicom_root)
    available = {p.name for p in root.iterdir() if p.is_dir()} if root.exists() else set()
    have = clinical["patient_id"].isin(available)
    logger.info("%d/%d clinical patients have a DICOM folder under %s", have.sum(), len(clinical), root)
    if have.sum() == 0:
        raise SystemExit("no patient folders found -- run scripts/01a_download_hnpc.py first")
    clinical = clinical[have].reset_index(drop=True)

    # --- inspection ------------------------------------------------------------
    if args.inspect:
        rows = []
        for pid in clinical["patient_id"]:
            try:
                s = select_series(root / pid, gtv.get(pid))
                rows.append({"patient_id": pid, "centre": pid.split("-")[1], "contours_on": s.contours_on,
                             "ct_slices": s.ct.n_files, "pet_slices": s.pet.n_files,
                             "primary_rois": "|".join(s.primary_rois), "node_rois": "|".join(s.node_rois),
                             "reg": bool(s.reg), "notes": "; ".join(s.notes), "error": ""})
            except Exception as exc:  # noqa: BLE001
                rows.append({"patient_id": pid, "centre": pid.split("-")[1], "error": str(exc)})
        table = pd.DataFrame(rows)
        out = Path(args.volumes_dir).parent / "series_selection.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        print(table.groupby("centre").agg(n=("patient_id", "size"),
                                          on_petct=("contours_on", lambda s: (s == "petct").sum()),
                                          on_planning=("contours_on", lambda s: (s == "planning").sum()),
                                          errors=("error", lambda s: (s != "").sum())).to_string())
        print(f"\nwrote {out}")
        return

    # --- conversion ---------------------------------------------------------------
    volumes_dir = Path(args.volumes_dir)
    if not args.skip_convert:
        ok, failed = 0, []
        for pid in clinical["patient_id"]:
            if (volumes_dir / pid / "GTV_on_PET.nii.gz").exists():
                ok += 1
                continue
            try:
                vol = convert_patient(root / pid, volumes_dir, gtv.get(pid))
                ok += 1
                if not vol.suv_ok:
                    logger.warning("%s: PET not SUV-converted", pid)
            except Exception as exc:  # noqa: BLE001
                logger.error("conversion failed for %s: %s", pid, exc)
                failed.append({"patient_id": pid, "error": str(exc)})
        logger.info("converted %d/%d patients (%d failures)", ok, len(clinical), len(failed))
        if failed:
            pd.DataFrame(failed).to_csv(volumes_dir.parent / "conversion_failures.csv", index=False)

    if args.convert_only:
        return

    # --- radiomics ----------------------------------------------------------------
    df = extract_cohort(volumes_dir=volumes_dir, clinical=clinical, config_path=args.config, out_path=args.out)
    print(f"\nwrote {args.out}: {df.shape[0]} patients x {df.shape[1]} columns")
    print(df.groupby("centre").agg(n=("patient_id", "size"), events=("PFS_event", "sum")).to_string())


if __name__ == "__main__":
    main()
