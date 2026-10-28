"""Build Fisher-z connectivity matrices for each atlas and write a cohort summary.

    .venv\\Scripts\\python.exe scripts\\02_build_connectomes.py

Output per atlas: data/processed/abide_<atlas>.npz with
    fc (N, R, R) float32, sub_id, file_id, site, dx (1 = ASD, 2 = control),
    age, sex (1 = M, 2 = F), mean_fd, n_timepoints, n_dead_rois
plus data/processed/cohort_summary.csv (subjects per site and group).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cxb.data import connectivity, load_roi_timeseries, select_subjects  # noqa: E402


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "data.yaml"))
    a, c = cfg["abide"], cfg["connectome"]
    raw = ROOT / cfg["paths"]["raw"]
    out = ROOT / cfg["paths"]["processed"]
    out.mkdir(parents=True, exist_ok=True)

    pheno = select_subjects(pd.read_csv(raw / a["phenotype_file"]), a["quality_checked"])
    kept_by_atlas = {}

    for atlas in a["atlases"]:
        folder = raw / a["pipeline"] / a["strategy"] / atlas
        fcs, rows, dropped = [], [], []
        for _, p in pheno.iterrows():
            path = folder / f"{p.FILE_ID}_{atlas}.1D"
            if not path.exists():
                dropped.append((p.FILE_ID, "missing file"))
                continue
            ts = load_roi_timeseries(path)
            if ts.shape[0] < c["min_timepoints"]:
                dropped.append((p.FILE_ID, f"{ts.shape[0]} timepoints"))
                continue
            fc, n_dead = connectivity(ts, c["fisher_z"], c["clip_r"])
            fcs.append(fc)
            rows.append(dict(sub_id=int(p.SUB_ID), file_id=p.FILE_ID, site=p.SITE_ID,
                             dx=int(p.DX_GROUP), age=float(p.AGE_AT_SCAN), sex=int(p.SEX),
                             mean_fd=float(p.get("func_mean_fd", np.nan)),
                             n_timepoints=ts.shape[0], n_dead_rois=n_dead))
        meta = pd.DataFrame(rows)
        fc = np.stack(fcs)
        np.savez_compressed(out / f"abide_{atlas}.npz", fc=fc,
                            **{k: meta[k].to_numpy() for k in meta.columns})
        kept_by_atlas[atlas] = meta
        print(f"{atlas}: {fc.shape[0]} subjects x {fc.shape[1]} ROIs; dropped {len(dropped)}; "
              f"subjects with >=1 dead ROI: {(meta.n_dead_rois > 0).sum()} "
              f"(max {meta.n_dead_rois.max()})")
        for fid, why in dropped[:10]:
            print(f"   dropped {fid}: {why}")

    main_meta = kept_by_atlas[a["main_atlas"]]
    summary = (main_meta.assign(group=main_meta.dx.map({1: "ASD", 2: "control"}))
               .pivot_table(index="site", columns="group", values="sub_id", aggfunc="count", fill_value=0))
    summary["total"] = summary.sum(axis=1)
    summary.loc["ALL"] = summary.sum()
    summary.to_csv(out / "cohort_summary.csv")
    print(f"\nCohort ({a['main_atlas']}):\n{summary.to_string()}")


if __name__ == "__main__":
    main()
