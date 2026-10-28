# ABIDE data summary (week 1)

Built 28 Oct 2026 by `scripts/01_download_abide.py` and `scripts/02_build_connectomes.py`.
Settings: `configs/data.yaml`; reasons: `decisions.md`.

## Counts

| Step | Subjects |
|---|---|
| Phenotype file | 1,112 |
| With preprocessed files and no QC fail | 873 (nilearn reports 871; small difference in the QC rule, to check) |
| After dropping scans under 100 timepoints | **848** (393 ASD, 455 controls, 19 sites) |

All 25 dropped subjects are the whole **OHSU** site (78 timepoints each). Open question: keep OHSU with a lower threshold?

## Per site (AAL; CC200 identical)

| Site | ASD | Control | Total |
|---|---|---|---|
| NYU | 74 | 98 | 172 |
| UM_1 | 36 | 52 | 88 |
| USM | 43 | 24 | 67 |
| UCLA_1 | 37 | 27 | 64 |
| PITT | 24 | 26 | 50 |
| MAX_MUN | 19 | 27 | 46 |
| TRINITY | 19 | 25 | 44 |
| YALE | 22 | 19 | 41 |
| UM_2 | 13 | 21 | 34 |
| KKI | 12 | 21 | 33 |
| LEUVEN_1 | 14 | 14 | 28 |
| LEUVEN_2 | 12 | 16 | 28 |
| OLIN | 14 | 14 | 28 |
| SDSU | 8 | 19 | 27 |
| SBL | 12 | 14 | 26 |
| STANFORD | 12 | 13 | 25 |
| UCLA_2 | 11 | 10 | 21 |
| CALTECH | 5 | 10 | 15 |
| CMU | 6 | 5 | 11 |
| **All** | **393** | **455** | **848** |

## Quality checks

| Check | AAL-116 | CC200 |
|---|---|---|
| Matrices finite and symmetric | Yes | Yes |
| Mean Fisher z (SD) | 0.324 (0.295) | 0.286 (0.275) |
| Subjects with at least one zero-signal ROI | 54 (SBL 19, PITT 16, others ≤ 5) | 15 |
| Most zero-signal ROIs in one subject | 12 | 12 |
| Timepoints | 116–316 | 116–316 |
| Mean framewise displacement: median; subjects > 0.2 mm | 0.079 mm; 90 | same |

Zero-signal ROIs are regions outside the scan's field of view (likely cerebellum for AAL). They are set to zero connectivity, not dropped.

## Files

- `data/processed/abide_rois_aal.npz`, `abide_rois_cc200.npz`: `fc` (N × R × R, float32) plus `sub_id, file_id, site, dx, age, sex, mean_fd, n_timepoints, n_dead_rois`.
- `data/processed/cohort_summary.csv`.
- Raw time series: `data/raw/abide/` (516 MB, not in git).
