# Decisions log

Fixed choices, with the date and the reason. Change a decision only by adding a new dated row.

| Date | Decision | Reason |
|---|---|---|
| 2026-10-28 | Dataset: ABIDE I Preprocessed Connectomes Project (public S3, no login) | Open, multi-site (17 sites before QC), used by BrainOOD, CI-GNN, BrainIB |
| 2026-10-28 | Pipeline: CPAC, band-pass filtered, no global signal regression (`filt_noglobal`) | Most common in brain-GNN papers; GSR distorts negative correlations |
| 2026-10-28 | Main atlas AAL-116; CC200 as sensitivity analysis | CI-GNN and BrainIB run with published settings |
| 2026-10-28 | Subjects: PCP quality-checked (any rater "fail" dropped); no motion cut-off | Standard nilearn subset; motion reported per subject (`mean_fd`) for later checks |
| 2026-10-28 | Connectivity: Pearson correlation, Fisher z, diagonal 0, dead ROIs set to 0 and counted | Standard; avoids NaN from ROIs outside the field of view |
| 2026-10-28 | Git: no automatic commits; the student handles git | Student's choice |
| 2026-10-28 | Names: benchmark **CXB** (Connectome eXplanation Benchmark), method **SiteSwap**; folder and package `cxb` | Student's choice; checked: no clash with published GNN explanation work |
