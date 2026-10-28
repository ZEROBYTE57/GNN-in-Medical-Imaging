# Centre-aware population graph construction for cross-institution HNC survival

Reference implementation for the dissertation problem statement: reproduce the
multi-level fusion GNN of **Peng et al., *Biomed. Signal Process. Control* 92
(2024) 106137**, change **exactly one component** — the population-graph edge
rule — and measure what that change does to cross-centre generalisation under a
strict leave-one-centre-out protocol, on the real TCIA Head-Neck-PET-CT cohort,
on the GPU.

## The claim

Peng et al. build a population graph by connecting any two patients whose fused
PET/CT similarity exceeds a fixed 0.2. Scanner and protocol effects make
same-centre patients look alike, so that rule connects patients mostly *within*
their own centre. Reported consequence: C-index **0.788** on held-out TCIA
cohorts against **0.674** on centres never present in the graph.

The proposed change is one function:

```python
centre_balanced_knn(X, centre, k)   # for patient i, the k most similar
                                    # patients from EVERY OTHER centre
```

| Arm | Rule | Isolates |
|---|---|---|
| `snf_threshold` | fused SNF similarity > 0.2 | **existing** — Peng et al., reproduced |
| `knn` | k most similar patients overall | control: "k-NN vs threshold" |
| `centre_balanced_knn` | k most similar **per other centre** | **novel** — "centre mixing" |

Features, GNN layers, MLP and Cox head are held identical across the three.

## The data

**No model in this repository has seen real patient data yet.** Every run so far
used `data/processed/features_synth.csv`, a simulated cohort that copies the
real centre sizes and encodes a site effect. Its numbers validate the pipeline
and mean nothing about the cohort.

The real cohort is TCIA **Head-Neck-PET-CT** (Vallières et al., Sci. Rep. 2017;
CC BY 3.0): 298 patients, 4 Québec centres, pre-treatment FDG-PET/CT with GTV
contours and outcomes. It is the 4-centre subset Peng et al. used for training.

| centre | patients | PFS events | median follow-up |
|---|---|---|---|
| CHUS | 101 | 29 | 40 mo |
| HGJ | 91 | 29 | 43 mo |
| CHUM | 65 | 9 | 39 mo |
| HMR | 41 | 22 | 35 mo |

(from `data/raw/INFOclinical_HN_Version2_30may2018.xlsx`, parsed by
`mlfgnn/data/hnpc.py`; CHUM's 9 events make it the noisiest held-out fold.)

### Getting it — the collection moved behind a login

As of September 2026 the collection is **not** anonymously downloadable: TCIA's
NBIA API answers "not in public domain", the Imaging Data Commons does not carry
it, and the CRDC index lists it under dbGaP accession **phs004225**. The
clinical spreadsheet, GTV-contour-name spreadsheet and series manifest are still
public and are already in `data/raw/`. For the images:

1. Sign in at <https://nci-crdc.datacommons.io> (Login.gov or eRA Commons).
2. Profile → *Create API key* → save as `credentials.json` in the repo root
   (git-ignored).
3. `python scripts/01a_download_hnpc.py` — 17 GB (CT + PT + RTSTRUCT), md5-verified,
   resumable, `--centres HMR` to start with the smallest centre.

If the portal says you lack access to phs004225, request it through dbGaP; the
data licence is unchanged. This is the one step that cannot be automated.

### What to do if access takes time

- **HECKTOR 2022** (7 centres, 524 training patients, PET/CT + GTV + RFS) is a
  larger multi-centre alternative with the same modality pair. Access is by
  request to the challenge organisers; the LOCO protocol here applies unchanged
  and the extension is one pattern in `mlfgnn/data/centres.py`.
- **Head-Neck-Radiomics-HN1** (MAASTRO, 137 patients, CT only) and
  **QIN-HEADNECK** (Iowa) are the extensions Peng et al. used; check their
  current hosting on the TCIA collection pages before planning around them.

## Two environments, one GPU

PyRadiomics' last Windows wheel is for Python 3.9; CUDA PyTorch and
scikit-survival want 3.11+. So stage 1 has its own interpreter and hands a CSV
to the rest:

```powershell
# main: graph construction, GNN, evaluation (Python 3.14, RTX 4060 / CUDA 13)
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install "torch==2.14.0+cu130" --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python.exe -m pip install --no-deps scikit-survival   # see requirements.txt

# radiomics: DICOM -> NIfTI -> features (Python 3.9)
py install 3.9
py -3.9 -m venv .venv-radiomics
.venv-radiomics\Scripts\python.exe -m pip install -r requirements-radiomics.txt
```

`--device auto` (the default) picks CUDA when present; the resolved device and
GPU name are written into every run's `config_used.json`.

## Pipeline

```
00  scripts/00_make_synthetic.py      simulated cohort (pipeline validation only)
01a scripts/01a_download_hnpc.py      CRDC download, manifest-driven, resumable
01  scripts/01_extract_features.py    [.venv-radiomics] --inspect, convert, extract
02  scripts/02_run_loco.py            [.venv] LOCO grid: 3 rules x k x 4 folds x 5 seeds + baselines
03  scripts/03_report.py              tables + paired bootstrap
04  scripts/04_analysis.py            novelty-vs-existing analysis -> <run>/analysis/ANALYSIS.md
```

Run `01_extract_features.py --inspect` first once images are present: it
applies the PET → CT → RTSTRUCT selection rule to every patient and reports who
has contours on the planning CT rather than the PET/CT, without converting
anything.

```powershell
.venv-radiomics\Scripts\python.exe scripts\01_extract_features.py --inspect
.venv-radiomics\Scripts\python.exe scripts\01_extract_features.py          # hours, CPU-bound
.venv\Scripts\python.exe scripts\02_run_loco.py --features data\processed\features.csv
.venv\Scripts\python.exe scripts\04_analysis.py --run results\loco_v1
```

## What the protocol enforces

Peng et al. fit selection once on a fixed training set and place every patient
in one graph. Under leave-one-centre-out that is not sound, so three rules are
enforced in `mlfgnn/experiments/fold.py`:

1. **Selection, standardisation and clinical encoding are refit inside every
   fold**, on the inner-training rows only.
2. **The training graph contains training-centre nodes only.**
3. **Held-out patients are attached by directed edges only** — messages flow
   train → held-out, never the reverse — so the evaluation is inductive and
   scoring one held-out patient cannot change another's.

Each fold reports `in_centre`, `held_out_centre`, and their fold-wise
difference, the **generalisation gap**, which is the quantity under study.

### A finding from the protocol itself

SNF is a transductive diffusion; to apply the 0.2 threshold to an unseen centre
at all, out-of-sample similarity must be quantile-mapped onto the training
distribution. k-NN rules are invariant to monotone transforms and need no such
step. That asymmetry argues for rank-based construction independently of the
accuracy result: **a rank rule transports across sites, an absolute threshold
does not.**

## The analysis (`04_analysis.py`)

Writes `<run>/analysis/` with six tables, matching figures, and `ANALYSIS.md`:

| | question |
|---|---|
| A1 | in-centre / held-out / gap per arm, every metric |
| A2 | which held-out centres are hard, per arm |
| A3 | paired bootstrap of every arm vs the existing rule, stratified by fold |
| A4 | sensitivity to k |
| A5 | mechanism: cross-centre edge ratio and assortativity vs held-out C and gap |
| A6 | every GNN arm vs the best tabular baseline (Mueller et al.'s critique) |

From the simulated cohort, the mechanism table alone is already informative:

| rule | cross-centre edges | centre assortativity | mean degree |
|---|---|---|---|
| `snf_threshold` | 0.51 | +0.30 | 42.2 |
| `knn` (k=3) | 0.27 | **+0.63** | 4.0 |
| `centre_balanced_knn` (k=3) | **1.00** | −0.34 | 9.5 |

## Layout

```
configs/    pyradiomics.yaml (pinned, hashed)  experiment.yaml (the whole protocol)
mlfgnn/
  data/       centres.py  hnpc.py  tcia.py  clinical.py  synthetic.py
  features/   extract.py [3.9 env]   select.py [main env]
  graphs/     similarity.py (SNF)  construct.py (the three rules)  metrics.py
  models/     layers.py  mlfgnn.py  losses.py  baselines.py
  eval/       metrics.py  bootstrap.py  report.py
  experiments/ fold.py (leakage control)  train.py  loco.py
  device.py
scripts/    00 .. 04
tests/      36 tests: pipeline invariants + real-cohort loader checks
```

## Reference

Peng J. et al. *Multi-level fusion graph neural network: application to PET and
CT imaging for risk stratification of head and neck cancer.* Biomed. Signal
Process. Control 92 (2024) 106137. <https://doi.org/10.1016/j.bspc.2024.106137>

Vallières M. et al. *Radiomics strategies for risk assessment of tumour failure
in head-and-neck cancer.* Sci. Rep. 7, 10117 (2017). Data: TCIA Head-Neck-PET-CT,
<https://doi.org/10.7937/K9/TCIA.2017.8oje5q00>
