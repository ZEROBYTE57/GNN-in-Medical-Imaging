# GNN-in-Medical-Imaging

Code and notes for an M.Tech dissertation on graph neural networks (GNNs) in medical imaging.

## Active project: CXB (Connectome eXplanation Benchmark)

**Question:** when a brain GNN highlights connections, has it found the disease or the scanner?

Brain-GNN papers usually validate their explanations by matching them to published findings, because real data has no known correct answer. This project creates one: it plants known connectivity effects and realistic scanner-site artefacts into real connectomes from the ABIDE dataset, then measures whether explainers and shortcut-robust brain GNNs recover the planted effect or the site artefact.

| Part | What it does |
|---|---|
| Benchmark | Known effects (strong, moderate, weak) planted into healthy-control connectomes; site decoys built from measured ComBat site effects |
| Models tested | MLP, logistic regression, random forest; GCN, GAT, GIN, graph transformer; BrainGNN, BrainIB, CI-GNN, BrainOOD |
| Explainers tested | Saliency, Integrated Gradients, GNNExplainer, PGExplainer, attention, built-in model explanations |
| Proposed method | **SiteSwap**: a GNN trained so its explanation stays the same across site-swapped copies of each subject |
| Main metric | How well each explanation recovers the planted connections (AUPRC) |
| Evaluation | Subject-level, inductive splits; random 5-fold and leave-one-site-out; 5 seeds; shuffled-label control |

Implemented so far: environment, ABIDE download, and connectivity matrices for 848 subjects from 19 sites (AAL-116 and CC200 atlases).

## Repository layout

| Folder | Contents | State |
|---|---|---|
| [CXB/](CXB/) | CXB benchmark and the SiteSwap method | **Active** |
| [Literature/](Literature/) | Reading notes and literature list (the papers themselves are not in the repo) | Active |
| [MLFGNN/](MLFGNN/) | Centre-aware population graph for head-and-neck cancer survival (TCIA); synthetic data only | Archived: data access |
| [HNC GNN/](HNC%20GNN/) | Earlier head-and-neck cancer pipeline | Archived |
| [AD-PopGraph/](AD-PopGraph/) | Leakage-free population graph for Alzheimer's disease (ADNI); protocol only | Archived: data access |
| [Retinal-GNN/](Retinal-GNN/) | Robustness benchmark for graph-based retinal vessel segmentation; metrics only | Archived |
| [GNN/](GNN/) | Early topic-search reports | Archived |

Inside the active project:

```text
CXB/
  configs/         settings (data.yaml)
  data/            raw and processed data (not in git)
  docs/            decisions log and data summary
  scripts/         numbered pipeline steps
  src/cxb/         the Python package
  tests/           unit tests
  results/         experiment outputs (not in git)
```

## Quick start

Windows, Python 3.14, NVIDIA GPU with CUDA 13.

```powershell
cd CXB
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install "torch==2.14.0+cu130" --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m pytest -q

.venv\Scripts\python.exe scripts\01_download_abide.py      # about 520 MB, no login
.venv\Scripts\python.exe scripts\02_build_connectomes.py   # writes data/processed/*.npz
```

Settings: [configs/data.yaml](CXB/configs/data.yaml). Reasons for each choice: [docs/decisions.md](CXB/docs/decisions.md). Cohort counts and quality checks: [docs/data-summary.md](CXB/docs/data-summary.md).

## Data

- **ABIDE I**, via the Preprocessed Connectomes Project (CPAC pipeline, band-pass filtered, no global signal regression). Downloaded by the script; never committed.
- ABIDE's terms require citing Di Martino et al., *Molecular Psychiatry* 19 (2014), and Craddock et al., *Frontiers in Neuroinformatics* (2013).
- Data and results folders are excluded by [.gitignore](.gitignore).
