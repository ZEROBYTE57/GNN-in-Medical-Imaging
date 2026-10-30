# GNN-in-Medical-Imaging

Code and notes for an M.Tech dissertation on graph neural networks (GNNs) in medical imaging.

## Active project: CXB (Connectome eXplanation Benchmark)

**Question:** when a brain GNN highlights connections, has it found the disease or the scanner?

Brain-GNN papers usually validate their explanations by matching them to published findings, because real data has no known correct answer. This project creates one. It plants a known connectivity effect into real healthy-control connectomes from ABIDE, adds a fake scanner shortcut, and measures whether explanations point at the planted effect or at the scanner. It also proposes **SiteSwap**, a simple training method that keeps explanations on the real signal.

| Part | Choice |
|---|---|
| Data | ABIDE I (PCP, CPAC, no global signal regression), AAL-116 atlas, 848 subjects, 19 sites |
| Planted effect | One shared signal added to the time series of 10 regions in fake "patients" (strengths x1, x2, x4); truth = connections inside the set |
| Scanner decoy | ComBat-style site shift, sized from real ABIDE sites, on the 10 regions where real sites differ most; linked to the label in training only (50/70/90%), plus a train-and-test comparison |
| Models | Own MLP and GCN; published brain GNNs added after the core experiment works |
| Explainers | Saliency, Integrated Gradients, GNNExplainer |
| Proposed method | **SiteSwap**: penalise the model when its explanation changes between a scan and its site-swapped (ComBat) copy; compared with ComBat harmonisation |
| Metrics | Disease AUPRC (planted connections), scanner share of the explanation, test accuracy |
| Evaluation | BrainOOD's 10 leave-site-out folds; 3 fake-label splits x 3 seeds; epoch chosen on validation data only |

Progress is checked at three gates: (1) explainers find a strong planted effect; (2) a scanner shortcut pulls explanations towards it; (3) SiteSwap reduces that pull more than ComBat.

Implemented so far:
- ABIDE download and connectivity matrices (848 subjects, 19 sites).
- BrainOOD site folds.
- Region-level planting, checked on real data: all matrices valid, effect within 5% of target.
- Own MLP and GCN, saliency and Integrated Gradients.
- **Gate 1 passed:** at strength × 4, the explanations find the planted connections (GCN AUPRC 1.00, MLP 0.99; chance 0.008).
- Scanner decoy, sized from real ABIDE site differences.
- **Gate 2 passed:** with a 90% shortcut in training, 45% (MLP) and 28% (GCN) of the top explanation connections move to the scanner regions (from 0%), and disease recovery falls. Test accuracy barely changes, or rises when the shortcut is also in the test data.
- ComBat harmonisation and SiteSwap.
- **Gate 3 (non-inferior under the pre-fixed rule):**
  - ComBat removes the scanner but also most of the disease signal.
  - SiteSwap with augmentation only gives the best disease recovery (MLP 0.55, GCN 0.34, against ComBat's 0.26 and 0.08).
  - The explanation-penalty version breaks the GCN.

- With a perfect swap (upper bound), SiteSwap removes the scanner completely (MLP scanner share 0.00, disease AUPRC 0.76). Its practical limit is the ComBat swap, which cannot copy how a scanner effect scales connections, person by person.

In progress: confirmation of augmentation-only SiteSwap on new fake-label splits (3-5, seed 1), with the pass rule fixed before running.

## Repository layout

| Folder | Contents | State |
|---|---|---|
| [CXB/](CXB/) | CXB benchmark and the SiteSwap method 
| [Literature/](Literature/) | Reading notes and literature list (the papers themselves are not in the repo) 
| [MLFGNN/](MLFGNN/) | Centre-aware population graph for head-and-neck cancer survival (TCIA); synthetic data only 
| [HNC GNN/](HNC%20GNN/) | Earlier head-and-neck cancer pipeline
| [AD-PopGraph/](AD-PopGraph/) | Leakage-free population graph for Alzheimer's disease (ADNI); protocol only
| [Retinal-GNN/](Retinal-GNN/) | Robustness benchmark for graph-based retinal vessel segmentation; metrics only
| [GNN/](GNN/) | Early topic-search reports

Inside the active project:

```text
CXB/
  configs/         settings (data.yaml)
  data/            raw and processed data (not in git)
  docs/            decisions log and data summary
  scripts/         numbered pipeline steps (01 download, 02 connectomes, 03 planting check, 04 gate 1, 05 gate 2, 06 gate 3)
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

- **ABIDE I**, via the Preprocessed Connectomes Project (CPAC pipeline, band-pass filtered, no global signal regression).
- ABIDE's terms require citing Di Martino et al., *Molecular Psychiatry* 19 (2014), and Craddock et al., *Frontiers in Neuroinformatics* (2013).
