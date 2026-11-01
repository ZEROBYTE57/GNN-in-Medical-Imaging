# GNN-in-Medical-Imaging

Code and notes for an M.Tech dissertation on graph neural networks (GNNs) in medical imaging.

## Active project: CXB (Connectome eXplanation Benchmark)

**Question:** when a brain GNN highlights connections, has it found the disease or the scanner?

Brain-GNN papers usually validate their explanations by matching them to published findings, because real data has no known correct answer. This project creates one. It plants a known connectivity effect into real healthy-control connectomes from ABIDE, adds a fake scanner shortcut, and measures whether explanations point at the planted effect or at the scanner. It then asks which fixes keep explanations on the disease: ComBat harmonisation (four versions) and **SiteSwap**, a simple training-side alternative.

| Part | Choice |
|---|---|
| Data | ABIDE I (PCP, CPAC, no global signal regression), AAL-116 atlas, 848 subjects, 19 sites |
| Planted effect | One shared signal added to the time series of 10 regions in fake "patients" (strengths x1, x2, x4); truth = connections inside the set |
| Scanner decoy | Site shift sized from real ABIDE sites, on the 10 regions where real sites differ most; linked to the label in training only (50-90%). Main version: person-specific (added in the time series); robustness version: a fixed shift per connection (exactly ComBat's model) |
| Models | Logistic regression, MLP, GCN (BrainGB set-up), BrainNetCNN (reimplemented from its paper). BrainGNN and Brain Network Transformer were also reimplemented but fail the sanity test at this sample size |
| Explainers | Saliency, Integrated Gradients, GNNExplainer-style input mask |
| Fixes compared | ComBat harmonisation (with and without empirical Bayes and label protection) and **SiteSwap**: each training scan gets a copy moved to the other scanner with ComBat's estimates (label kept), and the model trains on both. A perfect swap (true other-scanner scan) is the upper bound |
| Metrics | Disease AUPRC (planted connections), scanner share of the explanation, test accuracy |
| Evaluation | BrainOOD's 10 leave-site-out folds; 6 fake-label splits x 3 seeds for the methods comparison (3 x 3 elsewhere); epoch chosen on validation data only, after a 30-epoch warm-up; statistics per split (bootstrap CI, sign test, paired t-test; Holm correction for the main comparisons) |

Development went through three gates: (1) explainers find a strong planted effect; (2) a scanner shortcut pulls explanations towards it; (3) fixes compared. The final numbers below were rerun under one fixed protocol and analysis plan (see [CXB/docs/decisions.md](CXB/docs/decisions.md)). ComBat is the standard version (empirical Bayes), compared with and without the diagnosis protected.

Final results ([CXB/docs/stats.md](CXB/docs/stats.md), [CXB/docs/results.md](CXB/docs/results.md), figures in [CXB/docs/figures/](CXB/docs/figures/)):
- **Benchmark:** at strength x2, saliency recovers the planted connections as well as a univariate t-test on the training subjects, or slightly better (AUPRC MLP 0.80, GCN 0.81; t-test reference 0.78; chance 0.008). Integrated Gradients and the input mask find less.
- **Controls:** with shuffled labels, explainers score at chance (AUPRC 0.010-0.011; chance 0.008), so recovery reflects what a model learned, not explainer bias. A logistic regression's coefficients recover the planted effect best (0.98; the planted effect is a linear shift, which suits a linear model) and drift to the scanner under the shortcut like the neural models (0.52, scanner share 0.43).
- **Other models:** BrainNetCNN passes the sanity test (AUPRC 1.00 at x4) and fails differently: under the 90% shortcut its explanations stay on the disease (AUPRC 0.99, scanner share 0.03), but its accuracy drops by 0.12 (0.95 to 0.83; 3 splits x 2 seeds). BrainGNN and Brain Network Transformer do not learn even the strong planted effect from about 250 training subjects (accuracy about 0.60, AUPRC under 0.1, also after a fixed tuning budget), so their explanations cannot be judged at this sample size.
- **Explanations drift, accuracy does not:** going from a 50% to a 90% scanner shortcut, 46% (MLP) and 22% (GCN) of the top explanation connections move to the scanner regions and disease recovery drops by 0.27-0.37, while test accuracy falls only 0.04-0.06.
- **Fixes: the ComBat version decides the outcome.** Under the 90% shortcut (6 splits x 3 seeds), GCN disease recovery ranges from 0.02 to 0.63 across the four ComBat versions (MLP 0.19 to 0.59); only standard ComBat with the diagnosis protected works well. Without label protection, standard ComBat leaves the MLP on the scanner (share 0.49; plain 0.43).
- **SiteSwap is a safe training-side fix:** it beats the plain model clearly (disease AUPRC +0.15 for both models, Holm-adjusted p ≤ 0.001) and matches the best ComBat (MLP -0.00; GCN +0.10, a tie after Holm correction, p 0.07), with slightly higher accuracy (+0.03), whichever ComBat it takes its swap from (MLP 0.57-0.59, GCN 0.73-0.77), and new-site data need no harmonisation at test time. For the MLP it leaves more scanner share than protected ComBat (0.28 vs 0.16).
- **The limit is the scanner model, not the method:** SiteSwap and ComBat use the same estimate of the scanner effect, which assumes one fixed shift per connection; with the true other-scanner scan (perfect swap) disease recovery reaches 0.85 (MLP) and 0.90 (GCN) and the scanner share falls to 0.00.
- **The check works without ground truth:** the scanner-affected connections can be found from training data alone. Of the 45 connections that differ most between scanners in the training data, 93-100% touch the scanner regions and at most 5% are disease connections ([scripts/16_audit_check.py](CXB/scripts/16_audit_check.py)), so the scanner share of an explanation can be measured on real data.
- **Robustness:** with a ComBat-friendly decoy SiteSwap is slightly ahead of standard ComBat (MLP +0.02, GCN +0.09). With a second set of disease and decoy regions the two tie; there the plain GCN ignores the scanner, and SiteSwap costs it a little (-0.03 AUPRC), while for the MLP SiteSwap still beats plain by +0.13.
- **The real hospital signal is strong:** from the same connectomes, a logistic regression predicts the hospital with balanced accuracy 0.70 (chance 0.05, 19 hospitals) but autism only 0.64 (chance 0.50). The scanner is the easier thing to learn, which is what makes the shortcut a real risk.
- **Real autism labels:** explanations do not follow the hospital effect for any method (standard ComBat included); SiteSwap keeps the plain model's accuracy (MLP 0.629 vs 0.632; standard protected ComBat 0.603).
- **Label-protected ComBat** can inflate group differences (Nygaard et al., 2016): with the simplified ComBat the diagnosis becomes predictable from scanner-related connections (AUC 0.89 vs 0.69 raw), and its high accuracy when the shortcut is also in the test data is an artefact. Standard (empirical-Bayes) ComBat largely removes this (AUC 0.72).
- The explanation-consistency penalty (an early SiteSwap variant) hurts the GCN (disease AUPRC 0.20) and is reported only as an ablation.

## Repository layout

| Folder | Contents |
|---|---|
| [CXB/](CXB/) | CXB benchmark and the SiteSwap method |
| [Literature/](Literature/) | Reading notes and literature list (the papers themselves are not in the repo) |
| [MLFGNN/](MLFGNN/) | Centre-aware population graph for head-and-neck cancer survival (TCIA); synthetic data only |
| [HNC GNN/](HNC%20GNN/) | Earlier head-and-neck cancer pipeline |
| [AD-PopGraph/](AD-PopGraph/) | Leakage-free population graph for Alzheimer's disease (ADNI); protocol only |
| [Retinal-GNN/](Retinal-GNN/) | Robustness benchmark for graph-based retinal vessel segmentation; metrics only |
| [GNN/](GNN/) | Early topic-search reports |

Inside the active project:

```text
CXB/
  configs/         settings (data.yaml)
  data/            raw and processed data (not in git)
  docs/            decisions log, data summary, results tables and figures
  scripts/         numbered pipeline steps (01 download, 02 connectomes, 03 planting check, 04-06 gates 1-3, 07 benchmark table, 08 real labels, 09 ComBat check, 10 figures, 11 statistics, 12 standard ComBat check, 13 controls, 14 tuning, 15 hospital signal, 16 scanner-map check)
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

### Reproduce the paper

Run from `CXB/` after the quick start (NVIDIA GPU; several hours in total). Every run writes to `results/final/` (not in git); the last two steps turn those files into the tables and figures.

```powershell
$P = ".venv\Scripts\python.exe"
# Benchmark: can explainers find the planted effect? (x1, x2, x4; 3 splits x 3 seeds)
& $P scripts\07_benchmark.py --levels x1,x2,x4 --splits 3 --seeds 0,1,2 --out results/final/benchmark
# Drift: scanner shortcut from 50% to 90%
& $P scripts\05_gate2.py --splits 3 --seeds 0,1,2 --conditions train:0.5,train:0.6,train:0.7,train:0.8,train:0.9 --out results/final/drift
# Methods comparison, main setting (6 splits; repeat with --seed 1 and --seed 2)
& $P scripts\06_gate3.py --splits 6 --designs train --main siteswap_aug --out results/final/methods --seed 0
& $P scripts\06_gate3.py --splits 6 --designs train --main siteswap_eb --methods combat_eb,combat_protect_eb,siteswap_eb --suffix _eb --out results/final/methods --seed 0
# Robustness (3 splits; seeds 0-2): ComBat-friendly decoy and second region set
& $P scripts\06_gate3.py --splits 3 --designs train --decoy edge --main siteswap_aug --methods plain,combat,combat_protect,siteswap_aug,siteswap_oracle --out results/final/methods --seed 0
& $P scripts\06_gate3.py --splits 3 --designs train --region-set B --main siteswap_aug --methods plain,combat,combat_protect,siteswap_aug,siteswap_oracle --out results/final/methods --seed 0
& $P scripts\06_gate3.py --splits 3 --designs train --decoy edge --main siteswap_eb --methods combat_eb,combat_protect_eb,siteswap_eb --suffix _eb --out results/final/methods --seed 0
& $P scripts\06_gate3.py --splits 3 --designs train --region-set B --main siteswap_eb --methods combat_eb,combat_protect_eb,siteswap_eb --suffix _eb --out results/final/methods --seed 0
# Real autism labels, ComBat checks, controls
& $P scripts\08_real_labels.py --seeds 0,1,2 --out results/final/real_labels
& $P scripts\08_real_labels.py --seeds 0,1,2 --methods combat_protect_eb,siteswap_eb --out results/final/real_labels_eb
& $P scripts\09_combat_check.py --out results/final/combat_check
& $P scripts\12_combat_eb_check.py
& $P scripts\13_controls.py
& $P scripts\15_site_signal.py   # CPU only
& $P scripts\16_audit_check.py   # CPU only
# Other models: sanity test (x4), then BrainNetCNN drift and methods (seeds 0 and 2 were run)
& $P scripts\04_gate1.py --level x4 --splits 3 --seed 0 --models brainnetcnn --out results/final/models/gate1_brainnetcnn   # same for braingnn, bnt
& $P scripts\05_gate2.py --models brainnetcnn --splits 3 --seeds 0 --conditions train:0.5,train:0.9 --out results/final/models/brainnetcnn/drift
& $P scripts\06_gate3.py --models brainnetcnn --splits 3 --designs train --main siteswap_aug --methods plain,combat_protect_eb,siteswap_aug --seed 0 --out results/final/models/brainnetcnn/methods
& $P scripts\14_tune.py --models bnt,braingnn   # fixed tuning budget for models that fail the sanity test
# Tables and figures
& $P scripts\11_stats.py      # docs/stats.md
& $P scripts\10_figures.py    # docs/figures/, docs/results.md
```

Settings: [configs/data.yaml](CXB/configs/data.yaml). Reasons for each choice: [docs/decisions.md](CXB/docs/decisions.md). Cohort counts and quality checks: [docs/data-summary.md](CXB/docs/data-summary.md).

## Data

- **ABIDE I**, via the Preprocessed Connectomes Project (CPAC pipeline, band-pass filtered, no global signal regression).
- ABIDE's terms require citing Di Martino et al., *Molecular Psychiatry* 19 (2014), and Craddock et al., *Frontiers in Neuroinformatics* (2013).
