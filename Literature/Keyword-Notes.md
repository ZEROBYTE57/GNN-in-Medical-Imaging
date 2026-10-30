Discovering robust biomarkers of psychiatric disorders from resting-state 
functional MRI via graph neural networks: A systematic review

Chan et al. 2025, NeuroImage

Biomarkers: a measurable biological characteristic

the paper studies shows the study of biomarkers like,
- functional connectivity between brain regions
- important brain nodes indentified by gnn explainers
- network led abnomalities in brain circuts
- connectivity patterns between healthy controls and patients

Among all the psychiatric disorders, common biomarkers identified are 
- DMN Default mode network
- Salience network
- Frontopariel/Executive control network
- limbic and emotional processing regions

---

# Project glossary (CXB / SiteSwap), plain language

Grouped by topic. Numbers are from our own runs (Oct 2026).

## Brain data

- **fMRI (resting-state):** a scan of brain activity while the person lies still, doing nothing. It gives one activity signal per brain location over time.
- **Region / ROI (region of interest):** a patch of brain. The scan is averaged inside each region.
- **Atlas:** the map that cuts the brain into regions.
  - **AAL-116:** 116 anatomical regions, our main atlas.
  - **CC200:** 200 regions derived from the data, our second ("sensitivity") atlas.
- **Time series:** one region's activity over the scan, for example 200 time points. Each subject has one time series per region.
- **Functional connectivity (FC):** how strongly two regions' time series rise and fall together, measured by **Pearson correlation** (r, from -1 to +1).
- **Fisher z:** a standard transform of r (z = arctanh r) that makes the values behave better in statistics. All our matrices are in Fisher z.
- **Connectome / connectivity matrix:** the 116 × 116 table of all region-to-region correlations for one subject, with the diagonal set to 0.
- **Dead region:** a region with no signal because it lies outside the scanned area. Its row is set to 0 and counted.
- **ABIDE I:** a public autism dataset, about 1,100 subjects from 17 sites (hospitals). **PCP** is the Preprocessed Connectomes Project, which provides cleaned versions.
- **CPAC:** the cleaning pipeline we use.
  - **filt_noglobal:** band-pass filtered, *without* **global signal regression (GSR)**.
  - GSR subtracts the whole-brain average, which distorts negative correlations.
- **Our cohort:** 848 quality-checked subjects from 19 sites. OHSU was dropped because its scans are too short.

## Graphs and models

- **Graph:** nodes joined by edges. Here, **node = brain region** and **edge = connection**; an edge's strength is the correlation.
- **GNN (graph neural network):** a neural network that works on graphs by passing information along edges. Examples: **GCN**, **GIN**.
- **MLP:** a plain neural network that ignores the graph and just reads the matrix values. It's the key "is the graph even needed?" comparison.
- **Label:** the answer the model must predict (here, autism = 1, control = 0).
- **Control:** a healthy participant, not someone "about to get the disease". Patients are people with the diagnosis.
- **Epoch:** one pass of training through all the training data.
- **Seed:** the number that fixes the randomness, so a run can be repeated exactly.

## Splitting and testing

- **Train / validation / test:**
  - *train*: the data the model learns from;
  - *validation*: the data used to choose settings, such as which epoch to stop at;
  - *test*: the data the final score is measured on, once. It must never influence any choice.
- **Fold:** one way of cutting the data into train, validation and test. **k-fold cross-validation** trains k times, each time with a different test part, and averages the scores.
- **Leave-site-out / out-of-distribution (OOD):** the test part is a whole hospital the model never saw. Much harder, and more realistic, than random folds. We use **BrainOOD's 10 site folds** (fold 0, for example: test SBL, validation STANFORD).
- **Test-set selection ("best over epochs"):** reporting the best test score across all epochs.
  - This uses the test set to make a choice, so the score is inflated, often by several points.
  - We never do this: the epoch is always chosen on validation data.
- **Majority baseline:** always predicting the bigger group. On ABIDE this scores about 52%, the floor any model must beat.

## Sites and shortcuts (the "scanner" half)

- **Site effect / scanner effect:** differences in the data caused by the hospital's scanner and settings, not by the disease.
- **Shortcut:** when a model predicts from site differences (or another unrelated cue) instead of the disease. The model can look accurate while being right for the wrong reasons.
- **ComBat / harmonisation:** a statistical method that estimates and removes site effects. We use it to *add* them instead, fitted on training folds only.
- **Site decoy:** a fake site effect we plant on purpose, to test whether explanations point at it.
- **SiteSwap (our method):** trains the model so its explanation stays the same when a scan's site is swapped (via ComBat). An explanation that changes with the site is site-driven.

## Planted ground truth (the benchmark, CXB)

- **CXB (Connectome eXplanation Benchmark):** our benchmark: real scans with a *known*, planted correct answer, so explanations can be scored.
- **Ground truth:** the true answer. Real autism has no known ground truth, so we plant one.
- **Planting:** we take healthy controls only and split them randomly into fake "patients" and fake "controls". Then we add a known signal to the fake patients. Any model that separates the two groups must use that signal.
- **Fake labels:** the random patient/control labels. They're balanced within each site, so the site alone can't predict them.
- **Ranking set vs planting pool:**
  - The ranking set (20% of controls) is used only to decide *where* to plant, by comparing it with autism subjects.
  - The rest of the controls (the pool) get planted.
  - No subject is used twice.
- **Cohen's d:** effect size: the difference between two group means, in standard-deviation units. Real autism connection effects are small (top edge d = 0.41).
- **Winner's curse:** the strongest effects found in a small sample are overestimated, by chance.
- **Region-level planting (our design):** one shared signal is added to the time series of 7, 10 or 20 regions, then connectivity is recomputed. These regions become more connected with each other.
- **Orthogonal signal:** a signal made unrelated to all existing region time series, so it changes only the planted regions' connections.
- **Valid correlation matrix:** a matrix a real brain could produce (mathematically: positive semidefinite). Our first design, which shifted single connections, broke this for 100% of matrices. The current design breaks it for 0%.
- **Inside edges:** connections among the planted regions. They are the true answer ("positives").
- **Incident edges:** connections from a planted region to other regions. They weaken by about 25% as a side effect, so scoring ignores them ("don't care").
- **Untouched edges:** all other connections. They're unchanged ("negatives").
- **Level (× 0.5, × 1, × 2, × 4):** how strong the planted effect is, as a multiple of a real autism effect. × 4 is used only for gate 1.

## Explanations and scoring

- **Explainer:** a method that says which edges or regions drove the model's prediction.
  - **Saliency:** the size of the gradient.
  - **Integrated Gradients:** gradients averaged along a path from a blank input.
  - **GNNExplainer** and **PGExplainer:** learn a mask over edges.
- **Explanation mask:** a score per edge saying how important it is.
- **AUPRC (area under the precision-recall curve):** how well the explainer's ranking of edges puts the true (planted) edges on top. 1 is perfect; chance is the share of true edges, 0.003 to 0.03 here.
- **Recovery ceiling:** the AUPRC a plain statistical test gets (|t| per edge, no model). If even this fails, no explainer can succeed. Ours:
  - × 0.5: 0.03–0.08
  - × 1: 0.17–0.28
  - × 2: 0.80–0.83
  - × 4: about 1.0
- **Scanner share:** the share of an explanation's most important connections (top 45) that lie in the fake scanner regions. 0 means the explanation ignores the scanner; 0.45 means almost half of it points at the scanner.
- **Eta squared (η²):** the share of a connection's variation explained by which hospital a scan came from. We use it to find where real scanners leave the biggest fingerprints (median 0.08, decoy regions 0.14).
- **Fidelity:** does removing the edges an explainer marks as important actually change the prediction?
- **Stability:** does the explanation stay similar across seeds and small changes to the data?
- **Shuffled-label control:** train on randomly shuffled labels. A trustworthy explainer should then find nothing.

## Baselines and the base paper

- **Baseline:** an existing method we compare against.
- **BrainIB (TNNLS 2025):** the base paper. It's a brain GNN that learns an explanation (a subgraph).
  - It uses an **information bottleneck**: keep only the information needed to predict the label.
  - Its mask is sampled with **Gumbel-softmax**, a trick for picking edges at random but in a way that can still be trained.
- **Weight change:** how much a network's parameters moved during training. A part whose weights never move has learned nothing. We use this as a check that an explanation module is really trained.
- **Reimplementation:** we write our own BrainIB from the paper's description, because their code has no licence.
- **CI-GNN, BrainOOD, BrainGNN:** other brain-GNN baselines, still to be added.

## Project process

- **Gate:** a go/no-go check before investing more work. If a gate fails, we switch to a planned fallback.
- **Gate 1:** can a simple model plus explainer find a strong (× 4) planted effect at all? It passes if AUPRC ≥ 0.5. If not, the benchmark idea needs rethinking.
- **Pass rule fixed in advance (pre-registration):** writing down the success test *before* running an experiment, so it can't be adjusted to fit the results.
- **Non-inferior:** not clearly better than the competitor by the agreed margin, but not worse either.
- **Data augmentation:** training on extra, altered copies of the data. In SiteSwap, each scan is copied to the other scanner with the same label, so the scanner stops being a useful clue.
- **Confirmation run:** repeating an experiment on new, unseen splits to check that a choice made after seeing results still holds.
- **Oracle / upper bound:** a version of a method given perfect information that's unavailable in practice (here, the true scanner-B copy of each brain). It shows the best the method could ever do.
- **Ablation:** removing or changing one part of a method to see what that part contributes (here, SiteSwap with and without the explanation penalty).
- **Leakage:** label information reaching the model by an unintended route, making results look better or explanations point the wrong way.
- **Fallback:** a pre-agreed backup topic, such as an explanation trust score or atlas sensitivity.
- **Sensitivity analysis:** repeating the main result with a different choice (here, CC200 instead of AAL) to check the result doesn't depend on that choice.
