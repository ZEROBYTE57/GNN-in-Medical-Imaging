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

# Project glossary (CXB / CF-Site), plain language

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
- **GCN layer (Kipf & Welling 2017):** each region updates itself by averaging its neighbours' features, weighted by connection strength. Our GCN has two such layers.
- **Readout:** the step that turns all region outputs into one prediction per brain. Ours joins them in a fixed order (as in BrainGB), so the model knows which region is which.
- **BrainGB (IEEE TMI 2023):** a benchmark that standardised how brain GNNs are built; our GCN follows it.
- **BrainNetCNN (NeuroImage 2017):** a connectome model that slides cross-shaped filters over the connectivity matrix (edge-to-edge, edge-to-node, node-to-graph). The classic brain-network deep model.
- **BrainGNN (Medical Image Analysis 2021):** a brain GNN whose layers have region-specific weights and that keeps only the most useful regions (pooling). The kept-region scores are its built-in explanation.
- **Transformer / attention:** a model in which every region looks at every other region and learns how much to attend to each. **Brain Network Transformer (BNT, NeurIPS 2022)** applies this to connectivity; its attention weights can be read as an explanation.
- **ContrastPool (IEEE TMI 2024):** a recent explainable brain GNN that pools regions using differences between patient and control groups.
- **Built-in vs post-hoc explanation:** built-in = the model produces it while predicting (BrainGNN scores, BNT attention); post-hoc = computed afterwards by an explainer (saliency, IG). CXB can score both against the planted answer.
- **Reimplementation:** writing a published model ourselves from its paper, with the same layers and settings. The paper then says "our reimplementation of X" and lists any differences.
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
- **CF-Site, counterfactual site training (our method; called SiteSwap until 1 Nov 2026, code name `siteswap`):** adds a counterfactual copy of each training scan moved to another site (via ComBat), with the same label, so the site stops being a useful clue. Only the training data gets extra copies; the test data is not changed. (An early version also penalised explanations that changed with the site; it is now only an ablation.)

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
- **t-test reference (formerly "recovery ceiling"):** the AUPRC a plain statistical test gets (|t| per edge between the fake groups, training subjects only, no model). If even this fails, explainers are unlikely to succeed. It is *not* an upper bound: a trained model scored on more subjects can beat it slightly (saliency 0.80–0.81 vs 0.78 at x2).
  - × 0.5: 0.03–0.08
  - × 1: 0.17–0.28
  - × 2: 0.80–0.83
  - × 4: about 1.0
- **Scanner share:** the share of an explanation's most important connections (top 45) that lie in the fake scanner regions. 0 means the explanation ignores the scanner; 0.45 means almost half of it points at the scanner.
- **Eta squared (η²):** the share of a connection's variation explained by which hospital a scan came from. We use it to find where real scanners leave the biggest fingerprints (median 0.08, decoy regions 0.14).
- **GNNExplainer-style input mask:** our version of GNNExplainer. It learns a soft mask over the connectivity matrix that keeps the model's prediction while using as few connections as possible.
- **Site alignment (Spearman):** on real labels, how closely the explanation's ranking of connections follows the ranking by hospital effect (eta squared). Near 0 means the explanation doesn't follow the hospital.
- **Label probe:** a small logistic regression that tries to predict the label from a chosen set of connections. If it succeeds where it shouldn't, label information has leaked into those connections.
- **Fidelity:** does removing the edges an explainer marks as important actually change the prediction?
- **Stability:** does the explanation stay similar across seeds and small changes to the data?
- **Shuffled-label control:** train on randomly shuffled labels. A trustworthy explainer should then find nothing.

## Baselines and the base paper

- **Baseline:** an existing method we compare against.
- **BrainIB (TNNLS 2025):** the original base paper (now cited as related work only). It's a brain GNN that learns an explanation (a subgraph).
  - It uses an **information bottleneck**: keep only the information needed to predict the label.
  - Its mask is sampled with **Gumbel-softmax**, a trick for picking edges at random but in a way that can still be trained.
- **Weight change:** how much a network's parameters moved during training. A part whose weights never move has learned nothing. We use this as a check that an explanation module is really trained.
- **Reimplementation:** writing a model ourselves from its paper's description. Done for BrainNetCNN, BrainGNN and Brain Network Transformer. A BrainIB reimplementation was planned (its code has no licence) but dropped; the final paper does not use BrainIB.
- **CI-GNN, BrainOOD:** other brain GNNs, cited as related work. BrainOOD's leave-site-out folds are used.
- **Published models (1 Nov 2026):** BrainGNN, BrainNetCNN and Brain Network Transformer, reimplemented from their papers. Only BrainNetCNN passes the sanity test; the other two do not learn from about 250 training subjects.
- **Holm correction:** a way to adjust p-values when several tests are run together, so that one lucky result does not count as a finding. We use it for the main comparisons.
- **Scanner map (audit):** the connections that differ most between scanners in the training data. It stands in for the known decoy connections, so the scanner share of an explanation can be checked on real data without ground truth.

## Project process

- **Gate:** a go/no-go check before investing more work. If a gate fails, we switch to a planned fallback.
- **Gate 1:** can a simple model plus explainer find a strong (× 4) planted effect at all? It passes if AUPRC ≥ 0.5. If not, the benchmark idea needs rethinking.
- **Pass rule fixed in advance (pre-registration):** writing down the success test *before* running an experiment, so it can't be adjusted to fit the results.
- **Non-inferior:** not clearly better than the competitor by the agreed margin, but not worse either.
- **Data augmentation:** training on extra, altered copies of the data. In CF-Site, each scan is copied to the other scanner with the same label, so the scanner stops being a useful clue.
- **Confirmation run:** repeating an experiment on new, unseen splits to check that a choice made after seeing results still holds.
- **Oracle / upper bound:** a version of a method given perfect information that's unavailable in practice (here, the true scanner-B copy of each brain). It shows the best the method could ever do.
- **Counterfactual (scan):** what the same subject's scan would look like at another site, with everything else (brain, diagnosis) unchanged. CF-Site creates these with ComBat's estimates.
- **Oracle CF-Site:** CF-Site with the true other-scanner scan instead of the ComBat estimate; an upper bound that is not available in practice.
- **Ablation:** removing or changing one part of a method to see what that part contributes (here, CF-Site with and without the explanation penalty).
- **Leakage:** label information reaching the model by an unintended route, making results look better or explanations point the wrong way.
- **Drift curve:** the same experiment repeated as the shortcut grows stronger (50% to 90%), showing when explanations start to move to the scanner.
- **Covariate inflation:** when batch correction protects a variable such as the diagnosis, it can make the groups look more different than they are. It's a known pitfall of ComBat with unbalanced sites.
- **Empirical Bayes (in ComBat):** standard ComBat does not trust each site's estimated shift on its own; it pulls every connection's estimate towards the average over all connections of that site. This makes the correction steadier with small sites. Our first ComBat left it out and turned out to be a much weaker baseline.
- **Early stopping / chosen epoch:** we keep the model from the epoch with the lowest validation loss. If that comes too early, the model is kept before it has learned anything.
- **Fallback:** a pre-agreed backup topic, such as an explanation trust score or atlas sensitivity.
- **Sensitivity analysis:** repeating the main result with a different choice (here, CC200 instead of AAL) to check the result doesn't depend on that choice.
