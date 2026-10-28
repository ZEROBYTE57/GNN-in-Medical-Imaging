# Literature list: peer-reviewed only

Built 28 Oct 2026 for the paper "Do brain GNN explanations find the disease or the scanner?".
Preprints (arXiv only, no peer-reviewed version) are excluded and listed at the end, for awareness only.

## How to read the access column

| Code | Meaning |
|---|---|
| **OA** | Open access at the publisher: free to read and download |
| **Free proc.** | Free conference proceedings (NeurIPS, ICML/PMLR, ICLR/OpenReview, TMLR) |
| **IEEE (Amrita)** | IEEE Xplore; Amrita has access (confirmed: your BrainIB PDF came from it) |
| **Author copy** | Paywalled at the publisher, but the authors' accepted version is free on arXiv, PMC or bioRxiv |
| **Check** | Paywalled; try the Amrita library, then the authors' websites, then ask the authors by email |

**Citing:** every paper below is peer-reviewed and citable. If you read an author copy, cite the **published** version (journal, volume, pages, DOI), never the arXiv version.
**Status:** "Have" = already in this folder.

## A. Problem and motivation

| # | Paper | Venue | Use in your paper | Access | Status |
|---|---|---|---|---|---|
| A1 | Chan et al., Discovering robust biomarkers of psychiatric disorders from rs-fMRI via GNNs: a systematic review | NeuroImage 319, 2025 | Explanations judged by literature cross-referencing; no ground truth | OA | Have |
| A2 | Han et al., Rethinking functional brain connectome analysis: do graph deep learning models help? | npj Artificial Intelligence, 2026 | Simple models beat GNNs on connectomes | OA | Have |
| A3 | Mueller et al., Are population graphs really as powerful as believed? | TMLR, 2024 | GNNs vs simple baselines | Free proc. | Get |
| A4 | Geirhos et al., Shortcut learning in deep neural networks | Nature Machine Intelligence 2, 2020 | Defines shortcut learning | Author copy | Get |
| A5 | DeGrave et al., AI for radiographic COVID-19 detection selects shortcuts over signal | Nature Machine Intelligence 3, 2021 | Medical shortcut example | Author copy | Get |
| A6 | Rosenblatt et al., Data leakage inflates prediction performance in connectome-based machine learning models | Nature Communications 15, 2024 | Leakage in connectome ML; supports your inductive protocol | OA | Get |
| A7 | Abraham et al., Deriving reproducible biomarkers from multi-site resting-state data: an autism-based example | NeuroImage 147, 2017 | ABIDE site effects; leave-site-out | Author copy | Get |
| A8 | Mienye & Viriri, Graph neural networks in medical imaging: methods, applications and future directions | Information 16, 2025 | One background citation for GNNs in imaging | OA | Have (Back Up/) |

## B. Brain-network GNNs (background and baselines)

| # | Paper | Venue | Use | Access | Code | Status |
|---|---|---|---|---|---|---|
| B1 | Li et al., BrainGNN: interpretable brain graph neural network for fMRI analysis | Medical Image Analysis 74, 2021 | **Baseline** (built-in node explanations) | Author copy (bioRxiv) | Yes | Get |
| B2 | Cui et al., BrainGB: a benchmark for brain network analysis with GNNs | IEEE TMI 42, 2023 | Standard pipeline; accuracy-only benchmark | IEEE (Amrita) | Yes | Get |
| B3 | Kan et al., Brain Network Transformer | NeurIPS 2022 | **Baseline** (graph transformer) | Free proc. | Yes | Get |
| B4 | Cui et al., Interpretable GNNs for connectome-based brain disorder analysis (IBGNN) | MICCAI 2022 (arXiv v2 notes the MICCAI acceptance; cite the LNCS version) | **Baseline** (shared explanation mask) | Author copy | Yes | Get |
| B5 | Xu et al., Contrastive graph pooling for explainable classification of brain networks (ContrastPool) | IEEE TMI 43, 2024 | Interpretable brain GNN, judged by literature | IEEE (Amrita) | Yes | Optional |
| B6 | Zheng et al., BPI-GNN: interpretable brain network-based psychiatric diagnosis and subtyping | NeuroImage 293, 2024 | Prototype explanations; related work | OA | Yes | Optional |
| B7 | Wei et al., A brain graph foundation model: pre-training and prompt-tuning across broad atlases and disorders (BrainGFM) | ICLR 2026 | Newest direction; one sentence | Free proc. | Yes | Optional |

## C. Shortcut-robust brain GNNs (your contribution 2 tests these)

| # | Paper | Venue | Use | Access | Code | Status |
|---|---|---|---|---|---|---|
| C1 | Zheng et al., BrainIB: interpretable brain network-based psychiatric diagnosis with graph information bottleneck | IEEE TNNLS 36(7), 2025 | **Main baseline** | IEEE (Amrita) | Yes | Have |
| C2 | Zheng, Yu & Chen, CI-GNN: a Granger causality-inspired GNN for interpretable brain network-based psychiatric diagnosis | Neural Networks 172, 2024 | **Baseline**; quote: ground truth unavailable for real data | Author copy (arXiv) | Yes | Have |
| C3 | Xu et al., BrainOOD: out-of-distribution generalizable brain network analysis | ICLR 2025 | **Baseline**; reuse its ABIDE site splits | Free proc. | Yes | Have |
| C4 | Qiu et al., MetaExplainer: domain generalization of functional connectome analyses from the perspective of explainability | Medical Image Analysis 105, 2025 | **Baseline** if code runs | Check | Check | Get |
| C5 | Qiu et al., Towards GNNs with domain-generalizable explainability for fMRI-based brain disorder diagnosis (XG-GNN) | MICCAI 2024 (LNCS, pp. 454–464) | Related; earlier version of C4 | Check | Check | Get |
| C6 | Yu, Liu & Ji, Causal invariance-aware augmentation for brain graph contrastive learning (CIA-GCL) | ICML 2025 (PMLR 267:73221–73240) | Site-invariant brain subgraph; related | Free proc. | Check | Get |

## D. GNN explainers and how they are evaluated

| # | Paper | Venue | Use | Access | Status |
|---|---|---|---|---|---|
| D1 | Ying et al., GNNExplainer | NeurIPS 2019 | **Explainer** in the benchmark | Free proc. | Get |
| D2 | Luo et al., Parameterized explainer for GNN (PGExplainer) | NeurIPS 2020 | **Explainer** | Free proc. | Get |
| D3 | Sundararajan et al., Axiomatic attribution for deep networks (Integrated Gradients) | ICML 2017 (PMLR 70) | **Explainer** | Free proc. | Get |
| D4 | Miao, Liu & Li, Interpretable and generalizable graph learning via stochastic attention (GSAT) | ICML 2022 (PMLR 162) | Basis of SiteSwap's edge mask | Free proc. | Get |
| D5 | Agarwal et al., Evaluating explainability for graph neural networks (GraphXAI / ShapeGGen) | Scientific Data 10, 2023 | Closest benchmark (synthetic graphs only) | OA | Get |
| D6 | Agarwal, Zitnik & Lakkaraju, Probing GNN explainers: a rigorous theoretical and empirical analysis | AISTATS 2022 (PMLR 151) | Faithfulness and stability metrics | Free proc. | Get |
| D7 | Faber, Moghaddam & Wattenhofer, When comparing to ground truth is wrong: on evaluating GNN explanation methods | KDD 2021 | Pitfalls of ground-truth evaluation; answer it in your Discussion | Author copy | Get |
| D8 | Yuan et al., Explainability in graph neural networks: a taxonomic survey | IEEE TPAMI 45, 2023 | Explainer taxonomy | IEEE (Amrita) | Get |

## E. Invariant learning and shortcut-aware training (SiteSwap's competitors)

| # | Paper | Venue | Use | Access | Status |
|---|---|---|---|---|---|
| E1 | Wu et al., Discovering invariant rationales for GNNs (DIR) | ICLR 2022 | Invariant subgraph; related | Free proc. | Get |
| E2 | Sui et al., Causal attention for interpretable and generalizable graph classification (CAL) | KDD 2022 | **Baseline**: shortcut-feature mixing | Author copy | Get |
| E3 | Sui et al., Enhancing OOD generalization on graphs via causal attention learning (CAL+) | ACM TKDD 18, 2024 | Journal version of E2 | Check | Optional |
| E4 | Liu et al., Rationalizing graph neural networks with data augmentation (RGDA) | ACM TKDD 18(4), 2024 | **Closest method to SiteSwap** | Check | Have |

## F. Site effects, harmonisation and planting design

| # | Paper | Venue | Use | Access | Status |
|---|---|---|---|---|---|
| F1 | Johnson, Li & Rabinovic, Adjusting batch effects in microarray expression data using empirical Bayes methods (ComBat) | Biostatistics 8, 2007 | Original ComBat | Check (widely available) | Get |
| F2 | Fortin et al., Harmonization of multi-site diffusion tensor imaging data | NeuroImage 161, 2017 | ComBat for neuroimaging | Author copy (PMC) | Get |
| F3 | Yu et al., Statistical harmonization corrects site effects in functional connectivity measurements from multi-site fMRI data | Human Brain Mapping 39, 2018 | **Recipe for the site-swap generator** | Author copy (PMC) | Partial (need Methods) |
| F4 | Chen et al., Mitigating site effects in covariance for machine learning in neuroimaging data (CovBat) | Human Brain Mapping 43, 2022 | Covariance-level variant of the generator | OA | Get |
| F5 | Beer et al., Longitudinal ComBat | NeuroImage 220, 2020 | **Template for planting effects** into real data | OA | Have |
| F6 | Reynolds et al., ComBat harmonization: empirical Bayes versus fully Bayes approaches | NeuroImage: Clinical 39, 2023 | Precedent: ComBat for synthetic data and planted effects | OA | Have |
| F7 | Wang, Chen & Yan, Comprehensive evaluation of harmonization on functional brain imaging for multisite data-fusion | NeuroImage 274, 2023 | Harmonisation for fMRI; supports generator choices | OA | Optional |
| F8 | Xin et al., ComBat-Predict enhances generalizability of neuroimaging models to new sites | Human Brain Mapping 47, 2025/26 | Applying ComBat to unseen sites (leave-site-out) | OA | Optional |

## G. Datasets, atlases and tools (must cite if used)

| # | Paper | Venue | Why you must cite it | Access |
|---|---|---|---|---|
| G1 | Di Martino et al., The autism brain imaging data exchange (ABIDE I) | Molecular Psychiatry 19, 2014 | Required by ABIDE's terms | Author copy (PMC) |
| G2 | Craddock et al., The Neuro Bureau Preprocessing Initiative: open sharing of preprocessed neuroimaging data and derivatives | Frontiers in Neuroinformatics, 2013 (conference abstract) | Required for ABIDE Preprocessed (PCP) | OA |
| G3 | ADHD-200 Consortium, The ADHD-200 Consortium: a model to advance the translational potential of neuroimaging | Frontiers in Systems Neuroscience 6, 2012 | Replication dataset | OA |
| G4 | Bellec et al., The Neuro Bureau ADHD-200 Preprocessed repository | NeuroImage 144, 2017 | Preprocessed ADHD-200 | OA |
| G5 | Tzourio-Mazoyer et al., Automated anatomical labeling of activations in SPM (AAL atlas) | NeuroImage 15, 2002 | Main atlas | Check (cite without full read) |
| G6 | Craddock et al., A whole brain fMRI atlas generated via spatially constrained spectral clustering (CC200) | Human Brain Mapping 33, 2012 | Sensitivity atlas | Author copy (PMC) |
| G7 | Yeo et al., The organization of the human cerebral cortex estimated by intrinsic functional connectivity (Yeo-7) | Journal of Neurophysiology 106, 2011 | Network prior in SiteSwap | OA (free at journal) |
| G8 | Fey & Lenssen, Fast graph representation learning with PyTorch Geometric | ICLR Workshop 2019 | Software; workshop paper, standard to cite | Free |
| G9 | Abraham et al., Machine learning for neuroimaging with scikit-learn (nilearn) | Frontiers in Neuroinformatics 8, 2014 | Software for data loading and connectivity | OA |

Captum (explainer library) exists only on arXiv: cite it as software (GitHub URL and version), not as a paper.

## H. 2026 papers (peer-reviewed, found 28 Oct 2026)

Searched with Consensus (2026 only, preprints excluded) and web search. None plants known effects into real connectomes to test explanations, so the novelty still holds.

| # | Paper | Venue | Use in your paper | Access | Code | Priority |
|---|---|---|---|---|---|---|
| H1 | Lee et al., GNN-based modeling of functional brain connectivity in ASD using ABIDE-I and II | Psychiatry Research: Neuroimaging, 2026 | **Benchmark reference:** GCN/GAT/SAGPool on 2,013 ABIDE subjects, CC200 atlas, ComBat; about 61% balanced accuracy over 100 × 10-fold CV. Sets a realistic accuracy range; uses permutation importance for explanations | Check | Check | **High** |
| H2 | Hu et al., BrainIB++: GNNs and information bottleneck for functional brain biomarkers in schizophrenia | Biomedical Signal Processing and Control, 2026 | Newest information-bottleneck brain GNN; possible extra baseline; biomarkers judged by match to clinical findings | Check | Yes (github.com/TianzhengHU/BrainIB_coding) | **High** |
| H3 | Zheng et al., Functional connectivity-based classification and subtyping of major depression: an ensemble GNN approach | PLOS Digital Health, 2026 | Compares against BrainIB and CI-GNN under leave-one-site-out; uses ComBat | OA | Check | Medium |
| H4 | Wang et al., ADHD classification with GCN via joint feature learning among nodes and edges (JNEL-GCN) | IEEE TMI, 2026 | Reports 97% accuracy on ADHD-200 and ABIDE-I, with biomarkers validated against the literature. Example of why ground-truth checks are needed; cite neutrally | IEEE (Amrita) | Check | Medium |
| H5 | Xia et al., Meta-analysis prompt guided multi-task graph transformer for cognitive deficits and psychiatric disorders | Medical Image Analysis, 2026 | Newest interpretable brain graph transformer; biomarkers judged by match to findings | Check | Check | Low |
| H6 | Zhao et al., A unified graph domain adaptation framework for cross-site brain network analysis | IEEE TMI, 2026 | Cross-site GNN with adversarial site alignment (related to SiteSwap's site head) | IEEE (Amrita) | Check | Medium |
| H7 | Ji et al., Prototypical representation learning for multi-site domain generalization in schizophrenia diagnosis | IEEE TBME, 2026 | Site-invariant brain-network learning (HSIC); related | IEEE (Amrita) | Check | Low |
| H8 | Ji et al., LBFC-MDG: learnable brain functional connectivity with multi-site domain generalization | ISBI 2026 | Meta-learning across sites on a GCN; related | IEEE (Amrita) | Check | Low |
| H9 | Lan et al., TopProNet: brain disease diagnosis via topological features and class prototypes | ISBI 2026 | Robustness to site shift; related | IEEE (Amrita) | Check | Low |
| H10 | He et al., DPGR-Net: disentangled population-guided GNN for MDD identification | Neural Networks, 2026 | Separates site from disease; reports 92.4% on REST-meta-MDD | Check | Check | Low |
| H11 | Zhang, Tuzhilina & Park, Sparse covariate-driven factorization of brain connectivity with application to site effect correction (SLACC) | Biostatistics 27, 2026 | Models localised site effects on ABIDE connectivity; alternative to ComBat for the site generator | Check (may be OA) | Check | **High** |
| H12 | Fang et al., Out-of-distribution-resistant evaluations for explanations of GNNs (OAR/OAR+) | IEEE TPAMI, 2026 | Explanation-evaluation metric; possible extra metric in the benchmark | IEEE (Amrita) | Check | **High** |
| H13 | Kurashkin et al., Evaluation rigor from GNNs to graph foundation models: a systematic review and four-axis reporting standard | Machine Learning and Knowledge Extraction, 2026 | Reporting checklist (seeds, significance tests, tuned trivial baselines, leakage). Follow it and say so | OA | n/a | **High** |
| H14 | Jeon, Systematic validation of GNN explanations against adverse outcome pathway reactive centers | Journal of Cheminformatics, 2026 | **Method to borrow:** a shuffled-label retraining control showed most explanation-to-ground-truth agreement came from explainer biases. Add this control to your benchmark | OA | Check | **High** |

Removed from earlier lists: C7 (DPGR-Net) and C8 (Zhao TMI) are now H10 and H6.

## Excluded: preprints (awareness only; recheck for a published version before submission)

| Paper | Why it matters | Status as of 28 Oct 2026 |
|---|---|---|
| EGS (Zhang, Betala & Agarwal, arXiv 2602.07708) | Closest conceptual neighbour | arXiv only |
| XIGL (Younesian et al., arXiv 2608.14121) | Decoy training with human feedback | arXiv only |
| CORE (Wang et al., arXiv 2605.06050) | Site deconfounding for brain networks | arXiv only |
| Artemis (Dai et al., arXiv 2606.18287) | Causal confounder removal | arXiv only |
| Patel et al., GNN connectome harmonisation (arXiv 2507.13992) | Site translation with GNNs | arXiv only |
| BAGEL (Rathee et al., arXiv 2206.13983) | Explainer benchmark | One repository lists ECML PKDD; not confirmed, so treated as preprint |
| BrainNNExplainer (Cui et al., arXiv 2107.05097) | Shared explanation mask | Workshop / arXiv only; cite IBGNN (B4) instead |
| Captum (Kokhlikyan et al., arXiv 2009.07896) | Explainer library | Cite as software |

## Reading order (about 25 items to get)

1. **Baselines with code** (decide what runs): B1, B3, B4, C4, C6, E2. Check each GitHub repo runs on AAL/CC200 inputs.
2. **Generator and planting recipe:** F3 Methods, F4, F2.
3. **Explainers and evaluation:** D1–D7.
4. **Motivation:** A3–A7.
5. **Datasets and tools:** G1–G9 (read the data sections only).
