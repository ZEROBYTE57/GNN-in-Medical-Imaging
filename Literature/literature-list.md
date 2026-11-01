# Literature list for the CXB paper

Paper: "Do brain GNN explanations find the disease or the scanner?" (CXB benchmark + CF-Site).
First built 28 Oct 2026; rebuilt 1 Nov 2026 for the final scope (planting, explainers, published brain models, site effects).

**Rules**
- Peer-reviewed only, except the preprint list at the end (awareness only, or cited clearly as a preprint).
- Every entry with a DOI was checked on OpenAlex on 1 Nov 2026 (first author, venue, year, volume, first page, DOI); "ff" = last page not yet checked, take it from the publisher page when writing the bibliography. Entries without a DOI are carried over from the 28 Oct check. "Cites" = OpenAlex citation count on that date (a rough guide to standing, not quality).
- If you read an arXiv or author copy, cite the **published** version given here.
- IDs from the 28 Oct list are kept, so older notes still match; new papers have new IDs.
- **Reading order is at the end.**

Access: **OA** = free at the publisher · **Proc.** = free conference proceedings · **IEEE/Amrita** = via the Amrita library · **Author copy** = free accepted version on arXiv/PMC/bioRxiv · **Check** = try the library first.

## A. The problem: brain-GNN explanations are judged without a known answer

| # | Reference | Cites | Use in the paper | Access | Status |
|---|---|---|---|---|---|
| A1 | Chan et al., Discovering robust biomarkers of psychiatric disorders from resting-state fMRI via graph neural networks: a systematic review. *NeuroImage* 319:121422, 2025. doi:10.1016/j.neuroimage.2025.121422 | 14 | **Key motivation:** 65 brain-GNN studies; explanations checked only against the literature | OA | Have |
| C2 | Zheng, Yu & Chen, CI-GNN: a Granger causality-inspired graph neural network for interpretable brain network-based psychiatric diagnosis. *Neural Networks* 172:106147, 2024 | – | **Quote for the introduction:** real brain data has no ground-truth subgraph, so explanations were scored only on synthetic graphs | Author copy | Have |
| A2 | Han et al., Rethinking functional brain connectome analysis: do graph deep learning models help? *npj Artificial Intelligence* 2, 2026. doi:10.1038/s44387-025-00067-x | 8 | Simple models match or beat GNNs on connectomes; supports keeping the MLP | OA | Have |
| A3 | Mueller et al., Are population graphs really as powerful as believed? *TMLR*, 2024 | – | GNNs vs simple baselines | Proc. | Optional |
| A8 | Mienye & Viriri, Graph neural networks in medical imaging: methods, applications and future directions. *Information* 16, 2025 | – | One background sentence on GNNs in imaging | OA | Have (Back Up/) |

## P. Planting a known answer into real data (the benchmark's method)

| # | Reference | Cites | Use in the paper | Access | Status |
|---|---|---|---|---|---|
| F5 | Beer et al., Longitudinal ComBat: a method for harmonizing longitudinal multi-scanner imaging data. *NeuroImage* 220:117129, 2020. doi:10.1016/j.neuroimage.2020.117129 | 323 | **Template:** fake patient/control split of real controls, effects sized from real disease data | OA | Have |
| F6 | Reynolds et al., ComBat harmonization: empirical Bayes versus fully Bayes approaches. *NeuroImage: Clinical* 39:103472, 2023. doi:10.1016/j.nicl.2023.103472 | 37 | Precedent: planted disease effects in real ADNI features; also on EB ComBat | OA | Have |
| D12 | Zhou, Booth, Ribeiro & Shah, Do feature attribution methods correctly attribute features? *AAAI* 36(9):9623ff, 2022. doi:10.1609/aaai.v36i9.21196 | 109 | **Closest idea outside the brain:** planting known features into real images to score attribution methods ("semi-natural" data) | OA | Get |
| D11 | Adebayo, Muelly, Abelson & Kim, Post hoc explanations may be ineffective for detecting unknown spurious correlation. *ICLR*, 2022 | 28 | **Very close:** plants spurious signals and asks whether explanations reveal them. Our scanner decoy is the brain-connectome version | Proc. | Get |
| D13 | Arras, Osman & Samek, CLEVR-XAI: a benchmark dataset for the ground truth evaluation of neural network explanations. *Information Fusion* 81:14ff, 2022. doi:10.1016/j.inffus.2021.11.008 | 183 | Ground-truth evaluation of explanations (synthetic images) | Author copy | Get |
| D5 | Agarwal et al., Evaluating explainability for graph neural networks (GraphXAI / ShapeGGen). *Scientific Data* 10:144, 2023. doi:10.1038/s41597-023-01974-x | 186 | **Closest graph benchmark:** ground-truth GNN explanations on synthetic graphs only | OA | Get |

## D. Explainers and how explanations are evaluated

| # | Reference | Cites | Use in the paper | Access | Status |
|---|---|---|---|---|---|
| D9 | Simonyan, Vedaldi & Zisserman, Deep inside convolutional networks: visualising image classification models and saliency maps. *ICLR Workshop*, 2014 | 5365 | **Explainer used:** saliency (gradients) | Proc. | Get |
| D3 | Sundararajan, Taly & Yan, Axiomatic attribution for deep networks (Integrated Gradients). *ICML*, PMLR 70, 2017 | 3934 | **Explainer used** | Proc. | Get |
| D1 | Ying et al., GNNExplainer: generating explanations for graph neural networks. *NeurIPS*, 2019 | 850 | **Explainer used** in simplified form ("GNNExplainer-style input mask"); name it that way | Proc. | Get |
| D10 | Adebayo et al., Sanity checks for saliency maps. *NeurIPS* 31, 2018 | 1293 | Explanations can look plausible yet ignore the model; motivates ground-truth tests | Proc. | Get |
| D7 | Faber, Moghaddam & Wattenhofer, When comparing to ground truth is wrong: on evaluating GNN explanation methods. *KDD*, pp. 332ff, 2021. doi:10.1145/3447548.3467283 | 41 | **Answer it in the Discussion:** a model may solve the task without the planted signal; our t-test reference, accuracy checks and shuffled-label control address this | Author copy | Get |
| D8 | Yuan, Yu, Gui & Ji, Explainability in graph neural networks: a taxonomic survey. *IEEE TPAMI* 45, 2023. doi:10.1109/tpami.2022.3204236 | 594 | Explainer taxonomy for the related-work section | IEEE/Amrita | Get |
| D6 | Agarwal, Zitnik & Lakkaraju, Probing GNN explainers: a rigorous theoretical and empirical analysis. *AISTATS*, PMLR 151, 2022 | – | Faithfulness and stability metrics | Proc. | Optional |
| H14 | Jeon, Systematic validation of GNN explanations against adverse outcome pathway reactive centers. *Journal of Cheminformatics*, 2026 | – | Shuffled-label control: much ground-truth agreement came from explainer bias | OA | Get |
| H12 | Fang et al., Out-of-distribution-resistant evaluations for explanations of GNNs (OAR/OAR+). *IEEE TPAMI*, 2026 | – | Alternative explanation metric | IEEE/Amrita | Optional |

## M. Models: GNNs and published brain-network models

| # | Reference | Cites | Role in the paper | Access | Status |
|---|---|---|---|---|---|
| B9 | Kipf & Welling, Semi-supervised classification with graph convolutional networks. *ICLR*, 2017 | 23978 | **Used:** our GCN's layer | Proc. | Get |
| B11 | Veličković, Cucurull, Casanova, Romero, Liò & Bengio, Graph attention networks. *ICLR*, 2018 | – | GAT code is ready but **not in the paper** (held in reserve in case a reviewer asks for a second GNN) | Proc. | Optional |
| B2 | Cui et al., BrainGB: a benchmark for brain network analysis with graph neural networks. *IEEE TMI* 42:493ff, 2023. doi:10.1109/tmi.2022.3218745 | 218 | **Used:** our GCN follows its design (node features = connectivity rows, concatenation readout) | IEEE/Amrita | Get |
| B1 | Li et al., BrainGNN: interpretable brain graph neural network for fMRI analysis. *Medical Image Analysis* 74:102233, 2021. doi:10.1016/j.media.2021.102233 | 831 | **Reimplemented** from the paper; fails our sanity test at about 250 training subjects (reported). Its own region scores are a built-in explanation to score | Author copy (bioRxiv) | Get |
| B8 | Kawahara et al., BrainNetCNN: convolutional neural networks for brain networks; towards predicting neurodevelopment. *NeuroImage* 146:1038ff, 2017. doi:10.1016/j.neuroimage.2016.09.046 | 813 | **Used** (reimplemented from the paper); the classic connectome-specific model | Author copy | Get |
| B3 | Kan, Dai, Cui, Zhang, Guo & Yang, Brain Network Transformer. *NeurIPS*, pp. 25586ff, 2022. doi:10.52202/068431-1855 | 160 | **Reimplemented;** fails our sanity test (reported). The standard transformer baseline; attention as a built-in explanation | Proc. | Get |
| B5 | Xu et al., Contrastive graph pooling for explainable classification of brain networks (ContrastPool). *IEEE TMI* 43:3292ff, 2024. doi:10.1109/tmi.2024.3392988 | 55 | Related work: a recent explainable brain GNN (not used) | IEEE/Amrita | Optional |
| B4 | Cui et al., Interpretable graph neural networks for connectome-based brain disorder analysis (IBGNN). *MICCAI*, LNCS, pp. 375ff, 2022. doi:10.1007/978-3-031-16452-1_36 | 114 | Related work: interpretable brain GNN | Author copy | Optional |
| B10 | Yu et al., Long-range brain graph transformer (ALTER). *NeurIPS*, pp. 24472ff, 2024. doi:10.52202/079017-0771 | 20 | Related work; alternative to ContrastPool | Proc. | Optional |
| B6 | Zheng et al., BPI-GNN: interpretable brain network-based psychiatric diagnosis and subtyping. *NeuroImage* 293, 2024 | – | Related work (prototype explanations) | OA | Optional |
| B7 | Wei et al., A brain graph foundation model (BrainGFM). *ICLR*, 2026 | – | One sentence on the newest direction | Proc. | Optional |
| H1 | Lee et al., Graph neural network-based modeling of functional brain connectivity in autism spectrum disorder using the ABIDE-I and II datasets. *Psychiatry Research: Neuroimaging*, 2026 | – | Realistic ABIDE accuracy range (GCN/GAT/SAGPool, about 61% balanced accuracy) | Check | Get |

## S. Shortcuts and site effects

| # | Reference | Cites | Use in the paper | Access | Status |
|---|---|---|---|---|---|
| A4 | Geirhos et al., Shortcut learning in deep neural networks. *Nature Machine Intelligence* 2:665ff, 2020. doi:10.1038/s42256-020-00257-z | 2227 | Defines shortcut learning | Author copy | Get |
| A9 | Zech et al., Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs. *PLOS Medicine* 15:e1002683, 2018. doi:10.1371/journal.pmed.1002683 | 1702 | **Hospital as a shortcut** in medical imaging | OA | Get |
| A5 | DeGrave, Janizek & Lee, AI for radiographic COVID-19 detection selects shortcuts over signal. *Nature Machine Intelligence* 3:610ff, 2021. doi:10.1038/s42256-021-00338-7 | 631 | Shortcut found only through explanations | Author copy | Get |
| A10 | Brown et al., Detecting shortcut learning for fair medical AI using shortcut testing. *Nature Communications* 14:4314, 2023. doi:10.1038/s41467-023-39902-7 | 107 | Varying the shortcut strength to test reliance, like our 50–90% drift experiment | OA | Get |
| A7 | Abraham et al., Deriving reproducible biomarkers from multi-site resting-state data: an autism-based example. *NeuroImage* 147:736ff, 2017. doi:10.1016/j.neuroimage.2016.10.045 | 701 | ABIDE site effects; leave-site-out testing | Author copy | Get |
| F10 | Wachinger et al., Detect and correct bias in multi-site neuroimaging datasets. *Medical Image Analysis* 67:101879, 2021. doi:10.1016/j.media.2020.101879 | 115 | Site bias in multi-site neuroimaging; harmonisation as a fix | Author copy | Get |
| F11 | Dinsdale, Jenkinson & Namburete, Deep learning-based unlearning of dataset bias for MRI harmonisation and confound removal. *NeuroImage* 228:117689, 2021. doi:10.1016/j.neuroimage.2020.117689 | 178 | Adversarial scanner unlearning; named as future comparison | OA | Get |
| F12 | Tian et al., A deep learning-based multisite neuroimage harmonization framework established with a traveling-subject dataset. *NeuroImage*, 2022 | – | Learned harmonisation from travelling subjects; a possible better scanner model for CF-Site (future work) | OA | Optional |
| F13 | Hu et al., DeepComBat: a statistically motivated, hyperparameter-robust, deep learning approach to harmonization of neuroimaging data. *Human Brain Mapping*, 2024 (PMC11273293) | – | ComBat inside a conditional VAE; a possible better scanner model for CF-Site (future work) | OA | Optional |

## F. Harmonisation (ComBat and its pitfalls)

| # | Reference | Cites | Use in the paper | Access | Status |
|---|---|---|---|---|---|
| F1 | Johnson, Li & Rabinovic, Adjusting batch effects in microarray expression data using empirical Bayes methods (ComBat). *Biostatistics* 8:118ff, 2007. doi:10.1093/biostatistics/kxj037 | 8663 | **ComBat** (standard EB version used) | Check | Get |
| F2 | Fortin et al., Harmonization of multi-site diffusion tensor imaging data. *NeuroImage* 161:149ff, 2017. doi:10.1016/j.neuroimage.2017.08.047 | 1261 | ComBat in neuroimaging | Author copy (PMC) | Get |
| F3 | Yu et al., Statistical harmonization corrects site effects in functional connectivity measurements from multi-site fMRI data. *Human Brain Mapping* 39:4213ff, 2018. doi:10.1002/hbm.24241 | 458 | **ComBat on functional connectivity** (our setting) | Author copy (PMC) | Partial |
| F9 | Nygaard, Rødland & Hovig, Methods that remove batch effects while retaining group differences may lead to exaggerated confidence in downstream analyses. *Biostatistics* 17:29ff, 2016. doi:10.1093/biostatistics/kxv027 | 379 | **Explains our label-protected ComBat result** (inflated group differences) | OA | Get |
| F4 | Chen et al., Mitigating site effects in covariance for machine learning in neuroimaging data (CovBat). *Human Brain Mapping* 43, 2022 | – | Alternative harmonisation (limitations) | OA | Optional |
| F8 | Xin et al., ComBat-Predict enhances generalizability of neuroimaging models to new sites. *Human Brain Mapping* 47, 2026 | – | ComBat for unseen sites | OA | Optional |
| H11 | Zhang, Tuzhilina & Park, Sparse covariate-driven factorization of brain connectivity with application to site effect correction (SLACC). *Biostatistics* 27, 2026 | – | Localised site effects in ABIDE connectivity; supports our localised decoy | Check | Optional |

## W. CF-Site's neighbours: augmentation against spurious features

| # | Reference | Cites | Use in the paper | Access | Status |
|---|---|---|---|---|---|
| E5 | Kaushik, Hovy & Lipton, Learning the difference that makes a difference with counterfactually-augmented data. *ICLR*, 2020 | 253 | Counterfactual augmentation: change the spurious part, keep the label | Proc. | Get |
| E6 | Gao, Sagawa, Koh, Hashimoto & Liang, Out-of-domain robustness via targeted augmentations. *ICML*, PMLR 202, 2023 | – | **Closest general method:** randomise domain-dependent features, keep the label. CF-Site is this idea with ComBat's site model | Proc. | Get |
| E7 | Ilse, Tomczak & Forré, Selecting data augmentation for simulating interventions. *ICML*, PMLR 139, 2021 | 35 | Augmentation as an intervention on the spurious variable (theory) | Proc. | Get |
| E8 | Wang, Chaudhari & Davatzikos, Harmonization with flow-based causal inference. *MICCAI* (LNCS 12903):181ff, 2021. doi:10.1007/978-3-030-87199-4_17 | – | **Counterfactual harmonisation:** site counterfactuals from a causal normalising flow, used to remove site effects | Proc. | Get |
| E9 | Vigneshwaran, Wilms, Camacho, Souza & Forkert, Improved multi-site Parkinson's disease classification using neuroimaging data with counterfactual inference. *MIDL*, PMLR 227:1304ff, 2023 | – | **Closest to CF-Site:** trains a classifier on site counterfactuals (causal flow) and gains accuracy on an unseen site. CF-Site differs: ComBat counterfactuals, connectomes, and judged by explanations against a ground truth | Proc. (OA) | Must read |
| E10 | Pawlowski, Coelho de Castro & Glocker, Deep structural causal models for tractable counterfactual inference. *NeurIPS*, 2020 | – | Foundation of image counterfactuals in medical imaging | Proc. | Optional |
| E4 | Liu et al., Rationalizing graph neural networks with data augmentation (RGDA). *ACM TKDD* 18(4), 2024. doi:10.1145/3638781 | 15 | Closest graph method: swaps environment subgraphs | Check | Have |
| E2 | Sui et al., Causal attention for interpretable and generalizable graph classification (CAL). *KDD*, pp. 1696ff, 2022. doi:10.1145/3534678.3539366 | 195 | Shortcut-feature mixing on graphs | Author copy | Optional |
| E1 | Wu et al., Discovering invariant rationales for graph neural networks (DIR). *ICLR*, 2022 | – | Invariant rationales; related | Proc. | Optional |

## R. Site-robust and interpretable brain GNNs (related work)

| # | Reference | Use in the paper | Access | Status |
|---|---|---|---|---|
| C3 | Xu et al., BrainOOD: out-of-distribution generalizable brain network analysis. *ICLR*, 2025 | **Our leave-site-out folds come from it**; related site-shift method | Proc. | Have |
| C1 | Zheng et al., BrainIB: interpretable brain network-based psychiatric diagnosis with graph information bottleneck. *IEEE TNNLS* 36(7), 2025 | Base paper of the dissertation; information-bottleneck explanations | IEEE/Amrita | Have |
| C4 | Qiu et al., MetaExplainer: domain generalization of functional connectome analyses from the perspective of explainability. *Medical Image Analysis* 105, 2025 | Explanations under site shift | Check | Optional |
| C5 | Qiu et al., Towards GNNs with domain-generalizable explainability for fMRI-based brain disorder diagnosis (XG-GNN). *MICCAI*, LNCS, pp. 454–464, 2024 | Earlier version of C4 | Check | Optional |
| C6 | Yu, Liu & Ji, Causal invariance-aware augmentation for brain graph contrastive learning (CIA-GCL). *ICML*, PMLR 267, 2025 | Site-invariant brain subgraph | Proc. | Optional |
| H6 | Zhao et al., A unified graph domain adaptation framework for cross-site brain network analysis. *IEEE TMI*, 2026 | Adversarial cross-site alignment | IEEE/Amrita | Optional |
| H2 | Hu et al., BrainIB++. *Biomedical Signal Processing and Control*, 2026 | Newest information-bottleneck brain GNN | Check | Optional |
| H3, H5, H7–H10 | Zheng (PLOS Digit Health), Xia (MedIA), Ji (TBME), Ji (ISBI), Lan (ISBI), He DPGR-Net (Neural Networks), all 2026 | One sentence of related work at most | – | Optional |
| H4 | Wang et al., JNEL-GCN for ADHD. *IEEE TMI*, 2026 | Reports 97% accuracy with literature-checked biomarkers; cite neutrally as an example of why ground-truth checks are needed | IEEE/Amrita | Optional |

## G. Data, atlas, software and reporting (must cite)

| # | Reference | Cites | Why | Access |
|---|---|---|---|---|
| G1 | Di Martino et al., The autism brain imaging data exchange: towards a large-scale evaluation of the intrinsic brain architecture in autism. *Molecular Psychiatry* 19:659ff, 2014. doi:10.1038/mp.2013.78 | 2780 | Required by ABIDE's terms | Author copy (PMC) |
| G2 | Craddock et al., The Neuro Bureau Preprocessing Initiative: open sharing of preprocessed neuroimaging data and derivatives. *Frontiers in Neuroinformatics*, 2013 (conference abstract) | – | Required for ABIDE Preprocessed (PCP) | OA |
| G5 | Tzourio-Mazoyer et al., Automated anatomical labeling of activations in SPM using a macroscopic anatomical parcellation of the MNI MRI single-subject brain. *NeuroImage* 15:273ff, 2002. doi:10.1006/nimg.2001.0978 | 16319 | AAL atlas | Check |
| G11 | Craddock, James, Holtzheimer, Hu & Mayberg, A whole brain fMRI atlas generated via spatially constrained spectral clustering. *Human Brain Mapping* 33:1914ff, 2012. doi:10.1002/hbm.21333 | – | **Used:** CC200 atlas (second-atlas replication) | Paywalled | Get |
| G9 | Abraham et al., Machine learning for neuroimaging with scikit-learn (nilearn). *Frontiers in Neuroinformatics* 8, 2014 | – | Data download and connectivity | OA |
| G10 | Paszke et al., PyTorch: an imperative style, high-performance deep learning library. *NeurIPS*, 2019 | – | Software | Proc. |
| A6 | Rosenblatt et al., Data leakage inflates prediction performance in connectome-based machine learning models. *Nature Communications* 15:1829, 2024. doi:10.1038/s41467-024-46150-w | 236 | Supports our leakage-free protocol (ComBat fitted on training data only) | OA |
| H13 | Kurashkin et al., Evaluation rigor from GNNs to graph foundation models: a systematic review and four-axis reporting standard. *Machine Learning and Knowledge Extraction*, 2026 | – | Reporting checklist (seeds, significance tests, simple baselines, leakage); say we follow it | OA |

All models are plain PyTorch code of our own, so PyTorch Geometric (G8) is not cited.

## Preprints (awareness only; recheck for a published version before submission)

| Paper | Why it matters | Status (1 Nov 2026) |
|---|---|---|
| CFBrainAug (arXiv 2610.32184): modularity-induced counterfactual brain-network augmentation for interpretable autism diagnosis | Closest brain-network augmentation work (for interpretability, not scanner effects). **Cite as a preprint** in related work, because a reviewer will know it | arXiv only |
| CORE, "When brain networks travel: learning beyond site" (arXiv 2605.06050) | Site deconfounding for brain networks | arXiv only |
| EGS (Zhang, Betala & Agarwal, arXiv 2602.07708) | Scores explainers by retraining gains | arXiv only |
| XIGL (Younesian et al., arXiv 2608.14121) | Decoy training with human feedback | arXiv only |
| Artemis (arXiv 2606.18287), Patel et al. (arXiv 2507.13992), BAGEL (arXiv 2206.13983), BrainNNExplainer (arXiv 2107.05097) | Related; cite IBGNN (B4) instead of BrainNNExplainer | arXiv only |

## No longer used (kept for the record)

D2 PGExplainer and D4 GSAT (no learned edge masks in the final method), E3 CAL+ (journal version of E2), F7 Wang 2023 harmonisation review, G3–G4 ADHD-200 (no replication dataset), G6 (replaced by G11: CC200 is used only for the hospital-signal replication) and G7 Yeo-7 (CF-Site uses no network prior), Captum (our explainers are our own code).

## Reading order for the paper

Read in this order. Tier 1 is what the paper cannot be written without; for Tier 3, abstract and the relevant section are enough.

**Tier 1: core (about 16 papers; read fully)**

| Order | Paper | Why first | Paper section |
|---|---|---|---|
| 1 | A1 Chan 2025 | The gap: brain-GNN explanations are judged only against the literature | Introduction |
| 2 | C2 CI-GNN (the quote) | Authors admit there is no ground truth for real brain data | Introduction |
| 3 | F5 Beer 2020 | The planting template we follow | Methods: planting |
| 4 | D11 Adebayo 2022 | Closest experiment: planted spurious signals vs explanations | Related work, Discussion |
| 5 | D12 Zhou 2022 | Planting known features into real data to score attributions | Related work |
| 6 | D5 GraphXAI 2023 | Closest graph benchmark (synthetic only) | Related work |
| 7 | D9 Simonyan 2014, D3 IG 2017, D1 GNNExplainer 2019 | The three explainers we use | Methods: explainers |
| 8 | D7 Faber 2021 | The main objection to ground-truth evaluation; answer it | Discussion |
| 9 | B2 BrainGB 2023 | How our GCN is built; the standard brain-GNN pipeline | Methods: models |
| 10 | B1 BrainGNN 2021 | Reimplemented; fails the sanity test (report why) | Methods: models, Scope |
| 11 | B8 BrainNetCNN 2017 | Used as the second architecture family | Methods: models |
| 12 | B3 BNT 2022 | Reimplemented; fails the sanity test | Methods: models, Scope |
| 13 | F1 Johnson 2007 + F3 Yu 2018 | ComBat, and ComBat on connectivity | Methods: ComBat, CF-Site |
| 14 | F9 Nygaard 2016 | Why label-protected ComBat inflates results | Results, Discussion |
| 15 | E6 Gao 2023 | Closest general method to CF-Site; state the difference | Related work: CF-Site |
| 16 | E9 Vigneshwaran 2023 (+ E8 Wang 2021) | Closest neuroimaging work: training on site counterfactuals; state the difference (we judge explanations against a ground truth) | Related work: CF-Site |

**Tier 2: support (read the key sections)**

A4 Geirhos 2020, A9 Zech 2018, A10 Brown 2023 (shortcuts) · A7 Abraham 2017, C3 BrainOOD (ABIDE sites, our folds) · E5 Kaushik 2020, E7 Ilse 2021, E4 RGDA 2024 (augmentation) · F11 Dinsdale 2021, F10 Wachinger 2021 (scanner effects) · D10 Adebayo 2018, D13 CLEVR-XAI, D8 Yuan survey (explanation evaluation) · A2 Han 2026, H1 Lee 2026 (realistic ABIDE accuracy; simple models) · CFBrainAug preprint.

**Tier 3: cite only (abstract enough)**

B9 Kipf 2017, G11 Craddock 2012, E10 Pawlowski 2020, F12 Tian 2022, F13 Hu 2024, G1, G2, G5, G9, G10, A6 Rosenblatt 2024, H13 Kurashkin 2026, F2 Fortin 2017, F6 Reynolds 2023, B4 IBGNN, B10 ALTER, C1 BrainIB, C4–C6, H2–H12, E1, E2, D6.
