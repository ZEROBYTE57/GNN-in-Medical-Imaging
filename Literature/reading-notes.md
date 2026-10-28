# Literature verification notes

Last updated 28 Oct 2026. Full list of papers to cite: [literature-list.md](literature-list.md).
Verdict = what the paper means for our novelty claims:
C1 = planted-ground-truth benchmark, C2 = ground-truth test of shortcut-robust brain GNNs, C3 = GNN vs simple-model explanations, C4 = SiteSwap.

## Status at a glance

- **Novelty:** holds after full-text checks of 11 papers and a 2026 search (14 new peer-reviewed papers, abstracts only). No paper plants known effects into real connectomes to test GNN explanations.
- **Claim wording:** "first to test GNN explanations with planted effects in real connectomes", never "first to plant effects in real data" (Beer 2020 and Reynolds 2023 did that for harmonisation).
- **Preprints in this folder** (EGS, XIGL): read for awareness only; do not cite in the literature review unless a peer-reviewed version appears.

## Batch 1: full text checked

| File | Paper | Venue | What it does | What it does not do | Code | Verdict |
|---|---|---|---|---|---|---|
| 1-s2.0-S1053811925004252-main.pdf | Chan et al., Discovering robust biomarkers of psychiatric disorders from rs-fMRI via GNNs: a systematic review | NeuroImage 319 (2025) 121422 | Reviews 65 GNN studies (ADHD, ASD, MDD, SZ, up to 9 Oct 2024). Finds salient features vary greatly across studies; evaluation is mostly cross-referencing the literature. Proposes the Hi-5 robustness framework. Suggests synthetic data with known ground truth for testing attributors. | No planted effects in real connectomes; no experiments of its own on attributor correctness | Data on OSF (osf.io/wza6b) | **Supports C1 strongly.** The "over 80%" figure from Consensus is NOT in the paper (82.1% is the share of studies using brain graphs). Do not quote it. |
| 2502.01688v1.pdf | Xu et al., BrainOOD | ICLR 2025 | First OOD benchmark for brain networks: unseen-site splits on ABIDE (1,025 subjects, 17 sites) and ADNI. GIB-based causal subgraph. | No ground truth, no synthetic data, no explanation metrics; interpretation judged by agreement with the literature | github.com/AngusMonroe/BrainOOD | **C1/C2 safe.** Baseline; reuse its ABIDE site splits. Cite as ICLR 2025, not arXiv. |
| 2602.07708v1.pdf | Zhang, Betala & Agarwal, EGS | **Preprint** (arXiv 2026) | Scores explainers by the OOD gain of explanation-guided retraining; 11,200 models, 8 explainers | No brain, medical or clinical graphs | Not found in PDF | **C1 safe.** Awareness only; could become an extra metric if published. |
| 2608.14121v1.pdf | Younesian et al., XIGL | **Preprint** (arXiv 2026) | Human-in-the-loop explanation feedback against shortcuts; decoys are coloured nodes (ER-color) and superpixel colours (CPatchMNIST) | No brain data; decoys generic, not realistic site effects | github.com/TYounesian/xilgraph (full code "upon acceptance") | **C4 safe.** Awareness only. Our difference: measured site decoys, no human feedback. |
| Liu et al. - 2024 - Rationalizing GNNs with Data Augmentation.pdf | Liu et al., RGDA | ACM TKDD 18(4), Art. 86 (Feb 2024) | Separates rationale and environment subgraphs; augments by swapping environments across a batch; citation, polymer and molecule data | No site model; no connectomes | Not found in PDF | **Closest method to C4.** Our difference: the swapped "environment" is a measured scanner-site signature. Cite as 2024. |
| 1-s2.0-S2213158223001638-main.pdf | Reynolds et al., ComBat: empirical vs fully Bayes | NeuroImage: Clinical 39 (2023) 103472 | Fully Bayesian ComBat; posterior samples augment data and improve AD classification; simulation plants disease effects into real ADNI features (after Beer 2020) | Structural volume/thickness only; no connectivity, GNNs or explanations | Not checked | **Precedent for the site generator and for planting.** Answers "planting is too artificial". |
| s44387-025-00067-x.pdf | Han et al., Rethinking functional brain connectome analysis | npj Artificial Intelligence (2026; accepted 31 Dec 2025) | ABIDE, PNC, HCP, ABCD: classical ML beats GDL; message aggregation degrades performance as graph density rises; dual-pathway linear + GAT model | No explanation ground truth | github.com/LearningKeqi/RethinkingBCA | **Motivation for C3.** Cite the journal version. |

## Batch 2: full text checked

| File | Paper | Venue | What it does | What it does not do | Code | Verdict |
|---|---|---|---|---|---|---|
| 2301.01642v3.pdf (+ CI-GNN-RESTRICTED.txt) | Zheng, Yu & Chen, CI-GNN | Neural Networks 172 (Apr 2024) 106147 | Built-in causal edge explanation (GraphVAE + conditional MI). ABIDE, REST-meta-MDD, SRPBS; AAL 116 ROIs, Pearson FC; 10-fold and leave-one-site-out (appendix) | Ground-truth test only on synthetic BA-2Motif. Quote: "Because the ground truth causal subgraph for real-world brain disease datasets are not available, the explanation performance is compared only on the synthetic dataset." | github.com/ZKZ-Brain/CI-GNN | **Supports C1/C2; use the quote in the introduction.** Baseline. Read from arXiv, cite the journal version. |
| BrainIB_...pdf | Zheng et al., BrainIB | IEEE TNNLS 36(7), July 2025, 13066 | Graph information bottleneck subgraph; ABIDE, REST-meta-MDD (1,604), SRPBS; AAL, Pearson, binarised adjacency; 10-fold and leave-one-site-out | No ground truth, no synthetic data; biomarkers judged "consistent with clinical and neuroimaging findings" | github.com/SJYuCNEL/brain-and-Information-Bottleneck | **C2 safe.** Main baseline. Cite as 2025. |
| 1-s2.0-S1053811920306157-main.pdf | Beer et al., Longitudinal ComBat | NeuroImage 220 (2020) | Simulation: 197 ADNI cognitively normal subjects randomly split into fake CN/AD groups; effects added to 6 of 62 cortical-thickness features, sizes estimated from real AD data (2 strong, 2 moderate, 2 weak); 1,000 iterations | Harmonisation test only; no ML, explanations or connectivity | R (lme4) | **Template and precedent for planting.** Borrow: effect sizes from real disease data. |
| New Text Document.txt (Yu et al. note, renamed) | Yu et al., ComBat for functional connectivity | Hum Brain Mapp 39 (2018) 4213 | ComBat removes site effects in FC and network measures (EMBARC, 4 sites); site-effect size depends on connectivity metric and atlas | Note covers abstract and introduction only | n/a | **Still need the Methods section** (per-edge ComBat, Fisher z or not). Suggest renaming the file back to "Yu et al paper.txt". |

## 2026 literature: abstracts only, not yet read

From the 28 Oct 2026 search (Consensus, 2026 only, preprints excluded). Full details and links: literature-list.md, section H.

| ID | Paper | Venue | Why it matters | Priority |
|---|---|---|---|---|
| H14 | Jeon, GNN explanations vs adverse outcome pathway reactive centres | J Cheminformatics 2026 | Shuffled-label retraining showed most explanation-to-ground-truth agreement came from explainer biases | **High: adopt the control** |
| H1 | Lee et al., GNNs for ASD on ABIDE-I and II | Psychiatry Res Neuroimaging 2026 | GCN/GAT/SAGPool on 2,013 subjects, CC200, ComBat: about 61% balanced accuracy | **High: realistic accuracy range** |
| H13 | Kurashkin et al., four-axis reporting standard for GNN evaluation | MAKE 2026 | Checklist: seeds, significance tests, tuned trivial baselines, leakage; only 2 of 25 core studies ran a significance test | **High: follow and cite** |
| H12 | Fang et al., OAR/OAR+ explanation evaluation | IEEE TPAMI 2026 | Explanation metric that avoids out-of-distribution inputs | **High: candidate metric** |
| H11 | Zhang, Tuzhilina & Park, SLACC site effect correction | Biostatistics 27, 2026 | Localised site effects on ABIDE connectivity; ComBat alternative | **High: justify generator choice** |
| H2 | Hu et al., BrainIB++ | BSPC 2026 | Newest information-bottleneck brain GNN; code public | **High: candidate baseline** |
| H6 | Zhao et al., unified graph domain adaptation, cross-site | IEEE TMI 2026 | Adversarial site alignment (like SiteSwap's site head) | Medium |
| H3 | Zheng et al., ensemble GNN for MDD subtyping | PLOS Digit Health 2026 | Leave-one-site-out vs BrainIB and CI-GNN; ComBat | Medium |
| H4 | Wang et al., JNEL-GCN for ADHD | IEEE TMI 2026 | Claims 97% on ADHD-200 and ABIDE-I (vs about 61% in H1); biomarkers checked against literature | Medium: cite neutrally |
| H5, H7–H10 | Xia (MedIA), Ji (TBME), Ji (ISBI), Lan (ISBI), He DPGR-Net (Neural Networks) | 2026 | Interpretable or site-robust brain GNNs; related work | Low |

## Design consequences (running list)

1. **Atlas:** CI-GNN and BrainIB use AAL (116 ROIs); Lee 2026 and BrainOOD use CC200-type parcellations. Proposal: AAL main, CC200 sensitivity (both in ABIDE Preprocessed). **Decision pending before week 1.**
2. **Effect sizes:** strong, moderate and weak planted effects estimated from real ABIDE ASD vs control edge differences (Beer 2020 approach).
3. **Site decoys:** report decoy strength per atlas (Yu 2018: site effects vary by atlas and metric). Mention SLACC (H11) as an alternative generator and why ComBat was chosen.
4. **New control (from H14):** retrain every model on shuffled labels and run the same explainers; explanation agreement above this baseline is what counts.
5. **Reporting (from H13):** 5 seeds, paired significance tests, tuned MLP / logistic regression / random forest baselines, leakage-free splits; state compliance in the paper.
6. **Metrics:** primary stays AUPRC of planted-edge recovery; consider OAR (H12) as a secondary metric if its code is available.
7. **Expected accuracy:** about 60–65% on real ABIDE is normal (H1, BrainOOD). Do not chase higher numbers; the benchmark measures explanations.

## Still to obtain

| Priority | Paper | Why needed |
|---|---|---|
| 1 | Yu et al. 2018, full PDF (Methods) | Recipe for the site-swap generator |
| 1 | H14 Jeon 2026 (open access) | Shuffled-label control design |
| 1 | H1 Lee 2026 | ABIDE GNN accuracy reference, CC200 + ComBat pipeline |
| 1 | H13 Kurashkin 2026 (open access) | Reporting checklist |
| 2 | H2 BrainIB++, H12 OAR (IEEE via Amrita), H11 SLACC | Baseline, metric, generator alternative |
| 2 | Sui et al., CAL (KDD 2022) / CAL+ (TKDD 2024) | Baseline; check code |
| 2 | B1 BrainGNN, B3 Brain Network Transformer, B4 IBGNN, C4 MetaExplainer, C6 CIA-GCL | Baselines; check whether code runs on AAL/CC200 |
| 3 | D1–D8 explainers and evaluation papers | Methods section |
| 3 | G1–G9 dataset, atlas and tool papers | Required citations |
