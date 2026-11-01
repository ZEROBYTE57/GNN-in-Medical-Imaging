# Literature verification notes

Last updated 1 Nov 2026. Full list of papers to cite, with the reading order: [literature-list.md](literature-list.md).
Verdict = what the paper means for our novelty claims:
C1 = planted-ground-truth benchmark, C2 = ground-truth test of shortcut-robust brain GNNs, C3 = GNN vs simple-model explanations, C4 = CF-Site.

## Status at a glance

- **Novelty:** holds after full-text checks of 11 papers and a 2026 search (14 new peer-reviewed papers, abstracts only). No paper plants known effects into real connectomes to test GNN explanations.
- **Claim wording:** "first to test GNN explanations with planted effects in real connectomes", never "first to plant effects in real data" (Beer 2020 and Reynolds 2023 did that for harmonisation).
- **Preprints in this folder** (EGS, XIGL): read for awareness only; do not cite in the literature review unless a peer-reviewed version appears.
- **1 Nov 2026 update:** list rebuilt for the final scope. Added and checked on OpenAlex: the planting and explanation-evaluation precedents closest to CXB (Adebayo et al. ICLR 2022, Zhou et al. AAAI 2022, CLEVR-XAI, Sanity checks), the saliency and GCN papers, BrainNetCNN, shortcut papers (Zech 2018, Brown 2023), ComBat pitfalls (Nygaard 2016, Wachinger 2021, Dinsdale 2021) and CF-Site's closest general methods (Gao et al. ICML 2023 targeted augmentations, Kaushik 2020, Ilse 2021). None plants effects into real connectomes, so the novelty claim stands. **Gao et al. 2023 is now the closest method to CF-Site** and must be named in the related work: CF-Site is targeted augmentation with ComBat's site model, applied to connectomes and judged by explanations.

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
| H2 | Hu et al., BrainIB++ | BSPC 2026 | Newest information-bottleneck brain GNN; code public | Related work (no new baselines; decided 1 Nov 2026) |
| H6 | Zhao et al., unified graph domain adaptation, cross-site | IEEE TMI 2026 | Adversarial site alignment (like CF-Site's site head) | Medium |
| H3 | Zheng et al., ensemble GNN for MDD subtyping | PLOS Digit Health 2026 | Leave-one-site-out vs BrainIB and CI-GNN; ComBat | Medium |
| H4 | Wang et al., JNEL-GCN for ADHD | IEEE TMI 2026 | Claims 97% on ADHD-200 and ABIDE-I (vs about 61% in H1); biomarkers checked against literature | Medium: cite neutrally |
| H5, H7–H10 | Xia (MedIA), Ji (TBME), Ji (ISBI), Lan (ISBI), He DPGR-Net (Neural Networks) | 2026 | Interpretable or site-robust brain GNNs; related work | Low |

## Design consequences (final, 1 Nov 2026)

1. **Atlas:** AAL for all experiments. CC200 is used only to replicate the real hospital-signal check (0.71 vs 0.66); one atlas for the benchmark is stated as scope.
2. **Effect sizes:** done as planned (Beer 2020 approach): x1, x2, x4 the real ABIDE autism effect.
3. **Site decoys:** sized from real ABIDE site differences; two versions (person-specific and ComBat's own model). SLACC (H11) supports a localised decoy.
4. **Shuffled-label control (from H14): done.** Explainers score at chance with shuffled labels (AUPRC 0.010-0.011; chance 0.008).
5. **Reporting (from H13):** 6 splits x 3 seeds (methods) and 3 x 3 (elsewhere); split-level bootstrap CI, sign test and paired t-test, Holm-corrected for the main family. Logistic regression baseline (C tuned on validation): **done** (0.98 without shortcut; drifts like the others).
6. **Metrics:** disease AUPRC (primary), scanner share, accuracy. OAR (H12) not used.
7. **Expected accuracy:** real ABIDE leave-site-out accuracy about 0.60–0.65 (ours: MLP 0.632, GCN 0.597), in line with H1 and BrainOOD.
8. **Models (1 Nov):** reimplemented from their papers: BrainNetCNN (B8) passes the sanity test and is used; BrainGNN (B1) and Brain Network Transformer (B3) fail it at about 250 training subjects and are reported as a sample-size caution. ContrastPool (B5) not used.

## Still to obtain (Tier 1 of the reading order first)

| Priority | Paper | Why needed |
|---|---|---|
| 1 | D11 Adebayo 2022, D12 Zhou 2022, D5 GraphXAI | Closest prior work to the benchmark; related-work section |
| 1 | B1 BrainGNN, B8 BrainNetCNN, B3 BNT | Needed to reimplement the models |
| 1 | E6 Gao 2023 | Closest method to CF-Site |
| 1 | F1 Johnson 2007, F3 Yu 2018 (Methods), F9 Nygaard 2016 | ComBat and its pitfalls |
| 1 | D9 Simonyan, D3 IG, D1 GNNExplainer, D7 Faber | Explainers and the main objection |
| 2 | A4, A9, A10, A7, F10, F11, E5, E7, D10, D13, D8, H1, H13, H14 | Supporting citations |
| 3 | G1, G2, G5, G9, G10, B9 and the rest of Tier 3 | Cite only |
