# Literature review (curated, Scopus-targeted) - retinal vessel graph study

Date: 2026-10-23. Supersedes the "Reading list" section of reading-list-and-data.md (that file stays as the raw list and data check).
Built from: problem-statement-v3.md, novelty-check-2026-10-11.md, vga-net-paper-audit.md, reading-list-and-data.md, the 4 project PDFs, and new Consensus searches on 2026-10-23.

## 1. Selection rule (what "helpful for my paper" means)
A paper stays only if it does at least one of these jobs in OUR paper:
- **Motivate** a research question (RQ1 leakage, RQ2 graph vs no-edge control on topology, RQ3 robustness/shift/label noise).
- **Define a method or metric** we actually use (clDice, Betti errors, corruption protocol, split protocol, CIs).
- **Be the closest prior work** we must cite and differentiate from.
- **Supply a template** for how to report a controlled/negative result.
Everything else is cut (section 6). Venue rule: journal/proceedings that appear in Scopus (checked via SCImago, which only lists Scopus sources). Preprints are for awareness only.

## 2. At a glance
| Tier | Count | Meaning |
|---|---|---|
| Core | 15 | Read fully, cite, and differentiate. |
| Supporting | 17 | Cite in one sentence each; skim for numbers or protocol details. |
| Cut | see section 6 (about 20 items, several grouped) | Do not cite (reasons in section 6). |

## 3. Review by theme (this is the skeleton of the paper's Related Work)

### A. Graph reasoning for retinal vessel segmentation -> motivates RQ2
Graph modules are now a common add-on to U-Net/Transformer backbones: connectivity graphs from vessel structure (Shin 2019), dual-encoder dynamic-channel GCN (Li 2022), EdgeConv GCN cascaded with U-Net (Wang 2024), GNN-guided ViT (Xu 2024), skeleton-keypoint graph with GAT in TransUNet skips (Qiu 2024), graph-conv attention with ViT (Li 2025), and patch-grid GAT in VGA-Net (Jalali 2024). Common claims: better vessel continuity and robustness. Common evidence: Dice/Acc/Se/Sp/AUC, single runs, no parameter-matched no-edge control, no stress tests.
**Gap we fill:** controlled test of whether the graph edges themselves matter (RQ2).

### B. Topology-aware metrics and losses -> defines RQ2 endpoints
clDice (Shit 2021) is the standard centreline metric/loss. Topology-aware add-ons for retinal vessels exist (Su 2024 connectivity booster; Tan 2022 skeletal prior with connectivity metrics; Song 2025 morphological correction). Betti-number errors and Betti matching (Stucki 2023) give a topology count that overlap metrics miss.
**Use:** clDice + Betti-0/1 errors as primary endpoints, not Dice.

### C. Evaluation protocol and data leakage -> motivates RQ1
Leakage inflates scores across medical ML (Kapoor 2023 survey of fields; Kannoth 2024 slice-level MRI up to 67%; Tampu 2022 and Marciniak 2025 for retinal OCT; Wen 2020 for Alzheimer's MRI). For retinal vessels, Kovacs 2021 audited FoV-mask handling in 100+ papers - a different flaw. Our Consensus searches (2026-10-23) again found no study that quantifies patch-level vs image-level split inflation for retinal vessel segmentation. Jamaludin 2026 (IEEE Access) is the reporting template: leakage-prone vs verified split, 5 seeds, Wilson CIs, parameter-matched control, framed as a controlled negative result.
**Gap we fill:** RQ1 quantification on DRIVE (open claim, not proven absent).

### D. Cross-dataset shift and domain generalisation -> motivates RQ3 (shift)
Galdran 2022 shows a minimalistic U-Net nearly matches SOTA and that cross-dataset transfer is far from solved (10 databases). Domain-generalisation methods target the same problem by augmentation or feature design (Lyu 2022 AADG; Hu 2024 Hessian vector field; Liu 2025 DGSSA). Su 2022 gives empirical DRIVE<->STARE transfer. GraphSeg (NeurIPS 2025) pairs a graph prior with cross-dataset testing but without topology metrics or error bars (per our earlier check).
**Gap we fill:** does graph reasoning help topology under zero-shot transfer, with CIs.

### E. Corruption robustness -> motivates RQ3 (degradation)
The protocol of severity-graded corruptions comes from Hendrycks & Dietterich (2019); medical-imaging use for retina: Wang 2023 (curriculum Fourier adaptation, corruption types with increasing severity). The 2026-10-23 search for severity-controlled corruption benchmarks on retinal vessels returned mostly augmentation/architecture papers claiming "robustness" without a corruption sweep (Sun 2020 is the closest and is a preprint).
**Gap we fill:** severity-controlled corruption sweep with topology metrics (largely missing, not proven absent).

### F. Annotation noise -> motivates RQ3 (labels)
Huang 2024 (Meta-tubular-net) synthesises vascular label noise with topology-aware reweighting; Xu 2022 handles real noisy labels; Li 2026 (Frontiers in Medicine) studies positional label noise on DRIVE. None test graph vs no-edge.
**Gap we fill:** positional noise sweep applied to the graph vs control comparison.

## 4. Paper tables

Status codes for venue: **V-now** = SCImago listing seen this session; **V-prev** = checked in earlier session (SCImago, 2025 quartile in reading-list doc); **unchecked** = widely known Scopus source but not individually re-checked (do the 2-minute check before submission).

### 4.1 Core (15)
| # | Paper | Venue / year | Scopus status | Role in our paper | Extract while reading |
|---|---|---|---|---|---|
| 1 | Jalali et al., VGA-Net | IET Image Processing 2024 | V-prev (Q2) | Named baseline; RQ1/RQ2 motivation | Split protocol wording, graph construction, params (audit doc has the issues list) |
| 2 | Li Y. et al., Dual encoder dynamic-channel GCN | IEEE TMI 2022 | unchecked | Alternative graph baseline (Option B) | Graph construction, code availability, datasets, whether any topology metric is reported |
| 3 | Wang Z. et al., PCAM-guided GCN cascaded U-Net | Computers in Biology and Medicine 2024 | V-prev (Q1) | Closest "graph + robustness claim" | What exactly is called robustness; datasets; any no-graph ablation |
| 4 | Xu H. et al., G2ViT | Neural Networks 2024 | unchecked | Graph-reasoning prior art, claims robustness | Ablation without graph module; metrics used |
| 5 | Qiu Y. et al., GKE-TUNet (NEW) | IEEE JBHI 2024 | V-now (journal listed) | Graph on skeleton keypoints with topology focus; closest to "graph + topology" | Whether clDice/Betti reported; ablation of graph |
| 6 | Shit et al., clDice | CVPR 2021 | conference: verify in Scopus | Defines primary metric | Exact formula; soft-skeleton details for implementation check |
| 7 | Stucki et al., Betti matching | ICML 2023 | conference: verify in Scopus (see note) | Betti-error definition | Definition and caveats of Betti error vs matching |
| 8 | Kovacs et al., new baseline for retinal vessels | Medical Image Analysis 2021 | unchecked | Evaluation-flaw audit precedent; FoV handling | Which flaws they list; what they say about splits |
| 9 | Kapoor & Narayanan, Leakage and reproducibility | Patterns 2023 | unchecked | General leakage taxonomy and citation anchor for RQ1 | Taxonomy categories to name our leakage type |
| 10 | Jamaludin et al., leakage audit hybrid quantum-classical | IEEE Access 2026 | V-prev (Q1) | Reporting template for a controlled negative result | Table layout, CI method, seed count, how they phrase the null |
| 11 | Galdran et al., minimalistic models | Scientific Reports 2022 | unchecked | Why a plain U-Net is the strong control; cross-dataset anchor | Training recipe, cross-dataset table, split rules |
| 12 | Huang S. et al., Meta-tubular-net | Biomedical Signal Processing and Control 2024 | V-prev (Q1) | Label-noise synthesis protocol | How noise is generated and its severity levels |
| 13 | Hendrycks & Dietterich, corruptions | ICLR 2019 | NOT reliably in Scopus; cite as protocol source only | Severity-graded corruption protocol | Corruption list and 5 severity levels |
| 14 | Wang A.-C. et al., Curriculum Fourier DA | IEEE Trans. Automation Sci. & Eng. 2023 | unchecked | Retina + corruption severity precedent | Which corruptions on which retinal tasks |
| 15 | Liu et al., GraphSeg | NeurIPS 2025 | conference: verify in Scopus | Closest graph + cross-dataset work | Train/test sets, metrics, absence of topology metrics (confirm in full text) |

### 4.2 Supporting (17) - one sentence each
| Paper | Venue / year | Status | Use |
|---|---|---|---|
| Shin et al., graphical connectivity | Medical Image Analysis 2019 | unchecked | Origin of graph-for-vessels idea |
| Li J. et al., GViT-RSNet | Pattern Recognition Letters 2025 | V-now | Recent graph-conv + ViT example |
| Shi C. et al., graph feature aggregation | Simulation Modelling Practice and Theory 2022 | V-now | Another graph-aggregation example (optional; drop if the intro is long) |
| Su H. et al., HFRF-Net + connectivity booster | IEEE Trans. Instrum. Meas. 2024 | unchecked | Topology-aware retinal method |
| Tan Y.-B. et al., SkelCon (NEW) | IEEE TMI 2022 | unchecked | Skeletal prior + connectivity metrics on 4+ datasets |
| Song D. et al., OVS-Net | IEEE Trans. Image Processing 2025 | unchecked | Connectivity across 17 datasets |
| Kannoth Veetil et al., Parkinson MRI leakage | Digital Signal Processing 2024 | unchecked | Slice-level leakage inflation (closest analogue to patch-level) |
| Marciniak et al., OCT duplicates | IEEE Access 2025 | V-prev | Retinal-imaging leakage example |
| Tampu et al., OCT leakage | Scientific Data 2022 | unchecked | 5-30% inflation figure (verify number in the paper before quoting) |
| Lyu et al., AADG | IEEE TMI 2022 | unchecked | Domain generalisation reference |
| Hu D. et al., Hessian vector field DG | Medical Image Analysis 2024 | unchecked | Domain generalisation reference with code |
| Liu B. et al., DGSSA | Neural Networks 2025 | unchecked | Recent DG reference |
| Su Y. et al., empirical design study | Biomedical Signal Processing and Control 2022 | V-prev | DRIVE<->STARE transfer numbers |
| Xu Z. et al., Mean-Teacher Confident Learning | IEEE TMI 2022 | unchecked | Real noisy-label retinal example |
| Li et al., positional label noise | Frontiers in Medicine 2026 | unchecked | Positional noise on DRIVE (recent, small citation base) |
| Chen C. et al., retinal vessel DL review | IEEE Access 2021 | V-prev | General background review |
| Mienye & Viriri, GNNs in medical imaging | Information (MDPI) 2025 | V-now (Information (Switzerland) listed) | One cite for "GNNs in medical imaging" in the introduction |

Newly found, optional (add only if a reviewer would expect them): Bansal et al., Retinal Vessel Segmentation review 1982-2025, Advanced Intelligent Systems 2026 (indexing unchecked); Bai et al., nnLoGoNet with Skeleton Recall Loss, Scientific Reports 2026 (topology loss comparator).

Datasets (cite whatever the venue): FIVES (Sci Data 2022), DRIVE (IEEE TMI 2004), STARE (IEEE TMI 2000), CHASE_DB1 (IEEE TBME 2012), HRF (Int J Biomed Imaging 2013).

## 5. The four project PDFs - verdict
| PDF | What it is | Verdict |
|---|---|---|
| Mienye & Viriri, GNNs in Medical Imaging, Information 2025 | Review | KEEP as one background citation. |
| Zhang et al., TransGraphNet, BSPC 2025 | Transformer + GCN segmentation on ACDC, Synapse, Coronacases | OPTIONAL: one sentence showing graph modules are spreading across segmentation tasks. Not retinal, no topology metrics in abstract. |
| Chai et al., text + image interpretable GNN segmentation, BSPC 2026 | Multimodal text-guided graph/hypergraph | CUT: needs text prompts; our data has none. |
| Peng et al., MLF-GNN PET/CT, BSPC 2024 | Prognosis from a population graph | CUT: different task (survival prediction) and graph type. |

## 6. Cut list (do not cite)
| Paper | Reason |
|---|---|
| Wen et al. 2020 (Alzheimer's CNN review, MedIA) | Leakage point already carried by Kapoor, Kannoth, Tampu; add back only if you want a second MRI example. |
| Kovacs et al. 2024 mlscorecheck (Neurocomputing) | About checking reported score consistency; tangential to our split question. Re-add only if you add a consistency check of VGA-Net numbers. |
| Karani et al. 2020 (test-time adaptation, MedIA) | Adaptation method, not a corruption or robustness benchmark. |
| Guo et al. 2024 (random colour transform, EAAI) | A DG augmentation method; Lyu/Hu/Liu already cover DG. |
| Chen C.-Y. et al. 2023 Retina-TransNet (JBHI) | Cross-dataset numbers duplicate Su 2022 and Galdran 2022. |
| Sun et al. 2020 robust augmentation | Preprint; awareness only. |
| SCOPE (arXiv 2023), Fadugba (arXiv 2024), Rabbani (2026, venue unstated), MedMNIST-C, Amarnath 2026, Widiyanto 2026 (JEEEMI, indexing unverified), TFFM | Preprint or unverified venue. Keep in a private awareness list; Fadugba and Widiyanto should still be mentioned in text as closest related work if a reviewer would ask (decision for your guide). |
| GA-TAN (Frontiers Med 2026), nnLoGoNet-like 3D/OCTA papers, ROP graph U-Net (ICSI 2026), fuzzy UNet++ (IEEE Access 2026), older Frangi/multifractal/classical papers | Different modality or task, or no relevance to graph/topology/robustness. |
| Peng 2024 and Chai 2026 PDFs | See section 5. |

## 7. What is and is not verified
- Verified this session: the SCImago listing of Pattern Recognition Letters, IEEE JBHI, Information (Switzerland) and Simulation Modelling Practice and Theory (SCImago lists Scopus sources only; its page did not show quartiles in our fetch). Abstracts for G2ViT, GKE-TUNet, HFRF-Net, nnLoGoNet, GViT-RSNet, Wang 2024, VGA-Net, Galdran, SkelCon came from Consensus.
- Earlier sessions: quartiles for IEEE Access, BSPC, CBM, IET IP.
- NOT verified: indexing of every other journal (all are established Elsevier/IEEE/Springer venues, but confirm each); whether CVPR/ICML/NeurIPS/ICLR proceedings are indexed in Scopus (ICLR in particular is probably not; check with the Scopus source list); full texts of most core papers (the "extract" column is what to look for, not what we found); volume/page numbers for the new entries.
- The "gaps" are search-based and not proof of absence. Re-run forward citations of VGA-Net, GraphSeg and Jamaludin 2026 before claiming novelty.

## 8. Suggested Related Work outline (about 1.5 pages)
1. Graph reasoning in retinal vessel segmentation (A) - end with: gains reported on overlap metrics, no controlled edge ablation.
2. Topology-aware evaluation (B) - justify clDice and Betti errors.
3. Evaluation protocols and leakage (C) - end with the RQ1 gap.
4. Robustness: shift, corruption, label noise (D-F) - end with the RQ3 gap.
5. Position statement: a controlled, leakage-free benchmark, not a new architecture.

## 9. Next steps (suggested order)
1. Quick check: run the Scopus verification for all "unchecked" venues (Scopus Sources list or SCImago, 20 minutes).
2. Read the 5 papers that most determine novelty first: VGA-Net, GraphSeg, GKE-TUNet, Li 2022 (TMI), Jamaludin 2026. Fill the "extract" column.
3. Ask the guide about the Tier-2 non-IEEE/Elsevier items (VGA-Net/Wiley, Galdran/Springer, GraphSeg/NeurIPS) and about citing Fadugba and Widiyanto as preprint/unindexed related work.
