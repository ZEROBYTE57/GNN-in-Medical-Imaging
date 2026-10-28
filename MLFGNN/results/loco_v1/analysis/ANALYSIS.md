# Novelty vs existing — analysis of `loco_v1`

Cohort: `data\processed\features_synth.csv` — **simulated data**; numbers exercise the pipeline and say nothing about the real cohort.

Hardware: NVIDIA GeForce RTX 4060 Laptop GPU, torch 2.14.0+cu130.

## 1. Headline (Harrell C, mean ± std over LOCO folds × seeds)

| arm | in-centre | held-out centre | gap |
|---|---|---|---|
| radiomics_coxph / none | 0.748 ± 0.040 | 0.773 ± 0.011 | -0.025 ± 0.040 |
| MLF-GAT / knn@k8 | 0.749 ± 0.038 | 0.768 ± 0.038 | -0.019 ± 0.050 |
| rsf / none | 0.743 ± 0.049 | 0.767 ± 0.034 | -0.024 ± 0.059 |
| MLF-GAT / snf_threshold | 0.754 ± 0.035 | 0.766 ± 0.037 | -0.012 ± 0.037 |
| MLF-GAT / centre_balanced_knn@k2 | 0.749 ± 0.036 | 0.766 ± 0.041 | -0.017 ± 0.049 |
| MLF-GAT / knn@k5 | 0.749 ± 0.037 | 0.764 ± 0.043 | -0.015 ± 0.045 |
| MLF-GAT / centre_balanced_knn@k5 | 0.750 ± 0.040 | 0.764 ± 0.042 | -0.014 ± 0.047 |
| MLF-GAT / centre_balanced_knn@k8 | 0.754 ± 0.039 | 0.764 ± 0.031 | -0.010 ± 0.043 |
| MLF-GAT / centre_balanced_knn@k3 | 0.751 ± 0.041 | 0.760 ± 0.042 | -0.009 ± 0.044 |
| cox_nnet / none | 0.729 ± 0.043 | 0.759 ± 0.018 | -0.030 ± 0.047 |
| MLF-GAT / knn@k3 | 0.752 ± 0.039 | 0.758 ± 0.050 | -0.006 ± 0.050 |
| MLF-GAT / knn@k2 | 0.751 ± 0.038 | 0.751 ± 0.051 | 0.000 ± 0.053 |
| clinical_cox / none | 0.698 ± 0.054 | 0.710 ± 0.044 | -0.012 ± 0.069 |

**Existing** (snf_threshold): held-out 0.766, gap -0.012.  
**Novel** (best centre_balanced_knn@k2): held-out 0.766, gap -0.017.  
Δ held-out C = -0.000; Δ gap = -0.005 (negative = the novel rule closes the gap).

## 2. Paired bootstrap on held-out patients (vs existing rule)

| arm | ΔC | 95% CI | p | significant |
|---|---|---|---|---|
| MLF-GAT/knn@k8 | -0.0007 | [-0.0089, +0.0072] | 0.9103 | no |
| MLF-GAT/knn@k3 | -0.0049 | [-0.0145, +0.0051] | 0.3389 | no |
| MLF-GAT/knn@k5 | -0.0051 | [-0.0130, +0.0042] | 0.2658 | no |
| MLF-GAT/centre_balanced_knn@k8 | -0.0073 | [-0.0174, +0.0025] | 0.1395 | no |
| MLF-GAT/centre_balanced_knn@k3 | -0.0082 | [-0.0209, +0.0033] | 0.1262 | no |
| MLF-GAT/centre_balanced_knn@k2 | -0.0087 | [-0.0200, +0.0017] | 0.0997 | no |
| MLF-GAT/centre_balanced_knn@k5 | -0.0091 | [-0.0224, +0.0011] | 0.1130 | no |
| MLF-GAT/knn@k2 | -0.0116 | [-0.0232, -0.0019] | 0.0266 | yes |

1 arm(s) differ significantly from the existing rule.

## 3. Per held-out centre

| arm | CHUM | CHUS | HGJ | HMR | mean | worst_centre |
|---|---|---|---|---|---|---|
| radiomics_coxph/none | 0.768 | 0.771 | 0.766 | 0.787 | 0.773 | 0.766 |
| MLF-GAT/knn@k8 | 0.740 | 0.753 | 0.757 | 0.822 | 0.768 | 0.740 |
| rsf/none | 0.743 | 0.749 | 0.757 | 0.819 | 0.767 | 0.743 |
| MLF-GAT/snf_threshold | 0.744 | 0.757 | 0.748 | 0.816 | 0.766 | 0.744 |
| MLF-GAT/centre_balanced_knn@k2 | 0.722 | 0.759 | 0.758 | 0.825 | 0.766 | 0.722 |
| MLF-GAT/knn@k5 | 0.735 | 0.754 | 0.746 | 0.822 | 0.764 | 0.735 |
| MLF-GAT/centre_balanced_knn@k5 | 0.736 | 0.758 | 0.740 | 0.821 | 0.764 | 0.736 |
| MLF-GAT/centre_balanced_knn@k8 | 0.743 | 0.756 | 0.747 | 0.808 | 0.764 | 0.743 |
| MLF-GAT/centre_balanced_knn@k3 | 0.728 | 0.760 | 0.739 | 0.815 | 0.760 | 0.728 |
| cox_nnet/none | 0.736 | 0.760 | 0.762 | 0.778 | 0.759 | 0.736 |
| MLF-GAT/knn@k3 | 0.715 | 0.753 | 0.739 | 0.824 | 0.758 | 0.715 |
| MLF-GAT/knn@k2 | 0.705 | 0.750 | 0.734 | 0.813 | 0.751 | 0.705 |
| clinical_cox/none | 0.692 | 0.707 | 0.666 | 0.775 | 0.710 | 0.666 |

## 4. Sensitivity to k

| rule | k | held-out C | gap |
|---|---|---|---|
| centre_balanced_knn | 2 | 0.766 ± 0.041 | -0.017 ± 0.049 |
| centre_balanced_knn | 3 | 0.760 ± 0.042 | -0.009 ± 0.044 |
| centre_balanced_knn | 5 | 0.764 ± 0.042 | -0.014 ± 0.047 |
| centre_balanced_knn | 8 | 0.764 ± 0.031 | -0.010 ± 0.043 |
| knn | 2 | 0.751 ± 0.051 | 0.000 ± 0.053 |
| knn | 3 | 0.758 ± 0.050 | -0.006 ± 0.050 |
| knn | 5 | 0.764 ± 0.043 | -0.015 ± 0.045 |
| knn | 8 | 0.768 ± 0.038 | -0.019 ± 0.050 |

## 5. Mechanism

Spearman correlations across (fold × seed × arm) between graph structure and outcome:

- `spearman(cross_centre_edge_ratio, held_out_C)` = -0.086
- `spearman(cross_centre_edge_ratio, gap)` = +0.060
- `spearman(centre_assortativity, held_out_C)` = +0.018
- `spearman(centre_assortativity, gap)` = -0.013
- `spearman(degree_mean, held_out_C)` = +0.102
- `spearman(degree_mean, gap)` = -0.057
- `spearman(risk_homophily, held_out_C)` = +0.441
- `spearman(risk_homophily, gap)` = -0.308

A positive correlation of cross-centre edge ratio with held-out C, and a negative one with the gap, is the signature the proposal predicts. If both are near zero, centre mixing is not what moves the score on this cohort.

## 6. Against the best tabular baseline

Best tabular model on the held-out centres: **radiomics_coxph** (C = 0.773).

| GNN arm | held-out C | Δ vs tabular |
|---|---|---|
| MLF-GAT/knn@k8 | 0.768 | -0.005 |
| MLF-GAT/snf_threshold | 0.766 | -0.007 |
| MLF-GAT/centre_balanced_knn@k2 | 0.766 | -0.007 |
| MLF-GAT/knn@k5 | 0.764 | -0.009 |
| MLF-GAT/centre_balanced_knn@k5 | 0.764 | -0.009 |
| MLF-GAT/centre_balanced_knn@k8 | 0.764 | -0.009 |
| MLF-GAT/centre_balanced_knn@k3 | 0.760 | -0.013 |
| MLF-GAT/knn@k3 | 0.758 | -0.015 |
| MLF-GAT/knn@k2 | 0.751 | -0.022 |

Mueller et al. (TMLR 2024) showed population GNNs often fail to beat tabular baselines when the graph is weak; a graph rule only earns its place if this column is positive.
