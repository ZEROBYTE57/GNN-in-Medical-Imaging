# Evaluation protocol (draft v0.1)

- Unit of splitting: subject (never scan/visit). One baseline visit per subject.
- Cohort: same subjects for every method and task (AD vs NC, sMCI vs pMCI). pMCI = converted to AD within 36 months, as in Zhang.
- CV: 5x5 repeated stratified nested CV. Tuning only in inner folds.
- Normative reference (NC mean/SD): fitted on training-fold NC only.
- Graph: built from training nodes; test nodes attached inductively (edges to training nodes only, no test-test edges).
- Phenotypes allowed in edges: age, sex, APOE4, education. MMSE excluded from the main graph (ablation only).
- Metrics: balanced accuracy, AUC, AUPRC, sensitivity, specificity, ECE. Bootstrap 95% CI, paired tests.
- Headline figure: leakage gap = metric(Zhang-style protocol) - metric(leakage-free protocol), per model and task.

## Baseline ladder
1 majority | 2 MMSE-only, APOE-only, age+sex-only (logistic regression) | 3 tabular LR/SVM/XGBoost/MLP on ROI features |
4 Zhang IFDCGCN reproduction | 5 MMGL-style, EV-GCN-style, GraphSAGE | 6 proposed leakage-free graph
