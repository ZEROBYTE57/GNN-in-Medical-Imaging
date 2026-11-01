# Paper outline (8 pages)

Final framing, decided 1 Nov 2026 (see [decisions.md](decisions.md)). Every number below comes from finished runs: [stats.md](stats.md), [results.md](results.md), [site_signal.md](site_signal.md) and `results/final/audit_check/`. No new experiments are planned. GAT and the CC200 atlas are held in reserve, for use only if a reviewer asks.

## Working title

**Accuracy hides the scanner: a planted ground truth for explanations of brain-connectome models**

Alternative: *Do brain-network explanations find the disease or the scanner?*

## Message in one paragraph

Brain-network papers check explanations against published findings, because real data has no known answer. We plant a known connectivity effect into real ABIDE control connectomes and add a scanner shortcut sized from real hospital differences. Explanations recover the planted effect as well as a univariate t-test. Under a 90% scanner shortcut, however, up to 46% of the top explanation connections move to the scanner regions, while test accuracy falls by only 0.04-0.06. So accuracy cannot warn about the shortcut, but the explanation can. On real ABIDE the hospital is far easier to predict than autism, and the scanner-affected connections can be estimated from training data alone, so the check is usable in practice. Among fixes, the ComBat version decides the outcome: three of the four versions harm disease recovery. A training-side alternative (SiteSwap) matches the best ComBat without depending on that choice. Both are limited by the same scanner model, which assumes one fixed shift per connection.

## Contributions

1. **Benchmark.** A planted ground truth in real connectomes, with region-level planting in the time series (matrices stay valid correlation matrices). The scanner decoy is calibrated to real ABIDE hospital differences, and evaluation uses leave-site-out folds (BrainOOD). Controls: shuffled labels, logistic regression, and a t-test reference.
2. **Finding.** A scanner shortcut moves explanations while accuracy hides it. This holds for linear, MLP and GCN models. BrainNetCNN fails in the opposite way: its explanations stay on the disease, but its accuracy drops.
3. **Practical check.** The real hospital signal is strong (hospital predicted at 0.70 vs chance 0.05; autism only 0.64). The scanner-affected connections can be found without ground truth, so the scanner share of an explanation can be measured on real data.
4. **Analysis of fixes.** The ComBat version decides the outcome (GCN disease recovery ranges from 0.02 to 0.63). SiteSwap is a safe training-side fix. The perfect swap shows that the bottleneck is the scanner model, not the training method.

## Claim → evidence

| Claim | Evidence | Source |
|---|---|---|
| Explainers recover a planted effect | Saliency group AUPRC at x2: MLP 0.80, GCN 0.81; t-test reference 0.78; chance 0.008 | results.md, Fig. 1 |
| Recovery is not explainer bias | Shuffled labels: AUPRC 0.010-0.011 | stats.md §7 |
| A linear model recovers it best and drifts too | Logistic regression: 0.98 without shortcut; 0.52 and scanner share 0.43 under the 90% shortcut | stats.md §7 |
| A shortcut moves explanations, accuracy hides it | From 50% to 90% shortcut: scanner share +46% (MLP) / +22% (GCN), disease AUPRC −0.27 to −0.37, accuracy −0.04 to −0.06 | results.md §2, Fig. 2; paired CIs in stats.md §4 |
| Models fail in different ways | BrainNetCNN: scanner share 0.03, AUPRC 0.99, accuracy 0.95 → 0.83 (3 splits × 2 seeds) | stats.md §10 |
| The real hospital signal is strong | Hospital balanced accuracy 0.70 (chance 0.05); autism 0.64 (chance 0.50); CC200 0.71 / 0.66 | site_signal.md |
| The check works without ground truth | Top-45 scanner map from training data: 93-100% of its edges touch the scanner regions, ≤ 5% are disease edges | audit_check.csv, decisions.md |
| The ComBat version decides the outcome | GCN 0.02-0.63, MLP 0.19-0.59 across 4 versions; unprotected standard ComBat leaves the MLP scanner share at 0.49 | stats.md §1, Fig. 3 |
| SiteSwap beats plain | +0.15 disease AUPRC for both models, Holm p ≤ 0.001, 6/6 splits | stats.md §0 |
| SiteSwap matches the best ComBat | MLP −0.00; GCN +0.10 (tie after Holm, p 0.07); accuracy +0.03; both swap versions are close (MLP 0.57-0.59, GCN 0.73-0.77) | stats.md §0-1 |
| The limit is the scanner model | Perfect swap: 0.85 / 0.90, scanner share 0.00 | stats.md §1 |
| Label protection can inflate group differences, mainly without EB | Probe AUC: simplified ComBat 0.89, standard 0.72, raw 0.69 | stats.md §6 |
| Results hold on real labels | No method makes explanations follow the hospital; SiteSwap keeps plain accuracy (0.629 vs 0.632) | stats.md §5, Fig. 4 |

## Structure

| Section | Pages | Content |
|---|---|---|
| 1 Introduction | 1 | Problem: explanations are checked against the literature; shortcuts; our question and the four contributions |
| 2 Related work | 0.75 | Table below |
| 3 Benchmark | 1.5 | Data and folds, region planting, decoy calibration, metrics, controls, models, explainers |
| 4 Fixes | 0.5 | ComBat versions, SiteSwap, perfect swap |
| 5 Results | 3 | 5.1 benchmark and controls (Fig. 1, Table 1); 5.2 drift (Fig. 2) and other models; 5.3 real hospital signal and the check without ground truth; 5.4 fixes (Fig. 3, Table 2); 5.5 real labels (Fig. 4) |
| 6 Discussion and scope | 0.75 | What it means for brain-network papers; scope (below) |
| 7 Conclusion | 0.25 | |
| References | 0.25+ | |

## Related work: how this paper differs

| Line of work | Examples | What they do | What we add |
|---|---|---|---|
| Interpretable brain networks | BrainGNN (Li 2021), BrainGB (Cui 2023), BrainNetCNN (Kawahara 2017) | Explanations checked against published findings | A known ground truth, with a scanner confound |
| Ground-truth tests of explanations | CLEVR-XAI (Arras 2022), Zhou et al. 2022 | Planted features in images or synthetic data | Real connectomes, valid planting, a calibrated scanner confound |
| Do explanations reveal spurious signals? | Adebayo 2018, 2022 | Explanations often miss unknown spurious signals | For a known, measured confounder (the site), explanations do reveal the shortcut, and accuracy does not |
| Shortcut learning in medical imaging | Zech 2018, Brown 2023 | Shortcuts detected through accuracy or shortcut testing | Accuracy hides this shortcut, and explanations show it |
| Harmonisation | ComBat (Johnson 2007; Fortin 2017; Yu 2018), Nygaard 2016, Wachinger 2021, Dinsdale 2021 | Judged by accuracy or group statistics | Judged by where explanations point; the choice of version matters most |
| Counterfactual and targeted augmentation | Kaushik 2020, Ilse 2021, Gao 2023 | Change the spurious part, keep the label | SiteSwap applies this with ComBat as the scanner model, showing both the gain and the limit |
| OOD brain-network evaluation | BrainOOD (Xu 2025) | Leave-site-out protocol | We use its folds |

## Scope (stated plainly in section 6)

- **One dataset (ABIDE I).** The hospital-signal result is replicated on a second atlas (CC200).
- **The planted effect is a linear connectivity shift** sized to real autism effects. The claim concerns reliance on the shortcut, which shows up for every model type, including a linear one.
- **The check targets a known confounder (the site)**, the main one in multi-site neuroimaging. It does not search for unknown shortcuts.
- **BrainGNN and BNT do not learn from about 250 training subjects** in our reimplementations (unit tests pass). This is reported as a sample-size caution, not as evidence about the published models.
- **BrainNetCNN uses fewer runs** (3 splits × 2 seeds).

## Questions reviewers will ask, and our answers

| Question | Answer |
|---|---|
| Why not just use protected ComBat? | It works in our setting, but three of the four ComBat versions fail, and protection can inflate group differences (Nygaard). SiteSwap needs no test-time harmonisation and matches the best version. |
| Is SiteSwap new? | It is targeted augmentation (Gao 2023) applied with ComBat. Our point is the analysis: why it ties, and where the limit is. |
| Real models do not follow the hospital: so does the problem matter? | Real-label models learn little (accuracy about 0.60). The hospital is about 13× above chance from the same connections, so the shortcut is available. The benchmark shows what happens once it is learned. |
| Isn't the check just your metric? | Yes, with the known decoy replaced by a map estimated from training data. That is what makes it usable without ground truth. |
| Why only one GNN? | The GCN follows BrainGB. GAT code is ready if needed. The finding holds across linear, MLP, GCN and CNN models. |
