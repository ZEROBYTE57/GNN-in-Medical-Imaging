- This approach evaluates whether GNN explainers can tell if a GNN model learned real disease-related connections or scanner/site shortcuts. We first test the explainers with planted ground truth on healthy-control data. The explainers that pass are then applied to real disease datasets (autism, ADHD, MDD, schizophrenia) to check each disease model for scanner dependence.

- First from the dataset we had, (ABIDE Preprocessed) containing samples from effected patients & controls(about to get the disease)
- Splits the dataset into random samples of 2 sets, labels as (patients/controls),
- Then we plant the known site 

---

# Implementation so far (written 30 Oct 2026)

Plain-language summary of what has been built and measured since implementation began. Terms are explained in Keyword-Notes.md (project glossary).

## Two corrections to the notes above

1. **Controls are healthy people, not people "about to get the disease".** ABIDE has autism subjects and healthy controls.
2. **We don't split the whole dataset, and we don't plant a site first.**
   - The benchmark uses **healthy controls only**. They are split randomly into fake "patients" and fake "controls", and a known **disease-like** signal is planted in the fake patients.
   - **Site decoys** (fake scanner effects) are a later, separate step. They go in on top of the planted signal, to see whether explanations point at the disease or at the scanner.

## The plan in one picture

```
ABIDE controls ──► fake patient / fake control labels ──► plant a known signal in fake patients
                                                                 │
                         (later) add a fake site effect ─────────┤
                                                                 ▼
                          train MLP / GNN ──► explainer ──► compare with the planted answer (AUPRC)
                                                                 │
                     SiteSwap (our method) should find the planted signal, not the site effect
```

Then the explainers that pass are used on real autism data (and other disorders if time allows; only ABIDE is set up so far).

## Step 1: Data (done)

- Downloaded ABIDE I from the Preprocessed Connectomes Project: CPAC pipeline, no global signal regression, two atlases (AAL-116 main, CC200 check).
- For each subject: region time series → Pearson correlation → Fisher z → 116 × 116 matrix (dead regions set to 0).
- Quality-checked subjects only. OHSU was dropped (scans too short). **Result: 848 subjects, 19 sites.**
- Code: `CXB/src/cxb/data.py`, scripts `01_download_abide.py`, `02_build_connectomes.py`.

## Step 2: Splits (done)

- We reuse the **10 leave-site-out folds** of the BrainOOD paper (ICLR 2025), recovered from their files, so our scores are comparable with a published benchmark.
- Each fold has a whole test site and a whole validation site. Everything else is for training.
- **Rule:** the epoch (and any setting) is chosen on validation data, never on test data.
- Code: `CXB/src/cxb/splits.py`, with tests.

## Step 3: Planting the ground truth (done and checked)

**How a planted dataset is made:**

1. Set aside 20% of controls (per site) as a **ranking set**.
2. Compare all autism subjects with the ranking set to find the regions most involved in real autism differences.
3. The remaining controls form the **planting pool**. Half of each site gets the fake label "patient".
4. For each fake patient, add one shared signal to the time series of the chosen 7, 10 or 20 regions, then recompute the connectivity matrix in exactly the same way.
5. **True answer:** the connections among those regions (21, 45 or 190 connections).

**Why this design:** our first design shifted single connections directly. That made every planted matrix impossible for a real brain (not a valid correlation matrix). Two repairs leaked the effect onto connections that weren't planted. Planting in the time series avoids both problems.

**Measured on real data (AAL):**

| Check | Result |
|---|---|
| Valid matrices | 100% |
| Effect size achieved | within about 5% of target |
| Connections outside planted regions | unchanged for 87% of patients |
| Side effect | connections from planted regions to the rest weaken about 25% (ignored in scoring) |

**Difficulty levels (recovery ceiling: best AUPRC a plain statistical test can get):**

| Level | × 0.5 | × 1 | × 2 | × 4 |
|---|---|---|---|---|
| AUPRC | 0.03–0.08 | 0.17–0.28 | 0.80–0.83 | about 1.0 |

So the levels run from almost undetectable to easy.

**Known limitations:**
- Planted effects only increase connectivity, whereas real autism effects go both ways.
- CC200 is harder: most of its scans have fewer time points than regions.

Code: `CXB/src/cxb/planting.py`, check script `03_plant_check.py`, 18 tests passing.

## Step 4: Base paper BrainIB (checked privately)

- We ran BrainIB's released code on the authors' own data (10 folds, 300 epochs), to understand the method and its numbers. The paper reports 70.2% on ABIDE (Table V).
- The results and observations are kept in the private file `CXB/docs/baseline-notes.md`, which is not on GitHub because this repo is public.
- **Decisions:**
  - We don't contact the authors, and the paper makes no claims about their code.
  - We **rewrite BrainIB ourselves** from the paper's description.
  - We report our numbers next to the paper's, marked "as reported (different protocol)".
- **Why honest scoring matters for us:** choosing the best epoch by test accuracy inflates results, so we always choose the epoch on validation data.

## Step 5: Repo hygiene (done)

The GitHub repo is public. Papers, data, results, other people's code, BrainIB's data file, the private BrainIB notes and the Word documents are all excluded from git.

## Step 6: Gate 1 (done, passed, 30 Oct 2026)

**Question:** can a simple model plus explainer find a strong (× 4) planted effect at all?

**Setup:**
- AAL atlas, 10 planted regions;
- BrainOOD's 10 site folds × 3 fake-label splits;
- models: our own MLP and GCN, with the epoch chosen on the validation site;
- explainers: saliency and Integrated Gradients, applied to the test site;
- score: group AUPRC. Attributions are averaged over the test subjects, then compared with the planted connections.

**Result at × 4 (pass rule: AUPRC ≥ 0.5; chance 0.008):**

| Model | Test accuracy | Saliency AUPRC | Integrated Gradients AUPRC |
|---|---|---|---|
| GCN | 0.996 | **1.00** | **1.00** |
| MLP | 0.924 | 0.99 | 0.86 |

**Gate 1 passed.**

**Something we learned on the way:** a plain GCN couldn't learn the planted effect (0.68 validation accuracy). No setting of dropout, learning rate, edge density or size helped. It needed two changes together:
1. A separate weight for each region's own features. Otherwise averaging with neighbours cancels the signal: rows inside the planted set go up while their neighbours' rows go down.
2. A readout that keeps each region in its own place (concatenation) instead of averaging all regions.

Both changes were chosen using validation data only. This is a useful point for the paper: GNN design choices decide whether a GNN can even see a region-level effect.

**Weaker levels (same setup, best model + explainer):**

| Level | Best AUPRC | Ceiling (plain t-test) |
|---|---|---|
| × 4 | 1.00 | 1.00 |
| × 2 | 0.63 (MLP saliency); GCN varies a lot between folds | 0.78 |
| × 1 | 0.09 | 0.15 |

So the levels run from easy to almost impossible, and × 2 is where methods differ. Saliency beats Integrated Gradients at every level.

Code: `src/cxb/models.py`, `src/cxb/explain.py`, `scripts/04_gate1.py`. 23 tests pass.

## Step 7: Gate 2 (done, passed, 30 Oct 2026)

**Question:** does a fake scanner shortcut pull explanations away from the disease?

**Setup:**
- **Disease signal:** strength × 2. At × 4 the disease is too easy, so no model would need a shortcut.
- **Fake scanner ("scanner B"):** a signal added to 10 other regions, where real ABIDE hospitals differ most.
  - Size: 0.11, the typical difference between two real hospitals. That's 1.4 times the disease effect (0.08).
  - Added the same way as the disease signal, so every matrix stays valid.
- **Shortcut:** in the training and validation hospitals, 50%, 70% or 90% of fake patients have scanner B, against 50%, 30% or 10% of fake controls. In the test hospital it's 50/50, so the shortcut is useless there.
- **Comparison design:** the shortcut is also present in the test hospital (90%).
- **Pass rule (fixed before running):** at 90%, the scanner share rises by ≥ 0.2, or disease AUPRC falls by ≥ 0.2.

**Results (saliency, averaged over 10 folds × 3 splits):**

| Shortcut | MLP accuracy | MLP disease AUPRC | MLP scanner share | GCN accuracy | GCN disease AUPRC | GCN scanner share |
|---|---|---|---|---|---|---|
| 50% (none) | 0.67 | 0.62 | 0.00 | 0.60 | 0.46 | 0.00 |
| 70% | 0.66 | 0.62 | 0.08 | 0.59 | 0.28 | 0.00 |
| 90% | 0.69 | 0.38 | **0.45** | 0.59 | 0.18 | **0.28** |
| 90%, also in test | **0.78** | 0.38 | 0.44 | **0.70** | 0.21 | 0.36 |

*Scanner share:* the share of the 45 most important connections (by the explanation) that lie in the scanner regions.

**Gate 2 passed** for both models.

**What it shows:**
1. With a strong shortcut, almost half of the MLP's top connections come from the scanner, up from none, and it finds the disease less well.
2. **Accuracy doesn't warn you.** In the train-only design, test accuracy hardly changes (0.67 → 0.69) while the explanation shifts to the scanner.
3. **When the shortcut is also in the test data, accuracy goes up** (MLP 0.78), so the model looks *better* while relying on the scanner. Only the explanation reveals it. This is the paper's core message.
4. The GCN loses the disease signal faster than the MLP, but it's unstable at × 2 (disease AUPRC 0.24–0.80 across splits even without a shortcut).
5. Saliency again works better than Integrated Gradients.

Code: `src/cxb/decoy.py`, `scripts/05_gate2.py`. 27 tests pass.

## Step 8: Gate 3 (done, 31 Oct 2026): non-inferior, not a clean pass

**Question:** does SiteSwap keep explanations on the disease better than the obvious fix, ComBat harmonisation?

**Setup:**
- Same data as gate 2 (disease × 2, 90% scanner shortcut).
- The scanner (A or B) is known for every subject, as hospital names are in practice.
- **Five methods:**
  1. **Plain:** no fix.
  2. **ComBat:** removes scanner differences before training.
  3. **ComBat, label protected:** removes scanner differences while trying to keep the label effect.
  4. **SiteSwap-augmentation:** also trains on a copy of each scan moved to the other scanner (with ComBat), with the same label, so the scanner stops being a useful clue.
  5. **SiteSwap (full):** augmentation plus a penalty when the explanation differs between a scan and its copy.

**Results (shortcut in training only, saliency, 10 folds × 3 splits):**

| Method | MLP disease AUPRC | MLP scanner share | GCN disease AUPRC | GCN scanner share |
|---|---|---|---|---|
| Plain | 0.39 | 0.44 | 0.21 | 0.29 |
| ComBat | 0.10 | 0.01 | 0.02 | 0.01 |
| ComBat, label protected | 0.26 | 0.01 | 0.08 | 0.01 |
| **SiteSwap-augmentation** | **0.55** | 0.23 | **0.34** | **0.005** |
| SiteSwap (full) | 0.51 | 0.23 | 0.02 | 0.004 |

**Official verdict (rule fixed before running): non-inferior.**
- The rule judged the *full* SiteSwap.
- For the MLP, it found the disease far better than ComBat (+0.25), but kept more scanner share (+0.22), so it doesn't count as a pass.
- For the GCN, the explanation penalty broke the model (0.02).

**What it shows:**
1. **ComBat's dilemma is real.** It removes the scanner, but takes most of the disease signal with it, even when protecting the label.
2. **The simple version of SiteSwap (augmentation only) is the best method for both models.** It recovers the disease better than every other method, including the plain model, and removes the scanner completely from the GCN's explanation.
3. **The explanation penalty hurts.** The simpler method is better, matching the earlier rule: simplify, don't add.
4. The MLP still keeps about 0.23 scanner share with SiteSwap, so the shortcut isn't fully removed for it.

**Important:** we picked augmentation-only *after* seeing these results. To be valid, it must be confirmed on **new** fake-label splits and seeds, with a pass rule fixed in advance.

Code: `src/cxb/combat.py`, `src/cxb/siteswap.py`, `scripts/06_gate3.py`. 31 tests pass.

### Why the MLP still looks at the scanner (investigation, 31 Oct 2026)

- **Test:** we have the *true* scanner-B version of every brain, so SiteSwap can be trained with a perfect swap. Result for the MLP: scanner share **0.00**, disease AUPRC **0.76** (with the ComBat swap: 0.21 and 0.65).
- **So the SiteSwap idea works.** The problem is the swap.
- **Where the swap fails:** on the ~960 connections from the scanner regions to the rest of the brain. There, the scanner effect *shrinks* values towards zero, a multiplication rather than a shift.
  - ComBat adjusts each connection separately and can't copy this exactly for each person.
  - A simple classifier can still tell the label from those connections after the swap (AUC 0.88; perfect swap 0.43).
- **Fixes tried, none worked:**
  - ComBat on correlations instead of Fisher z;
  - ComBat estimated on other people only (cross-fitting);
  - matching within each label group (0.98, worse).
- **Conclusion for the paper:**
  - SiteSwap is only as good as its swap.
  - Report the perfect swap as the upper bound, and the ComBat swap as the practical version.
  - Better swap models are future work.

### Confirmation run (running)

- New fake-label splits (3–5) and a new seed, because the simple version was picked after seeing results.
- Same pass rule, now applied to augmentation-only SiteSwap.
- The penalty version is reported as an ablation; the perfect swap as the upper bound.

## What comes next

1. **Gate 1 (done, passed):** train a simple MLP and GCN on the strongest planted effect (× 4, 10 regions), explain them with saliency and Integrated Gradients, and score against the planted answer.
   - **Pass** if AUPRC ≥ 0.5 (chance is about 0.01, the ceiling about 1.0).
   - **If it passes**, the benchmark idea works.
   - **If it fails**, either the models can't learn the effect, or the explainers can't find it. We check which before moving to a fallback.
2. **Gate 2 (done, passed): does a fake scanner shortcut pull explanations away from the disease?** (details below)
3. **Gate 3 (done, non-inferior): does SiteSwap keep explanations on the disease better than ComBat harmonisation?** Next: confirm augmentation-only SiteSwap on new splits.
4. **Then:** pick 3–4 published baselines together and run them.
5. **Paper:** 8-page conference format.

# Simplified plan (decided 30 Oct 2026)

**The paper in one sentence:** using brain scans with a planted, known answer, we show that GNN explanations follow scanner shortcuts, and that a simple training trick (SiteSwap) keeps them on the real signal.

**Contributions:**
1. A test with a known answer, including a fake scanner shortcut.
2. A simple new method (SiteSwap), plus its evaluation.

| Part | Choice |
|---|---|
| Data | ABIDE only, AAL atlas only (claim scoped to autism) |
| Planting | 10 regions; strengths × 1, × 2, × 4 |
| Models | our own MLP and GCN; published baselines later |
| Explainers | saliency, Integrated Gradients, GNNExplainer |
| Metrics | disease AUPRC; scanner share; test accuracy |
| Repeats | 10 site folds × 3 fake-label splits × 3 seeds |
| Compute | about 5–8 GPU-hours |

## The scanner decoy

- **Why a fake one:** fake patients are balanced within every real hospital, so real hospitals carry no label information. To test "disease or scanner?", we create a scanner effect that does carry label information, at a place we know.
- **What it is:** a "scanner B" fingerprint, a shift on connection values whose size is copied from real ABIDE hospitals (ComBat estimates).
- **Where:** the 10 regions outside the planted set where real hospitals differ most.
- **Main design (train only):**
  - In training, 50%, 70% or 90% of fake patients carry scanner B, against the mirror share of fake controls.
  - In test it's 50/50, so the shortcut carries no information.
  - A model using the shortcut loses accuracy on test, and its explanation points at the decoy.
- **Comparison design (train and test):** the shortcut is present everywhere at 90%. Accuracy stays high, and only the explanation reveals the shortcut.
- **What we measure:**
  - **Disease AUPRC:** does the explanation still find the planted connections?
  - **Scanner share:** what fraction of the top explanation connections fall in the decoy regions?
  - **Accuracy drop.**
- **Headline figure:** as the shortcut strengthens (50 → 70 → 90%), explanations drift from disease to scanner.

## SiteSwap and its competitor

- **ComBat harmonisation (competitor):** removes scanner differences before training. When scanner and label are mixed (e.g. 80/20), it faces a dilemma:
  - If it protects the label, it keeps the shortcut.
  - If it doesn't, it erases part of the disease.
- **SiteSwap (ours):** during training, make a site-swapped copy of each scan with ComBat, and penalise the model if its explanation (saliency) changes between the two. Nothing is removed from the data.
- **Simplest fallback version:** train on both the original and the swapped copy with the same label (augmentation only, no explanation penalty).

## Gates

1. **Gate 1:** models plus explainers find the × 4 planted effect (AUPRC ≥ 0.5).
2. **Gate 2:** with the decoy present, explanations shift towards it.
3. **Gate 3:** SiteSwap reduces that shift more than ComBat does, without losing accuracy.
   - If not: simplify SiteSwap, rather than add parts.