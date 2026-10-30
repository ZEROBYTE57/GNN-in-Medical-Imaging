Topic T2 · visual explainer

# Planted ground truth: testing whether brain-GNN explanations are right

Real patient data has no answer key, so nobody can check whether an explainer points at the true disease connections. T2 makes an answer key: take real healthy brains, secretly change a few chosen connections in a fake "patient" group, then see whether the explainer finds them.

1 · The whole experiment

## Six steps, one hidden answer key

1**Real healthy brains**

About 470 ABIDE controls. Each gives a 200×200 connectivity matrix: 19,900 connections.

2**Coin-flip labels**

Half become fake "patients", half "controls". No real difference exists.

3**Plant the effect**

Change chosen connections in the "patients" only. Save the list.

4**Train the model**

GNN, MLP or random forest learns to separate the two groups.

5**Explain**

The explainer ranks all 19,900 connections by importance.

6**Score**

Compare the ranking with the saved list.

**Answer key** (planted list) goes straight from step 3 to step 6hidden from the model and the explainer

2 · What planting does to the data

## Only the chosen cells change, only in fake patients

A toy version with 12 brain regions instead of 200. Darker cells mean stronger connection. The real study does exactly this on 200×200 matrices.

Connection strength (darker = stronger) Planted cells (DMN to DMN)

Regions: DMN = PCC, mPFC, AG-L, AG-R · Visual = V1-L, V1-R, LO-L, LO-R · Salience = aINS-L, aINS-R, dACC, aPFC. The difference panel is the answer key: zero everywhere except the planted block.

3 · The same thing as a graph

## Known connections are the edges you edited

Node

A brain region (ROI). 200 in the real study.

Edge

How in-sync two regions are (correlation). 19,900 in the real study.

Planted

Teal edges: the set S you weakened in fake patients. This is the true signal.

Decoy

Amber dashed edges: the set D shifted to mimic a scanner artefact. Used only in the shortcut test (section 6).

Untouched

Every other edge. The explainer should give these low importance.

4 · How big is the change

## Effect size δ: how far the patient group shifts

For one planted connection, here are its values across people. Move the slider to change δ, measured in standard deviations (Cohen's d). Small δ means the groups overlap a lot and the effect is hard to find.

**δ =** 0.30

88%groups overlap

smallconventional label

realisticvs typical psychiatric FC effects (≈0.1–0.3, verify)

Fake controls (untouched)Fake patients (after planting)

Planting formula per patient and planted edge: new value = old value − δ × (that edge's standard deviation across controls). The minus sign means "weakened".

5 · Scoring an explainer

## Look at its top 100 and count how many you planted

Suppose 100 edges were planted. Each square is one of the explainer's top-100 edges, highest-ranked first. A teal square is a planted edge it found.

rank 1rank 100

0.65precision@100

130×better than chance (0.005)

0decoy edges in top 100

Planted edge (hit)Decoy edgeUntouched edge (miss)

Chance level: 100 planted out of 19,900 edges = 0.005. Numbers here are illustrative, not results.

6 · The shortcut test

## Does the explainer notice when the model cheats?

Scanner sites in "group A" get the decoy shift. In training, most fake patients come from group A, so the model can cheat by detecting the scanner. In testing, that link is broken.

### Training set: shortcut available

20 fake patients (16 from site group A)

20 fake controls (4 from site group A)

### Test set: shortcut broken

20 fake patients (10 from site group A)

20 fake controls (10 from site group A)

Fake patientFake controlFrom site group A (decoy shift)

### The four possible outcomes

Explainer highlights planted (teal) edges

Explainer highlights decoy (amber) edges

Model learned the real effect\
test accuracy holds

ideal\
Model and explainer are both right.

explainer wrong\
Explainer blames the scanner for nothing.

Model used the shortcut\
test accuracy collapses

most dangerous\
Explainer hides the cheating and shows a neat DMN story. Strongest possible finding.

explainer honest\
Explainer exposes the shortcut. Useful as a safety check.

7 · Three kinds of "good explanation"

## Planting is what makes correctness measurable

Plausible

Looks like what neuroscientists expect, e.g. highlights the DMN.

Checkable on real data, but circular.

Faithful

Matches what the **model** used. Test by deleting top edges and watching the prediction drop.

Checkable on real data.

Correct

Matches the **true** group difference.

Only checkable with planted ground truth.

8 · Terms

## Glossary

Planting

Deliberately editing chosen connections in real data so the true difference is known.

Planted set S

The list of edited connections. The answer key.

Decoy set D

A second edited list tied to scanner site, used to tempt the model into a shortcut.

δ (effect size)

How much the planted connections change, in standard deviations.

ρ (shortcut strength)

How strongly training labels go with site group A (16 of 20 above means 0.8).

Explainer

A method that scores every connection by how much it mattered to the prediction.

Precision@k

Of the explainer's top k connections, the fraction that were planted.

AUPRC

Main score: rewards ranking planted edges at the very top. Chance = 0.005 here.

Stability

How much the top-k list overlaps across random seeds (Jaccard; target ≥ 0.5).

Shortcut

Getting the right answer for the wrong reason, e.g. recognising the scanner instead of the disease.