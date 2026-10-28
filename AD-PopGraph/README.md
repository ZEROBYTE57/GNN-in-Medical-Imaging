# AD-PopGraph

Leakage-free, shortcut-aware evaluation of multi-modal population-graph GNNs for Alzheimer's diagnosis.
Base paper: Zhang et al., Comput Biol Med 164 (2023) 107328 (sMRI + FDG-PET, ADNI).

## Problem statement
Multi-modal population-graph GNNs for AD diagnosis report very high accuracy, but it is unclear how much comes
from the graph and imaging versus a single diagnostic phenotype (MMSE) used to build the edges, versus evaluation
choices that may leak test information. We build a leakage-free inductive protocol, quantify what the graph and each
modality add over strong simple baselines, and propose a diagnostic-variable-free, train-only-normative population graph.

## Research questions
1. Leakage gap: how much reported accuracy survives a leakage-free inductive protocol?
2. Shortcut: how much is explained by MMSE alone?
3. Graph value: at equal inputs, does the graph beat tabular models?

## Contributions
C1 audit protocol + leakage gap | C2 diagnostic-variable-free, train-only-normative inductive graph | C3 open baseline ladder

## Layout
docs/ notes and protocol | data/raw (ADNI downloads, never commit) | data/processed | src/ code | configs/ | results/ | paper/
