"""Centre-aware population graph construction for cross-institution HNC survival.

Reproduces the MLF-GNN pipeline of Peng et al. (Biomed. Signal Process. Control
92, 2024, 106137) and changes exactly one component -- the population-graph edge
rule -- so that the effect of graph construction on cross-centre generalisation
can be measured in isolation.
"""

__version__ = "0.1.0"
