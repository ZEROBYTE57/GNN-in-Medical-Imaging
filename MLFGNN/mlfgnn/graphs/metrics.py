"""Graph assessment metrics.

These are the numbers that explain *why* a graph rule wins or loses, and they
are cheap enough to log for every fold. Cross-centre edge ratio in particular
is the direct measurement of the mechanism the proposal claims: a threshold
graph should score low (edges stay inside centres) and a centre-balanced graph
scores 1.0 by construction.

Following Mueller et al. (MICCAI-W 2023) on extended graph assessment metrics,
homophily is reported for the *outcome* as well, since a graph that mixes
centres but scrambles risk would be no improvement.
"""

from __future__ import annotations

import numpy as np

from .construct import Graph


def cross_centre_edge_ratio(graph: Graph, centres: np.ndarray) -> float:
    """Fraction of edges whose endpoints are in different centres."""
    src, dst = graph.edge_index
    if src.size == 0:
        return float("nan")
    return float(np.mean(np.asarray(centres)[src] != np.asarray(centres)[dst]))


def centre_assortativity(graph: Graph, centres: np.ndarray) -> float:
    """Newman's assortativity coefficient for the categorical centre label.

    +1: edges only within centre (a fully site-segregated graph).
     0: edges are indifferent to centre.
    <0: edges preferentially cross centres.
    """
    src, dst = graph.edge_index
    if src.size == 0:
        return float("nan")
    centres = np.asarray(centres)
    labels, encoded = np.unique(centres, return_inverse=True)
    n_lab = len(labels)

    e = np.zeros((n_lab, n_lab))
    np.add.at(e, (encoded[src], encoded[dst]), 1.0)
    e /= e.sum()

    a = e.sum(axis=1)
    b = e.sum(axis=0)
    trace, expected = np.trace(e), float(a @ b)
    if abs(1.0 - expected) < 1e-12:
        return float("nan")
    return float((trace - expected) / (1.0 - expected))


def risk_homophily(graph: Graph, time: np.ndarray, event: np.ndarray, n_bins: int = 2) -> float:
    """Fraction of edges joining patients in the same outcome stratum.

    Patients are binned by observed time among those with an event; censored
    patients are assigned to the bin their censoring time falls in. This is a
    coarse proxy -- it is meant to detect a graph that mixes centres by
    destroying prognostic structure, not to be a fine-grained quality score.
    """
    src, dst = graph.edge_index
    if src.size == 0:
        return float("nan")
    edges = np.asarray(time)[np.asarray(event) == 1]
    if edges.size < n_bins:
        return float("nan")
    cuts = np.quantile(edges, np.linspace(0, 1, n_bins + 1)[1:-1])
    strata = np.digitize(time, cuts)
    return float(np.mean(strata[src] == strata[dst]))


def degree_stats(graph: Graph) -> dict[str, float]:
    deg = graph.degrees()
    return {
        "degree_mean": float(deg.mean()),
        "degree_std": float(deg.std()),
        "degree_min": float(deg.min()),
        "degree_max": float(deg.max()),
        "isolated_nodes": int((deg == 0).sum()),
    }


def edge_density(graph: Graph) -> float:
    n = graph.n_nodes
    possible = n * (n - 1) / 2
    return float(graph.n_edges / possible) if possible else float("nan")


def reachability_to_train(graph: Graph, train_mask: np.ndarray) -> float:
    """Fraction of held-out nodes with at least one edge into a training node.

    A held-out patient with no path to a training node receives no learned
    message and is effectively predicted by the MLP head alone -- worth
    catching, since it silently weakens whichever arm produces it.
    """
    train_mask = np.asarray(train_mask, dtype=bool)
    test_idx = np.flatnonzero(~train_mask)
    if test_idx.size == 0:
        return float("nan")
    src, dst = graph.edge_index
    # Attachment edges run train -> held-out, so a reachable held-out node is a
    # *destination* of an edge whose source is a training node.
    connected = set(dst[train_mask[src]].tolist())
    return float(np.mean([i in connected for i in test_idx]))


def describe(
    graph: Graph,
    centres: np.ndarray,
    time: np.ndarray | None = None,
    event: np.ndarray | None = None,
    train_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """All diagnostics for one graph, ready to append to the results table."""
    out: dict[str, float] = {
        "mode": graph.mode,
        "n_nodes": graph.n_nodes,
        "n_edges": graph.n_edges,
        "edge_density": edge_density(graph),
        "cross_centre_edge_ratio": cross_centre_edge_ratio(graph, centres),
        "centre_assortativity": centre_assortativity(graph, centres),
        **degree_stats(graph),
    }
    if time is not None and event is not None:
        out["risk_homophily"] = risk_homophily(graph, time, event)
    if train_mask is not None:
        out["heldout_reachability"] = reachability_to_train(graph, train_mask)
    return out
