"""The three population-graph construction rules.

This module is the entire methodological contribution. Features, GNN layers,
MLP and Cox head are held identical to Peng et al.; only the function that
turns a similarity matrix into an edge set changes.

    snf_threshold        edge wherever fused similarity > 0.2      (Peng et al.)
    knn                  edge to the k most similar patients       (control)
    centre_balanced_knn  edge to the k most similar patients FROM
                         EACH OTHER CENTRE                          (proposed)

Why the third rule is the interesting one: scanner and protocol effects make
same-centre patients look alike, so a fixed similarity threshold connects
patients mostly within their own centre and the network learns centre-specific
neighbourhoods. The centre-balanced rule forbids within-centre edges outright,
forcing every message-passing step to cross an institutional boundary.

A masking detail that matters: an edge may only be *drawn* using information
available at training time. ``build_graph`` therefore takes an explicit
``allowed_targets`` mask, and the LOCO driver passes "training nodes only" so
that held-out-centre patients are attached to -- but never wired among --
the training graph.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

MODES = ("snf_threshold", "knn", "centre_balanced_knn")


@dataclass
class Graph:
    """A weighted undirected graph in COO form.

    ``edge_index`` is ``(2, E)`` with both directions stored, matching the
    convention message-passing layers expect.
    """

    edge_index: np.ndarray
    edge_weight: np.ndarray
    n_nodes: int
    mode: str
    params: dict

    @property
    def n_edges(self) -> int:
        """Undirected edge count."""
        return self.edge_index.shape[1] // 2

    def degrees(self) -> np.ndarray:
        """Neighbourhood size: the number of edges arriving at each node.

        In-degree rather than out-degree, because the two coincide on the
        symmetric training graph but only in-degree is meaningful on the
        evaluation graph, where held-out patients receive messages and send
        none. Counting out-edges there would report every attached test node as
        isolated.
        """
        deg = np.zeros(self.n_nodes)
        np.add.at(deg, self.edge_index[1], 1.0)
        return deg

    def out_degrees(self) -> np.ndarray:
        deg = np.zeros(self.n_nodes)
        np.add.at(deg, self.edge_index[0], 1.0)
        return deg

    def to_dense(self) -> np.ndarray:
        a = np.zeros((self.n_nodes, self.n_nodes))
        a[self.edge_index[0], self.edge_index[1]] = self.edge_weight
        return a


# ---------------------------------------------------------------------------
# Edge rules -- each returns a boolean (n, n) directed adjacency, row = source
# ---------------------------------------------------------------------------
def threshold_edges(
    similarity: np.ndarray, threshold: float, allowed: np.ndarray
) -> np.ndarray:
    """Peng et al.: connect whenever fused similarity exceeds a fixed value."""
    adj = (similarity > threshold) & allowed
    np.fill_diagonal(adj, False)
    return adj


def knn_edges(similarity: np.ndarray, k: int, allowed: np.ndarray) -> np.ndarray:
    """Connect each patient to its k most similar patients overall."""
    n = similarity.shape[0]
    adj = np.zeros((n, n), dtype=bool)
    masked = np.where(allowed, similarity, -np.inf)
    np.fill_diagonal(masked, -np.inf)
    for i in range(n):
        candidates = np.flatnonzero(np.isfinite(masked[i]))
        if candidates.size == 0:
            continue
        top = candidates[np.argsort(-masked[i, candidates])[:k]]
        adj[i, top] = True
    return adj


def centre_balanced_knn(
    similarity: np.ndarray,
    centres: np.ndarray,
    k: int,
    allowed: np.ndarray,
) -> np.ndarray:
    """**The proposed rule.**

    For patient *i*, select the *k* most similar patients from **each** centre
    *c != centre(i)*. Within-centre edges are never drawn, so the neighbourhood
    of every node is by construction an equal mixture of the other institutions.

    Two consequences worth stating explicitly in the paper:

    * The degree is ``k * (n_other_centres)`` before symmetrisation, so it is
      *fixed by construction* and cannot drift with cohort composition -- unlike
      a threshold rule, whose degree depends on how tightly the cohort clusters.
    * A patient from an unseen centre is attached by exactly the same rule
      against the training centres, which is what makes the model inductive.
    """
    n = similarity.shape[0]
    adj = np.zeros((n, n), dtype=bool)
    masked = np.where(allowed, similarity, -np.inf)
    np.fill_diagonal(masked, -np.inf)

    unique = np.unique(centres)
    centre_members = {c: np.flatnonzero(centres == c) for c in unique}

    for i in range(n):
        for c in unique:
            if c == centres[i]:
                continue                      # no within-centre edges, by design
            members = centre_members[c]
            scores = masked[i, members]
            finite = np.flatnonzero(np.isfinite(scores))
            if finite.size == 0:
                continue
            take = min(k, finite.size)
            best = finite[np.argsort(-scores[finite])[:take]]
            adj[i, members[best]] = True
    return adj


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def build_graph(
    similarity: np.ndarray,
    centres: np.ndarray,
    mode: str,
    k: int = 5,
    threshold: float = 0.2,
    allowed_targets: np.ndarray | None = None,
    symmetrise: str = "union",
    min_degree: int = 1,
) -> Graph:
    """Turn a similarity matrix into a :class:`Graph` under the chosen rule.

    Parameters
    ----------
    allowed_targets:
        Boolean ``(n,)`` mask of nodes an edge is permitted to *point at*. Under
        leave-one-centre-out this is the training-node mask, which is what stops
        held-out patients from forming a private sub-graph of their own and
        keeps every arm of the comparison inductive in the same way. ``None``
        allows everything (the transductive, Peng-style setting).
    symmetrise:
        ``union`` keeps an edge if either endpoint selected the other (standard
        for k-NN graphs); ``mutual`` requires both.
    min_degree:
        Nodes left isolated are reconnected to their ``min_degree`` most similar
        allowed neighbours. An isolated node in a population GNN silently falls
        back to its own features, which would quietly turn part of the held-out
        centre into an MLP and confound the comparison.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")

    n = similarity.shape[0]
    centres = np.asarray(centres)
    allowed = (
        np.ones((n, n), dtype=bool)
        if allowed_targets is None
        else np.broadcast_to(np.asarray(allowed_targets, dtype=bool), (n, n)).copy()
    )

    if mode == "snf_threshold":
        adj = threshold_edges(similarity, threshold, allowed)
        params = {"threshold": threshold}
    elif mode == "knn":
        adj = knn_edges(similarity, k, allowed)
        params = {"k": k}
    else:
        adj = centre_balanced_knn(similarity, centres, k, allowed)
        params = {"k_per_centre": k, "n_centres": int(len(np.unique(centres)))}

    adj = np.logical_or(adj, adj.T) if symmetrise == "union" else np.logical_and(adj, adj.T)
    np.fill_diagonal(adj, False)

    adj = _repair_isolated(adj, similarity, allowed, min_degree)

    src, dst = np.nonzero(adj)
    weight = similarity[src, dst].astype(np.float32)
    # Similarities are non-negative by construction, but a numerically tiny
    # weight would zero out a legitimate edge during normalisation.
    weight = np.clip(weight, 1e-6, None)

    graph = Graph(
        edge_index=np.vstack([src, dst]).astype(np.int64),
        edge_weight=weight,
        n_nodes=n,
        mode=mode,
        params={**params, "symmetrise": symmetrise},
    )
    logger.debug(
        "%s: %d nodes, %d undirected edges, mean degree %.1f",
        mode, n, graph.n_edges, graph.degrees().mean(),
    )
    return graph


def _repair_isolated(
    adj: np.ndarray, similarity: np.ndarray, allowed: np.ndarray, min_degree: int
) -> np.ndarray:
    """Give every isolated node at least ``min_degree`` neighbours."""
    if min_degree <= 0:
        return adj
    degree = adj.sum(axis=1)
    isolated = np.flatnonzero(degree == 0)
    if isolated.size == 0:
        return adj
    logger.info("reconnecting %d isolated node(s)", isolated.size)
    masked = np.where(allowed, similarity, -np.inf)
    np.fill_diagonal(masked, -np.inf)
    for i in isolated:
        candidates = np.flatnonzero(np.isfinite(masked[i]))
        if candidates.size == 0:
            continue
        top = candidates[np.argsort(-masked[i, candidates])[:min_degree]]
        adj[i, top] = True
        adj[top, i] = True
    return adj
