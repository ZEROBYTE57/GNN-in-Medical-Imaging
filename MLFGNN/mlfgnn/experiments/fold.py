"""Per-fold preprocessing: selection, similarity and graph assembly.

Isolated from the training loop because this is where every leakage decision
lives, and those decisions are the difference between a benchmark and a
flattering number. Three rules are enforced here:

1. **Selection, standardisation and clinical encoding are fit on the inner
   training rows only.** Not on the training centres as a whole (that would
   leak into the in-centre validation estimate) and certainly not on the full
   cohort.

2. **The training graph contains training-centre nodes only.** If held-out
   patients were nodes of the graph the model trains on, their features would
   flow into training-node embeddings through message passing, and the
   "held-out" centre would not be held out.

3. **Held-out patients are attached by directed edges only.** Messages flow
   train -> held-out, never the reverse, so scoring one held-out patient cannot
   change the score of another. This is what makes the evaluation genuinely
   inductive, and it applies identically to all three graph rules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..data.clinical import encode_clinical
from ..features.select import FeatureSelector
from ..graphs.construct import Graph, build_graph
from ..graphs.similarity import affinity_matrix, scaled_exponential_kernel, snf
from ..seed import rng

logger = logging.getLogger(__name__)


@dataclass
class FoldSpec:
    """Index masks defining one leave-one-centre-out fold."""

    held_out_centre: str
    train_idx: np.ndarray          # inner-train: fits everything
    val_idx: np.ndarray            # in-centre validation: early stopping + in-centre metric
    test_idx: np.ndarray           # the held-out centre
    seed: int

    @property
    def fit_idx(self) -> np.ndarray:
        """Rows any estimator is allowed to learn from."""
        return self.train_idx

    @property
    def graph_idx(self) -> np.ndarray:
        """Nodes of the training graph: inner-train plus in-centre validation."""
        return np.sort(np.concatenate([self.train_idx, self.val_idx]))


def make_folds(
    centres: np.ndarray,
    seed: int,
    inner_val_fraction: float = 0.2,
) -> list[FoldSpec]:
    """One fold per centre; the inner split is stratified by centre.

    Stratifying the inner split matters: an unstratified 20% draw from four
    centres of very different sizes can leave a small centre almost absent from
    validation, which makes the in-centre number -- the reference point of the
    generalisation gap -- unstable across seeds.
    """
    centres = np.asarray(centres)
    g = rng(seed)
    folds: list[FoldSpec] = []

    for held_out in sorted(set(centres.tolist())):
        pool = np.flatnonzero(centres != held_out)
        test_idx = np.flatnonzero(centres == held_out)

        train_parts, val_parts = [], []
        for c in sorted(set(centres[pool].tolist())):
            members = np.flatnonzero(centres == c)
            members = members[g.permutation(len(members))]
            n_val = max(1, int(round(inner_val_fraction * len(members))))
            val_parts.append(members[:n_val])
            train_parts.append(members[n_val:])

        folds.append(
            FoldSpec(
                held_out_centre=held_out,
                train_idx=np.sort(np.concatenate(train_parts)),
                val_idx=np.sort(np.concatenate(val_parts)),
                test_idx=test_idx,
                seed=seed,
            )
        )
    return folds


@dataclass
class FoldData:
    """Everything a model needs for one fold, with train/test separation baked in."""

    x_pet: np.ndarray                     # (n_all, d_pet) selected + standardised
    x_ct: np.ndarray
    clinical: np.ndarray
    time: np.ndarray
    event: np.ndarray
    centres: np.ndarray
    spec: FoldSpec
    pet_columns: list[str] = field(default_factory=list)
    ct_columns: list[str] = field(default_factory=list)
    selection_stages: dict = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return len(self.time)


def prepare_fold(
    features: np.ndarray,
    columns: list[str],
    pet_prefix: str,
    ct_prefix: str,
    clinical_df,
    clinical_cols: list[str],
    time: np.ndarray,
    event: np.ndarray,
    centres: np.ndarray,
    spec: FoldSpec,
    selection_cfg: dict,
) -> FoldData:
    """Run the selection cascade per modality and standardise the whole cohort."""
    pet_cols = [c for c in columns if c.startswith(pet_prefix)]
    ct_cols = [c for c in columns if c.startswith(ct_prefix)]
    if not pet_cols or not ct_cols:
        raise ValueError(
            f"no columns with prefix {pet_prefix!r} / {ct_prefix!r}; "
            "check the feature table schema"
        )
    index = {c: i for i, c in enumerate(columns)}

    selected: dict[str, np.ndarray] = {}
    kept: dict[str, list[str]] = {}
    stages: dict[str, dict] = {}

    for name, cols in (("pet", pet_cols), ("ct", ct_cols)):
        block = features[:, [index[c] for c in cols]]
        selector = FeatureSelector(
            variance_threshold=selection_cfg.get("variance_threshold", 0.005),
            univariate_repeats=selection_cfg.get("univariate_repeats", 50),
            univariate_splits=selection_cfg.get("univariate_splits", 3),
            top_k_univariate=selection_cfg.get("top_k_univariate", 50),
            spearman_threshold=selection_cfg.get("spearman_threshold", 0.8),
            lasso_cv_folds=selection_cfg.get("lasso_cv_folds", 5),
            max_features=selection_cfg.get("max_features_per_modality", 12),
            seed=spec.seed,
        )
        result = selector.fit(
            block[spec.fit_idx], cols, time[spec.fit_idx], event[spec.fit_idx]
        )
        selected[name] = result.transform(block, cols)
        kept[name] = result.columns
        stages[name] = result.stage_counts

    clinical = np.zeros((len(time), 0))
    if clinical_df is not None and clinical_cols:
        clinical, _ = encode_clinical(
            clinical_df, clinical_cols, reference=clinical_df.iloc[spec.fit_idx]
        )

    return FoldData(
        x_pet=selected["pet"],
        x_ct=selected["ct"],
        clinical=clinical,
        time=time,
        event=event,
        centres=centres,
        spec=spec,
        pet_columns=kept["pet"],
        ct_columns=kept["ct"],
        selection_stages=stages,
    )


# ---------------------------------------------------------------------------
# Similarity, fit on the training graph and extended out-of-sample
# ---------------------------------------------------------------------------
def fold_similarity(
    data: FoldData,
    k_snf: int = 20,
    mu: float = 0.5,
    n_iters: int = 20,
    scope: str = "train_fit",
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(S_train, S_cross)``.

    ``S_train`` is the fused similarity among training-graph nodes.
    ``S_cross`` is ``(n_test, n_train)``: each held-out patient against every
    training node.

    Out-of-sample similarity is the awkward part. SNF is a transductive
    diffusion -- there is no exact way to place a new patient on the fused
    network without recomputing it. ``scope="train_fit"`` computes SNF on the
    training nodes, then scores held-out patients with the plain multi-modality
    kernel and **quantile-maps** the result onto the training similarity
    distribution.

    That mapping is not cosmetic. A fixed threshold of 0.2 is only defined
    relative to a similarity scale, and the raw kernel and the diffused SNF
    output do not share one -- so without it, the threshold rule cannot be
    applied to a new centre at all. The k-NN rules are invariant to any monotone
    transform and are therefore unaffected, which is itself an argument for
    them: a rank-based rule transports across sites, an absolute threshold
    does not.

    ``scope="transductive"`` reproduces the original setting -- SNF over the
    whole cohort at once -- and is retained as an ablation.
    """
    gi, ti = data.spec.graph_idx, data.spec.test_idx

    if scope == "transductive":
        full = snf(
            [
                affinity_matrix(data.x_pet, k=k_snf, mu=mu),
                affinity_matrix(data.x_ct, k=k_snf, mu=mu),
            ],
            k=k_snf,
            n_iters=n_iters,
        )
        return full[np.ix_(gi, gi)], full[np.ix_(ti, gi)]

    s_train = snf(
        [
            affinity_matrix(data.x_pet[gi], k=k_snf, mu=mu),
            affinity_matrix(data.x_ct[gi], k=k_snf, mu=mu),
        ],
        k=k_snf,
        n_iters=n_iters,
    )
    if ti.size == 0:
        return s_train, np.zeros((0, len(gi)))

    cross = (
        scaled_exponential_kernel(data.x_pet[ti], data.x_pet[gi], k=k_snf, mu=mu)
        + scaled_exponential_kernel(data.x_ct[ti], data.x_ct[gi], k=k_snf, mu=mu)
    ) / 2.0
    cross = _quantile_map(cross, s_train)
    return s_train, cross


def _quantile_map(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map ``values`` onto the empirical distribution of ``reference``'s off-diagonal."""
    n = reference.shape[0]
    ref = np.sort(reference[~np.eye(n, dtype=bool)])
    if ref.size == 0:
        return values
    flat = values.ravel()
    ranks = np.argsort(np.argsort(flat)) / max(flat.size - 1, 1)
    mapped = np.quantile(ref, np.clip(ranks, 0.0, 1.0))
    return mapped.reshape(values.shape)


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def attach_edges(
    cross: np.ndarray,
    test_centres: np.ndarray,
    train_centres: np.ndarray,
    mode: str,
    k: int,
    threshold: float,
) -> np.ndarray:
    """Neighbour mask ``(n_test, n_train)`` under the same rule as the training graph.

    Directed: the returned edges only ever point *into* the held-out node.
    """
    n_test, n_train = cross.shape
    adj = np.zeros((n_test, n_train), dtype=bool)

    if mode == "snf_threshold":
        adj = cross > threshold
    elif mode == "knn":
        for i in range(n_test):
            top = np.argsort(-cross[i])[:k]
            adj[i, top] = True
    elif mode == "centre_balanced_knn":
        # A held-out patient's centre is by definition absent from the training
        # set, so every training centre qualifies as "another centre" and the
        # patient receives k neighbours from each -- the same rule, unchanged.
        for c in np.unique(train_centres):
            members = np.flatnonzero(train_centres == c)
            if members.size == 0:
                continue
            take = min(k, members.size)
            for i in range(n_test):
                best = members[np.argsort(-cross[i, members])[:take]]
                adj[i, best] = True
    else:
        raise ValueError(f"unknown mode {mode!r}")

    # Never leave a held-out patient unattached: an isolated node is scored by
    # the MLP head alone, which would quietly convert part of the test set into
    # a different model.
    for i in np.flatnonzero(~adj.any(axis=1)):
        adj[i, int(np.argmax(cross[i]))] = True
    return adj


def build_fold_graphs(
    data: FoldData,
    s_train: np.ndarray,
    cross: np.ndarray,
    mode: str,
    k: int,
    threshold: float,
    symmetrise: str = "union",
) -> tuple[Graph, Graph, np.ndarray]:
    """Build the training graph and the evaluation graph.

    Returns ``(train_graph, eval_graph, node_order)`` where ``node_order`` maps
    positions in the evaluation graph back to rows of the feature table:
    training-graph nodes first, held-out nodes appended.
    """
    gi, ti = data.spec.graph_idx, data.spec.test_idx
    train_centres = data.centres[gi]

    train_graph = build_graph(
        s_train,
        train_centres,
        mode=mode,
        k=k,
        threshold=threshold,
        allowed_targets=None,          # all nodes here are training nodes
        symmetrise=symmetrise,
    )

    n_train = len(gi)
    if ti.size == 0:
        return train_graph, train_graph, gi

    attach = attach_edges(
        cross, data.centres[ti], train_centres, mode=mode, k=k, threshold=threshold
    )
    test_pos, train_pos = np.nonzero(attach)
    src = train_pos                                  # message source: training node
    dst = n_train + test_pos                         # destination: held-out node

    eval_edge_index = np.hstack(
        [train_graph.edge_index, np.vstack([src, dst]).astype(np.int64)]
    )
    eval_edge_weight = np.concatenate(
        [train_graph.edge_weight, cross[test_pos, train_pos].astype(np.float32)]
    )
    eval_edge_weight = np.clip(eval_edge_weight, 1e-6, None)

    eval_graph = Graph(
        edge_index=eval_edge_index,
        edge_weight=eval_edge_weight,
        n_nodes=n_train + len(ti),
        mode=mode,
        params={**train_graph.params, "attached_test_nodes": int(len(ti))},
    )
    return train_graph, eval_graph, np.concatenate([gi, ti])
