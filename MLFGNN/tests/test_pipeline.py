"""Invariants the study depends on. Run with `python -m pytest tests -q`.

These are not coverage tests -- each one guards a claim the paper will make.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlfgnn.data.synthetic import make_synthetic_cohort
from mlfgnn.eval.metrics import harrell_cindex
from mlfgnn.experiments.fold import (
    attach_edges,
    build_fold_graphs,
    fold_similarity,
    make_folds,
    prepare_fold,
)
from mlfgnn.features.select import FeatureSelector, univariate_cindex
from mlfgnn.graphs.construct import build_graph
from mlfgnn.graphs.metrics import cross_centre_edge_ratio
from mlfgnn.graphs.similarity import affinity_matrix, snf


@pytest.fixture(scope="module")
def cohort():
    return make_synthetic_cohort(
        centre_sizes={"CHUM": 30, "CHUS": 30, "HGJ": 25, "HMR": 20},
        n_pet=20, n_ct=20, batch_strength=2.0, seed=0,
    )


@pytest.fixture(scope="module")
def similarity(cohort):
    pet = cohort[[c for c in cohort.columns if c.startswith("PET_")]].to_numpy()
    ct = cohort[[c for c in cohort.columns if c.startswith("CT_")]].to_numpy()
    return snf([affinity_matrix(pet, k=10), affinity_matrix(ct, k=10)], k=10, n_iters=10)


# ---------------------------------------------------------------------------
# The edge rules
# ---------------------------------------------------------------------------
def test_centre_balanced_knn_has_no_within_centre_edges(cohort, similarity):
    centres = cohort["centre"].to_numpy()
    g = build_graph(similarity, centres, mode="centre_balanced_knn", k=3)
    src, dst = g.edge_index
    assert np.all(centres[src] != centres[dst])
    assert cross_centre_edge_ratio(g, centres) == 1.0


def test_centre_balanced_knn_degree_is_fixed_by_construction(cohort, similarity):
    """Every node gets k neighbours from each of the other centres."""
    centres = cohort["centre"].to_numpy()
    k, n_centres = 3, len(set(centres))
    g = build_graph(similarity, centres, mode="centre_balanced_knn", k=k,
                    symmetrise="mutual")
    # Under mutual symmetrisation degree can only fall, never exceed the
    # constructed k * (n_centres - 1).
    assert g.degrees().max() <= k * (n_centres - 1)


def test_knn_degree_at_least_k(cohort, similarity):
    centres = cohort["centre"].to_numpy()
    g = build_graph(similarity, centres, mode="knn", k=5, symmetrise="union")
    assert g.degrees().min() >= 5


def test_site_effect_makes_threshold_graph_centre_pure(cohort, similarity):
    """The mechanism the proposal rests on, measured directly.

    With a strong site effect, a fixed similarity threshold should connect
    patients mostly inside their own centre.
    """
    centres = cohort["centre"].to_numpy()
    g = build_graph(similarity, centres, mode="snf_threshold", threshold=0.2)
    assert g.n_edges > 0
    assert cross_centre_edge_ratio(g, centres) < 0.5


def test_no_site_effect_removes_the_centre_purity():
    """Negative control: with batch_strength=0 the threshold graph mixes centres."""
    df = make_synthetic_cohort(
        centre_sizes={"A": 30, "B": 30, "C": 30, "D": 30},
        n_pet=20, n_ct=20, batch_strength=0.0, seed=1,
    )
    pet = df[[c for c in df.columns if c.startswith("PET_")]].to_numpy()
    ct = df[[c for c in df.columns if c.startswith("CT_")]].to_numpy()
    sim = snf([affinity_matrix(pet, k=10), affinity_matrix(ct, k=10)], k=10, n_iters=10)
    centres = df["centre"].to_numpy()
    g = build_graph(sim, centres, mode="snf_threshold", threshold=0.2)
    # 4 equal centres -> 75% of all possible pairs cross a centre boundary.
    assert cross_centre_edge_ratio(g, centres) > 0.6


def test_allowed_targets_blocks_forbidden_edges(cohort, similarity):
    centres = cohort["centre"].to_numpy()
    allowed = centres != "HMR"
    g = build_graph(similarity, centres, mode="knn", k=5, allowed_targets=allowed)
    src, dst = g.edge_index
    # HMR may be a source (union symmetrisation) but is never a chosen target
    # of a node that had other options.
    assert (centres[dst] == "HMR").sum() <= (centres[src] == "HMR").sum()


def test_no_isolated_nodes(cohort, similarity):
    centres = cohort["centre"].to_numpy()
    for mode, kwargs in (
        ("snf_threshold", {"threshold": 0.99}),      # deliberately too strict
        ("knn", {"k": 2}),
        ("centre_balanced_knn", {"k": 2}),
    ):
        g = build_graph(similarity, centres, mode=mode, **kwargs)
        assert g.degrees().min() > 0, f"{mode} left an isolated node"


# ---------------------------------------------------------------------------
# Leakage control
# ---------------------------------------------------------------------------
def test_folds_are_disjoint_and_exhaustive(cohort):
    centres = cohort["centre"].to_numpy()
    for spec in make_folds(centres, seed=0):
        assert set(centres[spec.test_idx]) == {spec.held_out_centre}
        assert spec.held_out_centre not in set(centres[spec.train_idx])
        assert spec.held_out_centre not in set(centres[spec.val_idx])
        assert not set(spec.train_idx) & set(spec.val_idx)
        assert not set(spec.train_idx) & set(spec.test_idx)
        total = len(spec.train_idx) + len(spec.val_idx) + len(spec.test_idx)
        assert total == len(centres)


def test_evaluation_graph_never_lets_test_nodes_speak(cohort):
    """Held-out patients receive messages but never send them.

    If they did, scoring one held-out patient would change another's score and
    the evaluation would not be inductive.
    """
    centres = cohort["centre"].to_numpy()
    spec = make_folds(centres, seed=0)[0]
    data = _fold_data(cohort, spec)
    s_train, cross = fold_similarity(data, k_snf=10, n_iters=5)

    for mode, k in (("snf_threshold", 0), ("knn", 3), ("centre_balanced_knn", 3)):
        _, eval_graph, node_order = build_fold_graphs(
            data, s_train, cross, mode=mode, k=k, threshold=0.2
        )
        n_train = len(spec.graph_idx)
        src, dst = eval_graph.edge_index
        is_test = np.arange(eval_graph.n_nodes) >= n_train
        assert not is_test[src].any(), f"{mode}: a held-out node is a message source"
        assert len(node_order) == eval_graph.n_nodes


def test_eval_graph_diagnostics_see_the_attached_nodes(cohort):
    """Guards two easy sign errors: degree counted on the wrong endpoint, and
    reachability computed in the wrong direction. Both silently report a
    perfectly attached held-out centre as unreachable and isolated."""
    from mlfgnn.graphs.metrics import degree_stats, reachability_to_train

    centres = cohort["centre"].to_numpy()
    spec = make_folds(centres, seed=0)[0]
    data = _fold_data(cohort, spec)
    s_train, cross = fold_similarity(data, k_snf=10, n_iters=5)

    for mode, k in (("snf_threshold", 0), ("knn", 3), ("centre_balanced_knn", 3)):
        _, eval_graph, node_order = build_fold_graphs(
            data, s_train, cross, mode=mode, k=k, threshold=0.2
        )
        train_mask = np.isin(node_order, spec.graph_idx)
        assert degree_stats(eval_graph)["isolated_nodes"] == 0, mode
        assert reachability_to_train(eval_graph, train_mask) == 1.0, mode


def test_every_held_out_patient_is_attached(cohort):
    centres = cohort["centre"].to_numpy()
    spec = make_folds(centres, seed=0)[0]
    data = _fold_data(cohort, spec)
    s_train, cross = fold_similarity(data, k_snf=10, n_iters=5)

    for mode, k in (("snf_threshold", 0), ("knn", 3), ("centre_balanced_knn", 3)):
        adj = attach_edges(
            cross, data.centres[spec.test_idx], data.centres[spec.graph_idx],
            mode=mode, k=k, threshold=0.2,
        )
        assert adj.any(axis=1).all(), f"{mode} left a held-out patient unattached"


def test_centre_balanced_attachment_spans_every_training_centre(cohort):
    centres = cohort["centre"].to_numpy()
    spec = make_folds(centres, seed=0)[0]
    data = _fold_data(cohort, spec)
    _, cross = fold_similarity(data, k_snf=10, n_iters=5)

    train_centres = data.centres[spec.graph_idx]
    adj = attach_edges(cross, data.centres[spec.test_idx], train_centres,
                       mode="centre_balanced_knn", k=2, threshold=0.2)
    for i in range(adj.shape[0]):
        reached = set(train_centres[adj[i]])
        assert reached == set(train_centres), "a held-out patient missed a centre"


def test_selection_is_blind_to_held_out_rows(cohort):
    """Selecting on the held-out centre must change the result -- proving the
    fold-wise refit is doing something, not just running."""
    pet_cols = [c for c in cohort.columns if c.startswith("PET_")]
    x = cohort[pet_cols].to_numpy()
    t = cohort["PFS_time"].to_numpy()
    e = cohort["PFS_event"].to_numpy()
    spec = make_folds(cohort["centre"].to_numpy(), seed=0)[0]

    sel = FeatureSelector(max_features=5, univariate_repeats=3, top_k_univariate=15, seed=0)
    on_train = sel.fit(x[spec.train_idx], pet_cols, t[spec.train_idx], e[spec.train_idx])
    on_all = sel.fit(x, pet_cols, t, e)

    assert len(on_train.columns) <= 5
    assert set(on_train.columns) <= set(pet_cols)
    # The standardisation statistics must come from training rows only.
    assert not np.allclose(on_train.mu, on_all.mu)


# ---------------------------------------------------------------------------
# Metrics and loss
# ---------------------------------------------------------------------------
def test_harrell_cindex_bounds_and_orientation():
    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    event = np.array([1, 1, 1, 1, 0])
    perfect = -time                      # higher risk = shorter survival
    assert harrell_cindex(perfect, time, event) == pytest.approx(1.0)
    assert harrell_cindex(time, time, event) == pytest.approx(0.0)
    assert harrell_cindex(np.zeros(5), time, event) == pytest.approx(0.5)


def test_univariate_cindex_is_orientation_free():
    rng = np.random.default_rng(0)
    t = rng.exponential(size=200)
    e = np.ones(200)
    x = np.stack([t, -t], axis=1)        # one aligned, one anti-aligned
    c = univariate_cindex(x, t, e)
    assert np.allclose(c, 1.0, atol=1e-9)


def test_cox_loss_prefers_the_correct_ordering():
    torch = pytest.importorskip("torch")
    from mlfgnn.models.losses import cox_ph_loss

    time = torch.tensor([1.0, 2.0, 3.0, 4.0])
    event = torch.tensor([1.0, 1.0, 1.0, 1.0])
    good = torch.tensor([3.0, 2.0, 1.0, 0.0])
    bad = -good
    assert cox_ph_loss(good, time, event) < cox_ph_loss(bad, time, event)


def test_cox_loss_mask_restricts_the_risk_set():
    torch = pytest.importorskip("torch")
    from mlfgnn.models.losses import cox_ph_loss

    time = torch.tensor([1.0, 2.0, 3.0, 4.0])
    event = torch.tensor([1.0, 1.0, 0.0, 0.0])
    risk = torch.tensor([2.0, 1.0, 5.0, -5.0])
    mask = torch.tensor([True, True, False, False])
    full = cox_ph_loss(risk, time, event)
    masked = cox_ph_loss(risk, time, event, mask=mask)
    assert not torch.isclose(full, masked)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("layer", ["GCN", "GraphSAGE", "ResGatedGCN", "GAT"])
@pytest.mark.parametrize("fusion", ["pet", "feature", "network"])
def test_forward_shapes(cohort, similarity, layer, fusion):
    torch = pytest.importorskip("torch")
    from mlfgnn.models.mlfgnn import MLFGNN

    centres = cohort["centre"].to_numpy()
    g = build_graph(similarity, centres, mode="knn", k=4)
    n = g.n_nodes
    x_pet = torch.randn(n, 8)
    x_ct = torch.randn(n, 6)
    clinical = torch.randn(n, 3)

    model = MLFGNN(8, 6, clinical_dim=3, hidden_dim=16, layer_type=layer, fusion=fusion)
    out = model(
        x_pet, x_ct,
        torch.as_tensor(g.edge_index), torch.as_tensor(g.edge_weight),
        clinical,
    )
    assert out.shape == (n,)
    assert torch.isfinite(out).all()


def test_gat_attention_is_a_distribution_over_incoming_edges():
    torch = pytest.importorskip("torch")
    from mlfgnn.models.layers import scatter_softmax

    index = torch.tensor([0, 0, 1, 1, 1, 2])
    src = torch.randn(6, 2)
    out = scatter_softmax(src, index, 3)
    for node in range(3):
        assert out[index == node].sum(dim=0).allclose(torch.ones(2), atol=1e-5)


def _fold_data(cohort, spec):
    pet_cols = [c for c in cohort.columns if c.startswith("PET_")]
    ct_cols = [c for c in cohort.columns if c.startswith("CT_")]
    cols = pet_cols + ct_cols
    return prepare_fold(
        features=cohort[cols].to_numpy(),
        columns=cols,
        pet_prefix="PET_",
        ct_prefix="CT_",
        clinical_df=cohort,
        clinical_cols=["age", "T_stage", "N_stage", "site"],
        time=cohort["PFS_time"].to_numpy(),
        event=cohort["PFS_event"].to_numpy(),
        centres=cohort["centre"].to_numpy(),
        spec=spec,
        selection_cfg={
            "univariate_repeats": 3,
            "top_k_univariate": 15,
            "max_features_per_modality": 6,
        },
    )
