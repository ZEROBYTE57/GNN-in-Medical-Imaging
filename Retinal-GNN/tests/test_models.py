import pytest
import torch

from retinal_gnn.models import GraphAttentionBlock, build_model, grid_adjacency


def test_grid_adjacency():
    a = grid_adjacency(4, 5)
    assert a.shape == (20, 20)
    assert a.diagonal().all()                 # self loops
    assert a.sum(1).max() == 9                # interior node: 8 neighbours + self
    assert a.sum(1).min() == 4                # corner: 3 neighbours + self
    assert (a == a.T).all()                   # undirected


def test_knn_mask_is_superset_of_grid():
    blk = GraphAttentionBlock(16, heads=2, knn=4)
    t = torch.randn(2, 36, 16)
    m = blk.build_mask(t, 6, 6)
    assert m.shape == (2, 36, 36)
    assert (m | grid_adjacency(6, 6)).equal(m)
    assert m.sum(-1).min() >= 4


@pytest.mark.parametrize("name", ["unet", "graph_unet_noedge", "graph_unet_grid", "graph_unet"])
def test_forward_backward(name):
    torch.manual_seed(0)
    m = build_model(name, base=8)
    x = torch.randn(2, 3, 64, 64)
    y = m(x)
    assert y.shape == (2, 1, 64, 64)
    y.mean().backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_graph_block_changes_output_and_is_finite():
    blk = GraphAttentionBlock(16, heads=2, knn=4).eval()
    x = torch.randn(1, 16, 6, 6)
    y = blk(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert not torch.allclose(x, y)


def test_noedge_control_matches_param_count():
    n = lambda k: sum(p.numel() for p in build_model(k, 8).parameters())
    assert n("graph_unet_noedge") == n("graph_unet_grid") == n("graph_unet")
    assert n("unet") < n("graph_unet")
