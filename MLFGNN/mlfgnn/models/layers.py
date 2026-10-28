"""Message-passing layers, implemented on plain PyTorch.

The four layer types compared in Peng et al. -- GCN, GraphSAGE, Residual Gated
GCN and GAT -- written directly with ``index_add_``/``scatter_reduce_`` rather
than pulled from torch-geometric. Population graphs here have a few hundred
nodes, so there is nothing to gain from a specialised library, and dropping the
dependency removes the single most fragile part of the environment (PyG wheels
are pinned to exact torch/CUDA builds).

Every layer returns ``(h, edge_weight_out)``. The second element is what the
network-fusion path averages across modality branches: for GAT it is the
learned attention, for Res-GGCN the edge gate, and for GCN/GraphSAGE -- which
learn no per-edge quantity -- the input similarity passed through unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# scatter helpers
# ---------------------------------------------------------------------------
def scatter_sum(src: Tensor, index: Tensor, n: int) -> Tensor:
    shape = (n,) + src.shape[1:]
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    return out.scatter_add_(0, idx, src)


def scatter_softmax(src: Tensor, index: Tensor, n: int) -> Tensor:
    """Softmax over all entries sharing a destination index."""
    shape = (n,) + src.shape[1:]
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)

    maxes = torch.full(shape, float("-inf"), dtype=src.dtype, device=src.device)
    maxes = maxes.scatter_reduce(0, idx, src, reduce="amax", include_self=True)
    maxes = torch.nan_to_num(maxes, neginf=0.0)

    exp = (src - maxes.gather(0, idx)).exp()
    denom = scatter_sum(exp, index, n).gather(0, idx)
    return exp / denom.clamp_min(1e-16)


def add_self_loops(
    edge_index: Tensor, edge_weight: Tensor, n: int, fill: float = 1.0
) -> tuple[Tensor, Tensor]:
    loop = torch.arange(n, device=edge_index.device)
    loops = torch.stack([loop, loop])
    w = torch.full((n,), fill, dtype=edge_weight.dtype, device=edge_weight.device)
    return torch.cat([edge_index, loops], dim=1), torch.cat([edge_weight, w])


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------
class GCNConv(nn.Module):
    """Kipf & Welling, with edge weights folded into the normalisation."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor):
        n = x.size(0)
        ei, ew = add_self_loops(edge_index, edge_weight, n)
        src, dst = ei[0], ei[1]

        deg = scatter_sum(ew, dst, n).clamp_min(1e-12)
        norm = ew * deg[src].pow(-0.5) * deg[dst].pow(-0.5)

        h = self.lin(x)
        out = scatter_sum(norm.unsqueeze(-1) * h[src], dst, n)
        return out, edge_weight


class SAGEConv(nn.Module):
    """GraphSAGE with weighted mean aggregation."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor):
        n = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        num = scatter_sum(edge_weight.unsqueeze(-1) * x[src], dst, n)
        den = scatter_sum(edge_weight, dst, n).clamp_min(1e-12).unsqueeze(-1)
        return self.lin_self(x) + self.lin_neigh(num / den), edge_weight


class ResGatedGCNConv(nn.Module):
    """Bresson & Laurent's residual gated graph ConvNet.

    The edge gate ``eta_ij = sigmoid(A x_i + B x_j)`` is a genuine learned edge
    quantity, which is what the network-fusion path consumes.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.a = nn.Linear(in_dim, out_dim)
        self.b = nn.Linear(in_dim, out_dim)
        self.u = nn.Linear(in_dim, out_dim)
        self.v = nn.Linear(in_dim, out_dim)
        self.res = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor):
        n = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        gate = torch.sigmoid(self.a(x)[dst] + self.b(x)[src])
        gate = gate * edge_weight.unsqueeze(-1)

        num = scatter_sum(gate * self.v(x)[src], dst, n)
        den = scatter_sum(gate, dst, n).clamp_min(1e-12)
        out = self.u(x) + num / den
        return out + self.res(x), gate.mean(dim=-1)


class GATConv(nn.Module):
    """Velickovic et al.'s multi-head attention.

    The population-graph similarity is injected as an additive log-bias on the
    attention logits, so a similarity of 0 suppresses an edge and the layer
    reduces to standard GAT when all weights are 1. This keeps the *edge
    weights produced by graph construction* meaningful rather than discarded --
    which matters here, since graph construction is the variable under study.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.concat = concat
        self.out_dim = out_dim
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor):
        n = x.size(0)
        ei, ew = add_self_loops(edge_index, edge_weight, n)
        src, dst = ei[0], ei[1]

        h = self.lin(x).view(n, self.heads, self.out_dim)
        alpha = (h * self.att_src).sum(-1)[src] + (h * self.att_dst).sum(-1)[dst]
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = alpha + ew.clamp_min(1e-8).log().unsqueeze(-1)
        alpha = scatter_softmax(alpha, dst, n)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = scatter_sum(alpha.unsqueeze(-1) * h[src], dst, n)
        out = out.reshape(n, -1) if self.concat else out.mean(dim=1)
        out = out + self.bias

        # Drop the self-loop entries so the returned weights align 1:1 with the
        # input edge list.
        n_real = edge_index.size(1)
        return out, alpha[:n_real].mean(dim=-1)


LAYERS: dict[str, type[nn.Module]] = {
    "GCN": GCNConv,
    "GraphSAGE": SAGEConv,
    "ResGatedGCN": ResGatedGCNConv,
    "GAT": GATConv,
}


def make_layer(kind: str, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.0):
    """Build one layer, hiding the GAT-only arguments from the caller.

    GAT heads are averaged rather than concatenated so that every layer type
    maps ``in_dim -> out_dim`` and the rest of the model is layer-agnostic.
    """
    if kind not in LAYERS:
        raise ValueError(f"unknown layer {kind!r}; expected one of {list(LAYERS)}")
    if kind == "GAT":
        return GATConv(in_dim, out_dim, heads=heads, concat=False, dropout=dropout)
    return LAYERS[kind](in_dim, out_dim)
