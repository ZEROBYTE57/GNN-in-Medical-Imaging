"""Models: a plain U-Net and the same U-Net with a graph-attention bottleneck.

IMPORTANT (honesty note for the paper):
    `GraphUNet` is "VGA-Net-inspired", NOT a reproduction of VGA-Net
    (Jalali et al., IET Image Process. 2024). The official repo is not runnable and
    the full text was not available when this was written. What it shares with
    VGA-Net: graph reasoning over feature-map nodes inside a U-Net. What it does
    not have: VGA-Net's DRIU pixel features, hybrid dilated convolutions,
    BConvLSTM / dual-attention fusion, DropBlock. Do not describe this model as
    VGA-Net in the paper unless those parts are added after reading the paper.

Graph definition
    nodes : every spatial position of the bottleneck feature map (H*W nodes)
    edges : (a) 8-neighbour spatial grid, plus self loops
            (b) optional top-k feature-similarity (cosine) edges, recomputed per image
    layer : multi-head dot-product attention restricted to the edge set
            (a graph-transformer style GAT; not the additive-attention original GAT)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


def grid_adjacency(h: int, w: int, device=None) -> torch.Tensor:
    """Boolean (N, N) matrix: 8-neighbourhood on an h x w grid, with self loops."""
    ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    ys, xs = ys.reshape(-1), xs.reshape(-1)
    dy = (ys[:, None] - ys[None, :]).abs()
    dx = (xs[:, None] - xs[None, :]).abs()
    return (dy <= 1) & (dx <= 1)


class GraphAttentionBlock(nn.Module):
    """Masked multi-head attention over a feature-map graph, with residual + FFN."""

    def __init__(self, dim: int, heads: int = 4, knn: int = 8, dropout: float = 0.1, use_grid: bool = True):
        super().__init__()
        assert dim % heads == 0
        self.dim, self.heads, self.knn, self.use_grid = dim, heads, knn, use_grid
        self.dh = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * dim, dim))
        self.drop = nn.Dropout(dropout)
        self._adj_cache: dict[tuple[int, int, str], torch.Tensor] = {}

    def _adjacency(self, h, w, device):
        key = (h, w, str(device))
        if key not in self._adj_cache:
            self._adj_cache[key] = grid_adjacency(h, w, device)
        return self._adj_cache[key]

    def build_mask(self, t: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """(B, N, N) boolean edge mask from the grid plus optional top-k similarity edges."""
        B, N, _ = t.shape
        if self.use_grid:
            mask = self._adjacency(h, w, t.device).unsqueeze(0).expand(B, N, N)
        else:  # self loops only: capacity-matched control with no message passing
            mask = torch.eye(N, dtype=torch.bool, device=t.device).unsqueeze(0).expand(B, N, N)
        if self.knn > 0:
            tn = F.normalize(t.detach(), dim=-1)
            sim = tn @ tn.transpose(1, 2)                     # (B, N, N)
            k = min(self.knn, N)
            idx = sim.topk(k, dim=-1).indices                 # (B, N, k)
            knn_mask = torch.zeros_like(mask)
            knn_mask.scatter_(2, idx, True)
            mask = mask | knn_mask
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)                      # (B, N, C) node features
        mask = self.build_mask(t, H, W)                       # (B, N, N)
        q, k, v = self.qkv(self.norm1(t)).chunk(3, dim=-1)
        q, k, v = (z.view(B, -1, self.heads, self.dh).transpose(1, 2) for z in (q, k, v))  # (B, h, N, dh)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)  # (B, h, N, N)
        att = att.masked_fill(~mask.unsqueeze(1), float("-inf"))
        att = self.drop(att.softmax(dim=-1))
        out = (att @ v).transpose(1, 2).reshape(B, -1, C)
        t = t + self.drop(self.proj(out))
        t = t + self.drop(self.ffn(self.norm2(t)))
        return t.transpose(1, 2).reshape(B, C, H, W)


class UNet(nn.Module):
    """U-Net. If `graph_layers > 0`, graph-attention blocks are inserted at the bottleneck."""

    def __init__(self, in_ch: int = 3, base: int = 32, depth: int = 4,
                 graph_layers: int = 0, heads: int = 4, knn: int = 8, use_grid: bool = True):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]      # e.g. 32,64,128,256,512
        self.enc = nn.ModuleList([ConvBlock(in_ch if i == 0 else chs[i - 1], chs[i]) for i in range(depth)])
        self.pool = nn.MaxPool2d(2)
        self.bott = ConvBlock(chs[depth - 1], chs[depth])
        self.graph = nn.Sequential(*[GraphAttentionBlock(chs[depth], heads, knn, use_grid=use_grid) for _ in range(graph_layers)]) \
            if graph_layers > 0 else nn.Identity()
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in range(depth)])
        self.dec = nn.ModuleList([ConvBlock(2 * chs[i], chs[i]) for i in range(depth)])
        self.head = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x):
        skips = []
        for blk in self.enc:
            x = blk(x)
            skips.append(x)
            x = self.pool(x)
        x = self.graph(self.bott(x))
        for i in reversed(range(len(self.enc))):
            x = self.up[i](x)
            x = self.dec[i](torch.cat([x, skips[i]], dim=1))
        return self.head(x)                                   # logits, shape (B, 1, H, W)


def build_model(name: str, base: int = 32) -> nn.Module:
    if name == "unet":
        return UNet(base=base, graph_layers=0)
    if name == "graph_unet_noedge":    # control: same parameters, no message passing
        return UNet(base=base, graph_layers=2, knn=0, use_grid=False)
    if name == "graph_unet_grid":      # ablation: spatial grid edges only
        return UNet(base=base, graph_layers=2, knn=0)
    if name == "graph_unet":
        return UNet(base=base, graph_layers=2, knn=8)
    raise ValueError(name)
