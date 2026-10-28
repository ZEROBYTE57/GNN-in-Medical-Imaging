"""Full-batch training of one MLF-GNN on one fold.

The Cox partial likelihood is not separable over patients, so there is no
mini-batching: one forward pass over the whole training graph per epoch, exactly
as in the original. At a few hundred nodes this is milliseconds.

Early stopping watches the in-centre validation C-index. It must never watch the
held-out centre -- that would turn the held-out estimate into a tuned one and
make the generalisation gap meaningless.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import numpy as np
import torch

from ..eval.metrics import harrell_cindex
from ..graphs.construct import Graph
from ..models.losses import cox_ph_loss
from ..models.mlfgnn import MLFGNN
from ..device import resolve_device
from ..seed import set_seed

logger = logging.getLogger(__name__)


@dataclass
class TrainedModel:
    model: MLFGNN
    best_epoch: int
    best_val_cindex: float
    history: list[dict]


def _tensors(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=device)


def _graph_tensors(graph: Graph, device: torch.device):
    return (
        torch.as_tensor(graph.edge_index, dtype=torch.long, device=device),
        torch.as_tensor(graph.edge_weight, dtype=torch.float32, device=device),
    )


def train_mlfgnn(
    x_pet: np.ndarray,
    x_ct: np.ndarray,
    clinical: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    graph: Graph,
    train_pos: np.ndarray,
    val_pos: np.ndarray,
    *,
    layer_type: str = "GAT",
    fusion: str = "network",
    hidden_dim: int = 32,
    heads: int = 4,
    dropout: float = 0.5,
    use_clinical: bool = True,
    epochs: int = 100,
    lr: float = 0.005,
    weight_decay: float = 0.005,
    patience: int = 30,
    min_epochs: int = 20,
    seed: int = 0,
    device="auto",
) -> TrainedModel:
    """Train and return the checkpoint with the best in-centre validation C-index."""
    set_seed(seed)
    dev = resolve_device(device) if isinstance(device, str) else device

    clinical = clinical if use_clinical else np.zeros((len(time), 0))
    xp, xc, cl = _tensors(x_pet, dev), _tensors(x_ct, dev), _tensors(clinical, dev)
    tt = _tensors(time, dev)
    ee = _tensors(event, dev)
    edge_index, edge_weight = _graph_tensors(graph, dev)

    train_mask = torch.zeros(len(time), dtype=torch.bool, device=dev)
    train_mask[torch.as_tensor(train_pos, dtype=torch.long, device=dev)] = True

    model = MLFGNN(
        pet_dim=xp.shape[1],
        ct_dim=xc.shape[1],
        clinical_dim=cl.shape[1],
        hidden_dim=hidden_dim,
        layer_type=layer_type,
        fusion=fusion,
        heads=heads,
        dropout=dropout,
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_c, best_epoch, stale = -np.inf, 0, 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        risk = model(xp, xc, edge_index, edge_weight, cl)
        loss = cox_ph_loss(risk, tt, ee, mask=train_mask)
        if not torch.isfinite(loss):
            logger.warning("non-finite loss at epoch %d -- stopping early", epoch)
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            scores = model(xp, xc, edge_index, edge_weight, cl).cpu().numpy()
        val_c = harrell_cindex(scores[val_pos], time[val_pos], event[val_pos])
        history.append({"epoch": epoch, "loss": float(loss.item()), "val_cindex": val_c})

        if np.isfinite(val_c) and val_c > best_c:
            best_c, best_epoch, stale = val_c, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if epoch >= min_epochs and stale >= patience:
                logger.debug("early stop at epoch %d (best %d)", epoch, best_epoch)
                break

    model.load_state_dict(best_state)
    return TrainedModel(model, best_epoch, float(best_c), history)


@torch.no_grad()
def score_graph(
    model: MLFGNN,
    x_pet: np.ndarray,
    x_ct: np.ndarray,
    clinical: np.ndarray,
    graph: Graph,
    use_clinical: bool = True,
    device="auto",
) -> np.ndarray:
    """Risk score for every node of ``graph``.

    Called twice per fold: once on the training graph (for the in-centre split)
    and once on the evaluation graph, which is the training graph plus the
    directed attachment edges carrying messages into the held-out patients.
    """
    dev = resolve_device(device) if isinstance(device, str) else device
    model.eval()
    clinical = clinical if use_clinical else np.zeros((x_pet.shape[0], 0))
    edge_index, edge_weight = _graph_tensors(graph, dev)
    out = model(
        _tensors(x_pet, dev), _tensors(x_ct, dev), edge_index, edge_weight,
        _tensors(clinical, dev),
    )
    return out.cpu().numpy()
