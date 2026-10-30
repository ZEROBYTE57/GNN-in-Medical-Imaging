import numpy as np
import torch

from cxb.explain import auprc, edge_scores, integrated_gradients, saliency, t_scores
from cxb.models import GCN, MLP, train_model
from cxb.planting import region_masks


def _toy(n=200, R=12, seed=0):
    """Fake connectivity with a learnable effect inside regions 0-2 for label 1."""
    rng = np.random.default_rng(seed)
    fc = rng.normal(0, 0.3, (n, R, R))
    fc = (fc + fc.transpose(0, 2, 1)) / 2
    y = np.arange(n) % 2
    inside, incident = region_masks(R, [0, 1, 2])
    fc[y == 1] += 0.6 * inside
    idx = np.arange(R)
    fc[:, idx, idx] = 0
    return torch.tensor(fc, dtype=torch.float32), torch.tensor(y), inside, incident


def test_models_output_two_logits():
    fc, *_ = _toy(n=4)
    m, s = fc.mean(0).numpy(), fc.std(0).numpy()
    for cls in (MLP, GCN):
        assert cls(12, m, s).eval()(fc).shape == (4, 2)


def test_trained_models_and_explainers_find_planted_edges():
    fc, y, inside, incident = _toy()
    m, s = fc[:150].mean(0).numpy(), fc[:150].std(0).numpy()
    for cls in (MLP, GCN):
        torch.manual_seed(0)
        model, _, _ = train_model(cls(12, m, s, dropout=0.1), fc[:150], y[:150], fc[150:], y[150:], epochs=40)
        acc = (model(fc[150:]).argmax(1) == y[150:]).float().mean().item()
        assert acc > (0.9 if cls is MLP else 0.75), cls.__name__  # GCN pools 3 of 12 regions: slower
        for explain in (saliency, integrated_gradients):
            score = edge_scores(explain(model, fc[150:])).mean(0).detach().numpy()
            assert auprc(score, inside, incident) > 0.8, (cls.__name__, explain.__name__)


def test_edge_scores_symmetrise_and_auprc_ignores_incident():
    R = 5
    inside, incident = region_masks(R, [0, 1])
    attr = torch.tensor(inside, dtype=torch.float32)[None]
    s = edge_scores(attr)[0].numpy()
    assert auprc(s, inside, incident) == 1.0


def test_t_scores_rank_planted_edges_first():
    fc, y, inside, incident = _toy()
    assert auprc(t_scores(fc.numpy(), y.numpy()), inside, incident) == 1.0
