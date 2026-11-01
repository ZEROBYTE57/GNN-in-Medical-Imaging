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


def test_mask_explainer_finds_planted_edges():
    from cxb.explain import mask_explainer
    fc, y, inside, incident = _toy()
    m, s = fc[:150].mean(0).numpy(), fc[:150].std(0).numpy()
    for cls in (MLP, GCN):
        torch.manual_seed(0)
        model, _, _ = train_model(cls(12, m, s, dropout=0.1), fc[:150], y[:150], fc[150:], y[150:], epochs=40)
        mask = mask_explainer(model, fc[150:])
        assert mask.shape == fc[150:].shape and float(mask.min()) >= 0 and float(mask.max()) <= 1
        score = edge_scores(mask).mean(0).numpy()
        assert auprc(score, inside, incident) > 0.5, cls.__name__


def test_warmup_rule_never_picks_an_early_epoch():
    import numpy as np
    import torch
    from cxb.models import MLP, train_model
    rng = np.random.default_rng(0)
    fc = rng.normal(0, 0.3, (80, 5, 5)).astype(np.float32)
    y = torch.tensor(np.arange(80) % 2)
    t = torch.tensor(fc)
    _, best, curve = train_model(MLP(5, fc.mean(0), fc.std(0)), t[:60], y[:60], t[60:], y[60:], epochs=40, min_epochs=30)
    assert best >= 30 and len(curve) == 40
    _, best, _ = train_model(MLP(5, fc.mean(0), fc.std(0)), t[:60], y[:60], t[60:], y[60:], epochs=10, min_epochs=30)
    assert best == 10          # fewer epochs than the warm-up: falls back to the last epoch


def test_published_models_give_two_logits_and_edge_gradients():
    from cxb.models import MODELS
    fc, *_ = _toy(n=6)
    m, s = fc.mean(0).numpy(), fc.std(0).numpy()
    for name in ("brainnetcnn", "braingnn", "bnt", "gat"):
        model = MODELS[name](12, m, s).eval()
        x = fc.clone().requires_grad_(True)
        out = model(x)
        assert out.shape == (6, 2), name
        g = torch.autograd.grad(out[:, 1].sum(), x)[0]
        assert torch.isfinite(g).all() and (g != 0).any(), name


def test_published_models_learn_a_toy_planted_effect():
    from cxb.models import MODELS
    fc, y, inside, incident = _toy(n=300)
    m, s = fc[:200].mean(0).numpy(), fc[:200].std(0).numpy()
    for name in ("brainnetcnn", "braingnn", "gat"):
        torch.manual_seed(0)
        model, _, _ = train_model(MODELS[name](12, m, s), fc[:200], y[:200], fc[200:], y[200:], epochs=60)
        assert (model(fc[200:]).argmax(1) == y[200:]).float().mean().item() > 0.9, name
        score = edge_scores(saliency(model, fc[200:])).mean(0).detach().numpy()
        assert auprc(score, inside, incident) > 0.9, name


def test_model_variants_set_width_and_learning_rate():
    from cxb.models import MODELS
    m, s = np.zeros((12, 12)), np.ones((12, 12))
    v = MODELS["bnt-half-lr0.0003"](12, m, s)
    assert v.train_lr == 0.0003 and v.head[0].out_features == 128
    assert not hasattr(MODELS["bnt"](12, m, s), "train_lr")
