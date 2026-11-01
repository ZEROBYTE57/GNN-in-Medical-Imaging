"""Two controls asked for by the 1 Nov review (rules in docs/decisions.md, fixed before running).

1. Shuffled-label control: the x2 benchmark setting, but the fake labels of the training
   and validation subjects are permuted, so there is no learnable effect. Disease AUPRC
   of saliency and IG here = what an explainer reaches without a learned effect.
2. Logistic regression baseline (L2; C chosen on validation loss from {0.01, 0.1, 1}),
   standardised upper-triangle edges; explanation = |coefficient| per edge. Run on x2
   without a shortcut and with the 90% train-only scanner shortcut (main decoy).

Splits 0-2, BrainOOD's 10 site folds; seeds 0-2 for the neural models (logistic
regression has no randomness, so seed 0 only).

    .venv\\Scripts\\python.exe -u scripts\\13_controls.py
"""
import argparse
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from cxb.data import load_roi_timeseries
from cxb.decoy import assign_scanner, build_decoy, top_share
from cxb.explain import ALL_EXPLAINERS, auprc, edge_scores
from cxb.models import MODELS, train_model
from cxb.planting import build_regions
from cxb.splits import BRAINOOD_FOLDS, site_fold

ROOT = Path(__file__).resolve().parents[1]


def logreg(x_tr, y_tr, x_va, y_va, iu):
    """Fit on standardised upper-triangle edges; C by validation log loss. Returns
    (model, mean, sd, |coef| per upper-triangle edge, same order as edge_scores)."""
    a, b = x_tr[:, iu[0], iu[1]], x_va[:, iu[0], iu[1]]
    mu, sd = a.mean(0), np.where(a.std(0) > 1e-6, a.std(0), 1.0)
    best = None
    for c in (0.01, 0.1, 1.0):
        clf = LogisticRegression(C=c, max_iter=2000).fit((a - mu) / sd, y_tr)
        loss = log_loss(y_va, clf.predict_proba((b - mu) / sd)[:, 1], labels=[0, 1])
        if best is None or loss < best[0]:
            best = (loss, clf)
    clf = best[1]
    return clf, mu, sd, np.abs(clf.coef_[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--models", default="mlp,gcn")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--out", default="results/final/controls")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    atlas = cfg["abide"]["main_atlas"]
    raw = ROOT / cfg["paths"]["raw"] / cfg["abide"]["pipeline"] / cfg["abide"]["strategy"] / atlas
    d = np.load(ROOT / "data" / "processed" / f"abide_{atlas}.npz", allow_pickle=True)

    @lru_cache(maxsize=None)
    def load_ts(i):
        return load_roi_timeseries(raw / f"{d['file_id'][i]}_{atlas}.1D")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    explainers = {k: ALL_EXPLAINERS[k] for k in ("saliency", "ig")}
    rows = []
    for split in range(args.splits):
        t0 = time.time()
        p = build_regions(d["fc"], d["dx"], d["site"], d["n_dead_rois"], load_ts, m=10, level="x2",
                          split_seed=split, keep_ts=True)
        dec = build_decoy(p, load_ts, d["fc"], d["dx"], d["site"], m=10, seed=split)
        R, k = p.fc.shape[1], int(p.inside.sum() // 2)
        iu = np.triu_indices(R, 1)
        fc_t = torch.tensor(p.fc, dtype=torch.float32, device=dev)
        for fold in range(len(BRAINOOD_FOLDS)):
            tr, va, te = site_fold(p.site, fold)
            base = {"split": split, "fold": fold, "n_test": len(te)}
            # 1. shuffled labels (train + validation permuted, test labels real)
            for seed in seeds:
                rng = np.random.default_rng([split, fold, seed, 7])
                y = p.y.copy()
                y[tr] = rng.permutation(y[tr])
                y[va] = rng.permutation(y[va])
                y_t = torch.tensor(y, dtype=torch.long, device=dev)
                for name in args.models.split(","):
                    torch.manual_seed(seed)
                    model = MODELS[name](R, p.fc[tr].mean(0), p.fc[tr].std(0)).to(dev)
                    model, be, _ = train_model(model, fc_t[tr], y_t[tr], fc_t[va], y_t[va], epochs=args.epochs, seed=seed)
                    with torch.no_grad():
                        acc = float((model(fc_t[te]).argmax(1).cpu().numpy() == p.y[te]).mean())
                    for ex, f in explainers.items():
                        s = edge_scores(f(model, fc_t[te])).detach().cpu().numpy().mean(0)
                        rows.append({**base, "control": "shuffled_labels", "seed": seed, "model": name,
                                     "explainer": ex, "test_acc": acc, "disease_auprc": auprc(s, p.inside, p.incident),
                                     "scanner_share": float("nan")})
            # 2. logistic regression, no shortcut and 90% train-only shortcut
            for cond in ("no_shortcut", "shortcut_90"):
                if cond == "no_shortcut":
                    x = p.fc
                else:   # same scanner assignment as scripts/06_gate3.py and 12_combat_eb_check.py
                    rng = np.random.default_rng([split, fold, 90, False])
                    b = np.zeros(len(p.y), bool)
                    for part, f in ((tr, 0.9), (va, 0.9), (te, 0.5)):
                        b[part] = assign_scanner(p.y[part], p.site[part], f, rng)
                    x = np.where(b[:, None, None], dec.fc_b, p.fc)
                clf, mu, sd, w = logreg(x[tr], p.y[tr], x[va], p.y[va], iu)
                acc = float((clf.predict((x[te][:, iu[0], iu[1]] - mu) / sd) == p.y[te]).mean())
                rows.append({**base, "control": f"logreg_{cond}", "seed": 0, "model": "logreg", "explainer": "coefficients",
                             "test_acc": acc, "disease_auprc": auprc(w, p.inside, p.incident),
                             "scanner_share": top_share(w, dec.inside, k)})
        sub = pd.DataFrame([r for r in rows if r["split"] == split])
        print(f"split {split} ({time.time() - t0:.0f}s) | " + " | ".join(
            f"{c}/{m}/{e}: AUPRC {g.disease_auprc.mean():.3f} acc {g.test_acc.mean():.2f}"
            for (c, m, e), g in sub.groupby(["control", "model", "explainer"])), flush=True)
        pd.DataFrame(rows).to_csv(out_dir / "controls.csv", index=False)
    print("chance AUPRC (share of planted edges among scored edges):",
          round(float(p.inside[iu].sum() / (~p.incident[iu]).sum()), 4))


if __name__ == "__main__":
    main()
