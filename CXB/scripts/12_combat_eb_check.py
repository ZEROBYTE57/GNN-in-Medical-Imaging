"""Does standard (empirical-Bayes) ComBat change the ComBat results?

Our ComBat leaves out empirical-Bayes shrinkage. Same setting as the main methods
comparison (region set A, x2, 90% train-only shortcut, person-specific decoy, saliency),
split 0, seed 0, 10 folds. For ComBat and label-protected ComBat, with and without EB:
  - how much EB changes the harmonised training data (mean |difference|, relative to the
    mean |correction| ComBat makes);
  - disease AUPRC, scanner share and test accuracy of MLP and GCN trained on each.

    .venv\\Scripts\\python.exe -u scripts\\12_combat_eb_check.py
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from cxb import combat
from cxb.data import load_roi_timeseries
from cxb.decoy import assign_scanner, build_decoy, top_share
from cxb.explain import auprc, edge_scores, saliency
from cxb.models import MODELS, train_model
from cxb.planting import build_regions
from cxb.splits import BRAINOOD_FOLDS, site_fold

ROOT = Path(__file__).resolve().parents[1]


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    atlas = cfg["abide"]["main_atlas"]
    raw = ROOT / cfg["paths"]["raw"] / cfg["abide"]["pipeline"] / cfg["abide"]["strategy"] / atlas
    d = np.load(ROOT / "data" / "processed" / f"abide_{atlas}.npz", allow_pickle=True)

    @lru_cache(maxsize=None)
    def load_ts(i):
        return load_roi_timeseries(raw / f"{d['file_id'][i]}_{atlas}.1D")

    split, seed, frac = 0, 0, 0.9
    p = build_regions(d["fc"], d["dx"], d["site"], d["n_dead_rois"], load_ts, m=10, level="x2", split_seed=split, keep_ts=True)
    dec = build_decoy(p, load_ts, d["fc"], d["dx"], d["site"], m=10, seed=split)
    R, k = p.fc.shape[1], int(p.inside.sum() // 2)
    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)  # noqa: E731
    yt = torch.tensor(p.y, device=dev)
    rows = []
    for fold in range(len(BRAINOOD_FOLDS)):
        tr, va, te = site_fold(p.site, fold)
        # same scanner assignment as the main comparison (scripts/06_gate3.py)
        rng = np.random.default_rng([split, fold, int(frac * 100), False])
        b = np.zeros(len(p.y), bool)
        for part, f in ((tr, frac), (va, frac), (te, 0.5)):
            b[part] = assign_scanner(p.y[part], p.site[part], f, rng)
        x, bi, y = np.where(b[:, None, None], dec.fc_b, p.fc), b.astype(int), p.y
        for protect in (False, True):
            cov = (lambda idx: y[idx, None]) if protect else (lambda idx: None)
            harm = {}
            for eb in (False, True):
                f = combat.fit(x[tr], bi[tr], covariates=cov(tr), eb=eb)
                harm[eb] = (combat.harmonise(f, x[tr], bi[tr], cov(tr)), combat.harmonise(f, x[va], bi[va], cov(va)),
                            combat.harmonise(f, x[te], bi[te]))
            change = np.abs(harm[True][0] - harm[False][0]).mean() / np.abs(harm[False][0] - x[tr]).mean()
            for eb, (xtr, xva, xte) in harm.items():
                method = ("combat_protect" if protect else "combat") + ("_eb" if eb else "")
                for name in ("mlp", "gcn"):
                    torch.manual_seed(seed)
                    model = MODELS[name](R, xtr.mean(0), xtr.std(0)).to(dev)
                    model, be, _ = train_model(model, T(xtr), yt[tr], T(xva), yt[va], epochs=150, seed=seed)
                    with torch.no_grad():
                        acc = float((model(T(xte)).argmax(1) == yt[te]).float().mean())
                    s = edge_scores(saliency(model, T(xte))).detach().cpu().numpy().mean(0)
                    rows.append({"fold": fold, "method": method, "model": name, "n_test": len(te), "test_acc": acc,
                                 "disease_auprc": auprc(s, p.inside, p.incident), "scanner_share": top_share(s, dec.inside, k),
                                 "eb_relative_change": change})
        print(f"fold {fold} done", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "final" / "combat_eb"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "combat_eb.csv", index=False)
    print(df.groupby(["model", "method"])[["disease_auprc", "scanner_share", "test_acc", "eb_relative_change"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
