"""Why does label-protected ComBat give the highest accuracy when the shortcut is also
in the test data, while finding the disease poorly?

Hypothesis: with scanner and label 90% linked, the protected fit attributes part of the
scanner effect to the label, so after harmonisation the shortcut survives, mainly on the
~960 edges from the decoy regions to the rest of the brain (not counted by the 45-edge
scanner share). Checks, per fold (split 0, MLP and GCN, saliency):
  - test accuracy in the train-only and the both design (simplified and standard EB ComBat);
  - scanner share on decoy edges only (45) and on decoy + incident edges ("wide"; chance
    about 0.15);
  - label probe (logistic regression, 5-fold CV) on decoy-incident edges of the
    harmonised training data;
  - remaining scanner gap: mean difference between scanner B and A on decoy-incident edges
    after harmonisation, within each label.

    .venv\\Scripts\\python.exe -u scripts\\09_combat_check.py
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from cxb import combat
from cxb.data import load_roi_timeseries
from cxb.decoy import assign_scanner, build_decoy, top_share
from cxb.explain import auprc, edge_scores, saliency
from cxb.models import MODELS, train_model
from cxb.planting import build_regions
from cxb.splits import BRAINOOD_FOLDS, site_fold

ROOT = Path(__file__).resolve().parents[1]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/combat_check")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    atlas = cfg["abide"]["main_atlas"]
    raw = ROOT / cfg["paths"]["raw"] / cfg["abide"]["pipeline"] / cfg["abide"]["strategy"] / atlas
    d = np.load(ROOT / "data" / "processed" / f"abide_{atlas}.npz", allow_pickle=True)

    @lru_cache(maxsize=None)
    def load_ts(i):
        return load_roi_timeseries(raw / f"{d['file_id'][i]}_{atlas}.1D")

    p = build_regions(d["fc"], d["dx"], d["site"], d["n_dead_rois"], load_ts, m=10, level="x2", split_seed=0, keep_ts=True)
    dec = build_decoy(p, load_ts, d["fc"], d["dx"], d["site"], m=10, seed=0)
    R, k = p.fc.shape[1], 45
    iu = np.triu_indices(R, 1)
    inc_only = (dec.incident & ~p.inside & ~p.incident)[iu]
    wide = dec.inside | dec.incident
    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)  # noqa: E731
    yt = torch.tensor(p.y, device=dev)
    rows = []
    for design in ("train", "both"):
        for fold in range(len(BRAINOOD_FOLDS)):
            tr, va, te = site_fold(p.site, fold)
            rng = np.random.default_rng([0, fold, 90, design == "both"])
            b = np.zeros(len(p.y), bool)
            for part, f in ((tr, .9), (va, .9), (te, .9 if design == "both" else .5)):
                b[part] = assign_scanner(p.y[part], p.site[part], f, rng)
            x, bi, y = np.where(b[:, None, None], dec.fc_b, p.fc), b.astype(int), p.y
            versions = {"plain": (x[tr], x[va], x[te])}
            for name_, eb in (("combat_protect", False), ("combat_protect_eb", True)):   # EB = standard ComBat (added 1 Nov)
                f = combat.fit(x[tr], bi[tr], covariates=y[tr, None], eb=eb)
                versions[name_] = (combat.harmonise(f, x[tr], bi[tr], y[tr, None]),
                                   combat.harmonise(f, x[va], bi[va], y[va, None]),
                                   combat.harmonise(f, x[te], bi[te]))
            for method, (xtr, xva, xte) in versions.items():
                flat = xtr[:, iu[0], iu[1]][:, inc_only]
                probe = cross_val_score(LogisticRegression(C=0.1, max_iter=2000), flat, y[tr], cv=5, scoring="roc_auc").mean()
                gap = np.mean([flat[(bi[tr] == 1) & (y[tr] == lab)].mean(0) - flat[(bi[tr] == 0) & (y[tr] == lab)].mean(0)
                               for lab in (0, 1)], axis=0).mean()
                for name in ("mlp", "gcn"):
                    torch.manual_seed(0)
                    model = MODELS[name](R, xtr.mean(0), xtr.std(0)).to(dev)
                    model, _, _ = train_model(model, T(xtr), yt[tr], T(xva), yt[va], epochs=150)
                    with torch.no_grad():
                        acc = float((model(T(xte)).argmax(1) == yt[te]).float().mean())
                    s = edge_scores(saliency(model, T(xte))).detach().cpu().numpy().mean(0)
                    rows.append({"design": design, "fold": fold, "method": method, "model": name, "n_test": len(te), "test_acc": acc,
                                 "scanner_share": top_share(s, dec.inside, k), "scanner_share_wide": top_share(s, wide, k),
                                 "probe_auc_incident": probe, "scanner_gap_incident": gap,
                                 "disease_auprc": auprc(s, p.inside, p.incident)})
            print(f"{design} fold {fold} done", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "combat_check.csv", index=False)
    print(df.groupby(["design", "model", "method"]).mean(numeric_only=True).drop(columns="fold").round(3).to_string())


if __name__ == "__main__":
    main()
