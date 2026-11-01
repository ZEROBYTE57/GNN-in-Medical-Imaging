"""Real-label check: autism vs controls (no planting), real hospitals as sites.

All 848 subjects (CMU left out by the folds), BrainOOD's 10 site folds, seeds 0-2.
Methods:
  plain           train on the data as is
  combat_protect  ComBat over the training hospitals, diagnosis protected; validation
                  and test hospitals harmonised from their own data (no labels)
  siteswap_aug    train on original + a copy of each subject moved (ComBat, diagnosis
                  protected) to a random other training hospital, same label
  combat_protect_eb, siteswap_eb   the same with standard (empirical-Bayes) ComBat
There is no ground truth here, so scores are:
  - test accuracy and AUC;
  - site alignment of the explanation: Spearman correlation between the group saliency
    map and the per-edge site effect (eta squared over the training hospitals), and the
    share of the top-45 explanation edges among the top-45 site-effect edges (chance 0.007).
Lower site alignment at similar accuracy = explanations depend less on the hospital.

    .venv\\Scripts\\python.exe -u scripts\\08_real_labels.py [--seeds 0,1,2]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

from cxb import combat
from cxb.decoy import site_effect_eta2, top_share
from cxb.explain import EXPLAINERS, edge_scores
from cxb.models import MODELS, train_model
from cxb.siteswap import train_siteswap
from cxb.splits import BRAINOOD_FOLDS, site_fold

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["plain", "combat_protect", "siteswap_aug"]


def partial_spearman(a, b, c):
    """Spearman correlation of a and b after removing the (rank) linear effect of c from both."""
    ra, rb, rc = (rankdata(v) for v in (a, b, c))
    res = [r - np.polyval(np.polyfit(rc, r, 1), rc) for r in (ra, rb)]
    return float(np.corrcoef(*res)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--resume", action="store_true", help="keep finished folds in the output CSV and skip them")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--models", default="mlp,gcn")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--out", default="results/real_labels")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]

    d = np.load(ROOT / "data" / "processed" / "abide_rois_aal.npz", allow_pickle=True)
    fc, site = d["fc"].astype(np.float64), d["site"]
    y = (d["dx"] == 1).astype(int)                      # 1 = autism, 0 = control
    R = fc.shape[1]
    iu = np.triu_indices(R, 1)
    yt = torch.tensor(y, dtype=torch.long, device=dev)
    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)  # noqa: E731

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, done = [], set()
    if args.resume and (out_dir / "real_labels.csv").exists():
        rows = pd.read_csv(out_dir / "real_labels.csv").to_dict("records")
        done = {r["fold"] for r in rows}
        print(f"resuming: folds {sorted(done)} already done", flush=True)
    print(f"real labels | AAL | {len(y)} subjects ({y.sum()} autism) | seeds {seeds} | {dev}", flush=True)
    for fold in range(len(BRAINOOD_FOLDS)):
        if fold in done:
            continue
        t0 = time.time()
        tr, va, te = site_fold(site, fold)
        names = sorted(set(site[tr]))
        sid = np.array([names.index(s) for s in site[tr]])
        eta, _, _ = site_effect_eta2(fc, site, tr, min_n=10)
        eta_e = eta[iu]
        str_e = np.abs(fc[tr]).mean(0)[iu]          # connection strength (training subjects)
        k = 45
        top_site = np.zeros_like(eta, dtype=bool)
        top_site[iu[0][np.argsort(-eta_e)[:k]], iu[1][np.argsort(-eta_e)[:k]]] = True
        f0 = combat.fit(fc[tr], sid, covariates=y[tr, None])
        f_eb = combat.fit(fc[tr], sid, covariates=y[tr, None], eb=True) if "_eb" in args.methods else None
        inputs = {}
        for method in args.methods.split(","):
            if method in ("combat_protect", "combat_protect_eb"):
                f = f_eb if method.endswith("_eb") else f0
                fv, v_id = combat.add_site(f, fc[va])
                ft, t_id = combat.add_site(f, fc[te])
                inputs[method] = (combat.harmonise(f, fc[tr], sid, y[tr, None]),
                                  combat.harmonise(fv, fc[va], np.full(len(va), v_id)),
                                  combat.harmonise(ft, fc[te], np.full(len(te), t_id)))
            else:
                inputs[method] = (fc[tr], fc[va], fc[te])
        for seed in seeds:
            rng = np.random.default_rng([fold, seed])
            other = (sid + rng.integers(1, len(names), len(sid))) % len(names)   # a different training hospital
            swapped = T(combat.swap(f0, fc[tr], sid, other, y[tr, None]))
            swapped_eb = T(combat.swap(f_eb, fc[tr], sid, other, y[tr, None])) if f_eb is not None else None
            for method, (xtr, xva, xte) in inputs.items():
                for name in args.models.split(","):
                    torch.manual_seed(seed)
                    model = MODELS[name](R, xtr.mean(0), xtr.std(0)).to(dev)
                    if method in ("siteswap_aug", "siteswap_eb"):
                        sw = swapped_eb if method == "siteswap_eb" else swapped
                        model, be, _ = train_siteswap(model, T(xtr), sw, yt[tr], T(xva), yt[va], lam=0.0,
                                                      epochs=args.epochs, seed=seed)
                    else:
                        model, be, _ = train_model(model, T(xtr), yt[tr], T(xva), yt[va], epochs=args.epochs, seed=seed)
                    with torch.no_grad():
                        prob = torch.softmax(model(T(xte)), 1)[:, 1].cpu().numpy()
                    for ex_name, explain in EXPLAINERS.items():
                        s = edge_scores(explain(model, T(xte))).detach().cpu().numpy().mean(0)
                        rows.append({"fold": fold, "test_site": BRAINOOD_FOLDS[fold][0], "seed": seed,
                                     "method": method, "model": name, "explainer": ex_name, "best_epoch": be,
                                     "n_test": len(te), "test_acc": float(((prob > 0.5) == y[te]).mean()),
                                     "test_auc": float(roc_auc_score(y[te], prob)),
                                     "site_spearman": float(spearmanr(s, eta_e)[0]),
                                     "site_spearman_partial": partial_spearman(s, eta_e, str_e),
                                     "site_top_share": top_share(s, top_site, k)})
        sub = pd.DataFrame([r for r in rows if r["fold"] == fold and r["explainer"] == "saliency"])
        print(f"fold {fold} {BRAINOOD_FOLDS[fold][0]} n={len(te)} ({time.time() - t0:.0f}s) | " + " | ".join(
            f"{m}/{n}: auc {g.test_auc.mean():.2f} site rho {g.site_spearman.mean():.2f}"
            for (m, n), g in sub.groupby(["method", "model"], sort=False)), flush=True)
        pd.DataFrame(rows).to_csv(out_dir / "real_labels.csv", index=False)

    df = pd.DataFrame(rows)
    summ = df.groupby(["model", "explainer", "method"], sort=False)[
        ["test_acc", "test_auc", "site_spearman", "site_top_share"]].agg(["mean", "std"]).round(3)
    summ.to_csv(out_dir / "real_labels_summary.csv")
    print("\n", summ.to_string())


if __name__ == "__main__":
    main()
