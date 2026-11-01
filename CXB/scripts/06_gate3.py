"""Gate 3 / fixes comparison: which fixes keep explanations on the disease (ComBat versions, CF-Site)?

Same data as gate 2 (disease x2 plus scanner-B decoy), with the 90% shortcut, in the
train-only and the train-and-test designs. Methods (scanner A/B known for every subject):
  plain          train on the data as is
  combat         ComBat harmonisation, no covariates (fitted on training subjects)
  combat_protect ComBat with the label protected (labels for train/val; average for test)
  siteswap_aug   CF-Site training (counterfactual site training): original + ComBat
                 counterfactual copy at the other scanner (same label)
  siteswap       siteswap_aug + saliency-consistency penalty (lam = 1)
  siteswap_oracle siteswap_aug with the true other-scanner version of each brain (upper
                 bound: what CF-Site achieves with a perfect swap; not usable in practice)
Scores as in gate 2 (test site, group attributions): disease AUPRC, scanner share, accuracy.
Pass rule: the decision log (fixed before running).

    .venv\\Scripts\\python.exe -u scripts\\06_gate3.py [--splits 3]
"""
import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from cxb import combat
from cxb.data import load_roi_timeseries
from cxb.decoy import assign_scanner, build_decoy, top_share
from cxb.explain import EXPLAINERS, auprc, edge_scores
from cxb.models import MODELS, train_model
from cxb.planting import build_regions
from cxb.siteswap import train_siteswap
from cxb.splits import BRAINOOD_FOLDS, site_fold

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["plain", "combat", "combat_protect", "siteswap_aug", "siteswap", "siteswap_oracle"]
# also available: combat_eb, combat_protect_eb (standard empirical-Bayes ComBat) and siteswap_eb (swap from it)
DESIGNS = [("train", 0.9), ("both", 0.9)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--first-split", type=int, default=0)
    ap.add_argument("--main", default="siteswap", help="CF-Site variant judged by the pass rule")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--designs", default="train,both")
    ap.add_argument("--decoy", default="timeseries", choices=["timeseries", "edge"])
    ap.add_argument("--region-set", default="A", choices=["A", "B"],
                    help="A: main regions; B: next 10 disease and next 10 decoy regions (robustness)")
    ap.add_argument("--level", default="x2")
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--models", default="mlp,gcn")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--out", default="results/gate3")
    ap.add_argument("--suffix", default="", help="added to the output file name (keeps runs with other methods apart)")
    ap.add_argument("--atlas", default=None, help="atlas name (default: main_atlas in configs/data.yaml), e.g. rois_cc200")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    atlas = args.atlas or cfg["abide"]["main_atlas"]
    raw = ROOT / cfg["paths"]["raw"] / cfg["abide"]["pipeline"] / cfg["abide"]["strategy"] / atlas
    d = np.load(ROOT / "data" / "processed" / f"abide_{atlas}.npz", allow_pickle=True)
    fc_all, dx, site, dead, fid = d["fc"], d["dx"], d["site"], d["n_dead_rois"], d["file_id"]
    R = fc_all.shape[1]

    @lru_cache(maxsize=None)
    def load_ts(i):
        return load_roi_timeseries(raw / f"{fid[i]}_{atlas}.1D")

    def T(a):
        return torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    methods = args.methods.split(",")
    print(f"gate 3 | {atlas} | disease m={args.m} {args.level} | splits {args.splits} | lam {args.lam} | {dev}", flush=True)
    for split in range(args.first_split, args.first_split + args.splits):
        skip, t_dis, t_dec, dec_a = 0, None, None, []
        if args.region_set == "B":          # same strengths as set A, only the location changes
            pa = build_regions(fc_all, dx, site, dead, load_ts, m=args.m, level=args.level, split_seed=split, keep_ts=True)
            t_dis = pa.meta["target_shift"]
            da = build_decoy(pa, load_ts, fc_all, dx, site, m=args.m, seed=split, mode="edge")
            t_dec, dec_a = da.meta["target_shift"], da.regions.tolist()
            skip = args.m
            del pa
        p = build_regions(fc_all, dx, site, dead, load_ts, m=args.m, level=args.level, split_seed=split, keep_ts=True,
                          skip=skip, target=t_dis)
        dec = build_decoy(p, load_ts, fc_all, dx, site, m=args.m, seed=split, mode=args.decoy,
                          exclude=p.meta["first_regions"], skip=skip, target=t_dec)
        sets = [set(p.regions), set(dec.regions), set(p.meta["first_regions"]), set(dec_a)]
        assert len(set.union(*sets)) == sum(map(len, sets)), "region sets overlap"
        print(f"split {split}: region set {args.region_set} | disease {p.regions.tolist()} (target {p.meta['target_shift']:.3f}) | "
              f"decoy {dec.regions.tolist()} (achieved {dec.meta['achieved_shift']:.3f})", flush=True)
        k = int(p.inside.sum() // 2)
        y_t = torch.tensor(p.y, dtype=torch.long, device=dev)
        for design, frac in [d for d in DESIGNS if d[0] in args.designs.split(",")]:
            t0 = time.time()
            for fold in range(len(BRAINOOD_FOLDS)):
                tr, va, te = site_fold(p.site, fold)
                rng = np.random.default_rng([split, fold, int(frac * 100), design == "both"])
                b = np.zeros(len(p.y), dtype=bool)
                for part, f in ((tr, frac), (va, frac), (te, frac if design == "both" else 0.5)):
                    b[part] = assign_scanner(p.y[part], p.site[part], f, rng)
                x = np.where(b[:, None, None], dec.fc_b, p.fc)
                bi, y = b.astype(int), p.y
                eb = any(mm.endswith("_eb") for mm in methods)
                fits = {"combat": combat.fit(x[tr], bi[tr]),
                        "combat_protect": combat.fit(x[tr], bi[tr], covariates=y[tr, None])}
                if eb:
                    fits["combat_eb"] = combat.fit(x[tr], bi[tr], eb=True)
                    fits["combat_protect_eb"] = combat.fit(x[tr], bi[tr], covariates=y[tr, None], eb=True)
                for method in methods:
                    if method in ("combat", "combat_eb"):
                        f = fits[method]
                        xin = {"tr": combat.harmonise(f, x[tr], bi[tr]), "va": combat.harmonise(f, x[va], bi[va]),
                               "te": combat.harmonise(f, x[te], bi[te])}
                    elif method in ("combat_protect", "combat_protect_eb"):
                        f = fits[method]
                        xin = {"tr": combat.harmonise(f, x[tr], bi[tr], y[tr, None]),
                               "va": combat.harmonise(f, x[va], bi[va], y[va, None]),
                               "te": combat.harmonise(f, x[te], bi[te])}
                    else:
                        xin = {"tr": x[tr], "va": x[va], "te": x[te]}
                    xtr, xva, xte = T(xin["tr"]), T(xin["va"]), T(xin["te"])
                    for name in args.models.split(","):
                        torch.manual_seed(args.seed)
                        model = MODELS[name](R, xin["tr"].mean(0), xin["tr"].std(0)).to(dev)
                        if method.startswith("siteswap"):
                            if method == "siteswap_oracle":
                                xs = T(np.where(b[tr][:, None, None], p.fc[tr], dec.fc_b[tr]))
                            else:
                                fk = "combat_protect_eb" if method == "siteswap_eb" else "combat_protect"
                                xs = T(combat.swap(fits[fk], x[tr], bi[tr], 1 - bi[tr], y[tr, None]))
                            lam = args.lam if method == "siteswap" else 0.0
                            model, best_epoch, _ = train_siteswap(model, xtr, xs, y_t[tr], xva, y_t[va], lam=lam,
                                                                  epochs=args.epochs, seed=args.seed)
                        else:
                            model, best_epoch, _ = train_model(model, xtr, y_t[tr], xva, y_t[va],
                                                               epochs=args.epochs, seed=args.seed)
                        with torch.no_grad():
                            acc = float((model(xte).argmax(1) == y_t[te]).float().mean())
                        for ex_name, explain in EXPLAINERS.items():
                            s = edge_scores(explain(model, xte)).detach().cpu().numpy().mean(0)
                            rows.append({"split": split, "seed": args.seed, "decoy": args.decoy, "region_set": args.region_set, "fold": fold, "design": design, "frac": frac,
                                         "method": method, "model": name, "explainer": ex_name,
                                         "best_epoch": best_epoch, "n_test": len(te), "test_acc": acc,
                                         "disease_auprc": auprc(s, p.inside, p.incident),
                                         "scanner_share": top_share(s, dec.inside, k),
                                         "disease_share": top_share(s, p.inside, k)})
            sub = pd.DataFrame([r for r in rows if r["split"] == split and r["design"] == design and r["explainer"] == "saliency"])
            print(f"split {split} {design} {frac} ({time.time() - t0:.0f}s)", flush=True)
            for (mth, mdl), g in sub.groupby(["method", "model"], sort=False):
                print(f"  {mth:15s} {mdl}: acc {g.test_acc.mean():.2f} disease AUPRC {g.disease_auprc.mean():.2f} "
                      f"scanner share {g.scanner_share.mean():.2f}", flush=True)

    df = pd.DataFrame(rows)
    tag = f"{args.level}_m{args.m}_seed{args.seed}_lam{args.lam:g}_splits{args.first_split}-{args.first_split + args.splits - 1}" + ("_edge" if args.decoy == "edge" else "") + ("_setB" if args.region_set == "B" else "") + args.suffix
    df.to_csv(out_dir / f"gate3_{tag}.csv", index=False)
    summ = (df.groupby(["design", "model", "explainer", "method"], sort=False)
              [["test_acc", "disease_auprc", "scanner_share", "disease_share"]].mean().round(3))
    print("\n", summ.to_string())
    sal = df[(df.explainer == "saliency") & (df.design == "train")].groupby(["model", "method"])[
        ["test_acc", "disease_auprc", "scanner_share"]].mean()
    checks = {}
    for m in df.model.unique():
        if args.main not in sal.loc[m].index:
            continue
        cb = sal.loc[m].loc[[c for c in sal.loc[m].index if c.startswith("combat")]]
        if cb.empty:
            continue
        best_cb = cb.disease_auprc.idxmax()
        ss, ref = sal.loc[(m, args.main)], cb.loc[best_cb]
        gain = float(ss.disease_auprc - ref.disease_auprc)
        ok = gain >= 0.05 and ss.scanner_share <= ref.scanner_share + 0.02 and ss.test_acc >= ref.test_acc - 0.03
        checks[m] = {"best_combat": best_cb, "disease_auprc_gain": round(gain, 3),
                     "scanner_share_diff": round(float(ss.scanner_share - ref.scanner_share), 3),
                     "acc_diff": round(float(ss.test_acc - ref.test_acc), 3),
                     "verdict": "pass" if ok else ("non-inferior" if gain >= -0.02 else "fail")}
    verdict = "PASSED" if any(c["verdict"] == "pass" for c in checks.values()) else (
        "NON-INFERIOR" if any(c["verdict"] == "non-inferior" for c in checks.values()) else "FAILED")
    (out_dir / f"gate3_{tag}.json").write_text(json.dumps({"checks": checks, "verdict": verdict}, indent=1))
    print("\n" + " | ".join(f"{m}: vs {c['best_combat']} disease AUPRC {c['disease_auprc_gain']:+.3f}, scanner share "
                            f"{c['scanner_share_diff']:+.3f}, acc {c['acc_diff']:+.3f} -> {c['verdict']}"
                            for m, c in checks.items()))
    print(f"gate 3 {verdict} (judged: {args.main})")


if __name__ == "__main__":
    main()
