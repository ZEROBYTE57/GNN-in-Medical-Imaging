"""Gate 2: does a fake scanner shortcut pull explanations away from the disease?

Data: region-level disease planting (10 regions, x2 by default) plus a scanner-B decoy on
10 other regions (cxb.decoy). Per fold, "scanner B" is linked to the fake label:
  - train-only design: in the training and validation sites, a share `frac` of fake
    patients and 1 - frac of fake controls get scanner B; the test site is 50/50;
  - both design: the same link everywhere, including the test site.
frac = 0.5 means no shortcut (the reference condition).

Scores per fold, on the test site (group attributions):
  - disease AUPRC (planted disease edges vs untouched edges; decoy edges count as negatives);
  - scanner share: share of the top-k explanation edges inside the decoy (k = number of
    disease edges); disease share likewise;
  - test accuracy.
Pass rule (docs/decisions.md, fixed before running): at 90% train-only, for at least one
model with saliency, scanner share rises by >= 0.2 over 50%, or disease AUPRC falls by >= 0.2.

    .venv\\Scripts\\python.exe scripts\\05_gate2.py [--splits 3] [--level x2]
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

from cxb.data import load_roi_timeseries
from cxb.decoy import assign_scanner, build_decoy, top_share
from cxb.explain import EXPLAINERS, auprc, edge_scores
from cxb.models import MODELS, train_model
from cxb.planting import build_regions
from cxb.splits import BRAINOOD_FOLDS, site_fold

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = [("train", 0.5), ("train", 0.7), ("train", 0.9), ("both", 0.9)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", default="x2")
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--models", default="mlp,gcn")
    ap.add_argument("--out", default="results/gate2")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    atlas = cfg["abide"]["main_atlas"]
    raw = ROOT / cfg["paths"]["raw"] / cfg["abide"]["pipeline"] / cfg["abide"]["strategy"] / atlas
    d = np.load(ROOT / "data" / "processed" / f"abide_{atlas}.npz", allow_pickle=True)
    fc_all, dx, site, dead, fid = d["fc"], d["dx"], d["site"], d["n_dead_rois"], d["file_id"]
    R = fc_all.shape[1]

    @lru_cache(maxsize=None)
    def load_ts(i):
        return load_roi_timeseries(raw / f"{fid[i]}_{atlas}.1D")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, meta = [], {}
    print(f"gate 2 | {atlas} | disease m={args.m} {args.level} | splits {args.splits} | seed {args.seed} | {dev}")
    for split in range(args.splits):
        t0 = time.time()
        p = build_regions(fc_all, dx, site, dead, load_ts, m=args.m, level=args.level, split_seed=split, keep_ts=True)
        dec = build_decoy(p, load_ts, fc_all, dx, site, m=args.m, seed=split)
        k = int(p.inside.sum() // 2)
        meta[split] = {"disease_regions": p.regions.tolist(), "decoy_regions": dec.regions.tolist(),
                       "disease_target": p.meta["target_shift"], **{f"decoy_{a}": b for a, b in dec.meta.items()}}
        print(f"split {split}: disease regions {p.regions.tolist()} (target {p.meta['target_shift']:.3f}) | "
              f"decoy regions {dec.regions.tolist()} (target {dec.meta['target_shift']:.3f}, achieved "
              f"{dec.meta['achieved_shift']:.3f}) | {time.time() - t0:.0f}s", flush=True)
        fc_a = torch.tensor(p.fc, dtype=torch.float32, device=dev)
        fc_b = torch.tensor(dec.fc_b, dtype=torch.float32, device=dev)
        y_t = torch.tensor(p.y, dtype=torch.long, device=dev)
        for design, frac in CONDITIONS:
            for fold in range(len(BRAINOOD_FOLDS)):
                tr, va, te = site_fold(p.site, fold)
                rng = np.random.default_rng([split, fold, int(frac * 100), design == "both"])
                b = np.zeros(len(p.y), dtype=bool)
                for part, f in ((tr, frac), (va, frac), (te, frac if design == "both" else 0.5)):
                    b[part] = assign_scanner(p.y[part], p.site[part], f, rng)
                bt = torch.tensor(b, device=dev)[:, None, None]
                fc = torch.where(bt, fc_b, fc_a)
                for name in args.models.split(","):
                    torch.manual_seed(args.seed)
                    model = MODELS[name](R, fc[tr].mean(0).cpu().numpy(), fc[tr].std(0).cpu().numpy()).to(dev)
                    model, best_epoch, _ = train_model(model, fc[tr], y_t[tr], fc[va], y_t[va],
                                                       epochs=args.epochs, seed=args.seed)
                    with torch.no_grad():
                        acc = float((model(fc[te]).argmax(1) == y_t[te]).float().mean())
                    for ex_name, explain in EXPLAINERS.items():
                        s = edge_scores(explain(model, fc[te])).detach().cpu().numpy().mean(0)
                        rows.append({"split": split, "fold": fold, "design": design, "frac": frac,
                                     "model": name, "explainer": ex_name, "best_epoch": best_epoch,
                                     "n_test": len(te), "test_acc": acc,
                                     "disease_auprc": auprc(s, p.inside, p.incident),
                                     "scanner_share": top_share(s, dec.inside, k),
                                     "disease_share": top_share(s, p.inside, k)})
            sub = pd.DataFrame([r for r in rows if r["split"] == split and r["design"] == design
                                and r["frac"] == frac and r["explainer"] == "saliency"])
            print("  " + design + f" {frac:.1f} | " + " | ".join(
                f"{m}: acc {g.test_acc.mean():.2f} disease AUPRC {g.disease_auprc.mean():.2f} "
                f"scanner share {g.scanner_share.mean():.2f}" for m, g in sub.groupby("model")), flush=True)

    df = pd.DataFrame(rows)
    tag = f"{args.level}_m{args.m}_seed{args.seed}"
    df.to_csv(out_dir / f"gate2_{tag}.csv", index=False)
    summ = (df.groupby(["design", "frac", "model", "explainer"])
              [["test_acc", "disease_auprc", "scanner_share", "disease_share"]].mean().round(3))
    print("\n", summ.to_string())
    sal = df[df.explainer == "saliency"].groupby(["design", "frac", "model"])[["scanner_share", "disease_auprc"]].mean()
    checks = {}
    for m in df.model.unique():
        ref, s90 = sal.loc[("train", 0.5, m)], sal.loc[("train", 0.9, m)]
        checks[m] = {"scanner_share_rise": round(float(s90.scanner_share - ref.scanner_share), 3),
                     "disease_auprc_drop": round(float(ref.disease_auprc - s90.disease_auprc), 3)}
        checks[m]["passed"] = bool(checks[m]["scanner_share_rise"] >= 0.2 or checks[m]["disease_auprc_drop"] >= 0.2)
    passed = any(c["passed"] for c in checks.values())
    (out_dir / f"gate2_{tag}.json").write_text(json.dumps({"checks": checks, "passed": passed, "meta": meta}, indent=1, default=str))
    print("\n" + " | ".join(f"{m}: scanner share +{c['scanner_share_rise']}, disease AUPRC -{c['disease_auprc_drop']}"
                            for m, c in checks.items()))
    print(f"gate 2 {'PASSED' if passed else 'FAILED'} (rise >= 0.2 or drop >= 0.2, 90% vs 50%, train-only, saliency)")


if __name__ == "__main__":
    main()
