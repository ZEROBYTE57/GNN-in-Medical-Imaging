"""Gate 1: can a simple model plus explainer find a strong planted effect?

For each fake-label split, healthy controls get region-level planting (10 regions, x4 by
default; planting.build_regions). For each of BrainOOD's 10 site folds, an MLP and a GCN
are trained (epoch chosen on the validation site), then explained on the held-out test
site with saliency and Integrated Gradients.

Scores (inside edges = positives, untouched = negatives, incident = don't care):
  - group AUPRC: attributions averaged over the test subjects, then scored (primary;
    comparable to the recovery ceiling);
  - subject AUPRC: scored per test subject, then averaged (stricter);
  - ceiling: |t| between the fake groups of the training subjects (what the model
    learns from); a plain statistical test, no model.
Pass rule (docs/decisions.md): best model + explainer reaches mean group AUPRC >= 0.5.

    .venv\\Scripts\\python.exe scripts\\04_gate1.py [--splits 3] [--level x4] [--m 10]
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
from sklearn.metrics import roc_auc_score

from cxb.data import load_roi_timeseries
from cxb.explain import EXPLAINERS, auprc, edge_scores, t_scores
from cxb.models import MODELS, train_model
from cxb.planting import build_regions
from cxb.splits import BRAINOOD_FOLDS, site_fold

ROOT = Path(__file__).resolve().parents[1]
PASS_AUPRC = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", default="x4")
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--models", default="mlp,gcn")
    ap.add_argument("--out", default="results/gate1")
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
    rows = []
    print(f"gate 1 | {atlas} | m={args.m} {args.level} | splits {args.splits} | seed {args.seed} | {dev}")
    for split in range(args.splits):
        t0 = time.time()
        p = build_regions(fc_all, dx, site, dead, load_ts, m=args.m, level=args.level, split_seed=split)
        print(f"split {split}: pool {p.meta['n_pool']} ({p.meta['n_patients']} fake patients), "
              f"regions {p.regions.tolist()}, planted in {time.time() - t0:.0f}s", flush=True)
        fc_t = torch.tensor(p.fc, dtype=torch.float32, device=dev)
        y_t = torch.tensor(p.y, dtype=torch.long, device=dev)
        for fold in range(len(BRAINOOD_FOLDS)):
            tr, va, te = site_fold(p.site, fold)
            ceiling = auprc(t_scores(p.fc[tr], p.y[tr]), p.inside, p.incident)  # what the model sees
            for name in args.models.split(","):
                torch.manual_seed(args.seed)
                mean, std = p.fc[tr].mean(0), p.fc[tr].std(0)
                model = MODELS[name](R, mean, std).to(dev)
                model, best_epoch, _ = train_model(model, fc_t[tr], y_t[tr], fc_t[va], y_t[va],
                                                   epochs=args.epochs, seed=args.seed)
                with torch.no_grad():
                    prob = torch.softmax(model(fc_t[te]), 1)[:, 1].cpu().numpy()
                acc = float(((prob > 0.5) == p.y[te]).mean())
                auc = float(roc_auc_score(p.y[te], prob)) if len(set(p.y[te])) == 2 else float("nan")
                for ex_name, explain in EXPLAINERS.items():
                    s = edge_scores(explain(model, fc_t[te])).detach().cpu().numpy()
                    rows.append({"split": split, "fold": fold, "test_site": BRAINOOD_FOLDS[fold][0],
                                 "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                                 "model": name, "explainer": ex_name, "best_epoch": best_epoch,
                                 "test_acc": acc, "test_auc": auc,
                                 "group_auprc": auprc(s.mean(0), p.inside, p.incident),
                                 "subject_auprc": float(np.mean([auprc(r, p.inside, p.incident) for r in s])),
                                 "ceiling_auprc": ceiling})
                r = rows[-2:]
                print(f"  split {split} fold {fold} {BRAINOOD_FOLDS[fold][0]:8s} n={len(te):2d} {name}: "
                      f"acc {acc:.2f} auc {auc:.2f} ep {best_epoch:3d} | group AUPRC saliency "
                      f"{r[0]['group_auprc']:.2f} ig {r[1]['group_auprc']:.2f} | ceiling {ceiling:.2f}", flush=True)

    df = pd.DataFrame(rows)
    tag = f"{args.level}_m{args.m}_seed{args.seed}"
    df.to_csv(out_dir / f"gate1_{tag}.csv", index=False)
    iu = R * (R - 1) // 2
    chance = (args.m * (args.m - 1) / 2) / (iu - args.m * (R - args.m))
    summ = (df.groupby(["model", "explainer"])
              [["test_acc", "test_auc", "group_auprc", "subject_auprc", "ceiling_auprc"]]
              .agg(["mean", "std"]).round(3))
    print("\n", summ.to_string())
    best = df.groupby(["model", "explainer"])["group_auprc"].mean()
    verdict = {"level": args.level, "m": args.m, "chance_auprc": round(chance, 4),
               "best": f"{best.idxmax()[0]} + {best.idxmax()[1]}", "best_group_auprc": round(float(best.max()), 3),
               "threshold": PASS_AUPRC, "passed": bool(best.max() >= PASS_AUPRC),
               "means": {f"{k[0]}+{k[1]}": round(float(v), 3) for k, v in best.items()}}
    (out_dir / f"gate1_{tag}.json").write_text(json.dumps(verdict, indent=1))
    print(f"\nchance AUPRC {chance:.4f} | best {verdict['best']}: {verdict['best_group_auprc']} | "
          + (f"gate 1 {'PASSED' if verdict['passed'] else 'FAILED'} (threshold {PASS_AUPRC})"
             if args.level == "x4" else "(gate 1 is judged at x4 only)"))


if __name__ == "__main__":
    main()
