"""Benchmark table: how well do explainers recover the planted disease effect?

Same setup as gate 1 (scripts/04_gate1.py), over all strengths, seeds and explainers:
levels x1, x2, x4; 10 planted regions; BrainOOD's 10 site folds; fake-label splits 0-2;
seeds 0-2; MLP and GCN; saliency, Integrated Gradients and a GNNExplainer-style mask.
Scores on the test site: group AUPRC (primary), subject AUPRC, test accuracy and AUC,
and the univariate t-test reference (|t| on the training subjects; not an upper bound).

    .venv\\Scripts\\python.exe -u scripts\\07_benchmark.py [--levels x1,x2,x4] [--splits 3] [--seeds 0,1,2]
"""
import argparse
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score

from cxb.data import load_roi_timeseries
from cxb.explain import ALL_EXPLAINERS, auprc, edge_scores, t_scores
from cxb.models import MODELS, train_model
from cxb.planting import build_regions
from cxb.splits import BRAINOOD_FOLDS, site_fold

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="x1,x2,x4")
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--models", default="mlp,gcn")
    ap.add_argument("--explainers", default="saliency,ig,gnnexplainer")
    ap.add_argument("--out", default="results/benchmark")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]
    explainers = {k: ALL_EXPLAINERS[k] for k in args.explainers.split(",")}

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
    print(f"benchmark | {atlas} | m={args.m} | levels {args.levels} | splits {args.splits} | seeds {seeds} | {dev}", flush=True)
    for level in args.levels.split(","):
        for split in range(args.splits):
            t0 = time.time()
            p = build_regions(fc_all, dx, site, dead, load_ts, m=args.m, level=level, split_seed=split)
            fc_t = torch.tensor(p.fc, dtype=torch.float32, device=dev)
            y_t = torch.tensor(p.y, dtype=torch.long, device=dev)
            for seed in seeds:
                for fold in range(len(BRAINOOD_FOLDS)):
                    tr, va, te = site_fold(p.site, fold)
                    ceiling = auprc(t_scores(p.fc[tr], p.y[tr]), p.inside, p.incident)
                    for name in args.models.split(","):
                        torch.manual_seed(seed)
                        model = MODELS[name](R, p.fc[tr].mean(0), p.fc[tr].std(0)).to(dev)
                        model, best_epoch, _ = train_model(model, fc_t[tr], y_t[tr], fc_t[va], y_t[va],
                                                           epochs=args.epochs, seed=seed)
                        with torch.no_grad():
                            prob = torch.softmax(model(fc_t[te]), 1)[:, 1].cpu().numpy()
                        acc = float(((prob > 0.5) == p.y[te]).mean())
                        auc = float(roc_auc_score(p.y[te], prob)) if len(set(p.y[te])) == 2 else float("nan")
                        for ex_name, explain in explainers.items():
                            s = edge_scores(explain(model, fc_t[te])).detach().cpu().numpy()
                            rows.append({"level": level, "split": split, "seed": seed, "fold": fold,
                                         "model": name, "explainer": ex_name, "best_epoch": best_epoch,
                                         "n_test": len(te), "test_acc": acc, "test_auc": auc,
                                         "group_auprc": auprc(s.mean(0), p.inside, p.incident),
                                         "subject_auprc": float(np.mean([auprc(r, p.inside, p.incident) for r in s])),
                                         "ceiling_auprc": ceiling})
            sub = pd.DataFrame([r for r in rows if r["level"] == level and r["split"] == split])
            print(f"{level} split {split} ({time.time() - t0:.0f}s) | " + " | ".join(
                f"{m}+{e} {g.group_auprc.mean():.2f}" for (m, e), g in sub.groupby(["model", "explainer"]))
                + f" | t-test ref {sub.ceiling_auprc.mean():.2f}", flush=True)
            pd.DataFrame(rows).to_csv(out_dir / "benchmark.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "benchmark.csv", index=False)
    summ = df.groupby(["level", "model", "explainer"])[["test_acc", "group_auprc", "subject_auprc", "ceiling_auprc"]].agg(["mean", "std"]).round(3)
    summ.to_csv(out_dir / "benchmark_summary.csv")
    print("\n", summ.to_string())


if __name__ == "__main__":
    main()
