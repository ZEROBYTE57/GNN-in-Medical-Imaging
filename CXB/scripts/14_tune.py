"""Fixed tuning budget for models that fail the gate-1 sanity test (rule in the decision log,
1 Nov 2026): learning rate {1e-3, 3e-4} x width {default, half}, chosen by mean validation
loss only (x4 planted effect, no shortcut, split 0, seed 0, BrainOOD's 10 site folds).
Test data is not looked at. Writes results/final/models/tune_<model>.json with the choice.

    .venv\\Scripts\\python.exe -u scripts\\14_tune.py --models bnt,braingnn
"""
import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import yaml

from cxb.data import load_roi_timeseries
from cxb.models import MODELS, train_model
from cxb.planting import build_regions
from cxb.splits import BRAINOOD_FOLDS, site_fold

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    atlas = cfg["abide"]["main_atlas"]
    raw = ROOT / cfg["paths"]["raw"] / cfg["abide"]["pipeline"] / cfg["abide"]["strategy"] / atlas
    d = np.load(ROOT / "data" / "processed" / f"abide_{atlas}.npz", allow_pickle=True)

    @lru_cache(maxsize=None)
    def load_ts(i):
        return load_roi_timeseries(raw / f"{d['file_id'][i]}_{atlas}.1D")

    p = build_regions(d["fc"], d["dx"], d["site"], d["n_dead_rois"], load_ts, m=10, level="x4", split_seed=0)
    R = p.fc.shape[1]
    fc_t = torch.tensor(p.fc, dtype=torch.float32, device=dev)
    y_t = torch.tensor(p.y, dtype=torch.long, device=dev)
    out = ROOT / "results" / "final" / "models"
    out.mkdir(parents=True, exist_ok=True)
    for base in args.models.split(","):
        res = {}
        for variant in (f"{base}-lr0.001", f"{base}-lr0.0003", f"{base}-half-lr0.001", f"{base}-half-lr0.0003"):
            losses = []
            for fold in range(len(BRAINOOD_FOLDS)):
                tr, va, _ = site_fold(p.site, fold)
                torch.manual_seed(0)
                model = MODELS[variant](R, p.fc[tr].mean(0), p.fc[tr].std(0)).to(dev)
                _, be, curve = train_model(model, fc_t[tr], y_t[tr], fc_t[va], y_t[va], epochs=args.epochs, seed=0)
                losses.append(curve[be - 1])
            res[variant] = float(np.mean(losses))
            print(f"{variant}: mean validation loss {res[variant]:.4f}", flush=True)
        best = min(res, key=res.get)
        (out / f"tune_{base}.json").write_text(json.dumps({"losses": res, "chosen": best}, indent=1))
        print(f"{base}: chosen {best}", flush=True)


if __name__ == "__main__":
    main()
