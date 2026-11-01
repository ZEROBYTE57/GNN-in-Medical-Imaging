"""Can the scanner-effect map be estimated from training data alone? (CPU only; the decision log, 1 Nov)

The site audit (a practical version of the scanner share) replaces the known decoy edges
by the edges where the training data differ most between scanners. This check measures,
in the main setting (x2, person-specific decoy, splits 0-2, BrainOOD's 10 folds), how well
the estimated top-45 map recovers the true 45 decoy edges, at the 50% and 90% shortcut:
  naive     |t| of scanner B vs A over training subjects;
  adjusted  |t| of the scanner coefficient in edge ~ scanner + label (training subjects),
            so that the label-linked shortcut does not pull disease edges into the map.
Reported: precision of the top 45 (share that are true decoy edges), the share that touch a
decoy region (edges from a decoy region to another region; also scanner-affected) and the share that are
planted disease edges (contamination). No model is trained.

    .venv\\Scripts\\python.exe -u scripts\\16_audit_check.py
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cxb.data import load_roi_timeseries
from cxb.decoy import assign_scanner, build_decoy
from cxb.planting import build_regions
from cxb.splits import BRAINOOD_FOLDS, site_fold

ROOT = Path(__file__).resolve().parents[1]


def scanner_t(x, b, y=None):
    """|t| per upper-triangle edge of the scanner coefficient (OLS; with the label as a covariate if y given)."""
    n = len(b)
    X = np.column_stack([np.ones(n), b] + ([y] if y is not None else [])).astype(np.float64)
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ x
    res = x - X @ beta
    s2 = (res ** 2).sum(0) / (n - X.shape[1])
    return np.abs(beta[1] / np.sqrt(s2 * xtx_inv[1, 1] + 1e-12))


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    atlas = cfg["abide"]["main_atlas"]
    raw = ROOT / cfg["paths"]["raw"] / cfg["abide"]["pipeline"] / cfg["abide"]["strategy"] / atlas
    d = np.load(ROOT / "data" / "processed" / f"abide_{atlas}.npz", allow_pickle=True)

    @lru_cache(maxsize=None)
    def load_ts(i):
        return load_roi_timeseries(raw / f"{d['file_id'][i]}_{atlas}.1D")

    rows = []
    for split in range(3):
        p = build_regions(d["fc"], d["dx"], d["site"], d["n_dead_rois"], load_ts, m=10, level="x2",
                          split_seed=split, keep_ts=True)
        dec = build_decoy(p, load_ts, d["fc"], d["dx"], d["site"], m=10, seed=split, mode="timeseries",
                          exclude=p.meta["first_regions"])
        R = p.fc.shape[1]
        iu = np.triu_indices(R, 1)
        dec_e, dis_e, inc_e, k = dec.inside[iu], p.inside[iu], dec.incident[iu], int(dec.inside.sum() // 2)
        for frac in (0.5, 0.9):
            for fold in range(len(BRAINOOD_FOLDS)):
                tr, va, te = site_fold(p.site, fold)
                rng = np.random.default_rng([split, fold, int(frac * 100), False])   # as scripts/06_gate3.py
                b = np.zeros(len(p.y), bool)
                for part, f in ((tr, frac), (va, frac), (te, 0.5)):
                    b[part] = assign_scanner(p.y[part], p.site[part], f, rng)
                x = np.where(b[:, None, None], dec.fc_b, p.fc)[tr][:, iu[0], iu[1]].astype(np.float64)
                for name, t in (("naive", scanner_t(x, b[tr])), ("adjusted", scanner_t(x, b[tr], p.y[tr]))):
                    top = np.argsort(-t)[:k]
                    rows.append({"split": split, "frac": frac, "fold": fold, "map": name,
                                 "precision_decoy": float(dec_e[top].mean()),
                                 "share_decoy_incident": float(inc_e[top].mean()),
                                 "share_disease": float(dis_e[top].mean())})
        print(f"split {split} done", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "final" / "audit_check"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "audit_check.csv", index=False)
    print(df.groupby(["map", "frac"])[["precision_decoy", "share_decoy_incident", "share_disease"]].agg(["mean", "min"]).round(3).to_string())


if __name__ == "__main__":
    main()
