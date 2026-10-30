"""Sanity check of region-level time-series planting on real ABIDE data.

For each atlas, region count and level, a few fake splits are built and reported:
  - recovery ceiling: AUPRC of ranking edges by a plain two-sample |t| on the fake labels,
    inside edges = positives, untouched edges = negatives, diluted edges ignored
    ("don't care"); also with diluted edges counted as negatives. No model is trained:
    if a simple test cannot find the planted edges at a level, no explainer should;
  - matrices that are no longer valid correlation matrices (expected: none);
  - inside shift achieved vs target, and dilution on incident edges;
  - share of patients whose signal could be made orthogonal to all regions.

    .venv\\Scripts\\python.exe scripts\\03_plant_check.py [--splits 3]
"""
import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import average_precision_score

from cxb.data import load_roi_timeseries
from cxb.planting import LEVELS, REGION_COUNTS, build_regions, invalid_fraction

ROOT = Path(__file__).resolve().parents[1]


def t_abs(fc, y):
    a, b = fc[y == 1].astype(np.float64), fc[y == 0].astype(np.float64)
    se = np.sqrt(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(np.where(se > 0, (a.mean(0) - b.mean(0)) / se, 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=3)
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    raw = ROOT / cfg["paths"]["raw"] / cfg["abide"]["pipeline"] / cfg["abide"]["strategy"]
    out = {}
    for atlas in cfg["abide"]["atlases"]:
        d = np.load(ROOT / "data" / "processed" / f"abide_{atlas}.npz", allow_pickle=True)
        fc, dx, site, dead, fid = d["fc"], d["dx"], d["site"], d["n_dead_rois"], d["file_id"]

        @lru_cache(maxsize=None)
        def load_ts(i, atlas=atlas, fid=fid):
            return load_roi_timeseries(raw / atlas / f"{fid[i]}_{atlas}.1D")

        iu = np.triu_indices(fc.shape[1], 1)
        print(f"\n== {atlas} ({fc.shape[1]} regions, {len(iu[0])} edges)")
        for m in REGION_COUNTS:
            for level in LEVELS:
                ap_dc, ap_all, bad, shift, dil, orth, t0 = [], [], [], [], [], [], time.time()
                for s in range(args.splits):
                    p = build_regions(fc, dx, site, dead, load_ts, m=m, level=level, split_seed=s)
                    t = t_abs(p.fc, p.y)[iu]
                    ins, inc = p.inside[iu], p.incident[iu]
                    keep = ~inc
                    ap_dc.append(average_precision_score(ins[keep], t[keep]))
                    ap_all.append(average_precision_score(ins, t))
                    bad.append(invalid_fraction(p.fc[p.y == 1]))
                    delta = p.fc[p.y == 1] - fc[p.pool_idx[p.y == 1]]
                    shift.append(delta[:, p.inside].mean())
                    dil.append(delta[:, p.incident].mean())
                    orth.append(p.meta["share_fully_orthogonal"])
                key = f"{atlas}|m={m}|{level}"
                out[key] = {"inside_edges": p.meta["n_inside_edges"], "auprc_dontcare": float(np.mean(ap_dc)),
                            "auprc_dontcare_min": float(np.min(ap_dc)), "auprc_incident_negative": float(np.mean(ap_all)),
                            "invalid_share": float(np.mean(bad)), "target_shift": p.meta["target_shift"],
                            "inside_shift": float(np.mean(shift)), "incident_shift": float(np.mean(dil)),
                            "share_fully_orthogonal": float(np.mean(orth)), "n_pool": p.meta["n_pool"]}
                r = out[key]
                print(f"m={m:2d} ({r['inside_edges']:3d} edges) {level:4s} | ceiling AUPRC {r['auprc_dontcare']:.3f} "
                      f"(min {r['auprc_dontcare_min']:.3f}; incident as negatives {r['auprc_incident_negative']:.3f}) | "
                      f"shift {r['inside_shift']:+.3f} (target {r['target_shift']:.3f}), dilution {r['incident_shift']:+.3f} | "
                      f"invalid {100 * r['invalid_share']:.1f}% | orthogonal {100 * r['share_fully_orthogonal']:.0f}% | "
                      f"{time.time() - t0:.0f}s", flush=True)
        print(f"   chance AUPRC about {21 / len(iu[0]):.4f} (21 edges) to {190 / len(iu[0]):.4f} (190 edges)")
    dest = ROOT / "results" / "plant_check_regions.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"\nsaved {dest}")


if __name__ == "__main__":
    main()
