"""Stage 3: the leave-one-centre-out graph-construction benchmark.

    python scripts/02_run_loco.py --config configs/experiment.yaml

    # fast smoke run on synthetic data
    python scripts/02_run_loco.py --features data/processed/features_synth.csv \
        --seeds 0 --k-values 5 --epochs 30 --out results/smoke

Writes ``results_raw.csv``, ``headline_table.csv``, ``graph_diagnostics.csv``,
``patient_scores.csv`` and the generalisation-gap figure to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlfgnn.config import load_config  # noqa: E402
from mlfgnn.eval.report import write_all  # noqa: E402
from mlfgnn.experiments.loco import run_loco  # noqa: E402

logger = logging.getLogger("loco")


def read_features(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--features", default=None, help="override data.feature_table")
    p.add_argument("--out", default=None, help="override output_dir")
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--k-values", type=int, nargs="*", default=None)
    p.add_argument("--modes", nargs="*", default=None)
    p.add_argument("--layers", nargs="*", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    p.add_argument("--no-baselines", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    if args.features:
        cfg.raw["data"]["feature_table"] = args.features
    if args.seeds is not None:
        cfg.raw["protocol"]["seeds"] = args.seeds
    if args.k_values is not None:
        cfg.raw["graphs"]["knn"]["k_values"] = args.k_values
    if args.modes is not None:
        cfg.raw["graphs"]["modes"] = args.modes
    if args.layers is not None:
        cfg.raw["model"]["layer_types"] = args.layers
    if args.epochs is not None:
        cfg.raw["train"]["epochs"] = args.epochs
    if args.no_baselines:
        cfg.raw["baselines"] = []
    if args.out:
        cfg.raw["output_dir"] = args.out

    feature_path = Path(cfg["data"]["feature_table"])
    if not feature_path.exists():
        raise SystemExit(
            f"feature table not found: {feature_path}\n"
            "Run scripts/00_make_synthetic.py for a simulated cohort, or "
            "scripts/01_extract_features.py once the TCIA download is in place."
        )

    df = read_features(feature_path)
    logger.info(
        "%d patients, %d centres, %.0f%% events",
        len(df), df[cfg["data"]["centre_col"]].nunique(),
        100 * df[cfg["data"]["event_col"]].mean(),
    )

    results, diagnostics = run_loco(df, cfg, device=args.device)

    out_dir = Path(cfg.get("output_dir", "results")) / cfg.get("run_name", "run")
    written = write_all(results, diagnostics, out_dir)

    scores = results.attrs.get("patient_scores")
    if scores is not None and len(scores):
        scores.to_csv(out_dir / "patient_scores.csv", index=False)
        written["patient_scores.csv"] = out_dir / "patient_scores.csv"

    with (out_dir / "config_used.json").open("w", encoding="utf-8") as fh:
        json.dump({"hash": cfg.hash, "device": results.attrs.get("device"), "config": cfg.raw}, fh, indent=2, default=str)

    print(f"\nconfig hash {cfg.hash} -> {out_dir}")
    for name, path in written.items():
        print(f"  {name:<32} {path}")

    from mlfgnn.eval.report import format_table, headline_table

    print("\nHeadline (Harrell C, mean +/- std over LOCO folds x seeds):\n")
    print(format_table(headline_table(results)).to_string(index=False))


if __name__ == "__main__":
    main()
