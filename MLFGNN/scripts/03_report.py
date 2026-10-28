"""Stage 4: statistics and tables from a finished run.

    python scripts/03_report.py --run results/loco_v1 --reference snf_threshold

Runs the paired bootstrap of every graph rule against the reference rule on the
held-out-centre patients, stratified by fold, and prints the table that goes in
the paper.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlfgnn.eval.bootstrap import paired_bootstrap_cindex  # noqa: E402
from mlfgnn.eval.report import (  # noqa: E402
    format_table,
    graph_diagnostics_table,
    headline_table,
    plot_generalisation_gap,
)

logger = logging.getLogger("report")


def arm_label(row) -> str:
    k = row.get("k")
    suffix = "" if pd.isna(k) else f"@k{int(k)}"
    return f"{row['model']}/{row['graph_mode']}{suffix}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="directory written by 02_run_loco.py")
    p.add_argument("--reference", default="snf_threshold",
                   help="graph mode every other arm is compared against")
    p.add_argument("--metric", default="harrell_c")
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run = Path(args.run)

    results = pd.read_csv(run / "results_raw.csv")
    print("\n=== Headline table ===\n")
    table = headline_table(results, args.metric)
    print(format_table(table).to_string(index=False))

    diag_path = run / "graph_diagnostics.csv"
    if diag_path.exists():
        print("\n=== What each rule did to the graph ===\n")
        diagnostics = pd.read_csv(diag_path)
        cols = [c for c in diagnostics.columns if c not in {"seed", "fold"}]
        print(graph_diagnostics_table(diagnostics[cols]).round(3).to_string(index=False))

    scores_path = run / "patient_scores.csv"
    if not scores_path.exists():
        print("\n(no patient_scores.csv -- skipping the paired bootstrap)")
        return

    scores = pd.read_csv(scores_path)
    scores = scores[scores["on"] == "held_out_centre"]
    outcomes = _outcomes(run, results, scores)
    if outcomes is None:
        return

    print(f"\n=== Paired bootstrap vs {args.reference} (held-out centres) ===\n")
    _bootstrap_table(scores, outcomes, args)


def _outcomes(run: Path, results: pd.DataFrame, scores: pd.DataFrame):
    """Recover PFS time/event for the scored rows from the feature table."""
    cfg_path = run / "config_used.json"
    if not cfg_path.exists():
        print("config_used.json missing -- cannot locate the feature table")
        return None
    import json

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))["config"]
    features = Path(cfg["data"]["feature_table"])
    if not features.exists():
        print(f"feature table {features} not found -- skipping the bootstrap")
        return None
    df = pd.read_parquet(features) if features.suffix == ".parquet" else pd.read_csv(features)
    return df[[cfg["data"]["time_col"], cfg["data"]["event_col"]]].rename(
        columns={cfg["data"]["time_col"]: "time", cfg["data"]["event_col"]: "event"}
    )


def _bootstrap_table(scores: pd.DataFrame, outcomes: pd.DataFrame, args) -> None:
    scores = scores.copy()
    scores["arm"] = scores.apply(arm_label, axis=1)

    # Average each patient's score over seeds so the comparison is one score per
    # patient per arm -- the seeds are replicates of the same arm, not more data.
    pivot = scores.pivot_table(index=["row", "fold"], columns="arm", values="risk", aggfunc="mean")
    pivot = pivot.dropna()
    if pivot.empty:
        print("no patients scored by every arm -- skipping")
        return

    rows = pivot.index.get_level_values("row").to_numpy()
    folds = pivot.index.get_level_values("fold").to_numpy()
    time = outcomes["time"].to_numpy()[rows]
    event = outcomes["event"].to_numpy()[rows]

    reference = [c for c in pivot.columns if args.reference in c]
    if not reference:
        print(f"reference {args.reference!r} not among arms: {list(pivot.columns)}")
        return
    ref_col = reference[0]

    out = []
    for arm in pivot.columns:
        if arm == ref_col:
            continue
        res = paired_bootstrap_cindex(
            pivot[arm].to_numpy(), pivot[ref_col].to_numpy(),
            time, event, fold=folds, n_iters=args.iters, alpha=args.alpha, seed=0,
        )
        out.append(
            {
                "arm": arm,
                "vs": ref_col,
                "delta_C": round(res.observed, 4),
                "ci_low": round(res.ci_low, 4),
                "ci_high": round(res.ci_high, 4),
                "p": round(res.p_value, 4),
                "significant": res.significant,
            }
        )
    print(pd.DataFrame(out).sort_values("delta_C", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
