"""Aggregation, tables and the headline figure.

The headline claim of the study is a *gap*, not a level:

    generalisation_gap = C_in_centre - C_held_out_centre

so the reporting is organised around that quantity rather than around a single
test-set number. A graph rule that lowers in-centre C but lowers the gap more is
the desired result, and a table of held-out C alone would hide that trade.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PRIMARY_METRICS = ["harrell_c", "uno_c", "auc_12m", "auc_24m", "auc_60m", "ibs"]

#: Sentinel standing in for "this arm has no k" while pivoting (see below).
_NO_K = -1.0


def aggregate(results: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Mean +/- std over folds and seeds.

    ``by`` defaults to the experimental arm (model, graph mode, k) crossed with
    the evaluation split, so one row is one cell of the results table.
    """
    by = by or ["model", "graph_mode", "k", "split"]
    by = [c for c in by if c in results.columns]
    metrics = [m for m in PRIMARY_METRICS if m in results.columns]

    grouped = results.groupby(by, dropna=False)[metrics]
    out = grouped.agg(["mean", "std", "count"])
    out.columns = [f"{m}_{stat}" for m, stat in out.columns]
    return out.reset_index()


def generalisation_gap(results: pd.DataFrame, metric: str = "harrell_c") -> pd.DataFrame:
    """In-centre minus held-out-centre, per arm, per fold, per seed.

    Computed fold-wise before averaging: averaging the two levels separately and
    subtracting would discard the pairing and inflate the variance of the gap.
    """
    keys = [c for c in ["model", "graph_mode", "k", "fold", "seed"] if c in results.columns]

    # pivot_table silently drops rows whose index contains NaN, and the
    # threshold arm has no k -- which would delete the reference arm from every
    # table. Carry NaN through as a sentinel instead.
    frame = results.copy()
    has_k = "k" in frame.columns
    if has_k:
        frame["k"] = frame["k"].fillna(_NO_K)

    wide = frame.pivot_table(index=keys, columns="split", values=metric)
    if not {"in_centre", "held_out_centre"} <= set(wide.columns):
        raise KeyError(
            "results must contain both 'in_centre' and 'held_out_centre' splits; "
            f"found {list(wide.columns)}"
        )
    wide = wide.reset_index()
    if has_k:
        wide["k"] = wide["k"].replace(_NO_K, np.nan)
    wide["gap"] = wide["in_centre"] - wide["held_out_centre"]
    return wide


def headline_table(results: pd.DataFrame, metric: str = "harrell_c") -> pd.DataFrame:
    """The table that goes in the paper: one row per graph rule."""
    gaps = generalisation_gap(results, metric)
    keys = [c for c in ["model", "graph_mode", "k"] if c in gaps.columns]
    table = (
        gaps.groupby(keys, dropna=False)
        .agg(
            in_centre_mean=("in_centre", "mean"),
            in_centre_std=("in_centre", "std"),
            held_out_mean=("held_out_centre", "mean"),
            held_out_std=("held_out_centre", "std"),
            gap_mean=("gap", "mean"),
            gap_std=("gap", "std"),
            n=("gap", "count"),
        )
        .reset_index()
        .sort_values("held_out_mean", ascending=False)
    )
    return table


def format_table(table: pd.DataFrame) -> pd.DataFrame:
    """Human-readable ``mean +/- std`` strings for pasting into a manuscript."""
    out = table.copy()
    for prefix in ("in_centre", "held_out", "gap"):
        mean, std = f"{prefix}_mean", f"{prefix}_std"
        if mean in out.columns:
            out[prefix] = [
                f"{m:.3f} +/- {0.0 if np.isnan(s) else s:.3f}"
                for m, s in zip(out[mean], out[std], strict=True)
            ]
            out = out.drop(columns=[mean, std])
    return out


def graph_diagnostics_table(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Mechanism table: what each rule actually did to the graph."""
    cols = [
        "cross_centre_edge_ratio",
        "centre_assortativity",
        "degree_mean",
        "edge_density",
        "risk_homophily",
        "heldout_reachability",
        "isolated_nodes",
    ]
    cols = [c for c in cols if c in diagnostics.columns]
    keys = [c for c in ["graph_mode", "k"] if c in diagnostics.columns]
    return diagnostics.groupby(keys, dropna=False)[cols].mean().reset_index()


def plot_generalisation_gap(
    results: pd.DataFrame,
    out_path: str | Path,
    metric: str = "harrell_c",
    title: str | None = None,
):
    """The headline figure: in-centre vs held-out-centre C-index per graph rule."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = headline_table(results, metric)
    label = table["graph_mode"].astype(str)
    if "k" in table.columns and table["k"].notna().any():
        label = label + table["k"].map(lambda k: "" if pd.isna(k) else f" (k={int(k)})")

    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(table) + 2.2))

    ax.barh(y - 0.2, table["in_centre_mean"], height=0.38,
            xerr=table["in_centre_std"].fillna(0), label="in-centre",
            color="#9ecae1", error_kw={"lw": 1, "capsize": 3})
    ax.barh(y + 0.2, table["held_out_mean"], height=0.38,
            xerr=table["held_out_std"].fillna(0), label="held-out centre",
            color="#e6842a", error_kw={"lw": 1, "capsize": 3})

    for i, row in enumerate(table.itertuples()):
        ax.annotate(
            f"gap {row.gap_mean:+.3f}",
            xy=(max(row.in_centre_mean, row.held_out_mean) + 0.012, i),
            va="center", fontsize=9, color="#444",
        )

    ax.set_yticks(y, label)
    ax.set_xlabel(f"{metric.replace('_', ' ')}  (mean +/- std over LOCO folds x seeds)")
    ax.axvline(0.5, color="#999", lw=1, ls="--")
    ax.set_xlim(0.4, min(1.0, float(table[["in_centre_mean", "held_out_mean"]].max().max()) + 0.14))
    ax.set_title(title or "Generalisation gap by population-graph construction rule")
    ax.legend(loc="lower right", frameon=False)
    ax.invert_yaxis()
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


def write_all(
    results: pd.DataFrame,
    diagnostics: pd.DataFrame | None,
    out_dir: str | Path,
    metric: str = "harrell_c",
) -> dict[str, Path]:
    """Write every artefact of a run to ``out_dir`` and return the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    def _csv(name: str, df: pd.DataFrame) -> None:
        path = out_dir / name
        df.to_csv(path, index=False)
        written[name] = path

    _csv("results_raw.csv", results)
    _csv("results_aggregated.csv", aggregate(results))
    table = headline_table(results, metric)
    _csv("headline_table.csv", table)
    _csv("headline_table_formatted.csv", format_table(table))
    _csv("generalisation_gap.csv", generalisation_gap(results, metric))

    if diagnostics is not None and len(diagnostics):
        # Raw rows (one per arm x fold x seed) feed the mechanism analysis;
        # the aggregated table is what goes in the paper.
        _csv("graph_diagnostics_raw.csv", diagnostics)
        _csv("graph_diagnostics.csv", graph_diagnostics_table(diagnostics))

    try:
        written["figure"] = plot_generalisation_gap(results, out_dir / "fig_generalisation_gap.png", metric)
    except Exception as exc:  # noqa: BLE001 - a missing backend must not lose the CSVs
        logger.warning("figure not written: %s", exc)

    return written
