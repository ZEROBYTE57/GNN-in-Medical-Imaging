"""Stage 4: the novelty-vs-existing analysis.

    python scripts/04_analysis.py --run results/loco_v1

Reads a finished LOCO run and writes ``<run>/analysis/`` with:

    A1_headline.csv          in-centre / held-out / gap per arm, all metrics
    A2_per_centre.csv        held-out C per held-out centre per arm
    A3_bootstrap.csv         paired bootstrap of every arm vs the reference
    A4_k_sensitivity.csv     held-out C and gap as a function of k
    A5_mechanism.csv         graph diagnostics joined to outcomes, per fold
    A6_vs_baselines.csv      every GNN arm against the best tabular baseline
    fig_*.png                one figure per table
    ANALYSIS.md              the numbers written up, with the verdict spelled out

"Existing" is the reproduced Peng et al. rule (``snf_threshold``); "novel" is
``centre_balanced_knn``. ``knn`` is the control that separates the two
ingredients of the novelty -- rank-based selection and centre mixing.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlfgnn.eval.bootstrap import paired_bootstrap_cindex  # noqa: E402
from mlfgnn.eval.report import generalisation_gap  # noqa: E402

logger = logging.getLogger("analysis")

EXISTING = "snf_threshold"
NOVEL = "centre_balanced_knn"
CONTROL = "knn"
METRICS = ["harrell_c", "uno_c", "auc_12m", "auc_24m", "auc_60m", "ibs"]


def arm_of(row) -> str:
    k = row["k"]
    return f"{row['graph_mode']}" + ("" if pd.isna(k) else f"@k{int(k)}")


def load_run(run: Path):
    results = pd.read_csv(run / "results_raw.csv")
    results["arm"] = results.apply(arm_of, axis=1)
    raw_diag = run / "graph_diagnostics_raw.csv"
    diag = pd.read_csv(raw_diag) if raw_diag.exists() else None
    if diag is not None:
        diag["arm"] = diag.apply(arm_of, axis=1)
    else:
        logger.warning("no graph_diagnostics_raw.csv -- mechanism analysis skipped (re-run 02_run_loco.py)")
    scores = pd.read_csv(run / "patient_scores.csv") if (run / "patient_scores.csv").exists() else None
    cfg = json.loads((run / "config_used.json").read_text(encoding="utf-8")) if (run / "config_used.json").exists() else {}
    return results, diag, scores, cfg


# ---------------------------------------------------------------------------
# A1 headline
# ---------------------------------------------------------------------------
def headline(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m in METRICS if m in results.columns]
    rows = []
    for (model, arm), g in results.groupby(["model", "arm"]):
        row = {"model": model, "arm": arm}
        for m in metrics:
            for split in ("in_centre", "held_out_centre"):
                v = g.loc[g["split"] == split, m]
                row[f"{m}_{split}_mean"] = v.mean()
                row[f"{m}_{split}_std"] = v.std()
        gaps = generalisation_gap(g, "harrell_c")
        row["gap_mean"], row["gap_std"], row["n_folds_x_seeds"] = gaps["gap"].mean(), gaps["gap"].std(), len(gaps)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("harrell_c_held_out_centre_mean", ascending=False)


# ---------------------------------------------------------------------------
# A2 per centre
# ---------------------------------------------------------------------------
def per_centre(results: pd.DataFrame) -> pd.DataFrame:
    held = results[results["split"] == "held_out_centre"]
    t = held.pivot_table(index=["model", "arm"], columns="fold", values="harrell_c", aggfunc="mean")
    t["mean"] = t.mean(axis=1)
    t["worst_centre"] = t.drop(columns="mean").min(axis=1)
    return t.reset_index().sort_values("mean", ascending=False)


# ---------------------------------------------------------------------------
# A3 paired bootstrap
# ---------------------------------------------------------------------------
def bootstrap_table(scores: pd.DataFrame, outcomes: pd.DataFrame, reference_arm: str, iters: int) -> pd.DataFrame:
    s = scores[scores["on"] == "held_out_centre"].copy()
    s["arm"] = s.apply(arm_of, axis=1)
    s["label"] = s["model"] + "/" + s["arm"]
    pivot = s.pivot_table(index=["row", "fold"], columns="label", values="risk", aggfunc="mean").dropna()
    if pivot.empty:
        return pd.DataFrame()
    rows_idx = pivot.index.get_level_values("row").to_numpy()
    folds = pivot.index.get_level_values("fold").to_numpy()
    time = outcomes["time"].to_numpy()[rows_idx]
    event = outcomes["event"].to_numpy()[rows_idx]

    ref_cols = [c for c in pivot.columns if c.endswith("/" + reference_arm)]
    if not ref_cols:
        return pd.DataFrame()
    ref = ref_cols[0]
    out = []
    for col in pivot.columns:
        if col == ref:
            continue
        r = paired_bootstrap_cindex(pivot[col].to_numpy(), pivot[ref].to_numpy(), time, event,
                                    fold=folds, n_iters=iters, seed=0)
        out.append({"arm": col, "vs": ref, "delta_C": r.observed, "ci_low": r.ci_low,
                    "ci_high": r.ci_high, "p": r.p_value, "significant": r.significant})
    return pd.DataFrame(out).sort_values("delta_C", ascending=False)


# ---------------------------------------------------------------------------
# A4 k sensitivity
# ---------------------------------------------------------------------------
def k_sensitivity(results: pd.DataFrame) -> pd.DataFrame:
    g = results[results["graph_mode"].isin([NOVEL, CONTROL])]
    gaps = generalisation_gap(g, "harrell_c")
    t = gaps.groupby(["graph_mode", "k"]).agg(
        held_out_mean=("held_out_centre", "mean"), held_out_std=("held_out_centre", "std"),
        in_centre_mean=("in_centre", "mean"), gap_mean=("gap", "mean"), gap_std=("gap", "std"), n=("gap", "size"),
    ).reset_index()
    return t


# ---------------------------------------------------------------------------
# A5 mechanism: diagnostics vs outcome, fold by fold
# ---------------------------------------------------------------------------
def mechanism(results: pd.DataFrame, diag: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    if diag is None:
        return pd.DataFrame(), {}
    held = results[(results["split"] == "held_out_centre") & (results["graph_mode"] != "none")]
    gaps = generalisation_gap(results[results["graph_mode"] != "none"], "harrell_c")
    gaps["arm"] = gaps.apply(arm_of, axis=1)
    keys = ["seed", "fold", "arm"]
    merged = diag.merge(held[keys + ["harrell_c"]], on=keys, how="inner").merge(
        gaps[keys + ["gap"]], on=keys, how="left"
    )
    corr = {}
    for col in ("cross_centre_edge_ratio", "centre_assortativity", "degree_mean", "risk_homophily"):
        if col in merged.columns and merged[col].nunique() > 2:
            corr[f"spearman({col}, held_out_C)"] = float(merged[[col, "harrell_c"]].corr(method="spearman").iloc[0, 1])
            corr[f"spearman({col}, gap)"] = float(merged[[col, "gap"]].corr(method="spearman").iloc[0, 1])
    cols = keys + ["cross_centre_edge_ratio", "centre_assortativity", "degree_mean", "risk_homophily", "harrell_c", "gap"]
    return merged[[c for c in cols if c in merged.columns]], corr


# ---------------------------------------------------------------------------
# A6 against tabular baselines (Mueller et al.'s critique)
# ---------------------------------------------------------------------------
def vs_baselines(results: pd.DataFrame) -> pd.DataFrame:
    held = results[results["split"] == "held_out_centre"]
    base = held[held["graph_mode"] == "none"].groupby("model")["harrell_c"].mean()
    if base.empty:
        return pd.DataFrame()
    best_name, best_c = base.idxmax(), base.max()
    gnn = held[held["graph_mode"] != "none"].groupby(["model", "arm"])["harrell_c"].mean().reset_index()
    gnn["best_tabular"] = best_name
    gnn["best_tabular_C"] = best_c
    gnn["delta_vs_tabular"] = gnn["harrell_c"] - best_c
    return gnn.sort_values("delta_vs_tabular", ascending=False)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figures(out: Path, head: pd.DataFrame, pc: pd.DataFrame, ks: pd.DataFrame, mech: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # gap by arm
    h = head[head["arm"] != "none"]
    if len(h):
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(h) + 1.5))
        y = np.arange(len(h))
        ax.barh(y - 0.2, h["harrell_c_in_centre_mean"], 0.38, xerr=h["harrell_c_in_centre_std"].fillna(0), label="in-centre", color="#9ecae1")
        ax.barh(y + 0.2, h["harrell_c_held_out_centre_mean"], 0.38, xerr=h["harrell_c_held_out_centre_std"].fillna(0), label="held-out centre", color="#e6842a")
        ax.set_yticks(y, h["model"] + " / " + h["arm"]); ax.invert_yaxis(); ax.axvline(0.5, ls="--", c="#999")
        ax.set_xlabel("Harrell C"); ax.legend(frameon=False); ax.set_title("A1  In-centre vs held-out centre")
        fig.tight_layout(); fig.savefig(out / "fig_A1_gap.png", dpi=180); plt.close(fig)

    # per-centre
    if len(pc):
        cols = [c for c in pc.columns if c not in {"model", "arm", "mean", "worst_centre"}]
        fig, ax = plt.subplots(figsize=(8, 4))
        for _, r in pc.iterrows():
            ax.plot(cols, [r[c] for c in cols], marker="o", label=f"{r['model']}/{r['arm']}", lw=1.5 if r["arm"].startswith(NOVEL) else 1, alpha=0.9)
        ax.set_ylabel("held-out Harrell C"); ax.set_xlabel("held-out centre"); ax.set_title("A2  Which centres are hard")
        ax.legend(fontsize=7, ncol=2, frameon=False); fig.tight_layout(); fig.savefig(out / "fig_A2_per_centre.png", dpi=180); plt.close(fig)

    # k sensitivity
    if len(ks):
        fig, ax = plt.subplots(figsize=(6, 4))
        for mode, g in ks.groupby("graph_mode"):
            ax.errorbar(g["k"], g["held_out_mean"], yerr=g["held_out_std"].fillna(0), marker="o", capsize=3, label=mode)
        ax.set_xlabel("k (per centre for centre_balanced_knn)"); ax.set_ylabel("held-out Harrell C"); ax.set_title("A4  k sensitivity")
        ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out / "fig_A4_k.png", dpi=180); plt.close(fig)

    # mechanism scatter
    if len(mech) and "cross_centre_edge_ratio" in mech.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        for arm, g in mech.groupby("arm"):
            ax.scatter(g["cross_centre_edge_ratio"], g["harrell_c"], label=arm, s=28, alpha=0.8)
        ax.set_xlabel("cross-centre edge ratio (eval graph)"); ax.set_ylabel("held-out Harrell C"); ax.set_title("A5  Mechanism: centre mixing vs transport")
        ax.legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig(out / "fig_A5_mechanism.png", dpi=180); plt.close(fig)


# ---------------------------------------------------------------------------
# Write-up
# ---------------------------------------------------------------------------
def fmt(m, s) -> str:
    return f"{m:.3f} ± {0.0 if pd.isna(s) else s:.3f}"


def write_markdown(out: Path, run: Path, cfg: dict, head: pd.DataFrame, pc: pd.DataFrame, boot: pd.DataFrame,
                   ks: pd.DataFrame, corr: dict, vsb: pd.DataFrame, data_note: str) -> None:
    gnn = head[head["arm"] != "none"].set_index("arm")
    L: list[str] = []
    L.append(f"# Novelty vs existing — analysis of `{run.name}`\n")
    L.append(data_note + "\n")
    dev = cfg.get("device") or {}
    if dev:
        L.append(f"Hardware: {dev.get('gpu_name', dev.get('device'))}, torch {dev.get('torch')}.\n")

    L.append("## 1. Headline (Harrell C, mean ± std over LOCO folds × seeds)\n")
    L.append("| arm | in-centre | held-out centre | gap |\n|---|---|---|---|")
    for _, r in head.iterrows():
        L.append(f"| {r['model']} / {r['arm']} | {fmt(r['harrell_c_in_centre_mean'], r['harrell_c_in_centre_std'])} | "
                 f"{fmt(r['harrell_c_held_out_centre_mean'], r['harrell_c_held_out_centre_std'])} | {fmt(r['gap_mean'], r['gap_std'])} |")

    ex = [a for a in gnn.index if a.startswith(EXISTING)]
    nv = [a for a in gnn.index if a.startswith(NOVEL)]
    if ex and nv:
        e = gnn.loc[ex[0]]
        best_nv = gnn.loc[nv].sort_values("harrell_c_held_out_centre_mean", ascending=False).iloc[0]
        d_held = best_nv["harrell_c_held_out_centre_mean"] - e["harrell_c_held_out_centre_mean"]
        d_gap = best_nv["gap_mean"] - e["gap_mean"]
        L.append(f"\n**Existing** ({ex[0]}): held-out {e['harrell_c_held_out_centre_mean']:.3f}, gap {e['gap_mean']:+.3f}.  ")
        L.append(f"**Novel** (best {best_nv.name}): held-out {best_nv['harrell_c_held_out_centre_mean']:.3f}, gap {best_nv['gap_mean']:+.3f}.  ")
        L.append(f"Δ held-out C = {d_held:+.3f}; Δ gap = {d_gap:+.3f} (negative = the novel rule closes the gap).\n")

    if len(boot):
        L.append("## 2. Paired bootstrap on held-out patients (vs existing rule)\n")
        L.append("| arm | ΔC | 95% CI | p | significant |\n|---|---|---|---|---|")
        for _, r in boot.iterrows():
            L.append(f"| {r['arm']} | {r['delta_C']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {r['p']:.4f} | {'yes' if r['significant'] else 'no'} |")
        sig = boot[boot["significant"]]
        L.append("\n" + (f"{len(sig)} arm(s) differ significantly from the existing rule." if len(sig) else
                         "No arm differs significantly from the existing rule at α = 0.05 — with this cohort size, "
                         "the study is powered for an effect of roughly the size Peng et al. report (0.788 → 0.674), not smaller."))

    if len(pc):
        L.append("\n## 3. Per held-out centre\n")
        cols = [c for c in pc.columns if c not in {"model", "arm"}]
        L.append("| arm | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
        for _, r in pc.iterrows():
            L.append(f"| {r['model']}/{r['arm']} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")

    if len(ks):
        L.append("\n## 4. Sensitivity to k\n")
        L.append("| rule | k | held-out C | gap |\n|---|---|---|---|")
        for _, r in ks.iterrows():
            L.append(f"| {r['graph_mode']} | {int(r['k'])} | {fmt(r['held_out_mean'], r['held_out_std'])} | {fmt(r['gap_mean'], r['gap_std'])} |")

    if corr:
        L.append("\n## 5. Mechanism\n")
        L.append("Spearman correlations across (fold × seed × arm) between graph structure and outcome:\n")
        for k_, v in corr.items():
            L.append(f"- `{k_}` = {v:+.3f}")
        L.append("\nA positive correlation of cross-centre edge ratio with held-out C, and a negative one with the gap, "
                 "is the signature the proposal predicts. If both are near zero, centre mixing is not what moves the score on this cohort.")

    if len(vsb):
        L.append("\n## 6. Against the best tabular baseline\n")
        L.append(f"Best tabular model on the held-out centres: **{vsb['best_tabular'].iloc[0]}** (C = {vsb['best_tabular_C'].iloc[0]:.3f}).\n")
        L.append("| GNN arm | held-out C | Δ vs tabular |\n|---|---|---|")
        for _, r in vsb.iterrows():
            L.append(f"| {r['model']}/{r['arm']} | {r['harrell_c']:.3f} | {r['delta_vs_tabular']:+.3f} |")
        L.append("\nMueller et al. (TMLR 2024) showed population GNNs often fail to beat tabular baselines when the graph is weak; "
                 "a graph rule only earns its place if this column is positive.")

    (out / "ANALYSIS.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--reference", default=EXISTING)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--data-note", default=None, help="one line stating which cohort produced this run")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # The write-up uses Δ and ±; a cp1252 Windows console would otherwise choke
    # on the final print AFTER every file has already been written.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    run = Path(args.run)
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    results, diag, scores, cfg = load_run(run)

    feature_table = Path(cfg.get("config", {}).get("data", {}).get("feature_table", ""))
    data_note = args.data_note or (
        f"Cohort: `{feature_table}` — **simulated data**; numbers exercise the pipeline and say nothing about the real cohort."
        if "synth" in feature_table.name else f"Cohort: `{feature_table}`."
    )

    head = headline(results); head.to_csv(out / "A1_headline.csv", index=False)
    pc = per_centre(results); pc.to_csv(out / "A2_per_centre.csv", index=False)

    boot = pd.DataFrame()
    if scores is not None and feature_table.exists():
        df = pd.read_parquet(feature_table) if feature_table.suffix == ".parquet" else pd.read_csv(feature_table)
        dcfg = cfg["config"]["data"]
        outcomes = df[[dcfg["time_col"], dcfg["event_col"]]].rename(columns={dcfg["time_col"]: "time", dcfg["event_col"]: "event"})
        boot = bootstrap_table(scores, outcomes, args.reference, args.iters)
        boot.to_csv(out / "A3_bootstrap.csv", index=False)

    ks = k_sensitivity(results); ks.to_csv(out / "A4_k_sensitivity.csv", index=False)
    mech, corr = mechanism(results, diag); mech.to_csv(out / "A5_mechanism.csv", index=False)
    vsb = vs_baselines(results); vsb.to_csv(out / "A6_vs_baselines.csv", index=False)

    figures(out, head, pc, ks, mech)
    write_markdown(out, run, cfg, head, pc, boot, ks, corr, vsb, data_note)
    print((out / "ANALYSIS.md").read_text(encoding="utf-8"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
