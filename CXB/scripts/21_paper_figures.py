"""Paper versions of the result figures, drawn at their printed size for Springer LNNS/LNCS.

Text width 12.2 cm = 4.8 in, so every font is its final size (at least 6.5 pt). Same data
and summaries as scripts/10_figures.py (whose helpers are reused), different layout:
  fig1_recovery_lncs.pdf/png   recovery of the planted edges by planted strength
  fig2_drift_lncs.pdf/png      scanner share, disease AUPRC and accuracy vs the shortcut
The README figures (fig1_recovery.png, fig2_drift.png) are left unchanged.
"""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("figs", ROOT / "scripts" / "10_figures.py")
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)

W = 4.8
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "lines.linewidth": 1.3,
                     "axes.linewidth": 0.6, "grid.linewidth": 0.4, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def save(fig, name):
    for ext, dpi in (("pdf", 300), ("png", 600)):
        fig.savefig(F.FIG / f"{name}.{ext}", dpi=dpi)
    plt.close(fig)
    print("written", name)


def recovery():
    df = pd.read_csv(F.RES / "benchmark" / "benchmark.csv")
    levels = ["x1", "x2", "x4"]
    s = F.per_run(df, ["model", "explainer", "level"], ["group_auprc", "ceiling_auprc"])
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.05), sharey=True)
    for ax, model in zip(axes, ["mlp", "gcn"]):
        ceil = [s.loc[(model, "saliency", lv), ("ceiling_auprc", "mean")] for lv in levels]
        ax.plot(levels, ceil, color=F.MUTED, ls="--", lw=1.0, label="t-test reference")
        ax.axhline(0.008, color=F.MUTED, ls=":", lw=0.9, label="Chance")
        for i, ex in enumerate([e for e in F.EXPL if (model, e, "x1") in s.index]):
            m = [s.loc[(model, ex, lv), ("group_auprc", "mean")] for lv in levels]
            sd = [s.loc[(model, ex, lv), ("group_auprc", "std")] for lv in levels]
            ax.errorbar(levels, m, yerr=F.clip_err(m, sd), color=F.SERIES[i], marker=F.MARKERS[i], ms=3.6, capsize=2,
                        elinewidth=0.8, label=F.EXPL[ex].replace("GNNExplainer-style mask", "Input mask"))
        ax.set_title(model.upper(), loc="left", color=F.INK)
        ax.set_xlabel("Planted strength")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Disease AUPRC")
    h, lab = axes[1].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1), handlelength=1.8, columnspacing=1.0)
    fig.subplots_adjust(wspace=0.08, bottom=0.25)
    save(fig, "fig1_recovery_lncs")


def drift():
    files = sorted((F.RES / "drift").glob("gate2_*drift.csv"))
    df = pd.read_csv(files[-1])
    df = df[(df.explainer == "saliency") & (df.design == "train")]
    s = F.per_run(df, ["model", "frac"], ["scanner_share", "disease_auprc", "test_acc"])
    fig, axes = plt.subplots(1, 3, figsize=(W, 1.85))
    panels = [("scanner_share", "Scanner share (top 45)"), ("disease_auprc", "Disease AUPRC"),
              ("test_acc", "Test accuracy")]
    fr = sorted(df.frac.unique())
    for ax, (col, title) in zip(axes, panels):
        for i, model in enumerate(["mlp", "gcn"]):
            m = [s.loc[(model, f), (col, "mean")] for f in fr]
            sd = [s.loc[(model, f), (col, "std")] for f in fr]
            ax.errorbar([int(100 * f) for f in fr], m, yerr=F.clip_err(m, sd), color=F.SERIES[i], marker=F.MARKERS[i],
                        ms=3.6, capsize=2, elinewidth=0.8, label=model.upper())
        ax.set_title(title, loc="left", color=F.INK)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([50, 70, 90])
    axes[1].set_xlabel(r"Shortcut strength $\rho$ in training (%)")
    axes[0].legend(loc="upper left", handlelength=1.6)
    fig.subplots_adjust(wspace=0.32)
    save(fig, "fig2_drift_lncs")


if __name__ == "__main__":
    recovery()
    drift()
