"""Brain-graph figures: the benchmark, the drift and the fixes drawn on the brain network itself.

These figures illustrate results already reported; they add no new numbers to the statistics.
Stage 1 (cached; one small GPU run): x2 planting, person-specific scanner shortcut, split 0,
BrainOOD's 10 folds, seed 0, MLP and GCN. Saliency group explanations (each fold's map scaled
to sum 1, then averaged over folds) for: plain at 50% and 90% shortcut; protected standard
ComBat, CF-Site and oracle CF-Site at 90%. Stage 2 (CPU): seven figures in docs/figures/.
Region positions: AAL atlas centres (nilearn).

    .venv\\Scripts\\python.exe -u scripts\\17_graph_figures.py [--recompute]
"""
import argparse
import os
import textwrap
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath

from cxb import combat
from cxb.data import load_roi_timeseries
from cxb.decoy import assign_scanner, build_decoy, site_effect_eta2
from cxb.explain import EXPLAINERS, edge_scores
from cxb.models import MODELS, train_model
from cxb.planting import build_regions, edge_effect_sizes
from cxb.siteswap import train_siteswap
from cxb.splits import BRAINOOD_FOLDS, site_fold

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
CACHE = ROOT / "results" / "final" / "graph_figures" / "cache.npz"
# Same style as scripts/10_figures.py; categories: disease = orange, scanner = blue, other = grey
INK, MUTED = "#1f1f1e", "#6b6a64"
DISEASE, SCANNER, OTHER = "#eb6834", "#2a78d6", "#b4b2aa"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "text.color": INK, "figure.dpi": 200,
                     "savefig.bbox": "tight", "legend.frameon": False})
K = 45
CONDS = ["plain_50", "plain_90", "combat_90", "cfsite_90", "oracle_90"]
COND_NAMES = {"plain_50": "Plain, 50% shortcut", "plain_90": "Plain, 90% shortcut",
              "combat_90": "ComBat (protected), 90%", "cfsite_90": "CF-Site, 90%", "oracle_90": "Oracle CF-Site, 90%"}


def compute(dev):
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    atlas = cfg["abide"]["main_atlas"]
    raw = ROOT / cfg["paths"]["raw"] / cfg["abide"]["pipeline"] / cfg["abide"]["strategy"] / atlas
    d = np.load(ROOT / "data" / "processed" / f"abide_{atlas}.npz", allow_pickle=True)

    @lru_cache(maxsize=None)
    def load_ts(i):
        return load_roi_timeseries(raw / f"{d['file_id'][i]}_{atlas}.1D")

    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)  # noqa: E731
    p = build_regions(d["fc"], d["dx"], d["site"], d["n_dead_rois"], load_ts, m=10, level="x2", split_seed=0, keep_ts=True)
    dec = build_decoy(p, load_ts, d["fc"], d["dx"], d["site"], m=10, seed=0, mode="timeseries",
                      exclude=p.meta["first_regions"])
    R = p.fc.shape[1]
    iu = np.triu_indices(R, 1)
    y, yt = p.y, torch.tensor(p.y, dtype=torch.long, device=dev)
    maps = {f"{m}_{c}": np.zeros(len(iu[0])) for m in ("mlp", "gcn") for c in CONDS}
    for fold in range(len(BRAINOOD_FOLDS)):
        tr, va, te = site_fold(p.site, fold)
        for frac in (0.5, 0.9):
            rng = np.random.default_rng([0, fold, int(frac * 100), False])   # as scripts/06_gate3.py
            b = np.zeros(len(y), dtype=bool)
            for part, f in ((tr, frac), (va, frac), (te, 0.5)):
                b[part] = assign_scanner(y[part], p.site[part], f, rng)
            x, bi = np.where(b[:, None, None], dec.fc_b, p.fc), b.astype(int)
            runs = {"plain_50": "plain"} if frac == 0.5 else {"plain_90": "plain", "combat_90": "combat",
                                                               "cfsite_90": "cfsite", "oracle_90": "oracle"}
            if frac == 0.9:
                f_cp = combat.fit(x[tr], bi[tr], covariates=y[tr, None])
                f_eb = combat.fit(x[tr], bi[tr], covariates=y[tr, None], eb=True)
            for cond, method in runs.items():
                if method == "combat":
                    xin = (combat.harmonise(f_eb, x[tr], bi[tr], y[tr, None]), combat.harmonise(f_eb, x[va], bi[va], y[va, None]),
                           combat.harmonise(f_eb, x[te], bi[te]))
                else:
                    xin = (x[tr], x[va], x[te])
                for name in ("mlp", "gcn"):
                    torch.manual_seed(0)
                    model = MODELS[name](R, xin[0].mean(0), xin[0].std(0)).to(dev)
                    if method in ("cfsite", "oracle"):
                        xs = (np.where(b[tr][:, None, None], p.fc[tr], dec.fc_b[tr]) if method == "oracle"
                              else combat.swap(f_cp, x[tr], bi[tr], 1 - bi[tr], y[tr, None]))
                        model, _, _ = train_siteswap(model, T(xin[0]), T(xs), yt[tr], T(xin[1]), yt[va], lam=0.0, seed=0)
                    else:
                        model, _, _ = train_model(model, T(xin[0]), yt[tr], T(xin[1]), yt[va], seed=0)
                    s = edge_scores(EXPLAINERS["saliency"](model, T(xin[2]))).detach().cpu().numpy().mean(0)
                    maps[f"{name}_{cond}"] += s / s.sum() / len(BRAINOOD_FOLDS)
        print(f"fold {fold} done", flush=True)
    patients = (p.fc[y == 1] - d["fc"][p.pool_idx][y == 1]).mean(0)   # exact planted change
    scanner = (dec.fc_b - p.fc).mean(0)
    eta, _, _ = site_effect_eta2(d["fc"], d["site"], np.arange(len(d["fc"])), min_n=20)
    dvals, _ = edge_effect_sizes(d["fc"], np.where(d["dx"] == 1)[0], np.where(d["dx"] == 2)[0])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, disease=p.inside, decoy=dec.inside, dis_regions=p.regions, dec_regions=dec.regions,
                        planted=patients, scanner=scanner, eta=eta, autism_d=dvals, example=d["fc"][p.pool_idx[0]],
                        **{f"map_{k}": v for k, v in maps.items()})


# ----------------------------------------------------------------------------------- drawing helpers

def atlas_info():
    from nilearn import datasets, plotting
    a = datasets.fetch_atlas_aal(version="SPM12")
    coords, vals = plotting.find_parcellation_cut_coords(a.maps, return_label_names=True)
    names = dict(zip([int(i) for i in a.indices[1:]], a.labels[1:]))
    return np.asarray(coords), [names[int(v)] for v in vals]


LOBES = [("Frontal", ("Precentral", "Frontal", "Rolandic", "Supp_Motor", "Olfactory", "Rectus", "Paracentral")),
         ("Cingulate & insula", ("Cingulum", "Insula")),
         ("Subcortical & limbic", ("Hippocampus", "ParaHippocampal", "Amygdala", "Caudate", "Putamen", "Pallidum", "Thalamus")),
         ("Temporal", ("Temporal", "Heschl")),
         ("Parietal", ("Postcentral", "Parietal", "SupraMarginal", "Angular", "Precuneus")),
         ("Occipital", ("Calcarine", "Cuneus", "Lingual", "Occipital", "Fusiform")),
         ("Cerebellum", ("Cerebelum", "Vermis"))]


def lobe_of(name):
    for i, (_, keys) in enumerate(LOBES):
        if name.startswith(keys):
            return i
    return len(LOBES) - 1


def edge_matrix(vec, R):
    m = np.zeros((R, R))
    iu = np.triu_indices(R, 1)
    m[iu] = vec
    return m + m.T


def top_edges(vec, k=K):
    return np.argsort(-vec)[:k]


def category_adjacency(R, idx, disease, decoy):
    """+1 disease, -1 scanner, small positive = other, for the chosen upper-triangle edges."""
    iu = np.triu_indices(R, 1)
    a = np.zeros((R, R))
    for e in idx:
        i, j = iu[0][e], iu[1][e]
        a[i, j] = a[j, i] = 1.0 if disease[i, j] else (-1.0 if decoy[i, j] else 0.02)
    return a


CAT_CMAP = LinearSegmentedColormap.from_list("cat", [SCANNER, OTHER, DISEASE])


def node_style(R, dis_regions, dec_regions, base=6):
    col, size = [OTHER] * R, np.full(R, base, float)
    for r in dis_regions:
        col[r], size[r] = DISEASE, 28
    for r in dec_regions:
        col[r], size[r] = SCANNER, 28
    return col, size


def glass(ax, adj, coords, col, size, title, mode="z"):
    from nilearn import plotting
    plotting.plot_connectome(adj, coords, node_color=col, node_size=size, edge_cmap=CAT_CMAP, edge_vmin=-1,
                             edge_vmax=1, edge_kwargs={"linewidth": 1.4, "alpha": 0.85}, display_mode=mode,
                             axes=ax, annotate=False, colorbar=False)
    ax.set_title(title, loc="left", fontsize=9, color=INK)


def legend_handles(other=True):
    h = [Line2D([0], [0], color=DISEASE, lw=2.5, label="Planted disease connection"),
         Line2D([0], [0], color=SCANNER, lw=2.5, label="Scanner-shortcut connection")]
    if other:
        h.append(Line2D([0], [0], color=OTHER, lw=2.5, label="Other connection"))
    return h


def note(fig, text, width, y=-0.05):
    fig.text(0.01, y, textwrap.fill(text, width), fontsize=7, color=MUTED, va="top")


def shares(vec, disease, decoy, R):
    iu = np.triu_indices(R, 1)
    t = top_edges(vec)
    return disease[iu][t].mean(), decoy[iu][t].mean()


# ----------------------------------------------------------------------------------- figures

def fig_benchmark(c, coords):
    R = c["disease"].shape[0]
    iu = np.triu_indices(R, 1)
    idx = np.where(c["disease"][iu] | c["decoy"][iu])[0]
    adj = category_adjacency(R, idx, c["disease"], c["decoy"])
    col, size = node_style(R, c["dis_regions"], c["dec_regions"])
    fig = plt.figure(figsize=(7.4, 2.6))
    ax = fig.add_axes([0, 0.12, 1, 0.8])
    glass(ax, adj, coords, col, size, "", mode="lzr")
    fig.suptitle("The benchmark on the brain: a known answer (45 planted connections) and a known trap "
                 "(45 scanner connections)", x=0.01, ha="left", fontsize=9.5, color=INK)
    fig.legend(handles=legend_handles(False), loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(FIG / "fig5_brain_benchmark.png")
    plt.close(fig)


def fig_grid(c, coords, conds, fname, title):
    R = c["disease"].shape[0]
    col, size = node_style(R, c["dis_regions"], c["dec_regions"], base=4)
    fig, axes = plt.subplots(2, len(conds), figsize=(2.6 * len(conds), 5.6))
    for r, model in enumerate(("mlp", "gcn")):
        for k, cond in enumerate(conds):
            vec = c[f"map_{model}_{cond}"]
            adj = category_adjacency(R, top_edges(vec), c["disease"], c["decoy"])
            ds, ss = shares(vec, c["disease"], c["decoy"], R)
            head = (COND_NAMES[cond] + "\n") if r == 0 else ""
            glass(axes[r, k], adj, coords, col, size, f"{head}{model.upper()}: disease {ds:.0%} · scanner {ss:.0%}",
                  mode="z")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    note(fig, "Each brain shows the model's 45 most important connections (saliency, averaged over 10 held-out "
              "hospitals; one split and seed, as an illustration). Percentages: share of those connections that are "
              "planted disease or scanner connections. Averages over all runs are in the result tables.",
         int(38 * len(conds)), y=-0.03)
    fig.savefig(FIG / fname)
    plt.close(fig)


def chord(ax, R, names, edges_cat, title):
    """Circular connectome: left hemisphere on the left, right on the right, lobes in the same order."""
    lobe = np.array([lobe_of(n) for n in names])
    side = np.array([0 if n.endswith("_L") else (1 if n.endswith("_R") else 2) for n in names])
    order_l = sorted([i for i in range(R) if side[i] == 0], key=lambda i: (lobe[i], names[i]))
    order_r = sorted([i for i in range(R) if side[i] == 1], key=lambda i: (lobe[i], names[i]))
    mid = sorted([i for i in range(R) if side[i] == 2], key=lambda i: names[i])
    n = len(order_l) + len(order_r) + len(mid)
    ang = {}
    step = 2 * np.pi / n
    for k, i in enumerate(order_l):                      # left half, top to bottom
        ang[i] = np.pi / 2 + (k + 0.5) * step
    for k, i in enumerate(mid):                          # vermis at the bottom
        ang[i] = np.pi / 2 + (len(order_l) + k + 0.5) * step
    for k, i in enumerate(reversed(order_r)):            # right half, bottom to top (mirror)
        ang[i] = np.pi / 2 + (len(order_l) + len(mid) + k + 0.5) * step
    pos = {i: (np.cos(a), np.sin(a)) for i, a in ang.items()}
    lobe_cols = ["#d9d7cf", "#c4c2b9"]
    for i in range(R):
        a = ang[i]
        ax.plot([1.0 * np.cos(a), 1.06 * np.cos(a)], [1.0 * np.sin(a), 1.06 * np.sin(a)],
                color=lobe_cols[lobe[i] % 2], lw=3.2, solid_capstyle="butt")
    for li in range(len(LOBES)):
        for half in (order_l, order_r):
            members = [ang[i] for i in half if lobe[i] == li]
            if members:
                a = np.mean(members)
                ax.text(1.16 * np.cos(a), 1.16 * np.sin(a), str(li + 1), ha="center", va="center", fontsize=6.5,
                        color=MUTED, fontweight="bold")
    for cat in ("other", "scanner", "disease"):          # draw coloured edges on top
        for (i, j), c in edges_cat.items():
            if c != cat:
                continue
            (x1, y1), (x2, y2) = pos[i], pos[j]
            path = MPath([(x1, y1), (0.1 * (x1 + x2), 0.1 * (y1 + y2)), (x2, y2)], [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3])
            color = {"disease": DISEASE, "scanner": SCANNER, "other": OTHER}[c]
            ax.add_patch(PathPatch(path, fc="none", ec=color, lw=1.1, alpha=0.85))
    ax.text(-1.25, 1.12, "Left", fontsize=7, color=MUTED)
    ax.text(0.95, 1.12, "Right", fontsize=7, color=MUTED)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=9, color=INK)


def fig_circle(c, names):
    R = c["disease"].shape[0]
    iu = np.triu_indices(R, 1)

    def cats(idx):
        out = {}
        for e in idx:
            i, j = iu[0][e], iu[1][e]
            out[(i, j)] = "disease" if c["disease"][i, j] else ("scanner" if c["decoy"][i, j] else "other")
        return out
    truth = cats(np.where(c["disease"][iu] | c["decoy"][iu])[0])
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    chord(axes[0], R, names, truth, "A. Planted answer and scanner trap")
    chord(axes[1], R, names, cats(top_edges(c["map_mlp_plain_50"])), "B. MLP explanation, 50% shortcut")
    chord(axes[2], R, names, cats(top_edges(c["map_mlp_plain_90"])), "C. MLP explanation, 90% shortcut")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    note(fig, "Regions around the circle, left hemisphere on the left and right on the right, grouped by lobe: "
              + " · ".join(f"{i + 1} {n}" for i, (n, _) in enumerate(LOBES))
              + ". B and C: the model's 45 most important connections (one split and seed, as an illustration).", 170, y=-0.08)
    fig.savefig(FIG / "fig8_circle.png")
    plt.close(fig)


def fig_matrices(c, names):
    R = c["disease"].shape[0]
    lobe = np.array([lobe_of(n) for n in names])
    order = np.lexsort((np.arange(R), lobe))
    bounds = np.where(np.diff(lobe[order]))[0] + 0.5
    mats = [("A. Planted effect\n(change added to fake patients)", c["planted"], "Oranges"),
            ("B. Scanner effect\n(scanner B − scanner A)", c["scanner"], "Blues"),
            ("C. GCN explanation,\n50% shortcut", edge_matrix(c["map_gcn_plain_50"], R), "Greys"),
            ("D. GCN explanation,\n90% shortcut", edge_matrix(c["map_gcn_plain_90"], R), "Greys")]
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 3.0))
    for ax, (title, m, cmap) in zip(axes, mats):
        m = np.abs(m[np.ix_(order, order)])
        np.fill_diagonal(m, 0)
        ax.imshow(m, cmap=cmap, vmin=0, vmax=np.percentile(m, 99.7), interpolation="nearest")
        for bnd in bounds:
            ax.axhline(bnd, color="white", lw=0.5)
            ax.axvline(bnd, color="white", lw=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, loc="left", fontsize=8.5, color=INK)
    starts = np.r_[0, np.ceil(bounds)].astype(int)
    for s, e in zip(starts, np.r_[starts[1:], R]):
        axes[0].text(-3, (s + e) / 2, LOBES[lobe[order][s]][0], ha="right", va="center", fontsize=5.6, color=MUTED)
    note(fig, "116 × 116 connection matrices, regions grouped by lobe (white lines). Darker = larger effect, or more "
              "important to the model. The explanation moves from the disease pattern (A) towards the scanner rows (B) "
              "as the shortcut strengthens.", 170, y=-0.02)
    fig.savefig(FIG / "fig9_matrices.png")
    plt.close(fig)


def fig_real(c, coords):
    R = c["disease"].shape[0]
    iu = np.triu_indices(R, 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for ax, vec, colr, title in ((axes[0], c["eta"][iu], SCANNER, "A. Where real hospitals differ most\n(top 45, 19 hospitals)"),
                                 (axes[1], np.abs(c["autism_d"][iu]), DISEASE, "B. Where autism differs most\n(top 45, all subjects)")):
        a = np.zeros((R, R))
        for e in top_edges(vec):
            a[iu[0][e], iu[1][e]] = a[iu[1][e], iu[0][e]] = 1.0
        nodes = np.unique(np.r_[iu[0][top_edges(vec)], iu[1][top_edges(vec)]])
        col, size = [OTHER] * R, np.full(R, 4.0)
        for r in nodes:
            col[r], size[r] = colr, 18
        from nilearn import plotting
        plotting.plot_connectome(a, coords, node_color=col, node_size=size, edge_cmap=LinearSegmentedColormap.from_list(
            "one", [colr, colr]), edge_vmin=0, edge_vmax=1, edge_kwargs={"linewidth": 1.3, "alpha": 0.85},
            display_mode="z", axes=ax, annotate=False, colorbar=False)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
    note(fig, "Real ABIDE data, no planting. The hospital signal is strong (hospital predicted at balanced accuracy "
              "0.70, chance 0.05) and sits in different connections from the autism signal (0.64, chance 0.50).", 120,
         y=0.02)
    fig.savefig(FIG / "fig10_real_data_brain.png")
    plt.close(fig)


def fig_connectome(c, coords, names):
    R = c["disease"].shape[0]
    lobe = np.array([lobe_of(n) for n in names])
    order = np.lexsort((np.arange(R), lobe))
    fc = np.tanh(c["example"])
    np.fill_diagonal(fc, 0)
    fig = plt.figure(figsize=(7.4, 2.9))
    ax0 = fig.add_axes([0.0, 0.08, 0.32, 0.8])
    im = ax0.imshow(fc[np.ix_(order, order)], cmap=LinearSegmentedColormap.from_list("div", [SCANNER, "#f2f1ec", DISEASE]),
                    vmin=-1, vmax=1, interpolation="nearest")
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_title("A. One subject's connectome\n(116 × 116 correlations)", loc="left", fontsize=9, color=INK)
    cb = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6)
    ax1 = fig.add_axes([0.37, 0.08, 0.63, 0.8])
    iu = np.triu_indices(R, 1)
    vals = np.abs(fc[iu])
    keep = np.argsort(-vals)[: int(0.02 * len(vals))]
    a = np.zeros((R, R))
    a[iu[0][keep], iu[1][keep]] = fc[iu][keep]
    a = a + a.T
    from nilearn import plotting
    plotting.plot_connectome(a, coords, node_color=[OTHER] * R, node_size=6,
                             edge_cmap=LinearSegmentedColormap.from_list("div", [SCANNER, "#f2f1ec", DISEASE]),
                             edge_vmin=-1, edge_vmax=1, edge_kwargs={"linewidth": 0.9, "alpha": 0.8},
                             display_mode="lzr", axes=ax1, annotate=False, colorbar=False)
    ax1.set_title("B. The same connectome as a brain graph\n(116 regions; strongest 2% of connections)", loc="left",
                  fontsize=9, color=INK)
    fig.savefig(FIG / "fig11_connectome_graph.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(2)
    if args.recompute or not CACHE.exists():
        compute(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    c = dict(np.load(CACHE))
    coords, names = atlas_info()
    fig_connectome(c, coords, names)
    fig_benchmark(c, coords)
    fig_grid(c, coords, ["plain_50", "plain_90"], "fig6_brain_drift.png",
             "Explanation drift: the same models at a 50% and a 90% scanner shortcut")
    fig_grid(c, coords, ["plain_90", "combat_90", "cfsite_90", "oracle_90"], "fig7_brain_fixes.png",
             "Fixes on the brain (90% shortcut): where each model's explanation points")
    fig_circle(c, names)
    fig_matrices(c, names)
    fig_real(c, coords)
    print("figures written")


if __name__ == "__main__":
    main()
