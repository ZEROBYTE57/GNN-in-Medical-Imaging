"""Method figures for the paper: the CXB overview (fig12) and the CF-Site schematic (fig13).

Every thumbnail is real data from the cached illustration run of scripts/17_graph_figures.py
(x2 planting, person-specific scanner effect, split 0, seed 0). Run that script first if
results/final/graph_figures/cache.npz is missing. CPU only, a few seconds.

    .venv\\Scripts\\python.exe scripts\\18_flowchart.py
"""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
CACHE = ROOT / "results" / "final" / "graph_figures" / "cache.npz"

# Colours as in the result figures (scripts/10_figures.py, 17_graph_figures.py)
INK, MUTED, LINE, PANEL = "#1f1f1e", "#5f5e58", "#8d8b84", "#f4f3ef"
DISEASE, SCANNER, OTHER = "#eb6834", "#2a78d6", "#b4b2aa"
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "text.color": INK,
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                     "mathtext.bf": "Arial:bold", "mathtext.cal": "cmsy10", "pdf.fonttype": 42})
CM_FC = "RdBu_r"
CM_DIS = LinearSegmentedColormap.from_list("dis", ["#ffffff", DISEASE])
CM_SCN = LinearSegmentedColormap.from_list("scn", ["#ffffff", SCANNER])
CM_SAL = LinearSegmentedColormap.from_list("sal", ["#ffffff", "#3b3a36"])


def helpers():
    spec = importlib.util.spec_from_file_location("gf", ROOT / "scripts" / "17_graph_figures.py")
    gf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gf)
    return gf


class Canvas:
    """Figure with an overlay axis in figure coordinates (0-1) for panels, arrows and text."""

    def __init__(self, w, h):
        self.fig = plt.figure(figsize=(w, h))
        self.r = w / h
        self.bg = self.fig.add_axes([0, 0, 1, 1], zorder=0)
        self.top = self.fig.add_axes([0, 0, 1, 1], zorder=10)
        for a in (self.bg, self.top):
            a.set_xlim(0, 1)
            a.set_ylim(0, 1)
            a.axis("off")
        self.top.patch.set_alpha(0)

    def panel(self, x, y, w, h, label, title):
        self.bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                         fc=PANEL, ec="none", mutation_aspect=self.r))
        if label:
            self.top.text(x + 0.008, y + h - 0.03, label, fontsize=8.5, fontweight="bold", va="top")
            self.top.text(x + 0.026, y + h - 0.03, title, fontsize=8, fontweight="bold", va="top")

    def matrix(self, x, y, w, m, cmap, vmin, vmax, edge=LINE):
        a = self.fig.add_axes([x, y, w, w * self.r], zorder=5)
        a.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor(edge)
            s.set_linewidth(0.6)
        return a

    def box(self, x, y, w, h, fc="#ffffff", ec=LINE, lw=0.7, ls="-"):
        self.top.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008",
                                          fc=fc, ec=ec, lw=lw, ls=ls, mutation_aspect=self.r, zorder=2))

    def arrow(self, p, q, color=INK, lw=0.9, ls="-", rad=0.0):
        self.top.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, color=color, lw=lw, ls=ls,
                                           shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}", zorder=3))

    def text(self, x, y, s, **kw):
        kw.setdefault("va", "top")
        kw.setdefault("ha", "left")
        self.top.text(x, y, s, zorder=4, **kw)

    def save(self, name, top=1.0):
        from matplotlib.transforms import Bbox
        w, h = self.fig.get_size_inches()
        crop = Bbox([[0, 0], [w, h * top]])
        for ext in ("png", "pdf"):
            self.fig.savefig(FIG / f"{name}.{ext}", dpi=300, facecolor="white", bbox_inches=crop)
        plt.close(self.fig)
        print("written", FIG / f"{name}.png")


def brain(cv, gf, coords, rect, adj, col, size):
    from nilearn import plotting
    a = cv.fig.add_axes(rect, zorder=5)
    plotting.plot_connectome(adj, coords, node_color=col, node_size=size, edge_cmap=gf.CAT_CMAP, edge_vmin=-1,
                             edge_vmax=1, edge_kwargs={"linewidth": 0.9, "alpha": 0.9}, display_mode="z",
                             axes=a, annotate=False, colorbar=False)


def graph_icon(cv, cx, cy, s):
    """A small brain-graph glyph: the model takes a graph (regions = nodes, connections = edges)."""
    rng = np.random.default_rng(3)
    pts = np.array([[np.cos(t), np.sin(t)] for t in np.linspace(0, 2 * np.pi, 7, endpoint=False)])
    pts = pts * [s, s * cv.r] + [cx, cy]
    for i in range(7):
        for j in range(i + 1, 7):
            if rng.random() < 0.45:
                cv.top.plot(*zip(pts[i], pts[j]), color=LINE, lw=0.5, zorder=3)
    cv.top.scatter(pts[:, 0], pts[:, 1], s=6, color=INK, zorder=4, lw=0)


def overview(c, gf, coords):
    R = c["disease"].shape[0]
    iu = np.triu_indices(R, 1)
    cv = Canvas(6.9, 2.75)
    cv.panel(0.0, 0.0, 0.475, 0.9, "a", "Benchmark: a known answer and a known trap")
    cv.panel(0.485, 0.0, 0.245, 0.9, "b", "Train and explain")
    cv.panel(0.74, 0.0, 0.26, 0.9, "c", "Score the explanation")

    # (a) control connectome -> planted disease / scanner shortcut -> on the brain
    mw = 0.075
    cv.matrix(0.02, 0.4, mw, c["example"], CM_FC, -1, 1)
    cv.text(0.02, 0.38, "Control\nconnectome $x_i$", fontsize=7, linespacing=1.2)
    cv.text(0.02, 0.26, "ABIDE I,\n116 regions;\n343 controls,\nhalf 'patients'", fontsize=6.3, color=MUTED,
            linespacing=1.25)
    dis = np.abs(c["planted"])
    scn = np.abs(c["scanner"])
    cv.matrix(0.16, 0.55, mw, dis, CM_DIS, 0, np.percentile(dis[c["disease"]], 90), edge=DISEASE)
    cv.matrix(0.16, 0.11, mw, scn, CM_SCN, 0, np.percentile(scn[c["decoy"]], 90), edge=SCANNER)
    cv.text(0.24, 0.77, "+ disease", fontsize=7, color=DISEASE, fontweight="bold")
    cv.text(0.24, 0.715, "patients only\n10 regions, $\\times\\ell$\n45 true edges", fontsize=6.3, color=MUTED,
            linespacing=1.25)
    cv.text(0.24, 0.33, "+ scanner", fontsize=7, color=SCANNER, fontweight="bold")
    cv.text(0.24, 0.275, "linked to $y$:\n$\\rho$ in training,\n0.5 at test", fontsize=6.3, color=MUTED,
            linespacing=1.25)
    cv.arrow((0.1, 0.52), (0.155, 0.66), rad=-0.2)
    cv.arrow((0.1, 0.45), (0.155, 0.26), rad=0.15)
    idx = np.where(c["disease"][iu] | c["decoy"][iu])[0]
    col, size = gf.node_style(R, c["dis_regions"], c["dec_regions"], base=0)
    brain(cv, gf, coords, [0.335, 0.2, 0.135, 0.56], gf.category_adjacency(R, idx, c["disease"], c["decoy"]),
          col, size * 0.35)
    cv.arrow((0.318, 0.64), (0.352, 0.57))
    cv.arrow((0.318, 0.25), (0.352, 0.36))
    cv.text(0.4025, 0.19, "343 benchmark scans,\n10 leave-site-out folds", fontsize=6.3, color=MUTED, ha="center",
            linespacing=1.25)

    # (b) fix -> model -> group explanation
    x0 = 0.495
    cv.text(x0, 0.76, "Training", fontsize=7, fontweight="bold")
    for k, name in enumerate(["plain", "ComBat (×4)","CF-Site", "oracle CF-Site"]):
        cv.box(x0, 0.655 - k * 0.075, 0.12, 0.058)
        cv.text(x0 + 0.006, 0.684 - k * 0.075, name, fontsize=6.2, va="center")
    cv.box(x0, 0.08, 0.12, 0.29, ec=INK)
    cv.text(x0 + 0.06, 0.355, "Model $f$", fontsize=7, fontweight="bold", ha="center")
    graph_icon(cv, x0 + 0.06, 0.245, 0.022)
    cv.text(x0 + 0.06, 0.17, "LR, MLP, GCN,\nBrainNetCNN", fontsize=6.2, color=MUTED, ha="center", linespacing=1.2)
    cv.arrow((x0 + 0.06, 0.43), (x0 + 0.06, 0.375))
    sal = gf.edge_matrix(c["map_mlp_plain_90"], R)
    cv.matrix(0.635, 0.30, mw, sal, CM_SAL, 0, np.percentile(sal, 98))
    cv.arrow((x0 + 0.123, 0.24), (0.632, 0.36))
    cv.text(0.635, 0.28, "Explanation $S$", fontsize=7)
    cv.text(0.635, 0.225, "per connection,\nmean over the\ntest site", fontsize=6.3, color=MUTED, linespacing=1.25)
    cv.text(0.635, 0.76, "Explainers", fontsize=7, fontweight="bold")
    cv.text(0.635, 0.705, "saliency\nInt. Gradients\ninput mask", fontsize=6.3, color=MUTED, linespacing=1.25)

    # (c) top 45 on the brain + the three scores
    cv.arrow((0.715, 0.38), (0.765, 0.52))
    top = gf.top_edges(c["map_mlp_plain_90"])
    col, size = gf.node_style(R, c["dis_regions"], c["dec_regions"], base=0)
    brain(cv, gf, coords, [0.755, 0.36, 0.125, 0.43], gf.category_adjacency(R, top, c["disease"], c["decoy"]),
          col, size * 0.35)
    cv.text(0.89, 0.72, "Top 45\nof $S$", fontsize=7, linespacing=1.2)
    cv.text(0.89, 0.6, "(MLP, $\\rho$=0.9)", fontsize=6.3, color=MUTED)
    rows = [(DISEASE, "Disease AUPRC", "$S$ vs. the 45 true edges"),
            (SCANNER, "Scanner share", "scanner edges in the top 45"),
            (INK, "Accuracy", "on the held-out site")]
    for k, (color, head, sub) in enumerate(rows):
        y = 0.31 - k * 0.1
        cv.top.add_patch(FancyBboxPatch((0.758, y - 0.075), 0.007, 0.075, boxstyle="square,pad=0",
                                        fc=color, ec="none", zorder=3))
        cv.text(0.773, y, head, fontsize=7, fontweight="bold")
        cv.text(0.773, y - 0.04, sub, fontsize=6.3, color=MUTED)
    cv.save("fig12_flowchart", top=0.905)


def cfsite(c, gf):
    cv = Canvas(6.9, 1.75)
    cv.panel(0.0, 0.0, 1.0, 1.0, "", "")
    mw = 0.062
    x = c["example"]
    xt = x + c["scanner"]                                 # illustration: the subject moved to the other scanner
    cv.matrix(0.02, 0.35, mw, x, CM_FC, -1, 1)
    cv.text(0.02, 0.31, "Training scan $x_i$", fontsize=7)
    cv.text(0.02, 0.215, "scanner $b_i$, label $y_i$", fontsize=6.3, color=MUTED)

    cv.box(0.15, 0.48, 0.17, 0.36, ec=INK)
    cv.text(0.235, 0.80, "Scanner model", fontsize=7, fontweight="bold", ha="center")
    cv.text(0.235, 0.70, "label-protected ComBat\nfit on training data:\n$\\gamma_b$, $\\delta_b$ per connection",
            fontsize=6.3, color=MUTED, ha="center", linespacing=1.25)
    cv.box(0.15, 0.08, 0.17, 0.27, ec=LINE, ls=(0, (3, 2)))
    cv.text(0.235, 0.31, "Oracle (upper bound)", fontsize=7, fontweight="bold", ha="center")
    cv.text(0.235, 0.22, "true scan at the\nother scanner", fontsize=6.3, color=MUTED, ha="center", linespacing=1.25)
    cv.arrow((0.085, 0.6), (0.148, 0.66))
    cv.arrow((0.085, 0.45), (0.148, 0.24), ls=(0, (3, 2)), color=LINE)

    cv.matrix(0.38, 0.35, mw, xt, CM_FC, -1, 1, edge=SCANNER)
    cv.arrow((0.322, 0.66), (0.376, 0.6))
    cv.arrow((0.322, 0.22), (0.376, 0.42), ls=(0, (3, 2)), color=LINE)
    cv.text(0.38, 0.31, "Counterfactual $\\tilde{x}_i$", fontsize=7)
    cv.text(0.38, 0.215, "scanner $1-b_i$, same $y_i$", fontsize=6.3, color=MUTED)

    cv.box(0.53, 0.28, 0.13, 0.5, ec=INK)
    cv.text(0.595, 0.74, "Model $f$", fontsize=7, fontweight="bold", ha="center")
    graph_icon(cv, 0.595, 0.52, 0.02)
    cv.text(0.595, 0.385, "shared weights", fontsize=6.3, color=MUTED, ha="center")
    cv.top.plot([0.051, 0.051, 0.595], [0.6, 0.93, 0.93], color=INK, lw=0.9, zorder=3)
    cv.arrow((0.595, 0.93), (0.595, 0.785))
    cv.arrow((0.446, 0.5), (0.528, 0.5))

    cv.box(0.70, 0.36, 0.28, 0.34, ec=INK)
    cv.text(0.84, 0.64, "Training loss", fontsize=7, fontweight="bold", ha="center")
    cv.text(0.84, 0.53, "$\\mathcal{L} = \\mathrm{CE}(f(x_i), y_i) + \\mathrm{CE}(f(\\tilde{x}_i), y_i)$",
            fontsize=7, ha="center")
    cv.arrow((0.662, 0.53), (0.698, 0.53))
    cv.text(0.70, 0.25, "At test: new sites are used as they are\n(no harmonisation, no counterfactuals)",
            fontsize=6.3, color=MUTED, linespacing=1.25)
    cv.save("fig13_cfsite")


def main():
    if not CACHE.exists():
        raise SystemExit(f"missing {CACHE}: run scripts/17_graph_figures.py first")
    gf = helpers()
    c = np.load(CACHE)
    coords, _ = gf.atlas_info()
    FIG.mkdir(parents=True, exist_ok=True)
    overview(c, gf, coords)
    cfsite(c, gf)


if __name__ == "__main__":
    main()
