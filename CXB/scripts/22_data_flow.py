"""Data-to-score flow (fig16): how one brain scan becomes a benchmark score, for the introduction.

Six stages in two rows, each drawn with real data: the 116 AAL regions, four real region
signals, one real connectivity matrix, the exact planted change, a model, and a real
explanation ranking with the planted edges marked. Sized for single-column paper templates
(4.8 in wide; text at least 6.5 pt). Uses the cache of scripts/17_graph_figures.py
(one split and seed, x2 planting, plain MLP at a 50% link = no shortcut).
Prints any text that runs out of its box.
"""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
CACHE = ROOT / "results" / "final" / "graph_figures" / "cache.npz"

INK, MUTED, LINE, OTHER = "#1f1f1e", "#4f4e48", "#9a988f", "#b9b7af"
DISEASE, DISEASE_BG = "#c4501b", "#fcebe2"
plt.rcParams.update({"font.family": "Arial", "font.size": 6.5, "text.color": INK, "mathtext.fontset": "custom",
                     "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "pdf.fonttype": 42})
FIGW, W, H = 4.8, 100.0, 62.0
TS, LS, STEP = 7.2, 6.5, 2.45
BW, GX = 29.4, 5.3                       # box width, horizontal gap
XS = [0.4, 0.4 + BW + GX, 0.4 + 2 * (BW + GX)]
PIC = 15.0                               # picture height (units)


def helpers():
    spec = importlib.util.spec_from_file_location("gf", ROOT / "scripts" / "17_graph_figures.py")
    gf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gf)
    return gf


class Fig:
    def __init__(self):
        self.fig = plt.figure(figsize=(FIGW, FIGW * H / W))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(0, H)
        self.ax.axis("off")
        self.checks = []

    def text(self, x, y, s, size=LS, color=MUTED, weight="normal", ha="center", box=None, **kw):
        t = self.ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va="top", zorder=6, **kw)
        if box:
            self.checks.append((t, box, s))

    def box(self, x, top, title, lines, fc="#ffffff", ec=LINE, tc=INK):
        h = 1.0 + 2.3 + PIC + 0.9 + len(lines) * STEP + 0.6
        y = top - h
        self.ax.add_patch(FancyBboxPatch((x, y), BW, h, boxstyle="round,pad=0,rounding_size=0.9", fc=fc, ec=ec,
                                         lw=0.7, zorder=1))
        b = (x, y, BW, h)
        self.text(x + BW / 2, top - 0.9, title, size=TS, color=tc, weight="bold", box=b)
        pic = (x + 1.2, top - 3.3 - PIC, BW - 2.4, PIC)            # picture area (x, y, w, h)
        ly = top - 3.3 - PIC - 0.9
        for k, s in enumerate(lines):
            self.text(x + BW / 2, ly - k * STEP, s, box=b)
        return y, pic

    def inset(self, rect, equal=False):
        x, y, w, h = rect
        if equal:                                                   # square, centred in the area
            s = min(w, h)
            x, y, w, h = x + (w - s) / 2, y + (h - s) / 2, s, s
        a = self.fig.add_axes([x / W, y / H, w / W, h / H], zorder=3)
        return a

    def arrow(self, p, q):
        self.ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, color=INK, lw=0.9, shrinkA=0,
                                          shrinkB=0, zorder=4))

    def check(self):
        self.fig.canvas.draw()
        r = self.fig.canvas.get_renderer()
        bad = 0
        for t, (x, y, w, h), s in self.checks:
            bb = t.get_window_extent(r)
            (x0, y0), (x1, y1) = self.ax.transData.transform([(x, y), (x + w, y + h)])
            if bb.x0 < x0 + 0.5 or bb.x1 > x1 - 0.5 or bb.y0 < y0 + 0.5 or bb.y1 > y1:
                print("OVERFLOW:", s)
                bad += 1
        print("text overflow:", bad)


def clean(a, frame=True):
    a.set_xticks([])
    a.set_yticks([])
    for s in a.spines.values():
        s.set_visible(frame)
        s.set_linewidth(0.4)
        s.set_edgecolor(LINE)


def region_signals():
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    atlas = cfg["abide"]["main_atlas"]
    raw = ROOT / cfg["paths"]["raw"] / cfg["abide"]["pipeline"] / cfg["abide"]["strategy"] / atlas
    d = np.load(ROOT / "data" / "processed" / f"abide_{atlas}.npz", allow_pickle=True)
    ts = np.loadtxt(raw / f"{d['file_id'][0]}_{atlas}.1D", skiprows=1)
    return ts


def graph_glyph(a):
    rng = np.random.default_rng(3)
    ang = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    pts = np.c_[np.cos(ang), np.sin(ang)]
    for i in range(9):
        for j in range(i + 1, 9):
            if rng.random() < 0.35:
                a.plot(*zip(pts[i], pts[j]), color=OTHER, lw=0.6, zorder=1)
    a.scatter(pts[:, 0], pts[:, 1], s=7, color=INK, zorder=2, lw=0)
    a.set_xlim(-1.4, 1.4)
    a.set_ylim(-1.25, 1.25)
    a.set_aspect("equal")


def main():
    from nilearn import plotting
    gf = helpers()
    c = np.load(CACHE)
    coords, _ = gf.atlas_info()
    R = c["disease"].shape[0]
    iu = np.triu_indices(R, 1)
    f = Fig()
    top1 = H - 0.4

    # row 1: data
    y1, p = f.box(XS[0], top1, "Brain scan", ["T 3D volumes per person", "(median T = 196; 116–316)",
                                              "116 AAL regions (dots)"])
    a = f.inset(p)
    plotting.plot_connectome(np.zeros((R, R)), coords, node_color=OTHER, node_size=0.9, display_mode="z", axes=a,
                             annotate=False, colorbar=False)
    _, p = f.box(XS[1], top1, "116 region signals", ["atlas: one signal per region", "(4 of 116 shown,",
                                                     "one real subject)"])
    a = f.inset(p)
    ts = region_signals()
    for k, r in enumerate((0, 30, 60, 90)):
        z = (ts[:, r] - ts[:, r].mean()) / (ts[:, r].std() + 1e-9)
        a.plot(z * 0.32 - k * 1.6, color=[INK, MUTED, "#6f6d66", "#8d8b84"][k], lw=0.6)
    a.set_xlim(0, len(ts))
    clean(a, frame=False)
    _, p = f.box(XS[2], top1, "116 × 116 matrix", ["correlate every pair, Fisher z", "→ 6,670 unique edges",
                                                   "(the “fc” array)"])
    a = f.inset(p, equal=True)
    a.imshow(c["example"], cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    clean(a)
    mid1 = top1 - 3.3 - PIC / 2
    for i in range(2):
        f.arrow((XS[i] + BW, mid1), (XS[i + 1], mid1))

    # row 2: benchmark
    top2 = y1 - 5.4
    y2, p = f.box(XS[0], top2, "Planted disease", ["shared signal in 10 regions,", "fake patients only →",
                                                   "45 inside edges stronger"], fc=DISEASE_BG, ec=DISEASE, tc=DISEASE)
    a = f.inset(p)
    idx = np.where(c["disease"][iu])[0]                     # the 45 planted (inside) edges
    col = [OTHER] * R
    size = np.full(R, 0.0)
    for r in c["dis_regions"]:
        col[r], size[r] = DISEASE, 5.0
    plotting.plot_connectome(gf.category_adjacency(R, idx, c["disease"], c["decoy"]), coords, node_color=col,
                             node_size=size, edge_cmap=gf.CAT_CMAP, edge_vmin=-1, edge_vmax=1,
                             edge_kwargs={"linewidth": 0.7, "alpha": 0.9}, display_mode="z", axes=a, annotate=False,
                             colorbar=False)
    _, p = f.box(XS[1], top2, "Model", ["trained on fake patients", "vs fake controls",
                                        "(LR, MLP, GCN, BrainNetCNN)"])
    a = f.inset(p, equal=True)
    graph_glyph(a)
    clean(a, frame=False)
    _, p = f.box(XS[2], top2, "Explanation, scored", ["explainer ranks all 6,670 edges;", "AUPRC: are the 45 on top?",
                                                       "(chance ≈ 0.008, perfect 1.0)"])
    a = f.inset(p)
    vec = c["map_mlp_plain_50"]
    order = np.argsort(-vec)[:110]
    inside = c["disease"][iu][order]
    a.bar(np.arange(len(order)), vec[order] / vec[order[0]], width=0.9, color=[DISEASE if v else OTHER for v in inside],
          lw=0)
    a.axvline(44.5, color=INK, lw=0.6, ls=(0, (2, 1.5)))
    a.text(46.5, 0.97, "top 45", fontsize=6.0, color=INK, va="top")
    a.text(len(order) - 1, 0.80, "orange =\nplanted edge", fontsize=6.0, color=DISEASE, ha="right", va="top")
    a.set_xlim(-1, len(order))
    a.set_ylim(0, 1.05)
    clean(a, frame=False)
    a.spines["bottom"].set_visible(True)
    mid2 = top2 - 3.3 - PIC / 2
    for i in range(2):
        f.arrow((XS[i] + BW, mid2), (XS[i + 1], mid2))

    # row 1 -> row 2: from the matrix down and back to the planting
    gy = (y1 + top2) / 2
    f.ax.plot([XS[2] + BW / 2, XS[2] + BW / 2, XS[0] + BW / 2], [y1, gy, gy], color=INK, lw=0.9, zorder=4,
              solid_joinstyle="miter")
    f.arrow((XS[0] + BW / 2, gy), (XS[0] + BW / 2, top2))
    f.check()
    bottom = y2 - 0.4
    bb = Bbox([[0, bottom * FIGW / W], [FIGW, H * FIGW / W]])
    for ext, dpi in (("png", 600), ("pdf", 300)):
        f.fig.savefig(FIG / f"fig16_data_flow.{ext}", dpi=dpi, facecolor="white", bbox_inches=bb)
    print("written", FIG / "fig16_data_flow.png", f"{FIGW} x {(H - bottom) * FIGW / W:.2f} in")


if __name__ == "__main__":
    main()
