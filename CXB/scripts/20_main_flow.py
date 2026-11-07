"""Main flow (fig15): the CXB pipeline as a two-column flowchart, drawn at its printed size.

Sized for Springer LNNS/LNCS (text width 12.2 cm = 4.8 in), so fonts are their final size
(at least 6.5 pt). (a) benchmark construction, top to bottom, in the order the code runs
it: data, connectome, disease, then scanner on the planted scans, then the per-fold
shortcut. (b) training and evaluation: fixes, models, explanation, scores, statistics.
Boxes are the data or method at each stage; the text beside each arrow is the operation.
Vector PDF with embedded fonts, PNG at 600 dpi. The thumbnail is one real control's matrix
from the cache of scripts/17_graph_figures.py. Prints any text that runs out of its box.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
CACHE = ROOT / "results" / "final" / "graph_figures" / "cache.npz"

INK, MUTED, LINE = "#1f1f1e", "#4f4e48", "#8d8b84"
DISEASE, DISEASE_BG = "#c4501b", "#fcebe2"
SCANNER, SCANNER_BG = "#2565b8", "#e5eefa"
DARK = "#1d2433"
plt.rcParams.update({"font.family": "Arial", "font.size": 6.8, "text.color": INK, "mathtext.fontset": "custom",
                     "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "pdf.fonttype": 42})
FIGW = 4.8                               # inches = LNCS text width
W, H = 100.0, 112.0
TS, LS, AS = 7.4, 6.8, 6.5               # title, line and arrow-label sizes (pt)
STEP = 2.55                              # line step (units; 1 unit = 0.048 in)
L0, R0, CW = 0.6, 51.9, 47.5             # left column x, right column x, column width
GAP = 6.6


class Flow:
    def __init__(self):
        self.fig = plt.figure(figsize=(FIGW, FIGW * H / W))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(0, H)
        self.ax.axis("off")
        self.checks = []

    def text(self, x, y, s, size=LS, color=MUTED, weight="normal", ha="left", box=None, **kw):
        t = self.ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va="top", zorder=5,
                         linespacing=1.1, **kw)
        if box:
            self.checks.append((t, box, s))

    def node(self, x, top, w, title, lines=(), fc="#ffffff", ec=LINE, tc=INK, lc=MUTED, lw=0.6, center=True,
             indent=0.0, ls="-"):
        h = 4.3 + len(lines) * STEP
        y = top - h
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9", fc=fc, ec=ec, lw=lw,
                                         ls=ls, zorder=2))
        cx, ha = (x + w / 2, "center") if center else (x + 1.3, "left")
        b = (x, y, w, h)
        self.text(cx, top - 0.9, title, size=TS, color=tc, weight="bold", ha=ha, box=b)
        for k, s in enumerate(lines):
            self.text(cx + indent, top - 3.6 - k * STEP, s, size=LS, color=lc, ha=ha, box=b)
        return y

    def arrow(self, p, q, lw=0.8):
        self.ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6.5, color=INK, lw=lw, shrinkA=0,
                                          shrinkB=0, zorder=3))

    def down(self, x, y0, label=None):
        self.arrow((x, y0), (x, y0 - GAP))
        if label:
            self.text(x + 1.4, y0 - 0.6, label, size=AS, color=INK, style="italic")
        return y0 - GAP

    def check(self):
        self.fig.canvas.draw()
        r = self.fig.canvas.get_renderer()
        bad = 0
        for t, (x, y, w, h), s in self.checks:
            bb = t.get_window_extent(r)
            (x0, y0), (x1, y1) = self.ax.transData.transform([(x, y), (x + w, y + h)])
            if bb.x0 < x0 + 0.5 or bb.x1 > x1 - 1 or bb.y0 < y0 + 0.5 or bb.y1 > y1:
                print("OVERFLOW:", s)
                bad += 1
        print("text overflow:", bad)


def left_column(f, example):
    cx = L0 + CW / 2
    f.text(L0, H - 0.6, "a", size=8.5, color=INK, weight="bold")
    f.text(L0 + 3.4, H - 0.6, "Benchmark construction", size=8.0, color=INK, weight="bold")
    y = H - 5.0
    b = f.node(L0, y, CW, "ABIDE I resting-state fMRI", ["PCP · CPAC · band-pass · no GSR",
                                                       "1,112 → 848 subjects (QC, ≥ 100 vols)",
                                                       "393 autism · 455 controls · 19 sites"])
    y = f.down(cx, b, "AAL-116, Pearson r,\nFisher z")
    b = f.node(L0, y, CW, "Connectome $x_i$ (116 × 116)", ["one brain graph per subject:",
                                                          "6,670 edges = correlations"], center=False)
    th = (y - b) - 1.2
    a = f.fig.add_axes([(L0 + CW - th - 1.0) / W, (b + 0.6) / H, th / W, th / H])
    a.imshow(example, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    a.set_xticks([])
    a.set_yticks([])
    for s in a.spines.values():
        s.set_linewidth(0.4)
        s.set_edgecolor(LINE)
    y = f.down(cx, b, "343 controls; half per\nsite called “patient”")
    b = f.node(L0, y, CW, "1 · Disease: the known answer", [
        "10 regions, ranked on a held-out set;",
        "shared signal in fake patients only,",
        "×1/×2/×4 real effect ($\\ell$) → 45 true edges"], fc=DISEASE_BG, ec=DISEASE, tc=DISEASE)
    y = f.down(cx, b, "then, on the\nplanted scans")
    b = f.node(L0, y, CW, "2 · Scanner: the known trap", [
        "10 other regions where sites differ most,",
        "sized from real hospital differences;",
        "scanner-B copy of each scan → 45 edges"], fc=SCANNER_BG, ec=SCANNER, tc=SCANNER)
    y = f.down(cx, b, "in each of 10 leave-\nsite-out folds")
    b = f.node(L0, y, CW, "3 · Shortcut $\\rho$", [
        "train + val.: scanner B for $\\rho$ of patients",
        "and $1-\\rho$ of controls ($\\rho$ = 0.5 → 0.9);",
        "unseen test site: always 50/50"])
    last_bottom, last_top = b, y
    f.node(L0, b - 2.4, CW, "Real-data checks (no planting)", [
        "hospital vs autism signal · scanner map",
        "from training data · real autism labels"], ls=(0, (3, 2)))
    return last_bottom, last_top


def right_column(f):
    cx = R0 + CW / 2
    f.text(R0, H - 0.6, "b", size=8.5, color=INK, weight="bold")
    f.text(R0 + 3.4, H - 0.6, "Training and evaluation", size=8.0, color=INK, weight="bold")
    top = H - 5.0
    b = f.node(R0, top, CW, "Fix (one per run)", ["plain │ ComBat ×4 (EB × diagnosis kept)",
                                                 "CF-Site: train on $x_i$ and $\\tilde{x}_i$ (other",
                                                 "scanner); oracle: true scanner-B scan"])
    first_mid = top - (top - b) / 2
    y = f.down(cx, b, "train $f$: fake patients\nvs fake controls")
    mw, mg = (CW - 1.4) / 2, 1.4
    models = [("Log. regression", ["6,670 edges, L2", "expl.: |coef.|"]),
              ("MLP", ["6,670 → 64 → 2", "ReLU, dropout 0.5"]),
              ("GCN (BrainGB)", ["top-20% graph", "2 × GraphConv 16"]),
              ("BrainNetCNN", ["E2E 32 → 32 → E2N 64", "N2G 256 → FC → 2"])]
    for k, (t, ls) in enumerate(models):
        x = R0 + (k % 2) * (mw + mg)
        if k == 2:
            y = row - 1.2
        bk = f.node(x, y, mw, t, ls, ec=INK)
        if k == 1:
            row = bk
    f.text(cx, bk - 0.8, "neural: Adam, lr $10^{-3}$, 150 epochs;", size=AS, color=MUTED, ha="center")
    f.text(cx, bk - 3.3, "BrainGNN and BNT fail the sanity test", size=AS, color=MUTED, ha="center")
    y = f.down(cx, bk - 6.0, "one score\nper edge")
    b = f.node(R0, y, CW, "Group explanation $S$", ["saliency (main) · IG · input mask,",
                                                    "averaged over the test site"])
    y = f.down(cx, b, "rank edges by $S$,\nscore the top 45")
    b = f.node(R0, y, CW, "Scores", ["disease AUPRC (chance 0.008)", "scanner share of the top 45",
                                     "accuracy on the unseen test site"], center=False, indent=3.0)
    for k, col in enumerate((DISEASE, SCANNER, INK)):
        f.ax.add_patch(Rectangle((R0 + 1.4, y - 5.25 - k * STEP), 1.6, 1.6, fc=col, ec="none", zorder=4))
    y = f.down(cx, b, "per split: 3–6\nsplits × 3 seeds")
    b = f.node(R0, y, CW, "Gain or tie", ["bootstrap CI · sign test · paired t · Holm",
                                          "gain: CI excludes 0 and Holm p < 0.05"],
               fc=DARK, ec=DARK, tc="#f7f6f2", lc="#d5dae3")
    return first_mid, b


def main():
    c = np.load(CACHE)
    f = Flow()
    lb, lt = left_column(f, c["example"])
    first_mid, rb = right_column(f)
    mid = lb + (lt - lb) / 2                          # connector: end of (a) to the start of (b)
    gx = (L0 + CW + R0) / 2
    f.ax.plot([L0 + CW, gx, gx], [mid, mid, first_mid], color=INK, lw=0.8, zorder=3, solid_joinstyle="miter")
    f.arrow((gx, first_mid), (R0, first_mid))
    f.check()
    bottom = min(lb, rb) - 0.5
    bb = Bbox([[0, bottom * FIGW / W], [FIGW, H * FIGW / W]])
    for ext, dpi in (("png", 600), ("pdf", 300)):
        f.fig.savefig(FIG / f"fig15_main_flow.{ext}", dpi=dpi, facecolor="white", bbox_inches=bb)
    print("written", FIG / "fig15_main_flow.png", f"{FIGW:.1f} x {(H - bottom) * FIGW / W:.2f} in;",
          f"left bottom {lb:.1f}, right bottom {rb:.1f}")


if __name__ == "__main__":
    main()
