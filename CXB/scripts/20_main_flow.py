"""Main flow (fig15): the CXB pipeline as a two-column flowchart, drawn at its printed size.

Sized for single-column paper templates such as Springer LNCS/LNNS (text width 12.2 cm =
4.8 in), so fonts are their final size (at least 6.5 pt). Panel (a) builds the benchmark in
the order the code runs it: data, connectome, disease, then scanner on the planted scans,
then the per-fold shortcut. Panel (b) trains, explains and scores. Options are shown as chips (one is used
per run); colour marks the disease (orange) and the scanner (blue) everywhere.
Vector PDF with embedded fonts, PNG at 600 dpi. The thumbnail is one real control's matrix
from the cache of scripts/17_graph_figures.py. Prints any text that runs out of its box.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
CACHE = ROOT / "results" / "final" / "graph_figures" / "cache.npz"

INK, MUTED, LINE, PANEL = "#1f1f1e", "#4f4e48", "#9a988f", "#f3f2ee"
DISEASE, DISEASE_BG = "#c4501b", "#fcebe2"
SCANNER, SCANNER_BG = "#2565b8", "#e5eefa"
DARK, CHIP = "#1d2433", "#ffffff"
plt.rcParams.update({"font.family": "Arial", "font.size": 6.8, "text.color": INK, "mathtext.fontset": "custom",
                     "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "pdf.fonttype": 42})
FIGW = 4.8                                # inches = LNCS/LNNS text width
W, H = 100.0, 118.0                       # drawing units (1 unit = 0.048 in in both directions)
TS, LS, AS = 7.4, 6.8, 6.5                # title, line and arrow-label sizes (pt)
STEP = 2.55                               # line step
PL, PR, PW = 0.0, 51.6, 48.4              # panels: left x, right x, width
PAD = 1.3                                 # box inset inside a panel
GAP = 6.4                                 # vertical gap between boxes (holds the arrow label)


class Flow:
    def __init__(self):
        self.fig = plt.figure(figsize=(FIGW, FIGW * H / W))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(0, H)
        self.ax.axis("off")
        self.checks = []

    # ---- primitives
    def text(self, x, y, s, size=LS, color=MUTED, weight="normal", ha="left", va="top", box=None, **kw):
        t = self.ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=6,
                         linespacing=1.1, **kw)
        if box:
            self.checks.append((t, box, s))

    def rbox(self, x, y, w, h, fc, ec, lw=0.6, ls="-", r=0.9, z=2):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                                         lw=lw, ls=ls, zorder=z))

    def badge(self, x, y, n):
        self.ax.add_patch(Circle((x, y), 1.75, fc=DARK, ec="white", lw=0.6, zorder=7))
        self.ax.text(x, y - 0.05, str(n), fontsize=6.5, color="white", fontweight="bold", ha="center",
                     va="center", zorder=8)

    def arrow(self, p, q, lw=0.8, color=INK):
        self.ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6.5, color=color, lw=lw,
                                          shrinkA=0, shrinkB=0, zorder=4))

    def down(self, x, y0, label=None):
        self.arrow((x, y0), (x, y0 - GAP))
        if label:
            self.text(x + 1.4, y0 - 0.55, label, size=AS, color=INK, style="italic")
        return y0 - GAP

    # ---- composites
    def panel(self, x, top, bottom, letter, title):
        self.rbox(x, bottom, PW, top - bottom, PANEL, "none", r=1.4, z=0)
        self.text(x + 1.4, top - 1.0, letter, size=8.5, color=INK, weight="bold")
        self.text(x + 4.6, top - 1.0, title, size=8.0, color=INK, weight="bold")

    def node(self, x, top, w, title, items=(), n=None, fc="#ffffff", ec=LINE, tc=INK, lc=MUTED, lw=0.6,
             ls="-", center=True, title_size=TS):
        """items: text lines, or ('chips', [(label, emphasised), ...]) rows."""
        h = 4.2 + sum(3.4 if isinstance(it, tuple) else STEP for it in items)
        y = top - h
        self.rbox(x, y, w, h, fc, ec, lw=lw, ls=ls)
        b = (x, y, w, h)
        cx, ha = (x + w / 2, "center") if center else (x + (3.4 if n is not None else 1.4), "left")
        self.text(cx, top - 0.9, title, size=title_size, color=tc, weight="bold", ha=ha, box=b)
        yy = top - 3.55
        for it in items:
            if isinstance(it, tuple):
                self.chips(x + 1.2, yy + 0.2, w - 2.4, it[1], b)
                yy -= 3.4
            else:
                self.text(cx, yy, it, size=LS, color=lc, ha=ha, box=b)
                yy -= STEP
        if n is not None:
            self.badge(x + 0.4, top - 0.4, n)
        return y

    def chips(self, x, top, w, labels, box):
        g = 0.9
        total = sum(len(lab) + 3 for lab, _ in labels)
        cx = x
        for lab, emph in labels:
            cw = (w - g * (len(labels) - 1)) * (len(lab) + 3) / total
            self.rbox(cx, top - 2.75, cw, 2.75, CHIP, INK if emph else LINE, lw=0.9 if emph else 0.5, r=0.7, z=3)
            self.text(cx + cw / 2, top - 1.37, lab, size=AS, color=INK, ha="center", va="center",
                      weight="bold" if emph else "normal", box=(cx, top - 2.75, cw, 2.75))
            cx += cw + g

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
    x, w = PL + PAD, PW - 2 * PAD
    cx = x + w / 2
    y = H - 5.4
    b = f.node(x, y, w, "ABIDE I resting-state fMRI", ["PCP · CPAC · band-pass · no GSR",
                                                     "1,112 → 848 subjects (QC, ≥ 100 vols)",
                                                     "393 autism · 455 controls · 19 sites"])
    y = f.down(cx, b, "AAL-116, Pearson r,\nFisher z")
    b = f.node(x, y, w, "Connectome $x_i$ (116 × 116)", ["one brain graph per subject:",
                                                         "6,670 edges = correlations"], center=False)
    th = (y - b) - 1.2
    a = f.fig.add_axes([(x + w - th - 1.0) / W, (b + 0.6) / H, th / W, th / H])
    a.imshow(example, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    a.set_xticks([])
    a.set_yticks([])
    for s in a.spines.values():
        s.set_linewidth(0.4)
        s.set_edgecolor(LINE)
    y = f.down(cx, b, "343 controls; half per\nsite called “patient”")
    b = f.node(x, y, w, "Disease: the known answer", [
        "10 regions, ranked on a held-out set;",
        "shared signal in fake patients only,",
        "×1/×2/×4 real effect ($\\ell$) → 45 true edges"], fc=DISEASE_BG, ec=DISEASE, tc=DISEASE, lw=0.8)
    y = f.down(cx, b, "then, on the\nplanted scans")
    b = f.node(x, y, w, "Scanner: the known trap", [
        "10 other regions where sites differ most,",
        "sized from real hospital differences;",
        "scanner-B copy of each scan → 45 edges"], fc=SCANNER_BG, ec=SCANNER, tc=SCANNER, lw=0.8)
    y = f.down(cx, b, "in each of 10 leave-\nsite-out folds")
    b = f.node(x, y, w, "Shortcut $\\rho$: scanner linked to label", [
        "train + val.: scanner B for $\\rho$ of patients",
        "and $1-\\rho$ of controls ($\\rho$ = 0.5 → 0.9);",
        "unseen test site: always 50/50"], ec=INK, lw=0.8)
    return b, y


def real_data_box(f, bottom):
    """Dashed box at the foot of panel (a), its bottom level with the end of panel (b)."""
    x, w = PL + PAD, PW - 2 * PAD
    h = 4.2 + 2 * STEP
    f.node(x, bottom + h, w, "Real-data checks (no planting)", [
        "hospital vs autism signal · scanner map",
        "from training data · real autism labels"], ls=(0, (3, 2)), lc=MUTED)


def right_column(f):
    x, w = PR + PAD, PW - 2 * PAD
    cx = x + w / 2
    top = H - 5.4
    b = f.node(x, top, w, "Fix (one per run)", [
        ("chips", [("plain", False), ("ComBat ×4", False), ("CF-Site", True), ("oracle", False)]),
        "ComBat: EB yes/no × diagnosis kept yes/no",
        "CF-Site: train on $x_i$ and $\\tilde{x}_i$ (other scanner);",
        "oracle: $\\tilde{x}_i$ = the true scanner-B scan"])
    entry_y = top - 1.6
    y = f.down(cx, b, "train $f$: fake patients\nvs fake controls")
    # model container: four alternatives
    models = [("Log. regression", ["6,670 edges, L2", "expl.: |coef.|"]),
              ("MLP", ["6,670 → 64 → 2", "ReLU, dropout 0.5"]),
              ("GCN (BrainGB)", ["top-20% graph", "2 × GraphConv 16"]),
              ("BrainNetCNN", ["2 × E2E 32 → E2N 64", "N2G 256 → FC → 2"])]
    sub_h = 4.2 + 2 * STEP
    cont_h = 4.0 + 2 * sub_h + 1.0 + 2 * STEP + 1.2
    f.rbox(x, y - cont_h, w, cont_h, "#ffffff", INK, lw=0.8)
    f.text(cx, y - 0.9, "Model $f$ (one of four)", size=TS, color=INK, weight="bold", ha="center",
           box=(x, y - cont_h, w, cont_h))
    mw, mg = (w - 2.4 - 1.2) / 2, 1.2
    for k, (t, ls) in enumerate(models):
        mx = x + 1.2 + (k % 2) * (mw + mg)
        mt = y - 3.6 - (k // 2) * (sub_h + 1.0)
        f.node(mx, mt, mw, t, ls, ec=LINE, lw=0.5, title_size=7.0)
    ny = y - 3.6 - 2 * sub_h - 1.0 - 0.4
    f.text(cx, ny, "neural: Adam, lr $10^{-3}$, 150 epochs;", size=AS, color=MUTED, ha="center",
           box=(x, y - cont_h, w, cont_h))
    f.text(cx, ny - STEP, "BrainGNN and BNT fail the sanity test", size=AS, color=MUTED, ha="center",
           box=(x, y - cont_h, w, cont_h))
    y = f.down(cx, y - cont_h, "one score\nper edge")
    b = f.node(x, y, w, "Group explanation $S$", ["saliency (main) · IG · input mask,",
                                                  "averaged over the test site"])
    y = f.down(cx, b, "rank edges by $S$,\nscore the top 45")
    b = f.node(x, y, w, "Scores", ["     disease AUPRC (chance 0.008)", "     scanner share of the top 45",
                                   "     accuracy on the unseen test site"], center=False)
    for k, col in enumerate((DISEASE, SCANNER, INK)):
        f.ax.add_patch(Rectangle((x + 1.5, y - 5.15 - k * STEP), 1.6, 1.6, fc=col, ec="none", zorder=5))
    y = f.down(cx, b, "per split: 3–6\nsplits × 3 seeds")
    b = f.node(x, y, w, "Gain or tie", ["bootstrap CI · sign test · paired t · Holm",
                                        "gain: CI excludes 0 and Holm p < 0.05"],
               fc=DARK, ec=DARK, tc="#f7f6f2", lc="#d5dae3")
    return entry_y, b


def main():
    c = np.load(CACHE)
    f = Flow()
    # panels first (drawn underneath), sized after a dry run of the columns
    eb, et = left_column(f, c["example"])
    entry_y, rb = right_column(f)
    real_data_box(f, rb)
    lb = rb
    bottom = rb - 1.3
    f.panel(PL, H, bottom, "a", "Benchmark construction")
    f.panel(PR, H, bottom, "b", "Training and evaluation")
    # connector: from step 5 (end of a) to step 6 (start of b), through the gap between the panels
    mid = eb + (et - eb) / 2
    gx = PL + PW + (PR - PL - PW) / 2
    f.ax.plot([PL + PW - PAD, gx, gx], [mid, mid, entry_y], color=INK, lw=0.9, zorder=4, solid_joinstyle="miter")
    f.arrow((gx, entry_y), (PR + PAD, entry_y), lw=0.9)
    f.check()
    bb = Bbox([[0, bottom * FIGW / W], [FIGW, H * FIGW / W]])
    for ext, dpi in (("png", 600), ("pdf", 300)):
        f.fig.savefig(FIG / f"fig15_main_flow.{ext}", dpi=dpi, facecolor="white", bbox_inches=bb)
    print("written", FIG / "fig15_main_flow.png", f"{FIGW:.1f} x {(H - bottom) * FIGW / W:.2f} in;",
          f"left end {lb:.1f}, right end {rb:.1f}")


if __name__ == "__main__":
    main()
