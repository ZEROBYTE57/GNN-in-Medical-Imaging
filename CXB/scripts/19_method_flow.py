"""Full-page method flowchart (fig14): every step of CXB, from raw data to the decision rule.

Six bands, top to bottom: data, benchmark construction, protocol and fixes, model
architectures, explanation and scoring, experiments. Every number is from the code
(src/cxb/*.py) and the paper draft. Also saves three landscape crops for slides
(fig14_part1..3). The only data used is one control's matrix for the thumbnail, from the
cache of scripts/17_graph_figures.py. The script prints any text that runs out of its box.

    .venv\\Scripts\\python.exe scripts\\19_method_flow.py
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

INK, MUTED, LINE, PANEL, PREP = "#1f1f1e", "#5f5e58", "#8d8b84", "#f4f3ef", "#e9e7e1"
DISEASE, DISEASE_BG = "#d4581f", "#fcebe2"
SCANNER, SCANNER_BG = "#2a6fc9", "#e5eefa"
plt.rcParams.update({"font.family": "Arial", "font.size": 6, "text.color": INK, "mathtext.fontset": "custom",
                     "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
                     "pdf.fonttype": 42})
W, H = 100.0, 140.0
FIGW = 6.7
UNIT = FIGW / W                      # inches per drawing unit
TS, LS = 6.4, 5.8                    # title and line font sizes (pt)
LSTEP = 1.72                         # line spacing in units


class Page:
    def __init__(self):
        self.fig = plt.figure(figsize=(FIGW, H * UNIT))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(0, H)
        self.ax.axis("off")
        self.checks = []

    # ---- primitives
    def rect(self, x, y, w, h, fc, ec="none", lw=0.6, ls="-", r=0.6, z=1):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                                         lw=lw, ls=ls, zorder=z))

    def text(self, x, y, s, size=LS, color=INK, weight="normal", ha="left", va="top", box=None, **kw):
        t = self.ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=5, **kw)
        if box is not None:
            self.checks.append((t, box, s))
        return t

    def arrow(self, p, q, color=INK, lw=0.7, ls="-"):
        self.ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=5.5, color=color, lw=lw, ls=ls,
                                          shrinkA=0, shrinkB=0, zorder=4))

    def poly(self, pts, color=INK, lw=0.7):
        xs, ys = zip(*pts)
        self.ax.plot(xs, ys, color=color, lw=lw, zorder=4, solid_capstyle="butt")

    # ---- composites
    def box(self, x, y, w, h, title=None, lines=(), fc="#ffffff", ec=LINE, tc=INK, lw=0.6, size=LS, ls="-"):
        self.rect(x, y, w, h, fc, ec, lw=lw, ls=ls, z=2)
        b = (x, y, w, h)
        ty = y + h - 0.7
        if title:
            self.text(x + 0.8, ty, title, size=TS, color=tc, weight="bold", box=b)
            ty -= 1.95
        for k, s in enumerate(lines):
            self.text(x + 0.8, ty - k * LSTEP, s, size=size, color=MUTED, box=b)

    def layer(self, x, y, w, s, fc="#ffffff", ec=INK):
        self.rect(x, y, w, 2.5, fc, ec, lw=0.5, r=0.4, z=2)
        self.text(x + w / 2, y + 1.25, s, size=5.5, ha="center", va="center", box=(x, y, w, 2.5))

    def band(self, y0, y1, label):
        self.rect(0, y0, 3.6, y1 - y0, INK, r=0.5)
        self.ax.text(1.8, (y0 + y1) / 2, label, rotation=90, ha="center", va="center", fontsize=6.6,
                     fontweight="bold", color="#ffffff", zorder=5)
        self.rect(4.2, y0, W - 4.2, y1 - y0, PANEL, r=0.6, z=0)

    def check(self):
        self.fig.canvas.draw()
        rend = self.fig.canvas.get_renderer()
        bad = 0
        for t, (x, y, w, h), s in self.checks:
            bb = t.get_window_extent(rend)
            (x0, y0), (x1, y1) = self.ax.transData.transform([(x, y), (x + w, y + h)])
            if bb.x0 < x0 - 0.5 or bb.x1 > x1 - 1.5 or bb.y0 < y0 + 0.5 or bb.y1 > y1 + 0.5:
                print("OVERFLOW:", repr(s))
                bad += 1
        print("text overflow:", bad)

    def save(self, name, crops=()):
        for ext in ("png", "pdf"):
            self.fig.savefig(FIG / f"{name}.{ext}", dpi=300, facecolor="white")
        for k, (ya, yb) in enumerate(crops, 1):          # landscape crops for slides (drawing units)
            bb = Bbox([[0, ya * UNIT], [FIGW, yb * UNIT]])
            self.fig.savefig(FIG / f"{name}_part{k}.png", dpi=300, facecolor="white", bbox_inches=bb)
        print("written", FIG / f"{name}.png")


def data_band(p, example):
    p.band(127.0, 140.0, "1  Data")
    y, h = 128.4, 10.4
    boxes = [(5.2, 16.2, "ABIDE I (PCP)", ["1,112 subjects", "CPAC pipeline,", "band-pass, no GSR"]),
             (23.4, 16.2, "Quality check", ["time series present,", "no rater marked fail", "→ 873 subjects"]),
             (41.6, 16.2, "Length filter", ["≥ 100 time points", "→ 848 subjects:", "393 autism, 455 controls,",
                                            "19 sites"]),
             (59.8, 16.8, "Connectome", ["AAL atlas, 116 regions", "Pearson r per region pair", "Fisher z, zero diagonal"])]
    for x, w, t, ls in boxes:
        p.box(x, y, w, h, t, ls)
    for (x, w, *_), (nx, *_) in zip(boxes[:-1], boxes[1:]):
        p.arrow((x + w, y + h / 2), (nx, y + h / 2))
    a = p.fig.add_axes([78.8 / W, 129.0 / H, 8.4 / W, 8.4 * UNIT / (H * UNIT) * 1.0])
    a.imshow(example, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    a.set_xticks([])
    a.set_yticks([])
    for s in a.spines.values():
        s.set_linewidth(0.5)
        s.set_edgecolor(LINE)
    p.arrow((76.6, y + h / 2), (78.6, y + h / 2))
    p.text(88.4, 138.0, "$x_i$: 116 × 116", size=TS, weight="bold")
    p.text(88.4, 135.9, "brain graph:", color=MUTED)
    p.text(88.4, 134.2, "nodes = regions,", color=MUTED)
    p.text(88.4, 132.5, "edges = 6,670", color=MUTED)
    p.text(88.4, 130.8, "region pairs", color=MUTED)


def benchmark_band(p):
    p.band(84.0, 125.6, "2  Benchmark construction")
    p.text(52.4, 125.0, "Input: the 848 connectomes (455 controls, 393 autism)", ha="center", weight="bold", size=TS)
    lanes = [(5.2, 46.4, DISEASE, DISEASE_BG, "Disease effect: the known answer", [
        ("Choose disease regions", ["ranking set: site-stratified 20% of controls + all autism subjects",
                                    "Cohen's d per edge → top 10 regions by mean |d| → truth: 45 edges"]),
        ("Fake labels", ["planting pool: the other 343 controls with no empty region",
                         "in each site, a random half labelled 'patient' (y = 1), per split"]),
        ("Plant the effect (fake patients only)", ["shared random signal, orthogonal to the subject's own signals,",
                                                   "added to the 10 regions' signals; connectivity recomputed"]),
        ("Calibrate the strength $\\ell$", ["mean shift of the 45 edges = $\\ell$ × median real autism effect",
                                           "$\\ell$ $\\in$ {1, 2, 4}; edges to other regions: 'don't care'"])]),
             (53.4, 46.0, SCANNER, SCANNER_BG, "Scanner effect: the known trap", [
        ("Choose scanner regions", ["$\\eta^2$ between sites per edge (controls; sites with ≥ 20 controls)",
                                    "top 10 regions outside the disease set → 45 scanner edges"]),
        ("Size from real hospitals", ["median over site pairs of the mean |difference of site means|",
                                      "on the scanner edges: a realistic hospital difference"]),
        ("Scanner-B version of every subject", ["main: shared signal added to the 10 regions (person-specific)",
                                                "robustness: one fixed shift per edge (ComBat's own model)"]),
        ("Two scans per subject", ["$x_i$ at scanner A and $x_i^B$ at scanner B, same label $y_i$",
                                   "the true pair is what oracle CF-Site trains with"])])]
    top, bh, gap = 120.9, 6.1, 1.4
    for x, w, col, bg, head, steps in lanes:
        p.text(x + 0.2, 123.2, head, size=TS, weight="bold", color=col)
        for k, (t, ls) in enumerate(steps):
            yb = top - bh - k * (bh + gap)
            p.box(x, yb, w, bh, f"{k + 1}  {t}", ls, fc=bg, ec=col, tc=col)
            if k:
                p.arrow((x + w / 2, yb + bh + gap), (x + w / 2, yb + bh))
    last = top - bh - 3 * (bh + gap)
    my, mh = 84.5, 6.4
    p.box(5.2, my, 94.2, mh, "Link the scanner to the label, per fold", [
        "training and validation sites: scanner B for a fraction $\\rho$ of fake patients and $1-\\rho$ of fake controls "
        "($\\rho$ = 0.5 to 0.9); the test site is always 50/50,",
        "so at test the scanner carries no information about the label   →   benchmark data $(x_i, y_i, b_i)$"],
        ec=INK, lw=0.8)
    for x, w, *_ in lanes:
        p.arrow((x + w / 2, last), (x + w / 2, my + mh))


def fixes_band(p):
    p.band(64.0, 82.6, "3  Protocol, fixes, training")
    p.box(5.2, 64.6, 23.4, 17.4, "Evaluation protocol", [
        "BrainOOD: 10 leave-site-out folds", "each fold: 1 test site, 1 validation", "site, training on the rest",
        "recovery, drift: 3 splits × 3 seeds", "fixes: 6 splits × 3 seeds", "every choice on validation data;",
        "test sites never used for a choice"], ec=INK)
    p.text(64.8, 82.0, "Fixes compared (one per run, fitted on the training subjects of each fold)", ha="center",
           weight="bold", size=TS)
    x0, w, gap = 30.4, 16.35, 1.25
    fixes = [("Plain", ["train on $x$ as it is"], LINE, 0.6),
             ("ComBat (4 versions)", ["scanner A/B as batch", "EB shrinkage: yes or no", "diagnosis kept: yes or no",
                                      "test: covariate at mean"], LINE, 0.6),
             ("CF-Site (ours)", ["$\\tilde{x}$ = $x$ at the other scanner", "(label-protected ComBat;",
                                 "EB-swap variant)", "loss: CE on $x$ and on $\\tilde{x}$"], INK, 1.1),
             ("Oracle CF-Site", ["$\\tilde{x}$ = the true $x^B$ scan", "not available in practice:",
                                 "an upper bound"], LINE, 0.6)]
    for k, (t, ls, ec, lw) in enumerate(fixes):
        p.box(x0 + k * (w + gap), 70.8, w, 9.4, t, ls, ec=ec, lw=lw)
        p.arrow((x0 + k * (w + gap) + w / 2, 70.8), (x0 + k * (w + gap) + w / 2, 70.0))
    p.box(30.4, 64.4, 69.0, 5.6, None, [
        "Training (neural models): Adam, learning rate $10^{-3}$, weight decay $10^{-3}$, batch 32, 150 epochs;",
        "weights of the epoch with the lowest validation loss after a 30-epoch warm-up;",
        "inputs standardised inside each model with the training mean and SD of every entry"], ec=INK)
    p.arrow((28.6, 73.3), (30.4, 73.3))


def models_band(p):
    p.band(38.0, 62.6, "4  Model architectures")
    p.text(52.4, 62.0, "Models $f$: the raw 116 × 116 matrix in, two class scores out (patient vs. control)",
           ha="center", weight="bold", size=TS)
    cols = [("Logistic regression", ["upper triangle: 6,670 edges", "standardise each edge", "L2 logistic regression",
                                     "C $\\in$ {0.01, 0.1, 1} on validation"], "explanation: |coefficients|"),
            ("MLP", ["upper triangle: 6,670 edges", "dropout 0.5", "linear 6,670 → 64, ReLU", "dropout 0.5",
                     "linear 64 → 2"], None),
            ("GCN (BrainGB set-up)", ["node features: rows of $x$", "graph: top 20% |r|, w = ReLU(r)",
                                      "self-loops, $D^{-1/2}WD^{-1/2}$", "GraphConv 116 → 16 + root",
                                      "GraphConv 16 → 16 + root", "concat 116 × 16 → linear 2"], None),
            ("BrainNetCNN", ["edge-to-edge 1 → 32", "edge-to-edge 32 → 32", "edge-to-node 32 → 64",
                             "node-to-graph 64 → 256", "FC 256 → 128 → 30 → 2"], "leaky ReLU 0.33, dropout 0.5")]
    x0, w, gap = 5.2, 19.4, 0.9
    for k, (t, layers, note) in enumerate(cols):
        x = x0 + k * (w + gap)
        p.rect(x, 38.6, w, 21.6, "#ffffff", LINE, lw=0.5, z=1)
        p.text(x + w / 2, 59.6, t, size=TS, weight="bold", ha="center")
        for i, s in enumerate(layers):
            yb = 55.0 - i * 3.25
            prep = i < 2 and k < 3 or (k == 2 and i == 2)
            p.layer(x + 0.9, yb, w - 1.8, s, fc=PREP if prep else "#ffffff", ec=LINE if prep else INK)
            if i:
                p.arrow((x + w / 2, yb + 3.25), (x + w / 2, yb + 2.5), lw=0.5)
        if note:
            p.text(x + w / 2, 55.0 - len(layers) * 3.25 + 2.2, note, size=5.4, color=MUTED, ha="center",
                   style="italic")
    gx = x0 + 4 * (w + gap)
    p.box(gx, 38.6, 99.4 - gx, 21.6, "Sanity test", [
        "×4 effect, no shortcut:", "saliency AUPRC ≥ 0.5", "", "pass: MLP, GCN,", "BrainNetCNN", "(LR: 0.98 at ×2)",
        "fail: BrainGNN, BNT", "(reported, not used", "further)"], ec=INK, size=5.6)


def explain_band(p):
    p.band(14.0, 36.6, "5  Explain and score")
    p.text(5.6, 36.0, "Explain (test-site subjects)", size=TS, weight="bold")
    ex = [("Saliency (main)", ["$|\\partial(z_1 - z_0)/\\partial x|$ per entry"]),
          ("Integrated Gradients", ["32 steps, zero baseline"]),
          ("Input mask", ["150 steps; size, entropy terms"])]
    for k, (t, ls) in enumerate(ex):
        yb = 29.6 - k * 5.0
        p.box(5.2, yb, 20.0, 4.5, t, ls)
        p.arrow((25.2, yb + 2.25), (27.0, 31.85 if k == 0 else (31.0 if k == 1 else 30.2)), lw=0.5)
    p.box(27.0, 29.6, 20.0, 4.5, "Edge scores", ["symmetrise: 6,670 per subject"])
    p.box(27.0, 24.6, 20.0, 4.5, "Group explanation $S$", ["mean over the test subjects"])
    p.arrow((37.0, 29.6), (37.0, 29.1))
    p.box(5.2, 14.6, 41.8, 4.7, None, [
        "Controls: shuffled training labels give each explainer's chance level;",
        "a univariate t-test on the training subjects gives a model-free reference"], ec=LINE, ls=(0, (3, 2)))

    p.text(49.8, 36.0, "Score and test", size=TS, weight="bold")
    mets = [("Disease AUPRC", ["$S$ ranks the 45 true edges", "above untouched ones;", "chance 0.008"],
             DISEASE, DISEASE_BG),
            ("Scanner share", ["share of the top 45", "edges of $S$ that are", "scanner edges"], SCANNER, SCANNER_BG),
            ("Accuracy", ["pooled over all", "test subjects"], INK, "#ffffff")]
    x0, w, gap = 49.6, 15.95, 0.95
    for k, (t, ls, ec, fc) in enumerate(mets):
        p.box(x0 + k * (w + gap), 26.6, w, 7.8, t, ls, fc=fc, ec=ec, tc=ec if ec != INK else INK)
    p.arrow((47.0, 26.85), (49.6, 30.7))
    stats = [("Aggregate", ["folds and seeds →", "one value per split"]),
             ("Statistics", ["95% bootstrap CI over", "splits; sign test;", "paired t-test; Holm"]),
             ("Decision", ["gain: CI excludes 0", "and Holm p < 0.05;", "otherwise a tie"])]
    for k, (t, ls) in enumerate(stats):
        x = x0 + k * (w + gap)
        p.box(x, 14.6, w, 9.6, t, ls, ec=INK)
        if k:
            p.arrow((x - gap, 19.4), (x, 19.4))
    p.poly([(x0 + w / 2, 26.6), (x0 + w / 2, 25.4)])
    p.poly([(x0 + 2.5 * w + 2 * gap, 26.6), (x0 + 2.5 * w + 2 * gap, 25.4), (x0 + w / 2, 25.4)])
    p.poly([(x0 + 1.5 * w + gap, 26.6), (x0 + 1.5 * w + gap, 25.4)])
    p.arrow((x0 + w / 2, 25.4), (x0 + w / 2, 24.2))


def experiments_band(p):
    p.band(0.0, 12.6, "6  Experiments")
    exps = [("Recovery (Sec. 5.1)", ["$\\ell$ = 1, 2, 4; $\\rho$ = 0.5", "all explainers;", "shuffled labels; LR"]),
            ("Drift (Sec. 5.2)", ["$\\ell$ = 2; $\\rho$ = 0.5 → 0.9", "MLP, GCN, LR,", "BrainNetCNN"]),
            ("Fixes (Sec. 5.4)", ["$\\ell$ = 2; $\\rho$ = 0.9", "plain, 4 ComBat,", "3 CF-Site, oracle"]),
            ("Robustness (Sec. 5.4)", ["fixed-shift scanner;", "second region set;", "IG; no small sites"]),
            ("Real data (5.3, 5.5)", ["no planting: hospital vs.", "autism (LR; AAL, CC200);",
                                      "scanner map from training;", "real labels; label probe"])]
    x0, gap = 5.2, 1.1
    w = (99.4 - x0 - 4 * gap) / 5
    for k, (t, ls) in enumerate(exps):
        p.box(x0 + k * (w + gap), 0.6, w, 11.4, t, ls, ec=INK if k < 4 else SCANNER, size=5.6)


def main():
    c = np.load(CACHE)
    p = Page()
    data_band(p, c["example"])
    benchmark_band(p)
    fixes_band(p)
    models_band(p)
    explain_band(p)
    experiments_band(p)
    for y_from, y_to in ((127.0, 125.6), (84.0, 82.6), (64.0, 62.6), (38.0, 36.6), (14.0, 12.6)):
        p.arrow((97.6, y_from + 0.1), (97.6, y_to - 0.1), lw=0.9)
    p.check()
    p.save("fig14_method_flow", crops=((83.7, 140.0), (37.7, 83.7), (-0.2, 37.7)))


if __name__ == "__main__":
    main()
