"""Smoke test: run the official BrainIB (V2 / SGSIB) code on our ABIDE AAL matrices.

Uses BrainIB's own model, subgraph generator, train() and test() unchanged; only the
data loading is replaced. Real ASD vs control labels, one BrainOOD-style held-out site.
Reports the test accuracy at the LAST epoch (the official script reports the best test
accuracy over epochs, which selects on the test set).

    .venv\\Scripts\\python.exe scripts\\baselines\\brainib_smoke.py --epochs 20 --test-site PITT
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "external" / "brain-and-Information-Bottleneck" / "BrainIB_V2"
sys.path.insert(0, str(REPO))
sys.path.insert(1, str(Path(__file__).resolve().parent))

import compat  # noqa: E402  (torch.symeig stand-in, shared with brainib_v1_check.py)

compat.install()  # must happen before BrainIB's utils are used

from torch_geometric.data import Data  # noqa: E402
from SGSIB.GNN import GNN  # noqa: E402
from SGSIB.sub_graph_generator import MLP_subgraph  # noqa: E402
import SGSIB.utils as sgsib_utils  # noqa: E402
from SGSIB.utils import test, train  # noqa: E402

_orig_calculate_mi = sgsib_utils.calculate_MI


def _calculate_mi_with_diagnostic(x, y, s_x, s_y):
    """BrainIB's calculate_MI, unchanged, but prints the embedding spread if it fails."""
    try:
        return _orig_calculate_mi(x, y, s_x, s_y)
    except (FloatingPointError, RuntimeError):
        print(f"\n[diagnostic] batch {x.shape[0]} | embedding std across subjects: "
              f"graph {x.detach().std(0).mean():.2e}, subgraph {y.detach().std(0).mean():.2e} | "
              f"sigma^2: {float(s_x):.2e}, {float(s_y):.2e} | "
              f"any NaN in embeddings: {bool(torch.isnan(x).any() or torch.isnan(y).any())}")
        raise


sgsib_utils.calculate_MI = _calculate_mi_with_diagnostic  # train() looks it up at call time


def to_graph(fc_z, label, keep=0.2):
    """BrainIB input (per its README): node features = Pearson r rows; adjacency = the top
    `keep` fraction of |r| set to 1 (binarised). Edge weights must be positive: BrainIB's
    GCNConv normalises by sqrt(weighted degree), which is NaN for signed weights."""
    r = np.tanh(fc_z)  # back from Fisher z to correlation
    n = r.shape[0]
    iu = np.triu_indices(n, 1)
    vals = np.abs(r[iu])
    thr = np.quantile(vals, 1 - keep)
    sel = vals >= thr
    src, dst = iu[0][sel], iu[1][sel]
    edge_index = np.concatenate([np.stack([src, dst]), np.stack([dst, src])], axis=1)
    edge_attr = np.ones((edge_index.shape[1], 1))  # binarised adjacency
    x = r.copy()
    np.fill_diagonal(x, 1.0)
    return Data(x=torch.tensor(x, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
                y=torch.tensor([label], dtype=torch.long))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--test-site", default="PITT")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--their-data", action="store_true",
                    help="reproduction check: BrainIB's own md_AAL_0.4.mat, loaded by its own "
                         "read_dataset(), split by its own 10-fold separate_data()")
    ap.add_argument("--fold", type=int, default=0, help="fold 0-9 for --their-data")
    args = ap.parse_args()
    # Attributes BrainIB's train()/test() read from args.
    args.iters_per_epoch, args.mi_weight, args.pos_weight = 1, 0.001, 0.001

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.their_data:
        import os
        from data.create_dataset import read_dataset  # BrainIB's loader; reads ./data/md_AAL_0.4.mat
        from SGSIB.utils import separate_data
        mat = REPO / "data" / "md_AAL_0.4.mat"
        if not mat.exists():
            sys.exit(f"Missing {mat}\nDownload md_AAL_0.4.mat from the Google Drive link in "
                     f"{REPO / 'README.md'} and put it there.")
        cwd = os.getcwd()
        os.chdir(REPO)
        try:
            graphs = read_dataset()
        finally:
            os.chdir(cwd)
        train_set, test_set = separate_data(graphs, args.seed, args.fold)
        print(f"THEIR DATA | subjects {len(graphs)} | fold {args.fold}: train {len(train_set)}, "
              f"test {len(test_set)} | test class share "
              f"{np.mean([int(g.y) for g in test_set]):.3f} (label 1)")
    else:
        d = np.load(ROOT / "data" / "processed" / "abide_rois_aal.npz", allow_pickle=True)
        keep = d["n_dead_rois"] == 0
        fc, dx, site = d["fc"][keep], d["dx"][keep], d["site"][keep]
        y = (dx == 1).astype(int)  # 1 = ASD, 0 = control

        graphs = [to_graph(f, l) for f, l in zip(fc, y)]
        test_idx = np.where(site == args.test_site)[0]
        train_idx = np.where(site != args.test_site)[0]
        train_set = [graphs[i] for i in np.random.permutation(train_idx)]
        test_set = [graphs[i] for i in test_idx]
        print(f"subjects {len(graphs)} | train {len(train_set)} | test site {args.test_site}: {len(test_set)}"
              f" | test control share {np.mean(dx[test_idx] == 2):.3f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNN(num_of_features=116, device=device).to(device)
    sg = MLP_subgraph(node_features_num=116, edge_features_num=1, device=device)
    opt = torch.optim.Adam([{"params": model.parameters(), "lr": 0.0005},
                            {"params": sg.parameters(), "lr": 0.001}])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

    # Check: does BrainIB's subgraph generator (the explanation mask) receive any training
    # signal? Its mask is stored as graph.attr, but the GNN only reads graph.edge_attr.
    sg_start = [p.detach().clone() for p in sg.parameters()]
    gnn_start = [p.detach().clone() for p in model.parameters()]

    t0, best = time.time(), 0.0
    for epoch in range(1, args.epochs + 1):
        loss, mi = train(args, model, train_set, opt, epoch, sg, device)
        acc_tr, acc_te, _ = test(args, model, train_set, test_set, sg, device)
        best = max(best, acc_te)
        sched.step()
        print(f"epoch {epoch:3d} loss {loss:.3f} | train {acc_tr:.3f} | test {acc_te:.3f}")
    print(f"\nlast-epoch test accuracy {acc_te:.3f} (official script would report best-over-epochs {best:.3f})")
    sg_change = max((p.detach() - q).abs().max().item() for p, q in zip(sg.parameters(), sg_start))
    gnn_change = max((p.detach() - q).abs().max().item() for p, q in zip(model.parameters(), gnn_start))
    print(f"largest weight change after training: GNN {gnn_change:.2e} | subgraph generator {sg_change:.2e}"
          f"  ({'mask was NOT trained' if sg_change == 0 else 'mask was trained'})")
    print(f"time {time.time() - t0:.0f}s, peak GPU {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
          if torch.cuda.is_available() else f"time {time.time() - t0:.0f}s (CPU)")


if __name__ == "__main__":
    main()
