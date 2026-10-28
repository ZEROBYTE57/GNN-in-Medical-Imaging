"""
M3 - Population-graph builders. The heart of the dissertation.

One node = one patient. An edge j -> i means "patient i receives a message
from patient j" when the GNN aggregates neighbours (PyG convention: row 0 =
source, row 1 = target).

Three construction rules
------------------------
  "snf"         Peng et al. 2024: per-modality affinity (PET, CT) -> Similarity
                Network Fusion -> keep pairs above a threshold. Frozen before training.
  "knn"         each patient receives from its k most similar training patients.
  "centre_knn"  YOUR NOVELTY: each patient receives from k training patients,
                of which round(alpha * k) MUST come from other hospitals.
                alpha = 1 -> all neighbours cross-centre; alpha = 0 -> all same-centre.

Leakage rules (inductive setting, all modes)
--------------------------------------------
  * training patients only receive messages from training patients
  * each test patient receives from training patients only
  * no test -> test edges, and no test -> train edges
  So a test patient never influences training, and test patients never see each other.

The key diagnostic: neighbourhood MISMATCH
------------------------------------------
Under leave-one-centre-out, a test patient's neighbours are ALL from other
hospitals (its own hospital is entirely held out). If training patients mostly
receive from their OWN hospital, the GNN is trained on one kind of neighbourhood
and tested on another.
    mismatch = (same-centre fraction of training neighbourhoods)
             - (same-centre fraction of test neighbourhoods = 0 under LOCO)
centre_knn with alpha = 1 makes mismatch = 0 by construction.

Run from the project root:   python -m src.build_graph
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ======================================================================= SNF
def _sq_dist(X):
    s = (X ** 2).sum(1)
    return np.maximum(s[:, None] + s[None, :] - 2 * X @ X.T, 0)


def _affinity(X, K=20, mu=0.5):
    """Scaled exponential kernel of Wang et al. (Nature Methods 2014)."""
    D2 = _sq_dist(X)
    D = np.sqrt(D2)
    T = np.sort(D, 1)[:, 1:K + 1].mean(1)                     # mean distance to K neighbours
    sigma = (T[:, None] + T[None, :] + D) / 3
    return np.exp(-D2 / (mu * sigma + 1e-12))


def _full_norm(W):
    W = W.copy()
    np.fill_diagonal(W, 0)
    P = W / (2 * W.sum(1, keepdims=True) + 1e-12)
    np.fill_diagonal(P, 0.5)
    return P


def _local_norm(W, K):
    S = np.zeros_like(W)
    Wn = W.copy()
    np.fill_diagonal(Wn, -np.inf)
    idx = np.argsort(-Wn, 1)[:, :K]
    rows = np.arange(len(W))[:, None]
    S[rows, idx] = W[rows, idx]
    return S / (S.sum(1, keepdims=True) + 1e-12)


def snf(X_list, K=20, mu=0.5, t=20):
    """Similarity Network Fusion for a list of feature matrices (same patients, same order)."""
    Ws = [_affinity(X, K, mu) for X in X_list]
    Ps = [_full_norm(W) for W in Ws]
    Ss = [_local_norm(W, K) for W in Ws]
    m = len(Ps)
    for _ in range(t):
        new = []
        for v in range(m):
            others = sum(Ps[u] for u in range(m) if u != v) / (m - 1)
            P = Ss[v] @ others @ Ss[v].T
            P = (P + P.T) / 2
            new.append(_full_norm(P))
        Ps = new
    F = sum(Ps) / m
    return (F + F.T) / 2


# ======================================================================= builders
def _nearest(D_row, pool, k):
    """k nearest candidates from `pool` (indices) given a distance row."""
    if k <= 0 or len(pool) == 0:
        return np.array([], int)
    pool = np.asarray(pool)
    return pool[np.argsort(D_row[pool])[:k]]


def build_graph(X_pet, X_ct, centres, train_mask, mode="centre_knn",
                k=10, alpha=1.0, threshold=0.2, snf_K=20):
    """
    X_pet, X_ct : (n, p) selected + z-scored features (from M2), all patients
    centres     : (n,) hospital label of each patient
    train_mask  : (n,) True for training patients
    Returns edge_index (2, E) int array [source; target].
    """
    centres = np.asarray(centres)
    train = np.flatnonzero(train_mask)
    test = np.flatnonzero(~np.asarray(train_mask))
    src, dst = [], []

    if mode == "snf":
        F = snf([X_pet, X_ct], K=snf_K)
        off = ~np.eye(len(F), dtype=bool)
        # DOCUMENTED DEVIATION: raw SNF values are ~1e-2, so an absolute 0.2 keeps
        # almost nothing. We min-max the off-diagonal to [0, 1] before thresholding.
        # threshold="d0.8" instead keeps the top 80% of train-train pairs (Peng's density).
        Fn = (F - F[off].min()) / (F[off].max() - F[off].min())
        if isinstance(threshold, str) and threshold.startswith("d"):
            vals = Fn[np.ix_(train, train)][~np.eye(len(train), dtype=bool)]
            thr = np.quantile(vals, 1 - float(threshold[1:]))
        else:
            thr = float(threshold)
        for i in train:                                   # train <- train
            for j in train[(Fn[i, train] > thr) & (train != i)]:
                src.append(j); dst.append(i)
        for i in test:                                    # test <- train only
            for j in train[Fn[i, train] > thr]:
                src.append(j); dst.append(i)
    else:
        X = np.hstack([X_pet, X_ct])
        D = np.sqrt(_sq_dist(X))
        test_set = set(test.tolist())
        for i in np.concatenate([train, test]):
            pool = train[train != i]
            if mode == "knn" or i in test_set:
                # test patients: k nearest training patients (all cross-centre under LOCO)
                nb = _nearest(D[i], pool, k)
            elif mode == "centre_knn":
                k_cross = int(round(alpha * k))
                cross = pool[centres[pool] != centres[i]]
                same = pool[centres[pool] == centres[i]]
                nb = np.concatenate([_nearest(D[i], cross, k_cross),
                                     _nearest(D[i], same, k - k_cross)])
            else:
                raise ValueError(f"unknown mode {mode!r}")
            src.extend(nb); dst.extend([i] * len(nb))

    return np.array([src, dst], dtype=int).reshape(2, -1)


# ======================================================================= diagnostics
def edge_stats(edge_index, centres, train_mask):
    centres = np.asarray(centres)
    train_mask = np.asarray(train_mask)
    s, d = edge_index
    n = len(centres)
    indeg = np.bincount(d, minlength=n)
    same = centres[s] == centres[d]

    def same_frac(nodes_mask):
        m = nodes_mask[d]
        return same[m].mean() if m.any() else np.nan

    a, b = train_mask[s], train_mask[d]
    assert not (~a & ~b).any(), "test -> test edge found"
    assert not (~a & b).any(), "test -> train edge found"
    tr_same, te_same = same_frac(train_mask), same_frac(~train_mask)
    return {"edges": len(s),
            "deg_train": indeg[train_mask].mean(),
            "deg_test": indeg[~train_mask].mean(),
            "isolated_test": int((indeg[~train_mask] == 0).sum()),
            "same_ctr_train": tr_same,
            "same_ctr_test": te_same,
            "mismatch": tr_same - (0 if np.isnan(te_same) else te_same)}


# ======================================================================= demo
if __name__ == "__main__":
    from src.select_features import RadiomicsSelector

    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "synth_shift1.5.csv")
    configs = [("snf@0.2", dict(mode="snf", threshold=0.2)),
               ("snf@d0.8", dict(mode="snf", threshold="d0.8")),
               ("knn", dict(mode="knn")),
               *[(f"centre_knn a={a}", dict(mode="centre_knn", alpha=a))
                 for a in (0.0, 0.5, 1.0)]]

    rows = []
    for held in sorted(df["centre"].unique()):
        train_mask = (df["centre"] != held).to_numpy()
        tr = df[train_mask]
        feats = {}
        for m in ("pet", "ct"):                       # M2: fit on training patients only
            cols = [c for c in df.columns if c.startswith(m + "_")]
            feats[m] = RadiomicsSelector(top_n=15).fit(tr, cols).transform(df)
        for name, kw in configs:
            ei = build_graph(feats["pet"], feats["ct"], df["centre"], train_mask, k=10, **kw)
            rows.append({"held_out": held, "graph": name,
                         **edge_stats(ei, df["centre"], train_mask)})

    res = pd.DataFrame(rows)
    out = res.groupby("graph", sort=False).mean(numeric_only=True).round(2)
    pd.set_option("display.width", 200)
    print("Graph statistics, averaged over the 4 leave-one-centre-out folds (synthetic, shift=1.5)\n")
    print(out.to_string())
    (PROJECT_ROOT / "results").mkdir(exist_ok=True)
    res.to_csv(PROJECT_ROOT / "results" / "m3_graph_stats_synth.csv", index=False)
