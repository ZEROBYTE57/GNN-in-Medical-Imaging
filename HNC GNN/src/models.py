"""
M4 - Survival models. Each one outputs ONE risk score per patient (higher = worse prognosis).

  "cox"   classic Cox proportional-hazards model on the selected features. No graph, no network.
  "mlp"   small neural network + Cox loss. No graph. This is the FAIR baseline for the GNNs:
          same features, same loss, same training loop - the only thing missing is the graph.
          (If a GNN can't beat this, the graph adds nothing - Mueller et al., TMLR 2024.)
  "gcn"   graph convolution: each patient's representation = weighted average of itself
          and the patients it listens to (its in-edges from M3).
  "gat"   graph attention: like GCN, but LEARNS how much to listen to each neighbour.
          (It can down-weight neighbours, but cannot add or remove edges.)

Cox loss (negative partial log-likelihood)
------------------------------------------
For every patient i who had an event at time t_i:
    loss_i = -( risk_i - log sum_{j still at risk at t_i} exp(risk_j) )
"Rank patient i above everyone who was still event-free when i had its event."
Censored patients never contribute their own term, but they do sit in others' risk sets.

Training protocol (per leave-one-centre-out fold)
-------------------------------------------------
  * the held-out centre is never used for training OR early stopping
  * 20% of TRAINING patients (stratified by event) are an inner validation set
  * train up to 300 epochs; keep the epoch with the best inner-validation C-index
  * full-batch: the whole graph is one forward pass (only ~300 nodes)

Run from the project root:   python -m src.models
"""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GATConv, GCNConv

from src.select_features import harrell_c

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cpu")          # the graph is tiny; CPU is deterministic and fast enough


# ------------------------------------------------------------------ loss
def cox_loss(risk, time, event):
    order = torch.argsort(time, descending=True)      # longest time first -> cumsum = risk set
    r, e = risk[order], event[order]
    log_risk_set = torch.logcumsumexp(r, dim=0)
    return -((r - log_risk_set) * e).sum() / e.sum().clamp(min=1.0)


# ------------------------------------------------------------------ networks
class MLPCox(nn.Module):
    def __init__(self, d_in, hidden=32, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 1))

    def forward(self, x, edge_index=None):
        return self.net(x).squeeze(-1)


class GNNCox(nn.Module):
    def __init__(self, d_in, conv="gcn", hidden=32, dropout=0.3, heads=4):
        super().__init__()
        if conv == "gcn":
            self.c1, self.c2 = GCNConv(d_in, hidden), GCNConv(hidden, hidden)
        elif conv == "gat":
            self.c1 = GATConv(d_in, hidden, heads=heads, concat=False, dropout=dropout)
            self.c2 = GATConv(hidden, hidden, heads=heads, concat=False, dropout=dropout)
        else:
            raise ValueError(conv)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        h = self.drop(torch.relu(self.c1(x, edge_index)))
        h = self.drop(torch.relu(self.c2(h, edge_index)))
        return self.head(h).squeeze(-1)


# ------------------------------------------------------------------ fit + predict
def fit_predict(model_name, X, time, event, train_idx, edge_index=None, seed=0,
                hidden=32, lr=5e-3, weight_decay=5e-3, epochs=300, val_frac=0.2):
    """Train on train_idx only. Returns a risk score for EVERY patient (numpy array)."""
    time, event = np.asarray(time, float), np.asarray(event, int)

    if model_name == "cox":
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.util import Surv
        y = Surv.from_arrays(event[train_idx].astype(bool), time[train_idx])
        return CoxPHSurvivalAnalysis(alpha=1.0).fit(X[train_idx], y).predict(X)

    torch.manual_seed(seed)
    fit_idx, val_idx = train_test_split(train_idx, test_size=val_frac, random_state=seed,
                                        stratify=event[train_idx])
    x = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    t = torch.tensor(time, dtype=torch.float32, device=DEVICE)
    e = torch.tensor(event, dtype=torch.float32, device=DEVICE)
    ei = None if edge_index is None else torch.tensor(edge_index, dtype=torch.long, device=DEVICE)
    fi = torch.tensor(fit_idx, device=DEVICE)

    if model_name == "mlp":
        model = MLPCox(X.shape[1], hidden)
    else:
        assert ei is not None, f"{model_name} needs an edge_index"
        model = GNNCox(X.shape[1], conv=model_name, hidden=hidden)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_c, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        risk = model(x, ei)
        loss = cox_loss(risk[fi], t[fi], e[fi])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            r = model(x, ei).cpu().numpy()
        c = harrell_c(time[val_idx], event[val_idx], r[val_idx])[0]
        if c > best_c:
            best_c, best_state = c, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(x, ei).cpu().numpy()


# ------------------------------------------------------------------ demo
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    from src.build_graph import build_graph
    from src.select_features import RadiomicsSelector

    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "synth_shift1.5.csv")
    time, event = df["time"].to_numpy(), df["event"].to_numpy()
    runs = [("Cox (no graph)", "cox", None), ("MLP (no graph)", "mlp", None),
            ("GCN + snf@d0.8", "gcn", dict(mode="snf", threshold="d0.8")),
            ("GCN + knn", "gcn", dict(mode="knn")),
            ("GCN + centre_knn a=1", "gcn", dict(mode="centre_knn", alpha=1.0)),
            ("GAT + knn", "gat", dict(mode="knn")),
            ("GAT + centre_knn a=1", "gat", dict(mode="centre_knn", alpha=1.0))]

    table = {}
    for held in sorted(df["centre"].unique()):
        train_mask = (df["centre"] != held).to_numpy()
        train_idx, test_idx = np.flatnonzero(train_mask), np.flatnonzero(~train_mask)
        f = {m: RadiomicsSelector(top_n=15).fit(df[train_mask], [c for c in df.columns
                                                                 if c.startswith(m + "_")]).transform(df)
             for m in ("pet", "ct")}
        X = np.hstack([f["pet"], f["ct"]])
        for label, model, gkw in runs:
            ei = None if gkw is None else build_graph(f["pet"], f["ct"], df["centre"],
                                                      train_mask, k=10, **gkw)
            risk = fit_predict(model, X, time, event, train_idx, edge_index=ei, seed=0)
            table.setdefault(label, {})[held] = harrell_c(time[test_idx], event[test_idx],
                                                          risk[test_idx])[0]
        print(f"fold {held} done")

    res = pd.DataFrame(table).T
    res["mean"] = res.mean(1)
    print("\nTest C-index on the held-out centre (synthetic, shift=1.5, ONE seed - a smoke test,")
    print("not a result; M5 adds repeated seeds, confidence intervals and significance tests)\n")
    print(res.round(3).to_string())
