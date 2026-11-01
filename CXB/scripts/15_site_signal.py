"""How strong is the real hospital signal in ABIDE connectomes? (CPU only; decisions.md, 1 Nov)

A shortcut is a risk when it is easier to learn than the disease. This check measures both
on the real data with the same simple model: L2 logistic regression (C = 0.1) on the
standardised upper-triangle Fisher-z edges, stratified 5-fold cross-validation over all
848 subjects (scaling fitted on the training folds only).
  - hospital (19 sites): balanced accuracy, against chance 1/19;
  - autism vs control:   balanced accuracy, against chance 0.5.
Within-sample folds are used for both, so the two numbers are comparable (site-held-out
folds cannot be used to predict the site). Writes docs/site_signal.md.

    .venv\\Scripts\\python.exe -u scripts\\15_site_signal.py
"""
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]


def cv_balanced_accuracy(x, y, seed=0):
    pred = np.empty_like(y)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(x, y):
        mu, sd = x[tr].mean(0), x[tr].std(0)
        sd = np.where(sd > 1e-6, sd, 1.0)
        clf = LogisticRegression(C=0.1, max_iter=3000).fit((x[tr] - mu) / sd, y[tr])
        pred[te] = clf.predict((x[te] - mu) / sd)
    return balanced_accuracy_score(y, pred)


def main():
    lines = ["# Real hospital signal vs disease signal (ABIDE, logistic regression, 5-fold CV)", "",
             "| Atlas | Hospital: balanced accuracy (chance) | Autism vs control: balanced accuracy (chance) |",
             "|---|---|---|"]
    for atlas in ("rois_aal", "rois_cc200"):
        d = np.load(ROOT / "data" / "processed" / f"abide_{atlas}.npz", allow_pickle=True)
        iu = np.triu_indices(d["fc"].shape[1], 1)
        x = d["fc"][:, iu[0], iu[1]].astype(np.float64)
        sites = sorted(set(d["site"]))
        site = np.array([sites.index(s) for s in d["site"]])
        dx = (d["dx"] == 1).astype(int)
        a_site = cv_balanced_accuracy(x, site)
        a_dx = cv_balanced_accuracy(x, dx)
        print(f"{atlas}: hospital {a_site:.3f} (chance {1 / len(sites):.3f}) | autism {a_dx:.3f} (chance 0.5)", flush=True)
        lines.append(f"| {atlas.replace('rois_', '').upper()} | {a_site:.2f} ({1 / len(sites):.2f}) | {a_dx:.2f} (0.50) |")
    lines += ["", "848 subjects, 19 hospitals. Same model, features and folds for both targets."]
    (ROOT / "docs" / "site_signal.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
