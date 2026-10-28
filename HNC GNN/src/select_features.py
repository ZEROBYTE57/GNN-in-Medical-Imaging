"""
M2 - Feature selection (Peng et al. 2024, Sec. 3.1), fitted on TRAINING patients only.

Why "training only" matters
---------------------------
If the held-out hospital's patients help choose the features, the model has
already "seen" the test set. Its test C-index is then optimistic (leakage).
So: selector.fit(train_df)  ->  selector.transform(any_df).
Every statistic (min/max, mean/std, rankings, LASSO) comes from train_df.

Peng's funnel (per modality, PET and CT separately)
---------------------------------------------------
  1. Variance filter        drop features with variance < 0.005
  2. Univariate Cox ranking 50 x 3-fold CV on training data; rank by mean
                            C-index on the validation folds; keep top N (50)
  3. Redundancy pruning     if Spearman |rho| > 0.8, drop the lower-C feature
  4. LASSO-Cox              L1-penalised Cox, penalty chosen by CV; keep non-zero
                            (Peng does not say how the penalty was chosen. Default
                            "max" = best CV C-index; "1se" = sparsest model within
                            one standard error of the best, a common alternative)

Documented deviation (goes in your reproduction findings)
---------------------------------------------------------
Peng z-scores FIRST and then drops variance < 0.005. After z-scoring every
non-constant feature has variance exactly 1, so that filter removes nothing.
We apply the variance filter on min-max scaled [0,1] features instead (a
common radiomics convention), then z-score. `peng_order_check()` shows this.

Run from the project root:   python -m src.select_features
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RepeatedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-12


# ---------------------------------------------------------------- helpers
def harrell_c(time, event, scores):
    """Harrell's C-index for many score columns at once.
    scores: (n,) or (n, p); higher score = higher predicted risk.
    A pair (i, j) is comparable if i had an event before j's time."""
    scores = np.asarray(scores, float).reshape(len(time), -1)
    comp = (event[:, None] == 1) & (time[:, None] < time[None, :])
    diff = scores[:, None, :] - scores[None, :, :]
    conc = (diff > 0) + 0.5 * (diff == 0)
    return (conc * comp[:, :, None]).sum((0, 1)) / max(comp.sum(), 1)


def cox_direction(time, event, X):
    """Sign of the univariate Cox coefficient for every column, without fitting.
    The Cox log-likelihood is concave, so sign(beta_hat) = sign(score at beta=0):
        U = sum over events of (x_i - mean of x over patients still at risk)."""
    order = np.argsort(-time, kind="stable")
    ts, es, Xs = time[order], event[order], X[order]
    last = np.searchsorted(-ts, -ts, side="right") - 1      # handles tied times
    risk_mean = np.cumsum(Xs, 0)[last] / (last + 1)[:, None]
    U = ((Xs - risk_mean) * es[:, None]).sum(0)
    return np.where(U >= 0, 1.0, -1.0)


# ---------------------------------------------------------------- selector
class RadiomicsSelector:
    def __init__(self, var_thresh=0.005, top_n=50, n_repeats=50, n_folds=3,
                 rho_max=0.8, lasso_rule="max", seed=0):
        self.lasso_rule = lasso_rule   # "max" = best CV C-index, "1se" = sparsest within 1 SE
        self.var_thresh, self.top_n, self.rho_max = var_thresh, top_n, rho_max
        self.n_repeats, self.n_folds, self.seed = n_repeats, n_folds, seed

    def fit(self, df, feature_cols):
        X = df[feature_cols].to_numpy(float)
        t, e = df["time"].to_numpy(float), df["event"].to_numpy(int)
        cols = np.array(feature_cols)
        report = {"start": len(cols)}

        # 1. variance filter on min-max scaled features
        lo, hi = X.min(0), X.max(0)
        keep = ((X - lo) / (hi - lo + EPS)).var(0) >= self.var_thresh
        X, cols = X[:, keep], cols[keep]
        report["after_variance"] = len(cols)

        # z-score with TRAINING statistics
        mu, sd = X.mean(0), X.std(0) + EPS
        Z = (X - mu) / sd

        # 2. repeated-CV univariate Cox C-index ranking
        rkf = RepeatedKFold(n_splits=self.n_folds, n_repeats=self.n_repeats,
                            random_state=self.seed)
        cs = [harrell_c(t[va], e[va], Z[va] * cox_direction(t[tr], e[tr], Z[tr]))
              for tr, va in rkf.split(Z)]
        cidx = np.mean(cs, 0)
        top = np.argsort(-cidx)[: self.top_n]
        report["after_univariate"] = len(top)

        # 3. Spearman redundancy: walk down the ranking, keep a feature only if
        #    it is not strongly correlated with any better feature already kept
        rho = np.abs(pd.DataFrame(Z[:, top]).corr(method="spearman").to_numpy())
        kept = []
        for i in range(len(top)):
            if all(rho[i, j] <= self.rho_max for j in kept):
                kept.append(i)
        top = top[kept]
        report["after_spearman"] = len(top)

        # 4. LASSO-Cox with CV-chosen penalty
        chosen = self._lasso_cox(Z[:, top], t, e)
        final = top[chosen] if len(chosen) else top[:1]
        report["after_lasso"] = len(final)

        self.selected_ = list(cols[final])
        self.mu_ = pd.Series(mu, index=cols)[self.selected_]
        self.sd_ = pd.Series(sd, index=cols)[self.selected_]
        self.univariate_c_ = pd.Series(cidx, index=cols).sort_values(ascending=False)
        self.report_ = report
        return self

    def _lasso_cox(self, Z, t, e):
        from sksurv.linear_model import CoxnetSurvivalAnalysis
        from sksurv.util import Surv
        y = Surv.from_arrays(e.astype(bool), t)
        path = CoxnetSurvivalAnalysis(l1_ratio=1.0, n_alphas=50,
                                      alpha_min_ratio=0.01).fit(Z, y)
        alphas = path.alphas_
        fold_scores = []
        for tr, va in KFold(5, shuffle=True, random_state=self.seed).split(Z):
            m = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=alphas).fit(Z[tr], y[tr])
            preds = np.column_stack([m.predict(Z[va], alpha=a) for a in alphas])
            fold_scores.append(harrell_c(t[va], e[va], preds))
        fold_scores = np.array(fold_scores)                 # (folds, alphas)
        mean, se = fold_scores.mean(0), fold_scores.std(0) / np.sqrt(len(fold_scores))
        i_best = int(np.argmax(mean))
        if self.lasso_rule == "1se":   # alphas are sorted large -> small penalty
            i_best = int(np.flatnonzero(mean >= mean[i_best] - se[i_best])[0])
        best = alphas[i_best]
        coef = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=alphas).fit(Z, y) \
            .coef_[:, list(alphas).index(best)]
        return np.flatnonzero(np.abs(coef) > 1e-8)

    def transform(self, df):
        """Z-score the selected features using TRAINING mean/std."""
        return ((df[self.selected_] - self.mu_) / self.sd_).to_numpy(float)


def peng_order_check(df, feature_cols):
    """Shows why 'z-score, then variance < 0.005' removes nothing."""
    X = df[feature_cols].to_numpy(float)
    z = (X - X.mean(0)) / (X.std(0) + EPS)
    return int((z.var(0) < 0.005).sum())


# ---------------------------------------------------------------- demo
if __name__ == "__main__":
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "synth_shift1.5.csv")
    truth = {f"{m}_f{j:02d}" for m in ("pet", "ct") for j in range(4)}  # the real signal

    for m in ("pet", "ct"):
        cols = [c for c in df.columns if c.startswith(m + "_")]
        print(f"Peng order check ({m}): features removed by z-score-then-variance "
              f"= {peng_order_check(df, cols)} of {len(cols)}")

    print("\nLeave-one-centre-out: selector fitted on 3 centres only\n")
    print(f"{'held-out':<9}{'mod':<5}{'start':>6}{'var':>5}{'uni':>5}{'spear':>6}"
          f"{'lasso':>6}  true found   selected")
    for held in sorted(df["centre"].unique()):
        train = df[df["centre"] != held]
        assert held not in set(train["centre"]), "leakage: held-out centre in training"
        for m in ("pet", "ct"):
            cols = [c for c in df.columns if c.startswith(m + "_")]
            sel = RadiomicsSelector(top_n=15).fit(train, cols)   # 15 because synth has 30/modality
            r = sel.report_
            hits = len(truth & set(sel.selected_))
            print(f"{held:<9}{m:<5}{r['start']:>6}{r['after_variance']:>5}"
                  f"{r['after_univariate']:>5}{r['after_spearman']:>6}"
                  f"{r['after_lasso']:>6}  {hits}/4        "
                  f"{', '.join(s.split('_')[1] for s in sel.selected_)}")
