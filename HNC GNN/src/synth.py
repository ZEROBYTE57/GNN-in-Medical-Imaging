"""
M1 - Synthetic multi-centre survival data generator.

Why this exists
---------------
Real data (TCIA / HECKTOR) is not available yet. This module makes a fake
dataset with the SAME structure and the SAME problem as the real one, so
every later module (feature selection, graphs, GNN, evaluation) can be built
and tested today. When real data arrives it replaces this file's output with
an identical column layout.

What it simulates
-----------------
1. Biology:   each patient has PET and CT features. A few of them truly drive
              risk; the rest are noise (like most of 3,000 radiomics features).
2. Survival:  time-to-event drawn from a Cox model on that true risk, with
              censoring (some patients leave follow-up before any event).
3. Centre effect ("scanner shift"): every hospital shifts and rescales the
              features differently. This changes the numbers but NOT the
              patient's real risk. This is the thing that breaks cross-site
              generalisation, and the thing centre-balanced k-NN targets.

Run from the project root:   python -m src.synth
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Approximate centre sizes of TCIA Head-Neck-PET-CT (~298 patients)
CENTRES = {"CHUS": 101, "HGJ": 91, "CHUM": 65, "HMR": 41}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class SynthConfig:
    centres: dict = field(default_factory=lambda: dict(CENTRES))
    n_pet: int = 30               # features per modality
    n_ct: int = 30
    n_informative: int = 4        # per modality: features that carry real risk
    risk_strength: float = 1.5    # how predictable survival is
    centre_shift: float = 1.5     # scanner effect strength (0 = none)
    target_event_rate: float = 0.35
    max_followup: float = 60.0    # months
    seed: int = 42


def make_synthetic(cfg: SynthConfig = SynthConfig()):
    """Return (df, true_risk). df has: patient_id, centre, time, event, pet_*, ct_*."""
    rng = np.random.default_rng(cfg.seed)
    centre = np.concatenate([[c] * n for c, n in cfg.centres.items()])
    n, k = len(centre), cfg.n_informative

    # 1. Biology: identical distribution in every hospital
    z_pet = rng.normal(size=(n, cfg.n_pet))
    z_ct = rng.normal(size=(n, cfg.n_ct))

    # 2. True risk uses only the first k features of each modality
    informative = np.c_[z_pet[:, :k], z_ct[:, :k]]
    w = rng.uniform(0.5, 1.0, 2 * k) * rng.choice([-1, 1], 2 * k)
    true_risk = cfg.risk_strength * informative @ w / np.sqrt(2 * k)

    # 3. Centre effect: per-hospital shift + rescale of the MEASURED features
    x_pet, x_ct = z_pet.copy(), z_ct.copy()
    for c in cfg.centres:
        m = centre == c
        for x in (x_pet, x_ct):
            shift = rng.normal(0, cfg.centre_shift, x.shape[1])
            scale = np.exp(rng.normal(0, 0.1 * cfg.centre_shift, x.shape[1]))
            x[m] = x[m] * scale + shift

    # 4. Survival times (exponential Cox model, baseline mean ~100 months)
    t_event = rng.exponential(100.0 / np.exp(true_risk))

    # 5. Censoring: random dropout + end of study; tune dropout to hit target event rate
    u = rng.random(n)
    best = None
    for s in np.geomspace(5, 2000, 200):
        t_cens = np.minimum(-s * np.log(u), cfg.max_followup)
        rate = (t_event <= t_cens).mean()
        if best is None or abs(rate - cfg.target_event_rate) < best[0]:
            best = (abs(rate - cfg.target_event_rate), t_cens)
    t_cens = best[1]

    df = pd.DataFrame({
        "patient_id": [f"{c}-{i:03d}" for i, c in enumerate(centre)],
        "centre": centre,
        "time": np.round(np.minimum(t_event, t_cens), 1),
        "event": (t_event <= t_cens).astype(int),
    })
    df = pd.concat([df,
                    pd.DataFrame(x_pet, columns=[f"pet_f{j:02d}" for j in range(cfg.n_pet)]),
                    pd.DataFrame(x_ct, columns=[f"ct_f{j:02d}" for j in range(cfg.n_ct)])], axis=1)
    return df, true_risk


def sanity_report(df, true_risk, k=5):
    """Four checks that the fake data behaves like the real problem."""
    from lifelines.utils import concordance_index
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    feats = [c for c in df.columns if c.startswith(("pet_", "ct_"))]
    X = StandardScaler().fit_transform(df[feats])
    y = df["centre"].to_numpy()

    # (a) best possible C-index if you knew the true risk
    oracle = concordance_index(df["time"], -true_risk, df["event"])
    # (b) can a classifier tell which hospital a patient came from?
    acc = cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=5).mean()
    chance = pd.Series(y).value_counts(normalize=True).max()
    # (c) plain k-NN: what fraction of each patient's neighbours share its hospital?
    idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)[1][:, 1:]
    same = (y[idx] == y[:, None]).mean()
    sizes = pd.Series(y).value_counts()
    same_random = ((sizes * (sizes - 1)).sum()) / (len(y) * (len(y) - 1))
    return {"n": len(df), "event_rate": df["event"].mean(), "oracle_cindex": oracle,
            "centre_acc": acc, "centre_chance": chance,
            "knn_same_centre": same, "same_centre_if_random": same_random}


if __name__ == "__main__":
    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'centre_shift':>12} {'events':>7} {'oracle C':>9} {'centre acc':>11}"
          f" {'(chance)':>9} {'kNN same-ctr':>13} {'(random)':>9}")
    for shift in (0.0, 0.5, 1.5):
        df, risk = make_synthetic(SynthConfig(centre_shift=shift))
        r = sanity_report(df, risk)
        print(f"{shift:>12.1f} {r['event_rate']:>7.2f} {r['oracle_cindex']:>9.3f}"
              f" {r['centre_acc']:>11.2f} {r['centre_chance']:>9.2f}"
              f" {r['knn_same_centre']:>13.2f} {r['same_centre_if_random']:>9.2f}")
        df.to_csv(out_dir / f"synth_shift{shift}.csv", index=False)

    print("\nPer-centre summary (shift=1.5):")
    df = pd.read_csv(out_dir / "synth_shift1.5.csv")
    print(df.groupby("centre").agg(n=("event", "size"), event_rate=("event", "mean"),
                                   median_time=("time", "median")).round(2))
    print(f"\nSaved CSVs to {out_dir}")
