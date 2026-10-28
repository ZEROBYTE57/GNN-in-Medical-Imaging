"""A simulated multi-centre PET/CT cohort.

Purpose: make the whole pipeline runnable and testable *before* the TCIA
download and radiomics extraction finish, and give the graph-construction study
a setting where the ground truth is known.

The generator encodes the causal story the proposal argues for:

    z        latent biology, shared across centres, the ONLY driver of hazard
    b_c      centre effect (scanner, reconstruction, protocol), drives features
             but NOT hazard
    x_pet    = A_pet z + b_c^pet + noise
    x_ct     = A_ct  z + b_c^ct  + noise

Because ``b_c`` enters the features but not the hazard, similarity computed on
raw features mixes a prognostic component with a nuisance component. Turning up
``batch_strength`` makes same-centre patients look artificially alike, which is
exactly the regime in which a fixed similarity threshold produces a
centre-pure graph and cross-centre generalisation collapses. Turning it to 0
should make all three edge rules behave the same -- a useful negative control.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..seed import rng

DEFAULT_CENTRE_SIZES: dict[str, int] = {
    # Matches the real Head-Neck-PET-CT cohort sizes (Vallieres et al.).
    "CHUS": 100,
    "HGJ": 90,
    "CHUM": 65,
    "HMR": 41,
}


def make_synthetic_cohort(
    centre_sizes: dict[str, int] | None = None,
    n_pet: int = 60,
    n_ct: int = 60,
    n_latent: int = 6,
    batch_strength: float = 1.5,
    noise: float = 0.7,
    censor_rate: float = 0.55,
    max_followup: float = 72.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate a feature table with the same schema as the real one.

    Parameters
    ----------
    batch_strength:
        Standard deviation of the per-centre feature offset, in units of the
        latent signal. 0 = no site effect; 1.5 = site effect comparable to
        biology, the regime reported for multi-centre radiomics.
    censor_rate:
        Target fraction of censored patients.

    Returns
    -------
    DataFrame with columns ``patient_id``, ``centre``, ``PFS_time``,
    ``PFS_event``, the clinical covariates, and ``PET_*`` / ``CT_*`` feature
    columns -- i.e. exactly what ``scripts/02_run_loco.py`` consumes.
    """
    centre_sizes = centre_sizes or DEFAULT_CENTRE_SIZES
    g = rng(seed)
    centres = list(centre_sizes)
    n_total = sum(centre_sizes.values())

    # --- latent biology ------------------------------------------------------
    z = g.normal(size=(n_total, n_latent))

    # --- centre membership ---------------------------------------------------
    centre_labels: list[str] = []
    for c, n in centre_sizes.items():
        centre_labels += [c] * n
    centre_labels = np.array(centre_labels)
    order = g.permutation(n_total)
    z, centre_labels = z[order], centre_labels[order]

    # --- modality loadings and centre effects -------------------------------
    a_pet = g.normal(scale=1.0, size=(n_latent, n_pet))
    a_ct = g.normal(scale=1.0, size=(n_latent, n_ct))
    b_pet = {c: g.normal(scale=batch_strength, size=n_pet) for c in centres}
    b_ct = {c: g.normal(scale=batch_strength, size=n_ct) for c in centres}

    offs_pet = np.vstack([b_pet[c] for c in centre_labels])
    offs_ct = np.vstack([b_ct[c] for c in centre_labels])

    x_pet = z @ a_pet + offs_pet + g.normal(scale=noise, size=(n_total, n_pet))
    x_ct = z @ a_ct + offs_ct + g.normal(scale=noise, size=(n_total, n_ct))

    # --- clinical covariates -------------------------------------------------
    # Age and stage carry a little prognostic signal of their own, as they do
    # in the real cohort, so the clinical-only Cox baseline is not trivial.
    age = 60 + 10 * z[:, 0] + g.normal(scale=6, size=n_total)
    t_stage = np.clip(np.round(2.5 + 0.8 * z[:, 1] + g.normal(scale=0.6, size=n_total)), 1, 4)
    n_stage = np.clip(np.round(1.8 + 0.7 * z[:, 2] + g.normal(scale=0.7, size=n_total)), 0, 3)
    sites = np.array(["Oropharynx", "Larynx", "Oral cavity", "Nasopharynx", "Hypopharynx"])
    site = g.choice(sites, size=n_total, p=[0.60, 0.20, 0.09, 0.06, 0.05])

    # --- hazard: latent biology + clinical only, never the centre effect -----
    beta = g.normal(scale=1.0, size=n_latent)
    beta /= np.linalg.norm(beta)
    log_hazard = (
        1.4 * (z @ beta)
        + 0.25 * (t_stage - t_stage.mean()) / t_stage.std()
        + 0.15 * (n_stage - n_stage.mean()) / n_stage.std()
    )
    log_hazard -= log_hazard.mean()

    # --- exponential survival with administrative + random censoring ---------
    baseline_scale = 30.0
    true_time = g.exponential(scale=baseline_scale * np.exp(-log_hazard))
    censor_scale = baseline_scale * (censor_rate / max(1e-6, 1 - censor_rate))
    censor_time = np.minimum(g.exponential(scale=censor_scale, size=n_total), max_followup)

    observed = np.minimum(true_time, censor_time)
    event = (true_time <= censor_time).astype(int)

    # --- assemble ------------------------------------------------------------
    counters: dict[str, int] = dict.fromkeys(centres, 0)
    pids = []
    for c in centre_labels:
        counters[c] += 1
        pids.append(f"HN-{c}-{counters[c]:03d}")

    df = pd.DataFrame(
        {
            "patient_id": pids,
            "centre": centre_labels,
            "PFS_time": np.round(observed, 3),
            "PFS_event": event,
            "age": np.round(age, 1),
            "T_stage": [f"T{int(t)}" for t in t_stage],
            "N_stage": [f"N{int(n)}" for n in n_stage],
            "site": site,
        }
    )
    df = pd.concat(
        [
            df,
            pd.DataFrame(x_pet, columns=[f"PET_f{j:03d}" for j in range(n_pet)]),
            pd.DataFrame(x_ct, columns=[f"CT_f{j:03d}" for j in range(n_ct)]),
        ],
        axis=1,
    )

    df.attrs["true_beta"] = beta
    df.attrs["batch_strength"] = batch_strength
    return df
