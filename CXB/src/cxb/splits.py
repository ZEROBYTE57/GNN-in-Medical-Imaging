"""Evaluation splits.

BrainOOD (Xu et al., ICLR 2025) leave-one-site-out protocol for ABIDE, recovered from its
released meta.json by matching subjects on age, sex and site (see docs/decisions.md). Each
fold holds out one site for testing and another for validation (model selection); all
other sites are used for training.
"""
import numpy as np

BRAINOOD_FOLDS = [  # (test site, validation site)
    ("SBL", "STANFORD"),
    ("CALTECH", "PITT"),
    ("SDSU", "MAX_MUN"),
    ("OLIN", "SBL"),
    ("STANFORD", "SDSU"),
    ("TRINITY", "MAX_MUN"),
    ("MAX_MUN", "TRINITY"),
    ("YALE", "CALTECH"),
    ("KKI", "STANFORD"),
    ("PITT", "TRINITY"),
]


DROPPED_SITES = ("CMU",)  # not in BrainOOD's protocol (docs/decisions.md, 29 Oct 2026)


def site_fold(site, fold, dropped=DROPPED_SITES):
    """Return (train_idx, val_idx, test_idx) index arrays for BrainOOD fold 0-9.

    Subjects from `dropped` sites are left out of all three parts."""
    test_site, val_site = BRAINOOD_FOLDS[fold]
    site = np.asarray(site)
    test = np.where(site == test_site)[0]
    val = np.where(site == val_site)[0]
    train = np.where((site != test_site) & (site != val_site) & ~np.isin(site, dropped))[0]
    return train, val, test
