"""Patient ID -> centre label.

The centre label is the unit of the whole study: it defines the LOCO folds and
it is the grouping variable the proposed edge rule balances over. In the TCIA
Head-Neck-PET-CT collection the centre is encoded in the patient ID prefix
(Vallieres et al., Sci. Rep. 7, 2017), e.g.

    HN-CHUM-001 -> CHUM   Centre hospitalier de l'Universite de Montreal
    HN-CHUS-010 -> CHUS   Centre hospitalier universitaire de Sherbrooke
    HN-HGJ-042  -> HGJ    Hopital general juif de Montreal
    HN-HMR-007  -> HMR    Hopital Maisonneuve-Rosemont

The extension collections named in the protocol are registered here too so that
adding them later is a data-drop, not a code change.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

# Ordered: first match wins, so put the more specific patterns first.
CENTRE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # --- Head-Neck-PET-CT (Vallieres et al.) -- the 4 centres used now -------
    ("CHUM", re.compile(r"^HN-?CHUM", re.I)),
    ("CHUS", re.compile(r"^HN-?CHUS", re.I)),
    ("HGJ", re.compile(r"^HN-?HGJ", re.I)),
    ("HMR", re.compile(r"^HN-?HMR", re.I)),
    # --- planned extensions --------------------------------------------------
    ("MAASTRO", re.compile(r"^HN1?-?\d+$|^HN1", re.I)),   # Head-Neck-Radiomics-HN1
    ("QIN", re.compile(r"^QIN", re.I)),                    # QIN-HEADNECK (Iowa)
    ("TCGA", re.compile(r"^TCGA-?", re.I)),                # TCGA-HNSC (Pittsburgh)
    # --- HECKTOR-2022, requested access --------------------------------------
    ("CHUP", re.compile(r"^CHUP", re.I)),
    ("CHUV", re.compile(r"^CHUV", re.I)),
    ("MDA", re.compile(r"^MDA", re.I)),
    ("USZ", re.compile(r"^USZ", re.I)),
]

#: Centres available in the stage-1 cohort. LOCO runs one fold per entry.
PRIMARY_CENTRES: tuple[str, ...] = ("CHUM", "CHUS", "HGJ", "HMR")


class UnknownCentreError(ValueError):
    """Raised when a patient ID matches no registered centre pattern."""


def centre_of(patient_id: str, strict: bool = True) -> str:
    """Map one patient ID to its centre label."""
    pid = str(patient_id).strip()
    for label, pattern in CENTRE_PATTERNS:
        if pattern.match(pid):
            return label
    if strict:
        raise UnknownCentreError(
            f"patient_id {pid!r} matches no centre pattern; "
            "add it to CENTRE_PATTERNS before running."
        )
    return "UNKNOWN"


def centres_of(patient_ids: Iterable[str], strict: bool = True) -> list[str]:
    """Vectorised :func:`centre_of`."""
    return [centre_of(p, strict=strict) for p in patient_ids]


def loco_folds(centres: Sequence[str]) -> list[tuple[str, ...]]:
    """One fold per distinct centre, in a stable (sorted) order.

    Returns a list of ``(held_out_centre,)`` tuples -- a tuple rather than a
    bare string so that grouped folds (e.g. leave-one-*country*-out) can be
    added later without changing the caller.
    """
    return [(c,) for c in sorted(set(centres))]
