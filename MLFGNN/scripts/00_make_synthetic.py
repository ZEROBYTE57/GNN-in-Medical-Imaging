"""Generate a simulated multi-centre cohort so the pipeline runs before TCIA lands.

    python scripts/00_make_synthetic.py --out data/processed/features_synth.csv

The site effect is a knob: ``--batch-strength 0`` removes it entirely, which is
the negative control -- with no site effect the three graph rules should be
statistically indistinguishable, and if they are not, the difference is coming
from something other than centre mixing.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlfgnn.data.synthetic import DEFAULT_CENTRE_SIZES, make_synthetic_cohort  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/processed/features_synth.csv")
    p.add_argument("--batch-strength", type=float, default=1.5,
                   help="per-centre feature offset; 0 = no site effect")
    p.add_argument("--n-pet", type=int, default=60)
    p.add_argument("--n-ct", type=int, default=60)
    p.add_argument("--noise", type=float, default=0.7)
    p.add_argument("--censor-rate", type=float, default=0.55)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scale", type=float, default=1.0,
                   help="multiply every centre's size (use <1 for a fast smoke run)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    sizes = {c: max(12, int(round(n * args.scale))) for c, n in DEFAULT_CENTRE_SIZES.items()}
    df = make_synthetic_cohort(
        centre_sizes=sizes,
        n_pet=args.n_pet,
        n_ct=args.n_ct,
        batch_strength=args.batch_strength,
        noise=args.noise,
        censor_rate=args.censor_rate,
        seed=args.seed,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"wrote {out}  ({len(df)} patients, {df['PFS_event'].mean():.0%} events)")
    print(df.groupby("centre").agg(n=("patient_id", "size"), events=("PFS_event", "sum")))


if __name__ == "__main__":
    main()
