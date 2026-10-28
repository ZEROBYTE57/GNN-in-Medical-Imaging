"""Feature extraction (stage 1, Python 3.9 env) and selection (stage 3, main env).

Imports are deliberately lazy: ``extract`` needs PyRadiomics and runs in the
radiomics environment, ``select`` needs scipy/scikit-survival and runs in the
main one. Neither environment should have to satisfy the other's dependencies
just to import this package.
"""

from __future__ import annotations

__all__ = ["FeatureSelector", "SelectionResult"]


def __getattr__(name: str):
    if name in __all__:
        from . import select

        return getattr(select, name)
    raise AttributeError(name)
