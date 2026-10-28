"""Non-graph baselines.

The brief calls for these so that a reviewer cannot answer the paper with "a
tabular model would have done as well" -- a live objection for population GNNs
since Mueller et al. (TMLR 2024) showed simple baselines matching advanced
population GNNs when graph construction is weak.

All four share one interface::

    model = ClinicalCox().fit(X_train, t_train, e_train)
    risk  = model.predict(X_test)          # higher = higher hazard

so ``experiments/loco.py`` can treat them exactly like the GNN arms and run them
through the same metrics and the same paired bootstrap.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class BaseSurvivalModel:
    name = "base"

    def fit(self, x: np.ndarray, time: np.ndarray, event: np.ndarray):
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class CoxPH(BaseSurvivalModel):
    """Penalised Cox proportional hazards (lifelines).

    A ridge penalty is on by default: the radiomics variant receives the same
    12+12 selected features as the GNN, and an unpenalised Cox fit on 24
    correlated radiomics features in a ~200-patient training fold routinely
    fails to converge.
    """

    name = "coxph"

    def __init__(self, penalizer: float = 0.1, l1_ratio: float = 0.0) -> None:
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self._model = None
        self._columns: list[str] = []

    def fit(self, x: np.ndarray, time: np.ndarray, event: np.ndarray):
        import pandas as pd
        from lifelines import CoxPHFitter

        self._columns = [f"x{i}" for i in range(x.shape[1])]
        df = pd.DataFrame(x, columns=self._columns)
        df["T"], df["E"] = time, event

        self._model = CoxPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
        try:
            self._model.fit(df, duration_col="T", event_col="E")
        except Exception as exc:  # noqa: BLE001 - convergence failures are expected
            logger.warning("CoxPH failed (%s); retrying with a stronger penalty", exc)
            self._model = CoxPHFitter(penalizer=max(self.penalizer * 10, 1.0))
            self._model.fit(df, duration_col="T", event_col="E")
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        import pandas as pd

        df = pd.DataFrame(x, columns=self._columns)
        return np.asarray(self._model.predict_partial_hazard(df)).ravel()


class ClinicalCox(CoxPH):
    """Cox on the clinical covariates only -- the model actually used in clinic."""

    name = "clinical_cox"


class RadiomicsCoxPH(CoxPH):
    """Cox on the selected radiomics features (Peng et al.'s Cox-PH comparator)."""

    name = "radiomics_coxph"


class CoxNnet(BaseSurvivalModel):
    """Cox-nnet: an MLP trained with the Cox partial likelihood.

    Ching et al.'s model, and the second comparator in Peng et al. Implemented
    on the same loss as the GNN so the only difference is the absence of the
    population graph -- which makes the GNN-vs-Cox-nnet gap attributable to
    message passing rather than to a different objective.
    """

    name = "cox_nnet"

    def __init__(
        self,
        hidden: int = 32,
        epochs: int = 200,
        lr: float = 0.005,
        weight_decay: float = 0.005,
        dropout: float = 0.5,
        seed: int = 0,
    ) -> None:
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.seed = seed
        self._model = None

    def fit(self, x: np.ndarray, time: np.ndarray, event: np.ndarray):
        import torch
        import torch.nn as nn

        from ..seed import set_seed
        from .losses import cox_ph_loss

        set_seed(self.seed)
        xt = torch.as_tensor(x, dtype=torch.float32)
        tt = torch.as_tensor(time, dtype=torch.float32)
        et = torch.as_tensor(event, dtype=torch.float32)

        self._model = nn.Sequential(
            nn.Linear(x.shape[1], self.hidden),
            nn.Tanh(),                       # Ching et al. use tanh hidden units
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, 1, bias=False),
        )
        opt = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self._model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = cox_ph_loss(self._model(xt).squeeze(-1), tt, et)
            loss.backward()
            opt.step()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        import torch

        self._model.eval()
        with torch.no_grad():
            out = self._model(torch.as_tensor(x, dtype=torch.float32)).squeeze(-1)
        return out.numpy()


class RandomSurvivalForest(BaseSurvivalModel):
    """Random survival forest (scikit-survival).

    Included because it is the strongest tabular model on small radiomics
    cohorts and is the baseline most likely to be raised in review.
    """

    name = "rsf"

    def __init__(
        self,
        n_estimators: int = 500,
        min_samples_leaf: int = 10,
        max_features: str = "sqrt",
        seed: int = 0,
    ) -> None:
        self.kwargs = dict(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=seed,
            n_jobs=-1,
        )
        self._model = None

    def fit(self, x: np.ndarray, time: np.ndarray, event: np.ndarray):
        from sksurv.ensemble import RandomSurvivalForest as _RSF
        from sksurv.util import Surv

        y = Surv.from_arrays(event=event.astype(bool), time=time)
        self._model = _RSF(**self.kwargs).fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(x)).ravel()


REGISTRY: dict[str, type[BaseSurvivalModel]] = {
    "clinical_cox": ClinicalCox,
    "radiomics_coxph": RadiomicsCoxPH,
    "cox_nnet": CoxNnet,
    "rsf": RandomSurvivalForest,
}


def build_baseline(name: str, seed: int = 0) -> BaseSurvivalModel:
    if name not in REGISTRY:
        raise ValueError(f"unknown baseline {name!r}; expected one of {list(REGISTRY)}")
    cls = REGISTRY[name]
    try:
        return cls(seed=seed)          # type: ignore[call-arg]
    except TypeError:
        return cls()
