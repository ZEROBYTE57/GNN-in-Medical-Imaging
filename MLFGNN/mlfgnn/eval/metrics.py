"""Survival metrics.

Harrell's C is the headline number in Peng et al. and is what the reproduction
target (0.788 / 0.674) refers to. The protocol adds three metrics that Harrell's
C cannot express and that matter specifically under domain shift:

* **Uno's C** -- inverse-probability-of-censoring weighted, so it does not drift
  when the held-out centre has a different censoring distribution from the
  training centres. With four centres whose follow-up regimes differ, an
  apparent C-index gap can be a censoring artefact; Uno's C is the check.
* **Time-dependent AUC** at 12/24/60 months -- the endpoints Peng et al. report.
* **Integrated Brier score** -- a proper scoring rule, so it penalises a model
  that ranks well but is badly calibrated on a new site. Discrimination alone
  does not establish transportability.

Harrell's C is implemented here directly; the other three come from
scikit-survival, which is the standard implementation reviewers will expect.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

from ..features.select import comparable_pairs, concordance_from_pairs

logger = logging.getLogger(__name__)

#: Missing-dependency notices fire once per process, not once per fold x arm x
#: metric -- otherwise a grid run buries its own progress log.
_WARNED: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(message)


def harrell_cindex(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Harrell's concordance index. Higher ``risk`` must mean shorter survival."""
    i_idx, j_idx = comparable_pairs(np.asarray(time), np.asarray(event))
    if len(i_idx) == 0:
        return float("nan")
    return float(concordance_from_pairs(np.asarray(risk), i_idx, j_idx)[0])


def _surv(time: np.ndarray, event: np.ndarray):
    from sksurv.util import Surv

    return Surv.from_arrays(event=np.asarray(event).astype(bool), time=np.asarray(time))


def uno_cindex(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    train_time: np.ndarray,
    train_event: np.ndarray,
    tau: float | None = None,
) -> float:
    """Uno's IPCW concordance, with the censoring model fit on TRAINING data."""
    try:
        from sksurv.metrics import concordance_index_ipcw
    except ImportError:
        _warn_once("uno", "scikit-survival missing -- Uno's C unavailable")
        return float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = concordance_index_ipcw(
                _surv(train_time, train_event),
                _surv(time, event),
                np.asarray(risk),
                tau=tau,
            )
        return float(result[0])
    except (ValueError, ZeroDivisionError) as exc:
        logger.warning("Uno's C failed: %s", exc)
        return float("nan")


def time_dependent_auc(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    train_time: np.ndarray,
    train_event: np.ndarray,
    times: list[float],
) -> dict[str, float]:
    """Cumulative/dynamic AUC at each requested horizon."""
    try:
        from sksurv.metrics import cumulative_dynamic_auc
    except ImportError:
        _warn_once("auc", "scikit-survival missing -- time-dependent AUC unavailable")
        return {f"auc_{t:g}m": float("nan") for t in times}

    valid = _valid_times(times, time, event, train_time)
    if not valid:
        return {f"auc_{t:g}m": float("nan") for t in times}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auc, _ = cumulative_dynamic_auc(
                _surv(train_time, train_event), _surv(time, event),
                np.asarray(risk), np.asarray(valid),
            )
        out = {f"auc_{t:g}m": float(a) for t, a in zip(valid, auc, strict=True)}
    except (ValueError, ZeroDivisionError) as exc:
        logger.warning("time-dependent AUC failed: %s", exc)
        out = {}
    for t in times:
        out.setdefault(f"auc_{t:g}m", float("nan"))
    return out


def integrated_brier_score(
    survival_probs: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    train_time: np.ndarray,
    train_event: np.ndarray,
    times: np.ndarray,
) -> float:
    """IBS over ``times``; ``survival_probs`` is ``(n_samples, n_times)``."""
    try:
        from sksurv.metrics import integrated_brier_score as _ibs
    except ImportError:
        _warn_once("ibs", "scikit-survival missing -- IBS unavailable")
        return float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(
                _ibs(
                    _surv(train_time, train_event),
                    _surv(time, event),
                    survival_probs,
                    np.asarray(times),
                )
            )
    except (ValueError, ZeroDivisionError) as exc:
        logger.warning("IBS failed: %s", exc)
        return float("nan")


def _valid_times(
    times: list[float], time: np.ndarray, event: np.ndarray, train_time: np.ndarray
) -> list[float]:
    """Horizons that are inside the follow-up support of both sets.

    scikit-survival raises rather than extrapolating, and with four centres of
    differing follow-up a 60-month horizon is often outside one of them. Times
    that do not qualify are reported as NaN and aggregated with nanmean, which
    is the honest treatment -- dropping the fold silently would bias the mean.
    """
    t_min = max(float(np.min(time)), float(np.min(train_time)))
    t_max = min(float(np.max(time[np.asarray(event) == 1], initial=0.0)),
                float(np.max(train_time)))
    return [t for t in times if t_min < t < t_max]


def breslow_survival(
    risk_train: np.ndarray,
    time_train: np.ndarray,
    event_train: np.ndarray,
    risk_test: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Survival curves from log-hazards via the Breslow baseline estimator.

    The Cox head emits a relative log-hazard, which has no absolute scale, so
    the Brier score needs a baseline cumulative hazard. It is estimated on the
    TRAINING fold and applied to the held-out centre -- deliberately, since
    borrowing the held-out centre's own baseline would hide exactly the kind of
    miscalibration this metric is meant to expose.
    """
    order = np.argsort(time_train)
    t_sorted = np.asarray(time_train)[order]
    e_sorted = np.asarray(event_train)[order]
    r_sorted = np.exp(np.asarray(risk_train)[order] - np.max(risk_train))

    # Breslow increment at each observed event time: d_i / sum_{j in R_i} exp(r_j)
    denom = np.cumsum(r_sorted[::-1])[::-1]
    increments = np.where(e_sorted == 1, 1.0 / np.maximum(denom, 1e-12), 0.0)
    cum_hazard = np.cumsum(increments)

    baseline = np.interp(times, t_sorted, cum_hazard, left=0.0)
    scaled = np.exp(np.asarray(risk_test) - np.max(risk_train))
    return np.exp(-np.outer(scaled, baseline))


def evaluate_survival(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    train_risk: np.ndarray | None = None,
    train_time: np.ndarray | None = None,
    train_event: np.ndarray | None = None,
    eval_times: list[float] | None = None,
) -> dict[str, float]:
    """All metrics for one evaluation set, as a flat dict."""
    eval_times = eval_times or [12.0, 24.0, 60.0]
    out: dict[str, float] = {"harrell_c": harrell_cindex(risk, time, event)}

    if train_time is None or train_event is None:
        return out

    out["uno_c"] = uno_cindex(risk, time, event, train_time, train_event)
    out.update(time_dependent_auc(risk, time, event, train_time, train_event, eval_times))

    if train_risk is not None:
        grid = _brier_grid(time, event, train_time)
        if grid.size >= 2:
            surv = breslow_survival(train_risk, train_time, train_event, risk, grid)
            out["ibs"] = integrated_brier_score(
                surv, time, event, train_time, train_event, grid
            )
        else:
            out["ibs"] = float("nan")
    return out


def _brier_grid(time: np.ndarray, event: np.ndarray, train_time: np.ndarray,
                n_points: int = 25) -> np.ndarray:
    """A time grid strictly inside the support both sets share."""
    events = np.asarray(time)[np.asarray(event) == 1]
    if events.size < 2:
        return np.array([])
    lo = max(float(np.min(time)), float(np.min(train_time)))
    hi = min(float(np.max(events)), float(np.max(train_time)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.array([])
    span = hi - lo
    return np.linspace(lo + 0.02 * span, hi - 0.02 * span, n_points)

