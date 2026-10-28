"""Cox partial likelihood loss.

Peng et al. Eq. (1) is the Cox partial log-likelihood

    l(beta, H) = sum_{i in U} ( H_i beta - log sum_{j in R_i} exp(H_j beta) )

with U the uncensored patients and R_i the risk set of i. Minimising its
negative is the training objective.

Two details that are easy to get wrong in the population-graph setting:

* the risk set must be restricted to the *training* nodes. Every node in the
  graph produces a risk score, but including held-out-centre nodes in the
  denominator would leak their survival times into the loss;
* the loss is not separable over samples, so it cannot be mini-batched over
  nodes. Full-graph, full-batch training -- as in the original -- is used here.
"""

from __future__ import annotations

import torch
from torch import Tensor


def cox_ph_loss(
    risk: Tensor,
    time: Tensor,
    event: Tensor,
    mask: Tensor | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Negative Breslow partial log-likelihood, averaged over events.

    Parameters
    ----------
    risk:
        ``(n,)`` log-hazard ratios (the Cox layer output, no activation).
    mask:
        Boolean ``(n,)`` selecting the nodes that take part in the loss.

    Returns
    -------
    Scalar loss. Returns 0 when the selected set contains no events, so a
    degenerate fold cannot produce NaNs that poison the optimiser.
    """
    if mask is not None:
        risk, time, event = risk[mask], time[mask], event[mask]

    if event.sum() == 0:
        return risk.sum() * 0.0

    # Sort by descending time so that the risk set R_i -- patients whose event
    # or last follow-up is at or after t_i -- is a prefix, making log-sum-exp a
    # cumulative operation.
    order = torch.argsort(time, descending=True)
    risk, event = risk[order], event[order]

    # Numerically stable cumulative log-sum-exp over the prefix.
    max_risk = risk.max()
    cum = torch.cumsum(torch.exp(risk - max_risk), dim=0)
    log_cum = torch.log(cum + eps) + max_risk

    uncensored = event.bool()
    ll = (risk - log_cum)[uncensored]
    return -ll.mean()


def cox_ph_loss_with_ties(
    risk: Tensor, time: Tensor, event: Tensor, mask: Tensor | None = None
) -> Tensor:
    """Breslow loss with exact handling of tied event times.

    The cumulative form above assigns tied events slightly different risk sets
    depending on sort order. With month-resolution follow-up times ties are
    common, so this variant groups them. Slower; use it for the final reported
    numbers if tie density is high.
    """
    if mask is not None:
        risk, time, event = risk[mask], time[mask], event[mask]
    if event.sum() == 0:
        return risk.sum() * 0.0

    max_risk = risk.max()
    exp_risk = torch.exp(risk - max_risk)

    event_times = torch.unique(time[event.bool()])
    total = risk.new_zeros(())
    n_events = 0
    for t in event_times:
        at_risk = time >= t
        dying = (time == t) & event.bool()
        d = int(dying.sum())
        log_denom = torch.log(exp_risk[at_risk].sum() + 1e-8) + max_risk
        total = total + (risk[dying].sum() - d * log_denom)
        n_events += d
    return -total / max(n_events, 1)
