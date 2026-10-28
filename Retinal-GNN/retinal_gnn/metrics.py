"""Evaluation metrics for retinal vessel segmentation.

Overlap metrics (pixel level, inside the field of view):
    accuracy, sensitivity, specificity, precision, F1/Dice, MCC, AUC
Topology metrics:
    clDice (Shit et al., CVPR 2021)
    Betti-0 error  : |#connected components(pred) - #connected components(gt)|
    Betti-1 error  : |#holes(pred) - #holes(gt)|

Notes
-----
* On a binarised mask, F1 and Dice are the SAME quantity. We report it once
  ("dice") and do not present F1 as a separate metric.
* All pixel metrics are computed only inside the field-of-view (FOV) mask,
  the usual protocol on DRIVE / STARE / CHASE_DB1 / HRF.
* Topology metrics are computed on the FOV-cropped binary masks, 8-connectivity
  for the vessel foreground.
"""
from __future__ import annotations

import numpy as np
from skimage.measure import euler_number, label
from skimage.morphology import skeletonize
from sklearn.metrics import roc_auc_score

EPS = 1e-8


def _as_bool(x) -> np.ndarray:
    return np.asarray(x).astype(bool)


def pixel_metrics(pred_bin, gt, fov=None) -> dict:
    """Confusion-matrix metrics inside the FOV."""
    pred_bin, gt = _as_bool(pred_bin), _as_bool(gt)
    if fov is None:
        fov = np.ones_like(gt, dtype=bool)
    fov = _as_bool(fov)
    p, g = pred_bin[fov], gt[fov]
    tp = float(np.sum(p & g))
    tn = float(np.sum(~p & ~g))
    fp = float(np.sum(p & ~g))
    fn = float(np.sum(~p & g))
    sens = tp / (tp + fn + EPS)
    spec = tn / (tn + fp + EPS)
    prec = tp / (tp + fp + EPS)
    dice = 2 * tp / (2 * tp + fp + fn + EPS)
    acc = (tp + tn) / (tp + tn + fp + fn + EPS)
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / (mcc_den + EPS)
    return dict(accuracy=acc, sensitivity=sens, specificity=spec,
                precision=prec, dice=dice, mcc=mcc)


def auc_score(prob, gt, fov=None) -> float:
    """ROC-AUC of the probability map inside the FOV."""
    gt = _as_bool(gt)
    if fov is None:
        fov = np.ones_like(gt, dtype=bool)
    fov = _as_bool(fov)
    g = gt[fov]
    if g.min() == g.max():  # only one class present: AUC undefined
        return float("nan")
    return float(roc_auc_score(g, np.asarray(prob)[fov]))


def cl_dice(pred_bin, gt, fov=None) -> float:
    """Centerline Dice (clDice), Shit et al. 2021.

    Tprec = |skel(P) & G| / |skel(P)|   (topology precision)
    Tsens = |skel(G) & P| / |skel(G)|   (topology sensitivity)
    clDice = 2 * Tprec * Tsens / (Tprec + Tsens)
    """
    pred_bin, gt = _as_bool(pred_bin), _as_bool(gt)
    if fov is not None:
        fov = _as_bool(fov)
        pred_bin, gt = pred_bin & fov, gt & fov
    sp, sg = skeletonize(pred_bin), skeletonize(gt)
    if sp.sum() == 0 or sg.sum() == 0:
        return 0.0
    tprec = float(np.sum(sp & gt)) / (float(sp.sum()) + EPS)
    tsens = float(np.sum(sg & pred_bin)) / (float(sg.sum()) + EPS)
    return 2 * tprec * tsens / (tprec + tsens + EPS)


def betti_numbers(mask) -> tuple[int, int]:
    """(b0, b1) of a binary mask, 8-connectivity foreground.

    b0 = number of connected components.
    b1 = number of holes, from the Euler characteristic: chi = b0 - b1.
    """
    mask = _as_bool(mask)
    b0 = int(label(mask, connectivity=2).max())
    chi = int(euler_number(mask, connectivity=2))
    b1 = b0 - chi
    return b0, b1


def betti_errors(pred_bin, gt, fov=None) -> dict:
    pred_bin, gt = _as_bool(pred_bin), _as_bool(gt)
    if fov is not None:
        fov = _as_bool(fov)
        pred_bin, gt = pred_bin & fov, gt & fov
    pb0, pb1 = betti_numbers(pred_bin)
    gb0, gb1 = betti_numbers(gt)
    return dict(betti0_pred=pb0, betti0_gt=gb0, betti0_err=abs(pb0 - gb0),
                betti1_pred=pb1, betti1_gt=gb1, betti1_err=abs(pb1 - gb1))


def evaluate_image(prob, gt, fov=None, thr: float = 0.5) -> dict:
    """All metrics for one image. `prob` is a probability map in [0, 1]."""
    prob = np.asarray(prob, dtype=np.float32)
    pred_bin = prob >= thr
    out = pixel_metrics(pred_bin, gt, fov)
    out["auc"] = auc_score(prob, gt, fov)
    out["cldice"] = cl_dice(pred_bin, gt, fov)
    out.update(betti_errors(pred_bin, gt, fov))
    return out


def aggregate(per_image: list[dict]) -> dict:
    """Mean and std over images (NaN-safe). Returns {metric: (mean, std)}."""
    keys = per_image[0].keys()
    res = {}
    for k in keys:
        v = np.array([d[k] for d in per_image], dtype=float)
        res[k] = (float(np.nanmean(v)), float(np.nanstd(v)))
    return res
