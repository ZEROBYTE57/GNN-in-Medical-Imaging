"""The leave-one-centre-out benchmark.

Full grid: ``graph_mode x k x layer_type x fold x seed``, plus the non-graph
baselines once per fold and seed. Every arm sees exactly the same folds, the
same selected features and the same trained-on patients, so any difference in
the held-out-centre C-index is attributable to the edge rule alone.

Two numbers come out of each run, and the second is the point of the study:

    in_centre        C-index on the held-out 20% of the TRAINING centres
    held_out_centre  C-index on the centre that was never seen
"""

from __future__ import annotations

import itertools
import logging
import time as _time

import numpy as np
import pandas as pd

from ..data.clinical import encode_clinical
from ..device import describe_device, resolve_device
from ..eval.metrics import evaluate_survival
from ..graphs import metrics as gmetrics
from ..models.baselines import build_baseline
from ..seed import set_seed
from .fold import FoldSpec, build_fold_graphs, fold_similarity, make_folds, prepare_fold
from .train import score_graph, train_mlfgnn

logger = logging.getLogger(__name__)


def _positions(subset: np.ndarray, universe: np.ndarray) -> np.ndarray:
    """Positions of ``subset`` within ``universe`` (both are global row indices)."""
    lookup = {int(v): i for i, v in enumerate(universe)}
    return np.array([lookup[int(v)] for v in subset], dtype=int)


def _arms(cfg_graphs: dict) -> list[tuple[str, int | None]]:
    """(mode, k) pairs to run. The threshold rule has no k."""
    modes = cfg_graphs.get("modes", ["snf_threshold", "knn", "centre_balanced_knn"])
    k_values = cfg_graphs.get("knn", {}).get("k_values", [5])
    out: list[tuple[str, int | None]] = []
    for mode in modes:
        if mode == "snf_threshold":
            out.append((mode, None))
        else:
            out += [(mode, int(k)) for k in k_values]
    return out


def run_loco(
    df: pd.DataFrame,
    cfg,
    device="auto",
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full benchmark.

    Returns ``(results, diagnostics)`` -- one row per (arm, fold, seed, split)
    and one row per (arm, fold, seed) graph respectively.
    """
    device = resolve_device(device) if isinstance(device, str) else device
    logger.info("device: %s", describe_device(device))

    data_cfg = cfg.section("data")
    sel_cfg = cfg.section("selection")
    graph_cfg = cfg.section("graphs")
    model_cfg = cfg.section("model")
    train_cfg = cfg.section("train")
    proto = cfg.section("protocol")

    id_col = data_cfg.get("id_col", "patient_id")
    centre_col = data_cfg.get("centre_col", "centre")
    time_col = data_cfg.get("time_col", "PFS_time")
    event_col = data_cfg.get("event_col", "PFS_event")
    pet_prefix = data_cfg.get("pet_prefix", "PET_")
    ct_prefix = data_cfg.get("ct_prefix", "CT_")
    clinical_cols = data_cfg.get("clinical_cols", [])

    feature_cols = [c for c in df.columns if c.startswith((pet_prefix, ct_prefix))]
    features = df[feature_cols].to_numpy(dtype=float)
    time_arr = df[time_col].to_numpy(dtype=float)
    event_arr = df[event_col].to_numpy(dtype=float)
    centres = df[centre_col].to_numpy()

    seeds = proto.get("seeds", [0])
    eval_times = proto.get("eval_times", [12, 24, 60])
    inner_frac = proto.get("inner_val_fraction", 0.2)
    snf_cfg = graph_cfg.get("snf", {})
    symmetrise = graph_cfg.get("symmetrise", "union")
    sim_scope = graph_cfg.get("similarity_scope", "train_fit")
    arms = _arms(graph_cfg)
    layer_types = model_cfg.get("layer_types", ["GAT"])

    results: list[dict] = []
    diagnostics: list[dict] = []
    total = len(seeds) * len(set(centres.tolist())) * (len(arms) * len(layer_types) + 1)
    done = 0
    started = _time.time()

    for seed in seeds:
        set_seed(seed)
        for spec in make_folds(centres, seed=seed, inner_val_fraction=inner_frac):
            fold_data = prepare_fold(
                features=features,
                columns=feature_cols,
                pet_prefix=pet_prefix,
                ct_prefix=ct_prefix,
                clinical_df=df,
                clinical_cols=clinical_cols,
                time=time_arr,
                event=event_arr,
                centres=centres,
                spec=spec,
                selection_cfg=sel_cfg,
            )

            # Similarity is computed once per fold and reused by every arm, so
            # the arms differ only in how edges are drawn from it.
            s_train, cross = fold_similarity(
                fold_data,
                k_snf=snf_cfg.get("k_neighbors", 20),
                mu=snf_cfg.get("mu", 0.5),
                n_iters=snf_cfg.get("n_iters", 20),
                scope=sim_scope,
            )

            gi = spec.graph_idx
            train_pos = _positions(spec.train_idx, gi)
            val_pos = _positions(spec.val_idx, gi)

            for (mode, k), layer_type in itertools.product(arms, layer_types):
                k_eff = k if k is not None else 0
                train_graph, eval_graph, node_order = build_fold_graphs(
                    fold_data,
                    s_train,
                    cross,
                    mode=mode,
                    k=k_eff,
                    threshold=snf_cfg.get("threshold", 0.2),
                    symmetrise=symmetrise,
                )

                diagnostics.append(
                    {
                        "seed": seed,
                        "fold": spec.held_out_centre,
                        "graph_mode": mode,
                        "k": k,
                        **gmetrics.describe(
                            eval_graph,
                            fold_data.centres[node_order],
                            time=time_arr[node_order],
                            event=event_arr[node_order],
                            train_mask=np.isin(node_order, gi),
                        ),
                    }
                )

                trained = train_mlfgnn(
                    x_pet=fold_data.x_pet[gi],
                    x_ct=fold_data.x_ct[gi],
                    clinical=fold_data.clinical[gi],
                    time=time_arr[gi],
                    event=event_arr[gi],
                    graph=train_graph,
                    train_pos=train_pos,
                    val_pos=val_pos,
                    layer_type=layer_type,
                    fusion=model_cfg.get("fusion", "network"),
                    hidden_dim=model_cfg.get("hidden_dim", 32),
                    heads=model_cfg.get("heads", 4),
                    dropout=model_cfg.get("dropout", 0.5),
                    use_clinical=model_cfg.get("use_clinical", True),
                    epochs=train_cfg.get("epochs", 100),
                    lr=train_cfg.get("lr", 0.005),
                    weight_decay=train_cfg.get("weight_decay", 0.005),
                    patience=train_cfg.get("patience", 30),
                    min_epochs=train_cfg.get("min_epochs", 20),
                    seed=seed,
                    device=device,
                )

                use_clin = model_cfg.get("use_clinical", True)
                scores_train_graph = score_graph(
                    trained.model, fold_data.x_pet[gi], fold_data.x_ct[gi],
                    fold_data.clinical[gi], train_graph, use_clin, device,
                )
                scores_eval_graph = score_graph(
                    trained.model,
                    fold_data.x_pet[node_order], fold_data.x_ct[node_order],
                    fold_data.clinical[node_order], eval_graph, use_clin, device,
                )
                test_scores = scores_eval_graph[len(gi):]

                base = {
                    "model": f"MLF-{layer_type}",
                    "graph_mode": mode,
                    "k": k,
                    "fold": spec.held_out_centre,
                    "seed": seed,
                    "best_epoch": trained.best_epoch,
                }
                results.append(
                    {
                        **base,
                        "split": "in_centre",
                        **evaluate_survival(
                            scores_train_graph[val_pos],
                            time_arr[spec.val_idx], event_arr[spec.val_idx],
                            train_risk=scores_train_graph[train_pos],
                            train_time=time_arr[spec.train_idx],
                            train_event=event_arr[spec.train_idx],
                            eval_times=eval_times,
                        ),
                    }
                )
                held_out_row = {
                    **base,
                    "split": "held_out_centre",
                    **evaluate_survival(
                        test_scores,
                        time_arr[spec.test_idx], event_arr[spec.test_idx],
                        train_risk=scores_train_graph[train_pos],
                        train_time=time_arr[spec.train_idx],
                        train_event=event_arr[spec.train_idx],
                        eval_times=eval_times,
                    ),
                }
                results.append(held_out_row)
                results.extend(
                    _per_patient_rows(base, spec, test_scores, scores_train_graph[val_pos])
                )

                done += 1
                if progress:
                    _log_progress(done, total, started, spec, mode, k, layer_type, held_out_row)

            results.extend(
                _run_baselines(
                    cfg, fold_data, df, clinical_cols, time_arr, event_arr,
                    spec, seed, eval_times,
                )
            )
            done += 1

    res = pd.DataFrame([r for r in results if "split" in r])
    res.attrs["device"] = describe_device(device)
    scores_df = pd.DataFrame([r for r in results if "split" not in r])
    if len(scores_df):
        res.attrs["patient_scores"] = scores_df
    return res, pd.DataFrame(diagnostics)


def _per_patient_rows(base, spec, test_scores, val_scores) -> list[dict]:
    """Per-patient risk scores, kept for the paired bootstrap.

    The bootstrap needs the individual scores from every arm on the same
    patients; recomputing them later would mean retraining, so they ride along
    in a side table.
    """
    rows = []
    for idx, score in zip(spec.test_idx, test_scores, strict=True):
        rows.append({**base, "row": int(idx), "risk": float(score), "on": "held_out_centre"})
    for idx, score in zip(spec.val_idx, val_scores, strict=True):
        rows.append({**base, "row": int(idx), "risk": float(score), "on": "in_centre"})
    return rows


def _run_baselines(
    cfg, fold_data, df, clinical_cols, time_arr, event_arr, spec, seed, eval_times
) -> list[dict]:
    """Non-graph comparators on the same fold, same features, same split."""
    out: list[dict] = []
    names = cfg.get("baselines", []) or []
    if not names:
        return out

    radiomics = np.hstack([fold_data.x_pet, fold_data.x_ct])
    clinical = np.zeros((len(time_arr), 0))
    if clinical_cols:
        clinical, _ = encode_clinical(df, clinical_cols, reference=df.iloc[spec.train_idx])

    inputs = {
        "clinical_cox": clinical,
        "radiomics_coxph": np.hstack([radiomics, clinical]),
        "cox_nnet": np.hstack([radiomics, clinical]),
        "rsf": np.hstack([radiomics, clinical]),
    }

    for name in names:
        x = inputs.get(name, radiomics)
        if x.shape[1] == 0:
            logger.warning("baseline %s has no input features -- skipped", name)
            continue
        try:
            model = build_baseline(name, seed=seed).fit(
                x[spec.train_idx], time_arr[spec.train_idx], event_arr[spec.train_idx]
            )
            train_risk = model.predict(x[spec.train_idx])
            for split, idx in (("in_centre", spec.val_idx), ("held_out_centre", spec.test_idx)):
                out.append(
                    {
                        "model": name,
                        "graph_mode": "none",
                        "k": None,
                        "fold": spec.held_out_centre,
                        "seed": seed,
                        "split": split,
                        **evaluate_survival(
                            model.predict(x[idx]), time_arr[idx], event_arr[idx],
                            train_risk=train_risk,
                            train_time=time_arr[spec.train_idx],
                            train_event=event_arr[spec.train_idx],
                            eval_times=eval_times,
                        ),
                    }
                )
        except Exception as exc:  # noqa: BLE001 - one broken baseline must not kill the grid
            logger.error("baseline %s failed on fold %s: %s", name, spec.held_out_centre, exc)
    return out


def _log_progress(done, total, started, spec, mode, k, layer_type, last_row) -> None:
    elapsed = _time.time() - started
    eta = (elapsed / max(done, 1)) * (total - done)
    logger.info(
        "[%3d/%3d] fold=%-5s %-20s k=%-4s %-12s  held-out C=%.3f  (eta %.0fs)",
        done, total, spec.held_out_centre, mode, k, layer_type,
        last_row.get("harrell_c", float("nan")), eta,
    )
