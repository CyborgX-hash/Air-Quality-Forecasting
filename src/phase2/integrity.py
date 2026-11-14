"""
Phase 2 integration integrity checks.

Author: Priyanshu Agrahari (Integration Engineer, Testing and Reproducibility)

Each check inspects *actual pipeline objects* (never hard-coded results) and
returns ``{"name", "status", "detail"}`` where status is ``PASS``, ``FAIL`` or
``SKIP``. They implement the "final quality checks" of the Phase 2 brief:

* all stations accounted for / every horizon evaluated
* chronological (never random) splits
* scalers fitted on training data only
* ground truth only from observed (non-imputed) PM2.5
* no cross-station sequences
* inverse transform correctness
* model selection uses validation data only
* metrics follow the Phase 1 conventions (``src.evaluate.calculate_metrics``)
* identical feature ordering across stations
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.evaluate import calculate_metrics

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


def _res(name: str, ok: Optional[bool], detail: str) -> Dict[str, str]:
    status = SKIP if ok is None else (PASS if ok else FAIL)
    return {"name": name, "status": status, "detail": detail}


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------
def check_stations_accounted(expected: Sequence[str], actual: Iterable[str],
                             allow_subset: bool = False) -> Dict[str, str]:
    exp, act = set(expected), set(actual)
    missing, extra = sorted(exp - act), sorted(act - exp)
    if extra:
        return _res("all_stations_accounted", False, f"unexpected stations: {extra}")
    if missing:
        if allow_subset:
            return _res("all_stations_accounted", None,
                        f"subset run ({len(act)}/{len(exp)}); missing {missing}")
        return _res("all_stations_accounted", False, f"missing stations: {missing}")
    return _res("all_stations_accounted", True, f"{len(act)}/{len(exp)} stations present")


def check_horizons_evaluated(suite: Any, horizons: Sequence[int]) -> Dict[str, str]:
    det = suite.detailed_metrics_df
    problems: List[str] = []
    for st in suite.stations:
        got = set(det.loc[det["station"] == st, "horizon_hours"].astype(int))
        if got != set(horizons):
            problems.append(f"{st}: {sorted(got)}")
    pooled = set(suite.pooled_summary_df["horizon_hours"].astype(int))
    macro = set(suite.macro_summary_df["horizon_hours"].astype(int))
    if pooled != set(horizons) or macro != set(horizons):
        problems.append(f"macro={sorted(macro)} pooled={sorted(pooled)}")
    return _res("every_horizon_evaluated", not problems,
                "all stations x horizons present" if not problems else "; ".join(problems))


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------
def check_chronological_split(splits_by_station: Mapping[str, Any]) -> Dict[str, str]:
    problems = []
    for st, s in splits_by_station.items():
        tr, va, te = (d["datetime"] for d in (s.train_df, s.val_df, s.test_df))
        if not (tr.is_monotonic_increasing and va.is_monotonic_increasing
                and te.is_monotonic_increasing):
            problems.append(f"{st}: partition not time-ordered")
        if not (tr.max() < va.min() and va.max() < te.min()):
            problems.append(f"{st}: partitions overlap / out of order")
    return _res("chronological_split_no_random", not problems,
                "train < val < test for every station" if not problems else "; ".join(problems))


def check_scaler_train_only(splits_by_station: Mapping[str, Any],
                            rtol: float = 1e-6) -> Dict[str, str]:
    """Scaler statistics must equal statistics of the TRAIN partition only."""
    problems = []
    for st, s in splits_by_station.items():
        sc = s.scaler
        y_tr = s.train_df[s.target_col].dropna().to_numpy(dtype=float)
        if not sc.is_fitted:
            problems.append(f"{st}: scaler not fitted")
            continue
        if sc.scaler_type == "standard":
            ok = np.isclose(sc.target_mean_, y_tr.mean(), rtol=rtol) and \
                np.isclose(sc.target_std_, y_tr.std(), rtol=1e-4)
            fe = sc.feature_scaler.mean_
            fe_ref = s.train_df[s.feature_cols].mean().to_numpy(dtype=float)
            ok = ok and np.allclose(fe, fe_ref, rtol=1e-5, atol=1e-8)
        else:
            ok = np.isclose(sc.target_min_, y_tr.min()) and np.isclose(sc.target_max_, y_tr.max())
            ok = ok and np.allclose(sc.feature_scaler.data_min_,
                                    s.train_df[s.feature_cols].min().to_numpy(dtype=float))
        if not ok:
            problems.append(f"{st}: scaler stats differ from train-only stats")
    return _res("scaler_fit_on_train_only", not problems,
                "scaler statistics == train-partition statistics" if not problems
                else "; ".join(problems))


def check_inverse_transform_roundtrip(splits_by_station: Mapping[str, Any]) -> Dict[str, str]:
    problems = []
    for st, s in splits_by_station.items():
        y = s.val_df[s.target_col].dropna().to_numpy(dtype=float)[:500]
        if len(y) == 0:
            problems.append(f"{st}: no validation target to test")
            continue
        back = np.asarray(s.scaler.inverse_transform_target(s.scaler.transform_target(y)),
                          dtype=float).ravel()
        if not np.allclose(back, y, rtol=1e-5, atol=1e-4):
            problems.append(f"{st}: max |inverse(transform(y)) - y| = {np.max(np.abs(back - y)):.3g}")
    return _res("inverse_transform_roundtrip", not problems,
                "inverse(transform(y)) == y for every station" if not problems
                else "; ".join(problems))


def check_feature_order_consistent(splits_by_station: Mapping[str, Any],
                                   expected: Sequence[str]) -> Dict[str, str]:
    bad = [st for st, s in splits_by_station.items() if list(s.feature_cols) != list(expected)]
    return _res("feature_order_consistent", not bad,
                f"{len(expected)} features, identical order in all stations" if not bad
                else f"feature order differs from config in: {bad}")


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------
def check_no_cross_station(bundles: Mapping[str, Any]) -> Dict[str, str]:
    problems = []
    for st, b in bundles.items():
        for part in ("train", "val", "test"):
            ds = getattr(b, part)
            if ds.station_name != st:
                problems.append(f"{st}/{part}: dataset labelled {ds.station_name}")
    return _res("no_cross_station_sequences", not problems,
                "each dataset belongs to exactly one station" if not problems
                else "; ".join(problems))


def check_sequence_target_bounds(bundles: Mapping[str, Any],
                                 splits_by_station: Mapping[str, Any]) -> Dict[str, str]:
    """Every target timestamp lies inside the partition it is used for."""
    problems = []
    for st, b in bundles.items():
        bnd = splits_by_station[st].boundaries
        for part, lo, hi in (("train", None, bnd["train_end"]),
                             ("val", bnd["val_start"], bnd["val_end"]),
                             ("test", bnd["test_start"], bnd["test_end"])):
            ds = getattr(b, part)
            if len(ds) == 0:
                problems.append(f"{st}/{part}: empty")
                continue
            tt = pd.DatetimeIndex(np.asarray(ds.target_timestamps).ravel())
            if lo is not None and tt.min() < lo:
                problems.append(f"{st}/{part}: target before partition start")
            if tt.max() > hi:
                problems.append(f"{st}/{part}: target after partition end")
    return _res("sequence_targets_inside_partition", not problems,
                "no target crosses a partition boundary" if not problems else "; ".join(problems))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def check_ground_truth_observed(suite: Any, splits_by_station: Mapping[str, Any],
                                tol: float = 1e-3) -> Dict[str, str]:
    """Every evaluated y_true must be an *observed* PM2.5 value (never imputed)."""
    problems = []
    for st, rec in suite.station_records.items():
        yt = np.asarray(rec.y_true, dtype=float)
        if not np.isfinite(yt).all():
            problems.append(f"{st}: non-finite ground truth")
            continue
        s = splits_by_station[st]
        obs = np.sort(s.test_with_context_df[s.target_col].dropna().to_numpy(dtype=float))
        flat = yt.ravel()
        idx = np.clip(np.searchsorted(obs, flat), 1, len(obs) - 1)
        nearest = np.minimum(np.abs(obs[idx] - flat), np.abs(obs[idx - 1] - flat))
        if (nearest > tol).any():
            problems.append(f"{st}: {(nearest > tol).sum()} ground-truth values are not observed PM2.5")
    return _res("ground_truth_is_observed_pm25", not problems,
                "all y_true values are observed (non-imputed) PM2.5" if not problems
                else "; ".join(problems))


def check_metrics_follow_phase1(suite: Any, atol: float = 1e-3) -> Dict[str, str]:
    """Recompute every station/horizon metric with src.evaluate and compare."""
    det = suite.detailed_metrics_df.set_index(["station", "horizon_hours"])
    problems = []
    for st, rec in suite.station_records.items():
        for j, h in enumerate(rec.horizons):
            m = calculate_metrics(rec.y_true[:, j], rec.y_pred_lstm[:, j])
            row = det.loc[(st, h)]
            for k in ("mae", "rmse", "r2", "mbe"):
                if not np.isclose(m[k], row[f"lstm_{k}"], atol=atol):
                    problems.append(f"{st} {h}h {k}: {row[f'lstm_{k}']} != {m[k]}")
        # MBE sign convention: mean(pred - true)
        mbe = float(np.mean(rec.y_pred_lstm[:, 0] - rec.y_true[:, 0]))
        if not np.isclose(mbe, det.loc[(st, rec.horizons[0])]["lstm_mbe"], atol=atol):
            problems.append(f"{st}: MBE sign convention violated")
    return _res("metrics_follow_phase1_conventions", not problems,
                "all station/horizon metrics reproduce src.evaluate.calculate_metrics" if not problems
                else "; ".join(problems[:5]))


def check_macro_pooled_consistent(suite: Any, atol: float = 1e-3) -> Dict[str, str]:
    """Macro MAE = mean of station MAE; pooled MAE = obs-weighted mean (pooled RMSE from squared errors)."""
    problems = []
    det = suite.detailed_metrics_df
    for h in suite.horizons:
        sub = det[det["horizon_hours"] == h]
        macro = suite.macro_summary_df.set_index("horizon_hours").loc[h, "lstm_macro_mae"]
        if not np.isclose(macro, sub["lstm_mae"].mean(), atol=atol):
            problems.append(f"{h}h macro MAE mismatch")
        yt = np.concatenate([r.y_true[:, suite.horizons.index(h)] for r in suite.station_records.values()])
        yp = np.concatenate([r.y_pred_lstm[:, suite.horizons.index(h)] for r in suite.station_records.values()])
        pooled = suite.pooled_summary_df.set_index("horizon_hours").loc[h]
        if not np.isclose(pooled["pooled_rmse_lstm"], float(np.sqrt(np.mean((yp - yt) ** 2))), atol=atol):
            problems.append(f"{h}h pooled RMSE is not sqrt(pooled MSE)")
        if int(pooled["total_observations"]) != len(yt):
            problems.append(f"{h}h pooled observation count mismatch")
    return _res("macro_and_pooled_metrics_consistent", not problems,
                "macro = mean of stations; pooled RMSE from pooled squared errors" if not problems
                else "; ".join(problems))


def check_selection_on_validation_only(training_rows: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    if not training_rows:
        return _res("test_not_used_for_selection", None, "no training performed in this run")
    bad = [r["station"] for r in training_rows if r.get("selection_partition") != "val"]
    return _res("test_not_used_for_selection", not bad,
                "checkpoint selection used val_loss on the validation partition only" if not bad
                else f"selection partition is not 'val' for: {bad}")


def check_predictions_match_saved(suite: Any, saved: Mapping[str, Mapping[str, np.ndarray]],
                                  atol: float) -> Dict[str, str]:
    if not saved:
        return _res("predictions_match_evaluation", None, "no saved predictions to compare")
    problems = []
    for st, rec in suite.station_records.items():
        if st not in saved:
            problems.append(f"{st}: no saved predictions")
            continue
        p = saved[st]
        if p["y_pred"].shape != rec.y_pred_lstm.shape or not np.allclose(p["y_pred"], rec.y_pred_lstm, atol=atol):
            problems.append(f"{st}: saved predictions differ from evaluation inputs")
        if not np.allclose(p["y_true"], rec.y_true, atol=atol):
            problems.append(f"{st}: saved ground truth differs from evaluation inputs")
    return _res("predictions_match_evaluation", not problems,
                "saved (inverse-scaled) predictions == predictions used by the evaluator" if not problems
                else "; ".join(problems))


def summarize(results: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
    n_fail = sum(r["status"] == FAIL for r in results)
    return {
        "passed": sum(r["status"] == PASS for r in results),
        "failed": n_fail,
        "skipped": sum(r["status"] == SKIP for r in results),
        "all_ok": n_fail == 0,
        "checks": list(results),
    }


__all__ = [n for n in dir() if n.startswith("check_")] + ["summarize", "PASS", "FAIL", "SKIP"]
