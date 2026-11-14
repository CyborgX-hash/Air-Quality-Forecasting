# Phase 2 Integration Design

**Role:** Integration Engineer, Testing and Reproducibility (Priyanshu Agrahari)

This document explains how the five Phase 2 modules connect, which interface each
connection relies on, and which integration risks were found. No other member's module
was modified; everything below is composed *around* their public interfaces.

## 1. Data flow

```
data/raw/PRSA_Data_<station>_*.csv                         configs/phase2_config.yaml
        │                                                          │ (single source of truth)
        ▼                                                          ▼
 [prepare]  src/phase2/data_pipeline.process_station  ──►  StationDataSplits
            (Saksham) clean → features → chronological split → train-only StationScaler
        │
        ▼
 [sequences] src/phase2/sequence_generator.generate_station_sequence_bundle
            (Ajit)   StationDataSplits ──► StationSequenceBundle (X, y, timestamps, exclusion report)
        │
        ▼
 [train]    src/phase2/train.train_station_model  (Harsh)
            bundle → LSTM (lstm_model.build_lstm_model) → best-val checkpoint + history
        │
        ▼
 [predict]  src/phase2/predict.predict_and_evaluate   (Harsh)
            checkpoint + splits → scaled prediction → scaler.inverse_transform → µg/m³ (.npz)
        │
        ▼
 [evaluate] src/phase2/evaluate.evaluate_all_stations (Mihika)
            station-wise, macro, pooled, tier/diurnal/monthly, Phase 1 comparison
            + src/phase2/integrity.py quality gates (Priyanshu)
        │
        ▼
 [visualize] src/phase2/visualize.generate_all_evaluation_figures (Mihika) ─► [report] report.md
```

`src/phase2/run_pipeline.py` drives the stages; `src/phase2/experiment_manager.py` owns the
run directory, seeding, config snapshot and manifest.

## 2. Interface contracts relied upon

| Producer → consumer | Contract | How it is enforced |
|---|---|---|
| Config → all modules | one YAML; `Phase2Config.from_dict` ignores the extra `experiment/evaluation/reproducibility` sections, which `ExperimentSettings` reads | `test_integration_config.py` |
| Saksham → Ajit | `StationDataSplits` fields `train_scaled`, `val/test_scaled_with_context`, `boundaries`, `scaler`, `feature_cols` | `test_integration_sequences.py` |
| Saksham → everyone | feature column order identical in all 12 stations and equal to `config.get_feature_list()` | `check_feature_order_consistent` |
| Ajit → Harsh | `bundle.X_train (N,W,D)`, `y_train (N,5)`, horizons `[1,3,6,12,24]`; `model.output_units == len(horizons)` | `test_integration_model.py` |
| Harsh → Mihika | checkpoint file name `lstm_<station>_w<W>_u<U>_l<L>_best.keras`; unscaled predictions via `splits.inverse_transform_target` | `evaluator_bound_to_run`, `check_predictions_match_saved` |
| Mihika ↔ Phase 1 | all metrics come from `src.evaluate.calculate_metrics`; MBE = mean(pred − true) | `check_metrics_follow_phase1` |

### Adapter: `evaluator_bound_to_run`
`evaluate_single_station` resolves its checkpoint from a hard-coded `u64_l1` path and calls
`process_station(station)` with the *default* config and data directory. To evaluate arbitrary
runs without editing Mihika's file, the orchestrator temporarily rebinds those two names to the
run's own checkpoints and station splits, then restores them (tested). The numbers therefore
come from Mihika's unmodified evaluation code.

## 3. Run directory layout (`experiments/phase2_runs/<run_id>/`)

`run_id = <run_name>_<UTC timestamp>_<config hash>`, e.g. `phase2_20261114T101500Z_9d24b223`.

```
config.yaml            effective configuration (after CLI overrides)
manifest.json          config hash, git commit/dirty flag, library+hardware versions,
                       seed, raw-data SHA-256, per-stage status and timings
scalers/               scaler_<station>.joblib   (fitted on train only)
checkpoints/           lstm_<station>_w<W>_u<U>_l<L>_best.keras
history/               *_history.json|csv, *_metadata.json, training_summary.csv
predictions/           <station>_w<W>_test.npz (y_true, y_pred in µg/m³, timestamps)
evaluation/            Mihika's CSV/Markdown tables
figures/               8 evaluation figures + per-station loss curves
audit/                 data_split_audit.csv, sequence_audit.json, integrity_*.json
integrity_checks.json  final quality gates
report.md              generated only from the files above
```

## 4. Quality gates (`src/phase2/integrity.py`)

All 12 stations accounted for · every horizon evaluated · chronological split (train < val < test) ·
scaler statistics equal train-only statistics · inverse transform round-trip · identical feature
order · no cross-station datasets · no target outside its partition · ground truth values are
observed (never imputed) PM2.5 · metrics reproduce `src.evaluate` · macro = mean of stations and
pooled RMSE from pooled squared errors · checkpoint selection used the validation partition ·
saved predictions equal the predictions the evaluator used.

Gates are tested with negative controls (shuffled split, leaky scaler, tampered ground truth).
A subset run (`--stations`) reports the 12-station gate as `SKIP`, never `PASS`.

## 5. Integration findings (not changed; owners should decide)

1. **Timestamp label offset between generators.** Ajit's `input_end_timestamps` is the window-end
   time *t*; Saksham's `get_*_sequences` (used by predict/evaluate) returns *t + 1 h*. The samples and targets are
   identical (asserted in `test_generators_agree_on_samples_with_one_hour_label_offset`), only the label differs.
2. **Diurnal/monthly profiles use one timestamp for all horizons.** `evaluate_diurnal_profile`
   bins every horizon by the hour of the 1-hour-ahead target, so for h = 3, 6, 12, 24 the
   `hour_of_day` is shifted by h − 1 hours relative to the hour actually forecast.
3. **Hard-coded values in modules.** `evaluate.py` (`u64_l1` path, `"LSTM (Champion W72)"` label,
   `Phase2Config.get_default_config()`), `visualize.py` (history path), `train.py`
   (ReduceLROnPlateau `factor`/`min_lr` ignore `training.reduce_lr_factor/min_lr`), and
   `config.py` (`raw_dir` default `"Data/Raw"`).
4. **Raw-data path casing.** The YAML said `Data/Raw`, the data README says `data/raw`; fixed to
   `data/raw` in the YAML (works on case-sensitive Linux). The dataclass default is unchanged.
5. **Real-data tests have no skip guard.** Five existing tests (`test_real_station_sequence_bundle` ×2,
   three in `test_phase2_evaluate.py`) fail with `FileNotFoundError` when `data/raw` is absent.
