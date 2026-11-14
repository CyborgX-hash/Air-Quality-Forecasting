# Phase 2 Execution Guide

## 1. Requirements

| Item | Requirement |
|---|---|
| Python | 3.11 (verified). TensorFlow 2.21 wheels exist for 3.10-3.13; the system 3.13/3.14 interpreters were not used. |
| Libraries | `requirements.txt` (minimums) or `requirements-phase2-lock.txt` (versions the integration tests ran against) |
| Data | the 12 UCI CSVs in `data/raw/` (see `data/README.md`); **not committed** |
| Hardware | CPU is sufficient: the synthetic smoke pipeline (3 stations, 2 epochs) runs in ~15 s on an Apple-silicon laptop. Full-data training time has **not been measured** here because the raw CSVs are not in the repository; the manifest records per-stage timings and hardware for every real run. 8 GB RAM is ample (each station is ~35k rows). |

```bash
python3.11 -m venv .venv
```
```bash
.venv/bin/pip install -r requirements-phase2-lock.txt
```

## 2. Run the pipeline

Complete workflow, all stations in the config, all horizons (1, 3, 6, 12, 24 h):

```bash
.venv/bin/python -m src.phase2.run_pipeline --config configs/phase2_config.yaml
```

Useful options:

| Option | Meaning |
|---|---|
| `--stages prepare,sequences,train,predict,evaluate,visualize,report` | run any subset, always in pipeline order |
| `--run-id <id>` | reuse/resume a run directory (needed to run later stages separately) |
| `--stations A,B` | subset run (integrity gate "all 12 stations" is reported as `SKIP`) |
| `--window-size 72` | input window (hours); default `lstm_windows.default_sequence_length` |
| `--epochs N` | cap epochs (smoke tests) |
| `--raw-dir`, `--output-root`, `--seed` | override config values |
| `--checkpoint-dir models/phase2/checkpoints` | evaluate existing checkpoints instead of training (file names must match `lstm_<station>_w<W>_u<U>_l<L>_best.keras`) |

Stage-by-stage example:

```bash
.venv/bin/python -m src.phase2.run_pipeline --stages prepare,sequences --run-id exp1
```
```bash
.venv/bin/python -m src.phase2.run_pipeline --stages train --run-id exp1
```
```bash
.venv/bin/python -m src.phase2.run_pipeline --stages predict,evaluate,visualize,report --run-id exp1
```

Exit code is `1` if any integrity gate fails, `2` for invalid stage names.

## 3. Tests

```bash
.venv/bin/python -m pytest tests/phase2 -q        # integration + end-to-end (synthetic data, ~25 s)
```
```bash
.venv/bin/python -m pytest -q                      # everything
```

Normal tests never train on 12 stations; the end-to-end tests use two synthetic stations,
2 epochs. Tests marked as needing real data fail with `FileNotFoundError` until `data/raw/` is populated.

## 4. Reproducibility

* Seed: `project.random_seed` (42) → Python, NumPy, TensorFlow (`tf.keras.utils.set_random_seed`);
  `PYTHONHASHSEED` is exported (effective for child processes only).
* Optional bit-level determinism: `reproducibility.deterministic_ops: true` (slower, GPU-relevant).
  On CPU, two same-seed synthetic runs matched within 1e-4 µg/m³ (tested), but exact equality
  across hardware or TensorFlow versions is not guaranteed.
* Experiment ID = `<run_name>_<UTC timestamp>_<8-char config hash>`; the hash covers the whole
  Phase 2 config and integration settings.
* Each run stores: `config.yaml` (effective config), `manifest.json` (git commit + dirty flag,
  Python/TensorFlow/NumPy/pandas/scikit-learn versions, hardware, seed, raw-data SHA-256),
  scalers, best checkpoints, training history (JSON/CSV), predictions, evaluation tables, report.
* To reproduce a run: check out the recorded commit, install the same library versions, verify the
  raw-data hashes in `manifest.json`, then run with the saved `config.yaml`:
  `python -m src.phase2.run_pipeline --config experiments/phase2_runs/<run_id>/config.yaml`.
