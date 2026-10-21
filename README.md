# Air Quality Forecasting via Gradient Boosting and LSTM Models

Forecasting PM2.5 concentration one hour ahead using the UCI Beijing Multi-Site Air Quality dataset (12 monitoring stations, 2013–2017).

Phase 1 establishes a reproducible, leakage-free benchmark pipeline utilizing Persistence baselines, Linear Regression, and Gradient Boosting Regressors. (Deep learning LSTM models are reserved for Phase 2).

---

## Team Roster & Responsibilities

| Member | Primary Responsibilities | Owned Core Modules |
|:---|:---|:---|
| **Saksham** | Data loading, validation, causal forward-fill cleaning, EDA | `src/data/load_data.py`, `src/preprocessing.py`, `notebooks/01_eda.ipynb` |
| **Ajit** | Leakage-safe time-series feature engineering, chronological splitting | `src/features.py`, `src/split_data.py`, `notebooks/02_feature_engineering.ipynb` |
| **Harsh** | Baseline models (naive & constant persistence, linear regression) & Gradient Boosting | `src/models.py`, `experiments/configs/gradient_boosting.json` |
| **Mihika** | Evaluation metrics suite, stratified error analysis, publication figures | `src/evaluate.py`, `src/visualization.py`, `notebooks/04_model_evaluation.ipynb` |
| **Priyanshu** | ML Pipeline integration, configuration management, reproducibility, test suite | `src/pipeline.py`, `src/config.py`, `experiments/run_experiments.py`, `tests/test_pipeline.py`, `docs/reproducibility.md` |

---

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── load_data.py          # Data loading, continuous timestamps, validation
│   ├── preprocessing.py          # Causal forward-fill imputation, wind encoding
│   ├── features.py               # AR lags, multi-pollutant lags, rolling stats, domain features
│   ├── split_data.py             # Chronological 70/15/15 splitting & anti-leakage audit
│   ├── models.py                 # Persistence baselines, Linear Regression, Gradient Boosting
│   ├── evaluate.py               # MAE, RMSE, R², MBE, stratified error analysis
│   ├── visualization.py          # Publication-quality figures (300 DPI, headless Agg backend)
│   ├── config.py                 # Typed configuration dataclasses for entire pipeline
│   └── pipeline.py               # Master end-to-end pipeline orchestrator & CLI
├── experiments/
│   ├── configs/
│   │   └── gradient_boosting.json # Hyperparameter configuration for Gradient Boosting
│   └── run_experiments.py       # Experiment runner CLI for single stations & 12-station sweeps
├── tests/
│   ├── test_data.py              # Unit tests for data loading & timestamps
│   ├── test_preprocessing.py     # Unit tests for causal imputation & wind encoding
│   ├── test_features.py          # Unit tests for time-series features & anti-leakage
│   ├── test_split_data.py        # Unit tests for chronological splitting & boundary checks
│   ├── test_evaluate.py          # Unit tests for metrics & error stratification
│   ├── test_visualization.py     # Unit tests for figure generation
│   └── test_pipeline.py          # End-to-end integration tests & schema validation
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb # Feature Engineering Analysis
│   └── 04_model_evaluation.ipynb # Phase 1 Model Evaluation & Error Analysis
├── reports/
│   ├── figures/                  # 18 publication-quality figures
│   └── results/                  # Structured CSV & Markdown comparison tables & metadata
├── docs/
│   ├── data_dictionary.md        # Column-level schema & units
│   ├── eda_findings.md           # Exploratory analytical findings & distributions
│   ├── feature_engineering.md    # Feature formulations & physical interpretations
│   ├── evaluation_protocol.md    # Evaluation metrics definitions & standards
│   └── reproducibility.md        # Integration checklist, anti-leakage safeguards & clean-env guide
├── data/
│   └── README.md                 # Dataset download and directory structure instructions
├── requirements.txt              # Pinned/versioned project dependencies
└── README.md
```

---

## Setup & Clean-Environment Verification

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Dataset Preparation
Place the 12 UCI Beijing Multi-Site Air Quality CSV files in `data/raw/`. Follow the download commands in [data/README.md](data/README.md):
```bash
mkdir -p data/raw
curl -o data/raw/air_quality.zip "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"
unzip -q data/raw/air_quality.zip -d data/raw_extracted/
mv data/raw_extracted/PRSA_Data_20130301-20170228/*.csv data/raw/
rm -rf data/raw/air_quality.zip data/raw_extracted/
```

### 3. Running Automated Tests
Execute the full test suite (60+ unit and integration tests):
```bash
pytest -v
```

---

## Execution Guide

### Run End-to-End Phase 1 Pipeline
Run the full 8-stage pipeline for a monitoring station (data loading $\to$ causal cleaning $\to$ feature engineering $\to$ chronological splitting $\to$ model training $\to$ validation selection $\to$ final test evaluation $\to$ result table and figure export $\to$ reproducibility metadata):

```bash
# Default run on primary benchmark station (Aotizhongxin)
python -m src.pipeline --station Aotizhongxin

# Run with hyperparameter grid search enabled for Gradient Boosting
python -m src.pipeline --station Aotizhongxin --tune-gb

# Run on a custom CSV file or specific output directory
python -m src.pipeline --data-file path/to/station.csv --output-dir reports/results
```

### Run Experiment Sweeps
Use the dedicated experiment runner for comparative studies:

```bash
# Run single station experiment
python experiments/run_experiments.py --station Tiantan

# Sweep across all 12 Beijing monitoring stations and generate aggregated benchmark table
python experiments/run_experiments.py --all-stations
```

---

## Phase 1 Benchmark Results Summary (Aotizhongxin Station)

### Final Test Partition Performance (Untouched Test Period, N=4,920)

| Model | MAE (µg/m³) | RMSE (µg/m³) | R² | MBE (µg/m³) | MAE Improvement vs. Persistence |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Gradient Boosting (Tuned)** | **10.2280** | **19.4863** | **0.9576** | **-0.5699** | **+7.72%** |
| Gradient Boosting (Default) | 10.6596 | 20.0683 | 0.9550 | -0.5517 | +3.82% |
| Linear Regression | 10.8088 | 19.5187 | 0.9574 | -0.5931 | +2.47% |
| **Persistence (Naive Lag-1)** | 11.0831 | 21.0616 | 0.9504 | +0.1254 | 0.0% (Baseline) |
| Persistence (Constant Mean) | 70.9329 | 113.9054 | -0.4496 | -63.4366 | -540.01% |

### 12-Station Network Benchmark Summary (Untouched Test Period, N=58,201 Across 12 Sites)

| Model | Macro MAE (µg/m³) | Pooled MAE (µg/m³) | Macro RMSE (µg/m³) | Pooled RMSE (µg/m³) | Macro R² | Macro MBE (µg/m³) | Macro MAE Imprv. vs Persistence | Inter-Station Std MAE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Gradient Boosting (Tuned)** | **10.0845** | **10.0863** | **20.1496** | **20.2691** | **0.9522** | **-0.0510** | **+5.92%** | ±0.95 |
| Gradient Boosting (Default) | 10.2872 | 10.2895 | 20.3508 | 20.4625 | 0.9512 | +0.0257 | +4.02% | ±0.94 |
| Linear Regression | 10.5147 | 10.5163 | 19.8596 | 19.9479 | 0.9534 | -0.1200 | +1.81% | ±0.88 |
| **Persistence (Naive Lag-1)** | 10.7237 | 10.7240 | 21.3120 | 21.4066 | 0.9466 | +0.0861 | 0.0% (Baseline) | ±1.01 |
| Persistence (Constant Mean) | 69.3400 | 69.3241 | 111.4981 | 112.1734 | -0.4509 | -61.6829 | -545.39% | ±8.76 |

For complete station-by-station breakdowns, metadata provenance, and publication figures, see `reports/results/multi_station_sweep/` and `reports/figures/multi_station_sweep/`.
For error stratification across pollution concentration tiers (e.g. severe smog episodes > 250 µg/m³) and diurnal profiles, see [docs/reproducibility.md](docs/reproducibility.md) and [docs/evaluation_protocol.md](docs/evaluation_protocol.md).
