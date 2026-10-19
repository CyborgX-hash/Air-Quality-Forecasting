# Air Quality Forecasting via Gradient Boosting and LSTM Models

Forecasting PM2.5 concentration one hour ahead using the UCI Beijing Multi-Site Air Quality dataset (12 monitoring stations, 2013–2017).

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── load_data.py          # Data loading, timestamp, validation
│   └── preprocessing.py          # Causal imputation, wind encoding
├── tests/
│   ├── test_data.py              # 10 unit tests for data loading
│   └── test_preprocessing.py     # 6 unit tests for preprocessing
├── notebooks/
│   └── 01_eda.ipynb              # Exploratory Data Analysis
├── reports/figures/               # 9 EDA visualizations
├── docs/
│   ├── data_dictionary.md        # Column-level schema
│   └── eda_findings.md           # Analytical findings & recommendations
└── scripts/                      # Utility scripts
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib seaborn pytest jupyter
```

## Running Tests

```bash
pytest -v tests/
```

## Dataset

Place the 12 UCI Beijing Multi-Site Air Quality CSV files in `data/raw/`. See [data/README.md](data/README.md) for details.

## Team

| Member | Responsibility |
|:---|:---|
| Saksham | Data loading, cleaning, validation, EDA |
| Ajit | Feature engineering, model training |
| Priyanshu | Model evaluation, API contracts |
