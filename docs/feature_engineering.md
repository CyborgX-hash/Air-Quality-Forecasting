# Time-Series Feature Engineering & Chronological Split Specification

**Project:** Air Quality Forecasting via Gradient Boosting and LSTM Models  
**Component:** Feature Engineering & Data Splitting (Phase 1 — Traditional ML Foundation)  
**Author:** Ajit (Time-Series Feature Engineering Engineer)  
**Collaborators:** Saksham (Data Processing & Cleaning), Harsh (Gradient Boosting Modeling), Priyanshu (Evaluation & API Contracts)  
**Forecasting Task:** 1-hour ahead PM2.5 concentration ($PM2.5_{t+1}$, $\mu\text{g/m}^3$)  
**Benchmark Station:** Aotizhongxin (Site ID: Urban Benchmark; extended to all 12 Beijing monitoring stations)  
**Dataset Scope:** March 1, 2013 00:00 to February 28, 2017 23:00 (Hourly, 35,064 records/station, 420,768 records total across 12 sites)

---

## 1. Executive Summary & Problem Formulation

In Phase 1 of our five-member project, our objective is to build a high-performance, leakage-safe supervised forecasting dataset for traditional gradient boosting models (LightGBM, XGBoost, CatBoost).

### Formal Forecasting Definition
At any discrete hourly timestamp $t$, the system predicts the fine particulate matter concentration at timestamp $t + 1\ \text{hour}$:
$$y_t = PM2.5_{t+1}$$
The feature vector $\mathbf{x}_t \in \mathbb{R}^D$ is constructed **strictly and exclusively** from observations recorded at timestamp $t$ and preceding hours ($t - 1, t - 2, \dots$):
$$\mathbf{x}_t = \phi(\{ \mathbf{z}_\tau \mid \tau \le t \})$$
where $\mathbf{z}_\tau$ represents the raw pollutant concentrations and meteorological parameters at time $\tau$. Under no circumstances does any component of $\mathbf{x}_t$ incorporate information from $t + 1$ or any future timestep.

---

## 2. Core Architecture & Pipeline Contracts

The feature engineering architecture is implemented across two core modules:
1. `src/features.py`: Causal feature generation, domain feature transformations, target formulation, and feature matrix extraction.
2. `src/split_data.py`: Multi-station synchronized chronological train/val/test splitting, leakage auditing, and learned scaler isolation.

```
Raw Station CSVs (data/raw/)
       │
       ▼
src/data/load_data.py (Schema validation, timestamp creation)
       │
       ▼
src/preprocessing.py (Causal ffill <= 6h, station medians, target un-imputed)
       │
       ▼
src/features.py
   ├── create_target(df, horizon=1)  ──> y_{t+1} verified on strict hourly grid
   ├── create_lag_features()         ──> PM2.5 (1, 2, 3, 6, 12, 24h) + Pollutants
   ├── create_rolling_features()     ──> Rolling Mean & Std (3, 6, 12, 24h)
   ├── create_calendar_features()    ──> hour, month, day, day_of_week, is_weekend
   ├── create_cyclical_features()    ──> sin/cos diurnal (24h) & seasonal (12m)
   └── create_domain_weather()       ──> Dew point depression, wind stagnation
       │
       ▼
src/split_data.py (70% Train / 15% Val / 15% Test — Chronological, zero shuffle)
       │
       ├──> X_train, y_train (2013-03-01 00:00 to 2015-12-18 15:00) ──> Harsh (Model Fitting)
       ├──> X_val,   y_val   (2015-12-18 16:00 to 2016-07-24 19:00) ──> Harsh (Hyperparameter Tuning)
       └──> X_test,  y_test  (2016-07-24 20:00 to 2017-02-28 23:00) ──> Priyanshu (Final Evaluation)
```

### Module Interface Contracts

#### 1. Input Contract with Saksham (`src/preprocessing.py`)
`src/features.py` expects a pandas DataFrame with:
- `datetime`: `datetime64[ns]` timestamp column.
- `station`: Categorical or string station identifier.
- `PM2.5`: Float concentration. Missing values MUST remain `NaN` (never imputed).
- `PM10`, `SO2`, `NO2`, `CO`, `O3`, `TEMP`, `PRES`, `DEWP`, `RAIN`, `WSPM`, `wd`: Cleaned numeric and categorical columns.

#### 2. Output Contract with Harsh (`src/features.py::extract_feature_matrix`)
To ensure Harsh's gradient boosting models have a clean training pipeline, `extract_feature_matrix()` separates the data into:
- `X`: A pure numeric `pd.DataFrame` containing exactly the 70 candidate features (no target, no future data, no string metadata).
- `y`: A continuous `pd.Series` of target values ($PM2.5_{t+1}$) filtered to valid rows.
- `metadata`: A `pd.DataFrame` with `[station, datetime]` preserved for error auditing, station slicing, and residual diagnostics.

---

## 3. Strict Prevention of Data Leakage

Data leakage in time-series forecasting introduces catastrophic optimistic bias during training and validation, resulting in failure in production deployment. We enforce seven strict anti-leakage barriers:

| Leakage Risk | Real-World Danger | Prevention Mechanism in Pipeline |
|:---|:---|:---|
| **Lookahead in Rolling Stats** | Rolling statistic at $t$ includes observation at $t+1$, enabling model to "cheat" using future target. | Rolling windows are computed backwards over historical indices $[t - W + 1, t]$ on the time series. Target $y_{t+1}$ is never passed to rolling functions. |
| **Blind Row Shifting across Gaps** | Shifting rows blindly when an hour is missing causes row $t$ to receive a target from $t+2$ or $t+3$ hours ahead. | `create_target()` and `create_lag_features()` reindex to a complete regular 1-hour temporal grid per station. If $t + 1\text{h}$ is missing, target is assigned `NaN`. |
| **Cross-Station Boundary Spillover** | Lag from the end of Station A propagates into the beginning of Station B. | All lag, target, and rolling computations are partitioned strictly by `station`. Cross-station bleed is mathematically impossible. |
| **Cold-Start Boundary Starvation** | Splitting raw data before feature creation causes initial validation records to have missing lags. | Full contiguous feature engineering is executed across the continuous timeline, ensuring validation rows have valid historical context up to $t$. |
| **Statistical Scaler Leakage** | Scaling with mean/std calculated across the whole dataset leaks test distribution into training. | Scalers (`StandardScaler`, `RobustScaler`) are fitted strictly on `train_X` (`fit()`) and applied downstream (`transform()`) to validation and test sets. |
| **Random Train/Test Shuffling** | Random cross-validation trains on future samples to predict past samples, creating extreme lookahead bias. | Random shuffling is prohibited. Partitions are split strictly by chronological timestamps. |
| **Target in Feature Matrix** | Target column inadvertently retained in feature set $X$. | `extract_feature_matrix()` audits and asserts that `target_pm25` is excluded from $X$. |

---

## 4. Complete Feature Dictionary

Our feature generation pipeline produces **70 candidate features**, organized into seven explainable domain categories:

| Feature Group | Feature Name | Formula / Definition | Physical & Meteorological Justification | Candidate Modeling Role |
|:---|:---|:---|:---|:---|
| **Target** | `target_pm25` | $PM2.5_{t+1}$ | Ground truth fine particulate concentration 1 hour ahead | Supervised Target Label |
| **Base Pollutants** | `PM2.5` | $PM2.5_t$ | Contemporaneous PM2.5 concentration at current hour $t$ (1h before target) | Primary Autoregressive Predictor ($r = 0.969$) |
| **Base Pollutants** | `PM10` | $PM10_t$ | Coarse inhalable particulate matter ($\le 10\ \mu\text{m}$) | Strong collinear tracer ($r = 0.858$) |
| **Base Pollutants** | `SO2` | $SO2_t$ | Sulfur dioxide; coal combustion and heavy industry emissions tracer | Winter smog & heating season marker |
| **Base Pollutants** | `NO2` | $NO2_t$ | Nitrogen dioxide; vehicle traffic exhaust and combustion | Urban rush-hour traffic emissions tracer |
| **Base Pollutants** | `CO` | $CO_t$ | Carbon monoxide; incomplete combustion tracer | Incomplete combustion tracer ($r = 0.78$) |
| **Base Pollutants** | `O3` | $O3_t$ | Surface tropospheric ozone; secondary photochemical oxidant | Daytime photochemical oxidation indicator |
| **Base Meteorology** | `TEMP` | $TEMP_t$ | Ambient dry-bulb temperature (°C) | Boundary layer height and thermal inversion driver |
| **Base Meteorology** | `PRES` | $PRES_t$ | Barometric surface atmospheric pressure (hPa) | Synoptic cold front vs low pressure trough tracker |
| **Base Meteorology** | `DEWP` | $DEWP_t$ | Dew point temperature (°C) | Absolute moisture proxy; relative humidity driver |
| **Base Meteorology** | `RAIN` | $RAIN_t$ | Hourly liquid precipitation accumulation (mm/h) | Wet deposition atmospheric scavenging wash-out |
| **Base Meteorology** | `WSPM` | $WSPM_t$ | Horizontal surface wind speed at 10m (m/s) | Key dispersion vs stagnation threshold |
| **Autoregressive Lags** | `pm25_lag_1h` | $PM2.5_{t-1}$ | PM2.5 1 hour prior to observation (2h before target) | Short-term inertial persistence ($r = 0.924$) |
| **Autoregressive Lags** | `pm25_lag_2h` | $PM2.5_{t-2}$ | PM2.5 2 hours prior to observation | 2-hour trajectory momentum ($r = 0.879$) |
| **Autoregressive Lags** | `pm25_lag_3h` | $PM2.5_{t-3}$ | PM2.5 3 hours prior to observation | Multi-hour smog event trend ($r = 0.838$) |
| **Autoregressive Lags** | `pm25_lag_6h` | $PM2.5_{t-6}$ | PM2.5 6 hours prior to observation | Quarter-day atmospheric trend |
| **Autoregressive Lags** | `pm25_lag_12h` | $PM2.5_{t-12}$ | PM2.5 12 hours prior to observation | Semi-diurnal cycle baseline |
| **Autoregressive Lags** | `pm25_lag_24h` | $PM2.5_{t-24}$ | PM2.5 same hour previous day | 24-hour diurnal seasonality anchor |
| **Multi-Pollutant Lags** | `pm10_lag_1h`, `3h`, `6h`, `24h` | $PM10_{t-k}$ | Historical coarse particulate matter lags | Dust transport and coarse particulate momentum |
| **Multi-Pollutant Lags** | `no2_lag_1h`, `3h`, `6h`, `24h` | $NO2_{t-k}$ | Historical nitrogen dioxide lags | Commuter peak evolution & nocturnal boundary trapping |
| **Multi-Pollutant Lags** | `co_lag_1h`, `3h`, `6h`, `24h` | $CO_{t-k}$ | Historical carbon monoxide lags | Regional combustion plume trajectory tracking |
| **Historical Rolling** | `pm25_roll_mean_3h`, `std_3h` | $\mu, \sigma$ over $[t-2:t]$ | 3-hour backward moving average and volatility of PM2.5 | Short-term baseline and sudden spike onset |
| **Historical Rolling** | `pm25_roll_mean_6h`, `std_6h` | $\mu, \sigma$ over $[t-5:t]$ | 6-hour backward moving average and volatility of PM2.5 | Sub-diurnal smoothed level and turbulence |
| **Historical Rolling** | `pm25_roll_mean_12h`, `std_12h` | $\mu, \sigma$ over $[t-11:t]$ | 12-hour backward moving average and volatility of PM2.5 | Half-day accumulation level |
| **Historical Rolling** | `pm25_roll_mean_24h`, `std_24h` | $\mu, \sigma$ over $[t-23:t]$ | 24-hour backward moving average and volatility of PM2.5 | Daily baseline pollution state |
| **Historical Rolling** | `pm10_roll_mean_6h`, `std_6h`, `24h` | $\mu, \sigma$ over $[t-k:t]$ | Moving average and variability of coarse particles | Coarse particulate background trend |
| **Historical Rolling** | `temp_roll_mean_24h` | $\mu$ over $[t-23:t]$ | 24-hour moving average temperature | Thermal inertia and cold air outbreak indicator |
| **Temporal Features** | `hour` | $t.\text{hour} \in [0, 23]$ | Discrete hour of day in local Beijing Time | Diurnal rush hour and boundary layer expansion |
| **Temporal Features** | `day_of_week` | $t.\text{dayofweek} \in [0, 6]$ | Day of week (0=Monday, 6=Sunday) | Weekly commercial and industrial work rhythm |
| **Temporal Features** | `day` | $t.\text{day} \in [1, 31]$ | Calendar day of month | Intra-month indexing |
| **Temporal Features** | `month` | $t.\text{month} \in [1, 12]$ | Calendar month of year | Strong winter heating season vs summer contrast |
| **Temporal Features** | `is_weekend` | $\mathbb{I}(\text{day\_of\_week} \ge 5)$ | Binary flag for Saturday and Sunday | Weekend emission reduction effect |
| **Cyclical Encodings** | `sin_hour`, `cos_hour` | $\sin, \cos(2\pi \cdot \text{hour} / 24)$ | Trigonometric circular encoding of 24h diurnal cycle | Resolves discontinuity between 23:00 and 00:00 |
| **Cyclical Encodings** | `sin_month`, `cos_month` | $\sin, \cos(2\pi \cdot (\text{month}-1) / 12)$ | Trigonometric circular encoding of annual seasonal cycle | Resolves discontinuity between December and January |
| **Cyclical Encodings** | `sin_dow`, `cos_dow` | $\sin, \cos(2\pi \cdot \text{dow} / 7)$ | Trigonometric circular encoding of weekly cycle | Continuous weekly periodicity representation |
| **Wind Encodings** | `wd_deg` | $\theta \in [0^\circ, 337.5^\circ]$ | Compass direction azimuth angle in degrees | Meteorological direction indicator |
| **Wind Encodings** | `wd_sin` | $\sin(\theta_{\text{rad}})$ | East-West wind vector component (+East, -West) | Identifies easterly/southeasterly industrial corridors |
| **Wind Encodings** | `wd_cos` | $\cos(\theta_{\text{rad}})$ | North-South wind vector component (+North, -South) | Identifies clean northerly Siberian gusts (+1) |
| **Domain Interactions** | `temp_dewp_spread` | $\text{TEMP} - \text{DEWP}$ | Dew point depression (°C) | Proxy for relative humidity; low spread ($\le 2^\circ\text{C}$) drives hygroscopic smog condensation |
| **Domain Interactions** | `wind_stagnation` | $\mathbb{I}(\text{WSPM} < 1.5\ \text{m/s})$ | Binary atmospheric stagnation indicator | Prevents vertical/horizontal dispersion of pollutants |
| **Domain Interactions** | `rain_clearing` | $\mathbb{I}(\text{RAIN} > 0.0\ \text{mm/h})$ | Binary wet deposition indicator | Wash-out scavenging of suspended particulate matter |

---

## 5. Actual Chronological Split Boundaries & Sample Counts

We implement a configurable chronological split allocating **70% Training**, **15% Validation**, and **15% Testing**.

### Exact Calculated Boundaries

| Split Partition | Temporal Fraction | Start Timestamp (UTC+8) | End Timestamp (UTC+8) | Total Hours per Station | Aotizhongxin Sample Count | Citywide (12 Stations) Sample Count | Target Status | Downstream Role |
|:---|:---:|:---|:---|:---:|:---:|:---:|:---|:---|
| **Training** | **70.0%** | `2013-03-01 00:00:00` | `2015-12-18 15:00:00` | 24,544 | 24,544 | 294,528 | 23,892 valid (652 missing) | Parameter fitting for LightGBM, XGBoost, CatBoost |
| **Validation** | **15.0%** | `2015-12-18 16:00:00` | `2016-07-24 19:00:00` | 5,260 | 5,260 | 63,120 | 5,134 valid (126 missing) | Hyperparameter tuning, feature selection, early stopping |
| **Testing** | **15.0%** | `2016-07-24 20:00:00` | `2017-02-28 23:00:00` | 5,260 | 5,260 | 63,120 | 5,112 valid (148 missing) | Untouched hold-out evaluation (Priyanshu) |
| **Total Corpus** | **100.0%** | `2013-03-01 00:00:00` | `2017-02-28 23:00:00` | 35,064 | 35,064 | 420,768 | 34,138 valid (926 missing) | Full dataset across 4 calendar years |

### Protocol for Overlapping Feature Histories Across Boundaries
1. **Cold-Start Elimination:** Features are calculated on the full unbroken time series before partitioning. The first validation row (`2015-12-18 16:00:00`) has a complete 24-hour lag and 24-hour rolling mean computed from preceding historical hours in late training. This replicates true production deployment where historical sensor data is available at inference time.
2. **Boundary Transition Row:** At the final training row (`2015-12-18 15:00:00`), its feature vector $\mathbf{x}$ is at 15:00, and its target is $PM2.5$ at 16:00 (the start of the validation partition). If strict boundary separation is desired, setting `embargo_hours=1` in `chronological_split()` purges this transition row, guaranteeing that no training target timestamp coincides with any validation feature timestamp.

---

## 6. Verification & Test Suite Summary

A comprehensive test suite containing **31 automated unit tests** (100% passing) validates every anti-leakage requirement:

```bash
$ pytest -v tests/
tests/test_data.py ..........                            [ 32%]
tests/test_features.py .........                         [ 61%]
tests/test_preprocessing.py ......                       [ 80%]
tests/test_split_data.py ......                          [100%]
====================== 31 passed in 15.48s ======================
```

### Key Anti-Leakage Tests Implemented
- `test_target_is_strictly_one_hour_ahead`: Confirms $y_t = PM2.5_{t+1}$ and verifies row $N$ produces `NaN`.
- `test_lag_features_use_only_historical_data`: Confirms lag $k$ uses observation at $t - k$ and contains initial `NaN`s.
- `test_rolling_features_exclude_future_target_values`: **Perturbation Test**: Perturbs $PM2.5_{t+1}$ with an extreme spike ($99,999.0\ \mu\text{g/m}^3$) and proves that historical rolling mean and std at time $t$ remain strictly invariant.
- `test_missing_timestamps_handled_correctly`: Introduces an intentional 1-hour gap in synthetic data (e.g., 02:00 missing). Confirms row 01:00 produces `target = NaN` (does NOT grab 03:00), and row 03:00 produces `lag_1h = NaN` (does NOT grab 01:00).
- `test_station_isolation_prevents_cross_boundary_leakage`: Confirms Station B row 00:00 never borrows Station A row 05:00 data.
- `test_extract_feature_matrix_prevents_target_leakage`: Audits that $X$ strictly excludes `target_pm25`, `target_valid`, and metadata columns.
- `test_learned_transformation_isolation`: Confirms scalers fit strictly on training features ($mean = 20.0$) and never pool validation/test statistics.

---

## 7. Viva Q&A Guide: Core Concepts & Explanations

Here are concise, high-impact responses for project viva and technical presentations:

### Q1: What is a lag feature, and why is it essential for time-series forecasting?
> **Answer:** A lag feature is an observation of a variable taken at a fixed past time step relative to the current forecast timestamp $t$ (e.g., $PM2.5_{t-1}$, $PM2.5_{t-24}$). In air quality forecasting, atmospheric processes exhibit strong physical inertia—particulates take hours to disperse or accumulate. Lags provide autoregressive memory that allows tree-based models to capture short-term persistence (1h–3h momentum) and diurnal seasonality (24h prior-day baseline), which achieved Pearson correlations of $0.92$ to $0.97$ with the target.

### Q2: What are rolling statistics, and how do they differ from simple lags?
> **Answer:** While a lag feature captures a single instantaneous point in the past, rolling statistics compute an aggregated summary (such as mean or standard deviation) over a historical sliding window (e.g., past 3, 6, 12, or 24 hours). The rolling mean filters out high-frequency sensor noise and captures the underlying atmospheric pollution level, while the rolling standard deviation measures atmospheric volatility and turbulence, serving as an early indicator of incoming severe haze episodes.

### Q3: Why is random cross-validation invalid for time-series, and why must we use chronological splitting?
> **Answer:** Standard k-fold cross-validation randomly shuffles records across time. If we train on data from Wednesday and Friday to predict Thursday, the model uses future information to predict the past. This introduces severe lookahead bias, inflating validation scores artificially while causing complete failure on unseen future data. Chronological splitting preserves the natural forward arrow of time: we train on the past ($t \in [T_0, T_1]$), tune on the intermediate future ($t \in [T_1, T_2]$), and evaluate on the distant future ($t \in [T_2, T_3]$).

### Q4: How could data leakage occur in 1-hour ahead forecasting, and how did you prevent it?
> **Answer:** Leakage could enter through four distinct failure modes:
> 1. **Target in rolling window:** If a rolling statistic at time $t$ inadvertently includes $t+1$, the target leaks directly into the features. We compute rolling windows strictly backwards over $[t - W + 1, t]$.
> 2. **Blind row shifting over gaps:** If 14:00 is missing, shifting blindly by -1 assigns the 15:00 measurement to 13:00, creating a 2-hour target instead of a 1-hour target. We reindex to a strict hourly grid so gaps produce `NaN` rather than lookahead shortcuts.
> 3. **Station boundary bleed:** In multi-station datasets, shifting across the boundary assigns Station B's morning data as Station A's evening target. We partition all shifts strictly by station.
> 4. **Statistical leakage:** Fitting scalers or imputers on the entire dataset leaks validation distribution parameters into the training phase. We fit all transformations strictly on `train_X`.
