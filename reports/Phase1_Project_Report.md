# AIR QUALITY FORECASTING VIA GRADIENT BOOSTING AND LSTM MODELS
## Phase 1 Project Report

**Submitted by:**  
- **Saksham Sontakke** (Lead: Data Loading, Causal Cleaning & EDA)  
- **Ajit Kumar Prasad** (Lead: Time-Series Feature Engineering & Chronological Splitting)  
- **Harsh Kumar** (Lead: Baselines & Gradient Boosting Models)  
- **Mihika Mathur** (Lead: Evaluation Suite, Error Stratification & Visualizations)  
- **Priyanshu Agrahari** (Lead: ML Pipeline Integration, Automation & Reproducibility)  

**Project Domain:** Machine Learning, Time-Series Forecasting and Environmental Data Analysis  
**Repository:** [https://github.com/CyborgX-hash/Air-Quality-Forecasting](https://github.com/CyborgX-hash/Air-Quality-Forecasting)  
**Dataset:** Beijing Multi-Site Air Quality Dataset — UCI Machine Learning Repository  

---

## ABSTRACT

Air pollution is a major environmental concern that affects human health, ecological systems, and overall quality of life. Accurate air-quality forecasting can help individuals, researchers, and authorities understand pollution trends and make informed decisions. This project focuses on developing a machine-learning-based system for forecasting particulate matter concentration ($\text{PM}_{2.5}$) using historical air-quality measurements and meteorological information.

Phase 1 implements a rigorous, leakage-free time-series forecasting pipeline using the Beijing Multi-Site Air Quality Dataset, which contains hourly observations collected from 12 monitoring stations between March 2013 and February 2017. The project includes data loading, cleaning, exploratory data analysis, time-series feature engineering, chronological data splitting, model training, evaluation, and visualization.

Gradient Boosting is evaluated against baseline approaches, including persistence forecasting and Linear Regression. Models are assessed using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), coefficient of determination ($R^2$), and Mean Bias Error (MBE). The reported 12-station benchmark gives tuned Gradient Boosting a macro MAE of **$10.0845\ \mu\text{g/m}^3$**, compared with **$10.7237\ \mu\text{g/m}^3$** for naive persistence, corresponding to a **5.92%** mean reduction in station MAE (and a **5.96%** reduction in network macro MAE).

Phase 1 establishes the data-processing, feature engineering, and evaluation foundation for the project. Long Short-Term Memory (LSTM) neural networks are planned for Phase 2 to investigate whether sequence-based deep learning can improve forecasting performance.

---

## CHAPTER 1: INTRODUCTION

### 1.1 Background
Air quality varies over time because of traffic emissions, industrial activity, weather conditions, seasonal changes, and other environmental factors. Pollutants such as $\text{PM}_{2.5}$ are especially important because their small particle size ($\le 2.5\ \mu\text{m}$) allows them to penetrate deeply into the lungs and enter the bloodstream.

Monitoring stations continuously collect air-quality measurements. However, historical measurements alone do not directly communicate what pollution levels are likely to be in the next hour. Machine-learning techniques can identify patterns in historical pollutant concentrations and related variables to estimate future values.

Time-series forecasting is appropriate for this problem because observations have a strict temporal order. Previous pollutant levels, lagged measurements, rolling statistics, and meteorological variables can be used to construct features for predicting future concentrations.

### 1.2 Problem Statement
The objective is to develop a reliable forecasting pipeline that predicts $\text{PM}_{2.5}$ concentration one hour ahead using historical air-quality data.

The system must process hourly observations, handle missing values appropriately without future lookahead, create features without target contamination, train and compare forecasting models, and evaluate their performance using an untouched chronological test partition.

The project also investigates performance across multiple monitoring stations to understand how forecasting accuracy varies across geographic and topographical locations.

### 1.3 Objectives
The main objectives of Phase 1 are:
1. Load and validate historical air-quality data across 12 monitoring sites.
2. Perform causal data cleaning and exploratory data analysis.
3. Engineer time-series features (autoregressive lags, multi-pollutant lags, rolling statistics, calendar, and weather interactions) using historical observations only.
4. Construct a strictly one-hour-ahead $\text{PM}_{2.5}$ prediction target ($y_{t+1} = \text{PM}_{2.5, t+1}$).
5. Split the data chronologically into training (70%), validation (15%), and test (15%) sets with zero timestamp overlap.
6. Implement persistence, Linear Regression, and Gradient Boosting models.
7. Select the champion model using validation performance exclusively.
8. Evaluate models using multiple regression metrics (MAE, RMSE, $R^2$, MBE).
9. Generate visualizations and station-level benchmark reports.
10. Establish a reproducible, automated foundation with a comprehensive test suite for future LSTM implementation.

### 1.4 Scope of Phase 1
Phase 1 concentrates on data preparation, feature engineering, conventional machine-learning models, baseline comparisons, and systematic evaluation.

The primary forecasting task is predicting $\text{PM}_{2.5}$ concentration for the next hour ($h=1$). The project uses historical observations from 12 Beijing monitoring stations.

LSTM implementation and comparison with conventional models are reserved for Phase 2. Consequently, this report does not claim that an LSTM model has already been trained or evaluated.

---

## CHAPTER 2: DATASET DESCRIPTION

### 2.1 Dataset Overview
The project uses the Beijing Multi-Site Air Quality Dataset from the UCI Machine Learning Repository.
The dataset contains hourly measurements collected at 12 monitoring stations over four full years (1 March 2013 to 28 February 2017).

| Dataset Characteristic | Description |
|:---|:---|
| **Monitoring Stations** | 12 sites across urban, suburban, and rural Beijing |
| **Records per Station** | 35,064 hourly records |
| **Total Raw Observations** | 420,768 records (12 stations $\times$ 35,064 hours) |
| **Observation Period** | March 1, 2013 00:00 – February 28, 2017 23:00 |
| **Primary Prediction Variable** | $\text{PM}_{2.5}$ concentration ($\mu\text{g/m}^3$) |
| **Forecast Horizon** | One hour ahead ($t+1$) |
| **Data Type** | Multivariate hourly time-series environmental measurements |

The raw dataset is organized into station-specific CSV files (`PRSA_Data_{Station}_20130301-20170228.csv`).

### 2.2 Data Preparation & Causal Cleaning
The data-processing pipeline performs timestamp construction and validation, checks chronological ordering, identifies duplicate timestamps and missing observations, and prepares the station data for downstream feature engineering.

- **Causal Forward Fill:** Missing input weather and gaseous pollutant features are handled using causal forward filling with an upper bound limit of six consecutive hours (`ffill(limit=6)`). This restricts earlier observations to fill short sensor communication dropouts while avoiding indefinite bleeding across multi-day outages.
- **Station-Partitioned Imputation:** Imputation operates strictly within each station grouping so that historical values from one station never contaminate another.
- **Target Invariance:** Missing $\text{PM}_{2.5}$ target values are **never** imputed. Imputing targets creates synthetic artifacts in ground-truth labels. Rows with missing targets are dropped during final supervised matrix extraction.

### 2.3 Exploratory Data Analysis
Exploratory data analysis reveals key environmental dynamics:
- High autocorrelation in hourly $\text{PM}_{2.5}$ ($r > 0.95$ at lag 1).
- Distinct diurnal patterns: morning rush-hour peaks (08:00–10:00) and evening boundary-layer compression peaks (20:00–23:00).
- Strong seasonal variations: severe winter smog episodes driven by coal heating and shallow atmospheric temperature inversions, contrasted with lower summer concentrations.
- Spatial gradients: rural northern mountain stations (Dingling, Huairou) exhibit substantially lower average concentrations than urban traffic centers (Dongsi, Wanliu, Wanshouxigong).

---

## CHAPTER 3: METHODOLOGY

### 3.1 Forecasting Formulation
Let $y_t$ represent the observed $\text{PM}_{2.5}$ concentration at time $t$. The forecasting task is defined as:

$$\hat{y}_{t+1} = f(\mathbf{X}_t)$$

where $\mathbf{X}_t$ denotes the feature vector constructed strictly from observations available at or before time $t$, and $f(\cdot)$ represents the trained regression model. The target is defined by reindexing the station time series to a continuous hourly frequency grid and shifting one step:

$$\text{target\_pm25}_t = y_{t+1}$$

### 3.2 Feature Engineering
A total of 70 candidate time-series features are engineered from historical observations:
1. **Contemporaneous Base Features:** Current readings of $\text{PM}_{2.5}$, coarse particulate matter ($\text{PM}_{10}$), sulfur dioxide ($\text{SO}_2$), nitrogen dioxide ($\text{NO}_2$), carbon monoxide ($\text{CO}$), and ozone ($\text{O}_3$).
2. **Autoregressive Lags:** $\text{PM}_{2.5}$ lags at $t-1, t-2, t-3, t-6, t-12, t-24$ hours.
3. **Multi-Pollutant Lags:** Lags at $t-1, t-3, t-6, t-24$ hours for $\text{PM}_{10}$, $\text{NO}_2$, and $\text{CO}$.
4. **Causal Rolling Statistics:** Backward-looking moving averages and standard deviations over 3h, 6h, 12h, and 24h windows for $\text{PM}_{2.5}$, $\text{PM}_{10}$, and temperature.
5. **Meteorological Interactions:** Dew point depression ($\text{TEMP} - \text{DEWP}$), relative air density proxy ($\text{PRES} / (\text{TEMP} + 273.15)$), and rain flags.
6. **Cyclical Temporal Encoding:** Sine and cosine transformations of hour-of-day ($\sin(2\pi h / 24), \cos(2\pi h / 24)$), day-of-week, and wind direction radians.

### 3.3 Chronological Data Splitting
The continuous timeline is partitioned chronologically into three forward-time intervals:
- **Training Set (70%):** 1 March 2013 00:00 to 18 December 2015 15:00
- **Validation Set (15%):** 18 December 2015 16:00 to 24 July 2016 19:00
- **Test Set (15%):** 24 July 2016 20:00 to 28 February 2017 23:00

Random cross-validation shuffling is strictly prohibited. Set intersection verification confirms zero timestamp overlap across partitions.

### 3.4 Forecasting Models

#### 3.4.1 Naive 1-Step Persistence
Predicts the next hour's concentration using the contemporaneous measurement available at forecast time:

$$\hat{y}_{t+1} = \text{PM}_{2.5, t}$$

Because hourly atmospheric pollution exhibits strong inertia, naive persistence represents a rigorous operational baseline ($R^2 \approx 0.947$).

#### 3.4.2 Constant-Mean Persistence Baseline
Outputs the historical mean target value memorized from the training set ($\hat{y}_{t+1} = \bar{y}_{\text{train}}$) across all future timesteps. Demonstrates the impact of seasonality and long-term distributional drift.

#### 3.4.3 Linear Regression
Models the target as a linear combination of input features:

$$\hat{y}_{t+1} = \mathbf{w}^T \mathbf{X}_t + b$$

Provides an interpretable parametric baseline to test whether linear combinations of multi-pollutant and weather indicators provide value over persistence.

#### 3.4.4 Gradient Boosting Regressor
Builds an additive ensemble of shallow regression trees sequentially to minimize squared-error loss:

$$\hat{y}^{(m)}(\mathbf{X}) = \hat{y}^{(m-1)}(\mathbf{X}) + \eta \cdot h_m(\mathbf{X})$$

where $\eta$ is the learning rate and $h_m$ is the $m$-th regression tree fit to the pseudo-residuals. The project evaluates:
- **Default GB:** $M = 200$ estimators, learning rate $\eta = 0.05$, max depth $= 3$, min samples leaf $= 5$, subsample $= 0.8$.
- **Tuned GB:** $M = 250$ estimators, learning rate $\eta = 0.04$, max depth $= 5$, min samples leaf $= 8$, subsample $= 0.8$.

Champion selection is governed strictly by the lowest **Validation MAE**.

### 3.5 Evaluation Metrics

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N \left| y_i - \hat{y}_i \right|$$

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N \left( y_i - \hat{y}_i \right)^2}$$

$$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$

$$\text{MBE} = \frac{1}{N} \sum_{i=1}^N \left( \hat{y}_i - y_i \right)$$

*Sign Convention Note:* In the codebase and IEEE environmental reporting standard, $\text{MBE} = \frac{1}{N} \sum (\hat{y}_i - y_i)$, where $\text{MBE} > 0$ denotes overprediction and $\text{MBE} < 0$ denotes underprediction.

---

## CHAPTER 4: IMPLEMENTATION

### 4.1 Technology Stack

| Component | Technology / Library | Purpose |
|:---|:---|:---|
| **Programming Language** | Python 3.12 | Core execution environment |
| **Data Manipulation** | Pandas 3.0.6, NumPy 2.5.4 | Causal data cleaning, arrays, time indexing |
| **Machine Learning** | Scikit-learn 1.9.1 | Estimators, metrics, tree ensembles |
| **Visualization** | Matplotlib 3.11.2, Seaborn 0.13.2 | Headless publication-quality figures (300 DPI) |
| **Automated Testing** | Pytest 9.1.1 | Unit and integration testing |
| **Concurrency** | Concurrent.futures (ProcessPoolExecutor) | Multi-process parallel station execution |
| **Version Control** | Git & GitHub | Distributed version control |

### 4.2 Pipeline Workflow
The automated pipeline executes in 8 discrete stages:
1. `load_station_data` / `validate_all_stations_schema`: Dataset loading and schema integrity audit.
2. `clean_features`: Causal forward-fill imputation and wind vector encoding.
3. `generate_features`: Creation of target ($t+1$) and 70 candidate time-series features.
4. `chronological_split`: Synchronized 70/15/15 partitioning and anti-leakage audit.
5. Model training: Fitting Persistence, Linear Regression, Default GB, and Tuned GB.
6. Validation selection: Champion determination based strictly on validation MAE.
7. Final test evaluation: Single out-of-sample generalization run on untouched test data.
8. Artifact export: Structured Markdown/CSV tables, 300-DPI publication figures, and JSON metadata logs.

### 4.3 Automated Test Suite
The automated test suite runs via `pytest -v` with **68 / 68 tests passing**:
- `tests/test_data.py`: 10 tests
- `tests/test_evaluate.py`: 17 tests
- `tests/test_features.py`: 9 tests
- `tests/test_multi_station.py`: 6 tests
- `tests/test_pipeline.py`: 5 tests
- `tests/test_preprocessing.py`: 6 tests
- `tests/test_split_data.py`: 6 tests
- `tests/test_visualization.py`: 9 tests

---

## CHAPTER 5: RESULTS AND DISCUSSION

### 5.1 Twelve-Station Benchmark Summary ($N = 58,201$)

The comprehensive benchmark across all 12 Beijing monitoring stations is summarized below:

| Model Architecture | Macro MAE ($\mu\text{g/m}^3$) | Pooled MAE ($\mu\text{g/m}^3$) | Macro RMSE ($\mu\text{g/m}^3$) | Pooled RMSE ($\mu\text{g/m}^3$) | Macro $R^2$ | Macro MBE ($\mu\text{g/m}^3$) | Macro MAE Imprv. vs Persistence | Inter-Station Std MAE | Total $N_{\text{obs}}$ |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Gradient Boosting (Tuned)** | **10.0845** | **10.0863** | **20.1496** | **20.2691** | **0.9522** | **-0.0510** | **+5.92%** | **±0.95** | 58,201 |
| Gradient Boosting (Default) | 10.2872 | 10.2895 | 20.3508 | 20.4625 | 0.9512 | +0.0257 | +4.02% | ±0.94 | 58,201 |
| Linear Regression | 10.5147 | 10.5163 | 19.8596 | 19.9479 | 0.9534 | -0.1200 | +1.81% | ±0.88 | 58,201 |
| **Persistence (Naive Lag-1)** | 10.7237 | 10.7240 | 21.3120 | 21.4066 | 0.9466 | +0.0861 | 0.00% (Baseline) | ±1.01 | 58,201 |
| Persistence (Constant Mean) | 69.3400 | 69.3241 | 111.4981 | 112.1734 | -0.4509 | -61.6829 | -545.39% | ±8.76 | 58,201 |

*Mathematical Distinction:*
- **Macro MAE:** $\frac{1}{12} \sum \text{MAE}_s = 10.0845\ \mu\text{g/m}^3$.
- **Macro MAE Improvement:** The average of station-specific percentage reductions $\frac{1}{12} \sum_{s=1}^{12} \left( \frac{\text{MAE}_{\text{pers}, s} - \text{MAE}_{\text{tuned}, s}}{\text{MAE}_{\text{pers}, s}} \times 100 \right) = \mathbf{+5.92\%}$.
- **Improvement on Macro Average:** Calculating reduction on the unweighted macro means yields $\frac{10.7237 - 10.0845}{10.7237} \times 100 = \mathbf{5.96\%}$. Both figures are consistent representations of the data.
- **Pooled RMSE:** Calculated correctly as $\sqrt{\frac{1}{N_{\text{total}}} \sum N_s \text{RMSE}_s^2} = \mathbf{20.2691}\ \mu\text{g/m}^3$, matching direct concatenation evaluation exactly.

### 5.2 Station-Wise Champion Breakdown

| Station | Selected Champion Model | Test MAE ($\mu\text{g/m}^3$) | Test RMSE ($\mu\text{g/m}^3$) | Test $R^2$ | Test MBE ($\mu\text{g/m}^3$) | Test $N_{\text{obs}}$ | MAE Imprv. vs Persistence |
|:---|:---|---:|---:|---:|---:|---:|---:|
| **Aotizhongxin** | Gradient Boosting (Tuned) | 10.2280 | 19.4863 | 0.9576 | -0.5699 | 4,920 | +7.72% |
| **Changping** | Gradient Boosting (Default) | 10.4691 | 19.8162 | 0.9412 | -0.0463 | 4,973 | +1.59% |
| **Dingling** | Gradient Boosting (Default) | **8.4147** | **16.3218** | 0.9505 | +0.4787 | 4,695 | +3.92% |
| **Dongsi** | Gradient Boosting (Tuned) | 11.1081 | 22.5228 | 0.9531 | -0.2121 | 4,723 | +6.07% |
| **Guanyuan** | Gradient Boosting (Tuned) | 10.2662 | 18.9944 | 0.9593 | +0.4770 | 4,846 | +8.48% |
| **Gucheng** | Gradient Boosting (Tuned) | 10.5440 | 21.6312 | 0.9549 | -1.1803 | 4,832 | +3.96% |
| **Huairou** | Gradient Boosting (Tuned) | 8.4206 | 18.9275 | 0.9399 | +0.3700 | 4,899 | +5.18% |
| **Nongzhanguan** | Gradient Boosting (Tuned) | 10.3369 | 20.3241 | 0.9577 | -0.2911 | 4,835 | +6.25% |
| **Shunyi** | Gradient Boosting (Tuned) | 10.2047 | 20.4124 | 0.9529 | +0.8147 | 4,725 | **+9.38%** |
| **Tiantan** | Gradient Boosting (Tuned) | 10.2698 | 18.8796 | **0.9608** | +0.1674 | 4,903 | +8.10% |
| **Wanliu** | Gradient Boosting (Tuned) | 9.5030 | 19.3150 | 0.9557 | +0.4587 | 4,951 | +4.84% |
| **Wanshouxigong** | Gradient Boosting (Tuned) | 11.5123 | 25.2797 | 0.9415 | -1.0843 | 4,899 | +2.87% |

### 5.3 Discussion of Results

1. **Superiority of Gradient Boosting on Absolute Deviation:**
   Tuned Gradient Boosting achieves the lowest macro MAE ($10.0845\ \mu\text{g/m}^3$) across the network, outperforming naive persistence across all 12 stations (up to $+9.38\%$ MAE improvement at Shunyi).
2. **Metric Divergence (MAE vs. RMSE):**
   Linear Regression yields a lower macro RMSE ($19.8596\ \mu\text{g/m}^3$) than Tuned GB ($20.1496\ \mu\text{g/m}^3$), despite exhibiting higher MAE ($10.5147$ vs. $10.0845\ \mu\text{g/m}^3$). This reflects loss function alignment: Ordinary Least Squares analytically minimizes squared error ($L_2$ norm), whereas deeper decision trees fit to MAE conditional medians can produce occasional tail-end outlier misses during abrupt meteorological transitions.
3. **Temporal Autoregressive Lag Effect:**
   Feature importance analysis reveals that contemporaneous $\text{PM}_{2.5, t}$ accounts for $>90\%$ of tree split gains. Consequently, during sudden weather clearing events (e.g., strong northwesterly gale arrival), tree models exhibit a 1-hour lag inertia before adjusting to the low-concentration baseline.
4. **Episodic Smog Underprediction:**
   Error stratification across pollution tiers shows near-zero bias in clean conditions ($<35\ \mu\text{g/m}^3$), but systematic underprediction during severe smog episodes ($>250\ \mu\text{g/m}^3$), indicating that tree ensembles struggle with extreme values outside the core empirical distribution.

### 5.4 Single-Station Aotizhongxin Baseline Reproduction
Under identical configuration, the Aotizhongxin results match identically across isolated and sweep runs:
- Tuned GB MAE: **10.2280** $\mu\text{g/m}^3$
- Tuned GB RMSE: **19.4863** $\mu\text{g/m}^3$
- Tuned GB $R^2$: **0.9576**
- Tuned GB MBE: **-0.5699** $\mu\text{g/m}^3$
- Test Observations: **4,920**

### 5.5 Limitations
- **1-Hour Horizon:** The operational scope is currently confined to $h=1$ hour ahead.
- **Localized Station Models:** Models are fit station-by-station; they do not model spatial advection or establish zero-shot transferability to unseen geographic coordinates.
- **Severe Episode Inelasticity:** As with most tree ensembles, tail-end peak episodes exhibit mean-reversion underprediction.

---

## CHAPTER 6: CONCLUSION AND FUTURE WORK

### 6.1 Conclusion
Phase 1 successfully develops a reproducible, leak-free, automated machine learning pipeline for one-hour-ahead $\text{PM}_{2.5}$ forecasting across the 12-station Beijing monitoring network. 

Key achievements include:
- Establishing a mathematically sound baseline where Tuned Gradient Boosting achieves $10.0845\ \mu\text{g/m}^3$ macro MAE, improving on naive persistence across all 12 stations.
- Demonstrating that Linear Regression achieves competitive RMSE, emphasizing that multi-metric evaluation is critical for environmental forecasting.
- Enforcing strict temporal causality, preventing target contamination, and ensuring zero test-set snooping during champion selection.
- Building an automated 68-test regression suite ensuring software reproducibility.

### 6.2 Future Work (Phase 2: Deep Sequence Modeling)
Phase 2 will focus on:
1. **LSTM Network Implementation:** Developing vanilla, stacked, and bidirectional Long Short-Term Memory networks to model temporal sequences without manual feature hand-crafting.
2. **Multi-Horizon Forecasting:** Extending predictions from 1 hour to 6, 12, and 24 hours ahead.
3. **Spatiotemporal Graph Networks:** Incorporating inter-station distance matrices and wind vectors to capture cross-city atmospheric pollutant advection.
4. **Extreme Episode Loss Weighting:** Investigating asymmetric or focal loss functions to mitigate underprediction during hazardous smog emergencies.

---

## REFERENCES
1. UCI Machine Learning Repository. *Beijing Multi-Site Air Quality Data*. [https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data)
2. Scikit-learn Documentation. *Ensemble Methods: Gradient Tree Boosting*. [https://scikit-learn.org/stable/modules/ensemble.html](https://scikit-learn.org/stable/modules/ensemble.html)
3. Scikit-learn Documentation. *Linear Models*. [https://scikit-learn.org/stable/modules/linear_model.html](https://scikit-learn.org/stable/modules/linear_model.html)
4. Scikit-learn Documentation. *Model Evaluation and Scoring Metrics*. [https://scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)
5. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735–1780.
