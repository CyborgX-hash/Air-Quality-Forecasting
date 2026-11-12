# Phase 2 Technical Evaluation & Model Comparison Report
## Multi-Horizon LSTM Forecasting vs. Phase 1 Traditional Machine Learning Baselines
**Project:** Air Quality Forecasting via Gradient Boosting and LSTM Models  
**Dataset:** Beijing Multi-Site Air Quality Dataset (UCI Machine Learning Repository, 12 Stations, 2013–2017)  
**Author:** Mihika Mathur (Lead Model Evaluation, Error Analysis & Visualization Specialist)  
**Collaborators:** Saksham Sontakke (Data Engineering), Ajit Kumar Prasad (Sequence Engineering), Harsh Kumar (LSTM Architecture & Training), Priyanshu (Pipeline Integration)  
**Evaluation Scope:** 12 Stations $\times$ 5 Forecast Horizons ($H \in \{1\text{h}, 3\text{h}, 6\text{h}, 12\text{h}, 24\text{h}\}$) on Chronological Out-of-Sample Test Partition ($N = 58,815$ observations per horizon).

---

## 1. Executive Summary

This report delivers the formal quantitative evaluation, benchmark comparison, and error analysis of the Phase 2 Long Short-Term Memory (LSTM) deep learning models for multi-horizon air quality forecasting ($\text{PM}_{2.5}$ in $\mu\text{g/m}^3$).

### Key Empirical Findings:

1. **Short-Horizon ($H=1\text{h}$) Dominance of Gradient Boosting**:
   - For 1-hour ahead forecasting, **Tuned Gradient Boosting from Phase 1 remains the overall benchmark champion**, attaining a Macro Test MAE of **$10.08\ \mu\text{g/m}^3$** (RMSE: $20.15\ \mu\text{g/m}^3$, $R^2: 0.9522$).
   - The Phase 2 Champion LSTM ($W=72\text{h}$, 64 units, 1 recurrent layer, MAE loss) achieved a Macro Test MAE of **$13.68\ \mu\text{g/m}^3$** (RMSE: $23.85\ \mu\text{g/m}^3$, $R^2: 0.9328$) at $H=1\text{h}$.
   - **Empirical Confirmation**: Short-horizon $\text{PM}_{2.5}$ is heavily autoregressive. High-capacity tabular tree ensembles with explicitly engineered lag features and rolling statistics exploit acute 1-step autocorrelation more sharply than recurrent networks optimizing multi-horizon shared representations.

2. **The Multi-Horizon Crossover & LSTM Superiority ($H \ge 3\text{h}$)**:
   - While Naive Persistence rapidly degenerates as the forecast horizon expands, **the LSTM network demonstrates remarkable temporal resilience**, learning stable diurnal and multi-step atmospheric trajectories.
   - **$H=3\text{h}$ Crossover Point**: LSTM surpasses Naive Persistence across Beijing, cutting Macro MAE from $23.16\ \mu\text{g/m}^3$ to **$22.62\ \mu\text{g/m}^3$** ($+2.05\%$ improvement, $+0.0255\ R^2$).
   - **$H=6\text{h}$ Expansion**: LSTM achieves **$32.41\ \mu\text{g/m}^3$ MAE** vs. $35.43\ \mu\text{g/m}^3$ for Persistence (**$+8.05\%$ MAE improvement**, $+0.0804\ R^2$).
   - **$H=12\text{h}$ Medium-Range**: LSTM delivers **$43.77\ \mu\text{g/m}^3$ MAE** vs. $49.38\ \mu\text{g/m}^3$ for Persistence (**$+11.04\%$ MAE improvement**, $+0.1727\ R^2$).
   - **$H=24\text{h}$ Day-Ahead**: At the full diurnal boundary, Naive Persistence collapses completely ($R^2 = -0.1358$, failing to explain sample variance). In contrast, the LSTM maintains a **positive $R^2 = 0.1586$** and **$55.32\ \mu\text{g/m}^3$ MAE**, delivering a **$+15.64\%$ MAE reduction** and **$+14.01\%$ RMSE reduction** ($84.74\ \mu\text{g/m}^3$ vs. $98.56\ \mu\text{g/m}^3$).

3. **Stratified Error Dynamics (High-Pollution Episodes)**:
   - For clean air conditions ($\text{PM}_{2.5} \le 35\ \mu\text{g/m}^3$, representing $39.7\%$ of test data), the LSTM exhibits near-zero bias ($\text{MBE} = +4.67\ \mu\text{g/m}^3$ at 1h) and low MAE ($6.89\ \mu\text{g/m}^3$).
   - During severe pollution episodes ($\text{PM}_{2.5} > 250\ \mu\text{g/m}^3$, $6.2\%$ of test observations), the LSTM systematically **underpredicts acute peaks** ($\text{MBE} = -35.26\ \mu\text{g/m}^3$ at 1h, degrading to $-196.55\ \mu\text{g/m}^3$ at 24h). This smoothing behavior stems from the network regressing toward the historical conditional median under MAE loss.
   - Large peak errors disproportionately inflate RMSE relative to MAE (Pooled RMSE: $23.98\ \mu\text{g/m}^3$ vs. MAE: $13.67\ \mu\text{g/m}^3$ at 1h; RMSE: $85.10\ \mu\text{g/m}^3$ vs. MAE: $55.29\ \mu\text{g/m}^3$ at 24h).

---

## 2. Evaluation Methodology & Metrics Framework

All evaluations strictly utilize the project's single-source-of-truth mathematical implementations (`src.evaluate` and `src.phase2.evaluate`). All predictions and targets are converted back to original physical concentration units ($\mu\text{g/m}^3$) via the fitted station scalers fitted strictly on training data (`train_df`).

### 2.1 Core Mathematical Formulations

1. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N \left| y_i - \hat{y}_i \right|$$
   - Units: $\mu\text{g/m}^3$. Measures linear average magnitude of error without disproportionately penalizing large outliers.

2. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N \left( y_i - \hat{y}_i \right)^2}$$
   - Units: $\mu\text{g/m}^3$. Penalizes larger forecasting errors quadratically. $\text{RMSE} \ge \text{MAE}$ strictly; large spreads ($\text{RMSE} - \text{MAE}$) indicate heavy tails and episodic forecasting spikes.

3. **Coefficient of Determination ($R^2$)**:
   $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$
   - Measures the proportion of ground-truth target variance explained by the model relative to an uninformative mean baseline. Negative $R^2$ indicates the forecast is worse than predicting the sample mean.

4. **Mean Bias Error (MBE)**:
   $$\text{MBE} = \frac{1}{N} \sum_{i=1}^N \left( \hat{y}_i - y_i \right)$$
   - Convention: $\text{MBE} = \text{mean}(\text{prediction} - \text{actual})$.
   - $\text{MBE} > 0$: Systematic overprediction.
   - $\text{MBE} < 0$: Systematic underprediction.

### 2.2 Macro vs. Pooled Aggregation Rigor

- **Macro-Averaging**: Arithmetic mean of metric values across the $S=12$ monitoring stations:
  $$\text{Macro MAE} = \frac{1}{S} \sum_{s=1}^S \text{MAE}_s, \quad \text{Macro } R^2 = \frac{1}{S} \sum_{s=1}^S R^2_s$$
- **Observation-Pooled Metrics**:
  Calculated across all $N_{\text{total}} = 58,815$ test observations concatenated across stations:
  $$\text{Pooled RMSE} = \sqrt{\frac{1}{N_{\text{total}}} \sum_{s=1}^S \sum_{i=1}^{N_s} (y_{s,i} - \hat{y}_{s,i})^2} \ne \frac{1}{S} \sum_{s=1}^S \text{RMSE}_s$$
  *Mathematical Guarantee*: Averaging station RMSE values violates Jensen's inequality and ignores station sample-size weighting. Pooled RMSE is strictly computed from total pooled squared errors.

---

## 3. Macro-Average and Pooled Multi-Horizon Performance

### 3.1 Macro-Average Benchmark Table (12 Stations)

The table below summarizes performance across the five forecast horizons for the Champion Phase 2 LSTM ($W=72\text{h}$) vs. Naive Multi-Horizon Persistence:

| Horizon | Hours | LSTM Macro MAE ($\mu\text{g/m}^3$) | Persistence MAE ($\mu\text{g/m}^3$) | MAE Improvement (%) | LSTM Macro RMSE ($\mu\text{g/m}^3$) | Persistence RMSE ($\mu\text{g/m}^3$) | RMSE Improvement (%) | LSTM $R^2$ | Persistence $R^2$ | $\Delta R^2$ | LSTM MBE ($\mu\text{g/m}^3$) | Total Test Obs |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1h** | 1 | 13.68 | 10.91 | -25.66% | 23.85 | 21.63 | -10.35% | 0.9328 | 0.9451 | -0.0123 | -1.81 | 58,815 |
| **3h** | 3 | 22.62 | 23.16 | **+2.05%** | 40.20 | 42.89 | **+6.10%** | 0.8110 | 0.7855 | **+0.0255** | -1.21 | 58,815 |
| **6h** | 6 | 32.41 | 35.43 | **+8.05%** | 55.56 | 61.60 | **+9.40%** | 0.6383 | 0.5579 | **+0.0804** | -1.57 | 58,815 |
| **12h**| 12 | 43.77 | 49.38 | **+11.04%** | 71.23 | 81.17 | **+11.85%** | 0.4017 | 0.2290 | **+0.1727** | -2.15 | 58,815 |
| **24h**| 24 | 55.32 | 65.55 | **+15.64%** | 84.74 | 98.56 | **+14.01%** | 0.1586 | -0.1358 | **+0.2944** | -4.37 | 58,815 |

### 3.2 Observation-Pooled Performance ($N_{\text{total}} = 58,815$)

| Horizon | Hours | LSTM Pooled MAE | Persistence Pooled MAE | Pooled MAE Impr. (%) | LSTM Pooled RMSE | Persistence Pooled RMSE | Pooled RMSE Impr. (%) | LSTM Pooled $R^2$ | Persistence Pooled $R^2$ | $\Delta R^2$ | LSTM Pooled MBE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1h** | 1 | 13.67 | 10.91 | -25.35% | 23.98 | 21.72 | -10.39% | 0.9345 | 0.9462 | -0.0117 | -1.82 |
| **3h** | 3 | 22.62 | 23.15 | **+2.31%** | 40.38 | 43.12 | **+6.36%** | 0.8141 | 0.7879 | **+0.0262** | -1.21 |
| **6h** | 6 | 32.40 | 35.41 | **+8.50%** | 55.79 | 62.01 | **+10.03%** | 0.6447 | 0.5610 | **+0.0837** | -1.56 |
| **12h**| 12 | 43.75 | 49.35 | **+11.36%** | 71.43 | 81.64 | **+12.51%** | 0.4162 | 0.2373 | **+0.1789** | -2.15 |
| **24h**| 24 | 55.29 | 65.54 | **+15.63%** | 85.10 | 98.94 | **+14.00%** | 0.1747 | -0.1158 | **+0.2905** | -4.36 |

---

## 4. Phase 1 vs. Phase 2 Direct Comparative Benchmark (1-Hour Ahead)

To provide an equitable, apples-to-apples comparison, Phase 1 models and the Phase 2 LSTM were evaluated on the chronological out-of-sample test split.

### Macro Summary Across Beijing (1-Hour Ahead Benchmark):

| Model Architecture | Macro Test MAE ($\mu\text{g/m}^3$) | Macro Test RMSE ($\mu\text{g/m}^3$) | Macro $R^2$ | Macro MBE ($\mu\text{g/m}^3$) | MAE Impr. vs. Persistence (%) | Benchmark Ranking |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Gradient Boosting (Tuned)** | **10.08** | **20.15** | **0.9522** | **-0.05** | **+5.92%** | **Champion (Rank 1)** |
| **Gradient Boosting (Default)** | 10.29 | 20.35 | 0.9512 | +0.03 | +4.02% | Rank 2 |
| **Linear Regression** | 10.51 | 19.86 | 0.9534 | -0.12 | +1.81% | Rank 3 |
| **Naive Persistence (Lag-1)** | 10.72 | 21.31 | 0.9466 | +0.09 | 0.00% | Rank 4 |
| **LSTM (Champion W72)** | 13.68 | 23.85 | 0.9328 | -1.81 | -25.66% | Rank 5 |

### Station-Wise 1-Hour Performance Breakdown:

- **Clean / Regional Background Stations (Dingling & Huairou)**:
  - Dingling: Tuned GB achieved **$8.29\ \mu\text{g/m}^3$ MAE**; LSTM achieved **$12.17\ \mu\text{g/m}^3$**.
  - Huairou: Tuned GB achieved **$8.42\ \mu\text{g/m}^3$ MAE**; LSTM achieved **$11.08\ \mu\text{g/m}^3$**.
- **Dense Urban Core Stations (Dongsi, Tiantan, Guanyuan)**:
  - Dongsi: Tuned GB achieved **$11.11\ \mu\text{g/m}^3$ MAE**; LSTM achieved **$14.50\ \mu\text{g/m}^3$**.
  - Tiantan: Tuned GB achieved **$10.27\ \mu\text{g/m}^3$ MAE**; LSTM achieved **$13.22\ \mu\text{g/m}^3$**.
- **Industrial & Traffic Impacted (Gucheng & Wanshouxigong)**:
  - Gucheng: Tuned GB achieved **$10.54\ \mu\text{g/m}^3$ MAE**; LSTM achieved **$16.75\ \mu\text{g/m}^3$**.
  - Wanshouxigong: Linear Regression achieved **$11.42\ \mu\text{g/m}^3$ MAE**; LSTM achieved **$15.22\ \mu\text{g/m}^3$**.

### Why Gradient Boosting Outperforms LSTM at 1-Hour:
1. **Direct Autoregressive Feature Alignment**: The Gradient Boosting model consumes explicit tabular features: $\text{PM}_{2.5}(t-1)$, $\text{PM}_{2.5}(t-2)$, 3h/6h rolling standard deviations, and explicit wind-interaction products. At $H=1\text{h}$, the correlation between $y_{t+1}$ and $y_t$ exceeds 0.97. Tree-based partitioning isolates this dominant feature cleanly.
2. **Multi-Task Objective Penalty in LSTM**: The Phase 2 LSTM uses a shared recurrent trunk and a 5-unit dense projection head predicting $[t+1, t+3, t+6, t+12, t+24]$ simultaneously. The loss function averages errors across all five horizons:
   $$\mathcal{L} = \frac{1}{5} \sum_{h \in \{1, 3, 6, 12, 24\}} \left| y_{t+h} - \hat{y}_{t+h} \right|$$
   This multi-task loss forces the recurrent hidden state to encode long-term multi-day trajectory context, slightly degrading acute 1-hour resolution to avoid overfitting the immediate lag.

---

## 5. Station-Wise Multi-Horizon Analysis Across All 12 Stations

The multi-horizon performance across all 12 stations highlights spatial heterogeneity in Beijing air quality:

| Station | 1h MAE | 3h MAE | 6h MAE | 12h MAE | 24h MAE | 1h $R^2$ | 24h $R^2$ | Best Horizon Advantage vs. Persistence |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Huairou** | **11.08** | **19.34** | **28.45** | **39.56** | **49.88** | 0.9288 | 0.2195 | +18.4% MAE reduction at 24h |
| **Dingling** | **12.17** | **20.21** | **29.12** | **40.11** | **51.22** | 0.9257 | 0.2014 | +17.9% MAE reduction at 24h |
| **Changping** | **13.09** | **21.55** | **30.88** | **41.76** | **52.45** | 0.9091 | 0.1742 | +16.2% MAE reduction at 24h |
| **Nongzhanguan**| 13.03 | 21.84 | 31.42 | 42.65 | 54.12 | 0.9472 | 0.1685 | +15.8% MAE reduction at 24h |
| **Tiantan** | 13.22 | 22.11 | 31.85 | 43.12 | 54.89 | 0.9484 | 0.1610 | +15.5% MAE reduction at 24h |
| **Aotizhongxin** | 13.56 | 23.39 | 33.43 | 44.88 | 55.96 | 0.9439 | 0.1869 | +17.6% MAE reduction at 24h |
| **Guanyuan** | 13.72 | 22.85 | 32.90 | 44.20 | 56.10 | 0.9391 | 0.1492 | +14.9% MAE reduction at 24h |
| **Wanliu** | 13.85 | 22.95 | 32.75 | 43.90 | 55.45 | 0.9317 | 0.1580 | +15.2% MAE reduction at 24h |
| **Shunyi** | 14.39 | 23.45 | 33.15 | 44.75 | 56.80 | 0.9269 | 0.1420 | +14.1% MAE reduction at 24h |
| **Dongsi** | 14.50 | 23.80 | 33.95 | 45.80 | 57.90 | 0.9411 | 0.1385 | +13.8% MAE reduction at 24h |
| **Wanshouxigong**| 15.22 | 24.65 | 34.80 | 46.90 | 58.70 | 0.9340 | 0.1250 | +12.9% MAE reduction at 24h |
| **Gucheng** | **16.75** | **27.15** | **38.20** | **50.85** | **63.20** | 0.9200 | 0.0890 | +11.2% MAE reduction at 24h |

---

## 6. Comprehensive Error Analysis & Diagnostics

### 6.1 Performance During High-Pollution Episodes (Tier Stratification)

Evaluating test errors across concentration tiers reveals how error scales with pollution severity:

| Concentration Tier | $\text{PM}_{2.5}$ Range ($\mu\text{g/m}^3$) | Test Sample % | 1h MAE ($\mu\text{g/m}^3$) | 1h MBE ($\mu\text{g/m}^3$) | 24h MAE ($\mu\text{g/m}^3$) | 24h MBE ($\mu\text{g/m}^3$) | Primary Diagnostic Insight |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Good** | $0 - 35$ | 39.7% | 6.89 | +4.67 | 49.59 | +48.88 | Slight overprediction; high baseline safety |
| **Moderate** | $35 - 75$ | 20.1% | 10.98 | +0.95 | 25.99 | +7.60 | Highly accurate, near-zero systematic bias |
| **Unhealthy for Sensitive** | $75 - 115$ | 13.6% | 14.65 | -3.55 | 34.55 | -17.15 | Onset of negative bias (underprediction) |
| **Unhealthy** | $115 - 150$ | 7.8% | 17.73 | -5.07 | 49.73 | -35.81 | Progressive underprediction of escalation |
| **Very Unhealthy** | $150 - 250$ | 12.6% | 21.27 | -6.37 | 76.09 | -67.80 | Model smooths steep regional accumulation |
| **Severe / Hazardous** | $> 250$ | 6.2% | **43.33** | **-35.26** | **197.48** | **-196.55** | Severe peak clipping; mean regression |

#### Key Diagnostic Insights:
1. **Asymmetric Error Profile**: Clean and moderate air exhibits positive bias ($\text{MBE} > 0$), while severe haze exhibits severe negative bias ($\text{MBE} < 0$).
2. **Mean-Regression at Day-Ahead ($H=24\text{h}$)**: For 24-hour predictions during episodes $> 250\ \mu\text{g/m}^3$, the LSTM predicts an average concentration of $\approx 85\ \mu\text{g/m}^3$ (close to the Beijing annual mean of $79.8\ \mu\text{g/m}^3$). This results in an MBE of $-196.55\ \mu\text{g/m}^3$. At 24h, without future numerical weather forecasts, the recurrent network settles into climatological probability.
3. **Squared Error Concentration**: The $6.2\%$ severe haze observations account for over $44\%$ of the total squared error, explaining the large gap between MAE and RMSE.

### 6.2 Diurnal Error Cycle (Hours 0–23)
- **Morning Rush Hour (07:00 – 09:00)**: Sharp spike in vehicular NOx and primary emissions induces a modest MAE increase ($14.8\ \mu\text{g/m}^3$ at 1h).
- **Afternoon Boundary Layer Ventilation (13:00 – 16:00)**: Solar insolation expands the planetary boundary layer, dispersing ground pollutants. Model error reaches its daily minimum (MAE: $11.2\ \mu\text{g/m}^3$).
- **Nocturnal Inversion (20:00 – 02:00)**: Surface radiation cooling traps stagnant emissions, causing elevated concentration and peak error variance.

### 6.3 Spatial Heterogeneity
- **Northern Suburban Stations (Dingling, Huairou)**: Shielded by the Yan Mountains, these sites experience cleaner background air. MAE is $20–35\%$ lower across all horizons.
- **Southern Urban & Industrial Stations (Gucheng, Wanshouxigong)**: Influenced by regional transport from Hebei province, these stations suffer from rapid pollution surges, resulting in higher MAE and RMSE.

---

## 7. Artifacts & Deliverables Registry

All generated evaluation outputs are persisted in the repository:

### Core Code Modules
- [`src/phase2/evaluate.py`](file:///Users/mihikamathur/Desktop/Placement/air-quality-forecasting/Air-Quality-Forecasting/src/phase2/evaluate.py): Multi-station evaluation engine, metric calculators, and tier analyzers.
- [`src/phase2/visualize.py`](file:///Users/mihikamathur/Desktop/Placement/air-quality-forecasting/Air-Quality-Forecasting/src/phase2/visualize.py): Publication visualization pipeline for the 8 core figures.
- [`tests/test_phase2_evaluate.py`](file:///Users/mihikamathur/Desktop/Placement/air-quality-forecasting/Air-Quality-Forecasting/tests/test_phase2_evaluate.py): Unit and mathematical verification suite (all 7 tests passing).
- [`notebooks/phase2_evaluation.ipynb`](file:///Users/mihikamathur/Desktop/Placement/air-quality-forecasting/Air-Quality-Forecasting/notebooks/phase2_evaluation.ipynb): Interactive evaluation and analysis walkthrough notebook.

### Tabular Datasets (`reports/results/phase2/`)
- [`macro_average_performance_by_horizon.csv`](file:///Users/mihikamathur/Desktop/Placement/air-quality-forecasting/Air-Quality-Forecasting/reports/results/phase2/macro_average_performance_by_horizon.csv) & [`.md`](file:///Users/mihikamathur/Desktop/Placement/air-quality-forecasting/Air-Quality-Forecasting/reports/results/phase2/macro_average_performance_by_horizon.md)
- [`pooled_performance_by_horizon.csv`](file:///Users/mihikamathur/Desktop/Placement/air-quality-forecasting/Air-Quality-Forecasting/reports/results/phase2/pooled_performance_by_horizon.csv) & [`.md`](file:///Users/mihikamathur/Desktop/Placement/air-quality-forecasting/Air-Quality-Forecasting/reports/results/phase2/pooled_performance_by_horizon.md)
- [`phase1_vs_phase2_1h_benchmark.csv`](file:///Users/mihikamathur/Desktop/Placement/air-quality-forecasting/Air-Quality-Forecasting/reports/results/phase2/phase1_vs_phase2_1h_benchmark.csv) & [`.md`](file:///Users/mihikamathur/Desktop/Placement/air-quality-forecasting/Air-Quality-Forecasting/reports/results/phase2/phase1_vs_phase2_1h_benchmark.md)
- [`pollution_tier_error_analysis.csv`](file:///Users/mihikamathur/Desktop/Placement/air-quality-forecasting/Air-Quality-Forecasting/reports/results/phase2/pollution_tier_error_analysis.csv) & [`.md`](file:///Users/mihikamathur/Desktop/Placement/air-quality-forecasting/Air-Quality-Forecasting/reports/results/phase2/pollution_tier_error_analysis.md)
- [`all_stations_multi_horizon_comparison.csv`](file:///Users/mihikamathur/Desktop/Placement/air-quality-forecasting/Air-Quality-Forecasting/reports/results/phase2/all_stations_multi_horizon_comparison.csv)

### Diagnostic Figures (`reports/figures/phase2/`)
1. [`01_actual_vs_predicted_timeseries.png`](file:///Users/mihikamathur/Desktop/Placement/air-quality-forecasting/Air-Quality-Forecasting/reports/figures/phase2/01_actual_vs_predicted_timeseries.png)
2. [`02_prediction_error_distribution.png`](file:///Users/mihikamathur/Desktop/Placement/air-quality-forecasting/Air-Quality-Forecasting/reports/figures/phase2/02_prediction_error_distribution.png)
3. [`03_training_vs_validation_loss.png`](file:///Users/mihikamathur/Desktop/Placement/air-quality-forecasting/Air-Quality-Forecasting/reports/figures/phase2/03_training_vs_validation_loss.png)
4. [`04_mae_comparison_across_models.png`](file:///Users/mihikamathur/Desktop/Placement/air-quality-forecasting/Air-Quality-Forecasting/reports/figures/phase2/04_mae_comparison_across_models.png)
5. [`05_rmse_comparison_across_models.png`](file:///Users/mihikamathur/Desktop/Placement/air-quality-forecasting/Air-Quality-Forecasting/reports/figures/phase2/05_rmse_comparison_across_models.png)
6. [`06_forecast_horizon_vs_mae.png`](file:///Users/mihikamathur/Desktop/Placement/air-quality-forecasting/Air-Quality-Forecasting/reports/figures/phase2/06_forecast_horizon_vs_mae.png)
7. [`07_station_wise_performance_comparison.png`](file:///Users/mihikamathur/Desktop/Placement/air-quality-forecasting/Air-Quality-Forecasting/reports/figures/phase2/07_station_wise_performance_comparison.png)
8. [`08_residual_analysis.png`](file:///Users/mihikamathur/Desktop/Placement/air-quality-forecasting/Air-Quality-Forecasting/reports/figures/phase2/08_residual_analysis.png)

---

## 8. Recommendations for the Final Phase 2 Report & Phase 3 Roadmap

1. **Deploy a Hybrid Cascaded Ensemble**:
   - For short-term predictions ($H=1\text{h}$): Route requests to the **Tuned Gradient Boosting** model (Macro MAE $10.08\ \mu\text{g/m}^3$).
   - For medium-to-long term forecasts ($H \ge 3\text{h}$): Route requests to the **Multi-Horizon LSTM** (Macro MAE $22.62\ \mu\text{g/m}^3$ at 3h, beating persistence by up to $+15.6\%$ at 24h).
2. **Asymmetric or Extreme-Value Loss Penalization**:
   - To counteract peak underprediction during hazardous haze events, incorporate an asymmetric loss penalty (e.g. Pinball / Quantile Loss or weighted Huber loss penalizing underprediction more severely than overprediction).
3. **Integration of Numerical Weather Predictions (NWP) for $H=24\text{h}$**:
   - At 24h, the lack of future meteorological forecasts forces the model to regress toward the seasonal mean. Feeding forecasted wind vectors and boundary layer height into future steps will dramatically reduce day-ahead error variance.

