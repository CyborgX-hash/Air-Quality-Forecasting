# Quantitative Model Evaluation Protocol & Error Analysis Guide

**Project:** Air Quality Forecasting via Gradient Boosting and LSTM Models  
**Role:** Model Evaluation and Error Analysis Engineer (Mihika)  
**Task:** One-hour-ahead PM2.5 forecasting ($PM2.5_{t+1}$, $\mu\text{g/m}^3$)  
**Dataset:** UCI Beijing Multi-Site Air Quality Dataset (12 Monitoring Sites, March 2013 – February 2017)  
**Phase 1 Models:** Persistence Baseline, Linear Regression Baseline, Gradient Boosting Regressor  

---

## 1. Evaluation Protocol Architecture

A rigorous, leakage-safe machine learning evaluation pipeline demands strict separation between model selection, hyperparameter tuning, and final generalization assessment.

```
Full Chronological Time Series (2013-03-01 00:00 to 2017-02-28 23:00)
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                                                                  │
│   TRAINING SET (70%)             VALIDATION SET (15%)         TEST SET (15%)     │
│   2013-03-01 to 2015-12-18       2015-12-18 to 2016-07-24     2016-07-24 to 2017 │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
            │                               │                           │
            ▼                               ▼                           ▼
    Model Training                  Model Selection             Final Generalization
    - Fit parameters                - Compare architectures     - Run once on winner
    - Fit scalers strictly          - Tune hyperparameters      - Never touch HPs
    - No access to Val/Test         - Select winner             - Reports true risk
```

### 1.1 Strict Separation: Validation vs. Test Sets

| Dimension | Validation Partition | Final Test Partition |
|:---|:---|:---|
| **Temporal Range** | Dec 18, 2015 16:00 – Jul 24, 2016 19:00 | Jul 24, 2016 20:00 – Feb 28, 2017 23:00 |
| **Primary Objective** | Model selection, feature pruning, hyperparameter tuning | Unbiased estimation of out-of-sample generalization error |
| **Feedback Loop** | Permitted (iteratively compare candidate models) | **Strictly Forbidden** (no parameter or hyperparameter adjustments) |
| **Data Leakage Risk** | Overfitting to validation dynamics through excessive trials | Information leakage if test performance is used to guide modeling |
| **Role in Protocol** | Determines which model is declared "Best Phase 1 Model" | Certified performance reported in the final IEEE paper |

### 1.2 Anti-Leakage Rules Enforced in Pipeline
1. **Zero Future Target in Features:** Target $y_t = PM2.5_{t+1}$ is never passed to feature generators or rolling estimators.
2. **Strict Chronological Slicing:** No random shuffling or standard k-fold CV. Forward time sequence is strictly maintained.
3. **Scaler & Imputation Isolation:** All feature statistics (medians, means, standard deviations) are computed exclusively on the training partition and transformed downstream onto validation and test sets.
4. **Missing Values Handling:** Missing target entries in test data are retained as identifiable exclusions rather than imputed, ensuring evaluation strictly measures verified observations.

---

## 2. Mathematical Definition & Physical Interpretation of Metrics

All metrics are implemented in `src/evaluate.py` as the single canonical source of truth.

### 2.1 Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N \left| y_i - \hat{y}_i \right|$$

- **Units:** $\mu\text{g/m}^3$ (identical to target concentration).
- **Mathematical Properties:** $L_1$ norm of the error vector. Linear loss penalty.
- **Physical Interpretation:** Represents the average expected deviation of the forecast from the true ground monitor reading. Unlike quadratic metrics, MAE treats a $20\ \mu\text{g/m}^3$ error exactly twice as severe as a $10\ \mu\text{g/m}^3$ error.
- **Optimal Predictor:** Minimizing MAE yields the conditional median: $\hat{y} = \text{Median}(y \mid \mathbf{x})$.

### 2.2 Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N \left( y_i - \hat{y}_i \right)^2}$$

- **Units:** $\mu\text{g/m}^3$.
- **Mathematical Properties:** $L_2$ norm. Quadratic penalty on residuals. Always $\text{RMSE} \ge \text{MAE}$.
- **Physical Interpretation:** Disproportionately penalizes large forecasting misses. In environmental protection and health advisories, missing an extreme $400\ \mu\text{g/m}^3$ haze outbreak causes severe public health emergencies; RMSE directly reflects the severity of these tail-end failures.
- **Diagnostic Spread ($\text{RMSE} - \text{MAE}$):** A large divergence between RMSE and MAE indicates high residual variance and erratic large outlier errors during abrupt weather transitions or industrial plume episodes.

### 2.3 Coefficient of Determination ($R^2$)
$$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2} = 1 - \frac{\text{MSE}}{\text{Var}(y)}$$

- **Interpretation:** The fraction of total ground-truth variance explained by the model relative to a naive baseline that continually predicts the historical sample mean $\bar{y}$.
- **Boundedness:** Maximum value is $1.0$ (perfect prediction). Can be negative ($R^2 < 0$) on validation or test sets if the model's mean squared error exceeds the natural variance of the data.
- **Why $R^2$ Can Be Negative:** If an out-of-sample model exhibits severe bias or high volatility, $\sum (y_i - \hat{y}_i)^2 > \sum (y_i - \bar{y})^2$, driving the second term $> 1.0$ and yielding $R^2 < 0$. This occurs with arbitrary constant baselines or overfitted models.

### 2.4 Mean Bias Error (MBE)
$$\text{MBE} = \frac{1}{N} \sum_{i=1}^N \left( \hat{y}_i - y_i \right)$$

- **Units:** $\mu\text{g/m}^3$.
- **Interpretation:** Measures directional systematic error:
  - $\text{MBE} > 0$: Systematic overprediction (model overestimates pollution).
  - $\text{MBE} < 0$: Systematic underprediction (model underestimates pollution).
  - $\text{MBE} \approx 0$: Unbiased on average (positive and negative errors balance).

### 2.5 Residuals Convention
$$e_i = y_i - \hat{y}_i$$
- $e_i > 0$: Ground truth was higher than predicted (underprediction).
- $e_i < 0$: Ground truth was lower than predicted (overprediction).

---

## 3. Baseline Formulations & Benchmark Models

### 3.1 Naive 1-Step Persistence Baseline ($PM2.5_t \to PM2.5_{t+1}$)
In discrete time series forecasting, the primary standard benchmark is the naive lag-1 forecast:
$$\hat{y}_{t+1} = PM2.5_t$$
Because hourly atmospheric pollution has substantial temporal autocorrelation ($r > 0.95$ at 1-hour lag), naive persistence provides a remarkably strong baseline ($R^2 \approx 0.94$). Any machine learning architecture (Linear Regression, Gradient Boosting, or LSTM) must demonstrate statistically significant improvement over this baseline to justify its computational complexity.

### 3.2 Constant Persistence Baseline
Memorizes the last observed target value from the training split and outputs that constant scalar across all future validation and test timesteps. Demonstrates the impact of seasonality and long-term distributional drift.

### 3.3 Linear Regression Baseline
Ordinary Least Squares (OLS) trained on all 70 engineered features (lags, rolling stats, cyclical time, and meteorology). Establishes the linear frontier of predictive accuracy.

### 3.4 Gradient Boosting Regressor
Ensemble of decision trees trained with gradient descent on Huber/squared loss. Captures non-linear thresholds (e.g., wind speed dispersion cutoff at $1.5\ \text{m/s}$, humidity-temperature dew point spread).

---

## 4. Viva Voce Technical Defense & Conceptual Explanations

### Q1: Why do we report both MAE and RMSE?
**Answer:** MAE measures the typical, expected magnitude of error on a linear scale, representing day-to-day forecast reliability. RMSE squares errors before averaging, penalizing catastrophic forecasting failures heavily. In air pollution management, failing to predict a toxic smog outbreak carries much higher public health consequences than a minor baseline error. Reporting both allows us to diagnose error variance: when RMSE is significantly larger than MAE, the model has intermittent extreme misses.

### Q2: Can $R^2$ be negative, and what does it mean?
**Answer:** Yes. In in-sample ordinary linear regression, $R^2$ is bounded between $0$ and $1$. However, on unseen validation or test partitions, $R^2$ is calculated as $1 - \frac{\text{MSE}}{\text{Var}(y_{\text{true}})}$. If the model's predictions on test data produce larger squared errors than simply predicting the mean of the test set, $\frac{\text{MSE}}{\text{Var}(y)} > 1$, resulting in a negative $R^2$. This signifies that the model performs worse than a simple horizontal line at the mean.

### Q3: Why is standard cross-validation prohibited for this dataset?
**Answer:** Standard cross-validation randomly partitions data into folds, violating the chronological arrow of time. This introduces lookahead data leakage, where a model trained on data from Tuesday predicts observations from Monday using rolling statistics and autoregressive lags that already incorporated the target. Only strict forward chronological splitting reflects real-world operational forecasting.

### Q4: Why must we never use test set performance to adjust model hyperparameters?
**Answer:** The test partition serves as our sole unbiased estimate of true generalization risk on future, unseen data. If test performance is consulted to select hyperparameters, select feature subsets, or modify architectures, information leaks from the test set into the modeling pipeline. The test set effectively becomes part of the training process, resulting in optimistic bias and invalidating the empirical findings of the study.

### Q5: What is the physical significance of residuals in air quality forecasting?
**Answer:** Residuals ($y - \hat{y}$) represent the unexplained variation. If residuals are normally distributed around zero with no autocorrelation, the model has extracted all available predictable signal. If residuals exhibit non-zero mean bias, skewness, or correlation with wind speed or hour of day, the model is systematically failing to represent physical atmospheric processes (such as planetary boundary layer collapse or nocturnal thermal inversions).
