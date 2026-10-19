# Exploratory Data Analysis (EDA) Findings & Quality Report

**Project:** Air Quality Forecasting via Gradient Boosting and LSTM Models  
**Phase:** Phase 1 — Traditional Machine Learning & Baseline Modeling  
**Author:** Saksham (Data Processing & EDA Engineer)  
**Target:** 1-hour ahead PM2.5 forecasting ($PM2.5_{t+1}$, $\mu\text{g/m}^3$)  
**Dataset:** UCI Beijing Multi-Site Air Quality Dataset (12 Monitoring Sites, 2013-03-01 to 2017-02-28)

---

## 1. Executive Summary

This report establishes the empirical data foundation for the Beijing air quality forecasting system. We analyzed 420,768 hourly records across 12 monitoring sites over four complete calendar years (March 1, 2013 to February 28, 2017). 

Our primary diagnostic findings are:
1. **High Structural Integrity:** All 12 stations have identical record counts (35,064 hourly observations each), exactly zero duplicate timestamps, and zero missing hourly time steps.
2. **Target Label Missingness:** Ground-truth PM2.5 is missing in 2.08% of records (8,739 rows citywide; 925 rows at Aotizhongxin). Following strict ML protocol, **target labels are never imputed** for supervised training.
3. **Severe Distribution Skewness:** PM2.5 displays heavy right-skewness (skewness = 1.39, mean = 79.8 $\mu\text{g/m}^3$, median = 55.0 $\mu\text{g/m}^3$), punctuated by episodic acute haze spikes exceeding 800 $\mu\text{g/m}^3$. These are validated physical phenomena (atmospheric stagnation) rather than sensor malfunctions.
4. **Dominant Temporal Cycles:** Pronounced wintertime surges (heating season coal burning + thermal inversions) and diurnal bimodal rush-hour/nocturnal peaks dominate the signal.
5. **Key Meteorological Levers:** Surface wind speed acts as an acute dispersion threshold ($< 2\ \text{m/s}$ causes heavy stagnation; $> 5\ \text{m/s}$ clears pollution), while wind direction defines a regional transport corridor (southern/eastern winds transport industrial emissions; north-northwesterly winds bring clean Siberian air).

---

## 2. Dataset Dimensions & Station Inventory

The dataset comprises 12 air monitoring stations distributed across the Beijing metropolitan area:

| Station Name | Site Classification | Total Records | Missing PM2.5 | Missing PM2.5 % | Mean PM2.5 ($\mu\text{g/m}^3$) | Median PM2.5 ($\mu\text{g/m}^3$) | Max PM2.5 ($\mu\text{g/m}^3$) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Aotizhongxin** | Urban / Benchmark | 35,064 | 925 | 2.64% | 82.8 | 58.0 | 898.0 |
| **Changping** | Suburban North-West | 35,064 | 774 | 2.21% | 71.1 | 46.0 | 882.0 |
| **Dingling** | Mountain / Background | 35,064 | 779 | 2.22% | 66.0 | 41.0 | 881.0 |
| **Dongsi** | Urban Central | 35,064 | 750 | 2.14% | 86.2 | 61.0 | 737.0 |
| **Guanyuan** | Urban Central | 35,064 | 616 | 1.76% | 82.9 | 59.0 | 680.0 |
| **Gucheng** | Urban South-West | 35,064 | 646 | 1.84% | 83.9 | 60.0 | 770.0 |
| **Huairou** | Mountain / Suburban | 35,064 | 953 | 2.72% | 69.6 | 47.0 | 762.0 |
| **Nongzhanguan** | Urban East | 35,064 | 628 | 1.79% | 84.8 | 59.0 | 844.0 |
| **Shunyi** | Suburban North-East | 35,064 | 913 | 2.60% | 79.5 | 55.0 | 941.0 |
| **Tiantan** | Urban South-Central | 35,064 | 677 | 1.93% | 82.1 | 59.0 | 821.0 |
| **Wanliu** | Urban North-West | 35,064 | 382 | 1.09% | 83.4 | 59.0 | 957.0 |
| **Wanshouxigong** | Urban South-West | 35,064 | 696 | 1.98% | 85.0 | 60.0 | 999.0 |
| **Total / Citywide** | **All 12 Stations** | **420,768** | **8,739** | **2.08%** | **79.8** | **55.0** | **999.0** |

---

## 3. Data Quality & Preprocessing Decisions

### 3.1 Missing Value Analysis
Missing rates across the entire multi-station corpus break down as follows:
- **CO:** 4.92% (20,701 missing values) — highest missingness among features.
- **O3:** 3.16% (13,277 missing values).
- **NO2:** 2.88% (12,116 missing values).
- **SO2:** 2.14% (9,021 missing values).
- **PM2.5 (Target):** 2.08% (8,739 missing values).
- **PM10:** 1.53% (6,449 missing values).
- **Meteorology (TEMP, PRES, DEWP, RAIN, WSPM):** $< 0.10\%$ missing ($318$ to $403$ missing values).
- **Wind Direction (`wd`):** 0.43% (1,822 missing values).

### 3.2 Imputation Rules Enforced in `src/preprocessing.py`
1. **Target Variable (`PM2.5`): Zero Imputation.**  
   Imputing missing target labels with synthetic values (e.g. forward-fill, linear interpolation, or mean) would distort the true loss function and train downstream models on artificial predictions. Missing targets are retained as `NaN` and flagged in `target_valid = False`. Downstream training scripts use `get_valid_training_mask(df, horizon=1)` to train strictly on verified pairs.
2. **Causal Forward Imputation for Features:**  
   To prevent lookahead bias (data leakage), backward-fill (`bfill`) and bidirectional spline/linear interpolation are strictly prohibited. Missing feature values are forward-filled up to a maximum threshold of $6$ hours within each station group.
3. **Long Outages & Leading Missing Values:**  
   Any feature NaNs remaining after forward-fill (e.g., sensor outages $> 6$ hours or missing initial records) are filled using pre-calculated station-specific historical medians.
4. **Wind Direction Transformation:**  
   The 16-point categorical wind direction is forward-filled and mapped to radians $\theta \in [0, 2\pi)$, then decomposed into orthogonal cyclical components:
   $$\text{wd\_sin} = \sin(\theta), \quad \text{wd\_cos} = \cos(\theta)$$
   This resolves the discontinuity between $350^\circ$ and $10^\circ$ around true North.

---

## 4. Deep Analytical Findings & Visualizations

### 4.1 Target Distribution & Outlier Legitimacy (Figures 02 & 03)
- **High Kurtosis & Positive Skew:** The distribution is non-Gaussian with a median of 58.0 $\mu\text{g/m}^3$ and a long right tail stretching to 898 $\mu\text{g/m}^3$ at Aotizhongxin. Over 38% of hours exceed China's 24-hour National Ambient Air Quality Standard of 75 $\mu\text{g/m}^3$.
- **Validation of Extreme Values:** Readings $> 300\ \mu\text{g/m}^3$ occur in prolonged multi-day clusters during winter anticyclones. Because adjacent pollutants ($CO, NO_2, SO_2$) simultaneously spike during these hours, these extremes represent true severe smog events rather than sensor dropouts or electrical noise. They must **not** be clipped or discarded.
- **Modeling Recommendation:** For linear baselines or neural nets, consider training on $\log(1 + PM2.5)$ to stabilize variance. For tree-based gradient boosting (LightGBM/XGBoost), monotonic tree splits handle raw values naturally without transformation.

### 4.2 Temporal Dynamics: Diurnal & Seasonal Cycles (Figures 04 & 05)
- **Diurnal Bimodal Profile:**
  - **Morning Peak (08:00–10:00):** Driven by morning vehicular traffic emitting $NO_x$ and primary carbonaceous particles into a shallow morning boundary layer.
  - **Afternoon Trough (14:00–16:00):** Solar radiation reaches peak intensity, inducing strong vertical thermal turbulence and convective mixing that dilutes surface pollutants across a deep boundary layer (often $> 1.5\ \text{km}$).
  - **Nocturnal/Evening Peak (20:00–23:00):** Planetary boundary layer (PBL) collapses to $< 300\ \text{m}$, creating a ground-level thermal inversion that traps heavy diesel freight traffic and domestic heating emissions.
- **Seasonal Heating Surge:**
  - Winter (Dec–Feb) PM2.5 averages $102\ \mu\text{g/m}^3$, compared to $48\ \mu\text{g/m}^3$ in Summer (Jun–Aug).
  - This 2.1x winter-to-summer ratio is driven by centralized coal combustion for municipal heating across Northern China coupled with frequent Siberian high-pressure inversions.

### 4.3 Multi-Pollutant Collinearity (Figure 06 & 09)
The correlation matrix reveals high structural collinearity among combustion byproducts:
- **$PM_{10}$ ($r = 0.88$):** Primary co-pollutant; fine particulates ($PM_{2.5}$) make up ~65–75% of total coarse mass ($PM_{10}$) in urban Beijing.
- **$CO$ ($r = 0.78$):** Inert tracer of incomplete fossil-fuel combustion. Its high correlation makes it one of the strongest features for predicting PM2.5 build-up.
- **$NO_2$ ($r = 0.66$):** Traffic tracer and precursor to secondary ammonium nitrate aerosols.
- **$SO_2$ ($r = 0.48$):** Coal-burning marker, showing strong seasonal correlation with PM2.5 in winter ($r > 0.70$).
- **$O_3$ ($r = -0.15$):** Demonstrates an inverse relationship with PM2.5 in cold seasons. Thick particulate haze blocks UV radiation, attenuating photochemical ozone generation ($O_3$), while high NO emissions titrate existing ozone ($NO + O_3 \to NO_2 + O_2$).

### 4.4 Meteorological Drivers & Transport Corridors (Figures 07 & 08)
- **Wind Speed Clearance Threshold:**
  - Stagnant air ($\text{WSPM} < 1.5\ \text{m/s}$): Average PM2.5 = $105\ \mu\text{g/m}^3$.
  - Moderate breeze ($2.0 \le \text{WSPM} < 4.0\ \text{m/s}$): Average PM2.5 = $62\ \mu\text{g/m}^3$.
  - Strong winds ($\text{WSPM} \ge 5.0\ \text{m/s}$): Average PM2.5 = $26\ \mu\text{g/m}^3$.
  - Wind speed exhibits an exponential decay relationship with particulate concentration.
- **Directional Transport Corridor:**
  - **South-Southeast Winds (SSE, S, SE):** Deliver the highest average PM2.5 ($102–115\ \mu\text{g/m}^3$). Southerly winds funnel industrial sulfur and nitrogen plumes from Hebei and Tianjin into the Beijing topographic basin (surrounded by the Yan and Taihang mountains on the north and west).
  - **North-Northwest Winds (NW, NNW, N):** Deliver the cleanest air ($42–52\ \mu\text{g/m}^3$). Cold, high-pressure air masses descending from the Mongolian plateau sweep clean air through the mountain passes, rapidly terminating haze episodes.
- **Precipitation Wash-out:**
  - Wet deposition significantly attenuates airborne particles. Hours with $\text{RAIN} > 0\ \text{mm}$ exhibit a median PM2.5 of $38\ \mu\text{g/m}^3$, compared to $56\ \mu\text{g/m}^3$ during dry conditions.
- **Dew Point & Humidity:**
  - Positive correlation ($r = 0.27$) with PM2.5. High atmospheric moisture accelerates heterogeneous aqueous-phase chemical reactions converting gaseous $SO_2$ and $NO_2$ into secondary sulfate and nitrate particles.

---

## 5. Modeling Implications for Phase 1 (Gradient Boosting)

For teammates Ajit and Priyanshu, the EDA directly supports the following feature engineering and validation architecture:

1. **Autoregressive Lag Features:**
   - Strong autocorrelation mandates lag terms: $PM2.5_{t-1}, PM2.5_{t-2}, PM2.5_{t-3}, PM2.5_{t-24}$.
   - Add $PM10_{t-1}, CO_{t-1}, NO2_{t-1}$ to capture concurrent precursor dynamics.
2. **Rolling Window Aggregations:**
   - Causal rolling statistics (mean, std, min, max) over past $3, 6, 12, 24$ hours to capture momentum and rate-of-change ($PM2.5_t - \text{rolling\_mean}_{24h}$).
3. **Wind Interaction Features:**
   - Wind speed and direction should be multiplied to model directional advection vectors:
     $$\text{wind\_u} = \text{WSPM} \times \cos(\theta), \quad \text{wind\_v} = \text{WSPM} \times \sin(\theta)$$
4. **Validation Strategy:**
   - **Do NOT use standard random K-Fold cross-validation.** Random splitting will cause extreme temporal data leakage due to high autocorrelation between consecutive hours.
   - **Recommended Split:**
     - Train: March 1, 2013 to February 29, 2016 (36 months)
     - Validation: March 1, 2016 to October 31, 2016 (8 months)
     - Test: November 1, 2016 to February 28, 2017 (4 months, evaluating the critical winter heating season)

---

## 6. Known Dataset Limitations

1. **Missing Precursor Tracers:** Volatile organic compounds (VOCs) and ammonia ($NH_3$), which are critical for secondary organic aerosol (SOA) formation, are not measured in the dataset.
2. **Boundary Layer Height Omission:** The planetary boundary layer (PBL) height is not directly included, requiring surface temperature, pressure, and diurnal hour to act as proxies.
3. **Spatial Aggregation:** Station sensors are localized point measurements that may experience micro-scale road or construction influence.
