# Data Dictionary: UCI Beijing Multi-Site Air Quality Dataset

**Project:** Air Quality Forecasting via Gradient Boosting and LSTM Models  
**Component:** Data Processing, Quality Validation & EDA  
**Author:** Saksham (Data Processing & EDA Engineer)  
**Target Variable:** 1-hour ahead PM2.5 concentration ($PM2.5_{t+1}$, $\mu\text{g/m}^3$)  
**Temporal Coverage:** March 1, 2013 00:00 to February 28, 2017 23:00 (Hourly, 35,064 records per station, 420,768 records total across 12 stations)

---

## 1. Raw Dataset Columns & Physical Characteristics

| Column Name | Source Type | Cleaned Type | Unit | Description | Plausible Physical Bounds | Observed Min | Observed Max | Missing Count (All 12 Stations) | Missing % (All) | Missing Count (Aotizhongxin) | Missing % (Aotizhongxin) | ML Feature Role / Imputation Rule |
|:---|:---|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `No` | `int64` | `int64` | Dimensionless | Row sequence number within station file (1 to 35,064) | $[1, \infty)$ | 1 | 35,064 | 0 | 0.000% | 0 | 0.000% | Metadata / index. Excluded from model features. |
| `year` | `int64` | `int64` | Years | Calendar year of measurement (2013–2017) | $[2013, 2017]$ | 2013 | 2017 | 0 | 0.000% | 0 | 0.000% | Temporal index. Used in building `datetime`. |
| `month` | `int64` | `int64` | Months | Month of year (1–12) | $[1, 12]$ | 1 | 12 | 0 | 0.000% | 0 | 0.000% | Seasonality indicator. Can be encoded cyclically ($\sin, \cos$). |
| `day` | `int64` | `int64` | Days | Day of month (1–31) | $[1, 31]$ | 1 | 31 | 0 | 0.000% | 0 | 0.000% | Temporal index. Used in building `datetime`. |
| `hour` | `int64` | `int64` | Hours | Hour of day in local Beijing Time (0–23) | $[0, 23]$ | 0 | 23 | 0 | 0.000% | 0 | 0.000% | Diurnal indicator. Highly predictive of rush hour and boundary layer height. |
| `PM2.5` | `float64` | `float64` | $\mu\text{g/m}^3$ | Fine particulate matter with aerodynamic diameter $\le 2.5\ \mu\text{m}$ | $[0, 1500]$ | 2.0 | 999.0 | 8,739 | 2.077% | 925 | 2.638% | **Target variable ($y_{t+1}$) & key autoregressive feature ($X_t$). NEVER imputed for supervised labels.** Marked in `target_valid`. |
| `PM10` | `float64` | `float64` | $\mu\text{g/m}^3$ | Inhalable particulate matter with aerodynamic diameter $\le 10\ \mu\text{m}$ | $[0, 1500]$ | 2.0 | 999.0 | 6,449 | 1.533% | 718 | 2.048% | Primary feature ($r = 0.88$ with PM2.5). Causally forward-filled ($\le 6$h), remaining filled with station median. |
| `SO2` | `float64` | `float64` | $\mu\text{g/m}^3$ | Sulfur dioxide concentration; indicator of coal combustion and industrial emissions | $[0, 1000]$ | 0.2856 | 500.0 | 9,021 | 2.144% | 935 | 2.667% | Predictive feature ($r = 0.48$). Causally forward-filled ($\le 6$h), remaining filled with station median. |
| `NO2` | `float64` | `float64` | $\mu\text{g/m}^3$ | Nitrogen dioxide concentration; vehicle exhaust and industrial combustion | $[0, 500]$ | 1.0265 | 290.0 | 12,116 | 2.879% | 1,023 | 2.918% | Predictive feature ($r = 0.66$). Causally forward-filled ($\le 6$h), remaining filled with station median. |
| `CO` | `float64` | `float64` | $\mu\text{g/m}^3$ | Carbon monoxide concentration; incomplete combustion tracer | $[0, 20000]$ | 100.0 | 10,000.0 | 20,701 | 4.920% | 1,776 | 5.065% | Predictive feature ($r = 0.78$). Causally forward-filled ($\le 6$h), remaining filled with station median. |
| `O3` | `float64` | `float64` | $\mu\text{g/m}^3$ | Ground-level surface ozone concentration; secondary photochemical oxidant | $[0, 1500]$ | 0.2142 | 1,071.0 | 13,277 | 3.155% | 1,719 | 4.902% | Secondary pollutant ($r = -0.15$). Causally forward-filled ($\le 6$h), remaining filled with station median. |
| `TEMP` | `float64` | `float64` | $^\circ\text{C}$ | Ambient dry-bulb air temperature at 2m above ground | $[-50, 60]$ | -19.9 | 41.6 | 398 | 0.095% | 20 | 0.057% | Meteorological driver. Governs vertical atmospheric stability and chemistry. Causal ffill + median. |
| `PRES` | `float64` | `float64` | $\text{hPa}$ | Atmospheric barometric surface pressure | $[900, 1100]$ | 982.4 | 1,042.8 | 393 | 0.093% | 20 | 0.057% | Meteorological driver. Tracks synoptic pressure fronts (cold Siberian anticyclones vs warm troughs). Causal ffill + median. |
| `DEWP` | `float64` | `float64` | $^\circ\text{C}$ | Dew point temperature; proxy for atmospheric moisture and relative humidity | $[-50, 50]$ | -43.4 | 29.1 | 403 | 0.096% | 20 | 0.057% | Meteorological driver. High dew point correlates with humid secondary aerosol growth. Causal ffill + median. |
| `RAIN` | `float64` | `float64` | $\text{mm/h}$ | Hourly accumulated liquid precipitation | $[0, 300]$ | 0.0 | 72.5 | 390 | 0.093% | 20 | 0.057% | Wash-out / wet deposition indicator. Causal ffill ($\le 6$h), remaining filled with 0.0 (dry). |
| `wd` | `object` / `str` | `category` / `str` | Compass direction | 16-point compass surface wind direction (`N`, `NNW`, `SSE`, etc.) | 16 discrete sectors | N/A | N/A | 1,822 | 0.433% | 81 | 0.231% | Categorical feature. Forward-filled causally and transformed to continuous radians (`wd_sin`, `wd_cos`). |
| `WSPM` | `float64` | `float64` | $\text{m/s}$ | Horizontal surface wind speed at 10m height | $[0, 50]$ | 0.0 | 13.2 | 318 | 0.076% | 14 | 0.040% | Key dispersion driver ($r = -0.27$). Stagnation when $< 2\ \text{m/s}$; clearance when $> 5\ \text{m/s}$. Causal ffill + median. |
| `station` | `object` / `str` | `category` / `str` | Categorical name | Name of air quality monitoring site (12 unique sites) | 12 fixed sites | N/A | N/A | 0 | 0.000% | 0 | 0.000% | Categorical spatial identifier. Preserved across all loaders and transformations. |

---

## 2. Engineered & Derived Columns

The data pipeline introduces the following deterministic columns:

| Column Name | Data Type | Formula / Origin | Purpose |
|:---|:---|:---|:---|
| `datetime` | `datetime64[ns]` | `pd.to_datetime(year, month, day, hour)` | Continuous timestamp enabling monotonic time-series ordering, lag creation, and temporal splits. |
| `target_valid` | `bool` | `df['PM2.5'].notnull()` | Boolean flag indicating whether the true target observation at $t$ is non-null. Downstream models must use this to filter out invalid supervised training pairs. |
| `wd_deg` | `float64` | Compass angle $\theta \in [0^\circ, 337.5^\circ]$ where $\text{N}=0^\circ, \text{E}=90^\circ, \text{S}=180^\circ, \text{W}=270^\circ$ | Continuous angular representation of wind direction. |
| `wd_sin` | `float64` | $\sin(\theta_{\text{radians}})$ | Cyclical East-West wind vector component (East = +1, West = -1). Solves the artificial $0^\circ/360^\circ$ boundary issue. |
| `wd_cos` | `float64` | $\cos(\theta_{\text{radians}})$ | Cyclical North-South wind vector component (North = +1, South = -1). |

---

## 3. Data-Quality Policies & Design Principles

1. **Target Safety (No Synthetic Labels):**
   - The target $PM2.5$ column is **never imputed** for supervised training. Imputing the target would train downstream gradient boosting models on artificial model artifacts, invalidating test set evaluation.
2. **Strict Time-Series Causality:**
   - Under no circumstances does the pipeline perform backward fill (`bfill`) or bidirectional interpolation. In real-time inference at hour $t$, observations at $t+1, t+2$ do not exist. Using future observations would introduce lookahead bias (data leakage).
   - Only causal forward-fill ($limit \le 6$ hours) and pre-computed station historical medians are permitted.
3. **Partition Isolation by Station:**
   - All grouped transformations are partitioned strictly by `station`. Sensor data from one monitoring site is never propagated into another site.
4. **Zero Silent Dropping:**
   - All 35,064 records per station (420,768 records total) are preserved. Rows are never dropped during feature cleaning. Downstream supervised pairing uses `get_valid_training_mask(df, horizon=1)` to filter invalid targets without altering the underlying dataset.
