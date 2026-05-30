# 🟢 FINAL ALIGNED MODEL PIPELINE (DEFENSE-READY)

## 🎯 Core Principle

Your pipeline must reflect this logic:

> “Each city is treated as an independent or semi-independent urban air system influenced by local anthropogenic activity and regional climate patterns.”

This directly matches:

* HUC classification (urban systems)
* population concentration (emission intensity)
* regional grouping (Luzon/Visayas/Mindanao)
* dataset constraint (available cities only)

---

# 🧭 1. DATA ORGANIZATION LAYER (CRITICAL ALIGNMENT STEP)

## Input Dataset

```
data/all/2026
```

Contains:

* hourly multivariate data
* multiple cities
* Jan–Feb 2026 timeline

---

## Step 1.1: City Stratification

You MUST explicitly structure data as:

### GROUP A — PER-CITY MODELS (PRIMARY APPROACH)

Each city becomes its own time series:

* Manila → Model 1
* Quezon City → Model 2
* Makati → Model 3
* ...

✔ This aligns with:

* HUC independence
* localized pollution dynamics
* avoids spatial leakage

---

### GROUP B — REGIONAL ANALYSIS LAYER (SECONDARY)

Aggregate results by:

* Luzon
* Visayas
* Mindanao

✔ Used ONLY for:

* result interpretation
* comparison tables
* visualization

NOT training.

---

# ⚙️ 2. PREPROCESSING PIPELINE (CITY-AWARE)

## Step 2.1 — City-wise Sorting

For each city:

```python id="city_sort"
sort by timestamp ascending
```

---

## Step 2.2 — Missing Data Handling (CITY LEVEL)

Apply per city:

* interpolation OR
* KNN imputation

✔ IMPORTANT:
Do NOT impute globally across cities

---

## Step 2.3 — Feature Scaling (MODEL-DEPENDENT)

* Required for LSTM only
* Optional for tree models

Applied per city dataset independently

---

# 🔁 3. FEATURE ENGINEERING (CITY-LOCALIZED)

All features must be computed:

## Per city, NOT global

### Lag Features (per city)

* AQI(t-1), AQI(t-2), AQI(t-24)

### Pollutants (per city)

* PM2.5, PM10, O3, CO, NO2, SO2, NH3

### Temporal Features

* hour
* day of week

### Optional Rolling Features

* 24-hour rolling mean (city-specific)

---

# 🤖 4. MODEL PIPELINE (ALIGNED TO CITY LOGIC)

You train models **per city OR shared model with city feature**.

You must explicitly choose one:

---

# ✅ OPTION A (RECOMMENDED — DEFENSE STRONGEST)

## 🟢 PER-CITY MODELING PIPELINE

Each city gets its own models:

For each city:

### Classical Models

* Naive
* ARIMA
* SARIMA (s = 24)
* Holt-Winters

### ML Models

* XGBoost
* LightGBM
* Random Forest

### Deep Learning

* LSTM

✔ Advantages:

* matches HUC logic
* avoids spatial contamination
* easiest to defend

---

# ⚠️ OPTION B (GLOBAL MODEL — SECONDARY)

Single model trained on all cities:

Add feature:

```text id="city_feature"
city_encoded
```

✔ Must be justified as:

> “shared atmospheric learning across urban systems”

BUT:

* higher risk in defense questions
* weaker interpretability

---

# 🧠 5. LSTM PIPELINE (CITY-AWARE VERSION)

For each city:

```text id="lstm_city"
Input: multivariate sequences (city-specific)
Output: AQI(t+1)
```

Architecture:

* LSTM(64)
* Dropout(0.2)
* LSTM(32)
* Dense(1)

Loss:

* MSE

---

# 📊 6. EVALUATION PIPELINE (CITY-FIRST DESIGN)

You compute metrics:

## Per city:

* MAE
* RMSE
* MAPE

---

## Then aggregate:

### Regional aggregation:

* Luzon avg MAE
* Visayas avg MAE
* Mindanao avg MAE

---

## Final comparison table:

| City | ARIMA | SARIMA | LSTM | Best Model |

---

# 🧭 7. CITY SELECTION LOGIC IN PIPELINE (IMPORTANT ALIGNMENT STATEMENT)

You explicitly embed this logic in methodology:

> “Modeling was performed at the city level to reflect Highly Urbanized City (HUC) independence, where each urban area exhibits distinct emission dynamics driven by localized population density, transportation activity, and industrial structure.”

Then:

> “Regional grouping (Luzon, Visayas, Mindanao) was used solely for result aggregation and comparative analysis, not for model training.”

---

# ⚠️ 8. CRITICAL DEFENSE ALIGNMENT POINTS

This pipeline ensures you can answer:

### “Why per-city modeling?”

✔ HUC independence
✔ localized pollution patterns
✔ avoids spatial leakage

---

### “Why include multiple cities?”

✔ improves external validity
✔ supports regional comparison
✔ reflects national coverage

---

### “Why not one global model?”

✔ cities have different:

* emission profiles
* traffic density
* meteorological behavior

---

# 🟢 FINAL SYSTEM STRUCTURE

Your final system is:

## Level 1 — City Models

* 10 independent time-series pipelines

## Level 2 — Model Comparison

* ARIMA vs SARIMA vs LSTM per city

## Level 3 — Regional Analysis

* aggregated performance summaries

---

# 🚀 RESULT: WHY THIS IS DEFENSE-PROOF

✔ matches PSA HUC logic
✔ matches dataset constraints
✔ avoids leakage
✔ avoids oversimplified global modeling
✔ scientifically consistent with urban air modeling literature
✔ easy to explain under questioning

---
