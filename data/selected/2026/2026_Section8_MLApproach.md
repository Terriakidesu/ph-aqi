SECTION 8: Potential Machine Learning Approach

Baseline Algorithm
Random Forest Classifier

Real Basis
- Target is integer multiclass AQI label (main.aqi). Rows: 14159.
- Time coverage: 2026-01-01 00:00:00+08:00 to 2026-02-28 23:00:00+08:00.
- Class distribution:
  - AQI 1: 4666 (32.95%)
  - AQI 2: 7993 (56.45%)
  - AQI 3: 1469 (10.38%)
  - AQI 4: 31 (0.22%)
- Mean AQI by city:
  - Davao: 1.21
  - Cagayan de Oro: 1.24
  - Cebu City: 1.53
  - Bacolod: 1.55
  - Iloilo City: 1.76
  - Makati City: 2.05
  - Pasig: 2.05
  - Taguig: 2.05
  - Quezon City: 2.14
  - Manila: 2.19
- Correlation with AQI (Pearson):
  - components.o3: 0.850
  - components.pm10: 0.556
  - components.so2: 0.516
  - components.pm2_5: 0.478
  - components.co: 0.104
  - components.no: 0.079
  - components.no2: -0.064
  - components.nh3: -0.152
- Random Forest is a strong baseline for nonlinear tabular relationships and gives feature importance.

Other Algorithms for Comparative Analysis
- XGBoost or LightGBM
- Multinomial Logistic Regression
- CatBoost
- Class-weighted / Balanced Random Forest
- Support Vector Machine (multiclass)

Expected Challenges
- Severe class imbalance (especially minority AQI classes).
- Time leakage risk if train/test split is random instead of time-based.
- City-level distribution shift (different AQI profiles across cities).
- Imputed rows may smooth extremes and affect model boundaries.
- Correlated pollutants may affect coefficient interpretation in linear models.