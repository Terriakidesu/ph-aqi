# Detailed Project Procedures: AQI Forecasting

This document provides a comprehensive technical explanation of the methodologies and steps implemented in the "Predictive Modeling of Hourly Air Quality Index" project.

---

## 1. Data Selection and Preparation
**Objective:** To select a representative and high-quality dataset of air quality in the Philippines.

*   **City Selection:** Based on the `city_selection_basis.md`, we prioritized Highly Urbanized Cities (HUCs) with high population density (e.g., Manila, Cebu, Davao). This ensures the model focuses on areas where human activity significantly impacts AQI.
*   **Data Sourcing:** The primary dataset was sourced from a 2026 Kaggle dataset containing hourly AQI readings. We used the subset located in `data/selected/202601/202601_SelectedData.csv`.
*   **Regional Balancing:** Cities were selected from Luzon, Visayas, and Mindanao to ensure the model's findings are geographically diverse and not biased toward the capital region.

## 2. Data Cleaning and Preprocessing
**Objective:** To ensure data integrity and prepare it for time series analysis.

*   **Datetime Parsing:** Raw strings were converted into Python `datetime` objects using `pandas.to_datetime`. This is critical for chronological sorting and seasonal analysis.
*   **Missing Value Imputation:** To handle gaps in sensor data, we implemented a grouped imputation strategy. For each city, missing values were filled using **Forward Fill (ffill)** followed by **Backward Fill (bfill)**. This maintains the local continuity of the time series.
*   **Chronological Sorting:** The dataset was sorted primarily by `city_name` and secondarily by `datetime` to prevent "look-ahead bias" during training.

## 3. Exploratory Data Analysis (EDA)
**Objective:** To understand the underlying patterns of the data.

*   **Statistical Profiling:** We analyzed 13,863 observations to determine the mean, variance, and distribution of AQI levels.
*   **Seasonality Detection:** We implemented an autocorrelation check (`acf`) at a 24-hour lag. A correlation coefficient > 0.5 was used as the threshold to classify a city's AQI pattern as "seasonal," which directly influenced whether the **SARIMA** model was deployed.
*   **Stationarity Check:** Visual and statistical checks were performed to ensure the data was suitable for models like ARIMA, which require stationary or differenced series.

## 4. Feature Engineering
**Objective:** To transform raw data into a format that maximizes model predictive power.

*   **Target Definition:** `main.aqi` was defined as the target variable for univariate forecasting.
*   **Time Series Splitting:** Instead of a random split, we used a **chronological split (80/20)** for each city. The first 80% of the time series was used for training, and the final 20% was reserved for testing to simulate real-world forecasting.
*   **Normalization:** For the LSTM model, we applied `MinMaxScaler` to scale AQI values between 0 and 1, facilitating faster and more stable neural network convergence.

## 5. Model Development and Training
**Objective:** To implement a diverse suite of forecasting algorithms.

*   **Naive & Average Baselines:** Implemented as "sanity checks." Naive assumes the next value is the same as the last; Average assumes it is the mean of all history.
*   **Moving Average:** Implemented with a 24-hour window to capture local trends.
*   **Holt-Winters (Exponential Smoothing):** Used to capture both level and additive trend components without assuming seasonality.
*   **ARIMA (1,1,1):** A statistical model that combines Auto-Regressive (AR) and Moving Average (MA) components with differencing (I).
*   **SARIMA:** A Seasonal ARIMA extension applied only when Step 3 detected seasonality. It uses a `(1,1,1)x(1,1,1,24)` configuration.
*   **LSTM (Long Short-Term Memory):** A Recurrent Neural Network (RNN) designed to remember long-term dependencies. We used a 10-hour lookback window and a 50-unit hidden layer.

## 6. Model Evaluation
**Objective:** To measure accuracy using metrics defined in the Business Analytics curriculum.

For every city and every model, we calculated:
*   **MFE (Mean Forecast Error):** Measures bias (whether the model consistently over or under predicts).
*   **MAE (Mean Absolute Error):** Measures the average magnitude of the errors.
*   **MSE (Mean Squared Error):** Penalizes larger errors more heavily than smaller ones.
*   **RMSE (Root Mean Squared Error):** Provides the error in the same units as the target variable.
*   **MAPE (Mean Absolute Percentage Error):** Provides a percentage-based perspective on accuracy.

## 7. Model Comparison and Selection
**Objective:** To identify the best-performing model for deployment.

*   **Aggregation:** Performance metrics were averaged across all 10 cities to find the most robust universal model.
*   **Outlier Analysis:** We identified cities (like Cagayan de Oro) where specific models (like LSTM) struggled, helping refine our recommendations.
*   **Exporting Findings:** All exhaustive results were exported to `data/predictions/forecast_model_comparison.csv` and a human-readable summary was created in `docs/FORECAST_RESULTS.md`.
