# Air Quality Index (AQI) Forecasting Model Comparison

This report compares the performance of various forecasting models applied to the Philippine Cities AQI dataset.

## 1. Statistical and Deep Learning Forecasting Models
*Evaluated across 10 cities using `src/modeling/forecasting_models.py`*

| Model | MFE (Bias) | MAE | MSE | MAPE (%) | RMSE | Cities Evaluated |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LSTM** | -0.1027 | 0.4421 | 0.3179 | 32.63% | 0.5639 | 10 |
| Average Forecast | -0.0928 | 0.4621 | 0.3212 | 34.06% | 0.5668 | 10 |
| Moving Average | 0.2224 | 0.4863 | 0.3867 | 28.99% | 0.6219 | 10 |
| ARIMA | 0.4021 | 0.5775 | 0.6013 | 30.72% | 0.7754 | 10 |
| SARIMA | 0.4864 | 0.6579 | 0.7073 | 34.15% | 0.8410 | 10 |
| Naïve Forecast | 0.5566 | 0.6099 | 0.7626 | 29.16% | 0.8733 | 10 |
| Holt-Winters | 0.5809 | 0.6352 | 0.8199 | 30.56% | 0.9055 | 10 |

## Summary
The forecasting analysis focuses on traditional time series techniques (Naïve, Average, Moving Average, Holt-Winters, ARIMA, SARIMA) and Deep Learning (LSTM).

Following the principles of the **Business Analytics Module**, we evaluated the models using:
- **Mean Forecast Error (MFE)**: To measure average forecasting bias.
- **Mean Absolute Error (MAE)**: To measure average magnitude of error.
- **Mean Squared Error (MSE)**: To penalize large errors heavily.
- **Mean Absolute Percentage Error (MAPE)**: For relative error comparison.
- **RMSE**: For a scale-dependent error magnitude.

### Key Findings:
- **LSTM (Long Short-Term Memory)** emerged as the most accurate model overall after stabilizing the training process and handling data edge cases.
- **Average Forecast** and **Moving Average** remain very strong and computationally efficient baselines.
- The **MFE** results indicate that while LSTM and Average methods have a slight under-forecasting bias (negative MFE), traditional methods like Naive and Holt-Winters tend to over-forecast (positive MFE) on this dataset.
