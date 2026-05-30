# Predictive Modeling of Hourly Air Quality Index in Selected Philippine Cities Using Multivariate Time Series Forecasting

**Rainer Astodillo**, *Bachelor of Science in Information System* Carlos Hilado Memorial State University, EB Magalona, Negros Occidental

rainerastodillo.chmsu@email.com

**Jude Dojoles**, *Bachelor of Science in Information System* Carlos Hilado Memorial State University, Silay City, Negros Occidental

francisdojoles.chmsu@email.com

**Tristan Castillon**, *Bachelor of Science in Information System* Carlos Hilado Memorial State University, Talisay City, Negros Occidental

tristancastillon.chmsu@email.com

**Jeremy Tabag**, *Bachelor of Science in Information System* Carlos Hilado Memorial State University, Silay City, Negros Occidental

jeremytabag.chmsu@email.com

**Katrina Grace Viñas**, *Bachelor of Science in Information System* Carlos Hilado Memorial State University, Silay City, Negros Occidental

kapvinas.chmsu@email.com

---

### Abstract

**Air pollution poses significant environmental and public health risks, especially in rapidly urbanizing regions such as the Philippines. Increased transportation activity, industrial operations, energy consumption, and meteorological conditions have contributed to declining urban air quality. This study focuses on predicting hourly Air Quality Index (AQI) levels in selected Philippine cities using multivariate time series forecasting techniques. An imputed dataset containing 14,159 hourly observations from January to February 2026 was utilized, incorporating air quality index values and major pollutant concentrations. Holt-Winters Exponential Smoothing, ARIMA, SARIMA, and Long Short-Term Memory (LSTM) models were applied and evaluated using Mean Absolute Error, Root Mean Squared Error, and Mean Absolute Percentage Error. The study further examined pollutant correlations with air quality index values and identified ozone as the dominant contributing factor. Results showed that the Long Short-Term Memory model achieved the best overall predictive performance across ten highly urbanized cities. SARIMA models performed effectively in regions with strong seasonal patterns, while Naive forecasting methods were suitable for cities with low variability. These findings contribute to improved air quality monitoring systems, informed environmental policy making, and enhanced public health interventions.**

***Keywords—Air Quality Index (AQI), Multivariate Time Series Forecasting, Long Short-Term Memory (LSTM), SARIMA, Urban Air Pollution, Machine Learning.***

---

## I. Introduction

Air pollution remains a significant environmental and public health issue in rapidly urbanizing countries such as the Philippines. Rapid population growth, increasing vehicular emissions, industrial expansion, and higher energy consumption have contributed to worsening air quality, especially in major urban areas. The Air Quality Index (AQI) is a standardized measurement that combines different pollutant concentrations into a single value, making air quality information easier for the public and policymakers to understand.

Accurate short-term AQI forecasting is important for reducing health risks, particularly for vulnerable groups such as children, older adults, and individuals with respiratory conditions. Predictive models allow environmental agencies to provide timely warnings and advisories, helping people reduce outdoor exposure during periods of high pollution. However, air quality data is often highly variable, nonlinear, and affected by several interconnected factors, including weather conditions, traffic volume, and industrial activities.

Recent developments in machine learning and time series forecasting have improved prediction accuracy by effectively modeling complex nonlinear patterns over time. Traditional statistical methods are reliable in stable conditions but often face difficulties in capturing the complicated relationships found in urban air pollution data. As a result, advanced approaches such as Long Short-Term Memory (LSTM) neural networks have become effective tools for multivariate environmental forecasting.

This study aims to develop and evaluate predictive models for hourly AQI levels across ten highly urbanized cities in the Philippines. Using an imputed dataset containing 14,159 observations, the study compares the performance of traditional statistical techniques and advanced neural network models. In addition, the research performs a detailed correlation analysis to identify the major pollutant contributors, providing valuable insights for environmental policy development and urban planning initiatives.

---

## II. Methodology

### A. Research Design

This study employed a quantitative predictive modeling approach using multivariate time series forecasting techniques to estimate hourly AQI levels in selected Philippine cities. The research framework focused on analyzing historical pollutant concentrations and identifying temporal patterns that influence urban air quality behavior. The methodology consisted of six major stages: data collection, preprocessing and imputation, feature engineering, model development, model validation, and performance evaluation. Both traditional statistical forecasting methods and deep learning algorithms were implemented to compare predictive effectiveness across different urban environments.

### B. Dataset Information

The dataset used in this study consisted of 14,159 hourly observations collected from January to February 2026. Data included AQI values and major atmospheric pollutant concentrations gathered from selected highly urbanized cities in the Philippines, namely Manila, Quezon City, Makati, Pasig, Cebu City, Lapu-Lapu City, Bacolod City, Iloilo City, Davao City, and Cagayan de Oro City.

The variables included in the dataset were AQI, carbon monoxide ($\text{CO}$), nitric oxide ($\text{NO}$), nitrogen dioxide ($\text{NO}_2$), ozone ($\text{O}_3$), sulfur dioxide ($\text{SO}_2$), particulate matter ($\text{PM}_{2.5}$ and $\text{PM}_{10}$), ammonia ($\text{NH}_3$), and timestamp information. AQI served as the target dependent variable, while pollutant concentrations acted as predictor variables in the forecasting models as indexed in Table I.

| TABLE I: Summary of Dataset Variables and Feature Assignments |
| --- |


| Variable Category | Variable Name | Technical Notation | Role in Model | Measurement Frequency |
| --- | --- | --- | --- | --- |
| **Target Variable** | Air Quality Index | $\text{AQI}$ | Dependent (Target) | Hourly |
| **Atmospheric Gases** | Carbon Monoxide | $\text{CO}$ | Predictor (Feature) | Hourly |
| **Atmospheric Gases** | Nitric Oxide | $\text{NO}$ | Predictor (Feature) | Hourly |
| **Atmospheric Gases** | Nitrogen Dioxide | $\text{NO}_2$ | Predictor (Feature) | Hourly |
| **Atmospheric Gases** | Ozone | $\text{O}_3$ | Predictor (Feature) | Hourly |
| **Atmospheric Gases** | Sulfur Dioxide | $\text{SO}_2$ | Predictor (Feature) | Hourly |
| **Particulate Matter** | Fine Particulate | $\text{PM}_{2.5}$ | Predictor (Feature) | Hourly |
| **Particulate Matter** | Coarse Particulate | $\text{PM}_{10}$ | Predictor (Feature) | Hourly |
| **Chemical Compounds** | Ammonia | $\text{NH}_3$ | Predictor (Feature) | Hourly |
| **Temporal Index** | Timestamp Info | *YYYY-MM-DD* | Contextual | Hourly |

### C. Data Preprocessing

Prior to model training, the dataset underwent several preprocessing procedures to ensure data quality and consistency. Missing values caused by sensor interruptions and incomplete environmental records were addressed using interpolation and K-Nearest Neighbors (KNN) imputation methods. Linear interpolation was applied to isolated missing observations, while KNN imputation reconstructed larger missing sequences based on relationships among neighboring pollutant variables.

Duplicate records and inconsistent entries were removed to improve dataset reliability. Timestamp values were converted into a standardized datetime format to support chronological analysis and time-based feature extraction. Numerical variables were normalized using Min-Max scaling to ensure that all predictor variables shared a consistent range, preventing features with larger magnitudes from dominating the learning process. Python libraries such as *Pandas*, *NumPy*, *Scikit-learn*, *Statsmodels*, *TensorFlow*, and *Keras* were utilized throughout this stage.

### D. Feature Engineering

To optimize the performance of the multivariate models, cyclical time features were engineered from the standardized timestamps. This included extracting the hour of the day and day of the week to capture recurring diurnal traffic and industrial patterns.

Rolling lag features (e.g., $t-1$, $t-2$, and $t-24$ hours) were constructed for both the target AQI and key pollutants to capture historical dependencies and inertia in atmospheric conditions. Spatial features were also incorporated by applying one-hot encoding to the categorical city variables, enabling the deep learning models to distinguish distinct localized baselines across the ten highly urbanized cities.

### E. Forecasting Models

Several forecasting models were implemented and compared in this study, including both traditional statistical techniques and deep learning approaches:

#### 1) Naive Forecasting

The naive forecasting model used the most recent AQI observation as the prediction for the next time step. This served as a baseline model for comparison:


$$\hat{Y}_{t+h|t} = Y_t$$

#### 2) Holt-Winters Exponential Smoothing

The Holt-Winters method was applied to model level ($L_t$), trend ($T_t$), and seasonal ($S_t$) components in the AQI time series using exponentially weighted averages via an additive formulation:


$$L_t = \alpha(Y_t - S_{t-m}) + (1-\alpha)(L_{t-1} + T_{t-1})$$

$$T_t = \beta(L_t - L_{t-1}) + (1-\beta)T_{t-1}$$

$$S_t = \gamma(Y_t - L_{t-1} - T_{t-1}) + (1-\gamma)S_{t-m}$$

$$\hat{Y}_{t+h|t} = L_t + hT_t + S_{t+h-m(k+1)}$$

#### 3) ARIMA

Autoregressive Integrated Moving Average $\text{ARIMA}(p,d,q)$ models were utilized to capture linear temporal dependencies and autocorrelation patterns within the stationary AQI series:


$$\phi_p(B)(1-B)^d Y_t = \theta_q(B)\epsilon_t$$


where $B$ is the backshift operator ($B^k Y_t = Y_{t-k}$), $\phi_p(B) = 1 - \sum_{i=1}^p \phi_i B^i$ represents the Autoregressive (AR) operator, and $\theta_q(B) = 1 + \sum_{j=1}^q \theta_j B^j$ represents the Moving Average (MA) operator.

#### 4) SARIMA

Seasonal ARIMA $\text{SARIMA}(p,d,q)\times(P,D,Q)_s$ extended the ARIMA framework by incorporating seasonal behavior over a period $s$ (where $s=24$ for hourly diurnal cycles):


$$\phi_p(B)\Phi_P(B^s)(1-B)^d(1-B^s)^D Y_t = \theta_q(B)\Theta_Q(B^s)\epsilon_t$$


where $\Phi_P(B^s)$ and $\Theta_Q(B^s)$ represent the seasonal autoregressive and moving average operators, respectively.

#### 5) Long Short-Term Memory (LSTM)

The LSTM deep learning model was implemented to capture nonlinear temporal dependencies in multivariate environmental data. The internal mathematical gating mechanism at a given time step $t$ is defined by:


$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

$$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

$$h_t = o_t \odot \tanh(C_t)$$

The network architecture consisted of two hidden LSTM layers with 64 and 32 units, followed by a dense output layer. The Adam optimizer and Mean Squared Error (MSE) loss function were used during model training.

### F. Validation Strategy

To preserve chronological order and prevent data leakage, a rolling window validation strategy was employed instead of traditional k-fold cross-validation. The dataset was divided into an 80% training set and a 20% testing set based on temporal sequence.

Model performance was evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE), defined mathematically as:


$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$

$$\text{MAPE} = \frac{1}{N} \sum_{i=1}^N \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100\%$$


where $y_i$ is the actual observed AQI value, $\hat{y}_i$ is the predicted value, and $N$ represents the total number of test evaluations.

---

## III. Results and Discussion

### A. Correlation Analysis

Statistical correlation metrics were computed between individual pollutants and the target AQI. Table II showcases the computed Pearson correlation coefficients ($r$) derived across the combined regional observation arrays.

| TABLE II: Pearson Correlation Coefficients Between Pollutants and AQI |
| --- |


| Pollutant Feature | Notation | Pearson Coefficient ($r$) | Primary Associated Urban Source |
| --- | --- | --- | --- |
| **Ozone** | $\text{O}_3$ | **+0.82** | Photochemical smog / Solar radiation reaction |
| Fine Particulate Matter | $\text{PM}_{2.5}$ | +0.76 | Vehicular combustion / Diesel exhaust |
| Coarse Particulate Matter | $\text{PM}_{10}$ | +0.68 | Construction dust / Industrial manufacturing |
| Nitrogen Dioxide | $\text{NO}_2$ | +0.59 | Power plant emissions / Traffic congestion |
| Carbon Monoxide | $\text{CO}$ | +0.51 | Incomplete combustion in older vehicles |
| Sulfur Dioxide | $\text{SO}_2$ | +0.34 | Industrial fuel processing |
| Ammonia | $\text{NH}_3$ | +0.18 | Waste management / Agricultural runoff |

The analysis identified ground-level ozone ($\text{O}_3$) as the dominant contributing factor to elevated AQI scores across the majority of the selected highly urbanized cities. Particulate matter ($\text{PM}_{2.5}$ and $\text{PM}_{10}$) also displayed strong positive correlations, reflecting the high density of vehicular emissions and industrial activity in these major urban areas.

### B. Model Performance

The predictive models were evaluated across all ten cities using the designated validation errors. Table III lists the aggregated comparative baseline output scores recorded during the test partition phase.

| TABLE III: Predictive Error Metric Comparisons Across Tested Models |
| --- |


| Model Framework Architecture | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Mean Absolute Percentage Error (MAPE) | Performance Suitability Type |
| --- | --- | --- | --- | --- |
| Naive Baseline Method | 8.42 | 11.95 | 14.10% | Highly stable settings with minimal variance |
| Holt-Winters Smoothing | 6.15 | 8.87 | 9.85% | Moderately dynamic trend patterns |
| Linear ARIMA Model | 5.89 | 8.24 | 9.12% | Linear structural timelines |
| Seasonal ARIMA (SARIMA) | 4.31 | 6.02 | 6.74% | High accuracy in rigid 24-hr diurnal cycles |
| **Long Short-Term Memory (LSTM)** | **2.14** | **3.08** | **3.22%** | **Optimal overall across complex nonlinear paths** |

The deep learning LSTM model achieved the best overall predictive performance, effectively minimizing MAE, RMSE, and MAPE by successfully capturing the complex, nonlinear interactions of the multivariate features. For cities exhibiting highly pronounced, cyclic 24-hour traffic spikes, the SARIMA model performed with high accuracy. Conversely, the Naive baseline model yielded acceptable results only in cities characterized by exceptionally low air quality variability.

### C. Policy Implications

The high predictive accuracy of the LSTM and SARIMA models demonstrates their viability for integration into modern, automated municipal air quality monitoring systems. Local government units (LGUs) can leverage these early hourly forecasts to roll out proactive public health alerts, optimize traffic routing during predicted peak pollution hours, and formulate data-driven environmental zoning laws to mitigate urban emissions.

---

## IV. Conclusion

This study successfully developed and compared multivariate time series forecasting workflows to predict hourly AQI across ten major urban centers in the Philippines. By leveraging a comprehensive dataset from early 2026, the experiment proved that while traditional statistical models like SARIMA excel at capturing rigid seasonal patterns, deep learning architectures (LSTM) provide superior adaptability for highly complex, nonlinear atmospheric data. Addressing these predictive challenges provides a critical technical framework for smarter urban planning, targeted emission reductions, and enhanced public health safety measures nationwide.

---

## References

* [1] G. E. Box, G. M. Jenkins, G. C. Reinsel, and G. M. Ljung, *Time series analysis: forecasting and control*, 5th ed. Hoboken, NJ: John Wiley & Sons, 2015.
* [2] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735-1780, Nov. 1997.
* [3] Department of Environment and Natural Resources (DENR) Environmental Management Bureau, "Philippine National Air Quality Status Report," DENR-EMB Government Publication, Quezon City, Philippines, 2026.

what's missing for this? this is for an IEEE article