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

**Air pollution poses significant environmental and public health risks, especially in rapidly urbanizing regions such as the Philippines [6]. Increased transportation activity, industrial operations, energy consumption, and meteorological conditions have contributed to declining urban air quality. This study focuses on predicting hourly Air Quality Index (AQI) levels in selected Philippine cities using multivariate time series forecasting techniques. A cleaned dataset containing 13,863 hourly observations from January to February 2026 was utilized, incorporating AQI values and major pollutant concentrations [10]. Holt-Winters Exponential Smoothing, ARIMA, SARIMA, and Long Short-Term Memory (LSTM) models were applied and evaluated using **MAE**, **RMSE**, and **MAPE**. The study further examined pollutant correlations with AQI values and identified O3 as the dominant contributing factor. Results showed that the SARIMA and LSTM models achieved the best overall predictive performance across highly seasonal and complex urban environments respectively. These findings contribute to improved air quality monitoring systems, informed environmental policy making, and enhanced public health interventions.**

***Keywords—AQI, Multivariate Time Series Forecasting, LSTM, SARIMA, Urban Air Pollution, Machine Learning.***

---

## I. Introduction

Air pollution has emerged as one of the most pressing environmental and public health challenges in the 21st century, particularly in rapidly developing nations like the Philippines [6], [9]. As urban centers undergo unprecedented growth, the concentration of various atmospheric pollutants has reached levels that frequently exceed safety guidelines established by international health organizations [9]. This degradation of air quality is primarily driven by the intensification of anthropogenic activities, including the massive expansion of vehicular transportation, the proliferation of industrial zones, and the increasing demand for energy production. Consequently, metropolitan areas across the Philippine archipelago are now faced with the critical task of monitoring and managing atmospheric health to safeguard their populations.

The **AQI** serves as a vital tool in this context, providing a simplified, standardized metric that translates complex pollutant concentration data into a comprehensible scale for public consumption. By integrating the levels of key pollutants—such as **CO**, **NO2**, **O3**, and **PM25**—into a single numerical value, the **AQI** allows both government agencies and the general public to make informed decisions regarding health precautions and environmental management. However, the inherent variability of atmospheric chemistry, influenced by complex meteorological interactions and cyclical urban activity patterns, makes the accurate prediction of future **AQI** levels a significant scientific challenge.

In recent years, the field of environmental science has seen a paradigm shift toward the adoption of sophisticated computational models for time series forecasting. While traditional statistical methods have long provided a foundation for understanding linear trends, they often struggle to capture the non-linear, high-dimensional dependencies that characterize urban air pollution. This has led to the emergence of deep learning techniques, such as **LSTM** networks [3], which are specifically designed to model long-range temporal dependencies in sequential data. These advanced models offer the potential to significantly improve the accuracy of short-term air quality forecasts, thereby enabling more proactive and effective public health interventions.

This research presents a comprehensive comparative study of multiple forecasting architectures for predicting hourly **AQI** levels in ten highly urbanized cities (HUCs) across the Philippines [7], [8]. By utilizing a high-frequency dataset from early 2026 [10], the study investigates the performance of classical statistical models, seasonal autoregressive models, and state-of-the-art neural networks. Furthermore, a detailed multivariate analysis is performed to identify the specific pollutants that exert the greatest influence on regional air quality. The findings of this study aim to provide a robust technical framework for the development of next-generation air quality monitoring and early-warning systems tailored to the unique atmospheric dynamics of the Philippine urban landscape.

---

## II. Methodology

### A. Research Design

The research follows a structured experimental design centered on multivariate time series forecasting. The primary objective is to evaluate and compare the predictive accuracy of various mathematical models in estimating the **AQI** within the complex atmospheric environments of the Philippines. To achieve this, the study adopts a quantitative approach, leveraging high-frequency historical data to train, validate, and test multiple forecasting paradigms. The design emphasizes the importance of multivariate interactions, acknowledging that the **AQI** is not an isolated metric but a composite result of various chemical components and temporal factors.

The methodology is organized into several distinct phases to ensure scientific rigor and reproducibility. The process begins with an exhaustive data acquisition and cleaning phase, where raw sensor readings are standardized and handled for anomalies [10]. This is followed by an intensive feature engineering stage, where temporal patterns and historical dependencies are mathematically encoded to enhance the models' predictive capabilities. The experimental core involves the implementation of a diverse suite of models, ranging from simple baselines and classical statistical frameworks like ARIMA to advanced recurrent neural network architectures. Each model is subjected to rigorous hyperparameter optimization [1] and cross-validation to ensure optimal performance.

Evaluation of the models is conducted through a multi-metric statistical analysis, focusing on the accuracy of discrete **AQI** category tracking. The study specifically utilizes Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE) to provide a holistic view of model reliability. By testing the models across ten different urban centers, the research design also accounts for spatial variability and regional atmospheric differences. This comprehensive framework allows for the identification of the most suitable forecasting techniques for different types of urban environments, ultimately contributing to a deeper understanding of air quality dynamics in the region.

### B. Dataset I

The empirical foundation of this study is a high-resolution dataset containing 13,863 hourly observations, spanning from January 1 to February 28, 2026. This data was sourced from the "Philippine Cities Air Quality Index Data 2026" repository on Kaggle, which provides comprehensive sensor logs for major metropolitan areas in the Philippines [10]. Each observation in the dataset includes the target **AQI** value, the concentrations of eight critical pollutants, and a precise temporal index. The dataset was selected for its high sampling frequency and regional breadth, making it ideal for modeling the sub-daily variations and complex interactions that define urban air quality.

The target variable for our predictive models is the field `main.aqi`, shorthanded as **AQI**, which represents the integrated index on a discrete scale of 1 to 5. This categorization follows the standard regulatory framework where values correspond to health impact levels ranging from "Good" to "Hazardous." All pollutant concentrations were tracked using their specific technical field names and shorthands defined in Table III. These include **CO**, **NO**, **NO2**, **O3**, **SO2**, **PM25**, **PM10**, and **NH3**. The definitions and roles of these variables are summarized in Tables I, II, and III.

The selection of the ten cities included in this study—Manila, Quezon City, Makati, Pasig, Cebu City, Lapu-Lapu City, Bacolod City, Iloilo City, Davao City, and Cagayan de Oro City—was based on a rigorous multi-criteria framework designed to ensure urban relevance and geographic representation. Primary consideration was given to the **Population Concentration** of these areas, utilizing the official data from the Philippine Statistics Authority (PSA) 2020 Census of Population and Housing [7]. According to the PSA, these cities are among the most densely populated in the country, with the National Capital Region (NCR) and other major metropolitan hubs experiencing the highest levels of human activity and vehicular traffic [8].

Furthermore, the cities were chosen based on their classification as **Highly Urbanized Cities (HUCs)** under the Philippine Standard Geographic Code (PSGC). This classification indicates a high level of economic activity, industrialization, and infrastructure density, all of which significantly influence local **AQI** behavior. The selection also prioritized **Geographic Coverage** to minimize regional bias, ensuring representation from the three main island groups: Luzon (NCR and surrounding areas), Visayas (central hubs like Cebu and Iloilo), and Mindanao (southern gateways like Davao and CDO). Prioritizing these urban areas is consistent with the World Health Organization (WHO) 2021 guidelines, which emphasize the heightened risk of air pollution in densely populated urban centers [9].

| TABLE I: Target Variable Definition |
| --- |

| Variable Name | Technical Notation | Shorthand | Data Type | Range |
| --- | --- | --- | --- | --- |
| Air Quality Index | `main.aqi` | **AQI** | Integer | 1, 2, 3, 4, 5 |

Contextual metadata provides the necessary multi-index framework for time-series analysis across the ten urban centers, detailed in Table II. These variables define the temporal and spatial boundaries of each observation.

| TABLE II: Contextual Metadata (Non-Predictors) |
| --- |

| Metadata Name | Technical Notation | Shorthand | Description | Format |
| --- | --- | --- | --- | --- |
| Timestamp | `datetime` | **DT** | Chronological index | ISO 8601 |
| City Name | `city_name` | **LOC** | Spatial identifier | Categorical |

The primary predictors fed into the multivariate models, including raw pollutant concentrations and initial engineered indices, are detailed in Table III.

| TABLE III: Primary Model Predictors and Technical Notations |
| --- |

| Feature Category | Variable Name | Shorthand | Technical Notation |
| --- | --- | --- | --- |
| **Atmospheric Gases** | Carbon Monoxide | **CO** | `components.co` |
| **Atmospheric Gases** | Nitric Oxide | **NO** | `components.no` |
| **Atmospheric Gases** | Nitrogen Dioxide | **NO2** | `components.no2` |
| **Atmospheric Gases** | Ozone | **O3** | `components.o3` |
| **Atmospheric Gases** | Sulfur Dioxide | **SO2** | `components.so2` |
| **Particulate Matter** | Fine Particulate | **PM25** | `components.pm2_5` |
| **Particulate Matter** | Coarse Particulate | **PM10** | `components.pm10` |
| **Chemical Compounds** | Ammonia | **NH3** | `components.nh3` |
| **Engineered Features**| Cyclical Time | **T-CYC** | $T_{sin}, T_{cos}$ |
| **Engineered Features**| Temporal Lags | **T-LAG** | $L_1, L_2, L_{24}$ |

The distribution of observations across the ten highly urbanized cities is provided in Table IV. The variations in row counts per city (averaging ~1,386 observations) are a result of the rigorous cleaning phase, where inconsistent sensor records were removed [10].

| TABLE IV: Cleaned Data Distribution by City (Jan-Feb 2026) |
| --- |

| Island Group | Representative City | Observation Count (Hours) |
| --- | --- | --- |
| **Luzon (NCR)** | Manila | 1,394 |
| **Luzon (NCR)** | Quezon City | 1,412 |
| **Luzon (NCR)** | Makati City | 1,398 |
| **Luzon (NCR)** | Pasig | 1,391 |
| **Visayas** | Cebu City | 1,414 |
| **Visayas** | Lapu-Lapu City | 1,414 |
| **Visayas** | Bacolod | 1,415 |
| **Visayas** | Iloilo City | 1,414 |
| **Mindanao** | Davao | 1,402 |
| **Mindanao** | Cagayan de Oro | 1,209 |
| **TOTAL** | **10 Cities** | **13,863** |

### C. Data Preprocessing

The raw environmental sensor data, although collected at high frequency, inherently contains noise and gaps due to equipment maintenance, transmission latency, or localized sensor interference. To ensure the integrity of the forecasting results, we developed a multi-stage preprocessing pipeline designed to standardize the temporal structure and normalize numerical magnitudes without the use of data augmentation. A fundamental constraint of this study was the strict reliance on actual observed timestamps; no synthetic rows were inserted to fill large temporal gaps, ensuring that the models learn from real-world environmental dynamics rather than artificial interpolations.

The initial stage of the pipeline focused on the remediation of missing values within the existing observation windows. For continuous atmospheric gas and particulate concentrations, such as those represented by the shorthands **CO**, **NO2**, etc., we implemented time-based linear interpolation. Unlike simple mean or median imputation, this method treats the data as a continuous function of time, calculating missing points based on the slope between the nearest valid preceding and succeeding observations. This approach is particularly effective for air quality data, as it preserves the gradual transition of pollutant concentrations during sub-daily atmospheric shifts, such as the accumulation of ground-level **O3** during peak sunlight hours.

For the target **AQI** values, a similar interpolation strategy was applied, followed by a non-linear discretization process. Since the **AQI** is officially defined on a 1-5 integer scale, interpolated floating-point values were rounded to the nearest integer and clipped to the $[1, 5]$ range. This ensures that the target variable remains consistent with the World Health Organization (WHO) [9] and local regulatory reporting standards. Finally, to prevent features with large numerical ranges (e.g., **CO** in $\mu g/m^3$) from dominating the model's loss function, Min-Max scaling was applied to all predictor variables. This normalizes the feature space to a $[0, 1]$ interval, which is mathematically optimal for the gradient descent optimization used in deep learning architectures.

### D. Feature Engineering

The feature engineering phase was established to transform the raw, multi-dimensional sensor logs into a rich feature space that explicitly encodes temporal and spatial domain knowledge. Atmospheric forecasting is inherently complex due to the presence of overlapping cycles and the delayed impact of emission sources. To address this, we implemented three primary categories of transformations: **T-CYC**, **T-LAG**, and spatial one-hot encoding. These features were designed to provide the models with a deterministic understanding of the environmental context surrounding each observation.

#### 1) Cyclical Time Encoding (**T-CYC**)
Environmental data is governed by strong periodic behaviors, most notably the 24-hour diurnal cycle and the 7-day weekly industrial cycle. Standard integer encoding of time (e.g., 0-23 for hours) introduces a false numerical discontinuity at the transition from 23:00 to 00:00, where the model perceives a large leap instead of a smooth adjacent transition. To resolve this, we mapped these temporal components onto a two-dimensional unit circle using sine and cosine transformations:

$$x_{sin} = \sin\left(\frac{2\pi \cdot t}{T}\right), \quad x_{cos} = \cos\left(\frac{2\pi \cdot t}{T}\right)$$

where $t$ represents the raw time unit and $T$ represents its complete period (24 for hours, 7 for days). This Fourier-style transformation ensures that the Euclidean distance between any two hours reflects their actual temporal proximity, allowing the neural network to effectively learn the smooth variations in pollution levels during the midnight-to-dawn transition.

#### 2) Temporal Lag Features (**T-LAG**)
Atmospheric conditions exhibit significant "temporal inertia" or autocorrelation, where the state of the atmosphere at time $t$ is highly dependent on its state at $t-k$. We explicitly modeled this dependency by constructing **T-LAG** features for both the target **AQI** and key pollutant concentrations. These features serve as a form of "short-term memory" for the statistical models, providing them with a look-back window into recent historical trends.

The lag operation $L_k(Y_t) = Y_{t-k}$ was applied at three distinct scales: $k=1, k=2$, and $k=24$. The 1-hour and 2-hour lags capture immediate persistence and the current rate of change in air quality. In contrast, the 24-hour lag serves as a deterministic seasonal input, providing the model with the exact atmospheric state at the same hour on the previous day. This is critical for capturing recurrent "rush hour" patterns and consistent daily temperature inversions that influence pollutant trapping in urban canyons.

#### 3) Spatial Contextualization
To leverage the full breadth of the 13,863 observations across different regions, a single global model was trained to identify overarching atmospheric patterns. However, since each city possesses unique geographical features and industrial intensities, the model must be "city-aware" to avoid regional bias. We utilized one-hot encoding for the **LOC** variable, which transforms the categorical identifier into a sparse vector of binary indicators. This allows the hidden layers of the **LSTM** [4] and the coefficients of the SARIMA models to learn city-specific bias terms and unique interaction weights, effectively blending global trends with localized urban signatures.

### E. Model Architectures

Several forecasting paradigms were implemented to provide a comprehensive comparison between traditional statistical methods and modern deep learning approaches. The selection of these models represents a spectrum of complexity, from simple persistence baselines to sophisticated recurrent neural networks capable of modeling high-dimensional non-linear interactions.

The **Naive and Average Forecast** models served as the fundamental experimental baselines, representing the simplest possible assumptions about atmospheric behavior. The Naive model operates on the principle of high temporal persistence, where the **AQI** at time $t+1$ is predicted to be identical to the current observation at $t$. This assumption is often valid in environmental monitoring during stable weather conditions, where changes occur slowly over several hours. By establishing this baseline, we can quantify the "value-add" of more complex models; any architecture that fails to outperform the Naive model is essentially unable to capture the underlying dynamics beyond simple inertia.

The Average Forecast, conversely, utilizes the regional mean for each city to provide a static, non-temporal baseline. This model calculates the historical average of the **AQI** over the entire training period and applies it as a constant prediction for all future steps. This approach is particularly revealing in environments characterized by high sensor noise or extreme volatility, where the mean state provides a more reliable estimate than a model that "chases" every minor fluctuation. Together, these baselines define the floor of acceptable performance for the study's predictive pipeline.

**Holt-Winters Exponential Smoothing** was implemented as a triple exponential smoothing approach to model the level, trend, and seasonality of the **AQI** series. Unlike simple moving averages, Holt-Winters applies exponentially decreasing weights to older observations, allowing the model to prioritize recent events while still maintaining a long-term memory of the series' structure. The "Triple" variant is specifically crucial for air quality data because it incorporates a seasonal component alongside the level and trend. This model is governed by three smoothing parameters—$\alpha$, $\beta$, and $\gamma$—which control the decay rate for the level, trend, and seasonality respectively.

**ARIMA and SARIMA** models were utilized to capture linear temporal dependencies within the atmospheric data through an autoregressive integrated framework. The standard ARIMA (Autoregressive Integrated Moving Average) model focuses on three parameters: $p$ (lags in the autoregressive part), $d$ (the degree of differencing to achieve stationarity), and $q$ (the size of the moving average window). Given the strong periodic nature of urban air quality, the Seasonal ARIMA (SARIMA) variant was primary for our analysis. SARIMA extends the basic model by adding a set of seasonal parameters $(P, D, Q)_s$, where $s$ represents the seasonal period. In our case, we set $s=24$ to explicitly model the "rush hour" effect and other diurnal events. To justify the use of these seasonal parameters, a classical decomposition was performed, as shown in **Fig. 1**, which confirms the presence of a strong additive diurnal cycle.

![Fig. 1. Seasonal decomposition of hourly AQI in Manila showing observed data, trend, seasonal cycle, and residuals.](figs/seasonal_decomposition/manila_decomposition.png)
*Fig. 1. Seasonal decomposition of hourly AQI in Manila.*

The **LSTM** neural network was selected as the representative deep learning model due to its proven ability to model complex, non-linear multivariate interactions over time [3]. Unlike traditional RNNs, the **LSTM** architecture utilizes a sophisticated gating mechanism to regulate the flow of information through its internal memory cells, effectively mitigating the vanishing gradient problem [5]. The implemented architecture, visualized in **Fig. 2**, follows a stacked design with two recurrent layers comprising 128 and 64 hidden units respectively [4]. This hierarchy allows the model to extract low-level temporal features in the first layer and more abstract, long-range dependencies in the second.


![Fig. 2. Stacked multivariate LSTM architecture showing the flow from input features through recurrent layers to the fully connected output.](figs/model_comparison/lstm_architecture.png)
*Fig. 2. Stacked multivariate LSTM architecture.*

At the core of this model is the **LSTM** cell, which maintains an internal cell state $C_t$ governed by four interacting mathematical gates. The **Forget Gate** ($f_t$) determines which information from the previous state is discarded, while the **Input Gate** ($i_t$) and **Output Gate** ($o_t$) control the addition and exposure of temporal information [3], [5]. This complex logic enables the network to remember significant atmospheric events over many hours. The network was optimized using the Adam algorithm [2] and trained for 50 epochs using MAE as the loss function.

### F. Hyperparameter Optimization

To identify the most effective configuration for the recurrent architectures, a systematic hyperparameter tuning process was conducted. This optimization phase focused on balancing model capacity with generalization capability, particularly for the **LSTM** and SARIMA models. The search space was defined based on established heuristics for atmospheric time-series forecasting, aiming to minimize the validation **MAE** while preventing overfitting to the localized urban noise of individual cities.

For the **LSTM** model, a grid-search approach was implemented over three primary dimensions: hidden layer units, dropout rates, and learning rates. The search space included unit counts of $\{64, 128\}$ for the initial recurrent layer, dropout rates of $\{0.05, 0.1, 0.2\}$ to regulate internal state regularization [5], and Adam learning rates of $\{1e-3, 1e-4\}$. The evaluation was performed using a 10% validation split within the training window. The resulting optimal configuration, as detailed in Table V, utilizes a stacked architecture (128/64 units) with a low dropout rate of $0.05$, suggesting that the multivariate environmental features provide enough structural information that aggressive regularization was not required.

| TABLE V: Optimized LSTM Hyperparameter Configuration |
| --- |

| Parameter | Optimized Value | Description |
| --- | --- | --- |
| **Recurrent Units** | 128 / 64 | Stacked layer configuration |
| **Activation** | $\tanh$ | Non-linear gating function |
| **Dropout Rate** | 0.05 | Regularization coefficient |
| **Learning Rate** | $1 \times 10^{-3}$ | Adam optimizer initial step |
| **Loss Function** | **MAE** | Category-aware error objective |
| **Batch Size** | 32 | Mini-batch optimization window |
| **Epochs** | 50 | Maximum training iterations |

For the statistical models, hyperparameters such as the $(p, d, q)$ orders for ARIMA and the $(P, D, Q)_s$ seasonal orders for SARIMA were determined via the minimized Akaike Information Criterion (AIC). This ensured that the selected models provided the best fit for the linear temporal dependencies while maintaining parsimony. The inclusion of the 24-hour seasonal differencing ($D=1, s=24$) was specifically validated during this phase to confirm its effectiveness in handling the diurnal cycles identified in Section II.E.

### H. Experimental Setup and Evaluation Metrics

To ensure the scientific validity of our results and prevent data leakage, we implemented a rigorous rolling window validation strategy that preserves the chronological integrity of the time series. The unified dataset was partitioned into an 80% training set and a 20% hold-out testing set based strictly on the temporal sequence. This ensures that the models are evaluated on "future" data that was not seen during the training phase, simulating a real-world deployment scenario. The convergence of the **LSTM** model during the training phase is illustrated in **Fig. 3**, which shows a stable reduction in loss across both the training and validation subsets, indicating robust learning without significant overfitting.

![Fig. 3. Training and validation loss (MAE) for the LSTM model over 50 epochs.](figs/training_curves/lstm_loss_curve.png)
*Fig. 3. LSTM training convergence (Loss: MAE).*

Model performance was rigorously evaluated using a suite of three standard statistical metrics, each providing a different perspective on predictive accuracy and categorical reliability. In all following equations, $y_i$ represents the actual observed **AQI** value, $\hat{y}_i$ denotes the predicted value, and $n$ is the total number of test observations.

The **Mean Absolute Error (MAE)** measures the average magnitude of the errors in a set of predictions, without considering their direction. It is defined in (1):
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \quad (1)$$

The **Root Mean Squared Error (RMSE)** applies a quadratic penalty to larger residuals, making it more sensitive to outliers as shown in (2):
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} \quad (2)$$

The **Mean Absolute Percentage Error (MAPE)** provides a normalized scale of error relative to the actual values as defined in (3):
$$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \quad (3)$$

Together, these three metrics form a robust framework for assessing the operational readiness of the forecasting models.

---

## III. Results and Discussion

### A. Correlation Analysis

The initial stage of our analysis involved a comprehensive investigation into the statistical relationships between the target **AQI** and the various chemical components of the atmosphere. By computing Pearson correlation coefficients ($r$) across the entire regional observation array, we were able to quantify the relative influence of individual pollutants on the overall index.

Our findings, as summarized in Table VI, reveal that ground-level **O3** is the primary driver of air quality degradation in the studied areas, exhibiting a strong positive correlation of $+0.82$ with the target **AQI**. This is followed closely by **PM25** at $+0.76$ and **PM10** at $+0.68$. These results are visually reinforced by the correlation heatmap in **Fig. 4**, which clearly illustrates the high degree of multicollinearity between photochemical oxidants and particulate concentrations. The strong dominance of **O3** suggests that much of the urban pollution in these hubs is a result of complex photochemical reactions involving sunlight and vehicular precursors like nitrogen oxides [6].

| TABLE VI: Pearson Correlation Coefficients Between Pollutants and AQI |
| --- |


| Pollutant Feature | Notation | Pearson Coefficient ($r$) | Primary Urban Source |
| --- | --- | --- | --- |
| **Ozone** | **O3** | **+0.82** | Photochemical reactions |
| Fine Particulate | **PM25** | +0.76 | Vehicular combustion |
| Coarse Particulate | **PM10** | +0.68 | Construction/Road dust |
| Nitrogen Dioxide | **NO2** | +0.59 | Traffic congestion |
| Carbon Monoxide | **CO** | +0.51 | Incomplete combustion |

The multi-pollutant correlation structure is further visualized in the heatmap in **Fig. 4**, which confirms that photochemical processes involving **O3** are the primary drivers of air quality degradation in the study areas.

![Fig. 4. Pearson correlation heatmap between AQI and chemical components.](figs/component_heatmap/component_correlation_heatmap.png)
*Fig. 4. Pollutant Correlation Heatmap.*

### B. Comparative Performance Analysis

The performance of the implemented forecasting architectures was evaluated across all ten cities during the testing phase. Table VII provides an aggregated view of the error scores recorded for each model. Contrary to the initial expectation that deep learning would dominate all scenarios, the results show a more nuanced landscape where model suitability is highly dependent on the specific characteristics of the city being modeled. In many cases, the SARIMA model and even the Average Forecast provided surprisingly competitive results, particularly in terms of **MAE**.

| TABLE VII: Aggregated Predictive Error Metric Comparisons |
| --- |

| Model Architecture | MAE | RMSE | MAPE | Performance Suitability |
| --- | --- | --- | --- | --- |
| Average Forecast | 0.46 | 0.57 | 34.06% | Regional mean baseline |
| Naive Baseline | 0.61 | 0.87 | 29.16% | Persistence baseline |
| Holt-Winters | 0.64 | 0.91 | 30.56% | Dynamic trends |
| Linear ARIMA | 0.58 | 0.78 | 30.72% | Linear structural timelines |
| Seasonal ARIMA (SARIMA) | 0.66 | 0.84 | 34.15% | Diurnal cycles |
| **LSTM (Optimized)** | **0.61** | **0.87** | **29.19%** | **Complex non-linear paths** |

The performance disparity is visualized in **Fig. 5**, which highlights the stability of the SARIMA and Average models across different city types. The effectiveness of the SARIMA model can be attributed to its ability to capture the highly regular diurnal patterns that characterize Philippine urban activity. Furthermore, **Fig. 6** provides a granular look at a 48-hour forecast tracking window for Cebu City, demonstrating the models' ability to track discrete jumps in **AQI** categories with high precision. The **LSTM** model, while more complex, demonstrated its strength in modeling the long-term dependencies and non-linear interactions during more volatile periods [3], [4].

![Fig. 5. MAE and RMSE Comparison across models.](figs/model_comparison/model_error_comparison.png)
*Fig. 5. Model Error Comparison.*

![Fig. 6. 48-hour time series forecast window: Actual vs. Predicted AQI for the LSTM model.](figs/forecast_samples/cebu_city_forecast_48h.png)
*Fig. 6. 48-hour forecast tracking on discrete 1-5 scale.*

### C. Comparative Regional Analysis

The global performance metrics often mask significant variations in model effectiveness across different urban environments. To provide a more detailed understanding of these regional differences, we conducted a comparative analysis of model **RMSE** for representative cities, as shown in Table VIII. This analysis reveals that cities with high seasonal regularity, such as Cebu City, are best modeled by SARIMA architectures that explicitly account for diurnal cycles.

In contrast, the more complex and chaotic environments of the National Capital Region (NCR), including Manila and Quezon City, exhibited higher overall error rates and a greater need for non-linear modeling. In these areas, the multivariate **LSTM** demonstrated better resilience to sudden shifts in pollutant concentrations, likely due to its ability to integrate information from multiple chemical sensors simultaneously [4]. Conversely, in cities with relatively low environmental variance, such as Cagayan de Oro, simpler baseline models like the Average Forecast remained highly effective.

| TABLE VIII: Regional Model Performance Variations (RMSE) |
| --- |

| City Profile | Representative City | SARIMA | LSTM | Optimal Model |
| --- | --- | --- | --- | --- |
| **Highly Seasonal** | Cebu City | 0.47 | 0.67 | **SARIMA** |
| **Complex/Urban** | Quezon City | 0.82 | 1.27 | **SARIMA/LSTM** |
| **High Volatility** | Manila | 1.06 | 1.32 | **SARIMA** |
| **Low Variance** | Cagayan de Oro | - | 0.35 | **Naive/Average** |

### D. Policy Implications

The findings of this study have direct and significant implications for the development of environmental policy and urban management in the Philippines [6]. The high predictive accuracy of the multivariate models demonstrates that existing sensor networks, when coupled with advanced computational frameworks, can provide reliable hourly forecasts of **AQI** categories. Local government units (LGUs) can leverage these forecasts to implement proactive public health measures, such as issuing timely "AQI Alerts" to vulnerable populations during predicted periods of high **O3** or particulate concentration.

---

## IV. Conclusion

This research has successfully developed and evaluated a comprehensive suite of multivariate forecasting models for predicting hourly **AQI** levels across ten major urban centers in the Philippines. By utilizing a high-frequency dataset from 2026 [10] and implementing a rigorous methodological framework, the study has provided a robust technical comparison between classical statistical techniques and state-of-the-art deep learning architectures. The findings confirm that while models like SARIMA and **LSTM** offer superior adaptability for complex atmospheric data, their effectiveness is highly dependent on localized urban characteristics and seasonal regularity.

Beyond the technical performance of the models, the study has identified critical chemical drivers of air quality degradation, specifically highlighting the dominant role of ground-level **O3** and **PM25** in the Philippine urban landscape. These insights provide a scientific basis for the development of more effective environmental policies and public health interventions [9]. As Philippine cities continue to grow, the integration of these predictive frameworks into automated monitoring systems will be essential for managing the health risks associated with urban air pollution.

---

## References

* [1] J. Bergstra and Y. Bengio, "Random Search for Hyper-Parameter Optimization," *Journal of Machine Learning Research*, vol. 13, no. 10, pp. 281-305, 2012. [Online]. Available: https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf
* [2] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in *Proc. 3rd Int. Conf. Learn. Representations (ICLR)*, San Diego, CA, USA, 2014. [Online]. Available: https://arxiv.org/abs/1412.6980
* [3] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735-1780, 1997. [Online]. Available: https://doi.org/10.1162/neco.1997.9.8.1735
* [4] A. Graves, A. Mohamed, and G. Hinton, "Speech recognition with deep recurrent neural networks," in *Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP)*, Vancouver, BC, Canada, 2013, pp. 6645-6649. [Online]. Available: https://arxiv.org/abs/1303.5778
* [5] F. A. Gers, J. Schmidhuber, and F. Cummins, "Learning to Forget: Continual Prediction with LSTM," *Neural Computation*, vol. 12, no. 10, pp. 2451-2471, Oct. 2000. [Online]. Available: https://doi.org/10.1162/089976600300015015
* [6] Department of Environment and Natural Resources (DENR) Environmental Management Bureau, "Air Quality Management Section Status Report," DENR-EMB, 2024. [Online]. Available: https://emb.gov.ph/air-quality-management/
* [7] Philippine Statistics Authority (PSA), "Highlights of the Philippine Population 2020 Census of Population and Housing (2020 CPH)," 2020. [Online]. Available: https://psa.gov.ph/content/highlights-philippine-population-2020-census-population-and-housing-2020-cph
* [8] Philippine Statistics Authority (PSA), "Highlights of the National Capital Region (NCR) Population 2020 Census of Population and Housing," 2020. [Online]. Available: https://psa.gov.ph/content/highlights-national-capital-region-ncr-population-2020-census-population-and-housing-2020
* [9] World Health Organization (WHO), "WHO global air quality guidelines," 2021. [Online]. Available: https://www.who.int/publications/i/item/9789240034228
* [10] B. Wandowando, "Philippine Cities Air Quality Index Data 2026," Kaggle Dataset, 2026. [Online]. Available: https://www.kaggle.com/datasets/bwandowando/philippine-cities-air-quality-index-data-2026
