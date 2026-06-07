Predictive Modeling of Hourly Air Quality Index in Selected Philippine Cities Using Multivaraite Time Series Forecasting

Rainer Astodillo  
_College of Computer Studies_  
_Carlos Hilado Memorial State University  
_Talisay City, Negros Occidental, Philippines  
rainerastodillo.chmsu@gmail.com

Jeremy Tabag  
_College of Computer Studies_  
_Carlos Hilado Memorial State University  
_Talisay City, Negros Occidental, Philippines  
jtabag.chmsu@gmail.comFrancis Jude Dojoles  
_College of Computer Studies_  
_Carlos Hilado Memorial State University  
_Talisay City, Negros Occidental, Philippines  
francisdojoles.chmsu@gmail.com

Katrina Grace Viñas  
_College of Computer Studies_  
_Carlos Hilado Memorial State University  
_Talisay City, Negros Occidental, Philippines  
kgpvinas.chmsu@gmail.com Tristan Castillon  
_College of Computer Studies_  
_Carlos Hilado Memorial State University  
_Talisay City, Negros Occidental, Philippines  
tgcastillon.chmsu@gmail.com

El Jireh Bibanco  
_College of Computer Studies_  
_Carlos Hilado Memorial State University  
_Talisay City, Negros Occidental, Philippines  
ej.bibanco@chmsu.edu.ph

_Abstract_—Air pollution poses significant environmental and public health risks, especially in rapidly urbanizing regions such as the Philippines \[6\]. Increased transportation activity, industrial operations, energy consumption, and meteorological conditions have contributed to declining urban air quality. This study focuses on predicting hourly Air Quality Index (AQI) levels in selected Philippine cities using multivariate time series forecasting techniques. A cleaned dataset containing 13,863 hourly observations from January to February 2026 was utilized, incorporating AQI values and major pollutant concentrations \[10\]. Holt-Winters Exponential Smoothing, ARIMA, SARIMA, and Long Short-Term Memory (LSTM) models were applied and evaluated using MAE, RMSE, and MAPE. The study further examined pollutant correlations with AQI values and identified O3 as the dominant contributing factor. Results showed that the SARIMA and LSTM models achieved the best overall predictive performance across highly seasonal and complex urban environments respectively. These findings contribute to improved air quality monitoring systems, informed environmental policy making, and enhanced public health interventions.

Keywords—AQI, Multivariate Time Series Forecasting, LSTM, SARIMA, Urban Air Pollution, Machine Learning

# Introduction

Air pollution is a growing environmental and public health concern in the Philippines, particularly in rapidly urbanizing cities where increasing transportation, industrial activities, and energy consumption contribute to deteriorating air quality \[6\], \[9\]. As pollutant concentrations frequently approach or exceed recommended safety thresholds, effective monitoring and management strategies have become essential for protecting public health. In this context, the Air Quality Index (AQI) serves as an important tool for translating complex atmospheric data into a standardized and easily understandable measure of air quality. By integrating the concentrations of major pollutants, including CO, NO₂, O₃, and PM₂.₅, the AQI enables government agencies and the public to assess pollution levels and make informed decisions regarding health and environmental protection.

Accurate AQI forecasting remains a significant challenge due to the complex interactions among atmospheric pollutants, meteorological conditions, and human activities. Traditional statistical forecasting approaches have been widely used to analyze air quality trends; however, they often struggle to capture the non-linear and high-dimensional relationships present in urban pollution data. Recent advances in artificial intelligence and deep learning have introduced more sophisticated forecasting techniques, particularly Long Short-Term Memory (LSTM) networks, which are capable of modeling long-range temporal dependencies and complex sequential patterns \[3\].

This study presents a comparative evaluation of multiple forecasting approaches for predicting hourly AQI levels across ten highly urbanized cities (HUCs) in the Philippines \[7\], \[8\]. Using a high-frequency air quality dataset collected during early 2026 \[10\], the research compares the performance of baseline forecasting methods, classical statistical models, seasonal autoregressive techniques, and deep learning architectures. In addition, a multivariate analysis is conducted to identify the pollutants that most strongly influence AQI variations. The findings provide insights into the effectiveness of different forecasting models and contribute to the development of data-driven air quality monitoring and early-warning systems tailored to the Philippine urban environment \[6\], \[9\].

# Methodology

## Research Design

This study employed a quantitative experimental research design focused on multivariate time series forecasting to evaluate and compare the accuracy of different models in predicting AQI across selected Philippine cities. The approach recognizes that AQI is influenced by multiple pollutants and temporal factors, requiring the analysis of their combined effects.

The methodology consisted of several stages, including data acquisition, preprocessing, feature engineering, model development, and evaluation. Raw environmental data were cleaned and standardized before being used to train forecasting models ranging from simple baseline methods and statistical approaches, such as ARIMA, to advanced deep learning architectures. Hyperparameter optimization was also performed to improve model performance \[1\], \[10\].

Model performance was assessed using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE). By evaluating the models across ten urban centers, the study accounted for regional differences in air quality patterns and identified the most suitable forecasting approaches for various urban environments.

## Dataset Information and City Selection Rationale

This study utilized a dataset of 13,863 hourly AQI observations collected from January to February 2026, obtained from the Philippine Cities Air Quality Index Data 2026 repository on Kaggle \[10\]. The dataset includes AQI values, concentrations of eight major pollutants (CO, NO, NO₂, O₃, SO₂, PM₂.₅, PM₁₀, and NH₃), and corresponding timestamps.

The analysis focused on ten highly urbanized Philippine cities representing Luzon, Visayas, and Mindanao: Manila, Quezon City, Makati, Pasig, Cebu City, Lapu-Lapu City, Bacolod City, Iloilo City, Davao City, and Cagayan de Oro City. These locations were selected based on population density, urbanization level, and geographic representation, using data from the PSA 2020 Census and the PSGC classification of Highly Urbanized Cities (HUCs) \[7\], \[8\]. The selection is consistent with WHO air quality guidelines, which identify densely populated urban areas as being at greater risk of air pollution exposure \[9\].

1.  Target Variable Definition

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
| Variables Name | Technical Notation | Shorthand | Data Type | Range |
| Air Quality Index | main.aqi | AQI | Integer | 1, 2, 3, 4, 5 |

Contextual metadata provides the necessary multi-index framework for time-series analysis across the ten urban centers, detailed in Table II. These variables define the temporal and spatial boundaries of each observation.

1.  Contextual Metadata (Non-Predictors)

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
| Variables Name | Technical Notation | Shorthand | Description | Format |
| Timestamp | datetime | DT  | Chronological index | ISO 8601 |
| City Name | city_name | LOC | Spatial identifier | Categorical |

The primary predictors fed into the multivariate models, including raw pollutant concentrations and initial engineered indices, are detailed in Table III.

1.  Primary Model Predictors and Technical Notations

|     |     |     |     |
| --- | --- | --- | --- |
| Feature Category | Variable Name | Shorthand | Technical Notation |
| Atmospheric Gases | Carbon Monoxide | CO  | components.co |
| Atmospheric Gases | Nitric Oxide | NO  | components.no |
| Atmospheric Gases | Nitrogen Dioxide | NO2 | components.no2 |
| Atmospheric Gases | Ozone | O3  | components.o3 |
| Atmospheric Gases | Sulfur Dioxide | SO2 | components.so2 |
| Particulate Matter | Fine Particulate | PM25 | components.pm2_5 |
| Particulate Matter | Coarse Particulate | PM10 | components.pm10 |
| Chemical Compounds | Ammonia | NH3 | components.nh3 |

The distribution of observations across the ten highly urbanized cities is provided in Table V. The variations in row counts per city (averaging ~1,386 observations) are a result of the rigorous cleaning phase, where inconsistent sensor records were removed \[10\].

1.  Cleaned Data Distribution by City (Jan-Feb 2026)  
    

|     |     |     |
| --- | --- | --- |
| Island Group | Representative City | Observation Count (Hours) |
| Luzon (NCR) | Manila | 1,394 |
| Luzon (NCR) | Quezon City | 1,412 |
| Luzon (NCR) | Makati City | 1,398 |
| Luzon (NCR) | Pasig | 1,391 |
| Visayas | Cebu City | 1,414 |
| Visayas | Lapu-Lapu City | 1,414 |
| Visayas | Bacolod | 1,415 |
| Visayas | Iloilo City | 1,414 |
| Mindanao | Davao | 1,402 |
| Mindanao | Cagayan de Oro | 1,209 |
| TOTAL | 10 Cities | 13,863 |

## Data Preprocessing

The A multi-stage preprocessing pipeline was implemented to improve data quality while preserving the original temporal structure of the dataset. To maintain data authenticity, no synthetic observations were generated; only actual recorded timestamps were used.

Missing values in atmospheric pollutant measurements (e.g., CO, NO₂, and other pollutants) were addressed using time-based linear interpolation, which estimates missing values from neighboring observations while preserving temporal trends. The same method was applied to missing AQI values. Since AQI is reported on a 1–5 discrete scale, interpolated values were rounded to the nearest integer and clipped within the valid range to ensure consistency with WHO and local air quality standards \[9\].

To improve model performance and prevent variables with larger numerical ranges from dominating the learning process, Min-Max scaling was applied to all predictor variables, normalizing them to a \[0,1\] range. This preprocessing step supports efficient optimization and enhances the effectiveness of machine learning and deep learning models..

## Feature Engineering

The feature engineering phase was established to transform the raw, multi-dimensional sensor logs into a rich feature space that explicitly encodes temporal and spatial domain knowledge. Atmospheric forecasting is inherently complex due to the presence of overlapping cycles and the delayed impact of emission sources. To address this, we implemented three primary categories of transformations: T-CYC, T-LAG, and spatial one-hot encoding. These features were designed to provide the models with a deterministic understanding of the environmental context surrounding each observation.

1.  Engineered Feature Summary

|     |     |     |     |
| --- | --- | --- | --- |
| Feature Category | Shorthand | Formula | Purpose |
| Cyclical Hour Encoding | T-CYC | ,   | Eliminates temporal discontinuity at period boundaries |
| 1-Hour Lag | T-LAG |     | Captures immediate temporal persistence |
| 2-Hour Lag | T-LAG |     | Captures short-term rate of change |
| 24-Hour Lag | T-LAG |     | Encodes diurnal seasonal pattern |
| Spatial Indicator | LOC-OHE |     | Provides city-specific bias terms |

1.  Cyclical Time Encoding (T-CYC)

Environmental data is governed by strong periodic behaviors, most notably the 24-hour diurnal cycle and the 7-day weekly industrial cycle. Standard integer encoding of time (e.g., 0-23 for hours) introduces a false numerical discontinuity at the transition from 23:00 to 00:00, where the model perceives a large leap instead of a smooth adjacent transition. To resolve this, we mapped these temporal components onto a two-dimensional unit circle using sine and cosine transformations:

,

where represents the raw time unit and _T_ represents its complete period (24 for hours, 7 for days). This Fourier-style transformation ensures that the Euclidean distance between any two hours reflects their actual temporal proximity, allowing the neural network to effectively learn the smooth variations in pollution levels during the midnight-to-dawn transition.

1.  Temporal Lag Features (T-LAG)

Atmospheric conditions exhibit significant “temporal inertia” or autocorrelation, where the state of the atmosphere at time is highly dependent on its state at _t − k_ . We explicitly modeled this dependency by constructing T-LAG features for both the target AQI and key pollutant concentrations. These features serve as a form of “short-term memory” for the statistical models, providing them with a look-back window into recent historical trends.

The lag operation was applied at three distinct scales: . The 1-hour and 2-hour lags capture immediate persistence and the current rate of change in air quality. In contrast, the 24-hour lag serves as a deterministic seasonal input, providing the model with the exact atmospheric state at the same hour on the previous day. This is critical for capturing recurrent “rush hour” patterns and consistent daily temperature inversions that influence pollutant trapping in urban canyons.

1.  Spatial Contextualization

To leverage the full breadth of the 13,863 observations across different regions, a single global model was trained to identify overarching atmospheric patterns. However, since each city possesses unique geographical features and industrial intensities, the model must be “city-aware” to avoid regional bias. We utilized one-hot encoding for the LOC variable, which transforms the categorical identifier into a sparse vector of binary indicators. This allows the hidden layers of the LSTM \[4\] and the coefficients of the SARIMA models to learn city-specific bias terms and unique interaction weights, effectively blending global trends with localized urban signatures.

## Data Splitting

To prevent data leakage and maintain the temporal structure of the time series, the dataset of 13,863 observations was partitioned chronologically rather than through random sampling. This approach reflects a real-world forecasting scenario in which models are trained on historical data and evaluated on future observations.

An 80/20 temporal split was applied, with the first 80% of observations used for training and the remaining 20% reserved for testing. Additionally, 10% of the training data was allocated for validation during model tuning and training using a rolling-forward approach. This strategy ensures that future information does not influence model training or evaluation, thereby preserving the reliability and validity of the forecasting results. The LSTM training and validation performance, shown in Fig. 3, indicates stable convergence with minimal signs of overfitting..

1.  LSTM training convergence (Loss: MAE).

## Forecasting Models

Several forecasting models were implemented to compare traditional statistical approaches and modern deep learning techniques for AQI prediction.

The Naive Forecast and Average Forecast served as baseline models, using the current AQI value and historical mean AQI, respectively. Holt-Winters Exponential Smoothing was applied to capture level, trend, and seasonal patterns in the data.

For statistical forecasting, ARIMA and SARIMA were used to model temporal dependencies, with SARIMA incorporating a 24-hour seasonal cycle to capture daily pollution patterns. Seasonal decomposition analysis (Fig. 1) confirmed the presence of significant daily seasonality.

Additionally, Long Short-Term Memory (LSTM) networks were implemented to model complex non-linear relationships among AQI and pollutant variables, enabling the capture of long-term temporal dependencies beyond the capabilities of traditional statistical models.

1.  Seasonal Analysis: Manila (HUC)

The LSTM neural network was selected as the representative deep learning model due to its proven ability to model complex, non-linear multivariate interactions over time \[3\]. Unlike traditional RNNs, the LSTM architecture utilizes a sophisticated gating mechanism to regulate the flow of information through its internal memory cells, effectively mitigating the vanishing gradient

problem \[5\]. The implemented architecture, visualized in Fig. 2, follows a stacked design with two recurrent layers comprising 128 and 64 hidden units respectively \[4\]. This hierarchy allows the model to extract low-level temporal features in the first layer and more abstract, long-range dependencies in the second.

1.  Stacked multivariate LSTM architecture.

At the core of this model is the LSTM cell, which maintains an internal cell state governed by four interacting mathematical gates. The Forget Gate determines which information from the previous state is discarded, while the Input Gate and Output Gate control the addition and exposure of temporal information \[3\], \[5\]. This complex logic enables the network to remember significant atmospheric events over many hours. The network was optimized using the Adam algorithm \[2\] and trained for 50 epochs using MAE as the loss function.

## Hyperparameter Tuning

A systematic hyperparameter tuning process was conducted to optimize the performance of the LSTM and SARIMA models while maintaining good generalization across different cities. The tuning aimed to minimize validation MAE and reduce the risk of overfitting to local pollution patterns and urban noise \[1\].

For the LSTM model, a grid search was performed on key parameters, including the number of hidden units, dropout rates, and Adam learning rates (1e-3 and 1e-4). Model performance was evaluated using a 10% validation split within the training data. The optimal configuration, shown in Table VI, consisted of a stacked LSTM architecture with 128 and 64 hidden units and a low dropout rate, indicating that the multivariate environmental features provided sufficient information without requiring strong regularization \[5\].

1.  Optimized LSTM Hyperparameter Configuration

|     |     |     |
| --- | --- | --- |
| Parameter | Optimized Value | Description |
| Recurrent Units | 128 / 64 | Stacked layer configuration |
| Activation |     | Activation |
| Dropout Rate | 0.05 | Regularization coefficient |
| Learning Rate |     | Adam optimizer initial step |
| Loss Function | MAE | Category-aware error objective |
| Batch Size | 32  | Mini-batch optimization window |
| Epochs | 50  | Maximum training iterations |

For the statistical models, hyperparameters such as the orders for ARIMA and the seasonal orders for SARIMA were determined via the minimized Akaike Information Criterion (AIC). This ensured that the selected models provided the best fit for the linear temporal dependencies while maintaining parsimony. The inclusion of the 24-hour seasonal differencing was specifically validated during this phase to confirm its effectiveness in handling the diurnal cycles identified in Section II.F.

## Evaluation Metrics

Model performance was evaluated using three widely used statistical metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE). In all equations, represents the actual AQI value, represents the predicted AQI value, and (n) is the total number of observations. These metrics provide a comprehensive assessment of forecasting accuracy and reliability \[16\], \[19\].

The **Mean Absolute Error (MAE)** measures the average magnitude of prediction errors and provides a straightforward indication of model accuracy. It is calculated as:

The **Root Mean Squared Error (RMSE)** measures prediction error while giving greater weight to larger errors, making it particularly useful for identifying models that fail to capture significant AQI spikes. It is defined as:.

The Mean Absolute Percentage Error (MAPE) expresses forecasting error as a percentage of the actual value, allowing performance comparisons across different datasets and conditions. It is calculated as:

Together, these metrics provide a balanced evaluation of model performance in forecasting AQI levels \[16\], \[19\].

# Results and Discussion

## Correlation Analysis

A correlation analysis was conducted to examine the relationship between AQI and the measured atmospheric pollutants using Pearson correlation coefficients (r) . The results, presented in Table VII, indicate that O₃ (ozone) has the strongest positive correlation with AQI (r = +0.82), followed by PM₂.₅ (r = +0.76) and PM₁₀ (r = +0.68). These findings suggest that ozone and particulate matter are the primary contributors to air quality deterioration in the selected urban areas.

The correlation heatmap shown in Fig. 4 further highlights the strong relationships among these pollutants and AQI. The dominance of O₃ indicates that urban air pollution is largely influenced by photochemical reactions involving sunlight and pollutant precursors such as nitrogen oxides \[6\].\].

1.  Pearson Correlation Coefficients Between Pollutants and AQI

|     |     |     |     |
| --- | --- | --- | --- |
| Pollutant Feature | Notation | Pearson Coefficient (r) | Pollutant Feature |
| Ozone | O3  | +0.82 | Photochemical reactions |
| Fine Particulate | PM25 | +0.76 | Vehicular combustion |
| Coarse Particulate | PM10 | +0.68 | Construction/Road dust |
| Nitrogen Dioxide | NO2 | +0.59 | Traffic congestion |
| Carbon Monoxide | CO  | +0.51 | Incomplete combustion |

The multi-pollutant correlation structure is further visualized in the heatmap in Fig. 4, which confirms that photochemical processes involving O3 are the primary drivers of air quality degradation in the study areas.

1.  Hourly AQI trend for Manila.

## Comparative Performance Analysis

The performance of the implemented forecasting architectures was evaluated across all ten cities during the testing phase. Table VIII provides an aggregated view of the error scores recorded for each model. Contrary to the initial expectation that deep learning would dominate all scenarios, the results show a more nuanced landscape where model suitability is highly dependent on the specific characteristics of the city being modeled. In many cases, the SARIMA model and even the Average Forecast provided surprisingly competitive results, particularly in terms of MAE.

1.  Aggregated Predictive Error Metric Comparisons

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
| Model Architecture | MAE | RMSE | MAPE | Performance Suitability |
| Average Forecast | 0.46 | 0.57 | 34.06% | Regional mean baseline |
| Naive Baseline | 0.61 | 0.87 | 29.16% | Persistence baseline |
| Holt-Winters | 0.64 | 0.91 | 30.56% | Dynamic trends |
| Linear ARIMA | 0.58 | 0.78 | 30.72% | Linear structural timelines |
| Seasonal ARIMA (SARIMA) | 0.66 | 0.84 | 34.15% | Diurnal cycles |
| LSTM (Optimized) | 0.61 | 0.87 | 29.19% | Complex non-linear paths |

The performance comparison shown in Fig. 5 highlights the consistent accuracy of the SARIMA and Average Forecast models across different cities. SARIMA performed particularly well because of its ability to capture the regular daily patterns present in urban air quality data.

A detailed 48-hour forecast analysis for Quezon City is presented in Fig. 6, where all predictions were rounded and clipped to the AQI's 1–5 categorical scale. The results show that the LSTM model was more effective at tracking sudden, non-linear changes in air quality, while simpler models often lagged behind or underestimated severe pollution events \[3\], \[4\].

1.  Model Error Comparison.

1.  48-hour forecast tracking on discrete 1-5 scale.

## Comparative Regional Analysis

The global performance metrics often mask significant variations in model effectiveness across different urban environments. To provide a more detailed understanding of these regional differences, we conducted a comparative analysis of model RMSE for representative cities, as shown in Table IX. This analysis reveals that cities with high seasonal regularity, such as Cebu City, are best modeled by SARIMA architectures that explicitly account for diurnal cycles.

In contrast, the more complex and chaotic environments of the National Capital Region (NCR), including Manila and Quezon City, exhibited higher overall error rates and a greater need for non-linear modeling. In these areas, the multivariate LSTM demonstrated better resilience to sudden shifts in pollutant concentrations, likely due to its ability to integrate information from multiple chemical sensors simultaneously \[4\]. Conversely, in cities with relatively low environmental variance, such as Cagayan de Oro, simpler baseline models like the Average Forecast remained highly effective.

1.  Regional Model Performance Variations (RMSE)

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
| City Profile | Representative City | SARIMA | LSTM | Optimal Model |
| Highly Seasonal | Cebu City | 0.47 | 0.67 | SARIMA |
| Complex/ Urban | Quezon City | 0.82 | 1.27 | SARIMA/LSTM |
| High Volatility | Manila | 1.06 | 1.32 | SARIMA |
| Low Variance | Cagayan de Oro | 0.33 | 0.35 | Naive/<br><br>Average |

## Policy Implications

The study demonstrates that accurate AQI forecasting models can support proactive environmental management and policy development in the Philippines \[6\]. Forecasting systems can help LGUs implement dynamic traffic management strategies to reduce pollution caused by vehicular emissions \[6\], \[22\], establish early warning systems that provide advance AQI alerts to the public, and support targeted industrial regulation and urban planning based on major pollution sources. Furthermore, the successful application of machine learning models highlights the potential for integrating predictive analytics into environmental policies and the implementation of the Philippine Clean Air Act \[6\].

# Conclusion

. This study developed and evaluated multiple multivariate forecasting models for predicting hourly AQI levels across ten highly urbanized Philippine cities using a high-frequency 2026 dataset \[10\]. The results demonstrated that both SARIMA and LSTM are effective forecasting approaches, with their performance varying according to local pollution patterns, seasonal behavior, and urban complexity. While SARIMA performed well in environments with strong seasonal regularity, LSTM showed greater capability in capturing complex and non-linear air quality dynamics.

The study also identified ground-level O₃ and PM₂.₅ as the primary contributors to air quality degradation in the selected urban areas. These findings provide valuable insights for air quality management and support the development of evidence-based environmental policies and public health strategies \[9\]. Overall, the integration of forecasting models into existing monitoring systems can enhance early warning capabilities and assist decision-makers in mitigating the impacts of urban air pollution in the Philippines.

# Recommendation

Based on the findings of this study, several recommendations are proposed for future research and practical implementation. First, future studies should utilize a longer observation period, ideally covering at least one full year, to capture seasonal variations and improve model robustness across changing environmental conditions \[20\].

Second, incorporating meteorological variables such as temperature, humidity, wind speed, and atmospheric pressure is recommended, as these factors significantly influence pollutant behavior and air quality patterns \[11\], \[13\]. Hybrid approaches that combine physical and data-driven models may further enhance forecasting accuracy.

Third, the geographic scope should be expanded to include more developing and rapidly urbanizing cities across the Philippines, particularly in Mindanao and the Cordillera Administrative Region, to improve the generalizability of forecasting models \[6\], \[7\].

Fourth, the best-performing models should be integrated into operational air quality monitoring systems. Connecting forecasting models with existing monitoring infrastructure through APIs could support real-time AQI prediction and public alert systems \[6\].

Finally, future research should investigate advanced deep learning approaches, such as transformer-based models and attention mechanisms, which have shown strong performance in capturing long-term and complex environmental patterns \[15\], \[17\].

# References

1.  J. Bergstra and Y. Bengio, "Random Search for Hyper-Parameter Optimization," Journal of Machine Learning Research, vol. 13, no. 10, pp. 281–305, 2012. \[Online\]. Available: https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf
2.  D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in Proc. 3rd Int. Conf. Learn. Representations (ICLR), San Diego, CA, USA, 2015. \[Online\]. Available: https://arxiv.org/abs/1412.6980
3.  S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997. \[Online\]. Available: https://doi.org/10.1162/neco.1997.9.8.1735
4.  A. Graves, A. Mohamed, and G. Hinton, "Speech recognition with deep recurrent neural networks," in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), Vancouver, BC, Canada, 2013, pp. 6645–6649. \[Online\]. Available: https://arxiv.org/abs/1303.5778
5.  F. A. Gers, J. Schmidhuber, and F. Cummins, "Learning to Forget: Continual Prediction with LSTM," Neural Computation, vol. 12, no. 10, pp. 2451–2471, Oct. 2000. \[Online\]. Available: https://doi.org/10.1162/089976600300015015
6.  Department of Environment and Natural Resources (DENR) Environmental Management Bureau, "Air Quality Management Section Status Report," DENR-EMB, 2024. \[Online\]. Available: https://emb.gov.ph/air-quality-management/
7.  Philippine Statistics Authority (PSA), "Highlights of the Philippine Population 2020 Census of Population and Housing (2020 CPH)," 2020. \[Online\]. Available: https://psa.gov.ph/content/highlights-philippine-population-2020-census-population-and-housing-2020-cph
8.  Philippine Statistics Authority (PSA), "Highlights of the National Capital Region (NCR) Population 2020 Census of Population and Housing," 2020. \[Online\]. Available: https://psa.gov.ph/content/highlights-national-capital-region-ncr-population-2020-census-population-and-housing-2020
9.  World Health Organization (WHO), "WHO global air quality guidelines," World Health Organization, Geneva, Switzerland, 2021. \[Online\]. Available: https://www.who.int/publications/i/item/9789240034228
10. B. Wandowando, "Philippine Cities Air Quality Index Data 2026," Kaggle Dataset, 2026. \[Online\]. Available: https://www.kaggle.com/datasets/bwandowando/philippine-cities-air-quality-index-data-2026
11. X. Feng, Q. Li, Y. Zhu, J. Hou, L. Jin, and J. Wang, "Artificial neural networks forecasting of PM2.5 pollution using air mass trajectory based geographic model and wavelet transformation," Atmospheric Environment, vol. 107, pp. 118–128, 2015. \[Online\]. Available: https://doi.org/10.1016/j.atmosenv.2015.02.030
12. C. J. Huang and P. H. Kuo, "A Deep CNN-LSTM Model for Particulate Matter (PM2.5) Forecasting in Smart Cities," Sensors, vol. 18, no. 7, p. 2220, 2018. \[Online\]. Available: https://doi.org/10.3390/s18072220
13. K. P. Moustris, I. C. Ziomas, and A. G. Paliatsos, "3-Day-Ahead Forecasting of Regional Pollution Index for the Pollutants NO2, CO, SO2, and O3 Using Artificial Neural Networks in Athens, Greece," Water, Air, & Soil Pollution, vol. 209, no. 1–4, pp. 29–43, 2010. \[Online\]. Available: https://doi.org/10.1007/s11270-009-0179-5
14. P. A. Dominick, D. Juahir, M. T. Latif, S. M. Zain, and A. Z. Aris, "Spatial assessment of air quality patterns in Malaysia using multivariate analysis," Atmospheric Environment, vol. 60, pp. 172–181, 2012. \[Online\]. Available: https://doi.org/10.1016/j.atmosenv.2012.06.021
15. A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is All You Need," in Proc. 31st Int. Conf. Neural Inf. Process. Syst. (NIPS), Long Beach, CA, USA, 2017, pp. 5998–6008. \[Online\]. Available: https://arxiv.org/abs/1706.03762
16. C. J. Willmott and K. Matsuura, "Advantages of the mean absolute error (MAE) over the root mean square error (RMSE) in assessing average model performance," Climate Research, vol. 30, no. 1, pp. 79–82, 2005. \[Online\]. Available: https://doi.org/10.3354/cr030079
17. H. Zhou, S. Zhang, J. Peng, S. Zhang, J. Li, H. Xiong, and W. Zhang, "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting," in Proc. 35th AAAI Conf. Artif. Intell., 2021, pp. 11106–11115. \[Online\]. Available: https://arxiv.org/abs/2012.07436
18. Y. Li, J. Peng, L. Zhang, and X. Zhang, "LSTM-based air quality prediction model," IOP Conf. Ser.: Earth Environ. Sci., vol. 440, no. 3, p. 032018, 2020. \[Online\]. Available: https://doi.org/10.1088/1755-1315/440/3/032018
19. T. M. Chai and R. R. Draxler, "Root mean square error (RMSE) or mean absolute error (MAE)? – Arguments against avoiding RMSE in the literature," Geoscientific Model Development, vol. 7, no. 3, pp. 1247–1250, 2014. \[Online\]. Available: https://doi.org/10.5194/gmd-7-1247-2014
20. G. P. Brasseur and D. J. Jacob, Modeling of Atmospheric Chemistry. Cambridge, U.K.: Cambridge University Press, 2017. \[Online\]. Available: https://doi.org/10.1017/9781316544754
21. J. B. Orosa, M. Á. Costa, A. Rodríguez-Fernández, and G. Roshan, "Effect of climate change on outdoor thermal comfort in humid climates," Journal of Environmental Engineering, vol. 140, no. 10, p. 04014037, 2014. \[Online\]. Available: [https://doi.org/10.1061/(ASCE)EE.1943-7870.0000877](https://doi.org/10.1061/%28ASCE%29EE.1943-7870.0000877)
22. R. V. Ramos and A. C. G. Varquez, "Improving prediction of PM2.5 in Metro Manila using XGBoost with Optuna hyperparameter optimization," ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences, 2024. \[Online\]. Available: https://doi.org/10.5194/isprs-annals-X-4-W4-2024-213-2024