Good day everyone. Today we will study **Time Series Analysis and Forecasting**, one of the most important techniques in business analytics.

Forecasting helps organizations make decisions such as:

- estimating sales
- planning inventory
- predicting demand
- preparing budgets
- anticipating risks

In this lesson, we will explore four major forecasting approaches:

1.  Naïve Forecasting
2.  Average of Past Values Method
3.  Moving Average Method
4.  Exponential Smoothing

Here is our lesson roadmap. We begin with an overview of forecasting concepts, then move step-by-step through four forecasting methods arranged from simplest to more refined:

Naïve → Average → Moving Average → Exponential Smoothing

Each method improves how we use historical data to predict future values.

Patrick Henry once said:

_“I know of no way of judging the future but by the past.”_

This statement perfectly summarizes **time series forecasting**. In business analytics, we assume past patterns help us understand future behavior. However, this assumption only works if data is reliable, patterns are stable, and environment does not drastically change.

Forecasting methods are divided into two categories: **First: Qualitative Forecasting**. These rely on expert opinion, experience, judgment, or intuition. Example: Predicting election results without historical data.

**Second: Quantitative Forecasting.** These rely on historical numerical data. We use them when past information exists, data can be measured, and future follows historical patterns. Time series forecasting belongs here.

Qualitative forecasting is used when historical data is unavailable. Examples include forecasting new product demand, predicting technology adoption, and estimating policy effects. These methods depend heavily on experience, expert judgment, or professional intuition. Sometimes we call them “educated guesses.”

A **time series** is a sequence of observations collected across time. Examples: daily stock prices, monthly sales, annual population, and weekly gasoline demand. For time series analysis to be valid, data must be comparable across:

- **Time** (same interval length). Example: monthly vs yearly cannot mix.
- **Space** (same geographic unit). Example: Bacolod vs Philippines cannot mix.
- **Methodology** (same measurement method).

The first step in analyzing time series data is always: Create a time series plot. This plot helps us visually identify patterns, movements, or behavior changes. Before applying formulas, we must first understand: What kind of pattern exists?

A **horizontal pattern** means values fluctuate around a constant average. Example: daily temperature inside a controlled environment This is also called a **stationary time series**.

Stationary means mean remains constant, variance remains constant, and behavior remains stable over time. This type of data is easiest to forecast.

A **trend pattern** shows long-term upward or downward movement. Examples: population growth, technology adoption, inflation rates, or online shopping demand. Trends are usually caused by economic changes, technology improvements, demographic shifts, and consumer behavior changes. Trend patterns require adjusted forecasting techniques.

Next, we have **seasonal patterns**. These repeat within a fixed period. Examples: Christmas sales increase, umbrella sales during rainy season, and electricity demand during summer.

**Cyclical patterns** are different. They occur over longer periods, usually several years.

Example: economic boom and recession cycles. Understanding whether a pattern is seasonal or cyclical helps us choose the correct forecasting model.

This slide tests whether a dataset is stationary. Example: Daily change in Dow Jones Index is stationary. Why? Because changes fluctuate around zero. However, Dow Jones Index itself is NOT stationary because it has a trend. Stationary data produces better forecasting reliability.

Now we begin our first forecasting method: **Naïve Forecasting**. This is the simplest forecasting method. Rule: Future value equals last observed value. Example: If today's sales = 100 units, forecast tomorrow's sales = 100 units. Very simple. Very fast. Surprisingly effective in stable environments.

Forecasting is not only about predicting values. We must also evaluate: How accurate is our forecast? Accuracy is measured using forecast error, which is equals to the actual value minus forecast value. Smaller errors mean better forecasts.

Two common accuracy measures are: **Mean Forecast Error (MFE)**, which measures average forecasting bias. It shows whether forecasts are too high or too low. **Mean Absolute Error (MAE)** measures average magnitude of error. It ignores positive or negative direction**.** This is widely used in business forecasting.

Next accuracy measures include: **Mean Squared Error (MSE)**, which squares errors before averaging. This penalizes large errors heavily, which is useful in model comparison. **Mean Absolute Percentage Error (MAPE)** expresses error as percentage, which is very useful when comparing different datasets. Example: weekly sales vs monthly sales**.** MAPE allows fair comparison.

Our second forecasting method is Average of Past Values. Instead of using only the last observation, we use average of ALL previous observations. Example: Week 1 = 17, Week 2 = 21, so we forecast Week 3 equals to (17 + 21) ÷ 2 = 19. This method reduces random variation effects.

Here we compare two methods: Naïve Method and Average Method. We evaluate them using MAE, MSE, and MAPE. Result: Average method performs better. Why? Because it uses more information. More information usually improves predictions.

Next method: **Moving Average Method.** Instead of averaging ALL previous values, we average only the most recent **k values**. Example: 3-period moving average uses last 3 observations only. Advantage: More responsive to recent changes. This makes forecasts more realistic in dynamic environments.

Here we compare three forecasting methods Naïve, Average, and Moving Average. Notice: Moving average improves accuracy compared to naïve method, but still depends heavily on chosen window size k. Choosing correct k is important.

Now we move to our most advanced method today: **Exponential Smoothing.** This method assigns larger weights to recent observations and smaller weights to older observations. This means that recent data influences forecasts more strongly. This reflects real-world behavior better.

Exponential smoothing uses a parameter called alpha (α), which controls responsiveness. Small alpha means slow adjustment and stable forecast. Large alpha means fast adjustment and sensitive forecast. Choosing alpha depends on data behavior.

Now we compare all four forecasting methods Naïve, Average, Moving Average, and Exponential Smoothing.

Result: Exponential smoothing produces the best accuracy among them. This is why it is widely used in inventory systems, sales prediction, demand forecasting, and financial planning.

Today we learned four important forecasting methods: Naïve Forecasting, Average of Past Values, Moving Average, and Exponential Smoothing. We also learned how to evaluate forecasting accuracy using MFE, MAE, MSE, and MAPE. These techniques form the foundation of predictive analytics in business decision-making.

Thank you everyone.