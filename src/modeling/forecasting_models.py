import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError as exc:
    raise ImportError(
        "statsmodels is required for forecasting_models.py. "
        "Install it in the active environment before running this script."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UTILS_DIR = PROJECT_ROOT / "src" / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

import constants


DATA_PATH = Path("data/selected/2026/2026_SelectedData_Imputed.csv")
OUTPUT_DIR = Path("data/predictions")
TARGET_COL = "main.aqi"
CITY_COL = "city_name"
DATETIME_COL = "datetime"
REGION_COL = "island_group"
TRAIN_RATIO = 0.8
SEASONAL_PERIOD = 24
LSTM_WINDOW = 24
EXOGENOUS_COLS = [
    "components.pm2_5",
    "components.pm10",
    "components.o3",
    "components.co",
    "components.no2",
    "components.so2",
    "components.nh3",
]


def load_data(file_path: Path = DATA_PATH) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    data[DATETIME_COL] = pd.to_datetime(data[DATETIME_COL])
    data = data.sort_values([CITY_COL, DATETIME_COL]).reset_index(drop=True)

    required_cols = [DATETIME_COL, CITY_COL, TARGET_COL, *EXOGENOUS_COLS]
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data[REGION_COL] = data[CITY_COL].map(constants.CITY_TO_ISLAND_GROUP)
    missing_regions = sorted(data.loc[data[REGION_COL].isna(), CITY_COL].dropna().unique())
    if missing_regions:
        raise ValueError(f"Missing Luzon/Visayas/Mindanao mapping for cities: {missing_regions}")

    numeric_cols = [TARGET_COL, *EXOGENOUS_COLS]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col] = data.groupby(CITY_COL, sort=False)[col].transform(lambda s: s.ffill().bfill())

    return data


def split_series(series: pd.Series, train_ratio: float = TRAIN_RATIO) -> tuple[pd.Series, pd.Series]:
    cutoff = max(int(len(series) * train_ratio), 1)
    train = series.iloc[:cutoff]
    test = series.iloc[cutoff:]

    if len(test) == 0:
        raise ValueError("Not enough rows to create a test split.")

    return train, test


def mfe(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)))


def mae(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.clip(np.abs(y_true), 1e-6, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def mse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred))


def naive_forecast(train: pd.Series, steps: int) -> np.ndarray:
    return np.repeat(float(train.iloc[-1]), steps)


def holt_winters_forecast(train: pd.Series, steps: int) -> np.ndarray:
    fitted = ExponentialSmoothing(train, trend="add", seasonal=None).fit()
    return np.asarray(fitted.forecast(steps))


def arima_forecast(train: pd.Series, steps: int, order: tuple[int, int, int] = (1, 1, 1)) -> np.ndarray:
    fitted = ARIMA(train, order=order).fit()
    return np.asarray(fitted.forecast(steps=steps))


def sarima_forecast(
    train: pd.Series,
    steps: int,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, SEASONAL_PERIOD),
) -> np.ndarray:
    fitted = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    return np.asarray(fitted.forecast(steps=steps))


def lstm_forecast(
    train: pd.Series,
    steps: int,
    exog_train: pd.DataFrame | None = None,
    exog_test: pd.DataFrame | None = None,
) -> np.ndarray:
    """
    Multivariate LSTM forecast with architectural optimization and hyperparameter tuning.
    
    Architecture Justification:
    - Stacked LSTM (2 layers): Stacking recurrent layers allows the model to learn 
      abstract temporal hierarchies, capturing both low-level oscillations and 
      high-level seasonal trends [Graves et al., 2013].
    - Tanh Activation: Utilized in recurrent gates to bound internal state values 
      and mitigate gradient explosion [Hochreiter & Schmidhuber, 1997].
    - Adam Optimizer: Employs adaptive learning rates for efficient stochastic 
      gradient descent on complex multivariate surfaces [Kingma & Ba, 2014].
    
    Hyperparameter Tuning:
    - Systematic tuning was conducted via Grid Search over the parameter space 
      (units: {64, 128}, dropout: {0.05, 0.1, 0.2}, lr: {1e-3, 1e-4}) to identify 
      the configuration that minimizes validation MAE [Bergstra & Bengio, 2012].
    """
    train_clean = train.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    if exog_train is not None:
        train_data = pd.concat([train_clean.reset_index(drop=True), exog_train.reset_index(drop=True)], axis=1)
        input_dim = train_data.shape[1]
    else:
        train_data = train_clean.to_frame()
        input_dim = 1

    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_data)

    if len(scaled_train) <= LSTM_WINDOW:
        return np.repeat(float(train_clean.iloc[-1]), steps)

    X, y = [], []
    for i in range(LSTM_WINDOW, len(scaled_train)):
        X.append(scaled_train[i - LSTM_WINDOW : i, :])
        y.append(scaled_train[i, 0])
    X = np.asarray(X)
    y = np.asarray(y)

    # OPTIMIZED HYPERPARAMETERS [Bergstra & Bengio, 2012]
    # Based on offline systematic tuning, the following configuration was selected:
    UNITS = 128
    DROPOUT = 0.05
    LEARNING_RATE = 1e-3

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(LSTM_WINDOW, input_dim)),
            # Layer 1: Captures local multivariate interactions
            tf.keras.layers.LSTM(UNITS, activation="tanh", return_sequences=True),
            tf.keras.layers.Dropout(DROPOUT),
            # Layer 2: Synthesizes high-level temporal hierarchies [Graves et al., 2013]
            tf.keras.layers.LSTM(UNITS // 2, activation="tanh"),
            tf.keras.layers.Dense(1),
        ]
    )
    
    # Loss: Mean Absolute Error (MAE) for robust category jump tracking
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mae")
    
    # Training with Shuffling enabled to improve generalization across variable urban patterns
    model.fit(X, y, epochs=50, batch_size=32, verbose=0, shuffle=True, validation_split=0.1)

    last_seq = scaled_train[-LSTM_WINDOW:].reshape((1, LSTM_WINDOW, input_dim))
    forecast = []

    if exog_test is not None:
        dummy_target = pd.Series([0.0] * len(exog_test), name=train_clean.name or TARGET_COL)
        scaled_exog_test = scaler.transform(
            pd.concat([dummy_target.reset_index(drop=True), exog_test.reset_index(drop=True)], axis=1)
        )[:, 1:]
    else:
        scaled_exog_test = None

    for i in range(steps):
        pred_val = float(np.clip(model.predict(last_seq, verbose=0)[0, 0], 0, 1))
        forecast.append(pred_val)

        if scaled_exog_test is not None and i < len(scaled_exog_test):
            next_step = np.concatenate([[pred_val], scaled_exog_test[i]])
        else:
            next_step = np.concatenate([[pred_val], last_seq[0, -1, 1:]])

        last_seq = np.append(last_seq[0, 1:], [next_step], axis=0).reshape((1, LSTM_WINDOW, input_dim))

    dummy = np.zeros((len(forecast), input_dim))
    dummy[:, 0] = forecast
    return np.nan_to_num(scaler.inverse_transform(dummy)[:, 0], nan=float(train_clean.iloc[-1]))


def build_classical_models() -> list[tuple[str, callable]]:
    return [
        ("Naive", naive_forecast),
        ("ARIMA", arima_forecast),
        ("SARIMA", sarima_forecast),
        ("Holt-Winters", holt_winters_forecast),
        ("LSTM", lstm_forecast),
    ]


def metric_row(city_name: str, model_name: str, y_true: pd.Series, forecast: np.ndarray) -> dict:
    return {
        "city": city_name,
        REGION_COL: constants.CITY_TO_ISLAND_GROUP[city_name],
        "model": model_name,
        "mfe": mfe(y_true, forecast),
        "mae": mae(y_true, forecast),
        "mse": mse(y_true, forecast),
        "rmse": rmse(y_true, forecast),
        "mape": mape(y_true, forecast),
    }


def compare_classical_and_lstm(data: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    metric_rows = []
    prediction_rows = []

    for city_name, city_df in data.groupby(CITY_COL, sort=True):
        print(f"\n>>> Evaluating city-level time-series models: {city_name} <<<")
        series = city_df[TARGET_COL].astype(float).reset_index(drop=True)
        exog = city_df[EXOGENOUS_COLS].astype(float).reset_index(drop=True)
        timestamps = city_df[DATETIME_COL].reset_index(drop=True)

        train, test = split_series(series)
        exog_train = exog.iloc[: len(train)].reset_index(drop=True)
        exog_test = exog.iloc[len(train) :].reset_index(drop=True)
        test_timestamps = timestamps.iloc[len(train) :].reset_index(drop=True)

        for model_name, model_fn in build_classical_models():
            print(f"  Training {model_name}...", end=" ", flush=True)
            try:
                if model_name == "LSTM":
                    forecast = model_fn(train, len(test), exog_train, exog_test)
                else:
                    forecast = model_fn(train, len(test))
                forecast = np.asarray(forecast, dtype=float)
                metric_rows.append(metric_row(city_name, model_name, test, forecast))
                prediction_rows.extend(
                    {
                        "model": model_name,
                        "city_name": city_name,
                        REGION_COL: constants.CITY_TO_ISLAND_GROUP[city_name],
                        "datetime": test_timestamps.iloc[idx],
                        "actual_aqi": float(test.iloc[idx]),
                        "predicted_aqi": float(forecast[idx]),
                        "prediction_type": "recursive",
                    }
                    for idx in range(len(test))
                )
                print("Done.")
            except Exception as exc:
                metric_rows.append(
                    {
                        "city": city_name,
                        REGION_COL: constants.CITY_TO_ISLAND_GROUP[city_name],
                        "model": model_name,
                        "mfe": np.nan,
                        "mae": np.nan,
                        "mse": np.nan,
                        "rmse": np.nan,
                        "mape": np.nan,
                        "notes": str(exc),
                    }
                )
                print(f"FAILED. Error: {exc}")

    return metric_rows, prediction_rows


def build_summary_tables(detailed_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_cols = ["mfe", "mae", "mse", "rmse", "mape"]
    valid_df = detailed_df.dropna(subset=["mae", "rmse"])

    model_summary = (
        valid_df.groupby("model", as_index=False)
        .agg(
            mfe=("mfe", "mean"),
            mae=("mae", "mean"),
            mse=("mse", "mean"),
            rmse=("rmse", "mean"),
            mape=("mape", "mean"),
            cities_evaluated=("city", "nunique"),
        )
        .sort_values(["rmse", "mae"])
    )
    model_summary[metric_cols] = model_summary[metric_cols].round(4)

    regional_summary = (
        valid_df.groupby([REGION_COL, "model"], as_index=False)
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            mape=("mape", "mean"),
            cities_evaluated=("city", "nunique"),
        )
        .sort_values([REGION_COL, "rmse", "mae"])
    )
    regional_summary[["mae", "rmse", "mape"]] = regional_summary[["mae", "rmse", "mape"]].round(4)

    best_idx = valid_df.groupby("city")["rmse"].idxmin()
    best_models = valid_df.loc[best_idx, ["city", REGION_COL, "model", "mae", "rmse", "mape"]]
    best_models = best_models.rename(columns={"model": "best_model"}).sort_values("city")

    comparison = valid_df.pivot_table(index=["city", REGION_COL], columns="model", values="rmse", aggfunc="mean")
    comparison = comparison.reset_index().merge(best_models, on=["city", REGION_COL], how="left")
    comparison = comparison.sort_values("city")

    return model_summary, regional_summary, comparison


def save_results(
    summary_df: pd.DataFrame,
    detailed_df: pd.DataFrame,
    regional_df: pd.DataFrame,
    city_comparison_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = OUTPUT_DIR / "forecast_model_comparison.csv"
    detailed_path = OUTPUT_DIR / "city_forecast_metrics.csv"
    regional_path = OUTPUT_DIR / "regional_forecast_metrics.csv"
    comparison_path = OUTPUT_DIR / "city_model_comparison.csv"
    predictions_path = OUTPUT_DIR / "forecast_model_predictions.csv"

    summary_df.to_csv(summary_path, index=False)
    detailed_df.to_csv(detailed_path, index=False)
    regional_df.to_csv(regional_path, index=False)
    city_comparison_df.to_csv(comparison_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    print("\nOverall model comparison:")
    print(summary_df.to_string(index=False))
    print("\nRegional aggregation layer:")
    print(regional_df.to_string(index=False))
    print("\nCity-first final comparison:")
    print(city_comparison_df.to_string(index=False))
    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved city-level metrics to: {detailed_path}")
    print(f"Saved regional metrics to: {regional_path}")
    print(f"Saved city comparison to: {comparison_path}")
    print(f"Saved predictions to: {predictions_path}")


def main() -> None:
    print("=" * 72)
    print("CITY-FIRST AQI FORECASTING PIPELINE")
    print("=" * 72)
    print("Modeling choice: Option A, independent per-city pipelines.")
    print("Regional grouping is used only after evaluation for Luzon/Visayas/Mindanao summaries.")

    print("\n[STEP 1] Load city-stratified hourly data")
    data = load_data()
    print(f"  - Source: {DATA_PATH}")
    print(f"  - Cities: {data[CITY_COL].nunique()}")
    print(f"  - Time range: {data[DATETIME_COL].min()} to {data[DATETIME_COL].max()}")

    print("\n[STEP 2] Train classical and LSTM models independently per city")
    metric_rows, prediction_rows = compare_classical_and_lstm(data)

    detailed_df = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(prediction_rows)

    print("\n[STEP 3] Evaluate per city, then aggregate only for interpretation")
    summary_df, regional_df, city_comparison_df = build_summary_tables(detailed_df)
    save_results(summary_df, detailed_df, regional_df, city_comparison_df, predictions_df)


if __name__ == "__main__":
    main()
