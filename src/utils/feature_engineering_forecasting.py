import math
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UTILS_DIR = PROJECT_ROOT / "src" / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

import constants


INPUT_PATH = Path("data/selected/2026/2026_SelectedData_Imputed.csv")
OUTPUT_PATH = Path("data/selected/2026/2026_SelectedData_Imputed_FeatureEngineered.csv")

DATETIME_COL = "datetime"
CITY_COL = "city_name"
TARGET_COL = "main.aqi"
POLLUTANT_COLS = [
    "components.co",
    "components.no",
    "components.no2",
    "components.o3",
    "components.so2",
    "components.pm2_5",
    "components.pm10",
    "components.nh3",
]
AQI_LAGS = [1, 2, 3, 6, 12, 24]
POLLUTANT_LAGS = [1, 3, 24]
ROLLING_WINDOWS = [3, 6, 12, 24]
FORECAST_HORIZON = 1


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH)
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
    df = df.sort_values([CITY_COL, DATETIME_COL]).reset_index(drop=True)
    df["region"] = df[CITY_COL].map(constants.CITY_TO_REGION)

    missing_regions = sorted(df.loc[df["region"].isna(), CITY_COL].dropna().unique())
    if missing_regions:
        raise ValueError(f"Missing region mapping for cities: {missing_regions}")

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df[DATETIME_COL].dt.hour
    df["day_of_week"] = df[DATETIME_COL].dt.dayofweek
    df["day_of_month"] = df[DATETIME_COL].dt.day
    df["month"] = df[DATETIME_COL].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = df["hour"].map(lambda h: math.sin(2 * math.pi * h / 24))
    df["hour_cos"] = df["hour"].map(lambda h: math.cos(2 * math.pi * h / 24))
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(CITY_COL, sort=False)

    for lag in AQI_LAGS:
        df[f"aqi_lag_{lag}"] = grouped[TARGET_COL].shift(lag)

    for col in POLLUTANT_COLS:
        short_name = col.split(".")[-1]
        for lag in POLLUTANT_LAGS:
            df[f"{short_name}_lag_{lag}"] = grouped[col].shift(lag)

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(CITY_COL, sort=False)

    for window in ROLLING_WINDOWS:
        df[f"aqi_roll_mean_{window}"] = grouped[TARGET_COL].transform(
            lambda s: s.rolling(window).mean()
        )
        df[f"aqi_roll_std_{window}"] = grouped[TARGET_COL].transform(
            lambda s: s.rolling(window).std()
        )
        df[f"pm2_5_roll_mean_{window}"] = grouped["components.pm2_5"].transform(
            lambda s: s.rolling(window).mean()
        )
        df[f"pm10_roll_mean_{window}"] = grouped["components.pm10"].transform(
            lambda s: s.rolling(window).mean()
        )

    return df


def add_change_features(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(CITY_COL, sort=False)
    df["aqi_diff_1"] = grouped[TARGET_COL].diff(1)
    df["aqi_diff_24"] = grouped[TARGET_COL].diff(24)
    df["pm2_5_diff_1"] = grouped["components.pm2_5"].diff(1)
    df["pm10_diff_1"] = grouped["components.pm10"].diff(1)
    return df


def add_forecast_target(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(CITY_COL, sort=False)
    df[f"target_aqi_t_plus_{FORECAST_HORIZON}"] = grouped[TARGET_COL].shift(-FORECAST_HORIZON)
    return df


def build_feature_engineered_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_change_features(df)
    df = add_forecast_target(df)

    feature_df = df.dropna().reset_index(drop=True)
    return feature_df


def save_dataset(df: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)


def main() -> None:
    df = load_data()
    feature_df = build_feature_engineered_dataset(df)
    save_dataset(feature_df)

    print(f"Saved feature-engineered dataset to: {OUTPUT_PATH}")
    print(f"Original imputed rows: {len(df)}")
    print(f"Rows after feature engineering: {len(feature_df)}")
    print(f"Dropped rows due to lag/rolling requirements: {len(df) - len(feature_df)}")
    print("\nSample columns:")
    print(feature_df.columns.to_list())


if __name__ == "__main__":
    main()
