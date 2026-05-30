import math
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UTILS_DIR = PROJECT_ROOT / "src" / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

import constants


SOURCE_PATH = Path("data/all/2026/2026_CombinedData.csv")
SELECTED_OUTPUT_PATH = Path("data/selected/2026/2026_CityAware_SelectedData.csv")
IMPUTED_OUTPUT_PATH = Path("data/selected/2026/2026_CityAware_Imputed.csv")
FEATURE_OUTPUT_PATH = Path("data/selected/2026/2026_CityAware_FeatureEngineered.csv")
REPORT_OUTPUT_PATH = Path("data/selected/2026/2026_CityAware_FeatureEngineeringReport.csv")

DATETIME_COL = "datetime"
CITY_COL = "city_name"
AQI_COL = "main.aqi"
REGION_COL = "island_group"
TARGET_COL = "target_aqi_t_plus_1"
NUMERIC_COLS = [
    "components.co",
    "components.no",
    "components.no2",
    "components.o3",
    "components.so2",
    "components.pm2_5",
    "components.pm10",
    "components.nh3",
]
ORDERED_BASE_COLS = [DATETIME_COL, AQI_COL, *NUMERIC_COLS, CITY_COL, REGION_COL]
AQI_LAGS = [1, 2, 24]
POLLUTANT_LAGS = [1, 24]
ROLLING_WINDOWS = [24]


def load_source_data() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_PATH)
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL]).dt.floor("h")
    df = df[df[CITY_COL].isin(constants.CITIES)].copy()
    df = df.drop_duplicates(subset=[CITY_COL, DATETIME_COL], keep="last")
    df[REGION_COL] = df[CITY_COL].map(constants.CITY_TO_ISLAND_GROUP)

    missing_regions = sorted(df.loc[df[REGION_COL].isna(), CITY_COL].dropna().unique())
    if missing_regions:
        raise ValueError(f"Missing Luzon/Visayas/Mindanao mapping for cities: {missing_regions}")

    return df.sort_values([CITY_COL, DATETIME_COL]).reset_index(drop=True)


def impute_city_frame(city_df: pd.DataFrame) -> pd.DataFrame:
    city_df = city_df.sort_values(DATETIME_COL).copy()
    full_index = pd.date_range(
        start=city_df[DATETIME_COL].min(),
        end=city_df[DATETIME_COL].max(),
        freq="h",
    )

    city_df = city_df.set_index(DATETIME_COL).reindex(full_index)
    city_df.index.name = DATETIME_COL

    city_name = city_df[CITY_COL].dropna().iloc[0]
    city_df[CITY_COL] = city_name
    city_df[REGION_COL] = constants.CITY_TO_ISLAND_GROUP[city_name]

    for col in NUMERIC_COLS:
        city_df[col] = pd.to_numeric(city_df[col], errors="coerce")
        city_df[col] = city_df[col].interpolate(method="time").ffill().bfill()

    city_df[AQI_COL] = pd.to_numeric(city_df[AQI_COL], errors="coerce")
    city_df[AQI_COL] = city_df[AQI_COL].interpolate(method="time").ffill().bfill()
    city_df[AQI_COL] = city_df[AQI_COL].round().clip(1, 5).astype(int)

    return city_df.reset_index()[ORDERED_BASE_COLS]


def build_imputed_dataset(selected_df: pd.DataFrame) -> pd.DataFrame:
    frames = [
        impute_city_frame(city_df)
        for _, city_df in selected_df.groupby(CITY_COL, sort=True)
        if not city_df.empty
    ]
    if not frames:
        raise ValueError("No selected city rows found for city-aware imputation.")

    out_df = pd.concat(frames, ignore_index=True)
    out_df = out_df.sort_values([CITY_COL, DATETIME_COL]).reset_index(drop=True)
    out_df[NUMERIC_COLS] = out_df[NUMERIC_COLS].round(2)
    return out_df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df[DATETIME_COL].dt.hour
    df["day_of_week"] = df[DATETIME_COL].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = df["hour"].map(lambda h: math.sin(2 * math.pi * h / 24))
    df["hour_cos"] = df["hour"].map(lambda h: math.cos(2 * math.pi * h / 24))
    return df


def add_city_local_lags(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(CITY_COL, sort=False)

    for lag in AQI_LAGS:
        df[f"aqi_lag_{lag}"] = grouped[AQI_COL].shift(lag)

    for col in NUMERIC_COLS:
        short_name = col.split(".")[-1]
        for lag in POLLUTANT_LAGS:
            df[f"{short_name}_lag_{lag}"] = grouped[col].shift(lag)

    return df


def add_city_local_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(CITY_COL, sort=False)

    for window in ROLLING_WINDOWS:
        df[f"aqi_roll_mean_{window}"] = grouped[AQI_COL].transform(lambda s: s.rolling(window).mean())
        df[f"pm2_5_roll_mean_{window}"] = grouped["components.pm2_5"].transform(
            lambda s: s.rolling(window).mean()
        )
        df[f"pm10_roll_mean_{window}"] = grouped["components.pm10"].transform(
            lambda s: s.rolling(window).mean()
        )

    return df


def add_forecast_target(df: pd.DataFrame) -> pd.DataFrame:
    df[TARGET_COL] = df.groupby(CITY_COL, sort=False)[AQI_COL].shift(-1)
    return df


def build_feature_engineered_dataset(imputed_df: pd.DataFrame) -> pd.DataFrame:
    feature_df = imputed_df.sort_values([CITY_COL, DATETIME_COL]).reset_index(drop=True).copy()
    feature_df = add_time_features(feature_df)
    feature_df = add_city_local_lags(feature_df)
    feature_df = add_city_local_rolling_features(feature_df)
    feature_df = add_forecast_target(feature_df)
    return feature_df.dropna().reset_index(drop=True)


def build_report(selected_df: pd.DataFrame, imputed_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for city_name in constants.CITIES:
        selected_count = int((selected_df[CITY_COL] == city_name).sum())
        imputed_count = int((imputed_df[CITY_COL] == city_name).sum())
        feature_count = int((feature_df[CITY_COL] == city_name).sum())
        rows.append(
            {
                "city_name": city_name,
                REGION_COL: constants.CITY_TO_ISLAND_GROUP[city_name],
                "selected_rows": selected_count,
                "imputed_rows": imputed_count,
                "inserted_hourly_rows": imputed_count - selected_count,
                "feature_engineered_rows": feature_count,
                "imputation_scope": "per city",
                "feature_scope": "per city",
                "regional_use": "aggregation only, not training",
            }
        )
    return pd.DataFrame(rows)


def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = df.copy()
    out_df[DATETIME_COL] = out_df[DATETIME_COL].astype(str)
    out_df.to_csv(output_path, index=False)


def main() -> None:
    selected_df = load_source_data()
    imputed_df = build_imputed_dataset(selected_df)
    feature_df = build_feature_engineered_dataset(imputed_df)
    report_df = build_report(selected_df, imputed_df, feature_df)

    save_csv(selected_df[ORDERED_BASE_COLS], SELECTED_OUTPUT_PATH)
    save_csv(imputed_df, IMPUTED_OUTPUT_PATH)
    save_csv(feature_df, FEATURE_OUTPUT_PATH)
    report_df.to_csv(REPORT_OUTPUT_PATH, index=False)

    print(f"Saved city-aware selected data to: {SELECTED_OUTPUT_PATH}")
    print(f"Saved city-aware imputed data to: {IMPUTED_OUTPUT_PATH}")
    print(f"Saved city-aware feature-engineered data to: {FEATURE_OUTPUT_PATH}")
    print(f"Saved city-aware report to: {REPORT_OUTPUT_PATH}")
    print("\nFeature engineering report:")
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    main()
