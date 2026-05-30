from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UTILS_DIR = PROJECT_ROOT / "src" / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

import constants


INPUT_PATH = Path("data/all/2026/2026_CombinedData.csv")
SELECTED_OUTPUT_PATH = Path("data/selected/2026/2026_SelectedData.csv")
IMPUTED_OUTPUT_PATH = Path("data/selected/2026/2026_SelectedData_Imputed.csv")
JAN_OUTPUT_PATH = Path("data/selected/202601/202601_SelectedData.csv")
REPORT_PATH = Path("data/selected/2026/2026_SelectedData_PreprocessingReport.csv")

DATETIME_COL = "datetime"
CITY_COL = "city_name"
AQI_COL = "main.aqi"
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
ORDERED_COLS = [DATETIME_COL, AQI_COL, *NUMERIC_COLS, CITY_COL]


def load_source_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH)
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL]).dt.floor("h")
    return df


def build_selected_dataset(df: pd.DataFrame) -> pd.DataFrame:
    selected_df = df[df[CITY_COL].isin(constants.CITIES)].copy()
    selected_df = selected_df.drop_duplicates(subset=[CITY_COL, DATETIME_COL], keep="last")
    selected_df = selected_df.sort_values([DATETIME_COL, CITY_COL]).reset_index(drop=True)
    return selected_df[ORDERED_COLS]


def impute_city_frame(city_df: pd.DataFrame) -> pd.DataFrame:
    city_df = city_df.sort_values(DATETIME_COL).copy()
    
    # Removed: reindexing to full_index (no row insertion/augmentation)
    city_df = city_df.set_index(DATETIME_COL)
    city_name = city_df[CITY_COL].dropna().iloc[0]
    city_df[CITY_COL] = city_name

    for col in NUMERIC_COLS:
        city_df[col] = pd.to_numeric(city_df[col], errors="coerce")
        # Interpolate existing rows only
        city_df[col] = city_df[col].interpolate(method="time").ffill().bfill()

    city_df[AQI_COL] = pd.to_numeric(city_df[AQI_COL], errors="coerce")
    city_df[AQI_COL] = city_df[AQI_COL].interpolate(method="time").ffill().bfill()
    city_df[AQI_COL] = city_df[AQI_COL].round().clip(1, 5).astype(int)

    return city_df.reset_index()[ORDERED_COLS]


def build_imputed_dataset(selected_df: pd.DataFrame) -> pd.DataFrame:
    imputed_frames = []

    for city_name, city_df in selected_df.groupby(CITY_COL, sort=True):
        if city_df.empty:
            continue
        imputed_frames.append(impute_city_frame(city_df))

    if not imputed_frames:
        raise ValueError("No selected city data was available for imputation.")

    out_df = pd.concat(imputed_frames, ignore_index=True)
    out_df = out_df.sort_values([DATETIME_COL, CITY_COL]).reset_index(drop=True)
    out_df[NUMERIC_COLS] = out_df[NUMERIC_COLS].round(2)
    out_df[DATETIME_COL] = out_df[DATETIME_COL].astype(str)
    return out_df[ORDERED_COLS]


def validate_city_hours(city_df: pd.DataFrame) -> dict:
    city_df = city_df.copy()
    city_df[DATETIME_COL] = pd.to_datetime(city_df[DATETIME_COL])
    city_df = city_df.sort_values(DATETIME_COL)

    city_name = city_df[CITY_COL].iloc[0]
    duplicate_count = int(city_df.duplicated(subset=[DATETIME_COL]).sum())

    start = city_df[DATETIME_COL].min()
    end = city_df[DATETIME_COL].max()
    expected_index = pd.date_range(start=start, end=end, freq="h")
    actual_index = pd.DatetimeIndex(city_df[DATETIME_COL])
    missing_hours = expected_index.difference(actual_index)

    return {
        "city_name": city_name,
        "start_datetime": start,
        "end_datetime": end,
        "row_count": int(len(city_df)),
        "duplicate_timestamps": duplicate_count,
        "expected_hourly_rows": int(len(expected_index)),
        "missing_hour_count": int(len(missing_hours)),
    }


def build_preprocessing_report(selected_df: pd.DataFrame, imputed_df: pd.DataFrame) -> pd.DataFrame:
    selected_checks = []
    imputed_checks = []

    for _, city_df in selected_df.groupby(CITY_COL, sort=True):
        selected_checks.append(validate_city_hours(city_df))

    imputed_df_for_check = imputed_df.copy()
    imputed_df_for_check[DATETIME_COL] = pd.to_datetime(imputed_df_for_check[DATETIME_COL])
    for _, city_df in imputed_df_for_check.groupby(CITY_COL, sort=True):
        imputed_checks.append(validate_city_hours(city_df))

    selected_report = pd.DataFrame(selected_checks).rename(
        columns={
            "row_count": "original_rows",
            "duplicate_timestamps": "original_duplicate_timestamps",
            "expected_hourly_rows": "expected_hourly_rows",
            "missing_hour_count": "original_missing_hour_count",
        }
    )
    imputed_report = pd.DataFrame(imputed_checks).rename(
        columns={
            "row_count": "imputed_rows",
            "duplicate_timestamps": "imputed_duplicate_timestamps",
            "missing_hour_count": "imputed_missing_hour_count",
        }
    )

    report_df = selected_report.merge(
        imputed_report[
            [
                "city_name",
                "imputed_rows",
                "imputed_duplicate_timestamps",
                "imputed_missing_hour_count",
            ]
        ],
        on="city_name",
        how="left",
    )
    report_df["inserted_hourly_rows"] = report_df["imputed_rows"] - report_df["original_rows"]
    report_df["ready_for_forecasting"] = (
        (report_df["original_duplicate_timestamps"] == 0)
        & (report_df["imputed_duplicate_timestamps"] == 0)
        & (report_df["imputed_missing_hour_count"] == 0)
    )
    report_df["pollutant_imputation_method"] = "Time-based linear interpolation with forward/backward fill"
    report_df["aqi_imputation_method"] = "Time-based interpolation, rounded to nearest integer, clipped to 1-5"

    ordered_report_cols = [
        "city_name",
        "start_datetime",
        "end_datetime",
        "original_rows",
        "expected_hourly_rows",
        "original_duplicate_timestamps",
        "original_missing_hour_count",
        "imputed_rows",
        "inserted_hourly_rows",
        "imputed_duplicate_timestamps",
        "imputed_missing_hour_count",
        "pollutant_imputation_method",
        "aqi_imputation_method",
        "ready_for_forecasting",
    ]
    report_df = report_df[ordered_report_cols].sort_values("city_name").reset_index(drop=True)

    totals_row = {
        "city_name": "TOTAL (10 CITIES)",
        "start_datetime": report_df["start_datetime"].min(),
        "end_datetime": report_df["end_datetime"].max(),
        "original_rows": int(report_df["original_rows"].sum()),
        "expected_hourly_rows": int(report_df["expected_hourly_rows"].sum()),
        "original_duplicate_timestamps": int(report_df["original_duplicate_timestamps"].sum()),
        "original_missing_hour_count": int(report_df["original_missing_hour_count"].sum()),
        "imputed_rows": int(report_df["imputed_rows"].sum()),
        "inserted_hourly_rows": int(report_df["inserted_hourly_rows"].sum()),
        "imputed_duplicate_timestamps": int(report_df["imputed_duplicate_timestamps"].sum()),
        "imputed_missing_hour_count": int(report_df["imputed_missing_hour_count"].sum()),
        "pollutant_imputation_method": "Time-based linear interpolation with forward/backward fill",
        "aqi_imputation_method": "Time-based interpolation, rounded to nearest integer, clipped to 1-5",
        "ready_for_forecasting": bool(report_df["ready_for_forecasting"].all()),
    }

    return pd.concat([report_df, pd.DataFrame([totals_row])], ignore_index=True)


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    source_df = load_source_data()
    selected_df = build_selected_dataset(source_df)
    imputed_df = build_imputed_dataset(selected_df)
    report_df = build_preprocessing_report(selected_df, imputed_df)
    total_original_rows = len(selected_df)
    total_imputed_rows = len(imputed_df)
    total_inserted_rows = total_imputed_rows - total_original_rows

    save_dataset(selected_df, SELECTED_OUTPUT_PATH)
    save_dataset(selected_df, JAN_OUTPUT_PATH)
    save_dataset(imputed_df, IMPUTED_OUTPUT_PATH)
    save_dataset(report_df, REPORT_PATH)

    print(f"Saved selected dataset to: {SELECTED_OUTPUT_PATH}")
    print(f"Saved selected dataset copy to: {JAN_OUTPUT_PATH}")
    print(f"Saved imputed dataset to: {IMPUTED_OUTPUT_PATH}")
    print(f"Saved preprocessing report to: {REPORT_PATH}")
    print("\nTotal rows across all selected cities:")
    print(f"Before imputation: {total_original_rows}")
    print(f"After imputation: {total_imputed_rows}")
    print(f"Inserted hourly rows: {total_inserted_rows}")
    print("\nPer-city preprocessing report:")
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    main()
