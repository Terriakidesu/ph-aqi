import pandas as pd
import numpy as np


def impute_city_frame(city_df: pd.DataFrame) -> pd.DataFrame:
    city_df = city_df.sort_values("datetime").copy()
    city_df["datetime"] = pd.to_datetime(city_df["datetime"]).dt.floor("h")

    # Build complete hourly index between first and last observed timestamp.
    full_index = pd.date_range(
        start=city_df["datetime"].min(),
        end=city_df["datetime"].max(),
        freq="h",
    )

    city_df = city_df.set_index("datetime")
    city_df = city_df.reindex(full_index)
    city_df.index.name = "datetime"

    # Constant for the group and for newly created rows.
    city_name = city_df["city_name"].dropna().iloc[0]
    city_df["city_name"] = city_name

    # Continuous pollutant components: time-aware interpolation then edge fill.
    numeric_components = [
        "components.co",
        "components.no",
        "components.no2",
        "components.o3",
        "components.so2",
        "components.pm2_5",
        "components.pm10",
        "components.nh3",
    ]

    for col in numeric_components:
        city_df[col] = pd.to_numeric(city_df[col], errors="coerce")
        city_df[col] = city_df[col].interpolate(method="time")
        city_df[col] = city_df[col].ffill().bfill()

    # AQI is ordinal categorical (1-5): interpolate over time, round, then clamp.
    city_df["main.aqi"] = pd.to_numeric(city_df["main.aqi"], errors="coerce")
    city_df["main.aqi"] = city_df["main.aqi"].interpolate(method="time")
    city_df["main.aqi"] = city_df["main.aqi"].ffill().bfill()
    city_df["main.aqi"] = city_df["main.aqi"].round().clip(1, 5).astype(int)

    return city_df.reset_index()


def main():
    input_path = "data/selected/2026/2026_SelectedData.csv"
    output_path = "data/selected/2026/2026_SelectedData_Imputed.csv"

    df = pd.read_csv(input_path)
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.floor("h")

    imputed_frames = []
    for city, city_df in df.groupby("city_name", sort=True):
        if city_df.empty:
            continue
        imputed_frames.append(impute_city_frame(city_df))

    if not imputed_frames:
        raise ValueError("No city data found to process.")

    out_df = pd.concat(imputed_frames, ignore_index=True)
    out_df = out_df.sort_values(["datetime", "city_name"]).reset_index(drop=True)

    # Keep consistent column order.
    ordered_cols = [
        "datetime",
        "main.aqi",
        "components.co",
        "components.no",
        "components.no2",
        "components.o3",
        "components.so2",
        "components.pm2_5",
        "components.pm10",
        "components.nh3",
        "city_name",
    ]
    out_df = out_df[ordered_cols]

    # Round continuous pollutant features for cleaner output.
    round_cols = [
        "components.co",
        "components.no",
        "components.no2",
        "components.o3",
        "components.so2",
        "components.pm2_5",
        "components.pm10",
        "components.nh3",
    ]
    out_df[round_cols] = out_df[round_cols].round(2)

    # Preserve timezone offset format in CSV.
    out_df["datetime"] = out_df["datetime"].astype(str)
    out_df.to_csv(output_path, index=False)

    inserted_rows = len(out_df) - len(df)
    print(f"Saved imputed dataset to: {output_path}")
    print(f"Original rows: {len(df)}")
    print(f"Rows after hourly completion: {len(out_df)}")
    print(f"Inserted (imputed) rows: {inserted_rows}")


if __name__ == "__main__":
    main()
