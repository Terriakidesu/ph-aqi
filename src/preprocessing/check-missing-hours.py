import pandas as pd


def main():
    input_path = "datasets/selected/2026/2026_SelectedData.csv"
    output_path = "datasets/selected/2026/2026_MissingHours.csv"

    df = pd.read_csv(input_path)
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.floor("h")

    missing_rows = []

    for city, city_df in df.groupby("city_name"):
        observed = city_df["datetime"].drop_duplicates().sort_values()
        if observed.empty:
            continue

        full_range = pd.date_range(
            start=observed.iloc[0],
            end=observed.iloc[-1],
            freq="h",
        )
        missing = full_range.difference(observed)

        for ts in missing:
            missing_rows.append(
                {
                    "city_name": city,
                    "missing_datetime": ts,
                }
            )

    missing_df = pd.DataFrame(missing_rows).sort_values(
        by=["city_name", "missing_datetime"]
    )
    missing_df.to_csv(output_path, index=False)

    print(f"Saved missing-hour report to: {output_path}")
    print(f"Total missing rows: {len(missing_df)}")


if __name__ == "__main__":
    main()
