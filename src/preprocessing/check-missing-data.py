import pandas as pd


def handling_method_for(feature: str, is_numeric: bool) -> str:
    if feature in {"datetime", "city_name"}:
        return "Drop rows with missing key fields"
    if is_numeric:
        return "Interpolate by city and hour, then forward/backward fill"
    return "Fill with mode per city"


def main():
    input_path = "datasets/selected/2026/2026_SelectedData.csv"
    output_path = "datasets/selected/2026/2026_MissingDataAnalysis.csv"

    df = pd.read_csv(input_path)
    total_rows = len(df)

    rows = []
    for feature in df.columns:
        missing_count = int(df[feature].isna().sum())
        if missing_count == 0:
            continue

        percentage = (missing_count / total_rows) * 100 if total_rows else 0
        rows.append(
            {
                "Feature": feature,
                "No. of Missing Values": missing_count,
                "Percentage": round(percentage, 2),
                "Handling Method": handling_method_for(
                    feature, pd.api.types.is_numeric_dtype(df[feature])
                ),
            }
        )

    report_df = pd.DataFrame(rows)

    if report_df.empty:
        report_df = pd.DataFrame(
            columns=[
                "Feature",
                "No. of Missing Values",
                "Percentage",
                "Handling Method",
            ]
        )

    report_df.to_csv(output_path, index=False)
    print(report_df.to_string(index=False))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
