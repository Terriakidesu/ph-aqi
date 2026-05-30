import pandas as pd


def build_observation(total_rows: int, unique_count: int, top_count: int) -> str:
    if total_rows == 0:
        return "No rows available."
    top_share = (top_count / total_rows) * 100
    if unique_count == total_rows:
        return "High-cardinality feature; mostly unique values."
    if top_share >= 80:
        return f"Highly imbalanced; top category is {top_share:.2f}% of rows."
    if top_share >= 50:
        return f"Moderately imbalanced; top category is {top_share:.2f}% of rows."
    return f"Relatively distributed categories; top category is {top_share:.2f}%."


def main():
    input_path = "datasets/selected/2026/2026_SelectedData.csv"
    output_path = "datasets/selected/2026/2026_CategoricalAnalysis.csv"

    df = pd.read_csv(input_path)
    # For this AQI dataset, treat OpenWeather AQI level (1-5) as the categorical feature.
    categorical_cols = ["main.aqi"] if "main.aqi" in df.columns else []

    rows = []
    for feature in categorical_cols:
        aqi_labels = {
            1: "Good",
            2: "Fair",
            3: "Moderate",
            4: "Poor",
            5: "Very Poor",
        }
        counts = (
            pd.to_numeric(df[feature], errors="coerce")
            .value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
        )
        top_n = counts.sort_values(ascending=False)

        top_categories = "; ".join(
            [f"{int(idx)} - {aqi_labels[int(idx)]} ({int(val)})" for idx, val in top_n.items()]
        )
        top_count = int(top_n.iloc[0]) if not top_n.empty else 0

        rows.append(
            {
                "Feature": feature,
                "No. of Categories": 5,
                "Top Categories": top_categories,
                "Observations": build_observation(len(df), 5, top_count),
            }
        )

    report_df = pd.DataFrame(rows)
    if not report_df.empty:
        report_df = report_df.sort_values(by="Feature").reset_index(drop=True)
    else:
        report_df = pd.DataFrame(
            columns=[
                "Feature",
                "No. of Categories",
                "Top Categories",
                "Observations",
            ]
        )

    report_df.to_csv(output_path, index=False)
    print(report_df.to_string(index=False))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
