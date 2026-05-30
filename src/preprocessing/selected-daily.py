import pandas as pd
import constants


def main():
    df = pd.read_csv("datasets/selected/202601/202601_SelectedData.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df[df["city_name"].isin(constants.CITIES)].copy()
    df["date"] = df["datetime"].dt.floor("D")

    numeric_columns = constants.COMPONENTS[:-1]
    daily_df = (
        df.groupby(["city_name", "date"], as_index=False)[numeric_columns]
        .mean()
        .rename(columns={"date": "datetime"})
    )
    daily_df["main.aqi"] = daily_df["main.aqi"].round().astype(int)

    ordered_columns = numeric_columns + ["city_name", "datetime"]
    daily_df = daily_df[ordered_columns].sort_values(["datetime", "city_name"])

    daily_df.to_csv("datasets/selected/202601/202601_SelectedDailyData.csv", index=False)


if __name__ == "__main__":
    main()
