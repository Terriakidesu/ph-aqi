import pandas as pd
from pathlib import Path
import constants


def main():
    df = pd.read_csv("datasets/all/2026/2026_CombinedData.csv")
    selected_df = df[df["city_name"].isin(constants.CITIES)].copy()

    selected_df["datetime"] = pd.to_datetime(selected_df["datetime"]).dt.floor("h")
    selected_df = selected_df.sort_values(by=["datetime", "city_name"])

    output_paths = [
        Path("datasets/selected/2026/2026_SelectedData.csv"),
        Path("datasets/selected/202601/202601_SelectedData.csv"),
    ]

    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        selected_df.to_csv(path, index=False)


if __name__ == "__main__":
    main()
