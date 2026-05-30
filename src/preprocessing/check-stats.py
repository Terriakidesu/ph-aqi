import pandas as pd
import numpy as np
import constants

def main():
    # input_path = "datasets/selected/2026/2026_SelectedData.csv"
    input_path = "datasets/selected/2026/2026_SelectedData_Imputed.csv"

    df = pd.read_csv(input_path)
    rows = []

    for component in constants.COMPONENTS[1:-1]:

        mean = np.mean(df[component])
        median = np.median(df[component])
        std_dev = np.std(df[component])
        _min = np.min(df[component])
        _max = np.max(df[component])

        rows.append(
            {
                "feature": component,
                "mean": mean,
                "median": median,
                "std_dev": std_dev,
                "min": _min,
                "max": _max, 
            }
        )

    stats_df = pd.DataFrame(rows).round(2)
    print(stats_df.to_string(index=False))

    stats_df.to_csv("stats.csv", index=False)


if __name__ == "__main__":
    main()
