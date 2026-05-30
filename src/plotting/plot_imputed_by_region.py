from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

import constants


matplotlib.use("Agg")
import matplotlib.pyplot as plt


INPUT_PATH = Path("datasets/selected/2026/2026_SelectedData_Imputed.csv")
OUTPUT_DIR = Path("figs/imputed_by_region")
COMPONENTS = constants.COMPONENTS[:-1]


def load_imputed_dataset() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(
        "Asia/Manila"
    )
    df["region"] = df["city_name"].map(constants.CITY_TO_REGION)

    unknown_cities = sorted(df.loc[df["region"].isna(), "city_name"].dropna().unique())
    if unknown_cities:
        raise ValueError(f"Missing region mapping for cities: {unknown_cities}")

    return df.sort_values(["region", "city_name", "datetime"]).reset_index(drop=True)


def build_region_plot(region_df: pd.DataFrame, region_name: str) -> Path:
    fig, axes = plt.subplots(3, 3, figsize=(20, 14), sharex=True)
    axes = axes.flatten()

    for idx, component in enumerate(COMPONENTS):
        ax = axes[idx]
        label = constants.COMPONENT_NAMES[component]

        for city_name, city_df in region_df.groupby("city_name", sort=True):
            ax.plot(
                city_df["datetime"],
                city_df[component],
                linewidth=1.8,
                alpha=0.9,
                label=city_name,
            )

        region_mean = (
            region_df.groupby("datetime", as_index=False)[component]
            .mean()
            .sort_values("datetime")
        )
        ax.plot(
            region_mean["datetime"],
            region_mean[component],
            color="dimgray",
            linewidth=1.5,
            alpha=0.75,
            label="Regional mean",
        )

        if len(region_mean) > 1:
            x_idx = np.arange(len(region_mean))
            coeffs = np.polyfit(x_idx, region_mean[component], 1)
            trend_line = np.poly1d(coeffs)(x_idx)
            ax.plot(
                region_mean["datetime"],
                trend_line,
                color="crimson",
                linewidth=2.6,
                linestyle="--",
                label="Trend line",
            )

        ax.set_title(label)
        ax.set_ylabel("AQI level" if component == "main.aqi" else constants.UNITS)
        if component == "main.aqi":
            ax.set_ylim(1, 5)
            ax.set_yticks([1, 2, 3, 4, 5])
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    for ax in axes[len(COMPONENTS) :]:
        ax.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(4, len(labels)),
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=0.95,
        fontsize=10,
    )
    fig.suptitle(f"Imputed Air Quality Trends by Region: {region_name}", fontsize=16)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{region_name.lower().replace(' ', '_')}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    df = load_imputed_dataset()

    for region_name, region_df in df.groupby("region", sort=False):
        output_path = build_region_plot(region_df, region_name)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
