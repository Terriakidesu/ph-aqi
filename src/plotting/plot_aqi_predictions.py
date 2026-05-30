import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"city_name", "target_datetime", "true_aqi", "pred_aqi"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["target_datetime"] = pd.to_datetime(df["target_datetime"], errors="coerce")
    df = df.dropna(subset=["target_datetime"]).sort_values(["city_name", "target_datetime"])
    return df


def plot_city(ax: plt.Axes, city_df: pd.DataFrame, city_name: str) -> None:
    ax.plot(city_df["target_datetime"], city_df["true_aqi"], label="True AQI", linewidth=1.8)
    ax.plot(city_df["target_datetime"], city_df["pred_aqi"], label="Pred AQI", linewidth=1.4, alpha=0.9)
    ax.set_title(city_name)
    ax.set_ylabel("AQI Category")
    ax.set_ylim(0.5, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.grid(alpha=0.25)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot predicted vs true AQI from hourly predictions CSV.")
    parser.add_argument(
        "--csv",
        default="datasets/predictions/hourly_aqi_predictions.csv",
        help="Path to hourly predictions CSV.",
    )
    parser.add_argument("--city", default=None, help="City name to plot. If omitted, plots all cities.")
    parser.add_argument("--start", default=None, help="Start datetime filter (e.g. 2026-01-26).")
    parser.add_argument("--end", default=None, help="End datetime filter (e.g. 2026-01-31).")
    parser.add_argument("--save", default=None, help="Optional output image path (e.g. aqi_plot.png).")
    parser.add_argument("--show", action="store_true", help="Show interactive plot window.")
    args = parser.parse_args()

    df = load_data(args.csv)

    if args.start:
        df = df[df["target_datetime"] >= pd.to_datetime(args.start)]
    if args.end:
        df = df[df["target_datetime"] <= pd.to_datetime(args.end)]

    if args.city:
        df = df[df["city_name"] == args.city]
        if df.empty:
            raise ValueError(f"No rows found for city='{args.city}' with current filters.")
        fig, ax = plt.subplots(figsize=(12, 4.5))
        plot_city(ax, df, args.city)
        ax.set_xlabel("Target Datetime")
        ax.legend(loc="upper left")
        fig.tight_layout()
    else:
        cities = sorted(df["city_name"].unique())
        if not cities:
            raise ValueError("No rows to plot after filtering.")
        fig, axes = plt.subplots(len(cities), 1, figsize=(12, max(4, 2.8 * len(cities))), sharex=True)
        if len(cities) == 1:
            axes = [axes]
        for ax, city in zip(axes, cities):
            plot_city(ax, df[df["city_name"] == city], city)
        axes[-1].set_xlabel("Target Datetime")
        axes[0].legend(loc="upper left")
        fig.tight_layout()

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        print(f"Saved plot to: {out_path}")

    if args.show or not args.save:
        plt.show()


if __name__ == "__main__":
    main()
