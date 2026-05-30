from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


INPUT_PATH = Path("datasets/selected/2026/2026_SelectedData_Imputed_FeatureEngineered.csv")
OUTPUT_DIR = Path("figs/feature_engineered")
DATETIME_COL = "datetime"
REGION_COL = "region"

CORRELATION_FEATURES = [
    "main.aqi",
    "target_aqi_t_plus_1",
    "aqi_lag_1",
    "aqi_lag_24",
    "aqi_roll_mean_24",
    "aqi_roll_std_24",
    "components.pm2_5",
    "components.pm10",
    "pm2_5_lag_24",
    "pm10_lag_24",
    "aqi_diff_1",
    "aqi_diff_24",
]

DISTRIBUTION_FEATURES = [
    "main.aqi",
    "target_aqi_t_plus_1",
    "aqi_lag_1",
    "aqi_lag_24",
    "aqi_roll_mean_24",
    "aqi_diff_1",
]

SCATTER_FEATURES = [
    "aqi_lag_1",
    "aqi_lag_24",
    "aqi_roll_mean_24",
    "components.pm2_5",
    "components.pm10",
    "components.no2",
]

TREND_FEATURES = [
    "main.aqi",
    "aqi_lag_24",
    "aqi_roll_mean_24",
    "aqi_diff_24",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH)
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
    return df.sort_values([REGION_COL, DATETIME_COL]).reset_index(drop=True)


def save_correlation_heatmap(df: pd.DataFrame) -> Path:
    corr = df[CORRELATION_FEATURES].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    ax.set_title("Feature Correlation Heatmap")

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "feature_correlation_heatmap.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_distribution_plots(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, feature in enumerate(DISTRIBUTION_FEATURES):
        ax = axes[idx]
        ax.hist(df[feature], bins=30, color="#2a6f97", edgecolor="white", alpha=0.9)
        ax.set_title(feature)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Feature Distributions", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = OUTPUT_DIR / "feature_distributions.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_target_scatter_plots(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, feature in enumerate(SCATTER_FEATURES):
        ax = axes[idx]
        sample = df[[feature, "target_aqi_t_plus_1"]].sample(
            n=min(2500, len(df)), random_state=42
        )
        ax.scatter(
            sample[feature],
            sample["target_aqi_t_plus_1"],
            s=10,
            alpha=0.35,
            color="#bc4749",
        )
        ax.set_title(f"{feature} vs target_aqi_t_plus_1")
        ax.set_xlabel(feature)
        ax.set_ylabel("target_aqi_t_plus_1")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Key Features vs Forecast Target", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = OUTPUT_DIR / "feature_target_scatter.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_region_trend_plots(df: pd.DataFrame) -> list[Path]:
    saved_paths = []

    for region_name, region_df in df.groupby(REGION_COL, sort=True):
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
        axes = axes.flatten()

        region_hourly = region_df.groupby(DATETIME_COL, as_index=False)[TREND_FEATURES].mean()

        for idx, feature in enumerate(TREND_FEATURES):
            ax = axes[idx]
            ax.plot(region_hourly[DATETIME_COL], region_hourly[feature], color="#1d3557", linewidth=1.6)
            ax.set_title(feature)
            ax.grid(True, alpha=0.2)

        fig.suptitle(f"Feature Trends by Region: {region_name}", fontsize=15)
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        out_path = OUTPUT_DIR / f"{region_name.lower().replace(' ', '_')}_feature_trends.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    saved = [
        save_correlation_heatmap(df),
        save_distribution_plots(df),
        save_target_scatter_plots(df),
        *save_region_trend_plots(df),
    ]

    print("Saved feature-engineered plots:")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
