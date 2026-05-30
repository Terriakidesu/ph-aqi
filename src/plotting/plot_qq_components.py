from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

try:
    from scipy import stats
except ImportError as exc:
    raise ImportError(
        "scipy is required for Q-Q plots. Install it in the active environment before running this script."
    ) from exc


INPUT_PATH = Path("datasets/selected/2026/2026_SelectedData_Imputed.csv")
OUTPUT_PATH = Path("figs/distribution/qq_plot_components.png")
COMPONENT_COLS = [
    "main.aqi",
    "components.co",
    "components.no",
    "components.no2",
    "components.o3",
    "components.so2",
    "components.pm2_5",
    "components.pm10",
    "components.nh3",
]


def load_data() -> pd.DataFrame:
    return pd.read_csv(INPUT_PATH, usecols=COMPONENT_COLS)


def make_plot(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    for idx, col in enumerate(COMPONENT_COLS):
        ax = axes[idx]
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        stats.probplot(series, dist="norm", plot=ax)
        ax.set_title(col)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Q-Q Plots for AQI and Pollutant Components", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return OUTPUT_PATH


def main() -> None:
    df = load_data()
    output_path = make_plot(df)
    print(f"Saved Q-Q plot figure to: {output_path}")


if __name__ == "__main__":
    main()
