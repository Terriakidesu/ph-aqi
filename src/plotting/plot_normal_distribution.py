from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_PATH = Path("datasets/selected/2026/2026_SelectedData_Imputed.csv")
OUTPUT_PATH = Path("figs/distribution/normal_distribution_components.png")
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


def normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std == 0:
        return np.zeros_like(x)
    scale = 1.0 / (std * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - mean) / std) ** 2
    return scale * np.exp(exponent)


def load_data() -> pd.DataFrame:
    return pd.read_csv(INPUT_PATH, usecols=COMPONENT_COLS)


def make_plot(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    for idx, col in enumerate(COMPONENT_COLS):
        ax = axes[idx]
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        mean = float(series.mean())
        std = float(series.std(ddof=0))

        ax.hist(series, bins=30, density=True, alpha=0.7, color="#4c78a8", edgecolor="white")

        if std > 0:
            x = np.linspace(float(series.min()), float(series.max()), 300)
            ax.plot(x, normal_pdf(x, mean, std), color="#e45756", linewidth=2)

        ax.set_title(col)
        ax.text(
            0.02,
            0.98,
            f"mean={mean:.2f}\nstd={std:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )
        ax.grid(True, alpha=0.2)

    fig.suptitle("Histogram with Fitted Normal Distribution", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return OUTPUT_PATH


def main() -> None:
    df = load_data()
    output_path = make_plot(df)
    print(f"Saved normal distribution plot to: {output_path}")


if __name__ == "__main__":
    main()
