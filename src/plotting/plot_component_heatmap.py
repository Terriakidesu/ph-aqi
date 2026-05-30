from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


INPUT_PATH = Path("data/selected/2026/2026_SelectedData_Imputed.csv")
OUTPUT_PATH = Path("figs/component_heatmap/component_correlation_heatmap.png")

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
    df = pd.read_csv(INPUT_PATH)
    return df[COMPONENT_COLS].copy()


def save_heatmap(df: pd.DataFrame) -> Path:
    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    plt.rcParams.update({'font.size': 11})
    
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.index, fontsize=9)
    ax.set_title("Pollutant Correlation Heatmap", fontsize=14, fontweight='bold', pad=20)

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Pearson Coefficient", rotation=-90, va="bottom")
    
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return OUTPUT_PATH


def main() -> None:
    df = load_data()
    output_path = save_heatmap(df)
    print(f"Saved component correlation heatmap to: {output_path}")


if __name__ == "__main__":
    main()
