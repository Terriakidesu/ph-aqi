import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    csv_path = Path("data/predictions/forecast_model_comparison.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run forecasting_models.py first.")
        return

    df = pd.read_csv(csv_path)
    # Sort by RMSE for better visualization
    df = df.sort_values("rmse")

    # Optimize for IEEE: compact and large fonts
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 7))
    plt.rcParams.update({'font.size': 13})

    # MAE Plot
    ax1.bar(df["model"], df["mae"], color="skyblue", edgecolor='black', linewidth=1.5)
    ax1.set_title("Mean Absolute Error (MAE)", fontsize=15, fontweight='bold')
    ax1.set_ylabel("Score", fontsize=13)
    ax1.set_xticks(range(len(df["model"])))
    ax1.set_xticklabels(df["model"], rotation=30, ha="right", fontsize=11)
    ax1.grid(axis='y', linestyle=':', alpha=0.8)

    # RMSE Plot
    ax2.bar(df["model"], df["rmse"], color="salmon", edgecolor='black', linewidth=1.5)
    ax2.set_title("Root Mean Squared Error (RMSE)", fontsize=15, fontweight='bold')
    ax2.set_ylabel("Score", fontsize=13)
    ax2.set_xticks(range(len(df["model"])))
    ax2.set_xticklabels(df["model"], rotation=30, ha="right", fontsize=11)
    ax2.grid(axis='y', linestyle=':', alpha=0.8)

    plt.tight_layout()
    
    output_dir = Path("figs/model_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "model_error_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved model comparison plot to: {save_path}")

if __name__ == "__main__":
    main()
