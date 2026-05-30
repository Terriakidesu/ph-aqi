import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    csv_path = Path("data/predictions/forecast_model_predictions.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"])

    city = "Manila"
    model = "Holt-Winters" # One of the worst models
    
    city_df = df[(df["city_name"] == city) & (df["model"] == model)].sort_values("datetime")
    
    if city_df.empty:
        print(f"No data found for city {city} and model {model}.")
        return

    start_idx = 0
    end_idx = min(len(city_df), 48)
    plot_df = city_df.iloc[start_idx:end_idx].copy()

    # Round and clip to discrete 1-5 scale
    plot_df["predicted_aqi"] = plot_df["predicted_aqi"].round().clip(1, 5)

    # Optimize for IEEE double column (compact width, large fonts)
    plt.figure(figsize=(5, 4))
    plt.rcParams.update({'font.size': 13})

    plt.step(plot_df["datetime"], plot_df["actual_aqi"], where='post', label="Actual", linewidth=2.5, color='black', alpha=0.8)
    plt.step(plot_df["datetime"], plot_df["predicted_aqi"], where='post', label=f"Pred ({model})", linestyle='--', linewidth=2.0, color='red')

    plt.title(f"Worst Case: {city}", fontsize=14, fontweight='bold')
    plt.xlabel("Hours", fontsize=12)
    plt.ylabel("AQI (1-5)", fontsize=12)
    plt.ylim(0.5, 5.5)
    plt.yticks([1, 2, 3, 4, 5])
    plt.legend(loc='upper right', frameon=True, fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(rotation=30, ha="right", fontsize=10)
    
    output_dir = Path("figs/forecast_samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{city.lower().replace(' ', '_')}_forecast_48h_worst.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved worst-case forecast sample plot to: {save_path}")

if __name__ == "__main__":
    main()
