import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pathlib import Path

DATA_PATH = Path("data/selected/2026/2026_SelectedData.csv")
OUTPUT_DIR = Path("figs/seasonal_decomposition")

def main():
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found.")
        return

    data = pd.read_csv(DATA_PATH)
    data["datetime"] = pd.to_datetime(data["datetime"])
    
    # Use Manila for demonstration
    city_name = "Manila"
    city_df = data[data["city_name"] == city_name].sort_values("datetime")
    
    # Set datetime as index and frequency
    city_df = city_df.set_index("datetime")
    # Resample to hourly just in case there are gaps, although the data should be hourly
    series = city_df["main.aqi"].resample("H").mean().ffill()
    
    print(f"Performing seasonal decomposition for {city_name}...")
    # Period 24 for daily seasonality in hourly data
    result = seasonal_decompose(series, model='additive', period=24)
    
    fig = result.plot()
    # Increase font sizes and make more compact for IEEE
    # Width ~5 inches as per previous calibrations
    fig.set_size_inches(5.5, 9)
    
    import matplotlib.dates as mdates
    
    axes = fig.get_axes()
    for i, ax in enumerate(axes):
        ax.tick_params(labelsize=10)
        ax.set_ylabel(ax.get_ylabel(), fontsize=11, fontweight='bold')
        # Thicken lines in subplots
        for line in ax.get_lines():
            line.set_linewidth(1.2)
        
        # Format X-axis for all subplots (or at least the bottom one)
        # To avoid clutter, use a weekly locator
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        
        # Only show x-labels on the bottom plot to save space
        if i < len(axes) - 1:
            ax.set_xlabel('')
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel('Timeline (Jan-Feb 2026)', fontsize=12, fontweight='bold')
            plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

    plt.suptitle(f"Fig. 1. Seasonal Analysis: {city_name} (HUC)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / f"{city_name.lower().replace(' ', '_')}_decomposition.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved seasonal decomposition plot to: {save_path}")

if __name__ == "__main__":
    main()
