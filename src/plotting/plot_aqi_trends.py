import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Config
DATA_PATH = "data/selected/202601/202601_SelectedData.csv"
OUTPUT_DIR = Path("figs/aqi_trends")
TARGET_COL = "main.aqi"
CITY_COL = "city_name"
DATETIME_COL = "datetime"

POLLUTANTS = [
    "components.co", "components.no", "components.no2", "components.o3",
    "components.so2", "components.pm2_5", "components.pm10", "components.nh3"
]

def generate_trends():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(DATA_PATH)
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
    
    # 1. Feature Correlation Analysis
    print("\n[ANALYSIS] Calculating correlations with AQI...")
    corr_df = df[POLLUTANTS + [TARGET_COL]].corr(numeric_only=True)[TARGET_COL].sort_values(ascending=False)
    print(corr_df)
    
    # Save correlation to a text file for documentation
    with open(OUTPUT_DIR / "aqi_correlations.txt", "w") as f:
        f.write("Pollutant Correlations with main.aqi\n")
        f.write("====================================\n")
        f.write(corr_df.to_string())

    # 2. Per-City Trend Plots
    print("\n[PLOTTING] Generating trend graphs per city...")
    for city_name, city_df in df.groupby(CITY_COL):
        city_df = city_df.sort_values(DATETIME_COL)
        
        plt.figure(figsize=(12, 6))
        plt.plot(city_df[DATETIME_COL], city_df[TARGET_COL], label='Hourly AQI', alpha=0.5, color='blue')
        
        # Add Trend Line
        # Convert datetime to numeric for polyfit
        x_numeric = np.arange(len(city_df))
        z = np.polyfit(x_numeric, city_df[TARGET_COL], 1)
        p = np.poly1d(z)
        plt.plot(city_df[DATETIME_COL], p(x_numeric), "r--", linewidth=2, label='Trend Line')
        
        plt.title(f"AQI Trend Over Time - {city_name}")
        plt.xlabel("Date")
        plt.ylabel("AQI Level")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Format X-axis
        plt.gcf().autofmt_xdate()
        
        save_path = OUTPUT_DIR / f"{city_name.lower().replace(' ', '_')}_aqi_trend.png"
        plt.savefig(save_path)
        plt.close()
        print(f"  - Saved trend for {city_name} to {save_path}")

if __name__ == "__main__":
    generate_trends()
