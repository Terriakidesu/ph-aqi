import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Load predictions
df = pd.read_csv('data/predictions/hourly_aqi_predictions.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime')

# Prepare output directory
output_dir = Path("figs/model_comparison/rolling_validation")
output_dir.mkdir(parents=True, exist_ok=True)

# Generate plot for each city
for city in df['city_name'].unique():
    city_df = df[df['city_name'] == city].copy()
    
    # Calculate Squared Error
    city_df['squared_error'] = (city_df['true_aqi'] - city_df['pred_aqi']) ** 2

    # Rolling Window RMSE (24-hour window)
    city_df['rolling_rmse'] = np.sqrt(city_df['squared_error'].rolling(window=24).mean())

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(city_df['datetime'], city_df['rolling_rmse'], label='Rolling 24h RMSE', color='darkblue')
    plt.title(f'Rolling Window Validation RMSE ({city})', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save
    save_path = output_dir / f"{city.lower().replace(' ', '_')}_rolling_rmse.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved plot for {city} to: {save_path}")
