import csv

# Read the data
with open('data/predictions/city_forecast_metrics.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    
    # Format the table
    markdown_table = "| City | Model | MAE | RMSE | MAPE |\\n"
    markdown_table += "| --- | --- | --- | --- | --- |\\n"
    
    for row in reader:
        markdown_table += f"| {row['city']} | {row['model']} | {float(row['mae']):.4f} | {float(row['rmse']):.4f} | {float(row['mape']):.2f} |\\n"

# Write to per-city-result.md
with open('per-city-result.md', 'w') as f:
    f.write("# Per-City Forecast Metrics\\n\\n" + markdown_table)
