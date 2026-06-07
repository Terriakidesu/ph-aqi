import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


OUTPUT_DIR = Path("figs/model_comparison")


def plot_merged_model_errors(df: pd.DataFrame) -> Path:
    """Plot MAE and RMSE together in one grouped bar chart."""
    df = df.sort_values("rmse")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    plt.rcParams.update({"font.size": 12})

    x = np.arange(len(df["model"]))
    width = 0.36

    ax.bar(
        x - width / 2,
        df["mae"],
        width,
        label="MAE",
        color="#4C78A8",
        edgecolor="black",
        linewidth=1.0,
    )
    ax.bar(
        x + width / 2,
        df["rmse"],
        width,
        label="RMSE",
        color="#F58518",
        edgecolor="black",
        linewidth=1.0,
    )

    ax.set_title("Forecast Model Error Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Error Score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=25, ha="right", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.75)
    ax.legend(frameon=False, ncol=2)

    fig.tight_layout()
    save_path = OUTPUT_DIR / "model_error_comparison.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_lstm_rolling_window_chart(prediction_path: Path) -> Path | None:
    save_path = OUTPUT_DIR / "rolling_window_validation_compact.png"
    if save_path.exists():
        print(f"LSTM rolling window chart already exists: {save_path}")
        return save_path

    if not prediction_path.exists():
        print(f"Warning: {prediction_path} not found. Skipping LSTM rolling window chart.")
        return None

    predictions = pd.read_csv(prediction_path)
    lstm_df = predictions[predictions["model"].str.upper() == "LSTM"].copy()
    if lstm_df.empty:
        print("Warning: no LSTM predictions found. Skipping LSTM rolling window chart.")
        return None

    lstm_df["datetime"] = pd.to_datetime(lstm_df["datetime"])
    lstm_df = lstm_df.sort_values(["city_name", "datetime"])
    lstm_df["squared_error"] = (
        lstm_df["actual_aqi"] - lstm_df["predicted_aqi"]
    ) ** 2
    lstm_df["rolling_rmse"] = lstm_df.groupby("city_name")["squared_error"].transform(
        lambda values: np.sqrt(values.rolling(window=24, min_periods=6).mean())
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for city, city_df in lstm_df.groupby("city_name"):
        ax.plot(
            city_df["datetime"],
            city_df["rolling_rmse"],
            linewidth=1.2,
            alpha=0.85,
            label=city,
        )

    ax.set_title("LSTM Rolling Window Validation RMSE", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("24-hour Rolling RMSE", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def main():
    csv_path = Path("data/predictions/forecast_model_comparison.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run forecasting_models.py first.")
        return

    df = pd.read_csv(csv_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_path = plot_merged_model_errors(df)
    print(f"Saved merged model comparison plot to: {save_path}")

    table_path = Path("per_city_metrics_table.md")
    if table_path.exists():
        print(f"Per-city metrics table available as Markdown: {table_path}")
    else:
        print(f"Warning: {table_path} not found.")

    rolling_path = plot_lstm_rolling_window_chart(
        Path("data/predictions/forecast_model_predictions.csv")
    )
    if rolling_path:
        print(f"Saved LSTM rolling window chart to: {rolling_path}")

if __name__ == "__main__":
    main()
