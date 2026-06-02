# -*- coding: utf-8 -*-
"""
Generate per-city 48-hour AQI forecast plots from real prediction data.
Saves all city figures and identifies the most visually interesting one.
"""

import sys
import os
import shutil

# Force UTF-8 stdout so arrow chars don't crash on Windows cp1252
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- paths --------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_CSV = os.path.join(BASE_DIR, "data", "predictions", "forecast_model_predictions.csv")
OUT_DIR  = os.path.join(BASE_DIR, "figs", "forecast_samples")
os.makedirs(OUT_DIR, exist_ok=True)

# -- style for IEEE compact double-column -------------------------------------
MODEL_STYLES = {
    "Actual":       dict(color="black",   lw=1.5, ls="-",  zorder=10, marker="o", ms=3, mew=0),
    "LSTM":         dict(color="#d9534f", lw=1.2, ls="-",  zorder=8,  marker=None),
    "SARIMA":       dict(color="#5bc0de", lw=1.2, ls="--", zorder=7,  marker=None),
    "ARIMA":        dict(color="#5cb85c", lw=1.2, ls="-.", zorder=6,  marker=None),
    "Holt-Winters": dict(color="#f0ad4e", lw=1.2, ls=":",  zorder=5,  marker=None),
    "Naive":        dict(color="#777777", lw=1.0, ls="-",  zorder=4,  marker=None),
    "Average":      dict(color="#cccccc", lw=1.0, ls="--", zorder=3,  marker=None),
}


def pick_best_48h_window(series: pd.Series):
    """Return start index of the 48-h window with the highest 'interest' score."""
    n = len(series)
    if n <= 48:
        return 0
    best_start, best_score = 0, -1.0
    for start in range(0, n - 48, 4):
        window = series.iloc[start:start + 48]
        score  = window.var() + (window.max() - window.min()) + (window.diff().abs() > 0).sum() * 0.3
        if score > best_score:
            best_score = score
            best_start = start
    return best_start


def plot_city(city_name: str, wide_window: pd.DataFrame, actual_col: str,
              model_order: list, out_path: str):
    hours = np.arange(len(wide_window))
    n     = len(hours)

    # Compact figure size for double-column IEEE format (approx 3.5 inches width)
    fig, ax = plt.subplots(figsize=(4.5, 2.5), facecolor="white")
    ax.set_facecolor("white")

    # Add light grid for y-axis (AQI values)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5, zorder=0)

    handles = []

    # Actual
    style = MODEL_STYLES["Actual"]
    markevery = max(1, n // 14)
    ln, = ax.plot(hours, wide_window[actual_col], label="Actual",
                  color=style["color"], lw=style["lw"], ls=style["ls"],
                  marker=style["marker"], markersize=style["ms"],
                  markevery=markevery, zorder=style["zorder"])
    handles.append(ln)

    # Models
    for model in model_order:
        if model not in wide_window.columns:
            continue
        st = MODEL_STYLES.get(model, dict(color="#aaaaaa", lw=1.0, ls="-", zorder=4, marker=None))
        ln, = ax.plot(hours, wide_window[model], label=model,
                      color=st["color"], lw=st["lw"], ls=st["ls"], zorder=st["zorder"])
        handles.append(ln)

    # Day divider (24h)
    if n >= 24:
        ax.axvline(24, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=2)

    ax.set_xlim(0, n - 1)
    ax.set_ylim(1, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    
    # Tick formatting
    ax.set_xticks(range(0, n, 12))
    ax.set_xticklabels([f"H{h}" for h in range(0, n, 12)], fontsize=8)
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8)
    
    ax.tick_params(axis="both", length=2, pad=2)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')

    # Labels
    ax.set_ylabel("AQI", fontsize=9, labelpad=4)
    ax.set_xlabel("Forecast Hour", fontsize=9, labelpad=4)
    ax.set_title(f"48h Forecast: {city_name}", fontsize=10, pad=6)

    # Compact Legend
    leg = ax.legend(handles=handles, loc="upper right", frameon=True,
                    fontsize=7, ncol=3, handlelength=1.5, columnspacing=1.0,
                    borderpad=0.3, handletextpad=0.3)
    leg.get_frame().set_linewidth(0.5)

    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def interest_score(series: pd.Series, pred_cols: dict, window_df: pd.DataFrame) -> float:
    actual      = series
    variance    = float(actual.var())
    transitions = int((actual.diff().abs() > 0).sum())
    divergences = []
    for col in pred_cols:
        if col in window_df.columns:
            divergences.append(float((actual - window_df[col]).abs().mean()))
    mean_div = float(np.mean(divergences)) if divergences else 0.0
    return variance * 2.0 + transitions * 0.5 + mean_div


def main():
    print("Loading:", PRED_CSV)
    df = pd.read_csv(PRED_CSV)
    print(f"  Shape : {df.shape}")

    actual_col   = "actual_aqi"
    model_col    = "model"
    city_col     = "city_name"
    datetime_col = "datetime"
    pred_col     = "predicted_aqi"

    cities      = sorted(df[city_col].unique())
    model_order = sorted(df[model_col].unique(),
                         key=lambda m: list(MODEL_STYLES.keys()).index(m)
                                       if m in MODEL_STYLES else 99)

    wide = df.pivot_table(
        index=[city_col, datetime_col],
        columns=model_col,
        values=pred_col,
        aggfunc="first"
    ).reset_index()

    actual_df = df[[city_col, datetime_col, actual_col]].drop_duplicates()
    wide = wide.merge(actual_df, on=[city_col, datetime_col], how="left")
    wide[datetime_col] = pd.to_datetime(wide[datetime_col])
    wide = wide.sort_values([city_col, datetime_col]).reset_index(drop=True)

    # Enforce discrete integer constraints [1, 5] on all models
    for model in model_order:
        if model in wide.columns:
            wide[model] = wide[model].round().clip(1, 5)

    scores  = {}
    windows = {}

    for city in cities:
        city_df = wide[wide[city_col] == city].reset_index(drop=True)
        start   = pick_best_48h_window(city_df[actual_col])
        win     = city_df.iloc[start:start + 48].reset_index(drop=True)

        safe    = city.lower().replace(" ", "_").replace("/", "_")
        out     = os.path.join(OUT_DIR, f"{safe}_forecast_48h.png")
        plot_city(city, win, actual_col, model_order, out)

        sc = interest_score(win[actual_col], {m: m for m in model_order}, win)
        scores[city]  = sc
        windows[city] = win

    best_city = max(scores, key=scores.get)
    
    safe_best = best_city.lower().replace(" ", "_").replace("/", "_")
    src = os.path.join(OUT_DIR, f"{safe_best}_forecast_48h.png")
    dst = os.path.join(OUT_DIR, "best_city_forecast_48h.png")
    shutil.copy2(src, dst)
    print(f"\nBest figure -> {dst}")
    print(f"Paper reference city: {best_city}")


if __name__ == "__main__":
    main()
