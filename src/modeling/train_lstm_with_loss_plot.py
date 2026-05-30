import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

DATA_PATH = Path("data/selected/2026/2026_SelectedData_Imputed.csv")
OUTPUT_DIR = Path("figs/training_curves")
TARGET_COL = "main.aqi"
CITY_COL = "city_name"
DATETIME_COL = "datetime"
EXOGENOUS_COLS = [
    "components.pm2_5",
    "components.pm10",
    "components.o3",
    "components.co",
    "components.no2",
    "components.so2",
    "components.nh3",
]
WINDOW = 24

def main():
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found.")
        return

    data = pd.read_csv(DATA_PATH)
    data[DATETIME_COL] = pd.to_datetime(data[DATETIME_COL])
    
    # Loss curve is shown for one city, while the production pipeline trains all cities.
    city_name = "Manila"
    city_df = data[data[CITY_COL] == city_name].sort_values(DATETIME_COL)
    model_cols = [TARGET_COL, *EXOGENOUS_COLS]
    series = city_df[model_cols].astype(float).ffill().bfill().values
    
    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series)
    
    X, y = [], []
    for i in range(WINDOW, len(scaled_series)):
        X.append(scaled_series[i-WINDOW:i, :])
        y.append(scaled_series[i, 0])
    X, y = np.array(X), np.array(y)
    
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(WINDOW, len(model_cols))),
        tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32, activation='tanh'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    print(f"Training LSTM for {city_name}...")
    history = model.fit(
        X_train,
        y_train,
        epochs=25,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1,
        shuffle=False,
    )
    
    # Plot loss optimized for IEEE
    plt.figure(figsize=(5, 4))
    plt.rcParams.update({'font.size': 13})
    plt.plot(history.history['loss'], label='Train', linewidth=2.5, color='blue')
    plt.plot(history.history['val_loss'], label='Val', linestyle='--', linewidth=2.5, color='red')
    plt.title(f'LSTM Loss Convergence', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.legend(loc='upper right', frameon=True, fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / "lstm_loss_curve.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved LSTM loss curve to: {save_path}")

if __name__ == "__main__":
    main()
