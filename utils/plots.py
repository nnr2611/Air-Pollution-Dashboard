import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for Flask-safe rendering

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from io import BytesIO
import base64
import numpy as np

def plot_to_base64(fig):
    """Convert a Matplotlib figure to a base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    base64_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return base64_str

def get_all_charts(df, lstm_preds, hybrid_preds):
    """Generate all required charts and metrics for dashboard."""

    # Filter valid predictions (ignore NaNs)
    mask = ~np.isnan(hybrid_preds)
    y_true = df.loc[mask, 'target_7_days_ahead']
    y_pred = hybrid_preds[mask]

    # === 1. Time Series Plot ===
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(df['datetime'], df['target_7_days_ahead'], label='True PM2.5', color='blue', alpha=0.5)
    ax1.plot(df['datetime'], lstm_preds, label='LSTM', color='red', alpha=0.6)
    ax1.plot(df['datetime'], hybrid_preds, label='Hybrid', color='green', alpha=0.6)
    ax1.set_title("PM2.5 Over Time (True vs LSTM vs Hybrid)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("PM2.5 (µg/m³)")
    ax1.legend()
    ax1.grid(True)
    time_series_chart = plot_to_base64(fig1)

    # === 2. Residual Plot ===
    residuals = y_true - y_pred
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    ax2.plot(df['datetime'][mask], residuals, color='purple', alpha=0.6)
    ax2.axhline(0, linestyle='--', color='black')
    ax2.set_title("Residuals (True - Hybrid Prediction)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Error (µg/m³)")
    ax2.grid(True)
    residual_chart = plot_to_base64(fig2)

    # === 3. Scatter Plot ===
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    ax3.scatter(y_true, y_pred, alpha=0.3, color='green')
    ax3.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    ax3.set_title("Actual vs Predicted (Hybrid)")
    ax3.set_xlabel("Actual PM2.5")
    ax3.set_ylabel("Predicted PM2.5")
    ax3.grid(True)
    scatter_chart = plot_to_base64(fig3)

    # === 4. Metrics ===
    rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)), 2)
    mae = round(mean_absolute_error(y_true, y_pred), 2)

    return {
        'time_series_chart': time_series_chart,
        'residual_chart': residual_chart,
        'scatter_chart': scatter_chart,
        'rmse': rmse,
        'mae': mae
    }

def plot_time_series(df, lstm_preds, hybrid_preds):
    """Generate a single base64-encoded PM2.5 time-series plot."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df['datetime'], df['target_7_days_ahead'], label='True PM2.5', color='blue', alpha=0.5)
    ax.plot(df['datetime'], lstm_preds, label='LSTM Prediction', color='red', alpha=0.6)
    ax.plot(df['datetime'], hybrid_preds, label='Hybrid Prediction', color='green', alpha=0.6)
    ax.set_title("PM2.5 Over Time (True vs LSTM vs Hybrid)")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend()
    ax.grid(True)
    return plot_to_base64(fig)