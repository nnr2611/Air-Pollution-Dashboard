import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from tensorflow.keras.models import load_model
import os
import requests

# -------------------------------
# Configuration
# -------------------------------
DATA_URL = "https://drive.google.com/uc?export=download&id=1WE8iA7NKVY5O2qckF0yQrpjFGSttYNhq"  # 🔁 Replace this with your actual URL
DATA_PATH = "data/combined_data_cleaned.csv"
LSTM_PRED_PATH = "data/lstm_preds_full.npy"
XGB_PRED_PATH = "data/xgb_preds_full.npy"
LSTM_MODEL_PATH = "models/best_lstm_model.keras"
XGB_MODEL_PATH = "models/best_xgb_model.json"
SCALER_PATH = "scaler.pkl"

# -------------------------------
# Helpers
# -------------------------------
def download_file_from_url(url, local_path):
    """Download a file from the given URL if it doesn't exist locally."""
    if not os.path.exists(local_path):
        print(f"🔽 Downloading {os.path.basename(local_path)}...")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        response = requests.get(url)
        with open(local_path, 'wb') as f:
            f.write(response.content)
        print("✅ Download complete.")
    else:
        print(f"📄 {os.path.basename(local_path)} already exists locally.")

# -------------------------------
# Loaders
# -------------------------------
def load_data():
    """Load the cleaned air quality dataset (downloads if missing)."""
    download_file_from_url(DATA_URL, DATA_PATH)
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    return df

def load_predictions():
    """Load LSTM and Hybrid model predictions."""
    lstm_preds = np.load(LSTM_PRED_PATH)
    xgb_preds = np.load(XGB_PRED_PATH)
    return lstm_preds, xgb_preds

def load_models():
    """Load the trained LSTM and XGBoost models."""
    lstm_model = load_model(LSTM_MODEL_PATH)
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(XGB_MODEL_PATH)
    return lstm_model, xgb_model

def load_scaler():
    """Load the saved StandardScaler used for XGBoost input features."""
    return joblib.load(SCALER_PATH)

def get_latest_prediction_info(df, hybrid_preds):
    """Get the most recent prediction and corresponding site/date."""
    latest_idx = np.where(~np.isnan(hybrid_preds))[0][-1]
    latest_date = df['datetime'].iloc[latest_idx]
    latest_site = df['site'].iloc[latest_idx]
    latest_prediction = round(hybrid_preds[latest_idx], 2)

    return {
        "latest_date": latest_date.strftime('%Y-%m-%d'),
        "latest_site": latest_site,
        "latest_prediction": latest_prediction
    }
