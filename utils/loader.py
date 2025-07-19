import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from tensorflow.keras.models import load_model
import os
import requests

# -------------------------------
# Google Drive URLs (replace with your actual IDs)
# -------------------------------
DATA_URL = "https://drive.google.com/uc?export=download&id=1WE8iA7NKVY5O2qckF0yQrpjFGSttYNhq"
LSTM_PRED_URL = "https://drive.google.com/uc?export=download&id=1nmy3iQ3vLlT1kFVmrvlHmHxGNxSYkFNS"
XGB_PRED_URL = "https://drive.google.com/uc?export=download&id=1u9Zlz0LCsD9vwzq6ki5l7mmdjqxaY_cz"
LSTM_MODEL_URL = "https://drive.google.com/uc?export=download&id=1TRunbkjnP3QZX3EmQoJbRIUOmfPdg2o6"
XGB_MODEL_URL = "https://drive.google.com/uc?export=download&id=1bLkiKVcYl5Ob0OSKWFwR3N8irDFYsHfm"


# -------------------------------
# Local Paths
# -------------------------------
DATA_PATH = "data/combined_data_cleaned.csv"
LSTM_PRED_PATH = "data/lstm_preds_full.npy"
XGB_PRED_PATH = "data/xgb_preds_full.npy"
LSTM_MODEL_PATH = "models/best_lstm_model.keras"
XGB_MODEL_PATH = "models/best_xgb_model.json"


# -------------------------------
# File Downloader
# -------------------------------
def download_file_from_url(url, local_path):
    """Download a file from a URL if it doesn't exist locally."""
    if not os.path.exists(local_path):
        print(f"🔽 Downloading {os.path.basename(local_path)}...")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        response = requests.get(url)
        with open(local_path, 'wb') as f:
            f.write(response.content)
        print(f"✅ Downloaded {os.path.basename(local_path)}")
    else:
        print(f"📄 {os.path.basename(local_path)} already exists locally.")

# -------------------------------
# Loaders
# -------------------------------
def load_data():
    download_file_from_url(DATA_URL, DATA_PATH)
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    return df

def load_predictions():
    download_file_from_url(LSTM_PRED_URL, LSTM_PRED_PATH)
    download_file_from_url(XGB_PRED_URL, XGB_PRED_PATH)
    lstm_preds = np.load(LSTM_PRED_PATH)
    xgb_preds = np.load(XGB_PRED_PATH)
    return lstm_preds, xgb_preds

def load_models():
    download_file_from_url(LSTM_MODEL_URL, LSTM_MODEL_PATH)
    download_file_from_url(XGB_MODEL_URL, XGB_MODEL_PATH)
    lstm_model = load_model(LSTM_MODEL_PATH)
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(XGB_MODEL_PATH)
    return lstm_model, xgb_model


def get_latest_prediction_info(df, hybrid_preds):
    latest_idx = np.where(~np.isnan(hybrid_preds))[0][-1]
    latest_date = df['datetime'].iloc[latest_idx]
    latest_site = df['site'].iloc[latest_idx]
    latest_prediction = round(hybrid_preds[latest_idx], 2)
    return {
        "latest_date": latest_date.strftime('%Y-%m-%d'),
        "latest_site": latest_site,
        "latest_prediction": latest_prediction
    }
