# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

# ---------------------------
# Config Paths
# ---------------------------
DATA_PATH = "data/combined_data_cleaned.csv"
LSTM_MODEL_PATH = "models/best_lstm_model.keras"
XGB_MODEL_PATH = "models/best_xgb_model.json"
SCALER_PATH = "scaler.pkl"

# ---------------------------
# Load Data, Models, Scaler
# ---------------------------
@st.cache_resource
def load_artifacts():
    df = pd.read_csv(DATA_PATH)
    lstm_model = load_model(LSTM_MODEL_PATH)
    xgb_model = xgb.Booster()
    xgb_model.load_model(XGB_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return df, lstm_model, xgb_model, scaler

df, lstm_model, xgb_model, scaler = load_artifacts()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🌫️ PM2.5 Air Quality Forecasting (Beijing)")

st.sidebar.header("🔧 Options")
station_names = df['station'].unique().tolist()
selected_station = st.sidebar.selectbox("Select Station", station_names)

model_choice = st.sidebar.selectbox("Model", ["LSTM", "XGBoost"])

forecast_hours = st.sidebar.slider("Forecast Horizon (hours)", 24, 168, 72, step=24)

st.markdown(f"### Showing data for station: `{selected_station}`")

# Filter station-specific data
station_df = df[df['station'] == selected_station]

st.dataframe(station_df.tail(5))

# Placeholder for predictions
st.markdown("📈 Forecast plot will appear here after prediction code is added.")

