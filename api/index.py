from flask import Flask, render_template
from utils.loader import load_data, load_predictions, get_latest_prediction_info
from utils.plots import get_all_charts, plot_time_series  # <-- FIXED

app = Flask(__name__)

@app.route("/")
def index():
    df = load_data()
    lstm_preds, hybrid_preds = load_predictions()
    info = get_latest_prediction_info(df, hybrid_preds)
    img = plot_time_series(df, lstm_preds, hybrid_preds)  # returns base64 string

    return render_template("index.html",
        chart=img,
        latest_prediction=info["latest_prediction"],
        latest_site=info["latest_site"],
        latest_date=info["latest_date"]
    )

@app.route("/charts")
def charts():
    # Load data and predictions
    df = load_data()
    lstm_preds, hybrid_preds = load_predictions()

    # Get metrics and latest forecast details
    info = get_latest_prediction_info(df, hybrid_preds)

    # Generate base64 charts
    charts = get_all_charts(df, lstm_preds, hybrid_preds)

    # Render template with all data
    return render_template(
        "charts.html",
        latest_prediction=info['latest_prediction'],
        latest_site=info['latest_site'],
        latest_date=info['latest_date'],
        rmse=charts['rmse'],
        mae=charts['mae'],
        time_series_chart=charts['time_series_chart'],
        residual_chart=charts['residual_chart'],
        scatter_chart=charts['scatter_chart']
    )

if __name__ == "__main__":
    app.run()

