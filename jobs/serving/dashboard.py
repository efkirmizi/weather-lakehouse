import os
import requests
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Weather AI Lakehouse", layout="wide")
st.title("🌤️ Weather Forecasting Lakehouse")
st.markdown("Live multi-model inference tracking and residual error evaluation.")

# API URL (Uses Docker networking or falls back to localhost)
API_URL = os.getenv("API_URL", "http://localhost:8000")

# --- CHART 1: LIVE FORECASTS ---
st.subheader("Latest 24-Hour Forecast Comparison")
try:
    forecast_res = requests.get(f"{API_URL}/api/v1/forecast/latest", timeout=30)
    forecast_res.raise_for_status()
    forecast_data = forecast_res.json()

    if "status" not in forecast_data:
        df_forecast = pd.DataFrame(forecast_data)
        df_forecast['forecast_timestamp'] = pd.to_datetime(df_forecast['forecast_timestamp'])
        
        fig_line = px.line(
            df_forecast, 
            x="forecast_timestamp", 
            y=[c for c in df_forecast.columns if c != "forecast_timestamp"],
            labels={"value": "Temperature (°C)", "variable": "Model Architecture", "forecast_timestamp": "Time"},
            template="plotly_white"
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("No forecast data available in the Lakehouse yet.")
except Exception as e:
    st.error(f"Failed to fetch forecast data from API: {e}")

st.divider()

# --- CHART 2: ERROR RESIDUALS ---
st.subheader("Model Performance Residuals")
try:
    metrics_res = requests.get(f"{API_URL}/api/v1/metrics/residuals", timeout=30)
    metrics_res.raise_for_status()
    metrics_data = metrics_res.json()

    if "status" not in metrics_data:
        df_metrics = pd.DataFrame(metrics_data)
        df_metrics["model"] = df_metrics["model_name"] + " v" + df_metrics["model_version"].astype(str)
        col1, col2 = st.columns(2)

        for column, metric, title in ((col1, "mae", "Mean Absolute Error (MAE)"),
                                      (col2, "rmse", "Root Mean Square Error (RMSE)")):
            with column:
                fig = px.line(
                    df_metrics.sort_values("horizon"),
                    x="horizon", y=metric, color="model", markers=True,
                    labels={"horizon": "Forecast horizon (hours ahead)", metric: "°C"},
                    title=f"{title} by horizon", template="plotly_white",
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Waiting for actual weather data to overlap with predictions to calculate errors.")
except Exception as e:
    st.error(f"Failed to fetch metrics data from API: {e}")