"""
Streamlit Dashboard for Time Series Forecasting
Interactive web application for forecast visualization and monitoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="ProbForecast Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }

    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }

    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'api_url' not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"


class APIClient:
    """Client for interacting with the FastAPI service"""

    def __init__(self, base_url: str):
        self.base_url = base_url

    def health_check(self) -> dict:
        """Check API health"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.json() if response.status_code == 200 else {}
        except:
            return {}

    def forecast(self, request_data: dict) -> dict:
        """Generate forecast"""
        try:
            response = requests.post(
                f"{self.base_url}/forecast",
                json=request_data,
                timeout=30
            )
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            st.error(f"Forecast request failed: {str(e)}")
            return {}


def main():
    """Main dashboard application"""

    # Header
    st.markdown('<h1 class="main-header">🔮 ProbForecast Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Initialize API client
    api_client = APIClient(st.session_state.api_url)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # API Configuration
        st.subheader("API Configuration")
        api_url = st.text_input(
            "API URL",
            value=st.session_state.api_url,
            help="URL of the FastAPI service"
        )
        st.session_state.api_url = api_url
        api_client.base_url = api_url

        # Health Check
        st.subheader("🏥 System Health")
        if st.button("Check Health"):
            with st.spinner("Checking API health..."):
                health = api_client.health_check()
                if health:
                    st.success("✅ API is healthy")
                    st.json(health)
                else:
                    st.error("❌ API is not responding")

        # Model Selection
        st.subheader("🤖 Model Configuration")
        model_type = st.selectbox(
            "Model Type",
            ["deepar", "tft", "nbeats"],
            help="Select the forecasting model"
        )

        model_version = st.selectbox(
            "Model Version",
            ["latest", "1", "2", "3"],
            help="Select model version"
        )

        # Forecast Parameters
        st.subheader("📊 Forecast Parameters")
        horizon = st.slider(
            "Forecast Horizon",
            min_value=1,
            max_value=365,
            value=30,
            help="Number of time steps to forecast"
        )

        quantiles = st.multiselect(
            "Prediction Intervals",
            [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
            default=[0.1, 0.5, 0.9],
            help="Quantiles for uncertainty estimation"
        )

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📈 Generate Forecast")

        # Series selection
        series_id = st.text_input(
            "Time Series ID",
            value="store_001",
            help="Enter the ID of the time series to forecast"
        )

        # Forecast button
        if st.button("🔮 Generate Forecast", type="primary"):
            if series_id:
                with st.spinner("Generating forecast..."):
                    request_data = {
                        "series_id": series_id,
                        "horizon": horizon,
                        "model_type": model_type,
                        "model_version": model_version,
                        "quantiles": quantiles,
                        "include_history": False
                    }

                    forecast_result = api_client.forecast(request_data)

                    if forecast_result:
                        st.session_state.forecast_data = forecast_result
                        st.success("✅ Forecast generated successfully!")
                    else:
                        st.error("❌ Failed to generate forecast")
            else:
                st.warning("Please enter a time series ID")

    with col2:
        st.header("ℹ️ Model Information")
        st.info("Model information will be displayed here")

    # Forecast visualization
    if st.session_state.forecast_data:
        st.header("📊 Forecast Visualization")

        # Parse forecast data
        forecast_data = st.session_state.forecast_data
        forecast_points = forecast_data['forecast']

        # Create DataFrame for plotting
        timestamps = [point['timestamp'] for point in forecast_points]
        means = [point['mean'] for point in forecast_points]

        df_forecast = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps),
            'mean': means
        })

        # Add quantiles to DataFrame
        for point in forecast_points:
            for q_name, q_value in point.get('quantiles', {}).items():
                if q_name not in df_forecast.columns:
                    df_forecast[q_name] = None
                df_forecast.loc[df_forecast['timestamp'] == pd.to_datetime(point['timestamp']), q_name] = q_value

        # Create plot
        fig = go.Figure()

        # Add mean forecast
        fig.add_trace(
            go.Scatter(
                x=df_forecast['timestamp'],
                y=df_forecast['mean'],
                mode='lines+markers',
                name='Forecast (Mean)',
                line=dict(color='blue', width=2)
            )
        )

        # Add confidence intervals
        quantile_cols = [col for col in df_forecast.columns if col.startswith('p')]
        quantile_cols.sort()

        if len(quantile_cols) >= 2:
            # Add confidence bands
            lower_q = quantile_cols[0]
            upper_q = quantile_cols[-1]

            fig.add_trace(
                go.Scatter(
                    x=df_forecast['timestamp'],
                    y=df_forecast[upper_q],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=df_forecast['timestamp'],
                    y=df_forecast[lower_q],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(68, 68, 68, 0.2)',
                    name=f'Confidence Interval ({lower_q}-{upper_q})',
                    hoverinfo='skip'
                )
            )

        # Update layout
        fig.update_layout(
            title=f"Forecast for {forecast_data['series_id']}",
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified',
            height=500,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

        # Forecast summary
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Series ID",
                forecast_data['series_id']
            )

        with col2:
            st.metric(
                "Horizon",
                f"{forecast_data['forecast_horizon']} steps"
            )

        with col3:
            st.metric(
                "Model",
                f"{forecast_data['model_type'].upper()}"
            )

        with col4:
            st.metric(
                "Version",
                forecast_data['model_version']
            )

        # Detailed forecast table
        with st.expander("📋 Detailed Forecast Data"):
            st.dataframe(df_forecast, use_container_width=True)

        # Download forecast
        csv = df_forecast.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name=f"forecast_{forecast_data['series_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "ProbForecast Dashboard v1.0.0 | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
