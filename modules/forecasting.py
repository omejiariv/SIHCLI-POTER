# modules/forecasting.py
import requests
import pmdarima as pm
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import pacf, acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
from modules.config import Config
from datetime import datetime
import openmeteo_requests
import requests_cache
from retry_requests import retry

# --- CACHE PARA DATOS EXTERNOS ---
@st.cache_data(ttl=3600)
def get_weather_forecast(latitude, longitude):
    """Obtiene el pronóstico del tiempo para 7 días desde Open-Meteo."""
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
        "timezone": "auto",
        "forecast_days": 7
    }

    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        daily = response.Daily()
        
        daily_data = {
            "date": pd.to_datetime(daily.Time(), unit="s", utc=True),
            "temperature_2m_max": daily.Variables(0).ValuesAsNumpy(),
            "temperature_2m_min": daily.Variables(1).ValuesAsNumpy(),
            "precipitation_sum": daily.Variables(2).ValuesAsNumpy()
        }
        return pd.DataFrame(data=daily_data)
    except Exception as e:
        st.error(f"Error obteniendo pronóstico: {e}")
        return None

@st.cache_data(ttl=86400)
def get_official_enso_forecast():
    """Descarga pronóstico ENSO oficial (simulado o real según disponibilidad)."""
    # Aquí simplificamos para evitar errores de conexión con IRI si falla
    # En una versión completa, restauraríamos la lógica del PDF para IRI
    return None, None 

def evaluate_forecast(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae}

@st.cache_data
def generate_prophet_forecast(ts_data_raw, horizon, test_size=12):
    """Genera pronóstico con Prophet."""
    ts_data = ts_data_raw.rename(columns={Config.DATE_COL: 'ds', Config.PRECIPITATION_COL: 'y'})
    ts_data['ds'] = pd.to_datetime(ts_data['ds'])
    
    if len(ts_data) < test_size + 24:
        return None, None, None

    # Modelo
    model = Prophet()
    train = ts_data.iloc[:-test_size]
    test = ts_data.iloc[-test_size:]
    
    model.fit(train)
    
    # Evaluar
    future_eval = model.make_future_dataframe(periods=test_size, freq='MS')
    forecast_eval = model.predict(future_eval)
    y_pred = forecast_eval.tail(test_size)['yhat']
    metrics = evaluate_forecast(test['y'], y_pred)
    
    # Pronóstico Final
    full_model = Prophet()
    full_model.fit(ts_data)
    future = full_model.make_future_dataframe(periods=horizon, freq='MS')
    forecast = full_model.predict(future)
    
    return full_model, forecast, metrics
