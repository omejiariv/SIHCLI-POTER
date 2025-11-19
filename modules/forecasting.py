# modules/forecasting.py
import streamlit as st
import pandas as pd
import numpy as np
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import pacf, acf
from prophet import Prophet
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
from modules.config import Config

# --- Funciones Auxiliares ---

def evaluate_forecast(y_true, y_pred):
    """Calcula RMSE y MAE para evaluar un pronóstico."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae}

@st.cache_data(show_spinner=False)
def get_decomposition_results(series, period=12, model='additive'):
    """Realiza la descomposición de la serie de tiempo."""
    # Interpolación para asegurar continuidad
    series_clean = series.asfreq('MS').interpolate(method='time').dropna()
    if len(series_clean) < 2 * period:
        raise ValueError("Serie demasiado corta o con demasiados nulos para la descomposición.")
    return seasonal_decompose(series_clean, model=model, period=period)

@st.cache_data(show_spinner=False)
def auto_arima_search(ts_data, test_size):
    """Encuentra los parámetros óptimos para un modelo SARIMA usando auto_arima."""
    ts_data_copy = ts_data.copy()
    if not pd.api.types.is_datetime64_any_dtype(ts_data_copy.index):
        ts_data_copy = ts_data_copy.set_index(Config.DATE_COL).sort_index()
    
    ts = ts_data_copy[Config.PRECIPITATION_COL].asfreq('MS').interpolate(method='time').dropna()
    train = ts[:-test_size]
    
    auto_model = pm.auto_arima(
        train,
        start_p=1, start_q=1,
        test='adf',
        max_p=3, max_q=3,
        m=12,
        start_P=0, seasonal=True,
        d=None, D=None,
        trace=False,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )
    return auto_model.order, auto_model.seasonal_order

# --- Funciones Principales de Pronóstico ---

@st.cache_data
def generate_sarima_forecast(ts_data_raw, order, seasonal_order, horizon, test_size=12, regressors=None):
    """
    Entrena, evalúa y genera un pronóstico con SARIMAX, incluyendo regresores opcionales.
    """
    exog, exog_train, exog_test, exog_future = None, None, None, None
    
    # 1. Preparar datos de precipitación (ts)
    ts_data_precip = ts_data_raw[[Config.DATE_COL, Config.PRECIPITATION_COL]].copy()
    ts_data_precip = ts_data_precip.drop_duplicates(subset=[Config.DATE_COL], keep='first')
    ts_data_precip = ts_data_precip.set_index(Config.DATE_COL).sort_index()
    
    # Asegurar frecuencia mensual e interpolar
    ts = ts_data_precip[Config.PRECIPITATION_COL].asfreq('MS').interpolate(method='time').dropna()

    # 2. Preparar regresores (si existen)
    if regressors is not None and not regressors.empty:
        # 'regressors' ya debe tener el formato correcto de fecha en Config.DATE_COL
        exog_full = regressors.set_index(Config.DATE_COL).sort_index()
        
        # Alinear el índice de los regresores con el de la precipitación (histórico)
        exog = exog_full.reindex(ts.index)
        exog = exog.interpolate(method='linear', limit_direction='both')
        
        # Preparar regresores futuros (para el pronóstico)
        future_index = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1), periods=horizon, freq='MS')
        exog_future = exog_full.reindex(future_index)
        exog_future = exog_future.interpolate(method='linear', limit_direction='both')
        
        # Validar que no queden NaNs en exog_future, si quedan, usar relleno hacia adelante
        if exog_future.isnull().values.any():
             exog_future = exog_future.fillna(method='ffill').fillna(method='bfill')

    if len(ts) < test_size + 24:
        raise ValueError(f"Se necesitan al menos {test_size + 24} meses de datos para el pronóstico.")

    # 3. Dividir datos para entrenamiento y prueba
    train, test = ts[:-test_size], ts[-test_size:]
    
    if exog is not None:
        exog_train, exog_test = exog.iloc[:-test_size], exog.iloc[-test_size:]

    # 4. Entrenar y Evaluar (Modelo Split)
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order, exog=exog_train,
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    
    # Predicción sobre el conjunto de prueba
    pred_test = results.get_forecast(steps=test_size, exog=exog_test)
    y_pred_test = pred_test.predicted_mean
    metrics = evaluate_forecast(test, y_pred_test)

    # 5. Entrenar modelo completo y Pronosticar (Modelo Full)
    full_model = SARIMAX(ts, order=order, seasonal_order=seasonal_order, exog=exog,
                         enforce_stationarity=False, enforce_invertibility=False)
    full_results = full_model.fit(disp=False)
    
    # Pronóstico a futuro
    forecast = full_results.get_forecast(steps=horizon, exog=exog_future)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    
    # Preparar DataFrame para exportación
    sarima_df_export = forecast_mean.reset_index().rename(columns={'index': 'ds', 'predicted_mean': 'yhat'})
    
    return ts, forecast_mean, forecast_ci, metrics, sarima_df_export

@st.cache_data
def generate_prophet_forecast(ts_data_raw, horizon, test_size=12, regressors=None):
    """Entrena, evalúa y genera un pronóstico con Prophet."""
    # 1. Preparar datos
    ts_data = ts_data_raw.rename(columns={Config.DATE_COL: 'ds', Config.PRECIPITATION_COL: 'y'})
    ts_data = ts_data.drop_duplicates(subset=['ds'], keep='first')
    ts_data['ds'] = pd.to_datetime(ts_data['ds'])
    ts_data = ts_data.set_index('ds').sort_index()
    ts_data['y'] = ts_data['y'].interpolate(method='time')
    ts_data = ts_data.reset_index()

    if len(ts_data) < test_size + 24:
        raise ValueError(f"Se necesitan al menos {test_size + 24} meses de datos para Prophet.")

    # 2. Configurar Regresores
    regressor_cols = []
    # Modelo para evaluación
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)

    if regressors is not None and not regressors.empty:
        regressor_names = [col for col in regressors.columns if col != 'ds']
        
        # Limpiar ts_data de columnas antiguas si existen para evitar duplicados
        cols_to_drop = [c for c in regressor_names if c in ts_data.columns]
        if cols_to_drop:
            ts_data = ts_data.drop(columns=cols_to_drop)
            
        # Unir regresores
        ts_data = pd.merge(ts_data, regressors, on='ds', how='left')
        
        for col in regressor_names:
            # Interpolar faltantes
            ts_data[col] = ts_data[col].interpolate(method='linear', limit_direction='both')
            model.add_regressor(col)
            regressor_cols.append(col)

    # 3. Entrenar y Evaluar
    train = ts_data.iloc[:-test_size]
    test = ts_data.iloc[-test_size:]
    
    model.fit(train)
    
    # Crear dataframe futuro para prueba
    test_dates = model.make_future_dataframe(periods=test_size, freq='MS').tail(test_size)
    
    # Añadir regresores al futuro de prueba
    if regressor_cols:
        test_regressors = ts_data[ts_data['ds'].isin(test_dates['ds'])][['ds'] + regressor_cols]
        test_dates = pd.merge(test_dates, test_regressors, on='ds', how='left')
        # Relleno de seguridad
        for col in regressor_cols:
             test_dates[col] = test_dates[col].interpolate(method='linear', limit_direction='both').fillna(method='ffill')

    y_pred_test = model.predict(test_dates)['yhat']
    metrics = evaluate_forecast(test['y'], y_pred_test)

    # 4. Modelo Final y Pronóstico Futuro
    full_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    for col in regressor_cols:
        full_model.add_regressor(col)
        
    full_model.fit(ts_data)
    
    future = full_model.make_future_dataframe(periods=horizon, freq='MS')
    
    if regressor_cols:
        # Unir regresores futuros (deben venir en el argumento 'regressors' con fechas futuras)
        future = pd.merge(future, regressors, on='ds', how='left')
        for col in regressor_cols:
            future[col] = future[col].interpolate(method='linear', limit_direction='both').fillna(method='ffill')

    forecast = full_model.predict(future)
    
    return full_model, forecast, metrics

# --- Gráficos de Autocorrelación (Plotly) ---

def create_acf_chart(series, max_lag):
    if len(series) <= max_lag:
        return go.Figure().update_layout(title="Datos insuficientes para ACF")
    
    acf_values = acf(series, nlags=max_lag)
    lags = list(range(max_lag + 1))
    conf_interval = 1.96 / np.sqrt(len(series))
    
    fig = go.Figure([
        go.Bar(x=lags, y=acf_values, name='ACF'),
        go.Scatter(x=lags, y=[conf_interval]*len(lags), mode='lines', line=dict(color='blue', dash='dash'), name='Límite Superior'),
        go.Scatter(x=lags, y=[-conf_interval]*len(lags), mode='lines', line=dict(color='blue', dash='dash'), showlegend=False)
    ])
    fig.update_layout(title='Función de Autocorrelación (ACF)', height=400)
    return fig

def create_pacf_chart(series, max_lag):
    if len(series) <= max_lag:
        return go.Figure().update_layout(title="Datos insuficientes para PACF")
        
    pacf_values = pacf(series, nlags=max_lag)
    lags = list(range(max_lag + 1))
    conf_interval = 1.96 / np.sqrt(len(series))
    
    fig = go.Figure([
        go.Bar(x=lags, y=pacf_values, name='PACF'),
        go.Scatter(x=lags, y=[conf_interval]*len(lags), mode='lines', line=dict(color='red', dash='dash'), name='Límite Superior'),
        go.Scatter(x=lags, y=[-conf_interval]*len(lags), mode='lines', line=dict(color='red', dash='dash'), showlegend=False)
    ])
    fig.update_layout(title='Autocorrelación Parcial (PACF)', height=400)
    return fig

# Se importa get_weather_forecast de un módulo separado o se define aquí si es necesario.
# Asumimos que visualizer.py lo necesita, así que lo incluimos por compatibilidad si no está en otro lado.
# En tu estructura original parecía estar en forecasting.py.

import openmeteo_requests
import requests_cache
from retry_requests import retry

@st.cache_data(ttl=3600)
def get_weather_forecast(latitude, longitude):
    """Obtiene el pronóstico del tiempo a 7 días desde Open-Meteo."""
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", 
                  "relative_humidity_2m_mean", "surface_pressure_mean", 
                  "et0_fao_evapotranspiration", "shortwave_radiation_sum", "wind_speed_10m_max"],
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
            "precipitation_sum": daily.Variables(2).ValuesAsNumpy(),
            "relative_humidity_2m_mean": daily.Variables(3).ValuesAsNumpy(),
            "surface_pressure_mean": daily.Variables(4).ValuesAsNumpy(),
            "et0_fao_evapotranspiration": daily.Variables(5).ValuesAsNumpy(),
            "shortwave_radiation_sum": daily.Variables(6).ValuesAsNumpy(),
            "wind_speed_10m_max": daily.Variables(7).ValuesAsNumpy()
        }
        
        return pd.DataFrame(data=daily_data).head(7)
    except Exception as e:
        st.error(f"Error Open-Meteo: {e}")
        return None
