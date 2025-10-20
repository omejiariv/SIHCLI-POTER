import pmdarima as pm
import streamlit as st
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import pacf, acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from modules.config import Config

# modules/forecast_api.py

import streamlit as st
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

@st.cache_data
def get_weather_forecast(latitude, longitude):
    """
    Obtiene el pronóstico del tiempo. Maneja elegantemente el error de límite de la API.
    """
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude, "longitude": longitude,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
        "timezone": "auto"
    }
    
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        
        daily = response.Daily()
        timezone_str = response.Timezone().decode('utf-8')
        daily_data = {
            "date": pd.to_datetime(daily.Time(), unit="s", utc=True).tz_convert(timezone_str),
            "temperature_2m_max": daily.Variables(0).ValuesAsNumpy(),
            "temperature_2m_min": daily.Variables(1).ValuesAsNumpy(),
            "precipitation_sum": daily.Variables(2).ValuesAsNumpy(),
        }
        return pd.DataFrame(data=daily_data)
        
    except OpenMeteoRequestsError as e:
        # Si la API nos dice que hemos excedido el límite, lo mostramos al usuario
        st.error(f"Error de la API de Pronóstico: Límite diario de solicitudes excedido. Por favor, intente de nuevo mañana. (Detalle: {e})")
        return None
    except Exception as e:
        # Para cualquier otro error inesperado
        st.error(f"Ocurrió un error inesperado al obtener el pronóstico: {e}")
        return None
    
@st.cache_data(show_spinner=False)
def get_decomposition_results(series, period=12, model='additive'):
    """Realiza la descomposición de la serie de tiempo."""
    series_clean = series.asfreq('MS').interpolate(method='time').dropna()
    if len(series_clean) < 2 * period:
        raise ValueError("Serie demasiado corta o con demasiados nulos para la descomposición.")
    return seasonal_decompose(series_clean, model=model, period=period)

def create_acf_chart(series, max_lag):
    """Genera el gráfico de la Función de Autocorrelación (ACF) usando Plotly."""
    if len(series) <= max_lag:
        return go.Figure().update_layout(title="Datos insuficientes para ACF")

    acf_values = acf(series, nlags=max_lag)
    lags = list(range(max_lag + 1))
    conf_interval = 1.96 / np.sqrt(len(series))
    
    fig_acf = go.Figure(data=[
        go.Bar(x=lags, y=acf_values, name='ACF'),
        go.Scatter(x=lags, y=[conf_interval] * (max_lag + 1), mode='lines', line=dict(color='blue', dash='dash'), name='Límite de Confianza'),
        go.Scatter(x=lags, y=[-conf_interval] * (max_lag + 1), mode='lines', line=dict(color='blue', dash='dash'), fill='tonexty', fillcolor='rgba(0,0,255,0.1)', showlegend=False)
    ])
    fig_acf.update_layout(title='Función de Autocorrelación (ACF)', xaxis_title='Rezagos (Meses)', yaxis_title='Correlación', height=400)
    return fig_acf

def create_pacf_chart(series, max_lag):
    """Genera el gráfico de la Función de Autocorrelación Parcial (PACF) usando Plotly."""
    if len(series) <= max_lag:
        return go.Figure().update_layout(title="Datos insuficientes para PACF")

    pacf_values = pacf(series, nlags=max_lag)
    lags = list(range(max_lag + 1))
    conf_interval = 1.96 / np.sqrt(len(series))

    fig_pacf = go.Figure(data=[
        go.Bar(x=lags, y=pacf_values, name='PACF'),
        go.Scatter(x=lags, y=[conf_interval] * (max_lag + 1), mode='lines', line=dict(color='red', dash='dash'), name='Límite de Confianza'),
        go.Scatter(x=lags, y=[-conf_interval] * (max_lag + 1), mode='lines', line=dict(color='red', dash='dash'), fill='tonexty', fillcolor='rgba(255,0,0,0.1)', showlegend=False)
    ])
    fig_pacf.update_layout(title='Función de Autocorrelación Parcial (PACF)', xaxis_title='Rezagos (Meses)', yaxis_title='Correlación', height=400)
    return fig_pacf

def evaluate_forecast(y_true, y_pred):
    """Calcula RMSE y MAE para evaluar un pronóstico."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae}

# --- INICIO DE LA CORRECCIÓN ---
def generate_sarima_forecast(ts_data_raw, order, seasonal_order, horizon, test_size=12, regressors=None):
    """Entrena, evalúa y genera un pronóstico con SARIMAX, incluyendo regresores opcionales."""
    ts_data = ts_data_raw[[Config.DATE_COL, Config.PRECIPITATION_COL]].copy()
    ts_data = ts_data.set_index(Config.DATE_COL).sort_index()
    ts = ts_data[Config.PRECIPITATION_COL].asfreq('MS').interpolate(method='time').dropna()

    if len(ts) < test_size + 24:
        raise ValueError(f"Se necesitan al menos {test_size + 24} meses de datos para el pronóstico y la evaluación.")

    exog, exog_train, exog_test, exog_future = None, None, None, None
    if regressors is not None and not regressors.empty:
        exog = regressors.set_index(Config.DATE_COL)
        exog = exog.reindex(ts.index).interpolate()

    train, test = ts[:-test_size], ts[-test_size:]
    if exog is not None:
        exog_train, exog_test = exog.iloc[:-test_size], exog.iloc[-test_size:]

    model = SARIMAX(train, order=order, seasonal_order=seasonal_order, exog=exog_train, enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    pred_test = results.get_forecast(steps=test_size, exog=exog_test)
    y_pred_test = pred_test.predicted_mean
    metrics = evaluate_forecast(test, y_pred_test)

    if exog is not None:
        last_regressor_values = exog.iloc[-1:].values
        future_index = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1), periods=horizon, freq='MS')
        exog_future = pd.DataFrame(np.tile(last_regressor_values, (horizon, 1)), index=future_index, columns=exog.columns)

    full_model = SARIMAX(ts, order=order, seasonal_order=seasonal_order, exog=exog, enforce_stationarity=False, enforce_invertibility=False)
    full_results = full_model.fit(disp=False)
    forecast = full_results.get_forecast(steps=horizon, exog=exog_future)

    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    
    # La variable 'sarima_df_export' ahora siempre se define
    sarima_df_export = forecast_mean.reset_index().rename(columns={'index': 'ds', 'predicted_mean': 'yhat'})

    return ts, forecast_mean, forecast_ci, metrics, sarima_df_export
# --- FIN DE LA CORRECCIÓN ---

def generate_prophet_forecast(ts_data_raw, horizon, test_size=12, regressors=None):
    """Entrena, evalúa y genera un pronóstico con Prophet, incluyendo regresores opcionales."""
    ts_data = ts_data_raw.rename(columns={Config.DATE_COL: 'ds', Config.PRECIPITATION_COL: 'y'})
    ts_data['y'] = ts_data['y'].interpolate()

    if len(ts_data) < test_size + 24:
        raise ValueError(f"Se necesitan al menos {test_size + 24} meses de datos para Prophet.")

    if regressors is not None and not regressors.empty:
        ts_data = pd.merge(ts_data, regressors.rename(columns={Config.DATE_COL: 'ds'}), on='ds', how='left')
        for col in regressors.columns:
            if col != Config.DATE_COL:
                ts_data[col] = ts_data[col].interpolate()

    train, test = ts_data.iloc[:-test_size], ts_data.iloc[-test_size:]

    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    if regressors is not None:
        for col in regressors.columns:
            if col != Config.DATE_COL:
                model.add_regressor(col)
    model.fit(train)

    test_dates = model.make_future_dataframe(periods=test_size, freq='MS').tail(test_size)
    if regressors is not None:
        test_regressors = ts_data[ts_data['ds'].isin(test_dates['ds'])]
        test_dates = pd.merge(test_dates, test_regressors, on='ds', how='left')
        test_dates.interpolate(inplace=True)
    
    y_pred_test = model.predict(test_dates)['yhat']
    metrics = evaluate_forecast(test['y'], y_pred_test)

    full_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    if regressors is not None:
        for col in regressors.columns:
            if col != Config.DATE_COL:
                full_model.add_regressor(col)
    full_model.fit(ts_data)

    future = full_model.make_future_dataframe(periods=horizon, freq='MS')
    if regressors is not None:
        last_regressor_values = regressors.iloc[-1:].drop(Config.DATE_COL, axis=1)
        future_regressors = pd.DataFrame(np.tile(last_regressor_values.values, (horizon, 1)), columns=last_regressor_values.columns)
        future_regressors['ds'] = future['ds'].iloc[-horizon:]
        future = pd.merge(future, future_regressors, on='ds', how='left')

    forecast = full_model.predict(future)
    return full_model, forecast, metrics

@st.cache_data(show_spinner=False)
def auto_arima_search(ts_data, test_size):
    """Encuentra los parámetros óptimos para un modelo SARIMA usando auto_arima."""
    ts_data_copy = ts_data.copy()
    if not pd.api.types.is_datetime64_any_dtype(ts_data_copy.index):
        ts_data_copy = ts_data_copy.set_index(Config.DATE_COL).sort_index()

    ts = ts_data_copy[Config.PRECIPITATION_COL].asfreq('MS').interpolate(method='time').dropna()
    train = ts[:-test_size]

    auto_model = pm.auto_arima(train,
                               start_p=1, start_q=1,
                               test='adf',
                               max_p=3, max_q=3,
                               m=12,
                               start_P=0, seasonal=True,
                               d=None, D=None,
                               trace=False,
                               error_action='ignore',
                               suppress_warnings=True,
                               stepwise=True)
    return auto_model.order, auto_model.seasonal_order