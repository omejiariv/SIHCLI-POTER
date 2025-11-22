# modules/forecasting.py

import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from modules.config import Config
import pmdarima as pm
import streamlit as st

def evaluate_forecast(y_true, y_pred):
    """Calcula RMSE y MAE."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae}

def get_decomposition_results(series, period=12, model='additive'):
    """Descomposición estacional."""
    # Limpieza para asegurar que no falle
    series_clean = series.asfreq('MS').interpolate(method='time').dropna()
    if len(series_clean) < period * 2:
        raise ValueError("Serie muy corta para descomponer.")
    return seasonal_decompose(series_clean, model=model, period=period)

def create_acf_chart(series, max_lag):
    import plotly.graph_objects as go
    acf_vals = acf(series, nlags=max_lag)
    fig = go.Figure(data=[go.Bar(x=list(range(len(acf_vals))), y=acf_vals)])
    fig.update_layout(title="Autocorrelación (ACF)", xaxis_title="Lag", yaxis_title="ACF")
    return fig

def create_pacf_chart(series, max_lag):
    import plotly.graph_objects as go
    pacf_vals = pacf(series, nlags=max_lag)
    fig = go.Figure(data=[go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals)])
    fig.update_layout(title="Autocorrelación Parcial (PACF)", xaxis_title="Lag", yaxis_title="PACF")
    return fig

@st.cache_data(show_spinner=False)
def auto_arima_search(ts_data, test_size):
    """Busca parámetros óptimos SARIMA."""
    train = ts_data[:-test_size]
    model = pm.auto_arima(train, start_p=0, start_q=0, m=12, seasonal=True, trace=False, error_action='ignore')
    return model.order, model.seasonal_order

# --- FUNCIÓN SARIMA CON REGRESORES ---
@st.cache_data
def generate_sarima_forecast(ts_data, order, seasonal_order, horizon, test_size=12, regressors=None):
    """Pronóstico SARIMA con soporte para regresores exógenos (ej. ONI pronosticado)."""
    
    # 1. Preparar Serie Principal
    ts = ts_data[Config.PRECIPITATION_COL].asfreq('MS').interpolate(method='time').dropna()
    
    # 2. Preparar Regresores (Exógenos)
    exog = None
    exog_future = None
    
    if regressors is not None and not regressors.empty:
        # Alinear índices
        regressors = regressors.set_index(Config.DATE_COL).sort_index()
        # Exog histórico (alineado con ts)
        exog = regressors.reindex(ts.index).interpolate(method='linear').bfill().ffill()
        
        # Exog futuro (para el horizonte de pronóstico)
        last_date = ts.index[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='MS')
        exog_future = regressors.reindex(future_dates).interpolate(method='linear').bfill().ffill()
        
        # Validación de seguridad: Si faltan datos futuros en el regresor, no se puede pronosticar
        if exog_future.isnull().any().any():
            exog, exog_future = None, None # Abortar uso de regresores si están incompletos

    # 3. Split Train/Test
    train = ts[:-test_size]
    exog_train = exog[:-test_size] if exog is not None else None
    test = ts[-test_size:]
    exog_test = exog[-test_size:] if exog is not None else None

    # 4. Entrenamiento y Evaluación
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order, exog=exog_train, 
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    
    # Predicción en Test
    pred = res.get_forecast(steps=test_size, exog=exog_test)
    metrics = evaluate_forecast(test, pred.predicted_mean)

    # 5. Modelo Final (Toda la data)
    full_model = SARIMAX(ts, order=order, seasonal_order=seasonal_order, exog=exog,
                         enforce_stationarity=False, enforce_invertibility=False)
    full_res = full_model.fit(disp=False)
    
    # Pronóstico Futuro
    forecast = full_res.get_forecast(steps=horizon, exog=exog_future)
    
    # Formatear salida
    fc_df = forecast.predicted_mean.reset_index()
    fc_df.columns = ['ds', 'yhat']
    
    # Intervalos de confianza
    ci = forecast.conf_int()
    ci.columns = ['yhat_lower', 'yhat_upper']
    
    return ts, fc_df['yhat'], ci, metrics, fc_df

# --- FUNCIÓN PROPHET CON REGRESORES ---
@st.cache_data
def generate_prophet_forecast(ts_data, horizon, test_size=12, regressors=None):
    """Pronóstico Prophet con soporte para regresores adicionales."""
    
    # 1. Preparar Datos
    df = ts_data.reset_index()[['fecha_mes_año', 'precipitation']].rename(columns={'fecha_mes_año':'ds', 'precipitation':'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    
    # 2. Preparar Regresores
    regressor_cols = []
    if regressors is not None and not regressors.empty:
        # Unir regresores al DF principal por fecha
        # regressors debe tener columna 'ds' (o fecha) y las variables
        r_copy = regressors.copy()
        if 'ds' not in r_copy.columns and Config.DATE_COL in r_copy.columns:
             r_copy = r_copy.rename(columns={Config.DATE_COL: 'ds'})
        
        df = pd.merge(df, r_copy, on='ds', how='left')
        
        # Identificar columnas de regresores (todas menos ds e y)
        regressor_cols = [c for c in r_copy.columns if c != 'ds']
        
        # Rellenar huecos en historia
        for col in regressor_cols:
            df[col] = df[col].interpolate(method='linear').bfill().ffill()

    # 3. Split
    train = df.iloc[:-test_size]
    test = df.iloc[-test_size:]

    # 4. Configurar Modelo
    m = Prophet(yearly_seasonality=True)
    for col in regressor_cols:
        m.add_regressor(col)
        
    m.fit(train)
    
    # 5. Evaluar
    # Para predecir en test, necesitamos los regresores futuros (que ya están en 'test' del merge)
    forecast_test = m.predict(test.drop(columns=['y']))
    metrics = evaluate_forecast(test['y'], forecast_test['yhat'])

    # 6. Modelo Final
    m_full = Prophet(yearly_seasonality=True)
    for col in regressor_cols:
        m_full.add_regressor(col)
    m_full.fit(df)

    # 7. Futuro
    future = m_full.make_future_dataframe(periods=horizon, freq='MS')
    
    # Si hay regresores, necesitamos sus valores futuros
    if regressor_cols:
        # Asumimos que 'regressors' trae los datos futuros (porque fue generado como pronóstico de índice)
        # Hacemos merge nuevamente con el futuro
        future = pd.merge(future, r_copy, on='ds', how='left')
        for col in regressor_cols:
             future[col] = future[col].interpolate(method='linear').bfill().ffill()
             
        # Si después de todo faltan datos futuros en regresores, Prophet fallará. 
        # Cortamos el futuro hasta donde haya datos de regresores
        future = future.dropna(subset=regressor_cols)

    forecast = m_full.predict(future)
    
    return m_full, forecast, metrics

