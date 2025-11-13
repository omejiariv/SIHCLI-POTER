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
from pandas.tseries.offsets import DateOffset
from datetime import datetime

# modules/forecast_api.py

import streamlit as st
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import requests

@st.cache_data(ttl=3600) # Cache por 1 hora
def get_weather_forecast(latitude, longitude):
    """Obtiene el pronóstico del tiempo para 7 días con variables adicionales."""
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    
    # --- Variables a solicitar ---
    daily_variables = [
        "temperature_2m_max",               # Temperatura máxima a 2m (°C)
        "temperature_2m_min",               # Temperatura mínima a 2m (°C)
        "precipitation_sum",                # Suma de precipitación diaria (mm)
        "relative_humidity_2m_mean",        # Humedad relativa media a 2m (%)
        "surface_pressure_mean",            # Presión superficial media (hPa) - Usamos esta en lugar de MSL para pronóstico
        "et0_fao_evapotranspiration",       # Evapotranspiración de referencia (mm)
        "shortwave_radiation_sum",          # Suma de radiación de onda corta diaria (MJ/m²)
        "wind_speed_10m_max"                # Velocidad máxima del viento a 10m (km/h o m/s, verificar unidad)
        # "precipitable_water"             # Agua precipitable (mm) - Si está disponible para diario
    ]
    # Nota: "Intensidad lumínica" (lux) no suele estar disponible directamente. Usamos radiación solar.
    # Nota: Vapor de agua se aproxima con Agua Precipitable si está disponible, o Humedad Relativa.

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": daily_variables, # Lista actualizada de variables
        "timezone": "auto",
        "forecast_days": 7 # Aseguramos que sean 7 días
    }

    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

        # Procesar la respuesta
        daily = response.Daily()
        if daily is None:
             st.error("La API no devolvió datos diarios.")
             return None
             
        timezone_str = response.Timezone().decode('utf-8')
        
        # --- Diccionario para construir el DataFrame ---
        daily_data = {
            "date": pd.to_datetime(daily.Time(), unit="s", utc=True).tz_convert(timezone_str)
        }
        
        # Iterar sobre las variables solicitadas para extraer los datos
        num_variables = daily.VariablesLength()
        if num_variables != len(daily_variables):
             st.warning(f"La API devolvió {num_variables} variables, pero se solicitaron {len(daily_variables)}. Algunas podrían faltar.")
             # Aun así intentamos procesar las que sí llegaron
        
        # Mapeo manual para asegurar el orden y manejar posibles faltantes
        variable_map = {var: i for i, var in enumerate(daily_variables)}
        
        if variable_map.get("temperature_2m_max", -1) < num_variables and variable_map.get("temperature_2m_max", -1) != -1:
            daily_data["temperature_2m_max"] = daily.Variables(variable_map["temperature_2m_max"]).ValuesAsNumpy()
        if variable_map.get("temperature_2m_min", -1) < num_variables and variable_map.get("temperature_2m_min", -1) != -1:
            daily_data["temperature_2m_min"] = daily.Variables(variable_map["temperature_2m_min"]).ValuesAsNumpy()
        if variable_map.get("precipitation_sum", -1) < num_variables and variable_map.get("precipitation_sum", -1) != -1:
            daily_data["precipitation_sum"] = daily.Variables(variable_map["precipitation_sum"]).ValuesAsNumpy()
        if variable_map.get("relative_humidity_2m_mean", -1) < num_variables and variable_map.get("relative_humidity_2m_mean", -1) != -1:
            daily_data["relative_humidity_2m_mean"] = daily.Variables(variable_map["relative_humidity_2m_mean"]).ValuesAsNumpy()
        if variable_map.get("surface_pressure_mean", -1) < num_variables and variable_map.get("surface_pressure_mean", -1) != -1:
            daily_data["surface_pressure_mean"] = daily.Variables(variable_map["surface_pressure_mean"]).ValuesAsNumpy()
        if variable_map.get("et0_fao_evapotranspiration", -1) < num_variables and variable_map.get("et0_fao_evapotranspiration", -1) != -1:
            daily_data["et0_fao_evapotranspiration"] = daily.Variables(variable_map["et0_fao_evapotranspiration"]).ValuesAsNumpy()
        if variable_map.get("shortwave_radiation_sum", -1) < num_variables and variable_map.get("shortwave_radiation_sum", -1) != -1:
            daily_data["shortwave_radiation_sum"] = daily.Variables(variable_map["shortwave_radiation_sum"]).ValuesAsNumpy()
        if variable_map.get("wind_speed_10m_max", -1) < num_variables and variable_map.get("wind_speed_10m_max", -1) != -1:
            daily_data["wind_speed_10m_max"] = daily.Variables(variable_map["wind_speed_10m_max"]).ValuesAsNumpy()

        # Crear DataFrame y devolverlo
        forecast_df = pd.DataFrame(data=daily_data)
        # Asegurarse de que solo tenemos 7 días si la API devuelve más
        return forecast_df.head(7) 

    except openmeteo_requests.ApiError as e: # Catch specific API errors
        st.error(f"Error de la API de Open-Meteo: {e}")
        return None
    except Exception as e:
        st.error(f"Ocurrió un error inesperado al obtener el pronóstico: {e}")
        return None

@st.cache_data(ttl=86400) # Cachear por 1 día
def get_official_enso_forecast():
    """
    Descarga y procesa el pronóstico consolidado de ENSO (IRI/CPC).
    
    Este pronóstico es estacional (ej. 'JFM', 'FMA'). La función lo interpola
    a una frecuencia mensual ('MS') para que sea compatible con los modelos.
    
    Retorna:
        - df_prophet: DataFrame listo para Prophet (cols 'ds', 'anomalia_oni')
        - df_sarima: DataFrame listo para SARIMA (cols 'fecha_mes_año', 'anomalia_oni')
    """
    try:
        DATA_URL = "https://iri.columbia.edu/climate/forecast/enso/ensostat.tsv"
        
        # --- INICIO DE LA CORRECCIÓN (Lector de texto V5) ---
        
        # 1. Descargar el archivo como texto crudo
        response = requests.get(DATA_URL)
        response.raise_for_status()
        data_text = response.text
        lines = data_text.split('\n')
        data_rows = []
        
        # 2. Encontrar la línea del encabezado (Header)
        header_line_index = -1
        # Buscar la línea que contenga "YEAR" Y "NINO3.4_ANOM_FCST"
        for i, line in enumerate(lines):
            if "YEAR" in line and "NINO3.4_ANOM_FCST" in line:
                header_line_index = i
                break
        
        if header_line_index == -1:
            raise ValueError("No se pudo encontrar la línea de encabezado (con 'YEAR' y 'NINO3.4_ANOM_FCST') en el archivo.")

        # 3. Encontrar la posición de las columnas que queremos
        headers = lines[header_line_index].split()
        
        # Buscar el nombre de la columna que *contiene* NINO3.4_ANOM_FCST
        forecast_header_name = None
        for h in headers:
            if "NINO3.4_ANOM_FCST" in h:
                forecast_header_name = h
                break
        
        if forecast_header_name is None:
            raise ValueError("No se pudo encontrar la columna de pronóstico 'NINO3.4_ANOM_FCST' en el encabezado.")

        # Buscar el nombre de la columna MON(TH)
        month_header_name = None
        if "MONTH" in headers:
            month_header_name = "MONTH"
        elif "MON" in headers:
            month_header_name = "MON"
        else:
            raise ValueError("No se pudo encontrar la columna 'MON' o 'MONTH' en el encabezado.")
        
        year_header_name = "YEAR" # Esta parece estable

        # Get the *indices*
        month_idx = headers.index(month_header_name)
        year_idx = headers.index(year_header_name)
        forecast_idx = headers.index(forecast_header_name)

        # 4. Iterar sobre las líneas de datos (después del encabezado)
        for line in lines[header_line_index + 1:]:
            items = line.split()
            
            # Si la línea está vacía o es muy corta, saltar
            if len(items) < max(month_idx, year_idx, forecast_idx) + 1:
                continue
                
            month = items[month_idx]
            year_str = items[year_idx]
            forecast_val = items[forecast_idx]
            
            # Solo nos interesan las filas que NO son 'OBS'
            if forecast_val != 'OBS':
                try:
                    year_int = int(year_str.split('-')[0])
                    data_rows.append({
                        "MONTH": month,
                        "YEAR": year_int,
                        "NINO3.4_ANOM_FCST": float(forecast_val)
                    })
                except ValueError:
                    continue # Ignorar líneas que no se puedan convertir

        df_forecast = pd.DataFrame(data_rows)
        
        if df_forecast.empty:
            raise ValueError("No se encontraron datos de pronóstico válidos en el archivo.")

        # --- FIN DE LA CORRECCIÓN ---

        # El resto de la función es idéntica
        df_forecast['anomalia_oni'] = pd.to_numeric(df_forecast['NINO3.4_ANOM_FCST'])
        
        season_to_month_map = {
            'JFM': 2, 'FMA': 3, 'MAM': 4, 'AMJ': 5, 'MJJ': 6, 'JJA': 7,
            'JAS': 8, 'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12, 'DJF': 1
        }
        
        df_forecast['month'] = df_forecast['MONTH'].map(season_to_month_map)
        
        df_forecast['year_adj'] = df_forecast['YEAR'].where(df_forecast['MONTH'] != 'DJF', df_forecast['YEAR'] + 1)
        
        # Asegurarse de que 'year_adj' y 'month' no tengan nulos
        df_forecast.dropna(subset=['year_adj', 'month'], inplace=True)
        
        df_forecast['ds'] = pd.to_datetime(
            df_forecast['year_adj'].astype(int).astype(str) + '-' + df_forecast['month'].astype(int).astype(str) + '-01'
        )
        
        df_clean = df_forecast[['ds', 'anomalia_oni']].sort_values(by='ds').drop_duplicates()
        
        df_clean = df_clean.set_index('ds')
        date_range = pd.date_range(start=df_clean.index.min(), end=df_clean.index.max(), freq='MS')
        
        df_monthly = df_clean.reindex(date_range)
        df_monthly['anomalia_oni'] = df_monthly['anomalia_oni'].interpolate(method='linear')
        
        df_monthly = df_monthly.reset_index().rename(columns={'index': 'ds'})

        df_prophet_out = df_monthly.rename(columns={'anomalia_oni': Config.ENSO_ONI_COL})
        
        df_sarima_out = df_monthly.rename(
            columns={'ds': Config.DATE_COL, 'anomalia_oni': Config.ENSO_ONI_COL}
        )
        
        return df_prophet_out, df_sarima_out
        
    except Exception as e:
        # Esto capturará el error y lo mostrará en Streamlit
        st.error(f"Error al descargar el pronóstico oficial del ENSO: {e}")
        return None, None
        
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
@st.cache_data
def generate_sarima_forecast(ts_data_raw, order, seasonal_order, horizon, test_size=12, regressors=None):
    """Entrena, evalúa y genera un pronóstico con SARIMAX, incluyendo regresores opcionales."""
    ts_data = ts_data_raw[[Config.DATE_COL, Config.PRECIPITATION_COL]].copy()
    ts_data = ts_data.drop_duplicates(subset=[Config.DATE_COL], keep='first')
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

@st.cache_data
def generate_prophet_forecast(ts_data_raw, horizon, test_size=12, regressors=None):
    """Entrena, evalúa y genera un pronóstico con Prophet, incluyendo regresores opcionales."""
    
    # 1. Preparar datos de precipitación
    ts_data = ts_data_raw.rename(columns={Config.DATE_COL: 'ds', Config.PRECIPITATION_COL: 'y'})
    ts_data = ts_data.drop_duplicates(subset=['ds'], keep='first')
    
    # Corrección para 'ValueError: time-weighted interpolation'
    ts_data['ds'] = pd.to_datetime(ts_data['ds'])
    ts_data = ts_data.set_index('ds')
    ts_data['y'] = ts_data['y'].interpolate(method='time')
    ts_data = ts_data.reset_index()

    if len(ts_data) < test_size + 24:
        raise ValueError(f"Se necesitan al menos {test_size + 24} meses de datos para Prophet.")

    # 2. Inicializar el modelo de *evaluación*
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)

    # 3. Preparar regresores (si existen)
    regressor_cols = []
    if regressors is not None and not regressors.empty:
        
        # --- INICIO DE LA CORRECCIÓN PARA 'KeyError: anomalia_oni' ---
        
        # 1. Obtener la lista de nombres de regresores (ej: ['anomalia_oni'])
        regressor_names = [col for col in regressors.columns if col != 'ds']
        
        # 2. Eliminar esas columnas de ts_data (los datos históricos)
        #    para evitar un conflicto de merge (col_x, col_y)
        cols_to_drop_from_ts = [col for col in regressor_names if col in ts_data.columns]
        if cols_to_drop_from_ts:
            ts_data = ts_data.drop(columns=cols_to_drop_from_ts)
        
        # 3. Ahora el merge es limpio. 'ts_data' solo tendrá una columna 'anomalia_oni' (la del pronóstico).
        ts_data = pd.merge(ts_data, regressors, on='ds', how='left')
        
        # 4. Iterar sobre los nombres de regresores (ej: 'anomalia_oni')
        for col in regressor_names:
            # Interpolar el regresor (que ahora es el pronosticado)
            ts_data[col] = ts_data[col].interpolate(method='linear', limit_direction='both')
            model.add_regressor(col)
            regressor_cols.append(col)
        # --- FIN DE LA CORRECCIÓN ---

    # 4. Dividir datos y entrenar modelo de evaluación
    train, test = ts_data.iloc[:-test_size], ts_data.iloc[-test_size:]
    model.fit(train)

    # 5. Evaluar modelo
    test_dates = model.make_future_dataframe(periods=test_size, freq='MS').tail(test_size)
    if regressor_cols: 
        test_regressors = ts_data[ts_data['ds'].isin(test_dates['ds'])][['ds'] + regressor_cols]
        test_dates = pd.merge(test_dates, test_regressors, on='ds', how='left')
        test_dates[regressor_cols] = test_dates[regressor_cols].interpolate(method='linear', limit_direction='both')
    
    y_pred_test = model.predict(test_dates)['yhat']
    metrics = evaluate_forecast(test['y'], y_pred_test)

    # 6. Entrenar modelo completo (Full Model)
    full_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    if regressor_cols:
        for col in regressor_cols: # Añadir los mismos regresores al modelo final
            full_model.add_regressor(col)
    
    full_model.fit(ts_data) # Entrenar con TODOS los datos

    # 7. Crear DataFrame futuro
    future = full_model.make_future_dataframe(periods=horizon, freq='MS')
    
    if regressor_cols:
        future = pd.merge(future, regressors, on='ds', how='left')
        future[regressor_cols] = future[regressor_cols].interpolate(method='linear', limit_direction='both')

    # 8. Generar pronóstico final
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














