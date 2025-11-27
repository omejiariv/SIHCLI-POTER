import streamlit as st
import pandas as pd
import numpy as np
import requests # Librería estándar
from datetime import date, datetime, timedelta
import time

# --- INTENTO DE IMPORTAR LIBRERÍAS AVANZADAS (Para Históricos) ---
try:
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry
    
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    HAS_ADVANCED_LIBS = True
except ImportError:
    HAS_ADVANCED_LIBS = False

# ==============================================================================
# 1. FUNCIÓN DE PRONÓSTICO (CON REINTENTOS ANTI-ERROR 429)
# ==============================================================================
@st.cache_data(ttl=3600)  # Cache de 1 hora
def get_weather_forecast_detailed(lat, lon):
    """
    Descarga el pronóstico detallado de 7 días (Open-Meteo).
    Incluye lógica de reintentos para evitar el error 429 (Too Many Requests).
    """
    url = "https://api.open-meteo.com/v1/forecast"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", 
                  "wind_speed_10m_max", "shortwave_radiation_sum", "et0_fao_evapotranspiration"],
        "hourly": ["relative_humidity_2m", "surface_pressure"],
        "timezone": "auto",
        "forecast_days": 7
    }
    
    # Headers para identificarnos correctamente (evita bloqueos por bot genérico)
    headers = {
        "User-Agent": "SIHCLI-App/1.0 (streamlit_client)"
    }

    # SISTEMA DE REINTENTOS MANUAL
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                daily = data.get("daily", {})
                hourly = data.get("hourly", {})
                
                # 1. Crear DataFrame Diario Base
                df = pd.DataFrame({
                    "Fecha": daily.get("time", []),
                    "T. Máx (°C)": daily.get("temperature_2m_max", []),
                    "T. Mín (°C)": daily.get("temperature_2m_min", []),
                    "Ppt. (mm)": daily.get("precipitation_sum", []),
                    "Viento Máx (km/h)": daily.get("wind_speed_10m_max", []),
                    "Radiación SW (MJ/m²)": daily.get("shortwave_radiation_sum", []),
                    "ET₀ (mm)": daily.get("et0_fao_evapotranspiration", [])
                })
                
                # 2. Procesar Datos Horarios
                if hourly:
                    try:
                        h_times = pd.to_datetime(hourly.get("time", []))
                        h_hum = hourly.get("relative_humidity_2m", [])
                        h_pres = hourly.get("surface_pressure", [])
                        
                        df_h = pd.DataFrame({'time': h_times, 'hr': h_hum, 'pres': h_pres})
                        df_h['date_str'] = df_h['time'].dt.date.astype(str)
                        
                        daily_avgs = df_h.groupby('date_str').mean().reset_index()
                        
                        df['Fecha'] = df['Fecha'].astype(str)
                        df = pd.merge(df, daily_avgs, left_on='Fecha', right_on='date_str', how='left')
                        
                        df['HR Media (%)'] = df['hr'].round(1).fillna(0)
                        df['Presión (hPa)'] = df['pres'].round(1).fillna(1013)
                    except Exception as e:
                        print(f"Advertencia procesando horarios: {e}")
                        df['HR Media (%)'] = 0; df['Presión (hPa)'] = 1013
                
                return df
            
            elif response.status_code == 429:
                # Si es error de límite, esperamos y reintentamos
                wait_time = 2 * (attempt + 1)
                print(f"⚠️ API Limit (429). Reintentando en {wait_time}s...")
                time.sleep(wait_time)
                continue # Volver al inicio del loop
                
            else:
                st.error(f"Error API Open-Meteo: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Error de conexión clima: {e}")
            return pd.DataFrame()
            
    # Si llega aquí es que fallaron todos los intentos
    st.error("❌ Servicio de clima ocupado (429). Intente en unos minutos.")
    return pd.DataFrame()

# ==============================================================================
# 2. FUNCIONES DE DATOS HISTÓRICOS (INTACTAS)
# ==============================================================================

@st.cache_data(ttl=3600)
def get_historical_climate_average(latitudes, longitudes, variable, start_date_str, end_date_str):
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_historical_monthly_series(latitudes, longitudes, start_date_str, end_date_str, variable="precipitation_sum"):
    if not HAS_ADVANCED_LIBS: return pd.DataFrame()
    if not latitudes or not longitudes: return pd.DataFrame()

    url = "https://archive-api.open-meteo.com/v1/archive"
    try:
        start = date.fromisoformat(start_date_str)
        end = date.fromisoformat(end_date_str)
    except ValueError: return pd.DataFrame()

    BATCH_SIZE = 20 
    total_points = len(latitudes)
    all_series = []
    progress_bar = st.progress(0, text="📡 Descargando series satelitales...")

    for i in range(0, total_points, BATCH_SIZE):
        lats_batch = latitudes[i : i + BATCH_SIZE]
        lons_batch = longitudes[i : i + BATCH_SIZE]
        params = {
            "latitude": lats_batch, "longitude": lons_batch,
            "start_date": start.isoformat(), "end_date": end.isoformat(),
            "daily": variable, "timezone": "auto"
        }
        try:
            responses = openmeteo.weather_api(url, params=params)
            for response in responses:
                lat = response.Latitude()
                lon = response.Longitude()
                daily = response.Daily()
                time_range = pd.to_datetime(np.arange(daily.Time(), daily.TimeEnd(), daily.Interval()), unit='s')
                values = daily.Variables(0).ValuesAsNumpy()
                
                df_temp = pd.DataFrame({'date': time_range, 'value': values})
                try: df_monthly = df_temp.resample('ME', on='date').sum().reset_index()
                except: df_monthly = df_temp.resample('M', on='date').sum().reset_index()

                df_monthly['date'] = df_monthly['date'].dt.to_period('M').dt.to_timestamp()
                df_monthly['latitude'] = lat
                df_monthly['longitude'] = lon
                all_series.append(df_monthly)
            time.sleep(0.05)
        except Exception as e:
            print(f"Error en lote {i}: {e}")
            continue
        progress_bar.progress(min((i + BATCH_SIZE) / total_points, 1.0), text=f"Procesando...")

    progress_bar.empty()
    if not all_series: return pd.DataFrame()
    final_df = pd.concat(all_series, ignore_index=True)
    final_df.rename(columns={'value': 'ppt_sat'}, inplace=True)
    return final_df
