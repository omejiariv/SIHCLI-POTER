import streamlit as st
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import date
import time

# Configuración del cliente con caché y reintentos
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

@st.cache_data(ttl=3600)
def get_historical_climate_average(latitudes, longitudes, variable, start_date_str, end_date_str):
    """
    Obtiene el promedio histórico dividiendo la petición en lotes pequeños
    para evitar el error '414 URI Too Long'.
    """
    if not latitudes or not longitudes:
        return pd.DataFrame(columns=['latitude', 'longitude', 'valor_promedio'])

    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # Validar fechas
    try:
        start = date.fromisoformat(start_date_str)
        end = date.fromisoformat(end_date_str)
    except ValueError:
        st.error("Formato de fecha inválido.")
        return pd.DataFrame()

    all_results = []
    
    # --- ESTRATEGIA DE LOTES (BATCHING) ---
    BATCH_SIZE = 10 # Procesar de 10 en 10 estaciones
    
    total_points = len(latitudes)
    
    # Barra de progreso si son muchas
    progress_bar = None
    if total_points > BATCH_SIZE:
        progress_bar = st.progress(0, text="Descargando datos satelitales por lotes...")

    for i in range(0, total_points, BATCH_SIZE):
        # Crear lote actual
        lats_batch = latitudes[i : i + BATCH_SIZE]
        lons_batch = longitudes[i : i + BATCH_SIZE]
        
        params = {
            "latitude": lats_batch,
            "longitude": lons_batch,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": variable,
            "timezone": "auto"
        }

        try:
            # Llamada a la API para el lote
            responses = openmeteo.weather_api(url, params=params)
            
            # Procesar respuestas del lote
            for j, response in enumerate(responses):
                daily = response.Daily()
                if daily is None: continue
                
                # Extraer datos
                daily_data = daily.Variables(0).ValuesAsNumpy()
                mean_val = pd.Series(daily_data).mean()
                
                all_results.append({
                    'latitude': response.Latitude(),
                    'longitude': response.Longitude(),
                    'valor_promedio': mean_val
                })
            
            # Pequeña pausa para ser amables con la API
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error en lote {i}: {e}") # Log silencioso para no interrumpir todo
            continue
        
        # Actualizar barra
        if progress_bar:
            progress = min((i + BATCH_SIZE) / total_points, 1.0)
            progress_bar.progress(progress, text=f"Procesando estaciones {i+1} a {min(i+BATCH_SIZE, total_points)} de {total_points}...")

    if progress_bar: progress_bar.empty()

    # Crear DataFrame final
    if not all_results:
        return pd.DataFrame(columns=['latitude', 'longitude', 'valor_promedio'])
        
    return pd.DataFrame(all_results)
