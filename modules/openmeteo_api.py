# modules/openmeteo_api.py
import streamlit as st
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import date

@st.cache_data(ttl=3600) # Cachea por 1 hora
def get_historical_climate_average(latitudes, longitudes, variable, start_date_str, end_date_str):
    """
    Obtiene el promedio de una variable climática para un rango de fechas y ubicaciones
    usando la API histórica de Open-Meteo.
    Retorna un DataFrame con 'latitude', 'longitude', 'valor_promedio'.
    """
    if not latitudes or not longitudes:
        return pd.DataFrame(columns=['latitude', 'longitude', 'valor_promedio'])

    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # URL para la API histórica/climática
    url = "https://archive-api.open-meteo.com/v1/archive" 
    
    # Asegúrate de que las fechas estén en el formato correcto
    try:
        start_date = date.fromisoformat(start_date_str)
        end_date = date.fromisoformat(end_date_str)
    except ValueError:
        st.error("Formato de fecha inválido. Use 'YYYY-MM-DD'.")
        return pd.DataFrame(columns=['latitude', 'longitude', 'valor_promedio'])

    params = {
        "latitude": latitudes,
        "longitude": longitudes,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": variable, # Pide la variable diaria
        "timezone": "auto"
    }

    try:
        responses = openmeteo.weather_api(url, params=params)
    except Exception as e:
        st.error(f"Error al llamar a la API de Open-Meteo: {e}")
        return pd.DataFrame(columns=['latitude', 'longitude', 'valor_promedio'])

    results = []
    for i, response in enumerate(responses):
        daily = response.Daily()
        if daily is None or daily.VariablesLength() == 0:
            # Si no hay datos para esta ubicación, añade NaN
            results.append({
                'latitude': response.Latitude(),
                'longitude': response.Longitude(),
                'valor_promedio': pd.NA 
            })
            continue 

        daily_variable = daily.Variables(0).ValuesAsNumpy()
        
        # Calcula el promedio de los valores diarios obtenidos
        mean_value = pd.Series(daily_variable).mean() 
        
        results.append({
            'latitude': response.Latitude(),
            'longitude': response.Longitude(),
            'valor_promedio': mean_value
        })

    result_df = pd.DataFrame(results)
    # Elimina filas donde no se pudo calcular el promedio
    result_df.dropna(subset=['valor_promedio'], inplace=True) 
    return result_df
