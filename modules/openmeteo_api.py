import requests
import pandas as pd
import numpy as np
import time
import streamlit as st
from datetime import datetime

def get_historical_monthly_series(lats, lons, start_date, end_date):
    """
    Descarga series de tiempo históricas de precipitación (ERA5-Land) usando Open-Meteo Archive API.
    
    Características:
    - Procesa en LOTES (Chunks) para evitar errores de URL muy larga o timeout.
    - Agrega automáticamente los datos diarios a mensuales.
    - Maneja reintentos básicos.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # Validaciones básicas
    if not lats or not lons or len(lats) != len(lons):
        return pd.DataFrame()
        
    # Asegurar formato lista
    if not isinstance(lats, list): lats = [lats]
    if not isinstance(lons, list): lons = [lons]
    
    # --- CONFIGURACIÓN DE LOTES ---
    # Open-Meteo permite múltiples puntos, pero URLs muy largas fallan.
    # Procesaremos de a 20 estaciones por llamada.
    BATCH_SIZE = 20
    all_series = []
    
    total_points = len(lats)
    
    # Barra de progreso auxiliar (opcional, solo si estamos en Streamlit)
    progress_bar = None
    if total_points > BATCH_SIZE:
        progress_bar = st.progress(0, text="📡 Descargando datos satelitales por lotes...")

    for i in range(0, total_points, BATCH_SIZE):
        # Seleccionar lote actual
        lats_batch = lats[i : i + BATCH_SIZE]
        lons_batch = lons[i : i + BATCH_SIZE]
        
        params = {
            "latitude": ",".join(map(str, lats_batch)),
            "longitude": ",".join(map(str, lons_batch)),
            "start_date": start_date,
            "end_date": end_date,
            "daily": "precipitation_sum",
            "timezone": "America/Bogota"
        }

        try:
            # Hacemos la petición con reintento simple
            for attempt in range(3):
                try:
                    response = requests.get(url, params=params, timeout=60)
                    if response.status_code == 200:
                        break
                    elif response.status_code == 429: # Rate limit
                        time.sleep(2 * (attempt + 1))
                except requests.exceptions.RequestException:
                    time.sleep(1)
            
            if response.status_code != 200:
                print(f"Error lote {i}: {response.status_code}")
                continue

            data = response.json()
            
            # Procesar respuesta (puede ser una lista de resultados o un solo objeto)
            results = data if isinstance(data, list) else [data]
            
            for j, res in enumerate(results):
                if "daily" not in res: continue
                
                # Crear DataFrame Diario
                df = pd.DataFrame({
                    "date": res["daily"]["time"],
                    "ppt_daily": res["daily"]["precipitation_sum"]
                })
                df["date"] = pd.to_datetime(df["date"])
                
                # Agregación Mensual (Suma)
                # Usamos periodo para agrupar y luego volvemos a timestamp inicio de mes
                df_monthly = df.groupby(df["date"].dt.to_period("M"))["ppt_daily"].sum().reset_index()
                df_monthly["date"] = df_monthly["date"].dt.to_timestamp()
                
                # Asignar coordenadas originales de este punto
                df_monthly["latitude"] = lats_batch[j]
                df_monthly["longitude"] = lons_batch[j]
                
                # Renombrar para consistencia
                df_monthly.rename(columns={"ppt_daily": "ppt_sat"}, inplace=True)
                all_series.append(df_monthly)
            
            # Pausa breve para ser amables con la API
            time.sleep(0.2)
            
            # Actualizar progreso
            if progress_bar:
                progress = min((i + BATCH_SIZE) / total_points, 1.0)
                progress_bar.progress(progress, text=f"Descargando satélite: {int(progress*100)}%")

        except Exception as e:
            print(f"Error procesando lote {i}: {e}")
            continue

    if progress_bar: progress_bar.empty()

    if not all_series: 
        return pd.DataFrame()
        
    final_df = pd.concat(all_series, ignore_index=True)
    return final_df
