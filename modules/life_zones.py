# modules/life_zones.py
import numpy as np
import pandas as pd
import streamlit as st
from scipy.interpolate import griddata
import plotly.graph_objects as go
from modules.config import Config

# --- Diccionario de Zonas de Vida (Simplificado) ---
holdridge_zone_map = {
    1: "Nival", 2: "Tundra pluvial alpino", 3: "Tundra húmeda alpino", 4: "Tundra seca alpino",
    5: "Páramo pluvial subalpino", 6: "Páramo muy húmedo subalpino", 7: "Páramo seco subalpino",
    8: "Bosque pluvial Montano", 9: "Bosque muy húmedo Montano", 10: "Bosque húmedo Montano",
    11: "Bosque seco Montano", 12: "Monte espinoso Montano",
    13: "Bosque pluvial Premontano", 14: "Bosque muy húmedo Premontano", 15: "Bosque húmedo Premontano",
    16: "Bosque seco Premontano", 17: "Monte espinoso Premontano",
    18: "Bosque pluvial Tropical", 19: "Bosque muy húmedo Tropical", 20: "Bosque húmedo Tropical",
    21: "Bosque seco Tropical", 22: "Monte espinoso Tropical", 0: "Desconocido"
}

def classify_life_zone(altitude, ppt):
    """Clasifica una celda según Holdridge (Altitud y Precipitación)."""
    if pd.isna(altitude) or pd.isna(ppt) or altitude < 0 or ppt <= 0: return 0
    
    # Lógica simplificada de Holdridge basada en pisos altitudinales
    if altitude > 4200: return 1 # Nival
    
    # Piso Alpino (3700 - 4200)
    if altitude >= 3700:
        if ppt >= 1500: return 2
        elif ppt >= 750: return 3
        else: return 4
        
    # Piso Subalpino (3200 - 3700)
    if altitude >= 3200:
        if ppt >= 2000: return 5
        elif ppt >= 1000: return 6
        else: return 7

    # Piso Montano (2000 - 3200)
    if altitude >= 2000:
        if ppt >= 4000: return 8
        elif ppt >= 2000: return 9
        elif ppt >= 1000: return 10
        elif ppt >= 500: return 11
        else: return 12

    # Piso Premontano (1000 - 2000)
    if altitude >= 1000:
        if ppt >= 4000: return 13
        elif ppt >= 2000: return 14
        elif ppt >= 1000: return 15
        elif ppt >= 500: return 16
        else: return 17

    # Piso Tropical (< 1000)
    if ppt >= 4000: return 18
    elif ppt >= 2000: return 19
    elif ppt >= 1000: return 20
    elif ppt >= 500: return 21
    else: return 22

def calculate_life_zones_grid(df_precip_mean, gdf_stations):
    """Genera una grilla clasificada de zonas de vida interpolando Ppt y Altitud."""
    try:
        # Unir precipitación con coordenadas y altitud
        df = df_precip_mean.merge(
            gdf_stations[[Config.STATION_NAME_COL, Config.LATITUDE_COL, Config.LONGITUDE_COL, Config.ALTITUDE_COL]], 
            on=Config.STATION_NAME_COL, how='inner'
        )
        
        # Limpiar datos (altitud debe ser numérica)
        df[Config.ALTITUDE_COL] = pd.to_numeric(df[Config.ALTITUDE_COL], errors='coerce')
        df = df.dropna(subset=[Config.ALTITUDE_COL, Config.PRECIPITATION_COL])
        
        if len(df) < 4:
            return None, "Datos insuficientes para interpolar zonas de vida (mínimo 4 estaciones con altitud y precipitación)."

        # Crear Grilla
        grid_lon = np.linspace(df[Config.LONGITUDE_COL].min(), df[Config.LONGITUDE_COL].max(), 100)
        grid_lat = np.linspace(df[Config.LATITUDE_COL].min(), df[Config.LATITUDE_COL].max(), 100)
        GX, GY = np.meshgrid(grid_lon, grid_lat)
        
        points = df[[Config.LONGITUDE_COL, Config.LATITUDE_COL]].values
        values_ppt = df[Config.PRECIPITATION_COL].values
        values_alt = df[Config.ALTITUDE_COL].values
        
        # Interpolar Precipitación y Altitud
        grid_ppt = griddata(points, values_ppt, (GX, GY), method='linear')
        grid_alt = griddata(points, values_alt, (GX, GY), method='linear')
        
        # Clasificar cada celda vectorizada
        vectorized_classify = np.vectorize(classify_life_zone)
        grid_zones = vectorized_classify(grid_alt, grid_ppt)
        
        return (GX, GY, grid_zones), None

    except Exception as e:
        return None, str(e)
