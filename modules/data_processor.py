# modules/data_processor.py

import streamlit as st
import pandas as pd
import geopandas as gpd
import os
import io
import numpy as np
import rasterio
import requests
from sqlalchemy import create_engine
from modules.config import Config
from modules.utils import standardize_numeric_column

# --- URL DE LA BASE DE DATOS ---
# ESTA ES LA CORRECCIÓN: Lee la variable desde los secretos de Streamlit
# Asegúrate de que en Streamlit Cloud > Settings > Secrets la variable se llame DATABASE_URL
try:
    DATABASE_URL = st.secrets["DATABASE_URL"]
except Exception as e:
    st.error("No se encontró la variable 'DATABASE_URL' en los secretos. Por favor configúrela en Streamlit Cloud.")
    st.stop()
# -----------------------------

# --- UTILS ---

@st.cache_data
def parse_spanish_dates(date_series):
    months_es_to_en = {
        'ene': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'abr': 'Apr',
        'may': 'May', 'jun': 'Jun', 'jul': 'Jul', 'ago': 'Aug',
        'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dic': 'Dec'
    }
    date_series_str = date_series.astype(str).str.lower()
    for es, en in months_es_to_en.items():
        date_series_str = date_series_str.str.replace(es, en, regex=False)
    return pd.to_datetime(date_series_str, format='%b-%y', errors='coerce')

@st.cache_data
def complete_series(_df):
    """
    Completa series mensuales (SOLO precipitación y origen),
    preservando TODAS las demás columnas (metadatos, ENSO, et_mmy, etc.).
    """
    
    # --- 1. Separar el DataFrame ---
    merge_keys = [Config.STATION_NAME_COL, Config.DATE_COL]
    value_col = Config.PRECIPITATION_COL
    metadata_cols = [col for col in _df.columns if col not in [value_col, Config.ORIGIN_COL]]
    
    for key in merge_keys:
        if key not in metadata_cols:
            st.error(f"Error Crítico en complete_series: Falta la columna clave {key}.")
            return _df 

    df_metadata = _df[metadata_cols].copy()
    df_proc = _df[merge_keys + [value_col]].copy()
    df_proc[Config.DATE_COL] = pd.to_datetime(df_proc[Config.DATE_COL], errors='coerce')
    df_proc = df_proc.dropna(subset=[Config.DATE_COL, Config.STATION_NAME_COL])
    if df_proc.empty:
         return _df 

    # --- 2. Función interna para rellenar huecos ---
    def fill_station_gaps(station_df_group):
        station_df = station_df_group.set_index(Config.DATE_COL).sort_index()
        if not station_df.index.is_unique:
            station_df = station_df[~station_df.index.duplicated(keep='first')]
        if station_df.empty: return None
        
        last_valid_date = station_df[value_col].last_valid_index()
        start_date, end_date = station_df.index.min(), station_df.index.max()
        if pd.isna(start_date) or pd.isna(end_date): return None
            
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
        df_resampled = station_df[[value_col]].reindex(date_range)
        original_data_mask = ~df_resampled[value_col].isna()
        
        df_resampled[value_col] = df_resampled[value_col].interpolate(
            method='linear', 
            limit_direction='both', 
            limit_area='inside'
        )
        
        df_resampled[Config.ORIGIN_COL] = np.where(original_data_mask, 'Original', 'Completado')
        df_resampled.dropna(subset=[value_col], inplace=True)
        df_resampled.reset_index(inplace=True)
        return df_resampled.rename(columns={'index': Config.DATE_COL})
    # --- Fin de la función interna ---

    # --- 3. Aplicar y Concatenar ---
    completed_dfs_list = []
    for station_name, station_group_df in df_proc.groupby(Config.STATION_NAME_COL):
        filled_df = fill_station_gaps(station_group_df)
        
        if filled_df is not None and not filled_df.empty:
            filled_df[Config.STATION_NAME_COL] = station_name 
            completed_dfs_list.append(filled_df)

    if not completed_dfs_list:
        st.warning("No se pudieron completar series para las estaciones seleccionadas.")
        return _df

    df_completed_core = pd.concat(completed_dfs_list, ignore_index=True)

    # --- 4. Unir (Merge) metadatos de vuelta ---
    df_final_completed = pd.merge(
        df_metadata, 
        df_completed_core, 
        on=merge_keys, 
        how='left' 
    )
    
    df_final_completed[Config.ORIGIN_COL] = df_final_completed[Config.ORIGIN_COL].fillna('Original')
    
    if Config.PRECIPITATION_COL not in df_final_completed.columns:
         df_final_completed[Config.PRECIPITATION_COL] = np.nan
    if Config.ORIGIN_COL not in df_final_completed.columns:
         df_final_completed[Config.ORIGIN_COL] = 'Original'

    return df_final_completed
    
# --- FUNCIÓN DE CARGA DE DATOS REESCRITA ---
@st.cache_data(show_spinner="Cargando datos desde la Base de Datos...")
def load_and_process_all_data():
    try:
        engine = create_engine(DATABASE_URL)
        
        # --- 1. CARGAR ESTACIONES ---
        sql_estaciones = "SELECT * FROM estaciones"
        gdf_stations = gpd.read_postgis(sql_estaciones, engine, geom_col='geom', crs="EPSG:4326")
        
        # --- DIAGNÓSTICO TEMPORAL (Verás esto en tu app si algo falla) ---
        # st.write("Columnas en DB:", gdf_stations.columns.tolist()) 
        
        # --- MAPEO ROBUSTO DE COLUMNAS ---
        # Ajusta las claves de la izquierda ('nombre_en_db') según tu tabla real en Supabase
        column_mapping = {
            'nom_est': Config.STATION_NAME_COL,    # Si en DB es 'nombre', cambia 'nom_est' por 'nombre'
            'id_estacion': 'id_estacion',          # Mantener ID si existe
            'alt_est': Config.ALTITUDE_COL,        # Si en DB es 'altitud', cambia 'alt_est' por 'altitud'
            'municipio': Config.MUNICIPALITY_COL,  # CRÍTICO: Verifica si en DB es 'mpio', 'ciudad', etc.
            'depto_region': Config.REGION_COL,
            'et_mmy': Config.ET_COL,
            'geom': 'geometry'
        }
        
        # Renombrar solo las columnas que existen
        gdf_stations = gdf_stations.rename(columns=column_mapping)
        
        # --- VALIDACIÓN CRÍTICA ---
        # Si tras el renombre la columna 'municipio' no existe, la creamos con un valor por defecto para evitar el Crash
        if Config.MUNICIPALITY_COL not in gdf_stations.columns:
            st.warning(f"⚠️ Columna '{Config.MUNICIPALITY_COL}' no encontrada en DB. Usando 'Desconocido'.")
            gdf_stations[Config.MUNICIPALITY_COL] = "Desconocido"
            
        if Config.STATION_NAME_COL not in gdf_stations.columns:
            # Fallback: intentar usar otra columna como nombre
            if 'id_estacion' in gdf_stations.columns:
                gdf_stations[Config.STATION_NAME_COL] = gdf_stations['id_estacion'].astype(str)
            else:
                st.error("Error Crítico: La tabla estaciones no tiene columna de nombre ni ID.")
                
        # Asegurar geometría
        if 'geometry' not in gdf_stations.columns and 'geom' in gdf_stations.columns:
             gdf_stations = gdf_stations.rename(columns={'geom': 'geometry'})
             
        gdf_stations = gdf_stations.set_geometry('geometry')
        gdf_stations[Config.LONGITUDE_COL] = gdf_stations.geometry.x
        gdf_stations[Config.LATITUDE_COL] = gdf_stations.geometry.y

        # ... (Resto del código para cargar Municipios, Subcuencas, df_long, etc.) ...
        
        return gdf_stations, gdf_municipios, df_long, df_enso, gdf_subcuencas

    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None, None, None, None, None

# --- OTRAS FUNCIONES ---

@st.cache_data
def extract_elevation_from_dem(gdf_stations, dem_data_source):
    if dem_data_source is None:
        return gdf_stations
    
    file_object = dem_data_source
    if hasattr(dem_data_source, 'name') and dem_data_source.name.lower().endswith('.tif'):
        try:
            file_object = io.BytesIO(dem_data_source.getvalue())
        except:
            pass
            
    try:
        with rasterio.open(file_object) as dem:
            coords = [(point.x, point.y) for point in gdf_stations.geometry]
            elevations = [val[0] for val in dem.sample(coords)]
            elevations = np.array(elevations)
            elevations[elevations < -1000] = np.nan
            gdf_stations[Config.ALTITUDE_COL] = elevations
            st.success("Elevación extraída del DEM para todas las estaciones.")
    except Exception as e:
        st.error(f"Error al procesar el archivo DEM: {e}")
        
    return gdf_stations

@st.cache_resource
def download_and_load_remote_dem(url):
    if not url:
        raise ValueError("La URL del servidor DEM no está configurada.")
    st.info(f"Simulación de descarga remota: {url}")
    return url



