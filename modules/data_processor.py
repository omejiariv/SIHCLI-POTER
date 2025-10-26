# modules/data_processor.py

import streamlit as st
import pandas as pd
import geopandas as gpd
import zipfile
import tempfile
import os
import io
import numpy as np
import rasterio
import requests
from modules.config import Config
from modules.utils import standardize_numeric_column

# --- UTILS ---

@st.cache_data
def load_geojson_from_github(url):
    try:
        return gpd.read_file(url)
    except Exception as e:
        st.error(f"No se pudo cargar el GeoJSON desde la URL: {e}")
        return None

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
def load_csv_data(file_uploader_object, sep=";", lower_case=True):
    if file_uploader_object is None: return None
    try:
        content = file_uploader_object.getvalue()
        if not content.strip():
            st.error(f"El archivo '{file_uploader_object.name}' parece estar vacío.")
            return None
    except Exception as e:
        st.error(f"Error al leer el archivo '{file_uploader_object.name}': {e}")
        return None
    encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(io.BytesIO(content), sep=sep, encoding=encoding)
            if lower_case:
                df.columns = [col.strip().lower() for col in df.columns]
            return df
        except Exception:
            continue
    st.error(f"No se pudo decodificar el archivo '{file_uploader_object.name}' con las codificaciones probadas.")
    return None

@st.cache_data
def load_shapefile(file_uploader_object):
    if file_uploader_object is None: return None
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(file_uploader_object, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            shp_files = [f for f in os.listdir(temp_dir) if f.endswith('.shp')]
            if not shp_files:
                st.error("No se encontró un archivo .shp en el archivo .zip.")
                return None
            shp_path = os.path.join(temp_dir, shp_files[0])
            gdf = gpd.read_file(shp_path)
            gdf.columns = gdf.columns.str.strip().str.lower()
            if gdf.crs is None:
                gdf.set_crs("EPSG:4686", inplace=True) # Origen Nacional
            return gdf.to_crs("EPSG:4326")
    except Exception as e:
        st.error(f"Error al procesar el shapefile: {e}")
        return None


@st.cache_data
def complete_series(_df):
    """Completa las series de tiempo mensuales para cada estación mediante interpolación."""
    all_completed_dfs = []
    station_list = _df[Config.STATION_NAME_COL].unique()
    
    metadata_cols_to_keep = [
        col for col in [Config.MUNICIPALITY_COL, Config.ALTITUDE_COL, Config.REGION_COL, Config.CELL_COL]
        if col in _df.columns
    ]

    # La lógica de la barra de progreso se manejará fuera de la función cacheada si es necesario,
    # pero para estabilidad y rendimiento, es mejor omitirla aquí.
    for i, station in enumerate(station_list):
        df_station = _df[_df[Config.STATION_NAME_COL] == station].copy()
        station_metadata = None
        if not df_station.empty and metadata_cols_to_keep:
            station_metadata = df_station[metadata_cols_to_keep].iloc[0]

        # Convertir a datetime ANTES de establecer el índice
        df_station[Config.DATE_COL] = pd.to_datetime(df_station[Config.DATE_COL])
        
        # Guardar el índice original ANTES de reindexar
        original_index = df_station.set_index(Config.DATE_COL).index
        
        # Establecer el índice para la lógica de reindexación y limpieza
        df_station.set_index(Config.DATE_COL, inplace=True)
        
        if not df_station.index.is_unique:
            df_station = df_station[~df_station.index.duplicated(keep='first')]
            original_index = df_station.index # Actualizar el índice original si se eliminaron duplicados
            
        # Crear el rango completo de fechas
        if not df_station.empty:
             start_date = df_station.index.min()
             end_date = df_station.index.max()
             if pd.isna(start_date) or pd.isna(end_date): continue # Saltar si no hay fechas válidas
             date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
             
             # Reindexar para crear el DataFrame completo con huecos (NaN)
             df_resampled = df_station.reindex(date_range)
        else:
             continue # Saltar si la estación no tiene datos

        # Interpolar la precipitación usando el método lineal (más estable)
        df_resampled[Config.PRECIPITATION_COL] = \
            df_resampled[Config.PRECIPITATION_COL].interpolate(method='linear')
            
        # --- INICIO DEL NUEVO BLOQUE DE ASIGNACIÓN DE ORIGEN ---
        
        # 1. Primero, marca TODAS las filas del DataFrame reindexado como 'Completado'
        df_resampled[Config.ORIGIN_COL] = 'Completado' 
        
        # 2. Luego, usa el índice original para sobrescribir y marcar solo esas como 'Original'
        #    Usamos intersect para asegurarnos de que solo marcamos índices que existen en df_resampled
        indices_a_marcar_original = original_index.intersection(df_resampled.index)
        df_resampled.loc[indices_a_marcar_original, Config.ORIGIN_COL] = 'Original'
        
        # --- FIN DEL NUEVO BLOQUE ---
        
        # Asignar nombre de estación (importante después de reindexar)
        df_resampled[Config.STATION_NAME_COL] = station
        
        # Volver a añadir metadatos si existían
        if station_metadata is not None:
            for col_name, value in station_metadata.items():
                df_resampled[col_name] = value

        # Añadir columnas de año y mes
        df_resampled[Config.YEAR_COL] = df_resampled.index.year
        df_resampled[Config.MONTH_COL] = df_resampled.index.month
        
        # Resetear el índice para devolver un DataFrame plano
        df_resampled.reset_index(inplace=True)
        df_resampled.rename(columns={'index': Config.DATE_COL}, inplace=True)
        all_completed_dfs.append(df_resampled)
        
    return pd.concat(all_completed_dfs, ignore_index=True) if all_completed_dfs else pd.DataFrame()

@st.cache_data
def load_and_process_all_data(uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile, uploaded_file_parquet):
    df_stations_raw = load_csv_data(uploaded_file_mapa)
    df_precip_raw = load_csv_data(uploaded_file_precip)
    gdf_municipios = load_shapefile(uploaded_zip_shapefile)
    gdf_subcuencas = load_geojson_from_github(Config.URL_SUBCUENCAS_GEOJSON)
    
    # CORRECCIÓN 1: Devolver 5 valores en caso de error
    if any(df is None for df in [df_stations_raw, df_precip_raw, gdf_municipios, gdf_subcuencas]):
        st.error("Fallo en la carga de uno o más archivos base. El proceso no puede continuar.")
        return None, None, None, None, None

    lon_col = next((col for col in df_stations_raw.columns if 'longitud' in col.lower() or 'lon' in col.lower()), None)
    lat_col = next((col for col in df_stations_raw.columns if 'latitud' in col.lower() or 'lat' in col.lower()), None)

    # CORRECCIÓN 2: Devolver 5 valores en caso de error
    if not all([lon_col, lat_col]):
        st.error("No se encontraron columnas de longitud y/o latitud en el archivo de estaciones.")
        return None, None, None, None, None

    df_stations_raw[lon_col] = standardize_numeric_column(df_stations_raw[lon_col])
    df_stations_raw[lat_col] = standardize_numeric_column(df_stations_raw[lat_col])

    for col in [Config.MUNICIPALITY_COL, Config.REGION_COL]:
        if col in df_stations_raw.columns:
            df_stations_raw[col] = df_stations_raw[col].astype(str).str.strip().replace('nan', 'Sin Dato')

    if Config.ET_COL in df_stations_raw.columns:
        df_stations_raw[Config.ET_COL] = standardize_numeric_column(df_stations_raw[Config.ET_COL])

    df_stations_raw.dropna(subset=[lon_col, lat_col], inplace=True)

    gdf_stations = gpd.GeoDataFrame(
        df_stations_raw,
        geometry=gpd.points_from_xy(df_stations_raw[lon_col], df_stations_raw[lat_col]),
        crs="EPSG:4326"
    )

    gdf_stations[Config.LONGITUDE_COL] = gdf_stations.geometry.x
    gdf_stations[Config.LATITUDE_COL] = gdf_stations.geometry.y

    if Config.ALTITUDE_COL in gdf_stations.columns:
        gdf_stations[Config.ALTITUDE_COL] = standardize_numeric_column(gdf_stations[Config.ALTITUDE_COL])

    # --- INICIO DEL REEMPLAZO ---
    st.info("Cargando datos de precipitación desde Parquet (¡rápido!)...")

    if uploaded_file_parquet is None:
        st.error("Por favor, carga el archivo 'datos_precipitacion_largos.parquet' para continuar.")
        return None, None, None, None, None

    df_long = uploaded_file_parquet
    
    # Renombramos la columna del Parquet a la que usa la app
    df_long.rename(columns={'precipitacion_mm': Config.PRECIPITATION_COL}, inplace=True)
    # --- FIN DEL REEMPLAZO ---

    cols_to_numeric = [Config.ENSO_ONI_COL, 'temp_sst', 'temp_media', Config.SOI_COL, Config.IOD_COL]
    for col in cols_to_numeric:
        if col in df_long.columns:
            df_long[col] = standardize_numeric_column(df_long[col])

    # df_long.dropna(subset=[Config.PRECIPITATION_COL], inplace=True)

    df_long[Config.DATE_COL] = parse_spanish_dates(df_long[Config.DATE_COL])
    df_long.dropna(subset=[Config.DATE_COL], inplace=True)

    df_long[Config.ORIGIN_COL] = 'Original'
    df_long[Config.YEAR_COL] = df_long[Config.DATE_COL].dt.year
    df_long[Config.MONTH_COL] = df_long[Config.DATE_COL].dt.month

    id_estacion_col_name = 'id_estacio'
    if id_estacion_col_name not in gdf_stations.columns:
        st.error(f"No se encontró la columna '{id_estacion_col_name}' en el archivo de estaciones.")
        return None, None, None, None

    gdf_stations[id_estacion_col_name] = gdf_stations[id_estacion_col_name].astype(str).str.strip()
    df_long['id_estacion'] = df_long['id_estacion'].astype(str).str.strip()

    station_mapping = gdf_stations.set_index(id_estacion_col_name)[Config.STATION_NAME_COL].to_dict()
    df_long[Config.STATION_NAME_COL] = df_long['id_estacion'].map(station_mapping)
    df_long.dropna(subset=[Config.STATION_NAME_COL], inplace=True)

    station_metadata_cols = [
        Config.STATION_NAME_COL, Config.MUNICIPALITY_COL, Config.REGION_COL,
        Config.ALTITUDE_COL, Config.CELL_COL, Config.LATITUDE_COL, Config.LONGITUDE_COL, Config.ET_COL
    ]
    existing_metadata_cols = [col for col in station_metadata_cols if col in gdf_stations.columns]
    gdf_metadata_unique = gdf_stations[existing_metadata_cols].drop_duplicates(subset=[Config.STATION_NAME_COL])

    cols_to_drop_from_long = [c for c in existing_metadata_cols if c != Config.STATION_NAME_COL and c in df_long.columns]
    df_long.drop(columns=cols_to_drop_from_long, inplace=True, errors='ignore')

    df_long = pd.merge(df_long, gdf_metadata_unique, on=Config.STATION_NAME_COL, how='left')

    enso_cols = ['id', Config.DATE_COL, Config.ENSO_ONI_COL, 'temp_sst', 'temp_media']
    existing_enso_cols = [col for col in enso_cols if col in df_precip_raw.columns]
    df_enso = df_precip_raw[existing_enso_cols].drop_duplicates().copy()

    if Config.DATE_COL in df_enso.columns:
        df_enso[Config.DATE_COL] = parse_spanish_dates(df_enso[Config.DATE_COL])
        df_enso.dropna(subset=[Config.DATE_COL], inplace=True)
    
    for col in [Config.ENSO_ONI_COL, 'temp_sst', 'temp_media']:
        if col in df_enso.columns:
            df_enso[col] = standardize_numeric_column(df_enso[col])

    return gdf_stations, gdf_municipios, df_long, df_enso, gdf_subcuencas

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
        st.error(f"Error al procesar el archivo DEM. Asegúrese de que es un GeoTIFF válido y el CRS coincide: {e}")
        st.session_state[f'original_{Config.ALTITUDE_COL}'] = st.session_state.get(f'original_{Config.ALTITUDE_COL}', None)
        if Config.ALTITUDE_COL in gdf_stations.columns and st.session_state[f'original_{Config.ALTITUDE_COL}'] is not None:
            gdf_stations[Config.ALTITUDE_COL] = st.session_state[f'original_{Config.ALTITUDE_COL}']
            
    return gdf_stations

@st.cache_resource
def download_and_load_remote_dem(url):
    if not url:
        raise ValueError("La URL del servidor DEM no está configurada.")
    st.info(f"Simulación de descarga remota. En un entorno real, se usaría un archivo temporal. Usando '{url}' como marcador.")
    return url

@st.cache_data
def load_parquet_from_url(url):
    if not url: return None
    try:
        return pd.read_parquet(url)
    except Exception as e:
        st.error(f"No se pudo cargar el Parquet desde la URL: {e}")
        return None



