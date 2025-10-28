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
    """Completa series mensuales rellenando huecos internos con interpolación lineal."""
    all_completed_dfs = []
    
    # Define columnas esenciales y de metadatos a intentar preservar
    id_cols = [Config.STATION_NAME_COL, Config.DATE_COL]
    value_col = Config.PRECIPITATION_COL
    origin_col = Config.ORIGIN_COL
    metadata_cols_to_keep = [
        Config.MUNICIPALITY_COL, Config.ALTITUDE_COL, Config.REGION_COL, 
        Config.CELL_COL, Config.ET_COL, # Incluir ET_COL aquí
        Config.LATITUDE_COL, Config.LONGITUDE_COL # Incluir Lat/Lon si son necesarios después
    ]
    # Filtrar solo las columnas que realmente existen en el DataFrame de entrada _df
    actual_metadata_cols = [col for col in metadata_cols_to_keep if col in _df.columns]
    cols_needed_input = id_cols + [value_col] + actual_metadata_cols
    
    # Trabajar con una copia limpia solo de las columnas necesarias
    df_input = _df[cols_needed_input].copy() 
    station_list = df_input[Config.STATION_NAME_COL].unique()
    
    # Pre-crear DataFrame de metadatos únicos para eficiencia (desde la copia limpia)
    df_metadata = df_input[[Config.STATION_NAME_COL] + actual_metadata_cols].drop_duplicates(subset=[Config.STATION_NAME_COL])

    for station in station_list:
        # Filtra datos de la estación desde df_input
        df_station = df_input[df_input[Config.STATION_NAME_COL] == station].copy()
        
        # Asegura tipo datetime, elimina filas sin fecha válida
        df_station[Config.DATE_COL] = pd.to_datetime(df_station[Config.DATE_COL], errors='coerce')
        df_station.dropna(subset=[Config.DATE_COL], inplace=True)
        if df_station.empty: continue 

        # Establece índice de fecha y elimina duplicados
        df_station.set_index(Config.DATE_COL, inplace=True)
        if not df_station.index.is_unique:
            df_station = df_station[~df_station.index.duplicated(keep='first')]
        if df_station.empty: continue 

        # Crear el rango completo de fechas ENTRE el mínimo y máximo real
        start_date, end_date = df_station.index.min(), df_station.index.max()
        if pd.isna(start_date) or pd.isna(end_date): continue 
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
             
        # Reindexar para crear el DataFrame completo con huecos (NaN)
        # Selecciona solo la columna de precipitación ANTES de reindexar
        df_precip_only = df_station[[value_col]]
        df_resampled = df_precip_only.reindex(date_range)

        # --- Lógica de Interpolación y Origen ---
        # 1. Guarda una máscara de dónde están los NaNs AHORA (antes de interpolar)
        nan_mask_before_interp = df_resampled[value_col].isna()

        # 2. Interpola usando 'linear' y limitando la dirección para rellenar solo huecos internos
        df_resampled[value_col] = df_resampled[value_col].interpolate(
            method='linear', 
            limit_direction='both', 
            limit_area='inside' 
        )

        # 3. Asigna origen basado en la máscara guardada
        df_resampled[origin_col] = np.where(nan_mask_before_interp, 'Completado', 'Original')
        # --- FIN Lógica ---
        
        # Añadir nombre de estación de nuevo
        df_resampled[Config.STATION_NAME_COL] = station
        
        # Añadir columnas de Año/Mes
        df_resampled[Config.YEAR_COL] = df_resampled.index.year
        df_resampled[Config.MONTH_COL] = df_resampled.index.month

        # Resetear índice
        df_resampled.reset_index(inplace=True)
        df_resampled.rename(columns={'index': Config.DATE_COL}, inplace=True)
        
        # Seleccionar solo las columnas procesadas explícitamente HASTA AHORA
        processed_cols = [Config.STATION_NAME_COL, Config.DATE_COL, value_col, origin_col, Config.YEAR_COL, Config.MONTH_COL]
        # Guardamos solo las columnas que hemos procesado explícitamente en el bucle
        # ¡IMPORTANTE! No incluir metadatos aquí, se añaden al final.
        all_completed_dfs.append(df_resampled[processed_cols]) 

    if not all_completed_dfs:
        st.warning("No se encontraron datos válidos para completar series.")
        return pd.DataFrame()
        
    # Concatenar todos los DataFrames procesados (solo con columnas esenciales)
    df_completed_core = pd.concat(all_completed_dfs, ignore_index=True)

    # Une los datos completados ('core') con la metadata usando el nombre de la estación
    if not df_metadata.empty and not df_completed_core.empty: # Check both are non-empty
        # Asegurarse que la columna de merge exista en ambos
        if Config.STATION_NAME_COL in df_completed_core.columns and Config.STATION_NAME_COL in df_metadata.columns:
            df_final_completed = pd.merge(df_completed_core, df_metadata, on=Config.STATION_NAME_COL, how='left')
        else:
             st.error("La columna clave para el merge de metadatos no existe en uno de los DataFrames.")
             df_final_completed = df_completed_core # Devolver sin metadatos si falla el merge
    elif not df_completed_core.empty: # If only core exists, use that
        df_final_completed = df_completed_core 
        st.warning("No se pudo crear df_metadata o estaba vacío, el resultado no tendrá metadatos extra.")
    else: # If core is also empty, return empty
         df_final_completed = pd.DataFrame()

    # (Añadir Año/Mes al DataFrame final - MOVIDO desde dentro del bucle)
    # Hacerlo aquí asegura que existan incluso si el merge falló
    if not df_final_completed.empty and Config.DATE_COL in df_final_completed.columns: 
        # Calcular solo si no existen ya (evita recalcular si estaban en df_completed_core)
        if Config.YEAR_COL not in df_final_completed.columns:
             df_final_completed[Config.YEAR_COL] = df_final_completed[Config.DATE_COL].dt.year
        if Config.MONTH_COL not in df_final_completed.columns:
             df_final_completed[Config.MONTH_COL] = df_final_completed[Config.DATE_COL].dt.month
    elif not df_final_completed.empty: # Si no está vacío pero falta fecha
         if Config.YEAR_COL not in df_final_completed.columns: df_final_completed[Config.YEAR_COL] = None
         if Config.MONTH_COL not in df_final_completed.columns: df_final_completed[Config.MONTH_COL] = None
    
    # Eliminar cualquier línea de depuración st.write/st.dataframe residual si la hubiera

    return df_final_completed
    
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

    # Define la lista de columnas deseadas (USING YOUR VARIABLE NAME)
    station_metadata_cols = [
        Config.STATION_NAME_COL, Config.MUNICIPALITY_COL, Config.ALTITUDE_COL, 
        Config.REGION_COL, Config.CELL_COL, Config.ET_COL 
        # Add LATITUDE_COL and LONGITUDE_COL if needed for other tabs, 
        # otherwise keep it minimal for the merge
        # Config.LATITUDE_COL, Config.LONGITUDE_COL 
    ]
    
    # Depuración: Ver columnas del input _df
    st.write("--- Debug `complete_series` Metadata Merge ---")
    st.write("Columnas disponibles en _df (input de complete_series):", _df.columns.tolist())
    st.write("Valor de Config.ET_COL:", Config.ET_COL)
    st.write("Columnas deseadas (station_metadata_cols):", station_metadata_cols) # Using your variable name
    
    # Filtrar columnas deseadas que REALMENTE existen en _df
    metadata_cols_to_use = []
    for col in station_metadata_cols: # Using your variable name
        col_exists = col in _df.columns
        st.write(f"Chequeando metadato: '{col}' -> Existe en _df? {col_exists}")
        if col_exists:
            metadata_cols_to_use.append(col)
        # Forzar inclusión si es ET_COL y no se encontró (último recurso)
        elif col == Config.ET_COL:
             st.warning(f"'{Config.ET_COL}' no se encontró con 'in', pero se intentará incluir forzadamente.")
             # Intentar añadirlo de todos modos si sabemos que debería estar
             # Check again casting column names to string maybe? Redundant if previous check failed reliably.
             # Let's trust the first check for now, but keep this logic in mind if needed.
             # if Config.ET_COL in _df.columns.astype(str): 
             #      metadata_cols_to_use.append(Config.ET_COL)

    st.write("Columnas de metadatos que se usarán para el merge:", metadata_cols_to_use)

    # Crear un DataFrame de metadatos únicos por estación
    if Config.STATION_NAME_COL in metadata_cols_to_use:
        # Asegurar STATION_NAME_COL primero
        unique_cols = [Config.STATION_NAME_COL] + [c for c in metadata_cols_to_use if c != Config.STATION_NAME_COL]
        # Make sure to select FROM _df (the original input with all columns)
        df_metadata = _df[unique_cols].drop_duplicates(subset=[Config.STATION_NAME_COL]) 
    else:
        st.error("La columna STATION_NAME_COL falta en los metadatos a usar!")
        df_metadata = pd.DataFrame() # Evitar error en merge

    # Une los datos completados ('core') con la metadata
    if not df_metadata.empty:
        df_final_completed = pd.merge(df_completed_core, df_metadata, on=Config.STATION_NAME_COL, how='left')
    else:
        df_final_completed = df_completed_core # No hacer merge si no hay metadata válida
        st.warning("No se pudo crear df_metadata, el resultado no tendrá metadatos extra.")

    # Depuración FINAL antes de retornar
    st.write("Columnas en df_final_completed ANTES de retornar:", df_final_completed.columns.tolist())
    st.write(f"¿Está '{Config.ET_COL}' en el resultado final?", Config.ET_COL in df_final_completed.columns)
    st.write("--- Fin Debug Metadata Merge ---")
    
    # Asegurar que la columna clave (Station Name) esté presente si existe en el original
    if Config.STATION_NAME_COL not in existing_metadata_cols and Config.STATION_NAME_COL in gdf_stations.columns:
         existing_metadata_cols.insert(0, Config.STATION_NAME_COL) # Ponerla al principio si faltaba

    # Crear gdf_metadata_unique usando la lista verificada
    if Config.STATION_NAME_COL in existing_metadata_cols:
         # Asegurar que Station Name es la primera columna para drop_duplicates
         cols_for_unique = [Config.STATION_NAME_COL] + [c for c in existing_metadata_cols if c != Config.STATION_NAME_COL]
         gdf_metadata_unique = gdf_stations[cols_for_unique].drop_duplicates(subset=[Config.STATION_NAME_COL])
    else:
         st.error(f"Error Crítico: La columna clave '{Config.STATION_NAME_COL}' no se encontró en gdf_stations después de cargar.")
         gdf_metadata_unique = pd.DataFrame() # Crear DF vacío para evitar error

    # Drop potential duplicate columns from df_long BEFORE merging
    # Usar la lista verificada 'existing_metadata_cols'
    cols_to_drop_from_long = [c for c in existing_metadata_cols if c != Config.STATION_NAME_COL and c in df_long.columns]
    df_long.drop(columns=cols_to_drop_from_long, inplace=True, errors='ignore')

    # THE MERGE
    if not gdf_metadata_unique.empty: # Solo hacer merge si tenemos metadata válida
        df_long = pd.merge(df_long, gdf_metadata_unique, on=Config.STATION_NAME_COL, how='left')
        # Debug AFTER merge (optional but good to keep for now)
        st.write("Debug: Columns in df_long AFTER merge:", df_long.columns.tolist())
        if Config.ET_COL in df_long.columns:
            st.write(f"Debug: First 5 non-null values of {Config.ET_COL} after merge:", df_long[Config.ET_COL].dropna().head().tolist())
        else:
            st.warning(f"Debug: Column '{Config.ET_COL}' NOT FOUND in df_long after merge!")

    # --- FIN BLOQUE CON DEBUG DETALLADO ---

    # --- ENSO Data Processing (Mantener esta parte) ---
    enso_cols = ['id', Config.DATE_COL, Config.ENSO_ONI_COL, 'temp_sst', 'temp_media']
    # Check against df_precip_raw columns as originally intended
    existing_enso_cols = [col for col in enso_cols if col in df_precip_raw.columns] 
    if existing_enso_cols: # Only proceed if columns exist in the original precip file
        df_enso = df_precip_raw[existing_enso_cols].drop_duplicates().copy()

        if Config.DATE_COL in df_enso.columns:
            df_enso[Config.DATE_COL] = parse_spanish_dates(df_enso[Config.DATE_COL])
            df_enso.dropna(subset=[Config.DATE_COL], inplace=True)
        
        for col in [Config.ENSO_ONI_COL, 'temp_sst', 'temp_media']:
            if col in df_enso.columns:
                df_enso[col] = standardize_numeric_column(df_enso[col])
    else:
         st.warning("No se encontraron columnas ENSO en el archivo de precipitación original. df_enso estará vacío.")
         df_enso = pd.DataFrame()
        
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
