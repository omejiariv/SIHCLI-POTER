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
    """
    Completa series mensuales rellenando huecos INTERNOS,
    detiene la extrapolación y asigna etiquetas de origen correctas.
    Preserva TODAS las columnas de metadatos (et_mmy, anomalia_oni, etc.).
    """
    
    # --- 1. Separar el DataFrame ---
    
    # Columnas clave para el merge
    merge_keys = [Config.STATION_NAME_COL, Config.DATE_COL]
    
    # Columnas a procesar (solo precipitación)
    cols_to_proc = [Config.PRECIPITATION_COL]
    
    # Columnas de metadatos (TODAS las demás)
    # Excluir 'origin' si existe, ya que la vamos a recalcular
    metadata_cols = [col for col in _df.columns if col not in cols_to_proc and col != Config.ORIGIN_COL]
    # Asegurarse que las claves de merge estén en la lista de metadatos
    for key in merge_keys:
        if key not in metadata_cols:
             st.error(f"Error Crítico en complete_series: Falta la columna clave {key}.")
             return _df # Devolver original si faltan claves

    # Crear el DataFrame de metadatos (contiene ENSO, et_mmy, año, mes, etc.)
    df_metadata = _df[metadata_cols].copy()
    
    # Crear el DataFrame de procesamiento (solo precip + claves)
    # Asegurarse de que las claves de merge estén en df_proc
    df_proc = _df[merge_keys + cols_to_proc].copy()
    df_proc[Config.DATE_COL] = pd.to_datetime(df_proc[Config.DATE_COL], errors='coerce')
    df_proc = df_proc.dropna(subset=[Config.DATE_COL, Config.STATION_NAME_COL])
    if df_proc.empty:
         return _df # Devolver original si no hay nada que procesar

    # --- 2. Función interna para rellenar huecos ---
    def fill_station_gaps(station_df):
        station_df = station_df.set_index(Config.DATE_COL).sort_index()
        if not station_df.index.is_unique:
            station_df = station_df[~station_df.index.duplicated(keep='first')]
        if station_df.empty: return None
        
        # Guardar el último dato real
        last_valid_date = station_df[Config.PRECIPITATION_COL].last_valid_index()
        
        start_date, end_date = station_df.index.min(), station_df.index.max()
        if pd.isna(start_date) or pd.isna(end_date): return None
            
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        # Reindexar *solo* la columna de precipitación
        df_resampled = station_df[[Config.PRECIPITATION_COL]].reindex(date_range)
        
        # 1. Guardar máscara de dónde estaban los datos originales
        original_data_mask = ~df_resampled[Config.PRECIPITATION_COL].isna()
        
        # 2. Interpolar SÓLO HUECOS INTERNOS
        df_resampled[Config.PRECIPITATION_COL] = df_resampled[Config.PRECIPITATION_COL].interpolate(
            method='linear', 
            limit_direction='both', 
            limit_area='inside'
        )
        
        # 3. Asignar Origen
        df_resampled[Config.ORIGIN_COL] = np.where(original_data_mask, 'Original', 'Completado')
        
        # 4. Eliminar filas que siguen siendo NaN (extrapolación no deseada)
        df_resampled.dropna(subset=[Config.PRECIPITATION_COL], inplace=True)
        
        df_resampled.reset_index(inplace=True)
        return df_resampled.rename(columns={'index': Config.DATE_COL})
    # --- Fin de la función interna ---

    # --- 3. Aplicar y Concatenar (Lógica de Bucle Corregida) ---
    completed_dfs_list = [] # Crear una lista vacía
    
    # Iterar manualmente sobre cada grupo de estación
    for station_name, station_group_df in df_proc.groupby(Config.STATION_NAME_COL):
        filled_df = fill_station_gaps(station_group_df)
        
        if filled_df is not None and not filled_df.empty:
            filled_df[Config.STATION_NAME_COL] = station_name # Re-añadir el nombre
            completed_dfs_list.append(filled_df)
    # --- FIN LÓGICA CORREGIDA ---

    if not completed_dfs_list:
        st.warning("No se pudieron completar series para las estaciones seleccionadas.")
        return _df # Devolver original si la completación falló

    # Concatenar la lista de DataFrames
    df_completed_core = pd.concat(completed_dfs_list, ignore_index=True)

    # --- 4. Unir (Merge) metadatos de vuelta ---
    df_final_completed = pd.DataFrame() # Inicializar
    if not df_metadata.empty:
        df_final_completed = pd.merge(
            df_metadata, # El DataFrame original con TODAS las otras columnas
            df_completed_core, # El DataFrame solo con los valores procesados
            on=merge_keys, # Unir por Estación y Fecha
            how='left' # Empezar con la metadata
        )
        
        # Rellenar 'origin' con 'Original' para las filas que no fueron interpoladas
        df_final_completed[Config.ORIGIN_COL] = df_final_completed[Config.ORIGIN_COL].fillna('Original')
        
        # Rellenar 'precipitation' (si es NaN) con la precipitación original
        # (df_metadata debería tener la precipitación original, así que necesitamos fusionar eso también)
        # Manera más simple: si 'precipitation_y' (del merge) es NaN, usar 'precipitation_x'
        # Esta parte es compleja, simplifiquemos:
        # La lógica de `dropna` en `fill_station_gaps` ya debería haber limpiado esto.
        # Vamos a asegurarnos de que el 'how' del merge sea 'left' para conservar
        # todas las filas de metadatos (incluyendo meses que no se interpolaron).
        
    else:
        df_final_completed = df_completed_core
        st.warning("No se pudo crear df_metadata, el resultado no tendrá metadatos extra.")

    # El DataFrame final ahora debería tener TODAS las columnas, con 'precipitation' y 'origin' actualizadas
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
        Config.STATION_NAME_COL, Config.MUNICIPALITY_COL, Config.REGION_COL,
        Config.ALTITUDE_COL, Config.CELL_COL, Config.LATITUDE_COL, Config.LONGITUDE_COL, Config.ET_COL
    ]
      
    existing_metadata_cols = [] # Usar este nombre consistentemente
    # Bucle para filtrar y depurar cada columna
    for col in station_metadata_cols:
        col_exists = col in gdf_stations.columns

        if col_exists:
            existing_metadata_cols.append(col) # Usar este nombre
        # Forzar inclusión si es ET_COL y no se encontró (último recurso)
        elif col == Config.ET_COL:
             st.warning(f"'{Config.ET_COL}' no se encontró con 'in', pero se intentará incluir forzadamente.")
             # Check again just in case
             if Config.ET_COL in gdf_stations.columns.astype(str): 
                  existing_metadata_cols.append(Config.ET_COL) # Usar este nombre
 
    # Asegurar que la columna clave (Station Name) esté presente si existe en el original
    if Config.STATION_NAME_COL not in existing_metadata_cols and Config.STATION_NAME_COL in gdf_stations.columns:
         existing_metadata_cols.insert(0, Config.STATION_NAME_COL) # Ponerla al principio si faltaba

    # Crear gdf_metadata_unique usando la lista verificada
    if Config.STATION_NAME_COL in existing_metadata_cols: # Usar este nombre
         # Asegurar que Station Name es la primera columna para drop_duplicates
         cols_for_unique = [Config.STATION_NAME_COL] + [c for c in existing_metadata_cols if c != Config.STATION_NAME_COL] # Usar este nombre
         gdf_metadata_unique = gdf_stations[cols_for_unique].drop_duplicates(subset=[Config.STATION_NAME_COL])
    else:
         st.error(f"Error Crítico: La columna clave '{Config.STATION_NAME_COL}' no se encontró en gdf_stations después de cargar.")
         gdf_metadata_unique = pd.DataFrame() # Crear DF vacío para evitar error

    # Drop potential duplicate columns from df_long BEFORE merging
    # Usar la lista verificada 'existing_metadata_cols'
    cols_to_drop_from_long = [c for c in existing_metadata_cols if c != Config.STATION_NAME_COL and c in df_long.columns] # Usar este nombre
    df_long.drop(columns=cols_to_drop_from_long, inplace=True, errors='ignore')

    # THE MERGE
    if not gdf_metadata_unique.empty: # Solo hacer merge si tenemos metadata válida
        df_long = pd.merge(df_long, gdf_metadata_unique, on=Config.STATION_NAME_COL, how='left')
        # Debug AFTER merge (optional but good to keep for now)
        # st.write("Debug: Columns in df_long AFTER merge:", df_long.columns.tolist()) # Keep commented unless debugging
        if Config.ET_COL in df_long.columns:
            # st.write(f"Debug: First 5 non-null values of {Config.ET_COL} after merge:", df_long[Config.ET_COL].dropna().head().tolist()) # Keep commented unless debugging
            pass # Placeholder if no action needed when ET_COL exists
        else:
            # st.warning(f"Debug: Column '{Config.ET_COL}' NOT FOUND in df_long after merge!") # Keep commented unless debugging
            pass # Placeholder if no action needed when ET_COL missing

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
         df_enso = pd.DataFrame() # Create empty DataFrame if no ENSO columns found
    # --- End ENSO Data Processing ---

    # Final return statement should be aligned with the start of the function blocks
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








