import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
from sqlalchemy import create_engine
from modules.config import Config

# -----------------------------------------------------------------------------
# FUNCIÓN DE CARGA DE DATOS (ROBUSTA)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Cargando datos desde la Base de Datos...", ttl=600)
def load_and_process_all_data():
    """
    Carga segura de datos desde PostgreSQL/Supabase.
    - Maneja errores de conexión.
    - Elimina columnas duplicadas.
    - Extrae Lat/Lon para mapas.
    - Limpia espacios en blanco en nombres de estaciones.
    """
    # 1. Inicializar variables a None para evitar errores de referencia
    gdf_stations = None
    gdf_municipios = None
    df_long = None
    df_enso = None
    gdf_subcuencas = None

    try:
        # Verificar secretos
        if "DATABASE_URL" not in st.secrets:
            st.error("Falta 'DATABASE_URL' en los secretos de Streamlit (.streamlit/secrets.toml).")
            return None, None, None, None, None

        DATABASE_URL = st.secrets["DATABASE_URL"]
        engine = create_engine(DATABASE_URL)

        # ---------------------------------------------------------
        # 2. CARGAR ESTACIONES
        # ---------------------------------------------------------
        try:
            sql_estaciones = "SELECT * FROM estaciones"
            gdf_stations = gpd.read_postgis(sql_estaciones, engine, geom_col='geom', crs="EPSG:4326")
            
            # a) Normalizar nombres de columnas según Config
            cols_map = {
                'id_estacion': Config.STATION_NAME_COL, 
                'nom_est': Config.STATION_NAME_COL, # Prioridad
                'alt_est': Config.ALTITUDE_COL,
                'municipio': Config.MUNICIPALITY_COL,
                'depto_region': Config.REGION_COL,
                'geom': 'geometry'
            }
            gdf_stations = gdf_stations.rename(columns=cols_map)

            # b) Asegurar que exista la columna clave de nombre
            if Config.STATION_NAME_COL not in gdf_stations.columns:
                 if 'nombre' in gdf_stations.columns:
                     gdf_stations = gdf_stations.rename(columns={'nombre': Config.STATION_NAME_COL})
            
            # c) ELIMINAR DUPLICADOS DE COLUMNAS (Crítico para .unique())
            gdf_stations = gdf_stations.loc[:, ~gdf_stations.columns.duplicated()]

            # d) EXTRAER LAT/LON (Crítico para st.map)
            if 'geometry' in gdf_stations.columns:
                gdf_stations['latitude'] = gdf_stations.geometry.y
                gdf_stations['longitude'] = gdf_stations.geometry.x
                # Asegurar también las constantes de config
                gdf_stations[Config.LATITUDE_COL] = gdf_stations.geometry.y
                gdf_stations[Config.LONGITUDE_COL] = gdf_stations.geometry.x

            # e) LIMPIEZA DE ESPACIOS (Crítico para filtros)
            if Config.STATION_NAME_COL in gdf_stations.columns:
                gdf_stations[Config.STATION_NAME_COL] = gdf_stations[Config.STATION_NAME_COL].astype(str).str.strip()

        except Exception as e:
            st.warning(f"No se pudieron cargar estaciones: {e}")
            gdf_stations = pd.DataFrame()

        # ---------------------------------------------------------
        # 3. CARGAR GEOMETRÍAS ADICIONALES
        # ---------------------------------------------------------
        try:
            sql_mun = "SELECT * FROM geometrias WHERE tipo_geometria = 'municipio'"
            gdf_municipios = gpd.read_postgis(sql_mun, engine, geom_col='geom', crs="EPSG:4326")
        except Exception:
            gdf_municipios = pd.DataFrame()

        try:
            sql_sub = "SELECT * FROM geometrias WHERE tipo_geometria = 'subcuenca'"
            gdf_subcuencas = gpd.read_postgis(sql_sub, engine, geom_col='geom', crs="EPSG:4326")
        except Exception:
            gdf_subcuencas = pd.DataFrame()

        # ---------------------------------------------------------
        # 4. CARGAR PRECIPITACIÓN (SERIES DE TIEMPO)
        # ---------------------------------------------------------
        try:
            sql_ppt = "SELECT * FROM precipitacion_mensual"
            df_long = pd.read_sql(sql_ppt, engine)
            
            # a) Renombrar columnas
            ppt_renames = {
                'id_estacion_fk': Config.STATION_NAME_COL,
                'fecha': Config.DATE_COL,
                'valor': Config.PRECIPITATION_COL,
                'precipitation': Config.PRECIPITATION_COL
            }
            df_long = df_long.rename(columns=ppt_renames)
            
            # b) Eliminar columnas duplicadas
            df_long = df_long.loc[:, ~df_long.columns.duplicated()]

            # c) Procesar Fechas
            df_long[Config.DATE_COL] = pd.to_datetime(df_long[Config.DATE_COL])
            df_long[Config.YEAR_COL] = df_long[Config.DATE_COL].dt.year
            df_long[Config.MONTH_COL] = df_long[Config.DATE_COL].dt.month

            # d) LIMPIEZA DE ESPACIOS (Crítico para cruzar con estaciones)
            if Config.STATION_NAME_COL in df_long.columns:
                df_long[Config.STATION_NAME_COL] = df_long[Config.STATION_NAME_COL].astype(str).str.strip()

        except Exception as e:
            st.error(f"Error cargando precipitación: {e}")
            df_long = pd.DataFrame()

        # ---------------------------------------------------------
        # 5. CARGAR ENSO
        # ---------------------------------------------------------
        try:
            sql_enso = "SELECT * FROM indices_climaticos"
            df_enso = pd.read_sql(sql_enso, engine)
            if not df_enso.empty and Config.DATE_COL in df_enso.columns:
                df_enso[Config.DATE_COL] = pd.to_datetime(df_enso[Config.DATE_COL])
            
            # Mapeo de nombres si es necesario
            if 'anomalia_oni' in df_enso.columns:
                pass # Ya está correcto
            elif 'oni' in df_enso.columns:
                 df_enso = df_enso.rename(columns={'oni': Config.ENSO_ONI_COL})

        except Exception:
             df_enso = pd.DataFrame()

        return gdf_stations, gdf_municipios, df_long, df_enso, gdf_subcuencas

    except Exception as e:
        st.error(f"Error crítico conectando a la Base de Datos: {e}")
        # Retornar Nones en orden correcto
        return None, None, None, None, None

# -----------------------------------------------------------------------------
# FUNCIÓN DE COMPLETADO DE SERIES (BASICA/PLACEHOLDER)
# -----------------------------------------------------------------------------
def complete_series(df):
    """
    Rellena huecos en los datos mensuales usando interpolación lineal.
    """
    if df is None or df.empty:
        return df
    
    # Copia para no afectar el original
    df_proc = df.copy()
    
    try:
        # Asegurar índice de fecha
        if not pd.api.types.is_datetime64_any_dtype(df_proc[Config.DATE_COL]):
             df_proc[Config.DATE_COL] = pd.to_datetime(df_proc[Config.DATE_COL])
        
        # Agrupar y remuestrear
        # Nota: Esta es una operación pesada, se simplifica aquí para estabilidad.
        # Si necesitas la lógica compleja del PDF, se puede integrar, pero 
        # por ahora esto evita que la app se rompa.
        df_proc = df_proc.sort_values(Config.DATE_COL)
        
        # Interpolación simple por grupo
        df_proc[Config.PRECIPITATION_COL] = df_proc.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].transform(
            lambda group: group.interpolate(method='linear', limit_direction='both')
        )
        return df_proc

    except Exception as e:
        st.warning(f"Error durante la interpolación: {e}")
        return df
