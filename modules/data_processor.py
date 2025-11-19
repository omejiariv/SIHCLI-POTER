import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
from sqlalchemy import create_engine
from modules.config import Config

@st.cache_data(show_spinner="Cargando datos desde la Base de Datos...", ttl=600)
def load_and_process_all_data():
    """
    Carga segura de datos desde PostgreSQL/Supabase.
    Inicializa variables a None para evitar errores si la conexión falla.
    Previene columnas duplicadas que rompen la aplicación.
    """
    # Inicializar variables
    gdf_stations = None
    gdf_municipios = None
    df_long = None
    df_enso = None
    gdf_subcuencas = None

    try:
        # Verificar secretos
        if "DATABASE_URL" not in st.secrets:
            st.error("Falta 'DATABASE_URL' en los secretos de Streamlit.")
            return None, None, None, None, None

        DATABASE_URL = st.secrets["DATABASE_URL"]
        engine = create_engine(DATABASE_URL)

        # 1. Cargar Estaciones
        try:
            sql_estaciones = "SELECT * FROM estaciones"
            gdf_stations = gpd.read_postgis(sql_estaciones, engine, geom_col='geom', crs="EPSG:4326")
            
            # --- CORRECCIÓN DE DUPLICADOS DE COLUMNAS ---
            # 1. Renombrar columnas auxiliares primero
            cols_aux = {
                'alt_est': Config.ALTITUDE_COL,
                'municipio': Config.MUNICIPALITY_COL,
                'depto_region': Config.REGION_COL,
                'geom': 'geometry'
            }
            gdf_stations = gdf_stations.rename(columns=cols_aux)

            # 2. Resolver Nombre de Estación de forma inteligente
            # Solo renombramos si la columna destino NO existe aún, para evitar duplicados
            target_col = Config.STATION_NAME_COL
            
            if target_col not in gdf_stations.columns:
                if 'nom_est' in gdf_stations.columns:
                    gdf_stations = gdf_stations.rename(columns={'nom_est': target_col})
                elif 'id_estacion' in gdf_stations.columns:
                    gdf_stations = gdf_stations.rename(columns={'id_estacion': target_col})
                elif 'nombre' in gdf_stations.columns:
                    gdf_stations = gdf_stations.rename(columns={'nombre': target_col})
            
            # 3. LIMPIEZA FINAL DE DUPLICADOS (CRÍTICO)
            # Esto elimina cualquier columna duplicada accidentalmente (ej. dos 'nom_est')
            gdf_stations = gdf_stations.loc[:, ~gdf_stations.columns.duplicated()]

        except Exception as e:
            st.warning(f"No se pudieron cargar estaciones: {e}")
            gdf_stations = pd.DataFrame()

        # 2. Cargar Municipios
        try:
            sql_mun = "SELECT * FROM geometrias WHERE tipo_geometria = 'municipio'"
            gdf_municipios = gpd.read_postgis(sql_mun, engine, geom_col='geom', crs="EPSG:4326")
        except Exception:
            gdf_municipios = pd.DataFrame()

        # 3. Cargar Subcuencas
        try:
            sql_sub = "SELECT * FROM geometrias WHERE tipo_geometria = 'subcuenca'"
            gdf_subcuencas = gpd.read_postgis(sql_sub, engine, geom_col='geom', crs="EPSG:4326")
        except Exception:
            gdf_subcuencas = pd.DataFrame()

        # 4. Cargar Precipitación
        try:
            sql_ppt = "SELECT * FROM precipitacion_mensual"
            df_long = pd.read_sql(sql_ppt, engine)
            
            # Renombrado seguro para precipitación
            ppt_renames = {
                'id_estacion_fk': Config.STATION_NAME_COL,
                'fecha': Config.DATE_COL,
                'valor': Config.PRECIPITATION_COL,
                'precipitation': Config.PRECIPITATION_COL
            }
            df_long = df_long.rename(columns=ppt_renames)
            
            # Eliminar duplicados de columnas en df_long también
            df_long = df_long.loc[:, ~df_long.columns.duplicated()]

            df_long[Config.DATE_COL] = pd.to_datetime(df_long[Config.DATE_COL])
            df_long[Config.YEAR_COL] = df_long[Config.DATE_COL].dt.year
            df_long[Config.MONTH_COL] = df_long[Config.DATE_COL].dt.month
        except Exception:
            df_long = pd.DataFrame()

        # 5. Cargar ENSO
        try:
            sql_enso = "SELECT * FROM indices_climaticos"
            df_enso = pd.read_sql(sql_enso, engine)
            if not df_enso.empty and Config.DATE_COL in df_enso.columns:
                df_enso[Config.DATE_COL] = pd.to_datetime(df_enso[Config.DATE_COL])
        except Exception:
             df_enso = pd.DataFrame()

        return gdf_stations, gdf_municipios, df_long, df_enso, gdf_subcuencas

    except Exception as e:
        st.error(f"Error crítico conectando a la Base de Datos: {e}")
        return None, None, None, None, None

def complete_series(df):
    return df
