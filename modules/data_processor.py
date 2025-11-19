import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
from sqlalchemy import create_engine
from modules.config import Config

@st.cache_data(show_spinner="Cargando datos desde la Base de Datos...", ttl=600)
def load_and_process_all_data():
    """
    Carga datos desde PostgreSQL/Supabase y prepara las coordenadas
    explícitas (lat/lon) que Streamlit necesita para los mapas.
    """
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
            
            # --- CORRECCIÓN CRÍTICA: EXTRAER LAT/LON ---
            # Streamlit .map() necesita columnas explícitas 'latitude' y 'longitude'
            # Extraemos esto de la columna geométrica
            gdf_stations['latitude'] = gdf_stations.geometry.y
            gdf_stations['longitude'] = gdf_stations.geometry.x
            
            # También aseguramos las columnas de configuración por si acaso se usan en otros lados
            gdf_stations[Config.LATITUDE_COL] = gdf_stations.geometry.y
            gdf_stations[Config.LONGITUDE_COL] = gdf_stations.geometry.x

            # Normalizar nombres de columnas
            cols_map = {
                'alt_est': Config.ALTITUDE_COL,
                'municipio': Config.MUNICIPALITY_COL,
                'depto_region': Config.REGION_COL
            }
            gdf_stations = gdf_stations.rename(columns=cols_map)

            # Resolver nombre de estación
            target_col = Config.STATION_NAME_COL
            if target_col not in gdf_stations.columns:
                if 'nom_est' in gdf_stations.columns:
                    gdf_stations = gdf_stations.rename(columns={'nom_est': target_col})
                elif 'id_estacion' in gdf_stations.columns:
                    gdf_stations = gdf_stations.rename(columns={'id_estacion': target_col})
                elif 'nombre' in gdf_stations.columns:
                    gdf_stations = gdf_stations.rename(columns={'nombre': target_col})
            
            # Limpieza final de duplicados
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
            
            ppt_renames = {
                'id_estacion_fk': Config.STATION_NAME_COL,
                'fecha': Config.DATE_COL,
                'valor': Config.PRECIPITATION_COL,
                'precipitation': Config.PRECIPITATION_COL
            }
            df_long = df_long.rename(columns=ppt_renames)
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
