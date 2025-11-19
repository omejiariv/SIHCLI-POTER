import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
from sqlalchemy import create_engine
from shapely import wkt
from modules.config import Config

@st.cache_data(show_spinner="Cargando datos desde la Base de Datos...", ttl=600)
def load_and_process_all_data():
    """
    Carga datos convirtiendo la geometría de PostGIS a Texto (WKT)
    para asegurar que Streamlit pueda leer las coordenadas sin errores binarios.
    """
    gdf_stations = None
    gdf_municipios = None
    df_long = None
    df_enso = None
    gdf_subcuencas = None

    try:
        if "DATABASE_URL" not in st.secrets:
            st.error("Falta 'DATABASE_URL' en los secretos de Streamlit.")
            return None, None, None, None, None

        DATABASE_URL = st.secrets["DATABASE_URL"]
        engine = create_engine(DATABASE_URL)

        # ---------------------------------------------------------
        # 1. CARGAR ESTACIONES (Estrategia WKT)
        # ---------------------------------------------------------
        try:
            # Usamos ST_AsText para que la BD nos de "POINT(-75.5 6.2)" en vez de binario ilegible
            sql_estaciones = """
            SELECT 
                id_estacion, 
                nom_est, 
                alt_est, 
                municipio, 
                depto_region, 
                ST_AsText(geom) as wkt_geom 
            FROM estaciones
            """
            
            # Leemos como DataFrame normal primero
            df_temp = pd.read_sql(sql_estaciones, engine)
            
            # Convertimos la columna de texto WKT a geometría real de Python
            if 'wkt_geom' in df_temp.columns:
                # Función segura para parsear
                def parse_wkt(x):
                    try:
                        return wkt.loads(x) if x else None
                    except:
                        return None

                df_temp['geometry'] = df_temp['wkt_geom'].apply(parse_wkt)
                
                # Convertir a GeoDataFrame
                gdf_stations = gpd.GeoDataFrame(df_temp, geometry='geometry', crs="EPSG:4326")
            else:
                gdf_stations = df_temp

            # --- NORMALIZACIÓN ---
            cols_map = {
                'id_estacion': Config.STATION_NAME_COL, 
                'nom_est': Config.STATION_NAME_COL, # Prioridad
                'alt_est': Config.ALTITUDE_COL,
                'municipio': Config.MUNICIPALITY_COL,
                'depto_region': Config.REGION_COL
            }
            gdf_stations = gdf_stations.rename(columns=cols_map)

            # Asegurar nombre de estación (Fallback)
            if Config.STATION_NAME_COL not in gdf_stations.columns:
                 if 'nombre' in gdf_stations.columns:
                     gdf_stations = gdf_stations.rename(columns={'nombre': Config.STATION_NAME_COL})

            # Limpiar duplicados de columnas
            gdf_stations = gdf_stations.loc[:, ~gdf_stations.columns.duplicated()]

            # EXTRAER LAT/LON (Para st.map)
            if 'geometry' in gdf_stations.columns:
                # Eliminar geometrías nulas antes de extraer coordenadas
                gdf_stations = gdf_stations.dropna(subset=['geometry'])
                gdf_stations['latitude'] = gdf_stations.geometry.y
                gdf_stations['longitude'] = gdf_stations.geometry.x
                
            # Limpiar espacios en nombres
            if Config.STATION_NAME_COL in gdf_stations.columns:
                gdf_stations[Config.STATION_NAME_COL] = gdf_stations[Config.STATION_NAME_COL].astype(str).str.strip()

        except Exception as e:
            st.error(f"Error cargando estaciones: {e}")
            gdf_stations = pd.DataFrame()

        # ---------------------------------------------------------
        # 2. CARGAR OTRAS TABLAS
        # ---------------------------------------------------------
        # Municipios (Solo placeholder para evitar error)
        gdf_municipios = pd.DataFrame() 
        gdf_subcuencas = pd.DataFrame()

        # ---------------------------------------------------------
        # 3. CARGAR PRECIPITACIÓN
        # ---------------------------------------------------------
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
            
            if Config.STATION_NAME_COL in df_long.columns:
                df_long[Config.STATION_NAME_COL] = df_long[Config.STATION_NAME_COL].astype(str).str.strip()

            df_long[Config.DATE_COL] = pd.to_datetime(df_long[Config.DATE_COL])
            df_long[Config.YEAR_COL] = df_long[Config.DATE_COL].dt.year
            df_long[Config.MONTH_COL] = df_long[Config.DATE_COL].dt.month
        except Exception:
            df_long = pd.DataFrame()

        # ---------------------------------------------------------
        # 4. CARGAR ENSO
        # ---------------------------------------------------------
        try:
            sql_enso = "SELECT * FROM indices_climaticos"
            df_enso = pd.read_sql(sql_enso, engine)
            if not df_enso.empty and Config.DATE_COL in df_enso.columns:
                df_enso[Config.DATE_COL] = pd.to_datetime(df_enso[Config.DATE_COL])
            elif not df_enso.empty and 'fecha' in df_enso.columns:
                 df_enso[Config.DATE_COL] = pd.to_datetime(df_enso['fecha'])
            
            if 'oni' in df_enso.columns: 
                df_enso = df_enso.rename(columns={'oni': Config.ENSO_ONI_COL})
        except:
             df_enso = pd.DataFrame()

        return gdf_stations, gdf_municipios, df_long, df_enso, gdf_subcuencas

    except Exception as e:
        st.error(f"Error crítico conectando a la Base de Datos: {e}")
        return None, None, None, None, None

def complete_series(df):
    return df
