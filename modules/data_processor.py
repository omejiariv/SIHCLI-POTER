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
    Soluciona problemas de geometría activa y renombrado de columnas.
    """
    # Inicializar variables
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
        # 1. CARGAR ESTACIONES (Con corrección de geometría)
        # ---------------------------------------------------------
        try:
            sql_estaciones = "SELECT * FROM estaciones"
            # Cargamos con Pandas normal primero para inspeccionar columnas
            df_temp = pd.read_sql(sql_estaciones, engine)
            
            # Identificar columna de geometría (usualmente 'geom' o 'geometry')
            geom_col = 'geom' if 'geom' in df_temp.columns else 'geometry'
            
            # Convertir a GeoDataFrame explícitamente
            if geom_col in df_temp.columns:
                # Asegurarnos de que sea WKB o WKT
                from shapely import wkb, wkt
                
                # Función auxiliar para parsear geometría
                def parse_geom(x):
                    try:
                        if isinstance(x, str): return wkt.loads(x)
                        return wkb.loads(x, hex=True)
                    except: return None

                # Si viene como objeto (string/bytes), intentamos convertirlo
                if df_temp[geom_col].dtype == 'object':
                     df_temp[geom_col] = df_temp[geom_col].apply(parse_geom)

                gdf_stations = gpd.GeoDataFrame(df_temp, geometry=geom_col, crs="EPSG:4326")
                
                # Renombrar la columna de geometría a 'geometry' si no lo es
                if geom_col != 'geometry':
                    gdf_stations = gdf_stations.rename_geometry('geometry')
            else:
                # Si no hay geometría, seguimos como DataFrame normal (para evitar crash)
                st.warning("No se encontró columna de geometría en 'estaciones'.")
                gdf_stations = df_temp

            # --- RENOMBRADO DE COLUMNAS ---
            # Verificamos qué columnas existen antes de renombrar
            rename_map = {}
            if 'id_estacion' in gdf_stations.columns: rename_map['id_estacion'] = Config.STATION_NAME_COL
            if 'nom_est' in gdf_stations.columns: rename_map['nom_est'] = Config.STATION_NAME_COL # Prioridad
            if 'alt_est' in gdf_stations.columns: rename_map['alt_est'] = Config.ALTITUDE_COL
            if 'municipio' in gdf_stations.columns: rename_map['municipio'] = Config.MUNICIPALITY_COL
            if 'depto_region' in gdf_stations.columns: rename_map['depto_region'] = Config.REGION_COL

            gdf_stations = gdf_stations.rename(columns=rename_map)

            # Asegurar columnas críticas si faltan
            if Config.REGION_COL not in gdf_stations.columns:
                gdf_stations[Config.REGION_COL] = "Desconocido"
            if Config.MUNICIPALITY_COL not in gdf_stations.columns:
                gdf_stations[Config.MUNICIPALITY_COL] = "Desconocido"

            # Limpiar duplicados de columnas
            gdf_stations = gdf_stations.loc[:, ~gdf_stations.columns.duplicated()]

            # Extraer Lat/Lon para visualización
            if 'geometry' in gdf_stations.columns:
                gdf_stations['latitude'] = gdf_stations.geometry.y
                gdf_stations['longitude'] = gdf_stations.geometry.x
                gdf_stations[Config.LATITUDE_COL] = gdf_stations.geometry.y
                gdf_stations[Config.LONGITUDE_COL] = gdf_stations.geometry.x

            # Limpiar espacios
            if Config.STATION_NAME_COL in gdf_stations.columns:
                gdf_stations[Config.STATION_NAME_COL] = gdf_stations[Config.STATION_NAME_COL].astype(str).str.strip()

        except Exception as e:
            st.error(f"Error cargando estaciones: {e}")
            # Crear un DataFrame dummy para que la app no explote
            gdf_stations = pd.DataFrame({
                Config.STATION_NAME_COL: [],
                Config.REGION_COL: [],
                Config.MUNICIPALITY_COL: []
            })

        # ---------------------------------------------------------
        # 2. CARGAR PRECIPITACIÓN
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
        except Exception as e:
            st.warning(f"Error cargando precipitación: {e}")
            df_long = pd.DataFrame(columns=[Config.STATION_NAME_COL, Config.YEAR_COL, Config.PRECIPITATION_COL])

        # ---------------------------------------------------------
        # 3. CARGAR OTRAS TABLAS (Simplificado)
        # ---------------------------------------------------------
        try:
            df_enso = pd.read_sql("SELECT * FROM indices_climaticos", engine)
            if not df_enso.empty and 'fecha' in df_enso.columns:
                 df_enso[Config.DATE_COL] = pd.to_datetime(df_enso['fecha'])
                 if 'oni' in df_enso.columns: df_enso = df_enso.rename(columns={'oni': Config.ENSO_ONI_COL})
        except: df_enso = pd.DataFrame()
        
        # Municipios y Subcuencas (Placeholders por ahora para velocidad)
        gdf_municipios = pd.DataFrame()
        gdf_subcuencas = pd.DataFrame()

        return gdf_stations, gdf_municipios, df_long, df_enso, gdf_subcuencas

    except Exception as e:
        st.error(f"Error crítico de conexión: {e}")
        return None, None, None, None, None

def complete_series(df):
    return df
