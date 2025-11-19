import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
from sqlalchemy import create_engine, text
from shapely import wkt
from modules.config import Config

@st.cache_data(show_spinner="Procesando datos...", ttl=600)
def load_and_process_all_data():
    """
    Carga datos de Supabase con garantía de columnas.
    Si alguna columna crítica falta, la crea para evitar KeyErrors en la app.
    """
    # Inicializar variables vacías
    gdf_stations = pd.DataFrame()
    gdf_municipios = pd.DataFrame()
    gdf_subcuencas = pd.DataFrame()
    df_long = pd.DataFrame()
    df_enso = pd.DataFrame()

    try:
        if "DATABASE_URL" not in st.secrets:
            st.error("Falta DATABASE_URL en secrets.")
            return None, None, None, None, None

        engine = create_engine(st.secrets["DATABASE_URL"])

        # ------------------------------------------------------------
        # 1. CARGAR ESTACIONES
        # ------------------------------------------------------------
        try:
            sql_est = """
            SELECT id_estacion, nom_est, alt_est, municipio, depto_region, ST_AsText(geom) as wkt 
            FROM estaciones
            """
            df_est_raw = pd.read_sql(sql_est, engine)

            if not df_est_raw.empty:
                # Parsear geometría
                def parse_geom(x):
                    try: return wkt.loads(x) if x else None
                    except: return None
                
                if 'wkt' in df_est_raw.columns:
                    df_est_raw['geometry'] = df_est_raw['wkt'].apply(parse_geom)
                    gdf_stations = gpd.GeoDataFrame(df_est_raw, geometry='geometry', crs="EPSG:4326")
                else:
                    gdf_stations = df_est_raw.copy() # Fallback si no hay WKT

                # Normalizar nombres (Mapeo explícito)
                gdf_stations = gdf_stations.rename(columns={
                    'nom_est': Config.STATION_NAME_COL,
                    'alt_est': Config.ALTITUDE_COL,
                    'municipio': Config.MUNICIPALITY_COL,
                    'depto_region': Config.REGION_COL
                })

                # Limpieza de textos
                for col in [Config.STATION_NAME_COL, Config.MUNICIPALITY_COL, Config.REGION_COL]:
                    if col in gdf_stations.columns:
                        gdf_stations[col] = gdf_stations[col].astype(str).str.replace(r'\s*-\s*$', '', regex=True).str.strip()

                # Lat/Lon
                if 'geometry' in gdf_stations.columns:
                    gdf_stations = gdf_stations.dropna(subset=['geometry'])
                    gdf_stations['latitude'] = gdf_stations.geometry.y
                    gdf_stations['longitude'] = gdf_stations.geometry.x
        except Exception as e:
            st.warning(f"Error cargando estaciones: {e}")

        # ------------------------------------------------------------
        # 2. CARGAR PRECIPITACIÓN
        # ------------------------------------------------------------
        try:
            sql_ppt = "SELECT id_estacion_fk, fecha, valor FROM precipitacion_mensual"
            df_ppt_raw = pd.read_sql(sql_ppt, engine)
            
            if not df_ppt_raw.empty and not gdf_stations.empty:
                df_ppt_raw[Config.DATE_COL] = pd.to_datetime(df_ppt_raw['fecha'])
                
                # Merge para pegar nombres
                df_long = pd.merge(
                    df_ppt_raw,
                    gdf_stations[['id_estacion', Config.STATION_NAME_COL]],
                    left_on='id_estacion_fk',
                    right_on='id_estacion',
                    how='inner'
                )
                
                df_long = df_long.rename(columns={'valor': Config.PRECIPITATION_COL})
                df_long[Config.YEAR_COL] = df_long[Config.DATE_COL].dt.year
                df_long[Config.MONTH_COL] = df_long[Config.DATE_COL].dt.month
        except Exception as e:
            st.warning(f"Error cargando precipitación: {e}")

        # ------------------------------------------------------------
        # 3. CARGAR OTRAS TABLAS
        # ------------------------------------------------------------
        try:
            sql_geo = "SELECT nombre, tipo_geometria, ST_AsText(geom) as wkt FROM geometrias"
            df_geo = pd.read_sql(sql_geo, engine)
            if not df_geo.empty:
                df_geo['geometry'] = df_geo['wkt'].apply(parse_geom)
                gdf_all = gpd.GeoDataFrame(df_geo, geometry='geometry', crs="EPSG:4326")
                gdf_municipios = gdf_all[gdf_all['tipo_geometria'] == 'municipio']
                gdf_subcuencas = gdf_all[gdf_all['tipo_geometria'].isin(['subcuenca', 'cuenca'])]
        except: pass

        try:
            df_enso = pd.read_sql("SELECT * FROM indices_climaticos", engine)
            if not df_enso.empty:
                df_enso.columns = [c.lower() for c in df_enso.columns]
                if 'fecha' in df_enso.columns:
                    df_enso[Config.DATE_COL] = pd.to_datetime(df_enso['fecha'])
                if 'oni' in df_enso.columns: df_enso = df_enso.rename(columns={'oni': Config.ENSO_ONI_COL})
                elif 'anomalia_oni' in df_enso.columns: df_enso = df_enso.rename(columns={'anomalia_oni': Config.ENSO_ONI_COL})
        except: pass

        # ------------------------------------------------------------
        # 4. GARANTÍA DE COLUMNAS (EL BLINDAJE)
        # ------------------------------------------------------------
        # Esto asegura que el sidebar NUNCA falle por KeyError, incluso si la BD falla
        
        required_cols_stations = [Config.STATION_NAME_COL, Config.REGION_COL, Config.MUNICIPALITY_COL]
        if gdf_stations is None or gdf_stations.empty:
            gdf_stations = pd.DataFrame(columns=required_cols_stations + [Config.ALTITUDE_COL])
        
        for col in required_cols_stations:
            if col not in gdf_stations.columns:
                gdf_stations[col] = "Desconocido"
                # st.warning(f"Columna faltante corregida: {col}") # Descomentar para depurar

        required_cols_long = [Config.DATE_COL, Config.PRECIPITATION_COL, Config.YEAR_COL, Config.STATION_NAME_COL]
        if df_long is None or df_long.empty:
            df_long = pd.DataFrame(columns=required_cols_long)
        
        for col in required_cols_long:
            if col not in df_long.columns:
                if col == Config.PRECIPITATION_COL: df_long[col] = 0.0
                else: df_long[col] = None

        return gdf_stations, gdf_municipios, df_long, df_enso, gdf_subcuencas

    except Exception as e:
        st.error(f"Error crítico general: {e}")
        # Retorno de emergencia seguro
        return pd.DataFrame(columns=[Config.STATION_NAME_COL, Config.REGION_COL]), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def complete_series(df):
    return df

