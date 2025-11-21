import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
from sqlalchemy import create_engine, text
from shapely import wkt
from modules.config import Config

@st.cache_data(show_spinner="Cargando y limpiando datos...", ttl=600)
def load_and_process_all_data():
    """
    Carga datos desde Supabase.
    Garantiza nombres únicos para evitar el error 'DataFrame has no attribute unique'.
    """
    gdf_stations = pd.DataFrame()
    gdf_municipios = pd.DataFrame()
    gdf_subcuencas = pd.DataFrame()
    gdf_predios = pd.DataFrame()
    df_long = pd.DataFrame()
    df_enso = pd.DataFrame()

    try:
        if "DATABASE_URL" not in st.secrets:
            st.error("Falta DATABASE_URL en secrets.")
            return None, None, None, None, None, None

        engine = create_engine(st.secrets["DATABASE_URL"])

        # ---------------------------------------------------------
        # 1. CARGAR ESTACIONES
        # ---------------------------------------------------------
        try:
            sql_est = text("""
                SELECT id_estacion, nom_est, alt_est, municipio, depto_region, ST_AsText(geom) as wkt 
                FROM estaciones
            """)
            df_est = pd.read_sql(sql_est, engine)
            
            # Geometría
            def parse_geom(x):
                try: return wkt.loads(x) if x else None
                except: return None
            
            if 'wkt' in df_est.columns:
                df_est['geometry'] = df_est['wkt'].apply(parse_geom)
                gdf_stations = gpd.GeoDataFrame(df_est, geometry='geometry', crs="EPSG:4326")
            else:
                gdf_stations = df_est.copy()

            # --- CORRECCIÓN INTELIGENTE DE ETIQUETAS ---

            def create_label(row):
                nom = str(row['nom_est']).strip()
                cod = str(row['id_estacion']).strip()
                if cod in nom:
                    return nom # Ya tiene el código, no lo añadimos de nuevo
                return f"{nom} [{cod}]" # No lo tiene, lo añadimos

            gdf_stations['station_label'] = gdf_stations.apply(create_label, axis=1)

            # Renombrar usando el mapa de configuración
            # station_label -> STATION_NAME_COL
            gdf_stations = gdf_stations.rename(columns={
                'station_label': Config.STATION_NAME_COL,
                'alt_est': Config.ALTITUDE_COL,
                'municipio': Config.MUNICIPALITY_COL,
                'depto_region': Config.REGION_COL
            })

            # Limpieza final de duplicados de columnas
            gdf_stations = gdf_stations.loc[:, ~gdf_stations.columns.duplicated()]

            # Lat/Lon
            if 'geometry' in gdf_stations.columns:
                gdf_stations = gdf_stations.dropna(subset=['geometry'])
                gdf_stations['latitude'] = gdf_stations.geometry.y
                gdf_stations['longitude'] = gdf_stations.geometry.x

        except Exception as e:
            st.warning(f"Error cargando estaciones: {e}")

        # ---------------------------------------------------------
        # 2. CARGAR PRECIPITACIÓN
        # ---------------------------------------------------------
        try:
            sql_ppt = text('SELECT id_estacion_fk, "fecha_mes_año", precipitation FROM precipitacion_mensual')
            df_ppt = pd.read_sql(sql_ppt, engine)
            df_ppt[Config.DATE_COL] = pd.to_datetime(df_ppt['fecha_mes_año'])
            
            if not gdf_stations.empty:
                # Merge usando ID
                df_long = pd.merge(
                    df_ppt,
                    gdf_stations[['id_estacion', Config.STATION_NAME_COL]], 
                    left_on='id_estacion_fk',
                    right_on='id_estacion',
                    how='inner'
                )
                
                df_long = df_long.rename(columns={'precipitation': Config.PRECIPITATION_COL})
                df_long[Config.YEAR_COL] = df_long[Config.DATE_COL].dt.year
                df_long[Config.MONTH_COL] = df_long[Config.DATE_COL].dt.month
        except Exception as e:
            # st.error(f"Error ppt: {e}") # Opcional
            pass

        # ---------------------------------------------------------
        # 3. GEOMETRÍAS ADICIONALES
        # ---------------------------------------------------------
        try:
            sql_geo = text("SELECT nombre, tipo_geometria, ST_AsText(geom) as wkt FROM geometrias")
            df_geo = pd.read_sql(sql_geo, engine)
            if not df_geo.empty:
                df_geo['geometry'] = df_geo['wkt'].apply(parse_geom)
                gdf_all = gpd.GeoDataFrame(df_geo, geometry='geometry', crs="EPSG:4326")
                
                gdf_municipios = gdf_all[gdf_all['tipo_geometria'] == 'municipio']
                gdf_subcuencas = gdf_all[gdf_all['tipo_geometria'].isin(['subcuenca', 'cuenca'])]
                gdf_predios = gdf_all[gdf_all['tipo_geometria'] == 'predio']
        except: pass

        # ---------------------------------------------------------
        # 4. ENSO
        # ---------------------------------------------------------
        try:
            df_enso = pd.read_sql(text("SELECT * FROM indices_climaticos"), engine)
            df_enso.columns = [c.lower() for c in df_enso.columns]
            if 'fecha' in df_enso.columns: df_enso[Config.DATE_COL] = pd.to_datetime(df_enso['fecha'])
            if 'oni' in df_enso.columns: df_enso = df_enso.rename(columns={'oni': Config.ENSO_ONI_COL})
            elif 'anomalia_oni' in df_enso.columns: df_enso = df_enso.rename(columns={'anomalia_oni': Config.ENSO_ONI_COL})
        except: pass

        return gdf_stations, gdf_municipios, df_long, df_enso, gdf_subcuencas, gdf_predios

    except Exception as e:
        st.error(f"Error crítico conexión: {e}")
        return None, None, None, None, None, None

def complete_series(df):
    if df is None or df.empty: return df
    df = df.sort_values(Config.DATE_COL)
    df[Config.PRECIPITATION_COL] = df[Config.PRECIPITATION_COL].interpolate(method='linear', limit_direction='both')
    return df

