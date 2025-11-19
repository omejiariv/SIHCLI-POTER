import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
from sqlalchemy import create_engine, text
from shapely import wkt
from modules.config import Config

@st.cache_data(show_spinner="Conectando a Supabase...", ttl=600)
def load_and_process_all_data():
    """
    Carga datos de Supabase con manejo robusto de errores.
    Si falla, devuelve estructuras vacías con las columnas correctas para evitar KeyErrors.
    """
    # Estructura de respaldo (Dummy) para evitar que la app explote si falla la BD
    dummy_stations = pd.DataFrame(columns=[
        Config.STATION_NAME_COL, Config.REGION_COL, Config.MUNICIPALITY_COL, 
        Config.ALTITUDE_COL, 'latitude', 'longitude'
    ])
    
    try:
        # 1. Verificar Secretos
        if "DATABASE_URL" not in st.secrets:
            st.error("❌ Falta 'DATABASE_URL' en los secretos (.streamlit/secrets.toml).")
            return dummy_stations, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        engine = create_engine(st.secrets["DATABASE_URL"])

        # 2. Cargar Estaciones (Usando ST_AsText para geometría segura)
        # Nota: Usamos text() para mayor compatibilidad con SQLAlchemy
        sql_est = text("""
            SELECT id_estacion, nom_est, alt_est, municipio, depto_region, ST_AsText(geom) as wkt 
            FROM estaciones
        """)
        
        df_est_raw = pd.read_sql(sql_est, engine)
        
        if df_est_raw.empty:
            st.warning("⚠️ La tabla 'estaciones' está vacía o no se pudo leer.")
            return dummy_stations, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Parsear Geometría
        def parse_geom(x):
            try: return wkt.loads(x) if x else None
            except: return None
        
        df_est_raw['geometry'] = df_est_raw['wkt'].apply(parse_geom)
        gdf_stations = gpd.GeoDataFrame(df_est_raw, geometry='geometry', crs="EPSG:4326")

        # Limpieza de Strings
        for col in ['municipio', 'depto_region', 'nom_est']:
            if col in gdf_stations.columns:
                gdf_stations[col] = gdf_stations[col].astype(str).str.replace(r'\s*-\s*$', '', regex=True).str.strip()

        # Renombrar Columnas (Mapeo exacto según tu DB)
        gdf_stations = gdf_stations.rename(columns={
            'nom_est': Config.STATION_NAME_COL,
            'alt_est': Config.ALTITUDE_COL,
            'municipio': Config.MUNICIPALITY_COL,
            'depto_region': Config.REGION_COL
        })

        # Extraer Lat/Lon para mapas
        gdf_stations = gdf_stations.dropna(subset=['geometry'])
        gdf_stations['latitude'] = gdf_stations.geometry.y
        gdf_stations['longitude'] = gdf_stations.geometry.x

        # 3. Cargar Precipitación
        sql_ppt = text("SELECT id_estacion_fk, fecha, valor FROM precipitacion_mensual")
        df_ppt_raw = pd.read_sql(sql_ppt, engine)
        df_ppt_raw[Config.DATE_COL] = pd.to_datetime(df_ppt_raw['fecha'])

        # Cruce (Merge) para tener nombres de estaciones en los datos de lluvia
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

        # 4. Cargar Geometrías (Municipios/Cuencas)
        try:
            sql_geo = text("SELECT nombre, tipo_geometria, ST_AsText(geom) as wkt FROM geometrias")
            df_geo_raw = pd.read_sql(sql_geo, engine)
            df_geo_raw['geometry'] = df_geo_raw['wkt'].apply(parse_geom)
            gdf_all = gpd.GeoDataFrame(df_geo_raw, geometry='geometry', crs="EPSG:4326")
            
            gdf_municipios = gdf_all[gdf_all['tipo_geometria'] == 'municipio']
            gdf_subcuencas = gdf_all[gdf_all['tipo_geometria'].isin(['subcuenca', 'cuenca'])]
        except:
            gdf_municipios = pd.DataFrame()
            gdf_subcuencas = pd.DataFrame()

        # 5. Cargar ENSO
        try:
            df_enso = pd.read_sql(text("SELECT * FROM indices_climaticos"), engine)
            if not df_enso.empty:
                # Normalizar nombres de columnas a minúsculas
                df_enso.columns = [c.lower() for c in df_enso.columns]
                if 'fecha' in df_enso.columns:
                    df_enso[Config.DATE_COL] = pd.to_datetime(df_enso['fecha'])
                # Mapeo flexible para ONI
                if 'oni' in df_enso.columns:
                    df_enso = df_enso.rename(columns={'oni': Config.ENSO_ONI_COL})
                elif 'anomalia_oni' in df_enso.columns:
                    df_enso = df_enso.rename(columns={'anomalia_oni': Config.ENSO_ONI_COL})
        except:
            df_enso = pd.DataFrame()

        return gdf_stations, gdf_municipios, df_long, df_enso, gdf_subcuencas

    except Exception as e:
        # AQUÍ ESTÁ LA CLAVE: Mostramos el error real y devolvemos una tabla vacía PERO con columnas
        st.error(f"❌ Error Crítico de Base de Datos: {e}")
        return dummy_stations, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def complete_series(df):
    """Función auxiliar para completar series"""
    if df is None or df.empty: return df
    df = df.sort_values(Config.DATE_COL)
    df[Config.PRECIPITATION_COL] = df.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL]\
        .transform(lambda x: x.interpolate(method='linear', limit_direction='both'))
    return df
