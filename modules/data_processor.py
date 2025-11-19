import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
from sqlalchemy import create_engine
from shapely import wkt
from modules.config import Config

@st.cache_data(show_spinner="Procesando datos hidrometeorológicos...", ttl=600)
def load_and_process_all_data():
    """
    Carga optimizada basada en la estructura real de Supabase.
    Realiza el cruce (JOIN) entre IDs y Nombres para que los gráficos funcionen.
    """
    # Inicializar
    gdf_stations = None
    gdf_municipios = pd.DataFrame()
    gdf_subcuencas = pd.DataFrame()
    df_long = None
    df_enso = None

    try:
        if "DATABASE_URL" not in st.secrets:
            st.error("Falta DATABASE_URL en secrets.")
            return None, None, None, None, None

        engine = create_engine(st.secrets["DATABASE_URL"])

        # ------------------------------------------------------------
        # 1. CARGAR ESTACIONES (Tabla 'estaciones')
        # ------------------------------------------------------------
        # Usamos ST_AsText para evitar errores binarios con la geometría
        sql_est = """
        SELECT id_estacion, nom_est, alt_est, municipio, depto_region, ST_AsText(geom) as wkt 
        FROM estaciones
        """
        df_est_raw = pd.read_sql(sql_est, engine)

        # Parsear geometría WKT a objetos reales
        def parse_geom(x):
            try: return wkt.loads(x) if x else None
            except: return None
        
        df_est_raw['geometry'] = df_est_raw['wkt'].apply(parse_geom)
        gdf_stations = gpd.GeoDataFrame(df_est_raw, geometry='geometry', crs="EPSG:4326")

        # Limpieza de textos (Quitar guiones y espacios extra)
        for col in ['municipio', 'depto_region', 'nom_est']:
            if col in gdf_stations.columns:
                gdf_stations[col] = gdf_stations[col].astype(str).str.replace(r'\s*-\s*$', '', regex=True).str.strip()

        # Mapeo a columnas estándar (CRÍTICO: Esto asegura que 'depto_region' exista)
        gdf_stations = gdf_stations.rename(columns={
            'nom_est': Config.STATION_NAME_COL,
            'alt_est': Config.ALTITUDE_COL,
            'municipio': Config.MUNICIPALITY_COL,
            'depto_region': Config.REGION_COL  # <--- ESTA LÍNEA ES LA CLAVE
        })

        # Asegurar que las columnas existan si el rename falló (Fallback)
        if Config.REGION_COL not in gdf_stations.columns:
             gdf_stations[Config.REGION_COL] = "Desconocido"

        # Extraer Lat/Lon para mapas rápidos
        gdf_stations = gdf_stations.dropna(subset=['geometry'])
        gdf_stations['latitude'] = gdf_stations.geometry.y
        gdf_stations['longitude'] = gdf_stations.geometry.x

        # ------------------------------------------------------------
        # 2. CARGAR PRECIPITACIÓN (Tabla 'precipitacion_mensual')
        # ------------------------------------------------------------
        # OJO: Esta tabla suele tener IDs, no nombres. Hacemos el MERGE aquí.
        sql_ppt = "SELECT id_estacion_fk, fecha, valor FROM precipitacion_mensual"
        df_ppt_raw = pd.read_sql(sql_ppt, engine)

        # Convertir fecha
        df_ppt_raw[Config.DATE_COL] = pd.to_datetime(df_ppt_raw['fecha'])

        # CRUCE MAESTRO: Unir Ppt con Nombres de Estación usando el ID
        # Asumimos que 'id_estacion_fk' en ppt coincide con 'id_estacion' en estaciones
        df_merged = pd.merge(
            df_ppt_raw,
            gdf_stations[['id_estacion', Config.STATION_NAME_COL]],
            left_on='id_estacion_fk',
            right_on='id_estacion',
            how='inner' # Solo mantener datos de estaciones que existen
        )

        # Preparar df_long final
        df_long = df_merged.rename(columns={'valor': Config.PRECIPITATION_COL})
        df_long[Config.YEAR_COL] = df_long[Config.DATE_COL].dt.year
        df_long[Config.MONTH_COL] = df_long[Config.DATE_COL].dt.month
        
        # Limpiar columnas sobrantes
        df_long = df_long[[Config.DATE_COL, Config.STATION_NAME_COL, Config.PRECIPITATION_COL, Config.YEAR_COL, Config.MONTH_COL]]

        # ------------------------------------------------------------
        # 3. CARGAR GEOMETRÍAS (Tabla 'geometrias')
        # ------------------------------------------------------------
        # Cargamos todo y filtramos en pandas para evitar múltiples queries lentos
        sql_geo = "SELECT nombre, tipo_geometria, ST_AsText(geom) as wkt FROM geometrias"
        df_geo_raw = pd.read_sql(sql_geo, engine)
        df_geo_raw['geometry'] = df_geo_raw['wkt'].apply(parse_geom)
        gdf_all_geoms = gpd.GeoDataFrame(df_geo_raw, geometry='geometry', crs="EPSG:4326")

        # Filtrar por tipo (según tu imagen, tienes 'predio', buscamos 'municipio' o 'subcuenca')
        if 'tipo_geometria' in gdf_all_geoms.columns:
            gdf_municipios = gdf_all_geoms[gdf_all_geoms['tipo_geometria'] == 'municipio']
            gdf_subcuencas = gdf_all_geoms[gdf_all_geoms['tipo_geometria'].isin(['subcuenca', 'cuenca'])]

        # ------------------------------------------------------------
        # 4. CARGAR ENSO
        # ------------------------------------------------------------
        try:
            df_enso = pd.read_sql("SELECT * FROM indices_climaticos", engine)
            # Intentar normalizar columnas
            cols_enso = [c.lower() for c in df_enso.columns]
            df_enso.columns = cols_enso
            
            if 'fecha' in df_enso.columns:
                df_enso[Config.DATE_COL] = pd.to_datetime(df_enso['fecha'])
            
            if 'oni' in df_enso.columns:
                df_enso = df_enso.rename(columns={'oni': Config.ENSO_ONI_COL})
            elif 'anomalia_oni' in df_enso.columns:
                df_enso = df_enso.rename(columns={'anomalia_oni': Config.ENSO_ONI_COL})

        except:
            df_enso = pd.DataFrame()

        return gdf_stations, gdf_municipios, df_long, df_enso, gdf_subcuencas

    except Exception as e:
        st.error(f"Error de conexión a BD: {e}")
        # Retornar vacíos para no romper la app
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def complete_series(df):
    """Interpolación simple para rellenar huecos"""
    if df is None or df.empty: return df
    df = df.sort_values(Config.DATE_COL)
    # Interpolación lineal por grupo (estación)
    df[Config.PRECIPITATION_COL] = df.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL]\
        .transform(lambda x: x.interpolate(method='linear', limit_direction='both'))
    return df

