import streamlit as st
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine, text
from shapely import wkt
from modules.config import Config

@st.cache_data(show_spinner="Procesando datos...", ttl=600)
def load_and_process_all_data():
    # ... (Inicializaciones igual que antes) ...
    gdf_stations = pd.DataFrame()
    gdf_municipios = pd.DataFrame()
    gdf_subcuencas = pd.DataFrame()
    gdf_predios = pd.DataFrame() # Nueva variable para predios
    df_long = pd.DataFrame()
    df_enso = pd.DataFrame()

    try:
        if "DATABASE_URL" not in st.secrets:
            st.error("Falta DATABASE_URL.")
            return None, None, None, None, None, None # Ojo: retornamos uno más

        engine = create_engine(st.secrets["DATABASE_URL"])

        # 1. CARGAR ESTACIONES (Igual que antes...)
        try:
            sql_est = "SELECT id_estacion, nom_est, alt_est, municipio, depto_region, ST_AsText(geom) as wkt FROM estaciones"
            df_est = pd.read_sql(sql_est, engine)
            # ... (Lógica de parseo y limpieza igual que la versión anterior) ...
            # (Copia aquí la lógica de parse_geom y limpieza de strings que ya tenías)
            from shapely import wkt
            def parse_geom(x):
                try: return wkt.loads(x) if x else None
                except: return None

            df_est['geometry'] = df_est['wkt'].apply(parse_geom)
            gdf_stations = gpd.GeoDataFrame(df_est, geometry='geometry', crs="EPSG:4326")
            
            # Mapeo columnas
            gdf_stations = gdf_stations.rename(columns={
                'nom_est': Config.STATION_NAME_COL,
                'alt_est': Config.ALTITUDE_COL,
                'municipio': Config.MUNICIPALITY_COL,
                'depto_region': Config.REGION_COL
            })
            # Lat/Lon
            gdf_stations = gdf_stations.dropna(subset=['geometry'])
            gdf_stations['latitude'] = gdf_stations.geometry.y
            gdf_stations['longitude'] = gdf_stations.geometry.x
            
            # Limpiar espacios
            for col in [Config.STATION_NAME_COL, Config.MUNICIPALITY_COL]:
                if col in gdf_stations.columns:
                     gdf_stations[col] = gdf_stations[col].astype(str).str.strip()

        except Exception: pass

        # 2. CARGAR PRECIPITACIÓN (Igual que antes...)
        try:
            sql_ppt = "SELECT id_estacion_fk, fecha_mes_año, precipitation FROM precipitacion_mensual"
            df_ppt = pd.read_sql(sql_ppt, engine)
            df_ppt[Config.DATE_COL] = pd.to_datetime(df_ppt['fecha_mes_año'])
            
            if not gdf_stations.empty:
                df_long = pd.merge(df_ppt, gdf_stations[['id_estacion', Config.STATION_NAME_COL]], 
                                   left_on='id_estacion_fk', right_on='id_estacion', how='inner')
                df_long = df_long.rename(columns={'precipitation': Config.PRECIPITATION_COL})
                df_long[Config.YEAR_COL] = df_long[Config.DATE_COL].dt.year
                df_long[Config.MONTH_COL] = df_long[Config.DATE_COL].dt.month
        except Exception: pass

        # 3. CARGAR GEOMETRÍAS (ACTUALIZADO PARA CAPAS)
        try:
            # Traemos tipo y nombre
            sql_geo = "SELECT nombre, tipo_geometria, ST_AsText(geom) as wkt FROM geometrias"
            df_geo = pd.read_sql(sql_geo, engine)
            if not df_geo.empty:
                df_geo['geometry'] = df_geo['wkt'].apply(parse_geom)
                gdf_all = gpd.GeoDataFrame(df_geo, geometry='geometry', crs="EPSG:4326")
                
                # Separar capas
                gdf_municipios = gdf_all[gdf_all['tipo_geometria'] == 'municipio']
                gdf_subcuencas = gdf_all[gdf_all['tipo_geometria'].isin(['subcuenca', 'cuenca'])]
                gdf_predios = gdf_all[gdf_all['tipo_geometria'] == 'predio'] # Nueva capa
        except Exception as e:
            print(f"Error cargas geometrias: {e}")

        # 4. ENSO (Igual que antes...)
        try:
            df_enso = pd.read_sql("SELECT * FROM indices_climaticos", engine)
            # ... (Lógica ENSO igual a la versión anterior) ...
            df_enso.columns = [c.lower() for c in df_enso.columns]
            if 'fecha' in df_enso.columns: df_enso[Config.DATE_COL] = pd.to_datetime(df_enso['fecha'])
            if 'oni' in df_enso.columns: df_enso = df_enso.rename(columns={'oni': Config.ENSO_ONI_COL})
        except: pass

        return gdf_stations, gdf_municipios, df_long, df_enso, gdf_subcuencas, gdf_predios

    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
def complete_series(df):
    if df is None or df.empty: return df
    df = df.sort_values(Config.DATE_COL)
    # Interpolación simple para evitar errores
    df[Config.PRECIPITATION_COL] = df[Config.PRECIPITATION_COL].interpolate(method='linear', limit_direction='both')
    return df

