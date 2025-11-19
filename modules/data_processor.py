import streamlit as st
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine, text
from shapely import wkt
from modules.config import Config

@st.cache_data(show_spinner="Cargando datos...", ttl=600)
def load_and_process_all_data():
    """
    Carga datos usando los nombres de columna EXACTOS de la imagen de Supabase.
    """
    # Inicializar vacíos por seguridad
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

        # 1. ESTACIONES (Mapeo: nom_est, depto_region...)
        try:
            sql_est = text("""
                SELECT id_estacion, nom_est, alt_est, municipio, depto_region, ST_AsText(geom) as wkt 
                FROM estaciones
            """)
            df_est = pd.read_sql(sql_est, engine)
            
            # Geometría
            df_est['geometry'] = df_est['wkt'].apply(lambda x: wkt.loads(x) if x else None)
            gdf_stations = gpd.GeoDataFrame(df_est, geometry='geometry', crs="EPSG:4326")
            
            # Limpieza
            for col in ['municipio', 'depto_region', 'nom_est']:
                if col in gdf_stations.columns:
                    gdf_stations[col] = gdf_stations[col].astype(str).str.replace(r'\s*-\s*$', '', regex=True).str.strip()

            # Lat/Lon explícitos
            gdf_stations = gdf_stations.dropna(subset=['geometry'])
            gdf_stations['latitude'] = gdf_stations.geometry.y
            gdf_stations['longitude'] = gdf_stations.geometry.x

        except Exception as e:
            st.warning(f"⚠️ Error cargando estaciones: {e}")

        # 2. PRECIPITACIÓN (Mapeo: fecha_mes_año, precipitation)
        try:
            # Usamos comillas dobles en "fecha_mes_año" por si PostgreSQL reclama por la ñ
            sql_ppt = text('SELECT id_estacion_fk, "fecha_mes_año", precipitation FROM precipitacion_mensual')
            df_ppt = pd.read_sql(sql_ppt, engine)
            
            # Asegurar fecha
            df_ppt[Config.DATE_COL] = pd.to_datetime(df_ppt['fecha_mes_año'])
            
            # Merge con nombres de estaciones
            if not gdf_stations.empty:
                df_long = pd.merge(
                    df_ppt,
                    gdf_stations[['id_estacion', 'nom_est']],
                    left_on='id_estacion_fk',
                    right_on='id_estacion',
                    how='inner'
                )
                # Columnas derivadas
                df_long[Config.YEAR_COL] = df_long[Config.DATE_COL].dt.year
                df_long[Config.MONTH_COL] = df_long[Config.DATE_COL].dt.month
                
        except Exception as e:
            st.error(f"❌ Error cargando precipitación (SQL): {e}")

        # 3. GEOMETRÍAS (Municipios)
        try:
            sql_geo = text("SELECT nombre, tipo_geometria, ST_AsText(geom) as wkt FROM geometrias")
            df_geo = pd.read_sql(sql_geo, engine)
            df_geo['geometry'] = df_geo['wkt'].apply(lambda x: wkt.loads(x) if x else None)
            gdf_all = gpd.GeoDataFrame(df_geo, geometry='geometry', crs="EPSG:4326")
            
            gdf_municipios = gdf_all[gdf_all['tipo_geometria'] == 'municipio']
            gdf_subcuencas = gdf_all[gdf_all['tipo_geometria'].isin(['subcuenca', 'cuenca'])]
        except: pass

        # 4. ENSO
        try:
            df_enso = pd.read_sql(text("SELECT * FROM indices_climaticos"), engine)
            if not df_enso.empty:
                # Normalizar nombres
                df_enso.columns = [c.lower() for c in df_enso.columns]
                # Buscar columna fecha
                date_col = next((c for c in df_enso.columns if 'fecha' in c or 'date' in c), None)
                if date_col:
                    df_enso[Config.DATE_COL] = pd.to_datetime(df_enso[date_col])
                
                # Buscar ONI
                if 'oni' in df_enso.columns:
                    df_enso = df_enso.rename(columns={'oni': Config.ENSO_ONI_COL})
                elif 'anomalia_oni' in df_enso.columns:
                     df_enso = df_enso.rename(columns={'anomalia_oni': Config.ENSO_ONI_COL})
        except: pass

        return gdf_stations, gdf_municipios, df_long, df_enso, gdf_subcuencas

    except Exception as e:
        st.error(f"Error general: {e}")
        return None, None, None, None, None

def complete_series(df):
    if df is None or df.empty: return df
    df = df.sort_values(Config.DATE_COL)
    # Interpolación simple para evitar errores
    df[Config.PRECIPITATION_COL] = df[Config.PRECIPITATION_COL].interpolate(method='linear', limit_direction='both')
    return df
