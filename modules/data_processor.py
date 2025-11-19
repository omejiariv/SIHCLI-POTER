import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
from sqlalchemy import create_engine, text
from shapely import wkt
from modules.config import Config

@st.cache_data(show_spinner="Cargando datos desde la Base de Datos...", ttl=600)
def load_and_process_all_data():
    """
    Carga robusta usando SELECT * para evitar errores de nombres de columnas.
    """
    # Inicializar DataFrames de seguridad (vacíos pero con estructura)
    dummy_stations = pd.DataFrame(columns=[Config.STATION_NAME_COL, Config.REGION_COL, Config.MUNICIPALITY_COL])
    dummy_long = pd.DataFrame(columns=[Config.DATE_COL, Config.PRECIPITATION_COL, Config.YEAR_COL])
    
    gdf_stations = dummy_stations
    df_long = dummy_long
    gdf_municipios = pd.DataFrame()
    df_enso = pd.DataFrame()
    gdf_subcuencas = pd.DataFrame()

    try:
        if "DATABASE_URL" not in st.secrets:
            st.error("❌ Falta 'DATABASE_URL' en secrets.")
            return gdf_stations, gdf_municipios, df_long, df_enso, gdf_subcuencas

        engine = create_engine(st.secrets["DATABASE_URL"])

        # ------------------------------------------------------------
        # 1. CARGAR ESTACIONES
        # ------------------------------------------------------------
        try:
            # Usamos SELECT * para ver qué hay realmente
            df_est_raw = pd.read_sql(text("SELECT * FROM estaciones"), engine)
            
            # Normalizar columnas (convertir a minúsculas para evitar errores de mayúsculas)
            df_est_raw.columns = [c.lower() for c in df_est_raw.columns]
            
            # Buscar columna de geometría
            geom_col = next((col for col in df_est_raw.columns if col in ['geom', 'geometry']), None)
            
            if geom_col:
                # Parsear WKB (binario) o WKT (texto)
                from shapely import wkb, wkt
                def parse_geom_safe(x):
                    try:
                        if isinstance(x, str): return wkt.loads(x)
                        if isinstance(x, (bytes, bytearray)): return wkb.loads(x, hex=True)
                        return None
                    except: return None
                
                df_est_raw['geometry'] = df_est_raw[geom_col].apply(parse_geom_safe)
                gdf_stations = gpd.GeoDataFrame(df_est_raw, geometry='geometry', crs="EPSG:4326")
                
                # Extraer Lat/Lon
                gdf_stations = gdf_stations.dropna(subset=['geometry'])
                gdf_stations['latitude'] = gdf_stations.geometry.y
                gdf_stations['longitude'] = gdf_stations.geometry.x
            else:
                gdf_stations = df_est_raw # Fallback sin geometría

            # Renombrar columnas críticas si tienen nombres distintos
            rename_map = {
                'nom_est': Config.STATION_NAME_COL,
                'nombre': Config.STATION_NAME_COL,
                'municipio': Config.MUNICIPALITY_COL,
                'depto_region': Config.REGION_COL
            }
            gdf_stations = gdf_stations.rename(columns=rename_map)
            
            # Limpieza de textos
            for col in [Config.STATION_NAME_COL, Config.MUNICIPALITY_COL, Config.REGION_COL]:
                if col in gdf_stations.columns:
                    gdf_stations[col] = gdf_stations[col].astype(str).str.replace(r'\s*-\s*$', '', regex=True).str.strip()

        except Exception as e:
            st.warning(f"⚠️ Error cargando estaciones: {e}")

        # ------------------------------------------------------------
        # 2. CARGAR PRECIPITACIÓN (El punto del fallo)
        # ------------------------------------------------------------
        try:
            # USA SELECT * PARA NO FALLAR POR NOMBRE DE COLUMNA
            df_ppt_raw = pd.read_sql(text("SELECT * FROM precipitacion_mensual"), engine)
            
            # Convertir columnas a minúsculas
            df_ppt_raw.columns = [c.lower() for c in df_ppt_raw.columns]
            
            # Identificar la columna de fecha dinámicamente
            date_candidates = ['fecha', 'date', 'fecha_registro', 'timestamp', 'dt', 'time']
            date_col_found = next((c for c in date_candidates if c in df_ppt_raw.columns), None)
            
            if date_col_found:
                df_ppt_raw[Config.DATE_COL] = pd.to_datetime(df_ppt_raw[date_col_found])
            else:
                st.error(f"❌ No se encontró columna de fecha. Columnas disponibles: {list(df_ppt_raw.columns)}")
                return gdf_stations, gdf_municipios, dummy_long, df_enso, gdf_subcuencas

            # Identificar columna de valor
            val_candidates = ['valor', 'precipitation', 'precipitacion', 'value', 'ppt']
            val_col_found = next((c for c in val_candidates if c in df_ppt_raw.columns), None)
            
            if val_col_found:
                df_ppt_raw = df_ppt_raw.rename(columns={val_col_found: Config.PRECIPITATION_COL})
            
            # Identificar FK de estación
            id_candidates = ['id_estacion_fk', 'id_estacion', 'station_id', 'codigo']
            id_col_found = next((c for c in id_candidates if c in df_ppt_raw.columns), None)

            # MERGE
            if id_col_found and not gdf_stations.empty:
                # Buscar la columna de ID en estaciones (probablemente 'id_estacion')
                est_id_col = 'id_estacion' if 'id_estacion' in gdf_stations.columns else 'id'
                
                if est_id_col in gdf_stations.columns:
                    df_long = pd.merge(
                        df_ppt_raw,
                        gdf_stations[[est_id_col, Config.STATION_NAME_COL]],
                        left_on=id_col_found,
                        right_on=est_id_col,
                        how='inner'
                    )
                    
                    df_long[Config.YEAR_COL] = df_long[Config.DATE_COL].dt.year
                    df_long[Config.MONTH_COL] = df_long[Config.DATE_COL].dt.month

        except Exception as e:
            st.error(f"❌ Error procesando precipitación: {e}")
            # Devolver vacío pero seguro
            df_long = dummy_long

        # ------------------------------------------------------------
        # 3. CARGAR OTRAS (Dummy por ahora para asegurar carga)
        # ------------------------------------------------------------
        try:
            df_enso = pd.read_sql(text("SELECT * FROM indices_climaticos"), engine)
            df_enso.columns = [c.lower() for c in df_enso.columns]
            if 'fecha' in df_enso.columns:
                 df_enso[Config.DATE_COL] = pd.to_datetime(df_enso['fecha'])
            if 'oni' in df_enso.columns:
                 df_enso = df_enso.rename(columns={'oni': Config.ENSO_ONI_COL})
        except: pass

        return gdf_stations, gdf_municipios, df_long, df_enso, gdf_subcuencas

    except Exception as e:
        st.error(f"❌ Error General de BD: {e}")
        return gdf_stations, gdf_municipios, df_long, df_enso, gdf_subcuencas

def complete_series(df):
    return df
