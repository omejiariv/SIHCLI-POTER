# -------------------------------------------------------------------
# MIGRACIÓN DE DATOS A POSTGRESQL (migracion.py)
# -------------------------------------------------------------------
# Este script se ejecuta UNA SOLA VEZ para poblar la base de datos
# desde los archivos planos existentes.
#
# Prerrequisitos:
# 1. Tener una base de datos PostgreSQL con PostGIS habilitado.
# 2. pip install sqlalchemy psycopg2-binary geoalchemy2 pandas geopandas
# -------------------------------------------------------------------

import pandas as pd
import geopandas as gpd
import os
import sys
import numpy as np
from datetime import datetime

# Importaciones de SQLAlchemy
from sqlalchemy import create_engine, text, Column, Integer, String, Float, Date, JSON, Text, Numeric
from sqlalchemy.orm import sessionmaker, declarative_base
from geoalchemy2 import Geometry # Para PostGIS

print("Iniciando el script de migración...")

# --- PASO 1: CONFIGURAR LA CONEXIÓN A LA BASE DE DATOS ---
# --------------------------------------------------------
# EDITA ESTA LÍNEA con tus credenciales de PostgreSQL
# Formato: "postgresql://[USUARIO]:[CONTRASEÑA]@[HOST]:[PUERTO]/[NOMBRE_DB]"
DATABASE_URL = "postgresql://postgres:123456@localhost:5432/sihcli-poter_db"
# --------------------------------------------------------

try:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
except Exception as e:
    print(f"ERROR: No se pudo conectar a la base de datos en {DATABASE_URL}")
    print(f"Detalle: {e}")
    print("Asegúrate de que PostgreSQL esté corriendo y que la extensión PostGIS esté habilitada ('CREATE EXTENSION postgis;').")
    sys.exit(1)

# --- PASO 2: DEFINIR EL ESQUEMA (ORM) ---
# Definimos la estructura de nuestras tablas como clases de Python
Base = declarative_base()

class Estacion(Base):
    __tablename__ = "estaciones"
    id_estacion = Column(Text, primary_key=True) # ej. "26230240"
    nom_est = Column(Text, unique=True, nullable=False)
    alt_est = Column(Numeric)
    municipio = Column(Text)
    depto_region = Column(Text)
    geom = Column(Geometry(geometry_type='POINT', srid=4326)) # 4326 = WGS84
    et_mmy = Column(Numeric)

class Precipitacion(Base):
    __tablename__ = "precipitacion_mensual"
    id_precip = Column(Integer, primary_key=True, autoincrement=True) # Un ID propio es más eficiente
    id_estacion_fk = Column(Text, nullable=False) # Referencia a estaciones.id_estacion
    fecha_mes_año = Column(Date, nullable=False)
    precipitation = Column(Numeric)
    origin = Column(Text, default='Original')
    
    # __table_args__ = (Index('idx_precip_estacion_fecha', 'id_estacion_fk', 'fecha_mes_año', unique=True),)

class IndiceClimatico(Base):
    __tablename__ = "indices_climaticos"
    fecha_mes_año = Column(Date, primary_key=True)
    anomalia_oni = Column(Numeric)
    soi = Column(Numeric)
    iod = Column(Numeric)

class Geometria(Base):
    __tablename__ = "geometrias"
    id_geom = Column(Integer, primary_key=True, autoincrement=True)
    nombre = Column(Text)
    tipo_geometria = Column(Text, nullable=False) # 'subcuenca', 'municipio', 'predio'
    geom = Column(Geometry(geometry_type='MULTIPOLYGON', srid=4326))
    metadatos = Column(JSON) # Para guardar propiedades extra

class Raster(Base):
    __tablename__ = "rasters"
    id_raster = Column(Integer, primary_key=True, autoincrement=True)
    nombre = Column(Text, unique=True, nullable=False)
    tipo_raster = Column(Text, nullable=False) # 'DEM', 'CoberturaSuelo', 'ZonaVida'
    ruta_archivo = Column(Text, nullable=False) # Ruta en el servidor

# --- PASO 3: FUNCIONES HELPER (Copiadas de tu proyecto) ---

def parse_spanish_dates(date_series):
    months_es_to_en = {
        'ene': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'abr': 'Apr',
        'may': 'May', 'jun': 'Jun', 'jul': 'Jul', 'ago': 'Aug',
        'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dic': 'Dec'
    }
    date_series_str = date_series.astype(str).str.lower()
    for es, en in months_es_to_en.items():
        date_series_str = date_series_str.str.replace(es, en, regex=False)
    return pd.to_datetime(date_series_str, format='%b-%y', errors='coerce')

def standardize_numeric_column(series):
    series_clean = series.astype(str).str.replace(',', '.', regex=False)
    return pd.to_numeric(series_clean, errors='coerce')

# --- PASO 4: SCRIPT PRINCIPAL DE MIGRACIÓN ---

def migrar_datos():
    try:
        print("Conectado a la base de datos. Verificando extensión PostGIS...")
        session.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
        session.commit()
        
        print("Creando todas las tablas (si no existen)...")
        Base.metadata.create_all(bind=engine)
        print("Tablas creadas/verificadas.")

        # --- A. Migrar Estaciones (CORREGIDO) ---
        print("\nIniciando migración de 'estaciones'...")
        df_estaciones_raw = pd.read_csv("data/mapaCVENSO.csv", sep=";", encoding='latin1')
        df_estaciones_raw.columns = [col.strip().lower() for col in df_estaciones_raw.columns]
        
        # --- INICIO DE LA CORRECCIÓN ---
        # La lógica en data_processor.txt  muestra que 'et_mmy' 
        # ya está en 'mapaCVENSO.csv' (df_estaciones_raw).
        
        # Limpiar y preparar estaciones (df_estaciones_raw es la única fuente)
        df_estaciones = df_estaciones_raw.copy()
        df_estaciones['id_estacio'] = df_estaciones['id_estacio'].astype(str).str.strip()
        df_estaciones['latitud_wgs84'] = standardize_numeric_column(df_estaciones['latitud_wgs84'])
        df_estaciones['longitud_wgs84'] = standardize_numeric_column(df_estaciones['longitud_wgs84'])
        df_estaciones['alt_est'] = standardize_numeric_column(df_estaciones['alt_est'])
        
        # Estandarizar 'et_mmy' si existe (como en data_processor.txt )
        if 'et_mmy' in df_estaciones.columns:
            df_estaciones['et_mmy'] = standardize_numeric_column(df_estaciones['et_mmy'])
        else:
            df_estaciones['et_mmy'] = np.nan # Crear la columna si no existe
        
        # Ya no se necesita el MERGE con df_et_mmy
        # --- FIN DE LA CORRECCIÓN ---
        
        gdf_estaciones = gpd.GeoDataFrame(
            df_estaciones,
            geometry=gpd.points_from_xy(df_estaciones['longitud_wgs84'], df_estaciones['latitud_wgs84']),
            crs="EPSG:4326"
        )
        
        # Convertir a formato SQL
        gdf_estaciones_sql = gdf_estaciones.rename(columns={
            'id_estacio': 'id_estacion',
            'nom_est': 'nom_est',
            'alt_est': 'alt_est',
            'municipio': 'municipio',
            'depto_region': 'depto_region',
            'geometry': 'geom',
            'et_mmy': 'et_mmy'
        })
        
        # Seleccionar solo las columnas de la tabla
        columnas_tabla_estacion = [c.name for c in Estacion.__table__.columns]
        
        # Asegurarse de que todas las columnas de la tabla existan en el DF
        for col in columnas_tabla_estacion:
            if col not in gdf_estaciones_sql.columns:
                gdf_estaciones_sql[col] = np.nan # Añadir columnas faltantes como NaN
                
        gdf_estaciones_sql = gdf_estaciones_sql[columnas_tabla_estacion]
        
        print(f"Cargando {len(gdf_estaciones_sql)} estaciones a la base de datos...")
        # Usar 'to_postgis' (que es como 'to_sql' pero para geopandas)
        gdf_estaciones_sql.to_postgis('estaciones', engine, if_exists='replace', index=False)
        print("Migración de 'estaciones' completada.")

        # --- B. Migrar Índices Climáticos ---
        print("\nIniciando migración de 'indices_climaticos'...")
        # AHORA SÍ cargamos y limpiamos df_precip_raw para los índices
        df_precip_raw = pd.read_csv("data/DatosPptnmes_ENSO.csv", sep=";", encoding='latin1')
        df_precip_raw.columns = [col.strip().lower() for col in df_precip_raw.columns]
        
        df_indices = df_precip_raw[['fecha_mes_año', 'anomalia_oni', 'soi', 'iod']].drop_duplicates()
        
        # Limpiar y estandarizar
        df_indices['anomalia_oni'] = standardize_numeric_column(df_indices['anomalia_oni'])
        df_indices['soi'] = standardize_numeric_column(df_indices['soi'])
        df_indices['iod'] = standardize_numeric_column(df_indices['iod'])
        
        df_indices = df_indices.dropna(subset=['fecha_mes_año']).drop_duplicates(subset=['fecha_mes_año'])
        
        print(f"Cargando {len(df_indices)} registros de índices climáticos...")
        df_indices.to_sql('indices_climaticos', engine, if_exists='replace', index=False)
        print("Migración de 'indices_climaticos' completada.")

        # --- C. Migrar Geometrías ---
        print("\nIniciando migración de 'geometrias'...")
        
        gdf_subcuencas = gpd.read_file("data/SubcuencasAinfluencia.geojson")
        gdf_predios = gpd.read_file("data/PrediosEjecutados.geojson")
        gdf_municipios = gpd.read_file("data/mapaCVENSO.zip", layer="MunicipiosAntioquia")
        
        def preparar_geometria(gdf, tipo, col_nombre):
            gdf['tipo_geometria'] = tipo
            gdf = gdf.rename(columns={col_nombre: 'nombre', 'geometry': 'geom'})
            # Guardar todas las otras propiedades en la columna JSON
            prop_cols = [col for col in gdf.columns if col not in ['geom', 'nombre', 'tipo_geometria']]
            gdf['metadatos'] = gdf[prop_cols].to_dict('records')
            return gdf[['nombre', 'tipo_geometria', 'geom', 'metadatos']]

        gdf_subcuencas_sql = preparar_geometria(gdf_subcuencas, 'subcuenca', 'SUBC_LBL')
        gdf_predios_sql = preparar_geometria(gdf_predios, 'predio', 'NOMBRE_PRE')
        gdf_municipios_sql = preparar_geometria(gdf_municipios, 'municipio', 'MPIO_CNMBR')
        
        gdf_geometrias_final = pd.concat([gdf_subcuencas_sql, gdf_predios_sql, gdf_municipios_sql], ignore_index=True)
        
        print(f"Cargando {len(gdf_geometrias_final)} geometrías (cuencas, predios, municipios)...")
        gdf_geometrias_final.to_postgis('geometrias', engine, if_exists='replace', index=False)
        print("Migración de 'geometrias' completada.")

        # --- D. Migrar Rasters (solo las rutas) ---
        print("\nIniciando migración de 'rasters'...")
        rasters_data = [
            {"nombre": "DEM Antioquia (Base)", "tipo_raster": "DEM", "ruta_archivo": "data/DemAntioquia_EPSG3116.tif"},
            {"nombre": "Zonas de Vida (Holdridge)", "tipo_raster": "ZonaVida", "ruta_archivo": "data/PPAMAnt.tif"},
            {"nombre": "Cobertura del Suelo", "tipo_raster": "CoberturaSuelo", "ruta_archivo": "data/Cob25m_WGS84.tif"}
        ]
        df_rasters = pd.DataFrame(rasters_data)
        
        print(f"Cargando {len(df_rasters)} referencias de rasters...")
        df_rasters.to_sql('rasters', engine, if_exists='replace', index=False)
        print("Migración de 'rasters' completada.")

        # --- E. Migrar Precipitación (La Tabla Grande) ---
        print("\nIniciando migración de 'precipitacion_mensual' (esto puede tardar)...")
        df_precip_long = pd.read_parquet("data/datos_precipitacion_largos.parquet")
        
        # Preparar datos
        df_precip_sql = df_precip_long.rename(columns={
            'id_estacion': 'id_estacion_fk',
            'precipitacion_mm': 'precipitation'
        })
        df_precip_sql['fecha_mes_año'] = parse_spanish_dates(df_precip_sql['fecha_mes_año'])
        df_precip_sql['precipitation'] = standardize_numeric_column(df_precip_sql['precipitation'])
        df_precip_sql['id_estacion_fk'] = df_precip_sql['id_estacion_fk'].astype(str).str.strip()
        df_precip_sql['origin'] = 'Original'
        
        # Dejar solo las columnas que necesitamos
        columnas_tabla_precip = ['id_estacion_fk', 'fecha_mes_año', 'precipitation', 'origin']
        df_precip_sql = df_precip_sql[columnas_tabla_precip].dropna(subset=['fecha_mes_año'])
        
        print(f"Cargando {len(df_precip_sql)} registros de precipitación (en lotes)...")
        
        # Usar 'to_sql' con 'chunksize' para manejar la memoria
        df_precip_sql.to_sql(
            'precipitacion_mensual', 
            engine, 
            if_exists='replace', 
            index=False, 
            chunksize=50000, # Carga en lotes de 50,000 filas
            method='multi'
        )
        
        print("Migración de 'precipitacion_mensual' completada.")
        
        print("\n¡MIGRACIÓN COMPLETADA CON ÉXITO!")
        print("Puedes cerrar este script.")

    except Exception as e:
        print("\n--- ¡ERROR DURANTE LA MIGRACIÓN! ---")
        print(e)
        import traceback
        traceback.print_exc()
        session.rollback() # Deshacer cambios si algo falló
    finally:
        session.close()
        print("Conexión a la base de datos cerrada.")

# --- EJECUTAR EL SCRIPT ---
if __name__ == "__main__":
    
    # Validar que los archivos existan
    archivos_necesarios = [
        "data/mapaCVENSO.csv",
        "data/DatosPptnmes_ENSO.csv",
        "data/datos_precipitacion_largos.parquet",
        "data/SubcuencasAinfluencia.geojson",
        "data/PrediosEjecutados.geojson",
        "data/mapaCVENSO.zip"
    ]
    
    archivos_faltantes = [f for f in archivos_necesarios if not os.path.exists(f)]
    
    if archivos_faltantes:
        print("ERROR: Faltan archivos de datos en la carpeta 'data/'. No se puede continuar.")
        for f in archivos_faltantes:
            print(f"- {f}")
    else:
        print("Todos los archivos de datos encontrados. Iniciando migración...")
        migrar_datos()
