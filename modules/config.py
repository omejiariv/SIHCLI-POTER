import os
import streamlit as st

class Config:
    """
    Configuración centralizada para SIHCLI-POTER.
    Ajustada a la estructura de Supabase y estructura de carpetas local.
    """
    APP_TITLE = "SIHCLI-POTER"
    
    # --- MAPEO EXACTO CON BASE DE DATOS (Supabase) ---
    DATE_COL = 'fecha_mes_año'
    PRECIPITATION_COL = 'precipitation'
    
    # Metadatos de Estaciones
    STATION_NAME_COL = 'nom_est'
    ALTITUDE_COL = 'alt_est'
    MUNICIPALITY_COL = 'municipio'
    REGION_COL = 'depto_region'
    
    # Columnas generadas internamente
    LATITUDE_COL = 'latitude'
    LONGITUDE_COL = 'longitude'
    YEAR_COL = 'año'
    MONTH_COL = 'mes'
    
    # Índices Climáticos
    ENSO_ONI_COL = 'anomalia_oni'
    SOI_COL = 'soi'
    IOD_COL = 'iod'
    
    # --- RUTAS DE ARCHIVOS Y ASSETS ---
    # Calculamos rutas absolutas para robustez
    _MODULES_DIR = os.path.dirname(__file__)
    _PROJECT_ROOT = os.path.abspath(os.path.join(_MODULES_DIR, '..'))
    
    ASSETS_DIR = os.path.join(_PROJECT_ROOT, 'assets')
    DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
    
    # Archivos de Imagen
    LOGO_PATH = os.path.join(ASSETS_DIR, 'CuencaVerde_Logo.jpg')
    CHAAC_IMAGE_PATH = os.path.join(ASSETS_DIR, 'chaac.png')
    
    # Archivos Raster (Necesarios para Zonas de Vida y Cobertura)
    # Asegúrate de que estos archivos existan en la carpeta 'data'
    LAND_COVER_RASTER_PATH = os.path.join(DATA_DIR, 'Cob25m_WGS84.tif')
    DEM_FILE_PATH = os.path.join(DATA_DIR, 'DemAntioquiaWgs84.tif') # Opcional si usas morfometría vectorial
    
    # --- TEXTOS ---
    WELCOME_TEXT = """
    **Sistema de Información Hidroclimática del Norte de la Región Andina**
    
    Esta plataforma integra datos históricos, análisis estadísticos y modelación espacial 
    para el apoyo en la toma de decisiones sobre el recurso hídrico.
    """
    QUOTE_TEXT = "El agua es la fuerza motriz de toda la naturaleza."
    QUOTE_AUTHOR = "Leonardo da Vinci"
    CHAAC_STORY = "Chaac es la deidad maya de la lluvia, relacionada con el agua y la fertilidad."

    # --- GESTIÓN DE SESIÓN ---
    @staticmethod
    def initialize_session_state():
        """Inicializa todas las variables de sesión para evitar KeyErrors."""
        # Lista completa de claves usadas en la app
        keys = [
            'data_loaded', 
            'apply_interpolation', 
            
            # Datos Base
            'gdf_stations', 'df_long', 'df_enso', 
            'gdf_municipios', 'gdf_subcuencas', 'gdf_predios',
            'unified_basin_gdf',
            
            # Resultados de Análisis (Persistencia)
            'basin_results',  # Mapas avanzados
            'sarima_res',     # Pronósticos
            'prophet_res',
            'res_cuenca',
            'current_coverage_stats' # Cobertura
        ]
        
        for k in keys:
            if k not in st.session_state:
                st.session_state[k] = None
