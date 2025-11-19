import os
import streamlit as st

class Config:
    """
    Configuración ajustada a la estructura REAL de Supabase (según imagen).
    """
    APP_TITLE = "SIHCLI-POTER"
    
    # --- MAPEO EXACTO CON BASE DE DATOS ---
    # Según tu imagen:
    DATE_COL = 'fecha_mes_año'       # <--- CAMBIO CRÍTICO
    PRECIPITATION_COL = 'precipitation' # <--- CAMBIO CRÍTICO
    
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
    
    # Rutas
    _MODULES_DIR = os.path.dirname(__file__)
    _PROJECT_ROOT = os.path.abspath(os.path.join(_MODULES_DIR, '..'))
    ASSETS_DIR = os.path.join(_PROJECT_ROOT, 'assets')
    DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
    
    LOGO_PATH = os.path.join(ASSETS_DIR, 'CuencaVerde_Logo.jpg')
    CHAAC_IMAGE_PATH = os.path.join(ASSETS_DIR, 'chaac.png')
    
    # Textos
    WELCOME_TEXT = "Sistema de Información Hidroclimática del Norte de la Región Andina"
    QUOTE_TEXT = "El agua es la fuerza motriz de toda la naturaleza."
    QUOTE_AUTHOR = "Leonardo da Vinci"
    CHAAC_STORY = "Chaac es la deidad maya de la lluvia..."

    @staticmethod
    def initialize_session_state():
        """Inicializa variables de sesión."""
        keys = [
            'data_loaded', 'apply_interpolation', 'gdf_stations', 'df_long', 
            'df_enso', 'gdf_municipios', 'gdf_subcuencas', 'unified_basin_gdf'
        ]
        for k in keys:
            if k not in st.session_state:
                st.session_state[k] = None
