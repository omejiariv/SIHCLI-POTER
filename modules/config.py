import os
import streamlit as st

class Config:
    """
    Clase de configuración centralizada para SIHCLI-POTER.
    Define constantes de columnas, rutas y textos.
    """
    
    # --- Configuración General ---
    APP_TITLE = "Sistema de Información de Lluvias y Clima en el norte de la región Andina"
    
    # --- Nombres de Columnas (Mapeo con Base de Datos Supabase) ---
    # Estos valores deben coincidir con los nombres finales en el DataFrame
    DATE_COL = 'fecha'
    PRECIPITATION_COL = 'precipitation'
    
    # Columnas de Metadatos de Estaciones
    STATION_NAME_COL = 'nom_est'      # En BD es 'nom_est'
    ALTITUDE_COL = 'alt_est'          # En BD es 'alt_est'
    MUNICIPALITY_COL = 'municipio'    # En BD es 'municipio'
    REGION_COL = 'depto_region'       # En BD es 'depto_region'
    
    # Columnas Geográficas (Generadas en data_processor)
    LATITUDE_COL = 'latitude'
    LONGITUDE_COL = 'longitude'
    
    # Columnas Temporales (Generadas en data_processor)
    YEAR_COL = 'año'
    MONTH_COL = 'mes'
    
    # Columnas de Índices Climáticos
    ENSO_ONI_COL = 'anomalia_oni'
    SOI_COL = 'soi'
    IOD_COL = 'iod'
    
    # Otras columnas
    ORIGIN_COL = 'origin'             # Para diferenciar datos originales/completados
    ET_COL = 'et_mmy'
    PERCENTAGE_COL = 'porc_datos'

    # --- Rutas de Archivos y Assets ---
    # Calculamos rutas absolutas para evitar errores en despliegue
    _MODULES_DIR = os.path.dirname(__file__)
    _PROJECT_ROOT = os.path.abspath(os.path.join(_MODULES_DIR, '..'))
    
    # Carpetas
    ASSETS_DIR = os.path.join(_PROJECT_ROOT, 'assets')
    DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
    
    # Archivos específicos
    LOGO_PATH = os.path.join(ASSETS_DIR, 'CuencaVerde_Logo.jpg')
    CHAAC_IMAGE_PATH = os.path.join(ASSETS_DIR, 'chaac.png')
    GIF_PATH = os.path.join(ASSETS_DIR, 'PPAM.gif')
    
    # Archivos Raster (Deben estar en la carpeta 'data' del repo)
    DEM_FILE_PATH = os.path.join(DATA_DIR, 'DemAntioquiaWgs84.tif')
    PRECIP_RASTER_PATH = os.path.join(DATA_DIR, 'PPAMAnt.tif')

    # --- Textos de la Interfaz ---
    QUOTE_TEXT = "El futuro también depende del pasado y de nuestra capacidad presente para anticiparlo."
    QUOTE_AUTHOR = "omr."
    
    CHAAC_STORY = """
    ### Chaac, el Señor de la Lluvia
    En la mitología maya, **Chaac** es una de las deidades más importantes.
    Reside en los cuatro puntos cardinales y blande su hacha de relámpagos
    para golpear las nubes y producir la lluvia, esencial para la vida.
    Esta plataforma lleva su nombre como homenaje a la vital importancia del agua en nuestra región.
    """
    
    WELCOME_TEXT = """
    Esta plataforma interactiva está diseñada para la visualización y análisis de datos históricos de
    precipitación y su relación con el fenómeno ENSO en el norte de la región Andina.
    
    #### ¿Cómo empezar?
    1. **Filtros:** Utilice el **Panel de Control** a la izquierda para seleccionar región, municipio y estaciones.
    2. **Pestañas:** Navegue por las pestañas superiores para ver mapas, gráficos, estadísticas y pronósticos.
    """

    # --- Gestión de Estado de Sesión ---
    @staticmethod
    def initialize_session_state():
        """Inicializa variables de sesión seguras."""
        default_state = {
            'data_loaded': False,
            'apply_interpolation': False,
            'gdf_stations': None,
            'df_long': None,
            'df_enso': None,
            'gdf_municipios': None,
            'gdf_subcuencas': None,
            'unified_basin_gdf': None,
            'df_monthly_processed': None,
            # Pronósticos
            'forecasted_regressors_prophet': {},
            'forecasted_regressors_sarima': {},
            'last_forecasted_index_name': None,
            'last_forecasted_index_model': None,
            'last_forecasted_index_data': None,
            # UI
            'meses_numeros': list(range(1, 13)),
            'selected_basins_title': "",
            'dem_file_path': None,
            'dem_crs_is_geographic': True,
            'morph_results': None,
            'balance_results': None,
            # Clima
            'forecast_df': None,
            'forecast_station_name': None,
            'last_climate_data': None
        }

        for key, value in default_state.items():
            if key not in st.session_state:
                st.session_state[key] = value

        # Intentar registrar la ruta del DEM en sesión si existe
        if os.path.exists(Config.DEM_FILE_PATH):
            st.session_state['dem_file_path'] = Config.DEM_FILE_PATH
