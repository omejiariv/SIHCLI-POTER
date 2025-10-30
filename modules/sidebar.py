# modules/sidebar.py

import streamlit as st
from modules.config import Config
import pandas as pd
import os
import rasterio # Importación necesaria

# --- Ruta al DEM base ---
# (Asegúrate que 'DemAntioquiaWgs84.tif' esté en 'modules/data/')
_THIS_FILE_DIR_SB = os.path.dirname(__file__)
BASE_DEM_FILENAME = "DemAntioquia_EPSG3116.tif"
BASE_DEM_PATH = os.path.abspath(os.path.join(_THIS_FILE_DIR_SB, '..', 'data', BASE_DEM_FILENAME))

# Guardar la ruta y estado del CRS del DEM base en la sesión
# Esto se ejecuta una vez al inicio o cuando el script cambia
dem_base_found = False
dem_base_is_geographic = True # Asumir geográfico (peor caso)

if os.path.exists(BASE_DEM_PATH):
    try:
        with rasterio.open(BASE_DEM_PATH) as src:
            if src.crs:
                # Corregido: Verificar si es geográfico
                dem_base_is_geographic = src.crs.is_geographic 
            else:
                dem_base_is_geographic = True # Asumir geográfico si no hay CRS
                print("Advertencia (sidebar): DEM base no tiene CRS definido.")
        st.session_state['dem_file_path'] = BASE_DEM_PATH
        st.session_state['dem_crs_is_geographic'] = dem_base_is_geographic
        dem_base_found = True
    except Exception as e_base_crs:
        print(f"Advertencia (sidebar): No se pudo leer CRS del DEM base: {e_base_crs}")
        st.session_state['dem_file_path'] = None
        st.session_state['dem_crs_is_geographic'] = True
else:
    st.session_state['dem_file_path'] = None
    st.session_state['dem_crs_is_geographic'] = True
# --- FIN Ruta DEM base ---


def apply_filters_to_stations(df, min_perc, altitudes, regions, municipios, celdas):
    """
    Aplica una serie de filtros al DataFrame de estaciones.
    """
    stations_filtered = df.copy()
    if Config.PERCENTAGE_COL in stations_filtered.columns:
        stations_filtered[Config.PERCENTAGE_COL] = pd.to_numeric(
            stations_filtered[Config.PERCENTAGE_COL].astype(str).str.replace(',', '.', regex=False),
            errors='coerce'
        ).fillna(0)
    if min_perc > 0:
        stations_filtered = stations_filtered[stations_filtered[Config.PERCENTAGE_COL] >= min_perc]
    if altitudes:
        conditions = []
        altitude_col_numeric = pd.to_numeric(stations_filtered[Config.ALTITUDE_COL], errors='coerce')
        for r in altitudes:
            # rangos ajustados según tu código
            if r == '0-500': conditions.append((altitude_col_numeric >= 0) & (altitude_col_numeric <= 500))
            elif r == '500-1000': conditions.append((altitude_col_numeric > 500) & (altitude_col_numeric <= 1000))
            elif r == '1000-1500': conditions.append((altitude_col_numeric > 1000) & (altitude_col_numeric <= 1500)) 
            elif r == '1500-2000': conditions.append((altitude_col_numeric > 1500) & (altitude_col_numeric <= 2000))
            elif r == '2000-3000': conditions.append((altitude_col_numeric > 2000) & (altitude_col_numeric <= 3000))
            elif r == '>3000': conditions.append(altitude_col_numeric > 3000)
        if conditions:
            stations_filtered = stations_filtered[pd.concat(conditions, axis=1).any(axis=1)]
    if regions:
        stations_filtered = stations_filtered[stations_filtered[Config.REGION_COL].isin(regions)]
    if municipios:
        stations_filtered = stations_filtered[stations_filtered[Config.MUNICIPALITY_COL].isin(municipios)]
    if celdas and Config.CELL_COL in stations_filtered.columns:
        stations_filtered = stations_filtered[stations_filtered[Config.CELL_COL].isin(celdas)]
    return stations_filtered


def create_sidebar(gdf_stations, df_long):
    """
    Crea y muestra widgets del sidebar, retornando selecciones filtradas.
    """
    st.sidebar.header("Panel de Control")

    if 'all_station_names' not in st.session_state:
        st.session_state['all_station_names'] = sorted(gdf_stations[Config.STATION_NAME_COL].unique())

    # --- Expander 1: Filtros Geográficos y de Datos ---
    with st.sidebar.expander("**1. Filtros Geográficos y de Datos**", expanded=True):
        min_data_perc = st.slider("Filtrar por % de datos mínimo:", 0, 100, st.session_state.get('min_data_perc_slider', 0), key="min_data_perc_slider")
        altitude_ranges = ['0-500', '500-1000', '1000-1500', '1500-2000', '2000-3000', '>3000']
        selected_altitudes = st.multiselect('Filtrar por Altitud (m)', options=altitude_ranges, key='altitudes_multiselect')

        # Dentro de create_sidebar -> expander 1

        gdf_base_for_options = gdf_stations.copy()
        
        # --- LÓGICA DE FILTROS EN CASCADA CORREGIDA ---
        
        # 1. Crear lista de Regiones y obtener selección
        regions_list = sorted(gdf_base_for_options[Config.REGION_COL].dropna().unique())
        selected_regions = st.multiselect('Filtrar por Depto/Región', options=regions_list, key='regions_multiselect')

        # 2. Crear lista de Municipios DINÁMICAMENTE
        if selected_regions:
            # Si se seleccionó una región, filtrar municipios
            municipios_filtrados_df = gdf_base_for_options[
                gdf_base_for_options[Config.REGION_COL].isin(selected_regions)
            ]
            municipios_options = sorted(municipios_filtrados_df[Config.MUNICIPALITY_COL].dropna().unique())
        else:
            # Si no, mostrar todos los municipios
            municipios_options = sorted(gdf_base_for_options[Config.MUNICIPALITY_COL].dropna().unique())
            
        selected_municipios = st.multiselect('Filtrar por Municipio', options=municipios_options, key='municipios_multiselect')
        
        # 3. (Opcional) Filtrar Celdas dinámicamente también
        celdas_df = gdf_base_for_options
        if selected_regions:
             celdas_df = celdas_df[celdas_df[Config.REGION_COL].isin(selected_regions)]
        if selected_municipios:
             celdas_df = celdas_df[celdas_df[Config.MUNICIPALITY_COL].isin(selected_municipios)]
             
        celdas_list = []
        if Config.CELL_COL in celdas_df.columns:
            celdas_list = sorted(celdas_df[Config.CELL_COL].dropna().unique())
        
        selected_celdas = []
        if celdas_list:
             selected_celdas = st.multiselect('Filtrar por Celda_XY', options=celdas_list, key='celdas_multiselect')
             
        # --- FIN DE LA LÓGICA CORREGIDA ---

        gdf_filtered_geo_data = apply_filters_to_stations(
            gdf_stations.copy(),
            min_data_perc, selected_altitudes,
            selected_regions, selected_municipios, selected_celdas
        )
        station_options_valid_now = sorted(gdf_filtered_geo_data[Config.STATION_NAME_COL].unique())

    # --- Expander 2: Selección de Estaciones y Período ---
    with st.sidebar.expander("**2. Selección de Estaciones y Período**", expanded=True):
        previous_selection = st.session_state.get('station_multiselect', [])
        default_selection = [
            station for station in previous_selection
            if station in station_options_valid_now
        ]

        def select_all_valid_stations_callback():
            if st.session_state.get('select_all_checkbox_main', False):
                st.session_state.station_multiselect = station_options_valid_now 
            else:
                st.session_state.station_multiselect = []

        st.checkbox(
            "Seleccionar/Deseleccionar todas las visibles",
            key='select_all_checkbox_main',
            on_change=select_all_valid_stations_callback
        )
        
        selected_stations_final = st.multiselect(
            'Seleccionar Estaciones',
            options=station_options_valid_now, 
            default=default_selection,         
            key='station_multiselect'          
        )

        # Rango de Años y Meses
        years_with_data = sorted(df_long[Config.YEAR_COL].dropna().unique())
        min_year_data = int(min(years_with_data)) if years_with_data else 1970
        max_year_data = int(max(years_with_data)) if years_with_data else 2025
        slider_max_year = max(max_year_data, 2025)
        year_range_default = (min_year_data, slider_max_year)
        
        year_range = st.slider("Rango de Años", 
                               min_value=min_year_data, 
                               max_value=slider_max_year,
                               value=st.session_state.get('year_range', year_range_default), 
                               key='year_range')
        
        meses_dict = {m: i + 1 for i, m in enumerate(['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'])}
        default_meses = st.session_state.get('meses_nombres_multiselect', list(meses_dict.keys()))
        meses_nombres = st.multiselect("Meses", list(meses_dict.keys()), default=default_meses, key='meses_nombres_multiselect')
        meses_numeros = [meses_dict[m] for m in meses_nombres]
        st.session_state['meses_numeros'] = meses_numeros

    # --- Expander 3: Preprocesamiento y DEM (SIMPLIFICADO) ---
    with st.sidebar.expander("3. Opciones de Preprocesamiento y DEM"):
        analysis_mode = st.radio("Modo de análisis", ("Usar datos originales", "Completar series (interpolación)"), key="analysis_mode")
        exclude_na = st.checkbox("Excluir datos nulos (NaN)", key='exclude_na')
        exclude_zeros = st.checkbox("Excluir valores cero (0)", key='exclude_zeros')
        st.markdown("---")
        st.markdown("##### Modelo de Elevación Digital (DEM)")

        # --- INICIO DEL CÓDIGO MODIFICADO ---
        # Simplemente lee la ruta y el flag del session_state (que se definirán en app.py)
        dem_path_from_state = st.session_state.get('dem_file_path', None)
        dem_is_geo_from_state = st.session_state.get('dem_crs_is_geographic', True) # Asume True si no está

        if dem_path_from_state:
            dem_filename = os.path.basename(dem_path_from_state)
            st.info(f"Usando DEM base: {dem_filename}")
            if dem_is_geo_from_state:
                st.warning("El DEM base está en grados geográficos. El cálculo de áreas será impreciso.")
        else:
            # Muestra un error más genérico si la ruta no se encontró al inicio en app.py
            st.error("DEM base no encontrado al iniciar la app. Funciones DEM no calcularán áreas.")
        # --- FIN DEL CÓDIGO MODIFICADO ---
        # --- Fin Lógica DEM ---

    # Retornar los valores FINALES
    final_gdf_to_return = gdf_filtered_geo_data[gdf_filtered_geo_data[Config.STATION_NAME_COL].isin(selected_stations_final)]
    
    return {
        "gdf_filtered": final_gdf_to_return,
        "selected_stations": selected_stations_final,
        "year_range": year_range,
        "meses_numeros": meses_numeros,
        "analysis_mode": analysis_mode,
        "exclude_na": exclude_na,
        "exclude_zeros": exclude_zeros,
        "selected_regions": selected_regions,
        "selected_municipios": selected_municipios,
        "selected_altitudes": selected_altitudes
    }



