# modules/sidebar.py

import streamlit as st
from modules.config import Config
import pandas as pd

def create_sidebar(gdf_stations, df_long):
    """
    Crea y muestra widgets del sidebar, retornando selecciones filtradas.
    REVISADO OTRA VEZ: Filtra opciones de estación dinámicamente, preserva selección válida.
    """
    st.sidebar.header("Panel de Control")

    # --- Expander 1: Filtros Geográficos y de Datos ---
    with st.sidebar.expander("**1. Filtros Geográficos y de Datos**", expanded=True):
        min_data_perc = st.slider("Filtrar por % de datos mínimo:", 0, 100, st.session_state.get('min_data_perc_slider', 0), key="min_data_perc_slider")
        altitude_ranges = ['0-500', '500-1000', '1000-1500', '1500-2000', '2000-3000', '>3000']
        selected_altitudes = st.multiselect('Filtrar por Altitud (m)', options=altitude_ranges, key='altitudes_multiselect')

        # Usar listas completas para opciones geográficas
        gdf_base_for_options = gdf_stations.copy()
        regions_list = sorted(gdf_base_for_options[Config.REGION_COL].dropna().unique())
        municipios_list = sorted(gdf_base_for_options[Config.MUNICIPALITY_COL].dropna().unique())
        celdas_list = []
        if Config.CELL_COL in gdf_base_for_options.columns:
            celdas_list = sorted(gdf_base_for_options[Config.CELL_COL].dropna().unique())

        selected_regions = st.multiselect('Filtrar por Depto/Región', options=regions_list, key='regions_multiselect')
        selected_municipios = st.multiselect('Filtrar por Municipio', options=municipios_list, key='municipios_multiselect')
        selected_celdas = []
        if celdas_list:
             selected_celdas = st.multiselect('Filtrar por Celda_XY', options=celdas_list, key='celdas_multiselect')

        # Aplicar filtros geo/datos AHORA para determinar estaciones VÁLIDAS
        gdf_filtered_geo_data = apply_filters_to_stations(
            gdf_stations.copy(),
            min_data_perc, selected_altitudes,
            selected_regions, selected_municipios, selected_celdas
        )
        # Opciones de estaciones A MOSTRAR en el multiselect
        station_options_valid_now = sorted(gdf_filtered_geo_data[Config.STATION_NAME_COL].unique())

    # --- Expander 2: Selección de Estaciones y Período ---
    with st.sidebar.expander("**2. Selección de Estaciones y Período**", expanded=True):
        # Obtener la selección PREVIA del usuario (si existe)
        previous_selection = st.session_state.get('station_multiselect', [])
        
        # Determinar el DEFAULT: Mantener solo las estaciones previamente seleccionadas que AÚN son válidas
        default_selection = [
            station for station in previous_selection
            if station in station_options_valid_now
        ]

        # Callback para seleccionar/deseleccionar todas las VÁLIDAS ACTUALMENTE
        def select_all_valid_stations_callback():
            if st.session_state.get('select_all_checkbox_main', False):
                # Seleccionar solo las que cumplen los filtros geo/data actuales
                st.session_state.station_multiselect = station_options_valid_now 
            else:
                st.session_state.station_multiselect = []

        st.checkbox(
            "Seleccionar/Deseleccionar todas las visibles", # Texto ajustado
            key='select_all_checkbox_main',
            on_change=select_all_valid_stations_callback
        )
        
        # El multiselect AHORA usa las opciones filtradas y el default filtrado
        selected_stations_final = st.multiselect(
            'Seleccionar Estaciones',
            options=station_options_valid_now, # <-- OPCIONES FILTRADAS
            default=default_selection,         # <-- DEFAULT FILTRADO
            key='station_multiselect'          # <-- Key mantiene el estado
        )

        # Rango de Años y Meses (sin cambios)
        years_with_data = sorted(df_long[Config.YEAR_COL].dropna().unique())
        year_range_default = (min(years_with_data), max(years_with_data)) if years_with_data else (1970, 2025)
        year_range = st.slider("Rango de Años", min_value=int(year_range_default[0]), max_value=int(year_range_default[1]),
                               value=st.session_state.get('year_range', (int(year_range_default[0]), int(year_range_default[1]))), key='year_range')
        meses_dict = {m: i + 1 for i, m in enumerate(['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'])}
        default_meses = st.session_state.get('meses_nombres_multiselect', list(meses_dict.keys()))
        meses_nombres = st.multiselect("Meses", list(meses_dict.keys()), default=default_meses, key='meses_nombres_multiselect')
        meses_numeros = [meses_dict[m] for m in meses_nombres]
        st.session_state['meses_numeros'] = meses_numeros

    # --- Expander 3: Preprocesamiento y DEM ---
    # ... (sin cambios) ...
    with st.sidebar.expander("3. Opciones de Preprocesamiento y DEM"):
        analysis_mode = st.radio("Modo de análisis", ("Usar datos originales", "Completar series (interpolación)"), key="analysis_mode")
        exclude_na = st.checkbox("Excluir datos nulos (NaN)", key='exclude_na')
        exclude_zeros = st.checkbox("Excluir valores cero (0)", key='exclude_zeros')
        st.markdown("---")
        st.markdown("##### Modelo de Elevación (Opcional)")
        dem_file = st.file_uploader("Sube tu archivo DEM (.tif)...", type=["tif", "tiff"], key="dem_uploader_sidebar")
        # ... (código manejo dem_file) ...

    # Retornar los valores FINALES
    # 'gdf_filtered_geo_data' contiene todas las estaciones que cumplen filtros geo/data
    # 'selected_stations_final' contiene las estaciones seleccionadas por el usuario DENTRO de ese grupo
    # Filtramos gdf_filtered_geo_data una vez más para obtener el GDF final a retornar
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
    
def apply_filters_to_stations(df, min_perc, altitudes, regions, municipios, celdas):
    # Esta es la misma función de filtrado que tenías en app.py
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



