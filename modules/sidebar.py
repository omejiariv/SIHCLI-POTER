# modules/sidebar.py

import streamlit as st
from modules.config import Config
import pandas as pd

def create_sidebar(gdf_stations, df_long):
    """
    Creates and displays sidebar widgets, returning selected values.
    REVISED AGAIN: Correctly filters station selection based on geo filters.
    """
    st.sidebar.header("Panel de Control")

    # Store full list of station names once
    if 'all_station_names' not in st.session_state:
        st.session_state['all_station_names'] = sorted(gdf_stations[Config.STATION_NAME_COL].unique())

    # --- Expander 1: Geographic and Data Filters ---
    with st.sidebar.expander("**1. Filtros Geográficos y de Datos**", expanded=True):
        min_data_perc = st.slider("Filtrar por % de datos mínimo:", 0, 100, st.session_state.get('min_data_perc_slider', 0), key="min_data_perc_slider")
        altitude_ranges = ['0-500', '500-1000', '1000-2000', '2000-3000', '>3000']
        selected_altitudes = st.multiselect('Filtrar por Altitud (m)', options=altitude_ranges, key='altitudes_multiselect')

        # Use full lists for geo options
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

        # *** Apply geo/data filters HERE to find valid stations ***
        gdf_filtered_geo_data = apply_filters_to_stations(
            gdf_stations.copy(), # Start fresh
            min_data_perc, selected_altitudes,
            selected_regions, selected_municipios, selected_celdas
        )
        # Set of station names meeting current geo/data criteria
        stations_meeting_geo_criteria = set(gdf_filtered_geo_data[Config.STATION_NAME_COL].unique())

    # --- Expander 2: Station Selection and Period ---
    with st.sidebar.expander("**2. Selección de Estaciones y Período**", expanded=True):
        # Use the FULL list of stations as OPTIONS
        station_options_all = st.session_state['all_station_names']

        # Callback for select/deselect all
        def select_all_stations_callback():
            # Select/deselect ALL stations, filtering happens later
            if st.session_state.get('select_all_checkbox_main', False):
                st.session_state.station_multiselect = station_options_all
            else:
                st.session_state.station_multiselect = []

        st.checkbox(
            "Seleccionar/Deseleccionar todas",
            key='select_all_checkbox_main',
            on_change=select_all_stations_callback
        )

        # Multiselect widget uses ALL stations as options
        selected_stations_raw = st.multiselect(
            'Seleccionar Estaciones',
            options=station_options_all, # Stable list
            key='station_multiselect' # Key manages the selection state
        )

        # *** Filter the user's raw selection ***
        # Keep only those selected stations that ALSO meet the geo/data criteria
        selected_stations_final = [
            station for station in selected_stations_raw
            if station in stations_meeting_geo_criteria # Filter against valid stations
        ]

        # Optional: Inform user if selections were excluded
        excluded_count = len(selected_stations_raw) - len(selected_stations_final)
        if excluded_count > 0:
            st.caption(f"{excluded_count} estaciones seleccionadas fueron excluidas por los filtros geográficos/datos.")

        # Year Range (no changes needed)
        years_with_data = sorted(df_long[Config.YEAR_COL].dropna().unique())
        year_range_default = (min(years_with_data), max(years_with_data)) if years_with_data else (1970, 2020)
        year_range = st.slider("Rango de Años", min_value=int(year_range_default[0]), max_value=int(year_range_default[1]),
                               value=st.session_state.get('year_range', (int(year_range_default[0]), int(year_range_default[1]))), key='year_range')

        # Months (no changes needed)
        meses_dict = {m: i + 1 for i, m in enumerate(['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'])}
        default_meses = st.session_state.get('meses_nombres_multiselect', list(meses_dict.keys()))
        meses_nombres = st.multiselect("Meses", list(meses_dict.keys()), default=default_meses, key='meses_nombres_multiselect')
        meses_numeros = [meses_dict[m] for m in meses_nombres]
        st.session_state['meses_numeros'] = meses_numeros

    # --- Expander 3: Preprocessing and DEM ---
    # ... (no changes needed here) ...
    with st.sidebar.expander("3. Opciones de Preprocesamiento y DEM"):
        analysis_mode = st.radio("Modo de análisis", ("Usar datos originales", "Completar series (interpolación)"), key="analysis_mode")
        exclude_na = st.checkbox("Excluir datos nulos (NaN)", key='exclude_na')
        exclude_zeros = st.checkbox("Excluir valores cero (0)", key='exclude_zeros')
        st.markdown("---")
        st.markdown("##### Modelo de Elevación (Opcional)")
        dem_file = st.file_uploader("Sube tu archivo DEM (.tif)...", type=["tif", "tiff"], key="dem_uploader_sidebar")
        if dem_file:
            st.session_state['dem_file'] = dem_file
            st.success(f"Archivo DEM '{dem_file.name}' cargado.")
        elif 'dem_file' in st.session_state and st.session_state['dem_file'] is not None:
             st.info(f"Usando DEM cargado: {st.session_state['dem_file'].name}")

    # Return the FINAL filtered values
    return {
        # Return the GeoDataFrame filtered ONLY by the FINAL valid stations
        "gdf_filtered": gdf_stations[gdf_stations[Config.STATION_NAME_COL].isin(selected_stations_final)],
        "selected_stations": selected_stations_final, # Return the FINAL list
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
            elif r == '1000-2000': conditions.append((altitude_col_numeric > 1000) & (altitude_col_numeric <= 2000))
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

