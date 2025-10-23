# modules/sidebar.py

import streamlit as st
from modules.config import Config
import pandas as pd

def create_sidebar(gdf_stations, df_long):
    """
    Crea y muestra todos los widgets de la barra lateral y retorna los valores seleccionados.
    MODIFICADO: Listas de opciones de Región/Municipio más estables.
    """
    st.sidebar.header("Panel de Control")

    # --- Expander de Filtros Geográficos y de Datos ---
    with st.sidebar.expander("**1. Filtros Geográficos y de Datos**", expanded=True):
        min_data_perc = st.slider(
            "Filtrar por % de datos mínimo:", 0, 100,
            st.session_state.get('min_data_perc_slider', 0), # Usa session state para persistencia
            key="min_data_perc_slider"
        )
        altitude_ranges = ['0-500', '500-1000', '1000-2000', '2000-3000', '>3000']
        selected_altitudes = st.multiselect('Filtrar por Altitud (m)', options=altitude_ranges, key='altitudes_multiselect') # Añadir key

        # --- Lógica Modificada para Opciones Estables ---
        # 1. Aplicar filtros iniciales (ej. % datos, altitud) para obtener GDF base
        #    (Usaremos una función apply_filters simplificada aquí o asumimos que gdf_stations ya está pre-filtrado si es necesario)
        #    Para simplificar, usaremos gdf_stations directamente por ahora.
        gdf_base_for_options = gdf_stations.copy()
        # Aquí podrías aplicar filtros de min_data_perc y selected_altitudes si fuera necesario
        # antes de generar las listas de opciones.

        # 2. Generar lista COMPLETA de regiones disponibles en el GDF base
        regions_list = sorted(gdf_base_for_options[Config.REGION_COL].dropna().unique())
        
        # 3. Generar lista COMPLETA de municipios disponibles en el GDF base
        municipios_list = sorted(gdf_base_for_options[Config.MUNICIPALITY_COL].dropna().unique())
        
        # 4. Generar lista COMPLETA de celdas si existe
        celdas_list = []
        if Config.CELL_COL in gdf_base_for_options.columns:
            celdas_list = sorted(gdf_base_for_options[Config.CELL_COL].dropna().unique())
        # --- Fin Lógica Modificada ---

        # Usar las listas COMPLETAS como opciones
        selected_regions = st.multiselect(
            'Filtrar por Depto/Región',
            options=regions_list, # Lista completa
            key='regions_multiselect' # Key existente
        )

        selected_municipios = st.multiselect(
            'Filtrar por Municipio',
            options=municipios_list, # Lista completa
            key='municipios_multiselect' # Key existente
        )

        selected_celdas = [] # Inicializar
        if celdas_list: # Solo mostrar si hay celdas
             selected_celdas = st.multiselect('Filtrar por Celda_XY', options=celdas_list, key='celdas_multiselect') # Key existente


        # Aplicamos TODOS los filtros al final para obtener el GDF filtrado final
        gdf_filtered_final = apply_filters_to_stations(
            gdf_stations.copy(), # Empezar con el original
            min_data_perc, selected_altitudes,
            selected_regions, selected_municipios, selected_celdas
        )
        stations_options = sorted(gdf_filtered_final[Config.STATION_NAME_COL].unique())

    # --- Expander de Selección de Estaciones y Período ---
    with st.sidebar.expander("**2. Selección de Estaciones y Período**", expanded=True):
        # Callback para seleccionar/deseleccionar todas las estaciones FILTRADAS
        def select_all_stations_filtered():
            # Usar 'stations_options' que ya tiene las estaciones filtradas
            if st.session_state.get('select_all_checkbox_main', False):
                st.session_state.station_multiselect = stations_options
            else:
                st.session_state.station_multiselect = []

        st.checkbox(
            "Seleccionar/Deseleccionar todas",
            key='select_all_checkbox_main',
            on_change=select_all_stations_filtered
        )
        selected_stations = st.multiselect(
            'Seleccionar Estaciones',
            options=stations_options, # Opciones basadas en GDF filtrado final
            key='station_multiselect' # Key existente
        )

        # Rango de Años (sin cambios)
        years_with_data = sorted(df_long[Config.YEAR_COL].dropna().unique())
        year_range_default = (min(years_with_data), max(years_with_data)) if years_with_data else (1970, 2020)
        year_range = st.slider(
            "Rango de Años",
            min_value=int(year_range_default[0]), # Asegurar que sean enteros
            max_value=int(year_range_default[1]), # Asegurar que sean enteros
            value=st.session_state.get('year_range', (int(year_range_default[0]), int(year_range_default[1]))), # Usar default como tupla de enteros
            key='year_range'
        )

        # Meses (sin cambios)
        meses_dict = {m: i + 1 for i, m in enumerate(['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'])}
        # Usar session state para persistencia de meses
        default_meses = st.session_state.get('meses_nombres_multiselect', list(meses_dict.keys()))
        meses_nombres = st.multiselect("Meses", list(meses_dict.keys()), default=default_meses, key='meses_nombres_multiselect')
        meses_numeros = [meses_dict[m] for m in meses_nombres]
        st.session_state['meses_numeros'] = meses_numeros # Guardar números también

    # --- Expander de Opciones de Preprocesamiento ---
    with st.sidebar.expander("3. Opciones de Preprocesamiento y DEM"): # Renombrado
        analysis_mode = st.radio("Modo de análisis", ("Usar datos originales", "Completar series (interpolación)"), key="analysis_mode")
        exclude_na = st.checkbox("Excluir datos nulos (NaN)", key='exclude_na')
        exclude_zeros = st.checkbox("Excluir valores cero (0)", key='exclude_zeros')

        st.markdown("---")
        st.markdown("##### Modelo de Elevación (Opcional)")
        dem_file = st.file_uploader("Sube tu archivo DEM (.tif) para análisis morfométrico", type=["tif", "tiff"], key="dem_uploader_sidebar")
        if dem_file:
            # Guardar el objeto UploadedFile directamente en session_state
            st.session_state['dem_file'] = dem_file
            st.success(f"Archivo DEM '{dem_file.name}' cargado.")
        elif 'dem_file' in st.session_state and st.session_state['dem_file'] is not None:
             # Mostrar si ya hay uno cargado
             st.info(f"Usando DEM cargado: {st.session_state['dem_file'].name}")


    # Retornar los valores finales seleccionados y el GDF filtrado final
    return {
        "gdf_filtered": gdf_filtered_final, # El GDF resultante de TODOS los filtros
        "selected_stations": selected_stations,
        "year_range": year_range,
        "meses_numeros": meses_numeros,
        "analysis_mode": analysis_mode,
        "exclude_na": exclude_na,
        "exclude_zeros": exclude_zeros,
        "selected_regions": selected_regions, # Valor actual del filtro región
        "selected_municipios": selected_municipios, # Valor actual del filtro municipio
        "selected_altitudes": selected_altitudes # Valor actual del filtro altitud
        # selected_celdas no se retorna explícitamente pero se usa en gdf_filtered_final
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


