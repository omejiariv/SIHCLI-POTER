# app.py
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime
import requests_cache
import time

# --- Importaciones de Módulos Propios ---
from modules.config import Config
from modules.data_processor import load_and_process_all_data, complete_series
from modules.visualizer import (
    display_welcome_tab, display_spatial_distribution_tab, display_graphs_tab,
    display_advanced_maps_tab, display_anomalies_tab, display_drought_analysis_tab,
    display_stats_tab, display_correlation_tab, display_enso_tab,
    display_trends_and_forecast_tab, display_downloads_tab, display_station_table_tab,
    display_weekly_forecast_tab,
    display_additional_climate_maps_tab, 
    display_satellite_imagery_tab,
    display_land_cover_analysis_tab,
    display_life_zones_tab
)
from modules.sidebar import create_sidebar
from modules.reporter import generate_pdf_report
from modules.analysis import calculate_monthly_anomalies, calculate_basin_stats
from modules.github_loader import load_csv_from_url, load_zip_from_url

# --- Desactivar Advertencias ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def apply_filters_to_stations(df, min_perc, altitudes, regions, municipios, celdas):
    """Aplica una serie de filtros al DataFrame de estaciones."""
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


def main():
    #--- Definiciones de Funciones Internas ---
    def process_and_store_data(file_mapa, file_precip, file_shape, file_parquet):
        with st.spinner("Procesando archivos y cargando datos..."):
            gdf_stations, gdf_municipios, df_long, df_enso, gdf_subcuencas = \
                load_and_process_all_data(file_mapa, file_precip, file_shape, file_parquet)

            if gdf_stations is not None and df_long is not None and gdf_municipios is not None:
                st.session_state.update({
                    'gdf_stations': gdf_stations, 'gdf_municipios': gdf_municipios,
                    'df_long': df_long, 'df_enso': df_enso,
                    'gdf_subcuencas': gdf_subcuencas,
                    'data_loaded': True
                })
                st.success("¡Datos cargados y listos!")
                time.sleep(1) # Pequeña pausa para que el usuario vea el mensaje
            else:
                st.error("Hubo un error al procesar los archivos.")
                st.session_state['data_loaded'] = False

    #--- Inicio de la Ejecución de la App ---
    Config.initialize_session_state()
    st.set_page_config(layout="wide", page_title=Config.APP_TITLE)
    st.markdown("""<style>div.block-container{padding-top:1rem;} [data-testid="stMetricValue"] {font-size: 1.8rem;} [data-testid="stMetricLabel"] {font-size: 1rem; padding-bottom:5px; }</style>""", unsafe_allow_html=True)

    #--- TÍTULO DE LA APP ---
    title_col1, title_col2 = st.columns([0.05, 0.95])
    with title_col1:
        if os.path.exists(Config.LOGO_PATH):
            st.image(Config.LOGO_PATH, width=60)
    with title_col2:
        st.markdown(f'<h1 style="font-size:28px; margin-top:1rem;">{Config.APP_TITLE}</h1>', unsafe_allow_html=True)

    #--- DEFINICIÓN DE PESTAÑAS ---
    tab_names = [
        "Bienvenida", "Distribución Espacial", "Gráficos", "Mapas Avanzados",
        "Variables Climáticas", "Imágenes Satelitales",
        "Análisis Cobertura Suelo",
        "Zonas de Vida",
        "Análisis de Anomalías", "Análisis de Extremos", "Estadísticas",
        "Correlación", "Análisis ENSO", "Tendencias y Pronósticos",
        "Pronóstico Semanal",
        "Descargas", "Análisis por Cuenca", "Comparación de Periodos",
        "Tabla de Estaciones", "Generar Reporte"
    ]
    tabs = st.tabs(tab_names)

    #--- PANEL DE CARGA DE DATOS ---
    with st.sidebar.expander("**Subir/Actualizar Archivos Base**", expanded=not st.session_state.get('data_loaded', False)):
        load_mode = st.radio("Modo de Carga", ("GitHub", "Manual"), key="load_mode", horizontal=True)
        if load_mode == "Manual":
            uploaded_file_mapa = st.file_uploader("1. Archivo de estaciones (CSV)", type="csv")
            uploaded_file_precip = st.file_uploader("2. Archivo de precipitación (CSV)", type="csv")
            uploaded_zip_shapefile = st.file_uploader("3. Shapefile de municipios (.zip)", type="zip")
            if st.button("Procesar Datos Manuales"):
                if all([uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile, uploaded_file_parquet]):
                    process_and_store_data(uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile, uploaded_file_parquet)
                else:
                    st.warning("Por favor, suba los 4 archivos requeridos (Estaciones, Precipitación, Municipios y Parquet).")
        else:
            st.info(f"Datos desde: **{Config.GITHUB_USER}/{Config.GITHUB_REPO}**")
            if st.button("Cargar Datos desde GitHub"):
                with st.spinner("Descargando archivos..."):
                    github_files = {
                        'mapa': load_csv_from_url(Config.URL_ESTACIONES_CSV),
                        'precip': load_csv_from_url(Config.URL_PRECIPITACION_CSV),
                        'shape': load_zip_from_url(Config.URL_SHAPEFILE_ZIP),
                        'parquet': load_parquet_from_url(Config.URL_PARQUET)
                    }
                    # Ahora, actualiza la condición y la llamada a la función
                    if all(github_files.values()):
                        process_and_store_data(
                            github_files['mapa'], 
                            github_files['precip'], 
                            github_files['shape'],
                            github_files['parquet']
                        )   
                    else:
                        st.error("No se pudieron descargar los archivos desde GitHub.")

    #--- LÓGICA DE CONTROL DE FLUJO ---
    if not st.session_state.get('data_loaded', False):
        with tabs[0]:
            display_welcome_tab()
        for i, tab in enumerate(tabs):
            if i > 0:
                with tab:
                    st.warning("Para comenzar, cargue los datos usando el panel de la izquierda.")
        st.stop()

    #--- SECCIÓN DE CONTROL DEL SIDEBAR (UNA VEZ CARGADOS LOS DATOS) ---
    st.sidebar.success("Datos cargados.")
    if st.sidebar.button("Limpiar Caché y Reiniciar"):
        st.cache_data.clear()
        st.cache_resource.clear()
        requests_cache.clear()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # Llamada a la función que crea los filtros
    sidebar_filters = create_sidebar(st.session_state.gdf_stations, st.session_state.df_long)
    
    # Extraemos los valores del diccionario retornado
    gdf_filtered = sidebar_filters["gdf_filtered"]
    stations_for_analysis = sidebar_filters["selected_stations"]
    year_range = sidebar_filters["year_range"]
    meses_numeros = sidebar_filters["meses_numeros"]
    
    # Detener si no hay estaciones seleccionadas
    if not stations_for_analysis:
        with tabs[0]:
            display_welcome_tab()
        for i, tab in enumerate(tabs):
            if i > 0:
                with tab:
                    st.info("Para comenzar, seleccione al menos una estación en el panel de la izquierda.")
        st.stop()

    #--- Procesamiento de Datos Post-Filtros ---
    df_monthly_filtered = st.session_state.df_long[
        (st.session_state.df_long[Config.STATION_NAME_COL].isin(stations_for_analysis)) &
        (st.session_state.df_long[Config.DATE_COL].dt.year >= year_range[0]) &
        (st.session_state.df_long[Config.DATE_COL].dt.year <= year_range[1]) &
        (st.session_state.df_long[Config.DATE_COL].dt.month.isin(meses_numeros))
    ].copy()

    if sidebar_filters["analysis_mode"] == "Completar series (interpolación)":
        with st.spinner("Interpolando series, por favor espera..."):
            df_monthly_filtered = complete_series(df_monthly_filtered)

    if sidebar_filters["exclude_na"]:
        df_monthly_filtered.dropna(subset=[Config.PRECIPITATION_COL], inplace=True)
    if sidebar_filters["exclude_zeros"]:
        df_monthly_filtered = df_monthly_filtered[df_monthly_filtered[Config.PRECIPITATION_COL] > 0]

    annual_agg = df_monthly_filtered.groupby([Config.STATION_NAME_COL, Config.YEAR_COL]).agg(
        precipitation_sum=(Config.PRECIPITATION_COL, 'sum'),
        meses_validos=(Config.MONTH_COL, 'nunique')
    ).reset_index()
    annual_agg.loc[annual_agg['meses_validos'] < 10, 'precipitation_sum'] = np.nan
    df_anual_melted = annual_agg.rename(columns={'precipitation_sum': Config.PRECIPITATION_COL})

    display_args = {
        "gdf_filtered": gdf_filtered,
        "stations_for_analysis": stations_for_analysis,
        "df_anual_melted": df_anual_melted,
        "df_monthly_filtered": df_monthly_filtered,
        "analysis_mode": sidebar_filters["analysis_mode"],
        "selected_regions": sidebar_filters["selected_regions"],
        "selected_municipios": sidebar_filters["selected_municipios"],
        "selected_altitudes": sidebar_filters["selected_altitudes"]
    }
    
    #--- Renderizado de Pestañas ---
    with tabs[0]:
        display_welcome_tab()
    with tabs[1]:
        display_spatial_distribution_tab(**display_args)
    with tabs[2]:
        display_graphs_tab(**display_args)
    with tabs[3]:
        display_advanced_maps_tab(**display_args)
    with tabs[4]: # Variables Climáticas (nuevo índice 4)
        display_additional_climate_maps_tab(**display_args)
    with tabs[5]: # Imágenes Satelitales (nuevo índice 5)
        display_satellite_imagery_tab(**display_args)
    with tabs[6]: # Análisis Cobertura Suelo
        display_land_cover_analysis_tab(**display_args)
    with tabs[7]: # Zonas de Vida
        display_life_zones_tab(**display_args)
    with tabs[8]:
        display_anomalies_tab(df_long=st.session_state.df_long, **display_args)
    with tabs[9]:
        display_drought_analysis_tab(df_long=st.session_state.df_long, **display_args)
    with tabs[10]:
        display_stats_tab(df_long=st.session_state.df_long, **display_args)
    with tabs[11]:
        display_correlation_tab(**display_args)
    with tabs[12]:
        display_enso_tab(df_enso=st.session_state.df_enso, **display_args)
    with tabs[13]:
        display_trends_and_forecast_tab(df_full_monthly=st.session_state.df_long, **display_args)
        
    with tabs[14]:
        display_weekly_forecast_tab(
        stations_for_analysis=stations_for_analysis,
        gdf_filtered=gdf_filtered
    )
    
    with tabs[15]:
        display_downloads_tab(
            df_anual_melted=df_anual_melted,
            df_monthly_filtered=df_monthly_filtered,
            stations_for_analysis=stations_for_analysis,
            analysis_mode=st.session_state.analysis_mode
        )

    with tabs[16]: 
        st.header("Análisis Agregado por Cuenca Hidrográfica")
        # Verificar si gdf_subcuencas está cargado
        if 'gdf_subcuencas' in st.session_state and st.session_state.gdf_subcuencas is not None and not st.session_state.gdf_subcuencas.empty:
            BASIN_NAME_COLUMN = 'SUBC_LBL' # Columna con nombres de cuenca
            # Verificar si la columna de nombres existe
            if BASIN_NAME_COLUMN in st.session_state.gdf_subcuencas.columns:

                # --- LÓGICA CORREGIDA PARA OBTENER NOMBRES DE CUENCAS ---
                basin_names = [] # Inicializar lista
                
                # 1. Filtrar cuencas por las regiones seleccionadas en el sidebar
                #    'sidebar_filters' es el diccionario devuelto por create_sidebar
                regions_from_sidebar = sidebar_filters.get("selected_regions", []) 
                basins_in_selected_regions = st.session_state.gdf_subcuencas.copy() # Empezar con todas

                if regions_from_sidebar: # Si el usuario seleccionó alguna región
                    # Asumiendo que gdf_subcuencas tiene una columna que coincide con Config.REGION_COL
                    if Config.REGION_COL in basins_in_selected_regions.columns:
                         basins_in_selected_regions = basins_in_selected_regions[
                             basins_in_selected_regions[Config.REGION_COL].isin(regions_from_sidebar)
                         ]
                         if basins_in_selected_regions.empty:
                              st.info("Ninguna subcuenca encontrada en las regiones seleccionadas.")
                    else:
                         st.warning(f"El archivo de subcuencas no tiene la columna '{Config.REGION_COL}'. No se puede filtrar por región.")
                # Si no hay regiones seleccionadas, 'basins_in_selected_regions' sigue conteniendo todas las cuencas.
                
                # 2. Hacer Sjoin entre las cuencas (filtradas por región o todas) y las estaciones filtradas
                #    'gdf_filtered' viene del sidebar y ya respeta TODOS los filtros (incluida región, municipio, etc.)
                if not basins_in_selected_regions.empty and 'gdf_filtered' in sidebar_filters and not sidebar_filters['gdf_filtered'].empty:
                     # Asegurarse que ambos GeoDataFrames tengan CRS antes del sjoin
                     if basins_in_selected_regions.crs is None: basins_in_selected_regions.set_crs(st.session_state.gdf_stations.crs, allow_override=True) # Asumir mismo CRS que estaciones
                     if sidebar_filters['gdf_filtered'].crs is None: sidebar_filters['gdf_filtered'].set_crs(st.session_state.gdf_stations.crs, allow_override=True)

                     # Reproyectar a un CRS común si son diferentes (ej. WGS84)
                     target_crs_sjoin = "EPSG:4326"
                     try:
                          basins_for_sjoin = basins_in_selected_regions.to_crs(target_crs_sjoin)
                          stations_for_sjoin = sidebar_filters['gdf_filtered'].to_crs(target_crs_sjoin)
                          
                          relevant_basins_gdf = gpd.sjoin(
                              basins_for_sjoin,
                              stations_for_sjoin,
                              how="inner", 
                              predicate="intersects" # O 'contains'
                          )
                          if not relevant_basins_gdf.empty:
                              # Obtener nombres únicos de las cuencas resultantes del sjoin
                              basin_names = sorted(relevant_basins_gdf[BASIN_NAME_COLUMN].dropna().unique())
                          # Si relevant_basins_gdf está vacío, basin_names permanece []
                     except Exception as e_sjoin:
                          st.error(f"Error durante la unión espacial (sjoin): {e_sjoin}")
                          basin_names = [] # Resetear en caso de error
                          
                # Si no hay estaciones filtradas o no hay cuencas en regiones, basin_names permanece []
                # --- FIN LÓGICA CORREGIDA ---

                # --- Mostrar resultados o mensajes ---
                if not basin_names:
                    st.info("Ninguna cuenca (en las regiones/filtros seleccionados) contiene estaciones que coincidan con todos los filtros actuales.")
                else:
                    selected_basin = st.selectbox(
                        "Seleccione una cuenca para analizar:",
                        options=basin_names,
                        key="basin_selector" # Mantener la misma key
                    )
                    if selected_basin:
                        # Llamar a la función de análisis (asegúrate que esté importada)
                        # from modules.analysis import calculate_basin_stats 
                        stats_df, stations_in_selected_basin, error_msg = calculate_basin_stats(
                            sidebar_filters['gdf_filtered'], # Pasar las estaciones ya filtradas
                            st.session_state.gdf_subcuencas, # Pasar el GDF original de subcuencas
                            df_monthly_filtered, # Pasar los datos mensuales filtrados por sidebar
                            selected_basin,
                            BASIN_NAME_COLUMN
                        )

                        if error_msg: 
                            st.warning(error_msg)
                        
                        if stations_in_selected_basin: # Verificar si se encontraron estaciones DENTRO de la cuenca seleccionada
                            st.subheader(f"Resultados para la cuenca: {selected_basin}")
                            st.metric("Número de Estaciones Filtradas en la Cuenca", len(stations_in_selected_basin))
                            with st.expander("Ver estaciones incluidas"): 
                                st.write(", ".join(stations_in_selected_basin))
                            
                            if stats_df is not None and not stats_df.empty:
                                st.markdown("---")
                                st.write("**Estadísticas de Precipitación Mensual (Agregada para estaciones filtradas en la cuenca)**")
                                st.dataframe(stats_df, use_container_width=True)
                            else:
                                # Este mensaje podría aparecer si las estaciones están en la cuenca pero no tienen datos en el periodo/meses seleccionados
                                st.info("Aunque se encontraron estaciones filtradas en la cuenca, no hay datos de precipitación válidos para el período/meses seleccionados.")
                        # else: # Este else ya no es necesario porque si no hay estaciones, basin_names estaría vacío
                        #     st.warning("No se encontraron estaciones (que cumplan todos los filtros) dentro de la cuenca seleccionada.")
            else:
                st.error(f"Error Crítico: No se encontró la columna de nombres '{BASIN_NAME_COLUMN}' en el archivo de subcuencas.")
        else:
            st.warning("Los datos de las subcuencas no están cargados o el archivo está vacío.")
            
    with tabs[17]:
        st.header("Comparación de Periodos de Tiempo")
        analysis_level = st.radio(
            "Seleccione el nivel de análisis para la comparación:",
            ("Promedio Regional (Todas las estaciones seleccionadas)", "Por Cuenca Específica"),
            key="compare_level_radio"
        )
        df_to_compare = pd.DataFrame()

        if analysis_level == "Por Cuenca Específica":
            st.markdown("---")
            if st.session_state.gdf_subcuencas is not None and not st.session_state.gdf_subcuencas.empty:
                BASIN_NAME_COLUMN = 'SUBC_LBL'
                if BASIN_NAME_COLUMN in st.session_state.gdf_subcuencas.columns:
                    relevant_basins_gdf = gpd.sjoin(st.session_state.gdf_subcuencas, gdf_filtered, how="inner", predicate="intersects")
                    if not relevant_basins_gdf.empty:
                        basin_names = sorted(relevant_basins_gdf[BASIN_NAME_COLUMN].dropna().unique())
                    else:
                        basin_names = []
                    if not basin_names:
                        st.warning("Ninguna cuenca contiene estaciones que coincidan con los filtros actuales.", icon="⚠️")
                    else:
                        selected_basin = st.selectbox(
                            "Seleccione la cuenca a comparar:",
                            options=basin_names,
                            key="compare_basin_selector"
                        )
                        target_basin_geom = st.session_state.gdf_subcuencas[st.session_state.gdf_subcuencas[BASIN_NAME_COLUMN] == selected_basin]
                        stations_in_basin = gpd.sjoin(gdf_filtered, target_basin_geom, how="inner", predicate="within")
                        station_names_in_basin = stations_in_basin[Config.STATION_NAME_COL].unique().tolist()
                        df_to_compare = df_monthly_filtered[df_monthly_filtered[Config.STATION_NAME_COL].isin(station_names_in_basin)]
                        st.info(f"Análisis para **{len(station_names_in_basin)}** estaciones encontradas en la cuenca **{selected_basin}**.", icon="ℹ️")
                else:
                    st.error(f"Error Crítico: No se encontró la columna de nombres '{BASIN_NAME_COLUMN}' en el archivo de subcuencas.")
            else:
                st.warning("Los datos de las subcuencas no están cargados.", icon="⚠️")
        else: # Promedio Regional
            df_to_compare = df_monthly_filtered
        
        st.markdown("---")
        if df_to_compare.empty:
            st.warning("Seleccione una opción con estaciones válidas para poder realizar la comparación.", icon="ℹ️")
        else:
            years_with_data = sorted(df_to_compare[Config.YEAR_COL].dropna().unique())
            min_year, max_year = int(years_with_data[0]), int(years_with_data[-1])
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Periodo 1")
                periodo1 = st.slider(
                    "Seleccione el rango de años para el Periodo 1",
                    min_year, max_year,
                    (min_year, min_year + 10 if min_year + 10 < max_year else max_year),
                    key="periodo1_slider_comp"
                )
            with col2:
                st.markdown("#### Periodo 2")
                periodo2 = st.slider(
                    "Seleccione el rango de años para el Periodo 2",
                    min_year, max_year,
                    (max_year - 10 if max_year - 10 > min_year else min_year, max_year),
                    key="periodo2_slider_comp"
                )
            df_periodo1 = df_to_compare[(df_to_compare[Config.DATE_COL].dt.year >= periodo1[0]) & (df_to_compare[Config.DATE_COL].dt.year <= periodo1[1])]
            df_periodo2 = df_to_compare[(df_to_compare[Config.DATE_COL].dt.year >= periodo2[0]) & (df_to_compare[Config.DATE_COL].dt.year <= periodo2[1])]
            st.markdown("---")
            st.subheader("Resultados Comparativos")
            if df_periodo1.empty or df_periodo2.empty:
                st.warning("Uno o ambos periodos seleccionados no contienen datos. Por favor, ajuste los rangos.")
            else:
                stats1_mean = df_periodo1[Config.PRECIPITATION_COL].mean()
                stats2_mean = df_periodo2[Config.PRECIPITATION_COL].mean()
                delta = ((stats2_mean - stats1_mean) / stats1_mean) * 100 if stats1_mean != 0 else 0
                st.metric(
                    label=f"Precipitación Media Mensual ({periodo1[0]}-{periodo1[1]} vs. {periodo2[0]}-{periodo2[1]})",
                    value=f"{stats2_mean:.1f} mm",
                    delta=f"{delta:.2f}% (respecto a {stats1_mean:.1f} mm del Periodo 1)"
                )
                st.markdown("##### Desglose Estadístico Completo")
                col1_stats, col2_stats = st.columns(2)
                with col1_stats:
                    st.write(f"**Periodo 1 ({periodo1[0]}-{periodo1[1]})**")
                    st.dataframe(df_periodo1[Config.PRECIPITATION_COL].describe().round(2))
                with col2_stats:
                    st.write(f"**Periodo 2 ({periodo2[0]}-{periodo2[1]})**")
                    st.dataframe(df_periodo2[Config.PRECIPITATION_COL].describe().round(2))
    
    with tabs[18]:
        display_station_table_tab(**display_args)
    
    with tabs[19]:
        st.header("Generación de Reporte PDF")
       
        # Opciones para el reporte
        st.subheader("Seleccionar Secciones para Incluir en el Reporte:")
        report_sections_options = [
            "Resumen General",
            "Tabla de Estaciones",
            "Mapa de Distribución Espacial",
            "Análisis de Precipitación Mensual y Anual",
            "Análisis de Anomalías",
            "Análisis de Extremos Hidrológicos (Percentiles)",
            "Análisis de Índices de Sequía (SPI/SPEI)",
            "Análisis de Frecuencia de Extremos",
            "Análisis de Correlación",
            "Análisis ENSO",
            "Análisis de Tendencias y Pronósticos",
            "Comparación de Periodos"
        ]

        # Checkbox para seleccionar todas las secciones
        select_all_checkbox = st.checkbox("Seleccionar todas las secciones", value=st.session_state.select_all_report_sections_checkbox, key="select_all_report_sections_checkbox")
        
        if select_all_checkbox:
            st.session_state.selected_report_sections_multiselect = report_sections_options
        
        selected_report_sections = st.multiselect(
            "Secciones disponibles:",
            options=report_sections_options,
            default=st.session_state.selected_report_sections_multiselect,
            key="selected_report_sections_multiselect"
        )

        st.markdown("---")
        st.subheader("Configuración Adicional")
        report_title = st.text_input("Título del Reporte", value="Reporte de Análisis Climatológico", key="report_title_input")
        author_name = st.text_input("Nombre del Autor", value="Generado por SIHCLI", key="author_name_input")
        
        if st.button("Generar Reporte PDF", key="generate_pdf_button"):
            if not selected_report_sections:
                st.warning("Por favor, seleccione al menos una sección para incluir en el reporte.")
            else:
                with st.spinner("Generando reporte PDF... Esto puede tardar unos minutos."):
                    try:
                        # Prepara los datos necesarios para el reporte
                        # (Asegúrate de que estas variables estén disponibles en el alcance global o pasadas a generate_pdf_report)
                        # Por ejemplo, df_anomalies, df_drought_extremes, etc., deben ser calculados previamente
                        # o pasados como argumentos. Para este ejemplo, solo paso los args mínimos.
                        # DEBES ASEGURARTE DE QUE TODOS LOS DATOS NECESARIOS PARA CADA SECCIÓN ESTÉN DISPONIBLES.

                        report_pdf_bytes = generate_pdf_report(
                            selected_report_sections=selected_report_sections,
                            report_title=report_title,
                            author_name=author_name,
                            gdf_filtered=gdf_filtered,
                            df_long=st.session_state.df_long,
                            df_anual_melted=df_anual_melted,
                            df_monthly_filtered=df_monthly_filtered,
                            stations_for_analysis=stations_for_analysis,
                            # AÑADIR OTROS DATAFRAMES Y OBJETOS NECESARIOS AQUÍ
                            # Por ejemplo:
                            # df_anomalies=df_anomalies, 
                            # df_drought_extremes=df_drought_extremes,
                            # df_thresholds=df_thresholds,
                            # df_enso=st.session_state.df_enso,
                            # sarima_forecast=st.session_state.sarima_forecast,
                            # prophet_forecast=st.session_state.prophet_forecast
                            # etc.
                        )
                        st.success("Reporte PDF generado exitosamente!")
                        st.download_button(
                            label="Descargar Reporte PDF",
                            data=report_pdf_bytes,
                            file_name=f"{report_title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf",
                            mime="application/pdf",
                            key="download_pdf_button"
                        )
                    except Exception as e:
                        st.error(f"Error al generar el reporte PDF: {e}")
                        st.exception(e)

if __name__ == "__main__":
    main()













