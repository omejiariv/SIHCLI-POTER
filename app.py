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
# Se eliminan las funciones de carga de archivos, solo se importa la nueva
from modules.data_processor import load_and_process_all_data, complete_series
from modules.visualizer import (
    display_welcome_tab,
    display_alerts_tab,
    display_spatial_distribution_tab,
    display_graphs_tab,
    display_advanced_maps_tab,
    display_anomalies_tab,
    display_drought_analysis_tab,
    display_stats_tab,
    display_correlation_tab,
    display_enso_tab,
    display_climate_forecast_tab,
    display_trends_and_forecast_tab,
    display_weekly_forecast_tab,
    display_additional_climate_maps_tab,
    display_satellite_imagery_tab,
    display_land_cover_analysis_tab,
    display_life_zones_tab,
    display_climate_scenarios_tab,
    display_station_table_tab
)
from modules.sidebar import create_sidebar
from modules.reporter import generate_pdf_report
from modules.analysis import calculate_monthly_anomalies, calculate_basin_stats
# (Importaciones de github_loader y load_parquet_from_url eliminadas)

# --- INICIO BLOQUE INICIALIZACIÓN DEM ---
DEM_FILENAME = "DemAntioquia_EPSG3116.tif" 
try:
    _APP_DIR = os.path.dirname(__file__) 
    _DATA_DIR = os.path.abspath(os.path.join(_APP_DIR, 'data'))
    _DEM_PATH_APP = os.path.join(_DATA_DIR, DEM_FILENAME)
except NameError:
     _DEM_PATH_APP = os.path.join('data', DEM_FILENAME) 

# --- Desactivar Advertencias ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# La función apply_filters_to_stations se queda igual (no necesita cambios)
def apply_filters_to_stations(df, min_perc, altitudes, regions, municipios, celdas):
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
            # (Tu lógica de altitudes completa va aquí)
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
    #--- Inicio de la Ejecución de la App ---
    Config.initialize_session_state()
    st.set_page_config(layout="wide", page_title=Config.APP_TITLE)
    
    # --- Validación segura del DEM (Sin cambios) ---
    try:
        if not st.session_state.get('dem_file_path_validated', False):
            if os.path.exists(_DEM_PATH_APP):
                try:
                    import rasterio
                    with rasterio.open(_DEM_PATH_APP) as src:
                        if src.crs:
                             st.session_state['dem_crs_is_geographic'] = bool(src.crs.is_geographic)
                        st.session_state['dem_file_path'] = _DEM_PATH_APP
                        st.session_state['dem_file_path_validated'] = True
                        st.session_state['dem_source_name'] = os.path.basename(_DEM_PATH_APP)
                except Exception as e_dem:
                    st.warning(f"No se pudo validar DEM base {_DEM_PATH_APP}: {e_dem}")
                    st.session_state['dem_file_path_validated'] = False
    except RuntimeError:
        pass
        
    st.markdown("""<style>div.block-container{padding-top:1rem;} [data-testid="stMetricValue"] {font-size: 1.8rem;} [data-testid="stMetricLabel"] {font-size: 1rem; padding-bottom:5px; }</style>""", unsafe_allow_html=True)

    #--- TÍTULO DE LA APP (Sin cambios) ---
    title_col1, title_col2 = st.columns([0.05, 0.95])
    with title_col1:
        if os.path.exists(Config.LOGO_PATH):
            st.image(Config.LOGO_PATH, width=60)
    with title_col2:
        st.markdown(f'<h1 style="font-size:28px; margin-top:1rem;">{Config.APP_TITLE}</h1>', unsafe_allow_html=True)

    #--- DEFINICIÓN DE PESTAÑAS (Sin cambios) ---
    tab_names = [
        "Bienvenida", "Alertas y Resumen", "Distribución Espacial", "Gráficos", "Mapas Avanzados",
        "Variables Climáticas", "Imágenes Satelitales", "Análisis Cobertura Suelo", "Zonas de Vida",
        "Escenarios Climáticos", "Análisis de Anomalías", "Análisis de Extremos", "Estadísticas",
        "Correlación", "Análisis ENSO", "Pronóstico Climático", "Tendencias y Pronósticos",
        "Pronóstico Semanal", "Análisis por Cuenca", "Comparación de Periodos",
        "Tabla de Estaciones", "Generar Reporte"
    ]
    tabs = st.tabs(tab_names)

    #--- PANEL DE CARGA DE DATOS (REEMPLAZADO) ---
    # Ya no mostramos un expander para cargar, solo el botón de limpiar caché
    # La carga se maneja automáticamente en la siguiente sección.
    with st.sidebar.expander("**Gestión de Datos**", expanded=True):
        st.info("Los datos se cargan automáticamente desde la base de datos.")
        if st.button("Limpiar Caché y Recargar Datos"):
            st.cache_data.clear()
            st.cache_resource.clear()
            requests_cache.clear()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    #--- LÓGICA DE CONTROL DE FLUJO Y CARGA DE DATOS (REEMPLAZADO) ---
    
    # 1. Comprobar si los datos ya están en la sesión
    if not st.session_state.get('data_loaded', False):
        try:
            # 2. Si no están, llamamos a la nueva función (que usa SQL)
            with st.spinner("Conectando a la Base de Datos y cargando datos..."):
                gdf_stations, gdf_municipios, df_long, df_enso, gdf_subcuencas = load_and_process_all_data()
            
            # 3. Chequeo de que la carga fue exitosa
            if df_long is None or gdf_stations is None or gdf_subcuencas is None:
                st.error("Error fatal: No se pudieron cargar los datos de la base de datos.")
                with tabs[0]:
                    display_welcome_tab()
                st.stop()
            
            # 4. Guardar en session_state (la corrección que hicimos)
            st.session_state.gdf_stations = gdf_stations
            st.session_state.df_long = df_long
            st.session_state.df_enso = df_enso
            st.session_state.gdf_municipios = gdf_municipios
            st.session_state.gdf_subcuencas = gdf_subcuencas
            st.session_state.data_loaded = True
            st.rerun() # Forzar un rerun para que el sidebar se pueble

        except Exception as e:
            st.error(f"Error fatal al conectar o cargar datos de la base de datos: {e}")
            st.exception(e)
            with tabs[0]:
                display_welcome_tab()
            st.stop()
            
    # 5. Si los datos SÍ están cargados, asignarlos a variables locales
    gdf_stations = st.session_state.gdf_stations
    gdf_municipios = st.session_state.gdf_municipios
    df_long = st.session_state.df_long
    df_enso = st.session_state.df_enso
    gdf_subcuencas = st.session_state.gdf_subcuencas
    
    # --- FIN DE LA LÓGICA DE CARGA ---

    #--- SECCIÓN DE CONTROL DEL SIDEBAR (Sin cambios) ---
    sidebar_filters = create_sidebar(gdf_stations, df_long)
    
    # Extraemos los valores del diccionario retornado
    gdf_filtered = sidebar_filters["gdf_filtered"]
    stations_for_analysis = sidebar_filters["selected_stations"]
    year_range = sidebar_filters["year_range"]
    meses_numeros = sidebar_filters["meses_numeros"]
    analysis_mode = sidebar_filters["analysis_mode"]
    exclude_na = sidebar_filters["exclude_na"]
    exclude_zeros = sidebar_filters["exclude_zeros"]
    
    # Detener si no hay estaciones seleccionadas después de filtrar
    if not stations_for_analysis:
        with tabs[0]:
            display_welcome_tab()
        for i, tab in enumerate(tabs):
             if i > 0:
                 with tab:
                     st.info("Para comenzar, seleccione al menos una estación en el panel de la izquierda.")
        st.stop()

    #--- Procesamiento de Datos Post-Filtros (Lógica Optimizada) ---

    # 1. Ejecutar complete_series SOLO UNA VEZ si es necesario y guardarlo en session_state
    if analysis_mode == "Completar series (interpolación)":
        if 'df_completed' not in st.session_state:
            with st.spinner("Procesando y cacheando series completadas por primera vez..."):
                # [CORRECCIÓN] Llama a complete_series con la variable local df_long
                st.session_state.df_completed = complete_series(df_long)
                if st.session_state.df_completed.empty:
                    st.warning("La completación de series no produjo resultados.")
                    st.session_state.df_completed = df_long # Fallback
        base_df_monthly = st.session_state.df_completed
    else:
        # [CORRECCIÓN] Usa la variable local df_long
        base_df_monthly = df_long
        if Config.ORIGIN_COL not in base_df_monthly.columns:
            base_df_monthly[Config.ORIGIN_COL] = 'Original'

    # 2. Aplicar TODOS los filtros (Estación, Fecha, Mes) al DataFrame base seleccionado
    if not base_df_monthly.empty:
        df_monthly_filtered = base_df_monthly[
            (base_df_monthly[Config.STATION_NAME_COL].isin(stations_for_analysis)) &
            (base_df_monthly[Config.DATE_COL].dt.year >= year_range[0]) &
            (base_df_monthly[Config.DATE_COL].dt.year <= year_range[1]) &
            (base_df_monthly[Config.DATE_COL].dt.month.isin(meses_numeros))
        ].copy()
    else:
        df_monthly_filtered = pd.DataFrame()

    # 3. Aplicar exclusión de NaN y Ceros
    if not df_monthly_filtered.empty:
        if exclude_na:
            df_monthly_filtered.dropna(subset=[Config.PRECIPITATION_COL], inplace=True)
        if exclude_zeros:
            df_monthly_filtered[Config.PRECIPITATION_COL] = pd.to_numeric(df_monthly_filtered[Config.PRECIPITATION_COL], errors='coerce')
            df_monthly_filtered = df_monthly_filtered.dropna(subset=[Config.PRECIPITATION_COL])
            df_monthly_filtered = df_monthly_filtered[df_monthly_filtered[Config.PRECIPITATION_COL] > 0]
    
    # 4. Calcular datos anuales
    df_anual_melted = pd.DataFrame()
    
    # --- INICIO DEL BLOQUE CORREGIDO (IndentationError) ---
    # (Asegúrate de que este bloque 'if' esté indentado 4 espacios)
    if not df_monthly_filtered.empty and Config.PRECIPITATION_COL in df_monthly_filtered.columns and Config.MONTH_COL in df_monthly_filtered.columns:
        # (Este 'try' debe estar indentado 8 espacios)
        try:
            annual_agg = df_monthly_filtered.groupby([Config.STATION_NAME_COL, Config.YEAR_COL]).agg(
                precipitation_sum=(Config.PRECIPITATION_COL, lambda x: pd.to_numeric(x, errors='coerce').sum()),
                meses_validos=(Config.MONTH_COL, 'nunique')
            ).reset_index()
            annual_agg.loc[annual_agg['meses_validos'] < 10, 'precipitation_sum'] = np.nan
            df_anual_melted = annual_agg.rename(columns={'precipitation_sum': Config.PRECIPITATION_COL})
            df_anual_melted = df_anual_melted[[Config.STATION_NAME_COL, Config.YEAR_COL, Config.PRECIPITATION_COL, 'meses_validos']]
        # (Este 'except' debe estar indentado 8 espacios, alineado con 'try')
        except Exception as e_agg:
            st.error(f"Error durante la agregación anual: {e_agg}")
            df_anual_melted = pd.DataFrame()
    # (Este 'elif' debe estar indentado 4 espacios, alineado con 'if')
    elif not df_monthly_filtered.empty:
         st.warning("Columnas necesarias ('precipitation', 'month') no encontradas en df_monthly_filtered para agregación anual.")
    # --- FIN DEL BLOQUE CORREGIDO ---

    #--- Preparar argumentos para las pestañas (código original tuyo) ---
    display_args = {
        "gdf_stations": gdf_stations,
        "gdf_municipios": gdf_municipios,
        "df_long": df_long,
        "df_enso": df_enso,
        "gdf_subcuencas": gdf_subcuencas,
        "gdf_filtered": gdf_filtered,
        "stations_for_analysis": stations_for_analysis,
        "df_anual_melted": df_anual_melted,
        "df_monthly_filtered": df_monthly_filtered,
        "analysis_mode": analysis_mode,
        "selected_regions": sidebar_filters["selected_regions"],
        "selected_municipios": sidebar_filters["selected_municipios"],
        "selected_altitudes": sidebar_filters["selected_altitudes"]
    }
    
    #--- Renderizado de Pestañas (CORREGIDO Y ALINEADO) ---
    
    with tabs[0]:  # Bienvenida
        display_welcome_tab()
    
    with tabs[1]:  # Alertas y Resumen
        display_alerts_tab(**display_args)
    
    with tabs[2]:  # Distribución Espacial
        display_spatial_distribution_tab(**display_args)
    
    with tabs[3]:  # Gráficos
        display_graphs_tab(**display_args)
    
    with tabs[4]:  # Mapas Avanzados
        display_advanced_maps_tab(**display_args)
    
    with tabs[5]:  # Variables Climáticas
        display_additional_climate_maps_tab(**display_args)
    
    with tabs[6]:  # Imágenes Satelitales
        display_satellite_imagery_tab(**display_args)
    
    with tabs[7]:  # Análisis Cobertura Suelo
        display_land_cover_analysis_tab(**display_args)
    
    with tabs[8]:  # Zonas de Vida
        display_life_zones_tab(**display_args)
        
    with tabs[9]:  # Escenarios Climáticos
        display_climate_scenarios_tab(**display_args)
    
    with tabs[10]: # Análisis de Anomalías
        display_anomalies_tab(**display_args)
    
    with tabs[11]: # Análisis de Extremos
        display_drought_analysis_tab(**display_args)
    
    with tabs[12]: # Estadísticas
        display_stats_tab(**display_args)
    
    with tabs[13]: # Correlación
        display_correlation_tab(**display_args)
    
    with tabs[14]: # Análisis ENSO
        display_enso_tab(**display_args)
    
    with tabs[15]: # Pronóstico Climático
        display_climate_forecast_tab(**display_args)
        
    with tabs[16]: # Tendencias y Pronósticos (LA LÍNEA DEL ERROR)
        # CORRECCIÓN: Pasa la variable local 'df_long', no 'st.session_state.df_long'
        display_trends_and_forecast_tab(df_full_monthly=df_long, **display_args)
        
    with tabs[17]: # Pronóstico Semanal
        display_weekly_forecast_tab(
            stations_for_analysis=stations_for_analysis,
            gdf_filtered=gdf_filtered
        )
    
    with tabs[18]: # Análisis por Cuenca
        st.header("Análisis Agregado por Cuenca Hidrográfica")
        if gdf_subcuencas is not None and not gdf_subcuencas.empty:
            BASIN_NAME_COLUMN = 'SUBC_LBL' 
            if BASIN_NAME_COLUMN in gdf_subcuencas.columns:
                basin_names = [] 
                regions_from_sidebar = sidebar_filters.get("selected_regions", []) 
                basins_in_selected_regions = gdf_subcuencas.copy() 

                if regions_from_sidebar: 
                    if Config.REGION_COL in basins_in_selected_regions.columns:
                         basins_in_selected_regions = basins_in_selected_regions[
                             basins_in_selected_regions[Config.REGION_COL].isin(regions_from_sidebar)
                         ]
                         if basins_in_selected_regions.empty:
                             st.info("Ninguna subcuenca encontrada en las regiones seleccionadas.")
                    else:
                         st.warning(f"El archivo de subcuencas no tiene la columna '{Config.REGION_COL}'. No se puede filtrar por región.")
                
                if not basins_in_selected_regions.empty and 'gdf_filtered' in sidebar_filters and not sidebar_filters['gdf_filtered'].empty:
                     if basins_in_selected_regions.crs is None: basins_in_selected_regions.set_crs(gdf_stations.crs, allow_override=True)
                     if sidebar_filters['gdf_filtered'].crs is None: sidebar_filters['gdf_filtered'].set_crs(gdf_stations.crs, allow_override=True)
                     target_crs_sjoin = "EPSG:4326"
                     try:
                          basins_for_sjoin = basins_in_selected_regions.to_crs(target_crs_sjoin)
                          stations_for_sjoin = sidebar_filters['gdf_filtered'].to_crs(target_crs_sjoin)
                          relevant_basins_gdf = gpd.sjoin(
                              basins_for_sjoin, stations_for_sjoin,
                              how="inner", predicate="intersects"
                          )
                          if not relevant_basins_gdf.empty:
                              basin_names = sorted(relevant_basins_gdf[BASIN_NAME_COLUMN].dropna().unique())
                     except Exception as e_sjoin:
                          st.error(f"Error durante la unión espacial (sjoin): {e_sjoin}")
                          basin_names = []
                
                if not basin_names:
                    st.info("Ninguna cuenca (en las regiones/filtros seleccionados) contiene estaciones que coincidan con todos los filtros actuales.")
                else:
                     selected_basin = st.selectbox(
                        "Seleccione una cuenca para analizar:",
                        options=basin_names,
                        key="basin_selector" 
                     )
                     if selected_basin:
                        stats_df, stations_in_selected_basin, error_msg = calculate_basin_stats(
                            sidebar_filters['gdf_filtered'], gdf_subcuencas,
                            df_monthly_filtered, selected_basin, BASIN_NAME_COLUMN
                        )
                        if error_msg: st.warning(error_msg)
                        if stations_in_selected_basin: 
                            st.subheader(f"Resultados para la cuenca: {selected_basin}")
                            st.metric("Número de Estaciones Filtradas en la Cuenca", len(stations_in_selected_basin))
                            with st.expander("Ver estaciones incluidas"): 
                                st.write(", ".join(stations_in_selected_basin))
                            if stats_df is not None and not stats_df.empty:
                                st.markdown("---")
                                st.write("**Estadísticas de Precipitación Mensual (Agregada para estaciones filtradas en la cuenca)**")
                                st.dataframe(stats_df, use_container_width=True)
                            else:
                                st.info("Aunque se encontraron estaciones filtradas en la cuenca, no hay datos de precipitación válidos para el período/meses seleccionados.")
            else:
                st.error(f"Error Crítico: No se encontró la columna de nombres '{BASIN_NAME_COLUMN}' en el archivo de subcuencas.")
        else:
           st.warning("Los datos de las subcuencas no están cargados o el archivo está vacío.")
    
    with tabs[19]: # Comparación de Periodos
        st.header("Comparación de Periodos de Tiempo")
        analysis_level = st.radio(
            "Seleccione el nivel de análisis para la comparación:",
            ("Promedio Regional (Todas las estaciones seleccionadas)", "Por Cuenca Específica"),
            key="compare_level_radio"
        )
        df_to_compare = pd.DataFrame()

        if analysis_level == "Por Cuenca Específica":
            st.markdown("---")
            if gdf_subcuencas is not None and not gdf_subcuencas.empty:
                BASIN_NAME_COLUMN = 'SUBC_LBL'
                if BASIN_NAME_COLUMN in gdf_subcuencas.columns:
                    relevant_basins_gdf = gpd.sjoin(gdf_subcuencas, gdf_filtered, how="inner", predicate="intersects")
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
                         target_basin_geom = gdf_subcuencas[gdf_subcuencas[BASIN_NAME_COLUMN] == selected_basin]
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
                periodo1 = st.slider("Seleccione el rango de años para el Periodo 1", min_year, max_year, (min_year, min_year + 10 if min_year + 10 < max_year else max_year), key="periodo1_slider_comp")
            with col2:
                st.markdown("#### Periodo 2")
                periodo2 = st.slider("Seleccione el rango de años para el Periodo 2", min_year, max_year, (max_year - 10 if max_year - 10 > min_year else min_year, max_year), key="periodo2_slider_comp")
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
                st.metric(label=f"Precipitación Media Mensual ({periodo1[0]}-{periodo1[1]} vs. {periodo2[0]}-{periodo2[1]})", value=f"{stats2_mean:.1f} mm", delta=f"{delta:.2f}% (respecto a {stats1_mean:.1f} mm del Periodo 1)")
                st.markdown("##### Desglose Estadístico Completo")
                col1_stats, col2_stats = st.columns(2)
                with col1_stats:
                    st.write(f"**Periodo 1 ({periodo1[0]}-{periodo1[1]})**")
                    st.dataframe(df_periodo1[Config.PRECIPITATION_COL].describe().round(2))
                with col2_stats:
                    st.write(f"**Periodo 2 ({periodo2[0]}-{periodo2[1]})**")
                    st.dataframe(df_periodo2[Config.PRECIPITATION_COL].describe().round(2))
    
    with tabs[20]: # Tabla de Estaciones
        display_station_table_tab(**display_args)
    
    with tabs[21]: # Generar Reporte
        st.header("Generación de Reporte PDF")
        
        # --- 1. Preparar los datos faltantes ---
        summary_data = {
            "total_stations_count": len(gdf_stations),
            "selected_stations_count": len(stations_for_analysis),
            "year_range": year_range,
            "selected_months_count": len(meses_numeros),
            "analysis_mode": analysis_mode,
            "selected_regions": sidebar_filters["selected_regions"],
            "selected_municipios": sidebar_filters["selected_municipios"],
            "selected_altitudes": sidebar_filters["selected_altitudes"]
        }
        df_anomalies = calculate_monthly_anomalies(df_monthly_filtered, df_long) 
        
        # Opciones para el reporte
        st.subheader("Seleccionar Secciones para Incluir en el Reporte:")
        report_sections_options = [
            "Resumen General", "Tabla de Estaciones", "Mapa de Distribución Espacial",
            "Análisis de Precipitación Mensual y Anual", "Análisis de Anomalías",
            "Análisis de Extremos Hidrológicos (Percentiles)",
            "Análisis de Índices de Sequía (SPI/SPEI)",
            "Análisis de Frecuencia de Extremos", "Análisis de Correlación", "Análisis ENSO",
            "Análisis de Tendencias y Pronósticos", "Comparación de Periodos"
        ]
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
                        # --- 2. Llamar a la función con los argumentos correctos ---
                        report_pdf_bytes = generate_pdf_report(
                            sections_to_include=selected_report_sections, # Nombre corregido
                            summary_data=summary_data,                   # Argumento añadido
                            df_anomalies=df_anomalies,                   # Argumento añadido
                            report_title=report_title,
                            author_name=author_name,
                            gdf_filtered=gdf_filtered,
                            df_long=df_long,                 # Usar variable local
                            df_anual_melted=df_anual_melted,
                            df_monthly_filtered=df_monthly_filtered,
                            stations_for_analysis=stations_for_analysis,
                            df_enso=df_enso                   # Usar variable local
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



