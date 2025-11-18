# modules/sidebar.py
import streamlit as st
from modules.config import Config

def create_sidebar(gdf_stations, df_long):
    """
    Crea la barra lateral con los filtros y devuelve las selecciones.
    """
    with st.sidebar:
        st.image(Config.LOGO_PATH, use_column_width=True)
        st.title("Panel de Control")
        
        st.subheader("Filtros de Estaciones")
        
        # --- Filtro 1: Región ---
        regiones = sorted(gdf_stations[Config.REGION_COL].unique())
        selected_regions = st.multiselect("Seleccione Región(es):", regiones, default=regiones)
        
        # Filtrar por región
        if selected_regions:
            gdf_filtered = gdf_stations[gdf_stations[Config.REGION_COL].isin(selected_regions)]
        else:
            gdf_filtered = gdf_stations

        # --- Filtro 2: Municipio ---
        municipios = sorted(gdf_filtered[Config.MUNICIPALITY_COL].unique())
        selected_municipios = st.multiselect("Seleccione Municipio(s):", municipios, default=municipios)
        
        # Filtrar por municipio
        if selected_municipios:
            gdf_filtered = gdf_filtered[gdf_filtered[Config.MUNICIPALITY_COL].isin(selected_municipios)]

        # --- Filtro 3: Altitud ---
        # (Simplificado para el ejemplo, puedes agregar lógica de rangos si lo tenías antes)
        selected_altitudes = [] # Placeholder si no usas filtro de altitud complejo
        
        # --- Selección de Estaciones Específicas ---
        estaciones_disponibles = sorted(gdf_filtered[Config.STATION_NAME_COL].unique())
        stations_for_analysis = st.multiselect(
            "Seleccione Estaciones para Análisis:", 
            estaciones_disponibles,
            default=estaciones_disponibles[:5] # Pre-seleccionar algunas
        )

        st.markdown("---")
        st.subheader("Periodo de Análisis")
        
        # --- Filtro de Fecha (Rango de Años) ---
        min_year = int(df_long[Config.YEAR_COL].min())
        max_year = int(df_long[Config.YEAR_COL].max())
        
        year_range = st.slider(
            "Seleccione Rango de Años:",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )

        st.markdown("---")
        st.subheader("Configuración de Análisis")
        analysis_mode = st.selectbox("Modo de Análisis:", ["Histórico Completo", "Comparativo ENSO"])

        # --- Filtrado de Datos ---
        # Filtrar los DataFrames de datos basados en la selección
        
        # 1. Filtrar por estaciones seleccionadas
        if stations_for_analysis:
            df_filtered_stations = df_long[df_long[Config.STATION_NAME_COL].isin(stations_for_analysis)]
        else:
            df_filtered_stations = df_long # Si no hay selección, no mostrar nada o todo (depende de la lógica)

        # 2. Filtrar por rango de años
        df_monthly_filtered = df_filtered_stations[
            (df_filtered_stations[Config.YEAR_COL] >= year_range[0]) &
            (df_filtered_stations[Config.YEAR_COL] <= year_range[1])
        ]
        
        # Crear df_anual_melted (agregado anual) para compatibilidad
        # Agrupamos por Estación y Año, sumando precipitación
        df_anual_melted = df_monthly_filtered.groupby(
            [Config.STATION_NAME_COL, Config.YEAR_COL]
        )[Config.PRECIPITATION_COL].sum().reset_index()

        # --- RETORNO DE 9 VALORES (CRÍTICO PARA QUE COINCIDA CON APP.PY) ---
        return (
            stations_for_analysis, 
            df_anual_melted, 
            df_monthly_filtered, 
            gdf_filtered, 
            analysis_mode, 
            selected_regions, 
            selected_municipios, 
            selected_altitudes, 
            year_range # <--- ¡ESTE ERA EL QUE FALTABA!
        )
