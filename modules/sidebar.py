# modules/sidebar.py
import streamlit as st
from modules.config import Config

def create_sidebar(gdf_stations, df_long):
    """
    Crea la barra lateral con los filtros y devuelve las selecciones.
    Retorna 9 valores para coincidir con app.py.
    """
    with st.sidebar:
        if hasattr(Config, 'LOGO_PATH'):
            # Intentar mostrar logo si existe, sino título simple
            try:
                st.image(Config.LOGO_PATH, use_column_width=True)
            except:
                pass
        
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
        # (Simplificado, retornamos lista vacía si no se usa lógica compleja aquí)
        selected_altitudes = [] 
        
        # --- Selección de Estaciones Específicas ---
        estaciones_disponibles = sorted(gdf_filtered[Config.STATION_NAME_COL].unique())
        # Pre-seleccionar hasta 5 estaciones por defecto para que no esté vacío
        default_stations = estaciones_disponibles[:5] if len(estaciones_disponibles) > 0 else []
        
        stations_for_analysis = st.multiselect(
            "Seleccione Estaciones para Análisis:", 
            estaciones_disponibles,
            default=default_stations
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
            df_filtered_stations = df_long 

        # 2. Filtrar por rango de años
        df_monthly_filtered = df_filtered_stations[
            (df_filtered_stations[Config.YEAR_COL] >= year_range[0]) &
            (df_filtered_stations[Config.YEAR_COL] <= year_range[1])
        ]
        
        # Crear df_anual_melted (agregado anual) para compatibilidad con visualizer
        df_anual_melted = df_monthly_filtered.groupby(
            [Config.STATION_NAME_COL, Config.YEAR_COL]
        )[Config.PRECIPITATION_COL].sum().reset_index()

        # --- RETORNO DE 9 VALORES (CRÍTICO) ---
        # El orden DEBE ser:
        # 1. stations_for_analysis
        # 2. df_anual_melted
        # 3. df_monthly_filtered
        # 4. gdf_filtered
        # 5. analysis_mode
        # 6. selected_regions
        # 7. selected_municipios
        # 8. selected_altitudes
        # 9. year_range
        
        return (
            stations_for_analysis, 
            df_anual_melted, 
            df_monthly_filtered, 
            gdf_filtered, 
            analysis_mode, 
            selected_regions, 
            selected_municipios, 
            selected_altitudes, 
            year_range 
        )
