import streamlit as st
from modules.config import Config
import pandas as pd

def create_sidebar(gdf_stations, df_long):
    with st.sidebar:
        st.title("🎛️ Panel de Control")
        
        # 1. Optimización de Interpolación
        st.markdown("### ⚙️ Procesamiento")
        run_complete_series = st.checkbox(
            "Interpolación de Datos", 
            value=False,
            help="⚠️ Activar esto puede hacer la app lenta si hay muchas estaciones."
        )
        
        # Guardar estado en session_state para no perderlo
        if run_complete_series != st.session_state.get('apply_interpolation'):
            st.session_state['apply_interpolation'] = run_complete_series
            st.rerun() # Recargar solo si cambia

        st.divider()

        # 2. Filtros en Cascada (Optimizados)
        st.markdown("### 📍 Filtros de Ubicación")
        
        # Región
        all_regions = sorted(gdf_stations[Config.REGION_COL].unique())
        selected_regions = st.multiselect("Región:", all_regions, default=all_regions[:1]) # Default solo 1 para velocidad
        
        # Filtrar municipios basado en región
        if selected_regions:
            mask_region = gdf_stations[Config.REGION_COL].isin(selected_regions)
            filtered_munis = sorted(gdf_stations[mask_region][Config.MUNICIPALITY_COL].unique())
        else:
            filtered_munis = []
            
        selected_municipios = st.multiselect("Municipio:", filtered_munis)
        
        # Filtrar estaciones basado en municipio
        if selected_municipios:
            mask_final = gdf_stations[Config.MUNICIPALITY_COL].isin(selected_municipios)
        elif selected_regions:
            mask_final = gdf_stations[Config.REGION_COL].isin(selected_regions)
        else:
            mask_final = [True] * len(gdf_stations)
            
        available_stations = sorted(gdf_stations[mask_final][Config.STATION_NAME_COL].unique())
        
        # LIMITAR selección por defecto para evitar crash
        default_stations = available_stations[:3] if len(available_stations) > 0 else []
        
        stations_for_analysis = st.multiselect(
            f"Estaciones ({len(available_stations)} disp.):", 
            available_stations,
            default=default_stations
        )
        
        if len(stations_for_analysis) > 20:
            st.warning("⚠️ Más de 20 estaciones seleccionadas. El rendimiento puede disminuir.")

        st.divider()

        # 3. Filtro de Tiempo
        st.markdown("### 📅 Periodo")
        min_y = int(df_long[Config.YEAR_COL].min())
        max_y = int(df_long[Config.YEAR_COL].max())
        year_range = st.slider("Años:", min_y, max_y, (max_y-10, max_y))

        # Lógica de filtrado de DataFrames
        gdf_filtered = gdf_stations[gdf_stations[Config.STATION_NAME_COL].isin(stations_for_analysis)]
        
        # Filtrar datos mensuales (lo pesado)
        # Optimizamos usando .loc y el índice si es posible, o mascara booleana simple
        mask_time = (df_long[Config.YEAR_COL] >= year_range[0]) & (df_long[Config.YEAR_COL] <= year_range[1])
        mask_station = df_long[Config.STATION_NAME_COL].isin(stations_for_analysis)
        df_monthly_filtered = df_long.loc[mask_time & mask_station]

        # Agrupar anual
        df_anual_melted = df_monthly_filtered.groupby(
            [Config.STATION_NAME_COL, Config.YEAR_COL]
        )[Config.PRECIPITATION_COL].sum().reset_index()

        analysis_mode = "Histórico" # Placeholder para compatibilidad
        selected_altitudes = []

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
