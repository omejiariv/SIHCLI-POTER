import streamlit as st
from modules.config import Config
import pandas as pd
import numpy as np

def create_sidebar(gdf_stations, df_long):
    with st.sidebar:
        st.image(Config.LOGO_PATH, width=150) if hasattr(Config, 'LOGO_PATH') else None
        st.title("Panel de Control")

        # --- 1. Filtros de Datos (Nulos y Ceros) ---
        with st.expander("🛠️ Procesamiento y Limpieza", expanded=False):
            run_complete_series = st.checkbox("Interpolación (Rellenar huecos)", value=False)
            exclude_nulls = st.checkbox("Excluir datos nulos (NaN)", value=False)
            exclude_zeros = st.checkbox("Excluir valores cero (0)", value=False)

            # Guardar estado de interpolación
            if run_complete_series != st.session_state.get('apply_interpolation'):
                st.session_state['apply_interpolation'] = run_complete_series
                st.rerun()

        st.divider()

        # --- 2. Filtros de Ubicación ---
        st.markdown("### 📍 Filtros de Ubicación")
        
        # A. Filtro por Altitud (Nuevo)
        altitude_options = ["Todos", "0-500", "500-1000", "1000-1500", "1500-2000", "2000-3000", ">3000"]
        selected_alt_range = st.selectbox("Filtrar por Altitud (m):", altitude_options)
        
        # Aplicar filtro de altitud a las estaciones base
        gdf_filtered_base = gdf_stations.copy()
        if selected_alt_range != "Todos":
            if ">" in selected_alt_range:
                min_alt = int(selected_alt_range.replace(">", ""))
                gdf_filtered_base = gdf_filtered_base[gdf_filtered_base[Config.ALTITUDE_COL] >= min_alt]
            else:
                min_alt, max_alt = map(int, selected_alt_range.split("-"))
                gdf_filtered_base = gdf_filtered_base[
                    (gdf_filtered_base[Config.ALTITUDE_COL] >= min_alt) & 
                    (gdf_filtered_base[Config.ALTITUDE_COL] < max_alt)
                ]

        # B. Región
        if Config.REGION_COL in gdf_filtered_base.columns:
            all_regions = sorted(gdf_filtered_base[Config.REGION_COL].unique())
            selected_regions = st.multiselect("Región:", all_regions)
            
            # Filtrar cascada
            if selected_regions:
                gdf_filtered_base = gdf_filtered_base[gdf_filtered_base[Config.REGION_COL].isin(selected_regions)]
        else:
            selected_regions = []

        # C. Municipio
        all_munis = sorted(gdf_filtered_base[Config.MUNICIPALITY_COL].unique())
        selected_municipios = st.multiselect("Municipio:", all_munis)
        
        if selected_municipios:
            gdf_filtered_base = gdf_filtered_base[gdf_filtered_base[Config.MUNICIPALITY_COL].isin(selected_municipios)]

        # D. Estaciones Finales
        available_stations = sorted(gdf_filtered_base[Config.STATION_NAME_COL].unique())
        stations_for_analysis = st.multiselect(
            f"Estaciones ({len(available_stations)}):", 
            available_stations,
            default=available_stations[:3] if len(available_stations) > 0 else []
        )

        gdf_final = gdf_stations[gdf_stations[Config.STATION_NAME_COL].isin(stations_for_analysis)]

        st.divider()

        # --- 3. Filtro de Tiempo y Aplicación de Limpieza ---
        st.markdown("### 📅 Periodo")
        try:
            min_y = int(df_long[Config.YEAR_COL].min())
            max_y = int(df_long[Config.YEAR_COL].max())
            year_range = st.slider("Años:", min_y, max_y, (max_y-10, max_y))
        except:
            year_range = (2000, 2024)

        # FILTRADO DEL DATAFRAME PRINCIPAL (Lo más importante)
        mask = (
            (df_long[Config.YEAR_COL] >= year_range[0]) & 
            (df_long[Config.YEAR_COL] <= year_range[1]) &
            (df_long[Config.STATION_NAME_COL].isin(stations_for_analysis))
        )
        df_monthly_filtered = df_long.loc[mask].copy()

        # Aplicar filtros de limpieza (Nulos y Ceros)
        if exclude_nulls:
            df_monthly_filtered = df_monthly_filtered.dropna(subset=[Config.PRECIPITATION_COL])
        if exclude_zeros:
            df_monthly_filtered = df_monthly_filtered[df_monthly_filtered[Config.PRECIPITATION_COL] != 0]

        # Agrupar anual
        df_anual_melted = df_monthly_filtered.groupby(
            [Config.STATION_NAME_COL, Config.YEAR_COL]
        )[Config.PRECIPITATION_COL].sum().reset_index()

        return (stations_for_analysis, df_anual_melted, df_monthly_filtered, gdf_final, 
                "Histórico", selected_regions, selected_municipios, [], year_range)
