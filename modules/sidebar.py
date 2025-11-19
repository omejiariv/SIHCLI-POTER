import streamlit as st
from modules.config import Config
import pandas as pd

def create_sidebar(gdf_stations, df_long):
    with st.sidebar:
        if hasattr(Config, 'LOGO_PATH'):
            try:
                st.image(Config.LOGO_PATH, width=150)
            except: pass
            
        st.title("Panel de Control")
        
        # 1. Optimización de Interpolación
        st.markdown("### ⚙️ Procesamiento")
        run_complete_series = st.checkbox("Interpolación de Datos", value=False)
        
        if run_complete_series != st.session_state.get('apply_interpolation'):
            st.session_state['apply_interpolation'] = run_complete_series
            st.rerun()

        st.divider()

        # 2. Filtros de Ubicación (Seguros)
        st.markdown("### 📍 Filtros de Ubicación")
        
        # Verificar si hay datos de estaciones
        if gdf_stations is not None and not gdf_stations.empty and Config.REGION_COL in gdf_stations.columns:
            all_regions = sorted(gdf_stations[Config.REGION_COL].unique())
            selected_regions = st.multiselect("Región:", all_regions) # Sin default para cargar rápido
            
            if selected_regions:
                mask_region = gdf_stations[Config.REGION_COL].isin(selected_regions)
                filtered_munis = sorted(gdf_stations[mask_region][Config.MUNICIPALITY_COL].unique())
            else:
                filtered_munis = []
                
            selected_municipios = st.multiselect("Municipio:", filtered_munis)
            
            # Lógica de filtrado
            if selected_municipios:
                mask_final = gdf_stations[Config.MUNICIPALITY_COL].isin(selected_municipios)
            elif selected_regions:
                mask_final = gdf_stations[Config.REGION_COL].isin(selected_regions)
            else:
                mask_final = [True] * len(gdf_stations)
                
            available_stations = sorted(gdf_stations[mask_final][Config.STATION_NAME_COL].unique())
            default_stations = available_stations[:1] if len(available_stations) > 0 else []
            
            stations_for_analysis = st.multiselect(
                f"Estaciones ({len(available_stations)}):", 
                available_stations,
                default=default_stations
            )
            
            gdf_filtered = gdf_stations[gdf_stations[Config.STATION_NAME_COL].isin(stations_for_analysis)]
        else:
            st.warning("⚠️ No se cargaron estaciones.")
            stations_for_analysis = []
            gdf_filtered = pd.DataFrame()
            selected_regions = []
            selected_municipios = []

        st.divider()

        # 3. Filtro de Tiempo (CORRECCIÓN DEL CRASH)
        st.markdown("### 📅 Periodo")
        
        # Verificación de seguridad: Si df_long está vacío o no tiene columna de año
        if df_long is not None and not df_long.empty and Config.YEAR_COL in df_long.columns:
            try:
                min_y = int(df_long[Config.YEAR_COL].min())
                max_y = int(df_long[Config.YEAR_COL].max())
                # Protección contra NaNs
                if pd.isna(min_y) or pd.isna(max_y):
                    min_y, max_y = 2000, 2024
            except:
                min_y, max_y = 2000, 2024
                
            year_range = st.slider("Años:", min_y, max_y, (max_y-5, max_y))
            
            # Filtrar datos
            mask_time = (df_long[Config.YEAR_COL] >= year_range[0]) & (df_long[Config.YEAR_COL] <= year_range[1])
            mask_station = df_long[Config.STATION_NAME_COL].isin(stations_for_analysis)
            df_monthly_filtered = df_long.loc[mask_time & mask_station]
            
            df_anual_melted = df_monthly_filtered.groupby(
                [Config.STATION_NAME_COL, Config.YEAR_COL]
            )[Config.PRECIPITATION_COL].sum().reset_index()
        else:
            # Valores por defecto si no hay datos (evita el crash)
            st.warning("⚠️ No hay datos de precipitación cargados.")
            year_range = (2000, 2024)
            df_monthly_filtered = pd.DataFrame()
            df_anual_melted = pd.DataFrame()

        analysis_mode = "Histórico"
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
