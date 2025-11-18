# modules/sidebar.py
import streamlit as st
from modules.config import Config

def create_sidebar(gdf_stations, df_long):
    """
    Crea la barra lateral con los filtros y devuelve las selecciones.
    Retorna 9 valores para coincidir con app.py.
    """
    with st.sidebar:
        # --- 1. LOGO ARREGLADO (Ancho controlado) ---
        if hasattr(Config, 'LOGO_PATH'):
            try:
                # Usamos width=150 en lugar de use_column_width para que no sea gigante
                st.image(Config.LOGO_PATH, width=150) 
            except:
                pass
        
        st.title("Panel de Control")
        
        # --- 2. SECCIÓN DE PREPROCESAMIENTO (REACTIVADA) ---
        st.subheader("🛠️ Preprocesamiento")
        run_complete_series = st.checkbox(
            "Completar Series Faltantes (Interpolación)", 
            value=False,
            help="Rellena huecos en los datos mensuales usando interpolación lineal."
        )
        
        st.markdown("---")
        st.subheader("📍 Filtros de Estaciones")
        
        # --- Filtro 1: Región ---
        regiones = sorted(gdf_stations[Config.REGION_COL].unique())
        selected_regions = st.multiselect("Región(es):", regiones, default=regiones)
        
        if selected_regions:
            gdf_filtered = gdf_stations[gdf_stations[Config.REGION_COL].isin(selected_regions)]
        else:
            gdf_filtered = gdf_stations

        # --- Filtro 2: Municipio ---
        municipios = sorted(gdf_filtered[Config.MUNICIPALITY_COL].unique())
        selected_municipios = st.multiselect("Municipio(s):", municipios, default=municipios)
        
        if selected_municipios:
            gdf_filtered = gdf_filtered[gdf_filtered[Config.MUNICIPALITY_COL].isin(selected_municipios)]

        selected_altitudes = [] 
        
        # --- Selección de Estaciones ---
        estaciones_disponibles = sorted(gdf_filtered[Config.STATION_NAME_COL].unique())
        default_stations = estaciones_disponibles[:5] if len(estaciones_disponibles) > 0 else []
        
        stations_for_analysis = st.multiselect(
            "Estaciones para Análisis:", 
            estaciones_disponibles,
            default=default_stations
        )

        st.markdown("---")
        st.subheader("📅 Periodo de Análisis")
        
        min_year = int(df_long[Config.YEAR_COL].min())
        max_year = int(df_long[Config.YEAR_COL].max())
        
        year_range = st.slider(
            "Rango de Años:",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )

        st.markdown("---")
        analysis_mode = st.selectbox("Modo de Análisis:", ["Histórico Completo", "Comparativo ENSO"])

        # --- Lógica de Filtrado ---
        if stations_for_analysis:
            df_filtered_stations = df_long[df_long[Config.STATION_NAME_COL].isin(stations_for_analysis)]
        else:
            df_filtered_stations = df_long 

        df_monthly_filtered = df_filtered_stations[
            (df_filtered_stations[Config.YEAR_COL] >= year_range[0]) &
            (df_filtered_stations[Config.YEAR_COL] <= year_range[1])
        ]
        
        # --- APLICAR PREPROCESAMIENTO SI SE SELECCIONÓ ---
        # Aquí está el truco: Si el usuario marcó el checkbox, llamamos a la función de procesado
        # Nota: Para aplicar esto realmente, necesitaríamos importar complete_series aquí o devolver el flag.
        # Para mantener la estructura de 9 valores simple, vamos a devolver el flag "run_complete_series"
        # "escondido" o usaremos st.session_state.
        
        if run_complete_series:
             st.session_state['apply_interpolation'] = True
        else:
             st.session_state['apply_interpolation'] = False
        
        df_anual_melted = df_monthly_filtered.groupby(
            [Config.STATION_NAME_COL, Config.YEAR_COL]
        )[Config.PRECIPITATION_COL].sum().reset_index()

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
