import streamlit as st
import pandas as pd
import warnings
from modules.config import Config
from modules.data_processor import load_and_process_all_data, complete_series
from modules.sidebar import create_sidebar
from modules.reporter import generate_pdf_report
from modules.visualizer import (
    display_current_filters,
    display_welcome_tab,
    display_realtime_dashboard,
    display_spatial_distribution_tab,
    display_graphs_tab,
    display_stats_tab,
    display_trends_and_forecast_tab,
    display_anomalies_tab,
    display_correlation_tab,
    # display_enso_tab,  <-- ELIMINADO (Ya integrado en Pronóstico Climático)
    display_drought_analysis_tab,
    display_advanced_maps_tab,
    display_life_zones_tab,
    display_land_cover_analysis_tab,
    display_climate_scenarios_tab,
    display_station_table_tab,
    display_weekly_forecast_tab,      
    display_satellite_imagery_tab,
    display_climate_forecast_tab,
    import modules.db_manager as db_manager
)

# Configuración de página
st.set_page_config(page_title="SIHCLI-POTER", page_icon="🌧️", layout="wide")
warnings.filterwarnings('ignore')

```python
def main():
    # Inicializar base de datos de preferencias
    db_manager.init_db()
    # --- INICIALIZACIÓN DE ESTADO (ZONAS DE VIDA) ---
    if 'lz_raster_result' not in st.session_state:
        st.session_state.lz_raster_result = None
    if 'lz_profile' not in st.session_state:
        st.session_state.lz_profile = None
    if 'lz_names' not in st.session_state:
        st.session_state.lz_names = None
    if 'lz_colors' not in st.session_state:
        st.session_state.lz_colors = None

    # 1. Cargar Datos
    gdf_stations, gdf_municipios, df_long, df_enso, gdf_subcuencas, gdf_predios = load_and_process_all_data()
    
    if gdf_stations is None or df_long is None:
        st.error("⚠️ Error Fatal: No se pudieron cargar los datos. Verifica la conexión a BD.")
        st.stop()

    # 2. Sidebar (AHORA RECIBE 9 VALORES DE RETORNO)
    (stations_for_analysis, df_anual_melted, df_monthly_filtered, gdf_filtered, analysis_mode, 
     sel_regions, sel_munis, selected_months, year_range) = create_sidebar(gdf_stations, df_long)

    # Lógica de interpolación (Aplica sobre el filtrado mensual)
    if st.session_state.get('apply_interpolation', False):
        with st.spinner("Procesando interpolación..."):
            df_monthly_filtered = complete_series(df_monthly_filtered)
            df_anual_melted = df_monthly_filtered.groupby([Config.STATION_NAME_COL, Config.YEAR_COL])[Config.PRECIPITATION_COL].sum().reset_index()

    # Calcular fechas
    try:
        start_date = pd.to_datetime(f"{year_range[0]}-01-01")
        end_date = pd.to_datetime(f"{year_range[1]}-12-31")
    except:
        start_date, end_date = None, None

    # 3. PREPARAR DATAFRAME COMPLETO (CONTINUO) PARA PRONÓSTICOS
    # Este DF se filtra por Años y Estaciones, pero MANTIENE todos los meses.
    mask_base = (
        (df_long[Config.YEAR_COL] >= year_range[0]) & 
        (df_long[Config.YEAR_COL] <= year_range[1]) &
        (df_long[Config.STATION_NAME_COL].isin(stations_for_analysis))
    )
    df_complete_filtered = df_long.loc[mask_base].copy()

    # 4. Argumentos Unificados (display_args)
    display_args = {
        "df_long": df_monthly_filtered,        # Por defecto: Filtrado (Mapas, Estadísticas)
        "df_complete": df_complete_filtered,   # Para Pronósticos (Serie continua)
        "gdf_stations": gdf_stations, 
        "gdf_filtered": gdf_filtered,
        "gdf_municipios": gdf_municipios, 
        "gdf_subcuencas": gdf_subcuencas, 
        "gdf_predios": gdf_predios,
        "df_enso": df_enso, 
        "stations_for_analysis": stations_for_analysis,
        "df_anual_melted": df_anual_melted, 
        "df_monthly_filtered": df_monthly_filtered,
        "analysis_mode": analysis_mode,
        "selected_regions": sel_regions,
        "selected_municipios": sel_munis,
        "selected_months": selected_months,    # Nueva variable: Meses seleccionados (lista de int)
        "year_range": year_range,
        "start_date": start_date,
        "end_date": end_date
    }

    # --- CAJA DE INFORMACIÓN GLOBAL ---
    display_current_filters(stations_for_analysis, sel_regions, sel_munis, year_range)      

    # 5. Pestañas (LISTA ACTUALIZADA - SIN "ENSO")
    tab_titles = [
        "🏠 Inicio", 
        "🚨 Monitoreo (Tiempo Real)", 
        "🗺️ Distribución", 
        "📈 Gráficos", 
        "📊 Estadísticas",
        "🔮 Pronóstico Climático", 
        "📉 Tendencias",
        "⚠️ Anomalías", 
        "🔗 Correlación", 
        # "🌊 ENSO",  <-- ELIMINADO
        "🌊 Extremos",
        "🌍 Mapas Avanzados",
        "🧪 Corrección de Sesgo",
        "🌿 Cobertura", 
        "🌱 Zonas Vida", 
        "🌡️ Clima Futuro", 
        "📄 Reporte"
    ]
    
    tabs = st.tabs(tab_titles)

    # 6. Renderizado (Usando siempre display_args)
    
    # TAB 0: INICIO
    with tabs[0]: 
        display_welcome_tab()
    
    # TAB 1: MONITOREO
    with tabs[1]: 
        display_realtime_dashboard(df_complete_filtered, gdf_stations, gdf_filtered)

    # TAB 2: DISTRIBUCIÓN (Con argumentos corregidos)
    with tabs[2]: 
        display_spatial_distribution_tab(
            user_loc=None, 
            gdf_region=None, 
            interpolacion="Si" if st.session_state.get('apply_interpolation') else "No", 
            **display_args
        )

    # TAB 3: GRÁFICOS
    with tabs[3]: 
        display_graphs_tab(**display_args)

    # TAB 4: ESTADÍSTICAS
    with tabs[4]: 
        display_stats_tab(**display_args)
        st.markdown("---")
        display_station_table_tab(**display_args)

    # TAB 5: PRONÓSTICO CLIMÁTICO (INTEGRADO: IRI + HISTORIA + PROPHET)
    with tabs[5]: 
        display_climate_forecast_tab(**display_args)

    # TAB 6: TENDENCIAS
    with tabs[6]: 
        display_trends_and_forecast_tab(**display_args)

    # TAB 7: ANOMALÍAS
    with tabs[7]: 
        display_anomalies_tab(**display_args)

    # TAB 8: CORRELACIÓN
    with tabs[8]: 
        display_correlation_tab(**display_args)

    # NOTA: TAB 9 ANTERIOR ERA ENSO. AHORA "EXTREMOS" TOMA SU LUGAR.
    
    # TAB 9: EXTREMOS
    with tabs[9]: 
        # display_enso_tab(**display_args) <-- ELIMINADO
        display_drought_analysis_tab(**display_args) # Asumo que esta es la de extremos/sequía

    # TAB 10: MAPAS AVANZADOS
    with tabs[10]: 
        display_advanced_maps_tab(**display_args)

    # TAB 11: CORRECCIÓN DE SESGO
    with tabs[11]: 
        from modules.visualizer import display_bias_correction_tab
        display_bias_correction_tab(**display_args)

    # TAB 12: COBERTURA
    with tabs[12]: 
        display_land_cover_analysis_tab(**display_args)

    # TAB 13: ZONAS DE VIDA
    with tabs[13]: 
        display_life_zones_tab(**display_args)

    # TAB 14: CLIMA FUTURO
    with tabs[14]: 
        display_climate_scenarios_tab(**display_args)

    # TAB 15: REPORTE
    with tabs[15]: 
        st.header("Generar Reporte PDF")
        if st.button("Generar Reporte Ejecutivo", type="primary"):
            with st.spinner("Generando..."):
                res = {"n_estaciones": len(stations_for_analysis), "rango": f"{year_range}"}
                pdf = generate_pdf_report(df_monthly_filtered, gdf_filtered, res)
                if pdf: 
                    st.download_button("📥 Descargar PDF", pdf, "reporte.pdf", "application/pdf")
                else:
                    st.error("Error al generar reporte.")

    # CSS Estético
    st.markdown("""
    <style>
        div[data-baseweb="tab-list"] { gap: 5px; }
        div[data-baseweb="tab"] { background-color: #f0f2f6; border-radius: 4px 4px 0 0; padding: 0 16px; border: 1px solid #e0e0e0; border-bottom: none; }
        div[aria-selected="true"] { background-color: white; border-top: 3px solid #1f77b4; }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
