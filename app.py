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
    display_enso_tab,
    display_drought_analysis_tab,
    display_advanced_maps_tab,
    display_life_zones_tab,
    display_land_cover_analysis_tab,
    display_climate_scenarios_tab,
    display_station_table_tab,
    display_weekly_forecast_tab,      # Importaciones de seguridad
    display_satellite_imagery_tab,
    display_climate_forecast_tab
)

# Configuración de página
st.set_page_config(page_title="SIHCLI-POTER", page_icon="🌧️", layout="wide")
warnings.filterwarnings('ignore')

def main():
    # 1. Cargar Datos
    gdf_stations, gdf_municipios, df_long, df_enso, gdf_subcuencas, gdf_predios = load_and_process_all_data()
    
    if gdf_stations is None or df_long is None:
        st.error("⚠️ Error Fatal: No se pudieron cargar los datos. Verifica la conexión a BD.")
        st.stop()

    # 2. Sidebar
    (stations_for_analysis, df_anual_melted, df_monthly_filtered, gdf_filtered, analysis_mode, 
     sel_regions, sel_munis, sel_alts, year_range) = create_sidebar(gdf_stations, df_long)

    # Lógica de interpolación
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

    # 3. Argumentos Unificados (display_args)
    display_args = {
        "df_long": df_long, 
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
        "selected_altitudes": sel_alts,
        "year_range": year_range,
        "start_date": start_date,
        "end_date": end_date
    }

    # --- NUEVO: CAJA DE INFORMACIÓN GLOBAL ---
    display_current_filters(stations_for_analysis, sel_regions, sel_munis, year_range)    

    # 4. Pestañas
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
        "🌊 ENSO", 
        "🌊 Extremos",
        "🌍 Mapas Avanzados",
        "🧪 Corrección de Sesgo",
        "🌿 Cobertura", 
        "🌱 Zonas Vida", 
        "🌡️ Clima Futuro", 
        "📄 Reporte"
    ]
    
    tabs = st.tabs(tab_titles)

    # 5. Renderizado (Usando siempre display_args)
    with tabs[0]: 
        display_welcome_tab()
    
    with tabs[1]: 
        display_realtime_dashboard(df_long, gdf_stations, gdf_filtered)

    with tabs[2]: 
        display_spatial_distribution_tab(**display_args)

    with tabs[3]: 
        display_graphs_tab(**display_args)

    with tabs[4]: 
        display_stats_tab(**display_args)
        st.markdown("---")
        display_station_table_tab(**display_args)

    with tabs[5]: # AHORA ES PRONÓSTICO CLIMÁTICO
        from modules.visualizer import display_climate_forecast_tab
        display_climate_forecast_tab(**display_args)

    with tabs[6]: # AHORA ES TENDENCIAS
        display_trends_and_forecast_tab(**display_args)

    with tabs[7]: 
        display_anomalies_tab(**display_args)

    with tabs[8]: 
        display_correlation_tab(**display_args)

    with tabs[9]: 
        display_enso_tab(**display_args)

    with tabs[10]: 
        display_drought_analysis_tab(**display_args)

    with tabs[11]: 
        display_advanced_maps_tab(**display_args)

    with tabs[12]: # Ajusta el índice según corresponda
        from modules.visualizer import display_bias_correction_tab
        display_bias_correction_tab(**display_args)

    with tabs[13]: 
        display_land_cover_analysis_tab(**display_args)

    with tabs[14]: 
        display_life_zones_tab(**display_args)

    with tabs[15]: 
        display_climate_scenarios_tab(**display_args)

    with tabs[16]: 
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



