import streamlit as st
import pandas as pd
import warnings

# Configuración de página (DEBE SER LA PRIMERA LÍNEA)
st.set_page_config(page_title="SIHCLI-POTER", page_icon="🌧️", layout="wide")
warnings.filterwarnings('ignore')

# Importaciones de módulos
from modules.config import Config
from modules.data_processor import load_and_process_all_data, complete_series
from modules.sidebar import create_sidebar
from modules.reporter import generate_pdf_report

# Importación segura de db_manager
try:
    import modules.db_manager as db_manager
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("Advertencia: db_manager no encontrado.")

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
    display_drought_analysis_tab,
    display_advanced_maps_tab,
    display_life_zones_tab,
    display_land_cover_analysis_tab,
    display_climate_scenarios_tab,
    display_station_table_tab,
    display_weekly_forecast_tab,      
    display_satellite_imagery_tab,
    display_climate_forecast_tab
)

def main():
    # --- 1. INICIALIZACIÓN ---
    if DB_AVAILABLE:
        try:
            db_manager.init_db()
        except Exception as e:
            print(f"Error iniciando DB (No crítico): {e}")

    # Inicialización de estado (Zonas de Vida)
    for key in ['lz_raster_result', 'lz_profile', 'lz_names', 'lz_colors']:
        if key not in st.session_state:
            st.session_state[key] = None

    # --- 2. CARGA DE DATOS ---
    with st.spinner("Cargando datos del sistema..."):
        gdf_stations, gdf_municipios, df_long, df_enso, gdf_subcuencas, gdf_predios = load_and_process_all_data()
    
    if gdf_stations is None or df_long is None:
        st.error("⚠️ Error Fatal: No se pudieron cargar los datos. Verifica la conexión a BD.")
        st.stop()

    # --- 3. SIDEBAR (FILTROS) ---
    (stations_for_analysis, df_anual_melted, df_monthly_filtered, gdf_filtered, analysis_mode, 
     sel_regions, sel_munis, selected_months, year_range) = create_sidebar(gdf_stations, df_long)

    # Procesamiento de Interpolación
    if st.session_state.get('apply_interpolation', False):
        with st.spinner("Procesando interpolación..."):
            df_monthly_filtered = complete_series(df_monthly_filtered)
            df_anual_melted = df_monthly_filtered.groupby([Config.STATION_NAME_COL, Config.YEAR_COL])[Config.PRECIPITATION_COL].sum().reset_index()

    # Fechas
    try:
        start_date = pd.to_datetime(f"{year_range[0]}-01-01")
        end_date = pd.to_datetime(f"{year_range[1]}-12-31")
    except:
        start_date, end_date = None, None

    # Datos Completos para Pronósticos
    mask_base = (
        (df_long[Config.YEAR_COL] >= year_range[0]) & 
        (df_long[Config.YEAR_COL] <= year_range[1]) &
        (df_long[Config.STATION_NAME_COL].isin(stations_for_analysis))
    )
    df_complete_filtered = df_long.loc[mask_base].copy()

    # Argumentos Unificados
    display_args = {
        "df_long": df_monthly_filtered,
        "df_complete": df_complete_filtered,
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
        "selected_months": selected_months,
        "year_range": year_range,
        "start_date": start_date,
        "end_date": end_date
    }

    # --- 4. RENDERIZADO PRINCIPAL ---
    
    # Pestañas
    tab_titles = [
        "🏠 Inicio", "🚨 Monitoreo (Tiempo Real)", "🗺️ Distribución", "📈 Gráficos", 
        "📊 Estadísticas", "🔮 Pronóstico Climático", "📉 Tendencias", "⚠️ Anomalías", 
        "🔗 Correlación", "🌊 Extremos", "🌍 Mapas Avanzados", "🧪 Corrección de Sesgo",
        "🌿 Cobertura", "🌱 Zonas Vida", "🌡️ Clima Futuro", "📄 Reporte"
    ]
    
    tabs = st.tabs(tab_titles)

    # --- FUNCIÓN HELPER PARA MOSTRAR CAJA DE FILTROS ---
    def show_summary():
        display_current_filters(
            stations_sel=stations_for_analysis, 
            regions_sel=sel_regions, 
            munis_sel=sel_munis, 
            year_range=year_range,
            interpolacion="Si" if st.session_state.get('apply_interpolation') else "No",
            df_data=df_monthly_filtered
        )

    # --- PESTAÑA 0: INICIO (LIMPIA) ---
    with tabs[0]:
        display_welcome_tab()

    # --- PESTAÑA 1: MONITOREO ---
    with tabs[1]:
        show_summary()
        display_realtime_dashboard(df_complete_filtered, gdf_stations, gdf_filtered)

    # --- PESTAÑA 2: DISTRIBUCIÓN ---
    with tabs[2]:
        show_summary()
        display_spatial_distribution_tab(
            user_loc=None, 
            interpolacion="Si" if st.session_state.get('apply_interpolation') else "No", 
            **display_args
        )

    # --- RESTO DE PESTAÑAS ---
    with tabs[3]: 
        show_summary()
        display_graphs_tab(**display_args)

    with tabs[4]: 
        show_summary()
        display_stats_tab(**display_args)
        st.markdown("---")
        display_station_table_tab(**display_args)

    with tabs[5]: 
        show_summary()
        display_climate_forecast_tab(**display_args)

    with tabs[6]: 
        show_summary()
        display_trends_and_forecast_tab(**display_args)

    with tabs[7]: 
        show_summary()
        display_anomalies_tab(**display_args)

    with tabs[8]: 
        show_summary()
        display_correlation_tab(**display_args)

    with tabs[9]: 
        show_summary()
        display_drought_analysis_tab(**display_args)

    with tabs[10]: 
        show_summary()
        display_advanced_maps_tab(**display_args)

    with tabs[11]: 
        show_summary()
        # Importación local segura
        try:
            from modules.visualizer import display_bias_correction_tab
            display_bias_correction_tab(**display_args)
        except Exception as e:
            st.error(f"Error cargando módulo de Sesgo: {e}")

    with tabs[12]: 
        show_summary()
        display_land_cover_analysis_tab(**display_args)

    with tabs[13]: 
        show_summary()
        display_life_zones_tab(**display_args)

    with tabs[14]: 
        show_summary()
        display_climate_scenarios_tab(**display_args)

    with tabs[15]: 
        show_summary()
        st.header("Generar Reporte PDF")
        if st.button("Generar Reporte Ejecutivo", type="primary"):
            with st.spinner("Generando..."):
                res = {"n_estaciones": len(stations_for_analysis), "rango": f"{year_range}"}
                pdf = generate_pdf_report(df_monthly_filtered, gdf_filtered, res)
                if pdf: st.download_button("📥 Descargar PDF", pdf, "reporte.pdf", "application/pdf")
                else: st.error("Error al generar reporte.")

    # Estilos CSS finales
    st.markdown("""
    <style>
        div[data-baseweb="tab-list"] { gap: 5px; }
        div[data-baseweb="tab"] { background-color: #f0f2f6; border-radius: 4px 4px 0 0; padding: 0 16px; border: 1px solid #e0e0e0; border-bottom: none; }
        div[aria-selected="true"] { background-color: white; border-top: 3px solid #1f77b4; }
        .stTabs [data-baseweb="tab-panel"] { padding-top: 1rem; }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
