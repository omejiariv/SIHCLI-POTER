import streamlit as st
import pandas as pd
import warnings

# --- 1. CONFIGURACIÓN DE PÁGINA (PRIMERA LÍNEA ABSOLUTA) ---
st.set_page_config(page_title="SIHCLI-POTER", page_icon="🌧️", layout="wide")
warnings.filterwarnings('ignore')

# --- 2. IMPORTACIONES ---
try:
    from modules.config import Config
    from modules.data_processor import load_and_process_all_data, complete_series
    from modules.sidebar import create_sidebar
    from modules.reporter import generate_pdf_report
    
    # Intentamos importar db_manager pero no dejamos que rompa la app
    try:
        import modules.db_manager as db_manager
        DB_AVAILABLE = True
    except ImportError:
        DB_AVAILABLE = False

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
except Exception as e:
    st.error(f"Error crítico importando módulos: {e}")
    st.stop()

def main():
    # --- A. INICIALIZACIÓN ---
    if DB_AVAILABLE:
        try: db_manager.init_db()
        except: pass

    for k in ['lz_raster_result', 'lz_profile', 'lz_names', 'lz_colors']:
        if k not in st.session_state: st.session_state[k] = None

    # --- B. CARGA DE DATOS (CON SPINNER VISIBLE) ---
    data_loaded = False
    with st.spinner("Cargando datos..."):
        try:
            gdf_stations, gdf_municipios, df_long, df_enso, gdf_subcuencas, gdf_predios = load_and_process_all_data()
            data_loaded = True
        except Exception as e:
            st.error(f"Error cargando datos: {e}")
            st.stop()
    
    if not data_loaded or gdf_stations is None or df_long is None:
        st.error("No se pudieron cargar los datos. Revise la conexión.")
        st.stop()

    # --- C. SIDEBAR ---
    try:
        (stations_for_analysis, df_anual_melted, df_monthly_filtered, gdf_filtered, analysis_mode, 
         sel_regions, sel_munis, selected_months, year_range) = create_sidebar(gdf_stations, df_long)
    except Exception as e:
        st.error(f"Error en Sidebar: {e}")
        st.stop()

    # Procesamiento adicional (Interpolación)
    if st.session_state.get('apply_interpolation', False):
        with st.spinner("Interpolando..."):
            df_monthly_filtered = complete_series(df_monthly_filtered)
            df_anual_melted = df_monthly_filtered.groupby([Config.STATION_NAME_COL, Config.YEAR_COL])[Config.PRECIPITATION_COL].sum().reset_index()

    # Fechas
    try:
        start_date = pd.to_datetime(f"{year_range[0]}-01-01")
        end_date = pd.to_datetime(f"{year_range[1]}-12-31")
    except:
        start_date, end_date = None, None

    mask_base = (
        (df_long[Config.YEAR_COL] >= year_range[0]) & 
        (df_long[Config.YEAR_COL] <= year_range[1]) &
        (df_long[Config.STATION_NAME_COL].isin(stations_for_analysis))
    )
    df_complete_filtered = df_long.loc[mask_base].copy()

    # Argumentos
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

    # --- D. RENDERIZADO (AQUÍ ESTÁ LA CLAVE) ---
    
    # 1. CAJA DE RESUMEN (LLAMADA DIRECTA Y SEGURA)
    try:
        display_current_filters(
            stations_sel=stations_for_analysis, 
            regions_sel=sel_regions, 
            munis_sel=sel_munis, 
            year_range=year_range,
            interpolacion="Si" if st.session_state.get('apply_interpolation') else "No",
            df_data=df_monthly_filtered
        )
    except Exception as e:
        st.warning(f"No se pudo mostrar el resumen de filtros: {e}")

    # 2. PESTAÑAS
    tab_titles = [
        "🏠 Inicio", "🚨 Monitoreo", "🗺️ Distribución", "📈 Gráficos", 
        "📊 Estadísticas", "🔮 Pronóstico Climático", "📉 Tendencias", "⚠️ Anomalías", 
        "🔗 Correlación", "🌊 Extremos", "🌍 Mapas Avanzados", "🧪 Sesgo",
        "🌿 Cobertura", "🌱 Zonas Vida", "🌡️ Clima Futuro", "📄 Reporte"
    ]
    
    tabs = st.tabs(tab_titles)

    # 3. CONTENIDO DE PESTAÑAS (PROTEGIDO)
    with tabs[0]: display_welcome_tab()
    with tabs[1]: display_realtime_dashboard(df_complete_filtered, gdf_stations, gdf_filtered)
    
    with tabs[2]: 
        display_spatial_distribution_tab(
            user_loc=None, 
            interpolacion="Si" if st.session_state.get('apply_interpolation') else "No", 
            **display_args
        )
    
    with tabs[3]: display_graphs_tab(**display_args)
    with tabs[4]: 
        display_stats_tab(**display_args)
        st.markdown("---")
        display_station_table_tab(**display_args)
    
    with tabs[5]: display_climate_forecast_tab(**display_args)
    with tabs[6]: display_trends_and_forecast_tab(**display_args)
    with tabs[7]: display_anomalies_tab(**display_args)
    with tabs[8]: display_correlation_tab(**display_args)
    with tabs[9]: display_drought_analysis_tab(**display_args)
    with tabs[10]: display_advanced_maps_tab(**display_args)
    
    with tabs[11]: 
        try:
            from modules.visualizer import display_bias_correction_tab
            display_bias_correction_tab(**display_args)
        except: st.info("Módulo Sesgo cargando...")
        
    with tabs[12]: display_land_cover_analysis_tab(**display_args)
    with tabs[13]: display_life_zones_tab(**display_args)
    with tabs[14]: display_climate_scenarios_tab(**display_args)
    
    with tabs[15]: 
        st.header("Reporte PDF")
        if st.button("Generar"):
            res = {"n_estaciones": len(stations_for_analysis), "rango": f"{year_range}"}
            pdf = generate_pdf_report(df_monthly_filtered, gdf_filtered, res)
            if pdf: st.download_button("Descargar", pdf, "reporte.pdf", "application/pdf")

    # CSS
    st.markdown("""<style>.stTabs [data-baseweb="tab-panel"] { padding-top: 1rem; }</style>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
