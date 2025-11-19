import streamlit as st
import pandas as pd
import warnings

# --- Configuración de la Página (Debe ser la primera instrucción) ---
st.set_page_config(
    page_title="SIHCLI-POTER",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ignorar advertencias
warnings.filterwarnings('ignore')

# --- Importaciones de Módulos Propios ---
from modules.config import Config
from modules.data_processor import load_and_process_all_data, complete_series
from modules.sidebar import create_sidebar
from modules.reporter import generate_pdf_report

# Importamos TODAS las funciones disponibles en el nuevo visualizer.py
from modules.visualizer import (
    display_welcome_tab,
    display_alerts_tab,
    display_spatial_distribution_tab,
    display_graphs_tab,
    display_advanced_maps_tab,
    display_climate_forecast_tab,     # Pronóstico de índices (ONI/SOI)
    display_trends_and_forecast_tab,  # SARIMA/Prophet para precipitación
    display_anomalies_tab,            # Nuevo: Análisis de Anomalías
    display_stats_tab,                # Nuevo: Estadísticas detalladas
    display_correlation_tab,          # Nuevo: Correlaciones
    display_enso_tab,                 # Nuevo: Análisis ENSO detallado
    display_life_zones_tab,
    display_drought_analysis_tab,     # Nuevo nombre (antes risk_tab)
    display_climate_scenarios_tab,    # Nuevo: Escenarios Cambio Climático
    display_station_table_tab,        # Tabla detallada
    display_weekly_forecast_tab,      # Pronóstico 7 días OpenMeteo
    display_satellite_imagery_tab,    # WMS Satelital
    display_land_cover_analysis_tab   # Coberturas
)

# --- FUNCIÓN PRINCIPAL ---
def main():
    # CSS Personalizado
    st.markdown("""
    <style>
        .main .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        h1 { color: #1f77b4; }
        h2 { color: #2ca02c; }
        div.stButton > button:first-child {
            background-color: #1f77b4;
            color: white;
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

    # 1. Cargar Datos
    gdf_stations, gdf_municipios, df_long, df_enso, gdf_subcuencas = load_and_process_all_data()

    if gdf_stations is None or df_long is None:
        st.error("⚠️ Error Fatal: No se pudieron cargar los datos. Verifica la conexión a BD.")
        st.stop()

    # 2. Sidebar y Filtros
    (stations_for_analysis, df_anual_melted, df_monthly_filtered,
     gdf_filtered, analysis_mode, selected_regions, selected_municipios,
     selected_altitudes, year_range) = create_sidebar(gdf_stations, df_long)

    # Lógica de interpolación (si se activó en sidebar)
    if st.session_state.get('apply_interpolation', False):
        with st.spinner("Completando series de tiempo..."):
            df_monthly_filtered = complete_series(df_monthly_filtered)
            # Recalcular anual con datos completados
            df_anual_melted = df_monthly_filtered.groupby(
                [Config.STATION_NAME_COL, Config.YEAR_COL]
            )[Config.PRECIPITATION_COL].sum().reset_index()
            st.toast("✅ Series completadas con interpolación")

    # Calcular fechas para filtros internos
    start_date = pd.to_datetime(f"{year_range[0]}-01-01")
    end_date = pd.to_datetime(f"{year_range[1]}-12-31")

    # 3. Diccionario de Argumentos (Para pasar a las funciones limpiamente)
    display_args = {
        "df_long": df_long, # DataFrame original completo
        "df_full_monthly": df_long, # Alias para algunas funciones nuevas
        "gdf_stations": gdf_stations,
        "gdf_municipios": gdf_municipios,
        "gdf_subcuencas": gdf_subcuencas,
        "df_enso": df_enso,
        "stations_for_analysis": stations_for_analysis,
        "gdf_filtered": gdf_filtered,
        "analysis_mode": analysis_mode,
        "selected_regions": selected_regions,
        "selected_municipios": selected_municipios,
        "selected_altitudes": selected_altitudes,
        "df_anual_melted": df_anual_melted,
        "df_monthly_filtered": df_monthly_filtered,
        "start_date": start_date,
        "end_date": end_date,
        "year_range": year_range
    }

    # 4. Definición de Pestañas (Estructura Completa)
    tab_titles = [
        "🏠 Bienvenida", 
        "🚨 Alertas", 
        "🗺️ Distribución", 
        "📈 Gráficos", 
        "📊 Estadísticas",
        "📉 Tendencias y Pronósticos",
        "⚠️ Anomalías",
        "🔗 Correlación",
        "🌊 ENSO",
        "🏜️ Sequía (SPI)",
        "🌍 Mapas Avanzados",
        "🌱 Zonas de Vida",
        "🌡️ Cambio Climático",
        "🛰️ Satélite/Clima",
        "📄 Reporte"
    ]
    
    tabs = st.tabs(tab_titles)

    # 5. Renderizado de Pestañas
    with tabs[0]:
        display_welcome_tab()
    
    with tabs[1]:
        display_alerts_tab(**display_args)

    with tabs[2]:
        display_spatial_distribution_tab(**display_args)

    with tabs[3]:
        display_graphs_tab(**display_args)

    with tabs[4]: # Estadísticas
        display_stats_tab(**display_args)
        st.markdown("---")
        display_station_table_tab(**display_args)

    with tabs[5]: # Tendencias y Pronósticos (SARIMA/Prophet)
        display_trends_and_forecast_tab(**display_args)
        st.markdown("---")
        display_climate_forecast_tab(**display_args) # Pronóstico de índices

    with tabs[6]: # Anomalías
        display_anomalies_tab(**display_args)

    with tabs[7]: # Correlación
        display_correlation_tab(**display_args)

    with tabs[8]: # ENSO
        display_enso_tab(**display_args)

    with tabs[9]: # Sequía / Análisis de Riesgo
        # Esta función es la que daba error. Ahora debe existir en visualizer.py
        display_drought_analysis_tab(**display_args)

    with tabs[10]: # Mapas Avanzados (Interpolación, Morfometría)
        display_advanced_maps_tab(**display_args)

    with tabs[11]: # Zonas de Vida
        display_life_zones_tab(**display_args)
        # Opcional: Agregar cobertura si tienes el raster
        # display_land_cover_analysis_tab(**display_args)

    with tabs[12]: # Escenarios Cambio Climático
        display_climate_scenarios_tab(**display_args)

    with tabs[13]: # Satélite y Pronóstico Semanal
        st.subheader("Herramientas de Tiempo Real")
        subtab1, subtab2 = st.tabs(["Pronóstico 7 Días (OpenMeteo)", "Imágenes Satelitales"])
        with subtab1:
            display_weekly_forecast_tab(stations_for_analysis, gdf_filtered)
        with subtab2:
            display_satellite_imagery_tab(gdf_filtered)

    with tabs[14]: # Reporte PDF
        st.header("Generar Reporte PDF")
        col_pdf_1, col_pdf_2 = st.columns([1, 3])
        with col_pdf_1:
            if st.button("Generar Reporte Ejecutivo", type="primary"):
                with st.spinner("Generando PDF..."):
                    analysis_results = {
                        "n_estaciones": len(stations_for_analysis),
                        "rango_fechas": f"{start_date.date()} a {end_date.date()}",
                        "modo_analisis": analysis_mode
                    }
                    pdf_bytes = generate_pdf_report(
                        df_long=df_monthly_filtered, 
                        gdf_stations=gdf_filtered, 
                        analysis_results=analysis_results
                    )
                    if pdf_bytes:
                        st.success("¡Reporte listo!")
                        st.download_button(
                            label="📥 Descargar PDF",
                            data=pdf_bytes,
                            file_name=f"Reporte_SIHCLI_{year_range[0]}-{year_range[1]}.pdf",
                            mime="application/pdf"
                        )
                    else:
                        st.error("Error generando el reporte.")

if __name__ == "__main__":
    main()
