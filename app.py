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

# Ignorar advertencias de librerías
warnings.filterwarnings('ignore')

# --- Importaciones de Módulos Propios ---
from modules.config import Config
from modules.data_processor import load_and_process_all_data, complete_series
from modules.sidebar import create_sidebar
from modules.reporter import generate_pdf_report

# Importamos TODAS las funciones de visualización
from modules.visualizer import (
    display_welcome_tab,
    display_realtime_dashboard,  # <--- NUEVA FUNCIÓN UNIFICADA
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
    display_station_table_tab
)

# --- FUNCIÓN PRINCIPAL ---
def main():
    # Estilos CSS Personalizados
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
        /* Ajuste para pestañas */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 4px 4px 0 0;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ffffff;
            border-bottom: 2px solid #1f77b4;
        }
    </style>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 1. CARGAR DATOS (Backend)
    # ---------------------------------------------------------
    # Ahora recibimos 6 valores (incluyendo predios)
    gdf_stations, gdf_municipios, df_long, df_enso, gdf_subcuencas, gdf_predios = load_and_process_all_data()

    if gdf_stations is None or df_long is None:
        st.error("⚠️ Error Fatal: No se pudieron cargar los datos. Verifica la conexión a la Base de Datos.")
        st.stop()

    # ---------------------------------------------------------
    # 2. BARRA LATERAL Y FILTROS
    # ---------------------------------------------------------
    (stations_for_analysis, df_anual_melted, df_monthly_filtered,
     gdf_filtered, analysis_mode, selected_regions, selected_municipios,
     selected_altitudes, year_range) = create_sidebar(gdf_stations, df_long)

    # Lógica de interpolación (si se activó el checkbox en sidebar)
    if st.session_state.get('apply_interpolation', False):
        with st.spinner("🔄 Completando series de tiempo (Interpolación)..."):
            df_monthly_filtered = complete_series(df_monthly_filtered)
            # Recalcular anual con datos completados para consistencia
            df_anual_melted = df_monthly_filtered.groupby(
                [Config.STATION_NAME_COL, Config.YEAR_COL]
            )[Config.PRECIPITATION_COL].sum().reset_index()
            st.toast("✅ Series completadas correctamente")

    # Calcular fechas exactas para procesos internos
    try:
        start_date = pd.to_datetime(f"{year_range[0]}-01-01")
        end_date = pd.to_datetime(f"{year_range[1]}-12-31")
    except:
        start_date, end_date = None, None

    # ---------------------------------------------------------
    # 3. DICCIONARIO DE ARGUMENTOS (Contexto)
    # ---------------------------------------------------------
    # Empaquetamos todo para pasarlo limpio a las funciones
    display_args = {
        "df_long": df_long,             # Histórico completo (para referencia)
        "gdf_stations": gdf_stations,   # Metadatos de estaciones
        "gdf_municipios": gdf_municipios,
        "gdf_subcuencas": gdf_subcuencas,
        "gdf_predios": gdf_predios,     # Nueva capa de predios
        "df_enso": df_enso,             # Índices climáticos
        
        # Datos Filtrados (Selección del usuario)
        "stations_for_analysis": stations_for_analysis,
        "gdf_filtered": gdf_filtered,
        "df_anual_melted": df_anual_melted,         # Para gráficos anuales
        "df_monthly_filtered": df_monthly_filtered, # Para gráficos mensuales
        
        # Parámetros de Filtro
        "analysis_mode": analysis_mode,
        "selected_regions": selected_regions,
        "selected_municipios": selected_municipios,
        "selected_altitudes": selected_altitudes,
        "year_range": year_range,
        "start_date": start_date,
        "end_date": end_date
    }

    # ---------------------------------------------------------
    # 4. DEFINICIÓN DE PESTAÑAS
    # ---------------------------------------------------------
    tab_titles = [
        "🏠 Bienvenida", 
        "🚨 Monitoreo y Tiempo Real",
        "🗺️ Distribución", 
        "📈 Gráficos", 
        "📊 Estadísticas",
        "📉 Tendencias",
        "⚠️ Anomalías",
        "🔗 Correlación",
        "🌊 ENSO",
        "🌊 Extremos Hidrológicos",
        "🌍 Mapas Avanzados",
        "🌿 Cobertura",
        "🌱 Zonas de Vida",
        "🌡️ Cambio Climático",
        "📄 Reporte"
    ]
    
    tabs = st.tabs(tab_titles)

    # ---------------------------------------------------------
    # 5. RENDERIZADO DE MÓDULOS
    # ---------------------------------------------------------
    
    with tabs[0]: # Bienvenida
        display_welcome_tab()
    
    with tabs[1]: # Monitoreo y Tiempo Real (Fusión de Alertas + Pronóstico + Satélite)
        from modules.visualizer import display_realtime_dashboard
        display_realtime_dashboard(df_long, gdf_stations, gdf_filtered)

    with tabs[2]: # Mapa Distribución
        display_spatial_distribution_tab(**display_args)

    with tabs[3]: # Gráficos
        display_graphs_tab(**display_args)

    with tabs[4]: # Estadísticas
        display_stats_tab(**display_args)
        st.markdown("---")
        display_station_table_tab(**display_args)

    with tabs[5]: # Tendencias
        display_trends_and_forecast_tab(**args)
    
    with tabs[6]: # Anomalías
        display_anomalies_tab(**display_args)

    with tabs[7]: # Correlación
        display_correlation_tab(**display_args)

    with tabs[8]: # ENSO
        display_enso_tab(**display_args)

    with tabs[9]: # Extremos Hidrológicos
        display_drought_analysis_tab(**display_args)

    with tabs[10]: # Mapas Avanzados
        display_advanced_maps_tab(**display_args)

    with tabs[11]: # Cobertura
        display_land_cover_analysis_tab(**display_args)

    with tabs[12]: # Zonas de Vida
        display_life_zones_tab(**display_args)

    with tabs[13]: # Escenarios
        display_climate_scenarios_tab(**display_args)

    with tabs[14]: # Reporte PDF
        st.header("Generar Reporte PDF")
        col_pdf_1, col_pdf_2 = st.columns([1, 3])
        with col_pdf_1:
            if st.button("Generar Reporte Ejecutivo", type="primary"):
                with st.spinner("Generando PDF..."):
                    analysis_results = {
                        "n_estaciones": len(stations_for_analysis),
                        "rango_fechas": f"{year_range[0]} - {year_range[1]}",
                        "modo_analisis": analysis_mode
                    }
                    pdf_bytes = generate_pdf_report(
                        df_long=df_monthly_filtered, 
                        gdf_stations=gdf_filtered, 
                        analysis_results=analysis_results
                    )
                    if pdf_bytes:
                        st.success("¡Reporte generado!")
                        st.download_button(
                            label="📥 Descargar PDF",
                            data=pdf_bytes,
                            file_name=f"Reporte_SIHCLI_{year_range[0]}-{year_range[1]}.pdf",
                            mime="application/pdf"
                        )
                    else:
                        st.error("No se pudo generar el reporte.")

if __name__ == "__main__":
    main()

# --- Estilos CSS para separar pestañas ---
st.markdown("""
<style>
    /* Separación entre botones de pestañas */
    div[data-baseweb="tab-list"] {
        gap: 8px;
    }
    /* Estilo opcional para que parezcan botones individuales */
    div[data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        padding-left: 16px;
        padding-right: 16px;
        border: 1px solid #e0e0e0;
        border-bottom: none;
    }
    /* Pestaña activa */
    div[aria-selected="true"] {
        background-color: white;
        border-top: 2px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)




