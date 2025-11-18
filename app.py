# app.py

import streamlit as st
import pandas as pd
import warnings

# --- Configuración de la Página (Debe ser la primera instrucción de Streamlit) ---
# Se define aquí dentro de main para asegurar el contexto, pero es lo primero que se ejecuta.
# (Nota: En algunos setups se prefiere fuera, pero dentro de main es seguro si es lo primero).

# Ignorar advertencias no críticas de librerías
warnings.filterwarnings('ignore')

# --- Importaciones de Módulos Propios ---
from modules.config import Config
from modules.data_processor import load_and_process_all_data, complete_series
from modules.visualizer import (
    display_welcome_tab,
    display_alerts_tab,
    display_spatial_distribution_tab,
    display_graphs_tab,
    display_advanced_maps_tab,
    display_climate_forecast_tab,
    display_life_zones_tab,
    display_drought_risk_tab,
)
from modules.sidebar import create_sidebar
from modules.reporter import generate_pdf_report
from modules.analysis import calculate_monthly_anomalies, calculate_basin_stats

# --- FUNCIÓN PRINCIPAL ---
def main():
    # 1. Configuración de la página
    st.set_page_config(
        page_title=Config.APP_TITLE,
        page_icon="🌦️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # CSS Personalizado para ajustar el estilo
    st.markdown("""
        <style>
        .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        h1 { color: #1f77b4; }
        h2 { color: #2ca02c; }
        div.stButton > button:first-child {
            background-color: #1f77b4;
            color: white;
            border-radius: 5px;
        }
        </style>
        """, unsafe_allow_html=True)

    # 2. Cargar Datos (Centralizado y Cacheado desde Supabase)
    # Esta función maneja la conexión y el procesamiento inicial
    gdf_stations, gdf_municipios, df_long, df_enso, gdf_subcuencas = load_and_process_all_data()

    # Verificación de Carga Exitosa
    if gdf_stations is None or df_long is None:
        st.error("⚠️ Error Fatal: No se pudieron cargar los datos de la base de datos. Verifique la conexión y los secretos en Streamlit Cloud.")
        st.stop()

    # 3. Crear Sidebar y Obtener Filtros
    # La función create_sidebar maneja la lógica de la barra lateral y devuelve las selecciones
    (stations_for_analysis, df_anual_melted, df_monthly_filtered, 
     gdf_filtered, analysis_mode, selected_regions, selected_municipios, 
     selected_altitudes, year_range) = create_sidebar(gdf_stations, df_long)

    # --- NUEVO: APLICAR COMPLETADO DE SERIES SI SE SOLICITÓ ---
    if st.session_state.get('apply_interpolation', False):
        with st.spinner("Completando series de tiempo..."):
            # Aplicamos complete_series al dataframe filtrado
            df_monthly_filtered = complete_series(df_monthly_filtered)
            # Recalculamos el anual basado en los datos completados
            df_anual_melted = df_monthly_filtered.groupby(
                [Config.STATION_NAME_COL, Config.YEAR_COL]
            )[Config.PRECIPITATION_COL].sum().reset_index()
            st.toast("✅ Series completadas con interpolación")
            
    # --- CORRECCIÓN CLAVE: CÁLCULO DE FECHAS ---
    # Calculamos las fechas de inicio y fin AQUI, justo después de obtener el rango del sidebar
    # y ANTES de crear el diccionario display_args. Esto soluciona el NameError.
    start_date = pd.to_datetime(f"{year_range[0]}-01-01")
    end_date = pd.to_datetime(f"{year_range[1]}-12-31")
    # -------------------------------------------

    # 4. Definir Pestañas de la Aplicación
    tabs = st.tabs([
        "🏠 Bienvenida", 
        "🚨 Alertas y Resumen", 
        "🗺️ Distribución Espacial", 
        "📈 Gráficos", 
        "🌐 Mapas Avanzados",
        "🔮 Pronósticos",
        "🌿 Zonas de Vida",
        "⚠️ Riesgo (SPI)",
        "📄 Reporte PDF"
    ])

    # 5. Preparar Argumentos Comunes
    # Creamos el diccionario de argumentos que se pasará a las funciones de visualización.
    # Ahora 'start_date' y 'end_date' ya están definidos y no darán error.
    display_args = {
        "df_long": df_long,
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
        "start_date": start_date,  # Variable definida en el paso 3
        "end_date": end_date,      # Variable definida en el paso 3
        "year_range": year_range
    }

    # 6. Renderizar Contenido de las Pestañas
    
    with tabs[0]:
        display_welcome_tab()

    with tabs[1]:
        # Pasamos los argumentos desempaquetados
        display_alerts_tab(**display_args)

    with tabs[2]:
        display_spatial_distribution_tab(**display_args)

    with tabs[3]:
        display_graphs_tab(**display_args)

    with tabs[4]:
        display_advanced_maps_tab(**display_args)

    with tabs[5]:
        display_climate_forecast_tab(**display_args)

    with tabs[6]:
        display_life_zones_tab(**display_args)
        
    with tabs[7]:
        display_drought_risk_tab(**display_args)
        
    with tabs[8]:
        st.header("📄 Generar Reporte PDF")
        st.info("Genere un reporte ejecutivo descargable con los datos filtrados y las visualizaciones clave.")
        
        col_pdf_1, col_pdf_2 = st.columns([1, 3])
        with col_pdf_1:
            generate_btn = st.button("Generar Reporte de Análisis", type="primary")
            
        if generate_btn:
            with st.spinner("Generando documento PDF... esto puede tardar unos segundos..."):
                # Preparar un resumen simple de resultados para el reporte
                analysis_results = {
                    "n_estaciones": len(stations_for_analysis),
                    "rango_fechas": f"{start_date.date()} a {end_date.date()}",
                    "modo_analisis": analysis_mode
                }
                
                # Llamar al módulo reporter para crear el PDF en memoria
                pdf_bytes = generate_pdf_report(
                    df_long=df_monthly_filtered, # Usamos los datos ya filtrados por el usuario
                    gdf_stations=gdf_filtered,   # Usamos las estaciones filtradas
                    analysis_results=analysis_results
                )
                
                if pdf_bytes:
                    st.success("¡Reporte generado exitosamente!")
                    st.download_button(
                        label="⬇️ Descargar Reporte PDF",
                        data=pdf_bytes,
                        file_name=f"Reporte_Hidroclimatico_{year_range[0]}-{year_range[1]}.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.error("Hubo un problema al generar el reporte. Por favor revise los logs.")

# --- PUNTO DE ENTRADA ---
if __name__ == "__main__":
    main()




