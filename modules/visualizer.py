import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import folium
import os
from streamlit_folium import st_folium
from modules.config import Config

# -----------------------------------------------------------------------------
# 1. FUNCIONES DE GRÁFICOS (Requeridas por reporter.py)
# -----------------------------------------------------------------------------

def create_enso_chart(enso_data):
    [cite_start]"""Crea el gráfico de Anomalía ONI (ENSO)[cite: 10]."""
    if enso_data is None or enso_data.empty or Config.ENSO_ONI_COL not in enso_data.columns:
        return go.Figure().update_layout(title="Datos ENSO no disponibles")

    data = enso_data.copy().sort_values(Config.DATE_COL)
    data.dropna(subset=[Config.ENSO_ONI_COL], inplace=True)

    # Definir colores según fase
    conditions = [
        data[Config.ENSO_ONI_COL] >= 0.5,
        data[Config.ENSO_ONI_COL] <= -0.5
    ]
    phases = ['El Niño', 'La Niña']
    colors = ['red', 'blue']
    data['color'] = np.select(conditions, colors, default='grey')
    data['phase'] = np.select(conditions, phases, default='Neutral')

    fig = go.Figure()
    
    # Barras de fondo
    fig.add_trace(go.Bar(
        x=data[Config.DATE_COL], 
        y=[data[Config.ENSO_ONI_COL].max() - data[Config.ENSO_ONI_COL].min()] * len(data),
        base=data[Config.ENSO_ONI_COL].min(),
        marker_color=data['color'], 
        opacity=0.3,
        hoverinfo='none',
        showlegend=False
    ))

    # Línea principal
    fig.add_trace(go.Scatter(
        x=data[Config.DATE_COL], 
        y=data[Config.ENSO_ONI_COL],
        mode='lines+markers', 
        name='Anomalía ONI',
        line=dict(color='black', width=2)
    ))

    # Líneas de umbral
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Umbral El Niño")
    fig.add_hline(y=-0.5, line_dash="dash", line_color="blue", annotation_text="Umbral La Niña")

    fig.update_layout(
        title="Evolución del Fenómeno ENSO (Índice ONI)",
        yaxis_title="Anomalía de Temperatura (°C)",
        height=400
    )
    return fig

def create_anomaly_chart(df_plot):
    [cite_start]"""Crea gráfico de anomalías de precipitación[cite: 11]."""
    if df_plot.empty:
        return go.Figure()
    
    df_plot['color'] = np.where(df_plot['anomalia'] < 0, 'red', 'blue')
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_plot[Config.DATE_COL],
        y=df_plot['anomalia'],
        marker_color=df_plot['color'],
        name='Anomalía Precipitación'
    ))
    fig.update_layout(title="Anomalías Mensuales de Precipitación", yaxis_title="Anomalía (mm)")
    return fig

# -----------------------------------------------------------------------------
# 2. FUNCIONES DE VISUALIZACIÓN (PESTAÑAS)
# -----------------------------------------------------------------------------

def display_welcome_tab():
    [cite_start]"""Pestaña de bienvenida con estilos CSS corregidos[cite: 15]."""
    st.header(f"Bienvenido a {Config.APP_TITLE}")
    
    # CORRECCIÓN DEL ERROR DE SINTAXIS (CSS dentro de triple comilla)
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css?family=Playfair+Display:wght@700&display=swap');
    .quote { font-family: 'Playfair Display', serif; font-weight: 700; font-size: 22px; text-align: center; padding: 20px; }
    .author { font-family: 'Playfair Display', serif; text-align: right; font-style: italic; font-size: 18px; padding-right: 20px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f'<p class="quote">{Config.QUOTE_TEXT}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="author">- {Config.QUOTE_AUTHOR}</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.markdown(Config.WELCOME_TEXT, unsafe_allow_html=True)
        with st.expander("La Inspiración: Chaac, Divinidad Maya", expanded=False):
            st.markdown(Config.CHAAC_STORY)
    with col2:
        if os.path.exists(Config.CHAAC_IMAGE_PATH):
            st.image(Config.CHAAC_IMAGE_PATH, caption="Representación de Chaac")
        elif os.path.exists(Config.LOGO_PATH):
            st.image(Config.LOGO_PATH, width=250)

def display_alerts_tab(df_long, **kwargs):
    st.subheader("Tablero de Alertas")
    if df_long is not None and not df_long.empty:
        # Ejemplo simple de alertas
        max_precip = df_long[Config.PRECIPITATION_COL].max()
        st.metric("Precipitación Máxima Histórica", f"{max_precip:.1f} mm")
    else:
        st.warning("No hay datos para generar alertas.")

def display_spatial_distribution_tab(gdf_filtered, **kwargs):
    st.subheader("Distribución Espacial")
    if gdf_filtered is not None and not gdf_filtered.empty:
        # Mapa simple usando st.map para evitar errores si Folium falla
        st.map(gdf_filtered)
        # NOTA: Aquí iría la lógica compleja de Folium del PDF visualizer-py.pdf
        # Te recomiendo pegar esa lógica aquí una vez se estabilice la app.
    else:
        st.warning("No hay estaciones seleccionadas.")

def display_graphs_tab(df_monthly_filtered, **kwargs):
    st.subheader("Análisis Gráfico")
    if not df_monthly_filtered.empty:
        fig = px.line(df_monthly_filtered, x=Config.DATE_COL, y=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay datos para graficar.")

# --- Funciones "Placeholder" para que app.py no falle al importar ---
# (Reemplaza el contenido de estas funciones con el código completo del PDF cuando puedas)

def display_advanced_maps_tab(**kwargs):
    st.subheader("Mapas Avanzados (Interpolación)")
    st.info("Módulo de interpolación espacial.")

def display_climate_forecast_tab(**kwargs):
    st.subheader("Pronóstico Climático (ONI/SOI)")
    st.info("Pronóstico de índices climáticos.")

def display_trends_and_forecast_tab(**kwargs):
    st.subheader("Tendencias y Pronósticos de Lluvia")
    st.info("Análisis SARIMA y Prophet.")

def display_anomalies_tab(**kwargs):
    st.subheader("Análisis de Anomalías")
    st.info("Cálculo de anomalías estandarizadas.")

def display_stats_tab(**kwargs):
    st.subheader("Estadísticas Descriptivas")
    st.info("Tablas de resumen estadístico.")

def display_correlation_tab(**kwargs):
    st.subheader("Matriz de Correlación")
    st.info("Correlación entre estaciones y variables.")

def display_enso_tab(**kwargs):
    st.subheader("Impacto ENSO")
    st.info("Análisis detallado del fenómeno del Niño.")

def display_life_zones_tab(**kwargs):
    st.subheader("Zonas de Vida de Holdridge")
    st.info("Clasificación bioclimática.")

def display_drought_analysis_tab(df_long, gdf_stations, **kwargs):
    st.subheader("Análisis de Sequía (SPI)")
    st.info("Cálculo del Índice Estandarizado de Precipitación.")

def display_climate_scenarios_tab(**kwargs):
    st.subheader("Escenarios de Cambio Climático")
    st.info("Simulación de escenarios futuros.")

def display_weekly_forecast_tab(stations, gdf):
    st.subheader("Pronóstico Semanal")
    st.info("Pronóstico a 7 días (OpenMeteo).")

def display_satellite_imagery_tab(gdf):
    st.subheader("Imágenes Satelitales")
    st.info("Visor WMS.")

def display_station_table_tab(**kwargs):
    st.subheader("Tabla de Datos")
    st.info("Datos tabulares.")

def display_land_cover_analysis_tab(**kwargs):
    st.subheader("Cobertura del Suelo")
    st.info("Análisis de coberturas.")
