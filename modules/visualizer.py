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
# 1. FUNCIONES DE GRÁFICOS (Requeridas por reporter.py y otros)
# -----------------------------------------------------------------------------

def create_enso_chart(enso_data):
    """Crea el gráfico de Anomalía ONI (ENSO)."""
    if enso_data is None or enso_data.empty or Config.ENSO_ONI_COL not in enso_data.columns:
        return go.Figure().update_layout(title="Datos ENSO no disponibles")

    data = enso_data.copy().sort_values(Config.DATE_COL)
    data.dropna(subset=[Config.ENSO_ONI_COL], inplace=True)

    conditions = [
        data[Config.ENSO_ONI_COL] >= 0.5,
        data[Config.ENSO_ONI_COL] <= -0.5
    ]
    phases = ['El Niño', 'La Niña']
    colors = ['red', 'blue']
    data['color'] = np.select(conditions, colors, default='grey')

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data[Config.DATE_COL], 
        y=[data[Config.ENSO_ONI_COL].max() - data[Config.ENSO_ONI_COL].min()] * len(data),
        base=data[Config.ENSO_ONI_COL].min(),
        marker_color=data['color'], 
        opacity=0.3,
        hoverinfo='none',
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=data[Config.DATE_COL], 
        y=data[Config.ENSO_ONI_COL],
        mode='lines+markers', 
        name='Anomalía ONI',
        line=dict(color='black', width=2)
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="red")
    fig.add_hline(y=-0.5, line_dash="dash", line_color="blue")
    
    fig.update_layout(title="Evolución del Fenómeno ENSO (Índice ONI)", height=400)
    return fig

def create_anomaly_chart(df_plot):
    """Crea gráfico de anomalías de precipitación."""
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
    fig.update_layout(title="Anomalías Mensuales", yaxis_title="Anomalía (mm)")
    return fig

# -----------------------------------------------------------------------------
# 2. FUNCIONES DE PESTAÑAS (VISUALIZACIÓN)
# -----------------------------------------------------------------------------

def display_welcome_tab():
    st.header(f"Bienvenido a {Config.APP_TITLE}")
    # CSS Limpio y corregido
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css?family=Playfair+Display:wght@700&display=swap');
    .quote { font-family: 'Playfair Display', serif; font-weight: 700; font-size: 22px; text-align: center; padding: 20px; }
    .author { font-family: 'Playfair Display', serif; text-align: right; font-style: italic; font-size: 18px; padding-right: 20px; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown(f'<p class="quote">{Config.QUOTE_TEXT}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="author">- {Config.QUOTE_AUTHOR}</p>', unsafe_allow_html=True)
    st.markdown(Config.WELCOME_TEXT)

def display_alerts_tab(df_long, **kwargs):
    st.subheader("Tablero de Alertas")
    if df_long is not None and not df_long.empty:
        st.metric("Registros Cargados", len(df_long))
    else:
        st.warning("Sin datos para alertas.")

def display_spatial_distribution_tab(gdf_filtered, **kwargs):
    st.subheader("Distribución Espacial")
    if gdf_filtered is not None and not gdf_filtered.empty:
        st.map(gdf_filtered)
    else:
        st.warning("No hay estaciones seleccionadas.")

def display_graphs_tab(df_monthly_filtered, **kwargs):
    st.subheader("Análisis Gráfico")
    if df_monthly_filtered is not None and not df_monthly_filtered.empty:
        st.line_chart(df_monthly_filtered, x=Config.DATE_COL, y=Config.PRECIPITATION_COL)
    else:
        st.info("Sin datos para graficar.")

# --- Funciones Placeholder para evitar errores de importación ---

def display_advanced_maps_tab(**kwargs):
    st.subheader("Mapas Avanzados")
    st.info("Funcionalidad de interpolación.")

def display_climate_forecast_tab(**kwargs):
    st.subheader("Pronóstico Climático")
    st.info("Pronósticos ONI/SOI.")

def display_trends_and_forecast_tab(**kwargs):
    st.subheader("Tendencias")
    st.info("Análisis de tendencias y pronósticos de lluvia.")

def display_anomalies_tab(**kwargs):
    st.subheader("Anomalías")
    st.info("Análisis de anomalías.")

def display_stats_tab(**kwargs):
    st.subheader("Estadísticas")
    st.info("Resumen estadístico.")

def display_correlation_tab(**kwargs):
    st.subheader("Correlación")
    st.info("Matriz de correlación.")

def display_enso_tab(**kwargs):
    st.subheader("ENSO")
    st.info("Análisis del fenómeno del Niño.")

def display_life_zones_tab(**kwargs):
    st.subheader("Zonas de Vida")
    st.info("Clasificación Holdridge.")

def display_drought_analysis_tab(df_long, gdf_stations, **kwargs):
    st.subheader("Análisis de Sequía")
    st.info("Cálculo de índices SPI.")

def display_climate_scenarios_tab(**kwargs):
    st.subheader("Escenarios")
    st.info("Simulación de cambio climático.")

def display_weekly_forecast_tab(stations, gdf):
    st.subheader("Pronóstico Semanal")
    st.info("Datos de OpenMeteo.")

def display_satellite_imagery_tab(gdf):
    st.subheader("Imágenes Satelitales")
    st.info("Visor WMS.")

def display_station_table_tab(**kwargs):
    st.subheader("Tabla de Datos")
    st.info("Vista tabular.")

def display_land_cover_analysis_tab(**kwargs):
    st.subheader("Cobertura")
    st.info("Análisis de cobertura del suelo.")
