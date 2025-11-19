import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import folium
import requests
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from modules.config import Config

# -----------------------------------------------------------------------------
# 1. FUNCIONES AUXILIARES
# -----------------------------------------------------------------------------

def get_weather_forecast_simple(lat, lon):
    """Obtiene pronóstico simple de Open-Meteo para evitar errores de importación."""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
            "timezone": "auto"
        }
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
        daily = data.get('daily', {})
        if not daily: return pd.DataFrame()

        df = pd.DataFrame({
            'Fecha': daily.get('time', []),
            'Temp. Máx (°C)': daily.get('temperature_2m_max', []),
            'Temp. Mín (°C)': daily.get('temperature_2m_min', []),
            'Lluvia (mm)': daily.get('precipitation_sum', [])
        })
        return df
    except Exception:
        return pd.DataFrame()

def create_enso_chart(enso_data):
    """Crea el gráfico de Anomalía ONI (ENSO)."""
    if enso_data is None or enso_data.empty or Config.ENSO_ONI_COL not in enso_data.columns:
        return go.Figure().update_layout(title="Datos ENSO no disponibles", height=300)

    data = enso_data.copy().sort_values(Config.DATE_COL).dropna(subset=[Config.ENSO_ONI_COL])
    
    conditions = [data[Config.ENSO_ONI_COL] >= 0.5, data[Config.ENSO_ONI_COL] <= -0.5]
    colors = ['red', 'blue']
    data['color'] = np.select(conditions, colors, default='gray')

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data[Config.DATE_COL], y=data[Config.ENSO_ONI_COL],
        marker_color=data['color'], name="Anomalía ONI"
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="red")
    fig.add_hline(y=-0.5, line_dash="dash", line_color="blue")
    fig.update_layout(title="Índice Oceánico El Niño (ONI)", height=400)
    return fig

# -----------------------------------------------------------------------------
# 2. FUNCIONES PRINCIPALES DE VISUALIZACIÓN
# -----------------------------------------------------------------------------

def display_welcome_tab():
    st.header(f"Bienvenido a {Config.APP_TITLE}")
    st.info("Sistema de Información Hidroclimática Integrada")
    st.markdown(Config.WELCOME_TEXT)

def display_alerts_tab(df_long, **kwargs):
    st.subheader("🚨 Monitor de Alertas")
    if df_long is not None and not df_long.empty:
        umbral = st.slider("Umbral de Lluvia Mensual (mm)", 100, 1000, 300)
        alertas = df_long[df_long[Config.PRECIPITATION_COL] > umbral]
        st.metric("Eventos Extremos Detectados", len(alertas))
        if not alertas.empty:
            st.dataframe(alertas.sort_values(Config.PRECIPITATION_COL, ascending=False).head(10), use_container_width=True)
    else:
        st.warning("No hay datos cargados para analizar alertas.")

def display_spatial_distribution_tab(gdf_filtered, **kwargs):
    st.subheader("🗺️ Distribución Espacial")
    if gdf_filtered is not None and not gdf_filtered.empty:
        # Intentar usar lat/lon explícitas si existen, sino geometry
        map_data = gdf_filtered.copy()
        if 'latitude' not in map_data.columns and 'geometry' in map_data.columns:
            map_data['latitude'] = map_data.geometry.y
            map_data['longitude'] = map_data.geometry.x
        
        # Mapa simple de Streamlit (muy robusto)
        st.map(map_data, size=20, color='#0000FF')
    else:
        st.warning("No hay estaciones seleccionadas para mostrar en el mapa.")

def display_graphs_tab(df_monthly_filtered, **kwargs):
    st.subheader("📈 Análisis Temporal")
    if df_monthly_filtered is not None and not df_monthly_filtered.empty:
        # Gráfico de líneas interactivo
        fig = px.line(df_monthly_filtered, x=Config.DATE_COL, y=Config.PRECIPITATION_COL, 
                      color=Config.STATION_NAME_COL, title="Serie de Tiempo de Precipitación")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No hay datos mensuales para el rango y estaciones seleccionadas.")

# --- FUNCIONES QUE CAUSABAN EL ERROR (Ahora corregidas) ---

def display_weekly_forecast_tab(stations_for_analysis, gdf_filtered):
    """Muestra el pronóstico semanal para una estación seleccionada."""
    st.subheader("🌦️ Pronóstico a 7 Días (Open-Meteo)")
    
    if not stations_for_analysis:
        st.warning("Seleccione estaciones en el panel lateral primero.")
        return

    selected_station = st.selectbox("Seleccionar Estación:", stations_for_analysis, key="wk_cast_sel")
    
    if selected_station and gdf_filtered is not None:
        station_data = gdf_filtered[gdf_filtered[Config.STATION_NAME_COL] == selected_station]
        if not station_data.empty:
            # Obtener lat/lon
            if 'latitude' in station_data.columns:
                lat = station_data.iloc[0]['latitude']
                lon = station_data.iloc[0]['longitude']
            else:
                lat = station_data.iloc[0].geometry.y
                lon = station_data.iloc[0].geometry.x
            
            df = get_weather_forecast_simple(lat, lon)
            if not df.empty:
                st.dataframe(df, use_container_width=True)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Fecha'], y=df['Temp. Máx (°C)'], name='Máx', line=dict(color='red')))
                fig.add_trace(go.Scatter(x=df['Fecha'], y=df['Temp. Mín (°C)'], name='Mín', line=dict(color='blue')))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No se pudo obtener el pronóstico.")

def display_satellite_imagery_tab(gdf_filtered):
    """Muestra imágenes satelitales WMS."""
    st.subheader("🛰️ Imágenes Satelitales")
    
    wms_url = "https://mesonet.agron.iastate.edu/cgi-bin/wms/goes/east04.cgi?"
    try:
        m = folium.Map(location=[4.5, -74.0], zoom_start=5)
        folium.raster_layers.WmsTileLayer(
            url=wms_url, layers='xc04', fmt='image/png', name='Infrarrojo',
            attr='IEM/GOES', transparent=True, overlay=True
        ).add_to(m)
        
        if gdf_filtered is not None and not gdf_filtered.empty:
            # Añadir puntos simples
            data = gdf_filtered.copy()
            if 'latitude' not in data.columns:
                data['latitude'] = data.geometry.y
                data['longitude'] = data.geometry.x
            
            for _, row in data.iterrows():
                folium.CircleMarker([row['latitude'], row['longitude']], radius=2, color='red').add_to(m)

        st_folium(m, height=500, width="100%")
    except Exception as e:
        st.error(f"Error cargando mapa: {e}")

# --- Placeholders para el resto de pestañas ---

def display_advanced_maps_tab(**kwargs):
    st.info("Módulo de Mapas Avanzados.")

def display_climate_forecast_tab(**kwargs):
    st.info("Módulo de Pronóstico Climático.")

def display_trends_and_forecast_tab(**kwargs):
    st.info("Módulo de Tendencias.")

def display_anomalies_tab(**kwargs):
    st.info("Módulo de Anomalías.")

def display_stats_tab(**kwargs):
    st.info("Módulo de Estadísticas.")

def display_correlation_tab(**kwargs):
    st.info("Módulo de Correlación.")

def display_enso_tab(df_enso, **kwargs):
    st.subheader("Fenómeno ENSO")
    if df_enso is not None:
        st.plotly_chart(create_enso_chart(df_enso), use_container_width=True)

def display_life_zones_tab(**kwargs):
    st.info("Módulo de Zonas de Vida.")

def display_drought_analysis_tab(**kwargs):
    st.info("Módulo de Sequía.")

def display_climate_scenarios_tab(**kwargs):
    st.info("Módulo de Escenarios.")

def display_station_table_tab(**kwargs):
    st.info("Tabla de Datos.")

def display_land_cover_analysis_tab(**kwargs):
    st.info("Módulo de Coberturas.")
