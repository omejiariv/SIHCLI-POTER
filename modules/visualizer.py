import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster, MiniMap
from streamlit_folium import st_folium
from modules.config import Config

# -----------------------------------------------------------------------------
# 1. FUNCIONES HELPER PARA MAPAS Y GRÁFICOS
# -----------------------------------------------------------------------------

def create_folium_map(location, zoom, base_map_config=None, overlays=None):
    """Crea un mapa base de Folium configurado."""
    if base_map_config is None:
        base_map_config = {"tiles": "cartodbpositron", "attr": "CartoDB"}
    
    m = folium.Map(location=location, zoom_start=zoom, tiles=base_map_config["tiles"], attr=base_map_config["attr"])
    return m

def generate_station_popup_html(row, df_anual):
    """Genera el HTML para el popup de la estación."""
    station_name = row.get(Config.STATION_NAME_COL, 'Estación')
    alt = row.get(Config.ALTITUDE_COL, 'N/A')
    mun = row.get(Config.MUNICIPALITY_COL, 'N/A')
    
    # Calcular promedio rápido si hay datos
    avg_ppt = "N/A"
    if not df_anual.empty:
        station_data = df_anual[df_anual[Config.STATION_NAME_COL] == station_name]
        if not station_data.empty:
            avg_ppt = f"{station_data[Config.PRECIPITATION_COL].mean():.0f}"

    html = f"""
    <div style="font-family: sans-serif; width: 200px;">
        <h4>{station_name}</h4>
        <p><b>Municipio:</b> {mun}</p>
        <p><b>Altitud:</b> {alt} m</p>
        <hr>
        <p><b>Ppt Media Anual:</b> {avg_ppt} mm</p>
    </div>
    """
    return folium.Popup(html, max_width=250)

def create_enso_chart(enso_data):
    """Gráfico del Fenómeno ENSO (ONI)."""
    if enso_data is None or enso_data.empty or Config.ENSO_ONI_COL not in enso_data.columns:
        return go.Figure().update_layout(title="Datos ENSO no disponibles")

    data = enso_data.copy().sort_values(Config.DATE_COL)
    data = data.dropna(subset=[Config.ENSO_ONI_COL])
    
    # Colores según fase
    conditions = [data[Config.ENSO_ONI_COL] >= 0.5, data[Config.ENSO_ONI_COL] <= -0.5]
    colors = ['red', 'blue']
    data['color'] = np.select(conditions, colors, default='gray')

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data[Config.DATE_COL], y=data[Config.ENSO_ONI_COL],
        marker_color=data['color'], name="Anomalía ONI"
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="El Niño")
    fig.add_hline(y=-0.5, line_dash="dash", line_color="blue", annotation_text="La Niña")
    fig.update_layout(title="Índice Oceánico El Niño (ONI)", height=350, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# -----------------------------------------------------------------------------
# 2. FUNCIONES DE PESTAÑAS PRINCIPALES
# -----------------------------------------------------------------------------

def display_welcome_tab():
    st.markdown(f"# 🌧️ {Config.APP_TITLE}")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(Config.WELCOME_TEXT)
        st.info("💡 **Tip:** Usa el panel lateral izquierdo para filtrar por región y municipio.")
    with col2:
        if hasattr(Config, 'CHAAC_IMAGE_PATH') and Config.CHAAC_IMAGE_PATH:
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Chaac.jpg/220px-Chaac.jpg", caption="Chaac - Deidad Maya de la Lluvia")

def display_spatial_distribution_tab(gdf_filtered, df_anual_melted, **kwargs):
    st.subheader("📍 Distribución Espacial de Estaciones")
    
    if gdf_filtered is None or gdf_filtered.empty:
        st.warning("No hay estaciones seleccionadas para mostrar en el mapa.")
        return

    # Contenedores de métricas
    m1, m2, m3 = st.columns(3)
    m1.metric("Estaciones Visibles", len(gdf_filtered))
    m2.metric("Municipios Cubiertos", gdf_filtered[Config.MUNICIPALITY_COL].nunique())
    
    # Mapa Folium
    try:
        # Centro aproximado (Antioquia)
        m = create_folium_map(location=[7.0, -75.5], zoom=8)
        
        # Cluster de marcadores para rendimiento
        marker_cluster = MarkerCluster().add_to(m)
        
        for idx, row in gdf_filtered.iterrows():
            # Obtener coordenadas de geometry o columnas explícitas
            lat, lon = None, None
            if 'latitude' in row and 'longitude' in row:
                lat, lon = row['latitude'], row['longitude']
            elif hasattr(row.geometry, 'y'):
                lat, lon = row.geometry.y, row.geometry.x
            
            if lat and lon:
                folium.Marker(
                    location=[lat, lon],
                    popup=generate_station_popup_html(row, df_anual_melted),
                    tooltip=row[Config.STATION_NAME_COL],
                    icon=folium.Icon(color="green", icon="cloud")
                ).add_to(marker_cluster)

        st_folium(m, width="100%", height=500)
        
    except Exception as e:
        st.error(f"Error al renderizar el mapa: {e}")
        # Fallback a mapa simple de Streamlit si Folium falla
        if 'latitude' in gdf_filtered.columns:
            st.map(gdf_filtered)

def display_graphs_tab(df_monthly_filtered, stations_for_analysis, **kwargs):
    st.subheader("📈 Análisis Temporal")
    
    if df_monthly_filtered.empty:
        st.warning("No hay datos mensuales para el rango seleccionado.")
        return

    # Limitar visualización si hay demasiadas estaciones para evitar bloqueo
    if len(stations_for_analysis) > 10:
        st.warning(f"⚠️ Has seleccionado {len(stations_for_analysis)} estaciones. Mostrando solo las primeras 10 para evitar lentitud.")
        stations_to_plot = stations_for_analysis[:10]
        df_plot = df_monthly_filtered[df_monthly_filtered[Config.STATION_NAME_COL].isin(stations_to_plot)]
    else:
        df_plot = df_monthly_filtered

    tab_lines, tab_box, tab_heat = st.tabs(["Serie de Tiempo", "Distribución Mensual", "Mapa de Calor"])
    
    with tab_lines:
        fig = px.line(df_plot, x=Config.DATE_COL, y=Config.PRECIPITATION_COL, 
                      color=Config.STATION_NAME_COL, title="Precipitación Mensual (mm)")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with tab_box:
        fig_box = px.box(df_plot, x=Config.MONTH_COL, y=Config.PRECIPITATION_COL, 
                         color=Config.STATION_NAME_COL, title="Distribución de Lluvias por Mes")
        st.plotly_chart(fig_box, use_container_width=True)

    with tab_heat:
        fig_heat = px.density_heatmap(df_plot, x=Config.DATE_COL, y=Config.STATION_NAME_COL, 
                                      z=Config.PRECIPITATION_COL, histfunc="avg", title="Intensidad de Lluvia")
        st.plotly_chart(fig_heat, use_container_width=True)

def display_alerts_tab(df_long, **kwargs):
    st.subheader("🚨 Monitor de Alertas")
    if df_long is not None and not df_long.empty:
        # Umbral simple
        umbral = st.slider("Definir umbral de alerta mensual (mm):", 100, 1000, 300)
        alertas = df_long[df_long[Config.PRECIPITATION_COL] > umbral]
        
        col1, col2 = st.columns(2)
        col1.metric("Meses que superan umbral", len(alertas))
        col1.markdown(f"**{len(alertas)/len(df_long)*100:.1f}%** del total de registros.")
        
        if not alertas.empty:
            top_stations = alertas[Config.STATION_NAME_COL].value_counts().head(10)
            fig = px.bar(x=top_stations.values, y=top_stations.index, orientation='h', 
                         title="Top Estaciones con Eventos Extremos")
            col2.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Carga datos para ver alertas.")

# --- Placeholders para funciones avanzadas (Para rellenar luego con el PDF completo si se necesita) ---

def display_advanced_maps_tab(**kwargs):
    st.info("🚧 Módulo de Interpolación (Kriging/IDW) en construcción. Requiere procesado pesado.")

def display_climate_forecast_tab(**kwargs):
    st.info("🚧 Módulo de Pronóstico Climático en construcción.")

def display_trends_and_forecast_tab(**kwargs):
    st.info("🚧 Módulo de Tendencias (Mann-Kendall) en construcción.")

def display_anomalies_tab(**kwargs):
    st.info("🚧 Módulo de Anomalías en construcción.")

def display_stats_tab(df_monthly_filtered, **kwargs):
    st.subheader("📊 Estadísticas Descriptivas")
    if not df_monthly_filtered.empty:
        desc = df_monthly_filtered.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].describe()
        st.dataframe(desc.style.format("{:.1f}"), use_container_width=True)

def display_correlation_tab(**kwargs):
    st.info("🚧 Módulo de Correlación en construcción.")

def display_enso_tab(df_enso, **kwargs):
    st.subheader("🌊 Fenómeno ENSO")
    if df_enso is not None:
        st.plotly_chart(create_enso_chart(df_enso), use_container_width=True)

def display_life_zones_tab(**kwargs):
    st.subheader("🌱 Zonas de Vida")
    st.info("Requiere archivos raster (DEM, PPT) en la carpeta 'data'.")

def display_drought_analysis_tab(**kwargs):
    st.info("🚧 Análisis de Sequía (SPI) en construcción.")

def display_climate_scenarios_tab(**kwargs):
    st.info("🚧 Escenarios de Cambio Climático en construcción.")

def display_weekly_forecast_tab(**kwargs):
    st.info("Pronóstico Semanal.")

def display_satellite_imagery_tab(**kwargs):
    st.info("Imágenes Satelitales.")

def display_station_table_tab(**kwargs):
    st.info("Tabla Detallada.")

def display_land_cover_analysis_tab(**kwargs):
    st.info("Coberturas.")
