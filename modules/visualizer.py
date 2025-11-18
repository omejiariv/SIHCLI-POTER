# modules/visualizer.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from modules.config import Config
import leafmap.foliumap as leafmap

# --- CONSTANTES DE ESTILO ---
COLOR_PRIMARY = "#1f77b4"
COLOR_SECONDARY = "#ff7f0e"
COLOR_ACCENT = "#2ca02c"
COLOR_BACKGROUND = "#f0f2f6"

# --- FUNCIONES AUXILIARES ---

def _filter_data_by_date(df, start_date, end_date):
    """Filtra un DataFrame por rango de fechas."""
    if df is None or df.empty: return df
    mask = (df[Config.DATE_COL] >= pd.to_datetime(start_date)) & \
           (df[Config.DATE_COL] <= pd.to_datetime(end_date))
    return df.loc[mask]

def _get_common_filtering_ui(gdf_stations, key_suffix=""):
    """Crea la UI común para filtrar por Municipio -> Estación."""
    st.markdown("#### Filtros de Selección")
    
    municipios = sorted(gdf_stations[Config.MUNICIPALITY_COL].unique())
    selected_municipio = st.selectbox(
        "Seleccione un Municipio:",
        ["Todos"] + municipios,
        key=f"municipio_select_{key_suffix}"
    )

    if selected_municipio != "Todos":
        stations_filtered = gdf_stations[gdf_stations[Config.MUNICIPALITY_COL] == selected_municipio]
    else:
        stations_filtered = gdf_stations

    station_options = sorted(stations_filtered[Config.STATION_NAME_COL].unique())
    selected_station = st.selectbox(
        "Seleccione una Estación:",
        station_options,
        key=f"station_select_{key_suffix}"
    )
    
    return selected_station

def create_enso_chart(df_enso_filtered):
    """Crea el gráfico de índices ENSO (ONI, SOI)."""
    if df_enso_filtered is None or df_enso_filtered.empty:
        return go.Figure().update_layout(title="Datos ENSO no disponibles")

    fig = go.Figure()
    if Config.ENSO_ONI_COL in df_enso_filtered.columns:
        fig.add_trace(go.Scatter(
            x=df_enso_filtered[Config.DATE_COL], 
            y=df_enso_filtered[Config.ENSO_ONI_COL], 
            mode='lines+markers', 
            name='ONI'
        ))
    fig.update_layout(title="Índice ONI y Meses Seleccionados", height=300)
    return fig

# --- PESTAÑA 1: BIENVENIDA ---
def display_welcome_tab(**kwargs):
    st.markdown(f"# {Config.APP_TITLE}")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"### {Config.WELCOME_TEXT}")
        st.info(Config.CHAAC_STORY)
    with col2:
        try:
            st.image(Config.CHAAC_IMAGE_PATH, caption="Representación de Chaac", use_column_width=True)
        except:
            st.warning("Imagen de Chaac no encontrada.")
    st.markdown("---")
    st.markdown(f"*{Config.QUOTE_TEXT}* — **{Config.QUOTE_AUTHOR}**")

# --- PESTAÑA 2: ALERTAS ---
def display_alerts_tab(df_long, gdf_stations, start_date, end_date, **kwargs):
    st.markdown("## 🚨 Tablero de Alertas y Resumen Hidrometeorológico")
    df_filtered = _filter_data_by_date(df_long, start_date, end_date)
    
    if df_filtered.empty:
        st.warning("No hay datos para el período seleccionado.")
        return

    total_precip = df_filtered[Config.PRECIPITATION_COL].sum()
    avg_precip = df_filtered[Config.PRECIPITATION_COL].mean()
    max_precip = df_filtered[Config.PRECIPITATION_COL].max()
    n_stations = df_filtered[Config.STATION_NAME_COL].nunique()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precipitación Total", f"{total_precip:,.0f} mm")
    col2.metric("Promedio Mensual", f"{avg_precip:.1f} mm")
    col3.metric("Máxima Mensual", f"{max_precip:.1f} mm")
    col4.metric("Estaciones", n_stations)
    
    st.markdown("---")
    st.subheader("Monitoreo de Umbrales")
    threshold = st.slider("Umbral de Lluvia Mensual (mm):", 0, 1000, 300, 10)
    alerts_df = df_filtered[df_filtered[Config.PRECIPITATION_COL] > threshold].copy()
    
    if not alerts_df.empty:
        top_alerts = alerts_df[Config.STATION_NAME_COL].value_counts().head(10).reset_index()
        top_alerts.columns = ["Estación", "Meses sobre Umbral"]
        fig_alerts = px.bar(top_alerts, x="Meses sobre Umbral", y="Estación", orientation='h', title=f"Alertas > {threshold} mm")
        st.plotly_chart(fig_alerts, use_container_width=True)
    else:
        st.success("Sin alertas para este umbral.")

# --- PESTAÑA 3: MAPAS ---
def display_spatial_distribution_tab(df_long, gdf_stations, start_date, end_date, **kwargs):
    st.markdown("## 🗺️ Distribución Espacial")
    df_filtered = _filter_data_by_date(df_long, start_date, end_date)
    
    if df_filtered.empty:
        st.warning("No hay datos.")
        return

    precip_per_station = df_filtered.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].sum().reset_index()
    
    # Merge seguro
    if Config.STATION_NAME_COL in gdf_stations.columns and Config.STATION_NAME_COL in precip_per_station.columns:
        gdf_map = gdf_stations.merge(precip_per_station, on=Config.STATION_NAME_COL, how='inner')
    else:
        st.error(f"Error de columnas. Asegúrese de que '{Config.STATION_NAME_COL}' exista en ambos datasets.")
        return

    m = leafmap.Map(center=[6.2, -75.5], zoom=8, draw_control=False)
    m.add_basemap("CartoDB.Positron")
    
    # Mapa de burbujas simple
    for _, row in gdf_map.iterrows():
        folium.CircleMarker(
            location=[row[Config.LATITUDE_COL], row[Config.LONGITUDE_COL]],
            radius=5 + (row[Config.PRECIPITATION_COL] / gdf_map[Config.PRECIPITATION_COL].max()) * 20,
            popup=f"{row[Config.STATION_NAME_COL]}: {row[Config.PRECIPITATION_COL]:,.0f} mm",
            color=COLOR_PRIMARY, fill=True, fill_opacity=0.7
        ).add_to(m)
        
    st_folium(m, width=800, height=500)

# --- PESTAÑA 4: GRÁFICOS (VERSIÓN CORREGIDA Y ROBUSTA) ---
def display_graphs_tab(df_anual_melted, df_monthly_filtered, stations_for_analysis,
                       gdf_filtered, analysis_mode, selected_regions, selected_municipios,
                       selected_altitudes, **kwargs):
    
    st.header("📈 Análisis Gráfico")
    
    # Obtenemos los datos originales completos para evitar problemas de merge previos
    gdf_stations = kwargs.get('gdf_stations', pd.DataFrame())
    df_long = kwargs.get('df_long', pd.DataFrame())
    
    col_sel_1, col_sel_2 = st.columns([1, 3])
    
    with col_sel_1:
        # Selector local para esta pestaña
        selected_station = _get_common_filtering_ui(gdf_stations, key_suffix="graphs")
        
    # Filtramos df_long directamente (esto evita el KeyError de merges fallidos)
    df_station = df_long[df_long[Config.STATION_NAME_COL] == selected_station].copy()
    
    # Filtro de fechas (usando el estado de la sesión o defaults)
    year_range = st.session_state.get('year_range', (2000, 2024))
    start_date = pd.to_datetime(f"{year_range[0]}-01-01")
    end_date = pd.to_datetime(f"{year_range[1]}-12-31")
    
    df_station_filtered = df_station[
        (df_station[Config.DATE_COL] >= start_date) & 
        (df_station[Config.DATE_COL] <= end_date)
    ]
    
    with col_sel_2:
        if df_station_filtered.empty:
            st.warning("No hay datos para esta estación en el rango seleccionado.")
            return
            
        # Gráfico de Serie de Tiempo
        fig_ts = px.line(
            df_station_filtered, 
            x=Config.DATE_COL, 
            y=Config.PRECIPITATION_COL,
            title=f"Serie Histórica: {selected_station}",
            color_discrete_sequence=[COLOR_PRIMARY]
        )
        st.plotly_chart(fig_ts, use_container_width=True)
        
    st.markdown("---")
    col_g2, col_g3 = st.columns(2)
    
    with col_g2:
        # Ciclo Anual (Boxplot)
        fig_box = px.box(
            df_station,
            x=Config.MONTH_COL,
            y=Config.PRECIPITATION_COL,
            title=f"Ciclo Anual: {selected_station}",
            color_discrete_sequence=[COLOR_SECONDARY]
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
    with col_g3:
        # Histograma
        fig_hist = px.histogram(
            df_station_filtered,
            x=Config.PRECIPITATION_COL,
            title=f"Distribución: {selected_station}",
            color_discrete_sequence=[COLOR_ACCENT]
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# --- PESTAÑA 5: MAPAS AVANZADOS ---
def display_advanced_maps_tab(df_long, gdf_stations, gdf_municipios, start_date, end_date, **kwargs):
    st.markdown("## 🌐 Interpolación (IDW)")
    
    col_param, col_viz = st.columns([1, 3])
    with col_param:
        years = sorted(df_long[Config.YEAR_COL].unique(), reverse=True)
        selected_year = st.selectbox("Año:", years)
        selected_month = st.slider("Mes:", 1, 12, 1)
        run_idw = st.button("Generar Mapa")

    with col_viz:
        if run_idw:
            df_month = df_long[
                (df_long[Config.YEAR_COL] == selected_year) & 
                (df_long[Config.MONTH_COL] == selected_month)
            ]
            
            if df_month.empty:
                st.error("Sin datos para la fecha.")
                return
            
            # Merge seguro para coordenadas
            if Config.STATION_NAME_COL in df_month.columns and Config.STATION_NAME_COL in gdf_stations.columns:
                df_geo = df_month.merge(gdf_stations[[Config.STATION_NAME_COL, Config.LATITUDE_COL, Config.LONGITUDE_COL]], on=Config.STATION_NAME_COL, how='inner')
            else:
                st.error("Error de columnas en merge.")
                return
            
            if len(df_geo) < 3:
                st.warning("Mínimo 3 estaciones requeridas.")
                return
                
            fig_contour = go.Figure(data =
                go.Contour(
                    z=df_geo[Config.PRECIPITATION_COL],
                    x=df_geo[Config.LONGITUDE_COL],
                    y=df_geo[Config.LATITUDE_COL],
                    colorscale='Blues',
                    line_smoothing=0.85
                ))
            
            fig_contour.add_trace(go.Scatter(
                x=df_geo[Config.LONGITUDE_COL], y=df_geo[Config.LATITUDE_COL],
                mode='markers', marker=dict(color='black', size=5),
                text=df_geo[Config.STATION_NAME_COL]
            ))
            
            fig_contour.update_layout(title=f"Lluvia {selected_month}/{selected_year}", width=800, height=600)
            st.plotly_chart(fig_contour)
