# modules/visualizer.py

import streamlit as st
import pandas as pd
import base64
import geopandas as gpd
import altair as alt
import folium
from rasterstats import zonal_stats
from folium.plugins import MarkerCluster, MiniMap
from folium.raster_layers import WmsTileLayer
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import branca.colormap as cm
import matplotlib.pyplot as plt
import pymannkendall as mk
from scipy import stats
from prophet.plot import plot_plotly
import io
from datetime import datetime, timedelta, date
import json
import requests
import traceback
import openmeteo_requests


from modules.analysis import calculate_all_station_trends
from modules.analysis import calculate_hydrological_balance
from modules.interpolation import interpolate_idw
from modules.interpolation import create_kriging_by_basin
import rasterio
from modules.life_zones import generate_life_zone_map, holdridge_zone_map_simplified, holdridge_int_to_name_simplified
from rasterio.transform import from_origin
from rasterio.mask import mask
from scipy.interpolate import griddata
import gstools as gs
import pyproj
from rasterio.warp import reproject, Resampling

#--- Importaciones de Módulos Propios
from modules.openmeteo_api import get_historical_climate_average
from modules.analysis import calculate_morphometry, calculate_hypsometric_curve
from modules.analysis import (
    calculate_spi,
    calculate_spei,
    calculate_monthly_anomalies,
    calculate_percentiles_and_extremes,
    analyze_events,
    calculate_climatological_anomalies
)
from modules.config import Config
from modules.utils import add_folium_download_button
from modules.interpolation import create_interpolation_surface, perform_loocv_for_all_methods
from modules.forecasting import (
    generate_sarima_forecast,
    generate_prophet_forecast,
    get_decomposition_results,
    create_acf_chart,
    create_pacf_chart,
    auto_arima_search,
    get_weather_forecast
)
from modules.data_processor import complete_series

#--- FUNCIONES DE UTILIDAD DE VISUALIZACIÓN

def display_filter_summary(total_stations_count, selected_stations_count, year_range,
                           selected_months_count, analysis_mode, selected_regions,
                           selected_municipios, selected_altitudes):

    if isinstance(year_range, tuple) and len(year_range) == 2:
        year_text = f"{year_range[0]}-{year_range[1]}"
    else:
        year_text = "N/A"

    mode_text = "Original (con huecos)"
    if analysis_mode == "Completar series (interpolación)":
        mode_text = "Completado (interpolado)"

    summary_parts = [
        f"**Estaciones:** {selected_stations_count}/{total_stations_count}",
        f"**Período:** {year_text}",
        f"**Datos:** {mode_text}"
    ]

    if selected_regions:
        summary_parts.append(f"**Región:** {', '.join(selected_regions)}")
    if selected_municipios:
        summary_parts.append(f"**Municipio:** {', '.join(selected_municipios)}")
    if selected_altitudes:
        summary_parts.append(f"**Altitud:** {', '.join(selected_altitudes)}")

    st.info(" | ".join(summary_parts))

# NUEVA FUNCIÓN para cargar y cachear los GeoJSON
@st.cache_data
def load_geojson_from_url(url):
    """Descarga y carga un archivo GeoJSON desde una URL."""
    try:
        response = requests.get(url)
        response.raise_for_status() # Lanza un error para respuestas malas (4xx o 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"No se pudo cargar el GeoJSON desde {url}: {e}")
        return None

# MODIFICACIÓN de la función display_map_controls
def display_map_controls(container_object, key_prefix):
    """Muestra los controles para seleccionar mapa base y capas adicionales."""
    base_map_options = {
        "CartoDB Positron": {"tiles": "cartodbpositron", "attr": "CartoDB"},
        "OpenStreetMap": {"tiles": "OpenStreetMap", "attr": "OpenStreetMap"},
        "Topografía (Open TopoMap)": {
            "tiles": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
            "attr": "Open TopoMap"
        },
    }

    # AÑADIR NUEVAS CAPAS GEOJSON AQUÍ
    base_url = "https://raw.githubusercontent.com/omejiariv/Chaac-SIHCLI/main/data/"
    overlay_map_options = {
        # Capa WMS existente
        "División Política Colombia (WMS IDEAM)": {
            "type": "wms",
            "url": "https://geoservicios.ideam.gov.co/geoserver/ideam/wms",
            "layers": "ideam:col_admin",
            "fmt": 'image/png',
            "transparent": True,
            "attr": "IDEAM",
        },
        # Nuevas capas GeoJSON
        "Predios Ejecutados": {
            "type": "geojson",
            "url": f"{base_url}PrediosEjecutados.geojson",
            "attr": "Predios",
            "style": {"color": "#ff7800", "weight": 2, "opacity": 0.7}
        },
        "Subcuencas de Influencia": {
            "type": "geojson",
            "url": f"{base_url}SubcuencasAinfluencia.geojson",
            "attr": "Subcuencas",
            "style": {"color": "#4682B4", "weight": 2, "opacity": 0.7}
        },
        "Municipios de Antioquia": {
            "type": "geojson",
            "url": f"{base_url}MunicipiosAntioquia.geojson",
            "attr": "Municipios Antioquia",
            "style": {"color": "#33a02c", "weight": 1, "opacity": 0.6}
        }
    }

    selected_base_map_name = container_object.selectbox(
        "Seleccionar Mapa Base",
        list(base_map_options.keys()),
        key=f"{key_prefix}_base_map"
    )

    selected_overlays_names = container_object.multiselect(
        "Seleccionar Capas Adicionales",
        list(overlay_map_options.keys()),
        key=f"{key_prefix}_overlays"
    )

    selected_base_map_config = base_map_options[selected_base_map_name]
    selected_overlays_config = [overlay_map_options[name] for name in selected_overlays_names]

    return selected_base_map_config, selected_overlays_config

def create_enso_chart(enso_data):
    if enso_data.empty or Config.ENSO_ONI_COL not in enso_data.columns:
        return go.Figure()

    data = enso_data.copy().sort_values(Config.DATE_COL)
    data.dropna(subset=[Config.ENSO_ONI_COL], inplace=True)

    if data.empty:
        return go.Figure()

    conditions = [
        data[Config.ENSO_ONI_COL] >= 0.5,
        data[Config.ENSO_ONI_COL] <= -0.5
    ]
    phases = ['El Niño', 'La Niña']
    colors = ['red', 'blue']
    data['phase'] = np.select(conditions, phases, default='Neutral')
    data['color'] = np.select(conditions, colors, default='grey')

    y_range = [data[Config.ENSO_ONI_COL].min() - 0.5, data[Config.ENSO_ONI_COL].max() + 0.5]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=data[Config.DATE_COL], y=[y_range[1] - y_range[0]] * len(data),
        base=y_range[0], marker_color=data['color'], opacity=0.3,
        hoverinfo='none', showlegend=False
    ))

    legend_map = {'El Niño': 'red', 'La Niña': 'blue', 'Neutral': 'grey'}
    for phase, color in legend_map.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=15, color=color, symbol='square', opacity=0.5), name=phase, showlegend=True))

    fig.add_trace(go.Scatter(x=data[Config.DATE_COL], y=data[Config.ENSO_ONI_COL],
                             mode='lines', name='Anomalía ONI', line=dict(color='black', width=2), showlegend=True))

    # CORRECCIÓN DE SINTAXIS: Se añadieron las comas faltantes.
    fig.add_hline(y=0.5, line_dash="dash", line_color="red")
    fig.add_hline(y=-0.5, line_dash="dash", line_color="blue")

    fig.update_layout(height=600, title="Fases del Fenómeno ENSO y Anomalía ONI",
                      yaxis_title="Anomalía ONI (°C)", xaxis_title="Fecha", showlegend=True,
                      legend_title_text='Fase', yaxis_range=y_range)

    return fig

def create_anomaly_chart(df_plot):
    if df_plot.empty:
        return go.Figure()

    df_plot['color'] = np.where(df_plot['anomalia'] < 0, 'red', 'blue')

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_plot[Config.DATE_COL],
        y=df_plot['anomalia'],
        marker_color=df_plot['color'],
        name='Anomalía de Precipitación'
    ))

    fig.update_layout(
        height=600,
        title="Anomalías Mensuales de Precipitación y Fases ENSO",
        yaxis_title="Anomalía de Precipitación (mm)",
        xaxis_title="Fecha",
        showlegend=True
    )

    return fig

def generate_station_popup_html(row, df_anual_melted):
    station_name = row.get(Config.STATION_NAME_COL, 'N/A')

    try:
        year_range_val = st.session_state.get('year_range', (2000, 2025))
        year_min, year_max = year_range_val

        total_years_in_period = year_max - year_min + 1

        df_station_data = df_anual_melted[df_anual_melted[Config.STATION_NAME_COL] == station_name]
        precip_media_anual = df_station_data['precipitation'].mean() if not df_station_data.empty else 0
        valid_years = df_station_data['precipitation'].count() if not df_station_data.empty else 0

        precip_formatted = f"{precip_media_anual:.0f}" if pd.notna(precip_media_anual) else "N/A"

        text_html = f"<h4>{station_name}</h4>"
        text_html += f"<p><b>Municipio:</b> {row.get(Config.MUNICIPALITY_COL, 'N/A')}</p>"
        text_html += f"<p><b>Altitud:</b> {row.get(Config.ALTITUDE_COL, 'N/A')} m</p>"
        text_html += f"<p><b>Promedio Anual:</b> {precip_formatted} mm</p>"
        text_html += f"<small>(Calculado con <b>{valid_years}</b> de <b>{total_years_in_period}</b> años del período)</small>"

        return folium.Popup(text_html, max_width=450)

    except Exception as e:
        st.warning(f"No se pudo generar el popup para '{station_name}'. Razón: {e}")
        return folium.Popup(f"<h4>{station_name}</h4><p>Error al cargar datos del popup.</p>", max_width=450)

def generate_annual_map_popup_html(row, df_anual_melted_full_period):
    station_name = row.get(Config.STATION_NAME_COL, 'N/A')
    municipality = row.get(Config.MUNICIPALITY_COL, 'N/A')
    altitude = row.get(Config.ALTITUDE_COL, 'N/A')
    precip_year = row.get(Config.PRECIPITATION_COL, 'N/A')

    station_full_data = df_anual_melted_full_period[df_anual_melted_full_period[Config.STATION_NAME_COL] == station_name]

    precip_avg, precip_max, precip_min = "N/A", "N/A", "N/A"
    if not station_full_data.empty:
        precip_avg = f"{station_full_data[Config.PRECIPITATION_COL].mean():.0f}"
        precip_max = f"{station_full_data[Config.PRECIPITATION_COL].max():.0f}"
        precip_min = f"{station_full_data[Config.PRECIPITATION_COL].min():.0f}"

    altitude_formatted = f"{altitude:.0f}" if isinstance(altitude, (int, float)) and np.isfinite(altitude) else "N/A"
    precip_year_formatted = f"{precip_year:.0f}" if isinstance(precip_year, (int, float)) and np.isfinite(precip_year) else "N/A"

    html = f"""
    <h4>{station_name}</h4>
    <p><b>Municipio:</b> {municipality}</p>
    <p><b>Altitud:</b> {altitude_formatted} m</p>
    <hr>
    <p><b>Precipitación del Año:</b> {precip_year_formatted} mm</p>
    <p><b>Promedio Anual (histórico):</b> {precip_avg} mm</p>
    <p><small><b>Máxima del período:</b> {precip_max} mm</small></p>
    <p><small><b>Mínima del período:</b> {precip_min} mm</small></p>
    """
    return folium.Popup(html, max_width=300)

def create_folium_map(location, zoom, base_map_config, overlays_config, fit_bounds_data=None):
    """Crea un mapa base de Folium y le añade capas de overlay de forma inteligente."""
    m = folium.Map(
        location=location,
        zoom_start=zoom,
        # CORRECCIÓN DE ATTRIBUTEERROR: Extraer solo la clave 'tiles' (que es str)
        tiles=base_map_config.get('tiles', 'OpenStreetMap'),
        attr=base_map_config.get('attr', 'OpenStreetMap')
    )

    if fit_bounds_data is not None and not fit_bounds_data.empty:
        if len(fit_bounds_data) > 1:
            bounds = fit_bounds_data.total_bounds
            if np.all(np.isfinite(bounds)):
                m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
        elif len(fit_bounds_data) == 1:
            point = fit_bounds_data.iloc[0].geometry
            m.location = [point.y, point.x]
            m.zoom_start = 12

    if overlays_config:
        for layer_config in overlays_config:
            layer_type = layer_config.get("type", "tile") # Asumir 'tile' por defecto
            url = layer_config.get("url")

            if not url:
                continue

            layer_name = layer_config.get("attr", "Overlay")

            # LÓGICA PARA DISTINGUIR TIPOS DE CAPA
            if layer_type == "wms":
                WmsTileLayer(
                    url=url,
                    layers=layer_config["layers"],
                    fmt=layer_config.get("fmt", 'image/png'),
                    transparent=layer_config.get("transparent", True),
                    overlay=True,
                    control=True,
                    name=layer_name
                ).add_to(m)

            elif layer_type == "geojson":
                geojson_data = load_geojson_from_url(url)
                if geojson_data:
                    style_function = lambda x: layer_config.get("style", {})

                    folium.GeoJson(
                        geojson_data,
                        name=layer_name,
                        style_function=style_function
                    ).add_to(m)

            else: # Asumir que es una capa de teselas (TileLayer)
                folium.TileLayer(
                    tiles=url,
                    attr=layer_name,
                    name=layer_name,
                    overlay=True,
                    control=True,
                    show=False
                ).add_to(m)

    return m

# MAIN TAB DISPLAY FUNCTIONS

def display_welcome_tab():
    st.header("Bienvenido al Sistema de Información de Lluvias y Clima")
    st.markdown("""
<style>
@import
url('https://fonts.googleapis.com/css?family=Playfair+Display:wght@700&display=swap');
.quote { font-family: 'Playfair Display', serif; font-weight: 700; font-size: 22px; text-align:
center; padding: 20px; }
.author { font-family: 'Playfair Display', serif; text-align: right; font-style: italic; font-size:
18px; padding-right: 20px; }
</style>
""", unsafe_allow_html=True)

    st.markdown(f'<p class="quote">{Config.QUOTE_TEXT}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="author">- {Config.QUOTE_AUTHOR}</p>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        st.markdown(Config.WELCOME_TEXT, unsafe_allow_html=True)
        with st.expander("La Inspiración: Chaac, Divinidad Maya", expanded=False):
            st.markdown(Config.CHAAC_STORY)

    with col2:
        if os.path.exists(Config.CHAAC_IMAGE_PATH):
            st.image(Config.CHAAC_IMAGE_PATH, caption="Representación de Chaac, Códice Dresde.")
        if os.path.exists(Config.LOGO_PATH):
            st.image(Config.LOGO_PATH, width=250, caption="Corporación Cuenca Verde")
        st.markdown("---")

    with st.expander("Conceptos Clave, Métodos y Ecuaciones", expanded=True):
        st.markdown("""
Esta sección proporciona una descripción de los métodos y conceptos analíticos
utilizados en la plataforma.

### Análisis de Anomalías

Una **anomalía** representa la diferencia entre el valor observado en un momento
dado y un valor de referencia o "normal".

**Anomalía vs. Período Seleccionado**: Compara la lluvia de cada mes con el
promedio histórico de todos los meses iguales (ej. todos los eneros, febreros, etc.) en el
conjunto de datos.

**Anomalía vs. Normal Climatológica**: Compara la lluvia de cada mes con el
promedio de un período de referencia de 30 años (ej. 1991-2020), según recomienda la
Organización Meteorológica Mundial.

### Métodos de Interpolación Espacial

La interpolación estima la precipitación en lugares sin estaciones de medición.

- **IDW (Inverso de la Distancia Ponderada)**: Método que asume que los puntos
más cercanos tienen más influencia.

- **Kriging**: Método geoestadístico que usa la autocorrelación espacial (variograma)
para estimaciones más precisas.

- **Spline (Thin Plate)**: Método matemático que ajusta una superficie flexible a los
datos.

### Índices de Sequía

Estandarizan la precipitación para comparar la intensidad de sequías y períodos
húmedos.

- **SPI (Índice de Precipitación Estandarizado)**: Mide la desviación de la
precipitación respecto a su media histórica.

- **SPEI (Índice Estandarizado de Precipitación-Evapotranspiración)**: Versión
avanzada del SPI que incluye la evapotranspiración, haciéndolo más relevante en estudios
de cambio climático.

### Análisis de Frecuencia de Extremos

- **Período de Retorno**: Estimación estadística de la probabilidad de que un evento
extremo ocurra. Un período de retorno de 100 años tiene una probabilidad del 1% de
ocurrir en cualquier año.

### Análisis de Tendencias

- **Prueba de Mann-Kendall**: Prueba no paramétrica para detectar tendencias
monótonas (crecientes o decrecientes) en una serie de tiempo.

- **Pendiente de Sen**: Cuantifica la magnitud de la tendencia detectada por Mann-
Kendall (ej. "aumento de $5~mm/a\\tilde{n}o$").

### Modelos de Pronóstico

- **SARIMA**: Modelo estadístico clásico para series de tiempo que descompone los
datos en tendencia, estacionalidad y ruido.

- **Prophet**: Modelo de Facebook, automático y robusto, ideal para series con
fuertes efectos estacionales y datos faltantes.
""")

def display_spatial_distribution_tab(gdf_filtered, stations_for_analysis, df_anual_melted,
                                     df_monthly_filtered, analysis_mode, selected_regions,
                                     selected_municipios,
                                     selected_altitudes, **kwargs):

    st.header("Distribución espacial de las Estaciones de Lluvia")

    display_filter_summary(total_stations_count=len(st.session_state.gdf_stations),
                           selected_stations_count=len(stations_for_analysis),
                           year_range=st.session_state.year_range,
                           selected_months_count=len(st.session_state.meses_numeros),
                           analysis_mode=analysis_mode, selected_regions=selected_regions,
                           selected_municipios=selected_municipios,
                           selected_altitudes=selected_altitudes)

    if not stations_for_analysis:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return

    gdf_display = gdf_filtered.copy()

    # CORRECCIÓN DE SYNTAX ERROR: La string literal debe estar completa en una sola línea
    sub_tab_mapa, sub_tab_grafico = st.tabs(["Mapa Interactivo", "Gráfico de Disponibilidad de Datos"])

    with sub_tab_mapa:
        controls_col, map_col = st.columns([1, 3])

        with controls_col:
            st.subheader("Controles del Mapa")
            selected_base_map_config, selected_overlays_config = display_map_controls(st,
                                                                                       "dist_esp")
            st.metric("Estaciones en Vista", len(gdf_display))

        with map_col:
            if not gdf_display.empty:
                m = create_folium_map(
                    location=[6.2, -75.5],
                    zoom=7,
                    base_map_config=selected_base_map_config,
                    overlays_config=selected_overlays_config,
                    fit_bounds_data=gdf_display
                )

                if 'gdf_municipios' in st.session_state and st.session_state.gdf_municipios is not \
                   None:
                    folium.GeoJson(st.session_state.gdf_municipios.to_json(),
                                   name='Municipios').add_to(m)

                marker_cluster = MarkerCluster(name='Estaciones').add_to(m)

                for _, row in gdf_display.iterrows():
                    popup_object = generate_station_popup_html(row, df_anual_melted)
                    folium.Marker(
                        location=[row['geometry'].y, row['geometry'].x],
                        tooltip=row[Config.STATION_NAME_COL],
                        popup=popup_object
                    ).add_to(marker_cluster)

                m.add_child(MiniMap(toggle_display=True))
                folium.LayerControl().add_to(m)

                folium_static(m, height=500, width=None)

                add_folium_download_button(m, "mapa_distribucion_espacial.html")

            else:
                st.warning("No hay estaciones seleccionadas para mostrar en el mapa.")

    with sub_tab_grafico:
        st.subheader("Disponibilidad y Composición de Datos por Estación")

        if not gdf_display.empty:
            if analysis_mode == "Completar series (interpolación)":
                st.info("Mostrando la composición de datos originales vs. completados para el período seleccionado.")

                if not df_monthly_filtered.empty and Config.ORIGIN_COL in \
                   df_monthly_filtered.columns:

                    data_composition = \
                        df_monthly_filtered.groupby([Config.STATION_NAME_COL,
                                                     Config.ORIGIN_COL]).size().unstack(fill_value=0)

                    if 'Original' not in data_composition: data_composition['Original'] = 0
                    if 'Completado' not in data_composition: data_composition['Completado'] = 0

                    data_composition['total'] = data_composition['Original'] + \
                        data_composition['Completado']

                    data_composition['% Original'] = (data_composition['Original'] /
                                                     data_composition['total']) * 100

                    data_composition['% Completado'] = (data_composition['Completado'] /
                                                       data_composition['total']) * 100

                    sort_order_comp = st.radio("Ordenar por:", ["% Datos Originales (Mayor a Menor)", "% Datos Originales (Menor a Mayor)", "Alfabético"], horizontal=True,
                                               key="sort_comp")

                    if "Mayor a Menor" in sort_order_comp: data_composition = \
                        data_composition.sort_values("% Original", ascending=False)

                    elif "Menor a Mayor" in sort_order_comp: data_composition = \
                        data_composition.sort_values("% Original", ascending=True)

                    else: data_composition = data_composition.sort_index(ascending=True)

                    df_plot = data_composition.reset_index().melt(
                        id_vars=Config.STATION_NAME_COL,
                        value_vars=['% Original', '% Completado'],
                        var_name='Tipo de Dato',
                        value_name='Porcentaje'
                    )

                    fig_comp = px.bar(
                        df_plot,
                        x=Config.STATION_NAME_COL,
                        y='Porcentaje',
                        color='Tipo de Dato',
                        title='Composición de Datos por Estación',
                        labels={Config.STATION_NAME_COL: 'Estación', 'Porcentaje': '% del Período'},
                        text_auto='.1f',
                        color_discrete_map={'% Original': '#1f77b4', '% Completado': '#ff7f0e'}
                    )

                    fig_comp.update_layout(height=500, xaxis={'categoryorder': 'trace'},
                                           barmode='stack')

                    st.plotly_chart(fig_comp, use_container_width=True)

                else:
                    st.warning("No hay datos mensuales procesados para mostrar la composición.")

            else:
                st.info("Mostrando el porcentaje de disponibilidad de datos según el archivo de estaciones.")

                sort_order_disp = st.radio("Ordenar estaciones por:", ["% Datos (Mayor a Menor)", "% Datos (Menor a Mayor)", "Alfabético"], horizontal=True, key="sort_disp")

                df_chart = gdf_display.copy()

                if Config.PERCENTAGE_COL in df_chart.columns:
                    if "% Datos (Mayor a Menor)" in sort_order_disp:
                        df_chart = df_chart.sort_values(Config.PERCENTAGE_COL, ascending=False)

                    elif "% Datos (Menor a Mayor)" in sort_order_disp:
                        df_chart = df_chart.sort_values(Config.PERCENTAGE_COL, ascending=True)

                    else:
                        df_chart = df_chart.sort_values(Config.STATION_NAME_COL,
                                                        ascending=True)

                    fig_disp = px.bar(
                        df_chart,
                        x=Config.STATION_NAME_COL,
                        y=Config.PERCENTAGE_COL,
                        title='Porcentaje de Disponibilidad de Datos Históricos',
                        labels={Config.STATION_NAME_COL: 'Estación', Config.PERCENTAGE_COL: '% de Datos Disponibles'},
                        color=Config.PERCENTAGE_COL,
                        color_continuous_scale=px.colors.sequential.Viridis
                    )

                    fig_disp.update_layout(height=500, xaxis={'categoryorder': 'trace'})

                    st.plotly_chart(fig_disp, use_container_width=True)

                else:
                    st.warning("La columna con el porcentaje de datos no se encuentra en el archivo de estaciones.")

        else:
            st.warning("No hay estaciones seleccionadas para mostrar el gráfico.")

def display_graphs_tab(df_anual_melted, df_monthly_filtered, stations_for_analysis,
                       gdf_filtered, analysis_mode, selected_regions, selected_municipios,
                       selected_altitudes, **kwargs):
    st.header("Visualizaciones de Precipitación")
    display_filter_summary(
        total_stations_count=len(st.session_state.gdf_stations),
        selected_stations_count=len(stations_for_analysis),
        year_range=st.session_state.year_range,
        selected_months_count=len(st.session_state.meses_numeros),
        analysis_mode=analysis_mode,
        selected_regions=selected_regions,
        selected_municipios=selected_municipios,
        selected_altitudes=selected_altitudes
    )

    if not stations_for_analysis:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return

    year_range_val = st.session_state.get('year_range', (2000, 2020))
    year_min, year_max = year_range_val

    metadata_cols = [Config.STATION_NAME_COL, Config.MUNICIPALITY_COL, Config.ALTITUDE_COL]
    gdf_metadata = gdf_filtered[metadata_cols].drop_duplicates(subset=[Config.STATION_NAME_COL]).copy()
    if Config.ALTITUDE_COL in gdf_metadata.columns:
        gdf_metadata[Config.ALTITUDE_COL] = pd.to_numeric(gdf_metadata[Config.ALTITUDE_COL], errors='coerce').fillna(-9999).astype(int).astype(str)
    if Config.MUNICIPALITY_COL in gdf_metadata.columns:
        gdf_metadata[Config.MUNICIPALITY_COL] = gdf_metadata[Config.MUNICIPALITY_COL].astype(str).str.strip().replace('nan', 'Sin Dato')
    
    cols_to_drop = [col for col in [Config.MUNICIPALITY_COL, Config.ALTITUDE_COL] if col != Config.STATION_NAME_COL]
    df_anual_pre_merge = df_anual_melted.drop(columns=cols_to_drop, errors='ignore')
    df_anual_rich = df_anual_pre_merge.merge(gdf_metadata, on=Config.STATION_NAME_COL, how='left')
    df_monthly_pre_merge = df_monthly_filtered.drop(columns=cols_to_drop, errors='ignore')
    df_monthly_rich = df_monthly_pre_merge.merge(gdf_metadata, on=Config.STATION_NAME_COL, how='left')

    tab_keys = [
        "Análisis Anual", "Análisis Mensual", "Comparación Rápida",
        "Boxplot Anual", "Distribución", "Acumulada",
        "Relación Altitud", "Serie Regional", "Descargar Datos"
    ]
    sub_tab_anual, sub_tab_mensual, sub_tab_comparacion, sub_tab_boxplot, \
    sub_tab_distribucion, sub_tab_acumulada, sub_tab_altitud, sub_tab_regional, \
    sub_tab_descarga = st.tabs(tab_keys)

    with sub_tab_anual:
        anual_graf_tab, anual_analisis_tab = st.tabs(["Gráfico de Serie Anual", "Análisis Multianual"])
        with anual_graf_tab:
            if not df_anual_rich.empty:
                st.subheader("Precipitación Anual (mm)")
                fig_anual = px.line(
                    df_anual_rich.dropna(subset=[Config.PRECIPITATION_COL]),
                    x=Config.YEAR_COL,
                    y=Config.PRECIPITATION_COL,
                    color=Config.STATION_NAME_COL,
                    markers=True,
                    title=f'Precipitación Anual por Estación ({year_min} - {year_max})',
                    labels={Config.YEAR_COL: 'Año', Config.PRECIPITATION_COL: 'Precipitación (mm)'},
                    hover_data=[Config.MUNICIPALITY_COL, Config.ALTITUDE_COL]
                )
                fig_anual.update_layout(height=500)
                st.plotly_chart(fig_anual, use_container_width=True)
            else:
                st.warning("No hay datos anuales para mostrar la serie.")
        
        with anual_analisis_tab:
            if not df_anual_rich.empty:
                st.subheader("Precipitación Media Multianual")
                st.caption(f"Período de análisis: {year_min} - {year_max}")

                chart_type_annual = st.radio(
                    "Seleccionar tipo de gráfico:",
                    ("Gráfico de Barras (Promedio)", "Gráfico de Cajas (Distribución)"),
                    key="avg_chart_type_annual",
                    horizontal=True
                )

                if chart_type_annual == "Gráfico de Barras (Promedio)":
                    df_summary = df_anual_rich.groupby(Config.STATION_NAME_COL,
                                                       as_index=False)[Config.PRECIPITATION_COL].mean().round(0)

                    sort_order = st.radio(
                        "Ordenar estaciones por:",
                        ["Promedio (Mayor a Menor)", "Promedio (Menor a Mayor)", "Alfabético"],
                        horizontal=True,
                        key="sort_annual_avg"
                    )

                    if "Mayor a Menor" in sort_order:
                        df_summary = df_summary.sort_values(Config.PRECIPITATION_COL,
                                                            ascending=False)

                    elif "Menor a Mayor" in sort_order:
                        df_summary = df_summary.sort_values(Config.PRECIPITATION_COL,
                                                            ascending=True)

                    else:
                        df_summary = df_summary.sort_values(Config.STATION_NAME_COL,
                                                            ascending=True)

                    fig_avg = px.bar(
                        df_summary,
                        x=Config.STATION_NAME_COL,
                        y=Config.PRECIPITATION_COL,
                        title=f'Promedio de Precipitación Anual por Estación ({year_min} - {year_max})',
                        labels={Config.STATION_NAME_COL: 'Estación', Config.PRECIPITATION_COL: 'Precipitación Media Anual (mm)'},
                        color=Config.PRECIPITATION_COL,
                        color_continuous_scale=px.colors.sequential.Blues_r
                    )

                    category_order = 'total descending' if "Mayor" in sort_order else ('total ascending' if "Menor" in sort_order else 'trace')
                    fig_avg.update_layout(height=500, xaxis={'categoryorder': category_order})
                    st.plotly_chart(fig_avg, use_container_width=True)

                else:
                    df_anual_filtered_for_box = \
                        df_anual_rich[df_anual_rich[Config.STATION_NAME_COL].isin(stations_for_analysis)]

                    fig_box_annual = px.box(
                        df_anual_filtered_for_box,
                        x=Config.STATION_NAME_COL,
                        y=Config.PRECIPITATION_COL,
                        color=Config.STATION_NAME_COL,
                        points='all',
                        title='Distribución de la Precipitación Anual por Estación',
                        labels={Config.STATION_NAME_COL: 'Estación', Config.PRECIPITATION_COL: 'Precipitación Anual (mm)'}
                    )

                    fig_box_annual.update_layout(height=500)
                    st.plotly_chart(fig_box_annual, use_container_width=True,
                                    key="box_anual_multianual")

            else:
                st.warning("No hay datos anuales para mostrar el análisis multianual.")

    with sub_tab_mensual:
        mensual_graf_tab, mensual_enso_tab, mensual_datos_tab = st.tabs([
            "Gráfico de Serie Mensual", "Análisis ENSO en el Período", "Tabla de Datos"
        ])

        with mensual_graf_tab:
            if not df_monthly_rich.empty:
                controls_col, chart_col = st.columns([1, 4])
                with controls_col:
                    st.markdown("##### Opciones del Gráfico")
                    chart_type = st.radio("Tipo de Gráfico:", ["Líneas y Puntos", "Nube de Puntos", "Gráfico de Cajas (Distribución Mensual)"], key="monthly_chart_type")
                    color_by_disabled = (chart_type == "Gráfico de Cajas (Distribución Mensual)")
                    color_by = st.radio("Colorear por:", ["Estación", "Mes"], key="monthly_color_by", disabled=color_by_disabled)
                
                with chart_col:
                    if chart_type != "Gráfico de Cajas (Distribución Mensual)":
                        fig_mensual = px.scatter(
                            df_monthly_rich,
                            x=Config.DATE_COL,
                            y=Config.PRECIPITATION_COL,
                            color=Config.STATION_NAME_COL if color_by == "Estación" else df_monthly_rich[Config.DATE_COL].dt.month,
                            title=f"Serie de Precipitación Mensual ({year_min} - {year_max})",
                            labels={
                                Config.DATE_COL: 'Fecha',
                                Config.PRECIPITATION_COL: 'Precipitación (mm)',
                                'color': 'Mes' if color_by == 'Mes' else 'Estación'
                            },
                            hover_data={
                                Config.STATION_NAME_COL: True,
                                Config.MUNICIPALITY_COL: True,
                                Config.ALTITUDE_COL: True
                            }
                        )
                        
                        if chart_type == "Líneas y Puntos":
                            fig_mensual.update_traces(mode='lines+markers')
                        
                        fig_mensual.update_layout(height=500)
                        st.plotly_chart(fig_mensual, use_container_width=True)
                    
                    else:
                        st.subheader("Distribución de la Precipitación Mensual")
                        fig_box_monthly = px.box(
                            df_monthly_rich,
                            x=Config.MONTH_COL,
                            y=Config.PRECIPITATION_COL,
                            color=Config.STATION_NAME_COL,
                            title='Distribución de la Precipitación por Mes',
                            labels={
                                Config.MONTH_COL: 'Mes',
                                Config.PRECIPITATION_COL: 'Precipitación Mensual (mm)',
                                Config.STATION_NAME_COL: 'Estación'
                            }
                        )
                        fig_box_monthly.update_layout(height=500)
                        st.plotly_chart(fig_box_monthly, use_container_width=True)
            else:
                st.warning("No hay datos mensuales para mostrar el gráfico.")

        with mensual_enso_tab:
            if 'df_enso' in st.session_state and st.session_state.df_enso is not None:
                enso_filtered = st.session_state.df_enso[
                    (st.session_state.df_enso[Config.DATE_COL].dt.year >= year_min) &
                    (st.session_state.df_enso[Config.DATE_COL].dt.year <= year_max) &
                    (st.session_state.df_enso[Config.DATE_COL].dt.month.isin(st.session_state.meses_numeros))
                ]
                fig_enso_mensual = create_enso_chart(enso_filtered)
                st.plotly_chart(fig_enso_mensual, use_container_width=True, key="enso_chart_mensual")
            else:
                st.info("No hay datos ENSO disponibles para este análisis.")

        with mensual_datos_tab:
            st.subheader("Datos de Precipitación Mensual Detallados")
            if not df_monthly_rich.empty:
                df_values = df_monthly_rich.pivot_table(
                    index=Config.DATE_COL,
                    columns=Config.STATION_NAME_COL,
                    values=Config.PRECIPITATION_COL
                ).round(1)
                st.dataframe(df_values, use_container_width=True)
            else:
                st.info("No hay datos mensuales detallados.")

        with sub_tab_comparacion:
            st.subheader("Comparación de Precipitación Mensual entre Estaciones")

            if len(stations_for_analysis) < 2:
                st.info("Seleccione al menos dos estaciones para comparar.")

            else:
                st.markdown("##### Precipitación Mensual Promedio")

                df_monthly_avg_rich = df_monthly_rich.groupby([Config.STATION_NAME_COL,
                                                               Config.MONTH_COL]).agg(
                    precip_promedio=(Config.PRECIPITATION_COL, 'mean'),
                    precip_max=(Config.PRECIPITATION_COL, 'max'),
                    precip_min=(Config.PRECIPITATION_COL, 'min'),
                    municipio=(Config.MUNICIPALITY_COL, 'first'),
                    altitud=(Config.ALTITUDE_COL, 'first')
                ).reset_index()

                fig_avg_monthly = px.line(
                    df_monthly_avg_rich,
                    x=Config.MONTH_COL,
                    y='precip_promedio',
                    color=Config.STATION_NAME_COL,
                    labels={'precip_promedio': 'Precipitación Promedio (mm)', Config.MONTH_COL: 'Mes'},
                    title='Promedio de Precipitación Mensual por Estación',
                    hover_data={'municipio': True, 'altitud': True, 'precip_max': ':.0f', 'precip_min': ':.0f'}
                )

                meses_dict = {
                    'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6,
                    'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
                }

                fig_avg_monthly.update_layout(
                    height=500,
                    xaxis=dict(tickmode='array', tickvals=list(meses_dict.values()),
                               ticktext=list(meses_dict.keys()))
                )

                st.plotly_chart(fig_avg_monthly, use_container_width=True)

        with sub_tab_boxplot:
            st.subheader("Distribución de Precipitación Anual por Estación")

            if len(stations_for_analysis) < 1:
                st.info("Seleccione al menos una estación para ver la distribución.")

            else:
                df_anual_filtered_for_box = \
                    df_anual_rich[df_anual_rich[Config.STATION_NAME_COL].isin(stations_for_analysis)]

                if not df_anual_filtered_for_box.empty:
                    fig_box_annual = px.box(
                        df_anual_filtered_for_box,
                        x=Config.STATION_NAME_COL,
                        y=Config.PRECIPITATION_COL,
                        color=Config.STATION_NAME_COL,
                        points='all',
                        title='Distribución de la Precipitación Anual por Estación',
                        labels={Config.STATION_NAME_COL: 'Estación', Config.PRECIPITATION_COL: 'Precipitación Anual (mm)'}
                    )

                    fig_box_annual.update_layout(height=500)
                    st.plotly_chart(fig_box_annual, use_container_width=True,
                                    key="box_anual_comparacion")

                else:
                    st.warning("No hay datos anuales para mostrar el gráfico de cajas.")

        with sub_tab_distribucion:
            st.subheader("Distribución de la Precipitación")

            distribucion_tipo = st.radio("Seleccionar tipo de distribución:", ("Anual", "Mensual"),
                                         horizontal=True)

            plot_type = st.radio(
                "Seleccionar tipo de gráfico:",
                ("Histograma", "Gráfico de Violin"),
                horizontal=True,
                key="distribucion_plot_type"
            )

            hover_info = [Config.MUNICIPALITY_COL, Config.ALTITUDE_COL]
            if distribucion_tipo == "Anual":
                if not df_anual_rich.empty:
                    if plot_type == "Histograma":
                        fig_hist_anual = px.histogram(
                            df_anual_rich,
                            x=Config.PRECIPITATION_COL,
                            color=Config.STATION_NAME_COL,
                            title=f'Distribución Anual de Precipitación ({year_min} - {year_max})',
                            labels={Config.PRECIPITATION_COL: 'Precipitación Anual (mm)', 'count': 'Frecuencia'},
                            hover_data=hover_info
                        )

                        fig_hist_anual.update_layout(height=500)
                        st.plotly_chart(fig_hist_anual, use_container_width=True)

                    else:
                        fig_violin_anual = px.violin(
                            df_anual_rich,
                            y=Config.PRECIPITATION_COL,
                            x=Config.STATION_NAME_COL,
                            color=Config.STATION_NAME_COL,
                            box=True,
                            points="all",
                            title='Distribución Anual con Gráfico de Violin',
                            labels={Config.PRECIPITATION_COL: 'Precipitación Anual (mm)',
                                    Config.STATION_NAME_COL: 'Estación'},
                            hover_data=hover_info
                        )

                        fig_violin_anual.update_layout(height=500)
                        st.plotly_chart(fig_violin_anual, use_container_width=True)

                else:
                    st.warning("No hay datos anuales para mostrar la distribución.")

            else:  # Mensual
                if not df_monthly_rich.empty:
                    if plot_type == "Histograma":
                        fig_hist_mensual = px.histogram(
                            df_monthly_rich,
                            x=Config.PRECIPITATION_COL,
                            color=Config.STATION_NAME_COL,
                            title=f'Distribución Mensual de Precipitación ({year_min} - {year_max})',
                            labels={Config.PRECIPITATION_COL: 'Precipitación Mensual (mm)', 'count': 'Frecuencia'},
                            hover_data=hover_info
                        )

                        fig_hist_mensual.update_layout(height=500)
                        st.plotly_chart(fig_hist_mensual, use_container_width=True)

                    else:
                        fig_violin_mensual = px.violin(
                            df_monthly_rich,
                            y=Config.PRECIPITATION_COL,
                            x=Config.MONTH_COL,
                            color=Config.STATION_NAME_COL,
                            box=True,
                            points="all",
                            title='Distribución Mensual con Gráfico de Violin',
                            labels={
                                Config.MONTH_COL: 'Mes',
                                Config.PRECIPITATION_COL: 'Precipitación Mensual (mm)',
                                Config.STATION_NAME_COL: 'Estación'
                            },
                            hover_data=hover_info
                        )

                        fig_violin_mensual.update_layout(height=500)
                        st.plotly_chart(fig_violin_mensual, use_container_width=True)

                else:
                    st.warning("No hay datos mensuales para mostrar el gráfico.")

        with sub_tab_acumulada:
            st.subheader("Precipitación Acumulada Anual")

            if not df_anual_rich.empty:
                df_acumulada = df_anual_rich.groupby([Config.YEAR_COL,
                                                      Config.STATION_NAME_COL]).agg(
                    precipitation_sum=(Config.PRECIPITATION_COL, 'sum'),
                    municipio=(Config.MUNICIPALITY_COL, 'first'),
                    altitud=(Config.ALTITUDE_COL, 'first')
                ).reset_index()

                fig_acumulada = px.bar(
                    df_acumulada,
                    x=Config.YEAR_COL,
                    y='precipitation_sum',
                    color=Config.STATION_NAME_COL,
                    title=f'Precipitación Acumulada por Año ({year_min} - {year_max})',
                    labels={Config.YEAR_COL: 'Año', 'precipitation_sum': 'Precipitación Acumulada (mm)'},
                    hover_data=['municipio', 'altitud']
                )

                fig_acumulada.update_layout(barmode='group', height=500)
                st.plotly_chart(fig_acumulada, use_container_width=True)

            else:
                st.info("No hay datos para calcular la precipitación acumulada.")

        with sub_tab_altitud:
            st.subheader("Relación entre Altitud y Precipitación")

            if not df_anual_rich.empty and not df_anual_rich[Config.ALTITUDE_COL].isnull().all():
                df_relacion = \
                    df_anual_rich.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean().reset_index()

                df_relacion = df_relacion.merge(
                    gdf_filtered[[Config.STATION_NAME_COL, Config.ALTITUDE_COL,
                                  Config.MUNICIPALITY_COL]].drop_duplicates(),
                    on=Config.STATION_NAME_COL,
                    how='left'
                )

                fig_relacion = px.scatter(
                    df_relacion,
                    x=Config.ALTITUDE_COL,
                    y=Config.PRECIPITATION_COL,
                    color=Config.STATION_NAME_COL,
                    title='Relación entre Precipitación Media Anual y Altitud',
                    labels={Config.ALTITUDE_COL: 'Altitud (m)', Config.PRECIPITATION_COL: 'Precipitación Media Anual (mm)'},
                    hover_data=[Config.MUNICIPALITY_COL]
                )

                fig_relacion.update_layout(height=500)
                st.plotly_chart(fig_relacion, use_container_width=True)

            else:
                st.info("No hay datos de altitud o precipitación disponibles para analizar la relación.")

        with sub_tab_regional:
            st.subheader("Serie de Tiempo Promedio Regional (Múltiples Estaciones)")

            if not stations_for_analysis:
                st.warning("Seleccione una o más estaciones en el panel lateral para calcular la serie regional.")

            elif df_monthly_rich.empty:
                st.warning("No hay datos mensuales para las estaciones seleccionadas para calcular la serie regional.")

            else:
                with st.spinner("Calculando serie de tiempo regional..."):
                    df_regional_avg = \
                        df_monthly_rich.groupby(Config.DATE_COL)[Config.PRECIPITATION_COL].mean().reset_index()

                    df_regional_avg.rename(columns={Config.PRECIPITATION_COL: 'Precipitación Promedio'}, inplace=True)

                    show_individual = st.checkbox("Superponer estaciones individuales",
                                                  value=False) if len(stations_for_analysis) > 1 else False

                    fig_regional = go.Figure()

                    if show_individual and len(stations_for_analysis) <= 10:
                        for station in stations_for_analysis:
                            df_s = df_monthly_rich[df_monthly_rich[Config.STATION_NAME_COL] == station]
                            fig_regional.add_trace(go.Scatter(
                                x=df_s[Config.DATE_COL],
                                y=df_s[Config.PRECIPITATION_COL],
                                mode='lines',
                                name=station,
                                line=dict(color='rgba(128, 128, 128, 0.5)', width=1.5),
                                showlegend=True
                            ))

                    elif show_individual:
                        st.info("Demasiadas estaciones seleccionadas para superponer (>10). Mostrando solo el promedio regional.")

                    fig_regional.add_trace(go.Scatter(
                        x=df_regional_avg[Config.DATE_COL],
                        y=df_regional_avg['Precipitación Promedio'],
                        mode='lines',
                        name='Promedio Regional',
                        line=dict(color='#1f77b4', width=3)
                    ))

                    fig_regional.update_layout(
                        title=f'Serie de Tiempo Promedio Regional ({len(stations_for_analysis)} Estaciones)',
                        xaxis_title="Fecha",
                        yaxis_title="Precipitación Mensual (mm)",
                        height=550
                    )

                    st.plotly_chart(fig_regional, use_container_width=True)

                    with st.expander("Ver Datos de la Serie Regional Promedio"):
                        df_regional_avg_display = df_regional_avg.rename(columns={'Precipitación Promedio': 'Precipitación Promedio Regional (mm)'})
                        st.dataframe(df_regional_avg_display.round(1), use_container_width=True)

        with sub_tab_descarga:
            st.subheader("Descargar Datos de la Pestaña Gráficos")
            st.markdown("Descarga los datos procesados y enriquecidos utilizados en esta pestaña de visualización.")

            @st.cache_data
            def convert_df_to_csv_local(df):
                return df.to_csv(index=False, sep=';').encode('utf-8')

            if not df_anual_rich.empty:
                st.markdown("#### Datos Anuales")
                csv_anual = convert_df_to_csv_local(df_anual_rich)
                st.download_button(
                    label="Descargar CSV Anual",
                    data=csv_anual,
                    file_name='datos_graficos_anual.csv',
                    mime='text/csv',
                    key='dl_anual_graphs'
                )
            else:
                st.info("No hay datos anuales para descargar.")

            if not df_monthly_rich.empty:
                st.markdown("#### Datos Mensuales")
                csv_mensual = convert_df_to_csv_local(df_monthly_rich)
                st.download_button(
                    label="Descargar CSV Mensual",
                    data=csv_mensual,
                    file_name='datos_graficos_mensual.csv',
                    mime='text/csv',
                    key='dl_monthly_graphs'
                )
            else:
                st.info("No hay datos mensuales para descargar.")

def create_interpolation_surface(year, method, variogram_model, bounds, gdf_metadata, df_anual):
    """Crea una superficie de interpolación robusta, con etiquetas y suavizado, diferenciando entre IDW y Spline."""
    try:
        points_year = df_anual[df_anual[Config.YEAR_COL] == year]
        if points_year.empty or points_year[Config.PRECIPITATION_COL].isnull().all():
            return None, None, f"No hay datos para el año {year}."
        
        merged_data = pd.merge(gdf_metadata, points_year, on=Config.STATION_NAME_COL)
        coords = np.array(merged_data[[Config.LONGITUDE_COL, Config.LATITUDE_COL]])
        values = merged_data[Config.PRECIPITATION_COL].values
        
        if len(values) < 4:
            return None, None, "Se necesitan al menos 4 estaciones."

        grid_lon = np.linspace(bounds[0], bounds[2], 200)
        grid_lat = np.linspace(bounds[1], bounds[3], 200)
        grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)
        fig_variogram = None

        if "Kriging" in method:
            bin_center, gamma = gs.vario_estimate(coords.T, values)
            model_class = {'linear': gs.Linear, 'spherical': gs.Spherical, 'exponential': gs.Exponential, 'gaussian': gs.Gaussian}.get(variogram_model, gs.Spherical)
            model = model_class(dim=2)
            model.fit_variogram(bin_center, gamma, nugget=True)
            kriging = gs.krige.Ordinary(model, cond_pos=coords.T, cond_val=values)
            grid_z, variance = kriging.structured((grid_lon, grid_lat))
            rmse = np.sqrt(np.mean(variance))
            ax = model.plot(x_max=max(bin_center))
            fig_variogram = ax.get_figure()
            fig_variogram.set_size_inches(6, 4)
        else:
            # Lógica diferenciada para IDW ('linear') y Spline ('cubic')
            interp_method = 'cubic' if "Spline" in method else 'linear'
            
            try:
                grid_z = griddata(coords, values, (grid_x, grid_y), method=interp_method)
            except Exception:
                grid_z = griddata(coords, values, (grid_x, grid_y), method='nearest') # Fallback

            nan_mask = np.isnan(grid_z)
            if np.any(nan_mask):
                fill_values = griddata(coords, values, (grid_x[nan_mask], grid_y[nan_mask]), method='nearest')
                grid_z[nan_mask] = fill_values
            
            predicted_values = griddata(coords, values, coords, method=interp_method)
            rmse = np.sqrt(np.mean((values - predicted_values)**2))
        
        merged_data['hover_text'] = merged_data.apply(
            lambda row: f"<b>{row[Config.STATION_NAME_COL]}</b><br>"
                        f"Municipio: {row[Config.MUNICIPALITY_COL]}<br>"
                        f"Precipitación: {row[Config.PRECIPITATION_COL]:.0f} mm",
            axis=1
        )
        
        fig = go.Figure(data=go.Contour(
            z=grid_z.T, x=grid_lon, y=grid_lat,
            colorscale='viridis',
            colorbar=dict(title='Precipitación (mm)'),
            # --- MEJORA 1: ETIQUETAS EN ISOLÍNEAS ---
            contours=dict(
                coloring='heatmap',
                showlabels=True,  # <-- Muestra las etiquetas
                labelfont=dict(size=10, color='white') # Estilo de la etiqueta
            ),
            # --- MEJORA 2: SUAVIZADO DE LÍNEAS ---
            line_smoothing=0.85, # <-- Suaviza las isolíneas
            line_color='black',
            line_width=0.5
        ))
        
        fig.add_trace(go.Scatter(
            x=coords[:, 0], y=coords[:, 1], mode='markers',
            marker=dict(color='red', size=5),
            hoverinfo='text', hovertext=merged_data['hover_text'],
            showlegend=False
        ))
        
        fig.update_layout(
            title=f"Precipitación en {year} ({method})",
            xaxis_title="Longitud", yaxis_title="Latitud",
            annotations=[dict(x=0.05, y=0.95, xref='paper', yref='paper',
                              text=f'RMSE: {rmse:.1f} mm', showarrow=False,
                              font=dict(size=12, color="black"),
                              bgcolor="yellow", opacity=0.8)]
        )
        return fig, fig_variogram, None
    except Exception as e:
        import traceback
        return None, None, f"Error en la interpolación: {e}\n{traceback.format_exc()}"

@st.cache_data
def create_climate_risk_map(df_anual, _gdf_filtered):
    """
    Calcula y visualiza un mapa de riesgo por variabilidad climática basado en tendencias.
    """
    with st.spinner("Calculando tendencias para todas las estaciones..."):
        # Llama a la función que calcula la tendencia para cada estación
        gdf_trends = calculate_all_station_trends(df_anual, _gdf_filtered)

    if gdf_trends.empty or len(gdf_trends) < 4:
        st.warning("No hay suficientes estaciones con datos de tendencia (>10 años) para generar un mapa de riesgo.")
        return None

    # Prepara los datos para la interpolación (coordenadas y valores de pendiente)
    coords = np.array(gdf_trends.geometry.apply(lambda p: (p.x, p.y)).tolist())
    values = gdf_trends['slope_sen'].values
    
    # Crea la grilla de interpolación
    bounds = _gdf_filtered.total_bounds
    grid_lon = np.linspace(bounds[0], bounds[2], 200)
    grid_lat = np.linspace(bounds[1], bounds[3], 200)
    grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)

    # Interpola la Pendiente de Sen para crear una superficie continua
    grid_z = griddata(coords, values, (grid_x, grid_y), method='cubic')
    
    # Rellena los vacíos en los bordes para un mapa completo
    nan_mask = np.isnan(grid_z)
    if np.any(nan_mask):
        fill_values = griddata(coords, values, (grid_x[nan_mask], grid_y[nan_mask]), method='nearest')
        grid_z[nan_mask] = fill_values

    # Crea el mapa de contorno con Plotly
    fig = go.Figure(data=go.Contour(
        z=grid_z.T, 
        x=grid_lon,
        y=grid_lat,
        colorscale='RdBu', # Escala de color Rojo-Azul para tendencias negativas/positivas
        colorbar=dict(title='Tendencia (mm/año)'),
        contours=dict(coloring='heatmap', showlabels=True, labelfont=dict(size=10, color='black')),
        line_smoothing=0.85
    ))

    # Añade los puntos de las estaciones con sus datos de tendencia
    fig.add_trace(go.Scatter(
            x=coords[:, 0], y=coords[:, 1], mode='markers',
            marker=dict(color='black', size=5, symbol='circle-open'),
            hoverinfo='text',
            hovertext=gdf_trends.apply(lambda row:
                                       f"<b>Estación: {row[Config.STATION_NAME_COL]}</b><br><br>"
                                       f"Municipio: {row.get(Config.MUNICIPALITY_COL, 'N/A')}<br>" # Added Municipio
                                       f"Altitud: {row.get(Config.ALTITUDE_COL, 'N/A'):.0f} m<br>" # Added Altitud
                                       f"Tendencia: {row['slope_sen']:.2f} mm/año<br>"
                                       f"Significancia (p-valor): {row['p_value']:.3f}", 
                                       axis=1),
            name='Estaciones con Tendencia' # Esta línea (1291) está bien indentada
        ))

    fig.update_layout(
        title="Mapa de Riesgo por Tendencias de Precipitación",
        xaxis_title="Longitud",
        yaxis_title="Latitud"
    )
    return fig

def create_hypsometric_figure_and_data(basin_gdf, dem_file_uploader):
    """
    Calcula los datos de la curva hipsométrica y genera la figura de Plotly.
    También prepara los datos para la descarga en formato CSV.
    """
    if basin_gdf is None or dem_file_uploader is None:
        return None, None

    # Guardar temporalmente el DEM para poder leerlo
    dem_path = os.path.join(os.getcwd(), dem_file_uploader.name)
    with open(dem_path, "wb") as f:
        f.write(dem_file_uploader.getbuffer())

    hypsometric_data = calculate_hypsometric_curve(basin_gdf, dem_path)
    os.remove(dem_path) # Limpiar el archivo temporal

    if hypsometric_data.get("error"):
        st.error(hypsometric_data["error"])
        return None, None

    # Crear la figura
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hypsometric_data['cumulative_area_percent'],
        y=hypsometric_data['elevations'],
        mode='lines',
        fill='tozeroy' # Rellena el área bajo la curva
    ))
    fig.update_layout(
        title="Curva Hipsométrica de la Cuenca Agregada",
        xaxis_title="Área Acumulada sobre la Elevación (%)",
        yaxis_title="Elevación (m)",
        xaxis=dict(range=[0, 100]), # Eje X de 0 a 100%
        template="plotly_white"
    )
    
    # Preparar los datos para el CSV
    df_hypsometric = pd.DataFrame({
        'Elevacion_m': hypsometric_data['elevations'],
        'Porcentaje_Area_Acumulada': hypsometric_data['cumulative_area_percent']
    })
    csv_data = df_hypsometric.to_csv(index=False).encode('utf-8')
    
    return fig, csv_data
    
def display_advanced_maps_tab(gdf_filtered, stations_for_analysis, df_anual_melted,
                             df_monthly_filtered, analysis_mode, selected_regions, selected_municipios,
                             selected_altitudes, **kwargs):
    st.header("Mapas Avanzados")
    display_filter_summary(
        total_stations_count=len(st.session_state.get('gdf_stations', [])),
        selected_stations_count=len(stations_for_analysis),
        year_range=st.session_state.get('year_range', (2000, 2020)),
        selected_months_count=len(st.session_state.get('meses_numeros', [])),
        analysis_mode=analysis_mode,
        selected_regions=selected_regions,
        selected_municipios=selected_municipios,
        selected_altitudes=selected_altitudes
    )

    if not stations_for_analysis:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return

    tab_names = [
        "Animación GIF", "Superficies de Interpolación", "Morfometría",
        "Mapa de Riesgo Climático", "Validación Cruzada (LOOCV)",
        "Visualización Temporal", "Gráfico de Carrera", "Mapa Animado", "Comparación de Mapas"
    ]
    gif_tab, kriging_tab, morph_tab, risk_map_tab, validation_tab, temporal_tab, race_tab, anim_tab, compare_tab = st.tabs(tab_names)

    # Variables de estado para compartir resultados entre pestañas
    session_state_vars = ['run_balance', 'fig_basin', 'error_msg', 'mean_precip', 'morph_results', 'unified_basin_gdf', 'selected_basins_title', 'balance_results']
    for var in session_state_vars:
        if var not in st.session_state:
            st.session_state[var] = None

    with gif_tab:
        st.subheader("Distribución Espacio-Temporal de la Lluvia en Antioquia")
        col_controls, col_gif = st.columns([1, 3])
        with col_controls:
            if st.button("Reiniciar Animación", key="reset_gif_button"):
                st.rerun()
        with col_gif:
            gif_path = Config.GIF_PATH
            if os.path.exists(gif_path):
                try:
                    with open(gif_path, "rb") as f:
                        gif_bytes = f.read()
                    gif_b64 = base64.b64encode(gif_bytes).decode("utf-8")
                    html_string = f'<img src="data:image/gif;base64,{gif_b64}" width="600" alt="Animación PPAM">'
                    st.markdown(html_string, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Ocurrió un error al intentar mostrar el GIF: {e}")
            else:
                st.error(f"No se pudo encontrar el archivo GIF en la ruta especificada: {gif_path}")

    with kriging_tab:
        st.subheader("Superficies de Interpolación de Precipitación Anual")
        analysis_mode_interp = st.radio(
            "Seleccione el modo de interpolación:",
            ("Regional (Toda la selección)", "Por Cuenca Específica"),
            key="interp_mode_radio", horizontal=True,
            help="El modo 'Por Cuenca' permite un análisis detallado con buffer y DEM."
        )
        st.markdown("---")
        
        if analysis_mode_interp == "Por Cuenca Específica":
            if 'gdf_subcuencas' not in st.session_state or st.session_state.gdf_subcuencas is None:
                st.warning("Los datos de cuencas no están disponibles.")
                st.stop()
                
            BASIN_NAME_COLUMN = 'SUBC_LBL'
            if BASIN_NAME_COLUMN not in st.session_state.gdf_subcuencas.columns:
                st.error(f"La columna '{BASIN_NAME_COLUMN}' no se encontró en los datos de cuencas.")
                st.stop()

            col_control, col_display = st.columns([1, 2])
            
            with col_control:
                st.markdown("#### Controles de Cuenca")
                
                if not gdf_filtered.empty:
                    relevant_regions = gdf_filtered[Config.REGION_COL].unique()
                    if Config.REGION_COL in st.session_state.gdf_subcuencas.columns:
                        relevant_basins = st.session_state.gdf_subcuencas[st.session_state.gdf_subcuencas[Config.REGION_COL].isin(relevant_regions)]
                        basin_names = sorted(relevant_basins[BASIN_NAME_COLUMN].dropna().unique())
                    else:
                        st.warning(f"El archivo de cuencas no tiene columna '{Config.REGION_COL}'.")
                        basin_names = sorted(st.session_state.gdf_subcuencas[BASIN_NAME_COLUMN].dropna().unique())
                else:
                    basin_names = sorted(st.session_state.gdf_subcuencas[BASIN_NAME_COLUMN].dropna().unique())
                
                selected_basins = st.multiselect("Seleccione una o más subcuencas:", options=basin_names, key="basin_multiselect")
                buffer_km = st.slider("Buffer de influencia (km):", 0, 50, 10, 5, key="buffer_slider")
                df_anual_non_na = df_anual_melted.dropna(subset=[Config.PRECIPITATION_COL])
                years = sorted(df_anual_non_na[Config.YEAR_COL].unique()) if not df_anual_non_na.empty else []
                
                if not years:
                    st.warning("No hay datos anuales disponibles para la interpolación.")
                else:
                    selected_year = st.selectbox("Seleccione un año:", options=years, index=len(years) - 1, key="year_select_basin")
                    method = st.selectbox("Método de interpolación:", options=["IDW (Lineal)", "Spline (Cúbico)"], key="interp_method_basin")
                    run_balance = st.toggle("Calcular Balance Hídrico", value=True)
                    show_dem_background = st.toggle("Visualizar DEM de fondo", value=True)
                    
                    # El DEM se carga desde el panel lateral (manteniendo la lógica del original)
                    dem_file = st.session_state.get('dem_file')

                    if st.button("Generar Mapa para Cuenca(s)", disabled=not selected_basins, key="generate_basin_map_button"):
                        # Reiniciar estados
                        st.session_state['run_balance'] = run_balance
                        st.session_state['fig_basin'] = None
                        st.session_state['error_msg'] = None
                        st.session_state['mean_precip'] = None
                        st.session_state['morph_results'] = None
                        
                        dem_path = None
                        
                        try:
                            # --- MANEJO DE ARCHIVO DEM (SI ESTÁ PRESENTE) ---
                            if show_dem_background and dem_file is not None:
                                dem_path = os.path.join(os.getcwd(), dem_file.name)
                                with open(dem_path, "wb") as f: f.write(dem_file.getbuffer())

                            with st.spinner("Preparando datos y realizando interpolación..."):
                                target_basins_gdf = st.session_state.gdf_subcuencas[st.session_state.gdf_subcuencas[BASIN_NAME_COLUMN].isin(selected_basins)]
                                unified_basin_gdf = gpd.GeoDataFrame(geometry=[target_basins_gdf.unary_union], crs=target_basins_gdf.crs)
                                target_basin_metric = unified_basin_gdf.to_crs("EPSG:3116")
                                basin_buffer_metric = target_basin_metric.buffer(buffer_km * 1000)
                                stations_metric = st.session_state.gdf_stations.to_crs("EPSG:3116")
                                stations_in_buffer = stations_metric[stations_metric.intersects(basin_buffer_metric.unary_union)]
                                station_names = stations_in_buffer[Config.STATION_NAME_COL].unique()
                                precip_data_year = df_anual_non_na[(df_anual_non_na[Config.YEAR_COL] == selected_year) & (df_anual_non_na[Config.STATION_NAME_COL].isin(station_names))]
                                cols_to_merge = [Config.STATION_NAME_COL, 'geometry', Config.MUNICIPALITY_COL, Config.ALTITUDE_COL]
                                points_data = gpd.GeoDataFrame(pd.merge(
                                    stations_in_buffer[cols_to_merge],
                                    precip_data_year[[Config.STATION_NAME_COL, Config.PRECIPITATION_COL]],
                                    on=Config.STATION_NAME_COL
                                )).dropna(subset=[Config.PRECIPITATION_COL])
                                points_data.rename(columns={Config.PRECIPITATION_COL: 'Valor'}, inplace=True)
                                
                                bounds = basin_buffer_metric.unary_union.bounds
                                grid_resolution = 500
                                grid_lon, grid_lat = np.arange(bounds[0], bounds[2], grid_resolution), np.arange(bounds[1], bounds[3], grid_resolution)
                                grid_z = None

                                if len(points_data) > 2:
                                    points = np.column_stack((points_data.geometry.x, points_data.geometry.y))
                                    values = points_data['Valor'].values
                                    grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)
                                    interp_method_call = 'linear' if method == "IDW (Lineal)" else 'cubic'
                                    grid_z = griddata(points, values, (grid_x, grid_y), method=interp_method_call)
                                    nan_mask = np.isnan(grid_z)
                                    if np.any(nan_mask):
                                        # Usar el método 'nearest' para rellenar los valores NaN de la interpolación.
                                        fill_values = griddata(points, values, (grid_x[nan_mask], grid_y[nan_mask]), method='nearest')
                                        grid_z[nan_mask] = fill_values
                                    grid_z = np.nan_to_num(grid_z) # Convertir cualquier NaN restante a 0
                                else:
                                    st.session_state['error_msg'] = "Se necesitan al menos 3 estaciones para la interpolación."
                                
                                if grid_z is not None:
                                    grid_z[grid_z < 0] = 0
                                    transform = from_origin(grid_lon[0], grid_lat[-1], grid_resolution, grid_resolution)
                                    with rasterio.io.MemoryFile() as memfile:
                                        with memfile.open(driver='GTiff', height=len(grid_lat), width=len(grid_lon), count=1, dtype=str(grid_z.dtype), crs="EPSG:3116", transform=transform) as dataset:
                                            dataset.write(np.flipud(grid_z), 1)
                                        with memfile.open() as dataset:
                                            masked_data, masked_transform = mask(dataset, target_basin_metric.geometry, crop=True, nodata=np.nan)
                                    masked_data = masked_data[0].astype(np.float32)
                                    mean_precip = np.nanmean(masked_data)
                                    
                                    map_traces = []
                                    dem_trace = None
                                    x_coords = np.arange(masked_transform.c, masked_transform.c + masked_data.shape[1] * masked_transform.a, masked_transform.a)
                                    y_coords = np.arange(masked_transform.f, masked_transform.f + masked_data.shape[0] * masked_transform.e, masked_transform.e)

                                    if dem_path and show_dem_background:
                                        with st.spinner("Procesando y reproyectando DEM..."):
                                            try:
                                                with rasterio.open(dem_path) as dem_src:
                                                    dem_reprojected, _ = reproject(
                                                        source=rasterio.band(dem_src, 1),
                                                        destination=np.empty_like(masked_data, dtype=np.float32),
                                                        src_transform=dem_src.transform, src_crs=dem_src.crs,
                                                        dst_transform=masked_transform, dst_crs="EPSG:3116",
                                                        resampling=Resampling.bilinear
                                                    )
                                                    dem_trace = go.Heatmap(z=dem_reprojected, x=x_coords, y=y_coords, colorscale='gray', showscale=False, name='Elevación')
                                                    map_traces.append(dem_trace)
                                            except Exception as e:
                                                st.warning(f"No se pudo procesar el DEM: {e}")
                                    
                                    precip_trace = go.Contour(
                                        z=masked_data, x=x_coords, y=y_coords, colorscale='viridis',
                                        contours=dict(coloring='heatmap', showlabels=True, labelfont=dict(size=10, color='white')),
                                        line=dict(color='black', width=0.5), colorbar=dict(title='Precipitación (mm)'),
                                        opacity=0.7 if dem_trace is not None else 1.0, name='Precipitación'
                                    )
                                    map_traces.append(precip_trace)
                                    fig_basin = go.Figure(data=map_traces)

                                    points_data['hover_text'] = points_data.apply(lambda row: f"<b>{row[Config.STATION_NAME_COL]}</b><br>Municipio: {row[Config.MUNICIPALITY_COL]}<br>Altitud: {row[Config.ALTITUDE_COL]:.0f} m<br>Precipitación: {row['Valor']:.0f} mm", axis=1)
                                    fig_basin.add_trace(go.Scatter(
                                        x=points_data.geometry.x, y=points_data.geometry.y, mode='markers',
                                        marker=dict(color='black', size=5, symbol='circle-open'),
                                        name='Estaciones', hoverinfo='text', hovertext=points_data['hover_text']
                                    ))
                                    fig_basin.update_layout(
                                        title=f"Precipitación Interpolada ({method}) para Cuenca(s) ({selected_year})",
                                        xaxis_title="Coordenada Este (m)", yaxis=dict(title="Coordenada Norte (m)", scaleanchor='x', scaleratio=1)
                                    )
                                    st.session_state['fig_basin'] = fig_basin
                                    st.session_state['mean_precip'] = mean_precip
                                    st.session_state['unified_basin_gdf'] = unified_basin_gdf
                                    st.session_state['selected_basins_title'] = ", ".join(selected_basins)
                                    
                                    # CÁLCULO DE MORFOMETRÍA
                                    if dem_path:
                                        st.session_state['morph_results'] = calculate_morphometry(unified_basin_gdf, dem_path)

                                else:
                                    if not st.session_state.get('error_msg'):
                                        st.session_state['error_msg'] = "La interpolación no generó resultados."
                            
                        except Exception as e:
                            import traceback
                            st.session_state['error_msg'] = f"Ocurrió un error crítico: {e}\n\n{traceback.format_exc()}"
                        
                        finally:
                            # --- ELIMINACIÓN DE ARCHIVO DEM TEMPORAL ---
                            if dem_path and os.path.exists(dem_path): 
                                os.remove(dem_path)
                                
            with col_display:
                fig_basin = st.session_state.get('fig_basin')
                error_msg = st.session_state.get('error_msg')
                mean_precip = st.session_state.get('mean_precip')
                
                # Obtener resultados de la sesión
                unified_basin_gdf = st.session_state.get('unified_basin_gdf')
                morph_results = st.session_state.get('morph_results')
                run_balance = st.session_state.get('run_balance')
                
                if error_msg:
                    st.error(error_msg)
                
                if fig_basin:
                    st.subheader(f"Resultados para: {st.session_state.get('selected_basins_title', '')}")
                    st.plotly_chart(fig_basin, use_container_width=True)

                # Mostramos el Balance Hídrico
                if mean_precip is not None and unified_basin_gdf is not None and run_balance:
                    st.markdown("---")
                    st.subheader("Balance Hídrico Estimado")
                    
                    # Extraemos la altitud promedio de los resultados de morfometría
                    alt_prom = morph_results.get('alt_prom_m') if morph_results else None
                    
                    with st.spinner("Calculando balance..."):
                        balance_results = calculate_hydrological_balance(mean_precip, alt_prom, unified_basin_gdf)
                        st.session_state['balance_results'] = balance_results
                        if balance_results.get("error"):
                            st.error(balance_results["error"])
                        else:
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Precipitación Media (P)", f"{balance_results['P_media_anual_mm']:.0f} mm/año")
                            c2.metric("Altitud Media", f"{balance_results['Altitud_media_m']:.0f} m" if balance_results['Altitud_media_m'] is not None else "N/A")
                            c3.metric("ET Media Estimada (ET)", f"{balance_results['ET_media_anual_mm']:.0f} mm/año" if balance_results['ET_media_anual_mm'] is not None else "N/A")
                            c4.metric("Escorrentía (Q=P-ET)", f"{balance_results['Q_mm']:.0f} mm/año" if balance_results['Q_mm'] is not None else "N/A")
                            st.success(f"Volumen de escorrentía anual estimado: **{balance_results['Q_m3_año']/1e6:.2f} millones de m³** sobre un área de **{balance_results['Area_km2']:.2f} km²**.")
                
                # Mostramos la Morfometría (si fue calculada)
                if morph_results:
                    st.markdown("---")
                    st.subheader("Morfometría de la Cuenca (Calculada con DEM)")
                    if morph_results.get("error"):
                        st.error(morph_results["error"])
                    else:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Área", f"{morph_results['area_km2']:.2f} km²")
                        c2.metric("Perímetro", f"{morph_results['perimetro_km']:.2f} km")
                        c3.metric("Índice de Forma", f"{morph_results['indice_forma']:.2f}")
                        c4, c5, c6 = st.columns(3)
                        c4.metric("Altitud Máxima", f"{morph_results.get('alt_max_m', 'N/A'):.0f} m")
                        c5.metric("Altitud Mínima", f"{morph_results.get('alt_min_m', 'N/A'):.0f} m")
                        c6.metric("Altitud Promedio", f"{morph_results.get('alt_prom_m', 'N/A'):.1f} m")
                
                elif st.session_state.get('dem_file') is None and run_balance:
                    st.info("Para ver el Balance Hídrico completo y la Morfometría, suba un archivo DEM en el panel lateral.")
                    
        else: # MODO REGIONAL
            df_anual_non_na = df_anual_melted.dropna(subset=[Config.PRECIPITATION_COL])
            if len(stations_for_analysis) < 3 or df_anual_non_na.empty:
                st.warning("No hay suficientes datos anuales o estaciones (mínimo 3) para realizar la interpolación regional.")
            else:
                min_year, max_year = int(df_anual_non_na[Config.YEAR_COL].min()), int(df_anual_non_na[Config.YEAR_COL].max())
                control_col, map_col1, map_col2 = st.columns([1, 2, 2])
                with control_col:
                    st.markdown("#### Controles de los Mapas")
                    interpolation_methods = ["Kriging Ordinario", "IDW", "Spline (Thin Plate)"]
                    st.markdown("**Mapa 1**")
                    year1 = st.slider("Seleccione el año", min_year, max_year, max_year, key="interp_year1")
                    method1 = st.selectbox("Método de interpolación", options=interpolation_methods, key="interp_method1")
                    variogram_model1 = None
                    if "Kriging" in method1:
                        variogram_options = ['linear', 'spherical', 'exponential', 'gaussian']
                        variogram_model1 = st.selectbox("Modelo de Variograma para Mapa 1", variogram_options, key="var_model_1")
                    st.markdown("---")
                    st.markdown("**Mapa 2**")
                    year2 = st.slider("Seleccione el año", min_year, max_year, max_year - 1 if max_year > min_year else max_year, key="interp_year2")
                    method2 = st.selectbox("Método de interpolación", options=interpolation_methods, index=1, key="interp_method2")
                    variogram_model2 = None
                    if "Kriging" in method2:
                        variogram_options = ['linear', 'spherical', 'exponential', 'gaussian']
                        variogram_model2 = st.selectbox("Modelo de Variograma para Mapa 2", variogram_options, key="var_model_2")
                
                gdf_bounds = gdf_filtered.total_bounds.tolist()
                gdf_metadata = pd.DataFrame(gdf_filtered.drop(columns='geometry', errors='ignore'))
                
                # Asumiendo que create_interpolation_surface maneja la reproyección/CRS a EPSG:3116 si es necesario
                fig1, fig_var1, error1 = create_interpolation_surface(year1, method1, variogram_model1, gdf_bounds, gdf_metadata, df_anual_non_na)
                fig2, fig_var2, error2 = create_interpolation_surface(year2, method2, variogram_model2, gdf_bounds, gdf_metadata, df_anual_non_na)

                with map_col1:
                    if fig1: st.plotly_chart(fig1, use_container_width=True)
                    else: st.info(error1)
                with map_col2:
                    if fig2: st.plotly_chart(fig2, use_container_width=True)
                    else: st.info(error2)
                st.markdown("---")
                st.markdown("##### Variogramas de los Mapas")
                col3, col4 = st.columns(2)
                with col3:
                    if fig_var1:
                        buf = io.BytesIO()
                        fig_var1.savefig(buf, format="png")
                        st.image(buf)
                        st.download_button(label="Descargar Variograma 1 (PNG)", data=buf.getvalue(), file_name=f"variograma_1_{year1}_{method1}.png", mime="image/png")
                        # Es importante cerrar la figura de matplotlib
                        # La importación de plt se asume en el archivo principal
                        # plt.close(fig_var1) 
                    else:
                        st.info("El variograma no está disponible para este método.")
                with col4:
                    if fig_var2:
                        buf = io.BytesIO()
                        fig_var2.savefig(buf, format="png")
                        st.image(buf)
                        st.download_button(label="Descargar Variograma 2 (PNG)", data=buf.getvalue(), file_name=f"variograma_2_{year2}_{method2}.png", mime="image/png")
                        # plt.close(fig_var2)
                    else:
                        st.info("El variograma no está disponible para este método.")

    with morph_tab:
        st.subheader("Análisis Morfométrico de Cuencas")
        st.info("Esta sección requiere que se haya generado un mapa para una cuenca en la pestaña **'Superficies de Interpolación'** y que se haya subido un archivo DEM en el panel lateral.")
        
        # Acceder a los resultados ya calculados en kriging_tab para evitar recalcular
        unified_basin_gdf = st.session_state.get('unified_basin_gdf')
        morph_results = st.session_state.get('morph_results')
        dem_file_from_sidebar = st.session_state.get('dem_file')
        
        # --- LÓGICA DE VISUALIZACIÓN SIMPLIFICADA (SOLO MUESTRA) ---
        if unified_basin_gdf is not None and dem_file_from_sidebar is not None and morph_results is not None:
            
            st.markdown(f"### Resultados para: **{st.session_state.get('selected_basins_title', '')}**")
            
            # Regenerar el path del DEM para usarlo en la curva hipsométrica
            dem_path = os.path.join(os.getcwd(), dem_file_from_sidebar.name)
            
            try:
                # Escribir el archivo DEM si es necesario para el cálculo de la curva hipsométrica
                # Se asume que morph_results se calculó exitosamente antes.
                with open(dem_path, "wb") as f: f.write(dem_file_from_sidebar.getbuffer())

                st.markdown("#### Parámetros Morfométricos")
                if morph_results.get("error"):
                    st.error(morph_results["error"])
                else:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Área", f"{morph_results['area_km2']:.2f} km²")
                    c2.metric("Perímetro", f"{morph_results['perimetro_km']:.2f} km")
                    c3.metric("Índice de Forma", f"{morph_results['indice_forma']:.2f}")
                    c4, c5, c6 = st.columns(3)
                    c4.metric("Altitud Máxima", f"{morph_results.get('alt_max_m', 'N/A'):.0f} m")
                    c5.metric("Altitud Mínima", f"{morph_results.get('alt_min_m', 'N/A'):.0f} m")
                    c6.metric("Altitud Promedio", f"{morph_results.get('alt_prom_m', 'N/A'):.1f} m")

                st.markdown("---")
                st.markdown("#### Curva Hipsométrica")
                with st.spinner("Calculando curva hipsométrica y ajuste polinomial..."):
                    hypsometric_data = calculate_hypsometric_curve(unified_basin_gdf, dem_path)
                
                if hypsometric_data.get("error"):
                    st.error(hypsometric_data["error"])
                else:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hypsometric_data['cumulative_area_percent'],
                        y=hypsometric_data['elevations'],
                        mode='lines',
                        fill='tozeroy',
                        name='Curva Hipsométrica (Datos DEM)',
                        opacity=0.7
                    ))
                    fig.add_trace(go.Scatter(
                        x=hypsometric_data['fit_x'],
                        y=hypsometric_data['fit_y'],
                        mode='lines',
                        name='Ajuste Polinomial (Grado 3)',
                        line=dict(color='red', dash='dash')
                    ))
                    fig.update_layout(
                        title="Curva Hipsométrica de la Cuenca Agregada",
                        xaxis_title="Área Acumulada sobre la Elevación (%)",
                        yaxis_title="Elevación (m)",
                        xaxis=dict(range=[0, 100]),
                        template="plotly_white",
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("##### Ecuaciones Derivadas")
                    st.latex(r'''A(h) = \int_{h}^{H_{max}} f(z) \, dz''')
                    st.caption("Ecuación integral de la curva hipsométrica.")
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Ecuación de Ajuste (x = % Área / 100)", hypsometric_data['equation'])
                    col2.metric("Coeficiente de Determinación (R²)", f"{hypsometric_data['r_squared']:.4f}")
                    st.caption("El R² indica qué tan bien la ecuación polinomial representa la forma de la curva (1 es un ajuste perfecto).")

            except Exception as e:
                st.error(f"Ocurrió un error al procesar el DEM para la curva hipsométrica: {e}")

            finally:
                if os.path.exists(dem_path): os.remove(dem_path)

        else:
            # Mensajes de advertencia más específicos
            if unified_basin_gdf is None:
                st.warning("Debe generar un mapa para una cuenca en 'Superficies de Interpolación' (Modo: Por Cuenca Específica) primero.")
            if dem_file_from_sidebar is None:
                st.warning("Debe subir un archivo DEM (.tif) en el panel lateral para el análisis.")
            if morph_results is None and unified_basin_gdf is not None and dem_file_from_sidebar is not None:
                st.warning("Los parámetros morfométricos no se han calculado. Por favor, haga clic en **'Generar Mapa para Cuenca(s)'** en la pestaña anterior.")

    with risk_map_tab:
        st.subheader("Mapa de Vulnerabilidad por Tendencias de Precipitación a Largo Plazo")
        st.info("""
        Este mapa interpola la tendencia (Pendiente de Sen) de todas las estaciones con datos suficientes (>10 años)
        para crear una superficie de riesgo.
        - **Zonas Azules:** Indican una tendencia a precipitación *creciente* (más mm/año).
        - **Zonas Rojas:** Indican una tendencia a precipitación *decreciente* (menos mm/año).
        """)
        
        # Llamamos a la función que SÍ interpola la superficie
        fig_risk_contour = create_climate_risk_map(df_anual_melted, gdf_filtered)
        
        if fig_risk_contour:
            st.plotly_chart(fig_risk_contour, use_container_width=True)
        else:
            st.warning("No hay suficientes datos de tendencia (>10 años) para generar el mapa.")

    with validation_tab:
        st.subheader("Validación Cruzada Comparativa de Métodos de Interpolación")
        if len(stations_for_analysis) < 4:
            st.warning("Se necesitan al menos 4 estaciones con datos para realizar una validación robusta.")

        else:
            df_anual_non_na = df_anual_melted.dropna(subset=[Config.PRECIPITATION_COL])
            all_years_int = sorted(df_anual_non_na[Config.YEAR_COL].unique())

            if not all_years_int:
                st.warning("No hay años con datos válidos para la validación.")
            else:
                selected_year = st.selectbox("Seleccione un año para la validación:",
                                             options=all_years_int, index=len(all_years_int) - 1, key="validation_year_select")

                if st.button(f"Ejecutar Validación para el año {selected_year}",
                             key="run_validation_button"):

                    with st.spinner("Realizando validación cruzada..."):

                        gdf_metadata = pd.DataFrame(gdf_filtered.drop(columns='geometry',
                                                                     errors='ignore'))

                        validation_results_df = perform_loocv_for_all_methods(selected_year,
                                                                              gdf_metadata, df_anual_non_na)

                    if not validation_results_df.empty:
                        st.subheader(f"Resultados de la Validación para el Año {selected_year}")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Error Cuadrático Medio (RMSE)**")
                            fig_rmse = px.bar(validation_results_df.sort_values("RMSE"),
                                             x="Método", y="RMSE", color="Método", text_auto='.2f')
                            fig_rmse.update_layout(showlegend=False)
                            st.plotly_chart(fig_rmse, use_container_width=True)

                        with col2:
                            st.markdown("**Error Absoluto Medio (MAE)**")
                            fig_mae = px.bar(validation_results_df.sort_values("MAE"), x="Método",
                                             y="MAE", color="Método", text_auto='.2f')
                            fig_mae.update_layout(showlegend=False)
                            st.plotly_chart(fig_mae, use_container_width=True)

                        st.markdown("**Tabla Comparativa de Errores**")
                        st.dataframe(validation_results_df.style.format({"RMSE": "{:.2f}", "MAE":
                                                                        "{:.2f}"}))

                        best_rmse = \
                            validation_results_df.loc[validation_results_df['RMSE'].idxmin()]

                        st.success(f"**Mejor método según RMSE:** {best_rmse['Método']} (RMSE: {best_rmse['RMSE']:.2f})")

                    else:
                        st.error("No se pudieron calcular los resultados de la validación.")

    with temporal_tab:
        st.subheader("Explorador Anual de Precipitación")

        df_anual_melted_non_na = \
            df_anual_melted.dropna(subset=[Config.PRECIPITATION_COL])

        if not df_anual_melted_non_na.empty:
            all_years_int = sorted(df_anual_melted_non_na[Config.YEAR_COL].unique())

            controls_col, map_col = st.columns([1, 3])

            with controls_col:
                st.markdown("##### Opciones de Visualización")
                selected_base_map_config, selected_overlays_config = display_map_controls(st,
                                                                                         "temporal")
                selected_year = None
                if len(all_years_int) > 1:
                    selected_year = st.slider('Seleccione un Año para Explorar',
                                             min_value=min(all_years_int),
                                             max_value=max(all_years_int),
                                             value=min(all_years_int),
                                             key="temporal_year_slider")

                elif len(all_years_int) == 1:
                    selected_year = all_years_int[0]
                    st.info(f"Mostrando único año disponible: {selected_year}")

                if selected_year:
                    st.markdown(f"#### Resumen del Año: {selected_year}")

                    df_year_filtered = \
                        df_anual_melted_non_na[df_anual_melted_non_na[Config.YEAR_COL] == selected_year]

                    if not df_year_filtered.empty:
                        num_stations = len(df_year_filtered)
                        st.metric("Estaciones con Datos", num_stations)

                        if num_stations > 1:
                            st.metric("Promedio Anual",
                                      f"{df_year_filtered[Config.PRECIPITATION_COL].mean():.0f} mm")
                            st.metric("Máximo Anual",
                                      f"{df_year_filtered[Config.PRECIPITATION_COL].max():.0f} mm")
                        else:
                            st.metric("Precipitación Anual",
                                      f"{df_year_filtered[Config.PRECIPITATION_COL].iloc[0]:.0f} mm")

            with map_col:
                if selected_year:
                    # CORRECCIÓN DE ATTRIBUTEERROR: Extraer solo la clave 'tiles' (YA ESTABA EN EL CÓDIGO ORIGINAL, PERO ES VÁLIDA)
                    m_temporal = create_folium_map([4.57, -74.29], 5,
                                                   {'tiles': selected_base_map_config['tiles'], 'attr': selected_base_map_config['attr']}, selected_overlays_config)

                    df_year_filtered = \
                        df_anual_melted_non_na[df_anual_melted_non_na[Config.YEAR_COL] == selected_year]

                    if not df_year_filtered.empty:
                        cols_to_merge = [Config.STATION_NAME_COL, Config.MUNICIPALITY_COL,
                                         Config.ALTITUDE_COL, 'geometry']

                        df_map_data = pd.merge(df_year_filtered,
                                               gdf_filtered[cols_to_merge].drop_duplicates(subset=[Config.STATION_NAME_COL]),
                                               on=Config.STATION_NAME_COL, how="inner")

                        if not df_map_data.empty:
                            min_val, max_val = \
                                df_anual_melted_non_na[Config.PRECIPITATION_COL].min(), \
                                df_anual_melted_non_na[Config.PRECIPITATION_COL].max()

                            if min_val >= max_val: max_val = min_val + 1

                            colormap = cm.LinearColormap(colors=plt.cm.viridis.colors, vmin=min_val,
                                                         vmax=max_val)

                            for _, row in df_map_data.iterrows():
                                popup_object = generate_annual_map_popup_html(row,
                                                                              df_anual_melted_non_na)

                                folium.CircleMarker(
                                    location=[row['geometry'].y, row['geometry'].x], radius=5,
                                    color=colormap(row[Config.PRECIPITATION_COL]), fill=True,
                                    fill_color=colormap(row[Config.PRECIPITATION_COL]), fill_opacity=0.8,
                                    tooltip=row[Config.STATION_NAME_COL], popup=popup_object
                                ).add_to(m_temporal)

                            temp_gdf = gpd.GeoDataFrame(df_map_data, geometry='geometry',
                                                         crs=gdf_filtered.crs)

                            if not temp_gdf.empty:
                                bounds = temp_gdf.total_bounds
                                if np.all(np.isfinite(bounds)):
                                    m_temporal.fit_bounds([[bounds[1], bounds[0]], [bounds[3],
                                                                                    bounds[2]]])

                            folium.LayerControl().add_to(m_temporal)
                            folium_static(m_temporal, height=700, width=None)

    with race_tab:
        st.subheader("Ranking Anual de Precipitación por Estación")
        df_anual_valid = df_anual_melted.dropna(subset=[Config.PRECIPITATION_COL])

        if not df_anual_valid.empty:
            fig_racing = px.bar(
                df_anual_valid,
                x=Config.PRECIPITATION_COL,
                y=Config.STATION_NAME_COL,
                animation_frame=Config.YEAR_COL,
                orientation='h',
                labels={
                    Config.PRECIPITATION_COL: 'Precipitación Anual (mm)',
                    Config.STATION_NAME_COL: 'Estación'
                },
                title="Evolución de Precipitación Anual por Estación"
            )

            # Bloque original: Ajuste de altura dinámico y ordenamiento correcto.
            fig_racing.update_layout(
                height=max(600, len(stations_for_analysis) * 35),
                yaxis=dict(categoryorder='total ascending')
            )

            st.plotly_chart(fig_racing, use_container_width=True)
        else:
            st.warning("No hay suficientes datos anuales con los filtros actuales para generar el Gráfico de Carrera.")

    with anim_tab:
        st.subheader("Mapa Animado de Precipitación Anual")

        df_anual_valid = df_anual_melted.dropna(subset=[Config.PRECIPITATION_COL])

        if not df_anual_valid.empty:
            # Asegurar que se tienen las coordenadas en el gdf_filtered
            gdf_coords = gdf_filtered[[Config.STATION_NAME_COL, Config.LATITUDE_COL, Config.LONGITUDE_COL]].drop_duplicates(subset=[Config.STATION_NAME_COL])
            
            df_anim_merged = pd.merge(
                df_anual_valid,
                gdf_coords, # Usamos el gdf con solo coordenadas
                on=Config.STATION_NAME_COL, how="inner"
            )

            if not df_anim_merged.empty:
                fig_mapa_animado = px.scatter_geo(
                    df_anim_merged,
                    lat=Config.LATITUDE_COL, lon=Config.LONGITUDE_COL,
                    color=Config.PRECIPITATION_COL, size=Config.PRECIPITATION_COL,
                    hover_name=Config.STATION_NAME_COL,
                    animation_frame=Config.YEAR_COL,
                    projection='natural earth',
                    title='Precipitación Anual por Estación',
                    color_continuous_scale=px.colors.sequential.Viridis # Añadir una escala de color
                )

                fig_mapa_animado.update_geos(fitbounds="locations", visible=True)
                st.plotly_chart(fig_mapa_animado, use_container_width=True)

            else:
                st.warning("No se pudieron combinar los datos anuales con la información geográfica de las estaciones.")

        else:
            st.warning("No hay suficientes datos anuales con los filtros actuales para generar el Mapa Animado.")

    with compare_tab:
        st.subheader("Comparación de Mapas Anuales")
        df_anual_valid = df_anual_melted.dropna(subset=[Config.PRECIPITATION_COL])
        all_years = sorted(df_anual_valid[Config.YEAR_COL].unique())

        if len(all_years) > 1:
            control_col, map_col1, map_col2 = st.columns([1, 2, 2])
            with control_col:
                st.markdown("##### Controles de Mapa")
                selected_base_map_config, selected_overlays_config = display_map_controls(st, "compare")
                min_year, max_year = int(all_years[0]), int(all_years[-1])
                
                st.markdown("**Mapa 1**")
                year1 = st.selectbox("Seleccione el primer año", options=all_years, index=len(all_years) - 1, key="compare_year1")
                
                st.markdown("**Mapa 2**")
                year2 = st.selectbox(
                    "Seleccione el segundo año",
                    options=all_years,
                    index=len(all_years) - 2 if len(all_years) > 1 else 0,
                    key="compare_year2"
                )

                min_precip, max_precip = int(df_anual_valid[Config.PRECIPITATION_COL].min()), int(df_anual_valid[Config.PRECIPITATION_COL].max())
                if min_precip >= max_precip: max_precip = min_precip + 1
                color_range = st.slider("Rango de Escala de Color (mm)", min_precip, max_precip, (min_precip, max_precip), key="color_compare")
                colormap = cm.LinearColormap(
                    colors=plt.cm.viridis.colors,
                    vmin=color_range[0],
                    vmax=color_range[1]
                )

            def create_compare_map(data, year, col, gdf_stations_info, df_anual_full):
                col.markdown(f"**Precipitación en {year}**")

                m = create_folium_map([6.24, -75.58], 6, {'tiles': selected_base_map_config['tiles'], 'attr': selected_base_map_config['attr']},
                                      selected_overlays_config)

                if not data.empty:
                    # Se incluye la geometría para el join
                    data_with_geom = pd.merge(data, gdf_stations_info,
                                              on=Config.STATION_NAME_COL)

                    gpd_data = gpd.GeoDataFrame(data_with_geom, geometry='geometry',
                                                 crs=gdf_stations_info.crs)

                    for _, row in gpd_data.iterrows():
                        if pd.notna(row[Config.PRECIPITATION_COL]):
                            popup_object = generate_annual_map_popup_html(row, df_anual_full)

                            folium.CircleMarker(
                                location=[row['geometry'].y, row['geometry'].x], radius=5,
                                color=colormap(row[Config.PRECIPITATION_COL]),
                                fill=True, fill_color=colormap(row[Config.PRECIPITATION_COL]), fill_opacity=0.8,
                                tooltip=row[Config.STATION_NAME_COL], popup=popup_object
                            ).add_to(m)

                    if not gpd_data.empty:
                        # Asegurar que los límites son válidos antes de usarlos
                        bounds = gpd_data.total_bounds
                        if np.all(np.isfinite(bounds)):
                           m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

                    folium.LayerControl().add_to(m)

                with col:
                    folium_static(m, height=450, width=None)

            # Necesitamos más columnas en gdf_geometries para el merge en create_compare_map
            gdf_geometries = gdf_filtered[[Config.STATION_NAME_COL,
                                           Config.MUNICIPALITY_COL, Config.ALTITUDE_COL,
                                           'geometry']].drop_duplicates(subset=[Config.STATION_NAME_COL])

            data_year1 = df_anual_valid[df_anual_valid[Config.YEAR_COL] == year1]
            data_year2 = df_anual_valid[df_anual_valid[Config.YEAR_COL] == year2]

            create_compare_map(data_year1, year1, map_col1, gdf_geometries,
                               df_anual_valid)

            create_compare_map(data_year2, year2, map_col2, gdf_geometries,
                               df_anual_valid)

        else:
            st.warning("Se necesitan datos de al menos dos años diferentes para generar la Comparación de Mapas.")

def display_drought_analysis_tab(df_long, df_monthly_filtered, stations_for_analysis,
                                 df_anual_melted, gdf_filtered, analysis_mode, selected_regions,
                                 selected_municipios, selected_altitudes, **kwargs):
    st.header("Análisis de Extremos Hidrológicos")
    display_filter_summary(
        total_stations_count=len(st.session_state.gdf_stations),
        selected_stations_count=len(stations_for_analysis),
        year_range=st.session_state.year_range,
        selected_months_count=len(st.session_state.meses_numeros),
        analysis_mode=analysis_mode,
        selected_regions=selected_regions,
        selected_municipios=selected_municipios,
        selected_altitudes=selected_altitudes
    )
    if not stations_for_analysis:
        st.warning("Seleccione al menos una estación.")
        return

    st.subheader("Configuración del Análisis por Percentiles")
    station_to_analyze_perc = st.selectbox(
        "Seleccione una estación para el análisis de percentiles:",
        options=sorted(stations_for_analysis),
        key="percentile_station_select"
    )
    col1, col2 = st.columns(2)
    p_lower = col1.slider("Percentil Inferior (Sequía):", 1, 40, 10, key="p_lower_perc")
    p_upper = col2.slider("Percentil Superior (Húmedo):", 60, 99, 90, key="p_upper_perc")
    df_extremes, df_thresholds = pd.DataFrame(), pd.DataFrame()
    if station_to_analyze_perc:
        df_long_state = st.session_state.get('df_long')
        if df_long_state is not None and not df_long_state.empty:
            try:
                with st.spinner(f"Calculando percentiles P{p_lower} y P{p_upper}..."):
                    df_extremes, df_thresholds = calculate_percentiles_and_extremes(
                        df_long_state, station_to_analyze_perc, p_lower, p_upper
                    )
            except Exception as e:
                st.error(f"Error al calcular el análisis de percentiles: {e}")

    percentile_series_tab, percentile_thresholds_tab, indices_sub_tab, frequency_sub_tab = st.tabs([
        "Serie de Tiempo por Percentiles",
        "Umbrales de Percentil Mensual",
        "Índices de Sequía (SPI/SPEI)",
        "Análisis de Frecuencia de Extremos"
    ])
    
    with percentile_series_tab:
        if not df_extremes.empty:
            year_range_val = st.session_state.get('year_range', (2000, 2020))
            year_min, year_max = year_range_val if isinstance(year_range_val, tuple) and len(year_range_val) == 2 else st.session_state.get('year_range_single', (2000, 2020))
            
            df_plot = df_extremes[
                (df_extremes[Config.DATE_COL].dt.year >= year_min) &
                (df_extremes[Config.DATE_COL].dt.year <= year_max) &
                (df_extremes[Config.DATE_COL].dt.month.isin(st.session_state.meses_numeros))
            ].copy()

            if not df_plot.empty:
                st.subheader(f"Serie de Tiempo con Eventos Extremos (P{p_lower} y P{p_upper} Percentiles)")
                color_map = {f'Sequía Extrema (<P{p_lower}%)': 'red', f'Húmedo Extremo (>P{p_upper}%)': 'blue', 'Normal': 'gray'}
                fig_series = px.scatter(
                    df_plot, x=Config.DATE_COL, y=Config.PRECIPITATION_COL,
                    color='event_type', color_discrete_map=color_map,
                    title=f"Precipitación Mensual y Eventos Extremos en {station_to_analyze_perc}",
                    labels={Config.PRECIPITATION_COL: "Precipitación (mm)", Config.DATE_COL: "Fecha"},
                    hover_data={'event_type': True, 'p_lower': ':.0f', 'p_upper': ':.0f'}
                )
                mean_precip_station = st.session_state.df_long[st.session_state.df_long[Config.STATION_NAME_COL] == station_to_analyze_perc][Config.PRECIPITATION_COL].mean()
                fig_series.add_hline(y=mean_precip_station, line_dash="dash", line_color="green", annotation_text="Media Histórica")
                fig_series.update_layout(height=500)
                st.plotly_chart(fig_series, use_container_width=True)
            else:
                st.warning("No hay datos que coincidan con los filtros de tiempo para la estación seleccionada.")
        else:
            st.info("Seleccione una estación para ver el análisis.")

    with percentile_thresholds_tab:
        if not df_thresholds.empty:
            st.subheader("Umbrales de Percentil Mensual (Climatología Histórica)")
            meses_map_inv = {1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'}
            df_thresholds['Month_Name'] = df_thresholds[Config.MONTH_COL].map(meses_map_inv)
            fig_thresh = go.Figure()
            fig_thresh.add_trace(go.Scatter(x=df_thresholds['Month_Name'], y=df_thresholds['p_upper'], mode='lines+markers', name=f'Percentil Superior (P{p_upper}%)', line=dict(color='blue')))
            fig_thresh.add_trace(go.Scatter(x=df_thresholds['Month_Name'], y=df_thresholds['p_lower'], mode='lines+markers', name=f'Percentil Inferior (P{p_lower}%)', line=dict(color='red')))
            fig_thresh.add_trace(go.Scatter(x=df_thresholds['Month_Name'], y=df_thresholds['mean_monthly'], mode='lines', name='Media Mensual', line=dict(color='green', dash='dot')))
            fig_thresh.update_layout(title='Umbrales de Precipitación por Mes (Basado en Climatología)', xaxis_title="Mes", yaxis_title="Precipitación (mm)", height=400)
            st.plotly_chart(fig_thresh, use_container_width=True)
        else:
            st.info("Seleccione una estación para ver los umbrales.")

    with indices_sub_tab:
        st.subheader("Análisis con Índices Estandarizados")

        with st.expander("¿Cómo interpretar los índices de sequía?"):
            st.markdown("""
            El **Índice de Precipitación Estandarizado (SPI)** mide la desviación de la precipitación respecto a su media histórica.
            * **Valores Positivos (azul):** Indican condiciones más húmedas que el promedio.
            * **Valores Negativos (rojo):** Indican condiciones más secas (sequía).
            * **Valores cercanos a 0:** Representan condiciones normales.
            """)

        col1_idx, col2_idx = st.columns([1, 3])
        index_values = pd.Series(dtype=float)

        with col1_idx:
            index_type = st.radio("Índice a Calcular:", ("SPI", "SPEI"), key="index_type_radio")
            station_to_analyze_idx = st.selectbox("Estación para análisis:", options=sorted(stations_for_analysis), key="index_station_select")
            index_window = st.select_slider("Escala de tiempo (meses):", options=[3, 6, 9, 12, 24], value=12, key="index_window_slider")

        df_station_idx = pd.DataFrame()
        if station_to_analyze_idx:
            df_station_idx = df_monthly_filtered[df_monthly_filtered[Config.STATION_NAME_COL] == station_to_analyze_idx].copy().set_index(Config.DATE_COL).sort_index()

        with col2_idx:
            if not df_station_idx.empty:
                with st.spinner(f"Calculando {index_type}-{index_window}..."):
                    
                    if index_type == "SPI":
                        precip_series = df_station_idx[Config.PRECIPITATION_COL]
                        if len(precip_series.dropna()) < index_window * 2:
                            st.warning(f"No hay suficientes datos ({len(precip_series.dropna())} meses) para calcular el SPI-{index_window}.")
                        else:
                            index_values = calculate_spi(precip_series, index_window)
                    
                    elif index_type == "SPEI":
                        if Config.ET_COL not in df_station_idx.columns or df_station_idx[Config.ET_COL].isnull().all():
                            st.error(f"No hay datos de evapotranspiración ('{Config.ET_COL}') disponibles.")
                        else:
                            precip_series, et_series = df_station_idx[Config.PRECIPITATION_COL], df_station_idx[Config.ET_COL]
                            if len(precip_series.dropna()) < index_window * 2 or len(et_series.dropna()) < index_window * 2:
                                st.warning(f"No hay suficientes datos de precipitación o ETP para calcular el SPEI-{index_window}.")
                            else:
                                index_values = calculate_spei(precip_series, et_series, index_window)

        if not index_values.empty and not index_values.isnull().all():
            df_plot = pd.DataFrame({'index_val': index_values}).dropna()
            
            conditions = [
                (df_plot['index_val'] < -2.0),
                (df_plot['index_val'] >= -2.0) & (df_plot['index_val'] < -1.5),
                (df_plot['index_val'] >= -1.5) & (df_plot['index_val'] < -1.0),
                (df_plot['index_val'] >= -1.0) & (df_plot['index_val'] < 1.0),
                (df_plot['index_val'] >= 1.0) & (df_plot['index_val'] < 1.5),
                (df_plot['index_val'] >= 1.5) & (df_plot['index_val'] < 2.0),
                (df_plot['index_val'] >= 2.0)
            ]
            colors = ['#b2182b', '#ef8a62', '#fddbc7', 'lightgrey', '#92c5de', '#4393c3', '#2166ac']
            df_plot['color'] = np.select(conditions, colors, default='grey')
            
            fig = go.Figure(go.Bar(
                x=df_plot.index, 
                y=df_plot['index_val'],
                marker_color=df_plot['color'], 
                name=index_type
            ))
            fig.update_layout(
                title=f"Índice {index_type}-{index_window} para {station_to_analyze_idx}",
                yaxis_title=f"Valor {index_type}",
                xaxis_title="Fecha",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # display_event_analysis(index_values, index_type)

    with frequency_sub_tab:
        st.subheader("Análisis de Frecuencia de Precipitaciones Anuales Máximas")
        st.markdown("Este análisis estima la probabilidad de ocurrencia de un evento de precipitación de cierta magnitud utilizando la distribución de Gumbel para calcular los **períodos de retorno**.")
        station_to_analyze = st.selectbox("Seleccione una estación para el análisis de frecuencia:", options=sorted(stations_for_analysis), key="freq_station_select")
        if station_to_analyze:
            station_data = df_anual_melted[df_anual_melted[Config.STATION_NAME_COL] == station_to_analyze].copy()
            annual_max_precip = station_data['precipitation'].dropna()
            if len(annual_max_precip) < 10:
                st.warning("Se recomiendan al menos 10 años de datos para un análisis de frecuencia confiable.")
            else:
                with st.spinner("Calculando períodos de retorno..."):
                    params = stats.gumbel_r.fit(annual_max_precip)
                    return_periods = np.array([2, 5, 10, 25, 50, 100, 200, 500])
                    non_exceed_prob = 1 - 1 / return_periods
                    precip_values = stats.gumbel_r.ppf(non_exceed_prob, *params)
                    results_df = pd.DataFrame({"Período de Retorno (años)": return_periods, "Precipitación Anual Esperada (mm)": precip_values})
                    st.subheader(f"Resultados para la estación: {station_to_analyze}")
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown("#### Tabla de Resultados")
                        st.dataframe(results_df.style.format({"Precipitación Anual Esperada (mm)": "{:.1f}"}))
                    with col2:
                        st.markdown("#### Curva de Frecuencia")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=station_data[Config.YEAR_COL], y=annual_max_precip, mode='markers', name='Máximos Anuales Observados'))
                        x_plot = np.linspace(annual_max_precip.min(), precip_values[-1] * 1.1, 100)
                        y_plot_prob = stats.gumbel_r.cdf(x_plot, *params)
                        y_plot_prob = np.clip(y_plot_prob, 0, 0.999999)
                        y_plot_return_period = 1 / (1 - y_plot_prob)
                        fig.add_trace(go.Scatter(x=y_plot_return_period, y=x_plot, mode='lines', name='Curva de Gumbel Ajustada', line=dict(color='red')))
                        fig.update_layout(title="Curva de Períodos de Retorno", xaxis_title="Período de Retorno (años)", yaxis_title="Precipitación Anual (mm)", xaxis_type="log")
                        st.plotly_chart(fig, use_container_width=True)
                        

def display_anomalies_tab(df_long, df_monthly_filtered, stations_for_analysis,
                             analysis_mode, selected_regions, selected_municipios, selected_altitudes, **kwargs):

    st.header("Análisis de Anomalías de Precipitación")

    display_filter_summary(
        total_stations_count=len(st.session_state.gdf_stations),
        selected_stations_count=len(stations_for_analysis),
        year_range=st.session_state.year_range,
        selected_months_count=len(st.session_state.meses_numeros),
        analysis_mode=analysis_mode,
        selected_regions=selected_regions,
        selected_municipios=selected_municipios,
        selected_altitudes=selected_altitudes
    )

    if not stations_for_analysis:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return

    st.subheader("Configuración del Análisis")

    analysis_type = st.radio(
        "Calcular anomalía con respecto a:",
        ("El promedio de todo el período", "Una Normal Climatológica (período base fijo)"),
        key="anomaly_type"
    )

    df_anomalias = pd.DataFrame()
    avg_col_name = ""

    if analysis_type == "Una Normal Climatológica (período base fijo)":
        years_in_long = sorted(df_long[Config.YEAR_COL].unique())
        default_start = 1991 if 1991 in years_in_long else years_in_long[0]
        default_end = 2020 if 2020 in years_in_long else years_in_long[-1]

        c1, c2 = st.columns(2)
        with c1:
            baseline_start = st.selectbox("Año de inicio del período base:", years_in_long,
                                          index=years_in_long.index(default_start))
        with c2:
            baseline_end = st.selectbox("Año de fin del período base:", years_in_long,
                                        index=years_in_long.index(default_end))

        if baseline_start >= baseline_end:
            st.error("El año de inicio del período base debe ser anterior al año de fin.")
            return

        with st.spinner(f"Calculando anomalías vs. normal climatológica ({baseline_start}-{baseline_end})..."):
            from modules.analysis import calculate_climatological_anomalies # Asume importación
            df_anomalias = calculate_climatological_anomalies(df_monthly_filtered, df_long,
                                                             baseline_start, baseline_end)
        avg_col_name = 'precip_promedio_climatologico'

    else:
        with st.spinner("Calculando anomalías vs. promedio de todo el período..."):
            from modules.analysis import calculate_monthly_anomalies # Asume importación
            df_anomalias = calculate_monthly_anomalies(df_monthly_filtered, df_long)
        avg_col_name = 'precip_promedio_mes'

    if df_anomalias.empty or df_anomalias['anomalia'].isnull().all():
        st.warning("No hay suficientes datos históricos para calcular y mostrar las anomalías con los filtros actuales.")
        return

    anom_graf_tab, anom_fase_tab, anom_extremos_tab = st.tabs(["Gráfico de Anomalías",
                                                               "Anomalías por Fase ENSO", "Tabla de Eventos Extremos"])

    with anom_graf_tab:
        df_plot = df_anomalias.groupby(Config.DATE_COL).agg(anomalia=('anomalia',
                                                                        'mean')).reset_index()
        from .visualizer import create_anomaly_chart # Asume importación local de la función
        fig = create_anomaly_chart(df_plot)
        st.plotly_chart(fig, use_container_width=True)

    with anom_fase_tab:
        if Config.ENSO_ONI_COL in df_anomalias.columns:
            df_anomalias_enso = df_anomalias.dropna(subset=[Config.ENSO_ONI_COL]).copy()

            conditions = [df_anomalias_enso[Config.ENSO_ONI_COL] >= 0.5,
                          df_anomalias_enso[Config.ENSO_ONI_COL] <= -0.5]
            phases = ['El Niño', 'La Niña']
            df_anomalias_enso['enso_fase'] = np.select(conditions, phases, default='Neutral')

            fig_box = px.box(df_anomalias_enso, x='enso_fase', y='anomalia',
                             color='enso_fase', title="Distribución de Anomalías de Precipitación por Fase ENSO",
                             labels={'anomalia': 'Anomalía de Precipitación (mm)', 'enso_fase': 'Fase ENSO'},
                             points='all')
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning(f"La columna '{Config.ENSO_ONI_COL}' no está disponible para este análisis.")

    with anom_extremos_tab:
        st.subheader("Eventos Mensuales Extremos (Basado en Anomalías)")
        df_extremos = df_anomalias.dropna(subset=['anomalia']).copy()
        df_extremos['fecha'] = df_extremos[Config.DATE_COL].dt.strftime('%Y-%m')

        if avg_col_name and avg_col_name in df_extremos.columns:
            cols_to_show = ['fecha', Config.STATION_NAME_COL, 'anomalia',
                            Config.PRECIPITATION_COL, avg_col_name]
            col_rename_dict = {
                Config.STATION_NAME_COL: 'Estación',
                'anomalia': 'Anomalía (mm)',
                Config.PRECIPITATION_COL: 'Ppt. (mm)',
                avg_col_name: 'Ppt. Media (mm)'
            }
        else:
            cols_to_show = ['fecha', Config.STATION_NAME_COL, 'anomalia',
                            Config.PRECIPITATION_COL]
            col_rename_dict = {
                Config.STATION_NAME_COL: 'Estación',
                'anomalia': 'Anomalía (mm)',
                Config.PRECIPITATION_COL: 'Ppt. (mm)'
            }

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 10 Meses más Secos")
            secos = df_extremos.nsmallest(10, 'anomalia')[cols_to_show]
            st.dataframe(secos.rename(columns=col_rename_dict).round(0),
                         use_container_width=True)

        with col2:
            st.markdown("##### 10 Meses más Húmedos")
            humedos = df_extremos.nlargest(10, 'anomalia')[cols_to_show]
            st.dataframe(humedos.rename(columns=col_rename_dict).round(0),
                         use_container_width=True)

def display_stats_tab(df_long, df_anual_melted, df_monthly_filtered,
                      stations_for_analysis, gdf_filtered, analysis_mode, selected_regions, 
                      selected_municipios, selected_altitudes, **kwargs):
    st.header("Estadísticas de Precipitación")
    display_filter_summary(
        total_stations_count=len(st.session_state.gdf_stations),
        selected_stations_count=len(stations_for_analysis),
        year_range=st.session_state.year_range,
        selected_months_count=len(st.session_state.meses_numeros),
        analysis_mode=analysis_mode,
        selected_regions=selected_regions,
        selected_municipios=selected_municipios,
        selected_altitudes=selected_altitudes
    )

    if not stations_for_analysis:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return

    matriz_tab, resumen_mensual_tab, series_tab, sintesis_tab = st.tabs(["Matriz de Disponibilidad", "Resumen Mensual", "Datos Series Pptn", "Síntesis General"])

    with matriz_tab:
        st.subheader("Matriz de Disponibilidad de Datos Anual")
        heatmap_df = pd.DataFrame()
        title_text = ""
        color_scale = "Greens"

        if analysis_mode == "Completar series (interpolación)":
            view_mode = st.radio(
                "Seleccione la vista de la matriz:",
                ("Porcentaje de Datos Originales", "Porcentaje de Datos Completados",
                 "Porcentaje de Datos Totales"),
                horizontal=True, key="matriz_view_mode"
            )

            if view_mode == "Porcentaje de Datos Completados":
                df_counts = df_monthly_filtered[df_monthly_filtered[Config.ORIGIN_COL] ==
                                                'Completado'].groupby([Config.STATION_NAME_COL,
                                                                      Config.YEAR_COL]).size().reset_index(name='count')
                df_counts['porc_value'] = (df_counts['count'] / 12) * 100
                heatmap_df = df_counts.pivot(index=Config.STATION_NAME_COL,
                                             columns=Config.YEAR_COL, values='porc_value').fillna(0)
                color_scale = "Reds"
                title_text = "Porcentaje de Datos Completados (Interpolados)"

            elif view_mode == "Porcentaje de Datos Totales":
                df_counts = df_monthly_filtered.groupby([Config.STATION_NAME_COL,
                                                        Config.YEAR_COL]).size().reset_index(name='count')
                df_counts['porc_value'] = (df_counts['count'] / 12) * 100
                heatmap_df = df_counts.pivot(index=Config.STATION_NAME_COL,
                                             columns=Config.YEAR_COL, values='porc_value').fillna(0)
                color_scale = "Blues"
                title_text = "Disponibilidad de Datos Totales (Original + Completado)"

            else:  # Porcentaje de Datos Originales
                df_original_filtered = \
                    df_long[(df_long[Config.STATION_NAME_COL].isin(stations_for_analysis)) &
                            (df_long[Config.DATE_COL].dt.year >= st.session_state.year_range[0]) &
                            (df_long[Config.DATE_COL].dt.year <= st.session_state.year_range[1])]
                df_counts = df_original_filtered.groupby([Config.STATION_NAME_COL,
                                                         Config.YEAR_COL]).size().reset_index(name='count')
                df_counts['porc_value'] = (df_counts['count'] / 12) * 100
                heatmap_df = df_counts.pivot(index=Config.STATION_NAME_COL,
                                             columns=Config.YEAR_COL, values='porc_value').fillna(0)
                title_text = "Disponibilidad de Datos Originales"

        else:  # Modo de datos originales
            df_counts = df_monthly_filtered.groupby([Config.STATION_NAME_COL,
                                                    Config.YEAR_COL]).size().reset_index(name='count')
            df_counts['porc_value'] = (df_counts['count'] / 12) * 100
            heatmap_df = df_counts.pivot(index=Config.STATION_NAME_COL,
                                         columns=Config.YEAR_COL, values='porc_value').fillna(0)
            title_text = "Disponibilidad de Datos Originales"

        if not heatmap_df.empty:
            st.markdown(f"**{title_text}**")
            styled_df = heatmap_df.style.background_gradient(cmap=color_scale, axis=None,
                                                             vmin=0, vmax=100).format("{:.0f}%", na_rep="-")
            st.dataframe(styled_df)
        else:
            st.info("No hay datos para mostrar en la matriz con la selección actual.")

    with resumen_mensual_tab:
        st.subheader("Resumen de Estadísticas Mensuales por Estación")

        if not df_monthly_filtered.empty:
            summary_data = []
            for station_name, group in \
                    df_monthly_filtered.groupby(Config.STATION_NAME_COL):
                if not group[Config.PRECIPITATION_COL].dropna().empty:
                    max_row = group.loc[group[Config.PRECIPITATION_COL].idxmax()]
                    min_row = group.loc[group[Config.PRECIPITATION_COL].idxmin()]
                    summary_data.append({
                        "Estación": station_name,
                        "Ppt. Máxima Mensual (mm)": max_row[Config.PRECIPITATION_COL],
                        "Fecha Máxima": max_row[Config.DATE_COL].strftime('%Y-%m'),
                        "Ppt. Mínima Mensual (mm)": min_row[Config.PRECIPITATION_COL],
                        "Fecha Mínima": min_row[Config.DATE_COL].strftime('%Y-%m'),
                        "Promedio Mensual (mm)": group[Config.PRECIPITATION_COL].mean()
                    })

            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df.round(1), use_container_width=True)
            else:
                st.info("No hay datos suficientes para calcular el resumen mensual.")

        else:
            st.info("No hay datos para mostrar el resumen mensual.")

    with series_tab:
        st.subheader("Series de Precipitación Anual por Estación (mm)")
        if not df_anual_melted.empty:
            ppt_series_df = df_anual_melted.pivot_table(index=Config.STATION_NAME_COL,
                                                        columns=Config.YEAR_COL, values=Config.PRECIPITATION_COL)
            st.dataframe(ppt_series_df.style.format("{:.0f}", na_rep="-").background_gradient(cmap='viridis', axis=1))
        else:
            st.info("No hay datos anuales para mostrar en la tabla.")

    with sintesis_tab:
        st.subheader("Síntesis General de Precipitación")

        if not df_monthly_filtered.empty and not df_anual_melted.empty:
            df_anual_valid = df_anual_melted.dropna(subset=[Config.PRECIPITATION_COL])
            df_monthly_valid = \
                df_monthly_filtered.dropna(subset=[Config.PRECIPITATION_COL])

            if not df_anual_valid.empty and not df_monthly_valid.empty and not \
               gdf_filtered.empty:
                # --- A. EXTREMOS DE PRECIPITACIÓN
                max_monthly_row = \
                    df_monthly_valid.loc[df_monthly_valid[Config.PRECIPITATION_COL].idxmax()]
                min_monthly_row = \
                    df_monthly_valid.loc[df_monthly_valid[Config.PRECIPITATION_COL].idxmin()]

                max_annual_row = \
                    df_anual_valid.loc[df_anual_valid[Config.PRECIPITATION_COL].idxmax()]
                min_annual_row = \
                    df_anual_valid.loc[df_anual_valid[Config.PRECIPITATION_COL].idxmin()]

                # B. PROMEDIOS REGIONALES/CLIMATOLÓGICOS ---
                df_yearly_avg = \
                    df_anual_valid.groupby(Config.YEAR_COL)[Config.PRECIPITATION_COL].mean().reset_index()
                year_max_avg = \
                    df_yearly_avg.loc[df_yearly_avg[Config.PRECIPITATION_COL].idxmax()]
                year_min_avg = \
                    df_yearly_avg.loc[df_yearly_avg[Config.PRECIPITATION_COL].idxmin()]

                df_monthly_avg = \
                    df_monthly_valid.groupby(Config.MONTH_COL)[Config.PRECIPITATION_COL].mean().reset_index()

                month_max_avg = \
                    df_monthly_avg.loc[df_monthly_avg[Config.PRECIPITATION_COL].idxmax()][Config.MONTH_COL]
                month_min_avg = \
                    df_monthly_avg.loc[df_monthly_avg[Config.PRECIPITATION_COL].idxmin()][Config.MONTH_COL]

                #--- C. EXTREMOS DE ALTITUD ---
                df_stations_valid = gdf_filtered.dropna(subset=[Config.ALTITUDE_COL])

                station_max_alt = None
                station_min_alt = None

                if not df_stations_valid.empty:
                    df_stations_valid[Config.ALTITUDE_COL] = \
                        pd.to_numeric(df_stations_valid[Config.ALTITUDE_COL], errors='coerce')

                    if not df_stations_valid[Config.ALTITUDE_COL].isnull().all():
                        station_max_alt = \
                            df_stations_valid.loc[df_stations_valid[Config.ALTITUDE_COL].idxmax()]
                        station_min_alt = \
                            df_stations_valid.loc[df_stations_valid[Config.ALTITUDE_COL].idxmin()]

                # --- D. CÁLCULO DE TENDENCIAS (SEN'S SLOPE)
            trend_results = []
            
            # --- LÍNEA CORREGIDA ---
            import pymannkendall as mk # Importar directamente desde la librería
            # --- FIN DE LA CORRECCIÓN ---

            for station in stations_for_analysis:
                station_data = df_anual_valid[df_anual_valid[Config.STATION_NAME_COL] == station].copy()
                if len(station_data) >= 4:
                    mk_result_table = mk.original_test(station_data[Config.PRECIPITATION_COL])
                    trend_results.append({'slope_sen': mk_result_table.slope, 'p_value': mk_result_table.p, Config.STATION_NAME_COL: station})

            df_trends = pd.DataFrame(trend_results)
            max_pos_trend_row, min_neg_trend_row = None, None
            if not df_trends.empty:
                df_pos_trends = df_trends[df_trends['slope_sen'] > 0]
                df_neg_trends = df_trends[df_trends['slope_sen'] < 0]
                if not df_pos_trends.empty: max_pos_trend_row = df_pos_trends.loc[df_pos_trends['slope_sen'].idxmax()]
                if not df_neg_trends.empty: min_neg_trend_row = df_neg_trends.loc[df_neg_trends['slope_sen'].idxmin()]

                meses_map = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio', 7:
                             'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}

                # DISPLAY DE RESULTADOS
                st.markdown("#### 1. Extremos de Precipitación")
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Máxima Ppt. Anual",
                                     f"{max_annual_row[Config.PRECIPITATION_COL]:.0f} mm",
                                     f"{max_annual_row[Config.STATION_NAME_COL]} ({int(max_annual_row[Config.YEAR_COL])})")
                with col2: st.metric("Mínima Ppt. Anual",
                                     f"{min_annual_row[Config.PRECIPITATION_COL]:.0f} mm",
                                     f"{min_annual_row[Config.STATION_NAME_COL]} ({int(min_annual_row[Config.YEAR_COL])})")
                with col3: st.metric("Máxima Ppt. Mensual",
                                     f"{max_monthly_row[Config.PRECIPITATION_COL]:.0f} mm",
                                     f"{max_monthly_row[Config.STATION_NAME_COL]} ({meses_map.get(max_monthly_row[Config.MONTH_COL])} {max_monthly_row[Config.DATE_COL].year})")
                with col4: st.metric("Mínima Ppt. Mensual",
                                     f"{min_monthly_row[Config.PRECIPITATION_COL]:.0f} mm",
                                     f"{min_monthly_row[Config.STATION_NAME_COL]} ({meses_map.get(min_monthly_row[Config.MONTH_COL])} {min_monthly_row[Config.DATE_COL].year})")

                st.markdown("#### 2. Promedios Históricos y Climatológicos")
                col5, col6, col7 = st.columns(3)
                with col5: st.metric("Año más Lluvioso (Promedio Regional)",
                                     f"{year_max_avg[Config.PRECIPITATION_COL]:.0f} mm", f"Año: {int(year_max_avg[Config.YEAR_COL])}")
                with col6: st.metric("Año menos Lluvioso (Promedio Regional)",
                                     f"{year_min_avg[Config.PRECIPITATION_COL]:.0f} mm", f"Año: {int(year_min_avg[Config.YEAR_COL])}")
                with col7: st.metric("Mes Climatológico más Lluvioso",
                                     f"{df_monthly_avg.loc[df_monthly_avg[Config.MONTH_COL] == month_max_avg, Config.PRECIPITATION_COL].iloc[0]:.0f} mm", f"{meses_map.get(month_max_avg)} (Min: {meses_map.get(month_min_avg)})")

                st.markdown("#### 3. Geografía y Tendencias")
                col8, col9, col10, col11 = st.columns(4)
                with col8:
                    if station_max_alt is not None: st.metric("Estación a Mayor Altitud",
                                                              f"{float(station_max_alt[Config.ALTITUDE_COL]):.0f} m",
                                                              f"{station_max_alt[Config.STATION_NAME_COL]}")
                    else: st.info("No hay datos de altitud.")
                with col9:
                    if station_min_alt is not None: st.metric("Estación a Menor Altitud",
                                                              f"{float(station_min_alt[Config.ALTITUDE_COL]):.0f} m",
                                                              f"{station_min_alt[Config.STATION_NAME_COL]}")
                    else: st.info("No hay datos de altitud.")
                with col10:
                    if max_pos_trend_row is not None: st.metric("Mayor Tendencia Positiva",
                                                                f"+{max_pos_trend_row['slope_sen']:.2f} mm/año",
                                                                f"{max_pos_trend_row[Config.STATION_NAME_COL]} (p={max_pos_trend_row['p_value']:.3f})")
                    else: st.info("No hay tendencias positivas.")
                with col11:
                    if min_neg_trend_row is not None: st.metric("Mayor Tendencia Negativa",
                                                                f"{min_neg_trend_row['slope_sen']:.2f} mm/año",
                                                                f"{min_neg_trend_row[Config.STATION_NAME_COL]} (p={min_neg_trend_row['p_value']:.3f})")
                    else: st.info("No hay tendencias negativas.")
            else:
                st.info("No hay datos anuales, mensuales o geográficos válidos para mostrar la síntesis.")
        else:
            st.info("No hay datos para mostrar la síntesis general.")

def display_correlation_tab(df_monthly_filtered, stations_for_analysis, analysis_mode,
                                selected_regions, selected_municipios, selected_altitudes, **kwargs):

    st.header("Análisis de Correlación")

    display_filter_summary(
        total_stations_count=len(st.session_state.gdf_stations),
        selected_stations_count=len(stations_for_analysis),
        year_range=st.session_state.year_range,
        selected_months_count=len(st.session_state.meses_numeros),
        analysis_mode=analysis_mode,
        selected_regions=selected_regions,
        selected_municipios=selected_municipios,
        selected_altitudes=selected_altitudes
    )

    if not stations_for_analysis:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return

    st.markdown("Esta sección cuantifica la relación lineal entre la precipitación y diferentes variables utilizando el coeficiente de correlación de Pearson.")

    # AÑADIMOS UNA NUEVA PESTAÑA A LA LISTA
    tab_names = ["Correlación con ENSO (ONI)", "Matriz entre Estaciones", "Comparación 1 a 1", "Correlación con Otros Índices"]

    enso_corr_tab, matrix_corr_tab, station_corr_tab, indices_climaticos_tab = \
        st.tabs(tab_names)

    with enso_corr_tab:
        if Config.ENSO_ONI_COL not in df_monthly_filtered.columns or \
           df_monthly_filtered[Config.ENSO_ONI_COL].isnull().all():
            st.warning(f"No se puede realizar el análisis de correlación con ENSO. La columna '{Config.ENSO_ONI_COL}' no fue encontrada o no tiene datos en el período seleccionado.")
            return

        st.subheader("Configuración del Análisis de Correlación con ENSO")

        lag_months = st.slider(
            "Seleccionar desfase temporal (meses)",
            min_value=0, max_value=12, value=0,
            help="Analiza la correlación de la precipitación con el ENSO de 'x' meses atrás. Un desfase de 3 significa correlacionar la lluvia de hoy con el ENSO de hace 3 meses."
        )

        df_corr_analysis = df_monthly_filtered.dropna(subset=[Config.PRECIPITATION_COL,
                                                             Config.ENSO_ONI_COL])

        if df_corr_analysis.empty:
            st.warning("No hay datos coincidentes entre la precipitación y el ENSO para la selección actual.")
            return

        analysis_level = st.radio("Nivel de Análisis de Correlación con ENSO", ["Promedio de la selección", "Por Estación Individual"], horizontal=True, key="enso_corr_level")

        df_plot_corr = pd.DataFrame()
        title_text = ""

        if analysis_level == "Por Estación Individual":
            station_to_corr = st.selectbox("Seleccione Estación:",
                                           options=sorted(df_corr_analysis[Config.STATION_NAME_COL].unique()),
                                           key="enso_corr_station")
            if station_to_corr:
                df_plot_corr = df_corr_analysis[df_corr_analysis[Config.STATION_NAME_COL] ==
                                                station_to_corr].copy()
                title_text = f"Correlación para la estación: {station_to_corr}"
            else:
                return  # Si no se selecciona estación

        else:
            df_plot_corr = df_corr_analysis.groupby(Config.DATE_COL).agg(
                precipitation=(Config.PRECIPITATION_COL, 'mean'),
                anomalia_oni=(Config.ENSO_ONI_COL, 'first')
            ).reset_index()
            title_text = "Correlación para el promedio de las estaciones seleccionadas"

        if not df_plot_corr.empty and len(df_plot_corr) > 2:

            if lag_months > 0:
                df_plot_corr['anomalia_oni_shifted'] = \
                    df_plot_corr['anomalia_oni'].shift(lag_months)
                df_plot_corr.dropna(subset=['anomalia_oni_shifted'], inplace=True)
                oni_column_to_use = 'anomalia_oni_shifted'
                lag_text = f" (con desfase de {lag_months} meses)"
            else:
                oni_column_to_use = 'anomalia_oni'
                lag_text = ""

            corr, p_value = stats.pearsonr(df_plot_corr[oni_column_to_use],
                                           df_plot_corr['precipitation'])

            st.subheader(title_text + lag_text)

            col1, col2 = st.columns(2)
            col1.metric("Coeficiente de Correlación (r)", f"{corr:.3f}")
            col2.metric("Significancia (valor p)", f"{p_value:.4f}")

            if p_value < 0.05:
                st.success("La correlación es estadísticamente significativa.")
            else:
                st.warning("La correlación no es estadísticamente significativa.")

            fig_corr = px.scatter(
                df_plot_corr, x=oni_column_to_use, y='precipitation', trendline='ols',
                title=f"Dispersión: Precipitación vs. Anomalía ONI{lag_text}",
                labels={oni_column_to_use: f'Anomalía ONI (°C) [desfase {lag_months}m]',
                        'precipitation': 'Precipitación Mensual (mm)'}
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        else:
            st.warning("No hay suficientes datos superpuestos para calcular la correlación.")

    with matrix_corr_tab:
        st.subheader("Matriz de Correlación de Precipitación entre Estaciones")
        if len(stations_for_analysis) < 2:
            st.info("Seleccione al menos dos estaciones para generar la matriz de correlación.")
        else:
            with st.spinner("Calculando matriz de correlación..."):
                df_pivot = df_monthly_filtered.pivot_table(
                    index=Config.DATE_COL,
                    columns=Config.STATION_NAME_COL,
                    values=Config.PRECIPITATION_COL
                )
                corr_matrix = df_pivot.corr()

            fig_matrix = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title="Mapa de Calor de Correlaciones de Precipitación Mensual"
            )
            fig_matrix.update_layout(height=max(400, len(stations_for_analysis) * 25))
            st.plotly_chart(fig_matrix, use_container_width=True)

    with station_corr_tab:
        if len(stations_for_analysis) < 2:
            st.info("Seleccione al menos dos estaciones para comparar la correlación entre ellas.")
        else:
            st.subheader("Correlación de Precipitación entre dos Estaciones")
            station_options = sorted(stations_for_analysis)
            col1, col2 = st.columns(2)
            station1_name = col1.selectbox("Estación 1:", options=station_options,
                                           key="corr_station_1")
            station2_name = col2.selectbox("Estación 2:", options=station_options, index=1 if
                                           len(station_options) > 1 else 0, key="corr_station_2")

            if station1_name and station2_name and station1_name != station2_name:
                df_station1 = \
                    df_monthly_filtered[df_monthly_filtered[Config.STATION_NAME_COL] ==
                                        station1_name][[Config.DATE_COL, Config.PRECIPITATION_COL]]
                df_station2 = \
                    df_monthly_filtered[df_monthly_filtered[Config.STATION_NAME_COL] ==
                                        station2_name][[Config.DATE_COL, Config.PRECIPITATION_COL]]

                df_merged = pd.merge(df_station1, df_station2, on=Config.DATE_COL,
                                     suffixes=('_1', '_2')).dropna()
                df_merged.rename(columns={f'{Config.PRECIPITATION_COL}_1': station1_name,
                                          f'{Config.PRECIPITATION_COL}_2': station2_name}, inplace=True)

                if not df_merged.empty and len(df_merged) > 2:
                    corr, p_value = stats.pearsonr(df_merged[station1_name],
                                                   df_merged[station2_name])
                    st.markdown(f"#### Resultados de la correlación ({station1_name} vs. {station2_name})")
                    st.metric("Coeficiente de Correlación (r)", f"{corr:.3f}")

                    if p_value < 0.05:
                        st.success(f"La correlación es estadísticamente significativa (p={p_value:.4f}).")
                    else:
                        st.warning(f"La correlación no es estadísticamente significativa (p={p_value:.4f}).")

                    slope, intercept, _, _, _ = stats.linregress(df_merged[station1_name],
                                                                 df_merged[station2_name])
                    st.info(f"Ecuación de regresión: y = {slope:.2f}x + {intercept:.2f}")

                    fig_scatter = px.scatter(
                        df_merged, x=station1_name, y=station2_name, trendline='ols',
                        title=f'Dispersión de Precipitación: {station1_name} vs. {station2_name}',
                        labels={station1_name: f'Precipitación en {station1_name} (mm)',
                                station2_name: f'Precipitación en {station2_name} (mm)'}
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.warning("No hay suficientes datos superpuestos para calcular la correlación para las estaciones seleccionadas.")

    with indices_climaticos_tab:
        st.subheader("Análisis de Correlación con Índices Climáticos (SOI, IOD)")
        available_indices = []
        if Config.SOI_COL in df_monthly_filtered.columns and not \
           df_monthly_filtered[Config.SOI_COL].isnull().all():
            available_indices.append("SOI")
        if Config.IOD_COL in df_monthly_filtered.columns and not \
           df_monthly_filtered[Config.IOD_COL].isnull().all():
            available_indices.append("IOD")

        if not available_indices:
            st.warning("No se encontraron columnas para los índices climáticos (SOI o IOD) en el archivo principal o no hay datos en el período seleccionado.")
        else:
            col1_corr, col2_corr = st.columns(2)
            selected_index = col1_corr.selectbox("Seleccione un índice climático:",
                                                 available_indices)
            selected_station_corr = col2_corr.selectbox("Seleccione una estación:",
                                                        options=sorted(stations_for_analysis), key="station_for_index_corr")

            if selected_index and selected_station_corr:
                index_col_map = {"SOI": Config.SOI_COL, "IOD": Config.IOD_COL}
                index_col_name = index_col_map.get(selected_index)

                df_merged_indices = \
                    df_monthly_filtered[df_monthly_filtered[Config.STATION_NAME_COL] ==
                                        selected_station_corr].copy()

                if index_col_name in df_merged_indices.columns:
                    df_merged_indices.dropna(subset=[Config.PRECIPITATION_COL,
                                                     index_col_name], inplace=True)
                else:
                    st.error(f"La columna para el índice '{selected_index}' no se encontró en los datos de la estación.")
                    return

                if not df_merged_indices.empty and len(df_merged_indices) > 2:
                    corr, p_value = stats.pearsonr(df_merged_indices[index_col_name],
                                                   df_merged_indices[Config.PRECIPITATION_COL])
                    st.markdown(f"#### Resultados de la correlación ({selected_index} vs. Precipitación de {selected_station_corr})")
                    st.metric("Coeficiente de Correlación (r)", f"{corr:.3f}")

                    if p_value < 0.05:
                        st.success("La correlación es estadísticamente significativa.")
                    else:
                        st.warning("La correlación no es estadísticamente significativa.")

                    fig_scatter_indices = px.scatter(
                        df_merged_indices, x=index_col_name, y=Config.PRECIPITATION_COL,
                        trendline='ols',
                        title=f'Dispersión: {selected_index} vs. Precipitación de {selected_station_corr}',
                        labels={index_col_name: f'Valor del índice {selected_index}',
                                Config.PRECIPITATION_COL: 'Precipitación Mensual (mm)'}
                    )
                    st.plotly_chart(fig_scatter_indices, use_container_width=True)
                else:
                    st.warning("No hay suficientes datos superpuestos entre la estación y el índice para calcular la correlación.")

def display_enso_tab(df_enso, df_monthly_filtered, gdf_filtered, stations_for_analysis,
                        analysis_mode, selected_regions, selected_municipios, selected_altitudes, **kwargs):
    st.header("Análisis de Precipitación y el Fenómeno ENSO")
    display_filter_summary(
        total_stations_count=len(st.session_state.gdf_stations),
        selected_stations_count=len(stations_for_analysis),
        year_range=st.session_state.year_range,
        selected_months_count=len(st.session_state.meses_numeros),
        analysis_mode=analysis_mode,
        selected_regions=selected_regions,
        selected_municipios=selected_municipios,
        selected_altitudes=selected_altitudes
    )
    if not stations_for_analysis:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return
    if df_enso is None or df_enso.empty:
        st.warning("No se encontraron datos del fenómeno ENSO en el archivo de precipitación cargado.")
        return

    enso_series_tab, enso_anim_tab = st.tabs(["Series de Tiempo ENSO", "Mapa Interactivo ENSO"])

    with enso_series_tab:
        enso_vars_available = {
            Config.ENSO_ONI_COL: 'Anomalía ONI',
            'temp_sst': 'Temp. Superficial del Mar (SST)',
            'temp_media': 'Temp. Media'
        }
        available_tabs = [name for var, name in enso_vars_available.items() if var in
                          df_enso.columns]
        if not available_tabs:
            st.warning("No hay variables ENSO disponibles en el archivo de datos para visualizar.")
        else:
            enso_variable_tabs = st.tabs(available_tabs)
            for i, var_name in enumerate(available_tabs):
                with enso_variable_tabs[i]:
                    var_code = [code for code, name in enso_vars_available.items() if name ==
                                var_name][0]
                    enso_filtered = df_enso
                    if not enso_filtered.empty and var_code in enso_filtered.columns and not \
                       enso_filtered[var_code].isnull().all():

                        fig_enso_series = px.line(enso_filtered, x=Config.DATE_COL, y=var_code,
                                                  title=f"Serie de Tiempo para {var_name}")
                        st.plotly_chart(fig_enso_series, use_container_width=True)
                    else:
                        st.warning(f"No hay datos disponibles para '{var_code}' en el período seleccionado.")

    with enso_anim_tab:
        st.subheader("Explorador Mensual del Fenómeno ENSO")
        if gdf_filtered.empty or Config.ENSO_ONI_COL not in df_enso.columns:
            st.warning("Datos insuficientes para generar esta visualización. Se requiere información de estaciones y la columna 'anomalia_oni'.")
            return

        controls_col, map_col = st.columns([1, 3])
        enso_anim_data = df_enso[[Config.DATE_COL,
                                  Config.ENSO_ONI_COL]].copy().dropna(subset=[Config.ENSO_ONI_COL])
        conditions = [
            enso_anim_data[Config.ENSO_ONI_COL] >= 0.5,
            enso_anim_data[Config.ENSO_ONI_COL] <= -0.5
        ]
        phases = ['El Niño', 'La Niña']
        enso_anim_data['fase'] = np.select(conditions, phases, default='Neutral')

        year_range_val = st.session_state.get('year_range', (2000, 2020))
        if isinstance(year_range_val, tuple) and len(year_range_val) == 2 and \
           isinstance(year_range_val[0], int):
            year_min, year_max = year_range_val
        else:
            year_min, year_max = st.session_state.get('year_range_single', (2000, 2020))

        enso_anim_data_filtered = enso_anim_data[
            (enso_anim_data[Config.DATE_COL].dt.year >= year_min) &
            (enso_anim_data[Config.DATE_COL].dt.year <= year_max)
        ]
        selected_date = None

        with controls_col:
            st.markdown("##### Controles de Mapa")
            selected_base_map_config, selected_overlays_config = display_map_controls(st,
                                                                                       "enso_anim")
            st.markdown("##### Selección de Fecha")
            available_dates = sorted(enso_anim_data_filtered[Config.DATE_COL].unique())
            if available_dates:
                selected_date = st.select_slider("Seleccione una fecha (Año-Mes)",
                                                 options=available_dates,
                                                 format_func=lambda date: pd.to_datetime(date).strftime('%Y-%m'))
                phase_info = \
                    enso_anim_data_filtered[enso_anim_data_filtered[Config.DATE_COL] == selected_date]
                if not phase_info.empty:
                    current_phase = phase_info['fase'].iloc[0]
                    current_oni = phase_info[Config.ENSO_ONI_COL].iloc[0]
                    st.metric(f"Fase ENSO en {pd.to_datetime(selected_date).strftime('%Y-%m')}",
                              current_phase, f"Anomalía ONI: {current_oni:.2f}°C")
                else:
                    st.warning("No hay datos de ENSO para el período seleccionado.")
            else:
                st.warning("No hay fechas con datos ENSO en el rango seleccionado.")

        with map_col:
            if selected_date:
                # CORRECCIÓN DE ATTRIBUTEERROR: Extraer solo la clave 'tiles'
                m_enso = create_folium_map([4.57, -74.29], 5, {'tiles': selected_base_map_config['tiles'], 'attr': selected_base_map_config['attr']},
                                           selected_overlays_config)
                phase_color_map = {'El Niño': 'red', 'La Niña': 'blue', 'Neutral': 'gray'}
                phase_info = \
                    enso_anim_data_filtered[enso_anim_data_filtered[Config.DATE_COL] == selected_date]
                current_phase_str = phase_info['fase'].iloc[0] if not phase_info.empty else "N/A"
                marker_color = phase_color_map.get(current_phase_str, 'black')
                for _, station in gdf_filtered.iterrows():
                    folium.Marker(
                        location=[station['geometry'].y, station['geometry'].x],
                        tooltip=f"{station[Config.STATION_NAME_COL]}<br>Fase: {current_phase_str}",
                        icon=folium.Icon(color=marker_color, icon='cloud')
                    ).add_to(m_enso)
                if not gdf_filtered.empty:
                    bounds = gdf_filtered.total_bounds
                    if np.all(np.isfinite(bounds)):
                        m_enso.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                folium.LayerControl().add_to(m_enso)
                folium_static(m_enso, height=700, width=None)

            else:
                st.info("Seleccione una fecha para visualizar el mapa.")

def display_trends_and_forecast_tab(df_full_monthly, stations_for_analysis,
                                    df_anual_melted, df_monthly_filtered, analysis_mode, selected_regions,
                                    selected_municipios, selected_altitudes, **kwargs):
    st.header("Análisis de Tendencias y Pronósticos")
    display_filter_summary(
        total_stations_count=len(st.session_state.gdf_stations),
        selected_stations_count=len(stations_for_analysis),
        year_range=st.session_state.year_range,
        selected_months_count=len(st.session_state.meses_numeros),
        analysis_mode=analysis_mode,
        selected_regions=selected_regions,
        selected_municipios=selected_municipios,
        selected_altitudes=selected_altitudes
    )
    if not stations_for_analysis:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return

    tab_names = ["Análisis Lineal", "Tendencia Mann-Kendall", "Tabla Comparativa",
                 "Descomposición de Series", "Autocorrelación (ACF/PACF)", "Pronóstico SARIMA",
                 "Pronóstico Prophet", "SARIMA vs Prophet"]
    tendencia_individual_tab, mann_kendall_tab, tendencia_tabla_tab, \
        descomposicion_tab, autocorrelacion_tab, pronostico_sarima_tab, \
        pronostico_prophet_tab, compare_forecast_tab = st.tabs(tab_names)

    with tendencia_individual_tab:
        st.subheader("Tendencia de Precipitación Anual (Regresión Lineal)")
        analysis_type = st.radio("Tipo de Análisis de Tendencia:", ["Promedio de la selección",
                                                                    "Estación individual"], horizontal=True, key="linear_trend_type")
        df_to_analyze = None

        if analysis_type == "Promedio de la selección":
            df_to_analyze = \
                df_anual_melted.groupby(Config.YEAR_COL)[Config.PRECIPITATION_COL].mean().reset_index()
        else:
            station_to_analyze = st.selectbox("Seleccione una estación para analizar:",
                                              options=stations_for_analysis, key="tendencia_station_select")
            if station_to_analyze:
                df_to_analyze = df_anual_melted[df_anual_melted[Config.STATION_NAME_COL] == station_to_analyze]

        if df_to_analyze is not None and \
           len(df_to_analyze.dropna(subset=[Config.PRECIPITATION_COL])) > 2:
            df_to_analyze['año_num'] = pd.to_numeric(df_to_analyze[Config.YEAR_COL])
            df_clean = df_to_analyze.dropna(subset=[Config.PRECIPITATION_COL])
            slope, intercept, r_value, p_value, std_err = stats.linregress(df_clean['año_num'],
                                                                           df_clean[Config.PRECIPITATION_COL])
            tendencia_texto = "aumentando" if slope > 0 else "disminuyendo"
            significancia_texto = "**estadísticamente significativa**" if p_value < 0.05 else "no es **estadísticamente significativa**"
            st.markdown(f"La tendencia de la precipitación es de **{slope:.2f} mm/año** (es decir, está {tendencia_texto}). Con un valor p de **{p_value:.3f}**, esta tendencia {significancia_texto}.")

            df_to_analyze['tendencia'] = slope * df_to_analyze['año_num'] + intercept

            fig_tendencia = px.scatter(df_to_analyze, x='año_num',
                                       y=Config.PRECIPITATION_COL, title='Tendencia de la Precipitación Anual')
            fig_tendencia.add_trace(go.Scatter(x=df_to_analyze['año_num'],
                                                 y=df_to_analyze['tendencia'], mode='lines', name='Línea de Tendencia',
                                                 line=dict(color='red')))
            fig_tendencia.update_layout(xaxis_title="Año", yaxis_title="Precipitación Anual (mm)")
            st.plotly_chart(fig_tendencia, use_container_width=True)
        else:
            st.warning("No hay suficientes datos en el período seleccionado para calcular una tendencia.")

    with mann_kendall_tab:
        st.subheader("Tendencia de Precipitación Anual (Prueba de Mann-Kendall y Pendiente de Sen)")
        with st.expander("¿Qué es la prueba de Mann-Kendall?"):
            st.markdown("""
            - **Prueba de Mann-Kendall**: Detecta si existe una tendencia (creciente o decreciente) en el tiempo.
            - **Valor p**: Si es < 0.05, la tendencia es estadísticamente significativa.
            - **Pendiente de Sen**: Cuantifica la magnitud de la tendencia (ej. "aumento de 5 mm/año"). Es un método robusto que no se ve muy afectado por valores atípicos.
            """)

        mk_analysis_type = st.radio("Tipo de Análisis de Tendencia:", ["Promedio de la selección", "Estación individual"], horizontal=True, key="mk_trend_type")
        df_to_analyze_mk = None

        if mk_analysis_type == "Promedio de la selección":
            df_to_analyze_mk = \
                df_anual_melted.groupby(Config.YEAR_COL)[Config.PRECIPITATION_COL].mean().reset_index()
        else:
            station_to_analyze_mk = st.selectbox("Seleccione una estación para analizar:",
                                                  options=stations_for_analysis, key="mk_station_select")
            if station_to_analyze_mk:
                df_to_analyze_mk = \
                    df_anual_melted[df_anual_melted[Config.STATION_NAME_COL] == station_to_analyze_mk]

        if df_to_analyze_mk is not None and \
           len(df_to_analyze_mk.dropna(subset=[Config.PRECIPITATION_COL])) > 3:
            df_clean_mk = \
                df_to_analyze_mk.dropna(subset=[Config.PRECIPITATION_COL]).sort_values(by=Config.YEAR_COL)
            mk_result = mk.original_test(df_clean_mk[Config.PRECIPITATION_COL])

            title = 'Promedio de la selección' if mk_analysis_type == 'Promedio de la selección' \
                else station_to_analyze_mk
            st.markdown(f"#### Resultados para: {title}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Tendencia Detectada", mk_result.trend.capitalize())
            col2.metric("Valor p", f"{mk_result.p:.4f}")
            col3.metric("Pendiente de Sen (mm/año)", f"{mk_result.slope:.2f}")

            df_clean_mk['año_num'] = pd.to_numeric(df_clean_mk[Config.YEAR_COL])
            median_x = df_clean_mk['año_num'].median()
            median_y = df_clean_mk[Config.PRECIPITATION_COL].median()
            intercept = median_y - (mk_result.slope * median_x)
            df_clean_mk['tendencia_sen'] = (mk_result.slope * df_clean_mk['año_num']) + intercept

            fig_mk = go.Figure()
            fig_mk.add_trace(go.Scatter(x=df_clean_mk['año_num'],
                                         y=df_clean_mk[Config.PRECIPITATION_COL], mode='markers', name='Datos Anuales'))
            fig_mk.add_trace(go.Scatter(x=df_clean_mk['año_num'],
                                         y=df_clean_mk['tendencia_sen'], mode='lines', name="Tendencia (Sen's Slope)",
                                         line=dict(color='orange')))
            fig_mk.update_layout(title=f"Tendencia de Mann-Kendall para {title}",
                                 xaxis_title="Año", yaxis_title="Precipitación Anual (mm)")
            st.plotly_chart(fig_mk, use_container_width=True)
        else:
            st.warning("No hay suficientes datos (se requieren al menos 4 puntos) para calcular la tendencia de Mann-Kendall.")

    with tendencia_tabla_tab:
        st.subheader("Tabla Comparativa de Tendencias de Precipitación Anual")
        st.info("Presione el botón para calcular los valores para todas las estaciones seleccionadas.")
        if st.button("Calcular Tendencias para Todas las Estaciones Seleccionadas"):
            with st.spinner("Calculando tendencias..."):
                results = []
                df_anual_calc = df_anual_melted.copy()
                for station in stations_for_analysis:
                    station_data = df_anual_calc[df_anual_calc[Config.STATION_NAME_COL] ==
                                                  station].dropna(subset=[Config.PRECIPITATION_COL]).sort_values(by=Config.YEAR_COL)

                    slope_lin, p_lin = np.nan, np.nan
                    trend_mk, p_mk, slope_sen = "Datos insuficientes", np.nan, np.nan
                    if len(station_data) > 2:
                        station_data['año_num'] = pd.to_numeric(station_data[Config.YEAR_COL])
                        res = stats.linregress(station_data['año_num'],
                                               station_data[Config.PRECIPITATION_COL])
                        slope_lin, p_lin = res.slope, res.pvalue
                    if len(station_data) > 3:
                        mk_result_table = \
                            mk.original_test(station_data[Config.PRECIPITATION_COL])
                        trend_mk = mk_result_table.trend.capitalize()
                        p_mk = mk_result_table.p
                        slope_sen = mk_result_table.slope
                    results.append({"Estación": station, "Años Analizados": len(station_data),
                                    "Tendencia Lineal (mm/año)": slope_lin, "Valor p (Lineal)": p_lin, "Tendencia MK":
                                    trend_mk, "Valor p (MK)": p_mk, "Pendiente de Sen (mm/año)": slope_sen})
                if results:
                    results_df = pd.DataFrame(results)
                    def style_p_value(val):
                        if pd.isna(val) or isinstance(val, str): return ""
                        color = 'lightgreen' if val < 0.05 else 'lightcoral'
                        return f'background-color: {color}'
                    st.dataframe(results_df.style.format({"Tendencia Lineal (mm/año)": "{:.2f}",
                                                           "Valor p (Lineal)": "{:.4f}", "Valor p (MK)": "{:.4f}", "Pendiente de Sen (mm/año)":
                                                           "{:.2f}"}).applymap(style_p_value, subset=['Valor p (Lineal)', 'Valor p (MK)']),
                                 use_container_width=True)

    with descomposicion_tab:
        st.subheader("Descomposición de Series de Tiempo Mensual")
        station_to_decompose = st.selectbox("Seleccione una estación para la descomposición:",
                                            options=stations_for_analysis, key="decompose_station_select")
        if station_to_decompose:
            df_station = df_monthly_filtered[df_monthly_filtered[Config.STATION_NAME_COL] ==
                                            station_to_decompose].copy()
            if not df_station.empty:
                df_station.set_index(Config.DATE_COL, inplace=True)
                try:
                    from modules.forecasting import get_decomposition_results
                    series_for_decomp = \
                        df_station[Config.PRECIPITATION_COL].asfreq('MS').interpolate(method='time')
                    result = get_decomposition_results(series_for_decomp)
                    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                        subplot_titles=("Observado", "Tendencia", "Estacionalidad", "Residuo"))
                    fig.add_trace(go.Scatter(x=result.observed.index, y=result.observed,
                                             mode='lines', name='Observado'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=result.trend.index, y=result.trend, mode='lines',
                                             name='Tendencia'), row=2, col=1)
                    fig.add_trace(go.Scatter(x=result.seasonal.index, y=result.seasonal,
                                             mode='lines', name='Estacionalidad'), row=3, col=1)
                    fig.add_trace(go.Scatter(x=result.resid.index, y=result.resid, mode='markers',
                                             name='Residuo'), row=4, col=1)
                    fig.update_layout(height=700, title_text=f"Descomposición de la Serie para {station_to_decompose}", showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"No se pudo realizar la descomposición. Error: {e}")

    with autocorrelacion_tab:
        st.subheader("Análisis de Autocorrelación (ACF) y Autocorrelación Parcial (PACF)")
        station_to_analyze_acf = st.selectbox("Seleccione una estación:",
                                                options=stations_for_analysis, key="acf_station_select")
        max_lag = st.slider("Número máximo de rezagos (meses):", min_value=12,
                            max_value=60, value=24, step=12)
        if station_to_analyze_acf:
            df_station_acf = \
                df_monthly_filtered[df_monthly_filtered[Config.STATION_NAME_COL] ==
                                    station_to_analyze_acf].copy()
            if not df_station_acf.empty:
                df_station_acf.set_index(Config.DATE_COL, inplace=True)
                df_station_acf = df_station_acf.asfreq('MS')
                df_station_acf[Config.PRECIPITATION_COL] = \
                    df_station_acf[Config.PRECIPITATION_COL].interpolate(method='time').dropna()
                if len(df_station_acf) > max_lag:
                    try:
                        from modules.forecasting import create_acf_chart, create_pacf_chart
                        fig_acf = create_acf_chart(df_station_acf[Config.PRECIPITATION_COL],
                                                   max_lag)
                        st.plotly_chart(fig_acf, use_container_width=True)
                        fig_pacf = create_pacf_chart(df_station_acf[Config.PRECIPITATION_COL],
                                                     max_lag)
                        st.plotly_chart(fig_pacf, use_container_width=True)
                    except Exception as e:
                        st.error(f"No se pudieron generar los gráficos de autocorrelación. Error: {e}")
                else:
                    st.warning(f"No hay suficientes datos para el análisis de autocorrelación.")

    with pronostico_sarima_tab:
        st.subheader("Pronóstico (Modelo SARIMA)")
        st.info("Los pronósticos se generan utilizando los datos procesados...")
        station_to_forecast = st.selectbox("Seleccione una estación:",
                                            options=stations_for_analysis, key="sarima_station_select")
        c1, c2 = st.columns(2)
        with c1:
            forecast_horizon = st.slider("Meses a pronosticar:", 12, 36, 12, step=12,
                                         key="sarima_horizon")
        with c2:
            test_size = st.slider("Meses para evaluación:", 12, 36, 12, step=6,
                                  key="sarima_test_size")
        use_auto_arima = st.checkbox("Encontrar parámetros óptimos automáticamente (Auto-ARIMA)",
                                      value=True)
        if station_to_forecast and st.button("Generar Pronóstico SARIMA"):
            ts_data_sarima = \
                df_monthly_filtered[df_monthly_filtered[Config.STATION_NAME_COL] ==
                                    station_to_forecast].copy()
            if len(ts_data_sarima.dropna(subset=[Config.PRECIPITATION_COL])) < test_size + 36:
                st.warning("No hay suficientes datos para un pronóstico confiable (se necesitan al menos 3 años más que el período de evaluación).")
            else:
                try:
                    from modules.forecasting import auto_arima_search, generate_sarima_forecast
                    if use_auto_arima:
                        with st.spinner("Buscando el mejor modelo Auto-ARIMA (esto puede tardar)..."):
                            order, seasonal_order = auto_arima_search(ts_data_sarima, test_size)
                        st.success(f"Modelo óptimo encontrado: orden={order}, orden estacional={seasonal_order}")
                    else:
                        order, seasonal_order = (1, 1, 1), (1, 1, 1, 12)
                    with st.spinner("Entrenando y evaluando modelo SARIMA..."):
                        ts_hist, forecast_mean, forecast_ci, metrics, sarima_df_export = \
                            generate_sarima_forecast(ts_data_sarima, order, seasonal_order, forecast_horizon,
                                                     test_size)
                        st.session_state['sarima_results'] = {'forecast': sarima_df_export, 'metrics':
                                                              metrics, 'history': ts_hist}
                        st.markdown("##### Resultados del Pronóstico")
                        fig_pronostico = go.Figure()
                        fig_pronostico.add_trace(go.Scatter(x=ts_hist.index, y=ts_hist, mode='lines',
                                                             name='Datos Históricos'))
                        fig_pronostico.add_trace(go.Scatter(x=forecast_mean.index, y=forecast_mean,
                                                             mode='lines', name='Pronóstico SARIMA', line=dict(color='red', dash='dash')))
                        fig_pronostico.add_trace(go.Scatter(x=forecast_ci.index, y=forecast_ci.iloc[:, 0],
                                                             mode='lines', line=dict(width=0), showlegend=False))
                        fig_pronostico.add_trace(go.Scatter(x=forecast_ci.index, y=forecast_ci.iloc[:, 1],
                                                             mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255,0,0,0.2)',
                                                             name='Intervalo de Confianza'))
                        st.plotly_chart(fig_pronostico, use_container_width=True)
                        st.markdown("##### Evaluación del Modelo")
                        st.info(f"El modelo se evaluó usando los últimos **{test_size} meses** de datos históricos como conjunto de prueba.")
                        m1, m2 = st.columns(2)
                        m1.metric("RMSE (Error Cuadrático Medio)", f"{metrics['RMSE']:.2f}")
                        m2.metric("MAE (Error Absoluto Medio)", f"{metrics['MAE']:.2f}")
                except Exception as e:
                    st.error(f"No se pudo generar el pronóstico SARIMA. Error: {e}")

    with pronostico_prophet_tab:
        st.subheader("Pronóstico (Modelo Prophet)")
        station_to_forecast_prophet = st.selectbox("Seleccione una estación:",
                                                    options=stations_for_analysis, key="prophet_station_select")
        c1, c2 = st.columns(2)
        with c1:
            forecast_horizon_prophet = st.slider("Meses a pronosticar:", 12, 36, 12, step=12,
                                                 key="prophet_horizon")
        with c2:
            test_size_prophet = st.slider("Meses para evaluación:", 12, 36, 12, step=6,
                                           key="prophet_test_size")
        if station_to_forecast_prophet and st.button("Generar Pronóstico Prophet"):
            with st.spinner(f"Preparando y completando datos para {station_to_forecast_prophet}..."):
                original_station_data = \
                    df_full_monthly[df_full_monthly[Config.STATION_NAME_COL] ==
                                    station_to_forecast_prophet].copy()
                from modules.data_processor import complete_series # Asume importación
                ts_data_prophet = complete_series(original_station_data)
                if len(ts_data_prophet.dropna(subset=[Config.PRECIPITATION_COL])) < \
                   test_size_prophet + 24:
                    st.warning(f"Incluso después de completar, no hay suficientes datos para un pronóstico confiable.")
                else:
                    try:
                        from modules.forecasting import generate_prophet_forecast
                        from prophet.plot import plot_plotly
                        with st.spinner("Entrenando y evaluando modelo Prophet..."):
                            model, forecast, metrics = generate_prophet_forecast(ts_data_prophet,
                                                                                 forecast_horizon_prophet, test_size_prophet, regressors=None)

                        st.session_state['prophet_results'] = {'forecast': forecast[['ds', 'yhat']], 'metrics':
                                                               metrics}
                        st.markdown("##### Resultados del Pronóstico")
                        fig_prophet = plot_plotly(model, forecast)
                        st.plotly_chart(fig_prophet, use_container_width=True)
                        st.markdown("##### Evaluación del Modelo")
                        st.info(f"El modelo se evaluó usando los últimos **{test_size_prophet} meses** de datos históricos como conjunto de prueba.")
                        m1, m2 = st.columns(2)
                        m1.metric("RMSE", f"{metrics['RMSE']:.2f}")
                        m2.metric("MAE", f"{metrics['MAE']:.2f}")
                    except Exception as e:
                        st.error(f"No se pudo generar el pronóstico con Prophet. Error: {e}")

    with compare_forecast_tab:
        st.subheader("Comparación de Pronósticos: SARIMA vs Prophet")
        sarima_results = st.session_state.get('sarima_results')
        prophet_results = st.session_state.get('prophet_results')
        if not sarima_results or not prophet_results:
            st.warning("Debe generar un pronóstico SARIMA y Prophet en sus respectivas pestañas para poder compararlos.")
        else:
            fig_compare = go.Figure()
            if sarima_results.get('history') is not None:
                hist_data = sarima_results['history']
                fig_compare.add_trace(go.Scatter(x=hist_data.index, y=hist_data, mode='lines',
                                                 name='Histórico', line=dict(color='gray')))

            if sarima_results.get('forecast') is not None:
                sarima_fc = sarima_results['forecast']
                fig_compare.add_trace(go.Scatter(x=sarima_fc['ds'], y=sarima_fc['yhat'],
                                                 mode='lines', name='Pronóstico SARIMA', line=dict(color='red', dash='dash')))

            if prophet_results.get('forecast') is not None:
                prophet_fc = prophet_results['forecast']
                fig_compare.add_trace(go.Scatter(x=prophet_fc['ds'], y=prophet_fc['yhat'],
                                                 mode='lines', name='Pronóstico Prophet', line=dict(color='blue', dash='dash')))

            fig_compare.update_layout(title="Pronóstico Comparativo", xaxis_title="Fecha",
                                     yaxis_title="Precipitación (mm)", height=500, legend=dict(x=0.01, y=0.99))
            st.plotly_chart(fig_compare, use_container_width=True)

            st.markdown("#### Comparación de Precisión (sobre el conjunto de prueba)")
            sarima_metrics = sarima_results.get('metrics')
            prophet_metrics = prophet_results.get('metrics')
            if sarima_metrics and prophet_metrics:
                m_data = {'Métrica': ['RMSE', 'MAE'], 'SARIMA': [sarima_metrics['RMSE'],
                                                                 sarima_metrics['MAE']], 'Prophet': [prophet_metrics['RMSE'], prophet_metrics['MAE']]}
                metrics_df = pd.DataFrame(m_data)
                st.dataframe(metrics_df.style.format({'SARIMA': '{:.2f}', 'Prophet': '{:.2f}'}))
                rmse_winner = 'SARIMA' if sarima_metrics['RMSE'] < prophet_metrics['RMSE'] \
                    else 'Prophet'
                mae_winner = 'SARIMA' if sarima_metrics['MAE'] < prophet_metrics['MAE'] else \
                    'Prophet'
                st.success(f"**Ganador (menor error):** **{rmse_winner}** basado en RMSE y **{mae_winner}** basado en MAE.")
            else:
                st.info("Genere ambos pronósticos (SARIMA y Prophet) para ver la comparación de precisión.")

def display_downloads_tab(df_anual_melted, df_monthly_filtered, stations_for_analysis,
                             analysis_mode):
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
    st.header("Opciones de Descarga")
    if not stations_for_analysis:
        st.warning("Por favor, seleccione al menos una estación para activar las descargas.")
        return
    st.markdown("Aquí puedes descargar los datos actualmente visualizados, según los filtros aplicados en el panel de control.")
    st.markdown("---")
    st.markdown("#### Datos de Precipitación Anual (Filtrados)")
    if not df_anual_melted.empty:
        csv_anual = convert_df_to_csv(df_anual_melted)
        st.download_button(label="  Descargar CSV Anual", data=csv_anual,
                           file_name='precipitacion_anual_filtrada.csv', mime='text/csv', key='download-anual')
    else:
        st.info("No hay datos anuales para descargar con los filtros actuales.")
    st.markdown("---")
    if analysis_mode == "Completar series (interpolación)":
        st.markdown("#### Datos de Series Mensuales Completas (Interpoladas)")
        st.info("Los datos a continuación han sido completados (interpolados) para rellenar los vacíos en las series de tiempo.")
        csv_completed = convert_df_to_csv(df_monthly_filtered)
        st.download_button(label="  Descargar CSV de Series Completas",
                           data=csv_completed, file_name='precipitacion_mensual_completa.csv', mime='text/csv',
                           key='download-completed')
    else:
        st.markdown("#### Datos de Precipitación Mensual (Originales Filtrados)")
        if not df_monthly_filtered.empty:
            csv_mensual = convert_df_to_csv(df_monthly_filtered)
            st.download_button(label="  Descargar CSV Mensual", data=csv_mensual,
                               file_name='precipitacion_mensual_filtrada.csv', mime='text/csv', key='download-mensual')
        else:
            st.info("No hay datos mensuales para descargar con los filtros actuales.")
            
def display_station_table_tab(gdf_filtered, df_anual_melted, df_monthly_filtered,
                              stations_for_analysis, **kwargs):
    st.header("Información Detallada de las Estaciones")
    if not stations_for_analysis:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return

    st.info("Presiona el botón para generar una tabla detallada con estadísticas calculadas para cada estación seleccionada.")

    if st.button("Calcular Estadísticas Detalladas"):
        with st.spinner("Realizando cálculos, por favor espera..."):
            try:
                # La función auxiliar para el cálculo de estadísticas completas debe estar en el visualizador o importada
                @st.cache_data
                def calculate_comprehensive_stats(_df_anual, _df_monthly, _stations):
                    """Calcula un conjunto completo de estadísticas para cada estación seleccionada."""
                    results = []
                    # Importamos numpy y scipy/pymannkendall localmente para robustez
                    import numpy as np
                    from scipy import stats
                    try:
                        import pymannkendall as mk
                    except ImportError:
                        st.warning("La librería pymannkendall no está disponible para cálculo de tendencias.")
                        mk = None
                    
                    for station in _stations:
                        stats_dict = {"Estación": station}
                        station_anual = _df_anual[_df_anual[Config.STATION_NAME_COL] ==
                                                  station].dropna(subset=[Config.PRECIPITATION_COL])
                        station_monthly = _df_monthly[_df_monthly[Config.STATION_NAME_COL] ==
                                                      station].dropna(subset=[Config.PRECIPITATION_COL])

                        if not station_anual.empty:
                            stats_dict['Años con Datos'] = int(station_anual[Config.PRECIPITATION_COL].count())
                            stats_dict['Ppt. Media Anual (mm)'] = station_anual[Config.PRECIPITATION_COL].mean()
                            stats_dict['Desv. Estándar Anual (mm)'] = station_anual[Config.PRECIPITATION_COL].std()
                            
                            max_anual_row = \
                                station_anual.loc[station_anual[Config.PRECIPITATION_COL].idxmax()]
                            stats_dict['Ppt. Máxima Anual (mm)'] = max_anual_row[Config.PRECIPITATION_COL]
                            stats_dict['Año Ppt. Máxima'] = int(max_anual_row[Config.YEAR_COL])

                            min_anual_row = \
                                station_anual.loc[station_anual[Config.PRECIPITATION_COL].idxmin()]
                            stats_dict['Ppt. Mínima Anual (mm)'] = min_anual_row[Config.PRECIPITATION_COL]
                            stats_dict['Año Ppt. Mínima'] = int(min_anual_row[Config.YEAR_COL])

                            if len(station_anual) >= 4 and mk is not None:
                                mk_result = mk.original_test(station_anual[Config.PRECIPITATION_COL])
                                stats_dict['Tendencia (mm/año)'] = mk_result.slope
                                stats_dict['Significancia (p-valor)'] = mk_result.p
                            else:
                                stats_dict['Tendencia (mm/año)'] = np.nan
                                stats_dict['Significancia (p-valor)'] = np.nan

                        if not station_monthly.empty:
                            meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
                            monthly_means = \
                                station_monthly.groupby(station_monthly[Config.DATE_COL].dt.month)[Config.PRECIPITATION_COL].mean()
                            for i, mes in enumerate(meses, 1):
                                stats_dict[f'Ppt Media {mes} (mm)'] = monthly_means.get(i, 0)
                        
                        results.append(stats_dict)
                    
                    return pd.DataFrame(results)


                detailed_stats_df = calculate_comprehensive_stats(df_anual_melted,
                                                                  df_monthly_filtered, stations_for_analysis)
                
                base_info_df = gdf_filtered[[Config.STATION_NAME_COL, Config.ALTITUDE_COL,
                                             Config.MUNICIPALITY_COL, Config.REGION_COL]].copy()
                base_info_df.rename(columns={Config.STATION_NAME_COL: 'Estación'},
                                    inplace=True)

                # Corregir el merge: Usar 'Estación' como columna de unión
                final_df = pd.merge(base_info_df.drop_duplicates(subset=['Estación']),
                                    detailed_stats_df, on="Estación", how="right")

                # Asegurar que las columnas de Config coincidan con los nombres reales antes de ordenar
                column_order = [
                    'Estación', Config.MUNICIPALITY_COL, Config.REGION_COL, Config.ALTITUDE_COL,
                    'Años con Datos', 'Ppt. Media Anual (mm)', 'Desv. Estándar Anual (mm)', 
                    'Ppt. Máxima Anual (mm)', 'Año Ppt. Máxima', 'Ppt. Mínima Anual (mm)', 
                    'Año Ppt. Mínima', 'Tendencia (mm/año)', 'Significancia (p-valor)', 
                    'Ppt Media Ene (mm)', 'Ppt Media Feb (mm)', 'Ppt Media Mar (mm)', 
                    'Ppt Media Abr (mm)', 'Ppt Media May (mm)', 'Ppt Media Jun (mm)', 
                    'Ppt Media Jul (mm)', 'Ppt Media Ago (mm)', 'Ppt Media Sep (mm)', 
                    'Ppt Media Oct (mm)', 'Ppt Media Nov (mm)', 'Ppt Media Dic (mm)'
                ]
                
                # Filtrar solo las columnas que existen en final_df
                display_columns = [col for col in column_order if col in final_df.columns]
                final_df_display = final_df[display_columns]
                
                # Formateo de los datos para la visualización
                format_dict = {
                    'Ppt. Media Anual (mm)': '{:.1f}', 'Desv. Estándar Anual (mm)': '{:.1f}', 
                    'Ppt. Máxima Anual (mm)': '{:.1f}', 'Ppt. Mínima Anual (mm)': '{:.1f}', 
                    'Tendencia (mm/año)': '{:.2f}', 'Significancia (p-valor)': '{:.3f}', 
                    'Ppt Media Ene (mm)': '{:.1f}', 'Ppt Media Feb (mm)': '{:.1f}', 
                    'Ppt Media Mar (mm)': '{:.1f}', 'Ppt Media Abr (mm)': '{:.1f}', 
                    'Ppt Media May (mm)': '{:.1f}', 'Ppt Media Jun (mm)': '{:.1f}', 
                    'Ppt Media Jul (mm)': '{:.1f}', 'Ppt Media Ago (mm)': '{:.1f}', 
                    'Ppt Media Sep (mm)': '{:.1f}', 'Ppt Media Oct (mm)': '{:.1f}', 
                    'Ppt Media Nov (mm)': '{:.1f}', 'Ppt Media Dic (mm)': '{:.1f}'
                }

                st.dataframe(final_df_display.style.format({k: v for k, v in format_dict.items() if k in final_df_display.columns}))

            except Exception as e:
                st.error(f"Ocurrió un error al calcular las estadísticas: {e}")


def display_weekly_forecast_tab(stations_for_analysis, gdf_filtered):
    st.header("Pronóstico del Tiempo a 7 Días (Open-Meteo)")

    if not stations_for_analysis:
        st.warning("Seleccione al menos una estación para obtener el pronóstico.")
        return

    station_options = sorted(stations_for_analysis)
    selected_station = st.selectbox(
        "Seleccione una estación para el pronóstico:",
        options=station_options,
        key="weekly_forecast_station"
    )

    # Botón para obtener/actualizar pronóstico
    if st.button("Obtener/Actualizar Pronóstico", key="get_forecast_button"):
        if selected_station:
            station_info = gdf_filtered[gdf_filtered[Config.STATION_NAME_COL] == selected_station].iloc[0]
            lat = station_info.geometry.y
            lon = station_info.geometry.x

            with st.spinner(f"Obteniendo pronóstico para {selected_station}..."):
                # Llama a la función MODIFICADA de forecasting.py
                forecast_df_new = get_weather_forecast(lat, lon) 

                if forecast_df_new is not None and not forecast_df_new.empty:
                    st.session_state['forecast_df'] = forecast_df_new # Guarda el DF en sesión
                    st.session_state['forecast_station_name'] = selected_station
                    st.success("Pronóstico obtenido con éxito.")
                else:
                    # El error ya se muestra dentro de get_weather_forecast
                    st.session_state['forecast_df'] = None # Limpia si falla
                    st.session_state['forecast_station_name'] = None
        else:
             st.warning("Por favor, seleccione una estación primero.")


    # Mostrar pronóstico si está en la sesión
    if 'forecast_df' in st.session_state and st.session_state.forecast_df is not None:
        forecast_df = st.session_state['forecast_df']
        station_name = st.session_state['forecast_station_name']

        st.subheader(f"Pronóstico para los próximos 7 días en: {station_name}")

        # --- Tabla con Nuevas Variables ---
        display_df = forecast_df.copy()
        
        # Generamos un rango de fechas limpio por si acaso
        try:
             start_date = pd.to_datetime(display_df['date'].iloc[0])
             display_df['date_corrected'] = pd.date_range(start=start_date, periods=len(display_df))
        except: # Si falla la conversión, usa el índice como último recurso
             display_df['date_corrected'] = display_df.index

        # Formatear columnas para la tabla
        display_df['Fecha'] = display_df['date_corrected'].dt.strftime('%A, %d %b') # Formato más corto
        
        # Diccionario para renombrar y seleccionar columnas
        column_rename_map = {
            'Fecha': 'Fecha',
            'temperature_2m_max': 'T. Máx (°C)',
            'temperature_2m_min': 'T. Mín (°C)',
            'precipitation_sum': 'Ppt. (mm)',
            'relative_humidity_2m_mean': 'HR Media (%)',
            'surface_pressure_mean': 'Presión (hPa)',
            'et0_fao_evapotranspiration': 'ET₀ (mm)',
            'shortwave_radiation_sum': 'Radiación SW (MJ/m²)',
            'wind_speed_10m_max': 'Viento Máx (km/h)' # Asumiendo km/h, verificar unidad API
        }
        
        # Seleccionar y renombrar solo las columnas que existen en el DataFrame
        cols_to_display = ['Fecha'] + [col for col in column_rename_map if col in display_df.columns and col != 'Fecha']
        df_for_table = display_df[cols_to_display].copy()
        df_for_table.rename(columns=column_rename_map, inplace=True)

        # Diccionario de formato para la tabla
        format_dict = {
             'T. Máx (°C)': '{:.1f}', 'T. Mín (°C)': '{:.1f}', 'Ppt. (mm)': '{:.1f}',
             'HR Media (%)': '{:.0f}', 'Presión (hPa)': '{:.1f}', 'ET₀ (mm)': '{:.2f}',
             'Radiación SW (MJ/m²)': '{:.1f}', 'Viento Máx (km/h)': '{:.1f}'
        }
        
        # Aplicar formato solo a columnas existentes
        valid_format_dict = {k: v for k, v in format_dict.items() if k in df_for_table.columns}

        st.dataframe(
            df_for_table.set_index('Fecha').style.format(valid_format_dict),
            use_container_width=True
        )

        # --- Gráfico Principal (Temperatura y Precipitación) ---
        st.markdown("---")
        st.subheader("Gráfico de Temperatura y Precipitación")
        fig_temp_ppt = make_subplots(specs=[[{"secondary_y": True}]])

        # Temperaturas (eje Y primario)
        if 'temperature_2m_max' in display_df.columns:
            fig_temp_ppt.add_trace(go.Scatter(
                x=display_df['date_corrected'], y=display_df['temperature_2m_max'],
                name='Temp. Máxima', mode='lines+markers', line=dict(color='red')
            ), secondary_y=False)
        if 'temperature_2m_min' in display_df.columns:
             fig_temp_ppt.add_trace(go.Scatter(
                x=display_df['date_corrected'], y=display_df['temperature_2m_min'],
                name='Temp. Mínima', mode='lines+markers', line=dict(color='blue'),
                fill='tonexty', fillcolor='rgba(173, 216, 230, 0.2)'
            ), secondary_y=False)

        # Precipitación (eje Y secundario)
        if 'precipitation_sum' in display_df.columns:
             fig_temp_ppt.add_trace(go.Bar(
                x=display_df['date_corrected'], y=display_df['precipitation_sum'],
                name='Precipitación', marker_color='lightblue', opacity=0.7
            ), secondary_y=True)

        fig_temp_ppt.update_layout(
            #title_text=f"Pronóstico de Temperatura y Precipitación", # Título redundante
            xaxis_title="Fecha",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        # Ajustar rangos de ejes si hay datos
        if 'temperature_2m_min' in display_df.columns and 'temperature_2m_max' in display_df.columns:
             min_temp = display_df['temperature_2m_min'].min()
             max_temp = display_df['temperature_2m_max'].max()
             if pd.notna(min_temp) and pd.notna(max_temp):
                 fig_temp_ppt.update_yaxes(title_text="Temperatura (°C)", secondary_y=False, range=[min_temp - 2, max_temp + 2])
             else:
                  fig_temp_ppt.update_yaxes(title_text="Temperatura (°C)", secondary_y=False)
        else:
             fig_temp_ppt.update_yaxes(title_text="Temperatura (°C)", secondary_y=False)

        fig_temp_ppt.update_yaxes(title_text="Precipitación (mm)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig_temp_ppt, use_container_width=True)

        # --- Gráficos Adicionales (Opcional) ---
        st.markdown("---")
        st.subheader("Gráficos Adicionales")
        
        # Crear columnas para poner gráficos lado a lado
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            # Gráfico de Humedad y ET₀
            fig_hr_et = make_subplots(specs=[[{"secondary_y": True}]])
            if 'relative_humidity_2m_mean' in display_df.columns:
                 fig_hr_et.add_trace(go.Scatter(
                      x=display_df['date_corrected'], y=display_df['relative_humidity_2m_mean'],
                      name='HR Media (%)', mode='lines+markers', line=dict(color='green')
                 ), secondary_y=False)
            if 'et0_fao_evapotranspiration' in display_df.columns:
                  fig_hr_et.add_trace(go.Scatter(
                      x=display_df['date_corrected'], y=display_df['et0_fao_evapotranspiration'],
                      name='ET₀ (mm)', mode='lines+markers', line=dict(color='orange', dash='dot')
                 ), secondary_y=True)
            fig_hr_et.update_layout(title="Humedad Relativa y ET₀", xaxis_title="Fecha")
            fig_hr_et.update_yaxes(title_text="Humedad Relativa (%)", secondary_y=False)
            fig_hr_et.update_yaxes(title_text="ET₀ (mm)", secondary_y=True, showgrid=False)
            st.plotly_chart(fig_hr_et, use_container_width=True)

        with col_g2:
             # Gráfico de Viento y Radiación
             fig_wind_rad = make_subplots(specs=[[{"secondary_y": True}]])
             if 'wind_speed_10m_max' in display_df.columns:
                  fig_wind_rad.add_trace(go.Scatter(
                      x=display_df['date_corrected'], y=display_df['wind_speed_10m_max'],
                      name='Viento Máx (km/h)', mode='lines+markers', line=dict(color='purple')
                 ), secondary_y=False)
             if 'shortwave_radiation_sum' in display_df.columns:
                  fig_wind_rad.add_trace(go.Bar(
                       x=display_df['date_corrected'], y=display_df['shortwave_radiation_sum'],
                       name='Radiación SW (MJ/m²)', marker_color='gold', opacity=0.6
                 ), secondary_y=True)
             fig_wind_rad.update_layout(title="Viento y Radiación Solar", xaxis_title="Fecha")
             fig_wind_rad.update_yaxes(title_text="Velocidad Viento (km/h)", secondary_y=False)
             fig_wind_rad.update_yaxes(title_text="Radiación SW (MJ/m²)", secondary_y=True, showgrid=False)
             st.plotly_chart(fig_wind_rad, use_container_width=True)

    # Si no hay pronóstico en la sesión, muestra un mensaje
    elif 'forecast_df' not in st.session_state or st.session_state.forecast_df is None:
        st.info("Presiona el botón 'Obtener/Actualizar Pronóstico' para ver los datos de la estación seleccionada.")

# --- Función para Mapas Climáticos Adicionales ---
def display_additional_climate_maps_tab(gdf_filtered, **kwargs):
    st.header("Mapas de Variables Climáticas Adicionales (Open-Meteo)")
    st.info("Estos mapas usan datos históricos promediados de Open-Meteo para las ubicaciones de las estaciones seleccionadas y los interpolan.")

    if gdf_filtered.empty:
        st.warning("Seleccione al menos una estación en el panel de control para ver estos mapas.")
        return

    # Diccionario de variables disponibles en la API diaria histórica de Open-Meteo
    variables_openmeteo = {
        "Velocidad Media del Viento (10m)": "wind_speed_10m_mean",
        "Temperatura Media del Aire (2m)": "temperature_2m_mean",
        "Radiación Global Diaria (Onda Corta)": "shortwave_radiation_sum",
        "Evapotranspiración de Referencia (ET₀)": "et0_fao_evapotranspiration",
        "Humedad Relativa Media (2m)": "relative_humidity_2m_mean",
        "Presión a Nivel del Mar": "pressure_msl_mean" # Mantener MSL si es lo que usas
    }
    
    # --- AÑADIR ESTE DICCIONARIO DE UNIDADES ---
    variable_units = {
        "Velocidad Media del Viento (10m)": "(km/h)", # O (m/s) - Verifica la API
        "Temperatura Media del Aire (2m)": "(°C)",
        "Radiación Global Diaria (Onda Corta)": "(MJ/m²)",
        "Evapotranspiración de Referencia (ET₀)": "(mm)",
        "Humedad Relativa Media (2m)": "(%)",
        "Presión a Nivel del Mar": "(hPa)"
    }
    
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        selected_variable_name = st.selectbox(
            "Seleccione la Variable Climática:", 
            list(variables_openmeteo.keys()),
            key="additional_var_select"
        )
    with col2:
        # Usa el rango de años de la sesión
        year_range = st.session_state.get('year_range', (2000, 2020))
        start_year = st.number_input("Año Inicio (Promedio):", min_value=1940, max_value=date.today().year, value=year_range[0], key="clim_start_year")
    with col3:
        end_year = st.number_input("Año Fin (Promedio):", min_value=1940, max_value=date.today().year, value=year_range[1], key="clim_end_year")

    if start_year > end_year:
        st.error("El año de inicio debe ser menor o igual al año de fin.")
        return

    variable_code = variables_openmeteo[selected_variable_name]
    interpolation_method_clim = st.radio(
        "Método de Interpolación:", 
        ("IDW (Rápido)", "Spline (Suave)"), 
        horizontal=True, 
        key="clim_interp_method"
    )

    if st.button(f"Generar Mapa Promedio ({start_year}-{end_year}) para {selected_variable_name}", key="gen_clim_map_btn"):
        st.session_state['last_climate_data'] = None # Clear previous data
        with st.spinner(f"Obteniendo datos de {start_year}-{end_year} y generando mapa..."):
            
            # Prepara coordenadas únicas para la API
            gdf_unique_coords = gdf_filtered.drop_duplicates(subset=['geometry'])
            lats = gdf_unique_coords.geometry.y.tolist()
            lons = gdf_unique_coords.geometry.x.tolist()
            
            start_date_str = f"{start_year}-01-01" 
            end_date_str = f"{end_year}-12-31" 
            
            # Llama a la función de la API para obtener los promedios
            df_climate_data = get_historical_climate_average(lats, lons, variable_code, start_date_str, end_date_str)
            # Store the fetched data in session state for download
            if df_climate_data is not None and not df_climate_data.empty:
                st.session_state['last_climate_data'] = df_climate_data
                st.session_state['last_climate_variable'] = selected_variable_name
                st.session_state['last_climate_period'] = f"{start_year}-{end_year}"

            if df_climate_data is not None and not df_climate_data.empty:
                
                # Prepara lons, lats, vals para interpolar (usando WGS84 - EPSG:4326)
                lons_data = df_climate_data['longitude'].values
                lats_data = df_climate_data['latitude'].values
                vals_data = df_climate_data['valor_promedio'].values 

                if len(vals_data) >= 4:
                    # Define la grilla de interpolación (en WGS84)
                    bounds = gdf_filtered.total_bounds # gdf_filtered está en WGS84
                    grid_lon = np.linspace(bounds[0] - 0.1, bounds[2] + 0.1, 100) # Menor resolución
                    grid_lat = np.linspace(bounds[1] - 0.1, bounds[3] + 0.1, 100)
                    
                    # Selecciona el método
                    method_call = 'cubic' if interpolation_method_clim == "Spline (Suave)" else 'linear'
                    
                    # Interpola usando griddata (más flexible que interpolate_idw)
                    grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)
                    points = np.column_stack((lons_data, lats_data))
                    
                    try:
                        z_grid = griddata(points, vals_data, (grid_x, grid_y), method=method_call)
                        # Rellenar NaNs con el vecino más cercano si es necesario
                        nan_mask = np.isnan(z_grid)
                        if np.any(nan_mask):
                             fill_values = griddata(points, vals_data, (grid_x[nan_mask], grid_y[nan_mask]), method='nearest')
                             z_grid[nan_mask] = fill_values
                        z_grid = np.nan_to_num(z_grid) # Asegura que no queden NaNs

                        # Crea la figura con go.Contour
                        # --- MODIFICAR LA CREACIÓN DE LA FIGURA ---
                        # Obtener la unidad
                        unit = variable_units.get(selected_variable_name, "") # Obtiene la unidad, "" si no se encuentra
                        
                        fig = go.Figure(data=go.Contour(
                            z=z_grid.T, 
                            x=grid_lon, 
                            y=grid_lat,
                            colorscale='viridis', 
                            # Modifica esta línea para añadir la unidad:
                            colorbar_title=f"{selected_variable_name} {unit}", 
                            contours=dict(showlabels=True, labelfont=dict(size=10, color='white')),
                            line_smoothing=0.85
                        ))
                        # --- FIN DE LA MODIFICACIÓN DE LA FIGURA ---
                        # Añade los puntos de las estaciones originales con sus valores
                        fig.add_trace(go.Scatter(
                            x=lons_data, y=lats_data, mode='markers', 
                            marker=dict(color='red', size=5, line=dict(width=1, color='black')),
                            name='Estaciones',
                            hoverinfo='text',
                            text=[f"Valor: {val:.2f}" for val in vals_data] # Texto simple para el hover
                        ))
                        fig.update_layout(
                            title=f"Promedio ({start_year}-{end_year}) de {selected_variable_name}",
                            xaxis_title="Longitud", yaxis_title="Latitud", height=600
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                         st.error(f"Error durante la interpolación o graficación: {e}")

                else:
                    st.warning("No hay suficientes datos válidos (<4) devueltos por la API para interpolar.")
            else:
                st.error(f"No se pudieron obtener datos históricos para '{variable_code}' de la API.")

# --- ADD THIS BLOCK OUTSIDE (AFTER) THE if st.button(...) BLOCK ---
    # Display download button if data exists in session state
    if 'last_climate_data' in st.session_state and st.session_state['last_climate_data'] is not None:
        st.markdown("---")
        st.subheader("Descargar Datos Climáticos")
        
        # Use a helper function (or define one if needed) to convert DataFrame to CSV
        @st.cache_data # Cache the conversion
        def convert_df_to_csv_bytes(df):
            return df.to_csv(index=False, sep=';').encode('utf-8') # Use semicolon separator

        csv_data = convert_df_to_csv_bytes(st.session_state['last_climate_data'])
        
        var_name_safe = st.session_state['last_climate_variable'].replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")
        period_safe = st.session_state['last_climate_period']
        
        st.download_button(
            label=f"Descargar Datos ({st.session_state['last_climate_variable']} {period_safe})",
            data=csv_data,
            file_name=f"datos_climaticos_{var_name_safe}_{period_safe}.csv",
            mime="text/csv",
            key="download_climate_data_btn"
        )
    # --- END ADD ---

# --- Función para Imágenes Satelitales ---
def display_satellite_imagery_tab(gdf_filtered, **kwargs):
    st.header("Imágenes Satelitales")
    st.info("Visualiza capas WMS de servicios meteorológicos. La disponibilidad y actualización dependen del proveedor.")

    # --- Configuración de Capas WMS (ACTUALIZADA) ---
    # Usando la información proporcionada
    wms_layers_options = {
        "GOES-East B13 Full Disk (SSEC/Wisc)": {
            "url": "https://sats-ftp.ssec.wisc.edu/wms/wms_goes_east.cgi?", # VERIFICAR si aún funciona
            "layers": "goes_east_abi_b13_fd", # VERIFICAR si el nombre es correcto
            "fmt": 'image/png',
            "transparent": True,
            "attr": "NOAA / SSEC-UWisc",
        },
        "IDEAM - GOES B13 (Nombre a Verificar)": {
             "url": "http://geoapps.ideam.gov.co:8080/geoserver/wms?", # VERIFICAR si el servicio está activo
             "layers": "ideam:goes16_abi_band13", # ¡NECESITA VERIFICACIÓN URGENTE! Buscar nombre real en GetCapabilities
             "fmt": 'image/png',
             "transparent": True,
             "attr": "IDEAM",
        },
        "EUMETSAT - Meteosat IR 10.8 (Ejemplo)": {
             "url": "https://eumetview.eumetsat.int/geoserv/wms", # VERIFICAR si aún funciona
             "layers": "meteosat:msg_ir108", # VERIFICAR si el nombre es correcto
             "fmt": 'image/png',
             "transparent": True,
             "attr": "EUMETSAT",
        },
         "EUMETSAT - Meteosat Vapor de Agua 6.2 (Ejemplo)": {
             "url": "https://eumetview.eumetsat.int/geoserv/wms", # VERIFICAR si aún funciona
             "layers": "meteosat:msg_wv062", # VERIFICAR si el nombre es correcto
             "fmt": 'image/png',
             "transparent": True,
             "attr": "EUMETSAT",
        },
    }
    # --- Fin Configuración ---

    # Columnas para controles y mapa
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("##### Opciones de Visualización")

        # --- Define base map options directly here ---
        base_map_options = {
            "CartoDB Positron": {"tiles": "cartodbpositron", "attr": "CartoDB"},
            "OpenStreetMap": {"tiles": "OpenStreetMap", "attr": "OpenStreetMap"},
            "Topografía (Open TopoMap)": {
                "tiles": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
                "attr": "Open TopoMap"
            },
            # Add other base maps if you used them elsewhere in display_map_controls
        }
        # --- Create only the base map selectbox directly ---
        selected_base_map_name = st.selectbox(
            "Seleccionar Mapa Base",
            list(base_map_options.keys()),
            key="satellite_base_map" # Ensure this unique key is used ONLY here in this tab
        )
        selected_base_map_config = base_map_options[selected_base_map_name]
        # --- End direct widget creation ---

        # Selector de capas WMS
        selected_wms_names = st.multiselect(
            "Seleccionar Capas Satelitales:",
            options=list(wms_layers_options.keys()),
            # Select the SSEC layer by default if it exists
            default=[list(wms_layers_options.keys())[0]] if wms_layers_options else []
        )

    with col2:
        # Crea el mapa base
        m = create_folium_map(
            location=[4.6, -74.0], # Centrado en Colombia
            zoom=5,
            base_map_config=selected_base_map_config, # Usa la selección directa
            overlays_config=[], # Los WMS se añaden manualmente
            fit_bounds_data=gdf_filtered if not gdf_filtered.empty else None
        )

        # Añade las capas WMS seleccionadas
        added_layers = False
        for name in selected_wms_names:
            if name in wms_layers_options:
                config = wms_layers_options[name]
                try:
                    WmsTileLayer(
                        url=config["url"],
                        layers=config["layers"],
                        fmt=config.get("fmt", 'image/png'),
                        transparent=config.get("transparent", True),
                        overlay=True, # Importante para que sea una capa superpuesta
                        control=True, # Para que aparezca en el control de capas
                        name=name, # Nombre en el control de capas
                        attr=config.get("attr", name)
                    ).add_to(m)
                    added_layers = True
                except Exception as e:
                    # Muestra un error si una capa específica falla, pero continúa
                    st.error(f"No se pudo añadir la capa '{name}'. Verifica la URL ('{config['url']}') y el nombre de la capa ('{config['layers']}'). Error: {e}")

        # Mensajes informativos
        if not added_layers and selected_wms_names:
             st.warning("No se pudo añadir ninguna de las capas WMS seleccionadas. Verifica la configuración o las URLs.")
        elif not selected_wms_names:
             st.info("Selecciona al menos una capa satelital para visualizar.")

        # Añade control de capas si se añadió alguna
        if added_layers:
            folium.LayerControl().add_to(m)

        # Muestra el mapa
        folium_static(m, height=700, width=None)

def display_land_cover_analysis_tab(gdf_filtered, **kwargs):
    st.header("Análisis de Cobertura del Suelo por Cuenca")

    # --- Configuración ---
    land_cover_raster_filename = "Cob25m_WGS84.tif"
    
    # --- LEYENDA ACTUALIZADA ---
    land_cover_legend = {
        1: "Zonas urbanizadas",
        2: "Zonas industriales o comerciales y redes de comunicacion",
        3: "Zonas de extraccion minera, escombreras y vertederos",
        4: "Zonas verdes artificializadas, no agricolas",
        5: "Cultivos transitorios",
        6: "Cultivos permanentes",
        7: "Pastos",
        8: "Areas Agricolas Heterogeneas",
        9: "Bosques",
        10: "Areas con vegetación herbácea y/o arbustiva",
        11: "Areas abiertas, sin o con poca vegetacion",
        12: "Areas húmedas continentales",
        13: "Aguas continentales",
        # Asegúrate de saber cuál es el valor NoData de tu raster y añádelo si es necesario
        # 0: "Sin Datos / Fuera de Área" # Ejemplo si 0 es NoData
    }
    # --- FIN LEYENDA ---
    
# Inside display_land_cover_analysis_tab

    # --- Configuración ---
    land_cover_raster_filename = "Cob25m_WGS84.tif" 
    projected_crs = "EPSG:3116" 
    # --- Fin Configuración ---

    # --- DEFINIR RUTA PRIMERO ---
    # Construir ruta al raster
    _THIS_FILE_DIR = os.path.dirname(__file__)
    land_cover_raster_path = os.path.abspath(os.path.join(_THIS_FILE_DIR, '..', 'data', land_cover_raster_filename))

    # Mensaje informativo
    st.info(f"Se utilizará el archivo raster de coberturas: '{os.path.basename(land_cover_raster_path)}'.")

    # Obtener la cuenca unificada de la sesión
    unified_basin_gdf = st.session_state.get('unified_basin_gdf')
    basin_name = st.session_state.get('selected_basins_title', 'Cuenca Seleccionada')

    if unified_basin_gdf is None or unified_basin_gdf.empty:
        st.warning("Primero debes generar un mapa para una cuenca específica en la pestaña 'Mapas Avanzados -> Superficies de Interpolación'.")
        return

    # Verificar existencia del raster de cobertura (CORREGIDO)
    if not os.path.exists(land_cover_raster_path):
        st.error(f"No se encontró el archivo raster de coberturas en la ruta: {land_cover_raster_path}")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(f"Cobertura del Suelo en: {basin_name}")
        try:
            with st.spinner("Cargando y procesando raster de coberturas..."):
                # Abrir el raster de cobertura
                with rasterio.open(land_cover_raster_path) as cover_src:
                    cover_crs = cover_src.crs
                    cover_transform = cover_src.transform
                    # Leer NoData del archivo
                    nodata_val = cover_src.nodata
                    # Usar 0 como default interno si no hay NoData definido en el archivo
                    internal_nodata = nodata_val if nodata_val is not None else 0
                    # Asegurar que la leyenda tenga entrada para el nodata interno
                    if internal_nodata not in land_cover_legend:
                        land_cover_legend[internal_nodata] = "Sin Datos / NoData"

                    # Reproyectar la geometría de la cuenca al CRS del raster
                    basin_reproj = unified_basin_gdf.to_crs(cover_crs)

                    # Recortar (enmascarar) el raster
                    out_image, out_transform = mask(cover_src, basin_reproj.geometry, crop=True, nodata=internal_nodata, all_touched=True)

                    if out_image.ndim > 2:
                        out_image = out_image[0]

                    # Calcular estadísticas (excluyendo nodata INTERNO)
                    valid_pixel_mask = (out_image != internal_nodata)
                    unique_values, counts = np.unique(out_image[valid_pixel_mask], return_counts=True)

                    if unique_values.size == 0:
                        st.warning("No se encontraron píxeles de cobertura válidos dentro del área de la cuenca (después de excluir NoData).")
                        st.session_state['current_coverage_stats'] = None
                        return

                    # Calcular área por clase
                    pixel_size_x = abs(out_transform.a)
                    pixel_size_y = abs(out_transform.e)
                    # Verificar si CRS es geográfico (grados) para advertir sobre cálculo de área
                    is_geographic = cover_src.crs.is_geographic
                    if is_geographic:
                         st.warning("El CRS del raster de coberturas está en grados. El cálculo de área puede ser impreciso. Se recomienda usar un raster en CRS proyectado (métrico).")
                         # Usar área de píxel aproximada en m² (muy impreciso)
                         # Asumir ~111km por grado en el ecuador -> ~111000m
                         # Esto es solo un PALIATIVO, lo ideal es reproyectar el raster de entrada
                         pixel_area_m2 = (pixel_size_x * 111000) * (pixel_size_y * 111000)
                    else:
                         pixel_area_m2 = pixel_size_x * pixel_size_y # Área en unidades cuadradas del CRS (metros^2)

                    coverage_stats_list = []
                    total_valid_pixels = counts.sum()
                    total_area_m2_calc = total_valid_pixels * pixel_area_m2

                    for value, count in zip(unique_values, counts):
                        class_name = land_cover_legend.get(value, f"Código Desconocido ({value})") # <-- Aquí se usa la leyenda
                        area_m2 = count * pixel_area_m2
                        percentage = (count / total_valid_pixels) * 100 if total_valid_pixels > 0 else 0
                        coverage_stats_list.append({
                            "ID_Clase": value, "Tipo de Cobertura": class_name,
                            "area_m2": area_m2, "area_km2": area_m2 / 1_000_000,
                            "percentage": percentage
                        })

                    coverage_stats = pd.DataFrame(coverage_stats_list).sort_values(by="percentage", ascending=False)

                # Guardar resultados
                st.session_state['current_coverage_stats'] = coverage_stats
                st.session_state['total_basin_area_km2'] = total_area_m2_calc / 1_000_000

                # Mostrar tabla
                st.dataframe(coverage_stats[['Tipo de Cobertura', 'area_km2', 'percentage']]
                             .rename(columns={'area_km2': 'Área (km²)', 'percentage': 'Porcentaje (%)'})
                             .style.format({'Área (km²)': '{:.2f}', 'Porcentaje (%)': '{:.1f}%'}),
                             use_container_width=True)

        except Exception as e:
            st.error(f"Error al procesar el raster de coberturas: {e}")
            import traceback
            st.error(traceback.format_exc())
            st.session_state['current_coverage_stats'] = None
            return

    with col2:
        st.subheader("Visualización y Relación con Escorrentía")
        # --- Esta sección no necesita cambios ---
        if 'current_coverage_stats' in st.session_state and st.session_state['current_coverage_stats'] is not None:
            stats_df = st.session_state['current_coverage_stats']
            # Asegurarse que stats_df no esté vacío antes de graficar
            if not stats_df.empty:
                # Comprobar si hay códigos desconocidos ANTES de graficar
                if any("Código Desconocido" in name for name in stats_df["Tipo de Cobertura"]):
                     st.warning("Hay códigos de cobertura desconocidos en la cuenca. Revisa la leyenda `land_cover_legend` en el código.")
                
                fig_pie = px.pie(stats_df, names='Tipo de Cobertura', values='percentage',
                                 title=f"Distribución de Coberturas (%)", hole=0.3)
                fig_pie.update_traces(textposition='inside', textinfo='percent+label', sort=False)
                st.plotly_chart(fig_pie, use_container_width=True)

                # Mostrar Escorrentía (Código sin cambios)
                # ... (resto del código para mostrar Q) ...
                balance_results = st.session_state.get('balance_results')
                if balance_results and not balance_results.get("error"):
                    q_mm = balance_results.get('Q_mm'); q_m3 = balance_results.get('Q_m3_año')
                    if q_mm is not None and q_m3 is not None: st.metric("Escorrentía Media Anual Estimada (Q)",f"{q_mm:.0f} mm/año", f"Volumen: {q_m3 / 1e6:.2f} Millones m³/año")
                    else: st.info("La escorrentía no se calculó previamente.")
                else:
                     mean_precip = st.session_state.get('mean_precip'); morph_results = st.session_state.get('morph_results')
                     if mean_precip and morph_results and morph_results.get('alt_prom_m'):
                          temp_balance = calculate_hydrological_balance(mean_precip, morph_results.get('alt_prom_m'), unified_basin_gdf)
                          if temp_balance and temp_balance.get('Q_mm') is not None: st.metric("Escorrentía Media Anual Estimada (Q)", f"{temp_balance['Q_mm']:.0f} mm/año")
                          else: st.info("No se pudo recalcular la escorrentía.")
                     else: st.info("Ejecuta el Balance Hídrico para ver la escorrentía aquí.")
            else:
                # Esto puede ocurrir si todos los píxeles eran NoData
                st.info("No hay estadísticas de cobertura válidas para visualizar.")
        else:
            st.info("Procesa las coberturas primero para ver la visualización.")

    # --- Sección para Escenarios Hipotéticos (Parte 2) ---
    st.markdown("---")
    st.subheader("Modelado de Escenarios Hipotéticos de Cobertura")
    st.info("""
    Define un escenario hipotético de distribución de coberturas para la cuenca
    y estima el posible cambio en la escorrentía media anual.
    **Nota:** Esta es una estimación simplificada basada en el método del Número de Curva (SCS)
    y promedios anuales. Los resultados reales pueden variar.
    """)

    # Verificar si tenemos la escorrentía actual calculada
    balance_results = st.session_state.get('balance_results')
    q_actual_mm = None
    p_actual_mm = None
    if balance_results and not balance_results.get("error"):
        q_actual_mm = balance_results.get('Q_mm')
        p_actual_mm = balance_results.get('P_media_anual_mm')

    # Solo mostrar escenarios si tenemos Q y P actuales
    if q_actual_mm is not None and p_actual_mm is not None:

        st.markdown("##### Define los Porcentajes de Cobertura:")

        # --- Valores CN Base (EJEMPLOS - ¡Necesitan calibración local!) ---
        # Asumiendo Grupo Hidrológico de Suelo C (promedio/moderado)
        cn_values = {
            "Bosque (Buena condición)": 70, # CN más bajo, mayor infiltración
            "Pasto (Buena condición)": 74,
            "Cultivos (Contorno, buena condición)": 78,
            "Suelo Desnudo": 86, # CN alto, poca infiltración
            "Áreas Urbanas/Impermeables": 92 # CN muy alto
        }
        with st.expander("Ver/Editar Números de Curva (CN) Base"):
             # Permitir editar CN (avanzado)
             cn_bosque = st.number_input("CN Bosque", value=cn_values["Bosque (Buena condición)"], min_value=30, max_value=100)
             cn_pasto = st.number_input("CN Pasto", value=cn_values["Pasto (Buena condición)"], min_value=30, max_value=100)
             cn_cultivo = st.number_input("CN Cultivos", value=cn_values["Cultivos (Contorno, buena condición)"], min_value=30, max_value=100)
             cn_desnudo = st.number_input("CN Suelo Desnudo", value=cn_values["Suelo Desnudo"], min_value=30, max_value=100)
             cn_urbano = st.number_input("CN Urbano/Impermeable", value=cn_values["Áreas Urbanas/Impermeables"], min_value=30, max_value=100)
             # Actualizar diccionario con valores editados
             cn_values_edited = {
                 "Bosque": cn_bosque,
                 "Pasto": cn_pasto,
                 "Cultivos": cn_cultivo,
                 "Suelo Desnudo": cn_desnudo,
                 "Áreas Urbanas": cn_urbano
             }

        # Sliders para definir porcentajes
        # Usamos claves únicas para evitar conflictos si esta función se llama en otro lugar
        perc_bosque = st.slider("🌲 % Bosque", 0, 100, 20, key="perc_bosque")
        perc_pasto = st.slider("🌾 % Pasto", 0, 100, 20, key="perc_pasto")
        perc_cultivo = st.slider("🌽 % Cultivos", 0, 100, 20, key="perc_cultivo")
        perc_desnudo = st.slider("⛰️ % Suelo Desnudo", 0, 100, 20, key="perc_desnudo")
        perc_urbano = st.slider("🏘️ % Áreas Urbanas", 0, 100, 20, key="perc_urbano")

        total_perc = perc_bosque + perc_pasto + perc_cultivo + perc_desnudo + perc_urbano

        st.metric("Suma de Porcentajes", f"{total_perc:.1f}%")

        if not np.isclose(total_perc, 100.0):
            st.warning("La suma de los porcentajes debe ser 100%. Ajusta los sliders.")
        else:
            if st.button("Estimar Escorrentía del Escenario", key="estimate_scenario_q"):
                with st.spinner("Estimando escorrentía hipotética..."):
                    try:
                        # Usar CN editados si existen, sino los base
                        cn_dict = cn_values_edited if 'cn_values_edited' in locals() else cn_values
                        
                        # Calcular CN ponderado hipotético
                        cn_hip = (perc_bosque * cn_dict["Bosque"] +
                                  perc_pasto * cn_dict["Pasto"] +
                                  perc_cultivo * cn_dict["Cultivos"] +
                                  perc_desnudo * cn_dict["Suelo Desnudo"] +
                                  perc_urbano * cn_dict["Áreas Urbanas"]) / 100.0

                        # Calcular S (Almacenamiento potencial máximo) para el escenario
                        # Asegurar CN >= 1 para evitar división por cero o S negativo
                        cn_hip_safe = max(1, cn_hip) 
                        s_hip = (1000 / cn_hip_safe) - 10 # S en pulgadas
                        s_hip_mm = s_hip * 25.4 # Convertir S a mm

                        # Calcular Ia (Abstracción inicial) = 0.2 * S
                        ia_hip_mm = 0.2 * s_hip_mm

                        # Calcular Escorrentía Hipotética (Q_hip) usando la fórmula SCS
                        q_hip_mm = 0.0 # Escorrentía es 0 si P <= Ia
                        if p_actual_mm > ia_hip_mm:
                            q_hip_mm = ((p_actual_mm - ia_hip_mm)**2) / (p_actual_mm - ia_hip_mm + s_hip_mm)

                        # Calcular cambio porcentual
                        cambio_perc = ((q_hip_mm - q_actual_mm) / q_actual_mm) * 100 if q_actual_mm != 0 else np.inf

                        # Mostrar resultados
                        st.success("Estimación Completada:")
                        col_res1, col_res2, col_res3 = st.columns(3)
                        col_res1.metric("CN Ponderado del Escenario", f"{cn_hip:.1f}")
                        col_res2.metric("Escorrentía Estimada (Q)", f"{q_hip_mm:.0f} mm/año")
                        col_res3.metric("Cambio vs. Actual", f"{cambio_perc:.1f}%", delta_color=("inverse" if cambio_perc < 0 else "normal"))

                        st.caption(f"Cálculo basado en P = {p_actual_mm:.0f} mm/año y Q actual = {q_actual_mm:.0f} mm/año.")

                    except Exception as e_scen:
                        st.error(f"Error al calcular el escenario: {e_scen}")

    else:
        st.info("""
        Para modelar escenarios, primero calcula el Balance Hídrico
        en la pestaña 'Mapas Avanzados -> Superficies de Interpolación -> Por Cuenca Específica'
        para obtener la precipitación y escorrentía actuales.
        """)

def display_life_zones_tab(**kwargs):
    st.header("Clasificación de Zonas de Vida (Holdridge)")
    st.info("""
    Genera un mapa de Zonas de Vida basado en precipitación media anual y biotemperatura 
    estimada (desde DEM). Puedes ajustar la resolución y aplicar una máscara de cuenca.
    """)

    # --- Configuración ---
    precip_raster_filename = "PPAMAnt.tif"
    # --- Fin Configuración ---

    # Rutas y DEM
    _THIS_FILE_DIR = os.path.dirname(__file__)
    precip_raster_path = os.path.abspath(os.path.join(_THIS_FILE_DIR, '..', 'data', precip_raster_filename))
    dem_file_info = st.session_state.get('dem_file')

    # --- NUEVOS CONTROLES ---
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        # Selector de Resolución
        resolution_options = {
            "Baja (Rápido)": 8,
            "Media": 4, # Default
            "Alta (Lento)": 2,
            "Original (Muy Lento)": 1
        }
        selected_resolution = st.select_slider(
            "Seleccionar Resolución del Mapa:",
            options=list(resolution_options.keys()),
            value="Baja (Rápido)", # Default
            key="lifezone_resolution"
        )
        downscale_factor = resolution_options[selected_resolution]

    with col_ctrl2:
        # Opción de Máscara de Cuenca
        apply_basin_mask = st.toggle("Aplicar Máscara de Cuenca", value=True, key="lifezone_mask_toggle")
        basin_to_mask = None
        mask_geometry_to_use = None
        if apply_basin_mask:
            basin_to_mask = st.session_state.get('unified_basin_gdf')
            basin_name_mask = st.session_state.get('selected_basins_title')
            if basin_to_mask is not None and not basin_to_mask.empty:
                st.success(f"Se usará la máscara: {basin_name_mask}")
                mask_geometry_to_use = basin_to_mask.geometry # Pasar la geometría
            else:
                st.warning("No hay cuenca seleccionada en 'Mapas Avanzados' para usar como máscara.")
                apply_basin_mask = False # Desactivar si no hay cuenca
    # --- FIN CONTROLES ---

    dem_path = None
    temp_dem_filename = None # Guardar nombre para eliminar después
    if dem_file_info:
         temp_dem_filename = f"temp_dem_for_lifezones_{dem_file_info.name}"
         dem_path = os.path.join(os.getcwd(), temp_dem_filename)
         try:
             # Escribir solo si el archivo no existe o si cambió (simple check por nombre)
             # Una mejor opción sería hashear el contenido, pero esto es más simple
             if not os.path.exists(dem_path) or st.session_state.get('last_dem_used_for_lz') != dem_file_info.name:
                 with open(dem_path, "wb") as f:
                     f.write(dem_file_info.getbuffer())
                 st.session_state['last_dem_used_for_lz'] = dem_file_info.name
             # st.success(f"DEM '{dem_file_info.name}' listo.") # Mensaje menos intrusivo
         except Exception as e_write:
             st.error(f"No se pudo escribir el archivo DEM temporal: {e_write}")
             dem_path = None
    else:
        st.warning("Sube un archivo DEM en el panel lateral izquierdo para generar el mapa.")

    if not os.path.exists(precip_raster_path):
        st.error(f"No se encontró el archivo raster de precipitación: {precip_raster_path}")
        if dem_path and os.path.exists(dem_path): os.remove(dem_path)
        return

    # Botón para generar
    if dem_path and os.path.exists(precip_raster_path):
        if st.button("Generar Mapa de Zonas de Vida", key="gen_life_zone_map"):
            
            # Determinar qué máscara pasar (geometría de cuenca o nada)
            mask_arg = mask_geometry_to_use if apply_basin_mask else None
            
            # Calcular latitud promedio (o usar fija)
            mean_latitude = 6.5 # Fijo para Antioquia
            # Opcional: Calcular de la cuenca si se usa máscara
            # if apply_basin_mask and basin_to_mask is not None:
            #     try:
            #          mean_latitude = basin_to_mask.to_crs("EPSG:4326").geometry.centroid.y.iloc[0]
            #     except: pass # Mantener valor fijo si falla

            # Llamar a la función de cálculo pasando resolución y máscara
            classified_raster, output_profile, name_map = generate_life_zone_map(
                dem_path,
                precip_raster_path,
                # mean_latitude, # <-- ARGUMENTO ELIMINADO
                _mask_geometry=mask_arg, # <-- USAR _mask_geometry aquí también para consistencia con caché
                downscale_factor=downscale_factor
            )

            # Limpiar DEM temporal (AHORA se hace aquí, después de la llamada)
            if temp_dem_filename and os.path.exists(dem_path):
                 try: os.remove(dem_path); st.session_state['last_dem_used_for_lz'] = None
                 except Exception as e_del: st.warning(f"No se pudo eliminar el DEM temporal: {e_del}")

            # --- Visualización con Plotly Heatmap (Versión Estable) ---
            if classified_raster is not None and output_profile is not None and name_map is not None:
                st.subheader("Mapa de Zonas de Vida Generado")

                # Obtener metadatos del perfil del raster CLASIFICADO y REESCALADO
                height, width = classified_raster.shape
                crs = output_profile.get('crs', 'EPSG:???')
                nodata_val = output_profile.get('nodata', 0)
                transform = rasterio.transform.Affine(*output_profile['transform'][:6])

                # --- Generación de Coordenadas (Estándar) ---
                x_start, y_start = transform.c, transform.f
                x_end = x_start + transform.a * width; y_end = y_start + transform.e * height
                x_coords = np.linspace(x_start + transform.a / 2, x_end - transform.a / 2, width)
                y_coords_raw = np.linspace(y_start + transform.e / 2, y_end - transform.e / 2, height)
                # --- Fin Coordenadas ---

                # --- Lógica para Leyenda y Colores (IDs) ---
                unique_zones_present = np.unique(classified_raster)
                present_zone_ids = sorted([zone_id for zone_id in unique_zones_present if zone_id != nodata_val])
                fig = None
                if not present_zone_ids:
                    st.warning("No se encontraron zonas de vida clasificadas en el área.")
                else:
                    tick_values = present_zone_ids
                    tick_texts = [str(val) for val in tick_values] # Leyenda con IDs
                    color_palette1=px.colors.qualitative.Plotly; color_palette2=px.colors.qualitative.Alphabet
                    color_palette_combined=color_palette1 + color_palette2
                    if len(present_zone_ids)>len(color_palette_combined): color_palette_combined=color_palette_combined*(len(present_zone_ids)//len(color_palette_combined)+1)
                    zone_color_map={zone_id: color_palette_combined[i] for i, zone_id in enumerate(present_zone_ids)}
                    color_scale_discrete=[]; min_id, max_id=min(present_zone_ids), max(present_zone_ids)
                    id_range=max(1, max_id - min_id); sorted_zone_ids=sorted(zone_color_map.keys())
                    # (Bucle for para crear color_scale_discrete - igual que antes)
                    for i, zone_id in enumerate(sorted_zone_ids):
                         color=zone_color_map[zone_id]; norm_pos=(zone_id - min_id)/id_range if id_range>0 else 0.5
                         epsilon=0.5/(id_range + 1e-6); norm_start=max(0.0, norm_pos - epsilon); norm_end=min(1.0, norm_pos + epsilon)
                         if norm_start>=norm_end: norm_start=max(0.0, norm_pos - 1e-6); norm_end=min(1.0, norm_pos + 1e-6)
                         if i == 0:
                             if norm_start>0.0: color_scale_discrete.append([0.0, color])
                             color_scale_discrete.append([norm_start, color])
                         elif i>0:
                             prev_norm_end=color_scale_discrete[-1][0]
                             color_scale_discrete.append([max(prev_norm_end, norm_start - epsilon), zone_color_map[sorted_zone_ids[i-1]]])
                             color_scale_discrete.append([norm_start, color])
                         color_scale_discrete.append([norm_end, color])
                    if color_scale_discrete and color_scale_discrete[-1][0]<1.0: color_scale_discrete.append([1.0, color_scale_discrete[-1][1]])
                    elif not color_scale_discrete and sorted_zone_ids: color_scale_discrete=[[0.0, zone_color_map[sorted_zone_ids[0]]], [1.0, zone_color_map[sorted_zone_ids[0]]]]
                    simplified_colorscale=[];
                    if color_scale_discrete:
                        last_val=-1.0;
                        for val, col in color_scale_discrete:
                             if val>last_val: simplified_colorscale.append([val, col]); last_val=val
                        color_scale_discrete=simplified_colorscale
                    # --- FIN LÓGICA LEYENDA/COLOR ---

                    # --- ASEGURAR DEFINICIÓN DE classified_raster_display Y y_coords ---
                    if transform.e < 0:
                        y_coords = y_coords_raw[::-1]
                        classified_raster_display = np.flipud(classified_raster)
                    else:
                        y_coords = y_coords_raw
                        classified_raster_display = classified_raster
                    # --- FIN ASEGURAR ---

                    # --- Creación del Heatmap (CON Hover) ---
                    get_zone_name=np.vectorize(lambda zid: name_map.get(zid, "NoData" if zid==nodata_val else f"ID {zid}?"))
                    hover_names_raster=get_zone_name(classified_raster_display)
                    fig=go.Figure(data=go.Heatmap(z=classified_raster_display, x=x_coords, y=y_coords, colorscale=color_scale_discrete,
                                             zmin=min(present_zone_ids)-0.5, zmax=max(present_zone_ids)+0.5, showscale=True,
                                             colorbar=dict(title="ID Zona de Vida", tickvals=tick_values, ticktext=tick_texts, tickmode='array'),
                                             # hovertext=hover_names_raster, # MANTENER COMENTADO
                                             hoverinfo='skip', # <-- ASEGURAR 'skip' AQUÍ
                                             # hovertemplate='<b>Zona:</b> %{hovertext}<extra></extra>' # MANTENER COMENTADO
                                             ))
                # (Fin del else después de 'if not present_zone_ids:')

                # Mostrar figura si se creó
                if fig is not None:
                    fig.update_layout(
                        title="Mapa de Zonas de Vida de Holdridge",
                        xaxis_title=f"Coordenada X ({crs})",
                        yaxis_title=f"Coordenada Y ({crs})",
                        yaxis_scaleanchor="x",
                        height=700
                    )
                    st.plotly_chart(fig, use_container_width=True) # <-- Muestra el Heatmap

                    # --- AÑADIR LEYENDA DETALLADA (Área por Proporción) ---
                    st.markdown("---")
                    st.subheader("Leyenda y Área por Zona de Vida Presente")

                    # --- Calcular Área en Hectáreas por Proporción ---
                    # --- REVISED AREA RETRIEVAL ---
                    total_area_km2 = None # Initialize as None
                    balance_results_check = st.session_state.get('balance_results') 
                    
                    # Try getting area primarily from balance_results if it exists
                    if balance_results_check and 'Area_km2' in balance_results_check:
                        total_area_km2 = balance_results_check['Area_km2']
                    # Fallback: Check if land cover analysis set the area (less likely needed now)
                    elif st.session_state.get('total_basin_area_km2') is not None:
                         total_area_km2 = st.session_state.get('total_basin_area_km2')
                         
                    # --- END REVISED AREA RETRIEVAL ---

                    # --- DEBUGGING (Keep temporarily) ---
                    st.write(f"DEBUG: Retrieved total_area_km2 = {total_area_km2}") 
                    # --- END DEBUGGING ---
                    
                    area_hectares = []
                    # ... (rest of the calculation code remains the same) ...
                    pixel_counts = [] 
                    total_valid_pixels_in_map = 0
                    nodata_val = output_profile.get('nodata', 0)
                    valid_pixel_mask = (classified_raster != nodata_val)
                    total_valid_pixels_in_map = np.count_nonzero(valid_pixel_mask)

                    # Check both conditions *before* the loop
                    if total_valid_pixels_in_map > 0 and total_area_km2 is not None and total_area_km2 > 0:
                        # ----- Start Calculations -----
                        area_hectares = []
                        pixel_counts = []
                        for zone_id in present_zone_ids:
                            count = np.count_nonzero(classified_raster == zone_id)
                            pixel_counts.append(count)
                            proportion = count / total_valid_pixels_in_map
                            area_ha = proportion * (total_area_km2 * 100.0)
                            area_hectares.append(area_ha)
                        # ----- End Calculations -----

                        # ----- Create DataFrame -----
                        legend_data = {
                            "ID": present_zone_ids,
                            "Zona de Vida": [name_map.get(zid, f"ID {zid} Desconocido") for zid in present_zone_ids],
                            "Área (ha)": area_hectares,
                            "% del Área": [(c / total_valid_pixels_in_map) * 100.0 for c in pixel_counts]
                        }
                        legend_df = pd.DataFrame(legend_data).sort_values(by="Área (ha)", ascending=False)
                        # ----- End DataFrame Creation -----

                        # ----- Display DataFrame -----
                        st.dataframe(
                            legend_df.set_index('ID').style.format({
                                'Área (ha)': '{:,.1f}',
                                '% del Área': '{:.1f}%'
                            }),
                            use_container_width=True
                        )
                        st.caption(f"Área total clasificada: {total_area_km2:.2f} km² ({total_valid_pixels_in_map} píxeles)")
                        # ----- End Display DataFrame -----

                    elif total_area_km2 is None or total_area_km2 <= 0:
                         st.warning("No se pudo obtener un área total válida de la cuenca desde otras pestañas para calcular las hectáreas.")
                         # Display legend without areas
                         legend_data = {"ID": present_zone_ids, "Zona de Vida": [name_map.get(zid, f"ID {zid} Desconocido") for zid in present_zone_ids]}
                         if present_zone_ids: # Check if list is not empty before creating DF
                             legend_df = pd.DataFrame(legend_data).sort_values(by="ID")
                             st.dataframe(legend_df.set_index('ID'), use_container_width=True)
                         else:
                              st.info("No hay zonas de vida presentes para mostrar en la leyenda.")


                    else: # total_valid_pixels_in_map is 0
                         st.warning("No hay píxeles válidos clasificados para calcular áreas.")
                         # Display legend without areas
                         legend_data = {"ID": present_zone_ids, "Zona de Vida": [name_map.get(zid, f"ID {zid} Desconocido") for zid in present_zone_ids]}
                         if present_zone_ids: # Check if list is not empty before creating DF
                             legend_df = pd.DataFrame(legend_data).sort_values(by="ID")
                             st.dataframe(legend_df.set_index('ID'), use_container_width=True)
                         else:
                              st.info("No hay zonas de vida presentes para mostrar en la leyenda.")

                    # --- FIN LEYENDA DETALLADA ---
                    # --- EXPANDER INFO (sin cambios) ---
                st.markdown("---")
                with st.expander("Sobre la Clasificación de Zonas de Vida de Holdridge"):
                    st.markdown("""
                    El sistema de Zonas de Vida de Holdridge es un esquema global de clasificación bioclimática desarrollado por Leslie Holdridge en 1947 y actualizado en 1967. Clasifica las áreas terrestres basándose en tres ejes climáticos principales:

                    1.  **Biotemperatura Media Anual (°C):** Es una medida de la temperatura relacionada con el crecimiento vegetal. Se calcula como el promedio de las temperaturas medias (diarias o mensuales) superiores a 0°C, tratando cualquier temperatura sobre 30°C como si fuera 30°C. En esta aplicación, se *estima* a partir de la temperatura media anual calculada por altitud.
                    2.  **Precipitación Total Anual (mm):** La cantidad total de lluvia recibida en un año.
                    3.  **Razón de Evapotranspiración Potencial (PET / PPA):** Es la relación entre la evapotranspiración potencial (la cantidad de agua que *podría* evaporarse y transpirarse si hubiera suficiente agua disponible) y la precipitación anual. Indica la aridez o humedad del clima. La PET se estima a partir de la biotemperatura (PET ≈ 58.93 * BAT).

                    La combinación de estos tres factores define hexágonos en un diagrama logarítmico, donde cada hexágono representa una **Zona de Vida**, caracterizada por un tipo particular de vegetación natural esperada bajo esas condiciones climáticas (ej., Bosque seco Tropical, Páramo pluvial Subalpino).

                    **En esta aplicación:**
                    * La **Biotemperatura** se estima a partir de un Modelo Digital de Elevación (DEM) y una tasa de lapso estándar.
                    * La **Precipitación** se obtiene de un raster de precipitación media anual (`PPAMAnt.tif`).
                    * La clasificación se realiza pixel a pixel usando los rangos definidos específicamente para Antioquia (basados en BAT y PPT).
                    """)

            else:
                st.error("La generación del mapa de zonas de vida falló. Revisa los errores anteriores.")
        
    elif not dem_path and os.path.exists(precip_raster_path):
         st.info("Sube un archivo DEM para habilitar la generación del mapa.")







