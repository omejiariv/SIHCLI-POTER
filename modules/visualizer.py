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
# from prophet.plot import plot_plotly # Comentado si no se usa
import io
from datetime import datetime, timedelta, date
import json
import requests
import traceback
import openmeteo_requests
import rasterio
from rasterio.transform import from_origin
from rasterio.mask import mask
from scipy.interpolate import griddata
import gstools as gs
import pyproj
from rasterio.warp import reproject, Resampling

#--- Importaciones de Módulos Propios ---
from modules.config import Config
from modules.openmeteo_api import get_historical_climate_average
from modules.analysis import (
    calculate_morphometry, calculate_hypsometric_curve,
    calculate_hydrological_balance,
    calculate_all_station_trends,
    calculate_spi, calculate_spei, calculate_monthly_anomalies,
    calculate_percentiles_and_extremes, analyze_events,
    calculate_climatological_anomalies,
    calculate_basin_stats # Asegurar que esta esté importada
)
from modules.utils import add_folium_download_button
from modules.interpolation import (
    create_interpolation_surface, 
    perform_loocv_for_all_methods,
    # create_kriging_by_basin, # Comentado si no se usa
    # interpolate_idw # Comentado si no se usa
)
from modules.forecasting import (
    generate_sarima_forecast, generate_prophet_forecast,
    get_decomposition_results, create_acf_chart, create_pacf_chart,
    auto_arima_search, get_weather_forecast
)
from modules.data_processor import complete_series
from modules.life_zones import generate_life_zone_map, holdridge_int_to_name_simplified


# --- DEFINICIÓN DE display_filter_summary ---
def display_filter_summary(total_stations_count, selected_stations_count, year_range,
                           selected_months_count, analysis_mode, selected_regions,
                           selected_municipios, selected_altitudes):
    """Muestra un resumen de los filtros activos."""
    if isinstance(year_range, (tuple, list)) and len(year_range) == 2:
        year_text = f"{year_range[0]}-{year_range[1]}"
    else:
        year_text = str(year_range) 

    mode_text = "Original (con huecos)"
    if analysis_mode == "Completar series (interpolación)":
        mode_text = "Completado (interpolado)"

    summary_parts = [
        f"**Estaciones:** {selected_stations_count}/{total_stations_count}",
        f"**Período:** {year_text}",
        f"**Datos:** {mode_text}"
    ]
    if selected_regions: summary_parts.append(f"**Región:** {', '.join(selected_regions)}")
    if selected_municipios: summary_parts.append(f"**Municipio:** {', '.join(selected_municipios)}")
    if selected_altitudes: summary_parts.append(f"**Altitud:** {', '.join(selected_altitudes)}")
    
    with st.expander("Resumen de Filtros Activos", expanded=False):
         st.info(" | ".join(summary_parts))
# --- FIN display_filter_summary ---
        
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
    """
    Crea un mapa de folium base con capas y ajuste de límites opcional.
    """
    m = folium.Map(location=location, zoom_start=zoom, tiles=None)
    
    # Añadir mapa base
    if base_map_config and 'tiles' in base_map_config and 'attr' in base_map_config:
        folium.TileLayer(tiles=base_map_config['tiles'], attr=base_map_config['attr'], name="Mapa Base").add_to(m)
    else:
        # Fallback si la config está mal
        folium.TileLayer(tiles="cartodbpositron", attr="CartoDB").add_to(m)

    # Lógica de Overlays (WMS, GeoJSON, etc. - como en tu código)
    if overlays_config:
        # Asumiendo que overlays_config es una LISTA de diccionarios
        for layer_config in overlays_config:
            # Asegurarse que layer_config sea un diccionario
            if not isinstance(layer_config, dict):
                # st.warning(f"Elemento de overlay no es un diccionario: {layer_config}")
                continue # Saltar si no es un diccionario

            layer_type = layer_config.get("type", "tile")
            url = layer_config.get("url")
            if not url: continue
            layer_name = layer_config.get("attr", layer_config.get("name", "Overlay")) # Usar 'attr' o 'name'

            try:
                if layer_type == "wms":
                    if "layers" not in layer_config:
                        # st.warning(f"Capa WMS '{layer_name}' no tiene 'layers' definidos.")
                        continue
                    WmsTileLayer(
                        url=url,
                        layers=layer_config["layers"], # Requiere 'layers'
                        fmt=layer_config.get("fmt", 'image/png'),
                        transparent=layer_config.get("transparent", True),
                        overlay=True, control=True, name=layer_name,
                        attr=layer_name
                    ).add_to(m)
                elif layer_type == "geojson":
                    # Asumiendo que load_geojson_from_url está definida en otra parte
                    geojson_data = load_geojson_from_url(url) 
                    if geojson_data:
                        style_function = lambda x: layer_config.get("style", {})
                        folium.GeoJson(geojson_data, name=layer_name, style_function=style_function).add_to(m)
                else: # Asumir 'tile'
                    folium.TileLayer(
                        tiles=url, attr=layer_name, name=layer_name,
                        overlay=True, control=True, show=False
                    ).add_to(m)
            except Exception as e_layer:
                st.warning(f"No se pudo añadir la capa overlay '{layer_name}': {e_layer}")
    
    # --- LÓGICA DE AJUSTE DE LÍMITES (MOVIDA DENTRO) ---
    if fit_bounds_data is not None and not fit_bounds_data.empty:
        try:
            if len(fit_bounds_data) > 1:
                bounds = fit_bounds_data.total_bounds
                if np.all(np.isfinite(bounds)):
                    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
            elif len(fit_bounds_data) == 1:
                # Extraer la primera geometría válida
                point = fit_bounds_data.geometry.iloc[0]
                if point and not point.is_empty:
                    m.location = [point.y, point.x]
                    m.zoom_start = 12
        except Exception as e_bounds:
            st.warning(f"Error al ajustar límites del mapa: {e_bounds}")
    # --- FIN LÓGICA DE LÍMITES ---
            
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

        # Nivel 1 de indentación: Verificar si hay datos para mostrar
        if not gdf_display.empty:
            
            # Nivel 2 de indentación: Comprobar el modo de análisis
            if analysis_mode == "Completar series (interpolación)":
                
                # Nivel 3 de indentación: Código para el modo "Completar"

                if Config.ORIGIN_COL in df_monthly_filtered.columns:
                     st.dataframe(df_monthly_filtered[Config.ORIGIN_COL].value_counts())
                else:
                     st.warning(f"'{Config.ORIGIN_COL}' NO está en df_monthly_filtered al entrar a visualizer.")
                # --- Fin Debug Inicial ---

                st.info("Mostrando la composición de datos originales vs. completados para el período seleccionado.")

                # Nivel 3 de indentación: Comprobar si hay datos y la columna 'origin'
                if not df_monthly_filtered.empty and Config.ORIGIN_COL in df_monthly_filtered.columns:

                    # Nivel 4 de indentación: Calcular composición
                    data_composition = \
                        df_monthly_filtered.groupby([Config.STATION_NAME_COL,
                                                     Config.ORIGIN_COL]).size().unstack(fill_value=0)
                    
                    # Asegurar que las columnas 'Original' y 'Completado' existan
                    if 'Original' not in data_composition: data_composition['Original'] = 0
                    if 'Completado' not in data_composition: data_composition['Completado'] = 0

                    # Calcular total
                    data_composition['total'] = data_composition['Original'] + data_composition['Completado']
                    
                    # --- START CORRECTION for INF ---
                    # Initialize percentage columns with 0.0
                    data_composition['% Original'] = 0.0
                    data_composition['% Completado'] = 0.0

                    # Create mask for rows where total > 0
                    mask_total_greater_than_zero = data_composition['total'] > 0
                    
                    # Calculate percentages ONLY where total > 0
                    if mask_total_greater_than_zero.any(): # Check if there are any valid rows
                        data_composition.loc[mask_total_greater_than_zero, '% Original'] = \
                            (data_composition.loc[mask_total_greater_than_zero, 'Original'] / 
                             data_composition.loc[mask_total_greater_than_zero, 'total']) * 100
                        
                        data_composition.loc[mask_total_greater_than_zero, '% Completado'] = \
                            (data_composition.loc[mask_total_greater_than_zero, 'Completado'] / 
                             data_composition.loc[mask_total_greater_than_zero, 'total']) * 100
                    
                    # (Optional but safe) Replace any remaining inf/-inf/NaN with 0 in percentage columns
                    # We need numpy for this
                    import numpy as np 
                    data_composition.replace([np.inf, -np.inf, np.nan], 0.0, inplace=True)
                    # --- END CORRECTION for INF ---

                    # Ordenamiento
                    sort_order_comp = st.radio("Ordenar por:", ["% Datos Originales (Mayor a Menor)", "% Datos Originales (Menor a Mayor)", "Alfabético"], horizontal=True,
                                               key="sort_comp")

                    # Sorting logic
                    if "Mayor a Menor" in sort_order_comp: data_composition = data_composition.sort_values("% Original", ascending=False)
                    elif "Menor a Mayor" in sort_order_comp: data_composition = data_composition.sort_values("% Original", ascending=True)
                    else: data_composition = data_composition.sort_index(ascending=True)

                    # Melt para preparar datos para el gráfico
                    df_plot = data_composition.reset_index().melt(
                        id_vars=Config.STATION_NAME_COL,
                        value_vars=['% Original', '% Completado'],
                        var_name='Tipo de Dato',
                        value_name='Porcentaje'
                    )
                    
                    # Crear el gráfico (este código no cambia)
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

                    fig_comp.update_layout(height=500, xaxis={'categoryorder': 'trace'}, barmode='stack')
                    st.plotly_chart(fig_comp, use_container_width=True)

                # Nivel 3 de indentación: Else para 'if not df_monthly_filtered.empty...'
                else:
                    st.warning("No hay datos mensuales procesados o falta la columna 'origin' para mostrar la composición.")

            # Nivel 2 de indentación: Else para 'if analysis_mode == ...' (Este bloque no se modifica)

            else: 
                st.info("Mostrando el porcentaje de disponibilidad de datos según el archivo de estaciones.")

                # Nivel 3 de indentación: Código para el modo "Original"
                sort_order_disp = st.radio("Ordenar estaciones por:", ["% Datos (Mayor a Menor)", "% Datos (Menor a Mayor)", "Alfabético"], horizontal=True, key="sort_disp")

                df_chart = gdf_display.copy()

                # Nivel 3: Comprobar si existe la columna de porcentaje
                if Config.PERCENTAGE_COL in df_chart.columns:
                    # Nivel 4: Lógica de ordenamiento y gráfico
                    # Asegurarse que la columna sea numérica
                    df_chart[Config.PERCENTAGE_COL] = pd.to_numeric(df_chart[Config.PERCENTAGE_COL], errors='coerce').fillna(0)

                    if "% Datos (Mayor a Menor)" in sort_order_disp:
                        df_chart = df_chart.sort_values(Config.PERCENTAGE_COL, ascending=False)
                    elif "% Datos (Menor a Mayor)" in sort_order_disp:
                        df_chart = df_chart.sort_values(Config.PERCENTAGE_COL, ascending=True)
                    else:
                        df_chart = df_chart.sort_values(Config.STATION_NAME_COL, ascending=True)

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
                # Nivel 3: Else para 'if Config.PERCENTAGE_COL...'
                else:
                    st.warning(f"La columna '{Config.PERCENTAGE_COL}' no se encuentra en el archivo de estaciones.")
        
        # Nivel 1 de indentación: Else para 'if not gdf_display.empty:'
        else:
            st.warning("No hay estaciones seleccionadas (después de aplicar filtros) para mostrar el gráfico.")

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

            # @st.cache_data
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

# --- OTRAS FUNCIONES AUXILIARES (create_climate_risk_map, etc.) ---
@st.cache_data
def create_climate_risk_map(df_anual, _gdf_stations): # Añadido _ para caché
    try:
        gdf_trends = calculate_all_station_trends(df_anual, _gdf_stations)
    except Exception as e_trend:
         st.warning(f"No se pudieron calcular tendencias para el mapa de riesgo: {e_trend}")
         return None
    if gdf_trends.empty:
        st.warning("No hay suficientes datos de tendencia (>10 años) para generar el mapa de riesgo.")
        return None
    
    # Filtrar geometrías nulas antes de acceder a coords
    gdf_trends_valid_geom = gdf_trends[gdf_trends.geometry.notna() & ~gdf_trends.geometry.is_empty]
    if gdf_trends_valid_geom.empty:
        st.warning("No se encontraron geometrías válidas en las estaciones con tendencia.")
        return None
        
    coords = np.array([geom.coords[0] for geom in gdf_trends_valid_geom.geometry])
    lons = coords[:, 0]; lats = coords[:, 1]; values = gdf_trends_valid_geom['slope_sen'].values
    
    grid_lon = np.linspace(lons.min() - 0.1, lons.max() + 0.1, 100)
    grid_lat = np.linspace(lats.min() - 0.1, lats.max() + 0.1, 100)
    grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)
    
    try:
        grid_z = griddata((lons, lats), values, (grid_x, grid_y), method='cubic')
        nan_mask = np.isnan(grid_z)
        if np.any(nan_mask):
             fill_values = griddata((lons, lats), values, (grid_x[nan_mask], grid_y[nan_mask]), method='nearest')
             grid_z[nan_mask] = fill_values
        grid_z = np.nan_to_num(grid_z)
    except Exception as e:
         st.error(f"Error en interpolación (griddata) para mapa de riesgo: {e}")
         return None

    fig = go.Figure(data=go.Contour(
        z=grid_z.T, x=grid_lon, y=grid_lat, colorscale='RdBu_r', zmid=0,
        colorbar=dict(title='Tendencia (mm/año)'), contours=dict(showlabels=True, labelfont=dict(size=10, color='white'))
    ))
    fig.add_trace(go.Scatter(
        x=lons, y=lats, mode='markers', marker=dict(color='black', size=5, symbol='circle-open'),
        hoverinfo='text',
        hovertext=gdf_trends_valid_geom.apply(lambda row:
                           f"<b>Estación: {row[Config.STATION_NAME_COL]}</b><br><br>"
                           f"Municipio: {row.get(Config.MUNICIPALITY_COL, 'N/A')}<br>"
                           f"Altitud: {row.get(Config.ALTITUDE_COL, 'N/A'):.0f} m<br>"
                           f"Tendencia: {row['slope_sen']:.2f} mm/año<br>"
                           f"Significancia (p-valor): {row['p_value']:.3f}", 
                           axis=1), name='Estaciones'
    ))
    fig.update_layout(title="Mapa de Tendencias de Precipitación (Pendiente de Sen)", xaxis_title="Longitud", yaxis_title="Latitud", height=600)
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

    try:
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
    except Exception as e_summary:
         st.warning(f"Error al mostrar resumen de filtros: {e_summary}")


    if not stations_for_analysis:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return

    # Pestañas implementadas (según tu código anterior)
    tab_names = [
        "Animación GIF", "Superficies de Interpolación", "Morfometría",
        "Mapa de Riesgo Climático", "Validación Cruzada (LOOCV)",
        "Visualización Temporal", "Gráfico de Carrera", "Mapa Animado", "Comparación de Mapas"
    ]
    try:
        # Asignar todas las pestañas
        gif_tab, kriging_tab, morph_tab, risk_map_tab, validation_tab, temporal_tab, race_tab, anim_tab, compare_tab = st.tabs(tab_names)
    except ValueError as e:
        st.error(f"Error al crear pestañas, verifica 'tab_names': {e}")
        return

    # --- Pestaña Animación GIF ---
    with gif_tab:
        st.subheader("Distribución Espacio-Temporal de la Lluvia en Antioquia")
        col_controls_gif, col_gif_display = st.columns([1, 3])
        with col_controls_gif:
            if st.button("Reiniciar Animación", key="reset_gif_button"):
                st.rerun()
        with col_gif_display:
            gif_path = Config.GIF_PATH
            if gif_path and os.path.exists(gif_path):
                try:
                    with open(gif_path, "rb") as f: gif_bytes = f.read()
                    gif_b64 = base64.b64encode(gif_bytes).decode("utf-8")
                    html_string = f'<img src="data:image/gif;base64,{gif_b64}" width="600" alt="Animación PPAM">'
                    st.markdown(html_string, unsafe_allow_html=True)
                except FileNotFoundError: st.error(f"No se pudo encontrar el archivo GIF en: {gif_path}")
                except Exception as e: st.error(f"Ocurrió un error al mostrar el GIF: {e}")
            elif gif_path: st.error(f"La ruta del GIF no existe: {gif_path}")
            else: st.info("Ruta del archivo GIF no configurada (Config.GIF_PATH).")

    # --- Pestaña Superficies de Interpolación ---
    with kriging_tab:
        st.subheader("Superficies de Interpolación de Precipitación Anual")
        analysis_mode_interp = st.radio(
            "Seleccione el modo de interpolación:",
            ("Regional (Toda la selección)", "Por Cuenca Específica"),
            key="interp_mode_radio", horizontal=True,
            help="El modo 'Por Cuenca' permite un análisis detallado con buffer y DEM."
        )
        st.markdown("---")

        # --- Modo Por Cuenca Específica ---
        if analysis_mode_interp == "Por Cuenca Específica":
            if 'gdf_subcuencas' not in st.session_state or st.session_state.gdf_subcuencas is None or st.session_state.gdf_subcuencas.empty:
                st.warning("Los datos de cuencas no están disponibles o están vacíos.")
                st.stop()

            BASIN_NAME_COLUMN = 'SUBC_LBL'
            if BASIN_NAME_COLUMN not in st.session_state.gdf_subcuencas.columns:
                st.error(f"La columna '{BASIN_NAME_COLUMN}' no se encontró en los datos de cuencas.")
                st.stop()

            col_control, col_display = st.columns([1, 2])

            with col_control:
                st.markdown("#### Controles de Cuenca")
                basin_names = []
                regions_from_sidebar = selected_regions
                if 'gdf_subcuencas' in st.session_state and st.session_state.gdf_subcuencas is not None:
                    gdf_subcuencas_local = st.session_state.gdf_subcuencas
                    if regions_from_sidebar:
                        if Config.REGION_COL in gdf_subcuencas_local.columns:
                            relevant_basins_by_region = gdf_subcuencas_local[gdf_subcuencas_local[Config.REGION_COL].isin(regions_from_sidebar)]
                            if not relevant_basins_by_region.empty: basin_names = sorted(relevant_basins_by_region[BASIN_NAME_COLUMN].dropna().unique())
                            else: st.info("Ninguna cuenca encontrada en las regiones seleccionadas.")
                        else:
                             st.warning(f"Archivo cuencas sin columna '{Config.REGION_COL}'. Mostrando todas."); basin_names = sorted(gdf_subcuencas_local[BASIN_NAME_COLUMN].dropna().unique())
                    else: basin_names = sorted(gdf_subcuencas_local[BASIN_NAME_COLUMN].dropna().unique())
                else: st.warning("Datos de subcuencas no disponibles.")

                selected_basins = st.multiselect("Seleccione una o más subcuencas:", options=basin_names, key="basin_multiselect")
                buffer_km = st.slider("Buffer de influencia (km):", 0, 50, 10, 5, key="buffer_slider")
                df_anual_non_na = df_anual_melted.dropna(subset=[Config.PRECIPITATION_COL])
                years = sorted(df_anual_non_na[Config.YEAR_COL].unique()) if not df_anual_non_na.empty else []

                if not years: st.warning("No hay datos anuales válidos para la interpolación en la selección actual.")
                else:
                    selected_year = st.selectbox("Seleccione un año:", options=years, index=len(years) - 1, key="year_select_basin")
                    method = st.selectbox("Método de interpolación:", options=["IDW (Lineal)", "Spline (Cúbico)"], key="interp_method_basin")
                    run_balance = st.toggle("Calcular Balance Hídrico", value=True, key="run_balance_toggle_cuenca")
                    show_dem_background = st.toggle("Visualizar DEM de fondo", value=True, key="show_dem_toggle_cuenca")
                    
                    # Obtener info del DEM (SOLO DESDE SESSION STATE)
                    dem_fixed_path = st.session_state.get('dem_file_path')

                    if st.button("Generar Mapa para Cuenca(s)", disabled=not selected_basins, key="generate_basin_map_button"):
                        st.session_state['run_balance'] = run_balance; st.session_state['fig_basin'] = None; st.session_state['error_msg'] = None
                        st.session_state['mean_precip'] = None; st.session_state['morph_results'] = None; st.session_state['balance_results'] = None
                        st.session_state['unified_basin_gdf'] = None; st.session_state['selected_basins_title'] = ""
                        
                        # Usar solo DEM base
                        effective_dem_path_in_use = dem_fixed_path if (dem_fixed_path and os.path.exists(dem_fixed_path)) else None
                        
                        try:
                            with st.spinner("Preparando datos y realizando interpolación..."):
                                target_basins_gdf = st.session_state.gdf_subcuencas[st.session_state.gdf_subcuencas[BASIN_NAME_COLUMN].isin(selected_basins)]
                                unified_basin_gdf = gpd.GeoDataFrame(geometry=[target_basins_gdf.unary_union], crs=target_basins_gdf.crs)
                                target_basin_metric = unified_basin_gdf.to_crs("EPSG:3116"); basin_buffer_metric = target_basin_metric.buffer(buffer_km * 1000)

                                if 'gdf_stations' not in st.session_state or st.session_state.gdf_stations is None: raise ValueError("Datos de estaciones no cargados.")
                                if st.session_state.gdf_stations.crs is None: st.session_state.gdf_stations.set_crs("EPSG:4326", inplace=True)

                                stations_metric = st.session_state.gdf_stations.to_crs("EPSG:3116")
                                stations_in_buffer = stations_metric[stations_metric.intersects(basin_buffer_metric.unary_union)]
                                station_names = stations_in_buffer[Config.STATION_NAME_COL].unique()
                                if len(station_names) == 0: raise ValueError(f"No se encontraron estaciones dentro del buffer de {buffer_km} km.")

                                precip_data_year = df_anual_non_na[(df_anual_non_na[Config.YEAR_COL] == selected_year) & (df_anual_non_na[Config.STATION_NAME_COL].isin(station_names))]
                                cols_to_merge = [Config.STATION_NAME_COL, 'geometry', Config.MUNICIPALITY_COL, Config.ALTITUDE_COL]
                                points_data = gpd.GeoDataFrame(pd.merge(stations_in_buffer[cols_to_merge], precip_data_year[[Config.STATION_NAME_COL, Config.PRECIPITATION_COL]], on=Config.STATION_NAME_COL), geometry='geometry', crs="EPSG:3116").dropna(subset=[Config.PRECIPITATION_COL])
                                points_data.rename(columns={Config.PRECIPITATION_COL: 'Valor'}, inplace=True)
                                if len(points_data) < 3: raise ValueError(f"Se necesitan al menos 3 estaciones con datos en {selected_year} dentro del buffer (encontradas: {len(points_data)}).")

                                bounds = basin_buffer_metric.unary_union.bounds; grid_resolution = 500
                                grid_lon = np.arange(bounds[0], bounds[2], grid_resolution); grid_lat = np.arange(bounds[1], bounds[3], grid_resolution)
                                if grid_lon.size == 0 or grid_lat.size == 0: raise ValueError("Buffer o grilla inválida.")

                                points = np.column_stack((points_data.geometry.x, points_data.geometry.y)); values = points_data['Valor'].values
                                grid_x, grid_y = np.meshgrid(grid_lon, grid_lat); interp_method_call = 'linear' if method == "IDW (Lineal)" else 'cubic'
                                grid_z = griddata(points, values, (grid_x, grid_y), method=interp_method_call)
                                nan_mask = np.isnan(grid_z)
                                if np.any(nan_mask): fill_values = griddata(points, values, (grid_x[nan_mask], grid_y[nan_mask]), method='nearest'); grid_z[nan_mask] = fill_values
                                grid_z = np.nan_to_num(grid_z); grid_z[grid_z < 0] = 0
                                transform = from_origin(grid_lon[0], grid_lat.max(), grid_resolution, grid_resolution)

                                with rasterio.io.MemoryFile() as memfile:
                                    profile = {'driver': 'GTiff', 'height': len(grid_lat), 'width': len(grid_lon), 'count': 1, 'dtype': str(grid_z.dtype), 'crs': "EPSG:3116", 'transform': transform, 'nodata': np.nan}
                                    with memfile.open(**profile) as dataset: dataset.write(np.flipud(grid_z), 1)
                                    with memfile.open() as dataset: masked_data, masked_transform = mask(dataset, target_basin_metric.geometry, crop=True, nodata=np.nan, all_touched=True)

                                masked_data = masked_data[0].astype(np.float32)
                                mean_precip_values = masked_data[~np.isnan(masked_data)]; mean_precip = np.mean(mean_precip_values) if mean_precip_values.size > 0 else 0.0

                                map_traces = []; dem_trace = None
                                # --- CÁLCULO DE COORDENADAS CORREGIDO (Error 3) ---
                                height_masked, width_masked = masked_data.shape
                                transform_masked = masked_transform
                                x_coords = [transform_masked.c + transform_masked.a * (i + 0.5) for i in range(width_masked)]
                                y_coords_raw = [transform_masked.f + transform_masked.e * (i + 0.5) for i in range(height_masked)]
                                # --- FIN CÁLCULO CORREGIDO ---
                                
                                if transform_masked.e < 0:
                                     y_coords = y_coords_raw[::-1]; masked_data_display = np.flipud(masked_data)
                                else:
                                     y_coords = y_coords_raw; masked_data_display = masked_data
                                
                                masked_data_display_nan = masked_data_display.astype(float)
                                masked_data_display_nan[np.isnan(masked_data_display_nan)] = np.nan # Asegurar que NaN siga siendo NaN

                                # --- CORRECCIÓN INDENTACIÓN (Error de indentación anterior) ---
                                if show_dem_background and effective_dem_path_in_use:
                                    with st.spinner("Procesando y reproyectando DEM..."): # INDENTADO
                                        try:
                                            with rasterio.open(effective_dem_path_in_use) as dem_src: # Usar path efectivo
                                                dem_reprojected = np.empty(masked_data.shape, dtype=rasterio.float32)
                                                reproject(source=rasterio.band(dem_src, 1), destination=dem_reprojected, src_transform=dem_src.transform, src_crs=dem_src.crs, dst_transform=masked_transform, dst_crs="EPSG:3116", dst_nodata=np.nan, resampling=Resampling.bilinear)
                                                if masked_transform.e < 0: dem_reprojected = np.flipud(dem_reprojected)
                                                dem_trace = go.Heatmap(z=dem_reprojected, x=x_coords, y=y_coords, colorscale='gray', showscale=False, name='Elevación')
                                                map_traces.append(dem_trace)
                                        except Exception as e_dem_viz:
                                            st.warning(f"No se pudo procesar DEM para fondo: {e_dem_viz}")
                                # --- FIN CORRECCIÓN INDENTACIÓN ---

                                precip_trace = go.Heatmap(z=masked_data_display_nan, x=x_coords, y=y_coords, colorscale='viridis', colorbar=dict(title='Precipitación (mm)'), opacity=0.7 if dem_trace is not None else 1.0, name='Precipitación', hoverinfo='skip')
                                map_traces.append(precip_trace)
                                fig_basin = go.Figure(data=map_traces)

                                points_data['hover_text'] = points_data.apply(lambda row: f"<b>{row[Config.STATION_NAME_COL]}</b><br>Municipio: {row[Config.MUNICIPALITY_COL]}<br>Altitud: {row[Config.ALTITUDE_COL]:.0f} m<br>Precipitación: {row['Valor']:.0f} mm", axis=1)
                                fig_basin.add_trace(go.Scatter(x=points_data.geometry.x, y=points_data.geometry.y, mode='markers', marker=dict(color='black', size=5, symbol='circle-open', line=dict(color='white', width=0.5)), name='Estaciones', hoverinfo='text', hovertext=points_data['hover_text']))
                                fig_basin.update_layout(title=f"Precipitación Interpolada ({method}) para Cuenca(s) ({selected_year})", xaxis_title="Coordenada Este (m)", yaxis=dict(title="Coordenada Norte (m)", scaleanchor='x', scaleratio=1), height=600)

                                st.session_state['fig_basin'] = fig_basin
                                st.session_state['mean_precip'] = mean_precip if mean_precip is not None and not np.isnan(mean_precip) else None
                                st.session_state['unified_basin_gdf'] = unified_basin_gdf
                                st.session_state['selected_basins_title'] = ", ".join(selected_basins)

                                if effective_dem_path_in_use:
                                    try:
                                         st.session_state['morph_results'] = calculate_morphometry(unified_basin_gdf, effective_dem_path_in_use)
                                    except Exception as e_morph: st.session_state['morph_results'] = {"error": f"Error calculando morfometría: {e_morph}"}
                                else: 
                                    st.session_state['morph_results'] = None

                            # --- CÁLCULO DE BALANCE (Movido fuera del spinner) ---
                            run_balance_state = st.session_state.get('run_balance', False)
                            if run_balance_state:
                                mean_p = st.session_state.get('mean_precip'); morph_r = st.session_state.get('morph_results'); basin_g = st.session_state.get('unified_basin_gdf')
                                if mean_p is not None and morph_r is not None and not morph_r.get("error") and basin_g is not None:
                                     alt_prom_balance = morph_r.get('alt_prom_m')
                                     if alt_prom_balance is not None:
                                          balance_results_calc = calculate_hydrological_balance(mean_p, alt_prom_balance, basin_g)
                                          st.session_state['balance_results'] = balance_results_calc
                                     else: st.session_state['balance_results'] = {"error": "Altitud promedio no disponible."}
                                elif mean_p is None: st.session_state['balance_results'] = {"error": "Precipitación media no calculada."}
                                elif morph_r is None: st.session_state['balance_results'] = {"error": "Morfometría no calculada (falta DEM?)."}
                                elif morph_r.get("error"): st.session_state['balance_results'] = {"error": f"Error morfometría: {morph_r.get('error')}"}
                                else: st.session_state['balance_results'] = {"error": "Faltan datos para balance."}
                            # --- FIN CÁLCULO BALANCE ---

                        except Exception as e:
                            import traceback
                            st.session_state['error_msg'] = f"Ocurrió un error crítico: {e}\n\n{traceback.format_exc()}"
                        # --- CORRECCIÓN Error 5: 'finally' block eliminado ---
                        # 'finally' ya no es necesario aquí porque no creamos 'temp_dem_to_delete'
            
            # --- Visualización (fuera del botón) ---
            with col_display:
                fig_basin_to_show = st.session_state.get('fig_basin'); error_msg_to_show = st.session_state.get('error_msg')
                balance_results_to_show = st.session_state.get('balance_results'); morph_results_to_show = st.session_state.get('morph_results')
                run_balance_display = st.session_state.get('run_balance', False)

                if error_msg_to_show: st.error(error_msg_to_show)
                if fig_basin_to_show:
                    st.subheader(f"Resultados para: {st.session_state.get('selected_basins_title', '')}")
                    st.plotly_chart(fig_basin_to_show, use_container_width=True)

                if run_balance_display and balance_results_to_show is not None:
                    st.markdown("---"); st.subheader("Balance Hídrico Estimado")
                    if balance_results_to_show.get("error"): st.error(balance_results_to_show["error"])
                    else:
                        c1, c2, c3, c4 = st.columns(4)
                        p_val = balance_results_to_show.get('P_media_anual_mm', 0); alt_val = balance_results_to_show.get('Altitud_media_m')
                        et_val = balance_results_to_show.get('ET_media_anual_mm'); q_val = balance_results_to_show.get('Q_mm')
                        c1.metric("Precipitación Media (P)", f"{p_val:.0f} mm/año"); c2.metric("Altitud Media", f"{alt_val:.0f} m" if alt_val is not None else "N/A")
                        c3.metric("ET Media Estimada (ET)", f"{et_val:.0f} mm/año" if et_val is not None else "N/A"); c4.metric("Escorrentía (Q=P-ET)", f"{q_val:.0f} mm/año" if q_val is not None else "N/A")
                        q_vol = balance_results_to_show.get('Q_m3_año'); area_km2 = balance_results_to_show.get('Area_km2')
                        if q_vol is not None and area_km2 is not None: st.success(f"Volumen de escorrentía anual estimado: **{q_vol/1e6:.2f} millones de m³** sobre un área de **{area_km2:.2f} km²**.")
                elif run_balance_display: st.warning("No se pudieron obtener resultados del balance hídrico.")

                if morph_results_to_show is not None:
                    st.markdown("---"); st.subheader("Morfometría de la Cuenca")
                    if morph_results_to_show.get("error"): st.error(morph_results_to_show["error"])
                    else:
                        c1m, c2m, c3m = st.columns(3); c1m.metric("Área", f"{morph_results_to_show.get('area_km2', 'N/A'):.2f} km²"); c2m.metric("Perímetro", f"{morph_results_to_show.get('perimetro_km', 'N/A'):.2f} km"); c3m.metric("Índice de Forma", f"{morph_results_to_show.get('indice_forma', 'N/A'):.2f}")
                        c4m, c5m, c6m = st.columns(3); alt_max=morph_results_to_show.get('alt_max_m'); alt_min=morph_results_to_show.get('alt_min_m'); alt_prom=morph_results_to_show.get('alt_prom_m')
                        c4m.metric("Altitud Máxima", f"{alt_max:.0f} m" if alt_max is not None else "N/A"); c5m.metric("Altitud Mínima", f"{alt_min:.0f} m" if alt_min is not None else "N/A"); c6m.metric("Altitud Promedio", f"{alt_prom:.1f} m" if alt_prom is not None else "N/A")
                elif run_balance_display: st.info("Para Morfometría, usa el DEM base.")
        # --- Fin Modo Por Cuenca ---

        # --- Modo Regional ---
        else:
            df_anual_non_na = df_anual_melted.dropna(subset=[Config.PRECIPITATION_COL])
            if not stations_for_analysis or df_anual_non_na.empty:
                st.warning("No hay suficientes datos anuales para interpolación regional.")
            else:
                min_year_reg = int(df_anual_non_na[Config.YEAR_COL].min())
                max_year_reg = int(df_anual_non_na[Config.YEAR_COL].max())
                control_col_reg, map_col1_reg, map_col2_reg = st.columns([1, 2, 2])

                with control_col_reg:
                    st.markdown("#### Controles de los Mapas")
                    interpolation_methods_reg = ["Kriging Ordinario", "IDW", "Spline (Thin Plate)"]
                    if Config.ELEVATION_COL in gdf_filtered.columns and not gdf_filtered[Config.ELEVATION_COL].isnull().all():
                         interpolation_methods_reg.insert(1, "Kriging con Deriva Externa (KED)")

                    st.markdown("**Mapa 1**"); year1_reg = st.slider("Año Mapa 1", min_year_reg, max_year_reg, max_year_reg, key="interp_year1_reg")
                    method1_reg = st.selectbox("Método Mapa 1", options=interpolation_methods_reg, key="interp_method1_reg")
                    variogram_model1_reg = None
                    if "Kriging" in method1_reg: variogram_model1_reg = st.selectbox("Variograma Mapa 1", ['linear', 'spherical', 'exponential', 'gaussian'], key="var_model_1_reg")

                    st.markdown("---"); st.markdown("**Mapa 2**"); year2_reg = st.slider("Año Mapa 2", min_year_reg, max_year_reg, max(min_year_reg, max_year_reg - 1), key="interp_year2_reg")
                    index_map2 = min(1, len(interpolation_methods_reg) - 1) if interpolation_methods_reg else 0 
                    method2_reg = st.selectbox("Método Mapa 2", options=interpolation_methods_reg, index=index_map2, key="interp_method2_reg")
                    variogram_model2_reg = None
                    if "Kriging" in method2_reg: variogram_model2_reg = st.selectbox("Variograma Mapa 2", ['linear', 'spherical', 'exponential', 'gaussian'], key="var_model_2_reg")

                cols_metadata = [col for col in [Config.STATION_NAME_COL, Config.MUNICIPALITY_COL, Config.ALTITUDE_COL, Config.LATITUDE_COL, Config.LONGITUDE_COL, Config.ELEVATION_COL] if col in gdf_filtered.columns]
                gdf_metadata_reg = gdf_filtered[cols_metadata].drop_duplicates(subset=[Config.STATION_NAME_COL])

                # Calcula bounds ANTES de las llamadas
                try:
                    if not gdf_filtered.empty:
                        gdf_bounds_reg = gdf_filtered.total_bounds
                    else:
                        gdf_bounds_reg = None # O maneja el caso de gdf_filtered vacío
                        st.warning("No hay estaciones filtradas para definir los límites del mapa regional.")
                        # Decide si quieres detenerte aquí o continuar con bounds=None
                except Exception as e_bounds:
                     st.error(f"Error al calcular los límites geográficos: {e_bounds}")
                     gdf_bounds_reg = None
                
                fig1_reg, fig_var1_reg, error1_reg = None, None, "No ejecutado"; fig2_reg, fig_var2_reg, error2_reg = None, None, "No ejecutado"
                
            if gdf_bounds_reg is not None:
                try:
                    gdf_bounds_reg = gdf_filtered.total_bounds
                    fig1_reg, fig_var1_reg, error1_reg = create_interpolation_surface(
                        year=year1_reg, method=method1_reg, variogram_model=variogram_model1_reg,
                        gdf_bounds=gdf_bounds_reg,
                        gdf_metadata=gdf_metadata_reg,
                        df_anual_non_na=df_anual_melted
                    )
                except ImportError: st.error("Función 'create_interpolation_surface' no encontrada."); error1_reg = "ImportError"
                except TypeError as te1: 
                     st.error(f"Error Mapa 1 (TypeError): {te1}. Verifica argumentos 'create_interpolation_surface'.")
                     error1_reg = str(te1)
                except Exception as e1: st.error(f"Error Mapa 1: {e1}"); error1_reg = str(e1)

            if gdf_bounds_reg is not None:
                try:
                    fig2_reg, fig_var2_reg, error2_reg = create_interpolation_surface(
                        year=year2_reg, method=method2_reg, variogram_model=variogram_model2_reg,
                        gdf_bounds=gdf_bounds_reg,
                        gdf_metadata=gdf_metadata_reg,
                        df_anual_non_na=df_anual_melted
                    )
                except ImportError: st.error("Función 'create_interpolation_surface' no encontrada."); error2_reg = "ImportError"
                except TypeError as te2:
                     st.error(f"Error Mapa 2 (TypeError): {te2}. Verifica argumentos 'create_interpolation_surface'.")
                     error2_reg = str(te2)
                except Exception as e2: st.error(f"Error Mapa 2: {e2}"); error2_reg = str(e2)
                # --- FIN CORRECCIÓN ---

                with map_col1_reg:
                    if fig1_reg: st.plotly_chart(fig1_reg, use_container_width=True)
                    elif error1_reg and error1_reg != "No ejecutado": st.info(f"Mapa 1: {error1_reg}")
                    else: st.info("No se pudo generar el mapa 1.")

                with map_col2_reg:
                    if fig2_reg: st.plotly_chart(fig2_reg, use_container_width=True)
                    elif error2_reg and error2_reg != "No ejecutado": st.info(f"Mapa 2: {error2_reg}")
                    else: st.info("No se pudo generar el mapa 2.")

                if fig_var1_reg or fig_var2_reg:
                    st.markdown("---"); st.markdown("##### Variogramas (si aplica Kriging)")
                    col_var1, col_var2 = st.columns(2)
                    with col_var1:
                        if fig_var1_reg:
                            st.pyplot(fig_var1_reg); buf_var1 = io.BytesIO(); fig_var1_reg.savefig(buf_var1, format="png", bbox_inches='tight')
                            st.download_button("Descargar Variograma 1", buf_var1.getvalue(), f"variograma_{year1_reg}_{method1_reg}.png", "image/png")
                            plt.close(fig_var1_reg)
                        else: st.caption("Variograma no disponible para Mapa 1.")
                    with col_var2:
                        if fig_var2_reg:
                            st.pyplot(fig_var2_reg); buf_var2 = io.BytesIO(); fig_var2_reg.savefig(buf_var2, format="png", bbox_inches='tight')
                            st.download_button("Descargar Variograma 2", buf_var2.getvalue(), f"variograma_{year2_reg}_{method2_reg}.png", "image/png")
                            plt.close(fig_var2_reg)
                        else: st.caption("Variograma no disponible para Mapa 2.")
        # --- Fin Sección Regional ---

    # --- Pestaña Morfometría ---
    with morph_tab:
        st.subheader("Análisis Morfométrico de Cuenca(s)")
        st.info("Calcula métricas y curva hipsométrica para la(s) cuenca(s) seleccionadas en 'Superficies -> Por Cuenca'. Usa el DEM base.")
        
        # Obtener datos de la sesión
        unified_basin_gdf_morph = st.session_state.get('unified_basin_gdf')
        morph_results_morph = st.session_state.get('morph_results')
        basin_title_morph = st.session_state.get('selected_basins_title', 'Cuenca no seleccionada')
        dem_fixed_path_morph = st.session_state.get('dem_file_path') # Usar ruta base validada

        # Proceder solo si hay una cuenca seleccionada
        if unified_basin_gdf_morph is not None and not unified_basin_gdf_morph.empty:
            st.markdown(f"**Cuenca(s):** {basin_title_morph}")
            recalculate_morph = False # Bandera para recalcular si es necesario
            
            # Verificar si los resultados ya existen o si hay error previo
            if morph_results_morph is None or morph_results_morph.get("error"):
                 # Intentar recalcular solo si tenemos un DEM válido
                 if dem_fixed_path_morph and os.path.exists(dem_fixed_path_morph): 
                     recalculate_morph = True
                 else: 
                     # Mostrar advertencia si no hay DEM pero sí hay cuenca
                     st.warning("DEM base no encontrado o inválido. No se puede calcular morfometría.")
                     # Asegurarse de que no se intente mostrar resultados viejos o erróneos
                     morph_results_morph = None 

            # Recalcular morfometría si es necesario
            if recalculate_morph and dem_fixed_path_morph:
                 st.info("Calculando morfometría...")
                 try:
                     # Llamar a la función de cálculo (asegúrate que esté importada)
                     # from modules.analysis import calculate_morphometry 
                     morph_results_morph = calculate_morphometry(unified_basin_gdf_morph, dem_fixed_path_morph)
                     st.session_state['morph_results'] = morph_results_morph # Guardar resultado
                 except ImportError:
                      morph_results_morph = {"error": "Función 'calculate_morphometry' no encontrada."}
                      st.session_state['morph_results'] = morph_results_morph
                 except Exception as e_morph_tab: 
                     morph_results_morph = {"error": f"Error calculando morfometría: {e_morph_tab}"}
                     st.session_state['morph_results'] = morph_results_morph

            # Mostrar resultados de morfometría si existen y no hay error
            if morph_results_morph is not None:
                if morph_results_morph.get("error"): 
                    st.error(morph_results_morph["error"]) # Mostrar error si lo hubo
                else:
                    # Mostrar métricas principales
                    st.markdown("##### Métricas Principales")
                    c1m, c2m, c3m = st.columns(3)
                    c1m.metric("Área", f"{morph_results_morph.get('area_km2', 'N/A'):.2f} km²")
                    c2m.metric("Perímetro", f"{morph_results_morph.get('perimetro_km', 'N/A'):.2f} km")
                    c3m.metric("Índice de Forma", f"{morph_results_morph.get('indice_forma', 'N/A'):.2f}")
                    
                    c4m, c5m, c6m = st.columns(3)
                    alt_max=morph_results_morph.get('alt_max_m')
                    alt_min=morph_results_morph.get('alt_min_m')
                    alt_prom=morph_results_morph.get('alt_prom_m')
                    c4m.metric("Altitud Máxima", f"{alt_max:.0f} m" if alt_max is not None else "N/A")
                    c5m.metric("Altitud Mínima", f"{alt_min:.0f} m" if alt_min is not None else "N/A")
                    c6m.metric("Altitud Promedio", f"{alt_prom:.1f} m" if alt_prom is not None else "N/A")

                    # --- INICIO BLOQUE CURVA HIPSOMÉTRICA CORREGIDO ---
                    # Proceder solo si tenemos un DEM válido
                    if dem_fixed_path_morph and os.path.exists(dem_fixed_path_morph):
                        st.markdown("##### Curva Hipsométrica")
                        fig_hypso_plot = None # Inicializar figura como None
                        with st.spinner("Calculando curva hipsométrica..."):
                             try:
                                 # 1. Obtener el DICCIONARIO de datos
                                 # Asegúrate de importar: from modules.analysis import calculate_hypsometric_curve
                                 hypso_data = calculate_hypsometric_curve(unified_basin_gdf_morph, dem_fixed_path_morph)

                                 # 2. Verificar si hubo un error en el cálculo
                                 if hypso_data and hypso_data.get("error"):
                                     st.error(f"Error calculando curva: {hypso_data['error']}")
                                 # 3. Verificar si se obtuvieron datos válidos
                                 elif hypso_data and 'cumulative_area_percent' in hypso_data and 'elevations' in hypso_data:
                                     # 4. CREAR la figura Plotly usando los datos del diccionario
                                     fig_hypso_plot = go.Figure()
                                     # Añadir curva original
                                     fig_hypso_plot.add_trace(go.Scatter(
                                         x=hypso_data['cumulative_area_percent'],
                                         y=hypso_data['elevations'],
                                         mode='lines',
                                         name='Curva Hipsométrica',
                                         fill='tozeroy' # Rellenar área bajo la curva
                                     ))
                                     # Añadir curva ajustada (si existe)
                                     if 'fit_x' in hypso_data and 'fit_y' in hypso_data:
                                          fig_hypso_plot.add_trace(go.Scatter(
                                              x=hypso_data['fit_x'],
                                              y=hypso_data['fit_y'],
                                              mode='lines',
                                              name='Ajuste Polinomial',
                                              line=dict(color='red', dash='dash')
                                          ))
                                     # Configurar layout
                                     fig_hypso_plot.update_layout(
                                         title="Curva Hipsométrica",
                                         xaxis_title="Área Acumulada sobre Elevación (%)",
                                         yaxis_title="Elevación (m)",
                                         xaxis=dict(range=[0, 100]), 
                                         yaxis=dict(rangemode='tozero'), # Asegurar que Y empiece en 0
                                         legend=dict(x=0.01, y=0.99)
                                     )
                                     # Añadir anotaciones para ecuación y R² (si existen)
                                     annotation_text = []
                                     if hypso_data.get("equation"):
                                          annotation_text.append(f"Ecuación: {hypso_data['equation']}")
                                     if hypso_data.get("r_squared") is not None:
                                           annotation_text.append(f"R²: {hypso_data['r_squared']:.3f}")
                                     if annotation_text:
                                          fig_hypso_plot.add_annotation(
                                                x=0.98, y=0.02, xref="paper", yref="paper",
                                                text="<br>".join(annotation_text), 
                                                showarrow=False, align='right',
                                                font=dict(size=10, color="black"),
                                                bgcolor="rgba(255, 255, 255, 0.7)",
                                                bordercolor="black", borderwidth=1
                                          )
                                 else: # Si hypso_data está vacío o None por alguna razón
                                     st.warning("No se pudieron obtener datos válidos para la curva hipsométrica.")

                             except ImportError: 
                                 st.error("Función 'calculate_hypsometric_curve' no encontrada.")
                             except Exception as e_hypso: 
                                 st.error(f"Error inesperado preparando curva hipsométrica: {e_hypso}")
                                 import traceback
                                 st.error(traceback.format_exc())
                        
                        # 5. Mostrar la figura Plotly (si se creó)
                        if fig_hypso_plot is not None: 
                            st.plotly_chart(fig_hypso_plot, use_container_width=True)
                        # (Si no se creó, ya se mostró un error o advertencia arriba)

                    # Else para 'if dem_fixed_path_morph...'
                    else: 
                        st.info("DEM base no encontrado, no se puede generar curva hipsométrica.")
            # Else para 'if morph_results_morph is not None:'
            # (No es necesario un else aquí, si morph_results es None, no se muestra nada)

        # Else para 'if unified_basin_gdf_morph is not None...'
        else: 
            st.info("Selecciona una o más cuenca(s) en 'Mapas Avanzados -> Superficies -> Por Cuenca Específica' y genera el mapa para ver la morfometría aquí.")

    # --- Pestaña Mapa de Riesgo Climático ---
    with risk_map_tab:
        st.subheader("Mapa de Vulnerabilidad por Tendencias de Precipitación")
        st.info("""Interpola la tendencia (Pendiente de Sen) de estaciones (>10 años) para crear superficie de riesgo.""")
        try:
            fig_risk_contour = create_climate_risk_map(df_anual_melted, gdf_filtered)
            if fig_risk_contour: st.plotly_chart(fig_risk_contour, use_container_width=True)
        except NameError: st.error("Función 'create_climate_risk_map' no encontrada.")
        except Exception as e_risk: st.error(f"Error al generar mapa de riesgo: {e_risk}")

    # --- Pestaña Validación Cruzada ---
    with validation_tab:
        st.subheader("Validación Cruzada Dejando Uno Fuera (LOOCV)")
        st.info("Evalúa el error (RMSE) de métodos de interpolación para un año.")
        df_anual_non_na_val = df_anual_melted.dropna(subset=[Config.PRECIPITATION_COL])
        years_val = sorted(df_anual_non_na_val[Config.YEAR_COL].unique()) if not df_anual_non_na_val.empty else []
        if not years_val: st.warning("No hay datos anuales suficientes para validación.")
        else:
            selected_year_val = st.selectbox("Seleccione año para validar:", options=years_val, index=len(years_val)-1, key="year_select_validation")
            required_cols_val = [Config.STATION_NAME_COL, Config.LATITUDE_COL, Config.LONGITUDE_COL, Config.ALTITUDE_COL, Config.ELEVATION_COL]
            cols_present_val = [col for col in required_cols_val if col in gdf_filtered.columns]
            if len(cols_present_val) < 3: st.error("Faltan columnas (Nombre, Lat, Lon) para validación.")
            else:
                if st.button("Ejecutar Validación LOOCV", key="run_validation_button"):
                    with st.spinner(f"Ejecutando LOOCV para {selected_year_val}..."):
                        results_df = None; error_val = None; import_tb = False
                        try: 
                            import traceback; import_tb = True
                            results_df = perform_loocv_for_all_methods(
                                year=selected_year_val,
                                gdf_metadata=gdf_filtered[cols_present_val].drop_duplicates(subset=[Config.STATION_NAME_COL]),
                                df_anual_non_na=df_anual_non_na_val
                            )
                        except ImportError: error_val = "Función 'perform_loocv_for_all_methods' no encontrada."
                        except Exception as e_val: error_val = f"Error validación: {e_val}\n{traceback.format_exc() if import_tb else ''}"

                        if results_df is not None and not results_df.empty:
                            st.dataframe(results_df.style.format({'RMSE': '{:.2f}'}), use_container_width=True)
                            fig_val = px.bar(results_df.sort_values('RMSE'), x='Método', y='RMSE', title=f"RMSE por Método (LOOCV - {selected_year_val})", labels={'RMSE': 'RMSE (mm)'})
                            fig_val.update_layout(xaxis_title="Método")
                            st.plotly_chart(fig_val, use_container_width=True)
                        elif error_val: st.error(error_val)
                        else: st.warning("No se obtuvieron resultados de validación.")

    # --- Pestañas restantes (DESCOMENTADAS) ---
    with temporal_tab:
        st.subheader("Explorador Anual de Precipitación")
        df_anual_melted_non_na = df_anual_melted.dropna(subset=[Config.PRECIPITATION_COL])
        if not df_anual_melted_non_na.empty:
            all_years_int = sorted(df_anual_melted_non_na[Config.YEAR_COL].unique())
            controls_col, map_col = st.columns([1, 3])
            with controls_col:
                st.markdown("##### Opciones de Visualización")
                try:
                    selected_base_map_config, selected_overlays_config = display_map_controls(st, "temporal")
                except NameError:
                    st.error("Función 'display_map_controls' no encontrada.")
                    selected_base_map_config = {'tiles': 'cartodbpositron', 'attr': 'CartoDB'}
                    selected_overlays_config = []
                
                selected_year = None
                if len(all_years_int) > 1:
                    selected_year = st.slider('Seleccione un Año para Explorar', min_value=min(all_years_int), max_value=max(all_years_int), value=min(all_years_int), key="temporal_year_slider")
                elif len(all_years_int) == 1:
                    selected_year = all_years_int[0]; st.info(f"Mostrando único año disponible: {selected_year}")
                
                if selected_year:
                    st.markdown(f"#### Resumen del Año: {selected_year}")
                    df_year_filtered = df_anual_melted_non_na[df_anual_melted_non_na[Config.YEAR_COL] == selected_year]
                    if not df_year_filtered.empty:
                        num_stations = len(df_year_filtered)
                        st.metric("Estaciones con Datos", num_stations)
                        if num_stations > 0:
                            st.metric("Promedio Anual", f"{df_year_filtered[Config.PRECIPITATION_COL].mean():.0f} mm")
                            st.metric("Máximo Anual", f"{df_year_filtered[Config.PRECIPITATION_COL].max():.0f} mm")
                    
            with map_col:
                if selected_year:
                    try:
                        m_temporal = create_folium_map([4.57, -74.29], 5, {'tiles': selected_base_map_config['tiles'], 'attr': selected_base_map_config['attr']}, selected_overlays_config)
                        df_year_filtered_map = df_anual_melted_non_na[df_anual_melted_non_na[Config.YEAR_COL] == selected_year]
                        if not df_year_filtered_map.empty:
                            cols_to_merge_temp = [Config.STATION_NAME_COL, Config.MUNICIPALITY_COL, Config.ALTITUDE_COL, 'geometry']
                            if 'geometry' in gdf_filtered.columns:
                                df_map_data = pd.merge(df_year_filtered_map, gdf_filtered[cols_to_merge_temp].drop_duplicates(subset=[Config.STATION_NAME_COL]), on=Config.STATION_NAME_COL, how="inner")
                                if not df_map_data.empty:
                                    min_val, max_val = df_anual_melted_non_na[Config.PRECIPITATION_COL].min(), df_anual_melted_non_na[Config.PRECIPITATION_COL].max()
                                    if min_val >= max_val: max_val = min_val + 1
                                    colormap = cm.LinearColormap(colors=plt.cm.viridis.colors, vmin=min_val, vmax=max_val)
                                    
                                    for _, row in df_map_data.iterrows():
                                        if 'geometry' in row and row['geometry'] is not None and not row['geometry'].is_empty and pd.notna(row[Config.PRECIPITATION_COL]):
                                            try:
                                                popup_object = generate_annual_map_popup_html(row, df_anual_melted_non_na)
                                            except NameError:
                                                popup_object = f"<b>{row[Config.STATION_NAME_COL]}</b><br>Ppt: {row[Config.PRECIPITATION_COL]:.0f} mm"
                                            folium.CircleMarker(
                                                location=[row['geometry'].y, row['geometry'].x], radius=5,
                                                color=colormap(row[Config.PRECIPITATION_COL]), fill=True,
                                                fill_color=colormap(row[Config.PRECIPITATION_COL]), fill_opacity=0.8,
                                                tooltip=row[Config.STATION_NAME_COL], popup=popup_object
                                            ).add_to(m_temporal)
                                    
                                    temp_gdf = gpd.GeoDataFrame(df_map_data, geometry='geometry', crs=gdf_filtered.crs)
                                    if not temp_gdf.empty:
                                        bounds = temp_gdf.total_bounds
                                        if np.all(np.isfinite(bounds)): m_temporal.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                                else:
                                    st.warning("No se pudieron combinar datos anuales con geometría.")
                            else:
                                st.warning("gdf_filtered no contiene 'geometry'.")
                                
                            folium.LayerControl().add_to(m_temporal)
                            folium_static(m_temporal, height=700, width=None)
                    except NameError as e_temp:
                         st.error(f"Error en Pestaña Temporal: {e_temp}. Asegúrate que 'create_folium_map' y 'generate_annual_map_popup_html' estén definidas.")
                    except Exception as e_temp_map:
                         st.error(f"Error al crear mapa temporal: {e_temp_map}")
        else:
            st.warning("No hay datos anuales suficientes para 'Visualización Temporal'.")

    with race_tab:
        st.subheader("Ranking Anual de Precipitación por Estación")
        df_anual_valid_race = df_anual_melted.dropna(subset=[Config.PRECIPITATION_COL])
        if not df_anual_valid_race.empty:
            fig_racing = px.bar(
                df_anual_valid_race,
                x=Config.PRECIPITATION_COL, y=Config.STATION_NAME_COL,
                animation_frame=Config.YEAR_COL, orientation='h',
                labels={Config.PRECIPITATION_COL: 'Precipitación Anual (mm)', Config.STATION_NAME_COL: 'Estación'},
                title="Evolución de Precipitación Anual por Estación"
            )
            fig_racing.update_layout(height=max(600, len(stations_for_analysis) * 35), yaxis=dict(categoryorder='total ascending'))
            st.plotly_chart(fig_racing, use_container_width=True)
        else: st.warning("No hay datos anuales suficientes para 'Gráfico de Carrera'.")

    with anim_tab:
        st.subheader("Mapa Animado de Precipitación Anual")
        df_anual_valid_anim = df_anual_melted.dropna(subset=[Config.PRECIPITATION_COL])
        if not df_anual_valid_anim.empty:
            gdf_coords_anim = gdf_filtered[[Config.STATION_NAME_COL, Config.LATITUDE_COL, Config.LONGITUDE_COL]].drop_duplicates(subset=[Config.STATION_NAME_COL])
            df_anim_merged = pd.merge(df_anual_valid_anim, gdf_coords_anim, on=Config.STATION_NAME_COL, how="inner")
            if not df_anim_merged.empty:
                fig_mapa_animado = px.scatter_geo(
                    df_anim_merged,
                    lat=Config.LATITUDE_COL, lon=Config.LONGITUDE_COL,
                    color=Config.PRECIPITATION_COL, size=Config.PRECIPITATION_COL,
                    hover_name=Config.STATION_NAME_COL, animation_frame=Config.YEAR_COL,
                    projection='natural earth', title='Precipitación Anual por Estación',
                    color_continuous_scale=px.colors.sequential.Viridis
                )
                fig_mapa_animado.update_geos(fitbounds="locations", visible=True)
                st.plotly_chart(fig_mapa_animado, use_container_width=True)
            else: st.warning("No se pudieron combinar datos anuales con coordenadas.")
        else: st.warning("No hay datos anuales suficientes para 'Mapa Animado'.")

    with compare_tab:
        st.subheader("Comparación de Mapas Anuales")
        df_anual_valid_comp = df_anual_melted.dropna(subset=[Config.PRECIPITATION_COL])
        all_years_comp = sorted(df_anual_valid_comp[Config.YEAR_COL].unique())
        if len(all_years_comp) > 1:
            control_col_comp, map_col1_comp, map_col2_comp = st.columns([1, 2, 2])
            with control_col_comp:
                st.markdown("##### Controles de Mapa")
                try:
                    selected_base_map_config_comp, selected_overlays_config_comp = display_map_controls(st, "compare")
                except NameError:
                    st.error("Función 'display_map_controls' no encontrada.")
                    selected_base_map_config_comp = {'tiles': 'cartodbpositron', 'attr': 'CartoDB'}
                    selected_overlays_config_comp = []
                
                min_year_comp, max_year_comp = int(all_years_comp[0]), int(all_years_comp[-1])
                st.markdown("**Mapa 1**"); year1_comp = st.selectbox("Primer año", options=all_years_comp, index=len(all_years_comp) - 1, key="compare_year1")
                st.markdown("**Mapa 2**"); year2_comp = st.selectbox("Segundo año", options=all_years_comp, index=max(0, len(all_years_comp) - 2), key="compare_year2")
                
                min_precip_comp = int(df_anual_valid_comp[Config.PRECIPITATION_COL].min())
                max_precip_comp = int(df_anual_valid_comp[Config.PRECIPITATION_COL].max())
                if min_precip_comp >= max_precip_comp: max_precip_comp = min_precip_comp + 1
                color_range_comp = st.slider("Rango de Escala de Color (mm)", min_precip_comp, max_precip_comp, (min_precip_comp, max_precip_comp), key="color_compare")
                colormap_comp = cm.LinearColormap(colors=plt.cm.viridis.colors, vmin=color_range_comp[0], vmax=color_range_comp[1])

            # Definir la función interna
            def create_compare_map(data, year, col, gdf_stations_info, df_anual_full):
                col.markdown(f"**Precipitación en {year}**")
                try:
                    m_comp = create_folium_map([6.24, -75.58], 6, {'tiles': selected_base_map_config_comp['tiles'], 'attr': selected_base_map_config_comp['attr']}, selected_overlays_config_comp)
                    if not data.empty:
                        data_with_geom = pd.merge(data, gdf_stations_info, on=Config.STATION_NAME_COL)
                        gpd_data = gpd.GeoDataFrame(data_with_geom, geometry='geometry', crs=gdf_stations_info.crs)
                        for _, row in gpd_data.iterrows():
                            if pd.notna(row[Config.PRECIPITATION_COL]) and 'geometry' in row and row['geometry'] is not None and not row['geometry'].is_empty:
                                try:
                                    popup_object_comp = generate_annual_map_popup_html(row, df_anual_full)
                                except NameError:
                                    popup_object_comp = f"<b>{row[Config.STATION_NAME_COL]}</b><br>Ppt: {row[Config.PRECIPITATION_COL]:.0f} mm"
                                    
                                folium.CircleMarker(
                                    location=[row['geometry'].y, row['geometry'].x], radius=5,
                                    color=colormap_comp(row[Config.PRECIPITATION_COL]),
                                    fill=True, fill_color=colormap_comp(row[Config.PRECIPITATION_COL]), fill_opacity=0.8,
                                    tooltip=row[Config.STATION_NAME_COL], popup=popup_object_comp
                                ).add_to(m_comp)
                        if not gpd_data.empty:
                            bounds_comp = gpd_data.total_bounds
                            if np.all(np.isfinite(bounds_comp)): m_comp.fit_bounds([[bounds_comp[1], bounds_comp[0]], [bounds_comp[3], bounds_comp[2]]])
                    folium.LayerControl().add_to(m_comp)
                    with col: folium_static(m_comp, height=450, width=None)
                except NameError as e_comp_inner:
                     st.error(f"Error en create_compare_map: {e_comp_inner}. Faltan 'create_folium_map' o 'generate_annual_map_popup_html'.")
                except Exception as e_comp_map:
                     st.error(f"Error al crear mapa de comparación: {e_comp_map}")

            # Preparar datos y llamar a la función interna
            try:
                cols_geom = [Config.STATION_NAME_COL, Config.MUNICIPALITY_COL, Config.ALTITUDE_COL, 'geometry']
                if 'geometry' in gdf_filtered.columns:
                    gdf_geometries = gdf_filtered[cols_geom].drop_duplicates(subset=[Config.STATION_NAME_COL])
                    data_year1 = df_anual_valid_comp[df_anual_valid_comp[Config.YEAR_COL] == year1_comp]
                    data_year2 = df_anual_valid_comp[df_anual_valid_comp[Config.YEAR_COL] == year2_comp]
                    
                    create_compare_map(data_year1, year1_comp, map_col1_comp, gdf_geometries, df_anual_valid_comp)
                    create_compare_map(data_year2, year2_comp, map_col2_comp, gdf_geometries, df_anual_valid_comp)
                else:
                    st.warning("gdf_filtered no contiene 'geometry', no se puede generar comparación de mapas.")
            except Exception as e_comp_call:
                 st.error(f"Error al preparar datos para comparación: {e_comp_call}")
        else:
            st.warning("Se necesitan datos de al menos dos años diferentes para Comparación de Mapas.")
    
# --- Fin de la función display_advanced_maps_tab ---
            
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

            # --- INICIO DEBUG ACF/PACF ---
            st.write(f"Debug ACF/PACF: Datos para {station_to_analyze_acf} (primeras 10 filas):")
            st.dataframe(df_station[[Config.DATE_COL, Config.PRECIPITATION_COL]].head(10))
            st.write(f"Debug ACF/PACF: Descripción de datos de precipitación (sin NaNs):")
            st.dataframe(df_station[Config.PRECIPITATION_COL].dropna().describe())
            # --- FIN DEBUG ACF/PACF ---
            
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

    with autocorrelacion_tab: # Ajusta el nombre de la variable 'autocorrelacion_tab' si es diferente
        st.subheader("Análisis de Autocorrelación (ACF) y Autocorrelación Parcial (PACF)")
        
        # Widget para seleccionar la estación
        station_to_analyze_acf = st.selectbox(
            "Seleccione una estación:",
            options=stations_for_analysis, 
            key="acf_station_select" # Asegúrate que la key sea única
        )
        
        # Widget para seleccionar el número de rezagos
        max_lag = st.slider(
            "Número máximo de rezagos (meses):", 
            min_value=12,
            max_value=60, 
            value=24, 
            step=12,
            key="acf_max_lag_slider" # Asegúrate que la key sea única
        )

        # Procesar SOLO si se seleccionó una estación
        if station_to_analyze_acf:
            # Filtrar datos para la estación seleccionada
            df_station_acf = \
                df_monthly_filtered[df_monthly_filtered[Config.STATION_NAME_COL] ==
                                    station_to_analyze_acf].copy()
            
            # Verificar si hay datos después de filtrar
            if not df_station_acf.empty:
                # Preparar la serie de tiempo (índice de fecha, interpolar, quitar NaNs)
                df_station_acf.set_index(Config.DATE_COL, inplace=True)
                # Usar asfreq para asegurar frecuencia mensual y luego interpolar
                series_acf = df_station_acf[Config.PRECIPITATION_COL].asfreq('MS').interpolate(method='linear')
                series_acf.dropna(inplace=True) # Quitar NaNs restantes (al principio/final)

                # --- DEBUG ACF/PACF (AHORA DENTRO DEL IF y DESPUÉS de preparar series_acf) ---
                st.write(f"Debug ACF/PACF: Datos para {station_to_analyze_acf} (primeras 10 filas de la serie procesada):")
                st.dataframe(series_acf.head(10))
                st.write(f"Debug ACF/PACF: Descripción de datos de precipitación (serie procesada):")
                st.dataframe(series_acf.describe())
                # --- FIN DEBUG ACF/PACF ---

                # Verificar si hay suficientes datos DESPUÉS de procesar
                if len(series_acf) > max_lag:
                    try:
                        # Asegúrate de que las funciones estén importadas al principio de visualizer.py
                        # from modules.forecasting import create_acf_chart, create_pacf_chart 
                        
                        fig_acf = create_acf_chart(series_acf, max_lag)
                        st.plotly_chart(fig_acf, use_container_width=True)
                        
                        fig_pacf = create_pacf_chart(series_acf, max_lag)
                        st.plotly_chart(fig_pacf, use_container_width=True)
                        
                    except ImportError:
                         st.error("Funciones 'create_acf_chart' o 'create_pacf_chart' no encontradas.")
                    except Exception as e:
                        st.error(f"No se pudieron generar los gráficos de autocorrelación. Error: {e}")
                # Else para if len(series_acf) > max_lag:
                else:
                    st.warning(f"No hay suficientes datos ({len(series_acf)}) para el análisis de autocorrelación con {max_lag} rezagos después de procesar la serie.")
            # Else para if not df_station_acf.empty:
            else:
                st.warning(f"No se encontraron datos mensuales para la estación '{station_to_analyze_acf}' con los filtros actuales.")
                
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
    # @st.cache_data
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
        # @st.cache_data # Cache the conversion
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
    st.header("Análisis de Cobertura del Suelo por Cuenca (Raster)")

    # --- Configuración ---
    land_cover_raster_filename = "Cob25m_WGS84.tif"

    # --- LEYENDA (Basada en tu lista 1-13 Y la imagen de error 0 y 16) ---
    land_cover_legend = {
        1: "Zonas urbanizadas",
        2: "Zonas industriales o comerciales y redes de comunicación",
        3: "Zonas de extracción mineras y escombreras",
        4: "Zonas verdes artificializadas, no agrícolas",
        5: "Cultivos transitorios",
        6: "Cultivos permanentes",
        7: "Pastos",
        8: "Áreas Agrícolas Heterogéneas",
        9: "Bosques",
        10: "Áreas con vegetación herbácea y/o arbustiva",
        11: "Áreas abiertas, sin o con poca vegetación",
        12: "Áreas húmedas continentales",
        13: "Aguas continentales",
        
        # --- ENTRADAS FALTANTES (BASADAS EN image_e21236.png) ---
        0: "Sin Datos / Fuera de Área", # 0 es comúnmente NoData
        16: "NOMBRE_DE_LA_COBERTURA_PARA_16" # !! REEMPLAZA ESTO con el nombre correcto !!
        # (Si hay más códigos desconocidos, añádelos aquí)
    }
    # --- FIN LEYENDA ---

    projected_crs = "EPSG:3116" # CRS para cálculo de área (Ej. MAGNA-SIRGAS Bogota)
    # --- Fin Configuración ---

    # Construir ruta al raster
    _THIS_FILE_DIR = os.path.dirname(__file__)
    land_cover_raster_path = os.path.abspath(os.path.join(_THIS_FILE_DIR, '..', 'data', land_cover_raster_filename))

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
                # --- OBTENER ÁREA TOTAL PRIMERO ---
                total_area_km2 = None
                balance_results_check = st.session_state.get('balance_results')
                
                if balance_results_check and 'Area_km2' in balance_results_check:
                    total_area_km2 = balance_results_check['Area_km2']
                
                if total_area_km2 is None or total_area_km2 <= 0:
                    st.error("No se pudo obtener el Área Total de la cuenca (calculada en 'Mapas Avanzados -> Balance Hídrico').")
                    st.warning("Por favor, genera primero el Balance Hídrico para la cuenca seleccionada.")
                    st.session_state['current_coverage_stats'] = None
                    return # Detener si no tenemos el área total
                # --- FIN OBTENER ÁREA ---

                # Abrir el raster de cobertura
                with rasterio.open(land_cover_raster_path) as cover_src:
                    cover_crs = cover_src.crs
                    nodata_val = cover_src.nodata
                    internal_nodata = nodata_val if nodata_val is not None else 0
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
                        st.warning("No se encontraron píxeles de cobertura válidos dentro del área de la cuenca.")
                        st.session_state['current_coverage_stats'] = None
                        return

                    # --- CÁLCULO DE ÁREA POR PROPORCIÓN ---
                    coverage_stats_list = []
                    total_valid_pixels = counts.sum() # Total de píxeles válidos

                    for value, count in zip(unique_values, counts):
                        value_int = int(value)
                        class_name = land_cover_legend.get(value_int, f"Código Desconocido ({value_int})")
                        
                        # Calcular porcentaje de píxeles
                        percentage = (count / total_valid_pixels) * 100 if total_valid_pixels > 0 else 0
                        
                        # Calcular área en km² basado en la proporción del área total
                        area_km2 = (count / total_valid_pixels) * total_area_km2
                        
                        coverage_stats_list.append({
                            "ID_Clase": value_int, 
                            "Tipo de Cobertura": class_name,
                            "area_km2": area_km2, # Área en km² calculada por proporción
                            "percentage": percentage
                        })

                    coverage_stats = pd.DataFrame(coverage_stats_list).sort_values(by="percentage", ascending=False)
                    # --- FIN CÁLCULO PROPORCIÓN ---

                # Guardar resultados
                st.session_state['current_coverage_stats'] = coverage_stats
                # Guardar el área total correcta que usamos
                st.session_state['total_basin_area_km2'] = total_area_km2 

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
        if 'current_coverage_stats' in st.session_state and st.session_state['current_coverage_stats'] is not None:
            stats_df = st.session_state['current_coverage_stats']
            if not stats_df.empty:
                if any("Código Desconocido" in name for name in stats_df["Tipo de Cobertura"]):
                     st.warning("Hay códigos de cobertura desconocidos en la cuenca. Revisa la leyenda `land_cover_legend`.")
                
                fig_pie = px.pie(stats_df, names='Tipo de Cobertura', values='percentage',
                                 title=f"Distribución de Coberturas (%)", hole=0.3)
                fig_pie.update_traces(textposition='inside', textinfo='percent+label', sort=False)
                st.plotly_chart(fig_pie, use_container_width=True)

                # Mostrar Escorrentía
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
                st.info("No hay estadísticas de cobertura válidas para visualizar.")
        else:
            st.info("Procesa las coberturas primero para ver la visualización.")

    # --- Sección Escenarios Hipotéticos (SIN CAMBIOS) ---
    st.markdown("---")
    st.subheader("Modelado de Escenarios Hipotéticos de Cobertura")
    # ... (El código de escenarios que ya tenías va aquí, sin cambios) ...
    # Asegúrate de que el resto de la función (escenarios) esté presente
    balance_results = st.session_state.get('balance_results')
    q_actual_mm = None
    p_actual_mm = None
    if balance_results and not balance_results.get("error"):
        q_actual_mm = balance_results.get('Q_mm')
        p_actual_mm = balance_results.get('P_media_anual_mm')

    if q_actual_mm is not None and p_actual_mm is not None:
        st.markdown("##### Define los Porcentajes de Cobertura:")
        cn_values = {
            "Bosque (Buena condición)": 70,
            "Pasto (Buena condición)": 74,
            "Cultivos (Contorno, buena condición)": 78,
            "Suelo Desnudo": 86,
            "Áreas Urbanas/Impermeables": 92
        }
        with st.expander("Ver/Editar Números de Curva (CN) Base"):
             cn_bosque = st.number_input("CN Bosque", value=cn_values["Bosque (Buena condición)"], min_value=30, max_value=100)
             cn_pasto = st.number_input("CN Pasto", value=cn_values["Pasto (Buena condición)"], min_value=30, max_value=100)
             cn_cultivo = st.number_input("CN Cultivos", value=cn_values["Cultivos (Contorno, buena condición)"], min_value=30, max_value=100)
             cn_desnudo = st.number_input("CN Suelo Desnudo", value=cn_values["Suelo Desnudo"], min_value=30, max_value=100)
             cn_urbano = st.number_input("CN Urbano/Impermeable", value=cn_values["Áreas Urbanas/Impermeables"], min_value=30, max_value=100)
             cn_values_edited = {
                 "Bosque": cn_bosque, "Pasto": cn_pasto, "Cultivos": cn_cultivo,
                 "Suelo Desnudo": cn_desnudo, "Áreas Urbanas": cn_urbano
             }
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
                        cn_dict = cn_values_edited if 'cn_values_edited' in locals() else cn_values
                        cn_hip = (perc_bosque * cn_dict["Bosque"] +
                                  perc_pasto * cn_dict["Pasto"] +
                                  perc_cultivo * cn_dict["Cultivos"] +
                                  perc_desnudo * cn_dict["Suelo Desnudo"] +
                                  perc_urbano * cn_dict["Áreas Urbanas"]) / 100.0
                        cn_hip_safe = max(1, cn_hip)
                        s_hip = (1000 / cn_hip_safe) - 10
                        s_hip_mm = s_hip * 25.4
                        ia_hip_mm = 0.2 * s_hip_mm
                        q_hip_mm = 0.0
                        if p_actual_mm > ia_hip_mm:
                            q_hip_mm = ((p_actual_mm - ia_hip_mm)**2) / (p_actual_mm - ia_hip_mm + s_hip_mm)
                        cambio_perc = ((q_hip_mm - q_actual_mm) / q_actual_mm) * 100 if q_actual_mm != 0 else np.inf
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

# --- INICIO FUNCIÓN display_life_zones_tab ---
def display_life_zones_tab(**kwargs):
    st.header("Clasificación de Zonas de Vida (Holdridge)")
    st.info("""
    Genera un mapa de Zonas de Vida basado en precipitación media anual y altitud
    (según clasificación experta local). Puedes ajustar la resolución y aplicar una máscara de cuenca.
    """)

    # --- Configuración ---
    precip_raster_filename = "PPAMAnt.tif"
    # --- Fin Configuración ---

    # Rutas y DEM (SOLO BASE)
    _THIS_FILE_DIR = os.path.dirname(__file__)
    precip_raster_path = os.path.abspath(os.path.join(_THIS_FILE_DIR, '..', 'data', precip_raster_filename))
    
    # Obtener la ruta del DEM base desde el sidebar
    effective_dem_path_for_function = st.session_state.get('dem_file_path')
    dem_is_geographic = st.session_state.get('dem_crs_is_geographic', True)

    # Verificar existencia de archivos base
    if not os.path.exists(precip_raster_path):
        st.error(f"No se encontró el archivo raster de precipitación: {precip_raster_path}")
        return
    if not effective_dem_path_for_function:
         st.warning("DEM base no encontrado o no cargado (revisa el sidebar). No se puede generar el mapa.")
         return # Detener si no hay DEM

    # --- Controles ---
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        resolution_options = {"Baja (Rápido)": 8, "Media": 4, "Alta (Lento)": 2, "Original (Muy Lento)": 1}
        # CONFIGURAR 'Baja' como default para evitar 'Bad message format'
        selected_resolution = st.select_slider("Seleccionar Resolución del Mapa:", options=list(resolution_options.keys()), value="Baja (Rápido)", key="lifezone_resolution")
        downscale_factor = resolution_options[selected_resolution]
    with col_ctrl2:
        apply_basin_mask = st.toggle("Aplicar Máscara de Cuenca", value=True, key="lifezone_mask_toggle")
        basin_to_mask = None; mask_geometry_to_use = None
        if apply_basin_mask:
            basin_to_mask = st.session_state.get('unified_basin_gdf')
            basin_name_mask = st.session_state.get('selected_basins_title')
            if basin_to_mask is not None and not basin_to_mask.empty:
                st.success(f"Se usará la máscara: {basin_name_mask}")
                mask_geometry_to_use = basin_to_mask.geometry
            else:
                st.warning("No hay cuenca seleccionada en 'Mapas Avanzados' para usar como máscara."); apply_basin_mask = False
    # --- Fin Controles ---

    if st.button("Generar Mapa de Zonas de Vida", key="gen_life_zone_map"):
        
        mask_arg = mask_geometry_to_use if apply_basin_mask else None
        
        # --- CORRECCIÓN LLAMADA (Error 4): SIN 'maskgeometry' ---
        classified_raster, output_profile, name_map = generate_life_zone_map(
            effective_dem_path_for_function,
            precip_raster_path,
            mask_geometry=mask_arg, # <-- ¡¡CORREGIDO!!
            downscale_factor=downscale_factor
        )
        # --- FIN CORRECCIÓN ---

        # --- Visualización ---
        if classified_raster is not None and output_profile is not None and name_map is not None:
            st.subheader("Mapa de Zonas de Vida Generado")

            height, width = classified_raster.shape
            crs_profile = output_profile.get('crs')
            crs_str = str(crs_profile) if crs_profile else "Desconocido"
            nodata_val = output_profile.get('nodata', 0)
            transform = rasterio.transform.Affine(*output_profile['transform'][:6])

            x_start, y_start = transform.c, transform.f
            x_end = x_start + transform.a * width; y_end = y_start + transform.e * height
            x_coords = np.linspace(x_start + transform.a / 2, x_end - transform.a / 2, width)
            y_coords_raw = np.linspace(y_start + transform.e / 2, y_end - transform.e / 2, height)

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

                if transform.e < 0:
                    y_coords = y_coords_raw[::-1]
                    classified_raster_display = np.flipud(classified_raster)
                else:
                    y_coords = y_coords_raw
                    classified_raster_display = classified_raster

                raster_for_heatmap = classified_raster_display.astype(float)
                raster_for_heatmap[raster_for_heatmap == nodata_val] = np.nan # Hacer NoData transparente

                # get_zone_name=np.vectorize(lambda zid: name_map.get(zid, "NoData" if zid==nodata_val else f"ID {zid}?"))
                # hover_names_raster=get_zone_name(classified_raster_display)

                fig=go.Figure(data=go.Heatmap(
                    z=raster_for_heatmap,
                    x=x_coords,
                    y=y_coords,
                    colorscale=color_scale_discrete,
                    zmin=min(present_zone_ids)-0.5 if present_zone_ids else 0,
                    zmax=max(present_zone_ids)+0.5 if present_zone_ids else 1,
                    showscale=True,
                    colorbar=dict(title="ID Zona de Vida", tickvals=tick_values, ticktext=tick_texts, tickmode='array'),
                    hoverinfo='skip', # Mantener hover desactivado por estabilidad
                ))

            if fig is not None:
                fig.update_layout(title="Mapa de Zonas de Vida", xaxis_title=f"Coordenada X ({crs_str})", yaxis_title=f"Coordenada Y ({crs_str})", yaxis_scaleanchor="x", height=700)
                st.plotly_chart(fig, use_container_width=True)

                # --- LEYENDA DETALLADA ID -> Nombre + Hectáreas (CORREGIDO check CRS - Error 2) ---
                st.markdown("---"); st.subheader("Leyenda y Área por Zona de Vida Presente")
                area_hectares = []; pixel_counts = []; total_area_ha_calc = 0.0
                can_calculate_area = False

                if crs_profile and transform:
                    try:
                        if crs_profile.is_projected:
                            crs_units = crs_profile.linear_units.lower()
                            if 'metre' in crs_units or 'meter' in crs_units:
                                can_calculate_area = True
                            else: st.warning(f"ADVERTENCIA: Unidades CRS ({crs_units}) no son metros. Cálculo de área impreciso.")
                        elif crs_profile.is_geographic:
                             # Esto ahora usa el flag del sidebar, es más fiable
                             st.warning("ADVERTENCIA: El CRS del DEM está en grados geográficos. No se puede calcular el área. Use un DEM métrico.")
                        else: st.warning(f"Tipo de CRS ({crs_str}) no reconocido.")
                    except AttributeError: # Fallback
                         if dem_is_geographic: # Usar el flag del sidebar
                             st.warning("ADVERTENCIA: El CRS del DEM está en grados geográficos (WGS84). No se puede calcular el área. Use un DEM métrico.")
                         else: # Si no es geográfico pero falla .linear_units
                             st.warning(f"No se pudieron determinar las unidades del CRS ({crs_str}). Asumiendo métrico.")
                             # Intentar calcular de todas formas si no es geográfico
                             if crs_profile.is_projected: can_calculate_area = True 
                    except Exception as e_crs_check: st.warning(f"Error al verificar unidades del CRS: {e_crs_check}")

                if can_calculate_area:
                    pixel_size_x = abs(transform.a); pixel_size_y = abs(transform.e)
                    pixel_area_m2 = pixel_size_x * pixel_size_y; pixel_area_ha = pixel_area_m2 / 10000.0
                    for zone_id in present_zone_ids:
                        count = np.count_nonzero(classified_raster == zone_id)
                        pixel_counts.append(count); area_ha = count * pixel_area_ha
                        area_hectares.append(area_ha); total_area_ha_calc += area_ha
                    legend_data = {"ID": present_zone_ids, "Zona de Vida": [name_map.get(zid, f"ID {zid} Desconocido") for zid in present_zone_ids], "Área (ha)": area_hectares}
                    legend_df = pd.DataFrame(legend_data).sort_values(by="Área (ha)", ascending=False)
                    st.dataframe(legend_df.set_index('ID').style.format({'Área (ha)': '{:,.1f}'}), use_container_width=True)
                    st.caption(f"Área total clasificada (visible en mapa): {total_area_ha_calc:,.1f} ha")
                else:
                    legend_data = {"ID": present_zone_ids, "Zona de Vida": [name_map.get(zid, f"ID {zid} Desconocido") for zid in present_zone_ids]}
                    if present_zone_ids:
                         legend_df = pd.DataFrame(legend_data).sort_values(by="ID"); st.dataframe(legend_df.set_index('ID'), use_container_width=True)
                # --- FIN LEYENDA DETALLADA ---

                # --- EXPANDER INFO ---
                st.markdown("---")
                with st.expander("Sobre la Clasificación de Zonas de Vida"):
                     st.markdown("""
                     El sistema de Zonas de Vida de Holdridge... (basado en Altitud y Precipitación según tabla local).
                     ... (Resto del texto explicativo) ...
                     """)
                # --- FIN EXPANDER ---

        else:
             st.error("La generación del mapa de zonas de vida falló.")
    
    elif not effective_dem_path_for_function and os.path.exists(precip_raster_path):
         st.info("DEM base no encontrado o no cargado (revisa el sidebar). No se puede generar el mapa.")

































