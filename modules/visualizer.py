

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
import requests
from io import BytesIO
import os
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
from modules.config import Config
from shapely.ops import unary_union
from shapely.geometry import Point
from rasterio.mask import mask
import pymannkendall as mk
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from scipy import stats
from scipy.interpolate import griddata
from scipy.interpolate import Rbf
from modules.analysis import estimate_temperature, calculate_water_balance_turc, classify_holdridge_point, calculate_morphometry, calculate_hydrological_balance, calculate_hypsometric_curve, calculate_spei
from modules.analysis import generate_life_zone_raster, calculate_return_periods, calculate_percentiles_extremes, calculate_duration_curve, calculate_climatic_indices
from modules.openmeteo_api import get_historical_climate_average

# -----------------------------------------------------------------------------
# 1. FUNCIONES AUXILIARES
# -----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_img_as_base64(url):
    """
    Descarga una imagen y la convierte a string Base64.
    Esto permite incrustarla directamente en el HTML, evitando bloqueos de hotlinking.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://google.com"
    }
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            # Codificar a Base64
            encoded = base64.b64encode(r.content).decode()
            return f"data:image/png;base64,{encoded}"
    except Exception as e:
        print(f"Error Base64: {e}")
    return None
    
def display_current_filters(stations, regions, municipios, years):
    """Muestra un resumen colapsable de los filtros activos en toda la app."""
    with st.expander("ℹ️ Resumen de Filtros Activos (Sidebar)", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"**Período:** {years[0]} - {years[1]}")
        c2.markdown(f"**Regiones:** {', '.join(regions) if regions else 'Todas'}")
        c3.markdown(f"**Municipios:** {', '.join(municipios) if municipios else 'Todos'}")
        c4.markdown(f"**Estaciones:** {len(stations)} seleccionadas")
        if len(stations) < 10:
            st.caption(f"Selección: {', '.join(stations)}")

def analyze_point_data(lat, lon, df_long, gdf_stations, gdf_municipios, gdf_subcuencas):
    """
    Analiza un punto geográfico:
    1. Toponimia (Municipio/Cuenca).
    2. Datos Históricos (Interpolados).
    3. Variables Ambientales (Raster).
    """
    results = {}
    point_geom = Point(lon, lat) # Ojo: Shapely usa (lon, lat)
    
    # 1. CONTEXTO GEOGRÁFICO (TOPONIMIA)
    results['Municipio'] = "Desconocido"
    results['Cuenca'] = "Fuera de cuencas principales"
    
    try:
        # Buscar Municipio
        if gdf_municipios is not None and not gdf_municipios.empty:
            # Asumiendo CRS WGS84 (EPSG:4326)
            matches = gdf_municipios[gdf_municipios.contains(point_geom)]
            if not matches.empty:
                results['Municipio'] = matches.iloc[0].get('nombre', 'Sin Nombre')
        
        # Buscar Cuenca Hidrográfica
        if gdf_subcuencas is not None and not gdf_subcuencas.empty:
            matches_c = gdf_subcuencas[gdf_subcuencas.contains(point_geom)]
            if not matches_c.empty:
                results['Cuenca'] = matches_c.iloc[0].get('nombre', 'Sin Nombre')
    except Exception as e:
        print(f"Error espacial: {e}")

    # 2. INTERPOLACIÓN DE LLUVIA
    try:
        df_locs = gdf_stations.set_index(Config.STATION_NAME_COL)[['latitude', 'longitude']].copy()
        df_locs['dist'] = np.sqrt((df_locs['latitude'] - lat)**2 + (df_locs['longitude'] - lon)**2)
        nearest = df_locs.nsmallest(5, 'dist')
        nearest['weights'] = 1 / (nearest['dist']**2).replace(0, 0.00001)
        
        # Ppt Anual
        df_vecinas = df_long[df_long[Config.STATION_NAME_COL].isin(nearest.index)]
        annual_sums = df_vecinas.groupby([Config.STATION_NAME_COL, Config.YEAR_COL])[Config.PRECIPITATION_COL].sum().reset_index()
        avg_annual_ppt = annual_sums.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean()
        
        df_calc = pd.concat([avg_annual_ppt, nearest['weights']], axis=1, join='inner')
        results['Ppt_Media'] = (df_calc[Config.PRECIPITATION_COL] * df_calc['weights']).sum() / df_calc['weights'].sum()
        
        # Tendencia
        slopes = []
        for stn in nearest.index:
            df_st = df_long[df_long[Config.STATION_NAME_COL] == stn]
            df_ann = df_st.groupby(Config.YEAR_COL)[Config.PRECIPITATION_COL].sum()
            if len(df_ann) > 10:
                try: slopes.append(mk.original_test(df_ann).slope)
                except: slopes.append(0.0)
            else: slopes.append(0.0)
        results['Tendencia'] = np.average(slopes, weights=nearest['weights'].values[:len(slopes)])
        
    except: 
        results['Ppt_Media'] = 0
        results['Tendencia'] = 0

    # 3. RASTERS (ALTITUD Y COBERTURA)
    results['Altitud'] = 1500 # Default
    results['Cobertura'] = "No disponible" # Mensaje por defecto
    
    try:
        import rasterio
        # Altitud
        if os.path.exists(Config.DEM_FILE_PATH):
            with rasterio.open(Config.DEM_FILE_PATH) as src:
                val = list(src.sample([(lon, lat)]))[0][0]
                if val > -1000: results['Altitud'] = val
        
        # Cobertura
        if os.path.exists(Config.LAND_COVER_RASTER_PATH):
            with rasterio.open(Config.LAND_COVER_RASTER_PATH) as src:
                val = list(src.sample([(lon, lat)]))[0][0]
                
                # Leyenda completa
                legend = {
                    1: "Zonas Urbanas", 2: "Cultivos Transitorios", 3: "Pastos", 
                    4: "Áreas Agrícolas Heterogéneas", 5: "Bosques", 
                    6: "Vegetación Herbácea/Arbustiva", 7: "Áreas Abiertas", 
                    8: "Aguas Continentales", 9: "Bosque Fragmentado",
                    10: "Vegetación Secundaria", 11: "Zonas Degradadas", 12: "Humedales"
                }
                # Si es nodata o 0
                if val != src.nodata and val != 0:
                    results['Cobertura'] = legend.get(int(val), f"Clase {val}")
                else:
                    results['Cobertura'] = "Fuera de rango / Sin Datos"
    except Exception as e:
        results['Cobertura'] = f"Error Raster ({str(e)})"

    # 4. ZONA DE VIDA
    results['Zona_Vida'] = classify_holdridge_point(results['Ppt_Media'], results['Altitud'])
    
    return results

def get_weather_forecast_detailed(lat, lon):
    """
    Obtiene pronóstico detallado de Open-Meteo con 9 variables agrometeorológicas.
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": [
                "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
                "relative_humidity_2m_mean", "surface_pressure_mean",
                "et0_fao_evapotranspiration", "shortwave_radiation_sum", "wind_speed_10m_max"
            ],
            "timezone": "auto"
        }
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
        daily = data.get('daily', {})
        if not daily: return pd.DataFrame()

        # Crear DataFrame
        df = pd.DataFrame({
            'Fecha': pd.to_datetime(daily.get('time', [])),
            'T. Máx (°C)': daily.get('temperature_2m_max', []),
            'T. Mín (°C)': daily.get('temperature_2m_min', []),
            'Ppt. (mm)': daily.get('precipitation_sum', []),
            'HR Media (%)': daily.get('relative_humidity_2m_mean', []),
            'Presión (hPa)': daily.get('surface_pressure_mean', []),
            'ET₀ (mm)': daily.get('et0_fao_evapotranspiration', []),
            'Radiación SW (MJ/m²)': daily.get('shortwave_radiation_sum', []),
            'Viento Máx (km/h)': daily.get('wind_speed_10m_max', [])
        })
        return df
    except Exception:
        return pd.DataFrame()
        
def create_enso_chart(enso_data):
    """
    Genera el gráfico avanzado de ENSO con franjas de fondo para las fases.
    """
    if enso_data is None or enso_data.empty or Config.ENSO_ONI_COL not in enso_data.columns:
        return go.Figure().update_layout(title="Datos ENSO no disponibles", height=300)

    # Preparar datos
    data = enso_data.copy().sort_values(Config.DATE_COL).dropna(subset=[Config.ENSO_ONI_COL])
    
    # Definir colores de fondo según el valor ONI
    # El Niño >= 0.5 (Rojo), La Niña <= -0.5 (Azul), Neutral (Gris)
    conditions = [
        data[Config.ENSO_ONI_COL] >= 0.5,
        data[Config.ENSO_ONI_COL] <= -0.5
    ]
    # Colores con transparencia (rgba) para el fondo
    colors = ['rgba(255, 0, 0, 0.2)', 'rgba(0, 0, 255, 0.2)'] 
    data['color'] = np.select(conditions, colors, default='rgba(200, 200, 200, 0.2)') # Gris transparente

    # Calcular rangos para que las barras cubran todo el alto del gráfico
    y_min = data[Config.ENSO_ONI_COL].min() - 0.5
    y_max = data[Config.ENSO_ONI_COL].max() + 0.5

    fig = go.Figure()

    # 1. Barras de Fondo (Fases)
    # Usamos un gráfico de barras ancho para simular las franjas de fondo
    fig.add_trace(go.Bar(
        x=data[Config.DATE_COL],
        y=[y_max - y_min] * len(data), # Altura total
        base=y_min, # Empezar desde abajo
        marker_color=data['color'],
        width=86400000 * 30, # Ancho aprox de 1 mes en milisegundos para que se peguen
        hoverinfo="skip", # No mostrar tooltip para el fondo
        showlegend=False,
        name="Fase"
    ))

    # 2. Línea Principal (ONI)
    fig.add_trace(go.Scatter(
        x=data[Config.DATE_COL], 
        y=data[Config.ENSO_ONI_COL], 
        mode='lines', 
        line=dict(color='black', width=2),
        name='Anomalía ONI'
    ))

    # 3. Líneas de Umbral
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Umbral El Niño (+0.5)")
    fig.add_hline(y=-0.5, line_dash="dash", line_color="blue", annotation_text="Umbral La Niña (-0.5)")
    fig.add_hline(y=0, line_width=1, line_color="black")

    # 4. Leyenda Personalizada (Ficticia para mostrar los colores de fase)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', 
                             marker=dict(symbol='square', size=10, color='rgba(255, 0, 0, 0.5)'), name='El Niño'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', 
                             marker=dict(symbol='square', size=10, color='rgba(0, 0, 255, 0.5)'), name='La Niña'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', 
                             marker=dict(symbol='square', size=10, color='rgba(200, 200, 200, 0.5)'), name='Neutral'))

    # Configuración del Layout
    fig.update_layout(
        title="Fases del Fenómeno ENSO y Anomalía ONI",
        yaxis_title="Anomalía ONI (°C)",
        xaxis_title="Fecha",
        height=500,
        hovermode="x unified",
        yaxis_range=[y_min, y_max], # Fijar rango Y
        barmode='overlay', # Superponer la línea a las barras
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig
    
# 1. FUNCIONES AUXILIARES DE PARSEO Y DATOS
# -----------------------------------------------------------------------------

def parse_spanish_date(x):
    if isinstance(x, str):
        x = x.lower().strip()
        trans = {
            'ene': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'abr': 'Apr',
            'may': 'May', 'jun': 'Jun', 'jul': 'Jul', 'ago': 'Aug',
            'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dic': 'Dec'
        }
        for es, en in trans.items():
            if es in x:
                x = x.replace(es, en)
                break
        try: return pd.to_datetime(x, format='%b-%y')
        except: return pd.to_datetime(x, errors='coerce')
    return pd.to_datetime(x, errors='coerce')
    
# -----------------------------------------------------------------------------
# 2. FUNCIONES PRINCIPALES DE VISUALIZACIÓN
# -----------------------------------------------------------------------------

def display_welcome_tab():
    st.header(f"Bienvenido a {Config.APP_TITLE}")
    st.info("Sistema de Información Hidroclimática Integrada")
    st.markdown(Config.WELCOME_TEXT)

# -----------------------------------------------------------------------------
# NUEVA FUNCIÓN: CONEXIÓN CON IRI (COLUMBIA UNIVERSITY)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=12*3600)
def get_iri_enso_forecast():
    url_prob = "https://iri.columbia.edu/our-expertise/climate/forecasts/enso/current/?enso_tab=enso-cpc_plume"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url_prob, headers=headers, timeout=15)
        if r.status_code == 200:
            dfs = pd.read_html(io.StringIO(r.text), match="Season")
            if dfs:
                df = dfs[0]
                df.columns = ['Trimestre', 'La Niña', 'Neutral', 'El Niño']
                df_melted = df.melt(id_vars=['Trimestre'], var_name='Evento', value_name='Probabilidad')
                if df_melted['Probabilidad'].dtype == 'O':
                    df_melted['Probabilidad'] = df_melted['Probabilidad'].astype(str).str.replace('%', '').astype(float)
                return df_melted
    except: pass
    return pd.DataFrame()
    
# 2. NUEVA PESTAÑA UNIFICADA: MONITOREO Y TIEMPO REAL
# -----------------------------------------------------------------------------

def display_realtime_dashboard(df_long, gdf_stations, gdf_filtered, **kwargs):
    st.header("🚨 Centro de Monitoreo y Tiempo Real")
    
    tab_fc, tab_sat, tab_alert = st.tabs(["🌦️ Pronóstico Semanal", "🛰️ Satélite en Vivo", "📊 Alertas Históricas"])

    # --- SUB-PESTAÑA 1: PRONÓSTICO COMPLETO (RESTAURADO) ---
    with tab_fc:
        if gdf_filtered is None or gdf_filtered.empty: st.warning("Seleccione estaciones."); return
        sel_st = st.selectbox("Estación:", sorted(gdf_filtered[Config.STATION_NAME_COL].unique()))
        
        if sel_st:
            st_dat = gdf_filtered[gdf_filtered[Config.STATION_NAME_COL] == sel_st].iloc[0]
            with st.spinner("Consultando satélites y modelos meteorológicos..."):
                lat = st_dat['latitude'] if 'latitude' in st_dat else st_dat.geometry.y
                lon = st_dat['longitude'] if 'longitude' in st_dat else st_dat.geometry.x
                df = get_weather_forecast_detailed(lat, lon)
            
            if not df.empty:
                # 1. TARJETAS DE RESUMEN (HOY)
                td = df.iloc[0]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("T. Máx/Mín", f"{td['T. Máx (°C)']}/{td['T. Mín (°C)']}°C")
                c2.metric("Lluvia Hoy", f"{td['Ppt. (mm)']}mm")
                c3.metric("Viento Máx", f"{td['Viento Máx (km/h)']}km/h")
                c4.metric("Radiación", f"{td['Radiación SW (MJ/m²)']}MJ/m²")
                
                # 2. GRÁFICO PRINCIPAL (Climograma)
                st.markdown("#### 🌡️ Temperatura y Precipitación")
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=df['Fecha'], y=df['T. Máx (°C)'], name='Max', line=dict(color='red')), secondary_y=False)
                fig.add_trace(go.Scatter(x=df['Fecha'], y=df['T. Mín (°C)'], name='Min', line=dict(color='blue'), fill='tonexty'), secondary_y=False)
                fig.add_trace(go.Bar(x=df['Fecha'], y=df['Ppt. (mm)'], name='Ppt', marker_color='green', opacity=0.6), secondary_y=True)
                fig.update_layout(height=400, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True) # Plotly siempre usa container_width, eso está bien

                # 3. GRÁFICOS SECUNDARIOS (LO QUE FALTABA)
                st.markdown("#### 🍃 Condiciones Atmosféricas")
                col_g1, col_g2 = st.columns(2)
                
                with col_g1:
                    # Humedad y Presión
                    fig_atm = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_atm.add_trace(go.Scatter(x=df['Fecha'], y=df['HR Media (%)'], name='Humedad', line=dict(color='teal')), secondary_y=False)
                    fig_atm.add_trace(go.Scatter(x=df['Fecha'], y=df['Presión (hPa)'], name='Presión', line=dict(color='purple', dash='dot')), secondary_y=True)
                    fig_atm.update_layout(title="Humedad y Presión", height=350, legend=dict(orientation="h"))
                    fig_atm.update_yaxes(title_text="HR (%)", secondary_y=False)
                    fig_atm.update_yaxes(title_text="hPa", secondary_y=True, showgrid=False)
                    st.plotly_chart(fig_atm, use_container_width=True)

                with col_g2:
                    # Energía y Agua (Radiación + ET0)
                    fig_nrg = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_nrg.add_trace(go.Bar(x=df['Fecha'], y=df['Radiación SW (MJ/m²)'], name='Radiación', marker_color='gold'), secondary_y=False)
                    fig_nrg.add_trace(go.Scatter(x=df['Fecha'], y=df['ET₀ (mm)'], name='Evapotranspiración', line=dict(color='green')), secondary_y=True)
                    fig_nrg.update_layout(title="Energía y Ciclo del Agua", height=350, legend=dict(orientation="h"))
                    fig_nrg.update_yaxes(title_text="MJ/m²", secondary_y=False)
                    fig_nrg.update_yaxes(title_text="mm", secondary_y=True, showgrid=False)
                    st.plotly_chart(fig_nrg, use_container_width=True)

                # 4. TABLA DETALLADA
                with st.expander("Ver Tabla de Datos Completa"): 
                    st.dataframe(df, use_container_width=True)

    # --- SUB-PESTAÑA 2: SATÉLITE (ESTABILIZADA) ---
    with tab_sat:
        st.subheader("Observación Satelital")
        
        # Controles
        c_sat1, c_sat2 = st.columns([1, 3])
        with c_sat1:
            sat_mode = st.radio("Modo:", ["Animación (Visible)", "Mapa Interactivo (Lluvia/Nubes)"], index=1)
            show_stations_sat = st.checkbox("Mostrar Estaciones", value=True)
        
        with c_sat2:
            if sat_mode == "Animación (Visible)":
                # GIF Oficial NOAA (GeoColor) - Muy estable
                st.image(
                    "https://cdn.star.nesdis.noaa.gov/GOES16/ABI/GIFS/GOES16-ABI-GEOCOLOR-1000x1000.gif", 
                    caption="GOES-16 GeoColor (Tiempo Real)", 
                    use_column_width=True 
                )
            else:
                # Mapa Interactivo
                try:
                    # Usamos OpenStreetMap por estabilidad, centrado en la zona de interés
                    m = folium.Map(location=[6.2, -75.5], zoom_start=7, tiles="OpenStreetMap")
                    
                    # Capa de Radar de Lluvia (RainViewer - Cobertura Global y Rápida)
                    folium.TileLayer(
                        tiles="https://tile.rainviewer.com/nowcast/now/256/{z}/{x}/{y}/2/1_1.png",
                        attr="RainViewer",
                        name="Radar de Lluvia (Tiempo Real)",
                        overlay=True,
                        opacity=0.7
                    ).add_to(m)

                    # Capa de Nubes (Infrarrojo) - Opcional, si RainViewer falla
                    folium.TileLayer(
                        tiles="https://mesonet.agron.iastate.edu/cache/tile.py/1.0.0/goes-east-ir-4km-900913/{z}/{x}/{y}.png",
                        attr="IEM/NOAA",
                        name="Nubes Infrarrojo",
                        overlay=True,
                        opacity=0.5,
                        show=False # Oculta por defecto para no saturar
                    ).add_to(m)

                    # Mostrar Estaciones (Lo que pediste recuperar)
                    if show_stations_sat and gdf_filtered is not None and not gdf_filtered.empty:
                        for _, row in gdf_filtered.dropna(subset=['latitude', 'longitude']).iterrows():
                            folium.CircleMarker(
                                location=[row['latitude'], row['longitude']],
                                radius=3,
                                color='red',
                                fill=True,
                                fill_opacity=1,
                                tooltip=row[Config.STATION_NAME_COL]
                            ).add_to(m)

                    folium.LayerControl().add_to(m)
                    st_folium(m, height=600, width="100%")
                    
                    st.caption("🔵 Capa de Lluvia: Radar Meteorológico (RainViewer). ☁️ Capa de Nubes: GOES-16 Infrarrojo.")
                except Exception as e:
                    st.error(f"Error cargando el mapa satelital: {e}")

    # --- SUB-PESTAÑA 3: ALERTAS ---
    with tab_alert:
        if df_long is not None:
            umb = st.slider("Umbral (mm):", 0, 1000, 300)
            alts = df_long[df_long[Config.PRECIPITATION_COL] > umb]
            st.metric("Eventos Extremos", len(alts))
            if not alts.empty: 
                st.dataframe(alts.sort_values(Config.PRECIPITATION_COL, ascending=False).head(100), use_container_width=True)
        
def display_spatial_distribution_tab(gdf_filtered, df_long, gdf_municipios, gdf_subcuencas, gdf_predios=None, **kwargs):
    st.subheader("🗺️ Distribución Espacial y Análisis Puntual")
    
    # CSS para métricas compactas
    st.markdown("""
    <style>
    div[data-testid="stMetricValue"] { font-size: 1.1rem !important; }
    </style>
    """, unsafe_allow_html=True)

    st.info("👆 **Haga clic en el mapa** o ingrese coordenadas para analizar un punto específico.")

    if 'selected_point' not in st.session_state: st.session_state.selected_point = None

    tab_map, tab_avail, tab_matrix = st.tabs(["📍 Mapa Interactivo", "📊 Disponibilidad", "📅 Series Anuales"])
    
    # --- PESTAÑA 1: MAPA ---
    with tab_map:
        col_ctrl, col_map = st.columns([1, 3])
        with col_ctrl:
            st.markdown("#### Configuración")
            with st.expander("📍 Ingresar Coordenadas", expanded=False):
                in_lat = st.number_input("Latitud:", value=6.2, format="%.5f", key="mlat")
                in_lon = st.number_input("Longitud:", value=-75.5, format="%.5f", key="mlon")
                if st.button("Analizar Coordenada"):
                    st.session_state.selected_point = {'lat': in_lat, 'lng': in_lon}

            st.markdown("#### Capas")
            show_munis = st.checkbox("Municipios", value=True)
            show_cuencas = st.checkbox("Subcuencas", value=False)
            show_predios = st.checkbox("Predios", value=False)
            
            base_map_options = {
                "CartoDB Positron": {"tiles":"cartodbpositron", "attr":None},
                "OpenStreetMap": {"tiles":"OpenStreetMap", "attr":None},
                "Esri Satellite": {"tiles":"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", "attr":"Esri"}
            }
            base_map_name = st.selectbox("Mapa Base:", list(base_map_options.keys()))
            sel_tile = base_map_options[base_map_name]
        
        with col_map:
            # Centrar
            if st.session_state.selected_point:
                lat_c, lon_c, z = st.session_state.selected_point['lat'], st.session_state.selected_point['lng'], 11
            elif gdf_filtered is not None and not gdf_filtered.empty:
                v = gdf_filtered.dropna(subset=['latitude'])
                lat_c, lon_c, z = (v.latitude.mean(), v.longitude.mean(), 9) if not v.empty else (6.2, -75.5, 9)
            else: lat_c, lon_c, z = 6.2, -75.5, 9
            
            m = folium.Map(location=[lat_c, lon_c], zoom_start=z, tiles=sel_tile["tiles"], attr=sel_tile["attr"])
            
            try:
                if show_munis and not gdf_municipios.empty:
                    g = gdf_municipios.copy(); g['geometry'] = g.geometry.simplify(0.001)
                    folium.GeoJson(g, name="Municipios", style_function=lambda x:{'color':'gray','weight':1,'fillOpacity':0.05}, tooltip=folium.GeoJsonTooltip(['nombre']) if 'nombre' in g.columns else None).add_to(m)
                if show_cuencas and not gdf_subcuencas.empty:
                    g = gdf_subcuencas.copy(); g['geometry'] = g.geometry.simplify(0.001)
                    folium.GeoJson(g, name="Subcuencas", style_function=lambda x:{'color':'blue','weight':2,'fillOpacity':0}, tooltip=folium.GeoJsonTooltip(['nombre']) if 'nombre' in g.columns else None).add_to(m)
                if show_predios and gdf_predios is not None:
                    g = gdf_predios.copy(); g['geometry'] = g.geometry.simplify(0.0001)
                    folium.GeoJson(g, name="Predios", style_function=lambda x:{'color':'orange','weight':2,'fillOpacity':0.2}, tooltip=folium.GeoJsonTooltip(['nombre']) if 'nombre' in g.columns else None).add_to(m)
            except: pass

            # --- INICIO BLOQUE REEMPLAZO ---
            # Estaciones (Puntos verdes con Popup Inteligente)
            # --- BLOQUE DE ESTACIONES (CORREGIDO Y ALINEADO) ---
            if gdf_filtered is not None:
                marker_cluster = MarkerCluster().add_to(m)
                
                # Iterar sobre las estaciones
                for _, r in gdf_filtered.dropna(subset=['latitude']).iterrows():
                    
                    # 1. Filtrar datos SOLO de esta estación
                    df_st = df_long[df_long[Config.STATION_NAME_COL] == r[Config.STATION_NAME_COL]]
                    
                    # 2. Contar meses con datos REALES (>0)
                    # (Esto evita contar meses vacíos o nulos como válidos)
                    df_valid = df_st[df_st[Config.PRECIPITATION_COL] > 0]
                    n_months_real = len(df_valid)
                    
                    # 3. Promedio MENSUAL (Siempre confiable si hay datos)
                    if n_months_real > 0:
                        avg_ppt_mensual = df_valid[Config.PRECIPITATION_COL].mean()
                    else:
                        avg_ppt_mensual = 0
                    
                    # 4. Promedio ANUAL (Lógica Inteligente)
                    if n_months_real > 0:
                        # Intentar calcular años con al menos 6 meses de datos
                        counts = df_valid.groupby(Config.YEAR_COL).size()
                        years_ok = counts[counts >= 6].index
                        
                        if len(years_ok) > 0:
                            # Si hay años decentes, usarlos para el promedio real
                            df_years = df_valid[df_valid[Config.YEAR_COL].isin(years_ok)]
                            avg_ppt_anual = df_years.groupby(Config.YEAR_COL)[Config.PRECIPITATION_COL].sum().mean()
                            n_years_valid = len(years_ok)
                        else:
                            # Si los datos son muy fragmentados, ESTIMAR (Mensual * 12)
                            # Esto evita mostrar "0 mm" que confunde al usuario
                            avg_ppt_anual = avg_ppt_mensual * 12
                            n_years_valid = 0 # 0 indica que es una estimación por falta de años completos
                    else:
                        avg_ppt_anual = 0
                        n_years_valid = 0

                    # 5. Construir Popup HTML
                    html = f"""
                    <div style='font-family:sans-serif; font-size:12px; min-width:200px'>
                        <h5 style='margin:0; color:#2c3e50'>{r[Config.STATION_NAME_COL]}</h5>
                        <hr style='margin:5px 0'>
                        <b>Años Completos:</b> {n_years_valid}<br>
                        <b>Meses con Datos:</b> {n_months_real}<br>
                        <b>Promedio Anual (Est):</b> {avg_ppt_anual:,.0f} mm<br>
                        <b>Promedio Mensual:</b> {avg_ppt_mensual:.1f} mm
                    </div>
                    """
                    
                    # 6. Añadir Marcador al Mapa
                    folium.Marker(
                        [r['latitude'], r['longitude']], 
                        tooltip=f"{r[Config.STATION_NAME_COL]} ({avg_ppt_anual:.0f} mm)", 
                        popup=folium.Popup(html, max_width=300),
                        icon=folium.Icon(color="green", icon="cloud")
                    ).add_to(marker_cluster)
            # ---------------------------------------------------
            
            if st.session_state.selected_point:
                folium.Marker([st.session_state.selected_point['lat'], st.session_state.selected_point['lng']], popup="Punto Seleccionado", icon=folium.Icon(color="red", icon="info-sign")).add_to(m)

            folium.LayerControl().add_to(m)
            map_data = st_folium(m, width="100%", height=600)

            if map_data and map_data.get("last_clicked"):
                clicked = map_data["last_clicked"]
                if st.session_state.selected_point is None or abs(clicked['lat'] - st.session_state.selected_point['lat']) > 0.0001:
                    st.session_state.selected_point = {'lat': clicked['lat'], 'lng': clicked['lng']}
                    st.rerun()

    # --- RESULTADOS DEL PUNTO ---
    if st.session_state.selected_point:
        clat, clon = st.session_state.selected_point['lat'], st.session_state.selected_point['lng']
        st.markdown("---")
        st.subheader(f"📍 Análisis de Punto ({clat:.4f}, {clon:.4f})")
        
        with st.spinner("Consultando datos..."):
            p_data = analyze_point_data(clat, clon, df_long, gdf_filtered, gdf_municipios, gdf_subcuencas)
            fc = get_weather_forecast_detailed(clat, clon)
            
            # FILA 1: Contexto
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"**Ubicación:**<br>{p_data['Municipio']}<br><span style='color:gray; font-size:0.8em'>{p_data['Cuenca']}</span>", unsafe_allow_html=True)
            c2.metric("Altitud", f"{p_data['Altitud']:.0f} m")
            c3.metric("Ppt Histórica", f"{p_data['Ppt_Media']:.0f} mm/año")
            t_val = p_data['Tendencia']
            c4.metric("Tendencia Histórica", f"{t_val:+.1f} mm/año", delta_color="normal" if t_val>0 else "inverse")

            # FILA 2: Ambiental
            c5, c6 = st.columns(2)
            c5.metric("Zona de Vida", p_data['Zona_Vida'])
            c6.metric("Cobertura", p_data['Cobertura'])
            
            # FILA 3: Meteorología
            if not fc.empty:
                st.markdown("##### 🌦️ Condiciones Actuales y Pronóstico")
                today = fc.iloc[0]
                
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Temp. Promedio", f"{(today['T. Máx (°C)'] + today['T. Mín (°C)'])/2:.1f} °C")
                m2.metric("Lluvia Hoy", f"{today['Ppt. (mm)']} mm")
                m3.metric("Humedad Rel.", f"{today['HR Media (%)']} %")
                m4.metric("Viento Máx", f"{today['Viento Máx (km/h)']} km/h")
                m5.metric("Radiación", f"{today['Radiación SW (MJ/m²)']} MJ/m²")

                with st.expander("Ver Gráficos de Pronóstico (7 Días)", expanded=True):
                    # 1. Climograma
                    st.markdown("**🌡️ Temperatura y Precipitación**")
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(go.Scatter(x=fc['Fecha'], y=fc['T. Máx (°C)'], name='Max', line=dict(color='red')), secondary_y=False)
                    fig.add_trace(go.Scatter(x=fc['Fecha'], y=fc['T. Mín (°C)'], name='Min', line=dict(color='blue'), fill='tonexty'), secondary_y=False)
                    fig.add_trace(go.Bar(x=fc['Fecha'], y=fc['Ppt. (mm)'], name='Lluvia', marker_color='blue', opacity=0.5), secondary_y=True)
                    fig.update_layout(height=350, margin=dict(t=10,b=0,l=0,r=0), hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 2. Variables Adicionales (Tu Solicitud)
                    c_g1, c_g2 = st.columns(2)
                    with c_g1:
                        st.markdown("**🍃 Atmósfera (Humedad y Presión)**")
                        fig_atm = make_subplots(specs=[[{"secondary_y": True}]])
                        fig_atm.add_trace(go.Scatter(x=fc['Fecha'], y=fc['HR Media (%)'], name='HR %', line=dict(color='teal')), secondary_y=False)
                        fig_atm.add_trace(go.Scatter(x=fc['Fecha'], y=fc['Presión (hPa)'], name='Presión', line=dict(color='purple', dash='dot')), secondary_y=True)
                        fig_atm.update_layout(height=300, margin=dict(t=10,b=0,l=0,r=0), hovermode="x unified")
                        st.plotly_chart(fig_atm, use_container_width=True)
                    
                    with c_g2:
                        st.markdown("**☀️ Energía y Agua (Radiación y ET₀)**")
                        fig_nrg = make_subplots(specs=[[{"secondary_y": True}]])
                        fig_nrg.add_trace(go.Bar(x=fc['Fecha'], y=fc['Radiación SW (MJ/m²)'], name='Radiación', marker_color='orange'), secondary_y=False)
                        fig_nrg.add_trace(go.Scatter(x=fc['Fecha'], y=fc['ET₀ (mm)'], name='ET₀', line=dict(color='green')), secondary_y=True)
                        fig_nrg.update_layout(height=300, margin=dict(t=10,b=0,l=0,r=0), hovermode="x unified")
                        st.plotly_chart(fig_nrg, use_container_width=True)
                    
                    # Viento
                    st.markdown("**💨 Velocidad del Viento**")
                    fig_w = px.line(fc, x='Fecha', y='Viento Máx (km/h)', markers=True)
                    fig_w.update_traces(line_color='grey')
                    fig_w.update_layout(height=250, margin=dict(t=10,b=0,l=0,r=0))
                    st.plotly_chart(fig_w, use_container_width=True)

            else:
                st.warning("No se pudieron obtener datos meteorológicos en tiempo real.")
                
    # --- PESTAÑA 2: DISPONIBILIDAD ---
    with tab_avail:
        if df_long is not None and not gdf_filtered.empty:
            target = gdf_filtered[Config.STATION_NAME_COL].unique()
            sub = df_long[df_long[Config.STATION_NAME_COL].isin(target)]
            if not sub.empty:
                cnt = sub.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].count().reset_index()
                cnt.columns = ["Estación", "Registros"]
                sort_opt = st.radio("Ordenar:", ["Mayor a Menor", "Menor a Mayor", "Alfabético"], horizontal=True, key="sort_avail_unique")
                if "Mayor" in sort_opt: cnt = cnt.sort_values("Registros", ascending=True)
                elif "Menor" in sort_opt: cnt = cnt.sort_values("Registros", ascending=False)
                else: cnt = cnt.sort_values("Estación", ascending=False)
                
                fig = px.bar(cnt, x="Registros", y="Estación", orientation='h', height=max(500, len(cnt)*25))
                st.plotly_chart(fig, use_container_width=True)
            else: st.warning("Sin datos.")
    
    # --- PESTAÑA 3: MATRIZ ---
    with tab_matrix:
        if df_long is not None and not gdf_filtered.empty:
            target = gdf_filtered[Config.STATION_NAME_COL].unique()
            sub = df_long[df_long[Config.STATION_NAME_COL].isin(target)]
            if not sub.empty:
                piv = sub.pivot_table(index=Config.STATION_NAME_COL, columns=Config.YEAR_COL, values=Config.PRECIPITATION_COL, aggfunc='sum')
                st.dataframe(piv.style.background_gradient(cmap='viridis', axis=None).format("{:.0f}", na_rep="-"), use_container_width=True, height=600)
            else: st.warning("Sin datos.")
            
def display_graphs_tab(df_monthly_filtered, df_anual_melted, stations_for_analysis, **kwargs):
    st.subheader("📊 Análisis Gráfico Detallado")
    
    if df_monthly_filtered is None or df_monthly_filtered.empty:
        st.warning("No hay datos para mostrar.")
        return
    
    # Definición de Pestañas
    tab_names = [
        "1. Serie Anual", 
        "2. Ranking Multianual",
        "3. Serie Mensual", 
        "4. Ciclo Anual",
        "5. Distribución y Frecuencia"
    ]
    tabs = st.tabs(tab_names)
    
    # -------------------------------------------------------------------------
    # 1. SERIE ANUAL
    # -------------------------------------------------------------------------
    with tabs[0]:
        st.markdown("##### Precipitación Total Anual")
        
        # 1. Crear Figura (Asignar a variable específica)
        fig_anual = px.line(
            df_anual_melted, 
            x=Config.YEAR_COL, 
            y=Config.PRECIPITATION_COL, 
            color=Config.STATION_NAME_COL, 
            markers=True,
            labels={Config.PRECIPITATION_COL: "Lluvia (mm)", Config.YEAR_COL: "Año"}
        )
        
        # 2. Mostrar
        st.plotly_chart(fig_anual, use_container_width=True)
        
        # 3. Guardar en Memoria para el Reporte PDF (CRÍTICO)
        st.session_state['report_fig_anual'] = fig_anual
        
        # Descarga
        st.download_button(
            "📥 Descargar Datos Anuales (CSV)",
            df_anual_melted.to_csv(index=False).encode('utf-8'),
            "serie_anual.csv", "text/csv"
        )

    # -------------------------------------------------------------------------
    # 2. RANKING MULTIANUAL
    # -------------------------------------------------------------------------
    with tabs[1]:
        st.markdown("##### Ranking de Precipitación Media")
        
        avg_ppt = df_anual_melted.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean().reset_index()
        col_val = "Precipitación Media (mm)"
        avg_ppt.rename(columns={Config.PRECIPITATION_COL: col_val}, inplace=True)
        
        c_sort, _ = st.columns([1, 2])
        with c_sort:
            sort_opt = st.radio("Ordenar:", ["Mayor a Menor", "Menor a Mayor", "Alfabético"], horizontal=True, label_visibility="collapsed")
        
        if sort_opt == "Mayor a Menor": avg_ppt = avg_ppt.sort_values(col_val, ascending=False)
        elif sort_opt == "Menor a Mayor": avg_ppt = avg_ppt.sort_values(col_val, ascending=True)
        else: avg_ppt = avg_ppt.sort_values(Config.STATION_NAME_COL)
            
        fig_rank = px.bar(
            avg_ppt, x=Config.STATION_NAME_COL, y=col_val, color=col_val,
            color_continuous_scale=px.colors.sequential.Blues, text_auto='.0f'
        )
        st.plotly_chart(fig_rank, use_container_width=True)
        
        # Guardar
        st.session_state['report_fig_ranking'] = fig_rank
        
        st.download_button("📥 Descargar Ranking", avg_ppt.to_csv(index=False).encode('utf-8'), "ranking.csv", "text/csv")

    # -------------------------------------------------------------------------
    # 3. SERIE MENSUAL
    # -------------------------------------------------------------------------
    with tabs[2]:
        st.markdown("##### Serie Histórica Mensual")
        
        col_opts, col_chart = st.columns([1, 4])
        with col_opts:
            show_regional = st.checkbox("Ver Promedio Regional", value=False)
            show_markers = st.checkbox("Mostrar Puntos", value=False)
            
        with col_chart:
            fig_mensual = px.line(
                df_monthly_filtered, x=Config.DATE_COL, y=Config.PRECIPITATION_COL, 
                color=Config.STATION_NAME_COL, markers=show_markers,
                title="Precipitación Mensual"
            )
            
            if show_regional:
                reg_mean = df_monthly_filtered.groupby(Config.DATE_COL)[Config.PRECIPITATION_COL].mean().reset_index()
                fig_mensual.add_trace(go.Scatter(
                    x=reg_mean[Config.DATE_COL], y=reg_mean[Config.PRECIPITATION_COL],
                    mode='lines', name='PROMEDIO REGIONAL',
                    line=dict(color='black', width=3, dash='dash')
                ))
            
            st.plotly_chart(fig_mensual, use_container_width=True)
            
            # Guardar
            st.session_state['report_fig_mensual'] = fig_mensual
            
        st.download_button("📥 Descargar Mensual", df_monthly_filtered.to_csv(index=False).encode('utf-8'), "mensual.csv", "text/csv")

    # -------------------------------------------------------------------------
    # 4. CICLO ANUAL
    # -------------------------------------------------------------------------
    with tabs[3]:
        st.markdown("##### Régimen de Lluvias (Ciclo Promedio)")
        ciclo = df_monthly_filtered.groupby([Config.STATION_NAME_COL, Config.MONTH_COL])[Config.PRECIPITATION_COL].mean().reset_index()
        
        fig_ciclo = px.line(
            ciclo, x=Config.MONTH_COL, y=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL, 
            markers=True,
            labels={Config.MONTH_COL: "Mes", Config.PRECIPITATION_COL: "Lluvia Promedio (mm)"}
        )
        fig_ciclo.update_xaxes(tickmode='linear', tick0=1, dtick=1)
        st.plotly_chart(fig_ciclo, use_container_width=True)
        
        # Guardar
        st.session_state['report_fig_ciclo'] = fig_ciclo
        
        st.download_button("📥 Descargar Ciclo", ciclo.to_csv(index=False).encode('utf-8'), "ciclo.csv", "text/csv")

    # -------------------------------------------------------------------------
    # 5. DISTRIBUCIÓN
    # -------------------------------------------------------------------------
    with tabs[4]:
        st.markdown("##### Análisis Estadístico")
        
        c1, c2, c3 = st.columns(3)
        with c1: data_src = st.radio("Datos:", ["Anual (Totales)", "Mensual (Detalle)"], horizontal=True)
        with c2: chart_typ = st.radio("Gráfico:", ["Violín", "Histograma", "ECDF"], horizontal=True)
        with c3: sort_ord = st.selectbox("Orden:", ["Alfabético", "Mayor a Menor"])

        df_plot = df_anual_melted if "Anual" in data_src else df_monthly_filtered
        
        cat_orders = {}
        if sort_ord != "Alfabético":
            medians = df_plot.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].median()
            order_list = medians.sort_values(ascending=False).index.tolist()
            cat_orders = {Config.STATION_NAME_COL: order_list}

        if "Violín" in chart_typ:
            fig_dist = px.violin(df_plot, x=Config.STATION_NAME_COL, y=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL, box=True, points="all", category_orders=cat_orders)
            fig_dist.update_layout(showlegend=False)
        elif "Histograma" in chart_typ:
            fig_dist = px.histogram(df_plot, x=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL, marginal="box", barmode="overlay", opacity=0.7, category_orders=cat_orders)
        else:
            fig_dist = px.ecdf(df_plot, x=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL)

        fig_dist.update_layout(height=600, title=f"Distribución {data_src} - {chart_typ}")
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Guardar
        st.session_state['report_fig_dist'] = fig_dist
        
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
    """
    Muestra imágenes satelitales en tiempo real.
    Versión Robusta: Descarga segura de imágenes y mapas ligeros.
    """
    st.subheader("🛰️ Monitoreo Satelital (Tiempo Real)")

    tab_map, tab_anim = st.tabs(["🗺️ Mapa de Nubes (Interactivo)", "▶️ Animación (Últimas Horas)"])

    # --- TAB 1: MAPA INTERACTIVO ---
    with tab_map:
        col_map, col_info = st.columns([3, 1])
        with col_map:
            try:
                # Centrar mapa
                if gdf_filtered is not None and not gdf_filtered.empty:
                    if 'latitude' not in gdf_filtered.columns:
                        gdf_filtered['latitude'] = gdf_filtered.geometry.y
                        gdf_filtered['longitude'] = gdf_filtered.geometry.x
                    center_lat = gdf_filtered['latitude'].mean()
                    center_lon = gdf_filtered['longitude'].mean()
                else:
                    center_lat, center_lon = 6.0, -75.0

                m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

                # 1. Base: CartoDB Positron (Carga muy rápido y es limpia)
                folium.TileLayer(
                    tiles='CartoDB positron',
                    attr='CartoDB', name='Mapa Base Claro', overlay=False
                ).add_to(m)

                # 2. Overlay: Nubes (GOES-16 IR) - NASA GIBS
                # Usamos una URL WMS estándar que suele ser muy compatible
                folium.raster_layers.WmsTileLayer(
                    url='https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi',
                    name='Nubes (Infrarrojo)',
                    layers='GOES-East_ABI_Band13_Clean_Infrared',
                    fmt='image/png', transparent=True, opacity=0.5,
                    attr='NASA GIBS'
                ).add_to(m)

                # 3. Estaciones
                if gdf_filtered is not None and not gdf_filtered.empty:
                    from folium.plugins import MarkerCluster
                    mc = MarkerCluster(name="Estaciones").add_to(m)
                    for _, row in gdf_filtered.iterrows():
                        folium.CircleMarker(
                            location=[row['latitude'], row['longitude']],
                            radius=4, color='blue', fill=True, fill_color='cyan', fill_opacity=0.8,
                            popup=row.get(Config.STATION_NAME_COL, 'Estación')
                        ).add_to(mc)

                folium.LayerControl().add_to(m)
                st_folium(m, height=500, use_container_width=True)

            except Exception as e:
                st.error(f"Error cargando mapa: {e}")

        with col_info:
            st.info("""
            **Capas:**
            1. **Fondo:** CartoDB (Ligero).
            2. **Nubes:** Infrarrojo GOES-16.
            """)

    # --- TAB 2: ANIMACIÓN (GIF NOAA - Descarga Segura) ---
    with tab_anim:
        st.markdown("#### 🎬 Animación GeoColor (Sector Norte de Suramérica)")
        
        # URL Oficial NOAA (Northern South America)
        url_gif = "https://cdn.star.nesdis.noaa.gov/GOES16/ABI/SECTOR/nsa/GEOCOLOR/GOES16-NSA-GEOCOLOR-1000x1000.gif"
        
        with st.spinner("Descargando animación de la NOAA..."):
            gif_data = fetch_secure_content(url_gif)
        
        if gif_data:
            st.image(gif_data, caption="Animación GeoColor (Tiempo Real)", use_container_width=False, width=700)
        else:
            st.error("⚠️ No se pudo descargar la animación automáticamente.")
            st.markdown(f"[Haga clic aquí para verla directamente en la NOAA]({url_gif})")

def display_advanced_maps_tab(df_long, gdf_stations, gdf_subcuencas, gdf_filtered, **kwargs):
    """
    Muestra mapas avanzados de interpolación y morfometría.
    Versión Final: Recupera comparación regional y análisis de cuenca con popups enriquecidos y lógica robusta de promedios.
    """
    st.subheader("🌍 Superficies de Interpolación y Análisis Hidrológico")
    mode = st.radio("Modo de Análisis:", ["Regional (Comparación)", "Por Cuenca (Detallado)"], horizontal=True)
    
    # --- Función Auxiliar de Interpolación ---
    def run_interp(df, meth, bounds):
        try:
            margin_x = (bounds[2] - bounds[0]) * 0.1
            margin_y = (bounds[3] - bounds[1]) * 0.1
            gx, gy = np.mgrid[
                bounds[0]-margin_x : bounds[2]+margin_x : 100j, 
                bounds[1]-margin_y : bounds[3]+margin_y : 100j
            ]
            pts = df[['longitude', 'latitude']].values
            vals = df[Config.PRECIPITATION_COL].values
            
            if "Kriging" in meth or "RBF" in meth:
                rbf = Rbf(pts[:,0], pts[:,1], vals, function='thin_plate')
                gz = rbf(gx, gy)
            else: 
                method_scipy = 'cubic' if 'Spline' in meth else 'linear'
                gz = griddata(pts, vals, (gx, gy), method=method_scipy)
            return gx, gy, gz
        except Exception:
            return None, None, None

    # --- MODO 1: REGIONAL (COMPARACIÓN) ---
    if mode == "Regional (Comparación)":
        st.markdown("#### 🆚 Comparación de Periodos Climáticos")
        
        c1, c2 = st.columns(2)
        with c1:
            r1 = st.slider("Rango Años P1:", 1980, 2024, (1990, 2000), key="pr1")
            m1 = st.selectbox("Método P1:", ["IDW", "Spline", "Kriging/RBF"], key="m1")
        with c2:
            r2 = st.slider("Rango Años P2:", 1980, 2024, (2010, 2020), key="pr2")
            m2 = st.selectbox("Método P2:", ["IDW", "Spline", "Kriging/RBF"], key="m2")
            
        if st.button("🚀 Generar Mapas Comparativos"):
             def plot_reg(rng, meth, col, key_suffix):
                mask = (df_long[Config.YEAR_COL] >= rng[0]) & (df_long[Config.YEAR_COL] <= rng[1])
                df_period = df_long[mask]
                if df_period.empty:
                    col.warning(f"Sin datos para {rng}")
                    return

                df_avg = df_period.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean().reset_index()
                df_m = pd.merge(df_avg, gdf_stations, on=Config.STATION_NAME_COL).dropna(subset=['latitude','longitude'])
                
                if len(df_m) > 2:
                     gx, gy, gz = run_interp(df_m, meth, gdf_stations.total_bounds)
                     if gz is not None:
                         fig = go.Figure(go.Contour(z=gz.T, x=gx[:,0], y=gy[0,:], colorscale='Viridis', colorbar=dict(title='mm/año')))
                         fig.add_trace(go.Scatter(
                             x=df_m.longitude, y=df_m.latitude, mode='markers', 
                             marker=dict(color='red', size=5),
                             text=df_m.apply(lambda x: f"{x[Config.STATION_NAME_COL]}<br>{x[Config.PRECIPITATION_COL]:.0f} mm", axis=1)
                         ))
                         fig.update_layout(title=f"Precipitación Media ({rng[0]}-{rng[1]})", margin=dict(l=0, r=0, b=0, t=30))
                         col.plotly_chart(fig, use_container_width=True)

             plot_reg(r1, m1, c1, "p1")
             plot_reg(r2, m2, c2, "p2")

    # --- MODO 2: POR CUENCA (ANÁLISIS DETALLADO) ---
    else:
        st.markdown("#### ⛰️ Análisis Hidrológico por Cuenca")
        
        if gdf_subcuencas is None or gdf_subcuencas.empty: 
            st.warning("Sin capa de cuencas cargada."); return
            
        col_name = next((c for c in gdf_subcuencas.columns if 'nombre' in c.lower() or 'cuenca' in c.lower()), gdf_subcuencas.columns[0])
        sel_cuencas = st.multiselect("Seleccionar Cuenca(s):", sorted(gdf_subcuencas[col_name].unique().astype(str)))
        
        if sel_cuencas:
            c1, c2 = st.columns(2)
            rng = c1.slider("Periodo de Análisis:", 1980, 2025, (2000, 2020))
            meth = c2.selectbox("Método Interpolación:", ["IDW", "Kriging/RBF"])
            
            if st.button("Analizar Cuenca"):
                with st.spinner("Procesando hidrología..."):
                    # 1. Filtrar Geometría y Buffer
                    sub = gdf_subcuencas[gdf_subcuencas[col_name].isin(sel_cuencas)]
                    geom_union = gpd.GeoDataFrame({'geometry':[sub.unary_union]}, crs=gdf_subcuencas.crs)
                    buf = geom_union.geometry.buffer(0.2).unary_union # Buffer ~20km
                    gdf_buf = gpd.GeoDataFrame({'geometry': [buf]}, crs=gdf_stations.crs)
                    
                    # 2. Estaciones en zona
                    stns_in_zone = gpd.sjoin(gdf_stations, gdf_buf, predicate='intersects')
                    
                    if not stns_in_zone.empty:
                        ids = stns_in_zone[Config.STATION_NAME_COL].unique()
                        
                        # 3. Datos de Lluvia (Raw para cálculos de popup)
                        mask = (df_long[Config.STATION_NAME_COL].isin(ids)) & (df_long[Config.YEAR_COL] >= rng[0]) & (df_long[Config.YEAR_COL] <= rng[1])
                        df_sub_raw = df_long[mask].copy() # Datos crudos para estadísticas detalladas
                        
                        # 4. Datos para Interpolación (Promedio simple para el mapa de calor)
                        # Aquí usamos un promedio simple para la interpolación visual rápida
                        df_pts = df_sub_raw.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean().reset_index()
                        df_m = pd.merge(df_pts, gdf_stations, on=Config.STATION_NAME_COL).dropna(subset=['latitude','longitude'])
                        
                        if len(df_m) >= 3:
                            # Interpolación
                            b = geom_union.total_bounds 
                            bounds_interp = [b[0]-0.05, b[2]+0.05, b[1]-0.05, b[3]+0.05]
                            gx, gy, gz = run_interp(df_m, meth, bounds_interp)
                            
                            ppt_media = np.nanmean(gz) if gz is not None else df_m[Config.PRECIPITATION_COL].mean()
                            if np.isnan(ppt_media) or ppt_media <= 0: ppt_media = df_m[Config.PRECIPITATION_COL].mean()

                            # Cálculos Hidrológicos
                            morph = calculate_morphometry(geom_union)
                            bal = calculate_hydrological_balance(ppt_media, morph.get('alt_prom_m', 1500), geom_union)
                            bs_ts = df_sub_raw.groupby(Config.DATE_COL)[Config.PRECIPITATION_COL].mean()
                            c_run = bal.get('Q_mm', 0) / bal.get('P', 1) if bal.get('P', 1) > 0 else 0.4
                            fdc = calculate_duration_curve(bs_ts, c_run, morph.get('area_km2', 100))
                            idx = calculate_climatic_indices(bs_ts, morph.get('alt_prom_m', 1500))

                            st.session_state['basin_res'] = {
                                'ready':True, 'gz':gz, 'gx':gx, 'gy':gy, 'df':df_m, 'df_raw': df_sub_raw, # Guardamos raw para popup
                                'gdf_union':geom_union, 'buffer':gdf_buf, 'bal':bal, 'morph':morph, 
                                'fdc':fdc, 'idx':idx, 'bounds':bounds_interp, 'names':", ".join(sel_cuencas)
                            }
                        else:
                            st.error("Se necesitan al menos 3 estaciones para interpolar.")

            # --- MOSTRAR RESULTADOS ---
            res = st.session_state.get('basin_res')
            if res and res.get('ready'):
                
                # 1. Mapa Isoyetas
                st.markdown(f"##### Resultados: {res['names']}")
                fig = go.Figure(go.Contour(z=res['gz'].T, x=res['gx'][:,0], y=res['gy'][0,:], colorscale='Blues', colorbar=dict(title='mm/año')))
                fig.add_trace(go.Scatter(x=res['df'].longitude, y=res['df'].latitude, mode='markers', marker_color='red'))
                st.plotly_chart(fig, use_container_width=True)

                # 2. Métricas
                b = res['bal']
                c1,c2,c3 = st.columns(3)
                c1.metric("🌧️ Ppt Media", f"{b.get('P',0):.0f} mm")
                c2.metric("💧 Caudal (Q)", f"{b.get('Q_m3s', 0):.2f} m³/s")
                c3.metric("📦 Volumen", f"{b.get('Vol',0):.2f} Mm³")

                # 3. Curva Hipsométrica (Restaurada)
                if 'hypsometric_curve' in res['morph']:
                    st.markdown("---")
                    st.markdown("##### ⛰️ Curva Hipsométrica")
                    hypso = res['morph']['hypsometric_curve']
                    if hypso:
                        fig_h = px.line(x=hypso['area_acum_percent'], y=hypso['elevacion'], labels={'x':'% Área Acumulado', 'y':'Elevación (m)'})
                        fig_h.update_layout(height=350)
                        st.plotly_chart(fig_h, use_container_width=True)

                # 4. Mapa Contexto con POPUPS ENRIQUECIDOS (Lógica Corregida)
                st.markdown("---")
                st.markdown("##### 🗺️ Contexto y Estaciones (Buffer 20km)")
                
                minx, maxx, miny, maxy = res['bounds']
                m_ctx = folium.Map([(miny+maxy)/2, (minx+maxx)/2], zoom_start=10, tiles="CartoDB positron")
                
                folium.GeoJson(res['gdf_union'], style_function=lambda x:{'color':'blue','weight':2, 'fillOpacity':0.1}).add_to(m_ctx)
                folium.GeoJson(res['buffer'], style_function=lambda x:{'color':'gray','dashArray':'5,5','fill':False}).add_to(m_ctx)
                
                # Iterar para Popups
                stats_sub = res['df_raw'] # Datos completos históricos de estas estaciones
                
                for _, row in res['df'].iterrows():
                    st_name = row[Config.STATION_NAME_COL]
                    
                    # Filtrar datos de ESTA estación
                    st_data = stats_sub[stats_sub[Config.STATION_NAME_COL] == st_name]
                    
                    # --- LÓGICA DE CÁLCULO CORREGIDA ---
                    val_mensual = 0
                    val_anual = 0
                    n_years_valid = 0

                    if not st_data.empty:
                        # 1. Promedio Mensual (de todos los datos > 0)
                        val_mensual = st_data[st_data[Config.PRECIPITATION_COL] > 0][Config.PRECIPITATION_COL].mean()
                        
                        # 2. Años Completos (>= 10 meses con datos)
                        counts = st_data[st_data[Config.PRECIPITATION_COL] > 0].groupby(Config.YEAR_COL).size()
                        years_ok = counts[counts >= 10].index
                        n_years_valid = len(years_ok)
                        
                        # 3. Promedio Anual (Solo de años completos)
                        if n_years_valid > 0:
                            df_years = st_data[st_data[Config.YEAR_COL].isin(years_ok)]
                            val_anual = df_years.groupby(Config.YEAR_COL)[Config.PRECIPITATION_COL].sum().mean()
                        else:
                            # Fallback: Si no hay años completos, estimar
                            val_anual = val_mensual * 12
                    # -----------------------------------

                    mun = row.get(Config.MUNICIPALITY_COL, 'N/A')
                    alt = row.get(Config.ALTITUDE_COL, 'N/A')
                    # FIX: Eliminamos la dependencia de basin_map y usamos un valor genérico o columna directa si existe
                    subcuenca = row.get('subcuenca', row.get('cuenca', 'N/A'))

                    # HTML Popup
                    html = f"""
                    <div style='font-family:sans-serif; font-size:12px; min-width:220px'>
                        <h5 style='margin:0; color:#2c3e50; border-bottom:1px solid #ccc; padding-bottom:4px'>{st_name}</h5>
                        <div style='margin-top:5px; line-height:1.4'>
                            <b>Municipio:</b> {mun}<br>
                            <b>Subcuenca:</b> {subcuenca}<br>
                            <b>Altitud:</b> {alt} msnm
                        </div>
                        <div style='background-color:#f8f9fa; padding:5px; margin-top:5px; border-radius:4px'>
                            <b>Ppt Media Anual:</b> {val_anual:,.0f} mm<br>
                            <b>Ppt Media Mensual:</b> {val_mensual:.1f} mm<br>
                            <b>Años Completos:</b> {n_years_valid}
                        </div>
                    </div>
                    """
                    
                    iframe = folium.IFrame(html, width=240, height=160)
                    popup = folium.Popup(iframe, max_width=240)
                    
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=5,
                        color='darkred',
                        fill=True,
                        fillColor='red',
                        fillOpacity=0.8,
                        weight=1,
                        tooltip=f"{st_name} ({val_anual:.0f} mm)",
                        popup=popup
                    ).add_to(m_ctx)

                st_folium(m_ctx, height=500, width="100%")
            
# PESTAÑA DE PRONÓSTICO CLIMÁTICO (INDICES + GENERADOR)
# -----------------------------------------------------------------------------
def display_climate_forecast_tab(**kwargs):
    st.subheader("🔮 Pronóstico Climático & Fenómenos Globales")
    
    df_enso = kwargs.get('df_enso')
    
    tab_hist, tab_iri, tab_gen = st.tabs(["📜 Historia Índices", "🌎 Pronóstico Oficial (IRI)", "⚙️ Generador Prophet"])
    
    # --- TAB 1: HISTORIA ---
    with tab_hist:
        if df_enso is not None:
            c1, _ = st.columns([1,3])
            idx = c1.selectbox("Índice:", [Config.ENSO_ONI_COL, Config.SOI_COL, Config.IOD_COL])
            if idx in df_enso.columns:
                d = df_enso.dropna(subset=[idx, Config.DATE_COL]).sort_values(Config.DATE_COL)
                fig = px.line(d, x=Config.DATE_COL, y=idx, title=f"Evolución: {idx.upper()}")
                if idx == Config.ENSO_ONI_COL:
                    fig.add_hline(y=0.5, line_dash="dot", line_color="red", annotation_text="El Niño")
                    fig.add_hline(y=-0.5, line_dash="dot", line_color="blue", annotation_text="La Niña")
                st.plotly_chart(fig, use_container_width=True)

    # --- TAB 2: PRONÓSTICO IRI ---
    with tab_iri:
        st.markdown("#### Pronóstico ENSO (IRI / CPC)")
        
        with st.spinner("Consultando IRI..."):
            df_iri = get_iri_enso_forecast()
        
        # URL Oficial
        url_plume = "https://iri.columbia.edu/climate/ENSO/current/info/figure3.png"
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Probabilidades**")
            if not df_iri.empty:
                fig = px.bar(df_iri, x='Trimestre', y='Probabilidad', color='Evento', barmode='group',
                             color_discrete_map={'La Niña': 'blue', 'Neutral': 'gray', 'El Niño': 'red'}, height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Datos de probabilidad no disponibles.")

        with c2:
            st.markdown("**Pluma de Modelos**")
            # Implementación Base64 (Invulnerable a bloqueos de hotlink)
            with st.spinner("Descargando imagen segura..."):
                img_b64 = get_img_as_base64(url_plume)
            
            if img_b64:
                st.markdown(f'<img src="{img_b64}" style="width:100%; border-radius:5px;">', unsafe_allow_html=True)
            else:
                st.error("No se pudo descargar la imagen.")
                st.markdown(f"[Ver enlace original]({url_plume})")

    # --- TAB 3: PROPHET ---
    with tab_gen:
        st.markdown("#### Generador Prophet")
        indices = {}
        if df_enso is not None:
            cols_map = {Config.ENSO_ONI_COL: 'ONI', Config.SOI_COL: 'SOI', Config.IOD_COL: 'IOD'}
            for col, name in cols_map.items():
                if col in df_enso.columns:
                    indices[name] = df_enso[[Config.DATE_COL, col]].rename(columns={Config.DATE_COL:'ds', col:'y'}).dropna()
        
        if indices:
            sel_idx = st.selectbox("Índice:", list(indices.keys()))
            hor = st.slider("Meses:", 6, 60, 24)
            if st.button("Generar"):
                try:
                    m = Prophet()
                    m.fit(indices[sel_idx])
                    fut = m.make_future_dataframe(periods=hor, freq='MS')
                    fc = m.predict(fut)
                    fig = px.line(fc, x='ds', y='yhat', title=f"Proyección {sel_idx}")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")
# -----------------------------------------------------------------------------

def display_trends_and_forecast_tab(**kwargs):
    st.subheader("📉 Tendencias, Pronósticos y Riesgo")
    
    # Recuperar datos
    df_monthly = kwargs.get('df_monthly_filtered')
    df_anual = kwargs.get('df_anual_melted')
    stations = kwargs.get('stations_for_analysis')
    gdf_stations = kwargs.get('gdf_stations')

    if not stations or df_monthly is None or df_monthly.empty:
        st.warning("Seleccione estaciones en el panel lateral.")
        return

    # 1. SELECTOR GLOBAL DE SERIE (Estación o Regional)
    st.markdown("##### Configuración de la Serie de Tiempo")
    mode_fc = st.radio("Modo de Análisis:", ["Estación Individual", "Serie Regional (Promedio)"], horizontal=True, key="fc_mode_selector")

    ts_clean = None
    station_name_title = ""

    try:
        if mode_fc == "Estación Individual":
            selected_station = st.selectbox("Seleccionar Estación:", stations, key="trend_st")
            if selected_station:
                station_data = df_monthly[df_monthly[Config.STATION_NAME_COL] == selected_station].sort_values(Config.DATE_COL).set_index(Config.DATE_COL)
                ts_clean = station_data[Config.PRECIPITATION_COL].asfreq('MS').interpolate(method='time').dropna()
                station_name_title = selected_station
        else:
            # Promedio Regional
            station_name_title = "Serie Regional (Promedio)"
            reg_data = df_monthly.groupby(Config.DATE_COL)[Config.PRECIPITATION_COL].mean()
            ts_clean = reg_data.asfreq('MS').interpolate(method='time').dropna()

        # Validación final
        if ts_clean is None or len(ts_clean) < 24:
            st.error(f"Datos insuficientes (<24 meses) para {station_name_title}. Intente seleccionar otras estaciones.")
            return

    except Exception as e:
        st.error(f"Error preparando los datos: {e}")
        return

    # --- DEFINICIÓN DE REGRESORES DISPONIBLES (CORRECCIÓN: AL INICIO) ---
    # Definimos esto aquí para que esté visible tanto para SARIMA como para Prophet
    avail_regs = list(st.session_state.get('forecasted_regressors', {}).keys())
    
    # 2. PESTAÑAS
    tabs = st.tabs(["Tendencias", "Descomposición", "Autocorrelación", "SARIMA", "Prophet", "Comparación", "Mapa Riesgo"])

    # --- TAB 1: TENDENCIAS ---
    with tabs[0]:
        st.markdown(f"###### Tendencia: {station_name_title}")
        try:
            res = mk.original_test(ts_clean)
            c1, c2, c3 = st.columns(3)
            c1.metric("Tendencia", res.trend, delta=f"{res.slope:.3f} mm/mes")
            c2.metric("P-Value", f"{res.p:.4f}")
            c3.metric("Tau", f"{res.Tau:.3f}")
            fig = px.scatter(ts_clean.reset_index(), x=Config.DATE_COL, y=Config.PRECIPITATION_COL, trendline="ols", title="Tendencia Lineal")
            st.plotly_chart(fig, use_container_width=True)
        except: st.warning("No se pudo calcular tendencia.")

    # --- TAB 2: DESCOMPOSICIÓN ---
    with tabs[1]:
        try:
            decomp = seasonal_decompose(ts_clean, model='additive', period=12)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts_clean.index, y=decomp.trend, name='Tendencia'))
            fig.add_trace(go.Scatter(x=ts_clean.index, y=decomp.seasonal, name='Estacionalidad'))
            fig.add_trace(go.Scatter(x=ts_clean.index, y=decomp.resid, name='Residuo', mode='markers'))
            fig.update_layout(title="Descomposición Estacional", height=500)
            st.plotly_chart(fig, use_container_width=True)
        except: st.warning("Error en descomposición.")

    # --- TAB 3: AUTOCORRELACIÓN ---
    with tabs[2]:
        try:
            from statsmodels.tsa.stattools import acf, pacf
            nlags = min(24, len(ts_clean)//2 - 1)
            lag_acf = acf(ts_clean, nlags=nlags); lag_pacf = pacf(ts_clean, nlags=nlags)
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.bar(x=range(len(lag_acf)), y=lag_acf, title="ACF"), use_container_width=True)
            c2.plotly_chart(px.bar(x=range(len(lag_pacf)), y=lag_pacf, title="PACF"), use_container_width=True)
        except: pass

    # --- TAB 4: SARIMA (CON REGRESORES) ---
    with tabs[3]:
        st.markdown("#### Pronóstico SARIMA")
        
        # Selector de Regresores (Usando avail_regs definido arriba)
        sel_regs = st.multiselect("Usar Regresor Externo (ONI/SOI/IOD):", avail_regs, key="sarima_regs_sel", help="Debe generarlos primero en la pestaña 'Pronóstico Climático'")
        
        reg_df = None
        if sel_regs:
            try:
                reg_list = [st.session_state['forecasted_regressors'][k] for k in sel_regs]
                from functools import reduce
                reg_df = reduce(lambda left,right: pd.merge(left,right,on='ds', how='outer'), reg_list)
                reg_df = reg_df.rename(columns={'ds': Config.DATE_COL})
                # Rellenar posibles huecos en los regresores
                reg_df = reg_df.sort_values(Config.DATE_COL).interpolate(method='linear').bfill().ffill()
            except Exception as e:
                st.error(f"Error preparando regresores: {e}")

        horizon = st.slider("Horizonte (Meses):", 12, 48, 12, key="h_sarima")
        
        if st.button("Calcular SARIMA"):
            from modules.forecasting import generate_sarima_forecast
            with st.spinner("Calculando SARIMA..."):
                try:
                    ts_in = ts_clean.reset_index()
                    # Test size seguro
                    t_size = max(1, min(12, int(len(ts_clean)*0.2)))
                    
                    _, fc, ci, met, _ = generate_sarima_forecast(
                        ts_in, order=(1,1,1), seasonal_order=(1,1,1,12), 
                        horizon=horizon, test_size=t_size, regressors=reg_df
                    )
                    st.success(f"Modelo Ajustado. RMSE: {met['RMSE']:.1f}")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ts_clean.index[-60:], y=ts_clean[-60:], name="Histórico"))
                    fig.add_trace(go.Scatter(x=fc.index, y=fc, name="Pronóstico", line=dict(color='red')))
                    
                    # Intervalo de confianza
                    if not ci.empty:
                        fig.add_trace(go.Scatter(
                            x=pd.concat([pd.Series(ci.index), pd.Series(ci.index)[::-1]]),
                            y=pd.concat([ci.iloc[:, 0], ci.iloc[:, 1][::-1]]),
                            fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(color='rgba(255,255,255,0)'),
                            name='Intervalo Confianza'
                        ))
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state['sarima_res'] = fc
                except Exception as e:
                    st.error(f"Error SARIMA: {e}")

    # --- TAB 5: PROPHET (CON REGRESORES) ---
    with tabs[4]:
        st.markdown("#### Pronóstico Prophet")
        
        # Selector de Regresores (Usando avail_regs definido arriba)
        sel_regs_p = st.multiselect("Usar Regresor Externo (ONI/SOI/IOD):", avail_regs, key="prophet_regs_sel")
        
        reg_df_p = None
        if sel_regs_p:
            try:
                reg_list = [st.session_state['forecasted_regressors'][k] for k in sel_regs_p]
                from functools import reduce
                reg_df_p = reduce(lambda left,right: pd.merge(left,right,on='ds', how='outer'), reg_list)
                # Rellenar huecos
                reg_df_p = reg_df_p.sort_values('ds').interpolate(method='linear').bfill().ffill()
            except: pass

        horizon_p = st.slider("Horizonte (Meses):", 12, 48, 12, key="h_prophet")
        
        if st.button("Calcular Prophet"):
            from modules.forecasting import generate_prophet_forecast
            with st.spinner("Calculando Prophet..."):
                try:
                    ts_in = ts_clean.reset_index()
                    # Test size seguro
                    t_size = max(1, min(12, int(len(ts_clean)*0.2)))
                    
                    _, fc, met = generate_prophet_forecast(ts_in, horizon_p, test_size=t_size, regressors=reg_df_p)
                    st.success(f"Modelo Ajustado. RMSE: {met['RMSE']:.1f}")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ts_clean.index[-60:], y=ts_clean[-60:], name="Histórico"))
                    fig.add_trace(go.Scatter(x=fc['ds'], y=fc['yhat'], name="Pronóstico", line=dict(color='green')))
                    
                    # Intervalo de confianza
                    fig.add_trace(go.Scatter(
                        x=pd.concat([fc['ds'], fc['ds'][::-1]]),
                        y=pd.concat([fc['yhat_upper'], fc['yhat_lower'][::-1]]),
                        fill='toself', fillcolor='rgba(0,255,0,0.1)', line=dict(color='rgba(255,255,255,0)'),
                        name='Intervalo Confianza'
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state['prophet_res'] = fc[['ds', 'yhat']].set_index('ds')['yhat']
                except Exception as e:
                    st.error(f"Error Prophet: {e}")

    # --- TAB 6: COMPARACIÓN ---
    with tabs[5]:
        s, p = st.session_state.get('sarima_res'), st.session_state.get('prophet_res')
        if s is not None and p is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s.index, y=s, name="SARIMA", line=dict(color='red')))
            fig.add_trace(go.Scatter(x=p.index, y=p, name="Prophet", line=dict(color='green')))
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("Ejecute ambos modelos.")

    # --- TAB 7: RIESGO (CORREGIDO NAMEERROR) ---
    with tabs[6]:
        st.markdown("#### Mapa de Vulnerabilidad (Tendencias de Lluvia)")
        st.info("Interpolación de la Pendiente de Sen (mm/año).")
        
        if st.button("Generar Mapa de Riesgo"):
            with st.spinner("Calculando tendencias regionales..."):
                trend_data = []
                # CORRECCIÓN: Usamos df_anual que ya recuperamos al inicio
                if df_anual is not None:
                    stations_pool = df_anual[Config.STATION_NAME_COL].unique()
                    for stn in stations_pool:
                        sub = df_anual[df_anual[Config.STATION_NAME_COL] == stn]
                        if len(sub) > 10:
                            try:
                                res = mk.original_test(sub[Config.PRECIPITATION_COL])
                                if gdf_stations is not None:
                                    loc = gdf_stations[gdf_stations[Config.STATION_NAME_COL] == stn]
                                    if not loc.empty:
                                        iloc = loc.iloc[0]
                                        trend_data.append({
                                            'lat': iloc['latitude'], 'lon': iloc['longitude'], 
                                            'slope': res.slope, 'name': stn
                                        })
                            except: pass
                
                if len(trend_data) >= 4:
                    df_trend = pd.DataFrame(trend_data)
                    from scipy.interpolate import griddata
                    
                    grid_x, grid_y = np.mgrid[df_trend.lon.min():df_trend.lon.max():100j, 
                                              df_trend.lat.min():df_trend.lat.max():100j]
                    
                    grid_z = griddata(
                        df_trend[['lon', 'lat']].values, 
                        df_trend['slope'].values, 
                        (grid_x, grid_y), 
                        method='linear'
                    )
                    
                    fig = go.Figure(data=go.Contour(
                        z=grid_z.T, x=grid_x[:,0], y=grid_y[0,:],
                        colorscale='RdBu', 
                        colorbar=dict(title='Tendencia (mm/año)'),
                        zmid=0
                    ))
                    fig.add_trace(go.Scatter(x=df_trend.lon, y=df_trend.lat, mode='markers', text=df_trend.name, marker=dict(color='black')))
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No hay suficientes estaciones (>10 años datos) para interpolar.")
                    
def display_anomalies_tab(df_long, df_monthly_filtered, stations_for_analysis, **kwargs):
    st.subheader("⚠️ Análisis de Anomalías de Precipitación")
    
    df_enso = kwargs.get('df_enso')
    
    if df_monthly_filtered is None or df_monthly_filtered.empty:
        st.warning("No hay datos de precipitación filtrados.")
        return

    # 1. CONFIGURACIÓN
    st.markdown("#### Configuración del Análisis")
    col_conf1, col_conf2 = st.columns([1, 2])
    
    with col_conf1:
        reference_method = st.radio(
            "Calcular anomalía con respecto a:",
            ["El promedio de todo el período", "Una Normal Climatológica (período base fijo)"],
            key="anomaly_ref_method"
        )
        
    start_base, end_base = None, None
    
    if reference_method == "Una Normal Climatológica (período base fijo)":
        with col_conf2:
            all_years = sorted(df_long[Config.YEAR_COL].unique())
            if not all_years:
                st.error("No hay datos anuales disponibles.")
                return
                
            min_y, max_y = all_years[0], all_years[-1]
            
            def_start = 1991 if 1991 in all_years else min_y
            def_end = 2020 if 2020 in all_years else max_y
            
            c_start, c_end = st.columns(2)
            start_base = c_start.selectbox("Año Inicio Período Base:", all_years, index=all_years.index(def_start))
            end_base = c_end.selectbox("Año Fin Período Base:", all_years, index=all_years.index(def_end))
            
            if start_base > end_base:
                st.error("El año de inicio debe ser menor al año de fin.")
                return

    # 2. CÁLCULO
    with st.spinner("Calculando anomalías..."):
        # A. Definir datos de referencia
        if reference_method == "Una Normal Climatológica (período base fijo)":
            mask_base = (df_long[Config.YEAR_COL] >= start_base) & (df_long[Config.YEAR_COL] <= end_base)
            df_reference = df_long[mask_base]
            ref_text = f"Normal {start_base}-{end_base}"
        else:
            df_reference = df_long
            ref_text = "Promedio Histórico Total"
            
        # B. Serie regional mensual (promedio de estaciones seleccionadas)
        df_regional = df_monthly_filtered.groupby(Config.DATE_COL)[Config.PRECIPITATION_COL].mean().reset_index()
        df_regional[Config.MONTH_COL] = df_regional[Config.DATE_COL].dt.month
        
        # C. Climatología regional
        stations_list = df_monthly_filtered[Config.STATION_NAME_COL].unique()
        df_ref_stations = df_reference[df_reference[Config.STATION_NAME_COL].isin(stations_list)]
        climatology = df_ref_stations.groupby(Config.MONTH_COL)[Config.PRECIPITATION_COL].mean().reset_index()
        climatology.rename(columns={Config.PRECIPITATION_COL: 'clim_mean'}, inplace=True)
        
        # D. Unir y Restar
        df_anom = pd.merge(df_regional, climatology, on=Config.MONTH_COL, how='left')
        df_anom['anomalia'] = df_anom[Config.PRECIPITATION_COL] - df_anom['clim_mean']
        
        df_anom['color'] = np.where(df_anom['anomalia'] >= 0, 'blue', 'red')

    # 3. VISUALIZACIÓN
    tab_ts, tab_enso, tab_table = st.tabs(["Gráfico de Anomalías", "Anomalías por Fase ENSO", "Tabla de Eventos Extremos"])
    
    # --- A. SERIE TEMPORAL ---
    with tab_ts:
        st.markdown(f"##### Anomalías Mensuales (Ref: {ref_text})")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_anom[Config.DATE_COL], y=df_anom['anomalia'],
            marker_color=df_anom['color'], name='Anomalía'
        ))
        fig.update_layout(yaxis_title="Anomalía (mm)", xaxis_title="Fecha", height=500, showlegend=False)
        fig.add_hline(y=0, line_color="black", line_width=1)
        st.plotly_chart(fig, use_container_width=True)
        
        csv = df_anom.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar Anomalías (CSV)", csv, "anomalias.csv", "text/csv")

    # --- B. DISTRIBUCIÓN POR FASE ENSO ---
    with tab_enso:
        st.subheader("Distribución por Fase Climática")
        if df_enso is None or df_enso.empty:
            st.warning("No hay datos ENSO.")
        else:
            c_idx, _ = st.columns([1, 2])
            idx_name = c_idx.selectbox("Índice:", ["ONI (El Niño)", "SOI", "IOD"])
            idx_col_map = {"ONI (El Niño)": Config.ENSO_ONI_COL, "SOI": Config.SOI_COL, "IOD": Config.IOD_COL}
            target_idx_col = idx_col_map[idx_name]
            
            if target_idx_col in df_enso.columns:
                enso_clean = df_enso.copy()
                # Parseo seguro de fechas
                if enso_clean[Config.DATE_COL].dtype == 'object':
                    enso_clean[Config.DATE_COL] = enso_clean[Config.DATE_COL].apply(parse_spanish_date)
                else:
                    enso_clean[Config.DATE_COL] = pd.to_datetime(enso_clean[Config.DATE_COL], errors='coerce')

                df_merged = pd.merge(df_anom, enso_clean[[Config.DATE_COL, target_idx_col]], on=Config.DATE_COL, how='inner')
                
                if not df_merged.empty:
                    if idx_name == "ONI (El Niño)":
                        conds = [df_merged[target_idx_col] >= 0.5, df_merged[target_idx_col] <= -0.5]
                        choices = ['El Niño', 'La Niña']
                        colors = {'El Niño': '#d62728', 'La Niña': '#1f77b4', 'Neutral': 'lightgrey'}
                    elif idx_name == "SOI":
                        conds = [df_merged[target_idx_col] <= -7, df_merged[target_idx_col] >= 7]
                        choices = ['El Niño', 'La Niña']
                        colors = {'El Niño': '#d62728', 'La Niña': '#1f77b4', 'Neutral': 'lightgrey'}
                    else:
                        conds = [df_merged[target_idx_col] >= 0.4, df_merged[target_idx_col] <= -0.4]
                        choices = ['Positivo', 'Negativo']
                        colors = {'Positivo': '#d62728', 'Negativo': '#1f77b4', 'Neutral': 'lightgrey'}
                        
                    df_merged['Fase'] = np.select(conds, choices, default='Neutral')
                    
                    fig_enso = px.box(
                        df_merged, x='Fase', y='anomalia', color='Fase', color_discrete_map=colors,
                        points="all", title=f"Anomalías según Fase {idx_name}",
                        category_orders={"Fase": choices + ["Neutral"]}
                    )
                    fig_enso.update_layout(height=600, showlegend=False, yaxis_title="Anomalía (mm)")
                    fig_enso.add_hline(y=0, line_width=1, line_color="black", line_dash="dot")
                    st.plotly_chart(fig_enso, use_container_width=True)
                else:
                    st.warning("No hay datos coincidentes.")
            else:
                st.error(f"Columna {target_idx_col} no encontrada.")

    # --- C. TABLA DE EXTREMOS (CORREGIDA) ---
    with tab_table:
        st.subheader("Eventos Extremos")
        
        # CORRECCIÓN: Usar variables de Config en lugar de strings fijos
        cols_to_select = [Config.DATE_COL, Config.PRECIPITATION_COL, 'clim_mean', 'anomalia']
        cols_rename = ['Fecha', 'Ppt Real', 'Ppt Normal', 'Diferencia']
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🔴 Top 10 Meses Más Secos**")
            driest = df_anom.nsmallest(10, 'anomalia')[cols_to_select]
            driest.columns = cols_rename
            driest['Fecha'] = driest['Fecha'].dt.strftime('%Y-%m')
            st.dataframe(driest.style.format("{:.1f}", subset=['Ppt Real', 'Ppt Normal', 'Diferencia']), use_container_width=True)
            
        with c2:
            st.markdown("**🔵 Top 10 Meses Más Húmedos**")
            wettest = df_anom.nlargest(10, 'anomalia')[cols_to_select]
            wettest.columns = cols_rename
            wettest['Fecha'] = wettest['Fecha'].dt.strftime('%Y-%m')
            st.dataframe(wettest.style.format("{:.1f}", subset=['Ppt Real', 'Ppt Normal', 'Diferencia']), use_container_width=True)

def display_stats_tab(**kwargs):
    st.subheader("📊 Estadísticas Hidrológicas Detalladas")
    
    # Recuperar datos
    df_monthly = kwargs.get('df_monthly_filtered')
    df_anual = kwargs.get('df_anual_melted')
    
    if df_monthly is None or df_monthly.empty:
        st.warning("No hay datos para calcular estadísticas.")
        return

    tab1, tab2 = st.tabs(["Resumen General", "Matriz de Disponibilidad"])

    with tab1:
        # 1. Tabla de Resumen Estadístico
        st.markdown("##### Estadísticas Descriptivas por Estación")
        
        # Agrupar por estación
        stats_df = df_monthly.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].describe()
        
        # Añadir suma total histórica
        sum_total = df_monthly.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].sum()
        stats_df['Total Histórico (mm)'] = sum_total
        
        # Formatear
        st.dataframe(stats_df.style.format("{:.1f}"), use_container_width=True)
        
        # 2. Descargar
        st.download_button(
            "📥 Descargar Estadísticas (CSV)",
            stats_df.to_csv().encode('utf-8'),
            "estadisticas_precipitacion.csv",
            "text/csv"
        )

    with tab2:
        st.markdown("##### Disponibilidad de Datos (Mapa de Calor)")
        st.info("Muestra qué meses tienen datos registrados. Útil para identificar huecos.")
        
        # Crear matriz: Filas=Años, Columnas=Meses, Valor=Conteo
        # Pivotear datos
        try:
            matrix = df_monthly.pivot_table(
                index=df_monthly[Config.DATE_COL].dt.year,
                columns=df_monthly[Config.DATE_COL].dt.month,
                values=Config.PRECIPITATION_COL,
                aggfunc='count'
            ).fillna(0)
            
            fig_matrix = px.imshow(
                matrix,
                labels=dict(x="Mes", y="Año", color="Registros"),
                x=['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'],
                title="Matriz de Densidad de Datos (Registros por Mes)",
                color_continuous_scale="Greens"
            )
            fig_matrix.update_layout(height=600)
            st.plotly_chart(fig_matrix, use_container_width=True)
        except Exception as e:
            st.warning(f"No se pudo generar la matriz de disponibilidad: {e}")

def display_correlation_tab(**kwargs):
    st.subheader("🔗 Análisis de Correlación")
    
    # Recuperar datos
    df_monthly = kwargs.get('df_monthly_filtered')
    df_enso = kwargs.get('df_enso')
    
    # Validaciones
    if df_monthly is None or df_monthly.empty:
        st.warning("Faltan datos de precipitación para el análisis.")
        return
    
    # Crear pestañas
    tab1, tab2 = st.tabs(["Fenómenos Globales (ENSO)", "Matriz entre Estaciones"])

    # -------------------------------------------------------------------------
    # PESTAÑA 1: RELACIÓN LLUVIA REGIONAL VS ENSO (ONI)
    # -------------------------------------------------------------------------
    with tab1:
        if df_enso is None or df_enso.empty:
            st.warning("No se han cargado datos del índice ENSO.")
        else:
            st.markdown("##### Correlación: Índice Oceánico El Niño (ONI) vs. Precipitación")
            st.info("Analiza cómo la temperatura superficial del mar afecta la lluvia en la zona seleccionada.")

            try:
                # 1. Preparar copias de datos para no alterar los originales
                ppt_data = df_monthly.copy()
                enso_data = df_enso.copy()
                
                # 2. Asegurar formato de fecha en Precipitación
                ppt_data[Config.DATE_COL] = pd.to_datetime(ppt_data[Config.DATE_COL], errors='coerce')

                # 3. Asegurar formato de fecha en ENSO (Manejo de 'ene-70', etc.)
                # Usamos la función auxiliar parse_spanish_date si existe, o lógica inline
                if enso_data[Config.DATE_COL].dtype == 'object':
                    # Intento de conversión directa primero
                    enso_data[Config.DATE_COL] = pd.to_datetime(enso_data[Config.DATE_COL], errors='coerce')
                    
                    # Si falló (quedaron NaTs), intentamos el parseo manual de español
                    if enso_data[Config.DATE_COL].isnull().any():
                        def manual_spanish_parse(x):
                            if isinstance(x, str):
                                x = x.lower().strip()
                                trans = {
                                    'ene': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'abr': 'Apr',
                                    'may': 'May', 'jun': 'Jun', 'jul': 'Jul', 'ago': 'Aug',
                                    'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dic': 'Dec'
                                }
                                for es, en in trans.items():
                                    if es in x:
                                        x = x.replace(es, en)
                                        break
                                try: return pd.to_datetime(x, format='%b-%y')
                                except: return pd.NaT
                            return x
                        
                        # Recargar columna original para parsear
                        enso_original = df_enso.copy()
                        enso_data[Config.DATE_COL] = enso_original[Config.DATE_COL].apply(manual_spanish_parse)

                # 4. Limpiar fechas nulas en ambos lados
                ppt_data = ppt_data.dropna(subset=[Config.DATE_COL])
                enso_data = enso_data.dropna(subset=[Config.DATE_COL])

                # 5. Calcular Promedio Regional de Lluvia (una sola serie de tiempo)
                regional_ppt = ppt_data.groupby(Config.DATE_COL)[Config.PRECIPITATION_COL].mean().reset_index()
                
                # 6. Unir las dos series por fecha
                merged = pd.merge(regional_ppt, enso_data, on=Config.DATE_COL, how='inner')
                
                if len(merged) > 12:
                    c1, c2 = st.columns([2, 1])
                    
                    # Gráfico de Dispersión
                    with c1:
                        if Config.ENSO_ONI_COL in merged.columns:
                            fig = px.scatter(
                                merged, 
                                x=Config.ENSO_ONI_COL, 
                                y=Config.PRECIPITATION_COL, 
                                trendline="ols",
                                title="Dispersión: ONI vs Lluvia Regional",
                                labels={
                                    Config.ENSO_ONI_COL: "Anomalía ONI (°C)", 
                                    Config.PRECIPITATION_COL: "Lluvia Mensual Promedio (mm)"
                                },
                                opacity=0.6
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"No se encontró la columna '{Config.ENSO_ONI_COL}' en los datos ENSO.")

                    # Métricas Estadísticas
                    with c2:
                        if Config.ENSO_ONI_COL in merged.columns:
                            corr = merged[Config.ENSO_ONI_COL].corr(merged[Config.PRECIPITATION_COL])
                            st.markdown("#### Estadísticas")
                            st.metric("Correlación (Pearson)", f"{corr:.2f}")
                            
                            if abs(corr) > 0.5:
                                st.success("Existe una **fuerte** correlación.")
                            elif abs(corr) > 0.3:
                                st.info("Existe una correlación **moderada**.")
                            else:
                                st.warning("La correlación es **débil** o inexistente.")
                                
                            st.caption(f"Basado en {len(merged)} meses coincidentes.")
                else:
                    st.warning("No hay suficientes datos coincidentes en el tiempo entre la Lluvia y el ENSO para calcular la correlación.")
            
            except Exception as e:
                st.error(f"Error en el cálculo de correlación ENSO: {e}")

    # -------------------------------------------------------------------------
    # PESTAÑA 2: MATRIZ DE CORRELACIÓN ENTRE ESTACIONES
    # -------------------------------------------------------------------------
    with tab2:
        st.markdown("##### Matriz de Correlación de Precipitación entre Estaciones")
        st.info("Muestra qué tan similar es el comportamiento de la lluvia entre las diferentes estaciones seleccionadas. (1.0 = Idéntico, 0.0 = Sin relación).")
        
        try:
            # 1. Pivotear datos: Fechas en filas, Estaciones en columnas
            # Esto crea una tabla donde cada columna es una estación
            df_pivot = df_monthly.pivot_table(
                index=Config.DATE_COL, 
                columns=Config.STATION_NAME_COL, 
                values=Config.PRECIPITATION_COL
            )
            
            # Validar que haya suficientes datos
            if df_pivot.shape[1] < 2:
                st.warning("Se necesitan al menos 2 estaciones seleccionadas para calcular una matriz de correlación.")
            else:
                # 2. Calcular Matriz de Correlación (Pearson)
                corr_matrix = df_pivot.corr()
                
                # 3. Heatmap Interactivo
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="RdBu", # Rojo a Azul
                    zmin=-1, zmax=1,
                    title="Mapa de Calor de Correlaciones"
                )
                fig_corr.update_layout(height=700)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # 4. Botón de Descarga (CSV)
                csv_corr = corr_matrix.to_csv().encode('utf-8')
                st.download_button(
                    label="📥 Descargar Matriz de Correlación (CSV)",
                    data=csv_corr,
                    file_name="matriz_correlacion_estaciones.csv",
                    mime="text/csv",
                    key="dl_corr_matrix"
                )
                
        except Exception as e:
            st.error(f"Error generando la matriz de correlación: {e}")
        
def display_enso_tab(**kwargs):
    st.subheader("🌊 Fenómeno ENSO")
    df_enso = kwargs.get('df_enso')
    
    if df_enso is not None and not df_enso.empty:
        # Aplicar el mismo parseo seguro
        data = df_enso.copy()
        if data[Config.DATE_COL].dtype == 'object':
            data[Config.DATE_COL] = data[Config.DATE_COL].astype(str).apply(parse_spanish_date)
        else:
            data[Config.DATE_COL] = pd.to_datetime(data[Config.DATE_COL], errors='coerce')
            
        data = data.dropna(subset=[Config.DATE_COL]).sort_values(Config.DATE_COL)
        
        if Config.ENSO_ONI_COL in data.columns:
            st.plotly_chart(create_enso_chart(data), use_container_width=True)
        else:
            st.warning("Columna ONI no encontrada.")
    else:
        st.info("No hay datos ENSO cargados.")

def display_life_zones_tab(df_long, gdf_stations, **kwargs):
    st.subheader("🌱 Zonas de Vida (Sistema Holdridge)")
    
    tab_raster, tab_puntos = st.tabs(["🗺️ Mapa Raster (Continuo)", "📍 Estaciones (Puntos)"])
    
    # --- PESTAÑA 1: MAPA RASTER ---
    with tab_raster:
        st.info("Genera una superficie continua de zonas de vida cruzando los mapas de Elevación y Precipitación.")
        
        col1, col2 = st.columns(2)
        with col1:
            res_option = st.select_slider("Resolución:", options=["Baja (Rápido)", "Media", "Alta (Lento)"], value="Baja (Rápido)")
            downscale = 8 if "Baja" in res_option else (4 if "Media" in res_option else 1)
            
        with col2:
            use_mask = st.checkbox("Recortar por Cuenca Seleccionada", value=True)
            
        # Verificar si hay cuenca en memoria (CORRECCIÓN KEYERROR)
        basin_geom = None
        if use_mask:
            res_basin = st.session_state.get('basin_results')
            if res_basin and res_basin.get('ready'):
                # Intentar obtener la geometría con ambas claves posibles
                basin_geom = res_basin.get('gdf_union', res_basin.get('geom'))
                
                if basin_geom is not None:
                    st.success(f"Máscara activa: {res_basin.get('names', 'Cuenca')}")
                else:
                    st.warning("Error: Geometría de cuenca no encontrada en memoria.")
            else:
                st.caption("⚠️ No hay cuenca seleccionada (Ver 'Mapas Avanzados'). Se mostrará toda la región.")

        if st.button("Generar Mapa de Zonas de Vida"):
            if not os.path.exists(Config.DEM_FILE_PATH) or not os.path.exists(Config.PRECIP_RASTER_PATH):
                st.error("Faltan los archivos raster base (DEM o PPT) en la carpeta 'data'.")
            else:
                with st.spinner("Procesando rasters..."):
                    # Importar función generadora (asegúrate de tenerla en analysis.py)
                    # Si no está, avísame y te paso analysis.py completo
                    try:
                        from modules.analysis import generate_life_zone_raster
                        lz_arr, transform, crs = generate_life_zone_raster(
                            Config.DEM_FILE_PATH, 
                            Config.PRECIP_RASTER_PATH, 
                            mask_geom=basin_geom, 
                            downscale_factor=downscale
                        )
                        
                        if isinstance(crs, str): # Error
                            st.error(f"Error: {crs}")
                        elif lz_arr is not None:
                            # Leyenda Holdridge
                            legend_map = {
                                1: "Bosque Seco Tropical", 2: "Bosque Húmedo Tropical", 
                                3: "Bosque Muy Húmedo Tropical", 4: "Bosque Pluvial Tropical",
                                5: "Bosque Seco Premontano", 6: "Bosque Húmedo Premontano", 
                                7: "Bosque Muy Húmedo Premontano", 8: "Bosque Pluvial Premontano",
                                9: "Bosque Seco Montano Bajo", 10: "Bosque Húmedo Montano Bajo", 
                                11: "Bosque Muy Húmedo Montano Bajo", 12: "Bosque Pluvial Montano Bajo",
                                13: "Bosque Húmedo Montano", 14: "Bosque Muy Húmedo Montano"
                            }
                            
                            # Plotly Heatmap
                            h, w = lz_arr.shape
                            x0, y0 = transform.c, transform.f
                            dx, dy = transform.a, transform.e
                            x_coords = np.linspace(x0, x0 + dx*w, w)
                            y_coords = np.linspace(y0, y0 + dy*h, h)
                            
                            # Flip si es necesario (coordenadas norte suelen ser negativas en transform)
                            if dy < 0: y_coords = y_coords[::-1]; lz_arr = np.flipud(lz_arr)

                            plot_arr = lz_arr.astype(float)
                            plot_arr[plot_arr == 0] = np.nan
                            
                            fig = go.Figure(data=go.Heatmap(
                                z=plot_arr, x=x_coords, y=y_coords,
                                colorscale='Jet', showscale=False, hoverongaps=False
                            ))
                            fig.update_layout(title="Mapa de Zonas de Vida (Holdridge)", height=600, yaxis_scaleanchor="x")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Tabla de Áreas (Aprox)
                            unique, counts = np.unique(lz_arr[lz_arr!=0], return_counts=True)
                            data = []
                            for v, c in zip(unique, counts):
                                data.append({"Zona de Vida": legend_map.get(v, f"Clase {v}"), "Píxeles": c, "%": c/counts.sum()*100})
                            st.dataframe(pd.DataFrame(data).sort_values("%", ascending=False).style.format({"%": "{:.1f}%"}), use_container_width=True)
                    
                    except ImportError:
                        st.error("Función 'generate_life_zone_raster' no encontrada en analysis.py")
                    except Exception as e:
                        st.error(f"Error inesperado: {e}")

    # --- PESTAÑA 2: PUNTOS (EXISTENTE) ---
    with tab_puntos:
        df_anual = kwargs.get('df_anual_melted')
        if df_anual is None or gdf_stations is None:
            st.warning("Datos insuficientes.")
        else:
            ppt_media = df_anual.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean().reset_index()
            merged = pd.merge(ppt_media, gdf_stations[[Config.STATION_NAME_COL, Config.ALTITUDE_COL, 'latitude', 'longitude']], on=Config.STATION_NAME_COL)
            merged['Zona de Vida'] = merged.apply(lambda row: classify_holdridge_point(row[Config.PRECIPITATION_COL], row[Config.ALTITUDE_COL]), axis=1)
            
            fig_map = px.scatter_mapbox(
                merged, lat="latitude", lon="longitude", color="Zona de Vida", size=Config.PRECIPITATION_COL,
                hover_name=Config.STATION_NAME_COL, zoom=8, mapbox_style="carto-positron", title="Clasificación en Estaciones"
            )
            st.plotly_chart(fig_map, use_container_width=True)
            st.dataframe(merged[[Config.STATION_NAME_COL, 'Zona de Vida', Config.PRECIPITATION_COL, Config.ALTITUDE_COL]], use_container_width=True)
            
def display_drought_analysis_tab(df_long, gdf_stations, **kwargs):
    st.subheader("🌊 Análisis de Extremos Hidrológicos")
    st.info("Evaluación de eventos extremos: Sequías (Déficit), Inundaciones (Exceso) y Frecuencia (Períodos de Retorno).")

    # Recuperar estaciones filtradas del sidebar
    stations_filtered = kwargs.get('stations_for_analysis', [])

    if df_long is None or df_long.empty or not stations_filtered:
        st.warning("No hay datos o estaciones seleccionadas en el panel lateral.")
        return

    # 1. SELECCIÓN DE ESTACIÓN (Sincronizada + Opción Regional)
    # Creamos la lista de opciones incluyendo la Serie Regional
    options = ["Serie Regional (Promedio)"] + sorted(stations_filtered)
    
    selected_station = st.selectbox(
        "Seleccionar Estación para Análisis:", 
        options, 
        key="extremes_station_sel"
    )
    
    # 2. PREPARACIÓN DE DATOS (Corregido el error de sort_values)
    if selected_station == "Serie Regional (Promedio)":
        # Filtrar df_long solo para las estaciones seleccionadas
        df_subset = df_long[df_long[Config.STATION_NAME_COL].isin(stations_filtered)]
        # Calcular promedio regional por fecha
        df_station = df_subset.groupby(Config.DATE_COL)[Config.PRECIPITATION_COL].mean().reset_index()
        alt = 1500 # Altura promedio genérica para regional
    else:
        # Filtrar por estación específica
        df_station = df_long[df_long[Config.STATION_NAME_COL] == selected_station].copy()
        # Obtener altitud
        try:
            alt = gdf_stations[gdf_stations[Config.STATION_NAME_COL] == selected_station].iloc[0][Config.ALTITUDE_COL]
        except: 
            alt = 1500

    # Asegurar orden cronológico (CORRECCIÓN DEL ERROR)
    # En lugar de sort_values(Config.DATE_COL), usamos sort_values(by=...) para ser explícitos
    df_station = df_station.sort_values(by=Config.DATE_COL).set_index(Config.DATE_COL)
    
    # Resamplear a mensual
    ts_ppt = df_station[Config.PRECIPITATION_COL].resample('MS').sum()

    # 3. PESTAÑAS DE ANÁLISIS
    tab1, tab2, tab3 = st.tabs([
        "Índices Estandarizados (SPI/SPEI)", 
        "Frecuencia de Máximos (Gumbel)", 
        "Umbrales Percentiles"
    ])

    # --- SUB-PESTAÑA 1: SPI / SPEI ---
    with tab1:
        c1, c2 = st.columns(2)
        idx_type = c1.radio("Índice:", ["SPI (Lluvia)", "SPEI (Balance)"], horizontal=True)
        scale = c2.selectbox("Escala (Meses):", [1, 3, 6, 12, 24], index=2)
        
        try:
            series_idx = None
            if "SPI" in idx_type:
                from modules.analysis import calculate_spi
                series_idx = calculate_spi(ts_ppt, window=scale)
            else:
                from modules.analysis import calculate_spei
                # Estimar temperatura base si no hay datos reales
                t_series = pd.Series([25 - (0.006*float(alt))]*len(ts_ppt), index=ts_ppt.index)
                series_idx = calculate_spei(ts_ppt, t_series, window=scale)

            if series_idx is not None and not series_idx.dropna().empty:
                df_vis = pd.DataFrame({'Val': series_idx})
                df_vis['Color'] = np.where(df_vis['Val'] >= 0, 'blue', 'red')
                
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df_vis.index, y=df_vis['Val'], marker_color=df_vis['Color'], name=idx_type))
                fig.add_hline(y=-1.5, line_dash="dash", line_color="red", annotation_text="Sequía Severa")
                fig.add_hline(y=1.5, line_dash="dash", line_color="blue", annotation_text="Humedad Severa")
                fig.update_layout(title=f"Evolución {idx_type}-{scale} ({selected_station})", height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                last_val = df_vis['Val'].iloc[-1]
                lbl = "Normal"
                if last_val <= -1.5: lbl = "SEQUÍA"
                elif last_val >= 1.5: lbl = "HUMEDAD"
                st.metric(f"Estado último mes ({df_vis.index[-1].strftime('%Y-%m')})", lbl, f"{last_val:.2f}")
            else:
                st.warning("Datos insuficientes para calcular el índice.")
        except Exception as e:
            st.error(f"Error calculando índice: {e}")

    # --- SUB-PESTAÑA 2: FRECUENCIA (GUMBEL) ---
    with tab2:
        st.markdown("#### Análisis de Frecuencia (Máximos Anuales)")
        
        from modules.analysis import calculate_return_periods
        
        # Para regional, necesitamos un DF con estructura estándar, ya lo tenemos en df_station (reseteado)
        df_for_gumbel = df_station.reset_index()
        # Añadir columna dummy de nombre para que la función interna funcione
        df_for_gumbel[Config.STATION_NAME_COL] = selected_station
        df_for_gumbel[Config.YEAR_COL] = df_for_gumbel[Config.DATE_COL].dt.year
        
        res_df, debug_data = calculate_return_periods(df_for_gumbel, selected_station)
        
        if res_df is not None:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.dataframe(res_df.style.format({"Ppt Máxima Esperada (mm)": "{:.1f}"}), use_container_width=True)
            with c2:
                annual_max = debug_data['data']
                params = debug_data['params']
                tr_plot = np.linspace(1.01, 100, 100)
                prob_plot = 1 - (1/tr_plot)
                ppt_plot = stats.gumbel_r.ppf(prob_plot, *params)
                
                fig_freq = go.Figure()
                fig_freq.add_trace(go.Scatter(x=tr_plot, y=ppt_plot, mode='lines', name='Curva Gumbel', line=dict(color='red')))
                
                # Puntos observados
                sorted_max = np.sort(annual_max.values)
                n = len(sorted_max)
                rank = np.arange(1, n+1)
                tr_obs = (n + 0.12) / (n + 1 - rank - 0.44)
                
                fig_freq.add_trace(go.Scatter(x=tr_obs, y=sorted_max, mode='markers', name='Observados', marker=dict(color='black')))
                fig_freq.update_layout(xaxis_title="Período de Retorno (Años)", yaxis_title="Precipitación Máxima (mm)", xaxis_type="log", height=400)
                st.plotly_chart(fig_freq, use_container_width=True)
        else:
            st.warning("Datos insuficientes para Gumbel (se requieren min. 10 años completos).")

    # --- SUB-PESTAÑA 3: PERCENTILES ---
    with tab3:
        st.markdown("#### Umbrales Climatológicos")
        c_p1, c_p2 = st.columns(2)
        p_low = c_p1.slider("Percentil Bajo:", 1, 20, 10, key="pl")
        p_high = c_p2.slider("Percentil Alto:", 80, 99, 90, key="ph")

        df_station['Mes'] = df_station.index.month
        climatology = df_station.groupby('Mes')[Config.PRECIPITATION_COL].quantile([p_low/100, 0.5, p_high/100]).unstack()
        climatology.columns = ['low', 'median', 'high']
        
        months = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months, y=climatology['high'], name=f'P{p_high}', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=months, y=climatology['median'], name='Mediana', line=dict(color='green', dash='dot')))
        fig.add_trace(go.Scatter(x=months, y=climatology['low'], name=f'P{p_low}', line=dict(color='red')))
        
        fig.update_layout(title=f"Umbrales Mensuales - {selected_station}", height=450)
        st.plotly_chart(fig, use_container_width=True)
            
def display_climate_scenarios_tab(**kwargs):
    st.subheader("🌡️ Simulador de Cambio Climático")
    st.markdown("""
    Este módulo simula cómo cambiaría el **Balance Hídrico (Oferta de Agua)** si aumentara la temperatura o cambiara la lluvia.
    *Modelo utilizado: Fórmula de Turc.*
    """)
    
    # Datos
    df_anual = kwargs.get('df_anual_melted')
    gdf_stations = kwargs.get('gdf_stations')
    stations = kwargs.get('stations_for_analysis')
    
    if not stations:
        st.warning("Seleccione estaciones.")
        return

    # 1. Controles de Escenario
    c1, c2 = st.columns(2)
    delta_temp = c1.slider("Aumento de Temperatura (°C):", 0.0, 5.0, 2.0, 0.1)
    delta_ppt = c2.slider("Cambio en Precipitación (%):", -50, 50, -10, 5)
    
    # 2. Análisis
    if st.button("Simular Escenario Futuro"):
        results = []
        
        # Filtrar datos de estaciones seleccionadas
        station_info = gdf_stations[gdf_stations[Config.STATION_NAME_COL].isin(stations)]
        
        for _, row in station_info.iterrows():
            name = row[Config.STATION_NAME_COL]
            alt = row[Config.ALTITUDE_COL]
            
            # Obtener lluvia actual promedio
            ppt_actual = df_anual[df_anual[Config.STATION_NAME_COL] == name][Config.PRECIPITATION_COL].mean()
            
            if pd.notna(ppt_actual):
                # Escenario BASE (Actual)
                temp_actual = estimate_temperature(alt)
                etr_base, q_base = calculate_water_balance_turc(ppt_actual, temp_actual)
                
                # Escenario FUTURO
                temp_futura = temp_actual + delta_temp
                ppt_futura = ppt_actual * (1 + delta_ppt/100)
                etr_futura, q_futura = calculate_water_balance_turc(ppt_futura, temp_futura)
                
                # Cambio porcentual en caudal (Q)
                delta_q_perc = ((q_futura - q_base) / q_base * 100) if q_base > 0 else 0
                
                results.append({
                    "Estación": name,
                    "Q Actual (mm)": round(q_base, 1),
                    "Q Futuro (mm)": round(q_futura, 1),
                    "Impacto (%)": round(delta_q_perc, 1)
                })
        
        if results:
            res_df = pd.DataFrame(results)
            
            # Métricas Globales
            avg_impact = res_df["Impacto (%)"].mean()
            st.metric("Impacto Promedio en Oferta Hídrica", f"{avg_impact:.1f}%", delta_color="normal" if avg_impact > 0 else "inverse")
            
            # Gráfico de Impacto
            fig = px.bar(
                res_df, y="Estación", x="Impacto (%)",
                color="Impacto (%)",
                title=f"Impacto en Escorrentía (Q) con +{delta_temp}°C y {delta_ppt}% Lluvia",
                color_continuous_scale="RdYlGn",
                orientation='h'
            )
            fig.add_vline(x=0, line_color="black")
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(res_df, use_container_width=True)

def display_station_table_tab(**kwargs):
    st.subheader("📋 Tabla Detallada de Datos")
    
    # Podemos mostrar los datos mensuales o anuales
    df_monthly = kwargs.get('df_monthly_filtered')
    
    if df_monthly is not None and not df_monthly.empty:
        st.write(f"Mostrando {len(df_monthly)} registros filtrados.")
        
        # Formatear fecha para que se vea bonita
        df_show = df_monthly.copy()
        df_show['Fecha'] = df_show[Config.DATE_COL].dt.strftime('%Y-%m-%d')
        
        # Selección de columnas limpias
        cols = ['Fecha', Config.STATION_NAME_COL, Config.PRECIPITATION_COL]
        st.dataframe(df_show[cols], use_container_width=True)
        
        # Botón de descarga
        csv = df_show[cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Descargar CSV",
            csv,
            "datos_precipitacion.csv",
            "text/csv",
            key='download-csv'
        )
    else:
        st.warning("No hay datos para mostrar.")

def display_land_cover_analysis_tab(**kwargs):
    st.subheader("🌿 Análisis de Cobertura del Suelo y Escenarios")
    
    # Verificar Cuenca
    res_basin = st.session_state.get('basin_results')
    if not res_basin or not res_basin.get('ready') or 'gdf_union' not in res_basin:
        st.info("ℹ️ Primero analice una cuenca en la pestaña **'Mapas Avanzados'**.")
        return

    gdf_basin = res_basin['gdf_union']
    basin_name = res_basin.get('names', 'Cuenca')
    
    # Recuperación segura de variables
    bal = res_basin.get('bal', {})
    ppt_anual = bal.get('P', 0)
    q_actual = bal.get('Q', 0) 
    if q_actual == 0 and 'Q_mm' in bal: q_actual = bal['Q_mm']
    vol_actual = bal.get('Vol', 0)

    # Recuperar Área de la cuenca de forma segura para usarla globalmente en la función
    area_total_km2 = res_basin['morph']['area_km2']

    st.markdown(f"Cuenca: **{basin_name}** (Ppt ref: {ppt_anual:.0f} mm/año)")

    try:
        if not os.path.exists(Config.LAND_COVER_RASTER_PATH):
            st.error(f"⚠️ Raster no encontrado: {Config.LAND_COVER_RASTER_PATH}")
            return

        import rasterio
        from rasterio.mask import mask

        with rasterio.open(Config.LAND_COVER_RASTER_PATH) as src:
            if gdf_basin.crs != src.crs: gdf_basin = gdf_basin.to_crs(src.crs)
            out_image, _ = mask(src, gdf_basin.geometry, crop=True)
            data = out_image[0]

        legend = {
            1: "Zonas Urbanas", 2: "Cultivos Transitorios", 3: "Pastos", 4: "Áreas Agrícolas",
            5: "Bosques", 6: "Vegetación Herbácea", 7: "Áreas Abiertas", 8: "Aguas",
            9: "Bosque Fragmentado", 10: "Vegetación Secundaria", 11: "Zonas Degradadas", 12: "Humedales"
        }
        
        valid_pixels = data[data != src.nodata]
        if valid_pixels.size == 0:
            st.warning("Cuenca fuera del raster.")
            return

        unique, counts = np.unique(valid_pixels, return_counts=True)
        
        rows = []
        for val, count in zip(unique, counts):
            perc = (count / counts.sum()) * 100
            area = (perc / 100) * area_total_km2
            rows.append({"Cobertura": legend.get(val, f"Clase {val}"), "Área (km²)": area, "%": perc})
            
        df_cover = pd.DataFrame(rows).sort_values("%", ascending=False)

        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("#### Distribución Actual")
            st.dataframe(df_cover.style.format({"Área (km²)": "{:.2f}", "%": "{:.1f}%"}), use_container_width=True)
        with c2:
            fig = px.pie(df_cover, values='Área (km²)', names='Cobertura', hole=0.4)
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Escorrentía Actual (Q)", f"{q_actual:.0f} mm/año", f"Vol: {vol_actual:.2f} Mm³")

        st.markdown("---")

        # --- SIMULADOR ---
        st.subheader("🎛️ Simulador de Escorrentía (SCS-CN)")
        
        with st.expander("Configuración de Números de Curva (CN)", expanded=False):
            c_cn = st.columns(5)
            cn_bosque = c_cn[0].number_input("CN Bosque", 30, 100, 55)
            cn_pasto = c_cn[1].number_input("CN Pasto", 30, 100, 75)
            cn_cultivo = c_cn[2].number_input("CN Cultivo", 30, 100, 85)
            cn_urbano = c_cn[3].number_input("CN Urbano", 30, 100, 95)
            cn_suelo = c_cn[4].number_input("CN Suelo", 30, 100, 90)
        
        st.write("**Defina el Escenario Futuro (% Área):**")
        s1, s2, s3, s4, s5 = st.columns(5)
        p_bosque = s1.slider("% Bosque", 0, 100, 40)
        p_pasto = s2.slider("% Pasto", 0, 100, 30)
        p_cultivo = s3.slider("% Cultivo", 0, 100, 20)
        p_urbano = s4.slider("% Urbano", 0, 100, 5)
        p_suelo = s5.slider("% Suelo", 0, 100, 5)
        
        total_p = p_bosque + p_pasto + p_cultivo + p_urbano + p_suelo
        
        if total_p != 100:
            st.warning(f"⚠️ Suma: {total_p}%. Debe ser 100%.")
        else:
            if st.button("Estimar Escorrentía del Escenario"):
                cn_comp = ((p_bosque*cn_bosque) + (p_pasto*cn_pasto) + (p_cultivo*cn_cultivo) + (p_urbano*cn_urbano) + (p_suelo*cn_suelo)) / 100
                S = (25400 / cn_comp) - 254
                Q_escenario = ((ppt_anual - 0.2 * S)**2) / (ppt_anual + 0.8 * S) if ppt_anual > 0.2 * S else 0
                
                # CORRECCIÓN AQUÍ: Usamos area_total_km2 definida arriba
                vol_escenario = (Q_escenario * area_total_km2) / 1000
                
                delta_q = Q_escenario - q_actual
                
                st.success("Escenario Calculado")
                col_res1, col_res2, col_res3 = st.columns(3)
                col_res1.metric("CN Ponderado", f"{cn_comp:.1f}")
                col_res2.metric("Escorrentía (Q)", f"{Q_escenario:.0f} mm/año", delta=f"{delta_q:+.0f} mm/año")
                col_res3.metric("Volumen Total", f"{vol_escenario:.2f} Mm³", delta=f"{(vol_escenario - vol_actual):+.2f} Mm³")
                
                fig_sim = go.Figure(data=[
                    go.Bar(name='Actual', x=['Escorrentía'], y=[q_actual], marker_color='#1f77b4'),
                    go.Bar(name='Escenario', x=['Escorrentía'], y=[Q_escenario], marker_color='#2ca02c')
                ])
                fig_sim.update_layout(title="Comparación Q (mm/año)", height=300)
                st.plotly_chart(fig_sim, use_container_width=True)

    except Exception as e:
        st.error(f"Error procesando cobertura: {e}")

# PESTAÑA: CORRECCIÓN DE SESGO (VERSIÓN BLINDADA)
# -----------------------------------------------------------------------------
def display_bias_correction_tab(df_long, gdf_stations, gdf_filtered, **kwargs):
    """
    Módulo de validación y corrección de sesgo (Estaciones vs Satélite ERA5).
    Versión optimizada para series temporales mensuales.
    """
    st.subheader("🛰️ Validación Mensual (Estaciones vs. Satélite)")

    # --- DOCUMENTACIÓN Y AYUDA (NUEVO BLOQUE) ---
    with st.expander("ℹ️ Guía Técnica: Fuentes, Metodología e Interpretación", expanded=False):
        st.markdown("""
        ### 1. ¿Qué hace este módulo?
        Este módulo permite comparar la **precipitación observada** (medida por pluviómetros en tierra) con la **precipitación estimada** por modelos satelitales/reanálisis (ERA5-Land) para evaluar la precisión de estos últimos en la región Andina.

        ### 2. Fuentes de Datos
        * **Estaciones (Observado):** Datos hidrometeorológicos reales cargados en el sistema (IDEAM/Particulares).
        * **Satélite (Estimado):** [ERA5-Land](https://cds.climate.copernicus.eu/), un reanálisis climático global de alta resolución (~9km) producido por el ECMWF.
            * *Ventaja:* Cobertura global continua y datos desde 1950.
            * *Desventaja:* Tiende a subestimar lluvias extremas en topografía compleja (montañas) debido a su resolución espacial.

        ### 3. Metodología de Procesamiento
        1.  **Agregación Temporal:** Se transforman los datos diarios a **acumulados mensuales** exactos.
        2.  **Emparejamiento Espacial (Nearest Neighbor):** * Para cada estación en tierra, el sistema busca el **píxel (celda) más cercano** del modelo satelital utilizando un algoritmo *KD-Tree*.
            * *Radio de búsqueda:* Máximo 0.1 grados (~11 km). Si no hay datos satelitales cerca, la estación se descarta.
        3.  **Cálculo de Diferencia:** `Dif = Obs - Sat`. 
            * Valores positivos indican que la estación midió más lluvia que el satélite (Subestimación del modelo).
            * Valores negativos indican lo contrario.

        ### 4. Interpretación de Gráficos
        * **📈 Series Temporales:** Permite ver si el satélite "sigue el ritmo" de la estación (captura las temporadas de lluvias y sequías) aunque los montos no sean exactos.
        * **🗺️ Mapa:** Muestra la ubicación real de las estaciones sobre el fondo interpolado del satélite. Útil para identificar zonas donde el modelo falla sistemáticamente.
        * **🔍 Correlación:** Un $R^2$ cercano a 1 indica que el satélite es un buen predictor. Si los puntos están muy dispersos, el uso de datos satelitales debe hacerse con precaución (Bias Correction requerido).
        """)
        
    st.info("Comparación de series temporales mensuales: Lluvia Observada vs. ERA5-Land.")

    # 1. Selección de Estaciones
    target_gdf = gdf_filtered if gdf_filtered is not None and not gdf_filtered.empty else gdf_stations

    if df_long.empty or target_gdf is None or target_gdf.empty:
        st.warning("Faltan datos para realizar el análisis.")
        return

    # 2. Controles de UI
    c1, c2 = st.columns([2, 1])
    with c1:
        # Obtener rango de años disponibles EN LOS DATOS OBSERVADOS
        years = sorted(df_long[Config.YEAR_COL].unique())
        if not years:
            st.error("El dataset no contiene información de años.")
            return
            
        min_y, max_y = int(min(years)), int(max(years))
        # Slider con valores por defecto inteligentes
        default_start = max(min_y, max_y - 5)
        start_year, end_year = st.slider(
            "Período de Análisis:", 
            min_y, max_y, 
            (default_start, max_y), 
            key="bias_rng"
        )
    with c2:
        st.write("") # Espaciador para alineación vertical
        calc_btn = st.button("🚀 Calcular Series", type="primary", use_container_width=True)

    # 3. Lógica de Cálculo (Solo si se presiona el botón)
    if calc_btn:
        # Importaciones locales
        from modules.openmeteo_api import get_historical_monthly_series
        from scipy.spatial import cKDTree
        from scipy.interpolate import griddata
        import geopandas as gpd # Necesario para exportar GeoJSON

        # --- PASO 1: PROCESAR DATOS OBSERVADOS ---
        with st.spinner("1/3. Procesando datos de estaciones (Agregación Mensual)..."):
            # Filtrar datos
            mask = (df_long[Config.YEAR_COL] >= start_year) & \
                   (df_long[Config.YEAR_COL] <= end_year) & \
                   (df_long[Config.STATION_NAME_COL].isin(target_gdf[Config.STATION_NAME_COL]))
            df_subset = df_long[mask].copy()

            if df_subset.empty:
                st.error("No se encontraron datos observados en el periodo seleccionado.")
                return

            # Construir fecha robusta
            try:
                cols_data = {'year': df_subset[Config.YEAR_COL], 'day': 1}
                if hasattr(Config, 'MONTH_COL') and Config.MONTH_COL in df_subset.columns:
                    cols_data['month'] = df_subset[Config.MONTH_COL]
                elif 'MONTH' in df_subset.columns:
                    cols_data['month'] = df_subset['MONTH']
                elif 'MES' in df_subset.columns:
                    cols_data['month'] = df_subset['MES']
                else:
                    pass 

                df_subset['date'] = pd.to_datetime(cols_data)
            except Exception:
                date_col = next((col for col in df_subset.columns if 'date' in col.lower() or 'fecha' in col.lower()), None)
                if date_col:
                    df_subset['date'] = pd.to_datetime(df_subset[date_col])
                else:
                    st.error("Error crítico: No se pudo construir la fecha. Verifique columnas Año/Mes.")
                    return

            # Normalizar fecha
            df_subset['date'] = df_subset['date'].dt.to_period('M').dt.to_timestamp()

            # Agrupar: Suma total por mes y estación
            df_obs = df_subset.groupby([Config.STATION_NAME_COL, 'date'])[Config.PRECIPITATION_COL].sum().reset_index()

        # --- PASO 2: DESCARGA SATELITAL ---
        with st.spinner("2/3. Descargando series satelitales (ERA5-Land)..."):
            unique_locs = target_gdf[[Config.STATION_NAME_COL, 'latitude', 'longitude']].drop_duplicates(Config.STATION_NAME_COL)
            lats = unique_locs['latitude'].tolist()
            lons = unique_locs['longitude'].tolist()
            
            df_sat = get_historical_monthly_series(
                lats, lons, 
                f"{start_year}-01-01", 
                f"{end_year}-12-31"
            )
            
            if df_sat.empty:
                st.error("La API satelital no retornó datos.")
                return

        # --- PASO 3: EMPAREJAMIENTO ---
        with st.spinner("3/3. Cruzando información espacial..."):
            obs_coords = np.column_stack((unique_locs['latitude'], unique_locs['longitude']))
            sat_unique = df_sat[['latitude', 'longitude']].drop_duplicates()
            sat_coords = np.column_stack((sat_unique['latitude'], sat_unique['longitude']))
            
            tree = cKDTree(sat_coords)
            dists, idxs = tree.query(obs_coords)
            
            map_data = []
            for i, station_name in enumerate(unique_locs[Config.STATION_NAME_COL]):
                if dists[i] < 0.1: 
                    map_data.append({
                        Config.STATION_NAME_COL: station_name,
                        'sat_lat': sat_coords[idxs[i]][0],
                        'sat_lon': sat_coords[idxs[i]][1],
                        'dist_deg': dists[i]
                    })
            
            df_map = pd.DataFrame(map_data)
            if df_map.empty:
                st.error("No se encontraron coincidencias espaciales.")
                return

            # MERGE 1: Obs + Map
            df_merged = pd.merge(df_obs, df_map, on=Config.STATION_NAME_COL)
            # MERGE 1b: Agregar coordenadas REALES
            df_merged = pd.merge(df_merged, unique_locs, on=Config.STATION_NAME_COL, how='left')

            # MERGE 2: + Satélite
            df_final = pd.merge(
                df_merged,
                df_sat.rename(columns={'latitude': 'sat_lat', 'longitude': 'sat_lon'}),
                on=['date', 'sat_lat', 'sat_lon'],
                how='inner'
            )
            
            df_final['diff_mm'] = df_final[Config.PRECIPITATION_COL] - df_final['ppt_sat']
            
            st.success("✅ Análisis completado exitosamente.")

            # --- VISUALIZACIÓN ---
            tab_series, tab_mapa, tab_datos = st.tabs(["📈 Series Temporales", "🗺️ Mapa Promedio", "📋 Datos & Descargas"])
            
            # TAB 1: SERIES
            with tab_series:
                c_sel, _ = st.columns([1, 2])
                with c_sel:
                    estaciones_disp = sorted(df_final[Config.STATION_NAME_COL].unique())
                    sel_st = st.selectbox("Seleccionar Visualización:", ["Promedio Regional"] + estaciones_disp)
                
                if sel_st == "Promedio Regional":
                    plot_df = df_final.groupby('date')[[Config.PRECIPITATION_COL, 'ppt_sat']].mean().reset_index()
                    title_plot = "Promedio Regional (Todas las Estaciones)"
                else:
                    plot_df = df_final[df_final[Config.STATION_NAME_COL] == sel_st]
                    title_plot = f"Estación: {sel_st}"
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=plot_df['date'], y=plot_df[Config.PRECIPITATION_COL], name='Observado (Real)', mode='lines+markers'))
                fig.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['ppt_sat'], name='Satélite (ERA5)', mode='lines+markers', line=dict(dash='dash')))
                fig.update_layout(title=title_plot, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

            # TAB 2: MAPA
            with tab_mapa:
                st.markdown("**Comparativa Espacial (Promedio del Periodo)**")
                # Agregamos por ubicación REAL y SATELITAL
                map_agg = df_final.groupby([Config.STATION_NAME_COL, 'latitude', 'longitude', 'sat_lat', 'sat_lon'])[['ppt_sat', Config.PRECIPITATION_COL]].mean().reset_index()
                
                # -- GENERACIÓN DE TEXTO PARA POPUP (HOVER) --
                map_agg['hover_text'] = map_agg.apply(
                    lambda row: f"<b>{row[Config.STATION_NAME_COL]}</b><br>💧 Obs: {row[Config.PRECIPITATION_COL]:.1f} mm<br>🛰️ Sat: {row['ppt_sat']:.1f} mm", 
                    axis=1
                )

                try:
                    # Interpolación Satélite (Fondo)
                    grid_x, grid_y = np.mgrid[
                        map_agg['sat_lon'].min():map_agg['sat_lon'].max():100j,
                        map_agg['sat_lat'].min():map_agg['sat_lat'].max():100j
                    ]
                    grid_z = griddata(
                        (map_agg['sat_lon'], map_agg['sat_lat']), 
                        map_agg['ppt_sat'], 
                        (grid_x, grid_y), 
                        method='cubic'
                    )
                    
                    fig_map = go.Figure()
                    fig_map.add_trace(go.Contour(
                        z=grid_z.T, x=grid_x[:,0], y=grid_y[0,:], 
                        colorscale='Blues', opacity=0.6, showscale=False, name='Satélite (Fondo)'
                    ))
                    # Puntos Reales con HOVER PERSONALIZADO
                    fig_map.add_trace(go.Scatter(
                        x=map_agg['longitude'], y=map_agg['latitude'], 
                        mode='markers', 
                        marker=dict(
                            size=10, 
                            color=map_agg[Config.PRECIPITATION_COL], 
                            colorscale='RdBu', 
                            showscale=True, 
                            line=dict(width=1, color='black')
                        ),
                        text=map_agg['hover_text'], # Usamos la columna formateada
                        hoverinfo='text',           # Forzamos a mostrar solo el texto
                        name='Estaciones'
                    ))
                    fig_map.update_layout(title="Fondo: Satélite | Puntos: Estaciones (Posición Real)", height=500)
                    st.plotly_chart(fig_map, use_container_width=True)
                except Exception as e:
                    st.warning(f"No se pudo interpolar: {e}")
                    st.map(map_agg)

            # TAB 3: DATOS Y GEOJSON
            with tab_datos:
                st.markdown("### Datos Tabulares")
                st.dataframe(
                    df_final[[Config.STATION_NAME_COL, 'date', Config.PRECIPITATION_COL, 'ppt_sat', 'diff_mm']]
                    .sort_values(by=[Config.STATION_NAME_COL, 'date']),
                    use_container_width=True
                )
                
                c_csv, c_geo = st.columns(2)
                
                # 1. Descarga CSV
                with c_csv:
                    csv = df_final.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Descargar Series (CSV)", 
                        csv, 
                        "validacion_mensual_satelite.csv", 
                        "text/csv"
                    )
                
                # 2. Descarga GEOJSON (Promedios Espaciales)
                with c_geo:
                    # Convertir el DataFrame agregado (map_agg) a GeoDataFrame
                    # map_agg ya tiene el promedio por estación calculado en el bloque anterior (Tab 2)
                    gdf_export = gpd.GeoDataFrame(
                        map_agg, 
                        geometry=gpd.points_from_xy(map_agg.longitude, map_agg.latitude),
                        crs="EPSG:4326"
                    )
                    geojson_data = gdf_export.to_json()
                    st.download_button(
                        "🌍 Descargar Mapa Promedio (GeoJSON)",
                        data=geojson_data,
                        file_name="estaciones_promedio_satelite.geojson",
                        mime="application/geo+json"
                    )















