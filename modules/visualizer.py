import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
import requests
import os
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
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
from modules.analysis import generate_life_zone_raster

# -----------------------------------------------------------------------------
# 1. FUNCIONES AUXILIARES
# -----------------------------------------------------------------------------

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
@st.cache_data(ttl=86400) # Cache de 24 horas (el IRI actualiza mensualmente)
def get_iri_enso_forecast():
    """
    Descarga el pronóstico oficial de la Pluma ENSO del IRI en formato JSON.
    Retorna un DataFrame listo para graficar.
    """
    json_url = "https://iri.columbia.edu/our-expertise/climate/forecasts/enso/graphics/ensoplume_full.json"
    try:
        # Intentar descargar con timeout para no bloquear la app
        response = requests.get(json_url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # Estructura del JSON: {'stat_fcst': {'model_name': [[tiempo, valor], ...]}, 'dyn_fcst': ...}
        models_data = []
        
        # Procesar Modelos Dinámicos
        if 'dyn_fcst' in data:
            for model_name, values in data['dyn_fcst'].items():
                for point in values:
                    # El tiempo viene en formato decimal (ej. 2024.5)
                    year = int(point[0])
                    month = int((point[0] - year) * 12) + 1
                    date_obj = pd.to_datetime(f"{year}-{month}-01")
                    models_data.append({'Fecha': date_obj, 'Valor': point[1], 'Modelo': model_name, 'Tipo': 'Dinámico'})

        # Procesar Modelos Estadísticos
        if 'stat_fcst' in data:
            for model_name, values in data['stat_fcst'].items():
                for point in values:
                    year = int(point[0])
                    month = int((point[0] - year) * 12) + 1
                    date_obj = pd.to_datetime(f"{year}-{month}-01")
                    models_data.append({'Fecha': date_obj, 'Valor': point[1], 'Modelo': model_name, 'Tipo': 'Estadístico'})
        
        # Observaciones (Historia reciente)
        if 'obs' in data:
            for point in data['obs']:
                year = int(point[0])
                month = int((point[0] - year) * 12) + 1
                date_obj = pd.to_datetime(f"{year}-{month}-01")
                models_data.append({'Fecha': date_obj, 'Valor': point[1], 'Modelo': 'OBSERVADO', 'Tipo': 'Observado'})

        return pd.DataFrame(models_data)

    except Exception as e:
        st.warning(f"No se pudo conectar con el servidor del IRI: {e}")
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

            if gdf_filtered is not None:
                marker_cluster = MarkerCluster().add_to(m)
                for _, r in gdf_filtered.dropna(subset=['latitude']).iterrows():
                    folium.Marker([r.latitude, r.longitude], tooltip=r[Config.STATION_NAME_COL], icon=folium.Icon(color="green", icon="cloud")).add_to(marker_cluster)
            
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
    
    # Definición de Pestañas (Optimizadas: 5 pestañas potentes)
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
        st.caption("Evolución de la lluvia total acumulada por año.")
        
        fig = px.line(
            df_anual_melted, x=Config.YEAR_COL, y=Config.PRECIPITATION_COL, 
            color=Config.STATION_NAME_COL, markers=True, 
            labels={Config.PRECIPITATION_COL: "Lluvia (mm)", Config.YEAR_COL: "Año"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
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
            
        fig_bar = px.bar(
            avg_ppt, x=Config.STATION_NAME_COL, y=col_val, color=col_val,
            color_continuous_scale=px.colors.sequential.Blues, text_auto='.0f'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.download_button(
            "📥 Descargar Ranking (CSV)",
            avg_ppt.to_csv(index=False).encode('utf-8'),
            "ranking_multianual.csv", "text/csv"
        )

    # -------------------------------------------------------------------------
    # 3. SERIE MENSUAL (Con Comparación Regional Integrada)
    # -------------------------------------------------------------------------
    with tabs[2]:
        st.markdown("##### Serie Histórica Mensual")
        
        col_opts, col_chart = st.columns([1, 4])
        with col_opts:
            show_regional = st.checkbox("Ver Promedio Regional", value=False, help="Superpone la línea promedio de todas las estaciones.")
            show_markers = st.checkbox("Mostrar Puntos", value=False)
            
        with col_chart:
            fig = px.line(
                df_monthly_filtered, x=Config.DATE_COL, y=Config.PRECIPITATION_COL, 
                color=Config.STATION_NAME_COL, markers=show_markers,
                title="Precipitación Mensual"
            )
            
            # Lógica Regional Integrada
            if show_regional:
                reg_mean = df_monthly_filtered.groupby(Config.DATE_COL)[Config.PRECIPITATION_COL].mean().reset_index()
                fig.add_trace(go.Scatter(
                    x=reg_mean[Config.DATE_COL], y=reg_mean[Config.PRECIPITATION_COL],
                    mode='lines', name='PROMEDIO REGIONAL',
                    line=dict(color='black', width=3, dash='dash')
                ))
            
            st.plotly_chart(fig, use_container_width=True)
            
        st.download_button(
            "📥 Descargar Datos Mensuales (CSV)",
            df_monthly_filtered.to_csv(index=False).encode('utf-8'),
            "serie_mensual.csv", "text/csv"
        )

    # -------------------------------------------------------------------------
    # 4. CICLO ANUAL (Comparación Rápida)
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
        
        st.download_button(
            "📥 Descargar Ciclo Anual (CSV)",
            ciclo.to_csv(index=False).encode('utf-8'),
            "ciclo_anual.csv", "text/csv"
        )

    # -------------------------------------------------------------------------
    # 5. DISTRIBUCIÓN (La versión Potenciada)
    # -------------------------------------------------------------------------
    with tabs[4]:
        st.markdown("##### Análisis Estadístico de Distribución")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            data_src = st.radio("Datos:", ["Anual (Totales)", "Mensual (Detalle)"], horizontal=True)
        with c2:
            chart_typ = st.radio("Gráfico:", ["Violín (Densidad)", "Histograma", "Probabilidad (ECDF)"], horizontal=True)
        with c3:
            sort_ord = st.selectbox("Orden:", ["Alfabético", "Mayor a Menor (Mediana)"])

        df_plot = df_anual_melted if "Anual" in data_src else df_monthly_filtered
        
        # Ordenamiento
        cat_orders = {}
        if sort_ord != "Alfabético":
            medians = df_plot.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].median()
            order_list = medians.sort_values(ascending=False).index.tolist()
            cat_orders = {Config.STATION_NAME_COL: order_list}

        if "Violín" in chart_typ:
            fig = px.violin(df_plot, x=Config.STATION_NAME_COL, y=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL, box=True, points="all", category_orders=cat_orders)
            fig.update_layout(showlegend=False)
        elif "Histograma" in chart_typ:
            fig = px.histogram(df_plot, x=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL, marginal="box", barmode="overlay", opacity=0.7, category_orders=cat_orders)
        else:
            # ECDF (Acumulada) integrada aquí
            fig = px.ecdf(df_plot, x=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL, title="Probabilidad Acumulada")

        fig.update_layout(height=600, title=f"Distribución {data_src} - {chart_typ}")
        st.plotly_chart(fig, use_container_width=True)
        
        st.download_button(
            f"📥 Descargar Datos {data_src} (CSV)",
            df_plot.to_csv(index=False).encode('utf-8'),
            f"distribucion_{data_src.split()[0].lower()}.csv", "text/csv"
        )
        
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

def display_advanced_maps_tab(df_long, gdf_stations, gdf_subcuencas, gdf_filtered, **kwargs):
    st.subheader("🌍 Superficies de Interpolación y Morfometría")
    
    if df_long.empty or gdf_stations.empty:
        st.warning("Faltan datos base.")
        return

    mode = st.radio("Modo de Análisis:", ["Regional (Comparativo)", "Por Cuenca Específica"], horizontal=True)
    
    # --- FUNCIÓN INTERNA DE INTERPOLACIÓN ---
    def run_interpolation(df_data, method, grid_bounds, grid_res=100):
        from scipy.interpolate import griddata, Rbf
        minx, maxx, miny, maxy = grid_bounds
        grid_x, grid_y = np.mgrid[minx:maxx:complex(grid_res), miny:maxy:complex(grid_res)]
        points = df_data[['longitude', 'latitude']].values
        values = df_data[Config.PRECIPITATION_COL].values
        try:
            if "Spline" in method:
                grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
            elif "Kriging" in method:
                rbf = Rbf(points[:,0], points[:,1], values, function='thin_plate')
                grid_z = rbf(grid_x, grid_y)
            else:
                grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
            return grid_x, grid_y, grid_z
        except Exception: return None, None, None

    # --- MODO 1: REGIONAL ---
    if mode == "Regional (Comparativo)":
        c1, c2 = st.columns(2)
        with c1:
            min_y, max_y = int(df_long[Config.YEAR_COL].min()), int(df_long[Config.YEAR_COL].max())
            range1 = st.slider("Período 1:", min_y, max_y, (1980, 1990), key="p1")
            method1 = st.selectbox("Método 1:", ["IDW (Lineal)", "Spline (Cúbico)", "Kriging (Simulado)"], key="m1")
        with c2:
            range2 = st.slider("Período 2:", min_y, max_y, (1991, 2000), key="p2")
            method2 = st.selectbox("Método 2:", ["IDW (Lineal)", "Spline (Cúbico)", "Kriging (Simulado)"], key="m2")
            
        if st.button("Generar Comparación"):
            def plot_map(rng, meth, col):
                mask = (df_long[Config.YEAR_COL] >= rng[0]) & (df_long[Config.YEAR_COL] <= rng[1])
                df_ann = df_long[mask].groupby([Config.STATION_NAME_COL, Config.YEAR_COL])[Config.PRECIPITATION_COL].sum().reset_index()
                df_avg = df_ann.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean().reset_index()
                df_map = pd.merge(df_avg, gdf_stations, on=Config.STATION_NAME_COL).dropna(subset=['latitude', 'longitude'])
                
                if len(df_map) < 3:
                    col.warning("Datos insuficientes.")
                    return
                pad = 0.05
                bounds = [df_map.longitude.min()-pad, df_map.longitude.max()+pad, df_map.latitude.min()-pad, df_map.latitude.max()+pad]
                gx, gy, gz = run_interpolation(df_map, meth, bounds)
                if gz is not None:
                    fig = go.Figure(data=go.Contour(z=gz.T, x=gx[:,0], y=gy[0,:], colorscale='Viridis', colorbar=dict(title='mm/año')))
                    fig.add_trace(go.Scatter(x=df_map.longitude, y=df_map.latitude, mode='markers', marker=dict(color='red', size=5)))
                    fig.update_layout(title=f"{meth} ({rng[0]}-{rng[1]})", height=400, margin=dict(l=0,r=0,t=40,b=0))
                    col.plotly_chart(fig, use_container_width=True)
            plot_map(range1, method1, c1)
            plot_map(range2, method2, c2)

    # --- MODO 2: POR CUENCA ---
    else:
        if gdf_subcuencas.empty: st.warning("No hay capa de subcuencas."); return
        
        all_cuencas = sorted(gdf_subcuencas['nombre'].unique())
        sel_cuencas = st.multiselect("Seleccione Subcuencas:", all_cuencas)
        
        if sel_cuencas:
            c_time, c_meth = st.columns(2)
            with c_time:
                min_y, max_y = int(df_long[Config.YEAR_COL].min()), int(df_long[Config.YEAR_COL].max())
                rng = st.slider("Período de Análisis:", min_y, max_y, (min_y, max_y), key="rng_c")
            with c_meth:
                meth = st.selectbox("Método de Interpolación:", ["IDW (Lineal)", "Spline (Cúbico)", "Kriging (Simulado)"], key="meth_c")

            if st.button("Analizar Cuenca (Radio 50km)"):
                with st.spinner("Procesando datos hidrológicos..."):
                    # A. Geometría
                    subset = gdf_subcuencas[gdf_subcuencas['nombre'].isin(sel_cuencas)]
                    gdf_union = gpd.GeoDataFrame({'geometry': [subset.unary_union]}, crs=gdf_subcuencas.crs)
                    
                    # Buffer 50km (aprox 0.45 grados)
                    buffer_geom = gdf_union.geometry.buffer(0.45).unary_union 
                    stations_in = gdf_stations[gdf_stations.geometry.intersects(buffer_geom)]
                    
                    if not stations_in.empty:
                        target_ids = stations_in[Config.STATION_NAME_COL].unique()
                        mask = (df_long[Config.STATION_NAME_COL].isin(target_ids)) & \
                               (df_long[Config.YEAR_COL] >= rng[0]) & (df_long[Config.YEAR_COL] <= rng[1])
                        
                        df_subset = df_long[mask].copy()
                        # Promedio Anual Real (mm/año)
                        df_ann = df_subset.groupby([Config.STATION_NAME_COL, Config.YEAR_COL])[Config.PRECIPITATION_COL].sum().reset_index()
                        df_points = df_ann.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean().reset_index()
                        df_map_data = pd.merge(df_points, gdf_stations, on=Config.STATION_NAME_COL).dropna(subset=['latitude', 'longitude'])

                        if len(df_map_data) >= 3:
                            bounds = buffer_geom.bounds
                            gx, gy, gz = run_interpolation(df_map_data, meth, [bounds[0], bounds[2], bounds[1], bounds[3]])
                            ppt_media = np.nanmean(gz) if gz is not None else df_map_data[Config.PRECIPITATION_COL].mean()
                            
                            morph = calculate_morphometry(gdf_union)
                            bal = calculate_hydrological_balance(ppt_media, morph['alt_prom_m'], gdf_union)
                            
                            # --- CÁLCULO CURVA DURACIÓN (FDC) ---
                            # Serie temporal MENSUAL PROMEDIO de la cuenca (para ver variabilidad en el tiempo)
                            basin_ts = df_subset.groupby(Config.DATE_COL)[Config.PRECIPITATION_COL].mean()
                            
                            # CORRECCIÓN KEYERROR 'Q_mm': Usamos .get() para buscar 'Q' o 'Q_mm'
                            q_val = bal.get('Q_mm', bal.get('Q', 0))
                            p_val = bal.get('P', 0)
                            
                            # Coeficiente Escorrentía (C)
                            runoff_c = q_val / p_val if p_val > 0 else 0.4 # Default 0.4 si falla
                            
                            try:
                                # Importar aquí por seguridad si no está arriba
                                from modules.analysis import calculate_duration_curve 
                                fdc_df = calculate_duration_curve(basin_ts, runoff_c, morph['area_km2'])
                            except:
                                fdc_df = pd.DataFrame()

                            st.session_state['basin_results'] = {
                                'ready': True, 'gx': gx, 'gy': gy, 'gz': gz, 'df': df_map_data,
                                'morph': morph, 'bal': bal, 'geom': gdf_union, 'buffer': buffer_geom,
                                'names': ", ".join(sel_cuencas), 'periodo': f"{rng[0]}-{rng[1]}", 'method': meth,
                                'bounds': [bounds[0], bounds[2], bounds[1], bounds[3]],
                                'fdc_data': fdc_df
                            }
                        else: st.error("Insuficientes estaciones (<3) en radio de 50km.")
                    else: st.error("No hay estaciones cercanas.")

            # --- RENDERIZADO ---
            res = st.session_state.get('basin_results')
            if res and res.get('ready'):
                if 'bounds' not in res: st.warning("Datos antiguos. Recalcule."); return

                st.success(f"Análisis **{res.get('periodo')}** completado.")

                # 1. Mapa Interpolado
                fig = go.Figure(data=go.Contour(
                    z=res['gz'].T, x=res['gx'][:,0], y=res['gy'][0,:],
                    colorscale='Viridis', colorbar=dict(title='mm/año'), contours=dict(coloring='heatmap', showlabels=True)
                ))
                fig.add_trace(go.Scatter(x=res['df'].longitude, y=res['df'].latitude, mode='markers', marker=dict(color='red', size=5), name="Estaciones"))
                try:
                    poly = res['geom'].geometry.iloc[0]
                    if poly.geom_type == 'Polygon':
                        x, y = poly.exterior.xy
                        fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines', line=dict(color='white', width=2), name='Cuenca'))
                except: pass
                
                fig.update_layout(height=600, title="Superficie de Lluvia (mm/año)", xaxis_range=[res['bounds'][0], res['bounds'][1]], yaxis_range=[res['bounds'][2], res['bounds'][3]])
                st.plotly_chart(fig, use_container_width=True)

                # 2. Datos
                st.markdown("---")
                b = res['bal']
                st.markdown(f"#### 💧 Balance Hídrico")
                
                # CORRECCIÓN DE CLAVES PARA VISUALIZACIÓN
                vol_val = b.get('Vol', b.get('Q_m3_año', 0))
                q_val = b.get('Q_mm', b.get('Q', 0))
                
                q_ls = (vol_val * 1_000_000_000) / 31536000 if vol_val > 0 else 0
                
                cols = st.columns(5)
                cols[0].metric("Ppt Media", f"{b['P']:.0f} mm/año")
                cols[1].metric("Altitud", f"{b['Alt']:.0f} m")
                cols[2].metric("ET", f"{b['ET']:.0f} mm/año")
                cols[3].metric("Q (mm)", f"{max(0, q_val):.0f} mm/año")
                cols[4].metric("Q (L/s)", f"{q_ls:.0f} L/s")
                st.info(f"**Volumen:** {vol_val:.2f} millones de m³.")

                # 3. CURVA DE DURACIÓN DE CAUDALES (FDC) - NUEVO
                if 'fdc_data' in res and not res['fdc_data'].empty:
                    st.markdown("---")
                    st.subheader("📉 Curva de Duración de Caudales (FDC)")
                    st.info("Muestra el porcentaje del tiempo que el caudal iguala o excede un valor específico.")
                    
                    fdc = res['fdc_data']
                    fig_fdc = go.Figure()
                    fig_fdc.add_trace(go.Scatter(
                        x=fdc["Probabilidad Excedencia (%)"], 
                        y=fdc["Caudal (m³/s)"],
                        mode='lines', fill='tozeroy', line=dict(color='#1f77b4', width=3)
                    ))
                    # Puntos Clave Q95 (Ecológico) y Q50 (Medio)
                    try:
                        q95 = fdc.iloc[int(len(fdc)*0.95)]["Caudal (m³/s)"]
                        q50 = fdc.iloc[int(len(fdc)*0.50)]["Caudal (m³/s)"]
                        fig_fdc.add_vline(x=95, line_dash="dash", annotation_text=f"Q95: {q95:.2f}")
                        fig_fdc.add_vline(x=50, line_dash="dash", annotation_text=f"Q50: {q50:.2f}")
                    except: pass
                    
                    fig_fdc.update_layout(
                        xaxis_title="Probabilidad de Excedencia (%)", 
                        yaxis_title="Caudal (m³/s)", 
                        height=400,
                        title="Disponibilidad Hídrica en el Tiempo"
                    )
                    st.plotly_chart(fig_fdc, use_container_width=True)

                # 4. Morfometría
                st.markdown("#### 📐 Morfometría")
                m = res['morph']
                cm = st.columns(6)
                cm[0].metric("Área", f"{m['area_km2']:.2f} km²")
                cm[1].metric("Perímetro", f"{m['perimetro_km']:.2f} km")
                cm[2].metric("Índice Forma", f"{m['indice_forma']:.2f}")
                cm[3].metric("Alt Máx", f"{m['alt_max_m']:.0f} m")
                cm[4].metric("Alt Mín", f"{m['alt_min_m']:.0f} m")
                cm[5].metric("Pendiente", f"{m['pendiente_prom']:.1f} %")
                
                # 5. Hipsometría
                hypso = calculate_hypsometric_curve(res['geom'])
                if hypso:
                    st.markdown("---")
                    st.subheader("⛰️ Curva Hipsométrica")
                    c_h1, c_h2 = st.columns([3, 1])
                    with c_h1:
                        fig_h = go.Figure()
                        fig_h.add_trace(go.Scatter(x=hypso['area_percent'], y=hypso['elevations'], fill='tozeroy', name='Perfil'))
                        x_tr = np.linspace(0, 100, 50)
                        fig_h.add_trace(go.Scatter(x=x_tr, y=hypso['poly_model'](x_tr), line=dict(dash='dash'), name='Ajuste'))
                        fig_h.update_layout(title="Curva Hipsométrica", xaxis_title="% Área", yaxis_title="Elevación (m)", height=350)
                        st.plotly_chart(fig_h, use_container_width=True)
                    with c_h2:
                        st.latex(hypso['equation'])

                # 6. Mapa Contexto
                st.markdown("---")
                st.subheader("📍 Contexto Espacial")
                minx, maxx, miny, maxy = res['bounds']
                map_c = folium.Map(location=[(miny+maxy)/2, (minx+maxx)/2], zoom_start=9, tiles="CartoDB positron")
                folium.GeoJson(res['geom'], name="Cuenca", style_function=lambda x: {'color':'blue', 'weight':3}).add_to(map_c)
                folium.GeoJson(res['buffer'], name="Radio 50km", style_function=lambda x: {'color':'gray', 'dashArray':'5,5', 'fill':False}).add_to(map_c)
                for _, r in res['df'].iterrows():
                    folium.CircleMarker([r.latitude, r.longitude], radius=3, color='red', fill=True).add_to(map_c)
                st_folium(map_c, height=500, width="100%")
        else:
            st.info("Seleccione cuencas.")
            
def display_climate_forecast_tab(**kwargs):
    st.subheader("🔮 Pronóstico Climático (Índices Globales)")
    
    df_enso = kwargs.get('df_enso')
    
    # Crear pestañas para separar Historia (tu código) de Pronóstico (nuevo)
    tab_hist, tab_iri = st.tabs(["📜 Evolución Histórica", "🌎 Pronóstico Oficial (IRI/CPC)"])

    # ---------------------------------------------------------
    # PESTAÑA 1: HISTÓRICO (Tu lógica original conservada)
    # ---------------------------------------------------------
    with tab_hist:
        if df_enso is None or df_enso.empty:
            st.warning("No hay datos de índices climáticos cargados.")
        else:
            st.info("Análisis de la evolución histórica de los índices que afectan la región.")
            index_col = st.selectbox("Seleccione Índice:", [Config.ENSO_ONI_COL, Config.SOI_COL, Config.IOD_COL], index=0)
            
            if index_col in df_enso.columns:
                data = df_enso.dropna(subset=[index_col]).sort_values(Config.DATE_COL)
                fig = px.line(data, x=Config.DATE_COL, y=index_col, title=f"Evolución Histórica: {index_col}")
                
                # Líneas de referencia solo para ONI
                if index_col == Config.ENSO_ONI_COL:
                    fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="El Niño")
                    fig.add_hline(y=-0.5, line_dash="dash", line_color="blue", annotation_text="La Niña")
                    
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Columna {index_col} no encontrada.")

    # ---------------------------------------------------------
    # PESTAÑA 2: PRONÓSTICO IRI (La nueva funcionalidad)
    # ---------------------------------------------------------
    with tab_iri:
        st.markdown("#### Pluma de Predicción ENSO (International Research Institute)")
        st.caption("Pronóstico consolidado de modelos globales sobre la temperatura del Pacífico (Niño 3.4).")
        
        with st.spinner("Conectando con servidores del IRI..."):
            df_iri = get_iri_enso_forecast()
            
        if not df_iri.empty:
            # Filtrar últimos meses y futuro
            last_obs = df_iri[df_iri['Tipo'] == 'Observado']['Fecha'].max()
            df_plot = df_iri[df_iri['Fecha'] >= (last_obs - pd.DateOffset(months=4))]
            
            fig_plume = go.Figure()
            
            # 1. Modelos individuales (Espagueti)
            for model in df_plot['Modelo'].unique():
                sub = df_plot[df_plot['Modelo'] == model]
                tipo = sub.iloc[0]['Tipo']
                
                # Estilo según tipo
                if tipo == 'Observado':
                    color, width, opac, name = 'black', 4, 1, "Observado"
                elif 'AVG' in model.upper(): # Promedios
                    color, width, opac, name = 'blue', 3, 1, model
                elif tipo == 'Dinámico':
                    color, width, opac, name = 'orange', 1, 0.3, "Dinámicos"
                else:
                    color, width, opac, name = 'green', 1, 0.3, "Estadísticos"
                
                show_leg = True if width > 1 else False # Solo leyenda para los importantes
                
                fig_plume.add_trace(go.Scatter(
                    x=sub['Fecha'], y=sub['Valor'], mode='lines',
                    line=dict(color=color, width=width), opacity=opac,
                    name=name, showlegend=show_leg, hoverinfo='text+y', text=model
                ))
            
            # 2. Umbrales
            fig_plume.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="El Niño")
            fig_plume.add_hline(y=-0.5, line_dash="dash", line_color="blue", annotation_text="La Niña")
            
            fig_plume.update_layout(
                title=f"Pronóstico Multimodelo ENSO - Actualizado a {last_obs.strftime('%Y-%m')}",
                yaxis_title="Anomalía SST (°C)", height=550, hovermode="x unified"
            )
            st.plotly_chart(fig_plume, use_container_width=True)
            
            with st.expander("Ver Tabla de Datos"):
                st.dataframe(df_plot.pivot(index='Fecha', columns='Modelo', values='Valor'), use_container_width=True)
        else:
            st.error("No se pudo descargar el pronóstico. Mostrando imagen de respaldo.")
            st.image("https://iri.columbia.edu/climate/ENSO/current/info/figure3.png", width=700)

def display_trends_and_forecast_tab(**kwargs):
    st.subheader("📉 Tendencias, Pronósticos y Riesgo")
    
    # Recuperar datos
    df_monthly = kwargs.get('df_monthly_filtered')
    df_anual = kwargs.get('df_anual_melted')
    stations = kwargs.get('stations_for_analysis')
    gdf_stations = kwargs.get('gdf_stations')

    if not stations or df_monthly.empty:
        st.warning("Seleccione estaciones.")
        return

    # Pestañas Principales
    tabs = st.tabs([
        "Análisis de Tendencias", 
        "Descomposición", 
        "Autocorrelación", 
        "SARIMA", 
        "Prophet", 
        "Comparación Modelos",
        "Mapa de Riesgo"
    ])
    
    selected_station = st.selectbox("Estación:", stations, key="trend_st")
    
    # --- CORRECCIÓN CRÍTICA: Preparación de la Serie de Tiempo ---
    station_data = df_monthly[df_monthly[Config.STATION_NAME_COL] == selected_station].sort_values(Config.DATE_COL).set_index(Config.DATE_COL)
    
    # 1. Asignar frecuencia mensual explícita
    # 2. Interpolar valores internos (huecos en el medio)
    # 3. .dropna() para eliminar huecos al inicio/fin que la interpolación no puede llenar
    ts = station_data[Config.PRECIPITATION_COL].asfreq('MS').interpolate(method='linear').dropna()

    # Validar longitud mínima
    if len(ts) < 24:
        st.error(f"Datos insuficientes para la estación {selected_station} (se requieren al menos 24 meses continuos).")
        return

    # 1. TENDENCIAS (Mann-Kendall)
    with tabs[0]:
        try:
            res = mk.original_test(ts)
            c1, c2, c3 = st.columns(3)
            c1.metric("Tendencia", res.trend, delta=f"Slope: {res.slope:.3f}")
            c2.metric("P-Value", f"{res.p:.4f}")
            c3.metric("Tau Kendall", f"{res.Tau:.3f}")
            
            fig = px.scatter(ts.reset_index(), x=Config.DATE_COL, y=Config.PRECIPITATION_COL, trendline="ols", title="Tendencia Lineal")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"No se pudo calcular Mann-Kendall: {e}")

    # 2. DESCOMPOSICIÓN
    with tabs[1]:
        try:
            # Ahora ts no tiene NaNs, por lo que esto funcionará
            decomp = seasonal_decompose(ts, model='additive', period=12)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts.index, y=decomp.trend, name='Tendencia'))
            fig.add_trace(go.Scatter(x=ts.index, y=decomp.seasonal, name='Estacionalidad'))
            fig.add_trace(go.Scatter(x=ts.index, y=decomp.resid, name='Residuo', mode='markers'))
            fig.update_layout(title="Descomposición de Serie Temporal")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error en descomposición: {e}")

    # 3. AUTOCORRELACIÓN (ACF/PACF)
    with tabs[2]:
        try:
            from statsmodels.tsa.stattools import acf, pacf
            # Limitar lags a 50% de la serie para evitar errores
            nlags = min(40, len(ts)//2 - 1)
            lag_acf = acf(ts, nlags=nlags)
            lag_pacf = pacf(ts, nlags=nlags)
            
            c1, c2 = st.columns(2)
            fig_acf = px.bar(x=range(len(lag_acf)), y=lag_acf, title="Autocorrelación (ACF)")
            c1.plotly_chart(fig_acf, use_container_width=True)
            
            fig_pacf = px.bar(x=range(len(lag_pacf)), y=lag_pacf, title="Autocorrelación Parcial (PACF)")
            c2.plotly_chart(fig_pacf, use_container_width=True)
        except Exception as e:
            st.error(f"Error en ACF/PACF: {e}")

    # 4. SARIMA
    with tabs[3]:
        st.markdown("#### Pronóstico SARIMA")
        horizon = st.slider("Horizonte:", 6, 36, 12, key="h_sarima")
        if st.button("Calcular SARIMA"):
            try:
                # Simulación visual rápida (o usar statsmodels real si está instalado)
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                
                with st.spinner("Ajustando modelo SARIMA..."):
                    # Modelo genérico robusto (1,1,1)(1,1,1,12)
                    model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
                    model_fit = model.fit(disp=False)
                    
                    # Predicción
                    forecast_res = model_fit.get_forecast(steps=horizon)
                    forecast_vals = forecast_res.predicted_mean
                    conf_int = forecast_res.conf_int()
                    
                    fig = go.Figure()
                    # Mostrar últimos 5 años de historia para claridad
                    tail_ts = ts.tail(60)
                    fig.add_trace(go.Scatter(x=tail_ts.index, y=tail_ts, name="Histórico"))
                    fig.add_trace(go.Scatter(x=forecast_vals.index, y=forecast_vals, name="Pronóstico SARIMA", line=dict(color='red', dash='dash')))
                    
                    # Intervalo de confianza
                    fig.add_trace(go.Scatter(
                        x=pd.concat([conf_int.index, conf_int.index[::-1]]),
                        y=pd.concat([conf_int.iloc[:, 0], conf_int.iloc[:, 1][::-1]]),
                        fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(color='rgba(255,255,255,0)'),
                        name='Intervalo Confianza'
                    ))

                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state['sarima_res'] = forecast_vals
            except Exception as e:
                st.error(f"Error en cálculo SARIMA: {e}")

    # 5. PROPHET
    with tabs[4]:
        st.markdown("#### Pronóstico Prophet")
        horizon_p = st.slider("Horizonte:", 6, 36, 12, key="h_prophet")
        if st.button("Calcular Prophet"):
            try:
                with st.spinner("Entrenando Prophet..."):
                    df_p = ts.reset_index().rename(columns={Config.DATE_COL: 'ds', Config.PRECIPITATION_COL: 'y'})
                    m = Prophet(yearly_seasonality=True)
                    m.fit(df_p)
                    future = m.make_future_dataframe(periods=horizon_p, freq='MS')
                    fcst = m.predict(future)
                    
                    fig = go.Figure()
                    # Historia
                    hist_tail = df_p.tail(60)
                    fig.add_trace(go.Scatter(x=hist_tail['ds'], y=hist_tail['y'], name="Real"))
                    
                    # Pronóstico
                    fc_tail = fcst.tail(horizon_p)
                    fig.add_trace(go.Scatter(x=fc_tail['ds'], y=fc_tail['yhat'], name="Prophet", line=dict(color='green')))
                    
                    # Banda de error
                    fig.add_trace(go.Scatter(
                        x=pd.concat([fc_tail['ds'], fc_tail['ds'][::-1]]),
                        y=pd.concat([fc_tail['yhat_upper'], fc_tail['yhat_lower'][::-1]]),
                        fill='toself', fillcolor='rgba(0,255,0,0.1)', line=dict(color='rgba(255,255,255,0)'),
                        name='Intervalo Confianza'
                    ))

                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state['prophet_res'] = fcst[['ds', 'yhat']].tail(horizon_p).set_index('ds')['yhat']
            except Exception as e:
                st.error(f"Error en Prophet: {e}")

    # 6. COMPARACIÓN
    with tabs[5]:
        s_res = st.session_state.get('sarima_res')
        p_res = st.session_state.get('prophet_res')
        
        if s_res is not None and p_res is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s_res.index, y=s_res, name="SARIMA"))
            # Asegurar alineación de índices si difieren ligeramente
            fig.add_trace(go.Scatter(x=p_res.index, y=p_res, name="Prophet"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Ejecute ambos pronósticos primero.")

    # 7. MAPA DE RIESGO
    with tabs[6]:
        st.markdown("#### Mapa de Vulnerabilidad (Tendencias de Lluvia)")
        if st.button("Generar Mapa de Riesgo"):
            with st.spinner("Calculando tendencias regionales..."):
                trend_data = []
                # Usar todas las estaciones disponibles en la selección anual
                stations_pool = df_anual[Config.STATION_NAME_COL].unique()
                
                for stn in stations_pool:
                    sub = df_anual[df_anual[Config.STATION_NAME_COL] == stn]
                    if len(sub) > 10: # Min 10 años
                        try:
                            res = mk.original_test(sub[Config.PRECIPITATION_COL])
                            # Buscar coordenadas en gdf_stations (usando el gdf completo pasado en kwargs si está, o filtrar)
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
                    
                    # Interpolar
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
                    fig.add_trace(go.Scatter(x=df_trend.lon, y=df_trend.lat, mode='markers', text=df_trend.name))
                    fig.update_layout(title="Mapa de Tendencias de Precipitación (Pendiente Sen)", height=600)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No hay suficientes estaciones con >10 años de datos para interpolar tendencias.")

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

def display_life_zones_tab(**kwargs):
    st.subheader("🌱 Zonas de Vida (Sistema Holdridge)")
    
    tab_raster, tab_puntos = st.tabs(["🗺️ Mapa Raster (Continuo)", "📍 Estaciones (Puntos)"])
    
    # --- PESTAÑA 1: MAPA RASTER (NUEVA) ---
    with tab_raster:
        st.info("Genera una superficie continua de zonas de vida cruzando los mapas de Elevación y Precipitación.")
        
        col1, col2 = st.columns(2)
        with col1:
            res_option = st.select_slider("Resolución:", options=["Baja (Rápido)", "Media", "Alta (Lento)"], value="Baja (Rápido)")
            downscale = 8 if "Baja" in res_option else (4 if "Media" in res_option else 1)
            
        with col2:
            use_mask = st.checkbox("Recortar por Cuenca Seleccionada", value=True)
            
        # Verificar si hay cuenca en memoria
        basin_geom = None
        if use_mask:
            res_basin = st.session_state.get('basin_results')
            if res_basin and res_basin.get('ready'):
                basin_geom = res_basin['gdf_union']
                st.caption(f"Máscara activa: {res_basin.get('names', 'Cuenca')}")
            else:
                st.caption("⚠️ No hay cuenca seleccionada (Ver 'Mapas Avanzados'). Se mostrará toda la región.")

        if st.button("Generar Mapa de Zonas de Vida"):
            if not os.path.exists(Config.DEM_FILE_PATH) or not os.path.exists(Config.PRECIP_RASTER_PATH):
                st.error("Faltan los archivos raster base (DEM o PPT) en la carpeta 'data'.")
            else:
                with st.spinner("Procesando rasters..."):
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
                        # Visualización
                        # Leyenda (Debe coincidir con los IDs de analysis.py)
                        legend_map = {
                            1: "Bosque Seco Tropical (bs-T)", 2: "Bosque Húmedo Tropical (bh-T)", 
                            3: "Bosque Muy Húmedo Tropical (bmh-T)", 4: "Bosque Pluvial Tropical (bp-T)",
                            5: "Bosque Seco Premontano", 6: "Bosque Húmedo Premontano (bh-PM)", 
                            7: "Bosque Muy Húmedo Premontano (bmh-PM)", 8: "Bosque Pluvial Premontano (bp-PM)",
                            9: "Bosque Seco Montano Bajo", 10: "Bosque Húmedo Montano Bajo (bh-MB)", 
                            11: "Bosque Muy Húmedo Montano Bajo (bmh-MB)", 12: "Bosque Pluvial Montano Bajo (bp-MB)",
                            13: "Bosque Húmedo Montano", 14: "Bosque Muy Húmedo Montano"
                        }
                        
                        # Calcular coordenadas para Plotly
                        h, w = lz_arr.shape
                        x0, y0 = transform.c, transform.f
                        dx, dy = transform.a, transform.e
                        x_coords = np.linspace(x0, x0 + dx*w, w)
                        y_coords = np.linspace(y0, y0 + dy*h, h)
                        
                        # Filtrar ceros para el gráfico
                        plot_arr = lz_arr.astype(float)
                        plot_arr[plot_arr == 0] = np.nan
                        
                        fig = go.Figure(data=go.Heatmap(
                            z=plot_arr, x=x_coords, y=y_coords,
                            colorscale='Jet', showscale=False,
                            hoverongaps=False
                        ))
                        fig.update_layout(
                            title="Mapa de Zonas de Vida (Holdridge)",
                            yaxis_scaleanchor="x",
                            height=600
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Tabla de Áreas
                        unique, counts = np.unique(lz_arr[lz_arr!=0], return_counts=True)
                        # Área aprox del pixel en km2 (suponiendo coordenadas proyectadas o aprox)
                        # Si es WGS84, esto es inexacto, pero sirve de referencia relativa
                        pixel_area = abs(dx * dy) * 111 * 111 # Aprox grados a km2
                        
                        data = []
                        for v, c in zip(unique, counts):
                            data.append({
                                "Zona de Vida": legend_map.get(v, f"Clase {v}"),
                                "Píxeles": c,
                                "%": c/counts.sum()*100
                            })
                        st.dataframe(pd.DataFrame(data).sort_values("%", ascending=False).style.format({"%": "{:.1f}%"}))

    # --- PESTAÑA 2: PUNTOS (EXISTENTE) ---
    with tab_puntos:
        # Recuperar datos necesarios
        df_anual = kwargs.get('df_anual_melted')
        gdf_stations = kwargs.get('gdf_stations')
        
        if df_anual is None or gdf_stations is None:
            st.warning("Datos insuficientes.")
        else:
            # 1. Calcular Precipitación Media Anual Histórica por Estación
            ppt_media = df_anual.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean().reset_index()
            
            # 2. Unir con Altitud
            merged = pd.merge(ppt_media, gdf_stations[[Config.STATION_NAME_COL, Config.ALTITUDE_COL, 'latitude', 'longitude']], on=Config.STATION_NAME_COL)
            
            # 3. Calcular Zona de Vida para cada punto
            merged['Zona de Vida'] = merged.apply(
                lambda row: classify_holdridge_point(row[Config.PRECIPITATION_COL], row[Config.ALTITUDE_COL]), axis=1
            )
            
            # 4. Mapa Interactivo de Puntos
            fig_map = px.scatter_mapbox(
                merged,
                lat="latitude", lon="longitude",
                color="Zona de Vida",
                size=Config.PRECIPITATION_COL,
                hover_name=Config.STATION_NAME_COL,
                hover_data={Config.ALTITUDE_COL: True, Config.PRECIPITATION_COL: ':.0f'},
                zoom=8, mapbox_style="carto-positron",
                title="Clasificación en Estaciones"
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


















































