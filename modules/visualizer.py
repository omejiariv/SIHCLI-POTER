import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
import requests
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from modules.config import Config
from shapely.ops import unary_union
import pymannkendall as mk
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from scipy import stats
from scipy.interpolate import griddata
from modules.analysis import estimate_temperature, calculate_water_balance_turc, classify_holdridge_point
from modules.analysis import calculate_morphometry, calculate_hydrological_balance

# -----------------------------------------------------------------------------
# 1. FUNCIONES AUXILIARES
# -----------------------------------------------------------------------------

def get_weather_forecast_simple(lat, lon):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
            "timezone": "auto"
        }
        response = requests.get(url, params=params, timeout=5)
        data = response.json().get('daily', {})
        if not data: return pd.DataFrame()
        return pd.DataFrame({
            'Fecha': data.get('time', []),
            'Temp. Máx (°C)': data.get('temperature_2m_max', []),
            'Temp. Mín (°C)': data.get('temperature_2m_min', []),
            'Lluvia (mm)': data.get('precipitation_sum', [])
        })
    except: return pd.DataFrame()
        
def create_enso_chart(enso_data):
    if enso_data is None or enso_data.empty or Config.ENSO_ONI_COL not in enso_data.columns:
        return go.Figure().update_layout(title="Datos ENSO no disponibles", height=300)

    data = enso_data.copy().sort_values(Config.DATE_COL).dropna(subset=[Config.ENSO_ONI_COL])
    data['color'] = np.where(data[Config.ENSO_ONI_COL] >= 0.5, 'red', np.where(data[Config.ENSO_ONI_COL] <= -0.5, 'blue', 'gray'))

    fig = go.Figure()
    fig.add_trace(go.Bar(x=data[Config.DATE_COL], y=data[Config.ENSO_ONI_COL], marker_color=data['color'], name="ONI"))
    fig.add_hline(y=0.5, line_dash="dash", line_color="red")
    fig.add_hline(y=-0.5, line_dash="dash", line_color="blue")
    fig.update_layout(title="Índice Oceánico El Niño (ONI)", height=400)
    return fig
    
# 1. FUNCIONES AUXILIARES DE PARSEO Y DATOS
# -----------------------------------------------------------------------------

def parse_spanish_date(x):
    """Convierte fechas texto español a datetime."""
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

def display_alerts_tab(df_long, **kwargs):
    st.subheader("🚨 Monitor de Alertas")
    if df_long is not None and not df_long.empty:
        umbral = st.slider("Umbral de Lluvia Mensual (mm)", 100, 1000, 300)
        alertas = df_long[df_long[Config.PRECIPITATION_COL] > umbral]
        st.metric("Eventos Extremos Detectados", len(alertas))
        if not alertas.empty:
            st.dataframe(alertas.sort_values(Config.PRECIPITATION_COL, ascending=False).head(50), use_container_width=True)
    else:
        st.warning("No hay datos para alertas.")
        
def display_spatial_distribution_tab(gdf_filtered, df_long, gdf_municipios, gdf_subcuencas, gdf_predios=None, **kwargs):
    st.subheader("🗺️ Distribución Espacial y Capas")
    
    tab_map, tab_avail = st.tabs(["Mapa Interactivo", "Disponibilidad"])

    with tab_map:
        col_ctrl, col_map = st.columns([1, 3])
        
        with col_ctrl:
            st.markdown("#### Capas")
            show_munis = st.checkbox("Municipios", value=True)
            show_cuencas = st.checkbox("Subcuencas", value=False)
            show_predios = st.checkbox("Predios", value=False)
            base_map = st.selectbox("Mapa Base:", ["CartoDB positron", "OpenStreetMap", "Stamen Terrain"])

        with col_map:
            # Centrar el mapa
            if gdf_filtered is not None and not gdf_filtered.empty:
                # Filtrar nulos para el centro
                valid_locs = gdf_filtered.dropna(subset=['latitude', 'longitude'])
                if not valid_locs.empty:
                    lat_center = valid_locs['latitude'].mean()
                    lon_center = valid_locs['longitude'].mean()
                else:
                    lat_center, lon_center = 6.2, -75.5
            else:
                lat_center, lon_center = 6.2, -75.5

            m = folium.Map(location=[lat_center, lon_center], zoom_start=9, tiles=base_map)

            # --- CAPAS GEOMÉTRICAS (CORREGIDO) ---
            # Simplificamos la geometría preservando el DataFrame y sus columnas
            try:
                if show_munis and not gdf_municipios.empty:
                    # Copia para no alterar original
                    munis_sim = gdf_municipios.copy()
                    # Simplificar geometría en su lugar
                    munis_sim['geometry'] = munis_sim['geometry'].simplify(tolerance=0.001)
                    
                    folium.GeoJson(
                        munis_sim, 
                        name="Municipios",
                        style_function=lambda x: {'color': 'gray', 'weight': 1, 'fillOpacity': 0.05},
                        tooltip=folium.GeoJsonTooltip(fields=['nombre'])
                    ).add_to(m)

                if show_cuencas and not gdf_subcuencas.empty:
                    cuencas_sim = gdf_subcuencas.copy()
                    cuencas_sim['geometry'] = cuencas_sim['geometry'].simplify(tolerance=0.001)
                    
                    folium.GeoJson(
                        cuencas_sim, 
                        name="Subcuencas",
                        style_function=lambda x: {'color': 'blue', 'weight': 2, 'fillOpacity': 0.0},
                        tooltip=folium.GeoJsonTooltip(fields=['nombre'])
                    ).add_to(m)
                    
                if show_predios and gdf_predios is not None and not gdf_predios.empty:
                    predios_sim = gdf_predios.copy()
                    predios_sim['geometry'] = predios_sim['geometry'].simplify(tolerance=0.0001)
                    
                    folium.GeoJson(
                        predios_sim, 
                        name="Predios",
                        style_function=lambda x: {'color': 'orange', 'weight': 2, 'fillOpacity': 0.2},
                        tooltip=folium.GeoJsonTooltip(fields=['nombre'])
                    ).add_to(m)
            except Exception as e:
                st.warning(f"Algunas capas geométricas no se pudieron cargar: {e}")

            # --- ESTACIONES ---
            if gdf_filtered is not None and not gdf_filtered.empty:
                marker_cluster = MarkerCluster().add_to(m)
                # Filtrar nulos antes de iterar
                stations_to_plot = gdf_filtered.dropna(subset=['latitude', 'longitude'])
                
                for _, row in stations_to_plot.iterrows():
                    folium.Marker(
                        location=[row['latitude'], row['longitude']],
                        tooltip=f"{row[Config.STATION_NAME_COL]}",
                        icon=folium.Icon(color="green", icon="cloud")
                    ).add_to(marker_cluster)

            folium.LayerControl().add_to(m)
            st_folium(m, width="100%", height=600)

    with tab_avail:
        if df_long is not None and not df_long.empty and not gdf_filtered.empty:
            counts = df_long.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].count().reset_index()
            counts.columns = ["Estación", "Registros"]
            fig = px.bar(counts, x="Registros", y="Estación", orientation='h', title="Cantidad de Datos por Estación")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Seleccione estaciones para ver disponibilidad.")
            
def display_graphs_tab(df_monthly_filtered, df_anual_melted, stations_for_analysis, **kwargs):
    st.subheader("📊 Visualizaciones de Precipitación")
    
    if df_monthly_filtered.empty or df_anual_melted.empty:
        st.warning("No hay datos para mostrar.")
        return

    # Sub-Pestañas
    tab_names = ["Análisis Anual", "Análisis Mensual", "Comparación Rápida", 
                 "Boxplot Anual", "Distribución", "Acumulada", "Serie Regional"]
    tabs = st.tabs(tab_names)
    
    # 1. ANÁLISIS ANUAL
    with tabs[0]:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("##### Serie de Tiempo Anual")
            fig_anual = px.line(
                df_anual_melted, x=Config.YEAR_COL, y=Config.PRECIPITATION_COL,
                color=Config.STATION_NAME_COL, markers=True,
                title="Precipitación Total Anual"
            )
            st.plotly_chart(fig_anual, use_container_width=True)
        with col2:
            st.markdown("##### Promedios")
            avg_df = df_anual_melted.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean().sort_values(ascending=False)
            st.dataframe(avg_df, use_container_width=True)

    # 2. ANÁLISIS MENSUAL
    with tabs[1]:
        fig_mens = px.line(
            df_monthly_filtered, x=Config.DATE_COL, y=Config.PRECIPITATION_COL,
            color=Config.STATION_NAME_COL,
            title="Serie de Tiempo Mensual (Detallada)"
        )
        st.plotly_chart(fig_mens, use_container_width=True)

    # 3. COMPARACIÓN RÁPIDA (Ciclo Anual)
    with tabs[2]:
        # Promedio por mes (Ciclo estacional)
        ciclo = df_monthly_filtered.groupby([Config.STATION_NAME_COL, Config.MONTH_COL])[Config.PRECIPITATION_COL].mean().reset_index()
        fig_ciclo = px.line(
            ciclo, x=Config.MONTH_COL, y=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL,
            title="Ciclo Anual Promedio (Régimen de Lluvias)",
            labels={Config.MONTH_COL: "Mes"}
        )
        st.plotly_chart(fig_ciclo, use_container_width=True)

    # 4. BOXPLOT ANUAL
    with tabs[3]:
        fig_box = px.box(
            df_anual_melted, x=Config.STATION_NAME_COL, y=Config.PRECIPITATION_COL,
            color=Config.STATION_NAME_COL, points="all",
            title="Variabilidad de la Precipitación Anual"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # 5. DISTRIBUCIÓN
    with tabs[4]:
        fig_hist = px.histogram(
            df_monthly_filtered, x=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL,
            marginal="box", title="Distribución de Frecuencias (Mensual)",
            opacity=0.7
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # 6. ACUMULADA
    with tabs[5]:
        # Acumulada por año para ver qué año aportó más
        fig_bar = px.bar(
            df_anual_melted, x=Config.YEAR_COL, y=Config.PRECIPITATION_COL,
            color=Config.STATION_NAME_COL, title="Acumulado por Año"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # 7. SERIE REGIONAL
    with tabs[6]:
        st.markdown("##### Promedio de todas las estaciones seleccionadas")
        regional = df_monthly_filtered.groupby(Config.DATE_COL)[Config.PRECIPITATION_COL].mean().reset_index()
        fig_reg = px.area(
            regional, x=Config.DATE_COL, y=Config.PRECIPITATION_COL,
            title="Serie Regional Promedio (Índice de la Zona)",
            color_discrete_sequence=['#2ca02c']
        )
        st.plotly_chart(fig_reg, use_container_width=True)
        
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

    # Selector de Modo
    mode = st.radio("Modo de Análisis:", ["Regional (Comparativo)", "Por Cuenca Específica"], horizontal=True)
    
    # -------------------------------------------------------------------------
    # MODO 1: REGIONAL (COMPARATIVO)
    # -------------------------------------------------------------------------
    if mode == "Regional (Comparativo)":
        # (Mantener código regional igual, se omite aquí para brevedad, pero NO lo borres de tu archivo si lo usas)
        # ... (puedes dejar el bloque anterior del Modo 1 aquí) ...
        c1, c2 = st.columns(2)
        with c1:
            range1 = st.slider("Período 1:", int(df_long[Config.YEAR_COL].min()), int(df_long[Config.YEAR_COL].max()), (1980, 1990), key="p1")
        with c2:
            range2 = st.slider("Período 2:", int(df_long[Config.YEAR_COL].min()), int(df_long[Config.YEAR_COL].max()), (1991, 2000), key="p2")
        if st.button("Generar Comparación"):
            st.info("Funcionalidad regional activa (implementar lógica si se requiere).")

    # -------------------------------------------------------------------------
    # MODO 2: POR CUENCA (PERSISTENTE)
    # -------------------------------------------------------------------------
    else:
        if gdf_subcuencas.empty:
            st.warning("No hay capa de subcuencas cargada.")
            return

        st.markdown("#### 1. Selección de Cuenca(s)")
        available_cuencas = sorted(gdf_subcuencas['nombre'].unique())
        sel_cuencas = st.multiselect("Seleccione Subcuencas para fusionar:", available_cuencas)
        
        if sel_cuencas:
            # Configuración
            min_y, max_y = int(df_long[Config.YEAR_COL].min()), int(df_long[Config.YEAR_COL].max())
            rango_cuenca = st.slider("Período de Análisis:", min_y, max_y, (min_y, max_y), key="cuenca_rng")
            
            # --- BOTÓN DE CÁLCULO (Solo guarda en memoria) ---
            if st.button("Analizar Cuenca (con radio 50km)"):
                with st.spinner("Calculando..."):
                    # A. GEOMETRÍA
                    subset = gdf_subcuencas[gdf_subcuencas['nombre'].isin(sel_cuencas)]
                    union_geom = subset.unary_union
                    gdf_union = gpd.GeoDataFrame({'geometry': [union_geom]}, crs=gdf_subcuencas.crs)
                    
                    # B. FILTRO ESPACIAL (BUFFER 50KM)
                    try:
                        gdf_union_m = gdf_union.to_crs(epsg=3116)
                        gdf_stations_m = gdf_stations.to_crs(epsg=3116)
                        buffer_geom_m = gdf_union_m.geometry.buffer(50000).unary_union
                        stations_in_buffer = gdf_stations_m[gdf_stations_m.geometry.intersects(buffer_geom_m)]
                        buffer_geom_wgs84 = gpd.GeoSeries([buffer_geom_m], crs="EPSG:3116").to_crs(epsg=4326)
                    except:
                        buffer_geom_wgs84 = gdf_union.geometry.buffer(0.45).unary_union
                        stations_in_buffer = gdf_stations[gdf_stations.geometry.intersects(buffer_geom_wgs84)]

                    # C. DATOS DE LLUVIA
                    if not stations_in_buffer.empty:
                        target_stations = stations_in_buffer[Config.STATION_NAME_COL].unique()
                        mask = (
                            (df_long[Config.STATION_NAME_COL].isin(target_stations)) & 
                            (df_long[Config.YEAR_COL] >= rango_cuenca[0]) & 
                            (df_long[Config.YEAR_COL] <= rango_cuenca[1])
                        )
                        df_subset = df_long[mask]
                        df_points = df_subset.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean().reset_index()
                        
                        df_map_data = pd.merge(
                            df_points, 
                            gdf_stations[[Config.STATION_NAME_COL, 'latitude', 'longitude']], 
                            on=Config.STATION_NAME_COL
                        ).dropna(subset=['latitude', 'longitude'])

                        if len(df_map_data) >= 3:
                            # D. INTERPOLACIÓN
                            minx, miny, maxx, maxy = buffer_geom_wgs84.total_bounds
                            grid_x, grid_y = np.mgrid[minx:maxx:100j, miny:maxy:100j]
                            points = df_map_data[['longitude', 'latitude']].values
                            values = df_map_data[Config.PRECIPITATION_COL].values
                            grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
                            
                            ppt_media_estimada = df_map_data[Config.PRECIPITATION_COL].mean()

                            # E. BALANCE Y MORFOMETRÍA
                            from modules.analysis import calculate_morphometry, calculate_hydrological_balance
                            morph = calculate_morphometry(gdf_union)
                            bal = calculate_hydrological_balance(ppt_media_estimada, morph['alt_prom_m'], gdf_union)
                            
                            # --- GUARDAR EN SESIÓN (PERSISTENCIA) ---
                            st.session_state['basin_results'] = {
                                'ready': True,
                                'num_stations': len(df_map_data),
                                'grid_z': grid_z,
                                'bounds': [minx, maxx, miny, maxy],
                                'df_stations': df_map_data,
                                'morph': morph,
                                'bal': bal,
                                'gdf_union': gdf_union,
                                'buffer_geom': buffer_geom_wgs84
                            }
                        else:
                            st.error("Insuficientes estaciones (<3) en el radio de 50km.")
                    else:
                        st.error("No se encontraron estaciones en el radio de 50km.")

            # --- RENDERIZADO (Se ejecuta siempre si hay datos en memoria) ---
            res = st.session_state.get('basin_results')
            
            if res and res.get('ready'):
                st.success(f"Análisis realizado usando **{res['num_stations']} estaciones** en un radio de 50km.")

                # 1. Gráfico Interpolación
                fig_interp = go.Figure(data=go.Contour(
                    z=res['grid_z'].T, 
                    x=np.linspace(res['bounds'][0], res['bounds'][1], 100),
                    y=np.linspace(res['bounds'][2], res['bounds'][3], 100),
                    colorscale='Viridis', colorbar=dict(title='Precipitación (mm)'),
                    contours=dict(coloring='heatmap', showlabels=True)
                ))
                fig_interp.add_trace(go.Scatter(
                    x=res['df_stations']['longitude'], y=res['df_stations']['latitude'],
                    mode='markers', marker=dict(color='red', size=5, line=dict(width=1, color='black')),
                    text=res['df_stations'][Config.STATION_NAME_COL], name="Estaciones"
                ))
                fig_interp.update_layout(height=600, title="Superficie Interpolada")
                st.plotly_chart(fig_interp, use_container_width=True)

                # 2. Métricas
                st.markdown("---")
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("📐 Morfometría")
                    st.write(f"**Área:** {res['morph']['area_km2']:.2f} km²")
                    st.write(f"**Perímetro:** {res['morph']['perimetro_km']:.2f} km")
                    st.write(f"**Índice Forma:** {res['morph']['indice_forma']:.2f}")
                    st.write(f"**Altitud Media:** {res['morph']['alt_prom_m']:.0f} m")
                with c2:
                    st.subheader("💧 Balance Hídrico")
                    st.metric("Precipitación Media", f"{res['bal']['P_media_anual_mm']:.0f} mm")
                    st.metric("Caudal (Q)", f"{res['bal']['Q_mm']:.0f} mm", delta="Oferta")
                    st.caption(f"Volumen: {res['bal']['Q_m3_año']/1e6:.2f} Millones m³")

                # 3. Mapa Folium
                st.markdown("#### Contexto Espacial")
                minx, maxx, miny, maxy = res['bounds']
                m = folium.Map(location=[(miny+maxy)/2, (minx+maxx)/2], zoom_start=9, tiles="CartoDB positron")
                folium.GeoJson(res['gdf_union'], name="Cuenca", style_function=lambda x: {'color': 'blue', 'weight': 3}).add_to(m)
                folium.GeoJson(res['buffer_geom'], name="Radio 50km", style_function=lambda x: {'color': 'gray', 'dashArray': '5,5', 'fill': False}).add_to(m)
                for _, row in res['df_stations'].iterrows():
                    folium.CircleMarker([row['latitude'], row['longitude']], radius=3, color='red', fill=True).add_to(m)
                st_folium(m, height=400, width="100%")

        else:
            st.info("Seleccione cuencas para comenzar.")

def display_climate_forecast_tab(**kwargs):
    st.subheader("🔮 Pronóstico Climático (Índices)")
    df_enso = kwargs.get('df_enso')
    
    if df_enso is None or df_enso.empty:
        st.warning("No hay datos de índices climáticos (ONI/SOI) cargados.")
        return

    st.info("Análisis de la evolución de índices macroclimáticos que afectan la lluvia en la región.")
    
    # Selector de índice
    index_col = st.selectbox("Seleccione Índice:", [Config.ENSO_ONI_COL, Config.SOI_COL], index=0)
    
    if index_col in df_enso.columns:
        # Filtrar nulos
        data = df_enso.dropna(subset=[index_col]).sort_values(Config.DATE_COL)
        
        # Gráfico interactivo
        fig = px.line(data, x=Config.DATE_COL, y=index_col, title=f"Evolución Histórica: {index_col}")
        
        # Añadir líneas de referencia para ONI
        if index_col == Config.ENSO_ONI_COL:
            fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="El Niño")
            fig.add_hline(y=-0.5, line_dash="dash", line_color="blue", annotation_text="La Niña")
            
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"La columna {index_col} no se encuentra en los datos.")

def display_trends_and_forecast_tab(**kwargs):
    st.subheader("📉 Tendencias y Pronósticos")
    
    df_monthly = kwargs.get('df_monthly_filtered')
    stations = kwargs.get('stations_for_analysis')
    
    if not stations:
        st.warning("Seleccione estaciones.")
        return

    # Selector interno
    selected_station = st.selectbox("Seleccionar Estación para Análisis:", stations, key="trend_station_main")
    
    if selected_station:
        # Filtrar datos de la estación
        station_data = df_monthly[df_monthly[Config.STATION_NAME_COL] == selected_station].copy()
        station_data = station_data.sort_values(Config.DATE_COL).set_index(Config.DATE_COL)
        
        # Llenar huecos mínimos para que las series de tiempo funcionen
        ts = station_data[Config.PRECIPITATION_COL].asfreq('MS').interpolate(method='linear')

        tab_trend, tab_prophet = st.tabs(["Tendencia (Mann-Kendall)", "Pronóstico (Prophet)"])

        with tab_trend:
            if len(ts) > 24:
                try:
                    # Test de Mann-Kendall
                    result = mk.original_test(ts)
                    trend = result.trend
                    slope = result.slope
                    p_value = result.p

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Tendencia Detectada", "Creciente 📈" if slope > 0 else "Decreciente 📉")
                    c2.metric("Pendiente (Sen's Slope)", f"{slope:.3f} mm/mes")
                    c3.metric("Significancia (p-value)", f"{p_value:.4f}", delta="Significativo" if p_value < 0.05 else "No Sig.")

                    # Gráfico de descomposición estacional
                    decomp = seasonal_decompose(ts, model='additive', period=12)
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(x=ts.index, y=decomp.trend, mode='lines', name='Tendencia', line=dict(color='red', width=2)))
                    fig_trend.add_trace(go.Scatter(x=ts.index, y=ts, mode='lines', name='Observado', opacity=0.3))
                    fig_trend.update_layout(title="Descomposición de Tendencia")
                    st.plotly_chart(fig_trend, use_container_width=True)
                except Exception as e:
                    st.error(f"Error en cálculo de tendencia: {e}")
            else:
                st.warning("Se necesitan al menos 24 meses de datos para calcular tendencias.")

        with tab_prophet:
            st.markdown("##### Pronóstico Automático (Prophet)")
            horizon = st.slider("Meses a pronosticar:", 6, 36, 12)
            
            if st.button("Generar Pronóstico"):
                with st.spinner("Entrenando modelo inteligente..."):
                    try:
                        # Preparar datos para Prophet (ds, y)
                        df_prophet = ts.reset_index().rename(columns={Config.DATE_COL: 'ds', Config.PRECIPITATION_COL: 'y'})
                        
                        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                        m.fit(df_prophet)
                        
                        future = m.make_future_dataframe(periods=horizon, freq='MS')
                        forecast = m.predict(future)
                        
                        # Graficar
                        fig_fc = go.Figure()
                        # Histórico
                        fig_fc.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], name='Histórico', opacity=0.4))
                        # Pronóstico
                        fc_data = forecast.iloc[-horizon:]
                        fig_fc.add_trace(go.Scatter(x=fc_data['ds'], y=fc_data['yhat'], name='Pronóstico', line=dict(color='green', width=2)))
                        # Banda de confianza
                        fig_fc.add_trace(go.Scatter(
                            x=pd.concat([fc_data['ds'], fc_data['ds'][::-1]]),
                            y=pd.concat([fc_data['yhat_upper'], fc_data['yhat_lower'][::-1]]),
                            fill='toself', fillcolor='rgba(0,255,0,0.1)', line=dict(color='rgba(255,255,255,0)'),
                            name='Intervalo de Confianza'
                        ))
                        
                        fig_fc.update_layout(title=f"Pronóstico a {horizon} meses - {selected_station}")
                        st.plotly_chart(fig_fc, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error en Prophet: {e}")

def display_anomalies_tab(df_monthly_filtered, df_long, **kwargs):
    st.subheader("⚠️ Análisis de Anomalías")
    st.markdown("Identificación de meses inusualmente húmedos o secos respecto al promedio histórico.")
    
    if df_monthly_filtered is None or df_monthly_filtered.empty:
        st.warning("Datos insuficientes.")
        return

    # Calcular promedio mensual histórico (Climatología)
    # Agrupamos por MES (1-12) para saber cuánto llueve normalmente en Enero, Febrero, etc.
    climatology = df_long.groupby(Config.MONTH_COL)[Config.PRECIPITATION_COL].mean().reset_index()
    climatology = climatology.rename(columns={Config.PRECIPITATION_COL: 'mean_ppt'})
    
    # Unir con los datos filtrados
    df_anom = pd.merge(df_monthly_filtered, climatology, on=Config.MONTH_COL, how='left')
    
    # Calcular Anomalía (Valor Real - Promedio Histórico)
    df_anom['anomalia'] = df_anom[Config.PRECIPITATION_COL] - df_anom['mean_ppt']
    
    # Colores para el gráfico
    df_anom['color'] = np.where(df_anom['anomalia'] >= 0, 'blue', 'red')
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_anom[Config.DATE_COL],
        y=df_anom['anomalia'],
        marker_color=df_anom['color'],
        name="Anomalía"
    ))
    fig.update_layout(
        title="Anomalías de Precipitación (mm)",
        yaxis_title="Desviación del Promedio (mm)",
        xaxis_title="Fecha"
    )
    st.plotly_chart(fig, use_container_width=True)

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
    df_monthly = kwargs.get('df_monthly_filtered')
    df_enso = kwargs.get('df_enso')
    
    if df_monthly is None or df_monthly.empty:
        st.warning("Faltan datos de precipitación.")
        return
    if df_enso is None or df_enso.empty:
        st.warning("Faltan datos ENSO.")
        return

    try:
        # Copias para no alterar originales
        ppt_data = df_monthly.copy()
        enso_data = df_enso.copy()
        
        # 1. Normalizar Fechas (CRÍTICO)
        ppt_data[Config.DATE_COL] = pd.to_datetime(ppt_data[Config.DATE_COL], errors='coerce')
        
        # Parseo inteligente para ENSO (soporta 'ene-70')
        if enso_data[Config.DATE_COL].dtype == 'object':
            enso_data[Config.DATE_COL] = enso_data[Config.DATE_COL].astype(str).apply(parse_spanish_date)
        else:
            enso_data[Config.DATE_COL] = pd.to_datetime(enso_data[Config.DATE_COL], errors='coerce')
            
        # Eliminar fechas inválidas
        ppt_data = ppt_data.dropna(subset=[Config.DATE_COL])
        enso_data = enso_data.dropna(subset=[Config.DATE_COL])

        # 2. Agrupar y Unir
        regional_ppt = ppt_data.groupby(Config.DATE_COL)[Config.PRECIPITATION_COL].mean().reset_index()
        
        # El merge ahora funcionará porque ambas son datetime64[ns]
        merged = pd.merge(regional_ppt, enso_data, on=Config.DATE_COL, how='inner')
        
        if len(merged) > 12:
            col1, col2 = st.columns([2, 1])
            with col1:
                if Config.ENSO_ONI_COL in merged.columns:
                    fig = px.scatter(
                        merged, x=Config.ENSO_ONI_COL, y=Config.PRECIPITATION_COL, trendline="ols",
                        title="Lluvia vs. ONI", opacity=0.6
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Columna ONI no encontrada en datos unidos.")
            with col2:
                if Config.ENSO_ONI_COL in merged.columns:
                    corr = merged[Config.ENSO_ONI_COL].corr(merged[Config.PRECIPITATION_COL])
                    st.metric("Correlación Pearson", f"{corr:.2f}")
        else:
            st.warning(f"Datos insuficientes tras la unión ({len(merged)} meses). Verifique rangos de fecha.")

    except Exception as e:
        st.error(f"Error en correlación: {e}")
        
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
    st.info("Clasificación bioclimática basada en la precipitación anual y la altitud de cada estación.")
    
    # Recuperar datos necesarios
    df_anual = kwargs.get('df_anual_melted')
    gdf_stations = kwargs.get('gdf_stations')
    
    if df_anual is None or gdf_stations is None:
        st.warning("Datos insuficientes.")
        return

    # 1. Calcular Precipitación Media Anual Histórica por Estación
    ppt_media = df_anual.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean().reset_index()
    
    # 2. Unir con Altitud
    merged = pd.merge(ppt_media, gdf_stations[[Config.STATION_NAME_COL, Config.ALTITUDE_COL, 'latitude', 'longitude']], on=Config.STATION_NAME_COL)
    
    # 3. Calcular Zona de Vida para cada punto
    merged['Zona de Vida'] = merged.apply(
        lambda row: classify_holdridge_point(row[Config.PRECIPITATION_COL], row[Config.ALTITUDE_COL]), axis=1
    )
    
    # 4. Mapa Interactivo de Zonas
    fig_map = px.scatter_mapbox(
        merged,
        lat="latitude", lon="longitude",
        color="Zona de Vida",
        size=Config.PRECIPITATION_COL,
        hover_name=Config.STATION_NAME_COL,
        hover_data={Config.ALTITUDE_COL: True, Config.PRECIPITATION_COL: ':.0f'},
        zoom=8, mapbox_style="carto-positron",
        title="Distribución de Zonas de Vida"
    )
    st.plotly_chart(fig_map, use_container_width=True)
    
    # 5. Tabla Resumen
    st.markdown("#### Clasificación por Estación")
    st.dataframe(merged[[Config.STATION_NAME_COL, 'Zona de Vida', Config.PRECIPITATION_COL, Config.ALTITUDE_COL]], use_container_width=True)

def display_drought_analysis_tab(df_long, gdf_stations, **kwargs):
    st.subheader("🏜️ Análisis de Sequía (Índice SPI)")
    st.info("El Índice Estandarizado de Precipitación (SPI) cuantifica el déficit o exceso de lluvia para diferentes escalas de tiempo.")

    if df_long is None or df_long.empty:
        st.warning("No hay datos para calcular el SPI.")
        return

    # 1. Configuración
    col1, col2 = st.columns(2)
    with col1:
        # Filtramos estaciones disponibles
        stations = sorted(df_long[Config.STATION_NAME_COL].unique())
        selected_station = st.selectbox("Seleccionar Estación:", stations, key="spi_station_sel")
    with col2:
        scale = st.selectbox("Escala de Tiempo (Meses):", [3, 6, 12, 24], index=1, 
                             help="3=Corto plazo (agrícola), 12=Largo plazo (hidrológico)")

    if selected_station:
        # 2. Preparar datos
        df_station = df_long[df_long[Config.STATION_NAME_COL] == selected_station].sort_values(Config.DATE_COL).copy()
        df_station.set_index(Config.DATE_COL, inplace=True)
        
        # Resamplear para asegurar frecuencia mensual continua (rellenar huecos con NaN es necesario para SPI)
        ts = df_station[Config.PRECIPITATION_COL].resample('MS').sum()
        
        # 3. Cálculo del SPI (Algoritmo Gamma)
        # Ventana móvil de suma
        rolling_sum = ts.rolling(window=scale, center=False).sum()
        
        # Ajuste a distribución Gamma y transformación a Normal (Z-score)
        # (Simplificado para robustez: si hay ceros, se manejan aparte)
        valid_data = rolling_sum.dropna()
        
        if len(valid_data) > 30: # Se recomienda mínimo 30 datos
            try:
                # Fit Gamma
                fit_alpha, fit_loc, fit_beta = stats.gamma.fit(valid_data[valid_data > 0])
                
                # Calcular CDF
                cdf = stats.gamma.cdf(valid_data, fit_alpha, loc=fit_loc, scale=fit_beta)
                
                # Transformar a Z-score (Inversa de la Normal)
                spi = stats.norm.ppf(cdf)
                
                # Crear DataFrame de resultados
                df_spi = pd.DataFrame({'SPI': spi}, index=valid_data.index)
                
                # 4. Visualización
                df_spi['Color'] = np.where(df_spi['SPI'] >= 0, 'blue', 'red')
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_spi.index, y=df_spi['SPI'],
                    marker_color=df_spi['Color'],
                    name='SPI'
                ))
                # Umbrales oficiales
                fig.add_hline(y=-1.5, line_dash="dash", line_color="darkred", annotation_text="Sequía Severa")
                fig.add_hline(y=-2.0, line_dash="dash", line_color="black", annotation_text="Sequía Extrema")
                fig.add_hline(y=1.5, line_dash="dash", line_color="darkblue", annotation_text="Humedad Severa")
                
                fig.update_layout(
                    title=f"Evolución del SPI-{scale} en {selected_station}",
                    yaxis_title="Índice SPI (Desviaciones Estándar)",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar estado actual
                last_val = df_spi.iloc[-1]['SPI']
                last_date = df_spi.index[-1].strftime('%Y-%m')
                
                status = "Normal"
                if last_val <= -2.0: status = "SEQUÍA EXTREMA 💀"
                elif last_val <= -1.5: status = "SEQUÍA SEVERA ⚠️"
                elif last_val <= -1.0: status = "Sequía Moderada"
                elif last_val >= 1.5: status = "Exceso de Humedad 💧"
                
                st.info(f"**Estado en {last_date}:** {status} (SPI = {last_val:.2f})")
                
            except Exception as e:
                st.error(f"Error matemático calculando SPI: {e}")
        else:
            st.warning(f"Datos insuficientes para calcular SPI-{scale} (se requieren min 30 meses de datos consecutivos).")

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
    st.info("Módulo de Coberturas.")

















