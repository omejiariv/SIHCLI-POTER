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
from scipy.interpolate import Rbf
from modules.analysis import estimate_temperature, calculate_water_balance_turc, classify_holdridge_point, calculate_morphometry, calculate_hydrological_balance

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
            if gdf_filtered is not None and not gdf_filtered.empty:
                valid_locs = gdf_filtered.dropna(subset=['latitude', 'longitude'])
                lat_center = valid_locs['latitude'].mean() if not valid_locs.empty else 6.2
                lon_center = valid_locs['longitude'].mean() if not valid_locs.empty else -75.5
            else:
                lat_center, lon_center = 6.2, -75.5
            m = folium.Map(location=[lat_center, lon_center], zoom_start=9, tiles=base_map)
            try:
                if show_munis and not gdf_municipios.empty:
                    # Simplificar para evitar crash, preservando columnas
                    g_mun = gdf_municipios.copy()
                    g_mun['geometry'] = g_mun.geometry.simplify(0.001)
                    folium.GeoJson(g_mun, name="Municipios", style_function=lambda x: {'color': 'gray', 'weight': 1, 'fillOpacity': 0.05}, tooltip=folium.GeoJsonTooltip(fields=['nombre'])).add_to(m)
                if show_cuencas and not gdf_subcuencas.empty:
                    g_cuenca = gdf_subcuencas.copy()
                    g_cuenca['geometry'] = g_cuenca.geometry.simplify(0.001)
                    folium.GeoJson(g_cuenca, name="Subcuencas", style_function=lambda x: {'color': 'blue', 'weight': 2, 'fillOpacity': 0.0}, tooltip=folium.GeoJsonTooltip(fields=['nombre'])).add_to(m)
                if show_predios and gdf_predios is not None and not gdf_predios.empty:
                    g_pred = gdf_predios.copy()
                    g_pred['geometry'] = g_pred.geometry.simplify(0.0001)
                    folium.GeoJson(g_pred, name="Predios", style_function=lambda x: {'color': 'orange', 'weight': 2, 'fillOpacity': 0.2}, tooltip=folium.GeoJsonTooltip(fields=['nombre'])).add_to(m)
            except: pass
            if gdf_filtered is not None and not gdf_filtered.empty:
                marker_cluster = MarkerCluster().add_to(m)
                stations_to_plot = gdf_filtered.dropna(subset=['latitude', 'longitude'])
                for _, row in stations_to_plot.iterrows():
                    folium.Marker([row['latitude'], row['longitude']], tooltip=f"{row[Config.STATION_NAME_COL]}", icon=folium.Icon(color="green", icon="cloud")).add_to(marker_cluster)
            folium.LayerControl().add_to(m)
            st_folium(m, width="100%", height=600)
    
    with tab_avail:
        if df_long is not None and not df_long.empty and not gdf_filtered.empty:
            counts = df_long.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].count().reset_index()
            fig = px.bar(counts, x="precipitation", y=Config.STATION_NAME_COL, orientation='h', title="Cantidad de Datos por Estación")
            st.plotly_chart(fig, use_container_width=True)
            
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

    mode = st.radio("Modo de Análisis:", ["Regional (Comparativo)", "Por Cuenca Específica"], horizontal=True)
    
    def run_interpolation(df_data, method, grid_bounds, grid_res=100):
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

    # --- MODO 2: POR CUENCA (CORREGIDO KEYERROR Y COMPLETADO) ---
    else:
        if gdf_subcuencas.empty:
            st.warning("No hay capa de subcuencas.")
            return
        
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
                with st.spinner("Procesando datos..."):
                    # Geometría y Buffer
                    subset = gdf_subcuencas[gdf_subcuencas['nombre'].isin(sel_cuencas)]
                    gdf_union = gpd.GeoDataFrame({'geometry': [subset.unary_union]}, crs=gdf_subcuencas.crs)
                    buffer_geom = gdf_union.geometry.buffer(0.5).unary_union 
                    stations_in = gdf_stations[gdf_stations.geometry.intersects(buffer_geom)]
                    
                    if not stations_in.empty:
                        target_ids = stations_in[Config.STATION_NAME_COL].unique()
                        mask = (df_long[Config.STATION_NAME_COL].isin(target_ids)) & (df_long[Config.YEAR_COL] >= rng[0]) & (df_long[Config.YEAR_COL] <= rng[1])
                        
                        # Promedio Anual Real (mm/año)
                        df_ann = df_long[mask].groupby([Config.STATION_NAME_COL, Config.YEAR_COL])[Config.PRECIPITATION_COL].sum().reset_index()
                        df_points = df_ann.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean().reset_index()
                        df_map_data = pd.merge(df_points, gdf_stations, on=Config.STATION_NAME_COL).dropna(subset=['latitude', 'longitude'])

                        if len(df_map_data) >= 3:
                            bounds = buffer_geom.bounds
                            gx, gy, gz = run_interpolation(df_map_data, meth, [bounds[0], bounds[2], bounds[1], bounds[3]])
                            ppt_media = np.nanmean(gz) if gz is not None else df_map_data[Config.PRECIPITATION_COL].mean()
                            
                            morph = calculate_morphometry(gdf_union)
                            bal = calculate_hydrological_balance(ppt_media, morph['alt_prom_m'], gdf_union)
                            
                            # GUARDAR EN SESIÓN (CORRECCIÓN: USAR CLAVES UNIFICADAS)
                            st.session_state['basin_results'] = {
                                'ready': True, 'gx': gx, 'gy': gy, 'gz': gz, 'df': df_map_data,
                                'morph': morph, 'bal': bal, 
                                'gdf_union': gdf_union, # Clave estandarizada
                                'buffer': buffer_geom,
                                'names': ", ".join(sel_cuencas), 'periodo': f"{rng[0]}-{rng[1]}", 
                                'bounds': [bounds[0], bounds[2], bounds[1], bounds[3]]
                            }
                        else: st.error("Insuficientes estaciones (<3).")
                    else: st.error("No hay estaciones cercanas.")

            # --- RENDERIZADO ---
            res = st.session_state.get('basin_results')
            if res and res.get('ready'):
                # Validación para evitar crash por datos viejos
                if 'gdf_union' not in res: 
                    st.warning("Datos antiguos. Recalcule."); return

                st.success(f"Análisis **{res.get('periodo')}** completado.")

                # 1. Mapa
                fig = go.Figure(data=go.Contour(
                    z=res['gz'].T, x=res['gx'][:,0], y=res['gy'][0,:],
                    colorscale='Viridis', colorbar=dict(title='mm/año'), contours=dict(coloring='heatmap', showlabels=True)
                ))
                fig.add_trace(go.Scatter(x=res['df'].longitude, y=res['df'].latitude, mode='markers', marker=dict(color='red', size=5), name="Estaciones"))
                try:
                    poly = res['gdf_union'].geometry.iloc[0]
                    if poly.geom_type == 'Polygon':
                        x, y = poly.exterior.xy
                        fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines', line=dict(color='white', width=3), name='Cuenca'))
                    elif poly.geom_type == 'MultiPolygon':
                         for p in poly.geoms:
                            x, y = p.exterior.xy
                            fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines', line=dict(color='white', width=3), showlegend=False))
                except: pass
                
                fig.update_layout(height=600, title="Superficie de Lluvia", xaxis_range=[res['bounds'][0], res['bounds'][1]], yaxis_range=[res['bounds'][2], res['bounds'][3]])
                st.plotly_chart(fig, use_container_width=True)

                # 2. Datos
                st.markdown("---")
                b = res['bal']
                st.markdown(f"#### 💧 Balance Hídrico")
                vol_mm3 = b.get('Vol', 0)
                q_ls = (vol_mm3 * 1_000_000_000) / 31536000 if vol_mm3 > 0 else 0
                
                cols = st.columns(5)
                cols[0].metric("Ppt Media", f"{b['P']:.0f} mm/año")
                cols[1].metric("Altitud", f"{b['Alt']:.0f} m")
                cols[2].metric("ET", f"{b['ET']:.0f} mm/año")
                cols[3].metric("Q (mm)", f"{max(0, b['Q']):.0f} mm/año")
                cols[4].metric("Q (L/s)", f"{q_ls:.0f} L/s")
                st.info(f"**Volumen:** {vol_mm3:.2f} millones de m³.")
                
                st.markdown("#### 📐 Morfometría")
                m = res['morph']
                cm = st.columns(6)
                cm[0].metric("Área", f"{m['area_km2']:.2f} km²")
                cm[1].metric("Perímetro", f"{m['perimetro_km']:.2f} km")
                cm[2].metric("Índice Forma", f"{m['indice_forma']:.2f}")
                cm[3].metric("Alt Máx", f"{m['alt_max_m']:.0f} m")
                cm[4].metric("Alt Mín", f"{m['alt_min_m']:.0f} m")
                cm[5].metric("Pendiente", f"{m['pendiente_prom']:.1f} %")

                # 3. HIPSOMETRÍA (RESTAURADA)
                st.markdown("---")
                st.subheader("⛰️ Curva Hipsométrica")
                hypso = calculate_hypsometric_curve(res['gdf_union'])
                if hypso:
                    ch1, ch2 = st.columns([3, 1])
                    with ch1:
                        fig_h = go.Figure()
                        fig_h.add_trace(go.Scatter(x=hypso['area_percent'], y=hypso['elevations'], fill='tozeroy', name='Perfil'))
                        x_tr = np.linspace(0, 100, 50)
                        fig_h.add_trace(go.Scatter(x=x_tr, y=hypso['poly_model'](x_tr), line=dict(dash='dash'), name='Ajuste'))
                        fig_h.update_layout(title="Curva Hipsométrica", xaxis_title="% Área", yaxis_title="Elevación (m)", height=350)
                        st.plotly_chart(fig_h, use_container_width=True)
                    with ch2:
                        st.latex(hypso['equation'])

                # 4. MAPA CONTEXTO
                st.markdown("---")
                st.subheader("📍 Contexto Espacial")
                minx, maxx, miny, maxy = res['bounds']
                map_c = folium.Map(location=[(miny+maxy)/2, (minx+maxx)/2], zoom_start=9, tiles="CartoDB positron")
                folium.GeoJson(res['gdf_union'], name="Cuenca", style_function=lambda x: {'color':'blue', 'weight':3}).add_to(map_c)
                folium.GeoJson(res['buffer'], name="Radio 50km", style_function=lambda x: {'color':'gray', 'dashArray':'5,5', 'fill':False}).add_to(map_c)
                for _, r in res['df'].iterrows():
                    folium.CircleMarker([r.latitude, r.longitude], radius=3, color='red', fill=True).add_to(map_c)
                st_folium(map_c, height=500, width="100%")
        else:
            st.info("Seleccione cuencas.")
            
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
    st.subheader("📉 Tendencias, Pronósticos y Riesgo")
    
    df_monthly = kwargs.get('df_monthly_filtered')
    df_anual = kwargs.get('df_anual_melted')
    stations = kwargs.get('stations_for_analysis')
    gdf_stations = kwargs.get('gdf_stations')

    if not stations or df_monthly.empty:
        st.warning("Seleccione estaciones en el panel lateral.")
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
    
    selected_station = st.selectbox("Estación para Análisis Individual:", stations, key="trend_st")
    
    # Preparación de datos para la estación seleccionada
    if selected_station:
        station_data = df_monthly[df_monthly[Config.STATION_NAME_COL] == selected_station].sort_values(Config.DATE_COL).set_index(Config.DATE_COL)
        # Interpolación para series de tiempo continuas (necesaria para modelos)
        ts = station_data[Config.PRECIPITATION_COL].asfreq('MS').interpolate(method='time')

    # 1. TENDENCIAS (Mann-Kendall)
    with tabs[0]:
        if selected_station and len(ts) > 24:
            try:
                res = mk.original_test(ts)
                c1, c2, c3 = st.columns(3)
                c1.metric("Tendencia", res.trend, delta=f"Pendiente: {res.slope:.3f}")
                c2.metric("P-Value", f"{res.p:.4f}")
                c3.metric("Tau Kendall", f"{res.Tau:.3f}")
                
                fig = px.scatter(ts.reset_index(), x=Config.DATE_COL, y=Config.PRECIPITATION_COL, trendline="ols", title=f"Tendencia Lineal - {selected_station}")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error en Mann-Kendall: {e}")
        else:
            st.info("Datos insuficientes para la estación seleccionada.")

    # 2. DESCOMPOSICIÓN
    with tabs[1]:
        if selected_station and len(ts) > 24:
            try:
                decomp = seasonal_decompose(ts, model='additive', period=12)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ts.index, y=decomp.trend, name='Tendencia'))
                fig.add_trace(go.Scatter(x=ts.index, y=decomp.seasonal, name='Estacionalidad'))
                fig.add_trace(go.Scatter(x=ts.index, y=decomp.resid, name='Residuo', mode='markers'))
                fig.update_layout(title=f"Descomposición de Serie Temporal - {selected_station}", height=600)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error en descomposición: {e}")

    # 3. AUTOCORRELACIÓN (ACF/PACF)
    with tabs[2]:
        if selected_station and len(ts) > 24:
            from statsmodels.tsa.stattools import acf, pacf
            lag_acf = acf(ts, nlags=min(40, len(ts)//2))
            lag_pacf = pacf(ts, nlags=min(40, len(ts)//2))
            
            c1, c2 = st.columns(2)
            fig_acf = px.bar(x=range(len(lag_acf)), y=lag_acf, title="Autocorrelación (ACF)")
            c1.plotly_chart(fig_acf, use_container_width=True)
            
            fig_pacf = px.bar(x=range(len(lag_pacf)), y=lag_pacf, title="Autocorrelación Parcial (PACF)")
            c2.plotly_chart(fig_pacf, use_container_width=True)

    # 4. SARIMA
    with tabs[3]:
        st.markdown("#### Pronóstico SARIMA")
        horizon = st.slider("Horizonte:", 6, 36, 12, key="h_sarima")
        
        if st.button("Calcular SARIMA"):
            with st.spinner("Ajustando modelo SARIMA..."):
                try:
                    from statsmodels.tsa.statespace.sarimax import SARIMAX
                    # Modelo simple (1,1,1)(1,1,1,12) como base robusta
                    model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
                    model_fit = model.fit(disp=False)
                    
                    # Predicción
                    forecast = model_fit.get_forecast(steps=horizon)
                    fc_mean = forecast.predicted_mean
                    conf_int = forecast.conf_int()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ts.index[-60:], y=ts.tail(60), name="Histórico (Últimos 5 años)"))
                    fig.add_trace(go.Scatter(x=fc_mean.index, y=fc_mean, name="Pronóstico SARIMA", line=dict(color='red', dash='dash')))
                    fig.add_trace(go.Scatter(
                        x=pd.concat([conf_int.index, conf_int.index[::-1]]),
                        y=pd.concat([conf_int.iloc[:, 0], conf_int.iloc[:, 1][::-1]]),
                        fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(color='rgba(255,255,255,0)'),
                        name='Intervalo Confianza'
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state['sarima_res'] = fc_mean
                except Exception as e:
                    st.error(f"Error en SARIMA: {e}")

    # 5. PROPHET
    with tabs[4]:
        st.markdown("#### Pronóstico Prophet")
        horizon_p = st.slider("Horizonte:", 6, 36, 12, key="h_prophet")
        if st.button("Calcular Prophet"):
            with st.spinner("Entrenando Prophet..."):
                try:
                    df_p = ts.reset_index().rename(columns={Config.DATE_COL: 'ds', Config.PRECIPITATION_COL: 'y'})
                    m = Prophet(yearly_seasonality=True)
                    m.fit(df_p)
                    future = m.make_future_dataframe(periods=horizon_p, freq='MS')
                    fcst = m.predict(future)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_p['ds'].tail(60), y=df_p['y'].tail(60), name="Histórico"))
                    fig.add_trace(go.Scatter(x=fcst['ds'].tail(horizon_p), y=fcst['yhat'].tail(horizon_p), name="Prophet", line=dict(color='green')))
                    fig.add_trace(go.Scatter(
                        x=pd.concat([fcst['ds'].tail(horizon_p), fcst['ds'].tail(horizon_p)[::-1]]),
                        y=pd.concat([fcst['yhat_lower'].tail(horizon_p), fcst['yhat_upper'].tail(horizon_p)[::-1]]),
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
            fig.add_trace(go.Scatter(x=s_res.index, y=s_res, name="SARIMA", line=dict(color='red')))
            fig.add_trace(go.Scatter(x=p_res.index, y=p_res, name="Prophet", line=dict(color='green')))
            fig.update_layout(title="Comparación de Modelos", yaxis_title="Precipitación (mm)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Ejecute ambos pronósticos (SARIMA y Prophet) primero para ver la comparación.")

    # 7. MAPA DE RIESGO (Tendencias Espaciales)
    with tabs[6]:
        st.markdown("#### Mapa de Vulnerabilidad (Tendencias de Lluvia)")
        st.info("Interpolación de la Pendiente de Sen (mm/año) para todas las estaciones con >10 años de datos.")
        
        if st.button("Generar Mapa de Riesgo"):
            with st.spinner("Calculando tendencias regionales..."):
                trend_data = []
                # Usar todas las estaciones disponibles en el dataset cargado, no solo las filtradas
                for stn in gdf_stations[Config.STATION_NAME_COL].unique():
                    # Buscar datos para esta estación en df_anual completo (sin filtrar por fecha del sidebar si es posible)
                    # Pero usamos df_anual (que ya viene filtrado) o df_long completo si se pasa
                    # Usaremos df_anual por coherencia con lo que ve el usuario
                    sub = df_anual[df_anual[Config.STATION_NAME_COL] == stn]
                    
                    if len(sub) > 10: # Min 10 años
                        try:
                            res = mk.original_test(sub[Config.PRECIPITATION_COL])
                            # Obtener lat/lon
                            loc = gdf_stations[gdf_stations[Config.STATION_NAME_COL] == stn].iloc[0]
                            trend_data.append({
                                'lat': loc['latitude'], 'lon': loc['longitude'], 
                                'slope': res.slope, 'name': stn
                            })
                        except: pass
                
                if len(trend_data) >= 4:
                    df_trend = pd.DataFrame(trend_data)
                    
                    # Interpolar superficie de "Pendiente" (Slope)
                    from scipy.interpolate import griddata
                    
                    # Grilla
                    pad = 0.1
                    grid_x, grid_y = np.mgrid[df_trend.lon.min()-pad:df_trend.lon.max()+pad:100j, 
                                              df_trend.lat.min()-pad:df_trend.lat.max()+pad:100j]
                    
                    grid_z = griddata(
                        df_trend[['lon', 'lat']].values, 
                        df_trend['slope'].values, 
                        (grid_x, grid_y), 
                        method='linear' # Linear es seguro para tendencias
                    )
                    
                    fig = go.Figure(data=go.Contour(
                        z=grid_z.T, x=grid_x[:,0], y=grid_y[0,:],
                        colorscale='RdBu', # Rojo=Negativo (Disminuye lluvia), Azul=Positivo (Aumenta)
                        colorbar=dict(title='Tendencia (mm/año)'),
                        zmid=0 # Centrar el blanco en 0 cambio
                    ))
                    fig.add_trace(go.Scatter(
                        x=df_trend.lon, y=df_trend.lat, mode='markers', 
                        text=df_trend.apply(lambda row: f"{row['name']}: {row['slope']:.2f}", axis=1),
                        marker=dict(color='black', size=4), name='Estaciones'
                    ))
                    fig.update_layout(title="Mapa de Tendencias de Precipitación (Pendiente Sen)", height=600)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No hay suficientes estaciones con >10 años de datos en la selección actual para interpolar un mapa de riesgo.")

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
    st.subheader("🏜️ Análisis de Sequía (SPI / SPEI)")
    st.info("""
    Esta herramienta permite monitorear eventos extremos:
    * **SPI (Standardized Precipitation Index):** Solo considera la lluvia. Útil para sequía meteorológica.
    * **SPEI (Standardized Precipitation-Evapotranspiration Index):** Considera el balance hídrico (Lluvia - Evaporación). Útil para sequía agrícola e hidrológica bajo calentamiento.
    """)

    if df_long is None or df_long.empty:
        st.warning("No hay datos de precipitación para el cálculo.")
        return

    # 1. Configuración del Análisis
    col1, col2, col3 = st.columns(3)
    
    with col1:
        stations = sorted(df_long[Config.STATION_NAME_COL].unique())
        selected_station = st.selectbox("Estación:", stations, key="drought_station_sel")
    
    with col2:
        index_type = st.selectbox("Índice:", ["SPI (Precipitación)", "SPEI (Balance Hídrico)"], key="drought_type")
        
    with col3:
        scale = st.selectbox("Escala (Meses):", [1, 3, 6, 12, 24], index=2, 
                             help="3: Suelos, 6: Caudales, 12: Embalses/Acuíferos")

    if selected_station:
        # 2. Preparación de Datos
        # Obtener datos de la estación y ordenar cronológicamente
        df_station = df_long[df_long[Config.STATION_NAME_COL] == selected_station].sort_values(Config.DATE_COL).copy()
        df_station.set_index(Config.DATE_COL, inplace=True)
        
        # Resamplear a mensual (MS) para asegurar continuidad (rellenar huecos es vital para series de tiempo)
        # Para SPI/SPEI, los huecos deben ser tratados con cuidado.
        ts_ppt = df_station[Config.PRECIPITATION_COL].resample('MS').sum()

        # Obtener altitud para estimar temperatura (necesario para SPEI si no hay datos de T)
        try:
            station_meta = gdf_stations[gdf_stations[Config.STATION_NAME_COL] == selected_station].iloc[0]
            altitude = station_meta[Config.ALTITUDE_COL] if Config.ALTITUDE_COL in station_meta else 1500
        except:
            altitude = 1500 # Fallback por defecto

        # 3. Cálculo del Índice
        try:
            final_index_series = None
            
            if "SPI" in index_type:
                # --- CÁLCULO SPI (Gamma) ---
                # Suma móvil
                rolling_ppt = ts_ppt.rolling(window=scale, center=False).sum()
                
                # Filtrar datos válidos para el ajuste
                valid_data = rolling_ppt.dropna()
                data_nonzero = valid_data[valid_data > 0]
                
                if len(data_nonzero) > 30:
                    # Ajuste Gamma
                    alpha, loc, beta = stats.gamma.fit(data_nonzero)
                    # Calcular Probabilidad Acumulada (CDF)
                    cdf = stats.gamma.cdf(valid_data, alpha, loc=loc, scale=beta)
                    # Transformar a Z-Score (Normal)
                    final_index_series = pd.Series(stats.norm.ppf(cdf), index=valid_data.index)
                else:
                    st.warning("Datos insuficientes (no nulos) para ajustar la distribución Gamma.")

            else:
                # --- CÁLCULO SPEI (Normal o Log-Logística sobre D) ---
                # 1. Estimar Temperatura Media Mensual (Si no hay datos reales, usamos estimación por altitud)
                # T = 28 - 0.006 * h
                temp_est = max(28.0 - (0.006 * altitude), 5.0) # Evitar temperaturas irreales
                
                # 2. Calcular PET Mensual (Método Thornthwaite simplificado o proporcional a T para este demo)
                # Una aprox simple para Colombia: PET ~ 4.0 * T (mm/mes aprox variable)
                # Mejor: Método de Hargreaves simplificado si tuviéramos Tmax/Tmin. 
                # Usamos constante * T como proxy de demanda atmosférica.
                pet_series = pd.Series([temp_est * 4.5] * len(ts_ppt), index=ts_ppt.index) # ~100-120 mm/mes en trópico
                
                # 3. Balance (D)
                d_series = ts_ppt - pet_series
                
                # 4. Acumulación
                d_rolled = d_series.rolling(window=scale).sum().dropna()
                
                # 5. Estandarización (Z-Score simple)
                # El SPEI oficial usa Log-Logística, pero la Normal es una buena aproximación para apps web rápidas
                if len(d_rolled) > 30:
                    final_index_series = (d_rolled - d_rolled.mean()) / d_rolled.std()
            
            # 4. Visualización
            if final_index_series is not None:
                # Preparar DF para Plotly
                df_vis = pd.DataFrame({'Valor': final_index_series})
                df_vis['Color'] = np.where(df_vis['Valor'] >= 0, '#1f77b4', '#d62728') # Azul/Rojo
                
                # Métricas del último mes
                last_date = df_vis.index[-1]
                last_val = df_vis.iloc[-1]['Valor']
                
                state_text = "Normal"
                if last_val <= -2.0: state_text = "Sequía Extrema"
                elif last_val <= -1.5: state_text = "Sequía Severa"
                elif last_val <= -1.0: state_text = "Sequía Moderada"
                elif last_val >= 1.5: state_text = "Humedad Severa"
                
                st.metric(f"Estado Actual ({last_date.strftime('%Y-%m')})", state_text, f"{last_val:.2f} σ")

                # Gráfico
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_vis.index, 
                    y=df_vis['Valor'],
                    marker_color=df_vis['Color'],
                    name=index_type.split()[0]
                ))
                
                # Líneas de umbral
                fig.add_hline(y=0, line_width=1, line_color="black")
                fig.add_hline(y=-1.0, line_dash="dot", line_color="orange", annotation_text="Moderada")
                fig.add_hline(y=-1.5, line_dash="dash", line_color="red", annotation_text="Severa")
                fig.add_hline(y=-2.0, line_width=2, line_color="darkred", annotation_text="Extrema")
                
                fig.update_layout(
                    title=f"Evolución Histórica del {index_type} - Escala {scale} meses",
                    yaxis_title="Índice Estandarizado (σ)",
                    xaxis_title="Fecha",
                    height=500,
                    hovermode="x"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Explicación
                with st.expander("📖 Guía de Interpretación"):
                    st.markdown("""
                    * **> 2.0**: Extremadamente Húmedo
                    * **1.5 a 1.99**: Muy Húmedo
                    * **-0.99 a 0.99**: Normal
                    * **-1.0 a -1.49**: Sequía Moderada
                    * **-1.5 a -1.99**: Sequía Severa
                    * **< -2.0**: Sequía Extrema
                    """)

        except Exception as e:
            st.error(f"Error en el cálculo estadístico: {e}")
            st.info("Verifique que la estación tenga suficientes datos históricos continuos (mínimo 30 meses).")

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






























