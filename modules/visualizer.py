import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
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
    st.subheader("🗺️ Distribución Espacial y Datos Históricos")
    
    # Creamos 3 pestañas ahora
    tab_map, tab_avail, tab_matrix = st.tabs(["📍 Mapa Interactivo", "📊 Disponibilidad de Datos", "📅 Series Anuales"])

    # -------------------------------------------------------------------------
    # PESTAÑA 1: MAPA (Se mantiene igual de robusto)
    # -------------------------------------------------------------------------
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
    
    # -------------------------------------------------------------------------
    # PESTAÑA 2: DISPONIBILIDAD (CON ORDENAMIENTO)
    # -------------------------------------------------------------------------
    with tab_avail:
        st.markdown("#### Cantidad de Datos por Estación")
        
        if df_long is not None and not df_long.empty and not gdf_filtered.empty:
            # 1. Filtrar datos para las estaciones seleccionadas
            target_stations = gdf_filtered[Config.STATION_NAME_COL].unique()
            df_subset = df_long[df_long[Config.STATION_NAME_COL].isin(target_stations)]
            
            # 2. Contar
            counts = df_subset.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].count().reset_index()
            counts.columns = ["Estación", "Registros"]
            
            # 3. Lógica de Ordenamiento
            sort_option = st.radio(
                "Ordenar por:", 
                ["Mayor a Menor", "Menor a Mayor", "Alfabético"], 
                horizontal=True,
                key="sort_avail"
            )
            
            if sort_option == "Mayor a Menor":
                counts = counts.sort_values("Registros", ascending=True) # Ascendente para que en h-bar quede el mayor arriba
            elif sort_option == "Menor a Mayor":
                counts = counts.sort_values("Registros", ascending=False)
            else:
                counts = counts.sort_values("Estación", ascending=False) # Alfabético inverso para graficar de A-Z arriba-abajo
            
            # 4. Graficar
            fig = px.bar(
                counts, 
                x="Registros", 
                y="Estación", 
                orientation='h', 
                text="Registros",
                height=max(500, len(counts) * 25) # Altura dinámica
            )
            fig.update_traces(marker_color='#1f77b4')
            fig.update_layout(xaxis_title="Número de Meses con Datos", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Seleccione estaciones para ver disponibilidad.")

    # -------------------------------------------------------------------------
    # PESTAÑA 3: SERIES ANUALES (MATRIZ DE CALOR) - NUEVA
    # -------------------------------------------------------------------------
    with tab_matrix:
        st.markdown("#### Series de Precipitación Anual por Estación (mm)")
        st.info("Tabla cruzada de precipitación total anual. Los colores indican la intensidad (Oscuro = Menos lluvia, Claro/Amarillo = Más lluvia).")

        if df_long is not None and not df_long.empty and not gdf_filtered.empty:
            # 1. Filtrar y Agrupar Anual
            target_stations = gdf_filtered[Config.STATION_NAME_COL].unique()
            df_subset = df_long[df_long[Config.STATION_NAME_COL].isin(target_stations)]
            
            # Pivote: Indice=Estación, Columnas=Año, Valor=Suma Precipitación
            df_pivot = df_subset.pivot_table(
                index=Config.STATION_NAME_COL, 
                columns=Config.YEAR_COL, 
                values=Config.PRECIPITATION_COL, 
                aggfunc='sum'
            )
            
            # 2. Estilizar como Heatmap (Gradiente)
            # Usamos 'viridis' para simular el estilo de la imagen (morado -> amarillo)
            st.dataframe(
                df_pivot.style
                .format("{:.0f}", na_rep="0")
                .background_gradient(cmap="viridis", axis=None, vmin=0, vmax=df_pivot.max().max())
                .highlight_null(color='black'), # Nulos/Ceros oscuros
                use_container_width=True,
                height=600
            )
        else:
            st.warning("No hay datos para generar la matriz anual.")
            
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
                        
                        # NUEVA NOTA EXPLICATIVA
                        with st.expander("ℹ️ Explicación de la Ecuación"):
                            st.markdown("""
                            Esta ecuación polinómica modela la relación entre la **Elevación (H)** y el **% de Área Acumulada (A)**.
                            
                            * **H**: Altitud en metros sobre el nivel del mar.
                            * **A**: Porcentaje del área de la cuenca que está *por encima* de esa altitud (0-100).
                            
                            Los coeficientes indican la concavidad/convexidad de la cuenca, lo que influye en su respuesta hidrológica (erosión vs. sedimentación).
                            """)

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

    if df_long is None or df_long.empty:
        st.warning("No hay datos de precipitación.")
        return

    # 1. Selección Global de Estación
    stations = sorted(df_long[Config.STATION_NAME_COL].unique())
    selected_station = st.selectbox("Seleccionar Estación para Análisis:", stations, key="extremes_station_sel")
    
    if not selected_station: return

    # 2. Pestañas Internas
    tab_idx, tab_freq, tab_perc = st.tabs([
        "Índices Estandarizados (SPI/SPEI)", 
        "Frecuencia de Máximos (Gumbel)", 
        "Umbrales y Percentiles"
    ])

    # -------------------------------------------------------------------------
    # SUB-PESTAÑA 1: SPI / SPEI (Lo que ya teníamos)
    # -------------------------------------------------------------------------
    with tab_idx:
        c1, c2 = st.columns(2)
        idx_type = c1.radio("Índice:", ["SPI (Lluvia)", "SPEI (Balance)"], horizontal=True)
        scale = c2.selectbox("Escala (Meses):", [1, 3, 6, 12, 24], index=2)
        
        # Preparar datos
        df_station = df_long[df_long[Config.STATION_NAME_COL] == selected_station].sort_values(Config.DATE_COL).set_index(Config.DATE_COL)
        ts_ppt = df_station[Config.PRECIPITATION_COL].resample('MS').sum()
        
        # Obtener altitud para SPEI
        try:
            alt = gdf_stations[gdf_stations[Config.STATION_NAME_COL] == selected_station].iloc[0][Config.ALTITUDE_COL]
        except: alt = 1500

        # Cálculo (Lógica resumida llamando a analysis.py)
        try:
            if "SPI" in idx_type:
                # SPI Gamma (Implementación rápida inline o llamar a función si la tienes separada)
                # Por robustez, repetimos la lógica de cálculo seguro aquí o llamamos a calculate_spi si existe
                from modules.analysis import calculate_spi # Asumiendo que existe o usando la lógica previa
                # Si no tienes calculate_spi en analysis, usa el bloque de la respuesta anterior
                # Aquí uso un placeholder funcional:
                if len(ts_ppt) > 30:
                    roll = ts_ppt.rolling(scale).sum().dropna()
                    fit_alpha, fit_loc, fit_beta = stats.gamma.fit(roll[roll > 0])
                    cdf = stats.gamma.cdf(roll, fit_alpha, loc=fit_loc, scale=fit_beta)
                    series_idx = pd.Series(stats.norm.ppf(cdf), index=roll.index)
                else: series_idx = None
            else:
                # SPEI (Lógica simplificada)
                from modules.analysis import calculate_spei
                t_series = pd.Series([25 - (0.006*alt)]*len(ts_ppt), index=ts_ppt.index)
                series_idx = calculate_spei(ts_ppt, t_series, window=scale)

            if series_idx is not None:
                df_vis = pd.DataFrame({'Val': series_idx})
                df_vis['Color'] = np.where(df_vis['Val'] >= 0, 'blue', 'red')
                
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df_vis.index, y=df_vis['Val'], marker_color=df_vis['Color'], name=idx_type))
                fig.add_hline(y=-1.5, line_dash="dash", line_color="red", annotation_text="Sequía Severa")
                fig.add_hline(y=1.5, line_dash="dash", line_color="blue", annotation_text="Humedad Severa")
                fig.update_layout(title=f"Evolución {idx_type}-{scale}", height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Datos insuficientes para el índice.")
        except Exception as e:
            st.error(f"Error calculando índice: {e}")

    # -------------------------------------------------------------------------
    # SUB-PESTAÑA 2: FRECUENCIA (GUMBEL) - ¡NUEVO!
    # -------------------------------------------------------------------------
    with tab_freq:
        st.markdown("#### Análisis de Frecuencia (Máximos Anuales)")
        st.info("Estimación de la precipitación máxima esperada para diferentes Períodos de Retorno (Tr) usando la distribución de Gumbel.")
        
        from modules.analysis import calculate_return_periods
        
        res_df, debug_data = calculate_return_periods(df_long, selected_station)
        
        if res_df is not None:
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.markdown("**Tabla de Diseño**")
                st.dataframe(res_df.style.format({"Ppt Máxima Esperada (mm)": "{:.1f}"}), use_container_width=True)
                
                # Botón descarga
                csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Descargar Tabla (CSV)", csv, f"frecuencia_{selected_station}.csv", "text/csv")

            with c2:
                st.markdown("**Curva de Frecuencia**")
                # Graficar Gumbel
                annual_max = debug_data['data']
                params = debug_data['params']
                
                # Eje X para la curva (Tr continuo)
                tr_plot = np.linspace(1.01, 100, 100)
                prob_plot = 1 - (1/tr_plot)
                ppt_plot = stats.gumbel_r.ppf(prob_plot, *params)
                
                fig_freq = go.Figure()
                
                # Línea Teórica
                fig_freq.add_trace(go.Scatter(
                    x=tr_plot, y=ppt_plot, mode='lines', 
                    name='Curva Gumbel', line=dict(color='red', width=2)
                ))
                
                # Puntos Observados (Posición de ploteo de Gringorten o Weibull)
                # Weibull: P = m / (n+1) -> Tr = (n+1)/m
                sorted_max = np.sort(annual_max.values)
                n = len(sorted_max)
                rank = np.arange(1, n+1)
                tr_obs = (n + 1) / (n + 1 - rank) # Tr para máximos (orden ascendente en Gumbel plot)
                
                # Ojo: Gumbel ppf usa probabilidad de NO excedencia.
                # Para graficar los puntos observados contra la curva ajustada:
                fig_freq.add_trace(go.Scatter(
                    x=tr_obs, y=sorted_max, mode='markers',
                    name='Datos Observados', marker=dict(color='black', size=6)
                ))
                
                fig_freq.update_layout(
                    xaxis_title="Período de Retorno (Años)",
                    yaxis_title="Precipitación Máxima Anual (mm)",
                    xaxis_type="log", # Escala logarítmica es estándar para Tr
                    height=500,
                    hovermode="x"
                )
                st.plotly_chart(fig_freq, use_container_width=True)
        else:
            st.error(debug_data) # Mostrar mensaje de error (ej. <10 años)

    # -------------------------------------------------------------------------
    # SUB-PESTAÑA 3: PERCENTILES (Configurable) - ¡NUEVO!
    # -------------------------------------------------------------------------
    with tab_perc:
        st.markdown("#### Análisis de Umbrales y Percentiles")
        
        c_p1, c_p2 = st.columns(2)
        p_low = c_p1.slider("Percentil Bajo (Seco):", 1, 20, 10)
        p_high = c_p2.slider("Percentil Alto (Húmedo):", 80, 99, 95)
        
        from modules.analysis import calculate_percentiles_extremes
        res_perc = calculate_percentiles_extremes(df_long, selected_station, p_low, p_high)
        
        if res_perc:
            df_p, t_low, t_high = res_perc
            
            c_m1, c_m2 = st.columns(2)
            c_m1.metric(f"Umbral Seco (P{p_low})", f"{t_low:.1f} mm")
            c_m2.metric(f"Umbral Húmedo (P{p_high})", f"{t_high:.1f} mm")
            
            # Gráfico de dispersión clasificado
            fig_p = px.scatter(
                df_p, x=Config.DATE_COL, y=Config.PRECIPITATION_COL,
                color='Tipo Evento',
                color_discrete_map={'Normal': 'gray', f'Bajo (<P{p_low})': 'red', f'Alto (>P{p_high})': 'blue'},
                title=f"Eventos Extremos según Percentiles"
            )
            fig_p.add_hline(y=t_high, line_dash="dot", line_color="blue")
            fig_p.add_hline(y=t_low, line_dash="dot", line_color="red")
            st.plotly_chart(fig_p, use_container_width=True)
        else:
            st.warning("Error calculando percentiles.")
            
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














