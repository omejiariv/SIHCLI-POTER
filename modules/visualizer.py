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
import pymannkendall as mk
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from scipy import stats
from scipy.interpolate import griddata
from modules.analysis import estimate_temperature, calculate_water_balance_turc, classify_holdridge_point

# -----------------------------------------------------------------------------
# 1. FUNCIONES AUXILIARES
# -----------------------------------------------------------------------------

def get_weather_forecast_simple(lat, lon):
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
        return pd.DataFrame({
            'Fecha': daily.get('time', []),
            'Temp. Máx (°C)': daily.get('temperature_2m_max', []),
            'Temp. Mín (°C)': daily.get('temperature_2m_min', []),
            'Lluvia (mm)': daily.get('precipitation_sum', [])
        })
    except: return pd.DataFrame()

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

# 1. FUNCIONES AUXILIARES DE PARSEO Y DATOS
# -----------------------------------------------------------------------------

def parse_spanish_date(x):
    """
    Convierte fechas en formato texto español (ej: 'ene-70', 'ago-23') a datetime.
    """
    if isinstance(x, str):
        x = x.lower().strip()
        # Mapa de traducción
        trans = {
            'ene': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'abr': 'Apr',
            'may': 'May', 'jun': 'Jun', 'jul': 'Jul', 'ago': 'Aug',
            'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dic': 'Dec'
        }
        for es, en in trans.items():
            if es in x:
                x = x.replace(es, en)
                break # Solo reemplazamos el mes
        try:
            return pd.to_datetime(x, format='%b-%y')
        except:
            return pd.to_datetime(x, errors='coerce')
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
            # Controles de capas
            show_munis = st.checkbox("Municipios", value=True)
            show_cuencas = st.checkbox("Subcuencas", value=False)
            show_predios = st.checkbox("Predios", value=False)
            base_map = st.selectbox("Mapa Base:", ["CartoDB positron", "OpenStreetMap", "Stamen Terrain"])

        with col_map:
            # Centrar el mapa
            if gdf_filtered is not None and not gdf_filtered.empty:
                lat_center = gdf_filtered.geometry.y.mean()
                lon_center = gdf_filtered.geometry.x.mean()
            else:
                lat_center, lon_center = 6.2, -75.5 # Medellín por defecto

            m = folium.Map(location=[lat_center, lon_center], zoom_start=9, tiles=base_map)

            # --- CAPAS GEOMÉTRICAS (Optimizadas) ---
            # Simplificamos la geometría un poco para evitar "Bad Message Format" si son muy pesadas
            try:
                if show_munis and not gdf_municipios.empty:
                    folium.GeoJson(
                        gdf_municipios[['geometry', 'nombre']].simplify(tolerance=0.001), # Simplificar
                        name="Municipios",
                        style_function=lambda x: {'color': 'gray', 'weight': 1, 'fillOpacity': 0.05},
                        tooltip=folium.GeoJsonTooltip(fields=['nombre'])
                    ).add_to(m)

                if show_cuencas and not gdf_subcuencas.empty:
                    folium.GeoJson(
                        gdf_subcuencas[['geometry', 'nombre']].simplify(tolerance=0.001),
                        name="Subcuencas",
                        style_function=lambda x: {'color': 'blue', 'weight': 2, 'fillOpacity': 0.0},
                        tooltip=folium.GeoJsonTooltip(fields=['nombre'])
                    ).add_to(m)
                    
                if show_predios and gdf_predios is not None and not gdf_predios.empty:
                    folium.GeoJson(
                        gdf_predios[['geometry', 'nombre']].simplify(tolerance=0.0001), # Menos tolerancia para predios
                        name="Predios",
                        style_function=lambda x: {'color': 'orange', 'weight': 2, 'fillOpacity': 0.2},
                        tooltip=folium.GeoJsonTooltip(fields=['nombre'])
                    ).add_to(m)
            except Exception as e:
                st.warning(f"Algunas capas geométricas no se pudieron cargar: {e}")

            # --- ESTACIONES (Marcadores Ligeros) ---
            if gdf_filtered is not None and not gdf_filtered.empty:
                marker_cluster = MarkerCluster().add_to(m)
                # Iterar sobre datos limpios (sin geometría compleja)
                for _, row in gdf_filtered.iterrows():
                    try:
                        folium.Marker(
                            location=[row.geometry.y, row.geometry.x],
                            tooltip=f"{row[Config.STATION_NAME_COL]}",
                            icon=folium.Icon(color="green", icon="cloud")
                        ).add_to(marker_cluster)
                    except: pass

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
            
def display_graphs_tab(df_monthly_filtered, stations_for_analysis, **kwargs):
    st.subheader("📈 Análisis Gráfico Detallado")
    
    if df_monthly_filtered is None or df_monthly_filtered.empty:
        st.info("Seleccione estaciones y un rango de fechas con datos para visualizar gráficos.")
        return

    # Pestañas internas para organizar
    tab1, tab2, tab3 = st.tabs(["Serie Temporal", "Comparativa Mensual", "Mapa de Calor"])
    
    with tab1:
        # Gráfico de Líneas (Serie de Tiempo)
        fig_line = px.line(
            df_monthly_filtered, 
            x=Config.DATE_COL, 
            y=Config.PRECIPITATION_COL,
            color=Config.STATION_NAME_COL,
            title="Evolución de la Precipitación Mensual (mm)",
            labels={Config.PRECIPITATION_COL: "Lluvia (mm)", Config.DATE_COL: "Fecha"}
        )
        fig_line.update_layout(hovermode="x unified", height=500)
        st.plotly_chart(fig_line, use_container_width=True)
    
    with tab2:
        # Boxplot (Distribución por Mes)
        fig_box = px.box(
            df_monthly_filtered,
            x=Config.MONTH_COL,
            y=Config.PRECIPITATION_COL,
            color=Config.STATION_NAME_COL,
            title="Distribución de Lluvias por Mes (Ciclo Anual)",
            labels={Config.MONTH_COL: "Mes (1=Ene, 12=Dic)", Config.PRECIPITATION_COL: "Lluvia (mm)"}
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with tab3:
        # Heatmap (Intensidad)
        fig_heat = px.density_heatmap(
            df_monthly_filtered,
            x=Config.DATE_COL,
            y=Config.STATION_NAME_COL,
            z=Config.PRECIPITATION_COL,
            histfunc="avg",
            title="Mapa de Calor de Intensidad de Lluvia",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

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
    st.subheader("🌍 Mapas Avanzados (Interpolación)")
    st.markdown("Generación de superficies continuas de precipitación a partir de datos puntuales.")

    # Recuperar datos
    df_long = kwargs.get('df_long')
    gdf_stations = kwargs.get('gdf_stations')

    if df_long is None or gdf_stations is None or gdf_stations.empty:
        st.warning("Faltan datos para interpolar.")
        return

    # 1. Controles
    col1, col2, col3 = st.columns(3)
    with col1:
        years = sorted(df_long[Config.YEAR_COL].unique(), reverse=True)
        sel_year = st.selectbox("Año:", years, index=0)
    with col2:
        sel_month = st.selectbox("Mes:", range(1, 13), index=0)
    with col3:
        method = st.selectbox("Método:", ["Lineal (Rápido)", "Cúbico (Suave)"])
        interp_method = 'linear' if "Lineal" in method else 'cubic'

    # 2. Filtrar datos para ese momento específico
    # Obtenemos los datos del mes y año seleccionados
    mask_time = (df_long[Config.YEAR_COL] == sel_year) & (df_long[Config.MONTH_COL] == sel_month)
    df_slice = df_long[mask_time]
    
    if df_slice.empty:
        st.warning("No hay registros de lluvia para esa fecha.")
        return

    # 3. Unir con coordenadas (Lat/Lon)
    # Usamos gdf_stations que ya tiene 'latitude' y 'longitude' gracias al data_processor
    df_map = pd.merge(
        df_slice, 
        gdf_stations[[Config.STATION_NAME_COL, 'latitude', 'longitude']], 
        on=Config.STATION_NAME_COL, 
        how='inner'
    ).dropna(subset=['latitude', 'longitude', Config.PRECIPITATION_COL])

    if len(df_map) < 4:
        st.warning(f"Se necesitan al menos 4 estaciones con datos para interpolar (Hay {len(df_map)}).")
        return

    with st.spinner("Calculando superficie de lluvia..."):
        try:
            # 4. Crear Grilla
            # Definimos los límites basados en los datos + un margen
            x_min, x_max = df_map['longitude'].min(), df_map['longitude'].max()
            y_min, y_max = df_map['latitude'].min(), df_map['latitude'].max()
            
            # Margen del 10%
            pad_x = (x_max - x_min) * 0.1
            pad_y = (y_max - y_min) * 0.1
            
            # Crear malla de 100x100 puntos
            grid_x, grid_y = np.mgrid[
                (x_min-pad_x):(x_max+pad_x):100j, 
                (y_min-pad_y):(y_max+pad_y):100j
            ]
            
            # 5. Interpolar (scipy.griddata)
            points = df_map[['longitude', 'latitude']].values
            values = df_map[Config.PRECIPITATION_COL].values
            
            grid_z = griddata(points, values, (grid_x, grid_y), method=interp_method)
            
            # 6. Visualizar (Contour Plot)
            fig = go.Figure(data=go.Contour(
                z=grid_z.T, # Transpuesta para alinear con x/y en Plotly
                x=np.linspace(x_min-pad_x, x_max+pad_x, 100),
                y=np.linspace(y_min-pad_y, y_max+pad_y, 100),
                colorscale='Viridis',
                colorbar=dict(title='Lluvia (mm)'),
                contours=dict(
                    coloring='heatmap',
                    showlabels=True,
                    labelfont=dict(size=10, color='white')
                )
            ))
            
            # Añadir puntos de estaciones reales
            fig.add_trace(go.Scatter(
                x=df_map['longitude'], y=df_map['latitude'],
                mode='markers+text',
                marker=dict(color='red', size=5, line=dict(width=1, color='black')),
                text=df_map[Config.PRECIPITATION_COL].astype(int),
                textposition="top center",
                name="Estaciones"
            ))
            
            fig.update_layout(
                title=f"Isoyetas de Precipitación - {sel_month}/{sel_year}",
                xaxis_title="Longitud", 
                yaxis_title="Latitud",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error en la interpolación: {e}")

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










