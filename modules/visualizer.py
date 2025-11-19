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

def display_spatial_distribution_tab(gdf_filtered, **kwargs):
    st.subheader("🗺️ Distribución Espacial")
    if gdf_filtered is not None and not gdf_filtered.empty:
        map_data = gdf_filtered.copy()
        # Asegurar columnas lat/lon
        if 'latitude' not in map_data.columns and 'geometry' in map_data.columns:
            map_data['latitude'] = map_data.geometry.y
            map_data['longitude'] = map_data.geometry.x
        
        # Filtrar nulos para evitar el error de StreamlitAPIException
        map_data = map_data.dropna(subset=['latitude', 'longitude'])
        
        if not map_data.empty:
            st.map(map_data, size=20, color='#0000FF')
        else:
            st.warning("Las estaciones seleccionadas no tienen coordenadas válidas.")
    else:
        st.warning("No hay estaciones seleccionadas.")

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
    st.info("Módulo de Mapas Avanzados.")

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
    # Recuperamos los datos del kwargs para no depender de argumentos posicionales
    df_anual = kwargs.get('df_anual_melted')
    stations = kwargs.get('stations_for_analysis')

    st.subheader("📉 Análisis de Tendencias (Mann-Kendall)")
    
    if df_anual is None or df_anual.empty:
        st.warning("No hay datos anuales suficientes para calcular tendencias.")
        return

    st.info("Este módulo calcula si la lluvia está aumentando o disminuyendo estadísticamente a lo largo de los años.")

    # Selección de estación para análisis detallado
    selected_station = st.selectbox("Analizar Tendencia de:", stations, key="trend_station_sel")
    
    if selected_station:
        station_data = df_anual[df_anual[Config.STATION_NAME_COL] == selected_station].sort_values(Config.YEAR_COL)
        
        if len(station_data) < 5:
            st.warning("Se necesitan al menos 5 años de datos para calcular una tendencia confiable.")
        else:
            # Regresión Lineal Simple (Visual)
            fig = px.scatter(
                station_data, 
                x=Config.YEAR_COL, 
                y=Config.PRECIPITATION_COL, 
                trendline="ols",
                title=f"Tendencia de Precipitación Anual - {selected_station}",
                labels={Config.PRECIPITATION_COL: "Precipitación Total Anual (mm)"}
            )
            
            # Obtener resultados de la regresión
            results = px.get_trendline_results(fig)
            model = results.px_fit_results.iloc[0]
            slope = model.params[1]
            p_value = model.pvalues[1]
            
            # Interpretación automática
            trend_desc = "AUMENTO" if slope > 0 else "DISMINUCIÓN"
            significance = "Significativo" if p_value < 0.05 else "No Significativo"
            
            c1, c2 = st.columns(2)
            c1.metric("Tasa de Cambio Estimada", f"{slope:.2f} mm/año", delta=trend_desc)
            c2.metric("Confianza Estadística", significance, help="P-value < 0.05 indica alta certeza")
            
            st.plotly_chart(fig, use_container_width=True)

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
    st.info("Módulo de Estadísticas.")

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
    st.info("Módulo de Zonas de Vida.")

def display_drought_analysis_tab(**kwargs):
    st.info("Módulo de Sequía.")

def display_climate_scenarios_tab(**kwargs):
    st.info("Módulo de Escenarios.")

def display_station_table_tab(**kwargs):
    st.info("Tabla de Datos.")

def display_land_cover_analysis_tab(**kwargs):
    st.info("Módulo de Coberturas.")





