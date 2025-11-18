# modules/visualizer.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from modules.config import Config

# --- CONSTANTES DE ESTILO ---
COLOR_PRIMARY = "#1f77b4"
COLOR_SECONDARY = "#ff7f0e"
COLOR_ACCENT = "#2ca02c"

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
    """Crea el gráfico de índices ENSO (ONI)."""
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
    st.markdown("## 🚨 Tablero de Alertas")
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
    threshold = st.slider("Umbral de Lluvia Mensual (mm):", 0, 1000, 300, 10)
    alerts_df = df_filtered[df_filtered[Config.PRECIPITATION_COL] > threshold].copy()
    
    if not alerts_df.empty:
        top_alerts = alerts_df[Config.STATION_NAME_COL].value_counts().head(10).reset_index()
        top_alerts.columns = ["Estación", "Meses sobre Umbral"]
        fig_alerts = px.bar(top_alerts, x="Meses sobre Umbral", y="Estación", orientation='h', title=f"Alertas > {threshold} mm")
        st.plotly_chart(fig_alerts, use_container_width=True)
    else:
        st.success("Sin alertas para este umbral.")

# --- PESTAÑA 3: MAPAS (CORREGIDO: USANDO FOLIUM PURO) ---
def display_spatial_distribution_tab(df_long, gdf_stations, start_date, end_date, **kwargs):
    st.markdown("## 🗺️ Distribución Espacial")
    df_filtered = _filter_data_by_date(df_long, start_date, end_date)
    
    if df_filtered.empty:
        st.warning("No hay datos para mostrar.")
        return

    precip_per_station = df_filtered.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].sum().reset_index()
    
    # Merge seguro
    if Config.STATION_NAME_COL in gdf_stations.columns and Config.STATION_NAME_COL in precip_per_station.columns:
        gdf_map = gdf_stations.merge(precip_per_station, on=Config.STATION_NAME_COL, how='inner')
    else:
        st.error("Error de columnas en los datos espaciales.")
        return

    # Crear Mapa Folium Base
    m = folium.Map(location=[6.2, -75.5], zoom_start=8, tiles="CartoDB positron")
    
    max_val = gdf_map[Config.PRECIPITATION_COL].max()
    
    # Añadir marcadores
    for _, row in gdf_map.iterrows():
        radius = 3 + (row[Config.PRECIPITATION_COL] / max_val) * 15
        
        folium.CircleMarker(
            location=[row[Config.LATITUDE_COL], row[Config.LONGITUDE_COL]],
            radius=radius,
            tooltip=f"{row[Config.STATION_NAME_COL]}: {row[Config.PRECIPITATION_COL]:,.0f} mm",
            color=COLOR_PRIMARY, 
            fill=True, 
            fill_color=COLOR_PRIMARY, 
            fill_opacity=0.6
        ).add_to(m)
        
    st_folium(m, width=800, height=500)

# --- PESTAÑA 4: GRÁFICOS ---
def display_graphs_tab(df_anual_melted, df_monthly_filtered, stations_for_analysis,
                       gdf_filtered, analysis_mode, selected_regions, selected_municipios,
                       selected_altitudes, **kwargs):
    
    st.header("📈 Análisis Gráfico")
    
    # Datos completos originales
    df_long = kwargs.get('df_long', pd.DataFrame())
    
    col_sel_1, col_sel_2 = st.columns([1, 3])
    
    with col_sel_1:
        # Usamos el DataFrame COMPLETO de estaciones para el selector (pasado en kwargs)
        # Si no está disponible, usamos el filtrado, pero idealmente queremos todas las opciones.
        gdf_stations_all = kwargs.get('gdf_stations', gdf_filtered)
        selected_station = _get_common_filtering_ui(gdf_stations_all, key_suffix="graphs")
        
    # Filtrar df_long
    df_station = df_long[df_long[Config.STATION_NAME_COL] == selected_station].copy()
    
    # Filtro de fechas
    year_range = st.session_state.get('year_range', (2000, 2024))
    start_date = pd.to_datetime(f"{year_range[0]}-01-01")
    end_date = pd.to_datetime(f"{year_range[1]}-12-31")
    
    df_station_filtered = df_station[
        (df_station[Config.DATE_COL] >= start_date) & 
        (df_station[Config.DATE_COL] <= end_date)
    ]
    
    with col_sel_2:
        if df_station_filtered.empty:
            st.warning("No hay datos para esta estación.")
            return
            
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
        fig_box = px.box(df_station, x=Config.MONTH_COL, y=Config.PRECIPITATION_COL, title="Ciclo Anual", color_discrete_sequence=[COLOR_SECONDARY])
        st.plotly_chart(fig_box, use_container_width=True)
    with col_g3:
        fig_hist = px.histogram(df_station_filtered, x=Config.PRECIPITATION_COL, title="Distribución", color_discrete_sequence=[COLOR_ACCENT])
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
                st.error("Sin datos.")
                return
            
            if Config.STATION_NAME_COL in df_month.columns and Config.STATION_NAME_COL in gdf_stations.columns:
                df_geo = df_month.merge(gdf_stations[[Config.STATION_NAME_COL, Config.LATITUDE_COL, Config.LONGITUDE_COL]], on=Config.STATION_NAME_COL, how='inner')
            else:
                st.error("Error de columnas.")
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

def display_climate_forecast_tab(df_long, gdf_stations, **kwargs):
    st.markdown("## 🔮 Pronósticos Climáticos")
    
    # Importar aquí para evitar errores circulares si forecasting falla
    try:
        from modules.forecasting import generate_prophet_forecast, get_weather_forecast
    except ImportError:
        st.error("El módulo de pronósticos no está disponible. Instale 'prophet' y 'openmeteo-requests'.")
        return

    col_sel, col_res = st.columns([1, 3])
    
    with col_sel:
        selected_station = _get_common_filtering_ui(gdf_stations, key_suffix="forecast")
        horizon = st.slider("Horizonte de Pronóstico (Meses):", 1, 24, 12)
        
    # Filtrar datos
    df_station = df_long[df_long[Config.STATION_NAME_COL] == selected_station].copy()
    
    with col_res:
        if df_station.empty:
            st.warning("Sin datos para pronosticar.")
            return

        tab1, tab2 = st.tabs(["Pronóstico Estacional (Prophet)", "Pronóstico 7 Días"])
        
        with tab1:
            with st.spinner("Calculando modelo Prophet..."):
                model, forecast, metrics = generate_prophet_forecast(df_station, horizon)
                
                if forecast is not None:
                    st.success(f"Modelo entrenado. RMSE: {metrics['RMSE']:.2f}")
                    
                    fig = go.Figure()
                    # Datos Históricos
                    fig.add_trace(go.Scatter(x=df_station[Config.DATE_COL], y=df_station[Config.PRECIPITATION_COL], name="Histórico"))
                    # Pronóstico
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Pronóstico", line=dict(color='red')))
                    # Intervalo de confianza
                    fig.add_trace(go.Scatter(
                        x=forecast['ds'], y=forecast['yhat_upper'], mode='lines',
                        line=dict(width=0), showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=forecast['ds'], y=forecast['yhat_lower'], mode='lines',
                        line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 0, 0, 0.2)', showlegend=False
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Datos insuficientes para el modelo.")
        
        with tab2:
            # Obtener coords
            station_data = gdf_stations[gdf_stations[Config.STATION_NAME_COL] == selected_station].iloc[0]
            lat, lon = station_data[Config.LATITUDE_COL], station_data[Config.LONGITUDE_COL]
            
            weather_df = get_weather_forecast(lat, lon)
            if weather_df is not None:
                st.dataframe(weather_df)
                st.line_chart(weather_df.set_index('date')['precipitation_sum'])
            else:
                st.warning("No se pudo obtener el pronóstico del tiempo.")

def display_life_zones_tab(df_long, gdf_stations, **kwargs):
    st.markdown("## 🌿 Zonas de Vida (Holdridge)")
    st.info("Clasificación bioclimática basada en la precipitación anual y la altitud.")

    try:
        from modules.life_zones import calculate_life_zones_grid, holdridge_zone_map
    except ImportError:
        st.error("Módulo de Zonas de Vida no encontrado.")
        return

    if df_long.empty or gdf_stations.empty:
        st.warning("Sin datos.")
        return

    # Calcular precipitación media anual por estación (para todo el histórico)
    df_annual_sum = df_long.groupby([Config.STATION_NAME_COL, Config.YEAR_COL])[Config.PRECIPITATION_COL].sum().reset_index()
    df_mean_annual = df_annual_sum.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean().reset_index()

    with st.spinner("Calculando zonas de vida..."):
        grid_data, error = calculate_life_zones_grid(df_mean_annual, gdf_stations)
        
        if error:
            st.error(f"Error: {error}")
            return
            
        GX, GY, grid_zones = grid_data
        
        # Mapa de calor discreto
        fig = go.Figure(data=go.Heatmap(
            z=grid_zones,
            x=GX[0],
            y=GY[:, 0],
            colorscale='Viridis',
            colorbar=dict(title="Código Zona")
        ))
        
        fig.update_layout(
            title="Mapa Aproximado de Zonas de Vida",
            xaxis_title="Longitud", yaxis_title="Latitud"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Leyenda
        st.markdown("#### Leyenda de Zonas")
        st.write(pd.DataFrame.from_dict(holdridge_zone_map, orient='index', columns=['Zona de Vida']))

def display_life_zones_tab(df_long, gdf_stations, **kwargs):
    st.markdown("## 🌿 Zonas de Vida (Holdridge)")
    st.info("Clasificación bioclimática basada en la precipitación anual promedio y la altitud.")

    try:
        from modules.life_zones import calculate_life_zones_grid, holdridge_zone_map
    except ImportError:
        st.error("Módulo de Zonas de Vida no encontrado.")
        return

    if df_long.empty or gdf_stations.empty:
        st.warning("Sin datos para calcular zonas de vida.")
        return

    # Calcular precipitación media anual por estación (para todo el histórico disponible)
    # Primero sumamos por año para obtener el total anual
    df_annual_sum = df_long.groupby([Config.STATION_NAME_COL, Config.YEAR_COL])[Config.PRECIPITATION_COL].sum().reset_index()
    # Luego promediamos esos totales anuales para obtener la media anual climática
    df_mean_annual = df_annual_sum.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean().reset_index()

    with st.spinner("Calculando mapa de zonas de vida..."):
        grid_data, error = calculate_life_zones_grid(df_mean_annual, gdf_stations)
        
        if error:
            st.error(f"Error al calcular zonas: {error}")
            return
            
        GX, GY, grid_zones = grid_data
        
        # Mapa de calor discreto para las zonas
        fig = go.Figure(data=go.Heatmap(
            z=grid_zones,
            x=GX[0],
            y=GY[:, 0],
            colorscale='Viridis',
            colorbar=dict(title="Código Zona", tickmode="array", tickvals=list(holdridge_zone_map.keys()), ticktext=list(holdridge_zone_map.values()))
        ))
        
        # Añadir estaciones como referencia
        fig.add_trace(go.Scatter(
            x=gdf_stations[Config.LONGITUDE_COL],
            y=gdf_stations[Config.LATITUDE_COL],
            mode='markers',
            marker=dict(color='red', size=5, line=dict(width=1, color='black')),
            text=gdf_stations[Config.STATION_NAME_COL],
            name="Estaciones"
        ))
        
        fig.update_layout(
            title="Mapa Aproximado de Zonas de Vida (Holdridge)",
            xaxis_title="Longitud", yaxis_title="Latitud",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Leyenda explicativa
        with st.expander("Ver Leyenda de Zonas de Vida"):
            st.dataframe(pd.DataFrame.from_dict(holdridge_zone_map, orient='index', columns=['Zona de Vida']), use_container_width=True)

# --- PESTAÑA 8: ANÁLISIS DE RIESGO (SPI) ---
def display_drought_risk_tab(df_long, gdf_stations, **kwargs):
    st.markdown("## ⚠️ Análisis de Riesgo (Sequía/Exceso)")
    
    try:
        from modules.analysis import calculate_spi
    except ImportError:
        st.error("Módulo de análisis no encontrado.")
        return

    col_sel, col_res = st.columns([1, 3])
    
    with col_sel:
        selected_station = _get_common_filtering_ui(gdf_stations, key_suffix="risk")
        spi_window = st.selectbox("Ventana SPI (Meses):", [3, 6, 12, 24], index=2, help="SPI-3 (corto plazo), SPI-12 (largo plazo)")
        
    df_station = df_long[df_long[Config.STATION_NAME_COL] == selected_station].copy()
    
    with col_res:
        if df_station.empty:
            st.warning("Sin datos.")
            return
            
        with st.spinner(f"Calculando SPI-{spi_window}..."):
            df_spi = calculate_spi(df_station, window=spi_window)
            
            if 'spi' in df_spi.columns and not df_spi['spi'].dropna().empty:
                # Gráfico de SPI
                fig_spi = go.Figure()
                
                # Barras positivas (Humedad)
                df_pos = df_spi[df_spi['spi'] >= 0]
                fig_spi.add_trace(go.Bar(x=df_pos[Config.DATE_COL], y=df_pos['spi'], name="Húmedo", marker_color='blue'))
                
                # Barras negativas (Sequía)
                df_neg = df_spi[df_spi['spi'] < 0]
                fig_spi.add_trace(go.Bar(x=df_neg[Config.DATE_COL], y=df_neg['spi'], name="Seco", marker_color='red'))
                
                # Líneas de umbral
                fig_spi.add_hline(y=-1.5, line_dash="dash", line_color="darkred", annotation_text="Sequía Severa")
                fig_spi.add_hline(y=1.5, line_dash="dash", line_color="darkblue", annotation_text="Humedad Severa")
                
                fig_spi.update_layout(title=f"Índice Estandarizado de Precipitación (SPI-{spi_window}): {selected_station}", yaxis_title="Valor SPI")
                st.plotly_chart(fig_spi, use_container_width=True)
                
                st.info("**Interpretación SPI:** Valores > 1 indican exceso de humedad. Valores < -1 indican sequía.")
            else:
                st.warning("No hay suficientes datos históricos para calcular el SPI.")

def display_drought_risk_tab(df_long, gdf_stations, **kwargs):
    st.markdown("## ⚠️ Análisis de Riesgo (Sequía/Exceso)")
    st.info("El Índice Estandarizado de Precipitación (SPI) permite identificar periodos de sequía (negativo) o exceso de humedad (positivo).")
    
    try:
        from modules.analysis import calculate_spi
    except ImportError:
        st.error("Módulo de análisis no encontrado.")
        return

    col_sel, col_res = st.columns([1, 3])
    
    with col_sel:
        # Reutilizamos el selector común
        gdf_stations_all = kwargs.get('gdf_stations', pd.DataFrame())
        selected_station = _get_common_filtering_ui(gdf_stations_all, key_suffix="risk")
        
        spi_window = st.selectbox(
            "Ventana SPI (Meses):", 
            [3, 6, 12, 24], 
            index=2, 
            help="SPI-3: Sequía meteorológica (corto plazo)\nSPI-12: Sequía hidrológica (largo plazo)"
        )
        
    # Filtrar datos
    df_station = df_long[df_long[Config.STATION_NAME_COL] == selected_station].copy()
    
    with col_res:
        if df_station.empty:
            st.warning("Sin datos para esta estación.")
            return
            
        with st.spinner(f"Calculando SPI-{spi_window}..."):
            df_spi = calculate_spi(df_station, window=spi_window)
            
            if 'spi' in df_spi.columns and not df_spi['spi'].dropna().empty:
                # Gráfico de SPI con Plotly
                fig_spi = go.Figure()
                
                # Barras positivas (Humedad - Azul)
                df_pos = df_spi[df_spi['spi'] >= 0]
                fig_spi.add_trace(go.Bar(
                    x=df_pos[Config.DATE_COL], y=df_pos['spi'], 
                    name="Húmedo", marker_color='blue', opacity=0.7
                ))
                
                # Barras negativas (Sequía - Rojo)
                df_neg = df_spi[df_spi['spi'] < 0]
                fig_spi.add_trace(go.Bar(
                    x=df_neg[Config.DATE_COL], y=df_neg['spi'], 
                    name="Seco", marker_color='red', opacity=0.7
                ))
                
                # Líneas de umbral
                fig_spi.add_hline(y=-1.5, line_dash="dash", line_color="darkred", annotation_text="Sequía Severa")
                fig_spi.add_hline(y=1.5, line_dash="dash", line_color="darkblue", annotation_text="Humedad Severa")
                
                fig_spi.update_layout(
                    title=f"Índice SPI-{spi_window}: {selected_station}", 
                    yaxis_title="Valor SPI (Desviaciones Estándar)",
                    xaxis_title="Fecha",
                    height=500
                )
                st.plotly_chart(fig_spi, use_container_width=True)
                
                with st.expander("¿Cómo interpretar el SPI?"):
                    st.markdown("""
                    * **> 2.0**: Extremadamente Húmedo
                    * **1.5 a 1.99**: Muy Húmedo
                    * **-0.99 a 0.99**: Normal
                    * **-1.0 a -1.49**: Sequía Moderada
                    * **-1.5 a -1.99**: Sequía Severa
                    * **< -2.0**: Sequía Extrema
                    """)
            else:
                st.warning("No hay suficientes datos históricos consecutivos para calcular el SPI.")

