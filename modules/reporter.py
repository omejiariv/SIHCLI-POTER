# modules/reporter.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from fpdf import FPDF
import io
import os
import tempfile
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
import pymannkendall as mk

from modules.config import Config
from modules.visualizer import create_folium_map, generate_station_popup_html

# --- Configuración para Selenium ---
def setup_driver():
    """Configura y retorna un driver de Selenium para usar Chromium en Streamlit Cloud."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1200,800")
    chrome_options.binary_location = "/usr/bin/chromium"
    
    try:
        service = ChromeService(executable_path="/usr/bin/chromedriver")
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
    except Exception as e:
        st.error(f"Error al iniciar WebDriver para generar mapa: {e}")
        return None

# --- Clase para generar el PDF ---
class PDF(FPDF):
    def header(self):
        if os.path.exists(Config.LOGO_PATH):
            self.image(Config.LOGO_PATH, 10, 8, 25)
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Reporte de Análisis Hidroclimático', 0, 1, 'C')
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')

    def add_section_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)

    def add_body_text(self, text):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, text)
        self.ln()

    def add_dataframe(self, df):
        if df.empty:
            self.add_body_text("No hay datos disponibles para esta tabla.")
            return
        
        self.set_font('Arial', 'B', 8)
        col_width = (self.w - 2 * self.l_margin) / len(df.columns)
        for col_name in df.columns:
            self.cell(col_width, 8, str(col_name), 1, 0, 'C')
        self.ln()
        
        self.set_font('Arial', '', 7)
        for _, row in df.iterrows():
            for item in row:
                self.cell(col_width, 8, str(item), 1, 0, 'L')
            self.ln()
        self.ln(5)

    def add_plotly_fig(self, fig, width=190):
        try:
            img_bytes = fig.to_image(format="png", width=1000, height=500, scale=2)
            self.image(io.BytesIO(img_bytes), w=width)
            self.ln(5)
        except Exception as e:
            self.add_body_text(f"Error al generar imagen del gráfico: {e}")

    def add_folium_map(self, map_obj, width=190):
        driver = setup_driver()
        if not driver:
            self.add_body_text("No se pudo generar la imagen del mapa (WebDriver no disponible).")
            return

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode='w', encoding='utf-8') as tmp_html:
            map_obj.save(tmp_html.name)
            html_path = tmp_html.name
        
        png_path = html_path.replace(".html", ".png")

        try:
            driver.get(f"file://{html_path}")
            driver.save_screenshot(png_path)
            self.image(png_path, w=width)
            self.ln(5)
        finally:
            driver.quit()
            if os.path.exists(html_path): os.unlink(html_path)
            if os.path.exists(png_path): os.unlink(png_path)

# --- Función Principal para Generar el Reporte ---
def generate_pdf_report(report_title, sections_to_include, summary_data, df_anomalies, **data):
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, report_title, 0, 1, 'C')
    pdf.ln(10)

    # --- Extracción de DataFrames para fácil acceso ---
    gdf_filtered = data.get('gdf_filtered', pd.DataFrame())
    df_anual_melted = data.get('df_anual_melted', pd.DataFrame())
    df_monthly_filtered = data.get('df_monthly_filtered', pd.DataFrame())
    stations_for_analysis = data.get('stations_for_analysis', [])

    # --- LÓGICA COMPLETA PARA TODAS LAS SECCIONES ---
    
    if "Resumen Ejecutivo" in sections_to_include:
        # NUEVA SECCIÓN: Resumen Ejecutivo (Texto de ejemplo)
        pdf.add_section_title("1. Resumen Ejecutivo")
        pdf.add_body_text(
            "Este reporte presenta un análisis hidroclimático detallado de las estaciones seleccionadas. "
            "Se evalúan las tendencias de precipitación, la ocurrencia de anomalías y eventos extremos, "
            "y se realizan análisis de correlación con fenómenos macroclimáticos. El objetivo es proporcionar "
            "una visión integral del comportamiento de la lluvia en la región de estudio."
        )

    if "Resumen de Filtros" in sections_to_include:
        pdf.add_section_title("2. Resumen de Filtros Aplicados")
        filter_text = ""
        for key, value in summary_data.items():
            filter_text += f"- {key}: {value}\n"
        pdf.add_body_text(filter_text)

    if "Tabla de Estaciones" in sections_to_include:
        # NUEVA SECCIÓN: Tabla de Estaciones
        pdf.add_section_title("3. Tabla de Estaciones Analizadas")
        if not gdf_filtered.empty:
            stations_table = gdf_filtered[[Config.STATION_NAME_COL, Config.MUNICIPALITY_COL, Config.REGION_COL, Config.ALTITUDE_COL]].copy()
            stations_table.rename(columns={
                Config.STATION_NAME_COL: "Estacion",
                Config.MUNICIPALITY_COL: "Municipio",
                Config.REGION_COL: "Region",
                Config.ALTITUDE_COL: "Altitud (m)"
            }, inplace=True)
            pdf.add_dataframe(stations_table)
        else:
            pdf.add_body_text("No hay datos de estaciones para mostrar.")
            
    if "Distribución Espacial" in sections_to_include:
        pdf.add_section_title("2. Mapa de Distribución Espacial")
        if not gdf_filtered.empty:
            m = create_folium_map(
                location=[4.57, -74.29], zoom=5, 
                base_map_config={"tiles": "cartodbpositron", "attr": "CartoDB"}, 
                overlays_config=[], 
                fit_bounds_data=gdf_filtered
            )
            for _, row in gdf_filtered.iterrows():
                popup = generate_station_popup_html(row, df_anual_melted)
                folium.Marker(
                    location=[row.geometry.y, row.geometry.x], 
                    tooltip=row[Config.STATION_NAME_COL], 
                    popup=popup
                ).add_to(m)
            pdf.add_folium_map(m)
        else:
            pdf.add_body_text("No hay estaciones seleccionadas para mostrar en el mapa.")
            
    if "Gráficos de Series Temporales" in sections_to_include:
        pdf.add_section_title("3. Gráficos de Series Temporales")
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 10, "Precipitación Anual por Estación", 0, 1, 'L')
        if not df_anual_melted.empty:
            fig_anual = px.line(df_anual_melted, x=Config.YEAR_COL, y=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL, markers=True)
            pdf.add_plotly_fig(fig_anual)
        else:
            pdf.add_body_text("No hay datos anuales para mostrar.")

    if "Análisis de Anomalías" in sections_to_include:
        pdf.add_section_title("4. Análisis de Anomalías")
        if not df_anomalies.empty:
            df_plot = df_anomalies.groupby(Config.DATE_COL).agg(anomalia=('anomalia', 'mean')).reset_index()
            df_plot['color'] = np.where(df_plot['anomalia'] < 0, 'red', 'blue')
            fig_anom = go.Figure(go.Bar(x=df_plot[Config.DATE_COL], y=df_plot['anomalia'], marker_color=df_plot['color']))
            fig_anom.update_layout(title="Anomalías Mensuales de Precipitación (Promedio Regional)")
            pdf.add_plotly_fig(fig_anom)
        else:
            pdf.add_body_text("No hay datos de anomalías para mostrar.")

    if "Estadísticas Descriptivas" in sections_to_include:
        pdf.add_section_title("7. Estadísticas Descriptivas Mensuales")
        if not df_monthly_filtered.empty:
            # CORRECCIÓN: Se añade .reset_index() para incluir la columna de la estación
            stats_df = df_monthly_filtered.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].describe().reset_index().round(1)
            pdf.add_dataframe(stats_df)
        else:
            pdf.add_body_text("No hay datos para calcular estadísticas.")

    if "Análisis de Correlación" in sections_to_include:
        # NUEVA SECCIÓN: Matriz de Correlación
        pdf.add_section_title("8. Matriz de Correlación entre Estaciones")
        if len(stations_for_analysis) > 1:
            df_pivot = df_monthly_filtered.pivot_table(index=Config.DATE_COL, columns=Config.STATION_NAME_COL, values=Config.PRECIPITATION_COL)
            corr_matrix = df_pivot.corr()
            fig = px.imshow(corr_matrix, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu_r', title="Correlación de Precipitación Mensual")
            pdf.add_plotly_fig(fig, width=180)
        else:
            pdf.add_body_text("Se necesitan al menos dos estaciones para generar la matriz de correlación.")

    if "Análisis de Tendencias y Pronósticos" in sections_to_include:
        # NUEVA SECCIÓN: Análisis de Tendencias
        pdf.add_section_title("9. Análisis de Tendencias (Mann-Kendall)")
        if not df_anual_melted.empty:
            results = []
            for station in stations_for_analysis:
                station_data = df_anual_melted[df_anual_melted[Config.STATION_NAME_COL] == station].dropna(subset=[Config.PRECIPITATION_COL])
                if len(station_data) > 3:
                    mk_result = mk.original_test(station_data[Config.PRECIPITATION_COL])
                    results.append({
                        "Estacion": station,
                        "Tendencia": mk_result.trend,
                        "p-valor": mk_result.p,
                        "Pendiente Sen (mm/año)": mk_result.slope
                    })
            if results:
                trends_df = pd.DataFrame(results)
                pdf.add_dataframe(trends_df)
            else:
                pdf.add_body_text("No hay suficientes datos para calcular las tendencias.")
        else:
            pdf.add_body_text("No hay datos anuales para el análisis de tendencias.")

    if "Metodología y Fuentes de Datos" in sections_to_include:
        # NUEVA SECCIÓN: Metodología (Texto de ejemplo)
        pdf.add_section_title("10. Metodología y Fuentes")
        pdf.add_body_text(
            "Los datos de precipitación fueron obtenidos de [Tu Fuente de Datos]. El análisis se realizó utilizando Python y las librerías Pandas, Plotly y Streamlit. "
            "Las tendencias se evaluaron con la prueba no paramétrica de Mann-Kendall y la pendiente de Sen. Las anomalías se calcularon con respecto al promedio histórico del período seleccionado."
        )

    return bytes(pdf.output(dest='S'))
