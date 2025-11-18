# modules/reporter.py

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from fpdf import FPDF
from modules.utils import standardize_numeric_column
from modules.config import Config

# --- CORRECCIÓN DE IMPORTACIONES ---
# Solo importamos lo que existe en el nuevo visualizer.py
# create_folium_map YA NO EXISTE, por eso se eliminó de aquí.
from modules.visualizer import create_enso_chart 

from modules.analysis import calculate_monthly_anomalies 

# --- CLASE PDF PERSONALIZADA ---
class PDFReport(FPDF):
    def header(self):
        # Logo
        if os.path.exists(Config.LOGO_PATH):
            try:
                self.image(Config.LOGO_PATH, 10, 8, 33)
            except Exception:
                pass 
                
        self.set_font('Arial', 'B', 15)
        self.cell(80)  # Mover a la derecha
        self.cell(30, 10, 'Reporte Hidroclimático', 0, 0, 'C')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}/{{nb}}', 0, 0, 'C')

    def chapter_title(self, label):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, label, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 5, body)
        self.ln()

# --- FUNCIÓN PRINCIPAL DE GENERACIÓN DE REPORTE ---
def generate_pdf_report(df_long, gdf_stations, analysis_results, **kwargs):
    """
    Genera un reporte PDF descargable con los análisis seleccionados.
    """
    try:
        pdf = PDFReport()
        pdf.alias_nb_pages()
        pdf.add_page()
        
        # Título del Reporte
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, Config.APP_TITLE, 0, 1, 'C')
        pdf.ln(10)
        
        # Sección 1: Resumen General
        pdf.chapter_title('1. Resumen General')
        
        n_estaciones = df_long[Config.STATION_NAME_COL].nunique()
        
        # Manejo seguro de fechas
        if not df_long.empty:
            fecha_min = df_long[Config.DATE_COL].min().strftime('%Y-%m-%d')
            fecha_max = df_long[Config.DATE_COL].max().strftime('%Y-%m-%d')
            total_precip = df_long[Config.PRECIPITATION_COL].sum()
        else:
            fecha_min = "N/A"
            fecha_max = "N/A"
            total_precip = 0
        
        resumen_texto = (
            f"Este reporte presenta un análisis de los datos hidrometeorológicos para el período "
            f"comprendido entre {fecha_min} y {fecha_max}.\n\n"
            f"Se analizaron datos de {n_estaciones} estaciones de monitoreo. "
            f"La precipitación total acumulada registrada en todas las estaciones para el período fue de {total_precip:,.0f} mm."
        )
        pdf.chapter_body(resumen_texto)
        
        # Sección 2: Estadísticas Básicas
        pdf.chapter_title('2. Estadísticas Descriptivas')
        
        if not df_long.empty:
            # Calcular promedios por estación
            df_avg = df_long.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean().reset_index()
            top_3_lluviosas = df_avg.sort_values(Config.PRECIPITATION_COL, ascending=False).head(3)
            
            stats_texto = "Las estaciones con mayor precipitación promedio mensual fueron:\n"
            for idx, row in top_3_lluviosas.iterrows():
                stats_texto += f"- {row[Config.STATION_NAME_COL]}: {row[Config.PRECIPITATION_COL]:.1f} mm/mes\n"
        else:
            stats_texto = "No hay datos suficientes para generar estadísticas."
            
        pdf.chapter_body(stats_texto)

        # Guardar en temporal y retornar bytes
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            pdf.output(tmp_file.name)
            with open(tmp_file.name, "rb") as f:
                pdf_bytes = f.read()
        
        return pdf_bytes

    except Exception as e:
        st.error(f"Error al generar el reporte PDF: {e}")
        return None
