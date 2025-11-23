import os
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
from modules.config import Config
import streamlit as st
import io

class PDFReport(FPDF):
    def header(self):
        if self.page_no() > 1:
            if os.path.exists(Config.LOGO_PATH):
                try: self.image(Config.LOGO_PATH, 10, 8, 25)
                except: pass
            self.set_font('Arial', 'B', 10)
            self.cell(0, 5, Config.APP_TITLE[:60]+"...", 0, 1, 'R')
            self.set_font('Arial', 'I', 8)
            self.cell(0, 5, f'Fecha: {datetime.now().strftime("%Y-%m-%d")}', 0, 1, 'R')
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Página {self.page_no()}/{{nb}}', 0, 0, 'C')

    def chapter_title(self, label):
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(220, 230, 250)
        self.set_text_color(0)
        self.cell(0, 10, label, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 5, body)
        self.ln()

    def add_plot_image(self, img_bytes, title="", w=170, h=90):
        if img_bytes:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    tmp.write(img_bytes)
                    tmp_path = tmp.name
                
                if self.get_y() + h > 260: self.add_page()
                
                if title:
                    self.set_font('Arial', 'B', 11)
                    self.cell(0, 8, title, 0, 1, 'C')
                
                x = (210 - w) / 2
                self.image(tmp_path, x=x, w=w, h=h)
                self.ln(5)
                os.remove(tmp_path)
            except: pass

def create_static_map(gdf_stations, gdf_subcuencas=None):
    """Genera mapa estático con Matplotlib."""
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Capas
        if gdf_subcuencas is not None and not gdf_subcuencas.empty:
            gdf_subcuencas.plot(ax=ax, color='#e6f2ff', edgecolor='blue', alpha=0.5, zorder=1)

        if gdf_stations is not None and not gdf_stations.empty:
            gdf_stations.plot(ax=ax, color='red', markersize=40, edgecolor='white', zorder=2)
            # Etiquetas simples
            if len(gdf_stations) < 25:
                for x, y, label in zip(gdf_stations.geometry.x, gdf_stations.geometry.y, gdf_stations[Config.STATION_NAME_COL]):
                    ax.annotate(label[:10], xy=(x, y), xytext=(3, 3), textcoords="offset points", fontsize=7)

        ax.set_title("Localización de Estaciones", fontsize=12)
        ax.set_axis_off()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf.read()
    except: return None

def generate_pdf_report(df_long, gdf_stations, analysis_results, **kwargs):
    try:
        pdf = PDFReport()
        pdf.alias_nb_pages()
        
        # PORTADA
        pdf.add_page()
        pdf.ln(60)
        if os.path.exists(Config.LOGO_PATH): pdf.image(Config.LOGO_PATH, x=75, w=60)
        pdf.ln(20)
        pdf.set_font('Arial', 'B', 24)
        pdf.cell(0, 10, "REPORTE TÉCNICO", 0, 1, 'C')
        pdf.set_font('Arial', '', 16)
        pdf.cell(0, 10, "Análisis Hidroclimático y Pronósticos", 0, 1, 'C')
        pdf.ln(20)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 8, f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1, 'C')
        
        # 1. RESUMEN
        pdf.add_page()
        pdf.print_chapter_title("1. Resumen Ejecutivo")
        n = len(gdf_stations) if gdf_stations is not None else 0
        pdf.chapter_body(f"Análisis realizado sobre {n} estaciones. Este documento consolida la estadística descriptiva, tendencias históricas y pronósticos generados por el sistema.")

        # 2. MAPA
        pdf.print_chapter_title("2. Contexto Espacial")
        subc = kwargs.get('gdf_subcuencas')
        map_img = create_context_map_static(gdf_stations, gdf_subcuencas=subc)
        if map_img: pdf.add_plot_image(map_img, "Mapa de Estaciones", h=120)

        # 3. GRÁFICOS CAPTURADOS (Session State)
        pdf.add_page()
        pdf.print_chapter_title("3. Análisis Visual")
        
        # Lista de claves de gráficos que guardamos en visualizer.py
        keys = [
            ('report_fig_anual', 'Serie Histórica Anual'),
            ('report_fig_mensual', 'Régimen Mensual'),
            ('report_fig_ciclo', 'Ciclo Anual'),
            ('report_fig_dist', 'Distribución Estadística')
        ]
        
        for k, title in keys:
            if k in st.session_state and st.session_state[k]:
                try:
                    # Intentar convertir Plotly a imagen (requiere Kaleido)
                    img = st.session_state[k].to_image(format="png", width=1000, height=500, scale=2)
                    pdf.add_plot_image(img, title, h=90)
                except: pass

        # 4. ESTADÍSTICAS
        pdf.add_page()
        pdf.print_chapter_title("4. Estadísticas Clave")
        if not df_long.empty:
            stats = df_long.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].agg(['mean', 'max']).reset_index()
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(100, 8, "Estación", 1); pdf.cell(40, 8, "Promedio", 1); pdf.cell(40, 8, "Máximo", 1); pdf.ln()
            pdf.set_font('Arial', '', 10)
            for _, r in stats.iterrows():
                pdf.cell(100, 7, str(r[Config.STATION_NAME_COL])[:45], 1)
                pdf.cell(40, 7, f"{r['mean']:.1f}", 1, 0, 'R')
                pdf.cell(40, 7, f"{r['max']:.1f}", 1, 0, 'R')
                pdf.ln()

        # Output
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf.output(tmp.name)
            return open(tmp.name, "rb").read()

    except Exception as e:
        st.error(f"Error reporte: {e}")
        return None
