import os
import tempfile
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Backend no interactivo para servidores
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
from modules.config import Config
import streamlit as st
import io
import geopandas as gpd

class PDFReport(FPDF):
    def header(self):
        if self.page_no() > 1:
            if os.path.exists(Config.LOGO_PATH):
                try: self.image(Config.LOGO_PATH, 10, 8, 25)
                except: pass
            self.set_font('Arial', 'B', 10)
            self.cell(0, 5, Config.APP_TITLE[:50]+"...", 0, 1, 'R')
            self.set_font('Arial', 'I', 8)
            self.cell(0, 5, f'Fecha: {datetime.now().strftime("%Y-%m-%d")}', 0, 1, 'R')
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Página {self.page_no()}/{{nb}}', 0, 0, 'C')

    def print_chapter_title(self, num, label):
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(220, 230, 250)
        self.set_text_color(0)
        title_text = f"{num}. {label}" if str(num).isdigit() else label
        self.cell(0, 10, title_text, 0, 1, 'L', 1)
        self.ln(4)

    def print_section_body(self, body):
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
                    self.set_font('Arial', 'B', 10)
                    self.cell(0, 8, title, 0, 1, 'C')
                
                x = (210 - w) / 2
                self.image(tmp_path, x=x, w=w, h=h)
                self.ln(5)
                os.remove(tmp_path)
            except: pass

    def add_matplotlib_figure(self, fig, title=""):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                fig.savefig(tmp.name, format='png', dpi=150, bbox_inches='tight')
                tmp_path = tmp.name
            
            if self.get_y() + 100 > 270: self.add_page()
            
            if title:
                self.set_font('Arial', 'B', 10)
                self.cell(0, 8, title, 0, 1, 'C')
            
            self.image(tmp_path, x=20, w=170)
            self.ln(5)
            os.remove(tmp_path)
            plt.close(fig)
        except Exception as e:
            self.print_section_body(f"[Error gráfico: {e}]")

# --- FUNCIÓN DE MAPA ESTÁTICO (DEFINIDA AQUÍ) ---
def create_context_map_static(gdf_stations, gdf_municipios=None, gdf_subcuencas=None):
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Capas Base
        if gdf_municipios is not None and not gdf_municipios.empty:
            gdf_municipios.plot(ax=ax, color='none', edgecolor='gray', linewidth=0.5, alpha=0.5)
        
        if gdf_subcuencas is not None and not gdf_subcuencas.empty:
            gdf_subcuencas.plot(ax=ax, color='#e6f2ff', edgecolor='blue', alpha=0.2, linewidth=1)

        # Estaciones
        if gdf_stations is not None and not gdf_stations.empty:
            gdf_stations.plot(ax=ax, color='red', markersize=40, edgecolor='white', zorder=3)
            
            # Etiquetas (si son pocas)
            if len(gdf_stations) < 40:
                for x, y, label in zip(gdf_stations.geometry.x, gdf_stations.geometry.y, gdf_stations[Config.STATION_NAME_COL]):
                    ax.annotate(str(label)[:10], xy=(x, y), xytext=(3, 3), textcoords="offset points", fontsize=6)

        ax.set_title("Mapa de Localización")
        ax.set_axis_off()
        return fig
    except Exception as e:
        print(f"Error mapa estático: {e}")
        return None

# --- GENERADOR PRINCIPAL ---
def generate_pdf_report(df_long, gdf_stations, analysis_results, **kwargs):
    try:
        pdf = PDFReport()
        pdf.alias_nb_pages()
        
        # PORTADA
        pdf.add_page()
        pdf.ln(60)
        if os.path.exists(Config.LOGO_PATH):
            try: pdf.image(Config.LOGO_PATH, x=75, w=60)
            except: pass
        pdf.ln(20)
        pdf.set_font('Arial', 'B', 24)
        pdf.cell(0, 10, "REPORTE TÉCNICO", 0, 1, 'C')
        pdf.set_font('Arial', '', 16)
        pdf.cell(0, 10, "Análisis Hidroclimático Regional", 0, 1, 'C')
        pdf.ln(20)
        pdf.set_font('Arial', 'I', 12)
        pdf.cell(0, 8, f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1, 'C')
        
        # 1. RESUMEN
        pdf.add_page()
        pdf.print_chapter_title(1, 'Resumen Ejecutivo')
        
        n_est = len(gdf_stations) if gdf_stations is not None else 0
        ppt_prom = df_long[Config.PRECIPITATION_COL].mean() if not df_long.empty else 0
        
        pdf.print_section_body(
            f"Informe generado automáticamente por la plataforma SIHCLI-POTER. "
            f"Se analizaron {n_est} estaciones con un promedio mensual histórico de {ppt_prom:.1f} mm."
        )

        # 2. MAPA (LLAMADA CORREGIDA)
        pdf.print_chapter_title(2, 'Contexto Espacial')
        gdf_munis = kwargs.get('gdf_municipios')
        gdf_subc = kwargs.get('gdf_subcuencas')
        
        fig_map = create_context_map_static(gdf_stations, gdf_munis, gdf_subc)
        if fig_map:
            pdf.add_matplotlib_figure(fig_map, "Ubicación de Estaciones")
        else:
            pdf.print_section_body("Mapa no disponible.")

        # 3. GRÁFICOS
        pdf.add_page()
        pdf.print_chapter_title(3, 'Análisis Gráfico')
        
        keys = [
            ('report_fig_anual', 'Serie Histórica Anual'),
            ('report_fig_mensual', 'Régimen Mensual'),
            ('report_fig_ciclo', 'Ciclo Anual'),
            ('report_fig_dist', 'Distribución Estadística')
        ]
        
        count = 0
        for k, title in keys:
            if k in st.session_state and st.session_state[k]:
                try:
                    img = st.session_state[k].to_image(format="png", width=1000, height=500, scale=2)
                    pdf.add_plot_image(img, title, h=90)
                    count += 1
                    if count % 2 == 0: pdf.add_page()
                except: pass
        
        if count == 0:
            pdf.print_section_body("Nota: Visualice los gráficos en la app para incluirlos aquí.")

        # 4. TABLA
        pdf.add_page()
        pdf.print_chapter_title(4, 'Estadísticas por Estación')
        
        if not df_long.empty:
            stats = df_long.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].agg(['mean', 'max']).reset_index()
            
            pdf.set_font('Arial', 'B', 9)
            pdf.set_fill_color(240, 240, 240)
            col_w = [95, 30, 30]
            
            pdf.cell(col_w[0], 8, "Estación", 1, 0, 'C', 1)
            pdf.cell(col_w[1], 8, "Media", 1, 0, 'C', 1)
            pdf.cell(col_w[2], 8, "Máx", 1, 1, 'C', 1)
            
            pdf.set_font('Arial', '', 9)
            for _, row in stats.iterrows():
                if pdf.get_y() > 270:
                    pdf.add_page()
                    pdf.set_font('Arial', 'B', 9)
                    pdf.cell(col_w[0], 8, "Estación (cont.)", 1, 0, 'C', 1)
                    pdf.ln()
                    pdf.set_font('Arial', '', 9)

                pdf.cell(col_w[0], 6, str(row[Config.STATION_NAME_COL])[:50], 1)
                pdf.cell(col_w[1], 6, f"{row['mean']:.1f}", 1, 0, 'R')
                pdf.cell(col_w[2], 6, f"{row['max']:.1f}", 1, 1, 'R')

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf.output(tmp.name)
            return open(tmp.name, "rb").read()

    except Exception as e:
        st.error(f"Error reporte: {e}")
        return None
