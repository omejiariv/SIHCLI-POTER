import os
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
from modules.config import Config
import io

class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        self.WIDTH = 210
        self.HEIGHT = 297
        
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

    def add_matplotlib_figure(self, fig, title=""):
        """Convierte una figura de Matplotlib a imagen y la pega en el PDF"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                fig.savefig(tmp.name, format='png', dpi=150, bbox_inches='tight')
                tmp_path = tmp.name
            
            if self.get_y() + 100 > 270: self.add_page()
            
            if title:
                self.set_font('Arial', 'B', 10)
                self.cell(0, 8, title, 0, 1, 'C')
            
            # Centrar (A4 width 210)
            img_w = 170
            x = (210 - img_w) / 2
            self.image(tmp_path, x=x, w=img_w)
            self.ln(5)
            os.remove(tmp_path)
            plt.close(fig)
        except Exception as e:
            self.chapter_body(f"[Error gráfico: {e}]")

def create_context_map_static(gdf_stations, gdf_municipios=None, gdf_subcuencas=None):
    """Genera mapa profesional con capas."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Capas Base
    if gdf_municipios is not None and not gdf_municipios.empty:
        gdf_municipios.plot(ax=ax, color='none', edgecolor='gray', linewidth=0.5, alpha=0.5, zorder=1)
    if gdf_subcuencas is not None and not gdf_subcuencas.empty:
        gdf_subcuencas.plot(ax=ax, color='#e6f2ff', edgecolor='blue', linewidth=0.8, alpha=0.5, zorder=2)

    # Estaciones
    if gdf_stations is not None and not gdf_stations.empty:
        gdf_stations.plot(ax=ax, color='red', markersize=40, edgecolor='white', linewidth=0.5, zorder=3, label='Estaciones')
        
        # Etiquetas inteligentes (solo si son pocas)
        if len(gdf_stations) < 30:
            for x, y, label in zip(gdf_stations.geometry.x, gdf_stations.geometry.y, gdf_stations[Config.STATION_NAME_COL]):
                ax.annotate(label[:15], xy=(x, y), xytext=(3, 3), textcoords="offset points", fontsize=6, alpha=0.8)

    ax.set_title("Mapa de Localización de Estaciones", fontsize=12)
    ax.set_axis_off() # Mapa limpio sin ejes lat/lon feos
    return fig

def generate_pdf_report(df_long, gdf_stations, analysis_results, **kwargs):
    try:
        pdf = PDFReport()
        pdf.alias_nb_pages()
        
        # --- PORTADA ---
        pdf.add_page()
        pdf.ln(60)
        if os.path.exists(Config.LOGO_PATH):
            pdf.image(Config.LOGO_PATH, x=75, w=60)
        pdf.ln(20)
        pdf.set_font('Arial', 'B', 24)
        pdf.cell(0, 10, "REPORTE TÉCNICO", 0, 1, 'C')
        pdf.set_font('Arial', '', 16)
        pdf.cell(0, 10, "Análisis Hidroclimático Regional", 0, 1, 'C')
        pdf.ln(20)
        pdf.set_font('Arial', 'I', 12)
        pdf.cell(0, 8, f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1, 'C')
        pdf.cell(0, 8, f"Estaciones Analizadas: {len(gdf_stations)}", 0, 1, 'C')

        # --- 1. RESUMEN ---
        pdf.add_page()
        pdf.print_chapter_title(1, 'Resumen Ejecutivo')
        
        ppt_total = df_long[Config.PRECIPITATION_COL].sum() if not df_long.empty else 0
        ppt_prom = df_long[Config.PRECIPITATION_COL].mean() if not df_long.empty else 0
        
        pdf.chapter_body(
            f"El presente informe consolida el análisis hidrometeorológico realizado mediante la plataforma SIHCLI-POTER. "
            f"Se evaluó un registro histórico consolidado con un promedio mensual global de {ppt_prom:.1f} mm. "
            f"Este documento sirve como soporte técnico para la toma de decisiones en la gestión del recurso hídrico."
        )

        # --- 2. MAPA ---
        pdf.print_chapter_title(2, 'Contexto Espacial')
        gdf_munis = kwargs.get('gdf_municipios')
        gdf_subc = kwargs.get('gdf_subcuencas')
        fig_map = create_context_map_static(gdf_stations, gdf_munis, gdf_subc)
        pdf.add_matplotlib_figure(fig_map, "Ubicación de Estaciones y Cuencas")

        # --- 3. GRÁFICOS ESTADÍSTICOS (REGENERADOS CON MATPLOTLIB) ---
        pdf.add_page()
        pdf.print_chapter_title(3, 'Análisis de Precipitación')
        
        if not df_long.empty:
            # 3.1 Serie Anual
            df_anual = df_long.groupby(Config.YEAR_COL)[Config.PRECIPITATION_COL].sum()
            fig_anual, ax_an = plt.subplots(figsize=(10, 4))
            df_anual.plot(kind='line', ax=ax_an, marker='o', color='#1f77b4')
            ax_an.set_title("Precipitación Total Anual (Promedio Regional)")
            ax_an.set_ylabel("mm")
            ax_an.grid(True, alpha=0.3)
            pdf.add_matplotlib_figure(fig_anual, "Evolución Anual")

            # 3.2 Ciclo Mensual
            df_mensual = df_long.groupby(Config.MONTH_COL)[Config.PRECIPITATION_COL].mean()
            fig_ciclo, ax_ci = plt.subplots(figsize=(10, 4))
            df_mensual.plot(kind='bar', ax=ax_ci, color='#2ca02c', alpha=0.7)
            ax_ci.set_title("Régimen de Lluvia (Ciclo Anual Promedio)")
            ax_ci.set_ylabel("mm/mes")
            ax_ci.set_xticklabels(['E','F','M','A','M','J','J','A','S','O','N','D'], rotation=0)
            pdf.add_matplotlib_figure(fig_ciclo, "Ciclo Estacional")

        # --- 4. TABLA DE DATOS ---
        pdf.add_page()
        pdf.print_chapter_title(4, 'Estadísticas por Estación')
        
        if not df_long.empty:
            stats = df_long.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].agg(['mean', 'max', 'min']).reset_index()
            
            # Configurar tabla
            pdf.set_font('Arial', 'B', 9)
            pdf.set_fill_color(240, 240, 240)
            col_w = [95, 30, 30, 30]
            
            pdf.cell(col_w[0], 8, "Estación", 1, 0, 'C', 1)
            pdf.cell(col_w[1], 8, "Media", 1, 0, 'C', 1)
            pdf.cell(col_w[2], 8, "Máx", 1, 0, 'C', 1)
            pdf.cell(col_w[3], 8, "Mín", 1, 1, 'C', 1)
            
            pdf.set_font('Arial', '', 9)
            for _, row in stats.iterrows():
                # Verificar salto de página
                if pdf.get_y() > 270:
                    pdf.add_page()
                    pdf.set_font('Arial', 'B', 9)
                    pdf.cell(col_w[0], 8, "Estación (cont.)", 1, 0, 'C', 1)
                    pdf.ln()
                    pdf.set_font('Arial', '', 9)

                name = str(row[Config.STATION_NAME_COL])[:50]
                pdf.cell(col_w[0], 6, name, 1)
                pdf.cell(col_w[1], 6, f"{row['mean']:.1f}", 1, 0, 'R')
                pdf.cell(col_w[2], 6, f"{row['max']:.1f}", 1, 0, 'R')
                pdf.cell(col_w[3], 6, f"{row['min']:.1f}", 1, 1, 'R')

        # Guardar
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            pdf.output(tmp_file.name)
            tmp_path = tmp_file.name
        
        with open(tmp_path, "rb") as f:
            pdf_bytes = f.read()
        os.remove(tmp_path)
        
        return pdf_bytes

    except Exception as e:
        print(f"Error reporte: {e}")
        return None
