import os
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from fpdf import FPDF
from datetime import datetime
from modules.config import Config
import streamlit as st
import geopandas as gpd

class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        self.WIDTH = 210
        self.HEIGHT = 297
        
    def header(self):
        # Solo poner encabezado si no es la portada (página 1)
        if self.page_no() > 1:
            if os.path.exists(Config.LOGO_PATH):
                try: self.image(Config.LOGO_PATH, 10, 8, 25)
                except: pass
            
            self.set_font('Arial', 'B', 10)
            self.cell(0, 5, Config.APP_TITLE, 0, 1, 'R')
            self.set_font('Arial', 'I', 8)
            self.cell(0, 5, f'Generado el: {datetime.now().strftime("%d/%m/%Y")}', 0, 1, 'R')
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')

    def print_chapter_title(self, num, label):
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(200, 220, 255)
        self.set_text_color(0)
        self.cell(0, 10, f"{num}. {label}", 0, 1, 'L', 1)
        self.ln(4)

    def print_section_body(self, text):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 5, text)
        self.ln()

    def add_plot_image(self, fig_bytes, title="Gráfico", w=170, h=90):
        """Inserta una imagen desde bytes (Plotly/Matplotlib)"""
        if fig_bytes:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    tmp.write(fig_bytes)
                    tmp_path = tmp.name
                
                # Verificar espacio
                if self.get_y() + h > 270: self.add_page()
                
                self.set_font('Arial', 'B', 10)
                self.cell(0, 6, title, 0, 1, 'C')
                self.image(tmp_path, x=(self.WIDTH - w)/2, w=w, h=h)
                self.ln(5)
                os.remove(tmp_path)
            except Exception as e:
                self.print_section_body(f"[Error insertando imagen: {e}]")

def create_context_map_static(gdf_stations, gdf_municipios=None, gdf_subcuencas=None):
    """Genera un mapa estático Matplotlib limpio con contexto."""
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 1. Capas Base (Municipios/Cuencas)
        if gdf_municipios is not None and not gdf_municipios.empty:
            gdf_municipios.plot(ax=ax, color='none', edgecolor='gray', linewidth=0.5, alpha=0.5, zorder=1)
        
        if gdf_subcuencas is not None and not gdf_subcuencas.empty:
            gdf_subcuencas.plot(ax=ax, color='lightblue', edgecolor='blue', alpha=0.2, linewidth=1, zorder=2)

        # 2. Estaciones
        if gdf_stations is not None and not gdf_stations.empty:
            # Ajustar límites
            bounds = gdf_stations.total_bounds
            margin = 0.1
            ax.set_xlim([bounds[0]-margin, bounds[2]+margin])
            ax.set_ylim([bounds[1]-margin, bounds[3]+margin])
            
            gdf_stations.plot(ax=ax, color='red', markersize=60, edgecolor='black', zorder=3, label='Estaciones')
            
            # Etiquetas (solo si hay pocas para no saturar)
            if len(gdf_stations) < 30:
                for x, y, label in zip(gdf_stations.geometry.x, gdf_stations.geometry.y, gdf_stations[Config.STATION_NAME_COL]):
                    ax.annotate(label[:15], xy=(x, y), xytext=(3, 3), textcoords="offset points", fontsize=7, clip_on=True)

        ax.set_title("Localización Espacial de Estaciones", fontsize=14)
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Guardar a bytes
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf.read()
    except Exception as e:
        print(f"Error mapa estático: {e}")
        return None

def generate_pdf_report(df_long, gdf_stations, analysis_results, **kwargs):
    """Generador Maestro de Reportes."""
    try:
        pdf = PDFReport()
        pdf.alias_nb_pages()
        
        # --- PÁGINA 1: PORTADA ---
        pdf.add_page()
        pdf.ln(60)
        if os.path.exists(Config.LOGO_PATH):
            pdf.image(Config.LOGO_PATH, x=75, w=60)
        pdf.ln(20)
        pdf.set_font('Arial', 'B', 24)
        pdf.cell(0, 10, "INFORME HIDROCLIMÁTICO", 0, 1, 'C')
        pdf.set_font('Arial', '', 16)
        pdf.cell(0, 10, "Análisis de Precipitación y Clima", 0, 1, 'C')
        pdf.ln(20)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f"Fecha de Generación: {datetime.now().strftime('%Y-%m-%d')}", 0, 1, 'C')
        pdf.cell(0, 10, f"Período Analizado: {analysis_results.get('rango_fechas')}", 0, 1, 'C')
        
        # --- PÁGINA 2: RESUMEN Y MAPA ---
        pdf.add_page()
        pdf.print_chapter_title(1, 'Resumen Ejecutivo')
        
        n_est = len(gdf_stations) if gdf_stations is not None else 0
        ppt_total = df_long[Config.PRECIPITATION_COL].sum() if not df_long.empty else 0
        
        intro = (f"Este documento presenta el análisis técnico de {n_est} estaciones de monitoreo. "
                 f"La base de datos procesada contiene un acumulado histórico de {ppt_total:,.0f} mm de precipitación. "
                 "A continuación se detallan los patrones espaciales y temporales identificados.")
        pdf.print_section_body(intro)
        
        pdf.print_chapter_title(2, 'Contexto Espacial')
        
        # Generar Mapa Mejorado (con municipios/cuencas si están disponibles en kwargs)
        gdf_munis = kwargs.get('gdf_municipios')
        gdf_subc = kwargs.get('gdf_subcuencas')
        map_img = create_context_map_static(gdf_stations, gdf_munis, gdf_subc)
        
        if map_img:
            pdf.add_plot_image(map_img, title="Mapa de Localización de Estaciones", w=160, h=120)
        else:
            pdf.print_section_body("Mapa no disponible.")

        # --- PÁGINA 3: GRÁFICOS DE ANÁLISIS ---
        # Intentamos recuperar los gráficos guardados en session_state
        # (Esto requiere que el usuario haya visitado la pestaña de gráficos primero, si no, estarán vacíos)
        # Una mejora futura sería regenerarlos aquí si no existen.
        
        pdf.add_page()
        pdf.print_chapter_title(3, 'Análisis Gráfico')
        
        # Recuperar figuras (Plotly -> Imagen estática)
        # Nota: Plotly requiere 'kaleido' instalado en el servidor para .to_image()
        figs_to_print = [
            ('report_fig_anual', 'Serie Histórica Anual'),
            ('report_fig_mensual', 'Régimen Mensual'),
            ('report_fig_ciclo', 'Ciclo Anual Promedio (Estacionalidad)')
        ]
        
        count = 0
        for key, title in figs_to_print:
            if key in st.session_state and st.session_state[key]:
                try:
                    fig = st.session_state[key]
                    # Convertir a bytes PNG
                    img_bytes = fig.to_image(format="png", width=1000, height=500, scale=2)
                    pdf.add_plot_image(img_bytes, title=title, w=190, h=90)
                    count += 1
                    if count % 2 == 0: pdf.add_page() # 2 gráficos por página
                except Exception as e:
                    pdf.print_section_body(f"No se pudo renderizar el gráfico {title}. (Requiere Kaleido instalado)")
        
        if count == 0:
            pdf.print_section_body("Nota: Para incluir gráficos, por favor visualícelos primero en la pestaña 'Gráficos' de la aplicación.")

        # --- PÁGINA 4: TABLA DE DATOS ---
        pdf.add_page()
        pdf.print_chapter_title(4, 'Estadísticas por Estación')
        
        if not df_long.empty:
            # Resumen
            stats = df_long.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].agg(['mean', 'sum', 'max']).reset_index()
            
            # Configuración de tabla PDF
            pdf.set_font('Arial', 'B', 9)
            col_widths = [90, 30, 30, 30] # Anchos
            headers = ['Estación', 'Prom. Mes', 'Total Hist.', 'Máx Mes']
            
            # Cabecera
            for i, h in enumerate(headers):
                pdf.cell(col_widths[i], 8, h, 1, 0, 'C', 1)
            pdf.ln()
            
            # Filas
            pdf.set_font('Arial', '', 9)
            for _, row in stats.iterrows():
                name = str(row[Config.STATION_NAME_COL])[:40]
                vals = [
                    f"{row['mean']:.1f}",
                    f"{row['sum']:,.0f}",
                    f"{row['max']:.1f}"
                ]
                
                pdf.cell(col_widths[0], 7, name, 1)
                for i, v in enumerate(vals):
                    pdf.cell(col_widths[i+1], 7, v, 1, 0, 'R')
                pdf.ln()

        # Generar Bytes
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            pdf.output(tmp_file.name)
            tmp_path = tmp_file.name
        
        with open(tmp_path, "rb") as f:
            pdf_bytes = f.read()
        os.remove(tmp_path)
        
        return pdf_bytes

    except Exception as e:
        st.error(f"Error crítico generando reporte: {e}")
        return None
