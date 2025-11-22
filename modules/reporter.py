import os
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from modules.config import Config
import streamlit as st
import io

class PDFReport(FPDF):
    def header(self):
        # Logo
        if os.path.exists(Config.LOGO_PATH):
            try:
                # Ajustar posición (x, y, w)
                self.image(Config.LOGO_PATH, 10, 8, 30)
            except: pass
            
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Reporte Hidroclimático Ejecutivo', 0, 0, 'R')
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Generado por SIHCLI-POTER - Página {self.page_no()}/{{nb}}', 0, 0, 'C')

    def chapter_title(self, label):
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(230, 240, 255) # Azul claro
        self.cell(0, 8, label, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 5, body)
        self.ln()
        
    def add_image_from_bytes(self, img_bytes, w=180, h=100, title=""):
        if img_bytes:
            try:
                # Guardar bytes en archivo temporal porque FPDF lo requiere
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                    tmp_img.write(img_bytes)
                    tmp_path = tmp_img.name
                
                if title:
                    self.set_font('Arial', 'B', 10)
                    self.cell(0, 6, title, 0, 1, 'C')
                
                # Centrar imagen
                self.image(tmp_path, x=(210-w)/2, w=w, h=h)
                self.ln(5)
                os.remove(tmp_path)
            except:
                self.cell(0, 10, "[Error al insertar imagen]", 0, 1)

def create_static_map(gdf_stations):
    """Genera un mapa estático simple con Matplotlib para el PDF."""
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Pintar estaciones
        gdf_stations.plot(ax=ax, color='blue', markersize=50, alpha=0.7, edgecolor='k')
        
        # Etiquetas
        for idx, row in gdf_stations.iterrows():
            ax.annotate(text=row[Config.STATION_NAME_COL][:10], xy=(row.geometry.x, row.geometry.y), 
                        xytext=(3, 3), textcoords="offset points", fontsize=8)
        
        ax.set_title("Ubicación de Estaciones Seleccionadas")
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Guardar en bytes
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=100, bbox_inches='tight')
        img_buf.seek(0)
        plt.close(fig)
        return img_buf.read()
    except:
        return None

def generate_pdf_report(df_long, gdf_stations, analysis_results, **kwargs):
    """Genera el PDF completo."""
    try:
        pdf = PDFReport()
        pdf.alias_nb_pages()
        pdf.add_page()

        # --- 1. RESUMEN EJECUTIVO ---
        pdf.chapter_title('1. Resumen Ejecutivo')
        
        n_est = analysis_results.get("n_estaciones", 0)
        rango = analysis_results.get("rango", "N/A")
        
        # Calcular totales rápidos
        ppt_total = df_long[Config.PRECIPITATION_COL].sum()
        ppt_prom = df_long[Config.PRECIPITATION_COL].mean()
        
        intro = (f"El presente informe detalla el análisis hidroclimático para el periodo {rango}. "
                 f"Se han procesado datos de {n_est} estaciones de monitoreo.\n\n"
                 f"Resumen Estadístico:\n"
                 f"- Precipitación Promedio Mensual del sistema: {ppt_prom:.1f} mm\n"
                 f"- Precipitación Total Acumulada (histórica): {ppt_total:,.0f} mm")
        pdf.chapter_body(intro)

        # --- 2. MAPA DE LOCALIZACIÓN ---
        pdf.chapter_title('2. Localización Espacial')
        map_bytes = create_static_map(gdf_stations)
        if map_bytes:
            pdf.add_image_from_bytes(map_bytes, w=140, h=100, title="Mapa de Estaciones")
        else:
            pdf.chapter_body("No se pudo generar el mapa estático.")

        # --- 3. ANÁLISIS GRÁFICO (Desde Session State) ---
        pdf.add_page()
        pdf.chapter_title('3. Análisis Gráfico')
        
        # Recuperar figuras guardadas
        fig_anual = st.session_state.get('report_fig_anual')
        fig_mensual = st.session_state.get('report_fig_mensual')
        fig_ciclo = st.session_state.get('report_fig_ciclo')

        # Convertir Plotly a Imagen (Requiere Kaleido instalado en el servidor)
        # Si falla, pondrá un mensaje de error pero generará el PDF
        if fig_anual:
            try:
                img_bytes = fig_anual.to_image(format="png", width=800, height=400, scale=1.5)
                pdf.add_image_from_bytes(img_bytes, w=170, h=80, title="Serie Anual")
            except:
                pdf.chapter_body("[Gráfico Anual no disponible: Falta librería de exportación]")

        if fig_ciclo:
            try:
                img_bytes = fig_ciclo.to_image(format="png", width=800, height=400, scale=1.5)
                pdf.add_image_from_bytes(img_bytes, w=170, h=80, title="Ciclo Anual Promedio")
            except: pass

        if fig_mensual:
            pdf.add_page()
            try:
                img_bytes = fig_mensual.to_image(format="png", width=800, height=400, scale=1.5)
                pdf.add_image_from_bytes(img_bytes, w=170, h=80, title="Serie Mensual Detallada")
            except: pass

        # --- 4. TABLA RESUMEN ---
        pdf.chapter_title('4. Resumen por Estación')
        
        # Crear tabla simple
        pdf.set_font('Arial', 'B', 9)
        # Encabezados
        pdf.cell(80, 8, 'Estación', 1)
        pdf.cell(30, 8, 'Altitud', 1)
        pdf.cell(40, 8, 'Ppt Media (mm)', 1)
        pdf.ln()
        
        # Datos
        pdf.set_font('Arial', '', 9)
        stats = df_long.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean().reset_index()
        
        # Unir altitud
        stats = stats.merge(gdf_stations[[Config.STATION_NAME_COL, Config.ALTITUDE_COL]], on=Config.STATION_NAME_COL, how='left')
        
        for _, row in stats.iterrows():
            name = str(row[Config.STATION_NAME_COL])[:35] # Recortar nombre largo
            alt = str(row.get(Config.ALTITUDE_COL, 'N/A'))
            val = f"{row[Config.PRECIPITATION_COL]:.1f}"
            
            pdf.cell(80, 7, name, 1)
            pdf.cell(30, 7, alt, 1)
            pdf.cell(40, 7, val, 1)
            pdf.ln()

        # Generar
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            pdf.output(tmp_file.name)
            tmp_path = tmp_file.name
        
        with open(tmp_path, "rb") as f:
            pdf_bytes = f.read()
        os.remove(tmp_path)
            
        return pdf_bytes

    except Exception as e:
        print(f"Error PDF: {e}")
        return None
