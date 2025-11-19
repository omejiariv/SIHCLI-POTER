import os
import tempfile
import pandas as pd
from fpdf import FPDF
from modules.config import Config

class PDFReport(FPDF):
    def header(self):
        # Logo
        if hasattr(Config, 'LOGO_PATH') and os.path.exists(Config.LOGO_PATH):
            try:
                # Ajustar posición y tamaño del logo
                self.image(Config.LOGO_PATH, 10, 8, 33)
            except: pass
            
        self.set_font('Arial', 'B', 15)
        self.cell(80) # Mover a la derecha
        self.cell(30, 10, 'Reporte Hidroclimático', 0, 0, 'C')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}/{{nb}} - Generado por SIHCLI-POTER', 0, 0, 'C')

    def chapter_title(self, label):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, label, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 5, body)
        self.ln()

def generate_pdf_report(df_long, gdf_stations, analysis_results, **kwargs):
    """
    Genera un reporte PDF dinámico basado en los datos filtrados.
    """
    try:
        pdf = PDFReport()
        pdf.alias_nb_pages()
        pdf.add_page()

        # --- 1. Título y Contexto ---
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, "Resumen Ejecutivo", 0, 1, 'L')
        pdf.ln(5)

        # Datos generales
        n_estaciones = analysis_results.get("n_estaciones", 0)
        rango_fechas = analysis_results.get("rango_fechas", "N/A")
        
        # Cálculos rápidos para el texto
        total_ppt = 0
        promedio_ppt = 0
        if df_long is not None and not df_long.empty:
            total_ppt = df_long[Config.PRECIPITATION_COL].sum()
            promedio_ppt = df_long[Config.PRECIPITATION_COL].mean()

        texto_intro = (
            f"El presente reporte resume el análisis hidroclimático realizado para el período {rango_fechas}.\n\n"
            f"Se analizaron un total de {n_estaciones} estaciones de monitoreo seleccionadas. "
            f"Durante este período, se registró una precipitación acumulada total de {total_ppt:,.0f} mm "
            f"en el conjunto de datos, con un promedio mensual regional de {promedio_ppt:.1f} mm."
        )
        pdf.chapter_body(texto_intro)

        # --- 2. Estaciones Analizadas ---
        pdf.chapter_title('Estaciones Incluidas')
        if gdf_stations is not None and not gdf_stations.empty:
            lista_est = ", ".join(gdf_stations[Config.STATION_NAME_COL].unique()[:20])
            if len(gdf_stations) > 20:
                lista_est += f", y {len(gdf_stations)-20} más..."
            pdf.chapter_body(f"Las estaciones consideradas en este análisis son: {lista_est}.")
        else:
            pdf.chapter_body("No se seleccionaron estaciones específicas.")

        # --- 3. Estadísticas Clave ---
        pdf.chapter_title('Estadísticas Destacadas')
        if df_long is not None and not df_long.empty:
            # Estación más lluviosa
            top_station = df_long.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean().idxmax()
            top_val = df_long.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean().max()
            
            # Estación menos lluviosa
            min_station = df_long.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean().idxmin()
            min_val = df_long.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean().min()

            stats_txt = (
                f"- Estación con mayor promedio mensual: {top_station} ({top_val:.1f} mm/mes)\n"
                f"- Estación con menor promedio mensual: {min_station} ({min_val:.1f} mm/mes)\n"
            )
            pdf.chapter_body(stats_txt)
        else:
            pdf.chapter_body("No hay datos suficientes para generar estadísticas.")

        # --- 4. Generación del Archivo ---
        # Usamos un archivo temporal para guardar el PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            pdf.output(tmp_file.name)
            tmp_path = tmp_file.name
        
        # Leemos los bytes para retornarlos a Streamlit
        with open(tmp_path, "rb") as f:
            pdf_bytes = f.read()
            
        return pdf_bytes

    except Exception as e:
        print(f"Error generando PDF: {e}")
        return None

