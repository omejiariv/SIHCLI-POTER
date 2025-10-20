# modules/github_loader.py

import streamlit as st
import pandas as pd
import geopandas as gpd
import requests
import io

@st.cache_data(ttl=3600)
def load_csv_from_url(url):
    """
    Carga un archivo CSV desde una URL a un DataFrame de Pandas,
    probando varias codificaciones comunes para evitar errores.
    """
    try:
        # 1. Descargamos el contenido del archivo una sola vez
        response = requests.get(url)
        response.raise_for_status()
        content_bytes = response.content

        # 2. Lista de codificaciones a probar (latin1 suele funcionar para español)
        encodings_to_try = ['utf-8', 'latin1', 'cp1252']
        df = None
        
        # 3. Bucle para encontrar la codificación correcta
        for encoding in encodings_to_try:
            try:
                # Usamos io.BytesIO para tratar los bytes como un archivo en memoria
                df = pd.read_csv(io.BytesIO(content_bytes), sep=";", encoding=encoding)
                # Si la lectura es exitosa, rompemos el bucle
                break
            except UnicodeDecodeError:
                # Si esta codificación falla, probamos con la siguiente
                continue
        
        # Si después de probar todas, ninguna funcionó
        if df is None:
            st.error(f"No se pudo decodificar el archivo CSV desde {url} con las codificaciones probadas.")
            return None

        # 4. Convertimos el DataFrame limpio de vuelta a un objeto de bytes en memoria
        #    para mantener la compatibilidad con el resto de la aplicación.
        output_csv_bytes = df.to_csv(index=False, sep=';').encode('utf-8')
        return io.BytesIO(output_csv_bytes)

    except Exception as e:
        st.error(f"Error al cargar el archivo CSV desde la URL: {url}\nError: {e}")
        return None

@st.cache_data(ttl=3600)
def load_zip_from_url(url):
    """Descarga un archivo ZIP (shapefile) desde una URL y lo retorna como un objeto de bytes en memoria."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Lanza un error si la descarga falla
        return io.BytesIO(response.content)
    except Exception as e:
        st.error(f"Error al descargar el archivo ZIP desde la URL: {url}\nError: {e}")
        return None
