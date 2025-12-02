import json
import os
import pandas as pd
import streamlit as st

# Ruta donde guardaste los archivos descargados manualmente
# Asegúrate de crear esta carpeta y poner los archivos ahí
LOCAL_DATA_PATH = "data/iri" 

def fetch_iri_data(filename):
    """
    Carga los datos del IRI desde archivos locales (GeoJSON/JSON)
    previamente descargados, en lugar de conectar en vivo.
    
    Esto evita errores de conexión SSL/Firewall y es ideal dado que
    el pronóstico solo cambia mensualmente.
    """
    file_path = os.path.join(LOCAL_DATA_PATH, filename)
    
    try:
        # Verificar si el archivo existe
        if not os.path.exists(file_path):
            # Intentar buscar en la raíz si no está en subcarpeta (por si acaso)
            if os.path.exists(filename):
                file_path = filename
            else:
                st.warning(f"⚠️ Archivo de datos no encontrado: `{filename}`. Verifica que esté en `{LOCAL_DATA_PATH}`.")
                return None

        # Leer archivo local
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
            
    except json.JSONDecodeError:
        st.error(f"❌ Error: El archivo `{filename}` no tiene un formato JSON válido.")
        return None
    except Exception as e:
        st.error(f"❌ Error leyendo archivo local `{filename}`: {e}")
        return None

# --- FUNCIONES DE PROCESAMIENTO (SE MANTIENEN IGUAL) ---

def process_iri_plume(data_json):
    """Procesa el JSON de plumas (modelos spaghetti)"""
    if not data_json or 'years' not in data_json: return None
    
    try:
        # Busca el último año disponible en el archivo
        last_year_entry = data_json['years'][-1]
        year = last_year_entry['year']
        
        if not last_year_entry['months']: return None
        # Busca el último mes disponible
        last_month_entry = last_year_entry['months'][-1]
        month_idx = last_month_entry['month']
        
        models_data = []
        if 'models' in last_month_entry:
            for m in last_month_entry['models']:
                # Limpieza de valores centinela (-999, etc)
                clean_values = [x if x is not None and x > -100 else None for x in m['data']]
                models_data.append({
                    'name': m['model'], 
                    'type': m['type'], 
                    'values': clean_values
                })
            
        seasons_base = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
        start_idx = (month_idx + 1) % 12
        forecast_seasons = [seasons_base[(start_idx + i) % 12] for i in range(9)]

        return {
            'year': year, 
            'month_idx': month_idx, 
            'seasons': forecast_seasons, 
            'models': models_data
        }
    except Exception as e:
        st.error(f"Error procesando estructura Plume: {e}")
        return None

def process_iri_probabilities(data_json):
    """Procesa el JSON de probabilidades (barras)"""
    if not data_json or 'years' not in data_json: return None
    
    try:
        last_year_entry = data_json['years'][-1]
        if not last_year_entry['months']: return None
        
        last_month_entry = last_year_entry['months'][-1]
        
        probs = []
        if 'probabilities' in last_month_entry:
            for p in last_month_entry['probabilities']:
                probs.append({
                    'Trimestre': p['season'],
                    'La Niña': p['lanina'],
                    'Neutral': p['neutral'],
                    'El Niño': p['elnino']
                })
            
        return pd.DataFrame(probs)
    except Exception as e:
        st.error(f"Error procesando estructura Probabilities: {e}")
        return None
