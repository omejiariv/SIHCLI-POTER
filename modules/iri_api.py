import requests
import pandas as pd
import streamlit as st

# URL Base del servicio de datos del IRI
IRI_BASE_URL = "https://ftp.iri.columbia.edu/ensodata"

# --- MODO PRODUCCIÓN: CACHÉ ACTIVADO ---
@st.cache_data(ttl=24*3600)  # Se actualiza cada 24 horas
def fetch_iri_data(filename):
    """
    Descarga los archivos JSON seguros del IRI usando credenciales de st.secrets.
    Maneja errores de forma silenciosa para el usuario final.
    """
    url = f"{IRI_BASE_URL}/{filename}"
    
    # Recuperación segura de credenciales
    try:
        user = st.secrets["iri"]["username"]
        pwd = st.secrets["iri"]["password"]
    except Exception:
        # Si faltan credenciales, retornamos None (la UI mostrará una advertencia amigable)
        return None

    try:
        # Timeout de 20s para no congelar la app si el FTP está lento
        response = requests.get(url, auth=(user, pwd), timeout=20)
        
        if response.status_code == 200:
            return response.json()
        else:
            # Log interno (opcional) pero no rompemos la UI
            print(f"IRI Error {response.status_code}: {response.reason}")
            return None
            
    except Exception as e:
        print(f"IRI Connection Error: {e}")
        return None

# --- FUNCIONES DE PROCESAMIENTO (ESTÁNDAR) ---
def process_iri_plume(data_json):
    """Procesa el JSON de plumas (modelos spaghetti)"""
    if not data_json or 'years' not in data_json: return None
    
    # Navegar al último dato disponible
    last_year_entry = data_json['years'][-1]
    year = last_year_entry['year']
    
    if not last_year_entry['months']: return None
    last_month_entry = last_year_entry['months'][-1]
    month_idx = last_month_entry['month']
    
    models_data = []
    if 'models' in last_month_entry:
        for m in last_month_entry['models']:
            # Limpieza de valores nulos (-999)
            clean_values = [x if x > -100 else None for x in m['data']]
            models_data.append({
                'name': m['model'], 
                'type': m['type'], 
                'values': clean_values
            })
            
    # Generar etiquetas de trimestres futuros
    seasons_base = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
    start_idx = (month_idx + 1) % 12
    forecast_seasons = [seasons_base[(start_idx + i) % 12] for i in range(9)]

    return {
        'year': year, 
        'month_idx': month_idx, 
        'seasons': forecast_seasons, 
        'models': models_data
    }

def process_iri_probabilities(data_json):
    """Procesa el JSON de probabilidades (barras)"""
    if not data_json or 'years' not in data_json: return None
    
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
