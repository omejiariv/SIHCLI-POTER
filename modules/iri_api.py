import requests
import pandas as pd
import streamlit as st
import datetime

# URL Base del servicio de datos del IRI
IRI_BASE_URL = "https://ftp.iri.columbia.edu/ensodata"

@st.cache_data(ttl=24*3600)  # Cache de 24 horas (se actualiza mensual)
def fetch_iri_data(filename):
    """
    Descarga los archivos JSON seguros del IRI usando credenciales de st.secrets.
    """
    url = f"{IRI_BASE_URL}/{filename}"
    
    # Recuperar credenciales de forma segura
    try:
        user = st.secrets["iri"]["username"]
        pwd = st.secrets["iri"]["password"]
    except Exception:
        # Si no hay secretos configurados, retornamos None para manejar el error visualmente
        return None

    try:
        response = requests.get(url, auth=(user, pwd), timeout=20)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error conectando a IRI ({response.status_code}): Verifica tus credenciales.")
            return None
    except Exception as e:
        st.error(f"Error de conexión con IRI: {e}")
        return None

def process_iri_plume(data_json):
    """
    Procesa 'enso_plumes.json' para obtener el gráfico de espagueti (Plume).
    Encuentra automáticamente el pronóstico más reciente.
    """
    if not data_json or 'years' not in data_json:
        return None
    
    # 1. Encontrar el último año y mes disponible
    last_year_entry = data_json['years'][-1]
    year = last_year_entry['year']
    
    if not last_year_entry['months']:
        return None
        
    last_month_entry = last_year_entry['months'][-1]
    month_idx = last_month_entry['month'] # 0=Ene, 1=Feb...
    
    # 2. Extraer Modelos
    models_data = []
    if 'models' in last_month_entry:
        for m in last_month_entry['models']:
            # Limpiar datos (-999 suele ser valor nulo en Fortran/Scientific data)
            clean_values = [x if x > -100 else None for x in m['data']]
            models_data.append({
                'name': m['model'],
                'type': m['type'], # Dynamical, Statistical, CPC
                'values': clean_values
            })
            
    # 3. Definir las etiquetas de los trimestres (seasons)
    # IRI predice 9 temporadas hacia adelante. 
    # Si el mes es Nov (10), las temporadas son NDJ, DJF, JFM...
    seasons_base = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
    start_idx = (month_idx + 1) % 12 # Empezamos la predicción al mes siguiente aprox
    forecast_seasons = []
    for i in range(9):
        forecast_seasons.append(seasons_base[(start_idx + i) % 12])

    return {
        'year': year,
        'month_idx': month_idx,
        'seasons': forecast_seasons,
        'models': models_data
    }

def process_iri_probabilities(data_json):
    """
    Procesa 'enso_cpc_prob.json' (Consenso) o 'enso_iri_prob.json' (Objetivo).
    """
    if not data_json or 'years' not in data_json:
        return None
        
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
