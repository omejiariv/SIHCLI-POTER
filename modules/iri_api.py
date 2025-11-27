import requests
import pandas as pd
import streamlit as st

# URL Base del servicio de datos del IRI
IRI_BASE_URL = "https://ftp.iri.columbia.edu/ensodata"

# --- QUITAMOS EL CACHÉ TEMPORALMENTE PARA DIAGNOSTICAR ---
# @st.cache_data(ttl=24*3600) 
def fetch_iri_data(filename):
    """
    Versión de DIAGNÓSTICO: Muestra errores explícitos en pantalla.
    """
    url = f"{IRI_BASE_URL}/{filename}"
    
    # 1. DIAGNÓSTICO DE SECRETOS
    if "iri" not in st.secrets:
        st.error("❌ CRÍTICO: No se encuentra la sección [iri] en secrets.toml. Revisa que no esté indentada dentro de [connections].")
        return None
    
    try:
        user = st.secrets["iri"]["username"]
        pwd = st.secrets["iri"]["password"]
    except KeyError as e:
        st.error(f"❌ ERROR DE CLAVE: Falta la variable '{e}' dentro de la sección [iri].")
        return None

    # 2. INTENTO DE CONEXIÓN
    try:
        # Imprimimos aviso para saber que está intentando (solo visible mientras carga)
        print(f"Intentando conectar a {url} con usuario: {user}...") 
        
        response = requests.get(url, auth=(user, pwd), timeout=20)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            st.error("🔒 ERROR 401 (NO AUTORIZADO): El usuario o contraseña son incorrectos. Verifica que no tengan espacios extra en secrets.toml.")
            return None
        else:
            st.error(f"⚠️ ERROR HTTP {response.status_code}: {response.reason}")
            return None
            
    except Exception as e:
        st.error(f"🔥 ERROR DE CONEXIÓN: {e}")
        return None

# --- FUNCIONES DE PROCESAMIENTO (SE MANTIENEN IGUAL) ---
def process_iri_plume(data_json):
    if not data_json or 'years' not in data_json: return None
    last_year_entry = data_json['years'][-1]
    year = last_year_entry['year']
    if not last_year_entry['months']: return None
    last_month_entry = last_year_entry['months'][-1]
    month_idx = last_month_entry['month']
    
    models_data = []
    if 'models' in last_month_entry:
        for m in last_month_entry['models']:
            clean_values = [x if x > -100 else None for x in m['data']]
            models_data.append({'name': m['model'], 'type': m['type'], 'values': clean_values})
            
    seasons_base = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
    start_idx = (month_idx + 1) % 12
    forecast_seasons = [seasons_base[(start_idx + i) % 12] for i in range(9)]

    return {'year': year, 'month_idx': month_idx, 'seasons': forecast_seasons, 'models': models_data}

def process_iri_probabilities(data_json):
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
