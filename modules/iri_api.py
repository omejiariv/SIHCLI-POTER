import requests
import pandas as pd
import streamlit as st

# URL Base del servicio de datos del IRI
IRI_BASE_URL = "https://ftp.iri.columbia.edu/ensodata"

# --- MODO DIAGNÓSTICO (SIN CACHÉ PARA FORZAR INTENTO) ---
# @st.cache_data(ttl=24*3600)  <-- Comentado para que reintente siempre
def fetch_iri_data(filename):
    """
    Descarga archivos JSON del IRI.
    MODO VERBOSO: Muestra errores en pantalla para depuración.
    """
    url = f"{IRI_BASE_URL}/{filename}"
    
    # 1. PRUEBA DE SECRETOS
    try:
        # Verificamos si existe la sección principal
        if "iri" not in st.secrets:
            st.error(f"❌ ERROR CRÍTICO: No se encuentra la sección `[iri]` en `secrets.toml`.")
            st.info("Asegúrate de que `[iri]` NO esté indentado (con espacios) dentro de otra sección.")
            return None
            
        user = st.secrets["iri"]["username"]
        pwd = st.secrets["iri"]["password"]
    except KeyError as e:
        st.error(f"❌ ERROR DE CLAVE: Falta el campo `{e}` dentro de la sección `[iri]` en secrets.toml.")
        return None
    except Exception as e:
        st.error(f"❌ ERROR LEYENDO SECRETS: {e}")
        return None

    # 2. PRUEBA DE CONEXIÓN
    try:
        # Intentamos conectar
        response = requests.get(url, auth=(user, pwd), timeout=20)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            st.error(f"🔒 ERROR 401 (NO AUTORIZADO) en `{filename}`.")
            st.warning("Tus credenciales (usuario/contraseña) son incorrectas o fueron rechazadas por el servidor IRI.")
            return None
        elif response.status_code == 404:
            st.error(f"🔍 ERROR 404 (NO ENCONTRADO): El archivo `{filename}` no existe en la URL.")
            return None
        else:
            st.error(f"⚠️ ERROR HTTP {response.status_code}: {response.reason}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("📡 ERROR DE CONEXIÓN: No se pudo contactar con `ftp.iri.columbia.edu`. Verifica tu internet.")
        return None
    except Exception as e:
        st.error(f"🔥 ERROR INESPERADO: {e}")
        return None

# --- FUNCIONES DE PROCESAMIENTO (SE MANTIENEN IGUAL) ---
def process_iri_plume(data_json):
    """Procesa el JSON de plumas (modelos spaghetti)"""
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
