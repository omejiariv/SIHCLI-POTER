import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import streamlit as st

# URL Base HTTPS (Tal como sugirió el soporte del IRI)
IRI_BASE_URL = "https://ftp.iri.columbia.edu/ensodata"

def get_iri_session():
    """
    Crea una sesión HTTP 'camuflada' como un navegador web.
    Esto es necesario porque los servidores universitarios suelen bloquear
    las peticiones que se identifican como scripts de Python.
    """
    session = requests.Session()
    
    # 1. MÁSCARA DE NAVEGADOR (User-Agent)
    # Hacemos creer al servidor que somos Chrome en Windows
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }
    session.headers.update(headers)
    
    # 2. ESTRATEGIA DE REINTENTOS ROBUSTA
    # Si falla, reintenta 3 veces esperando un poco entre intentos
    retry = Retry(
        total=3,
        backoff_factor=2,  # Espera 2s, 4s, 8s...
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    return session

# Usamos caché para no bombardear al servidor (12 horas)
@st.cache_data(ttl=12*3600, show_spinner=False)
def fetch_iri_data(filename):
    """
    Descarga archivos JSON del IRI usando autenticación y sesión robusta.
    """
    url = f"{IRI_BASE_URL}/{filename}"
    
    # 1. VALIDACIÓN Y LIMPIEZA DE CREDENCIALES
    try:
        if "iri" not in st.secrets:
            # Fallback silencioso o error controlado si no hay secrets locales
            return None
            
        # .strip() elimina espacios en blanco accidentales al inicio o final
        user = st.secrets["iri"]["username"].strip()
        pwd = st.secrets["iri"]["password"].strip()
        
    except Exception as e:
        st.error(f"❌ Error leyendo credenciales IRI: {e}")
        return None

    # 2. INTENTO DE DESCARGA
    try:
        session = get_iri_session()
        
        # Timeout de 30 segundos para redes lentas
        response = session.get(url, auth=(user, pwd), timeout=30)
        
        if response.status_code == 200:
            return response.json()
            
        elif response.status_code == 401:
            st.error(f"🔒 ACCESO DENEGADO (401) al archivo `{filename}`.")
            st.caption("El servidor rechazó la contraseña. Verifica en .streamlit/secrets.toml")
            return None
            
        elif response.status_code == 404:
            st.warning(f"🔍 Archivo no encontrado en el servidor: `{filename}`")
            return None
            
        else:
            st.error(f"⚠️ Error del Servidor IRI ({response.status_code}): {response.reason}")
            return None
            
    except requests.exceptions.SSLError:
        st.error("🔒 Error de seguridad SSL con el servidor del IRI.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("📡 No se pudo conectar a `ftp.iri.columbia.edu`. Posible bloqueo de firewall o sitio caído.")
        return None
    except Exception as e:
        st.error(f"🔥 Error inesperado: {e}")
        return None

# --- FUNCIONES DE PROCESAMIENTO (TU LÓGICA ESTÁ PERFECTA) ---

def process_iri_plume(data_json):
    """Procesa el JSON de plumas (modelos spaghetti)"""
    if not data_json or 'years' not in data_json: return None
    
    try:
        # Busca el último año disponible
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
        return None
