# modules/life_zones.py
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import streamlit as st
import math

# --- Constantes ---
# Ajustados según la fórmula/tablas proporcionadas
LAPSE_RATE = 6.0 # °C / 1000m
BASE_TEMP_SEA_LEVEL = 28.0 # °C (Según fórmula por defecto)
LATITUDE_ADJUSTMENT_FACTOR = 0.0 # Mantener en 0 por ahora

# --- Funciones de Cálculo (Ajustadas/Confirmadas) ---

def estimate_mean_annual_temp(elevation_m):
    # Calcula TMA según fórmula SI(...) por defecto
    estimated_temp = BASE_TEMP_SEA_LEVEL - (LAPSE_RATE * elevation_m / 1000.0)
    # Límite inferior más realista para alta montaña Nival/Alpino
    return np.maximum(estimated_temp, -15) 

def calculate_biotemperature(mean_annual_temp, latitude): # <-- AÑADE , latitude AQUÍ
    # Aproximación de BAT clampeando TMA entre 0-30...
    clamped_temp = np.clip(mean_annual_temp, 0, 30)
    # Ajuste simple por latitud (opcional)
    lat_adjustment = LATITUDE_ADJUSTMENT_FACTOR * np.abs(latitude) 
    biotemp = np.maximum(0, clamped_temp + lat_adjustment) # Aplicar ajuste
    # Si TMA < 0, BAT debe ser 0 según definición Holdridge
    biotemp = np.where(mean_annual_temp < 0, 0, biotemp)
    return biotemp

def calculate_pet(biotemperature):
    # Fórmula Holdridge estándar
    return 58.93 * biotemperature

def calculate_per(pet, precipitation):
    # Razón PET/PPA
    precipitation = np.maximum(precipitation, 1) # Evitar división por cero
    per = pet / precipitation
    return per

# --- Diccionario de Zonas de Vida (REVISADO según tablas y diagrama) ---
# Los nombres ahora intentan seguir la nomenclatura Piso Altitudinal + Provincia Humedad
# Los IDs se agrupan lógicamente
holdridge_zone_map = {
    # Nival (Piso 1: BAT < 1.5)
    "Nival / Hielo": 1,                     # Todas las humedades
    # Alpino (Piso 2: 1.5 <= BAT < 3)
    "Desierto Alpino (dA)": 10,             # Desecado
    "Tundra seca Alpina (tsA)": 11,         # Árido / Semiárido
    "Estepa húmeda Alpina (ehA)": 12,       # Subhúmedo / Húmedo (Tundra humeda)
    "Estepa pluvial Alpina (epA)": 13,      # Perhúmedo (Tundra muy humeda)
    "Tundra pluvial Alpina (tpA)": 14,      # Pluvial / Superpluvial
    # Subalpino (Piso 3: 3 <= BAT < 6)
    "Desierto Subalpino (dSA)": 20,         # Desecado
    "Matorral desértico Subalpino (mdSA)": 21, # Árido
    "Estepa espinosa Subalpina (eeSA)": 22, # Semiárido (Paramo seco?)
    "Páramo húmedo Subalpino (phSA)": 23,  # Subhúmedo / Húmedo (Paramo?)
    "Páramo pluvial Subalpino (ppSA)": 24, # Perhúmedo / Pluvial (Paramo muy humedo?)
    "Tundra pluvial Subalpina (tpSA)": 25, # Superpluvial (No común, pero posible)
    # Montano (Piso 4: 6 <= BAT < 12)
    "Desierto Montano (dM)": 30,           # Desecado / Perárido
    "Matorral desértico Montano (mdM)": 31, # Árido
    "Estepa espinosa Montano (eeM)": 32,   # Semiárido
    "Bosque seco Montano (bsM)": 33,       # Subhúmedo
    "Bosque húmedo Montano (bhM)": 34,      # Húmedo
    "Bosque muy húmedo Montano (bmhM)": 35, # Perhúmedo
    "Bosque pluvial Montano (bpM)": 36,    # Pluvial / Superpluvial
    # Premontano (Piso 5: 12 <= BAT < 18)
    "Desierto Premontano (dPM)": 40,       # Desecado / Perárido
    "Matorral desértico Premontano (mdPM)": 41, # Árido
    "Monte espinoso Premontano (mePM)": 42,# Semiárido
    "Bosque seco Premontano (bsPM)": 43,   # Subhúmedo
    "Bosque húmedo Premontano (bhPM)": 44,  # Húmedo
    "Bosque muy húmedo Premontano (bmhPM)": 45,# Perhúmedo
    "Bosque pluvial Premontano (bpPM)": 46,# Pluvial / Superpluvial
    # Basal (Piso 6: BAT >= 18) -- Incluye Tropical y Subtropical
    "Desierto Basal (dB)": 50,             # Desecado / Perárido
    "Matorral desértico Basal (mdB)": 51,   # Árido
    "Monte espinoso Basal (meB)": 52,      # Semiárido
    "Bosque seco Basal (bsB)": 53,         # Subhúmedo
    "Bosque húmedo Basal (bhB)": 54,        # Húmedo
    "Bosque muy húmedo Basal (bmhB)": 55,   # Perhúmedo
    "Bosque pluvial Basal (bpB)": 56,      # Pluvial / Superpluvial
    # Zona Desconocida / NoData
    "Zona Desconocida": 0
}

# Mapa inverso ID -> Nombre
holdridge_int_to_name = {v: k for k, v in holdridge_zone_map.items()}

# --- Función de Clasificación (Basada en Tablas BAT y Humedad[PPT/PER]) ---

def classify_holdridge_zone_detailed(bat, ppt, per):
    """
    Clasifica la Zona de Vida usando límites de BAT (Piso Altitudinal)
    y límites de PPT/PER (Provincia de Humedad) según las tablas proporcionadas.
    """
    if pd.isna(bat) or pd.isna(ppt) or pd.isna(per) or bat < 0 or ppt <= 0:
        return 0 # ID Zona Desconocida

    # 1. Determinar Piso Altitudinal (Según Tabla BAT)
    if bat < 1.5: piso = "Nival"
    elif bat < 3: piso = "Alpino"
    elif bat < 6: piso = "Subalpino"
    elif bat < 12: piso = "Montano"
    elif bat < 18: piso = "Premontano"
    else: piso = "Basal" # (>= 18)

    # 2. Determinar Provincia de Humedad (Según Tabla Humedad PPT y PER)
    # Usaremos PER como guía principal y PPT como desempate/confirmación
    
    # Nivel Superpluvial / Pluvial (Rain Forest)
    if per < 0.25: # Muy húmedo
        if ppt >= 8000: provincia = "Superpluvial"
        else: provincia = "Pluvial" # ppt < 8000
    # Nivel Perhúmedo (Wet Forest)
    elif per < 0.5:
        provincia = "Perhúmedo" # ppt >= 2000 (generalmente)
    # Nivel Húmedo (Moist Forest)
    elif per < 1.0:
        provincia = "Húmedo" # ppt >= 1000 (generalmente)
    # Nivel Subhúmedo (Dry Forest)
    elif per < 2.0:
        provincia = "Subhúmedo" # ppt >= 500 (generalmente)
    # Nivel Semiárido (Steppe / Very Dry Forest)
    elif per < 4.0:
        provincia = "Semiárido" # ppt >= 250 (generalmente)
    # Nivel Árido (Thorn Steppe / Desert Scrub)
    elif per < 8.0:
        provincia = "Árido" # ppt >= 125 (generalmente)
    # Nivel Perárido (Desert)
    elif per < 16.0:
        provincia = "Perárido" # ppt >= 62.5 (generalmente)
    # Nivel Superárido / Desecado (Ice / Desert)
    else: # per >= 16.0
        # Diferenciar Desecado (muy poca lluvia) de Nival (muy frío)
        if ppt < 62.5: provincia = "Desecado" # Extremadamente seco
        elif piso == "Nival": provincia = "Nival" # Hielo
        else: provincia = "Desecado" # Muy seco pero no necesariamente helado

    # 3. Asignar Zona de Vida Específica (Combinando Piso y Provincia)
    zone_id = 0 # Default

    if piso == "Nival": zone_id = 1 # Nival / Hielo
    
    elif piso == "Alpino":
        if provincia in ["Superpluvial", "Pluvial"]: zone_id = 14 # tpA
        elif provincia == "Perhúmedo": zone_id = 13 # epA? (Tundra muy humeda -> Estepa pluvial?)
        elif provincia == "Húmedo": zone_id = 12 # ehA? (Tundra humeda -> Estepa humeda?)
        elif provincia == "Subhúmedo": zone_id = 11 # tsA?
        elif provincia == "Semiárido": zone_id = 11 # tsA
        elif provincia == "Árido": zone_id = 11 # tsA
        else: zone_id = 10 # dA (Perárido, Desecado)

    elif piso == "Subalpino":
        if provincia == "Superpluvial": zone_id = 25 # tpSA? (Podría ser ppSA)
        elif provincia == "Pluvial": zone_id = 24 # ppSA
        elif provincia == "Perhúmedo": zone_id = 24 # ppSA (Asumiendo muy humedo = pluvial?)
        elif provincia == "Húmedo": zone_id = 23 # phSA
        elif provincia == "Subhúmedo": zone_id = 22 # eeSA? (Páramo seco?)
        elif provincia == "Semiárido": zone_id = 22 # eeSA
        elif provincia == "Árido": zone_id = 21 # mdSA
        else: zone_id = 20 # dSA

    elif piso == "Montano":
        if provincia in ["Superpluvial", "Pluvial"]: zone_id = 36 # bpM
        elif provincia == "Perhúmedo": zone_id = 35 # bmhM
        elif provincia == "Húmedo": zone_id = 34 # bhM
        elif provincia == "Subhúmedo": zone_id = 33 # bsM
        elif provincia == "Semiárido": zone_id = 32 # eeM
        elif provincia == "Árido": zone_id = 31 # mdM
        else: zone_id = 30 # dM

    elif piso == "Premontano":
        if provincia in ["Superpluvial", "Pluvial"]: zone_id = 46 # bpPM
        elif provincia == "Perhúmedo": zone_id = 45 # bmhPM
        elif provincia == "Húmedo": zone_id = 44 # bhPM
        elif provincia == "Subhúmedo": zone_id = 43 # bsPM
        elif provincia == "Semiárido": zone_id = 42 # mePM
        elif provincia == "Árido": zone_id = 41 # mdPM
        else: zone_id = 40 # dPM

    elif piso == "Basal":
        if provincia in ["Superpluvial", "Pluvial"]: zone_id = 56 # bpB
        elif provincia == "Perhúmedo": zone_id = 55 # bmhB
        elif provincia == "Húmedo": zone_id = 54 # bhB
        elif provincia == "Subhúmedo": zone_id = 53 # bsB
        elif provincia == "Semiárido": zone_id = 52 # meB
        elif provincia == "Árido": zone_id = 51 # mdB
        else: zone_id = 50 # dB

    return zone_id


# --- Función generate_life_zone_map (Usa la clasificación actualizada) ---
@st.cache_data(show_spinner="Generando mapa de Zonas de Vida...")
def generate_life_zone_map(dem_path, precip_raster_path, mean_latitude, downscale_factor=4):
    """Genera un mapa raster clasificado de Zonas de Vida de Holdridge."""
    try:
        # Reescalado, lectura DEM y Precip (sin cambios)
        if downscale_factor <= 0: downscale_factor = 1
        with rasterio.open(dem_path) as dem_src:
            src_profile=dem_src.profile; src_crs=dem_src.crs; src_transform=dem_src.transform; nodata_dem=dem_src.nodata
            dst_height=src_profile['height']//downscale_factor; dst_width=src_profile['width']//downscale_factor
            dst_transform=src_transform*src_transform.scale((src_profile['width']/dst_width),(src_profile['height']/dst_height))
            dst_profile=src_profile.copy(); dst_profile.update({'height':dst_height,'width':dst_width,'transform':dst_transform,'dtype':rasterio.float32,'nodata':np.nan})
            dem_data=np.empty((dst_height,dst_width),dtype=rasterio.float32)
            reproject(source=rasterio.band(dem_src,1),destination=dem_data,src_transform=src_transform,src_crs=src_crs,src_nodata=nodata_dem,dst_transform=dst_transform,dst_crs=src_crs,dst_nodata=np.nan,resampling=Resampling.average)
            dem_mask=np.isnan(dem_data)
        with rasterio.open(precip_raster_path) as precip_src:
            precip_data_aligned=np.empty((dst_height,dst_width),dtype=rasterio.float32)
            reproject(source=rasterio.band(precip_src,1),destination=precip_data_aligned,src_transform=precip_src.transform,src_crs=precip_src.crs,src_nodata=precip_src.nodata,dst_transform=dst_transform,dst_crs=src_crs,dst_nodata=np.nan,resampling=Resampling.bilinear)
            precip_mask=np.isnan(precip_data_aligned)

        # Cálculos biofísicos (ahora sí usamos PET/PER para clasificación)
        st.write("Calculando variables biofísicas...")
        with np.errstate(invalid='ignore'):
            tma_raster = estimate_mean_annual_temp(dem_data)
            bat_raster = calculate_biotemperature(tma_raster, mean_latitude)
            pet_raster = calculate_pet(bat_raster)
            per_raster = calculate_per(pet_raster, precip_data_aligned)
        st.write("Cálculos completados.")

        # Clasificar píxeles usando la función DETALLADA
        st.write("Clasificando Zonas de Vida...")
        classified_raster = np.full((dst_height, dst_width), 0, dtype=np.int16) # 0 para NoData
        valid_pixels = ~dem_mask & ~precip_mask & ~np.isnan(bat_raster) & ~np.isnan(per_raster) & ~np.isnan(precip_data_aligned)

        bat_values = bat_raster[valid_pixels]
        ppt_values = precip_data_aligned[valid_pixels]
        per_values = per_raster[valid_pixels] # PER es necesario ahora

        # Vectorizar la clasificación detallada
        vectorized_classify = np.vectorize(classify_holdridge_zone_detailed)
        zone_ints = vectorized_classify(bat_values, ppt_values, per_values) # Pasamos BAT, PPT, PER

        classified_raster[valid_pixels] = zone_ints.astype(np.int16)
        st.write("Clasificación completada.")

        # Preparar salida (sin cambios)
        output_profile = dst_profile.copy()
        output_profile.update({'dtype': rasterio.int16, 'nodata': 0, 'count': 1})

        return classified_raster, output_profile, holdridge_int_to_name

    except Exception as e:
        st.error(f"Error generando mapa de zonas de vida: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None
