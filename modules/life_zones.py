# modules/life_zones.py
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import streamlit as st
import math # Para logaritmos

# --- Constantes ---
# (Puedes mantener o ajustar LAPSE_RATE y BASE_TEMP_SEA_LEVEL si tienes mejores datos locales)
LAPSE_RATE = 6.5
BASE_TEMP_SEA_LEVEL = 28.0
LATITUDE_ADJUSTMENT_FACTOR = 0.0 # Mantener en 0 si BASE_TEMP_SEA_LEVEL ya considera la latitud

# --- Funciones de Cálculo (Sin cambios) ---

def estimate_mean_annual_temp(elevation_m):
    estimated_temp = BASE_TEMP_SEA_LEVEL - (LAPSE_RATE * elevation_m / 1000.0)
    return np.maximum(estimated_temp, -15)

def calculate_biotemperature(mean_annual_temp, latitude):
    clamped_temp = np.clip(mean_annual_temp, 0, 30)
    lat_adjustment = LATITUDE_ADJUSTMENT_FACTOR * np.abs(latitude)
    biotemp = np.maximum(0, clamped_temp + lat_adjustment)
    biotemp = np.where(mean_annual_temp < 0, 0, biotemp)
    return biotemp

# (calculate_pet y calculate_per se mantienen, aunque la clasificación principal no usará PER)
def calculate_pet(biotemperature):
    return 58.93 * biotemperature

def calculate_per(pet, precipitation):
    precipitation = np.maximum(precipitation, 1)
    per = pet / precipitation
    return per

# --- Diccionario de Zonas de Vida (AJUSTADO A TABLA ANTIOQUIA) ---
# Basado en image_0814d6.png
holdridge_zone_map = {
    # Nival (BAT < 1.5) - No en tabla, pero mantenido por si acaso
    "Nival / Hielo": 1,
    # Subnival / Alpino (1.5 <= BAT < 3)
    "Tundra pluvial alpino (tp-A)": 2,
    "Tundra húmeda alpino (th-A)": 3,
    "Tundra seca alpino (ts-A)": 4,
    # Páramo / Subalpino (3 <= BAT < 6)
    "Páramo pluvial subalpino (pp-SA)": 5,
    "Páramo muy húmedo subalpino (pmh-SA)": 6, # Ajustado nombre
    "Páramo seco subalpino (ps-SA)": 7,
    # Montano (6 <= BAT < 12)
    "Bosque pluvial Montano (bp-M)": 8,
    "Bosque muy húmedo Montano (bmh-M)": 9,
    "Bosque húmedo Montano (bh-M)": 10,
    "Bosque seco Montano (bs-M)": 11,
    # Premontano / Montano Bajo (12 <= BAT < 18)
    "Bosque pluvial Premontano (bp-PM)": 12, # Cambiado nombre base a Premontano
    "Bosque muy húmedo Premontano (bmh-PM)": 13,
    "Bosque húmedo Premontano (bh-PM)": 14,
    "Bosque seco Premontano (bs-PM)": 15,
    "Monte espinoso Premontano (me-PM)": 16, # Añadido desde tabla
    # Basal / Tropical (BAT >= 18)
    "Bosque pluvial Tropical (bp-T)": 17, # Cambiado nombre base a Tropical
    "Bosque muy húmedo Tropical (bmh-T)": 18,
    "Bosque húmedo Tropical (bh-T)": 19,
    "Bosque seco Tropical (bs-T)": 20,
    "Monte espinoso Tropical (me-T)": 21, # Añadido desde tabla
    # Zona Desconocida / NoData
    "Zona Desconocida": 0
}

# Mapa inverso ID -> Nombre
holdridge_int_to_name = {v: k for k, v in holdridge_zone_map.items()}

# --- Función de Clasificación REESCRITA (Basada en Tabla BAT y PPT) ---

def classify_holdridge_zone_antioquia(bat, ppt):
    """
    Clasifica la Zona de Vida usando los rangos de BAT y PPT
    específicos de la tabla para Antioquia (image_0814d6.png).
    """
    # Manejar valores no válidos
    if pd.isna(bat) or pd.isna(ppt) or bat < 0 or ppt <= 0:
        return 0 # ID para Zona Desconocida

    # Convertir a ID entero
    zone_id = 0 # Default Zona Desconocida

    # Nival (No en tabla, añadido por completitud)
    if bat < 1.5:
        zone_id = holdridge_zone_map["Nival / Hielo"]
    # Subnival / Alpino (1.5 <= BAT < 3)
    elif bat < 3:
        if ppt >= 1500: zone_id = holdridge_zone_map["Tundra pluvial alpino (tp-A)"]
        elif ppt >= 750: zone_id = holdridge_zone_map["Tundra húmeda alpino (th-A)"]
        else: zone_id = holdridge_zone_map["Tundra seca alpino (ts-A)"] # ppt < 750
    # Páramo / Subalpino (3 <= BAT < 6)
    elif bat < 6:
        if ppt >= 2000: zone_id = holdridge_zone_map["Páramo pluvial subalpino (pp-SA)"]
        elif ppt >= 1000: zone_id = holdridge_zone_map["Páramo muy húmedo subalpino (pmh-SA)"]
        else: zone_id = holdridge_zone_map["Páramo seco subalpino (ps-SA)"] # ppt < 1000
    # Montano (6 <= BAT < 12)
    elif bat < 12:
        if ppt >= 4000: zone_id = holdridge_zone_map["Bosque pluvial Montano (bp-M)"]
        elif ppt >= 2000: zone_id = holdridge_zone_map["Bosque muy húmedo Montano (bmh-M)"]
        elif ppt >= 1000: zone_id = holdridge_zone_map["Bosque húmedo Montano (bh-M)"]
        else: zone_id = holdridge_zone_map["Bosque seco Montano (bs-M)"] # ppt < 1000 (Monte espinoso no está en tabla para Montano)
    # Premontano / Montano Bajo (12 <= BAT < 18)
    elif bat < 18:
        if ppt >= 4000: zone_id = holdridge_zone_map["Bosque pluvial Premontano (bp-PM)"]
        elif ppt >= 2000: zone_id = holdridge_zone_map["Bosque muy húmedo Premontano (bmh-PM)"]
        elif ppt >= 1000: zone_id = holdridge_zone_map["Bosque húmedo Premontano (bh-PM)"]
        elif ppt >= 500: zone_id = holdridge_zone_map["Bosque seco Premontano (bs-PM)"]
        else: zone_id = holdridge_zone_map["Monte espinoso Premontano (me-PM)"] # ppt < 500
    # Basal / Tropical (BAT >= 18)
    else: # bat >= 18
        if ppt >= 4000: zone_id = holdridge_zone_map["Bosque pluvial Tropical (bp-T)"]
        elif ppt >= 2000: zone_id = holdridge_zone_map["Bosque muy húmedo Tropical (bmh-T)"]
        elif ppt >= 1000: zone_id = holdridge_zone_map["Bosque húmedo Tropical (bh-T)"]
        elif ppt >= 500: zone_id = holdridge_zone_map["Bosque seco Tropical (bs-T)"] # <-- ¡AQUÍ ESTÁ!
        else: zone_id = holdridge_zone_map["Monte espinoso Tropical (me-T)"] # ppt < 500

    return zone_id


# --- Función Principal para Generar el Mapa (MODIFICADA para usar nueva clasificación) ---
@st.cache_data(show_spinner="Generando mapa de Zonas de Vida...")
def generate_life_zone_map(dem_path, precip_raster_path, mean_latitude, downscale_factor=4):
    """
    Genera un mapa raster clasificado de Zonas de Vida de Holdridge.
    """
    try:
        # --- Factor de reescalado ---
        if downscale_factor <= 0: downscale_factor = 1

        # 1. Abrir DEM y remuestrear
        with rasterio.open(dem_path) as dem_src:
            src_profile = dem_src.profile
            src_crs = dem_src.crs
            src_transform = dem_src.transform
            nodata_dem = dem_src.nodata
            dst_height = src_profile['height'] // downscale_factor
            dst_width = src_profile['width'] // downscale_factor
            dst_transform = src_transform * src_transform.scale(
                (src_profile['width'] / dst_width), (src_profile['height'] / dst_height)
            )
            dst_profile = src_profile.copy()
            dst_profile.update({
                'height': dst_height, 'width': dst_width, 'transform': dst_transform,
                'dtype': rasterio.float32, 'nodata': np.nan
            })
            dem_data = np.empty((dst_height, dst_width), dtype=rasterio.float32)
            reproject(
                source=rasterio.band(dem_src, 1), destination=dem_data,
                src_transform=src_transform, src_crs=src_crs, src_nodata=nodata_dem,
                dst_transform=dst_transform, dst_crs=src_crs, dst_nodata=np.nan,
                resampling=Resampling.average
            )
            dem_mask = np.isnan(dem_data)

        # 2. Abrir Precipitación y alinear/remuestrear
        with rasterio.open(precip_raster_path) as precip_src:
            precip_data_aligned = np.empty((dst_height, dst_width), dtype=rasterio.float32)
            reproject(
                source=rasterio.band(precip_src, 1), destination=precip_data_aligned,
                src_transform=precip_src.transform, src_crs=precip_src.crs, src_nodata=precip_src.nodata,
                dst_transform=dst_transform, dst_crs=src_crs, dst_nodata=np.nan,
                resampling=Resampling.bilinear
            )
            precip_mask = np.isnan(precip_data_aligned)

        # 3. Cálculos (TMA y BAT)
        with np.errstate(invalid='ignore'):
            tma_raster = estimate_mean_annual_temp(dem_data)
            bat_raster = calculate_biotemperature(tma_raster, mean_latitude)
            # Ya no necesitamos PET ni PER para la clasificación principal

        # 4. Clasificar píxeles usando la función específica de ANTIOQUIA
        st.write("Clasificando Zonas de Vida (Antioquia)...")
        classified_raster = np.full((dst_height, dst_width), 0, dtype=np.int16) # 0 para NoData
        valid_pixels = ~dem_mask & ~precip_mask & ~np.isnan(bat_raster) & ~np.isnan(precip_data_aligned)

        bat_values = bat_raster[valid_pixels]
        ppt_values = precip_data_aligned[valid_pixels]

        # Vectorizar la nueva función de clasificación
        vectorized_classify = np.vectorize(classify_holdridge_zone_antioquia)
        zone_ints = vectorized_classify(bat_values, ppt_values) # Solo necesita BAT y PPT

        classified_raster[valid_pixels] = zone_ints.astype(np.int16)
        st.write("Clasificación completada.")

        # 5. Preparar salida
        output_profile = dst_profile.copy()
        output_profile.update({
            'dtype': rasterio.int16,
            'nodata': 0, # Usar 0 (Zona Desconocida) como nodata
            'count': 1
        })

        return classified_raster, output_profile, holdridge_int_to_name

    except Exception as e:
        st.error(f"Error generando mapa de zonas de vida: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None
