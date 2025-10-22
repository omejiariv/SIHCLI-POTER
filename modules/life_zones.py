# modules/life_zones.py

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask # Asegurar que mask esté importado
import streamlit as st
import math

# --- Constantes ---
LAPSE_RATE = 6.0 # °C / 1000m
BASE_TEMP_SEA_LEVEL = 28.0 # °C
LATITUDE_ADJUSTMENT_FACTOR = 0.0

# --- Funciones de Cálculo ---

def estimate_mean_annual_temp(elevation_m):
    estimated_temp = BASE_TEMP_SEA_LEVEL - (LAPSE_RATE * elevation_m / 1000.0)
    return np.maximum(estimated_temp, -15)

def calculate_biotemperature(mean_annual_temp, latitude): # Acepta latitude
    clamped_temp = np.clip(mean_annual_temp, 0, 30)
    lat_adjustment = LATITUDE_ADJUSTMENT_FACTOR * np.abs(latitude)
    biotemp = np.maximum(0, clamped_temp + lat_adjustment)
    biotemp = np.where(mean_annual_temp < 0, 0, biotemp)
    return biotemp

def calculate_pet(biotemperature):
    return 58.93 * biotemperature

def calculate_per(pet, precipitation):
    precipitation = np.maximum(precipitation, 1)
    per = pet / precipitation
    return per

# --- Diccionario de Zonas de Vida (AJUSTADO a tabla Antioquia) ---
holdridge_zone_map = {
    # Nival (BAT < 1.5)
    "Nival / Hielo": 1,
    # Alpino / Subnival (1.5 <= BAT < 3)
    "Tundra pluvial alpino (tp-A)": 2,      # PPT >= 1500
    "Tundra húmeda alpino (th-A)": 3,       # 750 <= PPT < 1500
    "Tundra seca alpino (ts-A)": 4,         # PPT < 750
    # Páramo / Subalpino (3 <= BAT < 6)
    "Páramo pluvial subalpino (pp-SA)": 5,   # PPT >= 2000
    "Páramo muy húmedo subalpino (pmh-SA)": 6,# 1000 <= PPT < 2000
    "Páramo seco subalpino (ps-SA)": 7,     # PPT < 1000
    # Montano (6 <= BAT < 12)
    "Bosque pluvial Montano (bp-M)": 8,     # PPT >= 4000
    "Bosque muy húmedo Montano (bmh-M)": 9, # 2000 <= PPT < 4000
    "Bosque húmedo Montano (bh-M)": 10,     # 1000 <= PPT < 2000
    "Bosque seco Montano (bs-M)": 11,       # 500 <= PPT < 1000
    "Monte espinoso Montano (me-M)": 12,    # PPT < 500
    # Premontano / Montano Bajo (12 <= BAT < 18)
    "Bosque pluvial Premontano (bp-PM)": 13, # PPT >= 4000
    "Bosque muy húmedo Premontano (bmh-PM)": 14,# 2000 <= PPT < 4000
    "Bosque húmedo Premontano (bh-PM)": 15,  # 1000 <= PPT < 2000
    "Bosque seco Premontano (bs-PM)": 16,    # 500 <= PPT < 1000
    "Monte espinoso Premontano (me-PM)": 17, # PPT < 500
    # Basal / Tropical (BAT >= 18)
    "Bosque pluvial Tropical (bp-T)": 18,    # PPT >= 4000
    "Bosque muy húmedo Tropical (bmh-T)": 19,# 2000 <= PPT < 4000
    "Bosque húmedo Tropical (bh-T)": 20,     # 1000 <= PPT < 2000
    "Bosque seco Tropical (bs-T)": 21,       # 500 <= PPT < 1000
    "Monte espinoso Tropical (me-T)": 22,    # PPT < 500
    # Zona Desconocida / NoData
    "Zona Desconocida": 0
}

# Mapa inverso ID -> Nombre
holdridge_int_to_name = {v: k for k, v in holdridge_zone_map.items()}

# --- Función de Clasificación (Basada en Tabla Antioquia BAT/PPT) ---
def classify_holdridge_zone_antioquia(bat, ppt):
    """
    Clasifica la Zona de Vida usando los rangos de BAT y PPT
    específicos de la tabla para Antioquia.
    """
    if pd.isna(bat) or pd.isna(ppt) or bat < 0 or ppt <= 0:
        return 0 # ID Zona Desconocida

    zone_id = 0 # Default

    # Nival
    if bat < 1.5: zone_id = 1
    # Alpino (Subnival)
    elif bat < 3:
        if ppt >= 1500: zone_id = 2
        elif ppt >= 750: zone_id = 3
        else: zone_id = 4
    # Subalpino (Páramo)
    elif bat < 6:
        if ppt >= 2000: zone_id = 5
        elif ppt >= 1000: zone_id = 6
        else: zone_id = 7
    # Montano
    elif bat < 12:
        if ppt >= 4000: zone_id = 8
        elif ppt >= 2000: zone_id = 9
        elif ppt >= 1000: zone_id = 10
        elif ppt >= 500: zone_id = 11
        else: zone_id = 12
    # Premontano (Montano Bajo)
    elif bat < 18:
        if ppt >= 4000: zone_id = 13
        elif ppt >= 2000: zone_id = 14
        elif ppt >= 1000: zone_id = 15
        elif ppt >= 500: zone_id = 16
        else: zone_id = 17
    # Basal (Tropical)
    else: # bat >= 18
        if ppt >= 4000: zone_id = 18
        elif ppt >= 2000: zone_id = 19
        elif ppt >= 1000: zone_id = 20
        elif ppt >= 500: zone_id = 21
        else: zone_id = 22

    return zone_id

# --- Función Principal para Generar el Mapa (CON GUION BAJO EN DEFINICIÓN Y USO INTERNO) ---
# @st.cache_data(show_spinner="Generando mapa de Zonas de Vida...")
def generate_life_zone_map(dem_path, precip_raster_path, mean_latitude, _mask_geometry=None, downscale_factor=4): # <-- GUION BAJO AÑADIDO AQUÍ
    """
    Genera un mapa raster clasificado de Zonas de Vida de Holdridge,
    con opción de máscara por geometría y resolución ajustable.
    _mask_geometry: Una serie de geometrías de GeoPandas (se ignora para caché).
    downscale_factor: 1 = original, 2 = mitad res, 4 = cuarto res, etc.
    """
    try:
        # Factor de reescalado
        if downscale_factor <= 0: downscale_factor = 1

        # 1. Abrir DEM y remuestrear
        with rasterio.open(dem_path) as dem_src:
            src_profile=dem_src.profile; src_crs=dem_src.crs; src_transform=dem_src.transform; nodata_dem=dem_src.nodata
            dst_height=src_profile['height']//downscale_factor; dst_width=src_profile['width']//downscale_factor
            dst_transform=src_transform*src_transform.scale((src_profile['width']/dst_width),(src_profile['height']/dst_height))
            dst_profile=src_profile.copy(); dst_profile.update({'height':dst_height,'width':dst_width,'transform':dst_transform,'dtype':rasterio.float32,'nodata':np.nan})
            dem_data=np.empty((dst_height,dst_width),dtype=rasterio.float32)
            reproject(source=rasterio.band(dem_src,1),destination=dem_data,src_transform=src_transform,src_crs=src_crs,src_nodata=nodata_dem,dst_transform=dst_transform,dst_crs=src_crs,dst_nodata=np.nan,resampling=Resampling.average)
            dem_mask=np.isnan(dem_data)

        # 2. Abrir Precipitación y alinear/remuestrear
        with rasterio.open(precip_raster_path) as precip_src:
            precip_data_aligned=np.empty((dst_height,dst_width),dtype=rasterio.float32)
            reproject(source=rasterio.band(precip_src,1),destination=precip_data_aligned,src_transform=precip_src.transform,src_crs=precip_src.crs,src_nodata=precip_src.nodata,dst_transform=dst_transform,dst_crs=src_crs,dst_nodata=np.nan,resampling=Resampling.bilinear)
            precip_mask=np.isnan(precip_data_aligned)

        # 3. Cálculos biofísicos
        st.write("Calculando variables biofísicas...")
        with np.errstate(invalid='ignore'):
            tma_raster = estimate_mean_annual_temp(dem_data)
            bat_raster = calculate_biotemperature(tma_raster, mean_latitude)
        st.write("Cálculos completados.")

        # 4. Clasificar píxeles
        st.write("Clasificando Zonas de Vida (Antioquia)...")
        classified_raster = np.full((dst_height, dst_width), 0, dtype=np.int16) # 0 para NoData
        valid_pixels = ~dem_mask & ~precip_mask & ~np.isnan(bat_raster) & ~np.isnan(precip_data_aligned)
        bat_values = bat_raster[valid_pixels]; ppt_values = precip_data_aligned[valid_pixels]
        vectorized_classify = np.vectorize(classify_holdridge_zone_antioquia); zone_ints = vectorized_classify(bat_values, ppt_values)
        classified_raster[valid_pixels] = zone_ints.astype(np.int16)
        st.write("Clasificación completada.")

        # --- APLICAR MÁSCARA (Usando _mask_geometry) ---
        if _mask_geometry is not None and not _mask_geometry.empty: # <-- GUION BAJO AÑADIDO AQUÍ
            st.write("Aplicando máscara de geometría...")
            try:
                mask_geometry_reproj = _mask_geometry.to_crs(dst_profile['crs']) # <-- GUION BAJO AÑADIDO AQUÍ
                temp_classified_path = "temp_classified_raster_mask.tif"
                output_profile_mask = dst_profile.copy(); output_profile_mask.update({'dtype': rasterio.int16, 'nodata': 0, 'count': 1})
                with rasterio.open(temp_classified_path, 'w', **output_profile_mask) as dst: dst.write(classified_raster, 1)
                with rasterio.open(temp_classified_path) as src:
                    masked_data, masked_transform = mask(src, mask_geometry_reproj, crop=False, nodata=0)
                os.remove(temp_classified_path)
                classified_raster = masked_data[0]
                st.write("Máscara aplicada.")
            except Exception as e_mask:
                st.warning(f"No se pudo aplicar la máscara de geometría: {e_mask}")
        # --- FIN BLOQUE MÁSCARA ---

        # 5. Preparar salida
        output_profile = dst_profile.copy()
        output_profile.update({'dtype': rasterio.int16, 'nodata': 0, 'count': 1})

        return classified_raster, output_profile, holdridge_int_to_name

    except Exception as e:
        st.error(f"Error generando mapa de zonas de vida: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None
