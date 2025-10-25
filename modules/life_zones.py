# modules/life_zones.py
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
import streamlit as st
import math
import os

# --- Constantes ---
LAPSE_RATE = 6.0
BASE_TEMP_SEA_LEVEL = 28.0

# --- Diccionario de Zonas de Vida (Según tabla Antioquia) ---
holdridge_zone_map_simplified = {
    "Nival": 1,
    "Tundra pluvial alpino (tp-A)": 2, "Tundra húmeda alpino (th-A)": 3, "Tundra seca alpino (ts-A)": 4,
    "Páramo pluvial subalpino (pp-SA)": 5, "Páramo muy húmedo subalpino (pmh-SA)": 6, "Páramo seco subalpino (ps-SA)": 7,
    "Bosque pluvial Montano (bp-M)": 8, "Bosque muy húmedo Montano (bmh-M)": 9, "Bosque húmedo Montano (bh-M)": 10,
    "Bosque seco Montano (bs-M)": 11, "Monte espinoso Montano (me-M)": 12,
    "Bosque pluvial Premontano (bp-PM)": 13, "Bosque muy húmedo Premontano (bmh-PM)": 14, "Bosque húmedo Premontano (bh-PM)": 15,
    "Bosque seco Premontano (bs-PM)": 16, "Monte espinoso Premontano (me-PM)": 17,
    "Bosque pluvial Tropical (bp-T)": 18, "Bosque muy húmedo Tropical (bmh-T)": 19, "Bosque húmedo Tropical (bh-T)": 20,
    "Bosque seco Tropical (bs-T)": 21, "Monte espinoso Tropical (me-T)": 22,
    "Zona Desconocida": 0
}
holdridge_int_to_name_simplified = {v: k for k, v in holdridge_zone_map_simplified.items()}

# --- Función de Clasificación (Altitud y PPT) ---
def classify_life_zone_alt_ppt(altitude, ppt):
    if pd.isna(altitude) or pd.isna(ppt) or altitude < 0 or ppt <= 0: return 0
    zone_id = 0
    if altitude > 4200: zone_id = 1
    elif altitude >= 3700:
        if ppt >= 1500: zone_id = 2
        elif ppt >= 750: zone_id = 3
        else: zone_id = 4
    elif altitude >= 3200:
        if ppt >= 2000: zone_id = 5
        elif ppt >= 1000: zone_id = 6
        else: zone_id = 7
    elif altitude >= 2000:
        if ppt >= 4000: zone_id = 8
        elif ppt >= 2000: zone_id = 9
        elif ppt >= 1000: zone_id = 10
        elif ppt >= 500: zone_id = 11
        else: zone_id = 12
    elif altitude >= 1000:
        if ppt >= 4000: zone_id = 13
        elif ppt >= 2000: zone_id = 14
        elif ppt >= 1000: zone_id = 15
        elif ppt >= 500: zone_id = 16
        else: zone_id = 17
    else: # altitude < 1000
        if ppt >= 4000: zone_id = 18
        elif ppt >= 2000: zone_id = 19
        elif ppt >= 1000: zone_id = 20
        elif ppt >= 500: zone_id = 21
        else: zone_id = 22
    return zone_id

# --- Función Principal (SIN CACHÉ, SIN GUION BAJO) ---
# @st.cache_data(show_spinner="Generando mapa de Zonas de Vida...") # Caché eliminado
def generate_life_zone_map(dem_path, precip_raster_path, mask_geometry=None, downscale_factor=4): # <-- SIN GUION BAJO
    """Genera un mapa raster clasificado de Zonas de Vida usando Altitud y PPT."""
    try:
        if downscale_factor <= 0: downscale_factor = 1
        with rasterio.open(dem_path) as dem_src:
            src_profile=dem_src.profile; src_crs=dem_src.crs; src_transform=dem_src.transform; nodata_dem=dem_src.nodata
            dst_height=src_profile['height']//downscale_factor; dst_width=src_profile['width']//downscale_factor
            dst_transform=src_transform*src_transform.scale((src_profile['width']/dst_width),(src_profile['height']/dst_height))
            dst_profile=src_profile.copy(); dst_profile.update({'height':dst_height,'width':dst_width,'transform':dst_transform,'dtype':rasterio.float32,'nodata':np.nan})
            altitude_data=np.empty((dst_height,dst_width),dtype=rasterio.float32)
            reproject(source=rasterio.band(dem_src,1),destination=altitude_data,src_transform=src_transform,src_crs=src_crs,src_nodata=nodata_dem,dst_transform=dst_transform,dst_crs=src_crs,dst_nodata=np.nan,resampling=Resampling.average)
            altitude_mask=np.isnan(altitude_data)
        with rasterio.open(precip_raster_path) as precip_src:
            precip_data_aligned=np.empty((dst_height,dst_width),dtype=rasterio.float32)
            reproject(source=rasterio.band(precip_src,1),destination=precip_data_aligned,src_transform=precip_src.transform,src_crs=precip_src.crs,src_nodata=precip_src.nodata,dst_transform=dst_transform,dst_crs=src_crs,dst_nodata=np.nan,resampling=Resampling.bilinear)
            precip_mask=np.isnan(precip_data_aligned)

        # Ocultar st.write para reducir "ruido"
        # st.write("Clasificando Zonas de Vida (Alt/PPT)...") 
        classified_raster = np.full((dst_height, dst_width), 0, dtype=np.int16)
        valid_pixels = ~altitude_mask & ~precip_mask & ~np.isnan(precip_data_aligned)
        alt_values = altitude_data[valid_pixels]; ppt_values = precip_data_aligned[valid_pixels]
        vectorized_classify = np.vectorize(classify_life_zone_alt_ppt); zone_ints = vectorized_classify(alt_values, ppt_values)
        classified_raster[valid_pixels] = zone_ints.astype(np.int16)
        # st.write("Clasificación completada.")

        # --- APLICAR MÁSCARA (Usando mask_geometry SIN guion bajo) ---
        if mask_geometry is not None and not mask_geometry.empty:
            # st.write("Aplicando máscara de geometría...")
            try:
                mask_geometry_reproj = mask_geometry.to_crs(dst_profile['crs'])
                temp_classified_path = "temp_classified_raster_mask.tif"
                output_profile_mask = dst_profile.copy(); output_profile_mask.update({'dtype': rasterio.int16, 'nodata': 0, 'count': 1})
                with rasterio.open(temp_classified_path, 'w', **output_profile_mask) as dst: dst.write(classified_raster, 1)
                with rasterio.open(temp_classified_path) as src:
                    masked_data, masked_transform = mask(src, mask_geometry_reproj, crop=False, nodata=0)
                if os.path.exists(temp_classified_path): os.remove(temp_classified_path)
                classified_raster = masked_data[0]
                # st.write("Máscara aplicada.")
            except Exception as e_mask:
                st.warning(f"No se pudo aplicar la máscara de geometría: {e_mask}")
                if 'temp_classified_path' in locals() and os.path.exists(temp_classified_path): os.remove(temp_classified_path)

        # --- Preparar salida ---
        output_profile = dst_profile.copy()
        output_profile.update({'dtype': rasterio.int16, 'nodata': 0, 'count': 1})
        return classified_raster, output_profile, holdridge_int_to_name_simplified

    except Exception as e:
        st.error(f"Error generando mapa de zonas de vida: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None
