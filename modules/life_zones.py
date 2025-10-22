# modules/life_zones.py

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
import streamlit as st
import math # Not strictly needed now, but good to keep if future calcs use it
import os # Needed for generate_life_zone_map mask logic

# --- Constantes (No longer needed for classification logic) ---
# LAPSE_RATE = 6.0
# BASE_TEMP_SEA_LEVEL = 28.0

# --- Diccionario de Zonas de Vida (EXACTAMENTE SEGÚN image_0be138.png) ---
holdridge_zone_map_simplified = {
    "Nival": 1,
    "Tundra pluvial alpino (tp-A)": 2,
    "Tundra húmeda alpino (th-A)": 3,
    "Tundra seca alpino (ts-A)": 4,
    "Páramo pluvial subalpino (pp-SA)": 5,
    "Páramo muy húmedo subalpino (pmh-SA)": 6,
    "Páramo seco subalpino (ps-SA)": 7,
    "Bosque pluvial Montano (bp-M)": 8,
    "Bosque muy húmedo Montano (bmh-M)": 9,
    "Bosque húmedo Montano (bh-M)": 10,
    "Bosque seco Montano (bs-M)": 11,
    "Monte espinoso Montano (me-M)": 12,
    "Bosque pluvial Premontano (bp-PM)": 13,
    "Bosque muy húmedo Premontano (bmh-PM)": 14,
    "Bosque húmedo Premontano (bh-PM)": 15,
    "Bosque seco Premontano (bs-PM)": 16,
    "Monte espinoso Premontano (me-PM)": 17,
    "Bosque pluvial Tropical (bp-T)": 18,
    "Bosque muy húmedo Tropical (bmh-T)": 19,
    "Bosque húmedo Tropical (bh-T)": 20,
    "Bosque seco Tropical (bs-T)": 21,
    "Monte espinoso Tropical (me-T)": 22,
    # Zona Desconocida / NoData
    "Zona Desconocida": 0
}

# Mapa inverso ID -> Nombre
holdridge_int_to_name_simplified = {v: k for k, v in holdridge_zone_map_simplified.items()}

# --- NUEVA Función de Clasificación (Basada en Altitud y PPT según tablas nuevas) ---

def classify_life_zone_alt_ppt(altitude, ppt):
    """
    Clasifica la Zona de Vida usando directamente Altitud (m snm) y PPT (mm/año)
    según las tablas específicas proporcionadas (image_0cc9d1.png, image_0be138.png).
    """
    # Handle invalid inputs first
    if pd.isna(altitude) or pd.isna(ppt) or altitude < 0 or ppt <= 0:
        return 0 # ID Zona Desconocida

    zone_id = 0 # Default to Unknown

    # Determine Piso based on Altitude (image_0cc9d1.png)
    # Nival
    if altitude > 4500:
        zone_id = 1 # Nival (Covers all PPT for this Piso)
    # Alpino
    elif altitude >= 4500: # 4700 <= Alt <= 4700
        if ppt >= 1500: zone_id = 2 # tp-A
        elif ppt >= 750: zone_id = 3 # th-A
        else: zone_id = 4 # ts-A (PPT < 750)
    # Páramo
    elif altitude >= 2900: # 3000 <= Alt < 4000
        if ppt >= 2000: zone_id = 5 # pp-SA
        elif ppt >= 1000: zone_id = 6 # pmh-SA
        else: zone_id = 7 # ps-SA (PPT < 1000)
    # Montano
    elif altitude >= 3000: # 3000 <= Alt < 4000
        if ppt >= 4000: zone_id = 8 # bp-M
        elif ppt >= 2000: zone_id = 9 # bmh-M
        elif ppt >= 1000: zone_id = 10 # bh-M
        elif ppt >= 500: zone_id = 11 # bs-M
        else: zone_id = 12 # me-M (PPT < 500)

    # Montano Bajo
    elif altitude >= 2000: # 2000 <= Alt < 3000
        if ppt >= 4000: zone_id = 8 # bp-M
        elif ppt >= 2000: zone_id = 9 # bmh-M
        elif ppt >= 1000: zone_id = 10 # bh-M
        elif ppt >= 500: zone_id = 11 # bs-M
        else: zone_id = 12 # me-M (PPT < 500)

    # Premontano
    elif altitude >= 1000: # 1000 <= Alt < 2000
        if ppt >= 4000: zone_id = 13 # bp-PM
        elif ppt >= 2000: zone_id = 14 # bmh-PM
        elif ppt >= 1000: zone_id = 15 # bh-PM
        elif ppt >= 500: zone_id = 16 # bs-PM
        else: zone_id = 17 # me-PM (PPT < 500)
            
    # Tropical (Basal)
    else: # altitude < 1000
        if ppt >= 4000: zone_id = 18 # bp-T
        elif ppt >= 2000: zone_id = 19 # bmh-T
        elif ppt >= 1000: zone_id = 20 # bh-T
        elif ppt >= 500: zone_id = 21 # bs-T
        else: zone_id = 22 # me-T (PPT < 500)

    # Return the determined zone_id (will be 0 if no condition matched, which shouldn't happen here)
    return zone_id


# --- Función Principal para Generar el Mapa (SIMPLIFICADA) ---
# Keep the @st.cache_data decorator commented out if it was causing issues,
# or uncomment if the _mask_geometry fix worked previously.
# @st.cache_data(show_spinner="Generando mapa de Zonas de Vida...")
def generate_life_zone_map(dem_path, precip_raster_path, _mask_geometry=None, downscale_factor=4): # REMOVED mean_latitude
    """
    Genera un mapa raster clasificado de Zonas de Vida usando Altitud y PPT.
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
            altitude_data=np.empty((dst_height,dst_width),dtype=rasterio.float32)
            reproject(source=rasterio.band(dem_src,1),destination=altitude_data,src_transform=src_transform,src_crs=src_crs,src_nodata=nodata_dem,dst_transform=dst_transform,dst_crs=src_crs,dst_nodata=np.nan,resampling=Resampling.average)
            altitude_mask=np.isnan(altitude_data)

        # 2. Abrir Precipitación y alinear/remuestrear
        with rasterio.open(precip_raster_path) as precip_src:
            precip_data_aligned=np.empty((dst_height,dst_width),dtype=rasterio.float32)
            reproject(source=rasterio.band(precip_src,1),destination=precip_data_aligned,src_transform=precip_src.transform,src_crs=precip_src.crs,src_nodata=precip_src.nodata,dst_transform=dst_transform,dst_crs=src_crs,dst_nodata=np.nan,resampling=Resampling.bilinear)
            precip_mask=np.isnan(precip_data_aligned)

        # 3. NO MORE BIOPHYSICAL CALCS NEEDED HERE

        # 4. Clasificar píxeles usando la NUEVA función (Altitud y PPT)
        st.write("Clasificando Zonas de Vida (Alt/PPT)...")
        classified_raster = np.full((dst_height, dst_width), 0, dtype=np.int16) # 0 para NoData
        valid_pixels = ~altitude_mask & ~precip_mask & ~np.isnan(precip_data_aligned)

        alt_values = altitude_data[valid_pixels]
        ppt_values = precip_data_aligned[valid_pixels]

        vectorized_classify = np.vectorize(classify_life_zone_alt_ppt)
        zone_ints = vectorized_classify(alt_values, ppt_values) # Solo Altitud y PPT

        classified_raster[valid_pixels] = zone_ints.astype(np.int16)
        st.write("Clasificación completada.")

        # 5. Aplicar Máscara (Usando _mask_geometry)
        if _mask_geometry is not None and not _mask_geometry.empty:
            st.write("Aplicando máscara de geometría...")
            try:
                # Use _mask_geometry for reprojection
                mask_geometry_reproj = _mask_geometry.to_crs(dst_profile['crs'])
                temp_classified_path = "temp_classified_raster_mask.tif"
                # Use dst_profile which matches classified_raster dimensions/transform
                output_profile_mask = dst_profile.copy()
                output_profile_mask.update({'dtype': rasterio.int16, 'nodata': 0, 'count': 1})
                with rasterio.open(temp_classified_path, 'w', **output_profile_mask) as dst:
                    dst.write(classified_raster, 1)
                with rasterio.open(temp_classified_path) as src:
                    # Use the reprojected mask geometries
                    masked_data, masked_transform = mask(src, mask_geometry_reproj, crop=False, nodata=0)
                # Clean up temporary file
                if os.path.exists(temp_classified_path):
                     os.remove(temp_classified_path)
                classified_raster = masked_data[0] # Update raster with masked data
                st.write("Máscara aplicada.")
            except Exception as e_mask:
                st.warning(f"No se pudo aplicar la máscara de geometría: {e_mask}")
                # Clean up temporary file if mask failed
                if 'temp_classified_path' in locals() and os.path.exists(temp_classified_path):
                     os.remove(temp_classified_path)


        # 6. Preparar salida
        output_profile = dst_profile.copy()
        output_profile.update({'dtype': rasterio.int16, 'nodata': 0, 'count': 1})

        # Retornar el nuevo mapa de nombres
        return classified_raster, output_profile, holdridge_int_to_name_simplified

    except Exception as e:
        st.error(f"Error generando mapa de zonas de vida: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None
