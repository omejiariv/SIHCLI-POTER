# modules/life_zones.py
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
import streamlit as st
import math

# --- Constantes (Ya no se usan para T°, pero se mantienen por si acaso) ---
LAPSE_RATE = 6.0
BASE_TEMP_SEA_LEVEL = 28.0

# --- Diccionario de Zonas de Vida (SEGÚN NUEVA TABLA image_0d3dfa.png) ---
holdridge_zone_map_simplified = {
    "Nival": 1,
    "Alpino": 2,
    "Páramo": 3,
    "Bosque húmedo Montano (bh-M)": 4,
    "Bosque seco Montano (bs-M)": 5,
    "Bosque muy seco Montano (bms-M)": 6, # Ajustado nombre
    "Bosque húmedo Premontano (bh-PM)": 7,
    "Bosque seco Premontano (bs-PM)": 8,
    "Bosque húmedo Tropical (bh-T)": 9,
    "Bosque seco Tropical (bs-T)": 10,
    # Zona Desconocida / NoData
    "Zona Desconocida": 0
}

# Mapa inverso ID -> Nombre
holdridge_int_to_name_simplified = {v: k for k, v in holdridge_zone_map_simplified.items()}

# --- NUEVA Función de Clasificación (Basada en Altitud y PPT) ---

def classify_life_zone_alt_ppt(altitude, ppt):
    """
    Clasifica la Zona de Vida usando directamente Altitud (m snm) y PPT (mm/año)
    según la tabla específica proporcionada (image_0d3dfa.png).
    """
    if pd.isna(altitude) or pd.isna(ppt) or altitude < 0 or ppt <= 0:
        return 0 # ID Zona Desconocida

    zone_id = 0 # Default

    # Nival
    if altitude > 4200:
        zone_id = 1
    # Alpino
    elif altitude >= 3700: # Altura <= 4200 ya implícita
        zone_id = 2
    # Páramo
    elif altitude >= 3200: # Altura < 3700 ya implícita
        zone_id = 3
    # Montano
    elif altitude >= 2000: # Altura < 3200 ya implícita
        if ppt >= 1000: zone_id = 4 # bh-M
        elif ppt >= 500: zone_id = 5 # bs-M
        else: zone_id = 6 # bms-M (PPT < 500)
    # Premontano
    elif altitude >= 1000: # Altura < 2000 ya implícita
        if ppt >= 1000: zone_id = 7 # bh-PM
        else: zone_id = 8 # bs-PM (PPT < 1000)
    # Tropical (Basal)
    else: # altitude < 1000
        if ppt >= 1000: zone_id = 9 # bh-T
        else: zone_id = 10 # bs-T (PPT < 1000)

    return zone_id


# --- Función Principal para Generar el Mapa (SIMPLIFICADA) ---
@st.cache_data(show_spinner="Generando mapa de Zonas de Vida...")
# REMOVIDO mean_latitude, AÑADIDO guion bajo a _mask_geometry
def generate_life_zone_map(dem_path, precip_raster_path, _mask_geometry=None, downscale_factor=4):
    """
    Genera un mapa raster clasificado de Zonas de Vida usando Altitud y PPT.
    """
    try:
        # Factor de reescalado
        if downscale_factor <= 0: downscale_factor = 1

        # 1. Abrir DEM y remuestrear (sin cambios en lógica, solo variable 'altitude_data')
        with rasterio.open(dem_path) as dem_src:
            src_profile=dem_src.profile; src_crs=dem_src.crs; src_transform=dem_src.transform; nodata_dem=dem_src.nodata
            dst_height=src_profile['height']//downscale_factor; dst_width=src_profile['width']//downscale_factor
            dst_transform=src_transform*src_transform.scale((src_profile['width']/dst_width),(src_profile['height']/dst_height))
            dst_profile=src_profile.copy(); dst_profile.update({'height':dst_height,'width':dst_width,'transform':dst_transform,'dtype':rasterio.float32,'nodata':np.nan})
            altitude_data=np.empty((dst_height,dst_width),dtype=rasterio.float32) # Renombrado
            reproject(source=rasterio.band(dem_src,1),destination=altitude_data,src_transform=src_transform,src_crs=src_crs,src_nodata=nodata_dem,dst_transform=dst_transform,dst_crs=src_crs,dst_nodata=np.nan,resampling=Resampling.average)
            altitude_mask=np.isnan(altitude_data) # Renombrado

        # 2. Abrir Precipitación y alinear/remuestrear (sin cambios)
        with rasterio.open(precip_raster_path) as precip_src:
            precip_data_aligned=np.empty((dst_height,dst_width),dtype=rasterio.float32)
            reproject(source=rasterio.band(precip_src,1),destination=precip_data_aligned,src_transform=precip_src.transform,src_crs=precip_src.crs,src_nodata=precip_src.nodata,dst_transform=dst_transform,dst_crs=src_crs,dst_nodata=np.nan,resampling=Resampling.bilinear)
            precip_mask=np.isnan(precip_data_aligned)

        # 3. Cálculos biofísicos YA NO SON NECESARIOS (TMA, BAT, PET, PER)

        # 4. Clasificar píxeles usando la NUEVA función (Altitud y PPT)
        st.write("Clasificando Zonas de Vida (Alt/PPT)...")
        classified_raster = np.full((dst_height, dst_width), 0, dtype=np.int16) # 0 para NoData
        # Píxeles válidos donde tenemos Altitud y PPT
        valid_pixels = ~altitude_mask & ~precip_mask & ~np.isnan(precip_data_aligned) # Simplificado

        alt_values = altitude_data[valid_pixels] # Usar altitud directamente
        ppt_values = precip_data_aligned[valid_pixels]

        # Vectorizar la NUEVA función de clasificación
        vectorized_classify = np.vectorize(classify_life_zone_alt_ppt)
        zone_ints = vectorized_classify(alt_values, ppt_values) # Solo Altitud y PPT

        classified_raster[valid_pixels] = zone_ints.astype(np.int16)
        st.write("Clasificación completada.")

        # 5. Aplicar Máscara (Usando _mask_geometry)
        if _mask_geometry is not None and not _mask_geometry.empty:
            st.write("Aplicando máscara de geometría...")
            try:
                mask_geometry_reproj = _mask_geometry.to_crs(dst_profile['crs'])
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
