import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine
import streamlit as st
import os

# --- Constantes y Diccionarios ---
holdridge_zone_map_simplified = {
    "Nival": 1,
    "Tundra pluvial (tp-A)": 2, "Tundra húmeda (th-A)": 3, "Tundra seca (ts-A)": 4,
    "Páramo pluvial subalpino (pp-SA)": 5, "Páramo muy húmedo subalpino (pmh-SA)": 6, "Páramo seco subalpino (ps-SA)": 7,
    "Bosque pluvial Montano (bp-M)": 8, "Bosque muy húmedo Montano (bmh-M)": 9, "Bosque húmedo Montano (bh-M)": 10,
    "Bosque seco Montano (bs-M)": 11, "Monte espinoso Montano (me-M)": 12,
    "Bosque pluvial Premontano (bp-PM)": 13, "Bosque muy húmedo Premontano (bmh-PM)": 14,
    "Bosque húmedo Premontano (bh-PM)": 15, "Bosque seco Premontano (bs-PM)": 16, "Monte espinoso Premontano (me-PM)": 17,
    "Bosque pluvial Tropical (bp-T)": 18, "Bosque muy húmedo Tropical (bmh-T)": 19, "Bosque húmedo Tropical (bh-T)": 20,
    "Bosque seco Tropical (bs-T)": 21, "Monte espinoso Tropical (me-T)": 22,
    "Zona Desconocida": 0
}

# Invertir diccionario para búsquedas rápidas por ID
holdridge_int_to_name_simplified = {v: k for k, v in holdridge_zone_map_simplified.items()}

def classify_life_zone_alt_ppt(altitude, ppt):
    """
    Clasifica una celda según su altitud (m) y precipitación anual (mm).
    Lógica corregida para incluir Zonas 1, 7, 17 y 20 explícitamente.
    """
    # 0. Validación de datos inválidos
    if pd.isna(altitude) or pd.isna(ppt) or altitude < 0 or ppt <= 0:
        return 0
        
    # --- 1. PISO NIVAL (Zona 1) ---
    # Altura extrema, temperatura < 1.5°C. 
    # Se ajusta a 4500m+ para Andes Colombianos (Nevados).
    if altitude >= 4500:
        return 1  # Nival
        
    # --- 2. PISO ALPINO (Tundra / Superpáramo) ---
    # Altitud: 3800 a 4500 aprox.
    if altitude >= 3800:
        if ppt >= 1000: return 2   # Tundra pluvial
        elif ppt >= 500: return 3  # Tundra húmeda
        else: return 4             # Tundra seca
        
    # --- 3. PISO SUBALPINO (Páramo) ---
    # Altitud: 3000 a 3800 (Bajamos límite a 3000 para capturar páramos reales)
    if altitude >= 3000:
        if ppt >= 2000: return 5   # Páramo pluvial (pp-SA)
        elif ppt >= 1000: return 6 # Páramo muy húmedo (pmh-SA) - El más común
        else: return 7             # Páramo seco (ps-SA) -> CUBRE LA ZONA 7 (<1000mm)
        
    # --- 4. PISO MONTANO ---
    # Altitud: 2000 a 3000
    if altitude >= 2000:
        if ppt >= 4000: return 8   # Pluvial
        elif ppt >= 2000: return 9 # Muy húmedo
        elif ppt >= 1000: return 10 # Húmedo (bh-M)
        elif ppt >= 500: return 11  # Seco (bs-M)
        else: return 12             # Espinoso (me-M)
        
    # --- 5. PISO PREMONTANO ---
    # Altitud: 1000 a 2000 (Zona Cafetera típica)
    if altitude >= 1000:
        if ppt >= 4000: return 13   # Pluvial
        elif ppt >= 2000: return 14 # Muy húmedo
        elif ppt >= 1000: return 15 # Húmedo (bh-PM)
        elif ppt >= 500: return 16  # Seco (bs-PM)
        else: return 17             # Espinoso (me-PM) -> CUBRE LA ZONA 17 (<500mm)
        
    # --- 6. PISO TROPICAL (Basal) ---
    # Altitud: < 1000 msnm
    # Aquí solucionamos la aparición del Bosque Seco y Húmedo
    if ppt >= 8000: return 18       # Pluvial (Chocó extremo)
    elif ppt >= 4000: return 19     # Muy Húmedo (bmh-T) -> (Urabá, Amazonía piedemonte)
    elif ppt >= 2000: return 20     # Húmedo (bh-T) -> CUBRE LA ZONA 20 (Magdalena Medio húmedo)
    elif ppt >= 1000: return 21     # Seco (bs-T) -> CUBRE LA ZONA 21 (Caribe, Valles interandinos)
    else: return 22                 # Espinoso (me-T) -> (Guajira, Desierto Tatacoa)

def _resample_raster_to_shape(src_dataset, dst_shape, dst_transform, dst_crs=None, resampling=Resampling.average):
    dest = np.empty(dst_shape, dtype=np.float32)
    if dst_crs is None:
        dst_crs = src_dataset.crs
        
    reproject(
        source=rasterio.band(src_dataset, 1),
        destination=dest,
        src_transform=src_dataset.transform,
        src_crs=src_dataset.crs,
        src_nodata=src_dataset.nodata,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=resampling
    )
    return dest

def generate_life_zone_map(dem_path, precip_raster_path, mask_geometry=None, downscale_factor=4):
    """
    Genera mapa raster clasificado usando Altitud (DEM) y PPT.
    """
    try:
        if downscale_factor is None or downscale_factor <= 0:
            downscale_factor = 1
            
        # Abrir DEM
        with rasterio.open(dem_path) as dem_src:
            src_width = dem_src.width
            src_height = dem_src.height
            src_transform = dem_src.transform
            dem_crs = dem_src.crs
            
            dst_width = max(1, src_width // downscale_factor)
            dst_height = max(1, src_height // downscale_factor)
            
            scale_x = src_width / dst_width
            scale_y = src_height / dst_height
            dst_transform = src_transform * Affine.scale(scale_x, scale_y)
            
            dem_resampled = _resample_raster_to_shape(
                dem_src, (dst_height, dst_width), 
                dst_transform, dst_crs=dem_crs, resampling=Resampling.bilinear
            )

        # Abrir Precipitación
        with rasterio.open(precip_raster_path) as ppt_src:
            ppt_resampled = _resample_raster_to_shape(
                ppt_src, (dst_height, dst_width), 
                dst_transform, dst_crs=dem_crs, resampling=Resampling.average
            )

        # Máscara válida
        dem_mask = np.isnan(dem_resampled)
        ppt_mask = np.isnan(ppt_resampled)
        valid_mask = (~dem_mask) & (~ppt_mask) & np.isfinite(ppt_resampled)
        
        classified_raster = np.zeros((dst_height, dst_width), dtype=np.int16)
        
        if np.any(valid_mask):
            alt_values = dem_resampled[valid_mask]
            ppt_values = ppt_resampled[valid_mask]
            
            vectorized_classify = np.vectorize(classify_life_zone_alt_ppt)
            zone_ints = vectorized_classify(alt_values, ppt_values)
            classified_raster[valid_mask] = zone_ints.astype(np.int16)

        # Aplicar máscara de geometría (si existe)
        if mask_geometry is not None and not mask_geometry.empty:
            try:
                if hasattr(mask_geometry, "crs") and mask_geometry.crs and dem_crs and mask_geometry.crs != dem_crs:
                    mask_reproj = mask_geometry.to_crs(dem_crs)
                else:
                    mask_reproj = mask_geometry
                    
                shapes = [(geom, 1) for geom in mask_reproj.geometry]
                mask_raster = rasterize(
                    shapes,
                    out_shape=(dst_height, dst_width),
                    transform=dst_transform,
                    fill=0,
                    dtype=np.uint8
                )
                classified_raster = np.where(mask_raster == 1, classified_raster, 0)
            except Exception as e_mask:
                st.warning(f"No se pudo aplicar la máscara de geometría: {e_mask}")

        output_profile = {
            'driver': 'GTiff',
            'dtype': rasterio.int16,
            'nodata': 0,
            'width': dst_width,
            'height': dst_height,
            'count': 1,
            'crs': dem_crs,
            'transform': dst_transform
        }
        
        return classified_raster, output_profile, holdridge_int_to_name_simplified

    except Exception as e:
        st.error(f"Error generando mapa de zonas de vida: {e}")
        return None, None, None
