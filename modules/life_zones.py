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
    "Tundra pluvial alpino (tp-A)": 2, "Tundra húmeda alpino (th-A)": 3, "Tundra seca alpino (ts-A)": 4,
    "Páramo pluvial subalpino (pp-SA)": 5, "Páramo muy húmedo subalpino (pmh-SA)": 6, "Páramo seco subalpino (ps-SA)": 7,
    "Bosque pluvial Montano (bp-M)": 8, "Bosque muy húmedo Montano (bmh-M)": 9, "Bosque húmedo Montano (bh-M)": 10,
    "Bosque seco Montano (bs-M)": 11, "Monte espinoso Montano (me-M)": 12,
    "Bosque pluvial Premontano (bp-PM)": 13, "Bosque muy húmedo Premontano (bmh-PM)": 14,
    "Bosque húmedo Premontano (bh-PM)": 15, "Bosque seco Premontano (bs-PM)": 16, "Monte espinoso Premontano (me-PM)": 17,
    "Bosque pluvial Tropical (bp-T)": 18, "Bosque muy húmedo Tropical (bmh-T)": 19, "Bosque húmedo Tropical (bh-T)": 20,
    "Bosque seco Tropical (bs-T)": 21, "Monte espinoso Tropical (me-T)": 22,
    "Zona Desconocida": 0
}

holdridge_int_to_name_simplified = {v: k for k, v in holdridge_zone_map_simplified.items()}

def classify_life_zone_alt_ppt(altitude, ppt):
    """Clasifica una celda según su altitud (m) y precipitación anual (mm)."""
    if pd.isna(altitude) or pd.isna(ppt) or altitude < 0 or ppt <= 0:
        return 0
        
    if altitude > 4200:
        return 1
        
    if altitude >= 3700:
        if ppt >= 1500: return 2
        elif ppt >= 750: return 3
        else: return 4
        
    if altitude >= 3200:
        if ppt >= 2000: return 5
        elif ppt >= 1000: return 6
        else: return 7
        
    if altitude >= 2000:
        if ppt >= 4000: return 8
        elif ppt >= 2000: return 9
        elif ppt >= 1000: return 10
        elif ppt >= 500: return 11
        else: return 12
        
    if altitude >= 1000:
        if ppt >= 4000: return 13
        elif ppt >= 2000: return 14
        elif ppt >= 1000: return 15
        elif ppt >= 500: return 16
        else: return 17
        
    # altitude < 1000
    if ppt >= 4000: return 18
    if ppt >= 2000: return 19
    if ppt >= 1000: return 20
    if ppt >= 500: return 21
    return 22

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
