import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
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

# --- PALETA DE COLORES OFICIAL (HEX) ---
holdridge_colors = {
    1: "#FFFFFF",  # Nival (Blanco)
    # Alpino (Azules hielo/Grises)
    2: "#B0E0E6", 3: "#87CEEB", 4: "#708090",
    # Páramo (Violetas/Morados - Típico en mapas IGAC)
    5: "#8A2BE2", 6: "#9370DB", 7: "#D8BFD8",
    # Montano (Verdes oscuros a amarillos)
    8: "#00008B", 9: "#006400", 10: "#228B22", 11: "#9ACD32", 12: "#F0E68C",
    # Premontano (Verdes medios)
    13: "#0000CD", 14: "#008000", 15: "#32CD32", 16: "#FFFF00", 17: "#DAA520",
    # Tropical (Verdes selva a Naranjas secos)
    18: "#191970", 19: "#2E8B57", 20: "#7CFC00", 
    21: "#FFA500", # BS-T (Naranja Clásico)
    22: "#FF4500", # Espinoso (Rojo/Naranja fuerte)
    0:  "#000000"  # Desconocido
}

# Invertir para buscar por ID
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
    Genera mapa raster clasificado de Zonas de Vida.
    
    MEJORA CRÍTICA: Fuerza la reproyección a WGS84 (EPSG:4326) para que el mapa
    pueda visualizarse correctamente sobre mapas base web (OpenStreetMap/Carto).
    """
    try:
        # Validación básica de factor de escala
        if downscale_factor is None or downscale_factor <= 0:
            downscale_factor = 1

        # Sistema de coordenadas objetivo: WGS84 (Latitud/Longitud)
        dst_crs = 'EPSG:4326'

        # --- 1. PROCESAR DEM (Base Maestra) ---
        with rasterio.open(dem_path) as dem_src:
            # Calcular nuevas dimensiones reducidas
            dst_width = max(1, dem_src.width // downscale_factor)
            dst_height = max(1, dem_src.height // downscale_factor)
            
            # Calcular la transformación afín para pasar de Metros (origen) a Grados (destino)
            # manteniendo el encuadre correcto.
            dst_transform, dst_width, dst_height = calculate_default_transform(
                dem_src.crs, 
                dst_crs, 
                dem_src.width, 
                dem_src.height, 
                *dem_src.bounds,
                dst_width=dst_width,
                dst_height=dst_height
            )

            # Crear array destino para DEM
            dem_resampled = np.empty((dst_height, dst_width), dtype=np.float32)
            
            # Reproyectar DEM
            reproject(
                source=rasterio.band(dem_src, 1),
                destination=dem_resampled,
                src_transform=dem_src.transform,
                src_crs=dem_src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear
            )

        # --- 2. PROCESAR PRECIPITACIÓN (Esclavo) ---
        # Usamos exactamente el mismo transform y shape del DEM para asegurar alineación pixel a pixel
        with rasterio.open(precip_raster_path) as ppt_src:
            ppt_resampled = np.empty((dst_height, dst_width), dtype=np.float32)
            
            reproject(
                source=rasterio.band(ppt_src, 1),
                destination=ppt_resampled,
                src_transform=ppt_src.transform,
                src_crs=ppt_src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.average # Promedio es mejor para lluvia
            )

        # --- 3. CÁLCULO DE ZONAS DE VIDA ---
        # Crear máscaras para ignorar datos inválidos (NaN o valores absurdos)
        dem_mask = np.isnan(dem_resampled)
        ppt_mask = np.isnan(ppt_resampled)
        valid_mask = (~dem_mask) & (~ppt_mask) & (dem_resampled > -500) & (ppt_resampled >= 0)
        
        classified_raster = np.zeros((dst_height, dst_width), dtype=np.int16)
        
        if np.any(valid_mask):
            alt_values = dem_resampled[valid_mask]
            ppt_values = ppt_resampled[valid_mask]
            
            # Aplicar clasificación vectorizada (Rápida)
            # Asegúrate que 'classify_life_zone_alt_ppt' esté definida arriba o importada
            vectorized_classify = np.vectorize(classify_life_zone_alt_ppt)
            zone_ints = vectorized_classify(alt_values, ppt_values)
            classified_raster[valid_mask] = zone_ints.astype(np.int16)

        # --- 4. APLICAR MÁSCARA DE GEOMETRÍA (Corte por Cuenca) ---
        if mask_geometry is not None and not mask_geometry.empty:
            try:
                # Asegurar que la geometría de corte también esté en Lat/Lon
                mask_reproj = mask_geometry
                if mask_geometry.crs and mask_geometry.crs.to_string() != dst_crs:
                    mask_reproj = mask_geometry.to_crs(dst_crs)

                shapes = [(geom, 1) for geom in mask_reproj.geometry]
                
                # Rasterizar la geometría sobre la grilla existente
                mask_raster = rasterize(
                    shapes,
                    out_shape=(dst_height, dst_width),
                    transform=dst_transform,
                    fill=0,
                    dtype=np.uint8
                )
                
                # Dejar en 0 todo lo que esté fuera de la máscara
                classified_raster = np.where(mask_raster == 1, classified_raster, 0)
                
            except Exception as e_mask:
                st.warning(f"Advertencia recortando máscara: {e_mask}")

        # --- 5. PREPARAR RESULTADOS ---
        output_profile = {
            'driver': 'GTiff',
            'dtype': rasterio.int16,
            'nodata': 0,
            'width': dst_width,
            'height': dst_height,
            'count': 1,
            'crs': dst_crs,
            'transform': dst_transform
        }
        
        return classified_raster, output_profile, holdridge_int_to_name_simplified, holdridge_colors

    except Exception as e:
        st.error(f"Error generando mapa de zonas de vida: {e}")
        return None, None, None
