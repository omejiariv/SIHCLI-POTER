# modules/life_zones.py
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import streamlit as st
import math # Para logaritmos

# --- Constantes (Ajustar según tu región si es necesario) ---
LAPSE_RATE = 6.5 # Tasa de lapso estándar (°C / 1000m)
# Ajustar T° base según datos locales si es posible
BASE_TEMP_SEA_LEVEL = 28.0 # Temp base estimada a nivel del mar (°C) 
# Factor de ajuste simple por latitud (puede requerir calibración)
# Holdridge original sugiere un ajuste, pero es complejo. Usaremos uno simple.
# Valor negativo porque T° disminuye al alejarse del ecuador.
# Podríamos usar 0 si asumimos que BASE_TEMP_SEA_LEVEL ya es para la latitud de interés.
LATITUDE_ADJUSTMENT_FACTOR = 0.0 # -0.1 # Grados C por grado de latitud desde el ecuador

# --- Funciones de Cálculo MEJORADAS ---

def estimate_mean_annual_temp(elevation_m):
    """Estima la Temperatura Media Anual (TMA) basada en la altitud."""
    estimated_temp = BASE_TEMP_SEA_LEVEL - (LAPSE_RATE * elevation_m / 1000.0)
    return np.maximum(estimated_temp, -15) # Límite inferior más realista para alta montaña

def calculate_biotemperature(mean_annual_temp, latitude):
    """
    Estima la Biotemperature Anual Media (BAT) a partir de la TMA y Latitud.
    Aproximación: Clampea TMA entre 0-30 Y aplica un ajuste simple por latitud.
    Holdridge: Suma de T medias diarias > 0 / 365. Si T>30, se trata como 30 en la suma.
    Esto es difícil sin datos diarios/mensuales.
    """
    # Clampear temperaturas base Holdridge (0 a 30)
    clamped_temp = np.clip(mean_annual_temp, 0, 30)
    
    # Ajuste simple por latitud (opcional, puede desactivarse con LATITUDE_ADJUSTMENT_FACTOR = 0)
    # Asume que la T° base es ecuatorial y disminuye con latitud.
    lat_adjustment = LATITUDE_ADJUSTMENT_FACTOR * np.abs(latitude)
    
    # Aplicar ajuste y asegurar que BAT no sea negativa
    biotemp = np.maximum(0, clamped_temp + lat_adjustment) 
    
    # Otra corrección Holdridge: si TMA < 0, BAT es 0.
    biotemp = np.where(mean_annual_temp < 0, 0, biotemp)
    
    return biotemp

# --- calculate_pet y calculate_per SIN CAMBIOS ---
def calculate_pet(biotemperature):
    return 58.93 * biotemperature

def calculate_per(pet, precipitation):
    precipitation = np.maximum(precipitation, 1) 
    per = pet / precipitation
    return per 

# --- Diccionario de Zonas de Vida (REVISADO Y AMPLIADO) ---
# Basado en el diagrama estándar y tu tabla. Reordenado por ID numérico.
holdridge_zone_map = {
    # Nival / Polar (BAT < 1.5)
    "Nival / Hielo": 1,
    # Alpino / Subnival (1.5 <= BAT < 3)
    "Tundra seca Alpina": 2,          # Semiárido a Superárido
    "Tundra húmeda Alpina": 3,       # Húmedo a Subhúmedo
    "Tundra pluvial Alpina": 4,        # Perhúmedo a Superhúmedo
    # Subalpino / Páramo (3 <= BAT < 6)
    "Páramo seco Subalpino": 5,        # Semiárido a Arido
    "Páramo húmedo Subalpino": 6,     # Húmedo a Subhúmedo (Tu "muy húmedo"?)
    "Páramo pluvial Subalpino": 7,     # Perhúmedo a Superhúmedo
    # Montano (6 <= BAT < 12)
    "Estepa Montano": 11,             # Semiárido
    "Bosque seco Montano": 12,        # Subhúmedo
    "Bosque húmedo Montano": 13,       # Húmedo
    "Bosque muy húmedo Montano": 14,   # Perhúmedo
    "Bosque pluvial Montano": 15,      # Superhúmedo
    "Estepa espinosa Montano": 10,     # Árido (Añadido)
    "Desierto Montano": 9,           # Perárido a Superárido (Añadido)
    # Montano Bajo / Premontano (12 <= BAT < 18)
    "Monte espinoso Montano Bajo": 18, # Árido
    "Estepa espinosa Montano Bajo": 17, # Semiárido
    "Bosque seco Montano Bajo": 16,    # Subhúmedo
    "Bosque húmedo Montano Bajo": 15,   # Húmedo
    "Bosque muy húmedo Montano Bajo": 14,# Perhúmedo
    "Bosque pluvial Montano Bajo": 13,  # Superhúmedo
    "Desierto Montano Bajo": 19,       # Perárido a Superárido (Añadido)
    # Subtropical (18 <= BAT < 24) - Equivalente a Basal cálido?
    "Desierto Subtropical": 28,         # Perárido a Superárido
    "Monte espinoso Subtropical": 27,   # Árido
    "Estepa espinosa Subtropical": 26,  # Semiárido
    "Bosque seco Subtropical": 25,     # Subhúmedo
    "Bosque húmedo Subtropical": 24,    # Húmedo
    "Bosque muy húmedo Subtropical": 23,# Perhúmedo
    "Bosque pluvial Subtropical": 22,   # Superhúmedo
    # Tropical (BAT >= 24) - Equivalente a Basal muy cálido?
    "Desierto Tropical": 35,            # Perárido a Superárido
    "Monte espinoso Tropical": 34,      # Árido
    "Estepa espinosa Tropical": 33,     # Semiárido
    "Bosque seco Tropical": 32,        # Subhúmedo
    "Bosque húmedo Tropical": 31,       # Húmedo
    "Bosque muy húmedo Tropical": 30,   # Perhúmedo
    "Bosque pluvial Tropical": 29,      # Superhúmedo
    # Otros
    "Zona Desconocida / NoData": 0
}

# Mapa inverso ID -> Nombre
holdridge_int_to_name = {v: k for k, v in holdridge_zone_map.items()}

# --- Función de Clasificación REESCRITA ---

def classify_holdridge_zone_detailed(bat, ppt, per):
    """
    Clasifica la Zona de Vida usando lógica más cercana al diagrama hexagonal.
    Utiliza umbrales logarítmicos para PPT y PER donde sea apropiado.
    """
    
    # Manejar valores no válidos o extremos
    if pd.isna(bat) or pd.isna(ppt) or pd.isna(per) or bat < 0 or ppt <= 0:
        return "Zona Desconocida / NoData"

    # 1. Determinar Piso Altitudinal / Región Latitudinal (Basado en BAT)
    if bat < 1.5: piso = "Nival"
    elif bat < 3: piso = "Alpino"
    elif bat < 6: piso = "Subalpino"
    elif bat < 12: piso = "Montano"
    elif bat < 18: piso = "Montano Bajo"
    elif bat < 24: piso = "Subtropical"
    else: piso = "Tropical"

    # 2. Determinar Provincia de Humedad (Basado en PER)
    # Umbrales del diagrama (aproximados, escala log base 2)
    if per > 32.0: humidity_province = "Superárido"
    elif per > 16.0: humidity_province = "Perárido"
    elif per > 8.0: humidity_province = "Árido"
    elif per > 4.0: humidity_province = "Semiárido"
    elif per > 2.0: humidity_province = "Subhúmedo"
    elif per > 1.0: humidity_province = "Húmedo"
    elif per > 0.5: humidity_province = "Perhúmedo"
    elif per > 0.25: humidity_province = "Superhúmedo (Pluvial)" # Bosque muy húmedo en algunos diagramas
    else: humidity_province = "Superhúmedo++ (Rain Forest)" # El más húmedo

    # 3. Asignar Zona de Vida Específica
    # Esta es la lógica principal que mapea combinaciones a nombres.
    # Es compleja y requiere cuidadosa traducción del diagrama.
    # Ejemplo parcial mejorado:

    zone_name = "Zona Desconocida / NoData" # Default

    if piso == "Nival": zone_name = "Nival / Hielo"
    
    elif piso == "Alpino":
        if humidity_province in ["Superhúmedo++", "Superhúmedo (Pluvial)"]: zone_name = "Tundra pluvial Alpina"
        elif humidity_province in ["Perhúmedo", "Húmedo"]: zone_name = "Tundra húmeda Alpina"
        else: zone_name = "Tundra seca Alpina" # Cubre Subhúmedo a Superárido

    elif piso == "Subalpino":
        if humidity_province in ["Superhúmedo++", "Superhúmedo (Pluvial)"]: zone_name = "Páramo pluvial Subalpino"
        elif humidity_province in ["Perhúmedo", "Húmedo"]: zone_name = "Páramo húmedo Subalpino"
        else: zone_name = "Páramo seco Subalpino" # Cubre Subhúmedo a Superárido

    elif piso == "Montano":
        if humidity_province == "Superhúmedo++": zone_name = "Bosque pluvial Montano"
        elif humidity_province == "Superhúmedo (Pluvial)": zone_name = "Bosque pluvial Montano" # O muy húmedo? Check diagram
        elif humidity_province == "Perhúmedo": zone_name = "Bosque muy húmedo Montano"
        elif humidity_province == "Húmedo": zone_name = "Bosque húmedo Montano"
        elif humidity_province == "Subhúmedo": zone_name = "Bosque seco Montano"
        elif humidity_province == "Semiárido": zone_name = "Estepa Montano" # O Estepa Espinosa? Check
        elif humidity_province == "Árido": zone_name = "Estepa espinosa Montano" # O Desierto? Check
        else: zone_name = "Desierto Montano" # Cubre Perárido, Superárido

    elif piso == "Montano Bajo":
        if humidity_province == "Superhúmedo++": zone_name = "Bosque pluvial Montano Bajo"
        elif humidity_province == "Superhúmedo (Pluvial)": zone_name = "Bosque pluvial Montano Bajo"
        elif humidity_province == "Perhúmedo": zone_name = "Bosque muy húmedo Montano Bajo"
        elif humidity_province == "Húmedo": zone_name = "Bosque húmedo Montano Bajo"
        elif humidity_province == "Subhúmedo": zone_name = "Bosque seco Montano Bajo"
        elif humidity_province == "Semiárido": zone_name = "Estepa espinosa Montano Bajo"
        elif humidity_province == "Árido": zone_name = "Monte espinoso Montano Bajo"
        else: zone_name = "Desierto Montano Bajo" # Cubre Perárido, Superárido

    elif piso == "Subtropical":
        if humidity_province == "Superhúmedo++": zone_name = "Bosque pluvial Subtropical"
        elif humidity_province == "Superhúmedo (Pluvial)": zone_name = "Bosque pluvial Subtropical"
        elif humidity_province == "Perhúmedo": zone_name = "Bosque muy húmedo Subtropical"
        elif humidity_province == "Húmedo": zone_name = "Bosque húmedo Subtropical"
        elif humidity_province == "Subhúmedo": zone_name = "Bosque seco Subtropical"
        elif humidity_province == "Semiárido": zone_name = "Estepa espinosa Subtropical"
        elif humidity_province == "Árido": zone_name = "Monte espinoso Subtropical"
        else: zone_name = "Desierto Subtropical" # Cubre Perárido, Superárido
        
    elif piso == "Tropical":
        if humidity_province == "Superhúmedo++": zone_name = "Bosque pluvial Tropical"
        elif humidity_province == "Superhúmedo (Pluvial)": zone_name = "Bosque pluvial Tropical"
        elif humidity_province == "Perhúmedo": zone_name = "Bosque muy húmedo Tropical"
        elif humidity_province == "Húmedo": zone_name = "Bosque húmedo Tropical"
        elif humidity_province == "Subhúmedo": zone_name = "Bosque seco Tropical"
        elif humidity_province == "Semiárido": zone_name = "Estepa espinosa Tropical"
        elif humidity_province == "Árido": zone_name = "Monte espinoso Tropical"
        else: zone_name = "Desierto Tropical" # Cubre Perárido, Superárido

    # Retorna el nombre encontrado o el default si algo falló
    return holdridge_zone_map.get(zone_name, 0) # Devuelve el ID entero


# --- Función Principal para Generar el Mapa ---
@st.cache_data(show_spinner="Generando mapa de Zonas de Vida...")
def generate_life_zone_map(dem_path, precip_raster_path, mean_latitude, downscale_factor=4): # Añadido downscale_factor
    """
    Genera un mapa raster clasificado de Zonas de Vida de Holdridge,
    con opción de reducir la resolución para visualización.
    downscale_factor: 1 = original, 2 = mitad res (1/4 pixeles), 4 = cuarto res (1/16 pixeles), etc.
    """
    try:
        # --- NUEVO: Factor de reescalado ---
        if downscale_factor <= 0:
            downscale_factor = 1 # Evitar factor inválido
        
        # 1. Abrir DEM y obtener metadatos originales
        with rasterio.open(dem_path) as dem_src:
            src_profile = dem_src.profile
            src_crs = dem_src.crs
            src_transform = dem_src.transform
            nodata_dem = dem_src.nodata

            # --- NUEVO: Calcular dimensiones y transform reescalado ---
            dst_height = src_profile['height'] // downscale_factor
            dst_width = src_profile['width'] // downscale_factor
            
            # Ajustar transform para la nueva resolución
            dst_transform = src_transform * src_transform.scale(
                (src_profile['width'] / dst_width),
                (src_profile['height'] / dst_height)
            )

            # Crear perfil destino para remuestreo
            dst_profile = src_profile.copy()
            dst_profile.update({
                'height': dst_height,
                'width': dst_width,
                'transform': dst_transform,
                'dtype': rasterio.float32, # Trabajar con floats
                'nodata': np.nan # Usar NaN para nodata internamente
            })

            # --- NUEVO: Leer y remuestrear DEM ---
            st.write(f"Remuestreando DEM a {dst_width}x{dst_height} píxeles...")
            dem_data = np.empty((dst_height, dst_width), dtype=rasterio.float32)
            reproject(
                source=rasterio.band(dem_src, 1),
                destination=dem_data,
                src_transform=src_transform,
                src_crs=src_crs,
                src_nodata=nodata_dem,
                dst_transform=dst_transform,
                dst_crs=src_crs, # Mantener CRS original por ahora
                dst_nodata=np.nan,
                resampling=Resampling.average # Usar promedio para altitud
            )
            dem_mask = np.isnan(dem_data)
            st.write("Remuestreo DEM completado.")

        # 2. Abrir Precipitación y reproyectar/remuestrear al DEM reescalado
        with rasterio.open(precip_raster_path) as precip_src:
            st.write(f"Remuestreando Precipitación a {dst_width}x{dst_height} píxeles...")
            precip_data_aligned = np.empty((dst_height, dst_width), dtype=rasterio.float32)
            reproject(
                source=rasterio.band(precip_src, 1),
                destination=precip_data_aligned,
                src_transform=precip_src.transform,
                src_crs=precip_src.crs,
                src_nodata=precip_src.nodata,
                dst_transform=dst_transform, # Usar el transform reescalado
                dst_crs=src_crs,         # Usar el CRS reescalado
                dst_nodata=np.nan,
                resampling=Resampling.bilinear # Bilinear es bueno para precipitación
            )
            precip_mask = np.isnan(precip_data_aligned)
            st.write("Remuestreo Precipitación completado.")


        # 3. Cálculos (Ahora sobre rasters de menor resolución)
        st.write("Calculando variables biofísicas...")
        with np.errstate(invalid='ignore'):
            tma_raster = estimate_mean_annual_temp(dem_data)
            bat_raster = calculate_biotemperature(tma_raster, mean_latitude)
            pet_raster = calculate_pet(bat_raster)
            per_raster = calculate_per(pet_raster, precip_data_aligned)
        st.write("Cálculos completados.")

        # 4. Clasificar píxeles
        st.write("Clasificando Zonas de Vida...")
        classified_raster = np.full((dst_height, dst_width), 0, dtype=np.int16) # 0 para NoData
        valid_pixels = ~dem_mask & ~precip_mask & ~np.isnan(bat_raster) & ~np.isnan(per_raster) & ~np.isnan(precip_data_aligned)

        bat_values = bat_raster[valid_pixels]
        ppt_values = precip_data_aligned[valid_pixels]
        per_values = per_raster[valid_pixels]

        vectorized_classify = np.vectorize(classify_holdridge_zone_detailed)
        zone_ints = vectorized_classify(bat_values, ppt_values, per_values)
        
        classified_raster[valid_pixels] = zone_ints.astype(np.int16)
        st.write("Clasificación completada.")

        # 5. Preparar salida (usando el perfil reescalado)
        output_profile = dst_profile.copy() # Usar perfil destino del remuestreo
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
