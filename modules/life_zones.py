# modules/life_zones.py
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import streamlit as st

# --- Constantes (Ajustar según tu región si es necesario) ---
LAPSE_RATE = 6.5 # Tasa de lapso estándar (°C / 1000m)
BASE_TEMP_SEA_LEVEL = 28.0 # Temp base estimada a nivel del mar (°C)

# --- Funciones de Cálculo (Sin cambios respecto a la versión anterior) ---

def estimate_mean_annual_temp(elevation_m):
    """Estima la Temperatura Media Anual (TMA) basada en la altitud."""
    estimated_temp = BASE_TEMP_SEA_LEVEL - (LAPSE_RATE * elevation_m / 1000.0)
    return np.maximum(estimated_temp, -10) # Límite inferior razonable

def estimate_biotemperature(mean_annual_temp):
    """
    Estima la Biotemperature Anual Media (BAT) a partir de la TMA.
    Holdridge define BAT como el promedio de temperaturas > 0°C y < 30°C.
    Aquí usamos una aproximación clampeando la TMA.
    """
    # Clampar temperaturas: T° < 0 se vuelve 0, T° > 30 se vuelve 30
    clamped_temp = np.clip(mean_annual_temp, 0, 30)
    # Corrección: Las temperaturas bajo cero sí se cuentan como 0 en la suma,
    # pero para el promedio se podría usar una corrección más compleja.
    # Por simplicidad, mantenemos el clampeo como aproximación.
    return clamped_temp

def calculate_pet(biotemperature):
    """Calcula la Evapotranspiración Potencial (PET) anual media (mm)."""
    # Fórmula simplificada de Holdridge: PET = 58.93 * BAT
    return 58.93 * biotemperature

def calculate_per(pet, precipitation):
    """Calcula la Razón de Evapotranspiración Potencial (PER)."""
    # Evitar división por cero
    precipitation = np.maximum(precipitation, 1) # Asegura Ppt mínima de 1mm
    per = pet / precipitation
    return per # No limitamos PER aquí, la clasificación lo manejará

# --- Mapeo de Zonas a Enteros ---
# Basado en los nombres de image_2cf16d.png
# Asignamos un entero único a cada zona. El 0 será para NoData/Desconocido.
holdridge_zone_map = {
    # Nival
    "Nival": 1,
    # Alpino (Subnival en tabla)
    "Tundra pluvial Alpina": 2,
    "Tundra húmeda Alpina": 3,
    "Tundra seca Alpina": 4,
    # Subalpino (Páramo en tabla)
    "Páramo pluvial Subalpino": 5,
    "Páramo húmedo Subalpino": 6, # Asumiendo este es "Paramo muy humedo"
    "Páramo seco Subalpino": 7,
    # Montano (Montano en tabla)
    "Bosque pluvial Montano": 8,
    "Bosque muy húmedo Montano": 9,
    "Bosque húmedo Montano": 10,
    "Bosque seco Montano": 11,
    "Estepa espinosa Montano": 12,
    # Montano Bajo (Premontano en tabla)
    "Bosque pluvial Montano Bajo": 13,
    "Bosque muy húmedo Montano Bajo": 14,
    "Bosque húmedo Montano Bajo": 15,
    "Bosque seco Montano Bajo": 16,
    "Estepa espinosa Montano Bajo": 17,
    "Monte espinoso Montano Bajo": 18, # Agregado basado en patrón Holdridge general
    # Tropical y Subtropical (Basal en tabla) - Holdridge original los separa por BAT > 24
    # Aquí los agrupamos como "Basal" según la tabla, pero podríamos separarlos si es necesario.
    # Asumiendo que > 18 es el piso Basal completo (Premontano/Tropical)
    "Bosque pluvial Tropical/Premontano": 19, # Combinado
    "Bosque muy húmedo Tropical/Premontano": 20, # Combinado
    "Bosque húmedo Tropical/Premontano": 21, # Combinado
    "Bosque seco Tropical/Premontano": 22, # Combinado
    "Estepa espinosa Tropical/Premontano": 23, # Combinado (antes Monte Espinoso)
    "Monte espinoso Tropical/Premontano": 24, # Renombrado/Agregado
    "Desierto Tropical/Premontano": 25, # Agregado basado en patrón
    # Zona Desconocida / NoData
    "Zona Desconocida": 0
}

# Crear el mapa inverso para obtener el nombre a partir del número
holdridge_int_to_name = {v: k for k, v in holdridge_zone_map.items()}

# --- Función de Clasificación ---

def classify_holdridge_zone(bat, ppt, per):
    """
    Clasifica la Zona de Vida de Holdridge basado en Biotemperatura (BAT),
    Precipitación Total Anual (PPT) y Razón PET (PER).
    Basado en los umbrales de las imágenes proporcionadas.
    """
    
    # 1. Determinar Región Latitudinal / Piso Altitudinal (Según BAT)
    # Límites de image_2cf96b.png
    if bat < 1.5:
        piso = "Nival"
    elif bat < 3:
        piso = "Alpino" # (Subnival en tabla)
    elif bat < 6:
        piso = "Subalpino" # (Paramo en tabla)
    elif bat < 12:
        piso = "Montano"
    elif bat < 18: # Límite inferior de Basal/Premontano
        piso = "Montano Bajo" # (Premontano en tabla)
    else: # bat >= 18
        # Aquí podrías diferenciar Tropical (>=24) si fuera necesario,
        # pero la tabla agrupa en "Basal"
        piso = "Tropical/Premontano" # (Basal en tabla)

    # 2. Determinar Provincia de Humedad (Según PER)
    # Límites de image_2cf96b.png
    if per <= 0.125: # Límite superior Superárido
         # Podríamos diferenciar aún más si ppt es muy bajo (Desierto o Hielo)
         # Simplificación: Asignar Superárido o related based on Piso
         if piso == "Nival": return "Nival" # Hielo/Nieve permanente
         elif piso == "Alpino": return "Tundra seca Alpina" # Muy seco
         elif piso in ["Subalpino", "Montano", "Montano Bajo"]: return "Desierto" # Muy seco
         else: return "Desierto Tropical/Premontano" # Muy seco
    elif per <= 0.25:
        humidity = "Perárido"
    elif per <= 0.5:
        humidity = "Árido"
    elif per <= 1.0:
        humidity = "Semiárido"
    elif per <= 2.0:
        humidity = "Subhúmedo"
    elif per <= 4.0:
        humidity = "Húmedo"
    elif per <= 8.0:
        humidity = "Perhúmedo"
    else: # per > 8.0
        humidity = "Superhúmedo"

    # 3. Combinar Piso y Humedad para obtener la Zona Específica
    #    (Usando los nombres de image_2cf16d.png y lógica de combinación)
    
    zone_name = "Zona Desconocida" # Default

    if piso == "Alpino":
        if humidity == "Superhúmedo": zone_name = "Tundra pluvial Alpina"
        elif humidity == "Perhúmedo": zone_name = "Tundra húmeda Alpina" # Asumiendo Perhumedo=humedo aqui
        elif humidity == "Húmedo": zone_name = "Tundra húmeda Alpina"
        else: zone_name = "Tundra seca Alpina" # Cubre Subhumedo a Perarido
    elif piso == "Subalpino":
        if humidity == "Superhúmedo": zone_name = "Páramo pluvial Subalpino"
        elif humidity == "Perhúmedo": zone_name = "Páramo húmedo Subalpino" # Asumiendo Perhumedo="muy humedo"
        elif humidity == "Húmedo": zone_name = "Páramo húmedo Subalpino" # Asumiendo Húmedo también es "muy humedo"
        else: zone_name = "Páramo seco Subalpino" # Cubre Subhumedo a Perarido
    elif piso == "Montano":
        if humidity == "Superhúmedo": zone_name = "Bosque pluvial Montano"
        elif humidity == "Perhúmedo": zone_name = "Bosque muy húmedo Montano"
        elif humidity == "Húmedo": zone_name = "Bosque húmedo Montano"
        elif humidity == "Subhúmedo": zone_name = "Bosque seco Montano"
        elif humidity == "Semiárido": zone_name = "Estepa espinosa Montano"
        # Faltarían Árido, Perárido (podrían mapear a Desierto Montano si existiera)
    elif piso == "Montano Bajo":
        if humidity == "Superhúmedo": zone_name = "Bosque pluvial Montano Bajo"
        elif humidity == "Perhúmedo": zone_name = "Bosque muy húmedo Montano Bajo"
        elif humidity == "Húmedo": zone_name = "Bosque húmedo Montano Bajo"
        elif humidity == "Subhúmedo": zone_name = "Bosque seco Montano Bajo"
        elif humidity == "Semiárido": zone_name = "Estepa espinosa Montano Bajo"
        elif humidity == "Árido": zone_name = "Monte espinoso Montano Bajo"
        # Faltaría Perárido (podría mapear a Desierto MB)
    elif piso == "Tropical/Premontano":
        if humidity == "Superhúmedo": zone_name = "Bosque pluvial Tropical/Premontano"
        elif humidity == "Perhúmedo": zone_name = "Bosque muy húmedo Tropical/Premontano"
        elif humidity == "Húmedo": zone_name = "Bosque húmedo Tropical/Premontano"
        elif humidity == "Subhúmedo": zone_name = "Bosque seco Tropical/Premontano"
        elif humidity == "Semiárido": zone_name = "Estepa espinosa Tropical/Premontano" # Renombrado de Monte espinoso
        elif humidity == "Árido": zone_name = "Monte espinoso Tropical/Premontano" # Renombrado/Agregado
        elif humidity == "Perárido": zone_name = "Desierto Tropical/Premontano"
        # Superárido ya fue manejado al inicio

    return zone_name


# --- Función Principal para Generar el Mapa (Sin cambios excepto el diccionario de mapeo) ---

@st.cache_data(show_spinner="Generando mapa de Zonas de Vida...")
def generate_life_zone_map(dem_path, precip_raster_path):
    """
    Genera un mapa raster clasificado de Zonas de Vida de Holdridge.
    """
    try:
        # 1. Abrir DEM y obtener metadatos
        with rasterio.open(dem_path) as dem_src:
            dem_data = dem_src.read(1).astype(np.float32) # Leer como float
            dem_profile = dem_src.profile
            dem_crs = dem_src.crs
            dem_transform = dem_src.transform
            nodata_dem = dem_src.nodata

            # Mascarar nodata si existe
            dem_mask = (dem_data == nodata_dem) if nodata_dem is not None else np.zeros(dem_data.shape, dtype=bool)
            dem_data[dem_mask] = np.nan # Usar NaN para nodata internamente

        # 2. Abrir raster de precipitación y reproyectar/remuestrear al DEM
        with rasterio.open(precip_raster_path) as precip_src:
            profile_dest = dem_profile.copy()
            profile_dest.update({'dtype': rasterio.float32, 'nodata': np.nan})
            precip_data_aligned = np.empty(dem_data.shape, dtype=rasterio.float32)

            reproject(
                source=rasterio.band(precip_src, 1),
                destination=precip_data_aligned,
                src_transform=precip_src.transform,
                src_crs=precip_src.crs,
                src_nodata=precip_src.nodata,
                dst_transform=dem_transform,
                dst_crs=dem_crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear
            )
            precip_mask = np.isnan(precip_data_aligned)

        # 3. Realizar cálculos pixel a pixel
        with np.errstate(invalid='ignore'): # Ignorar warnings de cálculos con NaN
            tma_raster = estimate_mean_annual_temp(dem_data)
            bat_raster = estimate_biotemperature(tma_raster)
            pet_raster = calculate_pet(bat_raster)
            per_raster = calculate_per(pet_raster, precip_data_aligned)

        # 4. Clasificar cada píxel
        classified_raster = np.full(dem_data.shape, holdridge_zone_map["Zona Desconocida"], dtype=np.int16)
        valid_pixels = ~dem_mask & ~precip_mask & ~np.isnan(bat_raster) & ~np.isnan(per_raster) & ~np.isnan(precip_data_aligned)

        bat_values = bat_raster[valid_pixels]
        ppt_values = precip_data_aligned[valid_pixels]
        per_values = per_raster[valid_pixels]

        # Aplicar la clasificación vectorizada (si es posible) o por bucle
        # Para mejorar rendimiento, reescribir classify_holdridge_zone con np.select o similar
        # Por ahora, usamos el bucle (puede ser lento)
        zone_names = [classify_holdridge_zone(b, p, pe) for b, p, pe in zip(bat_values, ppt_values, per_values)]
        zone_ints = [holdridge_zone_map.get(name, holdridge_zone_map["Zona Desconocida"]) for name in zone_names]
        
        classified_raster[valid_pixels] = zone_ints

        # 5. Preparar salida
        output_profile = dem_profile.copy()
        output_profile.update({
            'dtype': rasterio.int16,
            'nodata': holdridge_zone_map["Zona Desconocida"], # Usar 0 como nodata
            'count': 1
        })

        return classified_raster, output_profile, holdridge_int_to_name

    except Exception as e:
        st.error(f"Error generando mapa de zonas de vida: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None
