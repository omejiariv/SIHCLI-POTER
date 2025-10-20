# modules/life_zones.py
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import streamlit as st

# --- Constantes (Ajustar según tu región si es necesario) ---
# Tasa de Lapsos estándar (grados C por cada 1000m de aumento de altitud)
LAPSE_RATE = 6.5 
# Temperatura base a nivel del mar (estimación, podría ajustarse)
BASE_TEMP_SEA_LEVEL = 28.0 
# Latitud promedio representativa de tu área de interés (ej., Antioquia ~ 6.5 N)
# Usaremos esto para una estimación simplificada de biotemperatura
REPRESENTATIVE_LATITUDE = 6.5

# --- Funciones de Cálculo ---

def estimate_mean_annual_temp(elevation_m):
    """Estima la Temperatura Media Anual (TMA) basada en la altitud."""
    # Evitar temperaturas no realistas en altitudes muy altas
    estimated_temp = BASE_TEMP_SEA_LEVEL - (LAPSE_RATE * elevation_m / 1000.0)
    return np.maximum(estimated_temp, -10) # Poner un límite inferior razonable

def estimate_biotemperature(mean_annual_temp):
    """
    Estima la Biotemperatura Anual Media (BAT) a partir de la TMA.
    Simplificación: Asume que las T° bajo 0°C o sobre 30°C se ajustan en promedio.
    Una estimación MUY simple podría ser la TMA misma si está entre 0-30, 
    o usar correcciones más complejas. Empecemos con una aproximación.
    Referencia simplificada: BAT ≈ TMA - (Factor * Latitud) si TMA > 0
    Otra: Clampar TMA entre 0 y 30.
    """
    # Clampar temperaturas: T° < 0 se vuelve 0, T° > 30 se vuelve 30
    clamped_temp = np.clip(mean_annual_temp, 0, 30) 
    # Para esta implementación, usaremos la temperatura clampeada como aproximación directa.
    # Podría refinarse más adelante.
    return clamped_temp

def calculate_pet(biotemperature):
    """Calcula la Evapotranspiración Potencial (PET) anual media (mm)."""
    # Fórmula simplificada de Holdridge: PET = 58.93 * BAT
    return 58.93 * biotemperature

def calculate_per(pet, precipitation):
    """Calcula la Razón de Evapotranspiración Potencial (PER)."""
    # Evitar división por cero o valores muy pequeños de precipitación
    precipitation = np.maximum(precipitation, 1) # Asegura Ppt mínima de 1mm
    per = pet / precipitation
    # Limitar el valor de PER para evitar extremos irreales en zonas muy secas/húmedas
    return np.clip(per, 0.01, 128.0) 

# --- Función de Clasificación ---

def classify_holdridge_zone(bat, ppt, per):
    """
    Clasifica la Zona de Vida de Holdridge basado en BAT, PPT y PER.
    Esta es una implementación simplificada basada en umbrales comunes.
    Se requiere una tabla o diagrama detallado para una clasificación completa y precisa.
    Fuente de umbrales (ejemplo): https://www.researchgate.net/figure/Diagram-for-the-classification-of-Holdridge-life-zones_fig1_296683838
    (Necesitarás adaptar estos umbrales del diagrama visual a código)
    """
    # Convertir PPT a escala logarítmica base 2 para comparación con diagrama (aproximado)
    ppt_log2 = np.log2(np.maximum(ppt, 1))

    # Definir umbrales (Estos son EJEMPLOS y necesitan refinamiento basado en el diagrama real)
    # Umbrales de Biotemperatura (aproximados)
    TROPICAL_LIMIT = 24
    SUBTROPICAL_LIMIT = 18 # Aprox
    TEMPERATE_LIMIT = 12 # Aprox
    BOREAL_LIMIT = 6   # Aprox
    ALPINE_LIMIT = 3   # Aprox
    NIVAL_LIMIT = 1.5 # Aprox

    # Umbrales de Humedad (Basados en PER, más fáciles de codificar que ejes diagonales)
    SUPERHUMID_LIMIT = 0.25
    PERHUMID_LIMIT = 0.5
    HUMID_LIMIT = 1.0
    SUBHUMID_LIMIT = 2.0
    SEMIARID_LIMIT = 4.0
    ARID_LIMIT = 8.0
    PERARID_LIMIT = 16.0
    SUPERARID_LIMIT = 32.0

    # Clasificación por Latitud/Altitud (Pisos)
    if bat > TROPICAL_LIMIT:
        floor = "Tropical"
    elif bat > SUBTROPICAL_LIMIT:
        floor = "Subtropical"
    elif bat > TEMPERATE_LIMIT:
        floor = "Templado (Temperate)"
    elif bat > BOREAL_LIMIT:
        floor = "Boreal (Cool Temperate)"
    elif bat > ALPINE_LIMIT:
        floor = "Subalpino/Alpino (Subpolar/Alpine)"
    elif bat > NIVAL_LIMIT:
        floor = "Nival (Polar)"
    else:
        floor = "Nival (Polar)" # O Glacial

    # Clasificación por Humedad
    if per < SUPERHUMID_LIMIT:
        humidity = "Superhúmedo (Rain Forest/Pluvial)"
    elif per < PERHUMID_LIMIT:
        humidity = "Perhúmedo (Wet Forest)"
    elif per < HUMID_LIMIT:
        humidity = "Húmedo (Moist Forest)"
    elif per < SUBHUMID_LIMIT:
        humidity = "Subhúmedo (Dry Forest)"
    elif per < SEMIARID_LIMIT:
        humidity = "Semiárido (Very Dry Forest/Steppe)"
    elif per < ARID_LIMIT:
        humidity = "Árido (Thorn Steppe/Desert Scrub)"
    elif per < PERARID_LIMIT:
        humidity = "Perárido (Desert)"
    elif per < SUPERARID_LIMIT: # Aprox
        humidity = "Superárido (Ice/Desert)"
    else:
        humidity = "Superárido (Ice/Desert)"
        
    # Combinar (Simplificado - no incluye todos los nombres específicos como "Bosque Montano Bajo")
    # Una implementación completa usaría una matriz o diccionario basado en el diagrama.
    # Ejemplo muy básico:
    zone_name = f"{floor} - {humidity}" 
    
    # --- IMPLEMENTACIÓN MÁS DETALLADA (Ejemplo Parcial) ---
    # Esto requeriría muchos más elif basados en rangos de BAT y PPT/PER
    if floor == "Tropical":
        if humidity == "Húmedo (Moist Forest)": zone_name = "Bosque húmedo Tropical (bh-T)"
        elif humidity == "Subhúmedo (Dry Forest)": zone_name = "Bosque seco Tropical (bs-T)"
        # ... etc ...
    elif floor == "Subtropical":
         if humidity == "Húmedo (Moist Forest)": zone_name = "Bosque húmedo Subtropical (bh-ST)"
         # ... etc ...

    return zone_name # Retorna el nombre combinado (o el específico si se implementa)


# --- Función Principal para Generar el Mapa ---

# Usaremos un diccionario para mapear nombres de zona a valores enteros para el raster
holdridge_zone_map = {
    # Define aquí las zonas que tu función `classify_holdridge_zone` puede retornar
    # y asígnales un número entero único. Ejemplo:
    "Bosque húmedo Tropical (bh-T)": 1,
    "Bosque seco Tropical (bs-T)": 2,
    "Bosque húmedo Subtropical (bh-ST)": 3,
    # ... Añade TODAS las posibles zonas con números únicos ...
    "Zona Desconocida": 0 # Para píxeles sin clasificación
}
# Crear el mapa inverso para obtener el nombre a partir del número
holdridge_int_to_name = {v: k for k, v in holdridge_zone_map.items()}


@st.cache_data(show_spinner="Generando mapa de Zonas de Vida...")
def generate_life_zone_map(dem_path, precip_raster_path, target_crs="EPSG:4326"):
    """
    Genera un mapa raster clasificado de Zonas de Vida de Holdridge.
    """
    try:
        # 1. Abrir DEM y obtener metadatos
        with rasterio.open(dem_path) as dem_src:
            dem_data = dem_src.read(1)
            dem_profile = dem_src.profile
            dem_crs = dem_src.crs
            dem_transform = dem_src.transform
            nodata_dem = dem_src.nodata if dem_src.nodata is not None else -9999

            # Mascarar nodata si existe
            dem_mask = (dem_data == nodata_dem)
            dem_data = np.where(dem_mask, np.nan, dem_data)

        # 2. Abrir raster de precipitación y reproyectar/remuestrear al DEM
        with rasterio.open(precip_raster_path) as precip_src:
            # Crear perfil destino con la forma y CRS del DEM
            profile_dest = dem_profile.copy()
            profile_dest.update({'dtype': rasterio.float32, 'nodata': np.nan})
            
            # Crear array destino
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
                resampling=Resampling.bilinear # O nearest si prefieres
            )
            precip_mask = np.isnan(precip_data_aligned)

        # 3. Realizar cálculos pixel a pixel (o con álgebra de mapas)
        
        # Estimar Temperatura Media Anual
        tma_raster = estimate_mean_annual_temp(dem_data)
        
        # Estimar Biotemperatura
        bat_raster = estimate_biotemperature(tma_raster)
        
        # Calcular PET
        pet_raster = calculate_pet(bat_raster)
        
        # Calcular PER
        per_raster = calculate_per(pet_raster, precip_data_aligned)
        
        # 4. Clasificar cada píxel
        # Inicializar raster de salida con un valor para 'no clasificado' o 'nodata'
        classified_raster = np.full(dem_data.shape, holdridge_zone_map["Zona Desconocida"], dtype=np.int16) 
        
        # Iterar sobre píxeles válidos (donde tenemos DEM y Precipitación)
        valid_pixels = ~dem_mask & ~precip_mask & ~np.isnan(bat_raster) & ~np.isnan(per_raster)
        
        # Vectorizar la clasificación para eficiencia
        bat_values = bat_raster[valid_pixels]
        ppt_values = precip_data_aligned[valid_pixels]
        per_values = per_raster[valid_pixels]
        
        # Esta parte podría ser lenta si classify_holdridge_zone no está vectorizada
        # Una forma más rápida sería hacer la lógica de umbrales directamente con numpy arrays
        # Ejemplo simplificado con la función (puede ser lento para rasters grandes):
        zone_names = [classify_holdridge_zone(b, p, pe) for b, p, pe in zip(bat_values, ppt_values, per_values)]
        
        # Mapear nombres de zona a enteros usando el diccionario
        zone_ints = [holdridge_zone_map.get(name, holdridge_zone_map["Zona Desconocida"]) for name in zone_names]
        
        classified_raster[valid_pixels] = zone_ints

        # 5. Preparar salida
        output_profile = dem_profile.copy()
        output_profile.update({
            'dtype': rasterio.int16,
            'nodata': holdridge_zone_map["Zona Desconocida"], # Usar el valor 0 como nodata
            'count': 1
        })

        return classified_raster, output_profile, holdridge_int_to_name # Devolver el raster, perfil y mapa de nombres

    except Exception as e:
        st.error(f"Error generando mapa de zonas de vida: {e}")
        import traceback
        st.error(traceback.format_exc()) # Imprime el traceback completo para depurar
        return None, None, None
