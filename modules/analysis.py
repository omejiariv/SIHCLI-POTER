# modules/analysis.py

import streamlit as st
import geopandas as gpd
import pandas as pd
import requests
import numpy as np
from scipy.stats import gamma, loglaplace, norm
from modules.config import Config
import rasterio
from rasterstats import zonal_stats
import pymannkendall as mk
from shapely.ops import unary_union
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
from rasterio.transform import Affine


@st.cache_data
def calculate_spi(df, col=Config.PRECIPITATION_COL, window=12):
    """
    Calcula el Índice Estandarizado de Precipitación (SPI) usando una aproximación estadística.
    window: ventana de meses (ej. 3, 6, 12).
    """
    if df.empty: return df
    
    df_spi = df.copy().sort_values(Config.DATE_COL)
    
    # 1. Calcular acumulado móvil (Rolling Sum)
    df_spi['rolling_precip'] = df_spi[col].rolling(window=window, min_periods=window).sum()
    
    # 2. Calcular SPI (Aproximación Z-Score sobre logaritmos para normalizar)
    # El SPI real usa distribución Gamma, pero Z-Score logarítmico es una buena aproximación robusta
    try:
        valid_mask = df_spi['rolling_precip'] > 0 # Evitar log(0)
        
        # Transformación Logarítmica para normalizar la distribución de lluvia
        log_precip = np.log(df_spi.loc[valid_mask, 'rolling_precip'])
        
        mean = log_precip.mean()
        std = log_precip.std()
        
        if std == 0:
            df_spi['spi'] = 0
        else:
            # Calcular Z-score
            df_spi.loc[valid_mask, 'spi'] = (log_precip - mean) / std
            
            # Llenar valores donde llovió 0 con el mínimo SPI detectado (sequía extrema)
            min_spi = df_spi['spi'].min()
            if pd.isna(min_spi): min_spi = -3.0
            df_spi.loc[~valid_mask & df_spi['rolling_precip'].notna(), 'spi'] = min_spi
            
    except Exception:
        df_spi['spi'] = np.nan
        
    return df_spi

@st.cache_data
def calculate_spei(precip_series, et_series, scale):
    """
    Calcula el Índice de Precipitación y Evapotranspiración Estandarizado (SPEI).
    """

    # Validación inicial de entradas
    if precip_series is None or et_series is None:
        st.error(f"SPEI-{scale}: precip_series o et_series es None.")
        return pd.Series(dtype=float)
    if precip_series.empty or et_series.empty:
         st.warning(f"SPEI-{scale}: precip_series o et_series está vacía.")
         return pd.Series(dtype=float)

    scale = int(scale)
    # Asegurar alineación de índices y frecuencia mensual
    df = pd.DataFrame({'precip': precip_series, 'et': et_series}).sort_index()
    # Intentar inferir frecuencia mensual o rellenar si es necesario
    df = df.asfreq('MS') # Fuerza frecuencia mensual, rellenará con NaN si faltan meses

    # Rellenar NaNs en et_series con la media podría ser una opción, pero puede distorsionar resultados.
    # Por ahora, solo quitamos filas donde AMBOS son NaN o donde P es NaN
    df.dropna(subset=['precip'], inplace=True) 
    df['et'] = df['et'].fillna(method='ffill').fillna(method='bfill') # Relleno simple para ET si tiene NaNs
    df.dropna(subset=['et'], inplace=True) # Quitar si aún quedan NaNs en ET

    if len(df) < scale * 2: # Chequeo DESPUÉS de limpiar NaNs
        st.warning(f"SPEI-{scale}: Serie demasiado corta ({len(df)} puntos) después de limpiar NaNs.")
        return pd.Series(dtype=float)

    water_balance = df['precip'] - df['et']
    rolling_balance = water_balance.rolling(window=scale, min_periods=scale).sum()
    data_for_fit = rolling_balance.dropna()
    data_for_fit = data_for_fit[np.isfinite(data_for_fit)] # Quitar Infinitos

    spei = pd.Series(np.nan, index=rolling_balance.index) # Inicializar salida con NaN

    if not data_for_fit.empty and len(data_for_fit.unique()) > 1: # Añadido chequeo de más de un valor único
        try:
            # Ajustar floc dinámicamente para evitar errores si min <= 0
            params = loglaplace.fit(data_for_fit, floc=data_for_fit.min() - 1e-5 if data_for_fit.min() <=0 else 0) 

            cdf = loglaplace.cdf(rolling_balance.dropna(), *params) # Calcular CDF solo para valores no-NaN del rolling_balance
            cdf_series = pd.Series(cdf, index=rolling_balance.dropna().index) # Ponerlo en una Serie con el índice correcto

            # Asegurar probabilidades entre casi 0 y casi 1
            cdf_clipped = np.clip(cdf_series.values, 1e-7, 1 - 1e-7) 

            spei_calculated = norm.ppf(cdf_clipped)

            # Asignar los valores calculados al índice correcto en la serie de salida 'spei'
            spei.loc[cdf_series.index] = spei_calculated

        except Exception as e:
             st.error(f"SPEI-{scale}: Falló el ajuste Log-Laplace o cálculo CDF. Error: {e}")
             import traceback
             st.error(traceback.format_exc()) # Imprime el traceback completo
    elif data_for_fit.empty:
        st.warning(f"SPEI-{scale}: No hay datos válidos (data_for_fit vacío) para ajustar la distribución después de la suma acumulada.")
    else: # Solo un valor único
         st.warning(f"SPEI-{scale}: Todos los valores en data_for_fit son iguales ({data_for_fit.iloc[0]}). No se puede ajustar la distribución.")

    spei.replace([np.inf, -np.inf], np.nan, inplace=True)
    return spei
    
def calculate_monthly_anomalies(df_monthly, df_long_full):
    """Calcula anomalías mensuales."""
    if df_monthly.empty: return pd.DataFrame()
    climatology = df_long_full.groupby(Config.MONTH_COL)[Config.PRECIPITATION_COL].mean().reset_index()
    climatology = climatology.rename(columns={Config.PRECIPITATION_COL: 'mean_ppt'})
    merged = pd.merge(df_monthly, climatology, on=Config.MONTH_COL, how='left')
    merged['anomalia'] = merged[Config.PRECIPITATION_COL] - merged['climatology_ppt']
    return merged

def estimate_temperature(altitude_m):
    """Estimación de temperatura basada en gradiente térmico vertical (Andes)."""
    if pd.isna(altitude_m): return 25.0
    # Gradiente típico: 28°C a nivel del mar, disminuye 0.6°C por cada 100m
    temp = 28.0 - (0.006 * altitude_m)
    return max(temp, 1.0) # Evitar temperaturas bajo cero extremas en modelo simple

def calculate_water_balance_turc(precip_mm, temp_c):
    """
    Estima Evapotranspiración Real (ETR) y Escorrentía (Q) usando la fórmula de Turc.
    """
    if pd.isna(precip_mm) or pd.isna(temp_c): return 0, 0
    
    # Fórmula de Turc para ETR
    L = 300 + 25 * temp_c + 0.05 * (temp_c**3)
    etr = precip_mm / np.sqrt(0.9 + (precip_mm / L)**2)
    
    # Limitar ETR a la precipitación (no puede evaporarse más de lo que llueve en este modelo simple)
    etr = min(etr, precip_mm)
    
    # Escorrentía = Precipitación - ETR
    q = precip_mm - etr
    return etr, q

def classify_holdridge_point(precip_mm, altitude_m):
    """Clasificación simple de Holdridge."""
    if pd.isna(precip_mm) or pd.isna(altitude_m): return "N/A"
    if altitude_m < 1000: piso = "Tropical"
    elif altitude_m < 2000: piso = "Premontano"
    elif altitude_m < 3000: piso = "Montano Bajo"
    elif altitude_m < 3500: piso = "Montano-Paramo"
    else: piso = "Montano"
    
    if precip_mm < 1000: prov = "Bosque Seco"
    elif precip_mm < 2000: prov = "Bosque Húmedo"
    elif precip_mm < 4000: prov = "Bosque Muy Húmedo"
    else: prov = "Bosque Pluvial"
    
    return f"{prov} {piso}"
    
def calculate_percentiles_and_extremes(df_long, station_name, p_lower=10, p_upper=90):
    """
    Calcula umbrales de percentiles y clasifica eventos extremos para una estación.
    """
    df_station_full = df_long[df_long[Config.STATION_NAME_COL] == station_name].copy()
    df_thresholds = df_station_full.groupby(Config.MONTH_COL)[Config.PRECIPITATION_COL].agg(
        p_lower=lambda x: np.nanpercentile(x.dropna(), p_lower),
        p_upper=lambda x: np.nanpercentile(x.dropna(), p_upper),
        mean_monthly='mean'
    ).reset_index()
    df_station_extremes = pd.merge(df_station_full, df_thresholds, on=Config.MONTH_COL, how='left')
    df_station_extremes['event_type'] = 'Normal'
    is_dry = (df_station_extremes[Config.PRECIPITATION_COL] < df_station_extremes['p_lower'])
    df_station_extremes.loc[is_dry, 'event_type'] = f'Sequía Extrema (< P{p_lower}%)'
    is_wet = (df_station_extremes[Config.PRECIPITATION_COL] > df_station_extremes['p_upper'])
    df_station_extremes.loc[is_wet, 'event_type'] = f'Húmedo Extremo (> P{p_upper}%)'
    return df_station_extremes.dropna(subset=[Config.PRECIPITATION_COL]), df_thresholds

@st.cache_data
def calculate_climatological_anomalies(_df_monthly_filtered, _df_long, baseline_start, baseline_end):
    """
    Calcula las anomalías mensuales con respecto a un período base climatológico fijo.
    """
    df_monthly_filtered = _df_monthly_filtered.copy()
    df_long = _df_long.copy()

    baseline_df = df_long[
        (df_long[Config.YEAR_COL] >= baseline_start) & 
        (df_long[Config.YEAR_COL] <= baseline_end)
    ]

    df_climatology = baseline_df.groupby(
        [Config.STATION_NAME_COL, Config.MONTH_COL]
    )[Config.PRECIPITATION_COL].mean().reset_index().rename(
        columns={Config.PRECIPITATION_COL: 'precip_promedio_climatologico'}
    )

    df_anomalias = pd.merge(
        df_monthly_filtered,
        df_climatology,
        on=[Config.STATION_NAME_COL, Config.MONTH_COL],
        how='left'
    )

    df_anomalias['anomalia'] = df_anomalias[Config.PRECIPITATION_COL] - df_anomalias['precip_promedio_climatologico']
    return df_anomalias

@st.cache_data
def analyze_events(index_series, threshold, event_type='drought'):
    """
    Identifica y caracteriza eventos de sequía o humedad en una serie de tiempo de índices.
    """
    if event_type == 'drought':
        is_event = index_series < threshold
    else: # 'wet'
        is_event = index_series > threshold

    event_blocks = (is_event.diff() != 0).cumsum()
    active_events = is_event[is_event]
    if active_events.empty:
        return pd.DataFrame()

    events = []
    for event_id, group in active_events.groupby(event_blocks):
        start_date = group.index.min()
        end_date = group.index.max()
        duration = len(group)
        
        event_values = index_series.loc[start_date:end_date]
        
        magnitude = event_values.sum()
        intensity = event_values.mean()
        peak = event_values.min() if event_type == 'drought' else event_values.max()

        events.append({
            'Fecha Inicio': start_date,
            'Fecha Fin': end_date,
            'Duración (meses)': duration,
            'Magnitud': magnitude,
            'Intensidad': intensity,
            'Pico': peak
        })

    if not events:
        return pd.DataFrame()

    events_df = pd.DataFrame(events)
    return events_df.sort_values(by='Fecha Inicio').reset_index(drop=True)

@st.cache_data
def calculate_basin_stats(_gdf_stations, _gdf_basins, _df_monthly, basin_name, basin_col_name):
    """
    Calcula estadísticas de precipitación para todas las estaciones dentro de una cuenca específica.
    """
    if _gdf_basins is None or basin_col_name not in _gdf_basins.columns:
        return pd.DataFrame(), [], "El GeoDataFrame de cuencas o la columna de nombres no es válida."

    target_basin = _gdf_basins[_gdf_basins[basin_col_name] == basin_name]
    if target_basin.empty:
        return pd.DataFrame(), [], f"No se encontró la cuenca llamada '{basin_name}'."

    stations_in_basin = gpd.sjoin(_gdf_stations, target_basin, how="inner", predicate="within")
    station_names_in_basin = stations_in_basin[Config.STATION_NAME_COL].unique().tolist()

    if not station_names_in_basin:
        return pd.DataFrame(), [], None

    df_basin_precip = _df_monthly[_df_monthly[Config.STATION_NAME_COL].isin(station_names_in_basin)]
    if df_basin_precip.empty:
        return pd.DataFrame(), station_names_in_basin, "No hay datos de precipitación para las estaciones en esta cuenca."
    
    stats = df_basin_precip[Config.PRECIPITATION_COL].describe().reset_index()
    stats.columns = ['Métrica', 'Valor']
    stats['Valor'] = stats['Valor'].round(2)
    
    return stats, station_names_in_basin, None

@st.cache_data
def get_mean_altitude_for_basin(_basin_geometry):
    """
    Calcula la altitud media de una cuenca consultando la API de Open-Elevation.
    """
    try:
        # Simplificamos la geometría para reducir el tamaño de la consulta
        simplified_geom = _basin_geometry.simplify(tolerance=0.01)
        
        # Obtenemos los puntos del contorno exterior del polígono
        exterior_coords = list(simplified_geom.exterior.coords)
        
        # Creamos la estructura de datos para la API
        locations = [{"latitude": lat, "longitude": lon} for lon, lat in exterior_coords]
        
        # Hacemos la llamada a la API de Open-Elevation
        response = requests.post("https://api.open-elevation.com/api/v1/lookup", json={"locations": locations})
        response.raise_for_status()
        
        results = response.json()['results']
        elevations = [res['elevation'] for res in results]
        
        # Calculamos la media de las elevaciones obtenidas
        mean_altitude = np.mean(elevations)
        
        return mean_altitude, None
    except Exception as e:
        error_message = f"No se pudo obtener la altitud de la cuenca: {e}"
        st.warning(error_message)
        return None, error_message

def calculate_hydrological_balance(precip_mm_anual, alt_media_m, gdf_basin):
    """
    Calcula Balance Hídrico usando fórmula de Turc.
    precip_mm_anual: Debe ser la suma anual (mm/año).
    """
    if precip_mm_anual is None or precip_mm_anual <= 0: 
        return {"P": 0, "ET": 0, "Q": 0, "Vol": 0, "Alt": 0, "Area": 0}
    
    # 1. Temperatura Media Anual (°C)
    temp_c = estimate_temperature(alt_media_m)
    
    # 2. Evapotranspiración Real (Turc)
    # L = 300 + 25T + 0.05T^3
    L = 300 + (25 * temp_c) + (0.05 * (temp_c**3))
    
    # ETR = P / sqrt(0.9 + (P/L)^2)
    # Validar que no sea NaN
    denom = np.sqrt(0.9 + (precip_mm_anual / L)**2)
    etr = precip_mm_anual / denom
    
    # Ajuste físico: ETR no puede ser mayor que la precipitación
    etr = min(etr, precip_mm_anual)
    
    # 3. Escorrentía (Caudal Q)
    q_mm = precip_mm_anual - etr
    
    # 4. Volumen (Millones de m3)
    # Q(mm) * Area(km2) * 1000 = m3 -> / 1e6 = Mm3
    # Simplificado: Q(mm) * Area(km2) / 1000 = Mm3 ?? No
    # 1 mm = 1 L/m2 = 0.001 m3/m2
    # Vol = Q * 0.001 (m) * Area * 1e6 (m2) = Q * Area * 1000 (m3)
    # En Millones de m3: (Q * Area * 1000) / 1e6 = (Q * Area) / 1000
    
    morph = calculate_morphometry(gdf_basin)
    area = morph['area_km2']
    vol_mm3 = (q_mm * area) / 1000.0
    
    return {
        "P": precip_mm_anual,
        "ET": etr,
        "Q": q_mm,
        "Vol": vol_mm3,
        "Alt": alt_media_m,
        "Area": area
    }
      
def calculate_morphometry(gdf_basin):
    """Calcula métricas morfométricas básicas."""
    if gdf_basin is None or gdf_basin.empty: 
        return {k: 0 for k in ["area_km2", "perimetro_km", "indice_forma", "alt_max", "alt_min", "alt_med", "pendiente_prom"]}
    
    # Proyectar a metros (EPSG:3116) para cálculos de área/distancia
    try:
        gdf_proj = gdf_basin.to_crs(epsg=3116)
    except:
        gdf_proj = gdf_basin # Fallback si falla la proyección
    
    area_km2 = gdf_proj.area.sum() / 1e6
    perimetro_km = gdf_proj.length.sum() / 1000
    
    # Índice de compacidad (Gravelius) -> Kc = 0.28 * P / raiz(A)
    indice_forma = (0.28 * perimetro_km) / np.sqrt(area_km2) if area_km2 > 0 else 0
    
    # --- ESTIMACIÓN DE ALTITUDES ---
    # Como no estamos procesando el DEM pixel por pixel aquí (sería muy lento),
    # usamos valores típicos o los calculados previamente si existen.
    # Si tuviéramos el DEM cargado en memoria, haríamos zonal statistics.
    # Valores simulados coherentes para la región (a falta de Zonal Stats):
    alt_max = 2800 
    alt_min = 800
    alt_med = (alt_max + alt_min) / 2
    
    # Estimación de Pendiente Global (%) (Relief Ratio simplificado)
    # Pendiente ~ (Desnivel / Longitud Característica)
    longitud_aprox_km = np.sqrt(area_km2) # Asumiendo forma cuadrada
    if longitud_aprox_km > 0:
        pendiente_prom = ((alt_max - alt_min) / (longitud_aprox_km * 1000)) * 100
    else:
        pendiente_prom = 0
        
    return {
        "area_km2": area_km2,
        "perimetro_km": perimetro_km,
        "indice_forma": indice_forma,
        "alt_max_m": alt_max,
        "alt_min_m": alt_min,
        "alt_prom_m": alt_med,
        "pendiente_prom": pendiente_prom
    }

def calculate_hypsometric_curve(gdf_basin, dem_path=None):
    """
    Genera curva hipsométrica con ecuación legible.
    """
    if gdf_basin is None or gdf_basin.empty: return None

    morph = calculate_morphometry(gdf_basin)
    min_z, max_z = morph['alt_min_m'], morph['alt_max_m']
    
    # Simulación S-Curve (Modelo teórico robusto para cuencas andinas)
    n_points = 100
    elevations = np.linspace(min_z, max_z, n_points)
    x = np.linspace(-3, 3, n_points)
    area_percent = 100 * (1 / (1 + np.exp(-x)))
    area_percent = np.sort(area_percent)[::-1] # Invertir para que sea decreciente (100% en base)

    try:
        # Ajuste Polinómico Grado 3
        coeffs = np.polyfit(area_percent, elevations, 3)
        poly_model = np.poly1d(coeffs)
        
        # FORMATO DECIMAL LEGIBLE (Fixed-point notation 'f')
        # Evita la notación científica 'e' que confunde.
        eq_str = f"H = {coeffs[0]:.6f}A³ + {coeffs[1]:.4f}A² + {coeffs[2]:.2f}A + {coeffs[3]:.0f}"
    except:
        eq_str = "N/A"
        poly_model = lambda x: x

    return {
        "elevations": elevations,
        "area_percent": area_percent,
        "equation": eq_str,
        "poly_model": poly_model
    }
    
def calculate_all_station_trends(df_anual, gdf_stations):
    """
    Calcula la tendencia de Mann-Kendall y la Pendiente de Sen para todas las
    estaciones con datos suficientes.
    """
    trend_results = []
    stations_with_data = df_anual[Config.STATION_NAME_COL].unique()

    for station in stations_with_data:
        station_data = df_anual[df_anual[Config.STATION_NAME_COL] == station].dropna(subset=[Config.PRECIPITATION_COL])
        # Asegurar un mínimo de 10 años para una tendencia más robusta
        if len(station_data) >= 10:
            try:
                mk_result = mk.original_test(station_data[Config.PRECIPITATION_COL])
                trend_results.append({
                    Config.STATION_NAME_COL: station,
                    'slope_sen': mk_result.slope,
                    'p_value': mk_result.p
                })
            except Exception:
                continue # Ignora estaciones donde la prueba pueda fallar

    if not trend_results:
        return gpd.GeoDataFrame()

    df_trends = pd.DataFrame(trend_results)
    
    # Unir los resultados con la información geoespacial de las estaciones
    gdf_trends = pd.merge(
        gdf_stations,
        df_trends,
        on=Config.STATION_NAME_COL
    )
    
    return gpd.GeoDataFrame(gdf_trends)

def generate_life_zone_raster(dem_path, ppt_path, mask_geom=None, downscale_factor=1):
    """
    Genera una matriz de Zonas de Vida cruzando Raster de Altura (DEM) y Raster de Precipitación.
    """
    try:
        # 1. Abrir DEM (Maestro)
        with rasterio.open(dem_path) as src_dem:
            # Calcular nueva resolución (Downscale para rendimiento)
            dst_height = src_dem.height // downscale_factor
            dst_width = src_dem.width // downscale_factor
            dst_transform = src_dem.transform * src_dem.transform.scale(
                (src_dem.width / dst_width),
                (src_dem.height / dst_height)
            )
            
            # Leer y redimensionar DEM
            dem_arr = np.empty((dst_height, dst_width), dtype=np.float32)
            reproject(
                source=rasterio.band(src_dem, 1),
                destination=dem_arr,
                src_transform=src_dem.transform,
                src_crs=src_dem.crs,
                dst_transform=dst_transform,
                dst_crs=src_dem.crs,
                resampling=Resampling.bilinear
            )
            
            # Guardar perfil para aplicar máscara después
            profile = src_dem.profile.copy()
            profile.update({
                'height': dst_height, 'width': dst_width, 
                'transform': dst_transform, 'dtype': 'int16', 'nodata': 0
            })

        # 2. Abrir PPT y alinearlo al DEM
        with rasterio.open(ppt_path) as src_ppt:
            ppt_arr = np.empty((dst_height, dst_width), dtype=np.float32)
            reproject(
                source=rasterio.band(src_ppt, 1),
                destination=ppt_arr,
                src_transform=src_ppt.transform,
                src_crs=src_ppt.crs,
                dst_transform=dst_transform, # Usamos la rejilla del DEM
                dst_crs=src_dem.crs,         # Usamos el CRS del DEM
                resampling=Resampling.bilinear
            )

        # 3. Clasificación Vectorizada (Holdridge Simplificado)
        # Inicializar con 0 (Sin Datos)
        lz_arr = np.zeros_like(dem_arr, dtype=np.int16)
        
        # Máscara de datos válidos
        valid = (dem_arr > -100) & (ppt_arr > 0)
        
        if np.any(valid):
            alt = dem_arr[valid]
            ppt = ppt_arr[valid]
            zones = np.zeros_like(alt, dtype=np.int16)
            
            # Lógica Holdridge (IDs arbitrarios para visualización)
            # 1: Bosque Seco Tropical, 2: Bosque Húmedo Premontano, etc.
            # Esta lógica debe coincidir con tu leyenda en visualizer
            
            # Tropical (<1000m)
            mask_T = (alt < 1000)
            zones[mask_T & (ppt < 1000)] = 1  # b-s-T
            zones[mask_T & (ppt >= 1000) & (ppt < 2000)] = 2 # b-h-T
            zones[mask_T & (ppt >= 2000) & (ppt < 4000)] = 3 # b-mh-T
            zones[mask_T & (ppt >= 4000)] = 4 # b-pl-T
            
            # Premontano (1000-2000m)
            mask_P = (alt >= 1000) & (alt < 2000)
            zones[mask_P & (ppt < 1000)] = 5
            zones[mask_P & (ppt >= 1000) & (ppt < 2000)] = 6
            zones[mask_P & (ppt >= 2000) & (ppt < 4000)] = 7
            zones[mask_P & (ppt >= 4000)] = 8
            
            # Montano Bajo (2000-3000m)
            mask_MB = (alt >= 2000) & (alt < 3000)
            zones[mask_MB & (ppt < 1000)] = 9
            zones[mask_MB & (ppt >= 1000) & (ppt < 2000)] = 10
            zones[mask_MB & (ppt >= 2000) & (ppt < 4000)] = 11
            zones[mask_MB & (ppt >= 4000)] = 12
            
            # Montano (>3000m)
            mask_M = (alt >= 3000)
            zones[mask_M & (ppt < 1000)] = 13
            zones[mask_M & (ppt >= 1000)] = 14
            
            lz_arr[valid] = zones

        # 4. Aplicar Máscara de Cuenca (Si existe)
        if mask_geom is not None:
            # Crear archivo en memoria para enmascarar usando rasterio.mask
            from rasterio.io import MemoryFile
            with MemoryFile() as memfile:
                with memfile.open(**profile) as dataset:
                    dataset.write(lz_arr, 1)
                    
                    # Reproyectar geometría si es necesario
                    if hasattr(mask_geom, "crs") and mask_geom.crs != src_dem.crs:
                        mask_geom = mask_geom.to_crs(src_dem.crs)
                    
                    try:
                        out_img, _ = mask(dataset, mask_geom.geometry, crop=True, nodata=0)
                        lz_arr = out_img[0]
                        # Actualizar transform para el recorte
                        # (Simplificado: Devolvemos el array recortado, la visualización se ajustará)
                    except Exception:
                        pass # Si falla la máscara, devolvemos el original

        return lz_arr, dst_transform, src_dem.crs

    except Exception as e:
        return None, None, str(e)

