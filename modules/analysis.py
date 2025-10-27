# modules/analysis.py

import streamlit as st
import geopandas as gpd
import pandas as pd
import requests
import numpy as np
from scipy.stats import gamma, norm
from modules.config import Config
import rasterio
from rasterstats import zonal_stats
import pymannkendall as mk


@st.cache_data
def calculate_spi(series, window):
    """
    Calcula el Índice de Precipitación Estandarizado (SPI), manejando ceros.
    """
    # Asegurar que la serie tenga índice de fecha y esté ordenada
    series = series.sort_index()
    
    # Calcular la suma acumulada (rolling sum)
    rolling_sum = series.rolling(window=window, min_periods=window).sum()
    
    # --- Manejo de Ceros ---
    # 1. Separar los datos válidos (no NaN)
    data_valid = rolling_sum.dropna()
    data_valid = data_valid[np.isfinite(data_valid)] # Quitar inf si los hubiera
    
    if data_valid.empty:
        st.warning(f"SPI-{window}: No hay suficientes datos válidos para el cálculo.")
        return pd.Series(dtype=float)

    # 2. Calcular la probabilidad de cero (q)
    n_total = len(data_valid)
    n_zeros = (data_valid == 0).sum()
    q = n_zeros / n_total
    
    # 3. Separar los datos estrictamente positivos para el ajuste Gamma
    data_positive = data_valid[data_valid > 0]
    
    spi = pd.Series(np.nan, index=rolling_sum.index) # Serie de salida inicializada con NaN

    # --- Ajuste Gamma y Cálculo CDF (Solo si hay datos positivos) ---
    if not data_positive.empty:
        try:
            # 4. Ajustar la distribución Gamma SOLO a los datos positivos
            params = gamma.fit(data_positive, floc=0) 
            shape, loc, scale = params # loc debería ser 0 por floc=0
            
            # 5. Calcular el CDF Gamma (G(x)) para TODOS los datos válidos (incluyendo ceros)
            #    gamma.cdf(0, ...) dará 0, lo cual es correcto aquí.
            cdf_gamma = gamma.cdf(data_valid, shape, loc=loc, scale=scale)
            
            # 6. Calcular el CDF combinado H(x) = q + (1 - q) * G(x)
            prob_combined = q + (1 - q) * cdf_gamma
            
            # Asegurar que las probabilidades estén estrictamente entre 0 y 1 para norm.ppf
            prob_combined = np.clip(prob_combined, 1e-6, 1 - 1e-6) 
            
            # 7. Transformar a SPI usando la inversa de la normal estándar
            spi_calculated = norm.ppf(prob_combined)
            
            # Asignar los valores calculados al índice correcto en la serie de salida
            spi.loc[data_valid.index] = spi_calculated

        except Exception as e:
            st.warning(f"SPI-{window}: Falló el ajuste Gamma o cálculo CDF. Error: {e}")
            # Dejar los valores como NaN si falla el ajuste/cálculo
            pass # spi ya está inicializada con NaN

    # Manejar los casos donde solo hubo ceros (si q=1)
    elif q == 1:
         # Si todos los valores válidos son cero, el SPI técnicamente no está bien definido, 
         # pero asignar 0 es una convención razonable (precipitación exactamente igual a la media de ceros).
         spi.loc[data_valid.index] = 0.0

    # Reemplazar infinitos (por si acaso clip no fue suficiente) y devolver
    spi.replace([np.inf, -np.inf], np.nan, inplace=True)
    return spi

@st.cache_data
def calculate_spei(precip_series, et_series, scale):

    st.write(f"Debug calculate_spei: Recibida et_series. Es None? {et_series is None}")
    if et_series is not None:
        st.write(f"Debug calculate_spei: Primeros valores de et_series:\n{et_series.head()}")
        st.write(f"Debug calculate_spei: Hay valores no nulos en et_series? {et_series.notna().any()}")
        
    """
    Calcula el Índice de Precipitación y Evapotranspiración Estandarizado (SPEI).
    """
    from scipy.stats import loglaplace
    scale = int(scale)
    df = pd.DataFrame({'precip': precip_series, 'et': et_series})
    df = df.sort_index().asfreq('MS')
    df.dropna(inplace=True)
    if len(df) < scale * 2:
        return pd.Series(dtype=float)

    water_balance = df['precip'] - df['et']
    rolling_balance = water_balance.rolling(window=scale, min_periods=scale).sum()
    data_for_fit = rolling_balance.dropna()
    data_for_fit = data_for_fit[np.isfinite(data_for_fit)]

    if len(data_for_fit) > 0:
        params = loglaplace.fit(data_for_fit)
        cdf = loglaplace.cdf(rolling_balance, *params)
    else:
        return pd.Series(dtype=float)
        
    spei = norm.ppf(cdf)
    spei = np.where(np.isinf(spei), np.nan, spei)
    return pd.Series(spei, index=rolling_balance.index)
    
@st.cache_data
def calculate_monthly_anomalies(_df_monthly_filtered, _df_long):
    """
    Calcula las anomalías mensuales con respecto al promedio de todo el período de datos.
    """
    df_monthly_filtered = _df_monthly_filtered.copy()
    df_long = _df_long.copy()
    
    df_climatology = df_long[
        df_long[Config.STATION_NAME_COL].isin(df_monthly_filtered[Config.STATION_NAME_COL].unique())
    ].groupby([Config.STATION_NAME_COL, Config.MONTH_COL])[Config.PRECIPITATION_COL].mean() \
     .reset_index().rename(columns={Config.PRECIPITATION_COL: 'precip_promedio_mes'})

    df_anomalias = pd.merge(
        df_monthly_filtered,
        df_climatology,
        on=[Config.STATION_NAME_COL, Config.MONTH_COL],
        how='left'
    )
    df_anomalias['anomalia'] = df_anomalias[Config.PRECIPITATION_COL] - df_anomalias['precip_promedio_mes']
    return df_anomalias.copy()

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

def calculate_hydrological_balance(mean_precip_mm, mean_altitude_m, basin_geometry_input):
    """
    Calcula el balance hídrico (P - ET = Q) para una cuenca.
    Recibe la altitud media como parámetro (del DEM) en lugar de llamarla de una API.
    """
    results = {
        "P_media_anual_mm": mean_precip_mm,
        "Altitud_media_m": mean_altitude_m,
        "ET_media_anual_mm": None,
        "Q_mm": None,
        "Q_m3_año": None,
        "Area_km2": None,
        "error": None
    }

    if mean_altitude_m is None or pd.isna(mean_altitude_m):
        results["error"] = "No se pudo calcular el balance; la altitud media es desconocida (N/A). Verifique el DEM."
        return results

    # 1. Calcular Evapotranspiración (ET) usando la altitud
    eto_dia = 4.37 * np.exp(-0.0002 * mean_altitude_m)
    eto_anual_mm = eto_dia * 365.25
    results["ET_media_anual_mm"] = eto_anual_mm

    # 2. Calcular la Escorrentía (Q)
    q_mm = mean_precip_mm - eto_anual_mm
    results["Q_mm"] = q_mm

    # 3. Calcular el Caudal en Volumen
    try:
        basin_metric = basin_geometry_input.to_crs("EPSG:3116") # Proyección métrica para área
        area_m2 = basin_metric.area.sum()
        results["Area_km2"] = area_m2 / 1_000_000
        
        q_m = q_mm / 1000
        q_volumen_m3_anual = q_m * area_m2
        results["Q_m3_año"] = q_volumen_m3_anual
    except Exception as e:
        results["error"] = f"Error al calcular el área de la cuenca: {e}"

    return results
    
def calculate_morphometry(basin_gdf, dem_path):
    """
    Calcula la morfometría, re-proyectando la cuenca al CRS del DEM
    para asegurar la compatibilidad y evitar errores de 'N/A'.
    """
    if basin_gdf.empty or basin_gdf.iloc[0].geometry is None:
        return {"error": "Geometría de cuenca no válida."}

    # --- Cálculos Geométricos (en proyección métrica) ---
    basin_metric = basin_gdf.to_crs("EPSG:3116")
    geom_metric = basin_metric.iloc[0].geometry
    area_m2 = geom_metric.area
    perimetro_m = geom_metric.length
    indice_forma = perimetro_m / (2 * np.sqrt(np.pi * area_m2)) if area_m2 > 0 else 0

    # --- Cálculos de Elevación (en la proyección del DEM) ---
    stats = {}
    try:
        with rasterio.open(dem_path) as src:
            nodata_value = src.nodata
            dem_crs = src.crs # Obtener el CRS del DEM

        # Reproyectar la cuenca al CRS del DEM para el análisis zonal
        basin_reprojected = basin_gdf.to_crs(dem_crs)
        
        z_stats = zonal_stats(basin_reprojected, dem_path, stats="min max mean", nodata=nodata_value)
        
        if z_stats and z_stats[0].get('min') is not None:
            stats['alt_min'] = z_stats[0].get('min')
            stats['alt_max'] = z_stats[0].get('max')
            stats['alt_prom'] = z_stats[0].get('mean')
        else:
            stats['error_dem'] = "No se encontraron datos de elevación válidos en el área de la cuenca."

    except Exception as e:
        stats['error_dem'] = f"No se pudieron calcular las estadísticas de elevación: {e}"

    return {
        "area_km2": area_m2 / 1_000_000,
        "perimetro_km": perimetro_m / 1_000,
        "indice_forma": indice_forma,
        "alt_max_m": stats.get('alt_max'),
        "alt_min_m": stats.get('alt_min'),
        "alt_prom_m": stats.get('alt_prom'),
        "error": stats.get('error_dem')
    }

def calculate_hypsometric_curve(basin_gdf, dem_path):
    """
    Calcula los datos para la curva hipsométrica, ajusta un polinomio de grado 3
    y devuelve los datos, la ecuación y el R².
    """
    try:
        with rasterio.open(dem_path) as src:
            dem_crs = src.crs
            basin_reprojected = basin_gdf.to_crs(dem_crs)
            
            zonal_result = zonal_stats(
                basin_reprojected,
                dem_path,
                stats="count", # Pedimos 'count' para asegurar que obtenemos los datos
                raster_out=True, # ¡Clave! Extrae los valores crudos del ráster
                nodata=src.nodata
            )
            
            if not zonal_result or 'mini_raster_array' not in zonal_result[0]:
                 return {"error": "No se pudo extraer datos del ráster para la cuenca."}
                 
            elevations = zonal_result[0]['mini_raster_array'].compressed()
            
            if elevations.size == 0:
                return {"error": "No se encontraron valores de elevación válidos en la cuenca."}

        # Ordenar las elevaciones de menor a mayor
        elevations_sorted = np.sort(elevations)
        total_pixels = len(elevations_sorted)
        
        # Calcular el porcentaje de área acumulada que está POR ENCIMA de una elevación dada
        # (1 - [posición / total]) * 100
        cumulative_area_percent = (1 - np.arange(total_pixels) / total_pixels) * 100

        # --- AJUSTE DE CURVA POLINOMIAL (TU SOLICITUD) ---
        # Normalizamos el eje X (área) a un rango de [0, 1] para mejorar la estabilidad numérica
        x_norm = cumulative_area_percent / 100.0
        
        # Ajustamos un polinomio de grado 3 (puedes cambiar el '3' si quieres otro grado)
        coeffs = np.polyfit(x_norm, elevations_sorted, 3)
        p = np.poly1d(coeffs) # 'p' es ahora la función polinomial
        
        # Generamos los puntos de la curva ajustada para graficar
        x_fit = np.linspace(0, 100, 200) # 200 puntos para una curva suave
        y_fit = p(x_fit / 100.0)
        
        # Calculamos el R² (coeficiente de determinación)
        y_predicted = p(x_norm)
        ss_res = np.sum((elevations_sorted - y_predicted) ** 2)
        ss_tot = np.sum((elevations_sorted - np.mean(elevations_sorted)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Formateamos la ecuación para mostrarla (y = ax³ + bx² + cx + d)
        equation = f"y = {coeffs[0]:.2f}x³ + {coeffs[1]:.2f}x² + {coeffs[2]:.2f}x + {coeffs[3]:.0f}"
        # --- FIN DEL AJUSTE ---

        return {
            "elevations": elevations_sorted,
            "cumulative_area_percent": cumulative_area_percent,
            "fit_x": x_fit,
            "fit_y": y_fit,
            "equation": equation,
            "r_squared": r_squared
        }
    except Exception as e:
        return {"error": f"Error al calcular la curva hipsométrica: {e}"}

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


