# modules/interpolation.py

import pandas as pd
import geopandas as gpd
import numpy as np
import gstools as gs
import rasterio
from rasterio.transform import from_origin
from rasterio.mask import mask
import plotly.graph_objects as go
from modules.config import Config
import streamlit as st
from scipy.interpolate import Rbf
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.ndimage import gaussian_filter

def interpolate_idw(lons, lats, vals, grid_lon, grid_lat, method='cubic'):
    """
    Realiza una interpolación espacial utilizando scipy.griddata.
    Es mucho más rápido que una implementación manual de IDW.

    Args:
        lons (array): Longitudes de los puntos de datos.
        lats (array): Latitudes de los puntos de datos.
        vals (array): Valores en los puntos de datos.
        grid_lon (array): Coordenadas de longitud de la grilla de salida.
        grid_lat (array): Coordenadas de latitud de la grilla de salida.
        method (str): Método de interpolación ('cubic', 'linear', 'nearest').
                      'cubic' produce resultados más suaves.

    Returns:
        array: La grilla interpolada (grid_z).
    """
    # 1. Preparar los puntos de datos en el formato correcto (N, 2)
    points = np.column_stack((lons, lats))

    # 2. Crear una malla (meshgrid) a partir de los vectores de la grilla de salida
    grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)

    # 3. Realizar la interpolación
    # griddata es altamente optimizada y realiza el trabajo pesado.
    grid_z = griddata(points, vals, (grid_x, grid_y), method=method)

    # 4. griddata puede dejar NaNs en los bordes. Los rellenamos con 0.
    grid_z = np.nan_to_num(grid_z)

    # La salida de griddata ya tiene la forma (ny, nx), por lo que la retornamos directamente.
    return grid_z

# -----------------------------------------------------------------------------
# NUEVA FUNCIÓN PÚBLICA PARA LA PESTAÑA DE VALIDACIÓN
# -----------------------------------------------------------------------------
@st.cache_data
def perform_loocv_for_year(year, method, gdf_metadata, df_anual_non_na):
    """
    Realiza una Validación Cruzada Dejando Uno Afuera (LOOCV) para un año y método dados.
    Devuelve las métricas de error (RMSE y MAE).
    """
    df_year = pd.merge(
        df_anual_non_na[df_anual_non_na[Config.YEAR_COL] == year],
        gdf_metadata,
        on=Config.STATION_NAME_COL
    )
    
    clean_cols = [Config.LONGITUDE_COL, Config.LATITUDE_COL, Config.PRECIPITATION_COL]
    if method == "Kriging con Deriva Externa (KED)" and Config.ELEVATION_COL in df_year.columns:
        clean_cols.append(Config.ELEVATION_COL)

    df_clean = df_year.dropna(subset=clean_cols).copy()
    df_clean = df_clean[np.isfinite(df_clean[clean_cols]).all(axis=1)]
    df_clean = df_clean.drop_duplicates(subset=[Config.LONGITUDE_COL, Config.LATITUDE_COL])

    if len(df_clean) < 4:
        return {'RMSE': np.nan, 'MAE': np.nan}

    lons = df_clean[Config.LONGITUDE_COL].values
    lats = df_clean[Config.LATITUDE_COL].values
    vals = df_clean[Config.PRECIPITATION_COL].values
    elevs = df_clean[Config.ELEVATION_COL].values if Config.ELEVATION_COL in df_clean else None
    
    return _perform_loocv(method, lons, lats, vals, elevs)

@st.cache_data
def perform_loocv_for_all_methods(_year, _gdf_metadata, _df_anual_non_na):
    """Ejecuta LOOCV para todos los métodos de interpolación para un año dado."""
    methods = ["Kriging Ordinario", "IDW", "Spline (Thin Plate)"]
    if Config.ELEVATION_COL in _gdf_metadata.columns:
        methods.insert(1, "Kriging con Deriva Externa (KED)")
    
    results = []
    for method in methods:
        metrics = perform_loocv_for_year(_year, method, _gdf_metadata, _df_anual_non_na)
        if metrics:
            results.append({
                "Método": method,
                "Año": _year,
                "RMSE": metrics.get('RMSE'),
                "MAE": metrics.get('MAE')
            })
    return pd.DataFrame(results)
# -----------------------------------------------------------------------------
# FUNCIÓN ORIGINAL, AHORA ACTUALIZADA PARA USAR LA FUNCIÓN AUXILIAR
# -----------------------------------------------------------------------------
@st.cache_data
def create_interpolation_surface(year, method, variogram_model, gdf_bounds, gdf_metadata, df_anual_non_na):
    """Crea una superficie de interpolación y calcula el error RMSE."""

    fig_var = None # Renamed fig_variogram to fig_var to avoid potential conflicts if plt wasn't closed
    error_msg = None
    
    # --- Data Preparation ---
    df_year = pd.merge(
        df_anual_non_na[df_anual_non_na[Config.YEAR_COL] == year],
        gdf_metadata,
        on=Config.STATION_NAME_COL,
        how='inner' # Use inner merge to ensure only stations with metadata are kept
    )
    
    clean_cols = [Config.LONGITUDE_COL, Config.LATITUDE_COL, Config.PRECIPITATION_COL]
    if method == "Kriging con Deriva Externa (KED)" and Config.ELEVATION_COL in df_year.columns:
        clean_cols.append(Config.ELEVATION_COL)
        
    df_clean = df_year.dropna(subset=clean_cols).copy()
    # Convert coordinates to numeric robustly
    df_clean[Config.LONGITUDE_COL] = pd.to_numeric(df_clean[Config.LONGITUDE_COL], errors='coerce')
    df_clean[Config.LATITUDE_COL] = pd.to_numeric(df_clean[Config.LATITUDE_COL], errors='coerce')
    df_clean[Config.PRECIPITATION_COL] = pd.to_numeric(df_clean[Config.PRECIPITATION_COL], errors='coerce')
    df_clean = df_clean.dropna(subset=[Config.LONGITUDE_COL, Config.LATITUDE_COL, Config.PRECIPITATION_COL]) # Drop rows where conversion failed
    
    df_clean = df_clean[np.isfinite(df_clean[clean_cols]).all(axis=1)] # Check for inf/-inf
    df_clean = df_clean.drop_duplicates(subset=[Config.LONGITUDE_COL, Config.LATITUDE_COL])

    if len(df_clean) < 4:
        error_msg = f"Se necesitan al menos 4 estaciones con datos válidos para el año {year} para interpolar (encontradas: {len(df_clean)})."
        fig = go.Figure().update_layout(title=error_msg, xaxis_visible=False, yaxis_visible=False)
        return fig, None, error_msg # Devuelve 3 valores

    lons = df_clean[Config.LONGITUDE_COL].values
    lats = df_clean[Config.LATITUDE_COL].values
    vals = df_clean[Config.PRECIPITATION_COL].values
    elevs = df_clean[Config.ELEVATION_COL].values if Config.ELEVATION_COL in df_clean else None

    # Calculate RMSE using the auxiliary function
    rmse = None # Initialize rmse
    try:
        # Check if _perform_loocv exists before calling it
        if '_perform_loocv' in globals() or '_perform_loocv' in locals():
            metrics = _perform_loocv(method, lons, lats, vals, elevs) 
            rmse = metrics.get('RMSE')
        else:
            # st.warning("Función _perform_loocv no encontrada. RMSE no se calculará.") # Use print for backend logs
            print("Warning: Función _perform_loocv no encontrada. RMSE no se calculará.")
    except Exception as e_rmse:
         # st.warning(f"Error calculando RMSE con _perform_loocv: {e_rmse}") # Use print for backend logs
         print(f"Warning: Error calculando RMSE con _perform_loocv: {e_rmse}")

    # Define grid based on bounds
    if gdf_bounds is None or len(gdf_bounds) != 4 or not all(np.isfinite(gdf_bounds)):
         error_msg = "Límites geográficos (bounds) inválidos o no proporcionados."
         fig = go.Figure().update_layout(title=error_msg, xaxis_visible=False, yaxis_visible=False)
         return fig, None, error_msg

    grid_lon = np.linspace(gdf_bounds[0] - 0.1, gdf_bounds[2] + 0.1, 150) # Reduced points for speed
    grid_lat = np.linspace(gdf_bounds[1] - 0.1, gdf_bounds[3] + 0.1, 150) # Reduced points for speed
    z_grid, error_message = None, None

    # --- Interpolation Calculation ---
    try:
        if method in ["Kriging Ordinario", "Kriging con Deriva Externa (KED)"]:
            model_map = {'gaussian': gs.Gaussian(dim=2), 'exponential': gs.Exponential(dim=2),
                         'spherical': gs.Spherical(dim=2), 'linear': gs.Linear(dim=2)}
            model = model_map.get(variogram_model, gs.Spherical(dim=2))
            # Increased len_scale_max for potentially larger ranges
            bin_center, gamma = gs.vario_estimate((lons, lats), vals)
            # Added error handling for fit_variogram
            try:
                model.fit_variogram(bin_center, gamma, nugget=True)
            except ValueError as e_fit:
                 raise ValueError(f"Error ajustando variograma: {e_fit}. Datos insuficientes o sin varianza espacial?")

            # Variogram plot generation (keep if needed, otherwise comment out)
            try:
                fig_variogram_plt, ax = plt.subplots(figsize=(6, 4)) 
                ax.plot(bin_center, gamma, 'o', label='Experimental')
                model.plot(ax=ax, label='Modelo Ajustado')
                ax.set_xlabel('Distancia'); ax.set_ylabel('Semivarianza')
                ax.set_title(f'Variograma para {year}'); ax.legend()
                plt.tight_layout() 
                # Instead of assigning fig_variogram_plt, keep ax if you return it, or close explicitly
                # fig_var = fig_variogram_plt # Keep if you return the matplotlib figure
                plt.close(fig_variogram_plt) # Close the plot to free memory if not returned
            except Exception as e_plot:
                 print(f"Warning: No se pudo generar el gráfico del variograma: {e_plot}")
                 fig_var = None


            if method == "Kriging Ordinario":
                krig = gs.krige.Ordinary(model, (lons, lats), vals)
                z_grid, _ = krig.structured([grid_lon, grid_lat]) 
            else: # KED
                if elevs is None:
                     raise ValueError("Datos de elevación necesarios para KED no encontrados.")
                # Using griddata for elevation interpolation - RBF can be slow/memory intensive
                grid_x_elev, grid_y_elev = np.meshgrid(grid_lon, grid_lat)
                drift_grid = griddata((lons, lats), elevs, (grid_x_elev, grid_y_elev), method='linear')
                nan_mask_elev = np.isnan(drift_grid)
                if np.any(nan_mask_elev): # Fill NaNs if any
                    fill_values_elev = griddata((lons, lats), elevs, (grid_x_elev[nan_mask_elev], grid_y_elev[nan_mask_elev]), method='nearest')
                    drift_grid[nan_mask_elev] = fill_values_elev
                drift_grid = np.nan_to_num(drift_grid) # Ensure no NaNs remain

                krig = gs.krige.ExtDrift(model, (lons, lats), vals, drift_src=elevs)
                z_grid, _ = krig.structured([grid_lon, grid_lat], drift_tgt=drift_grid.T) 

        elif method == "IDW":
            grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)
            z_grid = griddata((lons, lats), vals, (grid_x, grid_y), method='linear') # Linear for speed
            nan_mask = np.isnan(z_grid)
            if np.any(nan_mask):
                fill_values = griddata((lons, lats), vals, (grid_x[nan_mask], grid_y[nan_mask]), method='nearest')
                z_grid[nan_mask] = fill_values
            
            if z_grid is not None:
                z_grid = np.nan_to_num(z_grid) # Replace any final NaNs with 0 AFTER checking range
                print(f"Output 'z_grid' (shape:{z_grid.shape}) Min: {np.nanmin(z_grid):.2f}, Max: {np.nanmax(z_grid):.2f}, Mean: {np.nanmean(z_grid):.2f}")
            else:
                print("Output 'z_grid' is None after griddata")

        elif method == "Spline (Thin Plate)":
            # Using griddata with 'cubic' as approximation for spline
            grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)
            z_grid = griddata((lons, lats), vals, (grid_x, grid_y), method='cubic')
            nan_mask = np.isnan(z_grid)
            if np.any(nan_mask): # Fill border NaNs
                fill_values = griddata((lons, lats), vals, (grid_x[nan_mask], grid_y[nan_mask]), method='nearest')
                z_grid[nan_mask] = fill_values

        # Ensure z_grid is not None and handle potential remaining NaNs
        if z_grid is not None:
             z_grid = np.nan_to_num(z_grid) # Replace any final NaNs with 0

    except Exception as e:
        error_message = f"Error al calcular {method}: {e}"
        import traceback
        print(traceback.format_exc()) # Print full traceback to logs
        fig = go.Figure().update_layout(title=error_message, xaxis_visible=False, yaxis_visible=False)
        # Ensure fig_variogram is handled if created
        # if 'fig_variogram_plt' in locals() and fig_variogram_plt: plt.close(fig_variogram_plt) 
        return fig, None, error_message # Devuelve 3 valores

    # --- Plotting Section ---
    if z_grid is not None:
        fig = go.Figure(data=go.Contour(
            z=z_grid.T, x=grid_lon, y=grid_lat,
            colorscale=px.colors.sequential.YlGnBu, 
            colorbar_title='Precipitación (mm)',
            contours=dict(
                coloring='heatmap', 
                showlabels=True, 
                labelfont=dict(size=10, color='white'), 
                labelformat=".0f" 
            ),
            line_smoothing=0.85, 
            line_color='black',  
            line_width=0.5       
        ))
        
        hover_texts = [
             f"<b>{row[Config.STATION_NAME_COL]}</b><br>" +
             f"Municipio: {row.get(Config.MUNICIPALITY_COL, 'N/A')}<br>" + 
             f"Altitud: {row.get(Config.ALTITUDE_COL, 'N/A')} m<br>" +     
             f"Precipitación: {row[Config.PRECIPITATION_COL]:.0f} mm"
             for _, row in df_clean.iterrows() 
        ]

        fig.add_trace(go.Scatter(
            x=lons, y=lats, mode='markers', marker=dict(color='red', size=5, line=dict(width=1,
            color='black')),
            name='Estaciones',
            hoverinfo='text',
            text=hover_texts 
        ))
        
        if rmse is not None:
            fig.add_annotation(
                x=0.01, y=0.99, xref="paper", yref="paper",
                text=f"<b>RMSE: {rmse:.1f} mm</b>", align='left',
                showarrow=False, font=dict(size=12, color="black"),
                bgcolor="rgba(255, 255, 255, 0.7)", bordercolor="black", borderwidth=1
            )
        fig.update_layout(
            title=f"Precipitación en {year} ({method})",
            xaxis_title="Longitud", yaxis_title="Latitud", height=600,
            legend=dict(x=0.01, y=0.01, bgcolor="rgba(0,0,0,0)")
        )
        # Decide whether to return the matplotlib fig_var or None
        # If returning matplotlib figure, ensure plt.close() is NOT called earlier
        # If NOT returning matplotlib figure, ensure plt.close() IS called earlier
        return fig, None, None # Returning None for fig_var for simplicity now

    # Fallback if z_grid ended up being None
    # Ensure matplotlib plot is closed if created
    # if 'fig_variogram_plt' in locals() and fig_variogram_plt: plt.close(fig_variogram_plt) 
    return go.Figure().update_layout(title="Error: No se pudo generar la superficie Z"), None, "Superficie Z es None"
    
@st.cache_data
def create_kriging_by_basin(gdf_points, grid_lon, grid_lat, value_col='Valor'):
    """
    Realiza Kriging. Si falla, usa un respaldo de interpolación lineal y relleno
    para asegurar una superficie con gradiente y sin vacíos.
    """
    lons = gdf_points.geometry.x
    lats = gdf_points.geometry.y
    vals = gdf_points[value_col].values
    
    valid_indices = ~np.isnan(vals)
    lons, lats, vals = lons[valid_indices], lats[valid_indices], vals[valid_indices]

    if len(vals) < 3:
        st.error("Se necesitan al menos 3 puntos con datos para realizar la interpolación.")
        ny, nx = len(grid_lat), len(grid_lon)
        return np.zeros((ny, nx)), np.zeros((ny, nx))

    try:
        st.write("🛰️ Intentando interpolación con Kriging Ordinario...")
        bin_center, gamma = gs.vario_estimate((lons, lats), vals)
        model = gs.Spherical(dim=2)
        model.fit_variogram(bin_center, gamma, nugget=True)
        kriging = gs.krige.Ordinary(model, cond_pos=(lons, lats), cond_val=vals)
        grid_z, variance = kriging.structured([grid_lon, grid_lat], return_var=True)
        st.success("✅ Interpolación con Kriging completada con éxito.")
    except (RuntimeError, ValueError) as e: # Capturamos más tipos de error
        st.warning(f"⚠️ El Kriging falló: '{e}'. Usando interpolación de respaldo (lineal + vecino cercano).")
        points = np.column_stack((lons, lats))
        grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)
        
        grid_z = griddata(points, vals, (grid_x, grid_y), method='linear')
        nan_mask = np.isnan(grid_z)
        if np.any(nan_mask):
            fill_values = griddata(points, vals, (grid_x[nan_mask], grid_y[nan_mask]), method='nearest')
            grid_z[nan_mask] = fill_values
        
        grid_z = np.nan_to_num(grid_z)
        variance = np.zeros_like(grid_z)

    return grid_z, variance









