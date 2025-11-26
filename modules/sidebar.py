import streamlit as st
from modules.config import Config
import pandas as pd
import numpy as np

def create_sidebar(gdf_stations, df_long):
    with st.sidebar:
        if hasattr(Config, 'LOGO_PATH'):
            try: st.image(Config.LOGO_PATH, width=150) 
            except: pass
            
        st.title("Panel de Control")

        # --- 1. Filtros de Procesamiento ---
        with st.expander("🛠️ Procesamiento y Calidad", expanded=False):
            run_complete_series = st.checkbox("Interpolación (Rellenar huecos)", value=False)
            exclude_nulls = st.checkbox("Excluir datos nulos (NaN)", value=False)
            exclude_zeros = st.checkbox("Excluir valores cero (0)", value=False)
            
            st.markdown("---")
            # FILTRO POR % DE DATOS (MANTENIDO)
            min_pct = st.slider("Mínimo % de Datos Disponibles:", 0, 100, 0, 
                                help="Filtra estaciones que tengan al menos este porcentaje de datos en el histórico.")

            if run_complete_series != st.session_state.get('apply_interpolation'):
                st.session_state['apply_interpolation'] = run_complete_series
                st.rerun()

        st.divider()

        # --- 2. Filtros de Ubicación ---
        st.markdown("### 📍 Filtros de Ubicación")
        
        # Lógica de Filtrado Inicial (MANTENIDA)
        valid_stations_by_pct = gdf_stations[Config.STATION_NAME_COL].unique()
        
        if min_pct > 0 and df_long is not None:
            counts = df_long.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].count()
            n_years = df_long[Config.YEAR_COL].max() - df_long[Config.YEAR_COL].min() + 1
            total_months = n_years * 12
            pcts = (counts / total_months) * 100
            valid_stations_by_pct = pcts[pcts >= min_pct].index.tolist()

        # A. Filtro por Altitud (MANTENIDO)
        altitude_options = ["Todos", "0-500", "500-1000", "1000-1500", "1500-2000", "2000-3000", ">3000"]
        selected_alt_range = st.selectbox("Filtrar por Altitud (m):", altitude_options)
        
        # Aplicar filtros base
        gdf_filtered_base = gdf_stations[gdf_stations[Config.STATION_NAME_COL].isin(valid_stations_by_pct)].copy()
        
        if selected_alt_range != "Todos":
            if ">" in selected_alt_range:
                min_alt = int(selected_alt_range.replace(">", ""))
                gdf_filtered_base = gdf_filtered_base[gdf_filtered_base[Config.ALTITUDE_COL] >= min_alt]
            else:
                min_alt, max_alt = map(int, selected_alt_range.split("-"))
                gdf_filtered_base = gdf_filtered_base[
                    (gdf_filtered_base[Config.ALTITUDE_COL] >= min_alt) & 
                    (gdf_filtered_base[Config.ALTITUDE_COL] < max_alt)
                ]

        # B. Región (MANTENIDO)
        if Config.REGION_COL in gdf_filtered_base.columns:
            all_regions = sorted(gdf_filtered_base[Config.REGION_COL].astype(str).unique())
            selected_regions = st.multiselect("Región:", all_regions)
            if selected_regions:
                gdf_filtered_base = gdf_filtered_base[gdf_filtered_base[Config.REGION_COL].isin(selected_regions)]
        else:
            selected_regions = []

        # C. Municipio (MANTENIDO)
        all_munis = sorted(gdf_filtered_base[Config.MUNICIPALITY_COL].astype(str).unique())
        selected_municipios = st.multiselect("Municipio:", all_munis)
        if selected_municipios:
            gdf_filtered_base = gdf_filtered_base[gdf_filtered_base[Config.MUNICIPALITY_COL].isin(selected_municipios)]

        # D. Selección de Estaciones (MANTENIDO)
        available_stations = sorted(gdf_filtered_base[Config.STATION_NAME_COL].astype(str).unique())
        
        with st.expander(f"Estaciones ({len(available_stations)} disp.)", expanded=True):
            select_all = st.checkbox("Seleccionar Todas las visibles")
            
            if select_all:
                default_stations = available_stations
                if len(available_stations) > 50:
                    st.caption("⚠️ Muchas estaciones seleccionadas. El rendimiento puede variar.")
            else:
                default_stations = available_stations[:3] if len(available_stations) > 0 else []

            stations_for_analysis = st.multiselect(
                "Seleccione específicas:",
                options=available_stations,
                default=default_stations,
                label_visibility="collapsed"
            )

        gdf_final = gdf_stations[gdf_stations[Config.STATION_NAME_COL].isin(stations_for_analysis)]

        st.divider()

        # --- 3. Filtro de Tiempo (MANTENIDO Y MEJORADO) ---
        st.markdown("### 📅 Periodo Temporal")
        try:
            min_y = int(df_long[Config.YEAR_COL].min())
            max_y = int(df_long[Config.YEAR_COL].max())
            year_range = st.slider("Años:", min_y, max_y, (max_y-10, max_y))
        except:
            year_range = (2000, 2020)

        # --- 4. FILTRO DE MESES (NUEVO AGREGADO) ---
        st.markdown("### 📆 Análisis Estacional")
        st.caption("Filtre por meses específicos (ej. solo Enero y Febrero).")
        
        meses_nombres = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
                         'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
        
        selected_months = st.multiselect(
            "Meses a incluir:", 
            options=meses_nombres, 
            default=meses_nombres,
            help="Seleccione los meses que desea incluir en el análisis."
        )
        
        # Mapear nombres a números (1-12)
        mapa_meses = {m: i+1 for i, m in enumerate(meses_nombres)}
        selected_months_nums = [mapa_meses[m] for m in selected_months]

        # FILTRADO MAESTRO
        # 1. Filtro de Años y Estaciones
        mask = (
            (df_long[Config.YEAR_COL] >= year_range[0]) & 
            (df_long[Config.YEAR_COL] <= year_range[1]) &
            (df_long[Config.STATION_NAME_COL].isin(stations_for_analysis))
        )
        df_temp = df_long.loc[mask].copy()

        # 2. Filtro de Meses (NUEVO)
        if selected_months_nums:
            mask_mes = df_temp[Config.MONTH_COL].isin(selected_months_nums)
            df_monthly_filtered = df_temp.loc[mask_mes].copy()
        else:
            # Si no hay meses seleccionados, devolver vacío o todo (decisión de diseño: devolvemos vacío para indicar que se necesita selección)
            # O mejor, devolvemos vacío para que los gráficos se limpien y no muestren error
            df_monthly_filtered = pd.DataFrame(columns=df_long.columns) 

        if exclude_nulls:
            df_monthly_filtered = df_monthly_filtered.dropna(subset=[Config.PRECIPITATION_COL])
        if exclude_zeros:
            df_monthly_filtered = df_monthly_filtered[df_monthly_filtered[Config.PRECIPITATION_COL] != 0]

        df_anual_melted = df_monthly_filtered.groupby(
            [Config.STATION_NAME_COL, Config.YEAR_COL]
        )[Config.PRECIPITATION_COL].sum().reset_index()

        # --- BOTÓN LIMPIAR CACHÉ (MANTENIDO) ---
        st.divider()
        if st.button("🧹 Limpiar Caché y Recargar"):
            st.cache_data.clear()
            st.rerun()

        # RETORNO ACTUALIZADO: Incluye selected_months_nums al final
        return (stations_for_analysis, df_anual_melted, df_monthly_filtered, gdf_final, 
                "Histórico", selected_regions, selected_municipios, selected_months_nums, year_range)
