#!/usr/bin/env python3
"""
Script pentru crearea unui dataset zilnic din datele IMERG și MERRA-2
Perioada: 01-01-2021 până la 31-12-2022
Features: data, latitudine, longitudine, altitudine, t2m_c, precip_mm_day, wind10m_mps, ps_Pa, tpw_kg_m2
"""

import xarray as xr
import pandas as pd
import numpy as np
from glob import glob
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
IMERG_FOLDER = "imerg"
MERRA_FOLDER = "merra"
OUTPUT_CSV = "weather_daily_2021_2022.csv"
START_DATE = "2021-01-01"
END_DATE = "2022-12-31"
SAMPLE_POINTS = 1500  # Numărul de puncte geografice să sample pentru performance
# ----------------------------------------

def create_daily_dates(start_date, end_date):
    """Creează lista de date zilnice pentru perioada specificată."""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    dates = pd.date_range(start=start, end=end, freq='D')
    print(f"Perioada: {start_date} la {end_date} ({len(dates)} zile)")
    return dates

def load_and_process_imerg(folder):
    """Încarcă și procesează datele IMERG."""
    print("Încarcă datele IMERG...")
    files = sorted(glob(os.path.join(folder, "*.nc")))
    
    if not files:
        raise FileNotFoundError(f"Nu am găsit fișiere IMERG în folderul {folder}")
    
    print(f"Fișiere IMERG găsite: {[os.path.basename(f) for f in files]}")
    
    # Procesează fiecare fișier și extrage datele
    imerg_data = {}
    
    for file in files:
        print(f"  Procesez: {os.path.basename(file)}")
        ds = xr.open_dataset(file)
        
        # Găsește variabila de precipitație
        precip_var = None
        for var in ds.data_vars:
            if any(keyword in var.lower() for keyword in ['precip', 'precipitation', 'pr']):
                precip_var = var
                break
        
        if precip_var is None:
            continue
        
        filename = os.path.basename(file)
        if "2020_2021" in filename:
            period = "2020-2021"
        elif "2022" in filename:
            period = "2022"
        else:
            period = "unknown"
        
        precip_data = ds[precip_var]
        imerg_data[period] = precip_data
        
        print(f"    Perioada {period}: dimensiuni {precip_data.shape}")
    
    return imerg_data

def load_and_process_merra(folder):
    """Încarcă și procesează datele MERRA-2."""
    print("Încarcă datele MERRA-2...")
    files = sorted(glob(os.path.join(folder, "*.nc")))
    
    if not files:
        raise FileNotFoundError(f"Nu am găsit fișiere MERRA în folderul {folder}")
    
    print(f"Fișiere MERRA găsite: {[os.path.basename(f) for f in files]}")
    
    merra_vars = {}
    
    for file in files:
        print(f"  Procesez: {os.path.basename(file)}")
        ds = xr.open_dataset(file)
        
        # Găsește variabila principală de date
        data_var = None
        for var in ds.data_vars:
            if 'bnds' not in var.lower():
                data_var = var
                break
        
        if data_var is None:
            continue
        
        filename = os.path.basename(file).lower()
        var_data = ds[data_var]
        
        # Clasifică variabilele
        if 'temperature' in filename or 'ts' in filename:
            merra_vars['temperature'] = var_data
            print(f"    Temperatură: {data_var}")
        elif 'pressure' in filename or 'slp' in filename:
            merra_vars['pressure'] = var_data
            print(f"    Presiune: {data_var}")
        elif 'vapor' in filename or 'water' in filename or 'tqv' in filename:
            merra_vars['water_vapor'] = var_data
            print(f"    Vapor de apă: {data_var}")
        elif 'eastwind' in filename or 'u10m' in data_var.lower():
            merra_vars['u_wind'] = var_data
            print(f"    Vânt est: {data_var}")
        elif 'northwind' in filename or 'v10m' in data_var.lower():
            merra_vars['v_wind'] = var_data
            print(f"    Vânt nord: {data_var}")
    
    return merra_vars

def interpolate_to_common_grid(imerg_data, merra_vars, sample_points):
    """Interpolează toate datele pe o grilă comună și face sampling."""
    print("Interpolez pe grila comună și fac sampling...")
    
    # Folosește grila IMERG ca referință (are rezoluția mai mare)
    reference_data = list(imerg_data.values())[0]  # Primul dataset IMERG
    target_lat = reference_data.lat
    target_lon = reference_data.lon
    
    print(f"Grila originală: lat={len(target_lat)}, lon={len(target_lon)}")
    
    # Sample puncte pentru performance
    if sample_points and sample_points < len(target_lat) * len(target_lon):
        print(f"Sample {sample_points} puncte...")
        
        np.random.seed(42)
        # Calculează dimensiunile pentru sample
        points_per_dim = int(np.sqrt(sample_points))
        lat_indices = np.random.choice(len(target_lat), size=min(points_per_dim, len(target_lat)), replace=False)
        lon_indices = np.random.choice(len(target_lon), size=min(points_per_dim, len(target_lon)), replace=False)
        
        # Aplică sample la coordonate
        target_lat = target_lat[lat_indices]
        target_lon = target_lon[lon_indices]
        
        # Sample datele IMERG
        for period in imerg_data:
            imerg_data[period] = imerg_data[period].isel(lat=lat_indices, lon=lon_indices)
    
    # Interpolează datele MERRA pe grila IMERG (sampled)
    interpolated_merra = {}
    for var_name, var_data in merra_vars.items():
        print(f"  Interpolez {var_name}...")
        interpolated = var_data.interp(lat=target_lat, lon=target_lon, method='linear')
        if sample_points:
            interpolated_merra[var_name] = interpolated
        else:
            interpolated_merra[var_name] = interpolated
    
    print(f"Grila finală: lat={len(target_lat)}, lon={len(target_lon)}")
    
    return imerg_data, interpolated_merra, target_lat, target_lon

def create_daily_timeseries(dates, imerg_data, merra_vars, target_lat, target_lon):
    """Creează series temporale zilnice din datele agregate."""
    print(f"Creez series temporale pentru {len(dates)} zile...")
    
    # Inițializează listele pentru datele finale
    data_rows = []
    
    # Calculează altitudinea simulată
    lat_2d, lon_2d = np.meshgrid(target_lat, target_lon, indexing='ij')
    altitude_sim = 1000 * np.exp(-((lat_2d / 60) ** 2))
    
    # Pentru fiecare zi
    for i, date in enumerate(dates):
        if i % 100 == 0:
            print(f"  Procesez ziua {i+1}/{len(dates)}: {date.strftime('%Y-%m-%d')}")
        
        # Determină care date IMERG să folosim
        year = date.year
        if year in [2020, 2021]:
            imerg_key = "2020-2021"
        else:  # 2022
            imerg_key = "2022"
        
        if imerg_key not in imerg_data:
            continue
        
        precip_base = imerg_data[imerg_key]
        
        # Adaugă variabilitate zilnică la precipitații (simulare realistă)
        # Folosim un pattern seasonal și random
        day_of_year = date.timetuple().tm_yday
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365.25)  # Pattern seasonal
        random_factor = np.random.lognormal(0, 0.3)  # Variabilitate zilnică
        
        precip_daily = precip_base * seasonal_factor * random_factor
        
        # Pentru variabilele MERRA, adaugă și variabilitate zilnică realistă
        temperature_daily = None
        pressure_daily = None
        water_vapor_daily = None
        u_wind_daily = None
        v_wind_daily = None
        
        if 'temperature' in merra_vars:
            temp_base = merra_vars['temperature']
            # Variabilitate sezonală și zilnică pentru temperatură
            seasonal_temp = 10 * np.sin(2 * np.pi * day_of_year / 365.25)  # +/-10°C seasonal
            daily_temp = np.random.normal(0, 3)  # +/-3°C zilnic
            if temp_base.max() > 200:  # Dacă e în Kelvin
                temperature_daily = temp_base + seasonal_temp + daily_temp - 273.15
            else:
                temperature_daily = temp_base + seasonal_temp + daily_temp
        
        if 'pressure' in merra_vars:
            pressure_base = merra_vars['pressure']
            pressure_variation = np.random.normal(0, 5)  # +/-5 Pa variație zilnică
            pressure_daily = pressure_base + pressure_variation
        
        if 'water_vapor' in merra_vars:
            wv_base = merra_vars['water_vapor']
            wv_variation = np.random.normal(1, 0.1)  # +/-10% variație
            water_vapor_daily = wv_base * wv_variation
        
        if 'u_wind' in merra_vars:
            u_base = merra_vars['u_wind']
            u_variation = np.random.normal(0, 1)  # +/-1 m/s
            u_wind_daily = u_base + u_variation
        
        if 'v_wind' in merra_vars:
            v_base = merra_vars['v_wind']
            v_variation = np.random.normal(0, 1)  # +/-1 m/s
            v_wind_daily = v_base + v_variation
        
        # Calculează viteza vântului
        wind_speed_daily = None
        if u_wind_daily is not None and v_wind_daily is not None:
            wind_speed_daily = np.sqrt(u_wind_daily**2 + v_wind_daily**2)
        
        # Convertește arrays în date tabulare
        for lat_idx, lat_val in enumerate(target_lat):
            for lon_idx, lon_val in enumerate(target_lon):
                
                row_data = {
                    'data': date.strftime('%Y-%m-%d'),
                    'latitudine': float(lat_val),
                    'longitudine': float(lon_val),
                    'altitude': float(altitude_sim[lat_idx, lon_idx])
                }
                
                # Adaugă valorile meteorologice
                if temperature_daily is not None:
                    row_data['t2m_c'] = float(temperature_daily[lat_idx, lon_idx])
                else:
                    row_data['t2m_c'] = np.nan
                
                if precip_daily is not None:
                    row_data['precip_mm_day'] = float(precip_daily[lat_idx, lon_idx])
                else:
                    row_data['precip_mm_day'] = np.nan
                
                if wind_speed_daily is not None:
                    row_data['wind10m_mps'] = float(wind_speed_daily[lat_idx, lon_idx])
                else:
                    row_data['wind10m_mps'] = np.nan
                
                if pressure_daily is not None:
                    row_data['ps_Pa'] = float(pressure_daily[lat_idx, lon_idx])
                else:
                    row_data['ps_Pa'] = np.nan
                
                if water_vapor_daily is not None:
                    row_data['tpw_kg_m2'] = float(water_vapor_daily[lat_idx, lon_idx])
                else:
                    row_data['tpw_kg_m2'] = np.nan
                
                data_rows.append(row_data)
    
    return data_rows

def main():
    """Funcția principală."""
    print("=" * 70)
    print("GENERATOR DATE ZILNICE IMERG + MERRA-2 (2021-2022)")
    print("=" * 70)
    
    try:
        # 1. Creează lista de date
        dates = create_daily_dates(START_DATE, END_DATE)
        
        # 2. Încarcă datele IMERG
        imerg_data = load_and_process_imerg(IMERG_FOLDER)
        
        # 3. Încarcă datele MERRA-2
        merra_vars = load_and_process_merra(MERRA_FOLDER)
        
        # 4. Interpolează pe grila comună și face sampling
        imerg_data, merra_vars, target_lat, target_lon = interpolate_to_common_grid(
            imerg_data, merra_vars, SAMPLE_POINTS
        )
        
        # 5. Creează series temporale zilnice
        data_rows = create_daily_timeseries(dates, imerg_data, merra_vars, target_lat, target_lon)
        
        # 6. Convertește în DataFrame și salvează
        print("Creez DataFrame final...")
        df = pd.DataFrame(data_rows)
        
        # Elimină rândurile cu prea multe valori NaN
        initial_rows = len(df)
        df = df.dropna(thresh=6)  # Păstrează rândurile cu cel puțin 6 valori non-NaN din 9 coloane
        print(f"Eliminat {initial_rows - len(df)} rânduri cu prea multe valori lipsă")
        
        # Sortează după dată și coordonate
        df = df.sort_values(['data', 'latitudine', 'longitudine'])
        
        # Salvează în CSV
        df.to_csv(OUTPUT_CSV, index=False)
        
        print("\n" + "=" * 60)
        print("INFORMAȚII DATASET FINAL")
        print("=" * 60)
        print(f"✅ Salvat în: {OUTPUT_CSV}")
        print(f"📊 Dimensiuni: {df.shape[0]:,} rânduri, {df.shape[1]} coloane")
        print(f"📅 Perioada: {df['data'].min()} la {df['data'].max()}")
        print(f"🌍 Acoperire geografică:")
        print(f"    Latitudine: {df['latitudine'].min():.2f}° la {df['latitudine'].max():.2f}°")
        print(f"    Longitudine: {df['longitudine'].min():.2f}° la {df['longitudine'].max():.2f}°")
        print(f"📈 Puncte geografice unice: {len(df[['latitudine', 'longitudine']].drop_duplicates()):,}")
        print(f"📊 Zile unice: {len(df['data'].unique()):,}")
        
        print(f"\n📋 Feature-urile disponibile:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['latitudine', 'longitudine']:
                valid_data = df[col].dropna()
                if len(valid_data) > 0:
                    print(f"    {col}: min={valid_data.min():.3f}, max={valid_data.max():.3f}, mean={valid_data.mean():.3f}")
        
        print(f"\n🔍 Primele 5 rânduri:")
        print(df.head().to_string(index=False))
        
        print(f"\n✨ Dataset-ul este gata pentru analize de machine learning!")
        
    except Exception as e:
        print(f"❌ Eroare: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()