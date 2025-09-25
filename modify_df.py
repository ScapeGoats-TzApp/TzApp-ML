import pandas as pd
import numpy as np
import datetime

def get_season(date_str, latitude):
    """
    Determină sezonul pe baza datei și hemisferei
    """
    try:
        date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        month = date.month
        
        # Emisfera nordică
        if latitude >= 0:
            if month in [12, 1, 2]:
                return 'iarna'
            elif month in [3, 4, 5]:
                return 'primavara'
            elif month in [6, 7, 8]:
                return 'vara'
            else:  # 9, 10, 11
                return 'toamna'
        # Emisfera sudică (sezoanele sunt inversate)
        else:
            if month in [6, 7, 8]:
                return 'iarna'
            elif month in [9, 10, 11]:
                return 'primavara'
            elif month in [12, 1, 2]:
                return 'vara'
            else:  # 3, 4, 5
                return 'toamna'
    except:
        return 'unknown'

def classify_weather_category_comprehensive(row):
    """
    Clasifică cu thresholduri ajustate pentru a acoperi toate datele din dataset.
    Folosește o abordare ierarhică: categorii stricte -> relaxate -> fallback geografic
    """
    # Extrage valorile necesare
    lat = abs(row['latitudine'])
    lat_signed = row['latitudine']
    alt = row['altitudine']
    temp = row['t2m_c']
    precip = row['precip_mm_day'] 
    wind_kmh = row['viteza_vant_kmh']
    pressure = row['presiune_hPa']
    humidity = row['umiditate_relativa']
    season = row['sezon']
    
    # === PASS 1: CATEGORII ORIGINALE (STRICTE) ===
    
    # 1. Storm / Tropical Cyclone ⛈
    if (5 <= lat <= 30 and 
        pressure < 995 and 
        wind_kmh >= 50 and 
        humidity > 70 and 
        season in ['vara', 'toamna']):
        return 'Storm / Tropical Cyclone ⛈'
    
    # 2. Mountain Snow 🏔❄
    if (alt >= 1500 and 
        temp <= 0 and 
        humidity >= 70):
        return 'Mountain Snow 🏔❄'
    
    # 3. Polar Winter Cold 🧊
    if (lat >= 60 and 
        temp <= 0 and 
        season == 'iarna'):
        return 'Polar Winter Cold 🧊'
    
    # 4. Desert Hot & Dry (Summer) 🏜
    if (15 <= lat <= 35 and 
        temp >= 30 and 
        humidity < 30 and 
        pressure >= 1005 and 
        season == 'vara'):
        return 'Desert Hot & Dry (Summer) 🏜'
    
    # 5. Monsoon / Equatorial Rain (Summer) 🌧
    if (lat < 20 and 
        humidity >= 80 and 
        pressure < 1010 and 
        wind_kmh < 30 and 
        season == 'vara'):
        return 'Monsoon / Equatorial Rain (Summer) 🌧'
    
    # 6. Equatorial Dry-Season Sunny ☀
    if (lat < 15 and 
        temp >= 28 and 
        humidity < 60 and 
        pressure >= 1010 and 
        wind_kmh < 20 and 
        season == 'iarna'):
        return 'Equatorial Dry-Season Sunny ☀'
    
    # 7. Mid-Latitude Rainy (Spring/Autumn) 🌦
    if (30 <= lat <= 60 and 
        humidity >= 75 and 
        pressure < 1010 and 
        wind_kmh < 30 and 
        season in ['primavara', 'toamna']):
        return 'Mid-Latitude Rainy (Spring/Autumn) 🌦'
    
    # 8. Mid-Latitude Snow (Winter) ❄
    if (30 <= lat <= 60 and 
        temp <= 0 and 
        humidity >= 80 and 
        season == 'iarna'):
        return 'Mid-Latitude Snow (Winter) ❄'
    
    # 9. Foggy (Cool Season) 🌫
    if (humidity >= 95 and 
        0 <= temp <= 15 and 
        wind_kmh < 10 and 
        season in ['toamna', 'iarna']):
        return 'Foggy (Cool Season) 🌫'
    
    # 10. Temperate Summer Clear 🌤
    if (30 <= lat <= 60 and 
        22 <= temp <= 35 and 
        humidity < 60 and 
        pressure >= 1012 and 
        wind_kmh < 25 and 
        season == 'vara'):
        return 'Temperate Summer Clear 🌤'
    
    # === PASS 2: CATEGORII RELAXATE (pentru acoperirea dataset-ului) ===
    
    # Foggy conditions (FOARTE relaxat pentru a fi găsit)
    if (humidity >= 65 and 
        temp >= -10 and temp <= 25 and 
        wind_kmh <= 20 and 
        season in ['toamna', 'iarna', 'primavara']):
        return 'Foggy (Cool Season) 🌫'
    
    # Mid-latitude rainy (FOARTE relaxat)
    if (20 <= lat <= 70 and 
        humidity >= 50 and 
        (pressure <= 1020 or precip > 0.1) and 
        wind_kmh <= 40 and 
        season in ['primavara', 'toamna']):
        return 'Mid-Latitude Rainy (Spring/Autumn) 🌦'
    
    # Mid-latitude snow (FOARTE relaxat)
    if (20 <= lat <= 70 and 
        temp <= 12 and 
        humidity >= 40 and 
        season in ['iarna', 'toamna'] and
        (alt >= 200 or lat >= 35)):  # fie altitudine, fie latitudine mai mare
        return 'Mid-Latitude Snow (Winter) ❄'
    
    # Polar/Arctic conditions (relaxat)
    if (lat >= 55 and 
        temp <= 5 and 
        season in ['iarna', 'toamna']):
        return 'Polar Winter Cold 🧊'
    
    # High altitude snow (relaxat)
    if (alt >= 800 and 
        temp <= 8 and 
        humidity >= 40):
        return 'Mountain Snow 🏔❄'
    
    # Desert conditions (relaxat)
    if (lat >= 10 and lat <= 40 and 
        temp >= 25 and 
        humidity <= 20 and 
        season in ['vara', 'primavara']):
        return 'Desert Hot & Dry (Summer) 🏜'
    
    # Tropical storms (relaxat)
    if (5 <= lat <= 35 and 
        (pressure < 1000 or wind_kmh >= 35) and 
        humidity > 50):
        return 'Storm / Tropical Cyclone ⛈'
    
    # Equatorial sunny (relaxat)
    if (lat <= 20 and 
        temp >= 22 and 
        humidity <= 45 and 
        pressure >= 1005 and 
        season in ['iarna', 'toamna']):
        return 'Equatorial Dry-Season Sunny ☀'
    
    # Equatorial rainy (relaxat)
    if (lat <= 25 and 
        humidity >= 60 and 
        pressure <= 1015 and 
        season in ['vara', 'primavara']):
        return 'Monsoon / Equatorial Rain (Summer) 🌧'
    
    # Temperate summer (relaxat)
    if (20 <= lat <= 65 and 
        temp >= 18 and temp <= 40 and 
        humidity <= 70 and 
        pressure >= 1005 and 
        season == 'vara'):
        return 'Temperate Summer Clear 🌤'
    
    # === PASS 3: CATEGORII DE FALLBACK GEOGRAFIC ===
    
    # Extreme cold (orice latitudine)
    if temp <= -20:
        return 'Arctic Extreme Cold ❄️'
    
    # Extreme heat (orice latitudine)
    if temp >= 40:
        return 'Desert Extreme Heat 🔥'
    
    # High wind conditions
    if wind_kmh >= 60:
        return 'Storm High Wind 💨'
    
    # Very low pressure (storm systems)
    if pressure < 970:
        return 'Storm Low Pressure 🌀'
    
    # Very high pressure (clear weather)
    if pressure > 1030:
        return 'High Pressure Clear 🌞'
    
    # Tropical zone general (0°-30°)
    if lat <= 30:
        if temp >= 20:
            return 'Tropical Warm 🌺'
        elif temp >= 10:
            return 'Tropical Mild 🌿'
        else:
            return 'Tropical Cool 🌊'
    
    # Mid-latitude general (30°-60°)
    if 30 < lat <= 60:
        if temp >= 15:
            return 'Temperate Warm 🍃'
        elif temp >= 0:
            return 'Temperate Cool 🌲'
        else:
            return 'Temperate Cold ❄️'
    
    # Polar zone general (>60°)
    if lat > 60:
        if temp >= -10:
            return 'Arctic Mild 🐧'
        else:
            return 'Arctic Severe 🧊'
    
    # Ultimate fallback (should never reach here)
    return 'General Weather ☁️'

def main():
    """
    Procesează CSV-ul existent și creează versiunea simplificată cu doar coloanele originale + category
    """
    print("=== CURĂȚARE CSV - PĂSTRARE DOAR COLOANE ORIGINALE + CATEGORY ===")
    
    # Fișierele
    input_file = 'weather.csv'  # fișierul existent cu toate coloanele
    output_file = 'weather_clean.csv'  # fișierul curat
    
    print(f"📂 Încarcă datele din '{input_file}'...")
    try:
        df = pd.read_csv(input_file)
        print(f"✅ {len(df):,} înregistrări încărcate")
    except FileNotFoundError:
        print(f"❌ Fișierul {input_file} nu a fost găsit!")
        return
    
    # Verifică coloanele disponibile
    print(f"\n📊 Coloane disponibile: {list(df.columns)}")
    
    # Verifică dacă categoria există
    if 'category' not in df.columns:
        print(f"❌ Coloana 'category' nu există! Rulați mai întâi clasificarea.")
        return
    
    # Selectează DOAR coloanele inițiale + category
    desired_columns = [
        'data', 'latitudine', 'longitudine', 'altitude', 'air_temp', 
        'precip_mm_day', 'wind_mps', 'pressure', 'relative_umidity',  # originale
        'category'  # doar categoria
    ]
    
    # Verifică că toate coloanele dorite există
    missing_cols = [col for col in desired_columns if col not in df.columns]
    if missing_cols:
        print(f"⚠️  Coloane lipsă: {missing_cols}")
        # Încearcă să mapeze coloanele lipsă
        print(f"Coloane disponibile: {list(df.columns)}")
        return
    
    # Creează dataframe-ul final doar cu coloanele dorite
    print(f"\n🧹 Curățarea datelor - păstrare doar coloane originale + category...")
    df_clean = df[desired_columns].copy()
    
    # Salvează fișierul curat
    print(f"\n💾 Salvarea rezultatelor în '{output_file}'...")
    df_clean.to_csv(output_file, index=False)
    
    print(f"✅ Dataset curat salvat cu succes!")
    print(f"📈 {len(df_clean):,} înregistrări")
    print(f"📊 Coloane în fișierul final: {len(df_clean.columns)}")
    
    # Verifică structura finală
    print(f"\n=== STRUCTURA FINALĂ CSV CURAT ===")
    for i, col in enumerate(df_clean.columns):
        print(f"{i+1:2d}. {col}")
    
    # Analiză categorii
    if 'category' in df_clean.columns:
        category_counts = df_clean['category'].value_counts()
        print(f"\n=== DISTRIBUȚIA CATEGORIILOR ===")
        print(f"Total categorii unice: {len(category_counts)}")
        
        # Afișează top 10 categorii
        print(f"\n=== TOP 10 CATEGORII ===")
        for i, (category, count) in enumerate(category_counts.head(10).items()):
            percentage = (count / len(df_clean)) * 100
            print(f"{i+1:2d}. {category}: {count:,} ({percentage:.2f}%)")
    
    # Preview primele 3 rânduri
    print(f"\n=== PREVIEW PRIMELE 3 RÂNDURI ===")
    print(df_clean.head(3).to_string(index=False))
    
    # Statistici finale
    print(f"\n=== STATISTICI FINALE ===")
    print(f"📁 Fișier de input: {input_file}")
    print(f"📁 Fișier de output: {output_file}")
    print(f"📊 Coloane originale: {len(df.columns)} -> {len(df_clean.columns)}")
    print(f"📈 Înregistrări: {len(df_clean):,}")
    print(f"🎯 Doar coloanele originale + category păstrate!")
    
    # Redenumire finală pentru a fi exact weather.csv
    print(f"\n🔄 Redenumire finală în 'weather.csv'...")
    df_clean.to_csv('weather.csv', index=False)
    print(f"✅ Fișierul 'weather.csv' a fost actualizat cu structura curățată!")

if __name__ == "__main__":
    main()
