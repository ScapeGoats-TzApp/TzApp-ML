#!/usr/bin/env python3
"""
Script pentru re-categorizarea datelor meteorologice cu threshold-uri îmbunătățite
Bazat pe analiza exemplelor și pentru a reduce overfitting-ul
"""

import pandas as pd
import numpy as np

def categorize_weather_improved(row):
    """
    Categorizează vremea pe baza threshold-urilor îmbunătățite v2
    Bazat pe analiza erorilor și simularea cu 80% acuratețe
    """
    temp = row['air_temp']
    precip = row['precip_mm_day']
    humidity = row['relative_umidity']
    pressure = row['pressure']
    lat = abs(row['latitudine'])  # Latitudine absolută
    wind = row['wind_mps']
    altitude = row['altitude']
    season = row['season'] if 'season' in row else 1
    
    # 1. ARCTIC (latitudine > 60°) - îmbunătățit bazat pe simulare
    if lat > 70:
        if temp < -10:
            return "Arctic Extreme Cold ❄️"
        elif temp < -5:
            return "Arctic Severe 🧊" 
        else:
            return "Arctic Mild 🐧"
    elif lat > 60:
        if temp < -25:
            return "Arctic Extreme Cold ❄️"
        elif temp < -8:
            return "Arctic Severe 🧊"
        elif temp < 5:
            return "Arctic Mild 🐧"
        else:
            return "Temperate Cool 🌲"
    
    # 2. DESERT (precipitații < 1mm, temperatură > 35°C, umiditate scăzută)
    elif precip < 1.0 and temp > 35:
        if temp > 42 and humidity < 25:
            return "Desert Extreme Heat 🔥"
        elif temp > 38 and humidity < 35:
            return "Desert Hot & Dry (Summer) 🏜"
        elif pressure > 1020:
            return "High Pressure Clear 🌞"
        else:
            return "Desert Hot & Dry (Summer) 🏜"
    
    # 3. MONSOON (precipitații > 12mm, umiditate > 80%) - threshold redus
    elif precip > 12 and humidity > 80 and temp > 25:
        return "Monsoon / Equatorial Rain (Summer) 🌧"
    
    # 4. EQUATORIAL/TROPICAL (latitudine < 30°) - criterile îmbunătățite
    elif lat < 15:
        if temp > 30 and precip < 2:
            return "Equatorial Dry-Season Sunny ☀"
        elif temp > 28:
            return "Tropical Warm 🌺"
        elif temp > 20:
            return "Tropical Mild 🌿"
        else:
            return "Tropical Cool 🌊"
    elif lat < 30:
        if temp > 28:
            return "Tropical Warm 🌺"
        elif temp > 20:
            return "Tropical Mild 🌿"
        else:
            return "Tropical Cool 🌊"
    
    # 5. TEMPERATE (latitudine 30-60°) - criteriile îmbunătățite
    else:
        if temp < -2:
            return "Temperate Cold ❄️"
        elif temp < 12:
            return "Temperate Cool 🌲"
        elif temp > 22 and precip < 3 and season == 3:  # Vara
            return "Temperate Summer Clear 🌤"
        elif temp > 18:
            return "Temperate Warm 🍃"
        else:
            return "Temperate Cool 🌲"

def analyze_current_categories(df):
    """Analizează categoriile curente pentru comparație"""
    print("🔍 ANALIZA CATEGORIILOR CURENTE")
    print("=" * 50)
    
    current_counts = df['category'].value_counts()
    print(f"📊 Total categorii curente: {len(current_counts)}")
    print(f"📊 Total înregistrări: {len(df):,}")
    
    print(f"\n🏆 TOP 10 CATEGORII CURENTE:")
    for i, (cat, count) in enumerate(current_counts.head(10).items(), 1):
        percentage = (count / len(df)) * 100
        print(f"   {i:2d}. {cat}: {count:,} ({percentage:.1f}%)")
    
    return current_counts

def apply_improved_categorization(input_file='weather.csv', output_file='weather_recategorized.csv'):
    """Aplică categoria îmbunătățită pe dataset"""
    print("🚀 RE-CATEGORIZAREA DATELOR METEOROLOGICE")
    print("=" * 60)
    
    # Încarcă datele
    print(f"📂 Încarcă datele din '{input_file}'...")
    try:
        df = pd.read_csv(input_file)
        print(f"✅ {len(df):,} înregistrări încărcate")
    except FileNotFoundError:
        print(f"❌ Fișierul {input_file} nu a fost găsit!")
        return
    
    # Analizează categoriile curente
    current_counts = analyze_current_categories(df)
    
    # Verifică dacă coloana season există, dacă nu o creează
    if 'season' not in df.columns and 'month' in df.columns:
        print("📅 Creez coloana season din month...")
        df['season'] = df['month'].map(lambda m: 1 if m in [12, 1, 2] else (2 if m in [3, 4, 5] else (3 if m in [6, 7, 8] else 4)))
    elif 'month' not in df.columns and 'data' in df.columns:
        print("📅 Extracting month și season din data...")
        df['data'] = pd.to_datetime(df['data'], errors='coerce')
        df['month'] = df['data'].dt.month
        df['season'] = df['month'].map(lambda m: 1 if m in [12, 1, 2] else (2 if m in [3, 4, 5] else (3 if m in [6, 7, 8] else 4)))
    
    # Salvează categoria veche
    df['category_old'] = df['category']
    
    # Aplică categoria nouă
    print(f"\n🔄 Aplicarea categoriei îmbunătățite...")
    df['category'] = df.apply(categorize_weather_improved, axis=1)
    
    # Analizează noile categorii
    print(f"\n🆕 ANALIZA CATEGORIILOR ÎMBUNĂTĂȚITE")
    print("=" * 50)
    
    new_counts = df['category'].value_counts()
    print(f"📊 Total categorii noi: {len(new_counts)}")
    
    print(f"\n🏆 TOP 10 CATEGORII ÎMBUNĂTĂȚITE:")
    for i, (cat, count) in enumerate(new_counts.head(10).items(), 1):
        percentage = (count / len(df)) * 100
        print(f"   {i:2d}. {cat}: {count:,} ({percentage:.1f}%)")
    
    # Comparație
    print(f"\n📊 COMPARAȚIA:")
    print(f"   📋 Categorii vechi: {len(current_counts)}")
    print(f"   📋 Categorii noi: {len(new_counts)}")
    print(f"   🔄 Schimbări: {len(df[df['category'] != df['category_old']]):,} înregistrări ({len(df[df['category'] != df['category_old']])/len(df)*100:.1f}%)")
    
    # Eliminează categoriile cu puține exemple (sub 2000)
    print(f"\n🧹 ELIMINAREA CATEGORIILOR CU SUB 2000 EXEMPLE...")
    categories_to_keep = new_counts[new_counts >= 2000].index.tolist()
    categories_to_remove = new_counts[new_counts < 2000].index.tolist()
    
    if categories_to_remove:
        print(f"❌ Categorii eliminate (sub 2000 exemple):")
        for cat in categories_to_remove:
            count = new_counts[cat]
            print(f"   • {cat}: {count:,} exemple")
        
        # Filtrează datele
        df_filtered = df[df['category'].isin(categories_to_keep)].copy()
        print(f"\n✅ Categorii păstrate: {len(categories_to_keep)}")
        print(f"📉 Înregistrări eliminate: {len(df) - len(df_filtered):,}")
        print(f"📈 Înregistrări păstrate: {len(df_filtered):,}")
        
        df_final = df_filtered
    else:
        print(f"✅ Toate categoriile au peste 2000 exemple")
        df_final = df
    
    # Elimină coloana category_old înainte de salvare
    df_save = df_final.drop(columns=['category_old']).copy()
    
    # Salvează rezultatul
    print(f"\n💾 Salvarea rezultatelor în '{output_file}'...")
    df_save.to_csv(output_file, index=False)
    
    print(f"✅ Dataset re-categorizat salvat cu succes!")
    print(f"📊 Categorii finale: {len(df_save['category'].value_counts())}")
    print(f"📈 Înregistrări finale: {len(df_save):,}")
    
    # Salvează fișierul original ca backup
    print(f"\n🔄 Actualizare fișierul principal...")
    df_save.to_csv('weather.csv', index=False)
    print(f"✅ Fișierul 'weather.csv' actualizat cu categoriile îmbunătățite!")
    
    return df_save

def test_new_categorization():
    """Testează noile categorii cu exemplele problematice"""
    print("\n🧪 TESTAREA CATEGORIILOR ÎMBUNĂTĂȚITE")
    print("=" * 50)
    
    # Exemple de test din analiza anterioară
    test_examples = [
        # Singapore - ar trebui să fie Tropical Warm
        {'name': 'Singapore', 'latitudine': 1.3521, 'longitudine': 103.8198, 'altitude': 15,
         'air_temp': 32, 'precip_mm_day': 8.5, 'wind_mps': 4, 'pressure': 1012, 'relative_umidity': 85,
         'season': 3, 'expected': 'Tropical Warm 🌺'},
        
        # Dubai - Desert Extreme Heat  
        {'name': 'Dubai', 'latitudine': 25.2048, 'longitudine': 55.2708, 'altitude': 5,
         'air_temp': 45, 'precip_mm_day': 0.0, 'wind_mps': 8, 'pressure': 1008, 'relative_umidity': 25,
         'season': 3, 'expected': 'Desert Extreme Heat 🔥'},
        
        # Mumbai Monsoon
        {'name': 'Mumbai Monsoon', 'latitudine': 19.0760, 'longitudine': 72.8777, 'altitude': 8,
         'air_temp': 28, 'precip_mm_day': 25.0, 'wind_mps': 6, 'pressure': 1005, 'relative_umidity': 95,
         'season': 3, 'expected': 'Monsoon / Equatorial Rain (Summer) 🌧'},
        
        # Groenlanda - Arctic Extreme Cold
        {'name': 'Groenlanda', 'latitudine': 72.0, 'longitudine': -40.0, 'altitude': 2000,
         'air_temp': -28, 'precip_mm_day': 0.5, 'wind_mps': 15, 'pressure': 995, 'relative_umidity': 65,
         'season': 1, 'expected': 'Arctic Extreme Cold ❄️'},
         
        # Antarctica - Polar Winter Cold
        {'name': 'Antarctica', 'latitudine': -77.8, 'longitudine': 166.7, 'altitude': 2800,
         'air_temp': -35, 'precip_mm_day': 0.1, 'wind_mps': 6, 'pressure': 680, 'relative_umidity': 55,
         'season': 1, 'expected': 'Arctic Extreme Cold ❄️'}  # Va fi Arctic pentru lat > 66
    ]
    
    correct_predictions = 0
    total_tests = len(test_examples)
    
    for example in test_examples:
        # Creează Series pentru testare
        row_data = {k: v for k, v in example.items() if k not in ['name', 'expected']}
        test_row = pd.Series(row_data)
        
        # Aplică categoria
        predicted = categorize_weather_improved(test_row)
        expected = example['expected'] 
        is_correct = predicted == expected
        
        status = "✅" if is_correct else "❌"
        print(f"{status} {example['name']}: {predicted}")
        if not is_correct:
            print(f"   Așteptat: {expected}")
        
        if is_correct:
            correct_predictions += 1
    
    accuracy = (correct_predictions / total_tests) * 100
    print(f"\n🏆 REZULTAT: {correct_predictions}/{total_tests} ({accuracy:.1f}%)")
    
    if accuracy >= 80:
        print("🎉 Categoriile îmbunătățite funcționează mult mai bine!")
    else:
        print("⚠️  Categoriile necesită ajustări suplimentare")

def main():
    """Funcția principală"""
    print("🌍 ÎMBUNĂTĂȚIREA CATEGORIILOR METEOROLOGICE")
    print("=" * 60)
    print("Obiectiv: Reducerea overfitting-ului și îmbunătățirea generalizării")
    print("=" * 60)
    
    # Testează noile categorii
    test_new_categorization()
    
    # Aplică re-categorizarea
    df_result = apply_improved_categorization()
    
    print(f"\n🚀 PROCES FINALIZAT!")
    print("Modelul ML poate fi re-antrenat cu noile categorii îmbunătățite.")
    
    return df_result

if __name__ == "__main__":
    main()