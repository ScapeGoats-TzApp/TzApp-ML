import pandas as pd
import numpy as np
import datetime
import argparse
import sys

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

def filter_problematic_categories(df, min_examples=2000):
    """
    Elimină categoriile cu puține exemple care pot încurca modelul ML
    """
    print(f"=== ELIMINARE CATEGORII PROBLEMATICE (SUB {min_examples} EXEMPLE) ===")
    
    category_counts = df['category'].value_counts().sort_values(ascending=False)
    print(f"📊 Distribuția inițială: {len(category_counts)} categorii, {len(df):,} înregistrări")
    
    # Găsește categoriile problematice
    categories_to_remove = category_counts[category_counts < min_examples].index.tolist()
    categories_to_keep = category_counts[category_counts >= min_examples].index.tolist()
    
    if categories_to_remove:
        print(f"\n❌ Categorii de eliminat (sub {min_examples} exemple):")
        for cat in categories_to_remove:
            count = category_counts[cat]
            percentage = (count / len(df)) * 100
            print(f"   • {cat}: {count} exemple ({percentage:.3f}%)")
        
        # Elimină categoriile problematice
        df_filtered = df[df['category'].isin(categories_to_keep)].copy()
        
        print(f"\n✅ Categorii păstrate (peste {min_examples} exemple): {len(categories_to_keep)}")
        print(f"📉 Înregistrări eliminate: {len(df) - len(df_filtered):,}")
        print(f"📈 Înregistrări păstrate: {len(df_filtered):,}")
        
        return df_filtered
    else:
        print(f"\n✅ Toate categoriile au peste {min_examples} exemple - nu se elimină nimic")
        return df

def main(min_examples=2000):
    """
    Procesează CSV-ul existent și creează versiunea simplificată cu doar coloanele originale + category
    + elimină categoriile cu puține exemple care încurcă modelul
    """
    print("=== CURĂȚARE CSV - PĂSTRARE DOAR COLOANE ORIGINALE + CATEGORY + ELIMINARE CATEGORII PROBLEMATICE ===")
    
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
    
    # ELIMINĂ CATEGORIILE PROBLEMATICE
    print(f"\n🧹 Eliminarea categoriilor cu puține exemple...")
    df_clean = filter_problematic_categories(df_clean, min_examples=min_examples)
    
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
    
    # Analiză categorii finale
    if 'category' in df_clean.columns:
        category_counts_final = df_clean['category'].value_counts().sort_values(ascending=False)
        print(f"\n=== DISTRIBUȚIA CATEGORIILOR FINALE ===")
        print(f"Total categorii unice: {len(category_counts_final)}")
        print(f"Toate categoriile au minim 50+ exemple pentru antrenarea robustă a modelului")
        
        # Afișează toate categoriile finale
        print(f"\n=== TOATE CATEGORIILE FINALE ===")
        for i, (category, count) in enumerate(category_counts_final.items()):
            percentage = (count / len(df_clean)) * 100
            status = "🟢" if count >= 1000 else "🟡" if count >= 100 else "🔵"
            print(f"{i+1:2d}. {status} {category}: {count:,} ({percentage:.2f}%)")
        
        # Verifică că toate categoriile au minim threshold exemple
        min_count = category_counts_final.min()
        print(f"\n📊 Categoria cu cele mai puține exemple: {min_count:,} (minimum recomandat: {min_examples})")
        if min_count >= min_examples:
            print("✅ Toate categoriile sunt potrivite pentru antrenarea ML robustă!")
        else:
            print(f"⚠️  Există încă categorii cu sub {min_examples} exemple!")
    
    # Preview primele 3 rânduri
    print(f"\n=== PREVIEW PRIMELE 3 RÂNDURI ===")
    print(df_clean.head(3).to_string(index=False))
    
    # Statistici finale
    print(f"\n=== STATISTICI FINALE ===")
    print(f"📁 Fișier de input: {input_file}")
    print(f"📁 Fișier de output: {output_file}")
    print(f"📊 Categorii finale: {len(category_counts_final)} (toate cu {min_examples}+ exemple)")
    print(f"📈 Înregistrări finale: {len(df_clean):,}")
    print(f"🎯 Dataset optimizat pentru antrenarea robustă a modelului ML!")
    
    # Redenumire finală pentru a fi exact weather.csv
    print(f"\n🔄 Redenumire finală în 'weather.csv'...")
    df_clean.to_csv('weather.csv', index=False)
    print(f"✅ Fișierul 'weather.csv' a fost actualizat cu structura curățată și categoriile optimizate!")
    
    print(f"\n🚀 GATA! Acum poți antrena modelul cu date curate și echilibrate!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Curățare dataset weather - elimină categoriile cu puține exemple')
    parser.add_argument('--min-examples', type=int, default=2000, 
                       help='Numărul minim de exemple pentru o categorie (default: 2000)')
    
    args = parser.parse_args()
    
    print(f"🚀 Rulează scriptul cu threshold = {args.min_examples} exemple minime per categorie")
    main(min_examples=args.min_examples)