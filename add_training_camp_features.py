import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import json
import time
import os
from tqdm import tqdm

# ========================= CONFIG =========================
INPUT_CSV = "pure_fight_data_with_travel_features_populated_final.csv"  # Update if needed
OUTPUT_CSV = "pure_fight_data_with_training_camp_features.csv"
CACHE_FILE = "fighter_training_camps.json"

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# City -> Altitude (feet)
CITY_ALTITUDE = {
    "Las Vegas, NV": 2000,
    "Denver, Colorado": 5280,
    "Mexico City": 7350,
    "Sparta, NJ": 800,
    "Albuquerque": 5300,
    "Rio de Janeiro": 20,
    "Chicago": 600,
    "New York": 30,
    "Los Angeles": 300,
    "San Diego": 60,
    "Auckland": 30,
    "Dublin": 150,
    "London": 80,
    "Coconut Creek, FL": 10,
    "Albuquerque, NM": 5300,
    "Pittsburgh": 1200,
    "Sacramento, CA": 25,
    # Add more as needed
}

# Gym name -> known location (key improvement)
GYM_TO_LOCATION = {
    "Xtreme Couture": "Las Vegas, NV",
    "American Top Team": "Coconut Creek, FL",
    "Jackson Wink": "Albuquerque, NM",
    "Team Alpha Male": "Sacramento, CA",
    "SBG Ireland": "Dublin",
    "Gracie Humaitá": "Rio de Janeiro",
    "Nova União": "Rio de Janeiro",
    "City Kickboxing": "Auckland",
    "Miller Brothers MMA": "Sparta, NJ",
    "BMF Ranch": "Las Vegas, NV",
    "4oz Fight Club": "Atlanta area",
    "Japan Top Team": "Tokyo area",
    # Add more gyms here as you discover them
}

def slugify(name):
    if not name:
        return ""
    name = re.sub(r'[^a-zA-Z0-9\s-]', '', str(name)).strip().lower()
    return re.sub(r'[\s]+', '-', name)

def parse_training_location(trains_at_text):
    if not trains_at_text:
        return None, None
    text = trains_at_text.strip()
    
    # 1. Gym mapping first
    for gym, location in GYM_TO_LOCATION.items():
        if gym.lower() in text.lower():
            alt = CITY_ALTITUDE.get(location, None)
            return text, alt
    
    # 2. Extract city from end
    match = re.search(r'-\s*([A-Za-z\s,]+?)(?:,|\s*$)', text)
    if match:
        city = match.group(1).strip()
        alt = CITY_ALTITUDE.get(city, None)
        if alt is None:
            for key, val in CITY_ALTITUDE.items():
                if key.lower() in city.lower():
                    alt = val
                    break
        return text, alt
    
    # 3. Keyword fallback on whole string
    for key, val in CITY_ALTITUDE.items():
        if key.lower() in text.lower():
            return text, val
    
    return text, None

def get_training_info(fighter_name, cache):
    if fighter_name in cache:
        return cache[fighter_name]
    
    slug = slugify(fighter_name)
    if not slug:
        result = {"trains_at": None, "altitude_ft": None}
        cache[fighter_name] = result
        return result
    
    url = f"https://www.ufc.com/athlete/{slug}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            result = {"trains_at": None, "altitude_ft": None}
            cache[fighter_name] = result
            return result
        
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator="\n")
        
        match = re.search(r'Trains at\s*\n?\s*(.+?)(?:\n|$|Place of Birth|Fighting style)', text, re.IGNORECASE | re.DOTALL)
        trains_at = match.group(1).strip() if match else None
        
        if not trains_at:
            for elem in soup.find_all(['div', 'p', 'span', 'li']):
                txt = elem.get_text(strip=True)
                if 'trains at' in txt.lower():
                    trains_at = re.sub(r'^.*?Trains at\s*', '', txt, flags=re.IGNORECASE).strip()
                    break
        
        trains_at_str, altitude = parse_training_location(trains_at)
        
        result = {"trains_at": trains_at_str, "altitude_ft": altitude}
        cache[fighter_name] = result
        time.sleep(1.0)
        return result
    except Exception:
        result = {"trains_at": None, "altitude_ft": None}
        cache[fighter_name] = result
        return result

# ========================= MAIN =========================
print("Loading fight data...")
df = pd.read_csv(INPUT_CSV)

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
else:
    cache = {}

fighters = pd.concat([df["r_fighter"], df["b_fighter"]]).dropna().unique()
print(f"Processing {len(fighters)} unique fighters...")

for fighter in tqdm(fighters):
    get_training_info(fighter, cache)

with open(CACHE_FILE, "w", encoding="utf-8") as f:
    json.dump(cache, f, indent=2)

print("Merging into dataframe...")

def add_training_cols(row):
    for side in ["r_", "b_"]:
        name = row.get(f"{side}fighter")
        info = cache.get(name, {"trains_at": None, "altitude_ft": None})
        row[f"{side}trains_at"] = info.get("trains_at")
        row[f"{side}training_altitude_ft"] = info.get("altitude_ft")
        
        event_alt = row.get("event_altitude")
        train_alt = info.get("altitude_ft")
        if pd.notna(event_alt) and train_alt is not None:
            row[f"{side}training_event_alt_diff"] = round(event_alt - train_alt, 1)
        else:
            row[f"{side}training_event_alt_diff"] = None
    return row

df = df.apply(add_training_cols, axis=1)

df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Done! Saved: {OUTPUT_CSV}")
print(f"r_trains_at populated: {df['r_trains_at'].notna().sum()}")
print(f"b_trains_at populated: {df['b_trains_at'].notna().sum()}")