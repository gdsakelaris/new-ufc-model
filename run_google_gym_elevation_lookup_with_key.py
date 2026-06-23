#!/usr/bin/env python3
"""
Fill training gym location/elevation columns using Google Maps Platform.

Setup:
  pip install requests pandas openpyxl
  No environment variable required in this version; a placeholder key is embedded.

APIs to enable in Google Cloud:
  - Places API (New)
  - Geocoding API (fallback)
  - Elevation API

Usage:
  # from the folder containing unique_ufc_gyms_elevation_google_ready.csv:
  python run_google_gym_elevation_lookup_with_key.py

  # optional custom paths:
  python run_google_gym_elevation_lookup_with_key.py --input input.csv --output output.csv
"""

import argparse
import os
import time
from urllib.parse import quote_plus

import pandas as pd
import requests

PLACES_URL = "https://places.googleapis.com/v1/places:searchText"
GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
ELEVATION_URL = "https://maps.googleapis.com/maps/api/elevation/json"

# Placeholder API key embedded per user request.
# For a real key, you can still override this by setting GOOGLE_MAPS_API_KEY.
DEFAULT_GOOGLE_MAPS_API_KEY = "AIzaSyBD93tpmEsQfwC85zJRYiBEG2ODArJIhdo"


def places_text_search(session, api_key, query):
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.location",
    }
    body = {"textQuery": query, "languageCode": "en"}
    resp = session.post(PLACES_URL, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    places = data.get("places", [])
    if not places:
        return None, "ZERO_RESULTS"
    place = places[0]
    loc = place.get("location", {})
    return {
        "address": place.get("formattedAddress", ""),
        "lat": loc.get("latitude"),
        "lng": loc.get("longitude"),
        "place_id": place.get("id", ""),
        "source": "places_text_search",
    }, "OK"


def geocode_fallback(session, api_key, query):
    params = {"address": query, "key": api_key}
    resp = session.get(GEOCODE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    status = data.get("status", "UNKNOWN")
    if status != "OK" or not data.get("results"):
        return None, status
    result = data["results"][0]
    loc = result["geometry"]["location"]
    return {
        "address": result.get("formatted_address", ""),
        "lat": loc.get("lat"),
        "lng": loc.get("lng"),
        "place_id": result.get("place_id", ""),
        "source": "geocoding_fallback",
    }, "OK"


def get_elevation(session, api_key, lat, lng):
    params = {"locations": f"{lat},{lng}", "key": api_key}
    resp = session.get(ELEVATION_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    status = data.get("status", "UNKNOWN")
    if status != "OK" or not data.get("results"):
        return None, status
    elevation_m = data["results"][0].get("elevation")
    return elevation_m, "OK"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="unique_ufc_gyms_elevation_google_ready.csv")
    parser.add_argument("--output", default="unique_ufc_gyms_with_elevation.csv")
    parser.add_argument("--sleep", type=float, default=0.15, help="Seconds to sleep between Google calls")
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows for testing")
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_MAPS_API_KEY") or DEFAULT_GOOGLE_MAPS_API_KEY
    if not api_key:
        raise SystemExit("No Google Maps API key found.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = args.input
    output_path = args.output

    if not os.path.isabs(input_path) and not os.path.exists(input_path):
        candidate = os.path.join(script_dir, input_path)
        if os.path.exists(candidate):
            input_path = candidate

    if not os.path.isabs(output_path):
        output_path = os.path.join(os.getcwd(), output_path)

    if input_path.lower().endswith(".xlsx"):
        df = pd.read_excel(input_path, sheet_name="Unique Gyms")
    else:
        df = pd.read_csv(input_path)

    for col in ["resolved_address", "latitude", "longitude", "elevation_m", "elevation_ft", "google_place_id", "lookup_status", "notes"]:
        if col not in df.columns:
            df[col] = ""

    session = requests.Session()
    n = len(df) if args.limit is None else min(args.limit, len(df))

    for idx in range(n):
        gym = str(df.at[idx, "gym"]).strip()
        if not gym or pd.notna(df.at[idx, "elevation_ft"]) and str(df.at[idx, "elevation_ft"]).strip():
            continue

        query = gym
        try:
            match, status = places_text_search(session, api_key, query)
            time.sleep(args.sleep)
            if match is None:
                match, status = geocode_fallback(session, api_key, query)
                time.sleep(args.sleep)

            if match is None:
                df.at[idx, "lookup_status"] = f"location_{status}"
                df.at[idx, "notes"] = "No Google location match found; review manually."
                continue

            elevation_m, elev_status = get_elevation(session, api_key, match["lat"], match["lng"])
            time.sleep(args.sleep)

            df.at[idx, "resolved_address"] = match["address"]
            df.at[idx, "latitude"] = match["lat"]
            df.at[idx, "longitude"] = match["lng"]
            df.at[idx, "google_place_id"] = match["place_id"]
            df.at[idx, "notes"] = f"location_source={match['source']}; google_search=https://www.google.com/search?q={quote_plus(gym + ' MMA gym')}"

            if elev_status == "OK" and elevation_m is not None:
                df.at[idx, "elevation_m"] = round(float(elevation_m), 2)
                df.at[idx, "elevation_ft"] = round(float(elevation_m) * 3.280839895, 1)
                df.at[idx, "lookup_status"] = "OK"
            else:
                df.at[idx, "lookup_status"] = f"elevation_{elev_status}"

        except Exception as exc:
            df.at[idx, "lookup_status"] = "ERROR"
            df.at[idx, "notes"] = str(exc)[:250]

        if idx % 25 == 0:
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"Processed {idx + 1}/{n}: {gym}")

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Done. Saved: {output_path}")


if __name__ == "__main__":
    main()
