import pandas as pd
import requests
import json
import time
import argparse  # <--- Added
from tqdm import tqdm

def fetch_raw_tmdb_data(api_key, tmdb_id, media_type):
    """
    Fetches raw details from TMDB and normalizes fields, 
    but DOES NOT structure the final dict order.
    """
    endpoint_type = "movie" if media_type == "Movie" else "tv"
    url = f"https://api.themoviedb.org/3/{endpoint_type}/{tmdb_id}"
    params = {"api_key": api_key, "language": "en-US"}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 429:
            time.sleep(1)
            return fetch_raw_tmdb_data(api_key, tmdb_id, media_type)
            
        if response.status_code != 200:
            return None

        data = response.json()

        # Normalization Logic
        is_movie = (media_type == "Movie")
        
        # Title
        title = data.get("title") if is_movie else data.get("name")
        
        # Year (Try to cast to int)
        date_str = data.get("release_date") if is_movie else data.get("first_air_date")
        year = None
        if date_str and len(date_str) >= 4:
            try:
                year = int(date_str.split("-")[0])
            except ValueError:
                year = date_str # Fallback to string if weird format

        # Image
        poster_path = data.get("poster_path")
        image_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None

        # Genres
        genres = [g["name"] for g in data.get("genres", [])]

        return {
            "title": title,
            "description": data.get("overview"),
            "year": year,
            "rating": data.get("vote_average"),
            "votes": data.get("vote_count"),
            "genres": genres,
            "language": data.get("original_language"),
            "popularity": data.get("popularity"),
            "image_url": image_url,
            "type": media_type
        }

    except Exception:
        return None

def build_ordered_dataset(api_key, input_csv, output_ndjson):
    print("Loading CSV and assigning new IDs...")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Input file '{input_csv}' not found.")
        return pd.DataFrame()

    # 1. Setup new sequential IDs
    df = df.reset_index(drop=True)
    df['new_id'] = df.index + 1
    
    processed_records = []
    
    print(f"Processing {len(df)} records...")
    
    with open(output_ndjson, 'w', encoding='utf-8') as f_out:
        for row in tqdm(df.itertuples(), total=len(df)):
            
            # Fetch the data
            raw_data = fetch_raw_tmdb_data(api_key, row.id, row.type)
            
            if raw_data:
                # 2. CONSTRUCT DICTIONARY IN EXACT ORDER
                final_record = {
                    "id": row.new_id,
                    "title": raw_data["title"],
                    "description": raw_data["description"],
                    "year": raw_data["year"],
                    "rating": raw_data["rating"],
                    "votes": raw_data["votes"],
                    "genres": raw_data["genres"],
                    "language": raw_data["language"],
                    "popularity": raw_data["popularity"],
                    "image_url": raw_data["image_url"],
                    "type": raw_data["type"]
                }
                
                # Write to file
                f_out.write(json.dumps(final_record) + '\n')
                
                # Save to list for DataFrame
                processed_records.append(final_record)

    return pd.DataFrame(processed_records)

if __name__ == "__main__":
    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="Fetch TMDB data and create source_data.ndjson")
    
    # Add arguments
    parser.add_argument("--api_key", required=True, help="Your TMDB API Key")
    parser.add_argument("--input", default="tmdb_ids_combined.csv", help="Input CSV filename")
    parser.add_argument("--output", default="source_data.ndjson", help="Output NDJSON filename")

    args = parser.parse_args()

    # Execution
    df_final = build_ordered_dataset(args.api_key, args.input, args.output)

    print(f"\nDone. First row preview:")
    if not df_final.empty:
        print(json.dumps(df_final.iloc[0].to_dict(), indent=None))