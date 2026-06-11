"""
Bama.ir Samand Car Scraper
Scrapes 50 Samand cars manufactured after 1385 from bama.ir API.

Usage:
    pip install requests
    python bama_scraper.py
"""

import requests
import time
import csv

BASE_URL = "https://bama.ir/cad/api/search"
TARGET_COUNT = 50
PAGE_SIZE = 25

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "fa-IR,fa;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://bama.ir/car/samand?year=1385-2006,",
    "Origin": "https://bama.ir",
}

def fetch_page(session: requests.Session, page_index: int) -> dict:
    """Fetches a single page of results from the API."""
    params = {
        "vehicle": "samand",
        "yearFrom": "1385-2006",
        "pageIndex": page_index,
        "pageSize": PAGE_SIZE,
    }
    response = session.get(BASE_URL, params=params, headers=HEADERS, timeout=20)
    response.raise_for_status()
    return response.json()

def parse_ad(ad: dict) -> dict:
    """Extracts strictly the required fields from the raw ad dictionary."""
    detail = ad.get("detail", {})
    price_info = ad.get("price", {})

    price_type = price_info.get("type", "")
    price_raw = price_info.get("price", "0")

    if price_type == "lumpsum" and price_raw != "0":
        price = price_raw  
    elif price_type == "negotiable":
        price = "توافقی"
    else:
        price = price_raw if price_raw != "0" else "نامشخص"

    return {
        "Price": price,
        "Mileage": detail.get("mileage", "نامشخص"),
        "Color": detail.get("body_color", "نامشخص"),
        "Production Year": detail.get("year", "نامشخص"),
        "Transmission Type": detail.get("transmission", "نامشخص"),
        "Description": (detail.get("description") or "").replace("\r\n", " ").replace("\n", " ").strip(),
    }

def scrape(target: int = TARGET_COUNT) -> list[dict]:
    """Handles the scraping loop, pagination, and polite delays."""
    session = requests.Session()
    
    # Warm-up request to acquire standard cookies
    try:
        session.get(
            "https://bama.ir/car/samand?year=1385-2006,",
            headers={**HEADERS, "Accept": "text/html"},
            timeout=15,
        )
        time.sleep(1)
    except Exception as e:
        print(f"[warn] Warm-up request failed: {e}")

    cars = []
    page = 1

    while len(cars) < target:
        print(f"Fetching page {page} ... ({len(cars)}/{target} collected)")
        try:
            data = fetch_page(session, page)
        except Exception as e:
            print(f"[error] Failed to fetch page {page}: {e}")
            break

        ads = data.get("data", {}).get("ads", [])
        if not ads:
            print("No more ads found.")
            break

        for ad in ads:
            if ad.get("type") == "ad":
                cars.append(parse_ad(ad))
                if len(cars) >= target:
                    break

        if not data.get("metadata", {}).get("has_next", False):
            print("Reached last page.")
            break

        page += 1
        time.sleep(1.5)  # Polite delay to prevent rate-limiting

    print(f"\nTotal collected: {len(cars)} cars")
    return cars

def save_to_csv(cars: list[dict], filename: str = "samand_cars.csv"):
    """Saves the extracted dictionaries to a CSV file."""
    if not cars:
        print("No data to save.")
        return

    # Note: 'utf-8-sig' ensures Persian/Farsi characters render correctly when opened in Microsoft Excel
    with open(filename, mode="w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=cars[0].keys())
        writer.writeheader()
        writer.writerows(cars)
        
    print(f"Saved → {filename}")

if __name__ == "__main__":
    scraped_cars = scrape(target=TARGET_COUNT)
    if scraped_cars:
        save_to_csv(scraped_cars)