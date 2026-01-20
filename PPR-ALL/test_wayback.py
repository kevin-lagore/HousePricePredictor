#!/usr/bin/env python3
"""Test Wayback Machine approach for finding archived Daft listings."""

import json
import re
import time
import requests
from urllib.parse import quote, urlencode

# Test addresses from our sold listings
TEST_ADDRESSES = [
    ("88 Carrigweir, Tuam, Galway", "H54FX48", "carrigweir-tuam"),
    ("64 Carysfort Park, Blackrock, Dublin", "A94PC96", "carysfort-park-blackrock"),
    ("21a Lea Road, Dublin 4", "D04W0F2", "lea-road-dublin"),
]

WAYBACK_CDX_API = "https://web.archive.org/cdx/search/cdx"
WAYBACK_URL = "https://web.archive.org/web"


def search_wayback_cdx(url_pattern: str, limit: int = 50) -> list[dict]:
    """Search Wayback CDX API for URLs matching pattern."""
    results = []

    params = {
        "url": url_pattern,
        "output": "json",
        "limit": limit,
        "fl": "timestamp,original,statuscode,mimetype",
        "filter": "statuscode:200",
        "filter": "mimetype:text/html",
    }

    try:
        print(f"  CDX query: {url_pattern}")
        resp = requests.get(WAYBACK_CDX_API, params=params, timeout=30)
        print(f"  Response: {resp.status_code}")

        if resp.status_code == 200 and resp.text.strip():
            data = resp.json()
            if len(data) > 1:  # First row is header
                headers = data[0]
                for row in data[1:]:
                    entry = dict(zip(headers, row))
                    if entry.get('statuscode') == '200':
                        results.append({
                            "timestamp": entry['timestamp'],
                            "url": entry['original'],
                            "wayback_url": f"{WAYBACK_URL}/{entry['timestamp']}/{entry['original']}"
                        })
    except Exception as e:
        print(f"  Error: {e}")

    return results


def test_wayback_snapshot(wayback_url: str) -> dict:
    """Fetch a Wayback snapshot and extract property details."""
    print(f"  Fetching snapshot...")

    try:
        resp = requests.get(wayback_url, timeout=30, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        if resp.status_code == 200:
            html = resp.text
            details = {"html_length": len(html)}

            # Look for property details in the HTML

            # Beds
            beds_match = re.search(r'(\d+)\s*(?:Bed|bedroom)', html, re.IGNORECASE)
            if beds_match:
                details['beds'] = int(beds_match.group(1))

            # Baths
            baths_match = re.search(r'(\d+)\s*(?:Bath|bathroom)', html, re.IGNORECASE)
            if baths_match:
                details['baths'] = int(baths_match.group(1))

            # Size
            size_match = re.search(r'([\d,]+)\s*(?:sq\.?\s*m|m²|sqm)', html, re.IGNORECASE)
            if size_match:
                details['size_sqm'] = float(size_match.group(1).replace(',', ''))

            # Price
            price_match = re.search(r'€\s*([\d,]+)', html)
            if price_match:
                details['asking_price'] = int(price_match.group(1).replace(',', ''))

            # BER
            ber_match = re.search(r'BER[:\s]*([A-G][1-3]?)\b', html, re.IGNORECASE)
            if ber_match:
                details['ber'] = ber_match.group(1).upper()

            # Check for __NEXT_DATA__
            next_data_match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
            if next_data_match:
                details['has_next_data'] = True
                try:
                    data = json.loads(next_data_match.group(1))
                    props = data.get('props', {}).get('pageProps', {})
                    listing = props.get('listing', {})
                    if listing:
                        details['listing_found'] = True
                        if listing.get('numBedrooms'):
                            details['beds'] = listing['numBedrooms']
                        if listing.get('numBathrooms'):
                            details['baths'] = listing['numBathrooms']
                        if listing.get('floorArea', {}).get('value'):
                            details['size_sqm'] = listing['floorArea']['value']
                        if listing.get('description'):
                            details['has_description'] = True
                        if listing.get('point'):
                            details['has_coords'] = True
                except json.JSONDecodeError:
                    pass

            return details
        else:
            return {"error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def main():
    print("=" * 70)
    print("WAYBACK MACHINE APPROACH TEST")
    print("=" * 70)

    # First, let's see what Daft.ie URLs are in Wayback
    print("\n1. Testing general Daft.ie coverage in Wayback...")

    # Search for any Daft.ie for-sale listings
    results = search_wayback_cdx("daft.ie/for-sale/*", limit=10)
    print(f"   Found {len(results)} general for-sale listings in Wayback")

    if results:
        print("   Sample URLs:")
        for r in results[:3]:
            print(f"     - {r['timestamp']}: {r['url'][:70]}")

    # Now test specific addresses
    for address, eircode, slug in TEST_ADDRESSES:
        print(f"\n{'='*70}")
        print(f"Testing: {address} ({eircode})")
        print("=" * 70)

        # Try different search patterns
        patterns = [
            f"daft.ie/for-sale/*{slug}*",
            f"daft.ie/*{slug}*",
            f"*daft.ie*{eircode.lower()}*",
        ]

        found = False
        for pattern in patterns:
            results = search_wayback_cdx(pattern, limit=5)
            if results:
                print(f"\n  Found {len(results)} matches with pattern: {pattern}")
                for r in results[:2]:
                    print(f"    - {r['timestamp']}: {r['url'][:60]}")

                    # Try to extract details
                    details = test_wayback_snapshot(r['wayback_url'])
                    if details and 'error' not in details:
                        print(f"    Extracted: {details}")
                        found = True
                break

            time.sleep(0.5)  # Rate limit

        if not found:
            print("  No archived listings found for this address")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
