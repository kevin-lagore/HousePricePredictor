#!/usr/bin/env python3
"""Test Daft.ie direct search to find listings with full details."""

import json
import re
import time
from urllib.parse import quote_plus
from playwright.sync_api import sync_playwright

# Test addresses - recently sold properties
TEST_ADDRESSES = [
    "64 Carysfort Park, Blackrock, Dublin",
    "21a Lea Road, Dublin 4",
    "88 Carrigweir, Tuam, Galway",
]


def extract_listing_details(page) -> dict:
    """Extract property details from a Daft listing page."""
    details = {}

    # Try __NEXT_DATA__ first
    try:
        script = page.locator('script#__NEXT_DATA__').first
        if script.count() > 0:
            content = script.text_content()
            data = json.loads(content)
            props = data.get('props', {}).get('pageProps', {})
            listing = props.get('listing', {})

            if listing:
                details['source'] = '__NEXT_DATA__'
                details['beds'] = listing.get('numBedrooms')
                details['baths'] = listing.get('numBathrooms')

                floor_area = listing.get('floorArea', {})
                if isinstance(floor_area, dict):
                    details['size_sqm'] = floor_area.get('value')

                ber = listing.get('ber', {})
                if isinstance(ber, dict):
                    details['ber'] = ber.get('rating')

                point = listing.get('point', {})
                if isinstance(point, dict):
                    details['lat'] = point.get('latitude')
                    details['lon'] = point.get('longitude')

                details['description'] = listing.get('description', '')[:100] + '...' if listing.get('description') else None
                details['property_type'] = listing.get('propertyType')

                price = listing.get('price', {})
                if isinstance(price, dict):
                    details['price'] = price.get('value') or price.get('displayValue')

                return details
    except Exception as e:
        print(f"    __NEXT_DATA__ error: {e}")

    # Fallback to DOM extraction
    details['source'] = 'DOM'

    # Try to get beds/baths from property overview
    try:
        overview = page.locator('[data-testid="property-overview"], .PropertyOverview').first
        if overview.count() > 0:
            text = overview.text_content().lower()
            beds_match = re.search(r'(\d+)\s*bed', text)
            baths_match = re.search(r'(\d+)\s*bath', text)
            if beds_match:
                details['beds'] = int(beds_match.group(1))
            if baths_match:
                details['baths'] = int(baths_match.group(1))
    except:
        pass

    return details


def search_daft_for_address(page, address: str) -> dict:
    """Search Daft.ie for an address and get listing details."""
    result = {
        "address": address,
        "found": False,
        "url": None,
        "details": {},
    }

    # Use Daft's search
    search_url = f"https://www.daft.ie/property-for-sale/ireland?searchSource=search&terms={quote_plus(address)}"
    print(f"  Searching: {search_url[:80]}...")

    try:
        page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
        time.sleep(3)  # Wait for results

        # Dismiss cookie popup
        try:
            for selector in ['button:has-text("Accept")', '#didomi-notice-agree-button']:
                btn = page.locator(selector).first
                if btn.is_visible(timeout=1000):
                    btn.click()
                    time.sleep(0.5)
                    break
        except:
            pass

        # Look for listing links
        links = page.locator('a[href*="/for-sale/"]').all()
        print(f"  Found {len(links)} for-sale links")

        for link in links[:5]:
            try:
                href = link.get_attribute("href")
                # Check if it's a listing (has ID at end)
                if href and re.search(r'/\d+/?$', href):
                    # Found a listing, navigate to it
                    listing_url = href if href.startswith('http') else f"https://www.daft.ie{href}"
                    print(f"  Checking listing: {listing_url[:60]}...")

                    page.goto(listing_url, wait_until="domcontentloaded", timeout=30000)
                    time.sleep(2)

                    # Check if address matches
                    title = page.title().lower()
                    if any(word.lower() in title for word in address.split()[:2]):
                        result['found'] = True
                        result['url'] = listing_url
                        result['details'] = extract_listing_details(page)
                        print(f"  [OK] Found matching listing!")
                        return result
            except Exception as e:
                print(f"    Link error: {e}")
                continue

    except Exception as e:
        result['error'] = str(e)
        print(f"  Search error: {e}")

    return result


def main():
    print("=" * 70)
    print("DAFT.IE DIRECT SEARCH TEST")
    print("=" * 70)
    print("Testing if we can find active listings with full property details")
    print("=" * 70)

    with sync_playwright() as p:
        browser = p.firefox.launch(headless=False)
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        )
        page = context.new_page()
        page.set_default_timeout(30000)

        for address in TEST_ADDRESSES:
            print(f"\n{'='*70}")
            print(f"Testing: {address}")
            print("=" * 70)

            result = search_daft_for_address(page, address)

            if result['found']:
                print(f"\n  URL: {result['url']}")
                print(f"  Details: {json.dumps(result['details'], indent=4)}")
            else:
                print(f"\n  [X] No active listing found")
                if result.get('error'):
                    print(f"  Error: {result['error']}")

            time.sleep(2)

        browser.close()

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
