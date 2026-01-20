#!/usr/bin/env python3
"""Test extracting full details from an active Daft.ie for-sale listing."""

import json
import time
from playwright.sync_api import sync_playwright


def main():
    print("=" * 70)
    print("ACTIVE LISTING DATA EXTRACTION TEST")
    print("=" * 70)
    print("Testing what data we can extract from an ACTIVE for-sale listing")
    print("=" * 70)

    with sync_playwright() as p:
        browser = p.firefox.launch(headless=False)
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        )
        page = context.new_page()
        page.set_default_timeout(30000)

        # Go to Daft for-sale listings
        print("\n1. Loading for-sale listings page...")
        page.goto("https://www.daft.ie/property-for-sale/dublin", wait_until="domcontentloaded")
        time.sleep(3)

        # Dismiss cookie
        try:
            for selector in ['button:has-text("Accept")', '#didomi-notice-agree-button']:
                btn = page.locator(selector).first
                if btn.is_visible(timeout=1000):
                    btn.click()
                    time.sleep(0.5)
                    break
        except:
            pass

        # Find first listing link
        print("\n2. Finding a listing...")
        links = page.locator('a[href*="/for-sale/"]').all()
        print(f"   Found {len(links)} for-sale links")

        listing_url = None
        for link in links:
            try:
                href = link.get_attribute("href")
                if href and "/for-sale/" in href and href.count('/') >= 3:
                    # Skip navigation links, get actual listings
                    if "ireland" not in href and "dublin" not in href.split('/')[-1]:
                        listing_url = href if href.startswith('http') else f"https://www.daft.ie{href}"
                        break
            except:
                continue

        if not listing_url:
            print("   No listing found!")
            browser.close()
            return

        print(f"   Selected: {listing_url}")

        # Navigate to listing
        print("\n3. Loading listing page...")
        page.goto(listing_url, wait_until="domcontentloaded")
        time.sleep(3)

        # Extract __NEXT_DATA__
        print("\n4. Extracting __NEXT_DATA__...")
        try:
            script = page.locator('script#__NEXT_DATA__').first
            if script.count() > 0:
                content = script.text_content()
                data = json.loads(content)

                props = data.get('props', {}).get('pageProps', {})
                listing = props.get('listing', {})

                if listing:
                    print("\n   [SUCCESS] Found listing data in __NEXT_DATA__!")
                    print("\n   Available fields:")

                    # Core fields
                    fields = {
                        'title': listing.get('title'),
                        'seoTitle': listing.get('seoTitle'),
                        'propertyType': listing.get('propertyType'),
                        'numBedrooms': listing.get('numBedrooms'),
                        'numBathrooms': listing.get('numBathrooms'),
                    }

                    # Price
                    price = listing.get('price', {})
                    if isinstance(price, dict):
                        fields['price'] = price.get('value') or price.get('displayValue')

                    # Floor area
                    floor_area = listing.get('floorArea', {})
                    if isinstance(floor_area, dict):
                        fields['size_sqm'] = floor_area.get('value')

                    # BER
                    ber = listing.get('ber', {})
                    if isinstance(ber, dict):
                        fields['ber_rating'] = ber.get('rating')
                        fields['ber_code'] = ber.get('code')
                        fields['ber_epi'] = ber.get('epi')

                    # Coordinates
                    point = listing.get('point', {})
                    if isinstance(point, dict):
                        fields['latitude'] = point.get('latitude')
                        fields['longitude'] = point.get('longitude')

                    # Description
                    desc = listing.get('description', '')
                    fields['description'] = desc[:100] + '...' if len(desc) > 100 else desc

                    # Features
                    fields['facilities'] = listing.get('facilities', [])[:5]

                    # Media
                    media = listing.get('media', [])
                    fields['num_images'] = len(media) if isinstance(media, list) else 0

                    # Seller/Agent
                    seller = listing.get('seller', {})
                    if isinstance(seller, dict):
                        fields['agent_name'] = seller.get('name')
                        fields['agent_phone'] = seller.get('phone')

                    # Address
                    address = listing.get('address')
                    if isinstance(address, str):
                        fields['address'] = address
                    elif isinstance(address, dict):
                        fields['address'] = address.get('displayAddress')

                    # Print all found fields
                    for key, value in fields.items():
                        if value is not None and value != '' and value != []:
                            print(f"     {key}: {value}")

                    print(f"\n   Total fields with data: {sum(1 for v in fields.values() if v)}")

                else:
                    print("   No 'listing' object in pageProps")
            else:
                print("   No __NEXT_DATA__ script found")
        except Exception as e:
            print(f"   Error: {e}")

        print("\n5. Keeping browser open for 15 seconds...")
        time.sleep(15)
        browser.close()

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
