#!/usr/bin/env python3
"""Check a specific sold listing on MyHome.ie to see what data is available."""

import json
import re
import time
from playwright.sync_api import sync_playwright


def main():
    url = "https://www.myhome.ie/residential/brochure/12-braemor-drive-churchtown-dublin-14/4706275"

    print("=" * 70)
    print("CHECKING SOLD LISTING ON MYHOME.IE")
    print("=" * 70)
    print(f"URL: {url}")
    print("=" * 70)

    with sync_playwright() as p:
        browser = p.firefox.launch(headless=False)
        page = browser.new_page()
        page.set_default_timeout(30000)

        print("\n1. Loading page...")
        page.goto(url, wait_until="domcontentloaded")
        time.sleep(3)

        # Dismiss cookies
        try:
            btn = page.locator('#onetrust-accept-btn-handler').first
            if btn.is_visible(timeout=2000):
                btn.click()
                time.sleep(1)
        except:
            pass

        print(f"\n2. Page title: {page.title()}")
        print(f"   Current URL: {page.url}")

        # Get body text
        body_text = page.locator('body').text_content() or ""

        # Check for "Sale Agreed" or "Sold" indicators
        print("\n3. Sale status indicators:")
        sale_indicators = ['sale agreed', 'sold', 'under offer', 'price on application']
        for indicator in sale_indicators:
            if indicator in body_text.lower():
                print(f"   Found: '{indicator}'")

        # Extract data
        print("\n4. Extracted data:")

        # Beds
        beds_match = re.search(r'\b([1-9]\d?)\s*(?:Bed|bedroom)s?\b', body_text, re.IGNORECASE)
        print(f"   Beds: {beds_match.group(1) if beds_match else 'Not found'}")

        # Baths
        baths_match = re.search(r'\b([1-9]\d?)\s*(?:Bath|bathroom)s?\b', body_text, re.IGNORECASE)
        print(f"   Baths: {baths_match.group(1) if baths_match else 'Not found'}")

        # Size
        sqm_match = re.search(r'([\d,.]+)\s*(?:sq\.?\s*m|m2|sqm)', body_text, re.IGNORECASE)
        print(f"   Size (sqm): {sqm_match.group(1) if sqm_match else 'Not found'}")

        sqft_match = re.search(r'([\d,.]+)\s*(?:sq\.?\s*ft|sqft)', body_text, re.IGNORECASE)
        print(f"   Size (sqft): {sqft_match.group(1) if sqft_match else 'Not found'}")

        # BER
        ber_match = re.search(r'BER[:\s]*([A-G][1-3]?)\b', body_text, re.IGNORECASE)
        print(f"   BER: {ber_match.group(1) if ber_match else 'Not found'}")

        # Eircode
        eircode_match = re.search(r'\b([A-Z]\d{2}\s?[A-Z0-9]{4})\b', body_text, re.IGNORECASE)
        print(f"   Eircode: {eircode_match.group(1) if eircode_match else 'Not found'}")

        # Price
        price_match = re.search(r'EUR\s*([\d,]+)', body_text, re.IGNORECASE)
        if not price_match:
            price_match = re.search(r'[^\d]([\d,]{6,})[^\d]', body_text)
        print(f"   Price: {price_match.group(1) if price_match else 'Not found'}")

        # Property type
        type_match = re.search(r'\b(Detached|Semi-Detached|Terrace|End of Terrace|Apartment|Bungalow|Duplex)\b', body_text, re.IGNORECASE)
        print(f"   Type: {type_match.group(1) if type_match else 'Not found'}")

        # Description
        print("\n5. Looking for description...")
        try:
            desc_selectors = [
                '[class*="description"]',
                '[class*="Description"]',
                '.property-description',
                '#description',
            ]
            for selector in desc_selectors:
                el = page.locator(selector).first
                if el.count() > 0:
                    desc = el.text_content()
                    if desc and len(desc) > 50:
                        print(f"   Found description ({len(desc)} chars):")
                        print(f"   {desc[:300]}...")
                        break
        except Exception as e:
            print(f"   Error: {e}")

        # Check for JSON-LD
        print("\n6. JSON-LD structured data:")
        try:
            scripts = page.locator('script[type="application/ld+json"]').all()
            for i, script in enumerate(scripts):
                content = script.text_content()
                data = json.loads(content)
                print(f"   Script {i+1}: @type = {data.get('@type', 'unknown')}")
                if data.get('@type') in ['Product', 'RealEstateListing', 'Residence']:
                    print(f"   Keys: {list(data.keys())}")
        except Exception as e:
            print(f"   Error: {e}")

        # Agent info
        print("\n7. Agent info:")
        # From title
        title = page.title()
        if " - " in title:
            parts = title.split(" - ")
            if len(parts) >= 2:
                print(f"   Agent (from title): {parts[1]}")

        # Number of images
        print("\n8. Images:")
        images = page.locator('img').all()
        property_images = [img for img in images if 'property' in (img.get_attribute('src') or '').lower() or 'brochure' in (img.get_attribute('src') or '').lower()]
        print(f"   Total images: {len(images)}")
        print(f"   Property images: {len(property_images)}")

        print("\n\n9. Browser open for 30 seconds for inspection...")
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            pass

        browser.close()

    print("\n" + "=" * 70)
    print("CHECK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
