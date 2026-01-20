#!/usr/bin/env python3
"""Explore MyHome.ie to understand data structure in depth."""

import json
import re
import time
from playwright.sync_api import sync_playwright


def main():
    print("=" * 70)
    print("MYHOME.IE DEEP EXPLORATION")
    print("=" * 70)

    with sync_playwright() as p:
        browser = p.firefox.launch(headless=False)
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        )
        page = context.new_page()
        page.set_default_timeout(30000)

        # 1. Check an active for-sale listing first
        print("\n1. Loading for-sale listings page...")
        page.goto("https://www.myhome.ie/residential/ireland/property-for-sale", wait_until="domcontentloaded")
        time.sleep(3)

        # Dismiss cookies
        try:
            for selector in ['#onetrust-accept-btn-handler', 'button:has-text("Accept")', 'button:has-text("I Accept")']:
                btn = page.locator(selector).first
                if btn.is_visible(timeout=2000):
                    btn.click()
                    print("   Dismissed cookie popup")
                    time.sleep(1)
                    break
        except:
            pass

        # Find listing links
        listing_links = page.locator('a[href*="/residential/brochure/"]').all()
        print(f"   Found {len(listing_links)} for-sale brochure links")

        if listing_links:
            href = listing_links[0].get_attribute("href")
            listing_url = href if href.startswith("http") else f"https://www.myhome.ie{href}"
            print(f"   Selected: {listing_url}")

            print("\n2. Loading listing page...")
            page.goto(listing_url, wait_until="domcontentloaded")
            time.sleep(4)

            print(f"   Title: {page.title()}")

            # Check all script tags
            print("\n3. Analyzing all script tags...")
            scripts = page.locator('script').all()
            print(f"   Total scripts: {len(scripts)}")

            json_data_found = []
            for i, script in enumerate(scripts):
                try:
                    content = script.text_content() or ""
                    # Look for JSON data patterns
                    if "window." in content and "{" in content:
                        # Try to extract variable assignments
                        matches = re.findall(r'window\.(\w+)\s*=\s*(\{[^;]+\});', content)
                        for name, json_str in matches:
                            try:
                                data = json.loads(json_str)
                                json_data_found.append((name, data))
                            except:
                                pass

                    # Look for inline JSON
                    if content.strip().startswith('{') and len(content) > 100:
                        try:
                            data = json.loads(content)
                            json_data_found.append((f"script_{i}", data))
                        except:
                            pass
                except:
                    continue

            print(f"   Found {len(json_data_found)} JSON data objects")
            for name, data in json_data_found:
                print(f"     - {name}: {list(data.keys())[:10] if isinstance(data, dict) else type(data)}")

            # Check for JSON-LD
            print("\n4. Checking JSON-LD...")
            ld_scripts = page.locator('script[type="application/ld+json"]').all()
            print(f"   Found {len(ld_scripts)} JSON-LD scripts")

            for i, script in enumerate(ld_scripts):
                try:
                    content = script.text_content()
                    data = json.loads(content)
                    print(f"\n   JSON-LD #{i+1}:")
                    print(f"     @type: {data.get('@type', 'unknown')}")

                    if data.get('@type') in ['Product', 'RealEstateListing', 'Residence', 'Place', 'SingleFamilyResidence']:
                        print(f"     Keys: {list(data.keys())}")
                        # Print interesting fields
                        for key in ['name', 'description', 'address', 'geo', 'numberOfRooms', 'floorSize']:
                            if key in data:
                                val = data[key]
                                if isinstance(val, str) and len(val) > 60:
                                    val = val[:60] + "..."
                                print(f"     {key}: {val}")
                except Exception as e:
                    print(f"   Error parsing JSON-LD: {e}")

            # Look for property data in DOM
            print("\n5. Extracting from DOM elements...")

            # Property details section
            details = {}

            # Try various selectors for common property info
            selector_map = [
                ('Beds', '[data-testid*="bed"], .beds, [class*="bed" i], span:has-text("Bed")'),
                ('Baths', '[data-testid*="bath"], .baths, [class*="bath" i], span:has-text("Bath")'),
                ('BER', '[data-testid*="ber"], .ber-rating, [class*="ber" i], img[alt*="BER"]'),
                ('Price', '[data-testid*="price"], .price, [class*="price" i]'),
                ('Size', '[data-testid*="floor"], [class*="floor-area" i], [class*="size" i]'),
                ('Type', '[data-testid*="type"], .property-type, [class*="property-type" i]'),
            ]

            for name, selector in selector_map:
                try:
                    elements = page.locator(selector).all()
                    if elements:
                        texts = []
                        for el in elements[:3]:
                            text = el.text_content()
                            alt = el.get_attribute('alt')
                            if text:
                                texts.append(text.strip()[:50])
                            elif alt:
                                texts.append(f"[alt: {alt[:30]}]")
                        if texts:
                            details[name] = texts
                except:
                    pass

            print("   Found via selectors:")
            for name, vals in details.items():
                print(f"     {name}: {vals}")

            # Parse full page text for data
            print("\n6. Pattern matching on page text...")
            body_text = page.locator('body').text_content()

            patterns = [
                ('Beds', r'(\d+)\s*(?:Bed|bedroom)', re.IGNORECASE),
                ('Baths', r'(\d+)\s*(?:Bath|bathroom)', re.IGNORECASE),
                ('Size_sqm', r'([\d,.]+)\s*(?:sq\.?\s*m|m2|sqm)', re.IGNORECASE),
                ('Size_sqft', r'([\d,.]+)\s*(?:sq\.?\s*ft|sqft)', re.IGNORECASE),
                ('BER', r'BER[:\s]*([A-G][1-3]?)\b', re.IGNORECASE),
                ('Price', r'EUR\s*([\d,]+)|euro\s*([\d,]+)', re.IGNORECASE),
                ('Eircode', r'\b([A-Z]\d{2}\s?[A-Z0-9]{4})\b', re.IGNORECASE),
            ]

            for name, pattern, flags in patterns:
                match = re.search(pattern, body_text, flags)
                if match:
                    print(f"     {name}: {match.group(1) or match.group(0)}")

            # Check for API endpoints in network or page
            print("\n7. Looking for API data sources...")

            # Check for data attributes on body/main
            main_el = page.locator('main, #__nuxt, #app, [data-v-]').first
            if main_el.count() > 0:
                attrs = page.evaluate('''
                    (selector) => {
                        const el = document.querySelector(selector);
                        if (!el) return {};
                        const attrs = {};
                        for (const attr of el.attributes) {
                            if (attr.name.startsWith('data-')) {
                                attrs[attr.name] = attr.value.substring(0, 100);
                            }
                        }
                        return attrs;
                    }
                ''', 'main, #__nuxt, #app')
                if attrs:
                    print(f"   Data attributes: {attrs}")

            # Check window object for data
            print("\n8. Checking window object...")
            window_data = page.evaluate('''
                () => {
                    const data = {};
                    const keys = ['__INITIAL_STATE__', '__PRELOADED_STATE__', 'pageData',
                                 'brochureData', 'propertyData', '__NUXT__', 'Brochure'];
                    for (const key of keys) {
                        if (window[key]) {
                            data[key] = typeof window[key];
                        }
                    }
                    // Also check for any large objects
                    for (const key in window) {
                        try {
                            if (typeof window[key] === 'object' &&
                                window[key] !== null &&
                                !key.startsWith('webkit') &&
                                !key.startsWith('on')) {
                                const str = JSON.stringify(window[key]);
                                if (str && str.length > 1000 && str.includes('address')) {
                                    data[key] = `object with ${Object.keys(window[key]).length} keys`;
                                }
                            }
                        } catch {}
                    }
                    return data;
                }
            ''')
            if window_data:
                print(f"   Window data sources: {window_data}")

        print("\n\n9. Keeping browser open for 30 seconds for manual inspection...")
        print("   Look at Network tab for API calls")
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            pass

        browser.close()

    print("\n" + "=" * 70)
    print("EXPLORATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
