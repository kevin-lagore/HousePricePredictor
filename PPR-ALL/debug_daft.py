#!/usr/bin/env python3
"""Debug script to see what's actually on the Daft.ie sold properties page."""

import json
import time
from playwright.sync_api import sync_playwright

def main():
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=False)
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        )
        page = context.new_page()

        # Go to a single sold listing page to see what data is available
        url = "https://www.daft.ie/sold/88-carrigweir-weir-rd-tuam-galway/EB3161BA3910287080258D710060257D"
        print(f"Navigating to: {url}")
        page.goto(url, wait_until="domcontentloaded")

        # Wait for any Cloudflare
        print("Waiting for page to settle...")
        time.sleep(5)

        # Try to dismiss cookie popup
        try:
            for selector in ['button:has-text("Accept")', '#didomi-notice-agree-button', 'button[id*="accept"]']:
                try:
                    btn = page.locator(selector).first
                    if btn.is_visible(timeout=1000):
                        btn.click()
                        print(f"Clicked cookie button: {selector}")
                        time.sleep(1)
                        break
                except:
                    continue
        except:
            pass

        # Wait for content
        time.sleep(3)

        # Print page title
        print(f"\nPage title: {page.title()}")
        print(f"Current URL: {page.url}")

        # Find all links
        print("\n=== ALL LINKS ===")
        links = page.locator('a').all()
        print(f"Total links: {len(links)}")

        # Look for property-related links
        print("\n=== PROPERTY LINKS ===")
        for link in links[:50]:  # First 50 links
            try:
                href = link.get_attribute("href")
                if href and any(x in href for x in ["/sold/", "/for-sale/", "property"]):
                    text = link.text_content()[:50] if link.text_content() else "no text"
                    print(f"  {href[:80]} | {text}")
            except:
                continue

        # Check for data-testid elements
        print("\n=== DATA-TESTID ELEMENTS ===")
        testid_elements = page.locator('[data-testid]').all()
        testids = set()
        for el in testid_elements[:30]:
            try:
                testid = el.get_attribute("data-testid")
                if testid and testid not in testids:
                    testids.add(testid)
                    print(f"  {testid}")
            except:
                continue

        # Print snippet of page content
        print("\n=== PAGE CONTENT SNIPPET ===")
        content = page.content()
        # Look for listing-related content
        if "sold" in content.lower():
            print("Page contains 'sold'")
        if "properties" in content.lower():
            print("Page contains 'properties'")

        # Check for __NEXT_DATA__
        print("\n=== __NEXT_DATA__ ===")
        try:
            script = page.locator('script#__NEXT_DATA__').first
            if script.is_visible(timeout=2000):
                content = script.text_content()
                data = json.loads(content)
                print(json.dumps(data, indent=2)[:5000])
        except Exception as e:
            print(f"No __NEXT_DATA__: {e}")

        # Check for JSON-LD
        print("\n=== JSON-LD ===")
        try:
            scripts = page.locator('script[type="application/ld+json"]').all()
            for script in scripts:
                content = script.text_content()
                data = json.loads(content)
                print(json.dumps(data, indent=2)[:2000])
        except Exception as e:
            print(f"No JSON-LD: {e}")

        # Keep browser open for manual inspection
        print("\n\nBrowser will stay open for 30 seconds for inspection...")
        print("Press Ctrl+C to close earlier")
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            pass

        browser.close()

if __name__ == "__main__":
    main()
