#!/usr/bin/env python3
"""Debug MyHome.ie link extraction."""

import time
from playwright.sync_api import sync_playwright


def main():
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=False)
        page = browser.new_page()

        print("Loading MyHome.ie for-sale page...")
        page.goto("https://www.myhome.ie/residential/ireland/property-for-sale", wait_until="domcontentloaded")
        time.sleep(3)

        # Dismiss cookies
        try:
            btn = page.locator('#onetrust-accept-btn-handler').first
            if btn.is_visible(timeout=2000):
                btn.click()
                time.sleep(1)
        except:
            pass

        print(f"Title: {page.title()}")
        print(f"URL: {page.url}")

        # Find all links
        print("\n=== ALL LINKS ===")
        all_links = page.locator('a').all()
        print(f"Total links: {len(all_links)}")

        # Look for property links
        print("\n=== PROPERTY LINKS ===")
        patterns = ['/residential/brochure/', '/brochure/', '/property/', '/priceregister/']

        for pattern in patterns:
            links = page.locator(f'a[href*="{pattern}"]').all()
            print(f"\n'{pattern}': {len(links)} links")
            for link in links[:3]:
                href = link.get_attribute("href")
                print(f"  - {href[:80] if href else 'none'}")

        # Check the actual search results
        print("\n=== SEARCH RESULTS AREA ===")
        results_selectors = [
            '[data-testid="results"]',
            '.search-results',
            '.property-results',
            '[class*="result"]',
            '[class*="listing"]',
            '[class*="card"]',
        ]

        for selector in results_selectors:
            try:
                count = page.locator(selector).count()
                if count > 0:
                    print(f"  {selector}: {count} elements")
            except:
                pass

        # Look at page structure
        print("\n=== LOOKING FOR LISTING CARDS ===")
        # Try finding article or div elements that look like cards
        cards = page.locator('article, [class*="PropertyCard"], [class*="property-card"]').all()
        print(f"Found {len(cards)} potential card elements")

        if cards:
            first_card = cards[0]
            # Get links within first card
            card_links = first_card.locator('a').all()
            print(f"First card has {len(card_links)} links:")
            for link in card_links[:5]:
                href = link.get_attribute("href")
                print(f"  - {href}")

        print("\n\nBrowser open for 30 seconds...")
        time.sleep(30)
        browser.close()


if __name__ == "__main__":
    main()
