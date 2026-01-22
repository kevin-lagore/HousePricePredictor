#!/usr/bin/env python3
"""Explore offr.io API directly to find bid data."""

import json
import os
import time
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright


def main():
    print("=" * 70)
    print("OFFR.IO API EXPLORATION")
    print("=" * 70)

    load_dotenv()

    profile_dir = os.path.join(os.path.dirname(__file__), ".offr_profile")
    os.makedirs(profile_dir, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            profile_dir,
            headless=False,
            viewport={"width": 1920, "height": 1080},
            args=["--disable-blink-features=AutomationControlled"],
        )

        page = browser.pages[0] if browser.pages else browser.new_page()

        # Capture all API responses
        api_responses = []

        def handle_response(response):
            url = response.url
            try:
                content_type = response.headers.get('content-type', '')
                if 'application/json' in content_type:
                    body = response.json()
                    api_responses.append({'url': url, 'body': body})
                    body_str = json.dumps(body)
                    if any(kw in body_str.lower() for kw in ['bid', 'offer', 'amount', 'buyer']):
                        print(f"\n*** INTERESTING API: {url}")
                        print(f"{json.dumps(body, indent=2)[:2000]}")
            except:
                pass

        page.on("response", handle_response)
        page.set_default_timeout(30000)

        # Known property asset_id from Primestown Broadway
        asset_id = 149198
        agent_hash = "60f4fa64322200a34d8b7f0d14b9412f"  # Keane Auctioneers

        print(f"\nAsset ID: {asset_id}")
        print(f"Agent hash: {agent_hash}")

        # Try various API endpoints
        api_endpoints = [
            f"https://offr.io/api/property/{asset_id}",
            f"https://offr.io/api/asset/{asset_id}",
            f"https://offr.io/api/bids/{asset_id}",
            f"https://offr.io/api/offers/{asset_id}",
            f"https://offr.io/property-data/{asset_id}",
            f"https://offr.io/widget-data/{agent_hash}?asset_id={asset_id}",
            f"https://offr.io/auction/{asset_id}",
            f"https://offr.io/api/auction/{asset_id}",
        ]

        print("\n1. Trying direct API endpoints...")
        for endpoint in api_endpoints:
            print(f"\n   Trying: {endpoint}")
            api_responses.clear()
            try:
                page.goto(endpoint, wait_until="domcontentloaded", timeout=10000)
                time.sleep(1)

                # Check what we got
                page_text = page.locator('body').text_content() or ""
                if page_text and len(page_text) < 5000:
                    # Try to parse as JSON
                    try:
                        data = json.loads(page_text)
                        print(f"   Response: {json.dumps(data, indent=2)[:500]}")
                    except:
                        print(f"   Response (text): {page_text[:200]}")
            except Exception as e:
                print(f"   Error: {str(e)[:50]}")

        # Now try the actual property page and interact with the offr panel
        print("\n\n2. Opening property page and interacting with offr panel...")
        api_responses.clear()

        page.goto("https://keaneauctioneers.com/properties/primestown-broadway-co-wexford/", wait_until="domcontentloaded")
        time.sleep(3)

        # Wait for offr launcher to load, then click it
        print("   Waiting for offr launcher...")
        time.sleep(3)

        # Try to click the launcher by clicking its frame directly
        launcher_frame = page.frame_locator('iframe[src*="offr.io/launcher"]')
        try:
            # The launcher frame should have a clickable element
            launcher_frame.locator('body').click()
            print("   Clicked launcher frame body")
            time.sleep(3)
        except Exception as e:
            print(f"   Launcher click failed: {e}")

        # Check for panel that might have opened
        panel_frames = page.locator('iframe[src*="offr.io"]').all()
        print(f"\n   Found {len(panel_frames)} offr iframes after click")
        for frame in panel_frames:
            src = frame.get_attribute('src') or ""
            print(f"   - {src}")

        # Now click on "Place a bid" in the bubble
        print("\n   Looking for 'Place a bid' button in bubble...")
        bubble_frame = page.frame_locator('iframe[src*="widget-bubble"]')
        try:
            bid_btn = bubble_frame.locator('button:has-text("bid"), a:has-text("bid"), [class*="bid"]').first
            if bid_btn.count() > 0:
                print("   Found bid button, clicking...")
                bid_btn.click()
                time.sleep(5)

                # Check for new iframes (the panel should open)
                new_frames = page.locator('iframe[src*="offr.io"]').all()
                print(f"   Now have {len(new_frames)} offr iframes")
                for frame in new_frames:
                    src = frame.get_attribute('src') or ""
                    print(f"   - {src}")
        except Exception as e:
            print(f"   Bubble interaction error: {e}")

        # Try clicking on the bubble iframe body to expand it
        print("\n   Clicking bubble frame to expand...")
        try:
            bubble_frame.locator('body').click()
            time.sleep(3)

            # Check for new iframes
            new_frames = page.locator('iframe[src*="offr.io"]').all()
            print(f"   Now have {len(new_frames)} offr iframes")
            for frame in new_frames:
                src = frame.get_attribute('src') or ""
                print(f"   - {src}")
        except Exception as e:
            print(f"   Error: {e}")

        # Look for the panel iframe specifically
        panel = page.frame_locator('iframe[src*="offr.io/panel"]')
        try:
            panel_body = panel.locator('body')
            if panel_body.count() > 0:
                print(f"\n   Panel found! Content: {panel_body.text_content()[:500]}")
        except:
            pass

        # Try clicking on the property itself to trigger panel
        print("\n3. Trying to trigger panel via property page...")

        # Click on any "Bid" or "Offer" text on the page
        bid_links = page.locator('a:has-text("Bid"), button:has-text("Bid"), a:has-text("Offer"), button:has-text("Offer")').all()
        print(f"   Found {len(bid_links)} bid/offer links")
        for link in bid_links[:3]:
            text = link.text_content() or ""
            print(f"   - '{text[:50]}'")

        # Print all API responses captured
        print(f"\n4. API responses captured during session: {len(api_responses)}")
        for resp in api_responses:
            url = resp['url']
            body = resp['body']
            if 'offr' in url.lower():
                print(f"\n   {url}")
                # Check for interesting data
                body_str = json.dumps(body)
                if any(kw in body_str.lower() for kw in ['bid', 'offer', 'amount', 'buyer', 'history']):
                    print(f"   >>> {json.dumps(body, indent=2)[:1000]}")

        print("\n5. Keeping browser open for manual exploration...")
        print("   Try clicking the red Offr button in the bottom right")
        print("   Then click 'Place a bid' to see the bidding interface")
        time.sleep(60)

        browser.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
