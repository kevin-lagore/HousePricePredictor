#!/usr/bin/env python3
"""Simple exploration of offr.io to find bid data on agent websites."""

import json
import os
import time
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright


def main():
    print("=" * 70)
    print("OFFR.IO SIMPLE EXPLORATION")
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

        # Capture API responses
        api_responses = []

        def handle_response(response):
            url = response.url
            if 'offr' in url.lower():
                try:
                    content_type = response.headers.get('content-type', '')
                    if 'application/json' in content_type:
                        body = response.json()
                        api_responses.append({'url': url, 'body': body})
                        body_str = json.dumps(body)
                        # Print full response for interesting endpoints
                        if 'config' in url or 'bid' in url.lower() or 'offer' in url.lower():
                            print(f"\n[API FULL] {url}")
                            print(f"{json.dumps(body, indent=2)}")
                        else:
                            print(f"\n[API] {url[:80]}")
                            print(f"      {body_str[:200]}...")
                except:
                    pass

        page.on("response", handle_response)
        page.set_default_timeout(30000)

        # Go directly to a Keane property with auction
        print("\n1. Visiting Keane Auctioneers property page...")
        page.goto("https://keaneauctioneers.com/properties/primestown-broadway-co-wexford/", wait_until="domcontentloaded")
        time.sleep(5)

        print(f"   Title: {page.title()}")

        # Check for offr elements
        offr_els = page.locator('[id*="offr"], [class*="offr"], iframe[src*="offr"]').all()
        print(f"   Found {len(offr_els)} offr elements")

        # List all iframes
        iframes = page.locator('iframe').all()
        print(f"   Found {len(iframes)} iframes")
        for iframe in iframes:
            src = iframe.get_attribute('src') or ""
            print(f"     - {src[:80]}")

        # Look for offr launcher iframe and click inside it
        launcher_iframe = page.frame_locator('iframe[src*="offr.io/launcher"]')
        try:
            # Click inside the launcher iframe
            launcher_btn = launcher_iframe.locator('button, [class*="launcher"], [class*="btn"], div').first
            if launcher_btn.count() > 0:
                print("   Found button in launcher iframe, clicking...")
                launcher_btn.click()
                time.sleep(5)

                # Check if panel opened
                panel_iframe = page.frame_locator('iframe[src*="offr.io/panel"], iframe[src*="offr.io/widget"]')
                try:
                    panel_content = panel_iframe.locator('body').text_content()
                    print(f"   Panel content: {panel_content[:500] if panel_content else 'empty'}...")
                except:
                    pass
        except Exception as e:
            print(f"   Launcher click error: {e}")

        # Check page text for bid info
        page_text = page.locator('body').text_content() or ""
        if 'auction' in page_text.lower():
            print("   Page mentions 'auction'")
        if '€' in page_text:
            # Find price mentions
            import re
            prices = re.findall(r'€[\d,]+', page_text)
            print(f"   Prices found: {prices[:10]}")

        # Try another property
        print("\n2. Trying Littletown property...")
        api_responses.clear()
        page.goto("https://keaneauctioneers.com/properties/premium-c-20-07-acres-holding-at-littletown-tomhaggard-co-wexford/", wait_until="domcontentloaded")
        time.sleep(5)

        print(f"   Title: {page.title()}")

        # Check API responses
        print(f"\n   API responses captured: {len(api_responses)}")
        for resp in api_responses:
            print(f"\n   URL: {resp['url']}")
            print(f"   Body: {json.dumps(resp['body'], indent=2)}")

        # Look for offr panel
        panel = page.locator('#offr_panel, [class*="offr-panel"]').first
        if panel.count() > 0:
            print("   Found offr panel!")
            print(f"   Content: {panel.text_content()[:500]}")

        # Check iframes for offr content
        for iframe in page.locator('iframe').all():
            src = iframe.get_attribute('src') or ""
            if 'offr' in src.lower():
                print(f"\n   OFFR iframe found: {src}")
                try:
                    frame = iframe.content_frame()
                    if frame:
                        frame_html = frame.locator('body').inner_html()
                        print(f"   Frame HTML: {frame_html[:500]}...")
                except Exception as e:
                    print(f"   Frame error: {e}")

        print("\n3. Keeping browser open for 60 seconds for manual exploration...")
        print("   - Open DevTools (F12) > Network tab")
        print("   - Look for requests to offr.io")
        print("   - Try clicking any 'Bid' or 'Offer' buttons")
        time.sleep(60)

        browser.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
