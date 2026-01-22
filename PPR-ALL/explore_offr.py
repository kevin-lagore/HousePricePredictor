#!/usr/bin/env python3
"""Explore offr.io to understand data structure and available offer information.

This version auto-logs in using credentials from .env file.
Uses persistent profile to avoid Cloudflare detection.
"""

import json
import os
import re
import time
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright


def main():
    print("=" * 70)
    print("OFFR.IO EXPLORATION - AUTO LOGIN")
    print("=" * 70)

    # Load credentials from .env
    load_dotenv()
    email = os.getenv("OFFR_EMAIL")
    password = os.getenv("OFFR_PASSWORD")

    if not email or not password:
        print("ERROR: Please set OFFR_EMAIL and OFFR_PASSWORD in .env file")
        return

    print(f"\nLogging in as: {email}")

    # Use a persistent profile directory so cookies/sessions are saved
    profile_dir = os.path.join(os.path.dirname(__file__), ".offr_profile")
    os.makedirs(profile_dir, exist_ok=True)
    print(f"Using persistent profile: {profile_dir}")

    with sync_playwright() as p:
        # Use Chrome with persistent context
        browser = p.chromium.launch_persistent_context(
            profile_dir,
            headless=False,
            viewport={"width": 1920, "height": 1080},
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )

        page = browser.pages[0] if browser.pages else browser.new_page()

        # Capture API responses
        api_responses = []

        def handle_response(response):
            url = response.url
            if '/api/' in url or 'offr.io' in url:
                try:
                    content_type = response.headers.get('content-type', '')
                    if 'application/json' in content_type:
                        body = response.json()
                        api_responses.append({
                            'url': url,
                            'status': response.status,
                            'body': body
                        })
                        body_str = json.dumps(body)
                        if any(kw in body_str.lower() for kw in ['bid', 'offer', 'buyer', 'bidder']):
                            print(f"\n   [API] Interesting response: {url[:80]}")
                            print(f"         Keys: {list(body.keys()) if isinstance(body, dict) else type(body)}")
                except:
                    pass

        page.on("response", handle_response)
        page.set_default_timeout(60000)

        # Step 1: Go to login page and log in
        print("\n" + "=" * 70)
        print("STEP 1: Logging in to offr.io")
        print("=" * 70)

        page.goto("https://offr.io/login", wait_until="networkidle")
        time.sleep(2)

        # Check if we hit Cloudflare
        if "challenge" in page.url.lower() or "cloudflare" in page.content().lower():
            print("\n>>> Cloudflare challenge detected! <<<")
            print(">>> Please solve the captcha in the browser, then press ENTER <<<")
            input()

        # Check if already logged in (might be from persistent profile)
        if "login" in page.url.lower():
            print("Filling in login form...")

            # Fill email
            email_input = page.locator('input[type="email"], input[name="email"], input[id="email"]').first
            if email_input.count() > 0:
                email_input.fill(email)

            # Fill password
            password_input = page.locator('input[type="password"], input[name="password"]').first
            if password_input.count() > 0:
                password_input.fill(password)

            # Click login button
            login_btn = page.locator('button[type="submit"], button:has-text("Log In"), button:has-text("Login")').first
            if login_btn.count() > 0:
                login_btn.click()

            # Wait for login to complete
            time.sleep(5)

            # Check if login succeeded
            if "login" in page.url.lower():
                print("Login may have failed. Check for errors in browser.")
                print("Press ENTER to continue anyway...")
                input()
        else:
            print("Already logged in (from persistent session)")

        # Clear previous responses
        api_responses.clear()

        # Step 2: Check what's visible on auction page when logged in
        print("\n" + "=" * 70)
        print("STEP 2: Exploring auction listings as logged-in user")
        print("=" * 70)

        page.goto("https://offr.io/property-for-auction/ireland", wait_until="networkidle")
        time.sleep(3)

        print(f"\nPage title: {page.title()}")

        # Find properties with bids
        cards = page.locator('.card, [class*="property"]').all()
        print(f"Found {len(cards)} property cards")

        properties_with_bids = []
        for card in cards:
            try:
                card_text = card.text_content() or ""
                link = card.locator('a[href*="/property/"]').first
                href = link.get_attribute('href') if link.count() > 0 else None

                if href and ('bid' in card_text.lower() or '€' in card_text):
                    bid_match = re.search(r'(?:Latest Bid|Current Bid)[:\s]*€?([\d,]+)', card_text, re.IGNORECASE)
                    guide_match = re.search(r'(?:Guide|Asking)[:\s]*€?([\d,]+)', card_text, re.IGNORECASE)

                    properties_with_bids.append({
                        'href': href,
                        'latest_bid': bid_match.group(1) if bid_match else None,
                        'guide_price': guide_match.group(1) if guide_match else None,
                    })
            except:
                continue

        print(f"\nProperties with visible bids: {len(properties_with_bids)}")
        for prop in properties_with_bids[:10]:
            print(f"  {prop['href']}")
            print(f"    Latest Bid: €{prop['latest_bid']} | Guide: €{prop['guide_price']}")

        # Step 3: Visit individual property pages to look for bid history
        print("\n" + "=" * 70)
        print("STEP 3: Checking individual property pages for bid data")
        print("=" * 70)

        for prop in properties_with_bids[:3]:
            if not prop['href']:
                continue

            full_url = prop['href'] if prop['href'].startswith('http') else f"https://offr.io{prop['href']}"
            print(f"\n--- Visiting: {full_url} ---")

            api_responses.clear()
            page.goto(full_url, wait_until="networkidle")
            time.sleep(2)

            # Look for bid-related elements
            bid_elements = []
            for selector in ['[class*="bid"]', '[class*="offer"]', '[class*="history"]',
                           '[class*="timeline"]', 'table', '[class*="activity"]']:
                els = page.locator(selector).all()
                for el in els:
                    text = (el.text_content() or "").strip()
                    if text and len(text) > 10 and ('€' in text or 'bid' in text.lower()):
                        bid_elements.append((selector, text[:200]))

            if bid_elements:
                print(f"  Found {len(bid_elements)} elements with bid/offer content:")
                for selector, text in bid_elements[:5]:
                    print(f"    [{selector}]: {text}...")

            # Check for action buttons
            action_buttons = page.locator('button, a').all()
            for btn in action_buttons:
                btn_text = (btn.text_content() or "").strip().lower()
                if any(kw in btn_text for kw in ['bid', 'offer', 'place', 'make']):
                    print(f"  Action button found: '{btn_text}'")

            # Check window object for bid data
            window_bid_data = page.evaluate('''
                () => {
                    const results = {};
                    for (const key in window) {
                        try {
                            if (typeof window[key] === 'object' && window[key] !== null) {
                                const str = JSON.stringify(window[key]);
                                if (str && str.length > 100) {
                                    const lower = str.toLowerCase();
                                    if (lower.includes('"bid') || lower.includes('"offer') ||
                                        lower.includes('bidder') || lower.includes('buyer_id')) {
                                        results[key] = {
                                            size: str.length,
                                            sample: str.substring(0, 1000)
                                        };
                                    }
                                }
                            }
                        } catch {}
                    }
                    return results;
                }
            ''')

            if window_bid_data:
                print(f"  Window data with bid info found:")
                for key, info in window_bid_data.items():
                    print(f"    {key} ({info['size']} chars):")
                    try:
                        sample_data = json.loads(info['sample'])
                        print(f"      {json.dumps(sample_data, indent=6)[:500]}...")
                    except:
                        print(f"      {info['sample'][:300]}...")

            if api_responses:
                print(f"  API responses captured: {len(api_responses)}")
                for resp in api_responses:
                    if 'bid' in resp['url'].lower() or 'offer' in resp['url'].lower():
                        print(f"    {resp['url']}")
                        print(f"    {json.dumps(resp['body'], indent=4)[:500]}...")

        # Step 4: Try clicking "Place Bid" to see what happens
        print("\n" + "=" * 70)
        print("STEP 4: Checking 'Place Bid' flow")
        print("=" * 70)

        if properties_with_bids:
            prop = properties_with_bids[0]
            full_url = prop['href'] if prop['href'].startswith('http') else f"https://offr.io{prop['href']}"

            print(f"\nGoing to: {full_url}")
            api_responses.clear()
            page.goto(full_url, wait_until="networkidle")
            time.sleep(2)

            # Look for bid button
            bid_btn = None
            for selector in ['button:has-text("Bid")', 'button:has-text("Offer")',
                           'a:has-text("Place Bid")', '[class*="bid"] button',
                           'button:has-text("Place")', '.btn:has-text("Bid")']:
                try:
                    btn = page.locator(selector).first
                    if btn.count() > 0 and btn.is_visible():
                        bid_btn = btn
                        print(f"Found bid button with selector: {selector}")
                        break
                except:
                    continue

            if bid_btn:
                print("Clicking bid button to see what data loads...")
                try:
                    bid_btn.click()
                    time.sleep(3)

                    page_text = page.locator('body').text_content()

                    if 'bid history' in page_text.lower():
                        print("  BID HISTORY section found!")
                        history_el = page.locator('[class*="history"], [class*="bids"]').first
                        if history_el.count() > 0:
                            print(f"  Content: {history_el.text_content()[:500]}...")

                    modal = page.locator('[class*="modal"], [class*="dialog"], [role="dialog"]').first
                    if modal.count() > 0 and modal.is_visible():
                        print(f"  Modal content: {modal.text_content()[:500]}...")

                    if api_responses:
                        print(f"  API calls made: {len(api_responses)}")
                        for resp in api_responses:
                            print(f"    URL: {resp['url']}")
                            body_str = json.dumps(resp['body'], indent=2)
                            print(f"    Response: {body_str[:1000]}...")

                except Exception as e:
                    print(f"  Error clicking bid button: {e}")
            else:
                print("No bid button found on this property")

        # Step 5: Check account/dashboard for bid history
        print("\n" + "=" * 70)
        print("STEP 5: Checking account dashboard")
        print("=" * 70)

        dashboard_urls = [
            "https://offr.io/dashboard",
            "https://offr.io/account",
            "https://offr.io/my-bids",
            "https://offr.io/my-offers",
            "https://offr.io/buyer/dashboard",
        ]

        for url in dashboard_urls:
            print(f"\nTrying: {url}")
            api_responses.clear()
            try:
                page.goto(url, wait_until="networkidle", timeout=10000)
                time.sleep(2)

                if 'login' not in page.url.lower():
                    print(f"  Page loaded: {page.title()}")

                    body_text = page.locator('body').text_content()
                    if any(kw in body_text.lower() for kw in ['bid', 'offer', 'property', 'auction']):
                        print(f"  Found relevant content on this page")

                        for selector in ['table', '[class*="list"]', '[class*="history"]']:
                            els = page.locator(selector).all()
                            for el in els[:2]:
                                text = (el.text_content() or "").strip()
                                if '€' in text and len(text) > 50:
                                    print(f"  [{selector}]: {text[:300]}...")

                    if api_responses:
                        print(f"  API responses: {len(api_responses)}")
                        for resp in api_responses[-3:]:
                            print(f"    {resp['url'][:80]}")
                else:
                    print(f"  Redirected to login")
            except Exception as e:
                print(f"  Error: {e}")

        # Step 6: Visit actual agent website with offr widget
        print("\n" + "=" * 70)
        print("STEP 6: Visiting agent website with offr widget")
        print("=" * 70)

        # Visit Keane Auctioneers online auctions page
        api_responses.clear()
        print("\nGoing to: https://keaneauctioneers.com/online-auctions/")
        page.goto("https://keaneauctioneers.com/online-auctions/", wait_until="networkidle")
        time.sleep(3)

        print(f"Page title: {page.title()}")

        # Look for property links
        property_links = page.locator('a[href*="properties/"], a[href*="property/"]').all()
        print(f"Found {len(property_links)} property links")

        for link in property_links[:5]:
            href = link.get_attribute('href')
            print(f"  {href}")

        # Check for offr iframe/widget
        iframes = page.locator('iframe').all()
        print(f"\nFound {len(iframes)} iframes")
        for iframe in iframes:
            src = iframe.get_attribute('src') or ""
            if 'offr' in src.lower():
                print(f"  OFFR iframe: {src}")

        # Try visiting a specific property (not the all-properties page)
        actual_property_links = [l for l in property_links
                                 if 'all-properties' not in (l.get_attribute('href') or '')]
        if actual_property_links:
            prop_url = actual_property_links[0].get_attribute('href')
            print(f"\n--- Visiting property: {prop_url} ---")
            api_responses.clear()
            page.goto(prop_url, wait_until="networkidle")
            time.sleep(3)

            # Look for offr widget/button
            offr_elements = page.locator('[id*="offr"], [class*="offr"], iframe[src*="offr"]').all()
            print(f"Found {len(offr_elements)} offr elements")

            # Try to find and click the offr launcher
            launcher = page.locator('#offr_launcher_frame, [id*="offr_launcher"]').first
            if launcher.count() > 0:
                print("Found offr launcher frame")
                # The launcher is likely an iframe, let's see what's inside
                try:
                    frame = launcher.content_frame()
                    if frame:
                        frame_content = frame.locator('body').text_content()
                        print(f"  Launcher content: {frame_content[:200] if frame_content else 'empty'}...")
                except Exception as e:
                    print(f"  Could not read frame: {e}")

            # Look for any visible offr panel
            panel = page.locator('#offr_panel, [id*="offr_panel"], .offr-panel').first
            if panel.count() > 0 and panel.is_visible():
                print("Found visible offr panel!")
                panel_text = panel.text_content()
                print(f"  Panel content: {panel_text[:500] if panel_text else 'empty'}...")

            # Check all iframes on this page
            for iframe in page.locator('iframe').all():
                src = iframe.get_attribute('src') or ""
                name = iframe.get_attribute('name') or iframe.get_attribute('id') or "unnamed"
                print(f"  iframe [{name}]: {src[:100]}")

                if 'offr' in src.lower():
                    try:
                        frame = iframe.content_frame()
                        if frame:
                            # Look for bid data in the frame
                            frame_text = frame.locator('body').text_content() or ""
                            if 'bid' in frame_text.lower() or '€' in frame_text:
                                print(f"    BID DATA FOUND: {frame_text[:500]}...")
                    except Exception as e:
                        print(f"    Frame error: {e}")

            # Print API responses
            if api_responses:
                print(f"\nAPI responses: {len(api_responses)}")
                for resp in api_responses:
                    url = resp['url']
                    if 'offr' in url.lower():
                        print(f"  OFFR API: {url}")
                        body = resp['body']
                        print(f"    Body: {json.dumps(body)[:1000]}...")

                        # If widget exists, try to access it
                        if isinstance(body, dict) and body.get('exists'):
                            print("    Widget EXISTS! Trying to interact...")

        # Step 7: Try a different property with known auction
        print("\n" + "=" * 70)
        print("STEP 7: Trying Primestown Broadway property (known active auction)")
        print("=" * 70)

        # This property showed €240k bid earlier
        api_responses.clear()
        page.goto("https://keaneauctioneers.com/properties/primestown-broadway-co-wexford/", wait_until="networkidle")
        time.sleep(3)

        print(f"Page title: {page.title()}")
        print(f"URL: {page.url}")

        # Check for offr widget
        offr_elements = page.locator('[id*="offr"], [class*="offr"], iframe[src*="offr"]').all()
        print(f"Found {len(offr_elements)} offr elements")

        # Check all iframes
        for iframe in page.locator('iframe').all():
            src = iframe.get_attribute('src') or ""
            name = iframe.get_attribute('name') or iframe.get_attribute('id') or "unnamed"
            print(f"  iframe [{name}]: {src[:100]}")

        # Print page content looking for bid info
        page_text = page.locator('body').text_content() or ""
        if 'bid' in page_text.lower():
            # Find the context around "bid"
            lower_text = page_text.lower()
            idx = lower_text.find('bid')
            print(f"  Found 'bid' in page: ...{page_text[max(0,idx-50):idx+100]}...")

        # Check API responses
        print(f"\nAPI responses: {len(api_responses)}")
        for resp in api_responses:
            url = resp['url']
            body = resp['body']
            print(f"  {url[:80]}")
            if 'offr' in url.lower():
                print(f"    Full body: {json.dumps(body, indent=2)}")

        # Final: Keep browser open for a bit then close
        print("\nWaiting 10 seconds before closing...")
        time.sleep(10)

        # Print summary
        print("\n" + "=" * 70)
        print("API RESPONSE SUMMARY")
        print("=" * 70)

        interesting_responses = [r for r in api_responses
                                if any(kw in json.dumps(r['body']).lower()
                                      for kw in ['bid', 'offer', 'buyer', 'bidder'])]

        if interesting_responses:
            print(f"\nFound {len(interesting_responses)} responses with bid/offer data:")
            for resp in interesting_responses:
                print(f"\n  URL: {resp['url']}")
                print(f"  Response: {json.dumps(resp['body'], indent=2)[:2000]}")
        else:
            print("\nNo API responses with bid/offer data captured.")
            print("The bid data might be:")
            print("  - Rendered server-side (in HTML, not via API)")
            print("  - Only visible to verified bidders on specific properties")
            print("  - Loaded via WebSocket instead of REST API")

        browser.close()

    print("\n" + "=" * 70)
    print("EXPLORATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
