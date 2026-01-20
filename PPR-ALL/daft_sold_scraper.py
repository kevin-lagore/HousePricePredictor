#!/usr/bin/env python3
"""
Daft.ie Sold Properties Scraper

Searches daft.ie/sold-properties for PPR addresses and extracts listing details.
Uses Playwright for browser automation to bypass anti-scraping measures.

Usage:
  # First install dependencies
  pip install playwright
  playwright install chromium

  # Match PPR addresses to Daft sold listings
  python daft_sold_scraper.py --match --in PPR-ALL.csv --ppr_format --out matched.csv

  # Extract details from matched listings
  python daft_sold_scraper.py --extract --in matched.csv --out enriched.csv

  # Resume from checkpoint
  python daft_sold_scraper.py --match --in PPR-ALL.csv --ppr_format --out matched.csv --resume

  # Use parallel processing (faster but higher detection risk)
  python daft_sold_scraper.py --match --in PPR-ALL.csv --ppr_format --out matched.csv --workers 3
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.parse import quote_plus

# Check for playwright
try:
    from playwright.sync_api import sync_playwright, Page, Browser
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False
    print("Playwright not installed. Run: pip install playwright && playwright install chromium")

# Stealth mode for bypassing anti-bot protection
try:
    from undetected_playwright import stealth_sync
    HAS_STEALTH = True
except ImportError:
    HAS_STEALTH = False
    print("Warning: undetected_playwright not installed. Run: pip install undetected-playwright")


@dataclass
class SoldListing:
    """Represents a sold property listing from Daft.ie"""
    address: str = ""
    daft_url: str = ""
    sold_price: Optional[float] = None
    asking_price: Optional[float] = None
    sold_date: str = ""

    # Property details
    property_type: str = ""
    beds: Optional[int] = None
    baths: Optional[int] = None
    size_sqm: Optional[float] = None
    ber_rating: str = ""

    # Location
    county: str = ""
    area: str = ""
    eircode: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    # Listing details
    description: str = ""
    features: List[str] = None
    images_count: int = 0

    # Match metadata
    ppr_address: str = ""
    match_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d.get('features'):
            d['features'] = json.dumps(d['features'])
        return d


class DaftSoldScraper:
    """
    Scraper for Daft.ie sold properties using Playwright.
    """

    SEARCH_URL = "https://www.daft.ie/sold-properties/ireland"
    BASE_URL = "https://www.daft.ie"

    def __init__(self, headless: bool = True, slow_mo: int = 0, timeout: int = 30000, worker_id: int = 0):
        self.headless = headless
        self.slow_mo = slow_mo  # Default to 0 for speed
        self.timeout = timeout  # Reduced from 60s to 30s
        self.worker_id = worker_id
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self._playwright = None
        self._cookies_dismissed = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        """Start the browser."""
        self._playwright = sync_playwright().start()

        self.browser = self._playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-images',  # Don't load images for speed
                '--disable-extensions',
            ]
        )
        # Create page directly from browser (not context) for stealth to work
        self.page = self.browser.new_page()
        self.page.set_default_timeout(self.timeout)

        # Block unnecessary resources for speed
        self.page.route("**/*.{png,jpg,jpeg,gif,webp,svg,ico}", lambda route: route.abort())
        self.page.route("**/*google*", lambda route: route.abort())
        self.page.route("**/*facebook*", lambda route: route.abort())
        self.page.route("**/*analytics*", lambda route: route.abort())

        # Apply stealth mode to bypass anti-bot protection
        if HAS_STEALTH:
            stealth_sync(self.page)

    def _dismiss_popups(self, force: bool = False):
        """Dismiss cookie consent and other popups.

        Args:
            force: If True, always attempt dismissal even if previously dismissed
        """
        # Aggressive JavaScript-based popup removal - runs every time
        # This handles Didomi and other consent managers completely
        try:
            self.page.evaluate("""
                () => {
                    // Remove Didomi completely - including shadow DOM and iframes
                    const removeDidomi = () => {
                        // Remove the main containers
                        ['#didomi-host', '#didomi-popup', '#didomi-notice', '.didomi-popup-container',
                         '.didomi-notice-popup', '.didomi-popup-backdrop', '#didomi-consent-popup-purposes'].forEach(sel => {
                            document.querySelectorAll(sel).forEach(el => el.remove());
                        });

                        // Remove any fixed/absolute positioned overlays that might block interaction
                        document.querySelectorAll('div').forEach(el => {
                            const style = window.getComputedStyle(el);
                            const zIndex = parseInt(style.zIndex) || 0;
                            // Remove high z-index overlays that are fixed/absolute and cover the page
                            if ((style.position === 'fixed' || style.position === 'absolute') &&
                                zIndex > 1000 &&
                                el.offsetWidth > window.innerWidth * 0.5) {
                                el.remove();
                            }
                        });

                        // Also hide via CSS as backup
                        const style = document.createElement('style');
                        style.textContent = `
                            #didomi-host, #didomi-popup, .didomi-popup-container,
                            .didomi-notice-popup, .didomi-popup-backdrop,
                            [class*="didomi"], [id*="didomi"] {
                                display: none !important;
                                visibility: hidden !important;
                                pointer-events: none !important;
                            }
                        `;
                        if (!document.querySelector('#didomi-hide-style')) {
                            style.id = 'didomi-hide-style';
                            document.head.appendChild(style);
                        }
                    };

                    // Run immediately
                    removeDidomi();

                    // Also set up a mutation observer to catch any popups that reappear
                    if (!window._didomiObserver) {
                        window._didomiObserver = new MutationObserver(() => removeDidomi());
                        window._didomiObserver.observe(document.body, { childList: true, subtree: true });
                    }
                }
            """)
        except:
            pass

        # If we've already dismissed once and not forcing, we're done
        if self._cookies_dismissed and not force:
            return True

        try:
            # Try to click the agree button if it exists (some sites need actual click)
            consent_selectors = [
                '#didomi-notice-agree-button',
                'button:has-text("Agree & Close")',
                'button:has-text("Accept All")',
                'button:has-text("Accept")',
            ]
            for selector in consent_selectors:
                try:
                    btn = self.page.query_selector(selector)
                    if btn and btn.is_visible():
                        btn.click(force=True)
                        self._cookies_dismissed = True
                        time.sleep(0.2)
                        return True
                except:
                    continue

            self._cookies_dismissed = True
            return True
        except:
            pass
        return False

    def _wait_for_cloudflare(self, max_wait: float = 5.0):
        """Wait for Cloudflare challenge to complete (with timeout)."""
        start = time.time()
        while time.time() - start < max_wait:
            if "moment" not in self.page.title().lower():
                return True
            time.sleep(0.3)
        return False

    def stop(self):
        """Stop the browser."""
        if self.browser:
            self.browser.close()
        if self._playwright:
            self._playwright.stop()

    def search_sold(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search for sold properties matching a query.

        Args:
            query: Address or search term
            max_results: Maximum results to return

        Returns:
            List of search result dicts with url, address, price
        """
        # Always navigate to the main sold properties search page for each search
        # This ensures a clean state with the search box available
        self.page.goto(self.SEARCH_URL, wait_until="domcontentloaded")
        self._wait_for_cloudflare(max_wait=5.0)

        # Check if we're blocked
        title = self.page.title().lower()
        if "unavailable" in title or "blocked" in title or "error" in title:
            raise Exception(f"Blocked by Daft.ie - {self.page.title()}")

        # Wait a moment for any popups to appear, then remove them aggressively
        time.sleep(0.5)
        self._dismiss_popups(force=True)

        # Use the address search input with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            search_box = self.page.query_selector('#address-search')
            if not search_box:
                # Try waiting for it
                try:
                    self.page.wait_for_selector('#address-search', timeout=3000)
                    search_box = self.page.query_selector('#address-search')
                except:
                    pass

            if search_box:
                try:
                    # Clear and dismiss popups again right before interaction
                    self._dismiss_popups(force=True)
                    search_box.click(force=True, timeout=5000)
                    search_box.fill('')  # Clear existing
                    search_box.type(query, delay=20)  # Reduced from 50ms
                    break  # Success!
                except Exception as e:
                    if attempt < max_retries - 1:
                        # Wait and dismiss popups again before retry
                        time.sleep(0.5)
                        self._dismiss_popups(force=True)
                        continue
                    else:
                        raise Exception(f"Could not interact with search box after {max_retries} attempts: {e}")
            else:
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue
                else:
                    raise Exception("Search box not found after retries")

        # Wait for autocomplete with smart polling
        try:
            self.page.wait_for_selector('[role="listbox"] [role="option"]', timeout=2000)
        except:
            pass  # No autocomplete, continue anyway

        # Find the best matching autocomplete suggestion
        suggestions = self.page.query_selector_all('[role="listbox"] [role="option"]')
        if suggestions:
            query_tokens = set(self._tokenize(query))
            best_suggestion = None
            best_score = 0.0

            for s in suggestions:
                s_text = s.inner_text()
                s_tokens = set(self._tokenize(s_text))
                if s_tokens:
                    intersection = len(query_tokens & s_tokens)
                    union = len(query_tokens | s_tokens)
                    score = intersection / union if union else 0
                    if score > best_score:
                        best_score = score
                        best_suggestion = s

            if best_suggestion:
                # Dismiss popups one more time before clicking suggestion
                self._dismiss_popups(force=True)
                best_suggestion.click(force=True)
                # Wait for results to load
                try:
                    self.page.wait_for_selector('a[href*="/sold/"]', timeout=3000)
                except:
                    pass
        else:
            # Fallback to pressing Enter
            self.page.keyboard.press('Enter')
            try:
                self.page.wait_for_selector('a[href*="/sold/"]', timeout=3000)
            except:
                pass

        results = []

        # Find all sold links
        links = self.page.query_selector_all('a[href*="/sold/"]')

        for link in links[:max_results]:
            try:
                href = link.get_attribute('href')
                if not href or '/sold/' not in href:
                    continue

                # Extract address from URL slug
                slug = href.split('/sold/')[-1].split('/')[0]
                address = slug.replace('-', ' ').title()

                # Try to find parent card for price
                parent = link.query_selector('xpath=ancestor::li')
                price = None
                price_text = ""
                if parent:
                    price_el = parent.query_selector('[data-testid="price"]')
                    if price_el:
                        price_text = price_el.inner_text()
                        price = self._parse_price(price_text)

                results.append({
                    'url': self.BASE_URL + href if href.startswith('/') else href,
                    'address': address,
                    'price': price,
                    'price_text': price_text
                })

            except Exception as e:
                continue

        return results

    def extract_listing(self, url: str) -> SoldListing:
        """
        Extract full details from a sold listing page.

        Args:
            url: Full URL to the listing page

        Returns:
            SoldListing with all available details
        """
        listing = SoldListing(daft_url=url)

        self.page.goto(url, wait_until="domcontentloaded")
        self._wait_for_cloudflare(max_wait=5.0)
        self._dismiss_popups(force=True)

        # Wait for key content to load
        try:
            self.page.wait_for_selector('[data-testid="price"], [data-testid="beds"]', timeout=3000)
        except:
            pass  # Continue even if not found

        # Try to find embedded JSON data (Next.js pages often have this)
        scripts = self.page.query_selector_all('script[type="application/ld+json"]')
        for script in scripts:
            try:
                data = json.loads(script.inner_text())
                if isinstance(data, dict):
                    listing.address = data.get('name', '') or data.get('address', '')
                    if 'geo' in data:
                        listing.latitude = data['geo'].get('latitude')
                        listing.longitude = data['geo'].get('longitude')
            except:
                pass

        # Extract from page elements
        # Address
        addr_el = self.page.query_selector('[data-testid="address"], h1')
        if addr_el:
            listing.address = addr_el.inner_text().strip()

        # Sold Price - look for the price element containing "Sold:"
        price_el = self.page.query_selector('[data-testid="price"]')
        if price_el:
            price_text = price_el.inner_text()
            # Extract sold price (format: "Sold: €343,000")
            sold_match = re.search(r'Sold[:\s]*€?([\d,]+)', price_text, re.I)
            if sold_match:
                listing.sold_price = float(sold_match.group(1).replace(',', ''))

        # Asking Price
        asking_el = self.page.query_selector('[data-testid="sold-asking-price"]')
        if asking_el:
            asking_text = asking_el.inner_text()
            # Extract asking price (format: "Asking: €399,000")
            asking_match = re.search(r'Asking[:\s]*€?([\d,]+)', asking_text, re.I)
            if asking_match:
                listing.asking_price = float(asking_match.group(1).replace(',', ''))

        # Beds - using correct data-testid
        beds_el = self.page.query_selector('[data-testid="beds"]')
        if beds_el:
            beds_text = beds_el.inner_text()
            match = re.search(r'(\d+)', beds_text)
            if match:
                listing.beds = int(match.group(1))

        # Baths - using correct data-testid
        baths_el = self.page.query_selector('[data-testid="baths"]')
        if baths_el:
            baths_text = baths_el.inner_text()
            match = re.search(r'(\d+)', baths_text)
            if match:
                listing.baths = int(match.group(1))

        # Property type - using correct data-testid
        type_el = self.page.query_selector('[data-testid="property-type"]')
        if type_el:
            listing.property_type = type_el.inner_text().strip()

        # Size - check the card-info area for size info
        info_el = self.page.query_selector('[data-testid="card-info"]')
        if info_el:
            info_text = info_el.inner_text()
            size_match = re.search(r'([\d,.]+)\s*m²', info_text)
            if size_match:
                listing.size_sqm = float(size_match.group(1).replace(',', ''))

        # BER Rating - look in page content
        page_content = self.page.content()
        ber_match = re.search(r'BER[:\s]*([A-G][0-9]?)', page_content, re.I)
        if ber_match:
            listing.ber_rating = ber_match.group(1).upper()

        # Description
        desc_el = self.page.query_selector('[data-testid="description"], .PropertyDescription')
        if desc_el:
            listing.description = desc_el.inner_text().strip()[:500]  # Truncate

        # Eircode
        eircode_match = re.search(r'\b([A-Z]\d{2}\s*[A-Z0-9]{4})\b', page_content, re.I)
        if eircode_match:
            listing.eircode = eircode_match.group(1).upper().replace(' ', '')

        # Image count
        images = self.page.query_selector_all('.PropertyImage img, [data-testid="gallery"] img, img[src*="daft"]')
        listing.images_count = len(images)

        # Features list
        features = []
        feature_els = self.page.query_selector_all('.PropertyFeatures li, [data-testid="features"] li')
        for f in feature_els:
            features.append(f.inner_text().strip())
        listing.features = features if features else None

        return listing

    def _parse_price(self, price_text: str) -> Optional[float]:
        """Parse price from text like '€450,000' or '€1.2m'"""
        if not price_text:
            return None

        clean = re.sub(r'[€,\s]', '', price_text)

        if 'm' in clean.lower():
            match = re.search(r'([\d.]+)m', clean.lower())
            if match:
                return float(match.group(1)) * 1_000_000

        match = re.search(r'[\d.]+', clean)
        if match:
            try:
                return float(match.group(0))
            except:
                pass

        return None

    def match_address(self, ppr_address: str, ppr_county: str = "") -> Optional[SoldListing]:
        """
        Find a Daft sold listing matching a PPR address.

        Args:
            ppr_address: Address from PPR
            ppr_county: County from PPR (optional, helps narrow search)

        Returns:
            Best matching SoldListing or None
        """
        # Just use the address for search (county can make it too specific)
        search_query = ppr_address

        # Search for matches - get more results for better matching
        results = self.search_sold(search_query, max_results=20)

        if not results:
            return None

        # Score each result
        ppr_tokens = set(self._tokenize(ppr_address))

        best_match = None
        best_score = 0.0

        for result in results:
            result_tokens = set(self._tokenize(result['address']))

            if not result_tokens:
                continue

            # Jaccard similarity
            intersection = len(ppr_tokens & result_tokens)
            union = len(ppr_tokens | result_tokens)
            score = intersection / union if union > 0 else 0

            # Lower threshold to 0.3 for more matches
            if score > best_score and score >= 0.3:
                best_score = score
                best_match = result

        if best_match:
            # Get full listing details
            listing = self.extract_listing(best_match['url'])
            listing.ppr_address = ppr_address
            listing.match_confidence = best_score
            return listing

        return None

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize address into words."""
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        return [t for t in tokens if len(t) > 1]


class Checkpoint:
    """Manages checkpointing for resume support."""

    def __init__(self, output_path: Path):
        self.checkpoint_path = output_path.with_suffix('.checkpoint.json')
        self.results_path = output_path.with_suffix('.partial.csv')
        self.lock = threading.Lock()
        self.processed_addresses: set = set()
        self.results: List[Dict] = []

    def load(self) -> bool:
        """Load checkpoint if it exists. Returns True if loaded."""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r') as f:
                    data = json.load(f)
                    self.processed_addresses = set(data.get('processed', []))

                # Load partial results
                if self.results_path.exists():
                    with open(self.results_path, 'r', encoding='utf-8', newline='') as f:
                        reader = csv.DictReader(f)
                        self.results = list(reader)

                print(f"Resuming from checkpoint: {len(self.processed_addresses)} already processed")
                return True
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
        return False

    def save(self):
        """Save current progress."""
        with self.lock:
            # Save checkpoint
            with open(self.checkpoint_path, 'w') as f:
                json.dump({'processed': list(self.processed_addresses)}, f)

            # Save partial results
            if self.results:
                fieldnames = list(self.results[0].keys())
                with open(self.results_path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.results)

    def add_result(self, address: str, result: Dict):
        """Thread-safe add result."""
        with self.lock:
            self.processed_addresses.add(address)
            self.results.append(result)

        # Auto-save every 10 results
        if len(self.results) % 10 == 0:
            self.save()

    def is_processed(self, address: str) -> bool:
        """Check if address was already processed."""
        return address in self.processed_addresses

    def cleanup(self):
        """Remove checkpoint files after successful completion."""
        try:
            if self.checkpoint_path.exists():
                self.checkpoint_path.unlink()
            if self.results_path.exists():
                self.results_path.unlink()
        except:
            pass


class BrowserWorkerPool:
    """
    Pool of browser workers that reuse browser instances.
    Each worker has its own browser and processes tasks from a shared queue.
    """

    def __init__(self, num_workers: int, headless: bool = True):
        self.num_workers = num_workers
        self.headless = headless
        self.task_queue: Queue = Queue()
        self.result_queue: Queue = Queue()
        self.workers: List[threading.Thread] = []
        self.stop_event = threading.Event()
        self.error_count = 0
        self.error_lock = threading.Lock()
        self.base_delay = 1.0  # Base delay between requests (conservative to avoid blocks)
        self.current_delay = 1.0  # Adaptive delay

    def _adaptive_delay(self, had_error: bool):
        """Adjust delay based on error rate."""
        with self.error_lock:
            if had_error:
                self.error_count += 1
                # Increase delay on errors (back off)
                self.current_delay = min(self.current_delay * 1.5, 5.0)
            else:
                # Gradually decrease delay on success
                self.current_delay = max(self.current_delay * 0.95, self.base_delay)
        return self.current_delay

    def _worker_loop(self, worker_id: int):
        """Main loop for a worker thread - reuses single browser instance."""
        scraper = None
        try:
            # Create ONE browser instance for this worker
            scraper = DaftSoldScraper(headless=self.headless, worker_id=worker_id)
            scraper.start()

            while not self.stop_event.is_set():
                try:
                    # Get task with timeout so we can check stop_event
                    task = self.task_queue.get(timeout=1.0)
                except Empty:
                    continue

                if task is None:  # Poison pill
                    break

                idx, row, total = task
                address = row.get("address", "")
                county = row.get("county", "")
                had_error = False

                try:
                    listing = scraper.match_address(address, county)

                    if listing:
                        result = listing.to_dict()
                        result.update(row)
                        result["match_status"] = "matched"
                        status = f"MATCHED ({listing.match_confidence:.2f})"
                    else:
                        result = dict(row)
                        result["match_status"] = "no_match"
                        result["daft_url"] = ""
                        status = "NO MATCH"

                except Exception as e:
                    result = dict(row)
                    result["match_status"] = f"error: {str(e)[:100]}"
                    result["daft_url"] = ""
                    status = f"ERROR"
                    had_error = True

                    # On error, try to recover the browser
                    try:
                        scraper.stop()
                        time.sleep(1)
                        scraper = DaftSoldScraper(headless=self.headless, worker_id=worker_id)
                        scraper.start()
                    except:
                        pass

                self.result_queue.put({
                    "address": address,
                    "result": result,
                    "status": status,
                    "idx": idx
                })
                self.task_queue.task_done()

                # Adaptive delay - shorter on success, longer on error
                delay = self._adaptive_delay(had_error)
                if not had_error:
                    time.sleep(delay)

        finally:
            if scraper:
                try:
                    scraper.stop()
                except:
                    pass

    def start(self):
        """Start all worker threads."""
        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            t.start()
            self.workers.append(t)
            # Stagger worker startup to avoid thundering herd
            time.sleep(0.5)

    def submit(self, idx: int, row: Dict, total: int):
        """Submit a task to the pool."""
        self.task_queue.put((idx, row, total))

    def get_result(self, timeout: float = None) -> Optional[Dict]:
        """Get a result from completed tasks."""
        try:
            return self.result_queue.get(timeout=timeout)
        except Empty:
            return None

    def shutdown(self):
        """Signal workers to stop and wait for them."""
        self.stop_event.set()
        # Send poison pills
        for _ in self.workers:
            self.task_queue.put(None)
        # Wait for workers to finish
        for t in self.workers:
            t.join(timeout=10)


def process_single_address(args) -> Dict:
    """Process a single address - for use with ThreadPoolExecutor."""
    idx, row, total, headless, worker_id = args
    address = row.get("address", "")
    county = row.get("county", "")

    try:
        with DaftSoldScraper(headless=headless, worker_id=worker_id) as scraper:
            listing = scraper.match_address(address, county)

            if listing:
                result = listing.to_dict()
                result.update(row)
                result["match_status"] = "matched"
                return {"address": address, "result": result, "status": "matched", "confidence": listing.match_confidence}
            else:
                result = dict(row)
                result["match_status"] = "no_match"
                result["daft_url"] = ""
                return {"address": address, "result": result, "status": "no_match"}

    except Exception as e:
        result = dict(row)
        result["match_status"] = f"error: {e}"
        result["daft_url"] = ""
        return {"address": address, "result": result, "status": "error", "error": str(e)}


def normalize_ppr_row(row: Dict[str, str]) -> Dict[str, str]:
    """Convert PPR format to standard format."""
    normalized = {}

    col_map = {
        "Date of Sale (dd/mm/yyyy)": "sale_date",
        "Address": "address",
        "County": "county",
        "Eircode": "eircode",
    }

    for ppr_col, std_col in col_map.items():
        if ppr_col in row:
            normalized[std_col] = row[ppr_col]

    # Handle price
    for col in row.keys():
        if "Price" in col:
            price_str = row[col]
            clean = re.sub(r'[€,\s]', '', price_str)
            match = re.search(r'[\d.]+', clean)
            if match:
                normalized["selling_price"] = match.group(0)
            break

    # Parse date
    if "sale_date" in normalized:
        date_str = normalized["sale_date"]
        if "/" in date_str:
            parts = date_str.split("/")
            if len(parts) == 3:
                d, m, y = parts
                normalized["sale_date"] = f"{y}-{m.zfill(2)}-{d.zfill(2)}"

    return normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--match", action="store_true",
                       help="Match PPR addresses to Daft sold listings")
    parser.add_argument("--extract", action="store_true",
                       help="Extract details from matched listings")
    parser.add_argument("--in", dest="input_csv", required=True,
                       help="Input CSV file")
    parser.add_argument("--out", required=True,
                       help="Output CSV path")
    parser.add_argument("--ppr_format", action="store_true",
                       help="Input is PPR format")
    parser.add_argument("--headless", action="store_true", default=False,
                       help="Run browser in headless mode (may not bypass anti-bot)")
    parser.add_argument("--limit", type=int, default=0,
                       help="Limit number of records to process (0=all)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint if available")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of parallel browser workers (1-5, default=1)")

    args = parser.parse_args()

    if not HAS_PLAYWRIGHT:
        print("Error: Playwright is required. Install with:")
        print("  pip install playwright")
        print("  playwright install chromium")
        sys.exit(1)

    # Validate workers (keep conservative to avoid rate limiting)
    args.workers = max(1, min(5, args.workers))

    input_csv = Path(args.input_csv)
    output_csv = Path(args.out)

    # Read input
    rows = []
    for encoding in ["utf-8-sig", "utf-8", "latin-1", "cp1252"]:
        try:
            with input_csv.open("r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            break
        except UnicodeDecodeError:
            continue

    if not rows:
        print(f"Error: Could not read {input_csv}")
        sys.exit(1)

    if args.ppr_format:
        rows = [normalize_ppr_row(row) for row in rows]

    if args.limit > 0:
        rows = rows[:args.limit]

    # Setup checkpoint
    checkpoint = Checkpoint(output_csv)
    if args.resume:
        checkpoint.load()

    # Filter out already processed
    remaining_rows = [(i, row) for i, row in enumerate(rows)
                      if not checkpoint.is_processed(row.get("address", ""))]

    total = len(rows)
    already_done = total - len(remaining_rows)

    print(f"\nTotal records: {total}")
    if already_done > 0:
        print(f"Already processed: {already_done}")
    print(f"Remaining: {len(remaining_rows)}")
    print(f"Workers: {args.workers}")
    print()

    if not remaining_rows:
        print("All records already processed!")
        results = checkpoint.results
    elif args.workers == 1:
        # Sequential processing (original behavior, but faster)
        results = list(checkpoint.results)  # Start with existing results

        with DaftSoldScraper(headless=args.headless) as scraper:
            for idx, row in remaining_rows:
                address = row.get("address", "")
                county = row.get("county", "")

                print(f"[{idx+1}/{total}] Searching: {address[:50]}...", end=" ", flush=True)

                try:
                    listing = scraper.match_address(address, county)

                    if listing:
                        result = listing.to_dict()
                        result.update(row)
                        result["match_status"] = "matched"
                        checkpoint.add_result(address, result)
                        results.append(result)
                        print(f"MATCHED (confidence: {listing.match_confidence:.2f})")
                    else:
                        result = dict(row)
                        result["match_status"] = "no_match"
                        result["daft_url"] = ""
                        checkpoint.add_result(address, result)
                        results.append(result)
                        print("NO MATCH")

                except Exception as e:
                    result = dict(row)
                    result["match_status"] = f"error: {e}"
                    result["daft_url"] = ""
                    checkpoint.add_result(address, result)
                    results.append(result)
                    print(f"ERROR: {e}")
                    # Back off on errors
                    time.sleep(2.0)
    else:
        # Parallel processing with browser pooling (much faster!)
        results = list(checkpoint.results)
        completed = already_done

        print(f"Starting {args.workers} browser workers (reusing browser instances)...")

        # Create and start the worker pool
        pool = BrowserWorkerPool(num_workers=args.workers, headless=args.headless)
        pool.start()

        # Submit all tasks
        for idx, row in remaining_rows:
            pool.submit(idx, row, total)

        # Collect results
        pending = len(remaining_rows)
        try:
            while pending > 0:
                result_data = pool.get_result(timeout=120)  # 2 min timeout per result
                if result_data is None:
                    print("Warning: Timeout waiting for result, continuing...")
                    continue

                address = result_data["address"]
                result = result_data["result"]
                status = result_data["status"]

                checkpoint.add_result(address, result)
                results.append(result)

                completed += 1
                pending -= 1
                print(f"[{completed}/{total}] {address[:40]}... {status}")

        except KeyboardInterrupt:
            print("\nInterrupted! Saving progress...")
        finally:
            pool.shutdown()
            checkpoint.save()

    # Write final output
    if results:
        fieldnames = list(results[0].keys())
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    # Cleanup checkpoint on success
    checkpoint.cleanup()

    # Summary
    matched = sum(1 for r in results if r.get("match_status") == "matched")
    matched_results = [r for r in results if r.get("match_status") == "matched"]

    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"Total processed: {len(results)}")
    print(f"Matched: {matched} ({matched/len(results)*100:.1f}%)")

    if matched_results:
        # Data quality stats
        with_sold_price = sum(1 for r in matched_results if r.get("sold_price"))
        with_asking_price = sum(1 for r in matched_results if r.get("asking_price"))
        with_beds = sum(1 for r in matched_results if r.get("beds"))
        with_baths = sum(1 for r in matched_results if r.get("baths"))
        with_ber = sum(1 for r in matched_results if r.get("ber_rating"))
        avg_confidence = sum(r.get("match_confidence", 0) for r in matched_results) / len(matched_results)

        print(f"\nData Quality:")
        print(f"  With sold price: {with_sold_price}/{matched}")
        print(f"  With asking price: {with_asking_price}/{matched}")
        print(f"  With beds: {with_beds}/{matched}")
        print(f"  With baths: {with_baths}/{matched}")
        print(f"  With BER: {with_ber}/{matched}")
        print(f"  Avg match confidence: {avg_confidence:.2f}")

    print(f"\nOutput: {output_csv}")


if __name__ == "__main__":
    main()
