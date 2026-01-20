#!/usr/bin/env python3
"""
MyHome.ie Brochure Scraper
===========================
Scrapes full property data from MyHome.ie brochure pages.

KEY INSIGHT: MyHome.ie keeps full brochure pages even for properties that have
been sold - sometimes for years. This means we can get rich data (beds, baths,
size, BER, description, images) for historical sales.

This scraper:
1. Takes addresses from the Property Price Register (PPR)
2. Searches MyHome.ie to find matching brochure pages
3. Extracts full property details from brochure pages

Usage:
    # Search for specific addresses (from file, one per line)
    python scrape_myhome_brochures.py --addresses addresses.txt --out enriched.csv

    # Scrape all brochure links from for-sale pages (bulk mode)
    python scrape_myhome_brochures.py --bulk --out all_brochures.csv --limit 100

    # Visible browser for debugging
    python scrape_myhome_brochures.py --bulk --out test.csv --no-headless --limit 10

Author: House Price Predictor Project
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import signal
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, quote_plus

try:
    from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext, TimeoutError as PlaywrightTimeout
except ImportError:
    print("ERROR: playwright not installed. Run: pip install playwright && playwright install firefox")
    sys.exit(1)


# ============================================================================
# CONSTANTS
# ============================================================================

BASE_URL = "https://www.myhome.ie"
SEARCH_URL = "https://www.myhome.ie/residential/search"
FOR_SALE_URL = "https://www.myhome.ie/residential/ireland/property-for-sale"

# Scraping settings
DEFAULT_CHECKPOINT_INTERVAL = 25
DEFAULT_MONITOR_INTERVAL = 10
MIN_DELAY_SECONDS = 2.0
MAX_DELAY_SECONDS = 5.0
PAGE_LOAD_TIMEOUT_MS = 45000

# CSV fields
CSV_FIELDS = [
    "myhome_id", "url", "address_full", "county", "area", "eircode",
    "sold_price", "asking_price", "sale_status",
    "property_type", "bedrooms", "bathrooms", "size_sqm", "size_sqft",
    "ber_rating", "ber_number",
    "latitude", "longitude",
    "agent_name", "agent_phone",
    "description", "features_list", "num_images",
    "scraped_at", "source_type", "ppr_address",
]

# Irish counties
COUNTIES = [
    "carlow", "cavan", "clare", "cork", "donegal", "dublin", "galway",
    "kerry", "kildare", "kilkenny", "laois", "leitrim", "limerick",
    "longford", "louth", "mayo", "meath", "monaghan", "offaly",
    "roscommon", "sligo", "tipperary", "waterford", "westmeath", "wexford", "wicklow"
]

# Regex patterns
EIRCODE_PATTERN = re.compile(r'\b([A-Z]\d{2}\s?[A-Z0-9]{4})\b', re.IGNORECASE)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PropertyListing:
    """Property listing data."""
    myhome_id: Optional[str] = None
    url: Optional[str] = None
    address_full: Optional[str] = None
    county: Optional[str] = None
    area: Optional[str] = None
    eircode: Optional[str] = None

    sold_price: Optional[int] = None
    asking_price: Optional[int] = None
    sale_status: Optional[str] = None  # 'For Sale', 'Sale Agreed', 'Sold'

    property_type: Optional[str] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    size_sqm: Optional[float] = None
    size_sqft: Optional[float] = None

    ber_rating: Optional[str] = None
    ber_number: Optional[str] = None

    latitude: Optional[float] = None
    longitude: Optional[float] = None

    agent_name: Optional[str] = None
    agent_phone: Optional[str] = None

    description: Optional[str] = None
    features_list: Optional[str] = None
    num_images: Optional[int] = None

    scraped_at: Optional[str] = None
    source_type: Optional[str] = None  # 'brochure'
    ppr_address: Optional[str] = None  # Original PPR address used for search

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CheckpointData:
    """Checkpoint for resume."""
    last_page: int = 0
    processed_count: int = 0
    processed_urls: set[str] = field(default_factory=set)
    started_at: str = ""
    last_saved_at: str = ""

    def to_dict(self) -> dict:
        return {
            "last_page": self.last_page,
            "processed_count": self.processed_count,
            "processed_urls": list(self.processed_urls),
            "started_at": self.started_at,
            "last_saved_at": self.last_saved_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CheckpointData:
        return cls(
            last_page=data.get("last_page", 0),
            processed_count=data.get("processed_count", 0),
            processed_urls=set(data.get("processed_urls", [])),
            started_at=data.get("started_at", ""),
            last_saved_at=data.get("last_saved_at", ""),
        )


@dataclass
class ScraperStats:
    """Scraping statistics."""
    total_found: int = 0
    processed: int = 0
    successful: int = 0
    failed: int = 0
    field_counts: dict[str, int] = field(default_factory=dict)
    start_time: float = 0.0

    def success_rate(self) -> float:
        if self.processed == 0:
            return 0.0
        return (self.successful / self.processed) * 100

    def speed(self) -> float:
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0.0
        return self.processed / elapsed


# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================

class CheckpointManager:
    """Manages checkpoint for resume."""

    def __init__(self, output_file: str):
        self.checkpoint_file = Path(output_file).with_suffix('.checkpoint.json')
        self.data = CheckpointData()
        self.logger = logging.getLogger(__name__)

    def exists(self) -> bool:
        return self.checkpoint_file.exists()

    def load(self) -> bool:
        if not self.exists():
            return False
        try:
            with open(self.checkpoint_file, 'r') as f:
                self.data = CheckpointData.from_dict(json.load(f))
            self.logger.info(f"Loaded checkpoint: {self.data.processed_count} processed")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False

    def save(self) -> None:
        self.data.last_saved_at = datetime.now().isoformat()
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.data.to_dict(), f)
        except IOError as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

    def update(self, page: int, url: str, count: int) -> None:
        self.data.last_page = page
        self.data.processed_count = count
        self.data.processed_urls.add(url)

    def is_processed(self, url: str) -> bool:
        return url in self.data.processed_urls

    def clear(self) -> None:
        if self.exists():
            self.checkpoint_file.unlink()
        self.data = CheckpointData()

    def initialize(self) -> None:
        self.data = CheckpointData(started_at=datetime.now().isoformat())


# ============================================================================
# PROGRESS MONITOR
# ============================================================================

class ProgressMonitor:
    """Progress monitoring."""

    TRACKED_FIELDS = ["bedrooms", "bathrooms", "size_sqm", "ber_rating", "eircode", "description"]

    def __init__(self, update_interval: int = DEFAULT_MONITOR_INTERVAL):
        self.stats = ScraperStats()
        self.stats.start_time = time.time()
        self.stats.field_counts = {f: 0 for f in self.TRACKED_FIELDS}
        self.update_interval = update_interval
        self.last_address = ""

    def update(self, listing: Optional[PropertyListing], success: bool) -> None:
        self.stats.processed += 1
        if success and listing:
            self.stats.successful += 1
            self.last_address = listing.address_full or "Unknown"
            d = listing.to_dict()
            for field in self.TRACKED_FIELDS:
                if d.get(field) is not None and d.get(field) != "" and d.get(field) != 0:
                    self.stats.field_counts[field] += 1
        else:
            self.stats.failed += 1

    def set_total(self, total: int) -> None:
        self.stats.total_found = total

    def should_display(self) -> bool:
        return self.stats.processed % self.update_interval == 0

    def display(self) -> None:
        stats = self.stats
        progress = (stats.processed / max(stats.total_found, 1)) * 100
        eta = (stats.total_found - stats.processed) / max(stats.speed(), 0.01)

        print("\n" + "=" * 70)
        print("MYHOME.IE BROCHURE SCRAPER")
        print("=" * 70)
        print(f"Progress: {stats.processed:,} / {stats.total_found:,} ({progress:.1f}%)")
        print(f"Speed: {stats.speed():.2f}/sec | ETA: {timedelta(seconds=int(eta))}")
        print(f"Success: {stats.successful:,} ({stats.success_rate():.1f}%) | Failed: {stats.failed:,}")
        print("\nData Coverage:")
        for field in self.TRACKED_FIELDS:
            count = stats.field_counts.get(field, 0)
            pct = (count / max(stats.successful, 1)) * 100
            print(f"  {field:12s}: {count:>5}/{stats.successful:<5} ({pct:5.1f}%)")
        print(f"\nLast: {self.last_address[:50]}")
        print("=" * 70)

    def final_summary(self) -> str:
        stats = self.stats
        elapsed = time.time() - stats.start_time
        return f"""
{'='*70}
SCRAPING COMPLETE
{'='*70}
Duration: {timedelta(seconds=int(elapsed))}
Processed: {stats.processed:,}
Successful: {stats.successful:,} ({stats.success_rate():.1f}%)
Failed: {stats.failed:,}
Speed: {stats.speed():.2f}/sec
{'='*70}
"""


# ============================================================================
# MYHOME BROCHURE SCRAPER
# ============================================================================

class MyHomeBrochureScraper:
    """Scraper for MyHome.ie brochure pages."""

    def __init__(
        self,
        output_file: str,
        addresses_file: Optional[str] = None,
        bulk_mode: bool = False,
        county: Optional[str] = None,
        limit: Optional[int] = None,
        headless: bool = True,
        checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    ):
        self.output_file = output_file
        self.addresses_file = addresses_file
        self.bulk_mode = bulk_mode
        self.county = county.lower() if county else None
        self.limit = limit
        self.headless = headless
        self.checkpoint_interval = checkpoint_interval

        self.checkpoint = CheckpointManager(output_file)
        self.monitor = ProgressMonitor()

        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

        self.current_delay = MIN_DELAY_SECONDS
        self.shutdown_requested = False
        self.listings_written = 0

        self.logger = logging.getLogger(__name__)

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame) -> None:
        self.logger.info("\nShutdown requested...")
        self.shutdown_requested = True

    def _init_browser(self) -> None:
        """Initialize browser."""
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.firefox.launch(headless=self.headless)
        self.context = self.browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            locale="en-IE",
        )
        self.page = self.context.new_page()
        self.page.set_default_timeout(PAGE_LOAD_TIMEOUT_MS)

    def _close_browser(self) -> None:
        """Clean up browser."""
        if self.page:
            self.page.close()
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

    def _dismiss_cookies(self) -> None:
        """Dismiss cookie popup."""
        selectors = ['#onetrust-accept-btn-handler', 'button:has-text("Accept")', 'button:has-text("I Accept")']
        for selector in selectors:
            try:
                btn = self.page.locator(selector).first
                if btn.is_visible(timeout=2000):
                    btn.click()
                    time.sleep(0.5)
                    return
            except:
                continue

    def _extract_brochure_urls_from_list_page(self) -> list[str]:
        """Extract brochure URLs from a listing page."""
        urls = []
        try:
            links = self.page.locator('a[href*="/residential/brochure/"]').all()

            for link in links:
                try:
                    href = link.get_attribute("href")
                    if href:
                        full_url = href if href.startswith("http") else urljoin(BASE_URL, href)
                        # Filter: must have numeric ID in URL
                        if re.search(r'/\d{5,}', full_url):
                            urls.append(full_url.rstrip('/'))
                except:
                    continue

            # Deduplicate
            unique = list(dict.fromkeys(urls))
            self.logger.debug(f"Extracted {len(unique)} brochure URLs")
            return unique
        except Exception as e:
            self.logger.error(f"Error extracting URLs: {e}")
            return []

    def _search_for_address(self, address: str) -> Optional[str]:
        """Search MyHome.ie for an address and return best matching brochure URL."""
        try:
            # Use MyHome.ie search
            search_url = f"{BASE_URL}/residential/search?q={quote_plus(address)}"
            self.page.goto(search_url, wait_until="domcontentloaded")
            time.sleep(2)

            # Look for brochure links in results
            brochure_links = self.page.locator('a[href*="/residential/brochure/"]').all()

            if brochure_links:
                # Return first match
                href = brochure_links[0].get_attribute("href")
                if href:
                    return href if href.startswith("http") else urljoin(BASE_URL, href)

            return None
        except Exception as e:
            self.logger.error(f"Error searching for {address}: {e}")
            return None

    def _extract_property_data(self, ppr_address: Optional[str] = None) -> PropertyListing:
        """Extract property data from current brochure page."""
        listing = PropertyListing()
        listing.scraped_at = datetime.now().isoformat()
        listing.url = self.page.url
        listing.source_type = "brochure"
        listing.ppr_address = ppr_address

        # Extract ID from URL (e.g., /4706275)
        url_match = re.search(r'/(\d+)/?$', self.page.url)
        if url_match:
            listing.myhome_id = url_match.group(1)

        # Get page text for pattern matching
        try:
            body_text = self.page.locator('body').text_content() or ""
        except:
            body_text = ""

        # Address from title (format: "Address - Agent - ID - MyHome.ie")
        try:
            title = self.page.title()
            if " - " in title:
                listing.address_full = title.split(" - ")[0].strip()
                parts = title.split(" - ")
                if len(parts) >= 2:
                    listing.agent_name = parts[1].strip()
        except:
            pass

        # Fallback: Address from h1
        if not listing.address_full:
            try:
                h1 = self.page.locator('h1').first
                if h1.count() > 0:
                    listing.address_full = h1.text_content().strip()
            except:
                pass

        # Sale status
        sale_indicators = [
            ('sold', 'Sold'),
            ('sale agreed', 'Sale Agreed'),
            ('under offer', 'Under Offer'),
        ]
        body_lower = body_text.lower()
        for indicator, status in sale_indicators:
            if indicator in body_lower:
                listing.sale_status = status
                break
        if not listing.sale_status:
            listing.sale_status = "For Sale"

        # Eircode
        eircode_match = EIRCODE_PATTERN.search(body_text)
        if eircode_match:
            listing.eircode = eircode_match.group(1).upper().replace(" ", "")

        # Beds - must be 1-9 followed by Bed/bedroom (avoid matching 0 or IDs)
        beds_match = re.search(r'\b([1-9]\d?)\s*(?:Bed|bedroom)s?\b', body_text, re.IGNORECASE)
        if beds_match:
            listing.bedrooms = int(beds_match.group(1))

        # Baths
        baths_match = re.search(r'\b([1-9]\d?)\s*(?:Bath|bathroom)s?\b', body_text, re.IGNORECASE)
        if baths_match:
            listing.bathrooms = int(baths_match.group(1))

        # Size in sqm
        sqm_match = re.search(r'([\d,.]+)\s*(?:sq\.?\s*m|m2|m\u00b2|sqm)', body_text, re.IGNORECASE)
        if sqm_match:
            try:
                listing.size_sqm = float(sqm_match.group(1).replace(',', ''))
            except:
                pass

        # Size in sqft
        sqft_match = re.search(r'([\d,.]+)\s*(?:sq\.?\s*ft|sqft|ft\u00b2)', body_text, re.IGNORECASE)
        if sqft_match:
            try:
                listing.size_sqft = float(sqft_match.group(1).replace(',', ''))
            except:
                pass

        # BER rating (A1, B2, C1, D1, E1, F, G)
        ber_match = re.search(r'BER[:\s]*([A-G][1-3]?)\b', body_text, re.IGNORECASE)
        if ber_match:
            listing.ber_rating = ber_match.group(1).upper()

        # BER number
        ber_num_match = re.search(r'BER\s*(?:No\.?|Number)?[:\s]*(\d{9,12})', body_text, re.IGNORECASE)
        if ber_num_match:
            listing.ber_number = ber_num_match.group(1)

        # Price - look for EUR followed by digits, excluding IDs
        # Try specific price element first
        try:
            price_el = self.page.locator('[class*="price" i], [data-testid*="price"]').first
            if price_el.count() > 0:
                price_text = price_el.text_content() or ""
                # Look for currency patterns
                price_match = re.search(r'[EUR\u20ac]\s*([\d,]+)', price_text)
                if price_match:
                    price_val = int(price_match.group(1).replace(',', ''))
                    # Sanity check: real prices are typically 50k-10M
                    if 50000 <= price_val <= 10000000:
                        listing.asking_price = price_val
        except:
            pass

        # Property type
        type_patterns = [
            (r'\bApartment\b', 'Apartment'),
            (r'\bFlat\b', 'Apartment'),
            (r'\bDetached\b', 'Detached'),
            (r'\bSemi-Detached\b', 'Semi-Detached'),
            (r'\bSemi Detached\b', 'Semi-Detached'),
            (r'\bEnd of Terrace\b', 'End of Terrace'),
            (r'\bTerrace\b', 'Terrace'),
            (r'\bBungalow\b', 'Bungalow'),
            (r'\bDuplex\b', 'Duplex'),
            (r'\bStudio\b', 'Studio'),
            (r'\bTownhouse\b', 'Townhouse'),
        ]
        for pattern, ptype in type_patterns:
            if re.search(pattern, body_text, re.IGNORECASE):
                listing.property_type = ptype
                break

        # County from address
        if listing.address_full:
            addr_lower = listing.address_full.lower()
            for county in COUNTIES:
                if county in addr_lower:
                    listing.county = county.title()
                    break
            # Special case for Dublin areas
            if not listing.county and 'dublin' in addr_lower:
                listing.county = 'Dublin'

        # Description
        try:
            desc_selectors = [
                '[class*="description" i]',
                '.property-description',
                '#description',
            ]
            for selector in desc_selectors:
                desc_el = self.page.locator(selector).first
                if desc_el.count() > 0:
                    desc_text = desc_el.text_content()
                    if desc_text and len(desc_text) > 50:
                        # Clean and truncate
                        listing.description = desc_text.strip()[:1000]
                        break
        except:
            pass

        # Image count
        try:
            images = self.page.locator('img[src*="property"], img[src*="brochure"], img[src*="myhome"]').all()
            listing.num_images = len(images)
        except:
            pass

        return listing

    def scrape_brochure(self, url: str, ppr_address: Optional[str] = None) -> Optional[PropertyListing]:
        """Scrape a single brochure page."""
        try:
            self.page.goto(url, wait_until="domcontentloaded", timeout=PAGE_LOAD_TIMEOUT_MS)
            time.sleep(1.5)

            # Check for error pages
            title = self.page.title().lower()
            if "404" in title or "not found" in title or "error" in title:
                return None

            return self._extract_property_data(ppr_address)
        except PlaywrightTimeout:
            self.logger.warning(f"Timeout: {url}")
            return None
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {e}")
            return None

    def run_bulk(self, resume: bool = False, fresh: bool = False) -> None:
        """Bulk mode: scrape all brochures from for-sale pages."""
        if fresh:
            self.checkpoint.clear()
        elif resume and self.checkpoint.exists():
            self.checkpoint.load()
        else:
            self.checkpoint.initialize()

        self.logger.info("Starting browser...")
        self._init_browser()

        try:
            # Build URL
            if self.county:
                base_url = f"{BASE_URL}/residential/{self.county}/property-for-sale"
            else:
                base_url = FOR_SALE_URL

            start_page = max(1, self.checkpoint.data.last_page)
            url = f"{base_url}?page={start_page}" if start_page > 1 else base_url

            self.logger.info(f"Loading: {url}")
            self.page.goto(url, wait_until="domcontentloaded")
            self._dismiss_cookies()
            time.sleep(2)

            # Estimate total
            total_estimate = self.limit or 5000
            self.monitor.set_total(total_estimate)

            # Init CSV
            file_exists = Path(self.output_file).exists() and resume
            mode = 'a' if file_exists else 'w'
            csv_file = open(self.output_file, mode, newline='', encoding='utf-8-sig')
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            if not file_exists:
                writer.writeheader()

            try:
                current_page = start_page
                processed = self.checkpoint.data.processed_count

                while not self.shutdown_requested:
                    # Get brochure URLs
                    brochure_urls = self._extract_brochure_urls_from_list_page()

                    if not brochure_urls:
                        self.logger.info(f"No brochures on page {current_page}")
                        current_page += 1
                        if current_page > 500:
                            break
                        url = f"{base_url}?page={current_page}"
                        self.page.goto(url, wait_until="domcontentloaded")
                        time.sleep(1)
                        continue

                    self.logger.info(f"Page {current_page}: {len(brochure_urls)} brochures")

                    for brochure_url in brochure_urls:
                        if self.shutdown_requested:
                            break

                        if self.limit and processed >= self.limit:
                            self.logger.info(f"Reached limit: {self.limit}")
                            self.shutdown_requested = True
                            break

                        if self.checkpoint.is_processed(brochure_url):
                            continue

                        listing = self.scrape_brochure(brochure_url)

                        if listing and listing.address_full:
                            writer.writerow(listing.to_dict())
                            self.listings_written += 1
                            self.monitor.update(listing, success=True)
                        else:
                            self.monitor.update(None, success=False)

                        processed += 1
                        self.checkpoint.update(current_page, brochure_url, processed)

                        if processed % self.checkpoint_interval == 0:
                            self.checkpoint.save()
                            csv_file.flush()

                        if self.monitor.should_display():
                            self.monitor.display()

                        time.sleep(self.current_delay)

                    # Next page
                    current_page += 1
                    url = f"{base_url}?page={current_page}"
                    try:
                        self.page.goto(url, wait_until="domcontentloaded")
                        time.sleep(1)
                    except:
                        break

            finally:
                self.checkpoint.save()
                csv_file.close()
                print(self.monitor.final_summary())
                print(f"Output: {self.output_file}")
                print(f"Listings written: {self.listings_written}")

        finally:
            self._close_browser()

    def run_address_search(self, resume: bool = False, fresh: bool = False) -> None:
        """Address mode: search for addresses and scrape matching brochures."""
        if not self.addresses_file:
            self.logger.error("No addresses file provided")
            return

        # Load addresses
        addresses_path = Path(self.addresses_file)
        if not addresses_path.exists():
            self.logger.error(f"Addresses file not found: {self.addresses_file}")
            return

        with open(addresses_path, 'r', encoding='utf-8') as f:
            addresses = [line.strip() for line in f if line.strip()]

        if not addresses:
            self.logger.error("No addresses found in file")
            return

        self.logger.info(f"Loaded {len(addresses)} addresses")

        if fresh:
            self.checkpoint.clear()
        elif resume and self.checkpoint.exists():
            self.checkpoint.load()
        else:
            self.checkpoint.initialize()

        self.logger.info("Starting browser...")
        self._init_browser()

        try:
            self._dismiss_cookies()

            total = min(len(addresses), self.limit) if self.limit else len(addresses)
            self.monitor.set_total(total)

            # Init CSV
            file_exists = Path(self.output_file).exists() and resume
            mode = 'a' if file_exists else 'w'
            csv_file = open(self.output_file, mode, newline='', encoding='utf-8-sig')
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            if not file_exists:
                writer.writeheader()

            try:
                processed = self.checkpoint.data.processed_count

                for i, address in enumerate(addresses):
                    if self.shutdown_requested:
                        break

                    if self.limit and processed >= self.limit:
                        break

                    # Use address as "URL" for checkpoint
                    if self.checkpoint.is_processed(address):
                        continue

                    self.logger.info(f"Searching for: {address}")

                    # Search for brochure URL
                    brochure_url = self._search_for_address(address)

                    if brochure_url:
                        listing = self.scrape_brochure(brochure_url, ppr_address=address)

                        if listing and listing.address_full:
                            writer.writerow(listing.to_dict())
                            self.listings_written += 1
                            self.monitor.update(listing, success=True)
                        else:
                            self.monitor.update(None, success=False)
                    else:
                        self.logger.info(f"  No match found")
                        self.monitor.update(None, success=False)

                    processed += 1
                    self.checkpoint.update(0, address, processed)

                    if processed % self.checkpoint_interval == 0:
                        self.checkpoint.save()
                        csv_file.flush()

                    if self.monitor.should_display():
                        self.monitor.display()

                    time.sleep(self.current_delay)

            finally:
                self.checkpoint.save()
                csv_file.close()
                print(self.monitor.final_summary())
                print(f"Output: {self.output_file}")
                print(f"Listings written: {self.listings_written}")

        finally:
            self._close_browser()


# ============================================================================
# CLI
# ============================================================================

def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape property data from MyHome.ie brochure pages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--out", "-o", required=True, help="Output CSV file")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--addresses", "-a", help="File with addresses to search (one per line)")
    mode_group.add_argument("--bulk", action="store_true", help="Bulk mode: scrape all brochures from for-sale pages")

    parser.add_argument("--county", "-c", choices=COUNTIES, help="Filter by county (bulk mode)")
    parser.add_argument("--limit", "-l", type=int, help="Maximum listings to scrape")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--fresh", "-f", action="store_true", help="Start fresh")
    parser.add_argument("--no-headless", action="store_true", help="Show browser")
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.resume and args.fresh:
        parser.error("Cannot use --resume and --fresh together")

    setup_logging(args.verbose)

    scraper = MyHomeBrochureScraper(
        output_file=args.out,
        addresses_file=args.addresses,
        bulk_mode=args.bulk,
        county=args.county,
        limit=args.limit,
        headless=not args.no_headless,
        checkpoint_interval=args.checkpoint_interval,
    )

    try:
        if args.bulk:
            scraper.run_bulk(resume=args.resume, fresh=args.fresh)
        else:
            scraper.run_address_search(resume=args.resume, fresh=args.fresh)
    except KeyboardInterrupt:
        print("\nInterrupted. Checkpoint saved.")
        sys.exit(0)


if __name__ == "__main__":
    main()
