#!/usr/bin/env python3
"""
Daft.ie Sold Properties Scraper
================================
A robust scraper for extracting comprehensive property data from Daft.ie's sold properties section.

Features:
- Extracts ALL available fields from listings
- Real-time progress monitoring with data quality stats
- Checkpoint/resume system for interrupted runs
- Anti-bot handling with stealth Playwright
- Adaptive rate limiting

Usage:
    # Scrape all sold properties
    python scrape_daft_sold.py --out sold_listings.csv

    # Limit to specific county
    python scrape_daft_sold.py --county dublin --out dublin_sold.csv

    # Limit number of listings
    python scrape_daft_sold.py --limit 1000 --out sample.csv

    # Resume interrupted run
    python scrape_daft_sold.py --out sold_listings.csv --resume

    # Fresh start (ignore checkpoint)
    python scrape_daft_sold.py --out sold_listings.csv --fresh

    # Visible browser for debugging
    python scrape_daft_sold.py --out test.csv --no-headless --limit 10

    # Adjust checkpoint frequency
    python scrape_daft_sold.py --out sold.csv --checkpoint-interval 100

Author: House Price Predictor Project
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import signal
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

# Third-party imports
try:
    from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext, TimeoutError as PlaywrightTimeout
except ImportError:
    print("ERROR: playwright not installed. Run: pip install playwright && playwright install chromium")
    sys.exit(1)

# ============================================================================
# CONSTANTS
# ============================================================================

# URLs
BASE_URL = "https://www.daft.ie"
SOLD_PROPERTIES_URL = "https://www.daft.ie/sold-properties/ireland"

# County URL slugs
COUNTY_SLUGS = {
    "carlow": "carlow",
    "cavan": "cavan",
    "clare": "clare",
    "cork": "cork",
    "donegal": "donegal",
    "dublin": "dublin",
    "galway": "galway",
    "kerry": "kerry",
    "kildare": "kildare",
    "kilkenny": "kilkenny",
    "laois": "laois",
    "leitrim": "leitrim",
    "limerick": "limerick",
    "longford": "longford",
    "louth": "louth",
    "mayo": "mayo",
    "meath": "meath",
    "monaghan": "monaghan",
    "offaly": "offaly",
    "roscommon": "roscommon",
    "sligo": "sligo",
    "tipperary": "tipperary",
    "waterford": "waterford",
    "westmeath": "westmeath",
    "wexford": "wexford",
    "wicklow": "wicklow",
}

# Scraping settings
DEFAULT_CHECKPOINT_INTERVAL = 50
DEFAULT_MONITOR_INTERVAL = 25
MIN_DELAY_SECONDS = 1.5
MAX_DELAY_SECONDS = 8.0
ADAPTIVE_DELAY_INCREASE = 1.5
ADAPTIVE_DELAY_DECREASE = 0.9
PAGE_LOAD_TIMEOUT_MS = 60000
LISTING_LOAD_TIMEOUT_MS = 45000

# Field names for CSV output
CSV_FIELDS = [
    # Core
    "daft_id", "url", "address_full", "address_line1", "address_line2", "address_line3",
    "county", "area", "locality", "eircode",
    # Prices
    "sold_price", "asking_price", "price_change",
    # Dates
    "sale_date", "date_listed", "date_entered_market",
    # Property type
    "property_type", "property_category",
    # Details
    "bedrooms", "bathrooms", "ensuites", "size_sqm", "size_sqft", "floor_level",
    # Energy
    "ber_rating", "ber_number", "ber_epi",
    # Building
    "year_built", "new_build", "refurbished", "condition",
    # Furnishing
    "furnished", "turnkey",
    # Amenities (boolean)
    "parking", "garage", "garden", "balcony", "terrace",
    "central_heating", "heating_type", "double_glazing",
    "alarm", "cctv", "wheelchair_accessible",
    # Location
    "latitude", "longitude",
    # Listing metadata
    "agent_name", "agent_phone", "agent_license",
    "num_images", "views_count", "description", "features_list",
    # Structured data
    "og_title", "og_description", "og_image",
    "schema_type", "schema_data",
    # Extraction metadata
    "scraped_at", "html_source",
]

# Regex patterns
EIRCODE_PATTERN = re.compile(r'\b([A-Z]\d{2}\s?[A-Z0-9]{4})\b', re.IGNORECASE)
PRICE_PATTERN = re.compile(r'€\s*([\d,]+(?:\.\d{2})?)', re.IGNORECASE)
PRICE_AMV_PATTERN = re.compile(r'AMV:\s*€\s*([\d,]+)', re.IGNORECASE)
BER_RATING_PATTERN = re.compile(r'\b([A-G][1-3]?)\b', re.IGNORECASE)
BER_NUMBER_PATTERN = re.compile(r'BER\s*(?:No\.?|Number)?\s*:?\s*(\d{9,12})', re.IGNORECASE)
SQFT_PATTERN = re.compile(r'([\d,]+)\s*(?:sq\.?\s*ft|sqft|square\s*feet)', re.IGNORECASE)
SQM_PATTERN = re.compile(r'([\d,]+(?:\.\d+)?)\s*(?:sq\.?\s*m|sqm|m²|square\s*met)', re.IGNORECASE)
YEAR_BUILT_PATTERN = re.compile(r'\b(19\d{2}|20[0-2]\d)\b')


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PropertyListing:
    """Comprehensive property listing data structure."""
    # Core identifiers
    daft_id: Optional[str] = None
    url: Optional[str] = None

    # Address components
    address_full: Optional[str] = None
    address_line1: Optional[str] = None
    address_line2: Optional[str] = None
    address_line3: Optional[str] = None
    county: Optional[str] = None
    area: Optional[str] = None
    locality: Optional[str] = None
    eircode: Optional[str] = None

    # Prices
    sold_price: Optional[int] = None
    asking_price: Optional[int] = None
    price_change: Optional[str] = None

    # Dates
    sale_date: Optional[str] = None
    date_listed: Optional[str] = None
    date_entered_market: Optional[str] = None

    # Property classification
    property_type: Optional[str] = None
    property_category: Optional[str] = None

    # Property details
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    ensuites: Optional[int] = None
    size_sqm: Optional[float] = None
    size_sqft: Optional[float] = None
    floor_level: Optional[str] = None

    # Energy rating
    ber_rating: Optional[str] = None
    ber_number: Optional[str] = None
    ber_epi: Optional[float] = None

    # Building info
    year_built: Optional[int] = None
    new_build: Optional[bool] = None
    refurbished: Optional[bool] = None
    condition: Optional[str] = None

    # Furnishing
    furnished: Optional[str] = None
    turnkey: Optional[bool] = None

    # Amenities
    parking: Optional[bool] = None
    garage: Optional[bool] = None
    garden: Optional[bool] = None
    balcony: Optional[bool] = None
    terrace: Optional[bool] = None
    central_heating: Optional[bool] = None
    heating_type: Optional[str] = None
    double_glazing: Optional[bool] = None
    alarm: Optional[bool] = None
    cctv: Optional[bool] = None
    wheelchair_accessible: Optional[bool] = None

    # Location coordinates
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    # Agent info
    agent_name: Optional[str] = None
    agent_phone: Optional[str] = None
    agent_license: Optional[str] = None

    # Listing metadata
    num_images: Optional[int] = None
    views_count: Optional[int] = None
    description: Optional[str] = None
    features_list: Optional[str] = None

    # Open Graph data
    og_title: Optional[str] = None
    og_description: Optional[str] = None
    og_image: Optional[str] = None

    # Schema.org data
    schema_type: Optional[str] = None
    schema_data: Optional[str] = None

    # Extraction metadata
    scraped_at: Optional[str] = None
    html_source: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for CSV export."""
        return asdict(self)


@dataclass
class CheckpointData:
    """Checkpoint data for resume functionality."""
    last_page: int = 0
    last_listing_url: str = ""
    processed_count: int = 0
    processed_urls: set[str] = field(default_factory=set)
    started_at: str = ""
    last_saved_at: str = ""
    total_listings_found: int = 0
    output_file: str = ""
    county_filter: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "last_page": self.last_page,
            "last_listing_url": self.last_listing_url,
            "processed_count": self.processed_count,
            "processed_urls": list(self.processed_urls),
            "started_at": self.started_at,
            "last_saved_at": self.last_saved_at,
            "total_listings_found": self.total_listings_found,
            "output_file": self.output_file,
            "county_filter": self.county_filter,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointData:
        """Create from dictionary."""
        return cls(
            last_page=data.get("last_page", 0),
            last_listing_url=data.get("last_listing_url", ""),
            processed_count=data.get("processed_count", 0),
            processed_urls=set(data.get("processed_urls", [])),
            started_at=data.get("started_at", ""),
            last_saved_at=data.get("last_saved_at", ""),
            total_listings_found=data.get("total_listings_found", 0),
            output_file=data.get("output_file", ""),
            county_filter=data.get("county_filter"),
        )


@dataclass
class ScraperStats:
    """Statistics for monitoring scraper progress."""
    total_found: int = 0
    processed: int = 0
    successful: int = 0
    failed: int = 0

    # Field coverage tracking
    field_counts: dict[str, int] = field(default_factory=dict)

    # Error tracking
    errors: dict[str, int] = field(default_factory=dict)

    # Timing
    start_time: float = 0.0
    last_update_time: float = 0.0

    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.processed == 0:
            return 0.0
        return (self.successful / self.processed) * 100

    def listings_per_second(self) -> float:
        """Calculate processing speed."""
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0.0
        return self.processed / elapsed

    def eta_seconds(self) -> float:
        """Estimate time remaining."""
        speed = self.listings_per_second()
        if speed == 0:
            return float('inf')
        remaining = self.total_found - self.processed
        return remaining / speed


# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================

class CheckpointManager:
    """Manages checkpoint saving and loading for resume functionality."""

    def __init__(self, output_file: str):
        self.output_file = output_file
        self.checkpoint_file = self._get_checkpoint_path(output_file)
        self.data = CheckpointData()
        self.logger = logging.getLogger(__name__)

    def _get_checkpoint_path(self, output_file: str) -> Path:
        """Generate checkpoint file path from output file."""
        output_path = Path(output_file)
        return output_path.parent / f"{output_path.stem}_checkpoint.json"

    def exists(self) -> bool:
        """Check if checkpoint file exists."""
        return self.checkpoint_file.exists()

    def load(self) -> bool:
        """Load checkpoint data. Returns True if successful."""
        if not self.exists():
            return False

        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.data = CheckpointData.from_dict(data)

            # Validate checkpoint matches current output file
            if self.data.output_file and self.data.output_file != self.output_file:
                self.logger.warning(
                    f"Checkpoint output file mismatch: {self.data.output_file} vs {self.output_file}"
                )

            self.logger.info(
                f"Loaded checkpoint: page {self.data.last_page}, "
                f"{self.data.processed_count} processed"
            )
            return True

        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False

    def save(self) -> None:
        """Save current checkpoint data."""
        self.data.last_saved_at = datetime.now().isoformat()
        self.data.output_file = self.output_file

        try:
            # Write to temp file first for atomic save
            temp_file = self.checkpoint_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.data.to_dict(), f, indent=2)

            # Atomic rename
            temp_file.replace(self.checkpoint_file)

        except IOError as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

    def update(self, page: int, url: str, count: int) -> None:
        """Update checkpoint with latest progress."""
        self.data.last_page = page
        self.data.last_listing_url = url
        self.data.processed_count = count
        self.data.processed_urls.add(url)

    def mark_url_processed(self, url: str) -> None:
        """Mark a URL as processed."""
        self.data.processed_urls.add(url)

    def is_url_processed(self, url: str) -> bool:
        """Check if URL was already processed."""
        return url in self.data.processed_urls

    def clear(self) -> None:
        """Delete checkpoint file for fresh start."""
        if self.exists():
            self.checkpoint_file.unlink()
            self.logger.info("Checkpoint file deleted")
        self.data = CheckpointData()

    def initialize_new_run(self, county: Optional[str] = None) -> None:
        """Initialize a new scraping run."""
        self.data = CheckpointData(
            started_at=datetime.now().isoformat(),
            output_file=self.output_file,
            county_filter=county,
        )


# ============================================================================
# PROGRESS MONITOR
# ============================================================================

class ProgressMonitor:
    """Real-time progress monitoring with data quality stats."""

    TRACKED_FIELDS = [
        "sold_price", "asking_price", "bedrooms", "bathrooms",
        "size_sqm", "ber_rating", "eircode", "latitude", "description"
    ]

    def __init__(self, update_interval: int = DEFAULT_MONITOR_INTERVAL):
        self.stats = ScraperStats()
        self.stats.start_time = time.time()
        self.stats.field_counts = {f: 0 for f in self.TRACKED_FIELDS}
        self.update_interval = update_interval
        self.last_address = ""
        self.logger = logging.getLogger(__name__)

    def update(self, listing: Optional[PropertyListing], success: bool, error: Optional[str] = None) -> None:
        """Update stats with a processed listing."""
        self.stats.processed += 1

        if success and listing:
            self.stats.successful += 1
            self.last_address = listing.address_full or "Unknown"

            # Track field coverage
            listing_dict = listing.to_dict()
            for field_name in self.TRACKED_FIELDS:
                value = listing_dict.get(field_name)
                if value is not None and value != "" and value != 0:
                    self.stats.field_counts[field_name] = self.stats.field_counts.get(field_name, 0) + 1
        else:
            self.stats.failed += 1
            if error:
                error_type = error.split(":")[0] if ":" in error else error[:50]
                self.stats.errors[error_type] = self.stats.errors.get(error_type, 0) + 1

        self.stats.last_update_time = time.time()

    def set_total(self, total: int) -> None:
        """Set total number of listings found."""
        self.stats.total_found = total

    def should_display(self) -> bool:
        """Check if it's time to display progress."""
        return self.stats.processed % self.update_interval == 0

    def display(self) -> None:
        """Display current progress to console."""
        self._clear_and_print(self._format_display())

    def _clear_and_print(self, text: str) -> None:
        """Clear screen and print progress."""
        # Use ANSI escape codes for clearing (works on most terminals)
        if sys.stdout.isatty():
            print("\033[2J\033[H", end="")  # Clear screen and move cursor to top
        print(text)

    def _format_display(self) -> str:
        """Format the progress display."""
        stats = self.stats

        # Calculate progress
        progress_pct = (stats.processed / max(stats.total_found, 1)) * 100
        bar_width = 40
        filled = int(bar_width * progress_pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Calculate ETA
        eta_sec = stats.eta_seconds()
        if eta_sec == float('inf'):
            eta_str = "calculating..."
        else:
            eta_str = str(timedelta(seconds=int(eta_sec)))

        # Format field coverage
        field_lines = []
        for field_name in self.TRACKED_FIELDS:
            count = stats.field_counts.get(field_name, 0)
            pct = (count / max(stats.successful, 1)) * 100
            check = "✓" if pct >= 50 else "○"
            field_lines.append(f"  {check} {field_name:14s}: {count:>6}/{stats.successful:<6} ({pct:5.1f}%)")

        # Format top errors
        error_lines = []
        sorted_errors = sorted(stats.errors.items(), key=lambda x: x[1], reverse=True)[:5]
        for err_type, count in sorted_errors:
            error_lines.append(f"    - {err_type}: {count}")

        # Build display
        lines = [
            "=" * 80,
            "DAFT.IE SOLD PROPERTIES SCRAPER",
            "=" * 80,
            f"Started: {datetime.fromtimestamp(stats.start_time).strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Progress: [{bar}] {progress_pct:5.1f}%  |  {stats.processed:,} / {stats.total_found:,} listings",
            f"Speed:    {stats.listings_per_second():.2f} listings/sec  |  ETA: {eta_str}",
            "",
            f"Success: {stats.successful:,} ({stats.success_rate():.1f}%)  |  Failed: {stats.failed:,} ({100-stats.success_rate():.1f}%)",
            "",
            "Data Coverage:",
            *field_lines,
            "",
        ]

        if error_lines:
            lines.extend([
                "Top Errors:",
                *error_lines,
                "",
            ])

        lines.extend([
            f"Last processed: {self.last_address[:60]}",
            "=" * 80,
        ])

        return "\n".join(lines)

    def final_summary(self) -> str:
        """Generate final summary report."""
        stats = self.stats
        elapsed = time.time() - stats.start_time

        lines = [
            "",
            "=" * 80,
            "SCRAPING COMPLETE",
            "=" * 80,
            f"Duration: {timedelta(seconds=int(elapsed))}",
            f"Total processed: {stats.processed:,}",
            f"Successful: {stats.successful:,} ({stats.success_rate():.1f}%)",
            f"Failed: {stats.failed:,}",
            f"Average speed: {stats.listings_per_second():.2f} listings/sec",
            "",
            "Final Data Coverage:",
        ]

        for field_name in self.TRACKED_FIELDS:
            count = stats.field_counts.get(field_name, 0)
            pct = (count / max(stats.successful, 1)) * 100
            lines.append(f"  {field_name:14s}: {count:>6}/{stats.successful:<6} ({pct:5.1f}%)")

        lines.append("=" * 80)

        return "\n".join(lines)


# ============================================================================
# DAFT SCRAPER
# ============================================================================

class DaftSoldScraper:
    """Main scraper class for Daft.ie sold properties."""

    def __init__(
        self,
        output_file: str,
        county: Optional[str] = None,
        limit: Optional[int] = None,
        headless: bool = True,
        checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
        monitor_interval: int = DEFAULT_MONITOR_INTERVAL,
    ):
        self.output_file = output_file
        self.county = county.lower() if county else None
        self.limit = limit
        self.headless = headless
        self.checkpoint_interval = checkpoint_interval

        # Managers
        self.checkpoint = CheckpointManager(output_file)
        self.monitor = ProgressMonitor(update_interval=monitor_interval)

        # Playwright objects
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

        # State
        self.current_delay = MIN_DELAY_SECONDS
        self.shutdown_requested = False
        self.listings_written = 0

        # Logging
        self.logger = logging.getLogger(__name__)

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals gracefully."""
        self.logger.info("\nShutdown requested, saving checkpoint...")
        self.shutdown_requested = True

    def _get_search_url(self, page_num: int = 1) -> str:
        """Build search URL for sold properties."""
        if self.county and self.county in COUNTY_SLUGS:
            base = f"{BASE_URL}/sold-properties/{COUNTY_SLUGS[self.county]}"
        else:
            base = SOLD_PROPERTIES_URL

        if page_num > 1:
            # Daft.ie uses 'from' parameter for pagination (0-indexed, 20 per page)
            offset = (page_num - 1) * 20
            return f"{base}?from={offset}"
        return base

    def _init_browser(self) -> None:
        """Initialize Playwright browser with stealth settings."""
        self.playwright = sync_playwright().start()

        # Use Firefox - better at avoiding Cloudflare detection than Chromium
        self.browser = self.playwright.firefox.launch(
            headless=self.headless,
        )

        self.context = self.browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) "
                "Gecko/20100101 Firefox/121.0"
            ),
            locale="en-IE",
            timezone_id="Europe/Dublin",
        )

        self.page = self.context.new_page()
        self.page.set_default_timeout(PAGE_LOAD_TIMEOUT_MS)

    def _wait_for_cloudflare(self) -> bool:
        """Wait for Cloudflare challenge to complete. Returns True if passed."""
        max_wait = 30  # seconds
        start = time.time()

        while time.time() - start < max_wait:
            # Check if we're on a Cloudflare challenge page
            title = self.page.title().lower()
            content = self.page.content().lower()

            # Cloudflare challenge indicators
            cf_indicators = [
                "just a moment" in title,
                "checking your browser" in content,
                "cloudflare" in content and "challenge" in content,
                "ray id" in content and "cloudflare" in content,
            ]

            if not any(cf_indicators):
                # Not on Cloudflare page, we're through
                return True

            self.logger.debug("Waiting for Cloudflare challenge...")
            time.sleep(2)

        self.logger.warning("Cloudflare challenge timeout")
        return False

    def _close_browser(self) -> None:
        """Clean up browser resources."""
        if self.page:
            self.page.close()
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

    def _dismiss_cookie_popup(self) -> None:
        """Attempt to dismiss cookie consent popups."""
        try:
            # Try common cookie consent selectors
            selectors = [
                'button[id*="accept"]',
                'button[class*="accept"]',
                'button:has-text("Accept")',
                'button:has-text("Accept All")',
                'button:has-text("I Accept")',
                '#didomi-notice-agree-button',
                '.didomi-continue-without-agreeing',
                '[data-testid="cookie-popup-accept"]',
            ]

            for selector in selectors:
                try:
                    button = self.page.locator(selector).first
                    if button.is_visible(timeout=1000):
                        button.click(timeout=2000)
                        self.logger.debug("Dismissed cookie popup")
                        time.sleep(0.5)
                        return
                except:
                    continue

        except Exception as e:
            self.logger.debug(f"Cookie dismiss attempt: {e}")

    def _adaptive_delay(self, success: bool) -> None:
        """Adjust delay based on success/failure."""
        if success:
            self.current_delay = max(MIN_DELAY_SECONDS, self.current_delay * ADAPTIVE_DELAY_DECREASE)
        else:
            self.current_delay = min(MAX_DELAY_SECONDS, self.current_delay * ADAPTIVE_DELAY_INCREASE)

        time.sleep(self.current_delay)

    def _extract_listing_urls_from_page(self) -> list[str]:
        """Extract all listing URLs from current search results page."""
        urls = []

        try:
            # Wait for listings to load - try multiple selectors
            try:
                self.page.wait_for_selector('[data-testid="results"]', timeout=10000)
            except PlaywrightTimeout:
                # Fallback: wait for any listing cards
                self.page.wait_for_selector('a[href*="/sold/"]', timeout=10000)

            # Find all sold listing links - Daft uses /sold/ for sold properties
            links = self.page.locator('a[href*="/sold/"]').all()

            for link in links:
                try:
                    href = link.get_attribute("href")
                    if href and "/sold/" in href:
                        # Normalize URL
                        full_url = urljoin(BASE_URL, href)
                        # Filter: must have /sold/ and end with alphanumeric listing ID
                        # Daft IDs look like: EB3161BA3910287080258 or 608829C1D312AADC80258
                        if "/sold/" in full_url and re.search(r'/[A-Z0-9]{10,}/?$', full_url, re.IGNORECASE):
                            urls.append(full_url.rstrip('/'))
                except:
                    continue

            # Deduplicate while preserving order
            seen = set()
            unique_urls = []
            for url in urls:
                if url not in seen:
                    seen.add(url)
                    unique_urls.append(url)

            self.logger.debug(f"Found {len(unique_urls)} listing URLs on page")
            return unique_urls

        except PlaywrightTimeout:
            self.logger.warning("Timeout waiting for search results")
            return []

    def _get_total_listings_count(self) -> int:
        """Extract total number of sold listings from search page."""
        try:
            # Look for results count text
            count_selectors = [
                '[data-testid="results-header"]',
                'h1:has-text("Properties")',
                '.styles__SearchH1',
            ]

            for selector in count_selectors:
                try:
                    element = self.page.locator(selector).first
                    if element.is_visible(timeout=2000):
                        text = element.text_content()
                        # Extract number from text like "1,234 Properties Sold"
                        match = re.search(r'([\d,]+)\s*(?:Properties|Results)', text, re.IGNORECASE)
                        if match:
                            return int(match.group(1).replace(",", ""))
                except:
                    continue

            # Fallback: estimate from pagination
            return 10000  # Default estimate

        except Exception as e:
            self.logger.debug(f"Could not get total count: {e}")
            return 10000

    def _extract_next_data(self) -> Optional[dict]:
        """Extract __NEXT_DATA__ JSON from page."""
        try:
            script = self.page.locator('script#__NEXT_DATA__').first
            if script.is_visible(timeout=2000):
                content = script.text_content()
                return json.loads(content)
        except:
            pass
        return None

    def _extract_json_ld(self) -> list[dict]:
        """Extract all JSON-LD schema data from page."""
        schemas = []
        try:
            scripts = self.page.locator('script[type="application/ld+json"]').all()
            for script in scripts:
                try:
                    content = script.text_content()
                    data = json.loads(content)
                    schemas.append(data)
                except:
                    continue
        except:
            pass
        return schemas

    def _extract_og_tags(self) -> dict[str, str]:
        """Extract Open Graph meta tags."""
        og_data = {}
        try:
            og_tags = [
                ("og:title", "og_title"),
                ("og:description", "og_description"),
                ("og:image", "og_image"),
            ]

            for og_prop, key in og_tags:
                try:
                    meta = self.page.locator(f'meta[property="{og_prop}"]').first
                    content = meta.get_attribute("content")
                    if content:
                        og_data[key] = content
                except:
                    continue
        except:
            pass
        return og_data

    def _parse_price(self, text: Optional[str]) -> Optional[int]:
        """Parse price string to integer."""
        if not text:
            return None

        # Remove currency symbols and whitespace
        text = text.strip()

        # Handle AMV prices
        amv_match = PRICE_AMV_PATTERN.search(text)
        if amv_match:
            return int(amv_match.group(1).replace(",", ""))

        # Standard price
        match = PRICE_PATTERN.search(text)
        if match:
            price_str = match.group(1).replace(",", "")
            return int(float(price_str))

        return None

    def _extract_property_details(self, next_data: Optional[dict]) -> PropertyListing:
        """Extract all property details from page and structured data."""
        listing = PropertyListing()
        listing.scraped_at = datetime.now().isoformat()
        listing.url = self.page.url

        # Extract Daft ID from URL (alphanumeric IDs like EB3161BA3910287080258)
        url_match = re.search(r'/([A-Z0-9]{10,})/?$', listing.url, re.IGNORECASE)
        if url_match:
            listing.daft_id = url_match.group(1)

        # Try to get data from __NEXT_DATA__ first (most reliable)
        if next_data:
            self._extract_from_next_data(listing, next_data)

        # Extract from JSON-LD schemas
        schemas = self._extract_json_ld()
        for schema in schemas:
            self._extract_from_schema(listing, schema)

        # Extract Open Graph tags
        og_data = self._extract_og_tags()
        listing.og_title = og_data.get("og_title")
        listing.og_description = og_data.get("og_description")
        listing.og_image = og_data.get("og_image")

        # Extract from page content (fills in gaps)
        self._extract_from_page_content(listing)

        return listing

    def _extract_from_next_data(self, listing: PropertyListing, next_data: dict) -> None:
        """Extract data from __NEXT_DATA__ JSON."""
        try:
            # Navigate to listing data (structure varies)
            props = next_data.get("props", {})
            page_props = props.get("pageProps", {})

            # Try different data locations
            listing_data = (
                page_props.get("listing") or
                page_props.get("property") or
                page_props.get("data", {}).get("listing") or
                {}
            )

            if not listing_data:
                return

            # Address
            address = listing_data.get("address") or listing_data.get("title") or ""
            if isinstance(address, dict):
                listing.address_line1 = address.get("line1")
                listing.address_line2 = address.get("line2")
                listing.address_line3 = address.get("line3")
                listing.county = address.get("county")
                listing.area = address.get("area")
                listing.locality = address.get("locality")
                listing.eircode = address.get("eircode")
                listing.address_full = ", ".join(filter(None, [
                    listing.address_line1,
                    listing.address_line2,
                    listing.address_line3
                ]))
            else:
                listing.address_full = address

            # Prices
            price_info = listing_data.get("price") or listing_data.get("salePrice") or {}
            if isinstance(price_info, dict):
                listing.sold_price = price_info.get("value") or price_info.get("soldPrice")
                listing.asking_price = price_info.get("askingPrice") or price_info.get("displayPrice")
            elif isinstance(price_info, (int, float)):
                listing.sold_price = int(price_info)

            # Property type
            listing.property_type = listing_data.get("propertyType") or listing_data.get("category")
            listing.property_category = listing_data.get("category")

            # Details
            listing.bedrooms = listing_data.get("numBedrooms") or listing_data.get("beds")
            listing.bathrooms = listing_data.get("numBathrooms") or listing_data.get("baths")

            # Size
            floor_area = listing_data.get("floorArea") or {}
            if isinstance(floor_area, dict):
                listing.size_sqm = floor_area.get("value")
                listing.size_sqft = floor_area.get("sqft")

            # BER
            ber = listing_data.get("ber") or listing_data.get("berRating") or {}
            if isinstance(ber, dict):
                listing.ber_rating = ber.get("rating")
                listing.ber_number = ber.get("code") or ber.get("number")
                listing.ber_epi = ber.get("epi")
            elif isinstance(ber, str):
                listing.ber_rating = ber

            # Location
            coords = listing_data.get("point") or listing_data.get("coordinates") or {}
            if isinstance(coords, dict):
                listing.latitude = coords.get("latitude") or coords.get("lat")
                listing.longitude = coords.get("longitude") or coords.get("lng") or coords.get("lon")

            # Agent
            agent = listing_data.get("seller") or listing_data.get("agent") or {}
            if isinstance(agent, dict):
                listing.agent_name = agent.get("name") or agent.get("displayName")
                listing.agent_phone = agent.get("phone") or agent.get("phoneNumber")
                listing.agent_license = agent.get("licenseNumber")

            # Media
            media = listing_data.get("media") or listing_data.get("images") or []
            if isinstance(media, list):
                listing.num_images = len(media)

            # Description
            listing.description = listing_data.get("description") or listing_data.get("fullDescription")

            # Features
            features = listing_data.get("features") or listing_data.get("facilities") or []
            if isinstance(features, list):
                listing.features_list = "; ".join(str(f) for f in features)

            # Amenities from features
            self._parse_features_for_amenities(listing, features)

            # Dates
            listing.sale_date = listing_data.get("saleDate") or listing_data.get("soldDate")
            listing.date_listed = listing_data.get("publishDate") or listing_data.get("firstPublished")

            # Store raw schema data
            listing.schema_data = json.dumps(listing_data)[:5000]  # Truncate to reasonable size

        except Exception as e:
            self.logger.debug(f"Error extracting from __NEXT_DATA__: {e}")

    def _extract_from_schema(self, listing: PropertyListing, schema: dict) -> None:
        """Extract data from JSON-LD schema."""
        try:
            schema_type = schema.get("@type", "")
            listing.schema_type = schema_type

            if schema_type in ["Product", "RealEstateListing", "Place", "Residence"]:
                # Address from schema
                address = schema.get("address") or {}
                if isinstance(address, dict):
                    if not listing.address_full:
                        listing.address_full = address.get("streetAddress")
                    if not listing.locality:
                        listing.locality = address.get("addressLocality")
                    if not listing.county:
                        listing.county = address.get("addressRegion")
                    if not listing.eircode:
                        listing.eircode = address.get("postalCode")

                # Geo coordinates
                geo = schema.get("geo") or {}
                if isinstance(geo, dict):
                    if not listing.latitude:
                        listing.latitude = geo.get("latitude")
                    if not listing.longitude:
                        listing.longitude = geo.get("longitude")

                # Price from schema
                offers = schema.get("offers") or {}
                if isinstance(offers, dict):
                    if not listing.sold_price:
                        price = offers.get("price")
                        if price:
                            listing.sold_price = int(float(price))

                # Images count
                images = schema.get("image") or []
                if isinstance(images, list) and not listing.num_images:
                    listing.num_images = len(images)

        except Exception as e:
            self.logger.debug(f"Error extracting from schema: {e}")

    def _extract_from_page_content(self, listing: PropertyListing) -> None:
        """Extract data directly from page HTML/DOM."""
        try:
            # Address from title/header
            if not listing.address_full:
                title_selectors = [
                    '[data-testid="address"]',
                    'h1[data-testid="address"]',
                    '.PropertyMainHeader__Address',
                    'h1',
                ]
                for selector in title_selectors:
                    try:
                        element = self.page.locator(selector).first
                        if element.is_visible(timeout=500):
                            listing.address_full = element.text_content().strip()
                            break
                    except:
                        continue

            # Sold price
            if not listing.sold_price:
                price_selectors = [
                    '[data-testid="price"]',
                    '.PropertyMainHeader__Price',
                    '[class*="Price"]',
                    'span:has-text("Sold")',
                ]
                for selector in price_selectors:
                    try:
                        element = self.page.locator(selector).first
                        if element.is_visible(timeout=500):
                            text = element.text_content()
                            listing.sold_price = self._parse_price(text)
                            if listing.sold_price:
                                break
                    except:
                        continue

            # Property details (beds, baths, size)
            self._extract_property_info_badges(listing)

            # BER rating
            if not listing.ber_rating:
                ber_selectors = [
                    '[data-testid="ber"]',
                    '[class*="BER"]',
                    'img[alt*="BER"]',
                    ':has-text("BER")',
                ]
                for selector in ber_selectors:
                    try:
                        element = self.page.locator(selector).first
                        if element.is_visible(timeout=500):
                            text = element.text_content() or element.get_attribute("alt") or ""
                            match = BER_RATING_PATTERN.search(text)
                            if match:
                                listing.ber_rating = match.group(1).upper()
                                break
                    except:
                        continue

            # Eircode from page text
            if not listing.eircode:
                try:
                    page_text = self.page.content()
                    match = EIRCODE_PATTERN.search(page_text)
                    if match:
                        listing.eircode = match.group(1).upper().replace(" ", "")
                except:
                    pass

            # Description
            if not listing.description:
                desc_selectors = [
                    '[data-testid="description"]',
                    '.PropertyDescription',
                    '[class*="description"]',
                ]
                for selector in desc_selectors:
                    try:
                        element = self.page.locator(selector).first
                        if element.is_visible(timeout=500):
                            listing.description = element.text_content().strip()
                            break
                    except:
                        continue

            # Features list
            if not listing.features_list:
                try:
                    features_elements = self.page.locator('[data-testid="facilities"] li, .PropertyFacilities li').all()
                    features = [el.text_content().strip() for el in features_elements if el.text_content()]
                    if features:
                        listing.features_list = "; ".join(features)
                        self._parse_features_for_amenities(listing, features)
                except:
                    pass

            # Agent info
            if not listing.agent_name:
                try:
                    agent_element = self.page.locator('[data-testid="seller-name"], .SellerInfo__Name').first
                    if agent_element.is_visible(timeout=500):
                        listing.agent_name = agent_element.text_content().strip()
                except:
                    pass

            if not listing.agent_phone:
                try:
                    phone_element = self.page.locator('[data-testid="seller-phone"], a[href^="tel:"]').first
                    if phone_element.is_visible(timeout=500):
                        phone_text = phone_element.get_attribute("href") or phone_element.text_content()
                        listing.agent_phone = phone_text.replace("tel:", "").strip()
                except:
                    pass

            # Extract county from address if not set
            if not listing.county and listing.address_full:
                for county in COUNTY_SLUGS.keys():
                    if county.lower() in listing.address_full.lower():
                        listing.county = county.title()
                        break

        except Exception as e:
            self.logger.debug(f"Error extracting from page content: {e}")

    def _extract_property_info_badges(self, listing: PropertyListing) -> None:
        """Extract beds, baths, size from property info badges."""
        try:
            # Look for property info section
            info_selectors = [
                '[data-testid="property-overview"] li',
                '.PropertyOverview li',
                '.QuickPropertyDetails span',
                '[class*="PropertyInfo"] span',
            ]

            for selector in info_selectors:
                try:
                    elements = self.page.locator(selector).all()
                    for element in elements:
                        text = element.text_content().strip().lower()

                        # Bedrooms
                        if not listing.bedrooms and ("bed" in text or "bedroom" in text):
                            num_match = re.search(r'(\d+)', text)
                            if num_match:
                                listing.bedrooms = int(num_match.group(1))

                        # Bathrooms
                        if not listing.bathrooms and ("bath" in text or "bathroom" in text):
                            num_match = re.search(r'(\d+)', text)
                            if num_match:
                                listing.bathrooms = int(num_match.group(1))

                        # Size
                        if not listing.size_sqm:
                            sqm_match = SQM_PATTERN.search(text)
                            if sqm_match:
                                listing.size_sqm = float(sqm_match.group(1).replace(",", ""))

                        if not listing.size_sqft:
                            sqft_match = SQFT_PATTERN.search(text)
                            if sqft_match:
                                listing.size_sqft = float(sqft_match.group(1).replace(",", ""))
                except:
                    continue

        except Exception as e:
            self.logger.debug(f"Error extracting property info badges: {e}")

    def _parse_features_for_amenities(self, listing: PropertyListing, features: list) -> None:
        """Parse features list to set amenity flags."""
        if not features:
            return

        features_lower = " ".join(str(f).lower() for f in features)

        # Parking
        if any(x in features_lower for x in ["parking", "car space", "car park"]):
            listing.parking = True

        # Garage
        if "garage" in features_lower:
            listing.garage = True

        # Garden
        if any(x in features_lower for x in ["garden", "rear garden", "front garden"]):
            listing.garden = True

        # Balcony
        if "balcony" in features_lower:
            listing.balcony = True

        # Terrace
        if "terrace" in features_lower:
            listing.terrace = True

        # Heating
        if any(x in features_lower for x in ["central heating", "gas heating", "oil heating"]):
            listing.central_heating = True
            if "gas" in features_lower:
                listing.heating_type = "Gas"
            elif "oil" in features_lower:
                listing.heating_type = "Oil"
            elif "electric" in features_lower:
                listing.heating_type = "Electric"

        # Double glazing
        if "double glaz" in features_lower:
            listing.double_glazing = True

        # Alarm
        if "alarm" in features_lower:
            listing.alarm = True

        # CCTV
        if "cctv" in features_lower:
            listing.cctv = True

        # Wheelchair
        if any(x in features_lower for x in ["wheelchair", "accessible", "disability"]):
            listing.wheelchair_accessible = True

        # New build
        if "new build" in features_lower or "newly built" in features_lower:
            listing.new_build = True

        # Refurbished
        if any(x in features_lower for x in ["refurbished", "renovated", "modernised"]):
            listing.refurbished = True

        # Furnished
        if "furnished" in features_lower:
            if "unfurnished" in features_lower:
                listing.furnished = "Unfurnished"
            elif "partly furnished" in features_lower:
                listing.furnished = "Partly Furnished"
            else:
                listing.furnished = "Furnished"

        # Turnkey
        if "turnkey" in features_lower:
            listing.turnkey = True

        # Ensuite count
        ensuite_match = re.search(r'(\d+)\s*ensuite', features_lower)
        if ensuite_match:
            listing.ensuites = int(ensuite_match.group(1))
        elif "ensuite" in features_lower:
            listing.ensuites = 1

    def scrape_listing(self, url: str) -> Optional[PropertyListing]:
        """Scrape a single listing page."""
        try:
            self.page.goto(url, wait_until="domcontentloaded", timeout=LISTING_LOAD_TIMEOUT_MS)

            # Wait for Cloudflare if needed
            self._wait_for_cloudflare()

            # Dismiss any popups
            self._dismiss_cookie_popup()

            # Wait for content to load
            time.sleep(0.5)

            # Check for error pages
            if "not found" in self.page.title().lower() or "404" in self.page.title():
                return None

            # Extract __NEXT_DATA__
            next_data = self._extract_next_data()

            # Extract all property details
            listing = self._extract_property_details(next_data)

            return listing

        except PlaywrightTimeout:
            self.logger.warning(f"Timeout loading listing: {url}")
            return None
        except Exception as e:
            self.logger.error(f"Error scraping listing {url}: {e}")
            return None

    def _init_csv_writer(self) -> tuple[Any, Any]:
        """Initialize CSV file and writer."""
        # Check if file exists for append mode
        file_exists = Path(self.output_file).exists()

        if self.checkpoint.data.processed_count > 0 and file_exists:
            # Append mode for resume
            f = open(self.output_file, 'a', newline='', encoding='utf-8-sig')
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        else:
            # New file
            f = open(self.output_file, 'w', newline='', encoding='utf-8-sig')
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
            writer.writeheader()

        return f, writer

    def _write_listing(self, writer: Any, listing: PropertyListing) -> None:
        """Write a listing to CSV."""
        writer.writerow(listing.to_dict())
        self.listings_written += 1

    def run(self, resume: bool = False, fresh: bool = False) -> None:
        """Main scraping loop."""
        # Handle checkpoint
        if fresh:
            self.checkpoint.clear()
        elif resume and self.checkpoint.exists():
            if not self.checkpoint.load():
                self.logger.error("Failed to load checkpoint, starting fresh")
                self.checkpoint.initialize_new_run(self.county)
        else:
            self.checkpoint.initialize_new_run(self.county)

        # Initialize browser
        self.logger.info("Starting browser...")
        self._init_browser()

        try:
            # Navigate to search page
            start_page = max(1, self.checkpoint.data.last_page)
            search_url = self._get_search_url(start_page)

            self.logger.info(f"Navigating to: {search_url}")
            self.page.goto(search_url, wait_until="domcontentloaded")

            # Wait for Cloudflare challenge to pass
            if not self._wait_for_cloudflare():
                self.logger.error("Failed to pass Cloudflare challenge. Try running with --no-headless and solve manually.")
                return

            # Dismiss cookie popup on first page
            self._dismiss_cookie_popup()
            time.sleep(1)

            # Get total listings count
            total = self._get_total_listings_count()
            if self.limit:
                total = min(total, self.limit)

            self.monitor.set_total(total)
            self.checkpoint.data.total_listings_found = total

            self.logger.info(f"Found approximately {total:,} sold listings")

            # Initialize CSV
            csv_file, csv_writer = self._init_csv_writer()

            try:
                current_page = start_page
                processed = self.checkpoint.data.processed_count

                # Main scraping loop
                while not self.shutdown_requested:
                    # Get listing URLs from current search page
                    listing_urls = self._extract_listing_urls_from_page()

                    if not listing_urls:
                        self.logger.info(f"No more listings found on page {current_page}")
                        # Try next page anyway in case of transient issue
                        current_page += 1
                        if current_page > 500:  # Safety limit
                            break
                        search_url = self._get_search_url(current_page)
                        self.page.goto(search_url, wait_until="domcontentloaded")
                        self._dismiss_cookie_popup()
                        time.sleep(1)
                        continue

                    self.logger.info(f"Page {current_page}: Found {len(listing_urls)} listings")

                    # Process each listing
                    for url in listing_urls:
                        if self.shutdown_requested:
                            break

                        # Check limit
                        if self.limit and processed >= self.limit:
                            self.logger.info(f"Reached limit of {self.limit} listings")
                            self.shutdown_requested = True
                            break

                        # Skip if already processed
                        if self.checkpoint.is_url_processed(url):
                            continue

                        # Scrape listing
                        listing = self.scrape_listing(url)

                        if listing and listing.address_full:
                            self._write_listing(csv_writer, listing)
                            self.monitor.update(listing, success=True)
                            self._adaptive_delay(success=True)
                        else:
                            self.monitor.update(None, success=False, error="Empty or failed listing")
                            self._adaptive_delay(success=False)

                        # Update checkpoint
                        processed += 1
                        self.checkpoint.update(current_page, url, processed)

                        # Save checkpoint periodically
                        if processed % self.checkpoint_interval == 0:
                            self.checkpoint.save()
                            csv_file.flush()

                        # Display progress
                        if self.monitor.should_display():
                            self.monitor.display()

                    # Move to next page
                    current_page += 1
                    search_url = self._get_search_url(current_page)

                    try:
                        self.page.goto(search_url, wait_until="domcontentloaded")
                        self._dismiss_cookie_popup()
                        time.sleep(1)
                    except Exception as e:
                        self.logger.error(f"Error navigating to page {current_page}: {e}")
                        break

            finally:
                # Final save
                self.checkpoint.save()
                csv_file.close()

                # Print final summary
                print(self.monitor.final_summary())
                print(f"\nOutput saved to: {self.output_file}")
                print(f"Total listings written: {self.listings_written}")

        finally:
            self._close_browser()


# ============================================================================
# CLI
# ============================================================================

def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Scrape sold property data from Daft.ie",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --out sold_listings.csv
  %(prog)s --county dublin --out dublin_sold.csv
  %(prog)s --limit 1000 --out sample.csv
  %(prog)s --out sold.csv --resume
  %(prog)s --out test.csv --no-headless --limit 10
        """,
    )

    parser.add_argument(
        "--out", "-o",
        required=True,
        help="Output CSV file path",
    )

    parser.add_argument(
        "--county", "-c",
        choices=list(COUNTY_SLUGS.keys()),
        help="Filter to specific county",
    )

    parser.add_argument(
        "--limit", "-l",
        type=int,
        help="Maximum number of listings to scrape",
    )

    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from checkpoint if available",
    )

    parser.add_argument(
        "--fresh", "-f",
        action="store_true",
        help="Start fresh, ignore any existing checkpoint",
    )

    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Show browser window (for debugging)",
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help=f"Save checkpoint every N records (default: {DEFAULT_CHECKPOINT_INTERVAL})",
    )

    parser.add_argument(
        "--monitor-interval",
        type=int,
        default=DEFAULT_MONITOR_INTERVAL,
        help=f"Update progress display every N records (default: {DEFAULT_MONITOR_INTERVAL})",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Validate args
    if args.resume and args.fresh:
        parser.error("Cannot use both --resume and --fresh")

    # Setup
    setup_logging(args.verbose)

    # Create scraper
    scraper = DaftSoldScraper(
        output_file=args.out,
        county=args.county,
        limit=args.limit,
        headless=not args.no_headless,
        checkpoint_interval=args.checkpoint_interval,
        monitor_interval=args.monitor_interval,
    )

    # Run
    try:
        scraper.run(resume=args.resume, fresh=args.fresh)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Checkpoint saved.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
