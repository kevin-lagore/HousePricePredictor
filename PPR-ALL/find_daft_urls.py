#!/usr/bin/env python3
"""
Daft.ie URL Finder
==================
Find Daft.ie listing URLs by searching for property addresses.

This script takes addresses (from PPR or manual input) and finds the corresponding
Daft.ie listing URL using multiple search strategies:
1. Google search: site:daft.ie "address"
2. Daft.ie internal search
3. DuckDuckGo fallback

Usage:
    # Search for a single address
    python find_daft_urls.py --address "123 Main Street, Dublin 4"

    # Process PPR CSV file and find Daft URLs for each address
    python find_daft_urls.py --ppr PPR-ALL.csv --out ppr_with_daft_urls.csv

    # Limit processing
    python find_daft_urls.py --ppr PPR-ALL.csv --out output.csv --limit 100

    # Resume from checkpoint
    python find_daft_urls.py --ppr PPR-ALL.csv --out output.csv --resume

Author: House Price Predictor Project
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote_plus, urljoin, urlparse

try:
    from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext, TimeoutError as PlaywrightTimeout
except ImportError:
    print("ERROR: playwright not installed. Run: pip install playwright && playwright install chromium")
    sys.exit(1)


# ============================================================================
# CONSTANTS
# ============================================================================

BASE_URL = "https://www.daft.ie"
GOOGLE_SEARCH_URL = "https://www.google.com/search"
DUCKDUCKGO_SEARCH_URL = "https://duckduckgo.com/"

# Daft URL patterns
DAFT_LISTING_PATTERN = re.compile(r'daft\.ie/(sold|for-sale)/[^/]+/(\d+)')
DAFT_URL_PATTERN = re.compile(r'https?://(?:www\.)?daft\.ie/(sold|for-sale)/[^\s"\'<>]+/\d+')

# Rate limiting
MIN_DELAY = 2.0
MAX_DELAY = 10.0
SEARCH_TIMEOUT_MS = 30000

# Checkpoint settings
CHECKPOINT_INTERVAL = 25


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SearchResult:
    """Result of a URL search."""
    address: str
    daft_url: Optional[str] = None
    search_method: Optional[str] = None  # 'google', 'duckduckgo', 'daft_search'
    confidence: float = 0.0
    error: Optional[str] = None
    searched_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "address": self.address,
            "daft_url": self.daft_url,
            "search_method": self.search_method,
            "confidence": self.confidence,
            "error": self.error,
            "searched_at": self.searched_at,
        }


@dataclass
class CheckpointData:
    """Checkpoint for resume functionality."""
    last_row: int = 0
    processed_count: int = 0
    found_count: int = 0
    started_at: str = ""
    last_saved_at: str = ""

    def to_dict(self) -> dict:
        return {
            "last_row": self.last_row,
            "processed_count": self.processed_count,
            "found_count": self.found_count,
            "started_at": self.started_at,
            "last_saved_at": self.last_saved_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CheckpointData:
        return cls(**data)


# ============================================================================
# URL FINDER CLASS
# ============================================================================

class DaftUrlFinder:
    """Find Daft.ie URLs for property addresses using web search."""

    def __init__(self, headless: bool = True):
        self.headless = headless
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.logger = logging.getLogger(__name__)
        self.current_delay = MIN_DELAY

    def _init_browser(self) -> None:
        """Initialize Playwright browser."""
        self.playwright = sync_playwright().start()

        # Use Firefox - better at avoiding detection
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
        self.page.set_default_timeout(SEARCH_TIMEOUT_MS)

    def _wait_for_cloudflare(self, max_wait: int = 30) -> bool:
        """Wait for Cloudflare challenge to complete."""
        start = time.time()

        while time.time() - start < max_wait:
            title = self.page.title().lower()
            content = self.page.content().lower()

            cf_indicators = [
                "just a moment" in title,
                "checking your browser" in content,
                "cloudflare" in content and "challenge" in content,
            ]

            if not any(cf_indicators):
                return True

            self.logger.debug("Waiting for Cloudflare...")
            time.sleep(2)

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

    def _normalize_address(self, address: str) -> str:
        """Normalize address for searching."""
        # Remove extra whitespace
        address = " ".join(address.split())

        # Remove common noise words that hurt search
        noise = ["Ireland", "Co.", "County"]
        for word in noise:
            address = address.replace(word, "")

        return address.strip()

    def _extract_daft_urls(self, page_content: str) -> list[str]:
        """Extract Daft.ie URLs from page content."""
        urls = DAFT_URL_PATTERN.findall(page_content)
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique.append(url)
        return unique

    def _extract_daft_urls_from_links(self) -> list[str]:
        """Extract Daft.ie URLs from page links."""
        urls = []
        try:
            links = self.page.locator('a[href*="daft.ie"]').all()
            for link in links:
                href = link.get_attribute("href")
                if href and DAFT_LISTING_PATTERN.search(href):
                    urls.append(href)
        except:
            pass

        # Also try to get URLs from Google's redirect links
        try:
            # Google wraps URLs - look in the visible text or data attributes
            links = self.page.locator('a').all()
            for link in links:
                try:
                    href = link.get_attribute("href") or ""
                    # Google uses /url?q= for redirects
                    if "daft.ie" in href:
                        # Extract actual URL from Google redirect
                        if "/url?q=" in href:
                            start = href.find("/url?q=") + 7
                            end = href.find("&", start)
                            if end == -1:
                                end = len(href)
                            actual_url = href[start:end]
                            if DAFT_LISTING_PATTERN.search(actual_url):
                                urls.append(actual_url)
                        elif DAFT_LISTING_PATTERN.search(href):
                            urls.append(href)
                except:
                    continue
        except:
            pass

        # Deduplicate
        seen = set()
        unique = []
        for url in urls:
            clean_url = url.split("&")[0] if "&" in url else url
            if clean_url not in seen:
                seen.add(clean_url)
                unique.append(clean_url)

        return unique

    def _dismiss_cookie_popup(self) -> None:
        """Dismiss cookie consent popups."""
        selectors = [
            'button[id*="accept"]',
            'button:has-text("Accept")',
            'button:has-text("Accept All")',
            'button:has-text("I agree")',
            '#L2AGLb',  # Google's accept button
            'button:has-text("Reject all")',  # Sometimes easier to reject
        ]

        for selector in selectors:
            try:
                button = self.page.locator(selector).first
                if button.is_visible(timeout=1000):
                    button.click(timeout=2000)
                    time.sleep(0.3)
                    return
            except:
                continue

    def search_google(self, address: str) -> Optional[str]:
        """Search Google for Daft.ie URL."""
        try:
            # Build search query
            query = f'site:daft.ie "{address}"'
            search_url = f"{GOOGLE_SEARCH_URL}?q={quote_plus(query)}"

            self.logger.debug(f"Google search: {query}")
            self.page.goto(search_url, wait_until="domcontentloaded")

            # Handle cookie popup
            self._dismiss_cookie_popup()
            time.sleep(1)

            # Check for CAPTCHA
            if "unusual traffic" in self.page.content().lower():
                self.logger.warning("Google CAPTCHA detected")
                return None

            # Extract URLs from search results
            urls = self._extract_daft_urls_from_links()

            if urls:
                # Return first (most relevant) result
                return urls[0]

            return None

        except PlaywrightTimeout:
            self.logger.warning("Google search timeout")
            return None
        except Exception as e:
            self.logger.error(f"Google search error: {e}")
            return None

    def search_duckduckgo(self, address: str) -> Optional[str]:
        """Search DuckDuckGo for Daft.ie URL (fallback)."""
        try:
            query = f'site:daft.ie "{address}"'
            search_url = f"{DUCKDUCKGO_SEARCH_URL}?q={quote_plus(query)}"

            self.logger.debug(f"DuckDuckGo search: {query}")
            self.page.goto(search_url, wait_until="domcontentloaded")
            time.sleep(2)  # DDG needs more time to load results

            # Extract URLs
            urls = self._extract_daft_urls_from_links()

            # Also check page content
            if not urls:
                content = self.page.content()
                urls = self._extract_daft_urls(content)

            if urls:
                return urls[0]

            return None

        except Exception as e:
            self.logger.error(f"DuckDuckGo search error: {e}")
            return None

    def search_daft_directly(self, address: str) -> Optional[str]:
        """Search Daft.ie directly."""
        try:
            # Daft's search URL
            query = quote_plus(address)
            search_url = f"{BASE_URL}/property-for-sale/ireland?searchSource=search&terms={query}"

            self.logger.debug(f"Daft direct search: {address}")
            self.page.goto(search_url, wait_until="domcontentloaded")

            # Dismiss cookie popup
            self._dismiss_cookie_popup()
            time.sleep(1.5)

            # Look for listing links
            try:
                # Wait for results
                self.page.wait_for_selector('a[href*="/for-sale/"], a[href*="/sold/"]', timeout=10000)
            except:
                pass

            # Extract listing URLs
            links = self.page.locator('a[href*="/for-sale/"], a[href*="/sold/"]').all()

            for link in links:
                try:
                    href = link.get_attribute("href")
                    if href and re.search(r'/\d+/?$', href):
                        return urljoin(BASE_URL, href)
                except:
                    continue

            return None

        except Exception as e:
            self.logger.error(f"Daft search error: {e}")
            return None

    def find_url(self, address: str) -> SearchResult:
        """Find Daft.ie URL for an address using multiple methods."""
        result = SearchResult(
            address=address,
            searched_at=datetime.now().isoformat()
        )

        normalized = self._normalize_address(address)

        # Strategy 1: Google search (most reliable)
        url = self.search_google(normalized)
        if url:
            result.daft_url = url
            result.search_method = "google"
            result.confidence = 0.9
            return result

        time.sleep(self.current_delay)

        # Strategy 2: DuckDuckGo fallback
        url = self.search_duckduckgo(normalized)
        if url:
            result.daft_url = url
            result.search_method = "duckduckgo"
            result.confidence = 0.85
            return result

        time.sleep(self.current_delay)

        # Strategy 3: Daft.ie direct search
        url = self.search_daft_directly(normalized)
        if url:
            result.daft_url = url
            result.search_method = "daft_search"
            result.confidence = 0.7  # Lower confidence - may not be exact match
            return result

        result.error = "No URL found with any method"
        return result

    def process_ppr_file(
        self,
        ppr_file: str,
        output_file: str,
        limit: Optional[int] = None,
        resume: bool = False,
    ) -> None:
        """Process PPR CSV file and find Daft URLs for each address."""

        # Checkpoint handling
        checkpoint_file = Path(output_file).with_suffix('.checkpoint.json')
        checkpoint = CheckpointData()
        start_row = 0

        if resume and checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                checkpoint = CheckpointData.from_dict(json.load(f))
            start_row = checkpoint.last_row
            self.logger.info(f"Resuming from row {start_row}")
        else:
            checkpoint.started_at = datetime.now().isoformat()

        # Initialize browser
        self._init_browser()

        try:
            # Read PPR file
            with open(ppr_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            total_rows = len(rows)
            if limit:
                total_rows = min(total_rows, start_row + limit)

            self.logger.info(f"Processing {total_rows - start_row} addresses")

            # Open output file
            output_exists = Path(output_file).exists() and resume
            mode = 'a' if output_exists else 'w'

            with open(output_file, mode, newline='', encoding='utf-8-sig') as f:
                # Determine fieldnames - PPR fields + our new fields
                ppr_fields = list(rows[0].keys()) if rows else []
                new_fields = ["daft_url", "search_method", "search_confidence", "searched_at"]
                fieldnames = ppr_fields + new_fields

                writer = csv.DictWriter(f, fieldnames=fieldnames)

                if not output_exists:
                    writer.writeheader()

                # Process each row
                for i, row in enumerate(rows[start_row:total_rows], start=start_row):
                    # Build address from PPR fields
                    address_parts = [
                        row.get("Address", ""),
                        row.get("County", ""),
                    ]
                    address = ", ".join(p for p in address_parts if p)

                    if not address.strip():
                        continue

                    # Find URL
                    result = self.find_url(address)

                    # Merge PPR row with search result
                    output_row = dict(row)
                    output_row["daft_url"] = result.daft_url or ""
                    output_row["search_method"] = result.search_method or ""
                    output_row["search_confidence"] = result.confidence
                    output_row["searched_at"] = result.searched_at

                    writer.writerow(output_row)

                    # Update stats
                    checkpoint.processed_count += 1
                    checkpoint.last_row = i + 1
                    if result.daft_url:
                        checkpoint.found_count += 1

                    # Progress
                    if checkpoint.processed_count % 10 == 0:
                        found_pct = (checkpoint.found_count / checkpoint.processed_count) * 100
                        self.logger.info(
                            f"Progress: {checkpoint.processed_count}/{total_rows - start_row} | "
                            f"Found: {checkpoint.found_count} ({found_pct:.1f}%)"
                        )

                    # Save checkpoint
                    if checkpoint.processed_count % CHECKPOINT_INTERVAL == 0:
                        checkpoint.last_saved_at = datetime.now().isoformat()
                        with open(checkpoint_file, 'w') as cf:
                            json.dump(checkpoint.to_dict(), cf)
                        f.flush()

                    # Rate limiting
                    time.sleep(self.current_delay)

            # Final stats
            found_pct = (checkpoint.found_count / max(checkpoint.processed_count, 1)) * 100
            self.logger.info(f"\nComplete! Processed: {checkpoint.processed_count}, Found: {checkpoint.found_count} ({found_pct:.1f}%)")

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
    parser = argparse.ArgumentParser(
        description="Find Daft.ie URLs for property addresses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --address "123 Main Street, Dublin 4"
  %(prog)s --ppr PPR-ALL.csv --out ppr_with_urls.csv
  %(prog)s --ppr PPR-ALL.csv --out output.csv --limit 100
  %(prog)s --ppr PPR-ALL.csv --out output.csv --resume
        """,
    )

    parser.add_argument(
        "--address", "-a",
        help="Single address to search for",
    )

    parser.add_argument(
        "--ppr", "-p",
        help="PPR CSV file to process",
    )

    parser.add_argument(
        "--out", "-o",
        help="Output CSV file (required with --ppr)",
    )

    parser.add_argument(
        "--limit", "-l",
        type=int,
        help="Maximum number of addresses to process",
    )

    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from checkpoint",
    )

    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Show browser window",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Validate args
    if not args.address and not args.ppr:
        parser.error("Must provide either --address or --ppr")

    if args.ppr and not args.out:
        parser.error("--out required when using --ppr")

    setup_logging(args.verbose)

    finder = DaftUrlFinder(headless=not args.no_headless)

    if args.address:
        # Single address search
        finder._init_browser()
        try:
            result = finder.find_url(args.address)
            print(f"\nAddress: {result.address}")
            print(f"Daft URL: {result.daft_url or 'Not found'}")
            print(f"Method: {result.search_method or 'N/A'}")
            print(f"Confidence: {result.confidence}")
        finally:
            finder._close_browser()

    elif args.ppr:
        # Batch process PPR file
        finder.process_ppr_file(
            ppr_file=args.ppr,
            output_file=args.out,
            limit=args.limit,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
