#!/usr/bin/env python3
"""
Daft Property Data Extraction Tool

Extracts detailed listing information from archived or local HTML files.
Designed for property price prediction with 200k+ records.

==============================================================================
TWO-STEP WORKFLOW
==============================================================================

STEP 1: Match PPR addresses to Daft URLs
  python extract_daft_data.py --match --in PPR-ALL.csv --ppr_format --out matched.csv

  This searches the Wayback Machine for Daft.ie listings matching each PPR
  address. Outputs a CSV with daft_url and match_confidence columns.

STEP 2: Extract data from matched listings
  python extract_daft_data.py --extract --in matched.csv --wayback --out enriched.csv

  This downloads HTML from Wayback Machine for each matched URL and extracts
  property features (beds, baths, size, BER rating, amenities, etc.)

==============================================================================
SUPPORTED INPUT FORMATS
==============================================================================

Irish Property Price Register (PPR) format (--ppr_format):
- Date of Sale (dd/mm/yyyy)
- Address
- County
- Eircode
- Price (€)
- Not Full Market Price
- VAT Exclusive
- Description of Property
- Property Size Description

Generic format:
- address (required)
- daft_url (required for extraction, populated by --match step)
- sale_date (optional, YYYY-MM-DD or dd/mm/yyyy)
- selling_price (optional; actual sale price for training data)
- local_html (optional; filename inside --local_html_dir)

==============================================================================
OUTPUTS
==============================================================================

- matched.csv (Step 1: addresses with matched Daft URLs)
- enriched.csv (Step 2: all extracted data)
- enriched.jsonl (Step 2: same data in JSON Lines format)
- *_run.log (monitoring log with timestamps and progress)
- html_cache/ (downloaded HTML files for offline use)
- match_cache/ (cached URL lookups to speed up re-runs)

==============================================================================
EXAMPLES
==============================================================================

  # Full workflow with PPR-ALL.csv
  python extract_daft_data.py --match --in PPR-ALL.csv --ppr_format --out matched.csv
  python extract_daft_data.py --extract --in matched.csv --wayback --out enriched.csv

  # Extract from local HTML files (if you already have them)
  python extract_daft_data.py --extract --in matched.csv --local_html_dir saved_pages/ --out enriched.csv

  # Resume interrupted extraction
  python extract_daft_data.py --extract --in matched.csv --wayback --out enriched.csv --resume enriched_checkpoint.json

==============================================================================
MONITORING
==============================================================================

When running in PowerShell/cmd, you'll see:
- Real-time progress bar with ETA
- Success/failure counts
- Rate (records/second)
- Graceful Ctrl+C handling (saves checkpoint)
- Final summary with top errors

Install:
  pip install requests beautifulsoup4 lxml python-dateutil colorama
"""

from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import json
import logging
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import requests
from bs4 import BeautifulSoup
from dateutil.parser import parse as dateparse

# Optional colorama for Windows console colors
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False
    class Fore:
        GREEN = RED = YELLOW = CYAN = WHITE = RESET = ""
    class Style:
        BRIGHT = RESET_ALL = ""


WAYBACK_CDX = "https://web.archive.org/cdx/search/cdx"
WAYBACK_PREFIX = "https://web.archive.org/web/"

# Global for graceful shutdown
SHUTDOWN_REQUESTED = False


# ============================================================================
# MONITORING & LOGGING
# ============================================================================

class RunMonitor:
    """
    Real-time monitoring for long-running extraction jobs.
    Provides console output, file logging, and progress tracking.
    """

    def __init__(self, log_file: Path, total_records: int, verbose: bool = True):
        self.log_file = log_file
        self.total = total_records
        self.processed = 0
        self.success = 0
        self.failed = 0
        self.wayback_hits = 0
        self.local_hits = 0
        self.no_source = 0
        self.start_time = time.time()
        self.last_update = time.time()
        self.verbose = verbose
        self.errors: List[str] = []

        # Setup file logging
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout) if verbose else logging.NullHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        self._log_start()

    def _log_start(self):
        """Log run start."""
        msg = f"{'='*60}\nEXTRACTION STARTED\n{'='*60}"
        msg += f"\nTotal records to process: {self.total:,}"
        msg += f"\nLog file: {self.log_file}"
        msg += f"\nStart time: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        msg += f"\n{'='*60}"
        self.logger.info(msg)

    def update(self, success: bool, source: str, error: str = ""):
        """Update progress after processing a record."""
        self.processed += 1

        if success:
            self.success += 1
        else:
            self.failed += 1
            if error:
                self.errors.append(error)

        if source == "wayback":
            self.wayback_hits += 1
        elif source == "local_html":
            self.local_hits += 1
        elif source == "":
            self.no_source += 1

        # Print progress every 100 records or every 30 seconds
        now = time.time()
        if self.processed % 100 == 0 or (now - self.last_update) > 30:
            self._print_progress()
            self.last_update = now

    def _print_progress(self):
        """Print current progress to console."""
        elapsed = time.time() - self.start_time
        rate = self.processed / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.processed) / rate if rate > 0 else 0

        pct = (self.processed / self.total * 100) if self.total > 0 else 0

        # Build status line
        status = (
            f"{Fore.CYAN}[{pct:5.1f}%]{Style.RESET_ALL} "
            f"Processed: {self.processed:,}/{self.total:,} | "
            f"{Fore.GREEN}OK: {self.success:,}{Style.RESET_ALL} | "
            f"{Fore.RED}Fail: {self.failed:,}{Style.RESET_ALL} | "
            f"Rate: {rate:.1f}/s | "
            f"ETA: {self._format_time(remaining)}"
        )

        if self.verbose:
            # Clear line and print status (works in PowerShell/cmd)
            print(f"\r{status}", end="", flush=True)

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"

    def log_error(self, address: str, error: str):
        """Log an error for a specific record."""
        self.logger.error(f"Failed: {address[:50]}... | Error: {error}")

    def log_checkpoint(self, checkpoint_file: Path, results: List[Dict]):
        """Save checkpoint for resume capability."""
        self.logger.info(f"Saving checkpoint: {len(results):,} records")
        with checkpoint_file.open("w", encoding="utf-8") as f:
            json.dump(results, f)

    def finalize(self):
        """Print final summary."""
        elapsed = time.time() - self.start_time

        print()  # New line after progress bar
        summary = f"""
{'='*60}
{Fore.GREEN}EXTRACTION COMPLETE{Style.RESET_ALL}
{'='*60}
Total processed: {self.processed:,}
  ✓ Successful:  {self.success:,} ({self.success/self.processed*100:.1f}%)
  ✗ Failed:      {self.failed:,} ({self.failed/self.processed*100:.1f}%)

Data sources:
  Wayback Machine: {self.wayback_hits:,}
  Local HTML:      {self.local_hits:,}
  No source:       {self.no_source:,}

Runtime: {self._format_time(elapsed)}
Average rate: {self.processed/elapsed:.1f} records/second
{'='*60}
"""
        print(summary)
        self.logger.info(summary)

        # Log top errors
        if self.errors:
            error_counts = {}
            for e in self.errors:
                error_counts[e] = error_counts.get(e, 0) + 1
            top_errors = sorted(error_counts.items(), key=lambda x: -x[1])[:5]

            print(f"\n{Fore.YELLOW}Top errors:{Style.RESET_ALL}")
            for err, count in top_errors:
                print(f"  {count:,}x: {err}")


def signal_handler(_signum, _frame):
    """Handle Ctrl+C gracefully."""
    global SHUTDOWN_REQUESTED
    SHUTDOWN_REQUESTED = True
    print(f"\n{Fore.YELLOW}Shutdown requested. Saving progress...{Style.RESET_ALL}")


# ============================================================================
# PPR FORMAT HANDLING
# ============================================================================

def normalize_ppr_row(row: Dict[str, str]) -> Dict[str, str]:
    """
    Convert PPR-ALL.csv format to standard format.

    PPR columns:
    - Date of Sale (dd/mm/yyyy) -> sale_date
    - Address -> address
    - County -> county
    - Eircode -> eircode
    - Price (€) -> selling_price
    - Not Full Market Price -> not_full_market_price
    - VAT Exclusive -> vat_exclusive
    - Description of Property -> property_description
    - Property Size Description -> size_description
    """
    normalized = {}

    # Map PPR columns to standard names
    col_map = {
        "Date of Sale (dd/mm/yyyy)": "sale_date",
        "Address": "address",
        "County": "county",
        "Eircode": "eircode",
        "Not Full Market Price": "not_full_market_price",
        "VAT Exclusive": "vat_exclusive",
        "Description of Property": "property_description",
        "Property Size Description": "size_description",
    }

    for ppr_col, std_col in col_map.items():
        if ppr_col in row:
            normalized[std_col] = row[ppr_col]

    # Handle price with € symbol and different formats
    price_col = None
    for col in row.keys():
        if "Price" in col or "price" in col:
            price_col = col
            break

    if price_col:
        price_str = row[price_col]
        # Remove €, commas, spaces and extract number
        clean = re.sub(r'[€,\s]', '', price_str)
        match = re.search(r'[\d.]+', clean)
        if match:
            normalized["selling_price"] = match.group(0)

    # Parse Irish date format (dd/mm/yyyy)
    if "sale_date" in normalized:
        date_str = normalized["sale_date"]
        if "/" in date_str:
            try:
                # Parse dd/mm/yyyy explicitly
                parts = date_str.split("/")
                if len(parts) == 3:
                    d, m, y = parts
                    normalized["sale_date"] = f"{y}-{m.zfill(2)}-{d.zfill(2)}"
            except:
                pass

    # Generate potential Daft URL from address
    # This is a heuristic - actual URL matching would need a database
    address = normalized.get("address", "")
    if address:
        # Create a search-friendly slug
        slug = re.sub(r'[^\w\s-]', '', address.lower())
        slug = re.sub(r'\s+', '-', slug.strip())
        # Daft URLs typically look like: daft.ie/for-sale/...
        normalized["daft_url_hint"] = f"https://www.daft.ie/for-sale/{slug}"

    # Copy any remaining columns
    for col, val in row.items():
        if col not in col_map and "Price" not in col:
            key = col.lower().replace(" ", "_").replace("(", "").replace(")", "")
            if key not in normalized:
                normalized[key] = val

    return normalized


# ============================================================================
# HELPERS
# ============================================================================

def norm(s: str) -> str:
    """Normalize text: lowercase, collapse whitespace."""
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def sha1(s: str) -> str:
    """Generate SHA-1 hash for string."""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def safe_date(s: str) -> Optional[dt.date]:
    """Parse date string, return None on failure."""
    if not s:
        return None
    try:
        return dateparse(s).date()
    except Exception:
        return None


def text_or_none(el) -> Optional[str]:
    """Extract text from BeautifulSoup element, None if empty."""
    if not el:
        return None
    t = el.get_text(" ", strip=True)
    return t if t else None


def count_matches(pattern: str, text: str, flags=0) -> int:
    """Count regex pattern matches in text."""
    return len(re.findall(pattern, text, flags=flags))


# ============================================================================
# ADDRESS MATCHING - PPR to Daft URL
# ============================================================================

class AddressMatcher:
    """
    Match PPR addresses to Daft listing URLs via Wayback Machine.

    Strategy:
    1. Normalize the PPR address into search terms
    2. Query Wayback CDX for daft.ie URLs containing those terms
    3. Score candidates by address similarity
    4. Return best match with confidence score
    """

    # Common Irish address abbreviations
    ABBREVIATIONS = {
        'rd': 'road', 'rd.': 'road',
        'st': 'street', 'st.': 'street',
        'ave': 'avenue', 'ave.': 'avenue',
        'dr': 'drive', 'dr.': 'drive',
        'ln': 'lane', 'ln.': 'lane',
        'pk': 'park', 'pk.': 'park',
        'sq': 'square', 'sq.': 'square',
        'cres': 'crescent', 'cres.': 'crescent',
        'ct': 'court', 'ct.': 'court',
        'pl': 'place', 'pl.': 'place',
        'tce': 'terrace', 'tce.': 'terrace',
        'gdns': 'gardens', 'gdns.': 'gardens',
        'hts': 'heights', 'hts.': 'heights',
        'est': 'estate', 'est.': 'estate',
        'co': 'county', 'co.': 'county',
    }

    # Irish county names for extraction
    COUNTIES = [
        'antrim', 'armagh', 'carlow', 'cavan', 'clare', 'cork', 'derry',
        'donegal', 'down', 'dublin', 'fermanagh', 'galway', 'kerry',
        'kildare', 'kilkenny', 'laois', 'leitrim', 'limerick', 'longford',
        'louth', 'mayo', 'meath', 'monaghan', 'offaly', 'roscommon',
        'sligo', 'tipperary', 'tyrone', 'waterford', 'westmeath',
        'wexford', 'wicklow'
    ]

    def __init__(self, cache_dir: Optional[Path] = None, timeout: int = 30):
        self.timeout = timeout
        self.cache_dir = cache_dir
        self.cache: Dict[str, List[Dict]] = {}

        # Load cache if exists
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_file = cache_dir / "url_match_cache.json"
            if self.cache_file.exists():
                try:
                    with self.cache_file.open("r", encoding="utf-8") as f:
                        self.cache = json.load(f)
                except:
                    self.cache = {}

    def normalize_address(self, address: str) -> Dict[str, Any]:
        """
        Parse and normalize an Irish address into components.

        Returns dict with:
        - tokens: list of normalized words
        - number: house/unit number if found
        - county: county name if found
        - area: locality/area name
        - slug: URL-friendly version
        """
        addr = address.lower().strip()

        # Expand abbreviations
        for abbr, full in self.ABBREVIATIONS.items():
            addr = re.sub(rf'\b{re.escape(abbr)}\b', full, addr)

        # Extract house number
        number_match = re.match(r'^(\d+[a-z]?)\s+', addr)
        number = number_match.group(1) if number_match else None
        if number:
            addr = addr[len(number_match.group(0)):]

        # Extract county
        county = None
        for c in self.COUNTIES:
            if re.search(rf'\b{c}\b', addr):
                county = c
                break

        # Remove county reference for cleaner matching
        addr_clean = re.sub(r',?\s*co\.?\s*\w+$', '', addr, flags=re.I)
        addr_clean = re.sub(r',?\s*county\s+\w+$', '', addr_clean, flags=re.I)

        # Tokenize
        tokens = re.findall(r'\b[a-z0-9]+\b', addr_clean)
        tokens = [t for t in tokens if len(t) > 1]  # Skip single chars

        # Create URL slug
        slug = '-'.join(tokens[:6])  # First 6 tokens for URL

        return {
            'original': address,
            'tokens': tokens,
            'number': number,
            'county': county,
            'slug': slug
        }

    def search_wayback_for_address(self, address_info: Dict,
                                    target_date: Optional[dt.date] = None) -> List[Dict]:
        """
        Search Wayback Machine CDX for Daft URLs matching this address.

        Returns list of candidate URLs with metadata.
        """
        slug = address_info['slug']
        cache_key = f"{slug}_{target_date.isoformat() if target_date else 'any'}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        # Build search URL pattern
        # Daft URLs look like: daft.ie/for-sale/house-1-example-road-dublin/123456
        search_patterns = [
            f"daft.ie/for-sale/*{slug}*",
            f"daft.ie/*{slug}*",
        ]

        # If we have a house number, try more specific patterns
        if address_info['number']:
            num = address_info['number']
            specific_slug = f"{num}-{slug}"
            search_patterns.insert(0, f"daft.ie/for-sale/*{specific_slug}*")

        candidates = []

        for pattern in search_patterns[:2]:  # Limit API calls
            params = {
                "url": pattern,
                "matchType": "domain",
                "output": "json",
                "fl": "timestamp,original,statuscode",
                "filter": "statuscode:200",
                "collapse": "urlkey",
                "limit": "100",
            }

            if target_date:
                # Search ±2 years around sale date
                start = target_date - dt.timedelta(days=730)
                end = target_date + dt.timedelta(days=365)
                params["from"] = start.strftime("%Y%m%d")
                params["to"] = end.strftime("%Y%m%d")

            try:
                r = requests.get(WAYBACK_CDX, params=params, timeout=self.timeout,
                               headers={"User-Agent": "property-research/1.0"})
                if r.status_code == 200:
                    data = r.json()
                    if isinstance(data, list) and len(data) > 1:
                        for row in data[1:]:  # Skip header
                            if len(row) >= 2:
                                candidates.append({
                                    'timestamp': row[0],
                                    'url': row[1],
                                    'pattern': pattern
                                })
            except Exception:
                continue

            # If we found candidates with the first pattern, don't need more
            if candidates:
                break

        # Cache results
        self.cache[cache_key] = candidates

        return candidates

    def score_url_match(self, url: str, address_info: Dict) -> float:
        """
        Score how well a Daft URL matches the address.
        Returns 0.0-1.0 confidence score.
        """
        url_lower = url.lower()

        # Extract the address part from URL
        # e.g., /for-sale/house-1-example-road-dublin/123456
        match = re.search(r'/for-sale/([^/]+)/', url_lower)
        if not match:
            match = re.search(r'/([^/]+)/\d+$', url_lower)

        if not match:
            return 0.0

        url_slug = match.group(1)
        url_tokens = set(re.findall(r'[a-z0-9]+', url_slug))
        addr_tokens = set(address_info['tokens'])

        if not addr_tokens:
            return 0.0

        # Calculate Jaccard similarity
        intersection = len(url_tokens & addr_tokens)
        union = len(url_tokens | addr_tokens)
        jaccard = intersection / union if union > 0 else 0

        # Bonus for matching house number
        score = jaccard
        if address_info['number']:
            if address_info['number'] in url_tokens:
                score += 0.2

        # Bonus for matching county
        if address_info['county']:
            if address_info['county'] in url_tokens:
                score += 0.1

        return min(score, 1.0)

    def find_best_match(self, address: str, sale_date: Optional[dt.date] = None,
                        min_confidence: float = 0.4) -> Optional[Dict]:
        """
        Find the best matching Daft URL for a PPR address.

        Returns dict with:
        - url: matched Daft URL
        - confidence: match score (0-1)
        - timestamp: Wayback timestamp
        - wayback_url: full Wayback URL

        Returns None if no match above min_confidence.
        """
        address_info = self.normalize_address(address)

        if not address_info['tokens']:
            return None

        candidates = self.search_wayback_for_address(address_info, sale_date)

        if not candidates:
            return None

        # Score all candidates
        scored = []
        for cand in candidates:
            score = self.score_url_match(cand['url'], address_info)
            if score >= min_confidence:
                scored.append({
                    'url': cand['url'],
                    'confidence': score,
                    'timestamp': cand['timestamp'],
                    'wayback_url': f"{WAYBACK_PREFIX}{cand['timestamp']}/{cand['url']}"
                })

        if not scored:
            return None

        # Return best match
        scored.sort(key=lambda x: (-x['confidence'], x['timestamp']))
        return scored[0]

    def save_cache(self):
        """Persist the URL match cache to disk."""
        if self.cache_dir and self.cache:
            with self.cache_file.open("w", encoding="utf-8") as f:
                json.dump(self.cache, f)


def match_ppr_addresses(input_csv: Path, output_csv: Path,
                        ppr_format: bool = True,
                        min_confidence: float = 0.4,
                        timeout: int = 30,
                        verbose: bool = True):
    """
    Step 1: Match PPR addresses to Daft URLs.

    Reads PPR-ALL.csv, searches Wayback Machine for matching Daft listings,
    and outputs a CSV with daft_url column populated.

    Args:
        input_csv: Input PPR CSV file
        output_csv: Output CSV with matched URLs
        ppr_format: Input is PPR-ALL.csv format
        min_confidence: Minimum match confidence (0-1)
        timeout: Request timeout in seconds
        verbose: Print progress
    """
    global SHUTDOWN_REQUESTED

    signal.signal(signal.SIGINT, signal_handler)

    # Read input - try multiple encodings
    rows = None
    for encoding in ["utf-8-sig", "utf-8", "latin-1", "cp1252"]:
        try:
            with input_csv.open("r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            break
        except UnicodeDecodeError:
            continue

    if rows is None:
        print(f"{Fore.RED}Error: Could not read {input_csv} with any encoding{Style.RESET_ALL}")
        return 0

    if ppr_format:
        rows = [normalize_ppr_row(row) for row in rows]

    # Setup matcher with caching
    cache_dir = output_csv.parent / "match_cache"
    matcher = AddressMatcher(cache_dir=cache_dir, timeout=timeout)

    # Setup monitoring
    log_file = output_csv.parent / f"{output_csv.stem}_match.log"
    monitor = RunMonitor(log_file, len(rows), verbose=verbose)

    results = []
    matched_count = 0

    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"ADDRESS MATCHING - PPR to Daft URLs")
    print(f"{'='*60}{Style.RESET_ALL}")
    print(f"Input:  {input_csv} ({len(rows):,} records)")
    print(f"Output: {output_csv}")
    print(f"Min confidence: {min_confidence}")
    print(f"{'='*60}\n")

    for idx, row in enumerate(rows):
        if SHUTDOWN_REQUESTED:
            print(f"\n{Fore.YELLOW}Shutdown requested, saving progress...{Style.RESET_ALL}")
            break

        address = row.get("address", "")
        sale_date = safe_date(row.get("sale_date", ""))

        result = dict(row)
        result["daft_url"] = ""
        result["match_confidence"] = 0.0
        result["wayback_url"] = ""
        result["match_status"] = "not_found"

        if address:
            try:
                match = matcher.find_best_match(address, sale_date, min_confidence)

                if match:
                    result["daft_url"] = match["url"]
                    result["match_confidence"] = round(match["confidence"], 3)
                    result["wayback_url"] = match["wayback_url"]
                    result["match_status"] = "matched"
                    matched_count += 1
                    monitor.update(success=True, source="wayback")
                else:
                    result["match_status"] = "no_match"
                    monitor.update(success=False, source="", error="no_match")

            except Exception as e:
                result["match_status"] = f"error: {e}"
                monitor.update(success=False, source="", error=str(e))
        else:
            result["match_status"] = "no_address"
            monitor.update(success=False, source="", error="no_address")

        results.append(result)

        # Rate limiting - be nice to Wayback Machine
        if idx > 0 and idx % 10 == 0:
            time.sleep(0.5)

    # Save cache
    matcher.save_cache()

    # Write output
    if results:
        fieldnames = list(results[0].keys())
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    # Summary
    print(f"\n{'='*60}")
    print(f"{Fore.GREEN}MATCHING COMPLETE{Style.RESET_ALL}")
    print(f"{'='*60}")
    print(f"Total processed: {len(results):,}")
    print(f"Successfully matched: {matched_count:,} ({matched_count/len(results)*100:.1f}%)")
    print(f"No match found: {len(results) - matched_count:,}")
    print(f"\nOutput: {output_csv}")
    print(f"{'='*60}\n")

    return matched_count


# ============================================================================
# WAYBACK MACHINE FUNCTIONS
# ============================================================================

def pick_best_snapshot(url: str, target_date: Optional[dt.date], 
                       timeout_s: int = 30) -> Optional[Dict[str, str]]:
    """
    Find closest Wayback Machine snapshot to target_date.
    If no target_date, returns most recent snapshot.
    
    Returns dict with: timestamp, original, statuscode, mimetype
    """
    params = {
        "url": url,
        "output": "json",
        "fl": "timestamp,original,statuscode,mimetype",
        "filter": "statuscode:200",
        "collapse": "digest",
        "limit": "50",
    }
    
    if target_date:
        # Search within ±1 year of target date
        start = target_date - dt.timedelta(days=365)
        end = target_date + dt.timedelta(days=365)
        params["from"] = start.strftime("%Y%m%d")
        params["to"] = end.strftime("%Y%m%d")
    
    try:
        r = requests.get(WAYBACK_CDX, params=params, timeout=timeout_s,
                        headers={"User-Agent": "property-research/1.0"})
        if r.status_code != 200:
            return None
        
        data = r.json()
        if not isinstance(data, list) or len(data) < 2:
            return None
        
        rows = data[1:]  # Skip header row
        
        if target_date:
            # Find snapshot closest to target date
            tgt = int(target_date.strftime("%Y%m%d000000"))
            def distance(timestamp: str) -> int:
                try:
                    return abs(int(timestamp) - tgt)
                except ValueError:
                    return 10**18
            rows.sort(key=lambda x: distance(x[0]))
        else:
            # Get most recent
            rows.sort(key=lambda x: x[0], reverse=True)
        
        ts, orig, status, mime = rows[0]
        return {
            "timestamp": ts,
            "original": orig,
            "status": status,
            "mimetype": mime
        }
        
    except Exception as e:
        return None


def fetch_wayback_html(snapshot_ts: str, original_url: str, 
                       out_path: Path, timeout_s: int = 30) -> Tuple[Optional[str], Optional[str]]:
    """
    Download HTML from Wayback Machine.
    
    Returns: (archive_url, html_text)
    Saves raw bytes to out_path for archival purposes.
    """
    archive_url = f"{WAYBACK_PREFIX}{snapshot_ts}/{original_url}"
    
    try:
        r = requests.get(archive_url, timeout=timeout_s,
                        headers={"User-Agent": "property-research/1.0"})
        if r.status_code != 200:
            return None, None
        
        # Save raw bytes
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(r.content)
        
        # Return text
        try:
            return archive_url, r.text
        except:
            return archive_url, r.content.decode("utf-8", errors="replace")
        
    except Exception:
        return None, None


# ============================================================================
# DATA EXTRACTION
# ============================================================================

def extract_price_value(price_str: Optional[str]) -> Optional[float]:
    """Extract numeric price from string like '€450,000' or '€1.2m'."""
    if not price_str:
        return None
    
    # Remove €, commas, spaces
    clean = re.sub(r'[€,\s]', '', price_str)
    
    # Handle 'm' for millions
    if 'm' in clean.lower():
        match = re.search(r'([\d.]+)m', clean.lower())
        if match:
            return float(match.group(1)) * 1_000_000
    
    # Extract numeric value
    match = re.search(r'[\d.]+', clean)
    if match:
        try:
            return float(match.group(0))
        except:
            return None
    
    return None


def extract_eircode(text: str) -> Optional[str]:
    """Extract Eircode from text."""
    # Irish Eircode format: A65 F4E2 (letter+2digits space letter+digit+letter+digit)
    match = re.search(r'\b[A-Z]\d{2}\s*[A-Z0-9]{4}\b', text, flags=re.IGNORECASE)
    return match.group(0).upper().replace(' ', '') if match else None


def extract_property_type(soup: BeautifulSoup) -> Optional[str]:
    """Extract property type: house, apartment, duplex, etc."""
    text = soup.get_text(" ", strip=True).lower()
    
    types = {
        'apartment': r'\bapartment\b|\bflat\b',
        'house': r'\bhouse\b|\bdetached\b|\bsemi-detached\b|\bterraced\b',
        'duplex': r'\bduplex\b',
        'bungalow': r'\bbungalow\b',
        'townhouse': r'\btownhouse\b',
        'studio': r'\bstudio\b'
    }
    
    for ptype, pattern in types.items():
        if re.search(pattern, text):
            return ptype
    
    return None


def extract_year_built(text: str) -> Optional[int]:
    """Extract year built from text."""
    # Look for patterns like "built in 2005" or "built 2005"
    match = re.search(r'\bbuilt\s+(?:in\s+)?(\d{4})\b', text, flags=re.IGNORECASE)
    if match:
        year = int(match.group(1))
        if 1800 <= year <= dt.datetime.now().year:
            return year
    
    return None


def extract_listing_fields(html: str) -> Dict[str, Any]:
    """
    Extract all relevant fields from Daft listing HTML.
    Optimized for property price prediction.
    """
    soup = BeautifulSoup(html, "lxml")
    out = {}
    
    # Get full body text for regex searches
    body_text = soup.get_text(" ", strip=True)
    
    # ========================================
    # BASIC INFO
    # ========================================
    
    # Title
    h1 = soup.find("h1")
    out["title"] = text_or_none(h1)
    
    # Meta tags
    meta_title = (soup.find("meta", attrs={"property": "og:title"}) or
                 soup.find("meta", attrs={"name": "twitter:title"}))
    if meta_title and meta_title.get("content"):
        out["meta_title"] = meta_title["content"]
    
    # ========================================
    # PRICE
    # ========================================
    
    # Find price snippet
    price_candidates = soup.find_all(string=re.compile(r"€"))
    if price_candidates:
        prices = sorted({norm(str(x)) for x in price_candidates 
                        if x and len(str(x)) < 80}, key=len)
        out["price_snippet"] = prices[0] if prices else None
    
    # Extract numeric asking price
    out["asking_price"] = extract_price_value(out.get("price_snippet"))
    
    # ========================================
    # PROPERTY DETAILS
    # ========================================
    
    # Bedrooms
    m_beds = re.search(r'\b(\d+)\s+bed', body_text, flags=re.IGNORECASE)
    out["beds"] = int(m_beds.group(1)) if m_beds else None
    
    # Bathrooms
    m_baths = re.search(r'\b(\d+)\s+bath', body_text, flags=re.IGNORECASE)
    out["baths"] = int(m_baths.group(1)) if m_baths else None
    
    # Size (square meters)
    m_sqm = (re.search(r'\b(\d+(?:,\d+)?(?:\.\d+)?)\s*m2\b', body_text, flags=re.IGNORECASE) or
            re.search(r'\b(\d+(?:,\d+)?(?:\.\d+)?)\s*sq\.?\s*m\b', body_text, flags=re.IGNORECASE) or
            re.search(r'\b(\d+(?:,\d+)?(?:\.\d+)?)\s*sqm\b', body_text, flags=re.IGNORECASE))
    if m_sqm:
        sqm_str = m_sqm.group(1).replace(',', '')
        try:
            out["size_sqm"] = float(sqm_str)
        except:
            out["size_sqm"] = None
    else:
        out["size_sqm"] = None
    
    # Property type
    out["property_type"] = extract_property_type(soup)
    
    # Year built
    out["year_built"] = extract_year_built(body_text)
    
    # Eircode
    out["eircode"] = extract_eircode(body_text)
    
    # ========================================
    # BER (Energy Rating)
    # ========================================
    
    m_ber = re.search(r'\bBER\s*(?:Rating)?[:\s]*([A-G][0-9]?)\b', body_text, flags=re.IGNORECASE)
    out["ber_rating"] = m_ber.group(1).upper() if m_ber else None
    
    # ========================================
    # AMENITIES (price drivers)
    # ========================================
    
    out["parking"] = bool(re.search(r'\bparking\b', body_text, re.I))
    out["garage"] = bool(re.search(r'\bgarage\b', body_text, re.I))
    out["garden"] = bool(re.search(r'\bgarden\b', body_text, re.I))
    out["balcony"] = bool(re.search(r'\bbalcony\b', body_text, re.I))
    out["ensuite"] = count_matches(r'\bensuite\b', body_text, re.I)
    
    # ========================================
    # CONDITION INDICATORS
    # ========================================
    
    out["new_build"] = bool(re.search(r'\bnew build\b|\bnewly built\b', body_text, re.I))
    out["refurbished"] = bool(re.search(r'\brefurbish|\brenovated\b|\brestored\b', body_text, re.I))
    out["furnished"] = bool(re.search(r'\bfurnished\b', body_text, re.I))
    
    # ========================================
    # LISTING QUALITY PROXIES
    # ========================================
    
    # Image count (better photos = better presentation = higher price)
    img_patterns = [
        soup.find_all('img', src=re.compile(r'daft.*property', re.I)),
        soup.find_all('img', attrs={'data-src': re.compile(r'daft', re.I)}),
        soup.find_all('img', class_=re.compile(r'property.*image', re.I))
    ]
    out["num_images"] = max(len(imgs) for imgs in img_patterns)
    
    # Description
    desc = (
        soup.find(attrs={"data-testid": re.compile(r".*description.*", re.I)}) or
        soup.find("section", attrs={"id": re.compile(r".*description.*", re.I)}) or
        soup.find("div", class_=re.compile(r".*description.*", re.I)) or
        soup.find("div", attrs={"id": re.compile(r".*description.*", re.I)})
    )
    out["description"] = text_or_none(desc)
    out["description_length"] = len(out.get("description") or "")
    
    # ========================================
    # FEATURES LIST
    # ========================================
    
    features = []
    for ul in soup.find_all("ul"):
        lis = [li.get_text(" ", strip=True) for li in ul.find_all("li")]
        # Heuristic: feature lists are 3-30 items, mostly short strings
        if 3 <= len(lis) <= 30:
            if sum(1 for x in lis if len(x) <= 60) / len(lis) > 0.7:
                features = lis
                break
    
    out["features_list"] = features if features else None
    out["num_features"] = len(features) if features else 0
    
    # ========================================
    # STRUCTURED DATA (JSON-LD)
    # ========================================
    
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(script.get_text(strip=True))
            out["jsonld"] = data
            
            # Extract lat/lon if present
            if isinstance(data, dict):
                geo = data.get("geo", {})
                if isinstance(geo, dict):
                    out["latitude"] = geo.get("latitude")
                    out["longitude"] = geo.get("longitude")
            
            break
        except:
            continue
    
    return out


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_batch(input_csv: Path, output_csv: Path, output_jsonl: Path,
                 use_wayback: bool = False, html_out_dir: Optional[Path] = None,
                 local_html_dir: Optional[Path] = None, timeout: int = 30,
                 verbose: bool = True, ppr_format: bool = False,
                 checkpoint_interval: int = 1000, resume_from: Optional[Path] = None):
    """
    Process batch of properties from CSV.

    Args:
        input_csv: Input CSV with address, daft_url, sale_date, selling_price columns
        output_csv: Output CSV path
        output_jsonl: Output JSONL path
        use_wayback: Fetch from Wayback Machine
        html_out_dir: Where to save downloaded HTML
        local_html_dir: Directory containing local HTML files
        timeout: Request timeout in seconds
        verbose: Print progress
        ppr_format: Input is in PPR-ALL.csv format
        checkpoint_interval: Save checkpoint every N records
        resume_from: Resume from checkpoint file
    """
    global SHUTDOWN_REQUESTED

    # Setup signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

    if html_out_dir:
        html_out_dir.mkdir(parents=True, exist_ok=True)

    # Read input
    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        input_rows = list(reader)

    # Normalize PPR format if needed
    if ppr_format:
        print(f"{Fore.CYAN}Detected PPR format, normalizing columns...{Style.RESET_ALL}")
        input_rows = [normalize_ppr_row(row) for row in input_rows]

    # Resume from checkpoint if provided
    results = []
    start_idx = 0
    if resume_from and resume_from.exists():
        print(f"{Fore.YELLOW}Resuming from checkpoint: {resume_from}{Style.RESET_ALL}")
        with resume_from.open("r", encoding="utf-8") as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"  Loaded {start_idx:,} previously processed records")

    # Setup monitoring
    log_file = output_csv.parent / f"{output_csv.stem}_run.log"
    checkpoint_file = output_csv.parent / f"{output_csv.stem}_checkpoint.json"
    monitor = RunMonitor(log_file, len(input_rows), verbose=verbose)

    # Process records
    for idx, row in enumerate(input_rows):
        # Skip already processed records
        if idx < start_idx:
            continue

        # Check for shutdown request
        if SHUTDOWN_REQUESTED:
            print(f"\n{Fore.YELLOW}Saving checkpoint before exit...{Style.RESET_ALL}")
            monitor.log_checkpoint(checkpoint_file, results)
            break

        address = (row.get("address") or "").strip()
        daft_url = (row.get("daft_url") or row.get("daft_url_hint") or "").strip()
        sale_date = safe_date((row.get("sale_date") or "").strip())
        selling_price_str = (row.get("selling_price") or "").strip()
        local_html_name = (row.get("local_html") or "").strip()

        # Parse selling price if provided
        selling_price = None
        if selling_price_str:
            try:
                selling_price = float(selling_price_str.replace(',', ''))
            except:
                pass

        result = {
            "input_address": address,
            "daft_url": daft_url,
            "sale_date": sale_date.isoformat() if sale_date else "",
            "selling_price": selling_price,
            "source": "",
            "wayback_timestamp": "",
            "wayback_url": "",
            "html_path": "",
            "extract_ok": 0,
            "error": "",
        }

        # Copy over any additional columns from input
        for col in row.keys():
            if col not in result:
                result[col] = row[col]

        html_text = None

        # ========================================
        # OPTION 1: Local HTML file
        # ========================================

        if local_html_dir and local_html_name:
            html_path = local_html_dir / local_html_name
            if html_path.exists():
                result["source"] = "local_html"
                result["html_path"] = str(html_path)
                try:
                    html_text = html_path.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    result["error"] = f"local_read_failed: {e}"

        # ========================================
        # OPTION 2: Wayback Machine
        # ========================================

        if html_text is None and use_wayback and daft_url:
            snap = pick_best_snapshot(daft_url, sale_date, timeout_s=timeout)

            if not snap:
                result["error"] = "no_wayback_snapshot"
            else:
                ts = snap["timestamp"]
                result["wayback_timestamp"] = ts

                # Save to html_out_dir
                filename = f"{sha1(daft_url)}_{ts}.html"
                html_path = html_out_dir / filename

                wb_url, html_text = fetch_wayback_html(ts, daft_url, html_path, timeout_s=timeout)

                result["source"] = "wayback"
                result["wayback_url"] = wb_url or ""
                result["html_path"] = str(html_path)

                if html_text is None:
                    result["error"] = "wayback_download_failed"

        # ========================================
        # EXTRACT DATA
        # ========================================

        if html_text is None:
            if not result["error"]:
                result["error"] = "no_html_source"
            results.append(result)
            monitor.update(success=False, source=result["source"], error=result["error"])
            if result["error"]:
                monitor.log_error(address, result["error"])
            continue

        try:
            fields = extract_listing_fields(html_text)

            # Merge extracted fields into result
            for k, v in fields.items():
                if isinstance(v, (dict, list)):
                    result[k] = json.dumps(v, ensure_ascii=False)
                else:
                    result[k] = v

            result["extract_ok"] = 1
            monitor.update(success=True, source=result["source"])

        except Exception as e:
            result["error"] = f"extract_failed: {e}"
            monitor.update(success=False, source=result["source"], error=result["error"])
            monitor.log_error(address, result["error"])

        results.append(result)

        # Save checkpoint periodically
        if len(results) % checkpoint_interval == 0:
            monitor.log_checkpoint(checkpoint_file, results)

    # ========================================
    # WRITE OUTPUTS
    # ========================================

    # CSV
    fieldnames = sorted({k for r in results for k in r.keys()})

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # JSONL
    with output_jsonl.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Final summary
    monitor.finalize()

    print(f"\n{Fore.GREEN}Outputs:{Style.RESET_ALL}")
    print(f"  CSV:  {output_csv}")
    print(f"  JSONL: {output_jsonl}")
    print(f"  Log:  {log_file}")

    # Cleanup checkpoint on successful completion
    if not SHUTDOWN_REQUESTED and checkpoint_file.exists():
        checkpoint_file.unlink()
        print(f"  Checkpoint cleaned up (run completed successfully)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Mode selection
    parser.add_argument("--match", action="store_true",
                       help="Step 1: Match PPR addresses to Daft URLs (run this first)")
    parser.add_argument("--extract", action="store_true",
                       help="Step 2: Extract data from matched Daft listings")

    # Input/Output
    parser.add_argument("--in", dest="input_csv", required=True,
                       help="Input CSV file (supports PPR-ALL.csv format with --ppr_format)")
    parser.add_argument("--out", required=True,
                       help="Output CSV path")

    # Data sources
    parser.add_argument("--wayback", action="store_true",
                       help="Fetch HTML from Wayback Machine")
    parser.add_argument("--html_out_dir", default="html_cache",
                       help="Directory to save downloaded HTML (default: html_cache)")
    parser.add_argument("--local_html_dir", default="",
                       help="Directory containing local HTML files")

    # Format options
    parser.add_argument("--ppr_format", action="store_true",
                       help="Input is Irish Property Price Register format (PPR-ALL.csv)")

    # Matching options
    parser.add_argument("--min_confidence", type=float, default=0.4,
                       help="Minimum confidence for address matching (0-1, default: 0.4)")

    # Performance options
    parser.add_argument("--timeout", type=int, default=30,
                       help="HTTP request timeout in seconds (default: 30)")
    parser.add_argument("--checkpoint_interval", type=int, default=1000,
                       help="Save checkpoint every N records (default: 1000)")
    parser.add_argument("--resume", dest="resume_from", default="",
                       help="Resume from checkpoint file")

    # Output options
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress progress output")

    args = parser.parse_args()

    # Validate inputs
    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        print(f"{Fore.RED}Error: Input file not found: {input_csv}{Style.RESET_ALL}")
        sys.exit(1)

    output_csv = Path(args.out)

    # Determine mode
    if args.match:
        # Step 1: Match addresses to Daft URLs
        match_ppr_addresses(
            input_csv=input_csv,
            output_csv=output_csv,
            ppr_format=args.ppr_format,
            min_confidence=args.min_confidence,
            timeout=args.timeout,
            verbose=not args.quiet
        )

    elif args.extract or args.wayback or args.local_html_dir:
        # Step 2: Extract data from listings
        output_jsonl = output_csv.with_suffix(".jsonl")
        html_out_dir = Path(args.html_out_dir) if args.wayback else None
        local_html_dir = Path(args.local_html_dir) if args.local_html_dir else None
        resume_from = Path(args.resume_from) if args.resume_from else None

        # Print startup banner
        print(f"""
{Fore.CYAN}{'='*60}
Daft Property Data Extraction Tool
{'='*60}{Style.RESET_ALL}
Input:  {input_csv}
Output: {output_csv}
Mode:   {'PPR format' if args.ppr_format else 'Standard format'}
Source: {'Wayback Machine' if args.wayback else ''} {'+ ' if args.wayback and local_html_dir else ''}{'Local HTML' if local_html_dir else ''}
{'='*60}
""")

        process_batch(
            input_csv=input_csv,
            output_csv=output_csv,
            output_jsonl=output_jsonl,
            use_wayback=args.wayback,
            html_out_dir=html_out_dir,
            local_html_dir=local_html_dir,
            timeout=args.timeout,
            verbose=not args.quiet,
            ppr_format=args.ppr_format,
            checkpoint_interval=args.checkpoint_interval,
            resume_from=resume_from
        )

    else:
        # No mode specified - show help
        print(f"""
{Fore.CYAN}Daft Property Data Extraction Tool{Style.RESET_ALL}

This tool has two steps:

{Fore.GREEN}Step 1: Match PPR addresses to Daft URLs{Style.RESET_ALL}
  python extract_daft_data.py --match --in PPR-ALL.csv --ppr_format --out matched.csv

{Fore.GREEN}Step 2: Extract data from matched listings{Style.RESET_ALL}
  python extract_daft_data.py --extract --in matched.csv --wayback --out enriched.csv

{Fore.YELLOW}Options:{Style.RESET_ALL}
  --match           Run Step 1 (address matching)
  --extract         Run Step 2 (data extraction)
  --ppr_format      Input is PPR-ALL.csv format
  --wayback         Fetch from Wayback Machine
  --min_confidence  Minimum match confidence (0-1, default: 0.4)
  --quiet           Suppress progress output

Run with --help for full options.
""")


if __name__ == "__main__":
    main()