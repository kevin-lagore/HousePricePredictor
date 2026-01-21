#!/usr/bin/env python3
"""
Extract addresses from PPR CSV for use with MyHome.ie scraper.

Usage:
    python extract_ppr_addresses.py PPR-ALL.csv ppr_addresses.txt
    python extract_ppr_addresses.py PPR-ALL.csv ppr_addresses.txt --county Dublin
    python extract_ppr_addresses.py PPR-ALL.csv ppr_addresses.txt --recent 365  # Last 365 days
    python extract_ppr_addresses.py PPR-ALL.csv ppr_addresses.txt --limit 5000
"""

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path


def parse_date(date_str: str) -> datetime | None:
    """Parse PPR date format dd/mm/yyyy."""
    try:
        return datetime.strptime(date_str.strip(), "%d/%m/%Y")
    except:
        return None


def main():
    parser = argparse.ArgumentParser(description="Extract addresses from PPR CSV")
    parser.add_argument("input_csv", help="Input PPR CSV file")
    parser.add_argument("output_txt", help="Output addresses file (one per line)")
    parser.add_argument("--county", "-c", help="Filter by county")
    parser.add_argument("--recent", "-r", type=int, help="Only include sales from last N days")
    parser.add_argument("--limit", "-l", type=int, help="Maximum addresses to extract")
    parser.add_argument("--min-price", type=int, help="Minimum sale price")
    parser.add_argument("--max-price", type=int, help="Maximum sale price")

    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"Error: {args.input_csv} not found")
        return 1

    cutoff_date = None
    if args.recent:
        cutoff_date = datetime.now() - timedelta(days=args.recent)
        print(f"Filtering sales after: {cutoff_date.strftime('%Y-%m-%d')}")

    if args.county:
        print(f"Filtering by county: {args.county}")

    addresses = []
    seen = set()

    # Try multiple encodings - PPR files are often Windows-encoded
    encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

    file_handle = None
    for enc in encodings:
        try:
            file_handle = open(input_path, 'r', encoding=enc)
            # Test read first line
            file_handle.readline()
            file_handle.seek(0)
            print(f"Using encoding: {enc}")
            break
        except (UnicodeDecodeError, UnicodeError):
            if file_handle:
                file_handle.close()
            continue

    if not file_handle:
        print("Error: Could not determine file encoding")
        return 1

    with file_handle as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Get address
            address = row.get('Address', '').strip()
            if not address:
                continue

            # Skip duplicates
            addr_lower = address.lower()
            if addr_lower in seen:
                continue

            # County filter
            if args.county:
                county = row.get('County', '').strip()
                if county.lower() != args.county.lower():
                    continue

            # Date filter
            if cutoff_date:
                sale_date = parse_date(row.get('Date of Sale (dd/mm/yyyy)', ''))
                if not sale_date or sale_date < cutoff_date:
                    continue

            # Price filter
            price_str = row.get('Price (€)', row.get('Price', '')).strip()
            if price_str:
                # Clean price: remove €, commas, .00
                price_clean = price_str.replace('€', '').replace(',', '').replace('.00', '').strip()
                try:
                    price = int(float(price_clean))
                    if args.min_price and price < args.min_price:
                        continue
                    if args.max_price and price > args.max_price:
                        continue
                except:
                    pass

            seen.add(addr_lower)

            # Append county for better search matching
            county = row.get('County', '').strip()
            if county and county.lower() not in address.lower():
                address = f"{address}, {county}"

            addresses.append(address)

            if args.limit and len(addresses) >= args.limit:
                break

    # Write output
    with open(args.output_txt, 'w', encoding='utf-8') as f:
        for addr in addresses:
            f.write(addr + '\n')

    print(f"Extracted {len(addresses):,} unique addresses to {args.output_txt}")


if __name__ == "__main__":
    main()
