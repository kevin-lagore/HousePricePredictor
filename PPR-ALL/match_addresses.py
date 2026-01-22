#!/usr/bin/env python3
"""Match addresses from ppr_enriched_main.csv to PPR-ALL.csv.

Uses fuzzy matching and address normalization to find the best PPR record
for each property listing.
"""

import csv
import re
import unicodedata
from pathlib import Path
from rapidfuzz import fuzz, process
from collections import defaultdict


# Common word synonyms/abbreviations - used to exclude from street similarity
ADDRESS_SYNONYMS = {
    'road': {'rd', 'road'},
    'street': {'st', 'str', 'street'},
    'avenue': {'ave', 'av', 'avenue'},
    'drive': {'dr', 'drv', 'drive'},
    'crescent': {'cres', 'cr', 'crescent'},
    'park': {'pk', 'park'},
    'lane': {'ln', 'lane'},
    'court': {'ct', 'crt', 'court'},
    'terrace': {'ter', 'terr', 'terrace'},
    'place': {'pl', 'place'},
    'gardens': {'gdns', 'gdn', 'gardens'},
    'grove': {'gr', 'grv', 'grove'},
    'close': {'cl', 'close'},
    'way': {'way'},
    'square': {'sq', 'square'},
    'heights': {'hts', 'heights'},
    'view': {'vw', 'view'},
    'green': {'grn', 'green'},
    'mount': {'mt', 'mount'},
    'hill': {'hl', 'hill'},
    'wood': {'wd', 'wood'},
    'meadow': {'mdw', 'meadow'},
}

# Flatten to lookup: word -> canonical form
SYNONYM_LOOKUP = {}
for canonical, variants in ADDRESS_SYNONYMS.items():
    for v in variants:
        SYNONYM_LOOKUP[v] = canonical

# Words to exclude when comparing street names (these are type words, not identifying)
STREET_TYPE_WORDS = set(SYNONYM_LOOKUP.keys())


def normalize_address(addr: str) -> str:
    """Normalize an address for comparison."""
    if not addr:
        return ""

    # Convert to lowercase
    addr = addr.lower().strip()

    # Normalize unicode characters
    addr = unicodedata.normalize('NFKD', addr).encode('ascii', 'ignore').decode('ascii')

    # Common replacements
    replacements = [
        (r'\bco\.?\s*', ''),  # Remove "Co." or "Co "
        (r'\bcounty\s+', ''),  # Remove "County "
        (r'\bdublin\s+(\d+)', r'dublin \1'),  # Standardize Dublin X
        (r'\bd(\d+)\b', r'dublin \1'),  # D13 -> dublin 13
        (r'\brd\.?\b', 'road'),
        (r'\bst\.?\b', 'street'),
        (r'\bave\.?\b', 'avenue'),
        (r'\bdr\.?\b', 'drive'),
        (r'\bcres\.?\b', 'crescent'),
        (r'\bpk\.?\b', 'park'),
        (r'\bln\.?\b', 'lane'),
        (r'\bct\.?\b', 'court'),
        (r'\bapt\.?\b', 'apartment'),
        (r'\bno\.?\s*(\d+)', r'\1'),  # "No. 5" -> "5"
        (r'[,\.\-\/]+', ' '),  # Replace punctuation with space
        (r'\s+', ' '),  # Collapse multiple spaces
    ]

    for pattern, replacement in replacements:
        addr = re.sub(pattern, replacement, addr)

    return addr.strip()


def extract_street_number(addr: str) -> str | None:
    """Extract street/house number from address."""
    match = re.match(r'^(\d+[a-z]?)\s', addr.lower())
    return match.group(1) if match else None


def extract_county(addr: str) -> str | None:
    """Extract county from address."""
    addr_lower = addr.lower()

    # Dublin with number
    match = re.search(r'dublin\s*(\d+)?', addr_lower)
    if match:
        return 'dublin'

    # County patterns
    counties = [
        'carlow', 'cavan', 'clare', 'cork', 'donegal', 'galway', 'kerry',
        'kildare', 'kilkenny', 'laois', 'leitrim', 'limerick', 'longford',
        'louth', 'mayo', 'meath', 'monaghan', 'offaly', 'roscommon',
        'sligo', 'tipperary', 'waterford', 'westmeath', 'wexford', 'wicklow'
    ]

    for county in counties:
        if county in addr_lower:
            return county

    return None


def load_ppr_data(filepath: Path) -> tuple[list[dict], dict[str, list[int]]]:
    """Load PPR data and create index by county."""
    records = []
    county_index = defaultdict(list)

    with open(filepath, 'r', encoding='latin-1') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            addr = row.get('Address', '')
            county = row.get('County', '').lower().strip()

            # Find the price column (has special character in name)
            price = ''
            for key in row.keys():
                if 'price' in key.lower():
                    price = row[key]
                    # Clean up the price string (remove euro symbol and special chars)
                    price = price.replace('\x80', '').replace('€', '').strip()
                    break

            records.append({
                'address': addr,
                'county': county,
                'date': row.get('Date of Sale (dd/mm/yyyy)', ''),
                'price': price,
                'normalized': normalize_address(addr),
            })

            county_index[county].append(i)

            # Also index by extracted county from address
            extracted = extract_county(addr)
            if extracted and extracted != county:
                county_index[extracted].append(i)

    return records, county_index


def load_myhome_data(filepath: Path) -> list[dict]:
    """Load MyHome enriched data."""
    records = []

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            addr = row.get('address_full', '')
            county = row.get('county', '').lower().strip()

            records.append({
                **row,
                'normalized_address': normalize_address(addr),
                'county_lower': county,
            })

    return records


def extract_street_name(addr: str) -> str | None:
    """Extract the street name INCLUDING the type (e.g., 'parkside crescent')."""
    addr = normalize_address(addr)
    # Remove number at start
    addr = re.sub(r'^\d+[a-z]?\s+', '', addr)
    # Get words
    parts = addr.split()
    if not parts:
        return None

    # Find the street name + type (first identifying word + first type word after it)
    location_words = {'dublin', 'cork', 'galway', 'limerick', 'waterford', 'kilkenny',
                      'meath', 'kildare', 'wicklow', 'wexford', 'carlow', 'laois',
                      'offaly', 'westmeath', 'longford', 'louth', 'monaghan', 'cavan',
                      'donegal', 'sligo', 'mayo', 'roscommon', 'leitrim', 'clare',
                      'tipperary', 'kerry', 'ireland', 'county', 'co'}

    result = []
    for i, part in enumerate(parts):
        if part in location_words:
            break  # Stop at location words
        if len(part) > 1:
            result.append(part)
        if len(result) >= 2:
            break  # Take first 2 words (name + type)

    return ' '.join(result) if result else parts[0]


def extract_key_words(addr: str) -> set[str]:
    """Extract key identifying words from an address (street name, area name)."""
    addr = normalize_address(addr)
    # Remove number at start
    addr = re.sub(r'^\d+[a-z]?\s+', '', addr)
    # Remove common suffixes and county names
    stopwords = {
        'road', 'street', 'avenue', 'drive', 'crescent', 'park', 'lane', 'court',
        'dublin', 'cork', 'galway', 'limerick', 'waterford', 'kilkenny',
        'county', 'co', 'ireland'
    }
    words = set(addr.split())
    # Keep meaningful words (street name, area name)
    return {w for w in words if w not in stopwords and len(w) > 2}


def find_best_match(
    myhome_record: dict,
    ppr_records: list[dict],
    county_index: dict[str, list[int]],
    threshold: float = 75.0
) -> tuple[dict | None, float]:
    """Find the best matching PPR record for a MyHome listing."""

    myhome_addr = myhome_record['normalized_address']
    myhome_county = myhome_record['county_lower']
    original_addr = myhome_record.get('address_full', '')

    if not myhome_addr:
        return None, 0.0

    # Get candidate indices based on county
    candidate_indices = set()

    # Add records from matching county
    if myhome_county and myhome_county in county_index:
        candidate_indices.update(county_index[myhome_county])

    # Also check Dublin variants
    if 'dublin' in myhome_county:
        candidate_indices.update(county_index.get('dublin', []))

    # Extract county from address if not in county field
    addr_county = extract_county(original_addr)
    if addr_county and addr_county in county_index:
        candidate_indices.update(county_index[addr_county])

    if not candidate_indices:
        # Fall back to all records if no county match
        candidate_indices = set(range(len(ppr_records)))

    # Filter by street number - this is critical for accurate matching
    street_num = extract_street_number(myhome_addr)
    if street_num:
        filtered = [
            i for i in candidate_indices
            if extract_street_number(ppr_records[i]['normalized']) == street_num
        ]
        if filtered:
            candidate_indices = set(filtered)

    # Build candidate list
    candidates = [(ppr_records[i]['normalized'], i) for i in candidate_indices]

    if not candidates:
        return None, 0.0

    # Use rapidfuzz to find best matches with token_set_ratio (better for address matching)
    # token_set_ratio handles word overlap well regardless of order
    results = process.extract(
        myhome_addr,
        [c[0] for c in candidates],
        scorer=fuzz.token_set_ratio,
        limit=10
    )

    if not results:
        return None, 0.0

    # Validate matches more carefully
    myhome_street = extract_street_name(myhome_addr)
    myhome_keywords = extract_key_words(myhome_addr)

    for match_text, score, match_idx in results:
        if score < threshold:
            continue

        ppr_street = extract_street_name(match_text)
        ppr_keywords = extract_key_words(match_text)

        # Check street name similarity - this is the primary validation
        if myhome_street and ppr_street:
            street_similarity = fuzz.ratio(myhome_street, ppr_street)

            # Exact street name match - only need reasonable overall score
            if street_similarity == 100 and score >= 70:
                original_idx = candidates[match_idx][1]
                return ppr_records[original_idx], score

            # Very high score (90+) AND good street similarity - trust it
            if score >= 90 and street_similarity >= 80:
                original_idx = candidates[match_idx][1]
                return ppr_records[original_idx], score

            # Good street match (85%+) with good overall score
            if street_similarity >= 85 and score >= 75:
                original_idx = candidates[match_idx][1]
                return ppr_records[original_idx], score

    # No valid match found above threshold
    best_score = results[0][1] if results else 0.0
    return None, best_score


def main():
    base_path = Path(__file__).parent
    ppr_path = base_path / 'PPR-ALL.csv'
    myhome_path = base_path / 'ppr_enriched_main.csv'
    output_path = base_path / 'ppr_enriched_matched.csv'

    print("Loading PPR-ALL.csv...")
    ppr_records, county_index = load_ppr_data(ppr_path)
    print(f"  Loaded {len(ppr_records):,} PPR records")
    print(f"  Counties indexed: {len(county_index)}")

    print("\nLoading ppr_enriched_main.csv...")
    myhome_records = load_myhome_data(myhome_path)
    print(f"  Loaded {len(myhome_records):,} MyHome records")

    print("\nMatching addresses...")
    matched = 0
    unmatched = 0
    results = []

    for i, mh in enumerate(myhome_records):
        ppr_match, score = find_best_match(mh, ppr_records, county_index)

        if ppr_match:
            matched += 1
            results.append({
                **mh,
                'ppr_address': ppr_match['address'],
                'ppr_date': ppr_match['date'],
                'ppr_price': ppr_match['price'],
                'match_score': f"{score:.1f}",
            })
        else:
            unmatched += 1
            results.append({
                **mh,
                'ppr_address': '',
                'ppr_date': '',
                'ppr_price': '',
                'match_score': f"{score:.1f}" if score > 0 else '',
            })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(myhome_records)} - Matched: {matched}, Unmatched: {unmatched}")

    print(f"\nResults: {matched} matched, {unmatched} unmatched out of {len(myhome_records)} total")

    # Write results
    print(f"\nWriting results to {output_path}...")

    # Get fieldnames from first result, ensuring new columns are at the end
    if results:
        base_fields = [k for k in results[0].keys() if k not in ['ppr_address', 'ppr_date', 'ppr_price', 'match_score', 'normalized_address', 'county_lower']]
        # Remove old ppr_address if it exists
        base_fields = [k for k in base_fields if k != 'ppr_address']
        new_fields = ['ppr_address', 'ppr_date', 'ppr_price', 'match_score']
        fieldnames = base_fields + new_fields

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)

        print(f"  Wrote {len(results)} records")

    # Print sample matches
    print("\nSample matches:")
    for r in results[:10]:
        addr = r.get('address_full', '')[:50]
        ppr = r.get('ppr_address', '')[:50]
        score = r.get('match_score', '')
        print(f"  {addr}")
        print(f"    -> {ppr} (score: {score})")
        print()


if __name__ == '__main__':
    main()
