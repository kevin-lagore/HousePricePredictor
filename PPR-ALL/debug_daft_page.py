#!/usr/bin/env python3
"""Test script to verify the updated scraper extracts data correctly."""

from daft_sold_scraper import DaftSoldScraper

# Test URLs from the matched.csv
TEST_URLS = [
    "https://www.daft.ie/sold/5-braemor-drive-churchtown-co-dublin-dublin/06B906E0ECB0C00480257A7D00555CAA",
    "https://www.daft.ie/sold/134-ashewood-walk-summerhill-lane-portlaoise-laois/E926FD0CB6E1B50480257A7D00558EBF",
]

def test_extraction():
    print("Testing updated scraper extraction...")
    print("="*60)

    with DaftSoldScraper(headless=False) as scraper:
        for url in TEST_URLS:
            print(f"\nTesting: {url[:60]}...")
            listing = scraper.extract_listing(url)

            print(f"\n  Address: {listing.address}")
            print(f"  Sold Price: {listing.sold_price}")
            print(f"  Asking Price: {listing.asking_price}")
            print(f"  Beds: {listing.beds}")
            print(f"  Baths: {listing.baths}")
            print(f"  Property Type: {listing.property_type}")
            print(f"  Size (sqm): {listing.size_sqm}")
            print(f"  BER Rating: {listing.ber_rating}")
            print(f"  Eircode: {listing.eircode}")
            print("-"*40)

    print("\n" + "="*60)
    print("Test complete!")

if __name__ == "__main__":
    test_extraction()
