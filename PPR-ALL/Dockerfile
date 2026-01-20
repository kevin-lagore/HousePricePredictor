# Dockerfile for MyHome.ie Scraper
# Uses Playwright with Firefox for headless scraping

FROM mcr.microsoft.com/playwright/python:v1.40.0-jammy

WORKDIR /app

# Copy scraper files
COPY scrape_myhome.py .
COPY requirements-scraper.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-scraper.txt

# Install Playwright browsers (Firefox only to save space)
RUN playwright install firefox

# Create output directory
RUN mkdir -p /app/output

# Default command
CMD ["python", "scrape_myhome.py", "--help"]
