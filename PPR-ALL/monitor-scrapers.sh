#!/bin/bash
# Monitor scraper progress and send updates to Discord every 5 minutes
#
# Usage:
#   ./monitor-scrapers.sh              # Run in foreground
#   nohup ./monitor-scrapers.sh &      # Run in background

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

WEBHOOK_URL="https://discord.com/api/webhooks/1463501941299744770/GFB8cL68SzT0FyReQ2pmyjNuxtHEpn0jopGWfaiUpQjWnSmsr4qH3dj66lAZDIexRMuC"
OUTPUT_DIR="$SCRIPT_DIR/output"
INTERVAL=300  # 5 minutes

send_discord() {
    local message="$1"
    curl -s -X POST -H 'Content-type: application/json' \
        --data "{\"content\":\"$message\"}" \
        "$WEBHOOK_URL" > /dev/null 2>&1
}

get_file_stats() {
    local file="$1"
    if [ -f "$file" ]; then
        local size=$(du -h "$file" | cut -f1)
        local rows=$(wc -l < "$file")
        echo "${rows} rows, ${size}"
    else
        echo "not found"
    fi
}

check_processes() {
    local count=$(pgrep -f "scrape_myhome" | wc -l)
    echo "$count"
}

send_discord "[Monitor] Starting scraper monitor - updates every 5 minutes"

while true; do
    # Check if scrapers are still running
    RUNNING=$(check_processes)

    # Get stats for output files
    BULK_FILE=$(ls -t "$OUTPUT_DIR"/bulk_*.csv 2>/dev/null | head -1)
    ADDR_FILE=$(ls -t "$OUTPUT_DIR"/ppr_enriched_*.csv 2>/dev/null | head -1)

    BULK_STATS=$(get_file_stats "$BULK_FILE")
    ADDR_STATS=$(get_file_stats "$ADDR_FILE")

    # Build message
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    MESSAGE="**[Scraper Update - $TIMESTAMP]**\n"
    MESSAGE+="Processes running: $RUNNING\n"
    MESSAGE+="Bulk scraper: $BULK_STATS\n"
    MESSAGE+="Address scraper: $ADDR_STATS"

    # Send to Discord
    send_discord "$MESSAGE"

    # Exit if no scrapers running
    if [ "$RUNNING" -eq 0 ]; then
        send_discord "[Monitor] No scrapers running - monitor stopping"
        exit 0
    fi

    sleep $INTERVAL
done
