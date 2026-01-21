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

# Track previous counts for rate calculation
PREV_BULK_ROWS=0
PREV_ADDR_ROWS=0
PREV_TIME=$(date +%s)

# Estimated totals
BULK_TOTAL=5000      # Estimated listings on MyHome.ie
ADDR_TOTAL=$(wc -l < "$SCRIPT_DIR/ppr_addresses.txt" 2>/dev/null || echo 100000)

send_discord() {
    local message="$1"
    curl -s -X POST -H 'Content-type: application/json' \
        --data "{\"content\":\"$message\"}" \
        "$WEBHOOK_URL" > /dev/null 2>&1
}

get_rows() {
    local file="$1"
    if [ -f "$file" ]; then
        wc -l < "$file"
    else
        echo "0"
    fi
}

get_file_size() {
    local file="$1"
    if [ -f "$file" ]; then
        du -h "$file" | cut -f1
    else
        echo "0"
    fi
}

format_eta() {
    local seconds="$1"
    if [ "$seconds" -le 0 ]; then
        echo "calculating..."
    elif [ "$seconds" -gt 86400 ]; then
        local days=$((seconds / 86400))
        local hours=$(((seconds % 86400) / 3600))
        echo "${days}d ${hours}h"
    elif [ "$seconds" -gt 3600 ]; then
        local hours=$((seconds / 3600))
        local mins=$(((seconds % 3600) / 60))
        echo "${hours}h ${mins}m"
    else
        local mins=$((seconds / 60))
        echo "${mins}m"
    fi
}

calc_eta() {
    local current="$1"
    local previous="$2"
    local total="$3"
    local elapsed="$4"

    local delta=$((current - previous))
    if [ "$delta" -le 0 ] || [ "$elapsed" -le 0 ]; then
        echo "0"
        return
    fi

    local rate=$(echo "scale=4; $delta / $elapsed" | bc)
    local remaining=$((total - current))

    if [ "$(echo "$rate > 0" | bc)" -eq 1 ]; then
        local eta_seconds=$(echo "scale=0; $remaining / $rate" | bc)
        echo "$eta_seconds"
    else
        echo "0"
    fi
}

check_processes() {
    local count=$(pgrep -f "scrape_myhome" | wc -l)
    echo "$count"
}

send_discord "[Monitor] Starting scraper monitor - updates every 5 minutes"

while true; do
    sleep $INTERVAL

    NOW=$(date +%s)
    ELAPSED=$((NOW - PREV_TIME))

    # Check if scrapers are still running
    RUNNING=$(check_processes)

    # Get stats for output files (use fixed filenames)
    BULK_FILE="$OUTPUT_DIR/bulk_main.csv"
    ADDR_FILE="$OUTPUT_DIR/ppr_enriched_main.csv"

    BULK_ROWS=$(get_rows "$BULK_FILE")
    ADDR_ROWS=$(get_rows "$ADDR_FILE")
    BULK_SIZE=$(get_file_size "$BULK_FILE")
    ADDR_SIZE=$(get_file_size "$ADDR_FILE")

    # Calculate ETAs
    BULK_ETA_SEC=$(calc_eta "$BULK_ROWS" "$PREV_BULK_ROWS" "$BULK_TOTAL" "$ELAPSED")
    ADDR_ETA_SEC=$(calc_eta "$ADDR_ROWS" "$PREV_ADDR_ROWS" "$ADDR_TOTAL" "$ELAPSED")
    BULK_ETA=$(format_eta "$BULK_ETA_SEC")
    ADDR_ETA=$(format_eta "$ADDR_ETA_SEC")

    # Calculate rates
    BULK_RATE=$(echo "scale=1; ($BULK_ROWS - $PREV_BULK_ROWS) * 60 / $ELAPSED" | bc 2>/dev/null || echo "0")
    ADDR_RATE=$(echo "scale=1; ($ADDR_ROWS - $PREV_ADDR_ROWS) * 60 / $ELAPSED" | bc 2>/dev/null || echo "0")

    # Build message
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    MESSAGE="**[Scraper Update - $TIMESTAMP]**\n"
    MESSAGE+="Processes running: $RUNNING\n\n"
    MESSAGE+="**Bulk scraper:**\n"
    MESSAGE+="  ${BULK_ROWS}/${BULK_TOTAL} rows (${BULK_SIZE}) | ${BULK_RATE}/min | ETA: ${BULK_ETA}\n\n"
    MESSAGE+="**Address scraper:**\n"
    MESSAGE+="  ${ADDR_ROWS}/${ADDR_TOTAL} rows (${ADDR_SIZE}) | ${ADDR_RATE}/min | ETA: ${ADDR_ETA}"

    # Send to Discord
    send_discord "$MESSAGE"

    # Update previous values
    PREV_BULK_ROWS=$BULK_ROWS
    PREV_ADDR_ROWS=$ADDR_ROWS
    PREV_TIME=$NOW

    # Exit if no scrapers running
    if [ "$RUNNING" -eq 0 ]; then
        send_discord "[Monitor] No scrapers running - monitor stopping"
        exit 0
    fi
done
