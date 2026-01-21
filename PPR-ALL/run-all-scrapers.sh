#!/bin/bash
# Run both scrapers with automatic restart and failure notification
#
# Usage:
#   ./run-all-scrapers.sh                    # Run both scrapers
#   ./run-all-scrapers.sh --bulk-only        # Run only bulk scraper
#   ./run-all-scrapers.sh --addresses-only   # Run only address scraper
#
# Set WEBHOOK_URL environment variable for Slack/Discord notifications:
#   export WEBHOOK_URL="https://hooks.slack.com/services/xxx"

set -e

# Discord webhook for notifications
WEBHOOK_URL="${WEBHOOK_URL:-https://discord.com/api/webhooks/1463501941299744770/GFB8cL68SzT0FyReQ2pmyjNuxtHEpn0jopGWfaiUpQjWnSmsr4qH3dj66lAZDIexRMuC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
MAX_RETRIES=10
RETRY_DELAY=60
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$SCRIPT_DIR/logs"
OUTPUT_DIR="$SCRIPT_DIR/output"

# Create directories
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# Notification function
notify() {
    local message="$1"
    echo "$(date): $message"
    echo "$(date): $message" >> "$LOG_DIR/notifications.log"

    if [ -n "$WEBHOOK_URL" ]; then
        curl -s -X POST -H 'Content-type: application/json' \
            --data "{\"content\":\"[EC2 Scraper] $message\"}" \
            "$WEBHOOK_URL" > /dev/null 2>&1 || true
    fi
}

# Run scraper with retries
run_scraper() {
    local name="$1"
    local cmd="$2"
    local log_file="$LOG_DIR/${name}_${TIMESTAMP}.log"

    notify "$name scraper starting"

    for i in $(seq 1 $MAX_RETRIES); do
        echo "$(date): Starting $name (attempt $i/$MAX_RETRIES)" | tee -a "$log_file"

        # Run the command
        if eval "$cmd" >> "$log_file" 2>&1; then
            notify "$name scraper completed successfully"
            return 0
        fi

        EXIT_CODE=$?
        echo "$(date): $name exited with code $EXIT_CODE" | tee -a "$log_file"

        if [ $i -lt $MAX_RETRIES ]; then
            echo "$(date): Restarting in ${RETRY_DELAY}s..." | tee -a "$log_file"
            sleep $RETRY_DELAY
        fi
    done

    notify "$name scraper FAILED after $MAX_RETRIES attempts"
    return 1
}

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "Error: venv not found. Run ec2-setup.sh first."
    exit 1
fi
source venv/bin/activate

# Parse arguments
RUN_BULK=true
RUN_ADDRESSES=true

case "${1:-}" in
    --bulk-only)
        RUN_ADDRESSES=false
        ;;
    --addresses-only)
        RUN_BULK=false
        ;;
esac

# Check for addresses file if needed
if [ "$RUN_ADDRESSES" = true ] && [ ! -f "ppr_addresses.txt" ]; then
    echo "Extracting PPR addresses (last 5 years)..."
    python extract_ppr_addresses.py PPR-ALL.csv ppr_addresses.txt --recent 1825
fi

echo "=========================================="
echo "Starting Scrapers - $(date)"
echo "=========================================="
echo "Bulk scraper: $RUN_BULK"
echo "Address scraper: $RUN_ADDRESSES"
echo "Logs: $LOG_DIR"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Run scrapers SEQUENTIALLY to avoid memory issues
# Use fixed output filenames so --resume works across restarts
FAILED=0

if [ "$RUN_BULK" = true ]; then
    BULK_OUTPUT="$OUTPUT_DIR/bulk_main.csv"
    echo "Running bulk scraper..."
    if ! run_scraper "bulk" "python scrape_myhome_brochures.py --bulk --out $BULK_OUTPUT --resume"; then
        FAILED=$((FAILED + 1))
    fi
fi

if [ "$RUN_ADDRESSES" = true ]; then
    ADDR_OUTPUT="$OUTPUT_DIR/ppr_enriched_main.csv"
    echo "Running address scraper..."
    if ! run_scraper "addresses" "python scrape_myhome_brochures.py --addresses ppr_addresses.txt --out $ADDR_OUTPUT --resume"; then
        FAILED=$((FAILED + 1))
    fi
fi

if [ $FAILED -eq 0 ]; then
    notify "All scrapers completed successfully"
else
    notify "$FAILED scraper(s) failed"
fi

echo ""
echo "=========================================="
echo "Finished - $(date)"
echo "=========================================="
