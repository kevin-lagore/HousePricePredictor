#!/bin/bash
# Run the MyHome.ie brochure scraper in a tmux session
# This allows you to disconnect from SSH and the scraper keeps running
#
# Usage:
#   ./run-scraper.sh                    # Scrape brochures (bulk mode)
#   ./run-scraper.sh --county dublin    # Filter by county
#   ./run-scraper.sh --limit 1000       # Limit number of listings
#
# NOTE: Uses scrape_myhome_brochures.py which gets FULL property data
# (beds, baths, size, BER, description) from brochure pages that persist
# even after properties are sold.

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install playwright
    playwright install firefox
else
    source venv/bin/activate
fi

SESSION_NAME="scraper"
OUTPUT_DIR="output"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p "$OUTPUT_DIR"

OUTPUT_FILE="$OUTPUT_DIR/myhome_brochures_$TIMESTAMP.csv"

# Build command - use brochure scraper in bulk mode
CMD="python scrape_myhome_brochures.py --bulk --out $OUTPUT_FILE $@"

echo "=========================================="
echo "MyHome.ie Scraper"
echo "=========================================="
echo "Output: $OUTPUT_FILE"
echo "Command: $CMD"
echo ""

# Check if tmux session exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists!"
    echo "Options:"
    echo "  - Attach: tmux attach -t $SESSION_NAME"
    echo "  - Kill:   tmux kill-session -t $SESSION_NAME"
    exit 1
fi

# Start in tmux
echo "Starting scraper in tmux session '$SESSION_NAME'..."
tmux new-session -d -s "$SESSION_NAME" "$CMD"

echo ""
echo "Scraper is running in the background!"
echo ""
echo "Commands:"
echo "  Attach to see progress:  tmux attach -t $SESSION_NAME"
echo "  Detach (keep running):   Ctrl+B, then D"
echo "  Check if running:        tmux list-sessions"
echo "  Stop scraper:            tmux kill-session -t $SESSION_NAME"
echo "  View output:             tail -f $OUTPUT_FILE"
echo ""
