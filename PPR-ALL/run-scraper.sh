#!/bin/bash
# Run the MyHome.ie scraper in a tmux session
# This allows you to disconnect from SSH and the scraper keeps running
#
# Usage:
#   ./run-scraper.sh                    # Scrape price register (sold properties)
#   ./run-scraper.sh --for-sale         # Scrape for-sale listings
#   ./run-scraper.sh --county dublin    # Filter by county

set -e

cd ~/scraper
source venv/bin/activate

SESSION_NAME="scraper"
OUTPUT_DIR="output"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Default output file based on mode
if [[ "$*" == *"--for-sale"* ]]; then
    OUTPUT_FILE="$OUTPUT_DIR/myhome_forsale_$TIMESTAMP.csv"
else
    OUTPUT_FILE="$OUTPUT_DIR/myhome_sold_$TIMESTAMP.csv"
fi

# Build command
CMD="python scrape_myhome.py --out $OUTPUT_FILE $@"

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
