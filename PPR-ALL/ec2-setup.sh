#!/bin/bash
# EC2 Setup Script for MyHome.ie Scraper
# Run this on a fresh Ubuntu 22.04 EC2 instance
#
# Usage:
#   chmod +x ec2-setup.sh
#   ./ec2-setup.sh

set -e

echo "=========================================="
echo "MyHome.ie Scraper - EC2 Setup"
echo "=========================================="

# Update system
echo "[1/5] Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and dependencies
echo "[2/5] Installing Python and system dependencies..."
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    tmux \
    htop \
    unzip

# Set up virtual environment in current directory
echo "[3/5] Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install --upgrade pip
pip install playwright

# Install Firefox browser and system dependencies
echo "[4/5] Installing Firefox browser..."
playwright install firefox

echo "[5/5] Installing browser system dependencies..."
sudo venv/bin/playwright install-deps

# Create output directory
mkdir -p output

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run: source venv/bin/activate"
echo "  2. Start scraping with: ./run-scraper.sh"
echo ""
