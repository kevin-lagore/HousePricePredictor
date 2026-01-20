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

# Install Playwright system dependencies
echo "[3/5] Installing browser dependencies..."
sudo apt-get install -y \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2t64 \
    libpango-1.0-0 \
    libcairo2 \
    libatspi2.0-0

# Set up virtual environment in current directory
echo "[4/5] Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install --upgrade pip
pip install playwright

# Install Firefox browser for Playwright
echo "[5/5] Installing Firefox browser..."
playwright install firefox

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
