# Running MyHome.ie Scraper on AWS EC2

## Step 1: Launch EC2 Instance

1. Go to AWS Console → EC2 → Launch Instance
2. Settings:
   - **Name**: `myhome-scraper`
   - **AMI**: Ubuntu Server 22.04 LTS (free tier eligible)
   - **Instance type**: `t3.small` (recommended) or `t2.micro` (free tier, slower)
   - **Key pair**: Create new or use existing (you'll need this to SSH)
   - **Security group**: Allow SSH (port 22) from your IP
   - **Storage**: 20 GB gp3 (default is fine)

3. Click "Launch instance"

## Step 2: Connect to Your Instance

```bash
# Replace with your key file and instance public IP
ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP>
```

## Step 3: Clone the Repo and Run Setup

On the EC2 instance:

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/House-Price-Predictor.git
cd House-Price-Predictor/PPR-ALL

# Run setup
chmod +x ec2-setup.sh run-scraper.sh
./ec2-setup.sh
```

This installs Python, Playwright, and Firefox (~5 minutes).

## Step 4: Start Scraping

```bash
cd ~/House-Price-Predictor/PPR-ALL
./run-scraper.sh
```

## Step 6: Disconnect and Let It Run

The scraper runs in tmux, so you can safely close your SSH connection:

```bash
# Detach from tmux (scraper keeps running)
# Press: Ctrl+B, then D

# Or just close your terminal - scraper continues running
```

## Useful Commands

```bash
# Reattach to see progress
tmux attach -t scraper

# Check if scraper is running
tmux list-sessions

# Stop the scraper
tmux kill-session -t scraper

# Watch output file grow
tail -f output/myhome_sold_*.csv

# Check disk space
df -h

# Check memory usage
htop
```

## Download Results

From your local machine:

```bash
# Download all CSV files
scp -i your-key.pem "ubuntu@<EC2-PUBLIC-IP>:~/scraper/output/*.csv" .
```

## Cost Estimate

- **t3.small**: ~$0.02/hour = ~$15/month if running 24/7
- **t2.micro** (free tier): Free for first year, then ~$8/month

💡 **Tip**: Stop the instance when not scraping to save money!

## Scraping Options

```bash
# Scrape sold properties (price register)
./run-scraper.sh

# Scrape for-sale listings
./run-scraper.sh --for-sale

# Filter by county
./run-scraper.sh --county dublin

# Limit number of listings
./run-scraper.sh --limit 1000

# Resume interrupted scrape
./run-scraper.sh --resume

# Combine options
./run-scraper.sh --for-sale --county cork --limit 500
```

## Troubleshooting

### "Browser closed unexpectedly"
The instance may be out of memory. Try `t3.small` instead of `t2.micro`.

### "Connection refused" on SSH
Check your security group allows SSH from your IP.

### Scraper stops when I close terminal
Make sure you're using `./run-scraper.sh` which runs in tmux, not running python directly.
