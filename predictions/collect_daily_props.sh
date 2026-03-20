#!/bin/bash
# Collect NBA player props from Sports Game Odds API
# Runs daily before games start. Caches lines for backtesting.
# API trial ends March 19, 2026 — collect as much as possible.

cd /Users/sneaky/Desktop/nba/predictions

echo "[$(date)] Starting SGO props collection..."
/usr/bin/python3 sgo_client.py --cache >> cache/sgo/collection.log 2>&1
echo "[$(date)] Done." >> cache/sgo/collection.log
