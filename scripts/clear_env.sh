#!/bin/bash
# Clear stale Breeze credentials from shell environment
# This ensures all scripts read fresh values from .env file

echo "Clearing stale environment variables..."

unset BREEZE_API_KEY
unset BREEZE_API_SECRET
unset BREEZE_SESSION_TOKEN
unset MODE

echo "✓ Cleared: BREEZE_API_KEY, BREEZE_API_SECRET, BREEZE_SESSION_TOKEN, MODE"
echo "✓ Scripts will now load fresh values from .env file"
echo ""
echo "Verify with: env | grep -E 'BREEZE|MODE'"
