#!/usr/bin/env bash
# Pull latest changes from origin/main.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[update] Pulling latest changes..."
git fetch origin main
git reset --hard origin/main
echo "[update] Done."
