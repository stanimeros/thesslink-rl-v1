#!/usr/bin/env bash
# Concurrent g32 training: v2_g32 (full setup) then v4_neg_g32 + v4_nav_g32 in parallel.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[train_g32] Starting v2_g32 (full setup)..."
"$SCRIPT_DIR/train.sh" --env v2_g32

echo "[train_g32] v2_g32 setup done. Launching v4_neg_g32 and v4_nav_g32 with --quick..."
"$SCRIPT_DIR/train.sh" --env v4_neg_g32 --quick
"$SCRIPT_DIR/train.sh" --env v4_nav_g32 --quick

echo "[train_g32] All three environments launched."
echo "[train_g32] Monitor with: ./train.sh --status"
