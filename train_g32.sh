#!/usr/bin/env bash
# Train v7 neg + v7 nav in parallel on g32.
#
# Kills any running training processes, wipes previous results for these two
# env labels, then launches e3_neg_v7_g32 and e3_nav_v7_g32 side-by-side.
#
# Optional first argument only:
#   ./train_g32.sh --detach   # re-exec under nohup and exit (whole pipeline in background)
#
# W&B defaults / overrides: see train.sh (WANDB_* env vars). No other flags here.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs/train_g32"
mkdir -p "$LOG_DIR"

if [[ "${1:-}" == "--detach" ]]; then
  shift
  launcher="$LOG_DIR/launcher_$(date +%Y%m%d_%H%M%S).out"
  echo "[train_g32] Detaching → $launcher"
  nohup bash "$SCRIPT_DIR/train_g32.sh" "$@" >>"$launcher" 2>&1 &
  echo "[train_g32] Started pid=$! — safe to close this terminal. tail -f $launcher"
  exit 0
fi

if [[ $# -gt 0 ]]; then
  echo "[train_g32] unknown argument: $1 (only --detach is accepted)" >&2
  exit 1
fi

# ── Kill + clean ─────────────────────────────────────────────────────────────
echo "[train_g32] Killing any running training processes..."
"$SCRIPT_DIR/train.sh" --kill || true

# ── Launch v7 neg + v7 nav in parallel (--quick wipes only their own outputs) ─
echo "[train_g32] Launching e3_neg_v7_g32 and e3_nav_v7_g32 in parallel..."
nohup "$SCRIPT_DIR/train.sh" --env thesslink_e3_neg_v7_g32 --quick >"$LOG_DIR/e3_neg_v7_g32.out" 2>&1 &
neg7_pid=$!
nohup "$SCRIPT_DIR/train.sh" --env thesslink_e3_nav_v7_g32 --quick >"$LOG_DIR/e3_nav_v7_g32.out" 2>&1 &
nav7_pid=$!

echo "[train_g32] PIDs: e3_neg_v7_g32=$neg7_pid  e3_nav_v7_g32=$nav7_pid"
echo "[train_g32] Logs:"
echo "  $LOG_DIR/e3_neg_v7_g32.out"
echo "  $LOG_DIR/e3_nav_v7_g32.out"
echo "[train_g32] Monitor: ./train.sh --status"
