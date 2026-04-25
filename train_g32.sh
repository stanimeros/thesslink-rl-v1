#!/usr/bin/env bash
# v2_g32 (full) then v4_neg_g32 + v4_nav_g32 (--quick) in parallel; each train.sh under nohup.
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
  echo "[train_g32] Detaching full run → $launcher"
  nohup bash "$SCRIPT_DIR/train_g32.sh" "$@" >>"$launcher" 2>&1 &
  echo "[train_g32] Started pid=$! — safe to close this terminal. tail -f $launcher"
  exit 0
fi

if [[ $# -gt 0 ]]; then
  echo "[train_g32] unknown argument: $1 (only optional first flag: --detach)" >&2
  exit 1
fi

echo "[train_g32] Logs: $LOG_DIR"
echo "[train_g32] Starting v2_g32 under nohup → v2_g32.out"
nohup "$SCRIPT_DIR/train.sh" --env v2_g32 >"$LOG_DIR/v2_g32.out" 2>&1 &
v2_pid=$!
wait "$v2_pid"

echo "[train_g32] v2_g32 finished. Launching v4_neg_g32 and v4_nav_g32 (--quick) in parallel under nohup..."
nohup "$SCRIPT_DIR/train.sh" --env v4_neg_g32 --quick >"$LOG_DIR/v4_neg_g32_quick.out" 2>&1 &
neg_pid=$!
nohup "$SCRIPT_DIR/train.sh" --env v4_nav_g32 --quick >"$LOG_DIR/v4_nav_g32_quick.out" 2>&1 &
nav_pid=$!
echo "[train_g32] PIDs: v4_neg_g32=$neg_pid v4_nav_g32=$nav_pid"
echo "[train_g32] v4 jobs are under nohup — safe to disconnect; logs: $LOG_DIR/v4_neg_g32_quick.out  $LOG_DIR/v4_nav_g32_quick.out"
echo "[train_g32] Monitor: ./train.sh --status"
