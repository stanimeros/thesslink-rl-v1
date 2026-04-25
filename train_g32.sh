#!/usr/bin/env bash
# Concurrent g32 training: v2_g32 (full setup) then v4_neg_g32 + v4_nav_g32 in parallel.
# Each train.sh is started with nohup so you can close the terminal/IDE and training keeps going.
#
# Usage:
#   ./train_g32.sh              # foreground launcher; each train.sh still nohup'd
#   ./train_g32.sh --detach   # re-exec under nohup and exit immediately (best for IDE Run)
# W&B: metrics only (see train.sh); defaults aid26006-university-of-macedonia / thesslink-rl.
#   ./train_g32.sh --no-wandb   # forwarded — disable W&B for this pipeline
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

# Optional args: only --wandb* (same as train.sh); passed to each train.sh invocation.
FORWARD_TRAIN=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --wandb|--no-wandb)
      FORWARD_TRAIN+=("$1")
      shift
      ;;
    --wandb-entity|--wandb-project|--wandb-mode)
      [[ -n "${2:-}" ]] || { echo "[train_g32] $1 requires a value" >&2; exit 1; }
      FORWARD_TRAIN+=("$1" "$2")
      shift 2
      ;;
    *)
      echo "[train_g32] unknown argument: $1 (supported: --detach, --wandb, --no-wandb, --wandb-entity, --wandb-project, --wandb-mode)" >&2
      exit 1
      ;;
  esac
done

echo "[train_g32] Logs: $LOG_DIR"
echo "[train_g32] Starting v2_g32 under nohup → v2_g32.out"
nohup "$SCRIPT_DIR/train.sh" "${FORWARD_TRAIN[@]}" --env v2_g32 >"$LOG_DIR/v2_g32.out" 2>&1 &
v2_pid=$!
wait "$v2_pid"

echo "[train_g32] v2_g32 finished. Launching v4_neg_g32 and v4_nav_g32 (--quick) in parallel under nohup..."
nohup "$SCRIPT_DIR/train.sh" "${FORWARD_TRAIN[@]}" --env v4_neg_g32 --quick >"$LOG_DIR/v4_neg_g32_quick.out" 2>&1 &
neg_pid=$!
nohup "$SCRIPT_DIR/train.sh" "${FORWARD_TRAIN[@]}" --env v4_nav_g32 --quick >"$LOG_DIR/v4_nav_g32_quick.out" 2>&1 &
nav_pid=$!
echo "[train_g32] PIDs: v4_neg_g32=$neg_pid v4_nav_g32=$nav_pid"

wait "$neg_pid"
wait "$nav_pid"

echo "[train_g32] All three runs finished."
echo "[train_g32] Per-algorithm logs: epymarl/results/logs/ — monitor: ./train.sh --status"
