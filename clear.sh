#!/usr/bin/env bash
# Kill running training processes and wipe all training outputs.
set -euo pipefail

_script="${BASH_SOURCE[0]:-$0}"
if [[ -L "$_script" ]] && command -v readlink >/dev/null; then
    if _r="$(readlink -f "$_script" 2>/dev/null)"; then _script="$_r"; fi
fi
SCRIPT_DIR="$(cd "$(dirname "$_script")" && pwd)"

RESULTS_DIR_ABS="$SCRIPT_DIR/epymarl/results"
EPYMARL_SRC="epymarl/src"

log() { echo -e "\033[1;32m[clear]\033[0m $*"; }

log "Killing running training processes..."
pkill -f "$EPYMARL_SRC/main.py" 2>/dev/null && log "Killed." || log "No processes found."

log "Wiping all training outputs (sacred, models, logs)..."
rm -rf "$RESULTS_DIR_ABS/sacred" "$RESULTS_DIR_ABS/models" "$RESULTS_DIR_ABS/logs"

log "Done."
