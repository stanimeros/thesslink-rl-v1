#!/usr/bin/env bash
#
# Remove every empty directory under results/ (repo root). Deepest-first so
# nested empties collapse. Does not delete results/ itself.
#
# Usage: ./clean_empty_results_dirs.sh
#
set -euo pipefail

_script="${BASH_SOURCE[0]:-$0}"
if [[ -L "$_script" ]] && command -v readlink >/dev/null; then
    if _r="$(readlink -f "$_script" 2>/dev/null)"; then
        _script="$_r"
    fi
fi
SCRIPT_DIR="$(cd "$(dirname "$_script")" && pwd)"
RESULTS="$SCRIPT_DIR/results"

if [[ ! -d "$RESULTS" ]]; then
    echo "No results/ directory at $RESULTS — nothing to do."
    exit 0
fi

# -mindepth 1: keep the top-level results/ folder; -depth: children before parents
n="$(find "$RESULTS" -mindepth 1 -depth -type d -empty -print | wc -l | tr -d ' ')"
find "$RESULTS" -mindepth 1 -depth -type d -empty -delete
echo "Removed $n empty director(y/ies) under results/"
