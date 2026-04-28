#!/usr/bin/env bash
# Pull latest changes, install deps, and set up EPyMARL.
set -euo pipefail

_script="${BASH_SOURCE[0]:-$0}"
if [[ -L "$_script" ]] && command -v readlink >/dev/null; then
    if _r="$(readlink -f "$_script" 2>/dev/null)"; then _script="$_r"; fi
fi
SCRIPT_DIR="$(cd "$(dirname "$_script")" && pwd)"
cd "$SCRIPT_DIR"

EPYMARL_SRC="epymarl/src"
VENV=".venv/bin/activate"

log() { echo -e "\033[1;32m[update]\033[0m $*"; }
err() { echo -e "\033[1;31m[update]\033[0m $*" >&2; }

# ── Pull ─────────────────────────────────────────────────────────────────

log "Pulling latest changes..."
git fetch origin main
git reset --hard origin/main

# ── Virtualenv + deps ────────────────────────────────────────────────────

if [ ! -f "$VENV" ]; then
    log "Creating virtualenv..."
    python3 -m venv .venv
fi
log "Activating virtualenv..."
source "$VENV"

log "Installing requirements..."
pip install -r requirements.txt

# ── EPyMARL ──────────────────────────────────────────────────────────────

if [ ! -d "epymarl" ]; then
    log "Cloning EPyMARL..."
    git clone https://github.com/uoe-agents/epymarl.git
    pip install -r epymarl/requirements.txt
    pip install -e .
fi

log "Applying EPyMARL patches..."
PATCH_DIR="$SCRIPT_DIR/epymarl_config/patches"
_PATCHES=(
    "$PATCH_DIR/epymarl_01_thesslink_base.patch"
    "$PATCH_DIR/epymarl_02_wandb_run_name.patch"
    "$PATCH_DIR/epymarl_03_wandb_tags_group.patch"
    "$PATCH_DIR/epymarl_04_thesslink_gymma_time_limit.patch"
)
for _p in "${_PATCHES[@]}"; do
    [[ -f "$_p" ]] || { err "Missing patch: $_p"; exit 1; }
done
git -C epymarl checkout -- . 2>/dev/null || true
for _patch in "${_PATCHES[@]}"; do
    git -C epymarl apply "$_patch" || { err "Patch failed: $_patch"; exit 1; }
done
log "Patches applied."

log "Copying env YAMLs into EPyMARL..."
_copied=0
for _f in epymarl_config/envs/*.yaml; do
    [[ -f "$_f" ]] || continue
    cp "$_f" "$EPYMARL_SRC/config/envs/"
    ((_copied++)) || true
done
((_copied > 0)) || { err "No YAMLs found in epymarl_config/envs/"; exit 1; }

log "Copying alg YAMLs into EPyMARL..."
_copied=0
for _f in epymarl_config/algs/*.yaml; do
    [[ -f "$_f" ]] || continue
    cp "$_f" "$EPYMARL_SRC/config/algs/"
    ((_copied++)) || true
done
((_copied > 0)) || { err "No YAMLs found in epymarl_config/algs/"; exit 1; }

log "Done."
