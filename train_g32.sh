#!/usr/bin/env bash
# ThessLink RL — w6 + w7 g32 training launcher.
# Runs: setup → smoke test (neg + nav + full) → clear → launch all algos in parallel.
#
# Environments launched:
#   w6: thesslink_e3_w6_neg_v1_g32   (negotiation-only, v6_neg rewards)
#       thesslink_e3_w6_nav_v1_g32   (navigation-only, v6_nav rewards, t_limit=512)
#   w7: thesslink_e3_w7_full_v1_g32  (full neg→nav,     v7_full,        t_limit=352)
#
# Usage:
#   ./train_g32.sh            # run in foreground
#   ./train_g32.sh --detach   # re-exec under nohup and exit
#
# W&B overrides: WANDB_ENTITY / WANDB_PROJECT / WANDB_MODE
# Seed override: THESSLINK_SEED (default 42)
set -euo pipefail

_script="${BASH_SOURCE[0]:-$0}"
if [[ -L "$_script" ]] && command -v readlink >/dev/null; then
    if _r="$(readlink -f "$_script" 2>/dev/null)"; then _script="$_r"; fi
fi
SCRIPT_DIR="$(cd "$(dirname "$_script")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs/train_g32_w6_w7"
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

RESULTS_DIR_ABS="$SCRIPT_DIR/epymarl/results"
LOGS_ROOT="$RESULTS_DIR_ABS/logs"
EPYMARL_SRC="epymarl/src"
VENV=".venv/bin/activate"

NEG_ENV="thesslink_e3_w6_neg_v1_g32"
NAV_ENV="thesslink_e3_w6_nav_v1_g32"
FULL_ENV="thesslink_e3_w7_full_v1_g32"

WANDB_ENTITY_VAL="${WANDB_ENTITY:-aid26006-university-of-macedonia}"
WANDB_PROJECT_VAL="${WANDB_PROJECT:-thesslink-rl}"
WANDB_MODE_VAL="${WANDB_MODE:-online}"
SEED="${THESSLINK_SEED:-42}"

log()  { echo -e "\033[1;32m[train_g32]\033[0m $*"; }
err()  { echo -e "\033[1;31m[train_g32]\033[0m $*" >&2; }

source "$VENV"

read -r -a ALL_ALGOS <<< "$(python3 -c "
import importlib.util, sys
from pathlib import Path
root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('_c', root / 'thesslink_rl' / 'constants.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
print(' '.join(mod.TRAINING_ALGOS))
" "$SCRIPT_DIR")"

# ── Smoke test ───────────────────────────────────────────────────────────

for env_cfg in "$NEG_ENV" "$NAV_ENV" "$FULL_ENV"; do
    log "Smoke test: $env_cfg"
    THESSLINK_ENV="$env_cfg" python smoke_test.py || { err "Smoke FAILED for $env_cfg — aborting."; exit 1; }
done

# ── Clear previous outputs ───────────────────────────────────────────────

log "Clearing previous results..."
"$SCRIPT_DIR/clear.sh"

# ── Launch training ──────────────────────────────────────────────────────

log "W&B: entity=${WANDB_ENTITY_VAL}  project=${WANDB_PROJECT_VAL}  mode=${WANDB_MODE_VAL}"
log "Seed: ${SEED}  Algos: ${ALL_ALGOS[*]}"

WANDB_WITH=(
    use_wandb=True
    "wandb_team=${WANDB_ENTITY_VAL}"
    "wandb_project=${WANDB_PROJECT_VAL}"
    "wandb_mode=${WANDB_MODE_VAL}"
    wandb_save_model=False
)

PIDS=()
for env_cfg in "$NEG_ENV" "$NAV_ENV" "$FULL_ENV"; do
    mkdir -p "$LOGS_ROOT/${env_cfg}"
    for alg in "${ALL_ALGOS[@]}"; do
        logfile="$LOGS_ROOT/${env_cfg}/${alg}.log"
        log "  Starting ${env_cfg}/${alg} → $logfile"
        nohup python "$EPYMARL_SRC/main.py" \
            --config="$alg" \
            --env-config="$env_cfg" \
            with \
            local_results_path="$RESULTS_DIR_ABS" \
            "seed=${SEED}" \
            "${WANDB_WITH[@]}" \
            > "$logfile" 2>&1 &
        PIDS+=($!)
    done
done

echo ""
log "All ${#PIDS[@]} training jobs launched."
log "Kill + clear: ./clear.sh"
log "Neg logs:     $LOGS_ROOT/${NEG_ENV}/<algo>.log"
log "Nav logs:     $LOGS_ROOT/${NAV_ENV}/<algo>.log"
log "Full logs:    $LOGS_ROOT/${FULL_ENV}/<algo>.log"
