#!/usr/bin/env bash
#
# ThessLink RL -- Parallel training launcher
#
# Usage:
#   ./train.sh --env 2          # IQL, QMIX, VDN, MAPPO, COMA on ThessLink v2 (10×10)
#   ./train.sh --env v4_neg     # v4 negotiation rewards (tiered agreement + timeout)
#   ./train.sh --env v4_nav     # v4 navigation (mixed agreed POI + timeout penalty)
#   ./train.sh --env v2_g32     # v2 on a 32×32 grid  (time_limit=480)
#   ./train.sh --env v2_g64     # v2 on a 64×64 grid  (time_limit=960)
#   ./train.sh --env v4_neg_g32 # v4-neg on a 32×32 grid
#   ./train.sh --env v4_nav_g32 # v4-nav on a 32×32 grid
#   ./train.sh # prompts for env selector (see env_catalog) if stdin is a TTY
#   ./train.sh --quick --env v2  # skip git-reset + kill; scoped wipe (safe for concurrent runs)
#   ./train.sh --kill     # kill all running training processes
#
# Weights & Biases (metrics only; checkpoints stay local under results/models):
#   Always on (entity aid26006-university-of-macedonia, project thesslink-rl). Override with
#   WANDB_ENTITY / WANDB_PROJECT / WANDB_MODE in the environment.
#   Optional: THESSLINK_WANDB_GROUP=custom_name  overrides W&B ``group`` (default is algo_env).
#   Runner/learner log intervals default 20k env steps (see epymarl_config/envs) so W&B
#   is not flooded every 2k steps; raise further if charts are still too dense.
#
# Reproducibility: Sacred ``seed`` (numpy, torch, env) defaults to THESSLINK_SEED or 42.
#
# Results layout (epymarl/results; Sacred local_results_path):
#   logs/v<N>/<alg>.log   — nohup (per env version; v2 and v3 runs are kept separate)
#   sacred/…/ThessLink-v<N>/…
#   models/…/ThessLink-v<N>_…/…
# Before smoke and again before full training: delete all of epymarl/results/{sacred,models,logs}.
#
set -euo pipefail

# Resolve repo root: dirname "$0" is wrong when invoked as `bash train.sh` from another
# directory ($0 is just "train.sh", so `cd .` follows the caller's cwd, not this file).
_script="${BASH_SOURCE[0]:-$0}"
if [[ -L "$_script" ]] && command -v readlink >/dev/null; then
    if _r="$(readlink -f "$_script" 2>/dev/null)"; then
        _script="$_r"
    fi
fi
SCRIPT_DIR="$(cd "$(dirname "$_script")" && pwd)"
cd "$SCRIPT_DIR"

# ``TRAINING_ALGOS`` lives in ``thesslink_rl/constants.py``. Load that file alone so
# this works before the venv exists (importing ``thesslink_rl`` would pull in gymnasium).
_training_algos_words() {
  python3 -c "
import importlib.util
from pathlib import Path
import sys
root = Path(sys.argv[1])
path = root / 'thesslink_rl' / 'constants.py'
spec = importlib.util.spec_from_file_location('_thesslink_constants', path)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)
print(' '.join(mod.TRAINING_ALGOS))
" "$SCRIPT_DIR"
}
read -r -a ALL_ALGOS <<< "$(_training_algos_words)"

_env_catalog_help() {
  python3 -c "
import importlib.util
from pathlib import Path
import sys
root = Path(sys.argv[1])
path = root / 'thesslink_rl' / 'env_catalog.py'
spec = importlib.util.spec_from_file_location('_thesslink_env_catalog', path)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)
print(mod.prompt_help())
" "$SCRIPT_DIR"
}
ENV_CHOICES="$(_env_catalog_help)"

RESULTS_DIR="epymarl/results"
# Absolute base; per-run nohup logs live in logs/<env-label>/.
RESULTS_DIR_ABS="$SCRIPT_DIR/$RESULTS_DIR"
LOGS_ROOT="$RESULTS_DIR_ABS/logs"
EPYMARL_SRC="epymarl/src"
VENV=".venv/bin/activate"

# ── Helpers ──────────────────────────────────────────────────────────────

log()  { echo -e "\033[1;32m[train]\033[0m $*"; }
warn() { echo -e "\033[1;33m[train]\033[0m $*"; }
err()  { echo -e "\033[1;31m[train]\033[0m $*" >&2; }

kill_training() {
    log "Killing running training processes..."
    pkill -f "$EPYMARL_SRC/main.py" 2>/dev/null && log "Killed." || log "No processes found."
}

prepare_results_tree() {
    local phase="$1"
    local label="${ENV_LABEL:-${THESSLINK_ENV:-${THESSLINK_ENV_VERSION:-unknown}}}"
    if [[ "${QUICK:-false}" == true ]]; then
        local cfg="${ENV_CONFIG:-$label}"
        log "Wiping version-scoped training outputs ($phase) for: ${label}"
        rm -rf "$LOGS_ROOT/${label}"
        find "$RESULTS_DIR_ABS/sacred" -mindepth 1 -maxdepth 5 \
            -type d -name "*${cfg}*" -exec rm -rf {} + 2>/dev/null || true
        find "$RESULTS_DIR_ABS/models" -mindepth 1 -maxdepth 5 \
            -type d -name "*${cfg}*" -exec rm -rf {} + 2>/dev/null || true
    else
        log "Wiping all training outputs ($phase): $RESULTS_DIR/{sacred,models,logs}"
        rm -rf "$RESULTS_DIR_ABS/sacred" "$RESULTS_DIR_ABS/models" "$RESULTS_DIR_ABS/logs"
    fi
    mkdir -p "$RESULTS_DIR_ABS/sacred" "$RESULTS_DIR_ABS/models" "$LOGS_ROOT/${label}"
}

# ── Parse arguments ──────────────────────────────────────────────────────

if [[ "${1:-}" == "--kill" ]]; then
    kill_training
    exit 0
fi

# Env selector + optional flags (--quick, --env; order-independent).
QUICK=false
WANDB_ENTITY_VAL="${WANDB_ENTITY:-aid26006-university-of-macedonia}"
WANDB_PROJECT_VAL="${WANDB_PROJECT:-thesslink-rl}"
WANDB_MODE_VAL="${WANDB_MODE:-online}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick) QUICK=true; shift ;;
        --env)
            [[ -n "${2:-}" ]] || { err "--env requires a value"; exit 1; }
            export THESSLINK_ENV="$2"; shift 2 ;;
        *) break ;;
    esac
done

if [[ -z "${THESSLINK_ENV:-}" && -z "${THESSLINK_ENV_VERSION:-}" ]]; then
    if [[ -t 0 ]]; then
        read -r -p "ThessLink env selector [${ENV_CHOICES}]: " THESSLINK_ENV
    else
        err "Env selector required: ./train.sh --env <${ENV_CHOICES}> or export THESSLINK_ENV."
        exit 1
    fi
fi

if [[ -z "${THESSLINK_ENV:-}" && -n "${THESSLINK_ENV_VERSION:-}" ]]; then
    THESSLINK_ENV="${THESSLINK_ENV_VERSION}"
fi

if ! python3 -c "
import importlib.util
from pathlib import Path
import sys
root = Path(sys.argv[1])
choice = sys.argv[2]
path = root / 'thesslink_rl' / 'env_catalog.py'
spec = importlib.util.spec_from_file_location('_thesslink_env_catalog', path)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)
mod.resolve_env_choice(choice)
" "$SCRIPT_DIR" "$THESSLINK_ENV" >/dev/null 2>&1; then
    err "Invalid env selector: ${THESSLINK_ENV} (available: ${ENV_CHOICES})"
    exit 1
fi
export THESSLINK_ENV

if [[ "$WANDB_MODE_VAL" != "online" && "$WANDB_MODE_VAL" != "offline" ]]; then
    err "WANDB_MODE must be online or offline (got: ${WANDB_MODE_VAL})."
    exit 1
fi

if [[ $# -gt 0 ]]; then
    err "This script always trains all algorithms (${ALL_ALGOS[*]}). Remove extra arguments: $*"
    exit 1
fi
ALGOS=("${ALL_ALGOS[@]}")

# ── Setup ────────────────────────────────────────────────────────────────

if [[ "$QUICK" == false ]]; then
    log "Killing previous training processes..."
    kill_training
fi

# ── Virtualenv ───────────────────────────────────────────────────────────

if [ ! -f "$VENV" ]; then
    log "Creating virtualenv..."
    python3 -m venv .venv
fi
log "Activating virtualenv..."
source "$VENV"

log "Installing project requirements (PyTorch is large — first install may take many minutes)..."
pip install -r requirements.txt

# Same ENV_VERSION / ENV_CONFIG as smoke_test.py and visualize.py (via import config).
eval "$(python3 <<'PY'
import config
print(f"export ENV_VERSION={config.ENV_VERSION}")
print(f"export ENV_LABEL={config.ENV_LABEL!r}")
print(f"export ENV_CONFIG={config.ENV_CONFIG!r}")
PY
)"

log "Environment: ${ENV_LABEL} (env-config=${ENV_CONFIG})"
log "Weights & Biases: project=${WANDB_PROJECT_VAL} entity=${WANDB_ENTITY_VAL} mode=${WANDB_MODE_VAL}"
SEED="${THESSLINK_SEED:-42}"
log "RL / env seed (THESSLINK_SEED): ${SEED}"

# ── EPyMARL ──────────────────────────────────────────────────────────────

if [ ! -d "epymarl" ]; then
    log "Cloning EPyMARL..."
    git clone https://github.com/uoe-agents/epymarl.git

    log "Installing EPyMARL dependencies..."
    pip install -r epymarl/requirements.txt
    pip install -e .
fi

log "Applying patches to EPyMARL..."
PATCH_DIR="$SCRIPT_DIR/epymarl_config/patches"
_PATCHES=(
  "$PATCH_DIR/epymarl_01_thesslink_base.patch"
  "$PATCH_DIR/epymarl_02_wandb_run_name.patch"
  "$PATCH_DIR/epymarl_03_wandb_tags_group.patch"
)
for _p in "${_PATCHES[@]}"; do
  if [[ ! -f "$_p" ]]; then
    err "Missing EPyMARL patch (commit it to the repo): $_p"
    exit 1
  fi
done
git -C epymarl checkout -- . 2>/dev/null || true
for _patch in "${_PATCHES[@]}"; do
  if ! git -C epymarl apply "$_patch"; then
    err "git apply failed: $_patch (fresh clone + upstream EPyMARL revision mismatch?)"
    exit 1
  fi
done
log "Patches applied (thesslink base + wandb run names + wandb tags/group)."

log "Copying ThessLink env YAMLs into EPyMARL (epymarl_config/envs/*.yaml)..."
_copied=0
for _f in epymarl_config/envs/*.yaml; do
    [[ -f "$_f" ]] || continue
    cp "$_f" "$EPYMARL_SRC/config/envs/"
    ((_copied++)) || true
done
if ((_copied == 0)); then
    err "No YAML files in epymarl_config/envs/ — add thesslink*.yaml (or new versions) there."
    exit 1
fi

# Validate algorithm names
for alg in "${ALGOS[@]}"; do
    if [ ! -f "$EPYMARL_SRC/config/algs/${alg}.yaml" ]; then
        err "Unknown algorithm: $alg"
        err "Available: ${ALL_ALGOS[*]}"
        exit 1
    fi
done

# ── Smoke test ───────────────────────────────────────────────────────────

prepare_results_tree "before smoke — short runs for all algorithms"
log "Smoke will use --env-config=${ENV_CONFIG} (see epymarl/src/config/envs/${ENV_CONFIG}.yaml)"

log "Running smoke test..."
if python smoke_test.py; then
    log "Smoke test passed!"
else
    err "Smoke test FAILED — aborting training."
    exit 1
fi

prepare_results_tree "after smoke — full multi-algo training"

LOGS_DIR="$LOGS_ROOT/${ENV_LABEL}"
mkdir -p "$LOGS_DIR"

# ── Launch training ──────────────────────────────────────────────────────

algo_extra_args() {
    # After venv: full package import is OK (gymnasium installed).
    # ``common_reward`` default lives in ``epymarl_config/envs/<ENV_CONFIG>.yaml``;
    # QMIX/VDN/COMA are always forced True inside ``thesslink_rl.constants``.
    PYTHONPATH="$SCRIPT_DIR" python -c "from thesslink_rl.constants import epymarl_common_reward_cli_flag; import sys; print(epymarl_common_reward_cli_flag(sys.argv[1], sys.argv[2]))" "$1" "$ENV_CONFIG"
}

log "Launching ${#ALGOS[@]} algorithm(s): ${ALGOS[*]}"
echo ""

# Metrics / config only; never upload checkpoints (local save_model unchanged).
WANDB_WITH=(
    use_wandb=True
    "wandb_team=${WANDB_ENTITY_VAL}"
    "wandb_project=${WANDB_PROJECT_VAL}"
    "wandb_mode=${WANDB_MODE_VAL}"
    wandb_save_model=False
)

PIDS=()
for alg in "${ALGOS[@]}"; do
    logfile="$LOGS_DIR/${alg}.log"
    extra=$(algo_extra_args "$alg")
    log "  Starting $alg -> $logfile ${extra:+(${extra})}"
    nohup python "$EPYMARL_SRC/main.py" \
        --config="$alg" \
        --env-config="$ENV_CONFIG" \
        with \
        local_results_path="$RESULTS_DIR_ABS" \
        "seed=${SEED}" \
        save_model=True \
        log_interval=20000 \
        runner_log_interval=20000 \
        learner_log_interval=20000 \
        test_interval=50000 \
        save_model_interval=400000 \
        t_max=2000000 \
        $extra \
        "${WANDB_WITH[@]}" \
        > "$logfile" 2>&1 &
    PIDS+=($!)
done

echo ""
log "All training jobs launched:"
for i in "${!ALGOS[@]}"; do
    echo "  ${ALGOS[$i]}  PID=${PIDS[$i]}  log=$LOGS_DIR/${ALGOS[$i]}.log"
done

echo ""
log "Kill all with: ./train.sh --kill"
log "Tail a log:    tail -f $LOGS_ROOT/<${ENV_CHOICES}>/<algo>.log"
