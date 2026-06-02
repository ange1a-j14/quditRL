#!/usr/bin/env bash
# Sync quditRL to a GCP GPU instance and launch training in tmux.
#
# Usage (from repo root):
#   ./scripts/gcp_run.sh sync          # push local code to the VM
#   ./scripts/gcp_run.sh setup         # sync + first-time venv/deps on the VM
#   ./scripts/gcp_run.sh run           # sync + run experiments from scripts/experiments.txt
#   ./scripts/gcp_run.sh attach        # attach to the tmux session
#   ./scripts/gcp_run.sh logs          # tail the run log
#   ./scripts/gcp_run.sh poll          # status + pull/plot metrics from the latest log
#   ./scripts/gcp_run.sh pull          # copy checkpoints/output back to local machine
#   ./scripts/gcp_run.sh ssh           # open an interactive shell on the VM
#
# Experiments (one command per line) live in scripts/experiments.txt.
# Override for a single run:
#   TRAIN_CMD='python research/train.py --d 4 --target haar --seed 1' ./scripts/gcp_run.sh run
# Continue after a failed experiment:
#   CONTINUE_ON_ERROR=1 ./scripts/gcp_run.sh run

set -euo pipefail

PROJECT="cs229-497921"
ZONE="us-central1-c"
INSTANCE="gpu-l4-1"
REMOTE_DIR="quditRL"
TMUX_SESSION="quditrl"
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXPERIMENTS_FILE="${EXPERIMENTS_FILE:-${LOCAL_ROOT}/scripts/experiments.txt}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

# # TRAIN_CMD="${TRAIN_CMD:-python research/train.py --algo bc --d 3 --target haar --hidden 256 --max-pulses 10 --seed 0 --bc-targets 200 --bc-updates-per-target 20 --cem-iters 100 --cem-population 256 --cem-elites 16 --cem-seq-len 10}"
# TRAIN_CMD="${TRAIN_CMD:-python research/train.py --algo ppo --d 3 --target haar --hidden 512 --seed 0 --total-timesteps 5000000 --max-pulses 25 --reward l1}"
# TRAIN_CMD="${TRAIN_CMD:-python research/train.py --algo cem --d 4 --target haar --hidden 512 --seed 0 --cem-targets 200 --cem-iters 25 --cem-population 128 --cem-elites 16 --cem-seq-len 10}"
# TRAIN_CMD="${TRAIN_CMD:-python research/train.py --algo bc --d 3 --target haar --hidden 256 --max-pulses 10 --seed 0 --bc-targets 200 --bc-updates-per-target 20 --cem-iters 100 --cem-population 256 --cem-elites 16 --cem-seq-len 10}"
# TRAIN_CMD="${TRAIN_CMD:-python research/train.py --algo amortized --d 4 --target haar --hidden 2048 --seq-len 20 --batch-targets 256 --amortized-iters 30000 --lr 1e-3 --loss infidelity --pulse-penalty 0 --target-curriculum --curriculum-frac 0.5 --curriculum-start-pulses 1 --curriculum-end-pulses 20 --seed 0}"
TRAIN_CMDS=()

load_train_cmds() {
  TRAIN_CMDS=()
  if [[ -n "${TRAIN_CMD:-}" ]]; then
    TRAIN_CMDS=("$TRAIN_CMD")
    return
  fi
  if [[ ! -f "${EXPERIMENTS_FILE}" ]]; then
    echo "ERROR: no TRAIN_CMD and experiments file not found: ${EXPERIMENTS_FILE}"
    exit 1
  fi
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ "${line}" =~ ^[[:space:]]*# ]] && continue
    [[ "${line}" =~ ^[[:space:]]*$ ]] && continue
    TRAIN_CMDS+=("${line}")
  done < "${EXPERIMENTS_FILE}"
  if [[ ${#TRAIN_CMDS[@]} -eq 0 ]]; then
    echo "ERROR: no experiments in ${EXPERIMENTS_FILE}"
    exit 1
  fi
}

GCLOUD_SSH=(gcloud compute ssh --zone "$ZONE" "$INSTANCE" --project "$PROJECT")
GCLOUD_SCP=(gcloud compute scp --recurse --zone "$ZONE" --project "$PROJECT")

remote() {
  "${GCLOUD_SSH[@]}" -- "$@"
}

sync() {
  echo "Syncing ${LOCAL_ROOT} -> ${INSTANCE}:~/${REMOTE_DIR}/"
  # gcloud compute ssh does not speak rsync's --server protocol; use tar over ssh instead
  tar czf - \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='.ipynb_checkpoints' \
    --exclude='checkpoints' \
    --exclude='output' \
    --exclude='notes' \
    --exclude='*.pt' \
    --exclude='.venv' \
    -C "${LOCAL_ROOT}" . \
  | "${GCLOUD_SSH[@]}" -- "mkdir -p ~/${REMOTE_DIR} && tar xzf - -C ~/${REMOTE_DIR}"
  echo "Sync complete."
}

setup() {
  echo "Setting up Python env on ${INSTANCE}..."
  remote bash -s <<EOF
set -euo pipefail
mkdir -p ~/${REMOTE_DIR}
cd ~/${REMOTE_DIR}
if ! dpkg -s python3-venv >/dev/null 2>&1; then
  echo "Installing python3-venv (requires sudo once on the VM)..."
  sudo apt-get update -qq
  sudo apt-get install -y python3-venv python3-pip
fi
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy gymnasium scipy matplotlib
mkdir -p checkpoints output
echo "Setup complete."
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
EOF
}

run() {
  load_train_cmds
  echo "Starting tmux session '${TMUX_SESSION}' on ${INSTANCE} with ${#TRAIN_CMDS[@]} experiment(s)..."
  local i
  for i in "${!TRAIN_CMDS[@]}"; do
    echo "  $((i + 1)). ${TRAIN_CMDS[$i]}"
  done

  local train_cmds_b64
  train_cmds_b64="$(printf '%s\n' "${TRAIN_CMDS[@]}" | base64 | tr -d '\n')"
  remote bash -s -- "${REMOTE_DIR}" "${TMUX_SESSION}" "${train_cmds_b64}" "${CONTINUE_ON_ERROR}" <<'EOF'
set -euo pipefail
REMOTE_DIR="$1"
TMUX_SESSION="$2"
TRAIN_CMDS_B64="$3"
CONTINUE_ON_ERROR="$4"
command -v tmux >/dev/null || { sudo apt-get update -qq && sudo apt-get install -y tmux; }
cd ~/"${REMOTE_DIR}"
mkdir -p checkpoints output logs
if [[ ! -f .venv/bin/activate ]]; then
  echo "ERROR: ~/quditRL/.venv not found. Run: ./scripts/gcp_run.sh setup"
  exit 1
fi
tmux has-session -t "${TMUX_SESSION}" 2>/dev/null && tmux kill-session -t "${TMUX_SESSION}"
LOG_FILE="logs/train_$(date +%Y%m%d_%H%M%S).log"
tmux new-session -d -s "${TMUX_SESSION}" \
  env REMOTE_DIR="${REMOTE_DIR}" LOG_FILE="${LOG_FILE}" TRAIN_CMDS_B64="${TRAIN_CMDS_B64}" CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR}" \
  bash -lc 'set -o pipefail; cd ~/"${REMOTE_DIR}" && source .venv/bin/activate && export PYTHONUNBUFFERED=1 && mapfile -t TRAIN_CMDS < <(printf "%s" "${TRAIN_CMDS_B64}" | base64 -d) && total=${#TRAIN_CMDS[@]} && start_ts=$(date +%s) && start_iso=$(date -Is) && { echo "START_TIME=${start_iso}"; echo "EXPERIMENTS=${total}"; echo "CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR}"; failed=0; for i in "${!TRAIN_CMDS[@]}"; do cmd="${TRAIN_CMDS[$i]}"; [[ -z "${cmd}" ]] && continue; n=$((i + 1)); echo; echo "=== EXPERIMENT ${n}/${total} ==="; echo "TRAIN_CMD=${cmd}"; exp_start=$(date +%s); eval "${cmd}"; status=$?; exp_elapsed=$(( $(date +%s) - exp_start )); echo "EXPERIMENT ${n}/${total} exit=${status} elapsed=${exp_elapsed}s"; if [[ ${status} -ne 0 ]]; then failed=$((failed + 1)); if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then exit "${status}"; fi; fi; done; if [[ ${failed} -gt 0 ]]; then exit 1; fi; } 2>&1 | tee "${LOG_FILE}"; status=${PIPESTATUS[0]}; end_ts=$(date +%s); end_iso=$(date -Is); elapsed=$((end_ts - start_ts)); { echo; echo "END_TIME=${end_iso}"; echo "ELAPSED_SECONDS=${elapsed}"; echo "Done with exit code ${status}. Log: ~/${REMOTE_DIR}/${LOG_FILE}"; } | tee -a "${LOG_FILE}"; exit "${status}"'
sleep 1
if ! tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  echo "ERROR: tmux session exited immediately."
  echo "On the VM, check: ls -lt ~/${REMOTE_DIR}/logs/ && tail ~/${REMOTE_DIR}/logs/*.log"
  exit 1
fi
echo "Training started in tmux session '${TMUX_SESSION}' (user: $(whoami))."
echo "Attach on VM:  tmux attach -t ${TMUX_SESSION}"
echo "Attach local:  ./scripts/gcp_run.sh attach"
EOF
}

attach() {
  "${GCLOUD_SSH[@]}" -- -t "tmux attach -t ${TMUX_SESSION}"
}

logs() {
  remote bash -s <<EOF
set -euo pipefail
cd ~/${REMOTE_DIR}/logs
latest="\$(ls -t train_*.log 2>/dev/null | head -1 || true)"
if [[ -z "\${latest}" ]]; then
  echo "No log files found in ~/${REMOTE_DIR}/logs"
  exit 1
fi
echo "Tailing \${latest} (Ctrl-C to stop)"
tail -f "\${latest}"
EOF
}

pull() {
  echo "Pulling checkpoints/ and output/ from ${INSTANCE}..."
  mkdir -p "${LOCAL_ROOT}/checkpoints" "${LOCAL_ROOT}/output"
  "${GCLOUD_SCP[@]}" "${INSTANCE}:~/${REMOTE_DIR}/checkpoints/*" "${LOCAL_ROOT}/checkpoints/" 2>/dev/null || true
  "${GCLOUD_SCP[@]}" "${INSTANCE}:~/${REMOTE_DIR}/output/*" "${LOCAL_ROOT}/output/" 2>/dev/null || true
  echo "Pull complete."
}

poll() {
  echo "=== Training status (${INSTANCE}) ==="
  local poll_info csv_remote csv_name png_path experiment latest_png
  poll_info=$(remote bash -s <<EOF
set -euo pipefail
cd ~/${REMOTE_DIR} 2>/dev/null || { echo "STATUS:missing"; exit 0; }

if tmux has-session -t ${TMUX_SESSION} 2>/dev/null; then
  echo "STATUS:running"
else
  echo "STATUS:stopped"
fi

latest_log=\$(ls -t logs/train_*.log 2>/dev/null | head -1 || true)
if [[ -n "\${latest_log}" ]]; then
  echo "LOG:\${latest_log}"
  experiment=\$(grep -oE '=== EXPERIMENT [0-9]+/[0-9]+ ===' "\${latest_log}" 2>/dev/null | tail -1 | sed 's/^=== //;s/ ===$//' || true)
  [[ -n "\${experiment}" ]] && echo "EXP:\${experiment}"
  while IFS= read -r csv_path; do
    [[ -n "\${csv_path}" ]] && echo "CSV:\${csv_path}"
  done < <(
    grep -oE 'Logging metrics to output/[^ ]+\.csv' "\${latest_log}" 2>/dev/null \
      | sed 's/^Logging metrics to //' \
      | awk '!seen[\$0]++'
  )
  latest_csv=\$(grep -oE 'Logging metrics to output/[^ ]+\.csv' "\${latest_log}" 2>/dev/null | sed 's/^Logging metrics to //' | tail -1 || true)
  grep -F -e '[iter ' -e '[target ' -e '    eval |' "\${latest_log}" 2>/dev/null | tail -3 || tail -3 "\${latest_log}"
else
  latest_csv=""
fi

if [[ -n "\${latest_csv}" ]]; then
  echo "METRICS:\$(tail -1 "\${latest_csv}")"
fi
EOF
)

  case "$(echo "${poll_info}" | sed -n 's/^STATUS://p' | head -1)" in
    running) echo "tmux:   running (${TMUX_SESSION})" ;;
    stopped) echo "tmux:   not running" ;;
    missing) echo "Project not found on VM — run ./scripts/gcp_run.sh setup"; return 1 ;;
  esac

  if experiment=$(echo "${poll_info}" | sed -n 's/^EXP://p' | head -1); then
    [[ -n "${experiment}" ]] && echo "queue:  ${experiment}"
  fi

  if metrics=$(echo "${poll_info}" | sed -n 's/^METRICS://p' | head -1); then
    [[ -n "${metrics}" ]] && echo "metrics: ${metrics}"
  fi

  if log_path=$(echo "${poll_info}" | sed -n 's/^LOG://p' | head -1); then
    [[ -n "${log_path}" ]] && echo "log:    ${log_path}"
  fi

  echo "${poll_info}" | grep -F -e '[iter ' -e '[target ' -e '    eval |' || true

  csv_remotes=()
  while IFS= read -r _csv; do
    [[ -n "${_csv}" ]] && csv_remotes+=("${_csv}")
  done < <(echo "${poll_info}" | sed -n 's/^CSV://p')
  if [[ ${#csv_remotes[@]} -eq 0 ]]; then
    echo ""
    echo "No metrics CSV found in the latest log yet."
    echo "Not pulling older output, to avoid showing a stale plot."
    return 0
  fi

  echo ""
  echo "Pulling ${#csv_remotes[@]} metrics file(s) from latest log..."
  mkdir -p "${LOCAL_ROOT}/output"
  for csv_remote in "${csv_remotes[@]}"; do
    csv_name=$(basename "${csv_remote}")
    echo "  output/${csv_name}"
    "${GCLOUD_SCP[@]}" "${INSTANCE}:~/${REMOTE_DIR}/${csv_remote}" "${LOCAL_ROOT}/output/"
    echo "  plotting output/${csv_name}"
    (cd "${LOCAL_ROOT}" && python3 -c "from metrics import plot_run; plot_run('output/${csv_name}')")
    latest_png="${LOCAL_ROOT}/output/${csv_name%.csv}.png"
  done

  if [[ ! -f "${latest_png}" ]]; then
    echo "Plot not created (is matplotlib installed locally?)"
    return 1
  fi

  echo "Opening ${latest_png}"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    open "${latest_png}"
  elif command -v xdg-open >/dev/null; then
    xdg-open "${latest_png}"
  else
    echo "(install xdg-open or open the file manually)"
  fi
}

ssh_shell() {
  "${GCLOUD_SSH[@]}"
}

usage() {
  sed -n '3,18p' "$0" | tr -d '#'
  exit 1
}

cmd="${1:-}"
case "${cmd}" in
  sync)   sync ;;
  setup)  sync && setup ;;
  run)    sync && run ;;
  attach) attach ;;
  logs)   logs ;;
  poll)   poll ;;
  pull)   pull ;;
  ssh)    ssh_shell ;;
  *)      usage ;;
esac
