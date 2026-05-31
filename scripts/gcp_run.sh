#!/usr/bin/env bash
# Sync quditRL to a GCP GPU instance and launch training in tmux.
#
# Usage (from repo root):
#   ./scripts/gcp_run.sh sync          # push local code to the VM
#   ./scripts/gcp_run.sh setup         # sync + first-time venv/deps on the VM
#   ./scripts/gcp_run.sh run           # start training in a detached tmux session
#   ./scripts/gcp_run.sh attach        # attach to the tmux session
#   ./scripts/gcp_run.sh logs          # tail the run log
#   ./scripts/gcp_run.sh poll          # status + pull latest metrics plot and open it
#   ./scripts/gcp_run.sh pull          # copy checkpoints/output back to local machine
#   ./scripts/gcp_run.sh ssh           # open an interactive shell on the VM
#
# Override the default train command:
#   TRAIN_CMD='python research/train.py --d 4 --target haar --seed 1' ./scripts/gcp_run.sh run

set -euo pipefail

PROJECT="cs229-497921"
ZONE="us-central1-c"
INSTANCE="gpu-l4-1"
REMOTE_DIR="quditRL"
TMUX_SESSION="quditrl"
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# TODO:
# Change the cmd below to change train settings
# Default training command (edit or override with TRAIN_CMD=...)
# # TRAIN_CMD="${TRAIN_CMD:-python research/train.py --algo bc --d 3 --target haar --hidden 256 --max-pulses 10 --seed 0 --bc-targets 200 --bc-updates-per-target 20 --cem-iters 100 --cem-population 256 --cem-elites 16 --cem-seq-len 10}"
# TRAIN_CMD="${TRAIN_CMD:-python research/train.py --algo ppo --d 3 --target haar --hidden 512 --seed 0 --total-timesteps 5000000 --max-pulses 25 --reward l1}"
# TRAIN_CMD="${TRAIN_CMD:-python research/train.py --algo cem --d 4 --target haar --hidden 512 --seed 0 --cem-targets 200 --cem-iters 25 --cem-population 128 --cem-elites 16 --cem-seq-len 10}"
# TRAIN_CMD="${TRAIN_CMD:-python research/train.py --algo bc --d 3 --target haar --hidden 256 --max-pulses 10 --seed 0 --bc-targets 200 --bc-updates-per-target 20 --cem-iters 100 --cem-population 256 --cem-elites 16 --cem-seq-len 10}"
TRAIN_CMD="${TRAIN_CMD:-python research/train.py --algo amortized --d 3 --target haar --hidden 512 --seq-len 6 --batch-targets 256 --amortized-iters 20000 --lr 1e-3 --loss infidelity --seed 0}"


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
  echo "Starting tmux session '${TMUX_SESSION}' on ${INSTANCE}..."
  local train_cmd_b64
  train_cmd_b64="$(printf '%s' "${TRAIN_CMD}" | base64 | tr -d '\n')"
  remote bash -s -- "${REMOTE_DIR}" "${TMUX_SESSION}" "${train_cmd_b64}" <<'EOF'
set -euo pipefail
REMOTE_DIR="$1"
TMUX_SESSION="$2"
TRAIN_CMD="$(printf '%s' "$3" | base64 -d)"
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
  env REMOTE_DIR="${REMOTE_DIR}" LOG_FILE="${LOG_FILE}" TRAIN_CMD="${TRAIN_CMD}" \
  bash -lc 'set -o pipefail; cd ~/"${REMOTE_DIR}" && source .venv/bin/activate && export PYTHONUNBUFFERED=1 && start_ts=$(date +%s); start_iso=$(date -Is); { echo "START_TIME=${start_iso}"; echo "TRAIN_CMD=${TRAIN_CMD}"; eval "${TRAIN_CMD}"; } 2>&1 | tee "${LOG_FILE}"; status=${PIPESTATUS[0]}; end_ts=$(date +%s); end_iso=$(date -Is); elapsed=$((end_ts - start_ts)); { echo; echo "END_TIME=${end_iso}"; echo "ELAPSED_SECONDS=${elapsed}"; echo "Done with exit code ${status}. Log: ~/${REMOTE_DIR}/${LOG_FILE}"; } | tee -a "${LOG_FILE}"; exit "${status}"'
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
  local poll_info csv_remote csv_name png_path
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
  latest_csv=\$(grep -oE 'Logging metrics to output/[^ ]+\.csv' "\${latest_log}" 2>/dev/null | sed 's/^Logging metrics to //' | tail -1 || true)
  grep -F -e '[iter ' -e '[target ' -e '    eval |' "\${latest_log}" 2>/dev/null | tail -3 || tail -3 "\${latest_log}"
else
  latest_csv=""
fi

if [[ -n "\${latest_csv}" ]]; then
  echo "CSV:\${latest_csv}"
  echo "METRICS:\$(tail -1 "\${latest_csv}")"
else
  echo "CSV:"
fi
EOF
)

  case "$(echo "${poll_info}" | sed -n 's/^STATUS://p' | head -1)" in
    running) echo "tmux:   running (${TMUX_SESSION})" ;;
    stopped) echo "tmux:   not running" ;;
    missing) echo "Project not found on VM — run ./scripts/gcp_run.sh setup"; return 1 ;;
  esac

  if metrics=$(echo "${poll_info}" | sed -n 's/^METRICS://p' | head -1); then
    [[ -n "${metrics}" ]] && echo "metrics: ${metrics}"
  fi

  if log_path=$(echo "${poll_info}" | sed -n 's/^LOG://p' | head -1); then
    [[ -n "${log_path}" ]] && echo "log:    ${log_path}"
  fi

  echo "${poll_info}" | grep -F -e '[iter ' -e '[target ' -e '    eval |' || true

  csv_remote=$(echo "${poll_info}" | sed -n 's/^CSV://p' | head -1)
  if [[ -z "${csv_remote}" ]]; then
    echo ""
    echo "No metrics CSV found in the latest log yet."
    echo "Not pulling older output, to avoid showing a stale plot."
    return 0
  fi

  csv_name=$(basename "${csv_remote}")
  echo ""
  echo "Pulling output/${csv_name}..."
  mkdir -p "${LOCAL_ROOT}/output"
  "${GCLOUD_SCP[@]}" "${INSTANCE}:~/${REMOTE_DIR}/${csv_remote}" "${LOCAL_ROOT}/output/"

  echo "Generating plot from latest metrics..."
  (cd "${LOCAL_ROOT}" && python3 -c "from metrics import plot_run; plot_run('output/${csv_name}')")

  png_path="${LOCAL_ROOT}/output/${csv_name%.csv}.png"
  if [[ ! -f "${png_path}" ]]; then
    echo "Plot not created (is matplotlib installed locally?)"
    return 1
  fi

  echo "Opening ${png_path}"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    open "${png_path}"
  elif command -v xdg-open >/dev/null; then
    xdg-open "${png_path}"
  else
    echo "(install xdg-open or open the file manually)"
  fi
}

ssh_shell() {
  "${GCLOUD_SSH[@]}"
}

usage() {
  sed -n '3,14p' "$0" | tr -d '#'
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
