#!/usr/bin/env bash
# Local orchestrator: push scripts to the Ascend host, set up env, run capture,
# pull graph.txt back. All paths assume this repo lives at /wks/default/hfc/.
#
# Usage: bash scripts/run_remote.sh [model_id] [seq_len] [past_len]
#   model_id  default gpt2
#   seq_len   default 32
#   past_len  default 0  (set >0 to exercise the KV-cache decode path)
set -euo pipefail

MODEL="${1:-gpt2}"
SEQ_LEN="${2:-32}"
PAST_LEN="${3:-0}"

REMOTE="kkrazy@47.103.62.251"
SSH_PORT=60008
SSH_OPTS=(-p "${SSH_PORT}" -o StrictHostKeyChecking=accept-new)
SCP_OPTS=(-P "${SSH_PORT}" -o StrictHostKeyChecking=accept-new)
RSYNC_SSH="ssh ${SSH_OPTS[*]}"

LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_OUT="${LOCAL_ROOT}/out"
mkdir -p "${LOCAL_OUT}"

REMOTE_ROOT="/home/kkrazy/hfc"

echo "[run] model=${MODEL}  seq_len=${SEQ_LEN}  past_len=${PAST_LEN}"
echo "[run] local=${LOCAL_ROOT}  remote=${REMOTE}:${REMOTE_ROOT}"

# 1) push scripts/ to remote ~/hfc/  (rsync if available, else scp)
# Note: NO --delete — ~/hfc/ also contains .venv/, .hf_cache/, env.sh, out/.
# We only want to update the script files, not wipe sibling state.
if command -v rsync >/dev/null 2>&1; then
    echo "[run] rsync scripts/ -> ${REMOTE}:${REMOTE_ROOT}/"
    ssh "${SSH_OPTS[@]}" "${REMOTE}" "mkdir -p ${REMOTE_ROOT}"
    rsync -az -e "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=accept-new" \
        "${LOCAL_ROOT}/scripts/" "${REMOTE}:${REMOTE_ROOT}/"
else
    echo "[run] rsync not found — falling back to scp"
    ssh "${SSH_OPTS[@]}" "${REMOTE}" "mkdir -p ${REMOTE_ROOT}"
    scp "${SCP_OPTS[@]}" "${LOCAL_ROOT}/scripts/"* "${REMOTE}:${REMOTE_ROOT}/"
fi

# 2) run setup (idempotent — fast path if venv already exists)
echo "[run] running setup_remote.sh on the host"
ssh "${SSH_OPTS[@]}" "${REMOTE}" "bash ${REMOTE_ROOT}/setup_remote.sh"

# 3) run capture under the env (sources CANN + venv) and stream output back
echo "[run] running capture_graph.py on NPU"
ssh "${SSH_OPTS[@]}" "${REMOTE}" \
    "source ${REMOTE_ROOT}/env.sh && \
     python ${REMOTE_ROOT}/capture_graph.py \
        --model '${MODEL}' --seq-len '${SEQ_LEN}' --past-len '${PAST_LEN}' \
        --out-dir ${REMOTE_ROOT}/out"

# 4) pull graph.txt back
echo "[run] fetching ${REMOTE_ROOT}/out/graph.txt -> ${LOCAL_OUT}/graph.txt"
scp "${SCP_OPTS[@]}" "${REMOTE}:${REMOTE_ROOT}/out/graph.txt" "${LOCAL_OUT}/graph.txt"

echo "[run] done. ${LOCAL_OUT}/graph.txt:"
ls -la "${LOCAL_OUT}/graph.txt"
