#!/usr/bin/env bash
# Orchestrator for Qwen3 family on the Ascend host. Bootstraps a per-family
# venv (~/hfc/.venv-qwen3) with newer transformers, then runs run_qwen3_moe.py.
#
# Usage:
#   bash scripts/run_qwen3_remote.sh [model_id] [scale] [policy] [extra args...]
#
# Defaults aimed at "does it capture at all" rather than full-quality:
#   model_id  Qwen/Qwen3-30B-A3B
#   scale     0.1   (shrink layers/experts so config builds in CPU RAM)
#   policy    kv
#
# Pass --no-run as an extra arg to stop after capture+rewrite verification.
set -euo pipefail

MODEL="${1:-Qwen/Qwen3-30B-A3B}"
SCALE="${2:-0.1}"
POLICY="${3:-kv}"
shift $(( $# > 3 ? 3 : $# ))
EXTRA_ARGS=("$@")

REMOTE="kkrazy@47.103.62.251"
SSH_PORT=60008
SSH_OPTS=(-p "${SSH_PORT}" -o StrictHostKeyChecking=accept-new)
SCP_OPTS=(-P "${SSH_PORT}" -o StrictHostKeyChecking=accept-new)

LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_OUT="${LOCAL_ROOT}/out"
mkdir -p "${LOCAL_OUT}"

REMOTE_ROOT="/home/kkrazy/hfc"
VENV_NAME="qwen3"
REQ_NAME="requirements-qwen3.txt"

echo "[run-qwen3] model=${MODEL}  scale=${SCALE}  policy=${POLICY}  extra=${EXTRA_ARGS[*]:-}"
echo "[run-qwen3] local=${LOCAL_ROOT}  remote=${REMOTE}:${REMOTE_ROOT}  venv=.venv-${VENV_NAME}"

# 1) push scripts/ + hfc package to remote
ssh "${SSH_OPTS[@]}" "${REMOTE}" "mkdir -p ${REMOTE_ROOT}"
if command -v rsync >/dev/null 2>&1; then
    rsync -az -e "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=accept-new" \
        "${LOCAL_ROOT}/scripts/" "${REMOTE}:${REMOTE_ROOT}/"
    rsync -az --delete -e "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=accept-new" \
        "${LOCAL_ROOT}/hfc/" "${REMOTE}:${REMOTE_ROOT}/hfc/"
else
    scp "${SCP_OPTS[@]}" "${LOCAL_ROOT}/scripts/"* "${REMOTE}:${REMOTE_ROOT}/"
    scp -r "${SCP_OPTS[@]}" "${LOCAL_ROOT}/hfc" "${REMOTE}:${REMOTE_ROOT}/"
fi

# 2) bootstrap per-family venv (idempotent)
echo "[run-qwen3] bootstrap .venv-${VENV_NAME} on host"
ssh "${SSH_OPTS[@]}" "${REMOTE}" \
    "HFC_VENV_NAME=${VENV_NAME} HFC_REQ_NAME=${REQ_NAME} bash ${REMOTE_ROOT}/setup_remote.sh"

# 3) execute under the qwen3 env
echo "[run-qwen3] running run_qwen3_moe.py on NPU"
ssh "${SSH_OPTS[@]}" "${REMOTE}" \
    "source ${REMOTE_ROOT}/env-${VENV_NAME}.sh && \
     cd ${REMOTE_ROOT} && \
     python ${REMOTE_ROOT}/run_qwen3_moe.py \
        --model '${MODEL}' --scale ${SCALE} --policy ${POLICY} \
        --out-dir ${REMOTE_ROOT}/out ${EXTRA_ARGS[*]:-}"

echo "[run-qwen3] done"
