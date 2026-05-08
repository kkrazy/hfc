#!/usr/bin/env bash
# Idempotent remote setup: venv + pinned deps + env.sh that sources CANN.
# Runs as user kkrazy on the Ascend host. Never touches system Python or CANN install.
# NOTE: no `set -u` — CANN's set_env.sh does `LD_LIBRARY_PATH=...:$LD_LIBRARY_PATH`
# without checking if the var is defined, which trips strict-mode unbound-variable
# errors. We use -e + pipefail and -x for visibility instead.
set -eo pipefail

HFC_ROOT="${HOME}/hfc"
# Per-model-family venv: pass HFC_VENV_NAME=qwen3 (etc.) to bootstrap a sibling
# venv at ~/hfc/.venv-qwen3 with its own pinned requirements file. Default empty
# preserves the original ~/hfc/.venv path used by OPT/Qwen2.
VENV_NAME="${HFC_VENV_NAME:-}"
if [[ -n "${VENV_NAME}" ]]; then
    VENV_DIR="${HFC_ROOT}/.venv-${VENV_NAME}"
    ENV_FILE="${HFC_ROOT}/env-${VENV_NAME}.sh"
else
    VENV_DIR="${HFC_ROOT}/.venv"
    ENV_FILE="${HFC_ROOT}/env.sh"
fi
REQ_FILE="${HFC_ROOT}/${HFC_REQ_NAME:-requirements.txt}"
PIP_INDEX="https://mirrors.aliyun.com/pypi/simple/"
PIP_HOST="mirrors.aliyun.com"
CANN_SET_ENV="/usr/local/Ascend/ascend-toolkit/set_env.sh"

mkdir -p "${HFC_ROOT}" "${HFC_ROOT}/out"

# Source CANN early so the version probe further down can import torch_npu
# (which dlopens libhccl.so from the CANN install).
if [[ -f "${CANN_SET_ENV}" ]]; then
    # shellcheck disable=SC1090
    source "${CANN_SET_ENV}"
fi

# 1) venv
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "[setup] creating venv at ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
fi

# 2) write env.sh (always overwrite — small file, source of truth)
cat > "${ENV_FILE}" <<EOF
# Sourced before running any capture_graph.py invocation.
set -e
if [[ -f "${CANN_SET_ENV}" ]]; then
    # shellcheck disable=SC1091
    source "${CANN_SET_ENV}"
fi
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="${HFC_ROOT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
# Activate venv last so its bin/ wins on PATH.
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
set +e
EOF

# 3) install deps (pip-only inside venv; no system changes)
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
python -m pip install --quiet --upgrade "pip>=24.0" "wheel" "setuptools" \
    --index-url "${PIP_INDEX}" --trusted-host "${PIP_HOST}"

if [[ ! -f "${REQ_FILE}" ]]; then
    echo "[setup] ERROR: ${REQ_FILE} missing — push it from local first" >&2
    exit 1
fi

echo "[setup] installing pinned deps from ${REQ_FILE}"
python -m pip install --requirement "${REQ_FILE}" \
    --index-url "${PIP_INDEX}" --trusted-host "${PIP_HOST}"

# 4) sanity print
echo "[setup] versions:"
python - <<'PY'
import importlib, sys
print(f"  python      = {sys.version.split()[0]}")
for mod in ("torch", "torch_npu", "transformers", "accelerate", "numpy"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:11s} = {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"  {mod:11s} = IMPORT FAILED ({e!r})")
PY

echo "[setup] CANN env:"
echo "  ASCEND_TOOLKIT_HOME = ${ASCEND_TOOLKIT_HOME:-unset}"
echo "  ASCEND_HOME_PATH    = ${ASCEND_HOME_PATH:-unset}"

echo "[setup] OK"
