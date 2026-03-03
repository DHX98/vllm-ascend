#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "python interpreter not found. Set PYTHON_BIN to a valid interpreter."
  exit 1
fi

echo "[check] verifying required python modules..."
"${PYTHON_BIN}" - <<'PY'
from importlib.util import find_spec
required = ["pytest", "numpy", "torch"]
missing = [m for m in required if find_spec(m) is None]
if missing:
    raise SystemExit(f"missing modules: {', '.join(missing)}")
PY

echo "[run] python syntax checks..."
"${PYTHON_BIN}" -m py_compile \
  vllm_ascend/utils.py \
  vllm_ascend/attention/utils.py \
  vllm_ascend/attention/sfa_v1.py \
  vllm_ascend/worker/model_runner_v1.py \
  vllm_ascend/worker/pcp_utils.py \
  tests/ut/attention/test_lightning_indexer_utils.py \
  tests/ut/worker/test_pcp_manager.py

echo "[run] targeted unit tests for this change..."
"${PYTHON_BIN}" -m pytest -sv \
  tests/ut/attention/test_lightning_indexer_utils.py \
  tests/ut/worker/test_pcp_manager.py::test_build_dual_chunk_swap_plan_prefill \
  tests/ut/worker/test_pcp_manager.py::test_build_dual_chunk_swap_plan_decode_and_prefill

if [[ "${RUN_SFA_TESTS:-0}" == "1" ]]; then
  echo "[run] optional sfa unit test suite..."
  "${PYTHON_BIN}" -m pytest -sv tests/ut/attention/test_sfa_v1.py
fi

echo "[done] all requested tests finished."
