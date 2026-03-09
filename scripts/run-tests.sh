#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  ./scripts/run-tests.sh
  ./scripts/run-tests.sh tests/test_provider_base.py
  ./scripts/run-tests.sh tests/test_provider_base.py::test_text_stream_collect_collects_all_events
  ./scripts/run-tests.sh -k provider -x

Provider benchmarks (opt-in):
  ./scripts/run-tests.sh tests/test_provider_benchmarks.py --run-provider-benchmarks -s

Notes:
  - For benchmarks, real provider API keys must be set in environment variables.
  - All arguments are forwarded directly to pytest.
EOF
  exit 0
fi

if [[ $# -eq 0 ]]; then
  exec "$PYTHON_BIN" -m pytest -q tests
fi

ARGS=("$@")
RUN_BENCHMARKS=false
HAS_CAPTURE_FLAG=false

for arg in "${ARGS[@]}"; do
  if [[ "$arg" == "--run-provider-benchmarks" ]]; then
    RUN_BENCHMARKS=true
  fi

  if [[ "$arg" == "-s" || "$arg" == --capture=* || "$arg" == "--capture" ]]; then
    HAS_CAPTURE_FLAG=true
  fi
done

if [[ "$RUN_BENCHMARKS" == true && "$HAS_CAPTURE_FLAG" == false ]]; then
  ARGS+=("--capture=tee-sys")
fi

exec "$PYTHON_BIN" -m pytest -q "${ARGS[@]}"
