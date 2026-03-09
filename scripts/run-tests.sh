#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

if [[ $# -eq 0 ]]; then
  exec "$PYTHON_BIN" -m pytest -q tests
fi

exec "$PYTHON_BIN" -m pytest -q "$@"
