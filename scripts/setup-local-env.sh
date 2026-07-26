#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV_BIN:-uv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"

if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
  printf 'uv is required. Install it from https://docs.astral.sh/uv/ and rerun.\n' >&2
  exit 1
fi

pushd "${ROOT_DIR}" >/dev/null

"${UV_BIN}" python install "${PYTHON_VERSION}"
"${UV_BIN}" venv --python "${PYTHON_VERSION}" --clear
"${UV_BIN}" sync --locked --python "${PYTHON_VERSION}" --extra dev --extra local-runtime
"${UV_BIN}" run python -m local_rag_assistant --help >/dev/null

popd >/dev/null

printf 'Environment ready at %s/.venv\n' "${ROOT_DIR}"
printf 'Run commands with: uv run local-rag-assistant --help\n'
printf 'Or activate it with: source %s/.venv/bin/activate\n' "${ROOT_DIR}"
