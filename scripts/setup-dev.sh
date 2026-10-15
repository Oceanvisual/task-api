#!/usr/bin/env bash
# Одноразовая настройка клона: venv как в проде (Python 3.11, ADR 0001/0002), mypy, git-хуки.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -x .venv/bin/python ]] || uv venv --python 3.11 .venv
uv pip install -q --python .venv/bin/python --index-strategy unsafe-best-match \
  --only-binary sqlalchemy -r requirements.txt mypy
git config core.hooksPath .githooks
echo "ok: .venv ($(.venv/bin/python --version)), hooksPath=$(git config core.hooksPath)"
