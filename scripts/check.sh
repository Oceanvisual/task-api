#!/usr/bin/env bash
# Локальный аналог CI (.github/workflows/deploy.yml): lint -> format(изменённые) -> mypy ratchet -> pytest.
# Использует .venv проекта (см. CLAUDE.md «Команды»). Флаг --fast: без pytest.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH" LLM_API_KEY="${LLM_API_KEY:-dummy-for-mocked-tests}" ENABLE_WEB_SEARCH=false

echo "▶ ruff check";  uvx ruff check app tests
base=$(git merge-base HEAD origin/main 2>/dev/null || echo HEAD)
files=()  # без mapfile: на macOS системный bash 3.2
while IFS= read -r f; do files+=("$f"); done < <( { git diff --name-only --diff-filter=ACMR "$base" -- '*.py'; git ls-files --others --exclude-standard -- '*.py'; } | sort -u)
echo "▶ ruff format --check (изменённые: ${#files[@]})"
(( ${#files[@]} == 0 )) || uvx ruff format --check "${files[@]}"
command -v mypy >/dev/null || uv pip install -q --python .venv/bin/python mypy
echo "▶ mypy ratchet"; scripts/mypy-ratchet.sh
[[ "${1:-}" == "--fast" ]] && exit 0
echo "▶ pytest"; python -m pytest -q -p no:cacheprovider
