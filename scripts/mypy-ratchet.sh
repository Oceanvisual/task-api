#!/usr/bin/env bash
# mypy-ratchet: падает, если ошибок mypy больше, чем в .mypy-baseline.
# Ошибок стало меньше -> обнови baseline: scripts/mypy-ratchet.sh --update
set -euo pipefail
cd "$(dirname "$0")/.."
baseline_file=.mypy-baseline
n=$(mypy app 2>&1 | grep -c ': error:' || true)
if [[ "${1:-}" == "--update" ]]; then echo "$n" > "$baseline_file"; echo "baseline = $n"; exit 0; fi
base=$(cat "$baseline_file")
echo "mypy errors: $n (baseline $base)"
if (( n > base )); then echo "::error::mypy: +$((n - base)) новых ошибок" >&2; exit 1; fi
if (( n < base )); then echo "::notice::mypy: ошибок меньше baseline — запусти scripts/mypy-ratchet.sh --update"; fi
