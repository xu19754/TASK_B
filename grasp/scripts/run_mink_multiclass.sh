#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec /usr/bin/python3 -u "$ROOT/experiments/mink_multiclass/run_suite.py" "$@"
