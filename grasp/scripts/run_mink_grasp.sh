#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/experiments/mink_grasp/runtime.sh"
cd "$ROOT"
# Standard-library supervisor; actual simulation and IK use Isaac Python 3.11.
exec /usr/bin/python3 -u "$ROOT/experiments/mink_grasp/supervise.py" "$@"
