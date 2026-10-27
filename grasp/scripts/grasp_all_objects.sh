#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec /usr/bin/python3 -u "$ROOT/scripts/banana_angles.py" --all-objects "$@"
